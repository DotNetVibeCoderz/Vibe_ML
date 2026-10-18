#!/usr/bin/env python3
"""Cross-checks SciNet's HDF5 writer (and the MATLAB v7.3 and NetCDF-4 writers built on it)
against the reference HDF5 library, through h5py and netCDF4-python, in both directions.

SciNet's tests can only show that the writer agrees with SciNet's own reader, which shares its
author's reading of the format. This script is what establishes that HDF5 itself accepts the
files: h5py opens and reads everything, HDF5 then *modifies* the files in place (which walks and
rewrites object headers, so a wrong checksum, size or message layout fails here even when a
read-only open would not), and SciNet reads the modified files back.

Run it after any change to Hdf5Writer.cs, MatV73Writer.cs or NetCdf4Writer.cs:

    pip install h5py netCDF4 numpy
    python tools/verify/hdf5_interop.py

It builds and drives tools/verify/Hdf5Interop (the C# half), which writes the files and dumps the
ones Python has modified. Checked with h5py 3.16 / HDF5 2.0.0 and netCDF4 1.7.5 (netCDF-C 4.10.1,
HDF5 2.2.0).
"""

import os
import shutil
import subprocess
import sys
import tempfile

try:
    import h5py
    import numpy as np
except ImportError:
    sys.exit("h5py and numpy are required: pip install h5py numpy")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARNESS = os.path.join(REPO, "tools", "verify", "Hdf5Interop")
DLL = os.path.join(HARNESS, "bin", "Release", "net10.0", "Hdf5Interop.dll")

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {name}{('  -- ' + str(detail)) if detail and not condition else ''}")
    if not condition:
        failures.append(name)


def build():
    r = subprocess.run(["dotnet", "build", HARNESS, "-c", "Release", "-v", "q", "-nologo"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit("building the C# harness failed")


def run_csharp(*args):
    r = subprocess.run(["dotnet", DLL, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        sys.exit(f"the C# harness failed on: {' '.join(args)}")
    return r.stdout


def read_everything(f):
    """Visits every object and reads every dataset and attribute; returns the errors."""
    errors = []

    def visit(name, obj):
        try:
            for k in obj.attrs:
                obj.attrs[k]
            if isinstance(obj, h5py.Dataset):
                obj[()]
        except Exception as e:  # noqa: BLE001 - any failure is the finding
            errors.append(f"{name}: {e}")

    for k in f.attrs:
        try:
            f.attrs[k]
        except Exception as e:  # noqa: BLE001
            errors.append(f"/@{k}: {e}")
    f.visititems(visit)
    return errors


# ---------------------------------------------------------------- 1. generic HDF5

def check_generic(work):
    print("\n=== 1. Hdf5Writer writes, h5py reads ===")
    path = os.path.join(work, "generic.h5")
    run_csharp("write-h5", path)
    with h5py.File(path, "r") as f:
        check("every object and attribute reads", not read_everything(f), read_everything(f))
        check("superblock version 2", f.id.get_create_plist().get_version()[0] == 2, f.id.get_create_plist().get_version())
        n = f["num"]
        check("f8 values and shape", np.array_equal(n["f8"][()], (np.arange(24) * 0.5 - 3).reshape(2, 3, 4)))
        check("f4", np.array_equal(n["f4"][()], np.array([1.5, -2.25, 3.0, 4.0e10], dtype="<f4")) and n["f4"].dtype == np.dtype("<f4"))
        check("f2", n["f2"].dtype == np.float16 and n["f2"][()].tolist() == [0.5, -1.25, 65504.0])
        check("i1", n["i1"].dtype == np.int8 and n["i1"][()].tolist() == [-128, -1, 0, 1, 127])
        check("u1", n["u1"].dtype == np.uint8 and n["u1"][()].tolist() == [0, 1, 200, 255])
        check("i2 2-D", n["i2"][()].tolist() == [[-32768, -2], [3, 32767]])
        check("u2", n["u2"][()].tolist() == [0, 1, 65535, 4660])
        check("i4", n["i4"][()].tolist() == [-2**31, 5, 7, 2**31 - 1])
        check("u4", n["u4"][()].tolist() == [0, 4294967295, 12345])
        check("i8", n["i8"][()].tolist() == [-(1 << 62), -1, 1 << 53, 9])
        check("u8", n["u8"][()].tolist() == [0, 1 << 63, (1 << 40) + 1, 2**64 - 1])
        check("bool is h5py's bool", n["bool"].dtype == np.bool_ and n["bool"][()].tolist() == [True, False, True, True])
        check("complex is h5py's complex", n["c16"].dtype == np.complex128
              and n["c16"][()].tolist() == [[1 + 2j, -3 + 0.5j], [-1j, 4.25 + 0j]])
        check("scalar dataspace", n["scalar"].shape == () and n["scalar"][()] == 2.5)
        check("empty (0, 3)", n["empty"].shape == (0, 3))
        check("reshaped int64", n["reshaped"][()].tolist() == [[0, 1], [2, 3], [4, 5]])

        s = f["str"]
        check("vlen utf-8 strings", s["vlen"].asstr()[()].tolist() == ["alpha", "", "日本語 €", "z"])
        check("fixed strings", s["fixed"][()].tolist() == [b"ab", b"cde", b""])
        check("scalar string", s["scalar"].asstr()[()] == "one string")

        refs = f["refs/objs"][()]
        check("object references", [f[r].name for r in refs] == ["/num/f8", "/str", "/"])
        check("reference attribute", f[f.attrs["ref"]].name == "/num/f8")

        c = f["chunked"]
        check("deflate dataset", c["deflate"].compression == "gzip" and c["deflate"].chunks == (37, 23)
              and np.array_equal(c["deflate"][()], (np.arange(37 * 23) % 17 - 8).reshape(37, 23)))
        check("extendible dataset", c["ext"].maxshape == (None, 3) and c["ext"][()].tolist() == np.arange(15.0).reshape(5, 3).tolist())
        check("extendible + deflate", c["extz"].maxshape == (None,) and c["extz"].compression == "gzip" and c["extz"][()].tolist() == [10, 20, 30, 40])
        check("extendible, no data", c["ext0"].shape == (0, 2) and c["ext0"].maxshape == (None, 2))

        m = f["many"]
        check("20 links in one header", sorted(m.keys()) == sorted([f"d{i:02}" for i in range(20)] + ["attrs"]))
        check("12 compact attributes", [m["attrs"].attrs[f"a{i:02}"] for i in range(12)] == [10 * i for i in range(12)])
        check("nested groups", f["a/b/c/deep"][()].tolist() == [42])
        check("utf-8 link name", "ünï" in f)

        a = f.attrs
        check("vlen string attribute", a["title"] == "SciNet HDF5 writer" and a["utf"] == "héllo €")
        check("fixed string attribute", a["fixed"] == b"fixed text")
        check("numeric attributes", a["version"] == 3 and a["version"].dtype == np.int32 and a["pi"] == np.pi)
        check("array attributes", a["arr"].tolist() == [1, -2, 3] and a["mat"].tolist() == [[1, 2, 3], [4, 5, 6]])
        check("bool / complex attributes", a["flag"] == True and a["z"] == 1 - 2j)  # noqa: E712
        check("string array attribute", list(a["names"]) == ["x", "yy", "zzz"])
        check("creation order kept", list(a.keys())[:3] == ["title", "utf", "fixed"] if f.id.get_create_plist().get_attr_creation_order() else True)
        check("dataset attribute", f["num/f8"].attrs["units"] == "m/s")

    # HDF5 modifies the file: attributes go dense, a dataset grows, links are added and deleted.
    mod = os.path.join(work, "modified.h5")
    shutil.copy(path, mod)
    with h5py.File(mod, "r+") as f:
        f["num/f8"].attrs["added"] = np.arange(5)
        for i in range(30):
            f["many/attrs"].attrs[f"b{i}"] = i
        f.create_dataset("new/x", data=np.arange(10.0))
        f["chunked/ext"].resize((7, 3))
        f["chunked/ext"][5:] = -1
        del f["num/u2"]
        for i in range(10):
            f["a"].create_group(f"g{i}")
    with h5py.File(mod, "r") as f:
        check("HDF5 can modify the file", not read_everything(f) and len(f["many/attrs"].attrs) == 42 and f["chunked/ext"].shape == (7, 3))
    out = run_csharp("dump-h5", mod)
    check("SciNet reads the HDF5-modified file", "D /new/x <f8 (10)" in out and "@added=[0,1,2,3,4]" in out
          and "D /chunked/ext <f8 (7,3) Chunked [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,-1,-1,-1,-1,-1,-1]" in out
          and "/num/u2" not in out and "@b29=29" in out and "g9" in out, out[:400])


# ---------------------------------------------------------------- 2. MATLAB v7.3

def mat73_load(f, obj):
    """A small mat73-style decoder (what the `mat73` package does), written against MATLAB's layout."""
    cls = obj.attrs.get("MATLAB_class", b"").decode()
    if isinstance(obj, h5py.Group):
        if "MATLAB_sparse" in obj.attrs:
            rows = int(obj.attrs["MATLAB_sparse"])
            jc = obj["jc"][()]
            ir = obj["ir"][()] if "ir" in obj else np.array([], dtype="u8")
            data = obj["data"][()] if "data" in obj else np.array([])
            if data.dtype.names:
                data = data["real"] + 1j * data["imag"]
            dense = np.zeros((rows, len(jc) - 1), dtype=data.dtype if data.size else float)
            for j in range(len(jc) - 1):
                for k in range(jc[j], jc[j + 1]):
                    dense[ir[k], j] = data[k]
            return ("sparse", cls, dense)
        fields = [b"".join(x).decode() for x in obj.attrs["MATLAB_fields"]]
        first = obj[fields[0]] if fields else None
        if isinstance(first, h5py.Dataset) and first.dtype == h5py.ref_dtype and "MATLAB_class" not in first.attrs:
            shape = first.shape[::-1]
            elems = np.empty(shape, dtype=object)
            cols = {k: obj[k][()].T for k in fields}
            for idx in np.ndindex(shape):
                elems[idx] = {k: mat73_load(f, f[cols[k][idx]]) for k in fields}
            return ("structarray", elems)
        return {k: mat73_load(f, obj[k]) for k in fields}
    if obj.attrs.get("MATLAB_empty", 0):
        return ("empty", cls, tuple(obj[()].tolist()))
    data = obj[()]
    if cls == "cell":
        cells = data.T
        out = np.empty(cells.shape, dtype=object)
        for idx in np.ndindex(cells.shape):
            out[idx] = mat73_load(f, f[cells[idx]])
        return out
    if cls == "char":
        a = data.T
        return ["".join(chr(c) for c in row) for row in np.atleast_2d(a)]
    if data.dtype.names:
        data = data["real"] + 1j * data["imag"]
    if cls == "logical":
        return data.T.astype(bool)
    return data.T


def check_mat(work):
    print("\n=== 2. MatFile.Save(format: \"7.3\") writes, h5py reads (MATLAB layout) ===")
    for variant in ("write-mat", "write-matz"):
        path = os.path.join(work, variant + ".mat")
        run_csharp(variant, path)
        head = open(path, "rb").read(128)
        with h5py.File(path, "r") as f:
            tag = f"[{variant}]"
            check(f"{tag} opens with a 512-byte user block", f.userblock_size == 512)
            check(f"{tag} MAT header text and version 0x0200 'IM'", head.startswith(b"MATLAB 7.3 MAT-file") and head[124:128] == b"\x00\x02IM")
            check(f"{tag} every object reads", not read_everything(f), read_everything(f))
            for name in f:
                if not name.startswith("#"):
                    a = f[name].attrs.get_id("MATLAB_class")
                    t = a.get_type()
                    ok = isinstance(t, h5py.h5t.TypeStringID) and t.get_strpad() == h5py.h5t.STR_NULLTERM and t.get_size() == len(f[name].attrs["MATLAB_class"])
                    if not ok:
                        check(f"{tag} MATLAB_class of {name} is a NUL-terminated fixed string", False)
            v = {k: mat73_load(f, f[k]) for k in f if not k.startswith("#")}
            check(f"{tag} dbl stored transposed", f["dbl"].shape == (4, 3) and np.array_equal(v["dbl"], np.arange(12.0).reshape(3, 4) * 1.5 - 2))
            check(f"{tag} 3-D", np.array_equal(v["nd"], np.arange(24.0).reshape(2, 3, 4)))
            check(f"{tag} single", f["sgl"].dtype == np.float32 and v["sgl"].tolist() == [[1.5, -2.25, 3.0], [4.0, 5.5, -6.75]])
            check(f"{tag} integer classes", f["i8"].dtype == np.int8 and v["i8"].tolist() == [[-128, 0, 127]]
                  and f["u16"].dtype == np.uint16 and f["i32"].attrs["MATLAB_class"] == b"int32"
                  and f["u64"].dtype == np.uint64 and v["u64"].tolist() == [[0, 1 << 63]])
            check(f"{tag} scalar is 1x1", v["scalar"].tolist() == [[3.5]])
            check(f"{tag} complex as {{real, imag}}", f["cplx"].dtype.names == ("real", "imag")
                  and np.array_equal(v["cplx"], np.array([[1 + 2j, -1], [3j, 0.5 - 0.25j]])))
            check(f"{tag} logical as uint8", f["lgc"].dtype == np.uint8 and v["lgc"].tolist() == [[True, False, True], [False, False, True]])
            check(f"{tag} char as UTF-16 with MATLAB_int_decode=2", v["str"] == ["hello"] and v["uni"] == ["héllo € 日本"]
                  and f["str"].attrs["MATLAB_int_decode"] == 2)
            check(f"{tag} char matrix", v["sarr"] == ["abc", "xyz"])
            check(f"{tag} empty arrays", v["emptystr"] == ("empty", "char", (0, 0)) and v["empty"] == ("empty", "double", (0, 3))
                  and v["emptycell"] == ("empty", "cell", (0, 0)))
            st = v["st"]
            check(f"{tag} struct", f["st"].attrs["MATLAB_class"] == b"struct" and st["a"].tolist() == [[1.0]] and st["b"] == ["text"]
                  and st["c"].tolist() == [[1, 2], [3, 4]])
            sa = v["starr"]
            check(f"{tag} struct array via #refs#", sa[0] == "structarray" and sa[1].shape == (1, 2)
                  and sa[1][0, 1]["y"] == ["second"] and sa[1][0, 1]["x"].tolist() == [[2.0, 3.0]])
            cell = v["cell"]
            check(f"{tag} cell via #refs#", cell.shape == (1, 3) and cell[0, 0].tolist() == [[1.0]] and cell[0, 1] == ["two"])
            nest = v["nest"]
            check(f"{tag} nested cell/struct", nest[0, 0]["q"][0, 1].tolist() == [[-7.0]] and nest[0, 1][0, 0] == ["inner"])
            sp = v["sp"]
            check(f"{tag} sparse", sp[1] == "double" and sp[2].tolist() == [[1, 0, 0, 2.5, 0], [0, 0, -3, 0, 0], [0, 4, 0, 0, 0], [0, 0, 0, 0, 5.5]])
            check(f"{tag} logical sparse", v["splog"][1] == "logical" and v["splog"][2].astype(bool).tolist() == [[True, False], [False, True], [True, False]])
            check(f"{tag} complex sparse", v["spc"][2].tolist() == [[1 + 2j, 0, 0], [0, 0, -1j], [3, 0, 0]])
            check(f"{tag} big", np.allclose(v["big"], np.sin(np.arange(6000) * 0.01).reshape(200, 30), rtol=0, atol=0))
            if variant == "write-matz":
                check(f"{tag} large datasets deflated, small ones not", f["big"].compression == "gzip" and f["dbl"].compression is None)
        out = run_csharp("dump-mat", path)
        check(f"[{variant}] SciNet reads it back", "starr=struct(1x2){0.x=double(1x1)[1];0.y=char(1x5)[first]" in out, out[:300])


# ---------------------------------------------------------------- 3. NetCDF-4

def check_netcdf(work):
    print("\n=== 3. NetCdfFile.Create(version: 4) writes, netCDF4-python reads ===")
    try:
        import netCDF4
    except ImportError:
        print("  SKIP  netCDF4 not installed")
        return
    path = os.path.join(work, "written.nc")
    run_csharp("write-nc", path)
    with netCDF4.Dataset(path) as ds:
        check("opens as NETCDF4", ds.data_model == "NETCDF4")
        dims = {k: (len(d), d.isunlimited()) for k, d in ds.dimensions.items()}
        check("dimensions, in order, with the unlimited one", list(dims.items()) == [("time", (2, True)), ("lat", (3, False)), ("lon", (4, False)), ("strlen", (5, False))], dims)
        check("global attributes", ds.title == "SciNet NetCDF-4 writer" and ds.version == 3 and ds.coeffs.tolist() == [1.5, -2.0] and list(ds.tags) == ["a", "bc"])
        check("variables in order", list(ds.variables) == ["time", "lat", "temp", "scalar", "name", "flags", "ub", "u4", "i8", "words", "nodata"], list(ds.variables))
        v = ds.variables
        check("coordinate variables", v["time"][:].tolist() == [0.0, 6.0] and v["time"].units == "hours since 2000-01-01" and v["lat"][:].tolist() == [-10, 0, 10])
        check("dimensions attached by DIMENSION_LIST", v["temp"].dimensions == ("time", "lat", "lon") and v["name"].dimensions == ("lat", "strlen"))
        check("_FillValue typed as the variable", v["temp"]._FillValue == -999 and v["temp"].getncattr("_FillValue").dtype == np.int16)
        temp = v["temp"][:]
        check("mask and scale through netCDF", np.ma.is_masked(temp[1, 2, 3]) and abs(float(temp[0, 0, 0]) - (-3.0)) < 1e-6 and v["temp"].long_name == "température")
        v["temp"].set_auto_maskandscale(False)
        raw = v["temp"][:].ravel().tolist()
        check("raw short data", raw == [i * 37 - 300 if i != 23 else -999 for i in range(24)])
        check("scalar", v["scalar"][()] == 42.5 and v["scalar"].dimensions == ())
        check("char variable", netCDF4.chartostring(v["name"][:]).tolist() == ["south", "eq", "north"])
        check("byte / ubyte / uint / int64", v["flags"][:].tolist() == [-1, 0, 1, 127] and v["ub"][:].data.tolist() == [0, 1, 254, 255]
              and v["u4"][:].data.tolist() == [0, 7, 4294967295] and v["i8"][:].tolist() == [-2**53, 0, 2**53]
              and v["ub"].dtype == np.uint8 and v["u4"].dtype == np.uint32 and v["i8"].dtype == np.int64)
        check("NC_STRING variable", v["words"].dtype == str and list(v["words"][:]) == ["un", "deux", "trois"])
        check("unset variable reads as the default fill", np.ma.getmaskarray(v["nodata"][:]).all())
    with h5py.File(path, "r") as f:
        check("h5py sees dimension scales", h5py.h5ds.is_scale(f["time"].id) and h5py.h5ds.is_scale(f["lon"].id)
              and f["temp"].dims[0][0].name == "/time" and f["temp"].dims[2][0].name == "/lon")
        check("dimension without a variable", f["lon"].attrs["NAME"].startswith(b"This is a netCDF dimension but not a netCDF variable.")
              and f["lon"].attrs["_Netcdf4Dimid"] == 2)

    # netCDF-C appends to the file, then SciNet reads it.
    mod = os.path.join(work, "appended.nc")
    shutil.copy(path, mod)
    with netCDF4.Dataset(mod, "a") as ds:
        ds.history = "appended by netCDF-C"
        x = ds.createVariable("added", "f8", ("lat", "lon"))
        x[:] = np.arange(12.0).reshape(3, 4)
        ds.variables["time"][2] = 12.0
    with netCDF4.Dataset(mod) as ds:
        check("netCDF-C can append to the file", len(ds.dimensions["time"]) == 3 and ds.variables["added"][:].sum() == 66)
    out = run_csharp("dump-nc", mod)
    check("SciNet reads the netCDF-C-modified file", "added:Double(lat,lon)=[0,1,2,3,4,5,6,7,8,9,10,11]" in out
          and "time:Double(time)=[0,6,12]" in out and "@history=appended by netCDF-C" in out, out[:400])


def main():
    build()
    work = tempfile.mkdtemp(prefix="hdf5-interop-")
    try:
        check_generic(work)
        check_mat(work)
        check_netcdf(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All HDF5 interop checks passed.")


if __name__ == "__main__":
    main()
