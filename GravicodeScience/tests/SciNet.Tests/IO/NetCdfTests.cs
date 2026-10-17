using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// NetCDF classic tests. nc_v1.nc / nc_v2.nc were written by scipy 1.17.1 netcdf_file
/// (gen_fixtures.py); scipy/example_*.nc come from SciPy's test suite (BSD-3-Clause). Expected
/// values are scipy 1.17.1's netcdf_file output. Files written by SciNet (both versions, record
/// and fixed variables, attributes, scalar and char variables) were read back by scipy during
/// development and matched exactly.
/// </summary>
public class NetCdfTests
{
    [Theory]
    [InlineData("nc_v1.nc", 1)]
    [InlineData("nc_v2.nc", 2)]
    public void ReadsScipyWrittenFiles(string file, int version)
    {
        using var f = NetCdfFile.Open(Fixture(file));
        Assert.Equal(version, f.Version);
        Assert.Equal(4, f.RecordCount);
        Assert.Null(f.Dimensions["time"]);
        Assert.Equal(3, f.Dimensions["x"]);
        Assert.Equal("Created for SciNet tests", f.Attributes["history"]);
        Assert.Equal(3, f.Attributes["version"]);
        Assert.Equal([0.5, 1.5, -2.0], (double[])f.Attributes["coeffs"]);

        var time = f.Variables["time"];
        Assert.True(time.IsRecord);
        Assert.Equal("days since 2000-01-01", time.Attributes["units"]);
        Close([0, 0.5, 1.0, 1.5], time.GetData().ToArray(), 0);

        var temp = f.Variables["temp"];
        Assert.Equal(NetCdfType.Float, temp.Type);
        Assert.Equal([4, 3], temp.Shape);
        Close(Enumerable.Range(0, 12).Select(i => i + 270.25).ToArray(), temp.GetData().ToArray(), 0);

        // Raw packed shorts, unmasked, as scipy gives them with maskandscale=False.
        Close([100, -999, 250, -32], f.Variables["packed"].GetData().ToArray(), 0);
        Assert.Equal(-999, (short)f.Variables["packed"].Attributes["_FillValue"]);
        Assert.Equal(0.01f, (float)f.Variables["packed"].Attributes["scale_factor"]);
        Close([-90, 0, 90], f.Variables["lat"].GetData().ToArray(), 0);
        Close([-1, 0, 127], f.Variables["flags"].GetData().ToArray(), 0);
        Assert.Equal(["alpha", "beta", "gamma"], f.Variables["name"].GetStrings());
        // (No scalar variable in these fixtures: scipy 1.17.1 places a 0-d variable inside the
        // record section and overwrites record 1 of the first record variable. SciNet's writer
        // keeps scalars with the fixed variables; see WriteThenRead_RoundTrips.)
    }

    [Fact]
    public void MaskAndScale_MatchesScipy()
    {
        using (var f = NetCdfFile.Open(Fixture("nc_v2.nc"), maskAndScale: true))
        {
            // scipy 1.17.1 maskandscale=True: [10.999999977648258, --, 12.499999944120646, 9.680000007152557]
            var p = f.Variables["packed"].GetData().ToArray();
            Close(10.999999977648258, p[0], 1e-15);
            Assert.True(double.IsNaN(p[1]));
            Close(12.499999944120646, p[2], 1e-15);
            Close(9.680000007152557, p[3], 1e-15);
        }
        using (var f = NetCdfFile.Open(Fixture(Path.Combine("scipy", "example_2.nc")), maskAndScale: true))
        {
            var t = f.Variables["Temperature"].GetData().ToArray();
            Assert.Equal(15, t.Length);
            Assert.Equal(20.0, t[0]);
            Close(20.709999984130263, t[1], 1e-14);
            Assert.True(double.IsNaN(t[3]));
            Close(29.999999776482582, t[14], 1e-14);
        }
        using (var f = NetCdfFile.Open(Fixture(Path.Combine("scipy", "example_3_maskedvals.nc")), maskAndScale: true))
        {
            static bool[] Nan(NetCdfVariable v) => v.GetData().ToArray().Select(double.IsNaN).ToArray();
            Assert.Equal([false, true, false], Nan(f.Variables["var1_fillval0"])); // 1e-10 is not the fill value 0
            Assert.Equal([false, false, false], Nan(f.Variables["var2_noFillval"]));
            Assert.Equal([true, false, false], Nan(f.Variables["var3_fillvalAndMissingValue"])); // _FillValue wins
            Assert.Equal([false, true, false], Nan(f.Variables["var4_missingValue"]));
            Assert.Equal([false, true, false], Nan(f.Variables["var5_fillvalNaN"]));
            Assert.Equal([false, true, false], Nan(f.Variables["var6_char"]));
            Assert.Equal([true, false, false, false, false, true], Nan(f.Variables["var7_2d"]));
        }
    }

    [Fact]
    public void ReadsScipyExample1()
    {
        using var f = NetCdfFile.Open(Fixture(Path.Combine("scipy", "example_1.nc")));
        Assert.Equal("Fictional Model Output", f.Attributes["source"]);
        Assert.Equal(1, f.RecordCount);
        Assert.Equal([1, 4, 5, 10], f.Variables["temp"].Shape);
        Close([20, 30, 40, 50, 60], f.Variables["lat"].GetData().ToArray(), 0);
        Close([1000, 850, 700, 500], f.Variables["level"].GetData().ToArray(), 0);
        Close([12], f.Variables["time"].GetData().ToArray(), 0);
        var rh = f.Variables["rh"].GetData();
        Close(21.5, rh.ToArray().Sum(), 1e-5); // scipy: rh.data.sum() = 21.5 (float32 data)
        Close(0.8, rh[0, 2, 7], 1e-7);
        Assert.Equal([0.0, 1.0], (double[])f.Variables["rh"].Attributes["valid_range"]);
        // temp was never written: every value is the default float fill, 9.96921e+36
        Assert.All(f.Variables["temp"].GetData().ToArray(), v => Close(9.96921e36, v, 1e31));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void WriteThenRead_RoundTrips(int version)
    {
        var path = TempFile(".nc");
        try
        {
            using (var f = NetCdfFile.Create(version))
            {
                f.Attributes["title"] = "round trip";
                f.Attributes["ints"] = new[] { 1, 2, 3 };
                f.Attributes["one"] = 1.5f;
                f.CreateDimension("t", null);
                f.CreateDimension("y", 2);
                f.CreateDimension("len", 4);
                var t = f.CreateVariable("t", NetCdfType.Double, "t");
                t.SetData(new NdArray([0.0, 1, 2], 3));
                var b = f.CreateVariable("b", NetCdfType.Byte, "t"); // byte record variable: per-record padding
                b.SetData(new NdArray([-128.0, 0, 127], 3));
                var s = f.CreateVariable("s", NetCdfType.Short, "t", "y");
                s.SetData(new NdArray([1.0, -2, 3, -4, 5, -6], 3, 2));
                s.Attributes["units"] = "m";
                var fixedVar = f.CreateVariable("grid", NetCdfType.Float, "y", "len");
                fixedVar.SetData(new NdArray([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], 2, 4));
                var names = f.CreateVariable("names", NetCdfType.Char, "y", "len");
                names.SetStrings(["ab", "cdef"]);
                var scalar = f.CreateVariable("k", NetCdfType.Int);
                scalar.SetData(new NdArray([42.0], 1));
                f.CreateVariable("unset", NetCdfType.Short, "y");
                Assert.Throws<ArgumentException>(() => b.SetData(new NdArray([300.0], 1)));
                f.Save(path);
            }
            using var g = NetCdfFile.Open(path);
            Assert.Equal(version, g.Version);
            Assert.Equal(3, g.RecordCount);
            Assert.Equal("round trip", g.Attributes["title"]);
            Assert.Equal([1, 2, 3], (int[])g.Attributes["ints"]);
            Assert.Equal(1.5f, (float)g.Attributes["one"]);
            Close([0, 1, 2], g.Variables["t"].GetData().ToArray(), 0);
            Close([-128, 0, 127], g.Variables["b"].GetData().ToArray(), 0);
            Close([1, -2, 3, -4, 5, -6], g.Variables["s"].GetData().ToArray(), 0);
            Assert.Equal("m", g.Variables["s"].Attributes["units"]);
            Close([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], g.Variables["grid"].GetData().ToArray(), 0);
            Assert.Equal(["ab", "cdef"], g.Variables["names"].GetStrings());
            Assert.Equal(42.0, g.Variables["k"].GetValue());
            Close([-32767, -32767], g.Variables["unset"].GetData().ToArray(), 0); // NC_FILL_SHORT

            // Copy an opened file: the reader keeps its own offsets.
            var copy = TempFile(".nc");
            try
            {
                g.Save(copy);
                using var h = NetCdfFile.Open(copy);
                Close([1, -2, 3, -4, 5, -6], h.Variables["s"].GetData().ToArray(), 0);
                Assert.Equal(["ab", "cdef"], h.Variables["names"].GetStrings());
            }
            finally { File.Delete(copy); }
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void SingleRecordVariable_HasNoRecordPadding()
    {
        // NetCDF spec special case: with exactly one record variable of a sub-4-byte type, records are packed.
        var path = TempFile(".nc");
        try
        {
            using (var f = NetCdfFile.Create())
            {
                f.CreateDimension("t", null);
                f.CreateVariable("c", NetCdfType.Byte, "t").SetData(new NdArray([1.0, 2, 3, 4, 5], 5));
                f.Save(path);
            }
            var bytes = File.ReadAllBytes(path);
            Assert.Equal([1, 2, 3, 4, 5], bytes[^5..]);
            using var g = NetCdfFile.Open(path);
            Close([1, 2, 3, 4, 5], g.Variables["c"].GetData().ToArray(), 0);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void MaskAndScale_Write_PacksAndRounds()
    {
        var path = TempFile(".nc");
        try
        {
            using (var f = NetCdfFile.Create(maskAndScale: true))
            {
                f.CreateDimension("n", 4);
                var v = f.CreateVariable("p", NetCdfType.Short, "n");
                v.Attributes["scale_factor"] = 0.1;
                v.Attributes["add_offset"] = 5.0;
                v.Attributes["_FillValue"] = (short)-1;
                // 5.3 packs to 2.9999999999999982 — truncation (SciPy) would store 2, rounding stores 3.
                v.SetData(new NdArray([5.0, 5.3, double.NaN, 6.0], 4));
                f.Save(path);
            }
            using (var raw = NetCdfFile.Open(path))
                Close([0, 3, -1, 10], raw.Variables["p"].GetData().ToArray(), 0);
            using var g = NetCdfFile.Open(path, maskAndScale: true);
            var d = g.Variables["p"].GetData().ToArray();
            Close(5.0, d[0], 1e-12);
            Close(5.3, d[1], 1e-12);
            Assert.True(double.IsNaN(d[2]));
            Close(6.0, d[3], 1e-12);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void TruncatedNetcdf4AndGarbage_AreRefused()
    {
        // NetCDF-4 is read now (see NetCdf4Tests); a bare HDF5 signature is a truncated file.
        var hdf = new byte[] { 0x89, (byte)'H', (byte)'D', (byte)'F', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0 };
        var e = Assert.Throws<InvalidDataException>(() => NetCdfFile.Open(new MemoryStream(hdf)));
        Assert.Contains("HDF5", e.Message);
        Assert.Throws<NotSupportedException>(() => NetCdfFile.Open(new MemoryStream("CDF\u0005\0\0\0\0"u8.ToArray())));
        Assert.Throws<InvalidDataException>(() => NetCdfFile.Open(new MemoryStream("hello world"u8.ToArray())));
        var truncated = File.ReadAllBytes(Fixture("nc_v1.nc"))[..40];
        Assert.Throws<InvalidDataException>(() => NetCdfFile.Open(new MemoryStream(truncated)));
    }
}
