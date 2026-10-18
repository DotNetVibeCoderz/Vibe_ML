using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// NetCDF-4 reading. nc4_classic.nc / nc4_enhanced.nc were written by netCDF4-python 1.7.5
/// (netCDF-C 4.10.1, HDF5 2.0.0) from gen_hdf5.py; expected values are what netCDF4-python reads
/// back from them.
/// </summary>
public class NetCdf4Tests
{
    [Fact]
    public void ClassicModel_DimensionsVariablesAttributes()
    {
        using var f = NetCdfFile.Open(Fixture("nc4_classic.nc"));
        Assert.Equal(4, f.Version);
        Assert.Equal("NETCDF4_CLASSIC", f.DataModel);
        Assert.Equal(["time", "lat", "lon", "strlen"], f.Dimensions.Keys);
        Assert.Null(f.Dimensions["time"]);
        Assert.Equal(3, f.Dimensions["lat"]);
        Assert.Equal(2, f.RecordCount);
        Assert.Equal("SciNet NetCDF-4 classic model", f.Attributes["title"]);
        Assert.Equal(3, f.Attributes["version"]);
        Assert.False(f.Attributes.ContainsKey("_NCProperties"));
        Assert.Equal(["time", "lat", "lon", "temp", "scalar", "name", "flags"], f.Variables.Keys);
        Assert.Empty(f.Groups);

        var time = f.Variables["time"];
        Assert.Equal(NetCdfType.Double, time.Type);
        Assert.True(time.IsRecord);
        Assert.Equal("hours since 2000-01-01", time.Attributes["units"]);
        Assert.Equal(new[] { 0.0, 6.0 }, time.GetData().ToArray());
        Assert.Equal(NetCdfType.Float, f.Variables["lat"].Type);
        Assert.Equal(new[] { -10.0, 0, 10 }, f.Variables["lat"].GetData().ToArray());

        var temp = f.Variables["temp"];
        Assert.Equal(NetCdfType.Short, temp.Type);
        Assert.Equal(["time", "lat", "lon"], temp.Dimensions);
        Assert.Equal(new[] { 2, 3, 4 }, temp.Shape);
        Assert.Equal((short)-999, temp.Attributes["_FillValue"]);
        Assert.Equal(0.01f, temp.Attributes["scale_factor"]);
        var raw = temp.GetData().ToArray();
        for (var i = 0; i < 24; i++) Assert.Equal(i == 23 ? -999 : i * 37 - 300, raw[i]);

        var sc = f.Variables["scalar"];
        Assert.Empty(sc.Shape);
        Assert.Equal(42.5, sc.GetValue());
        var name = f.Variables["name"];
        Assert.Equal(NetCdfType.Char, name.Type);
        Assert.Equal(["south", "eq", "north"], name.GetStrings());
        Assert.Equal(new[] { -1.0, 0, 1, 127 }, f.Variables["flags"].GetData().ToArray());
        Assert.Throws<NotSupportedException>(() => f.Save(new MemoryStream()));
        Assert.Throws<InvalidOperationException>(() => temp.SetData(temp.GetData()));
    }

    [Fact]
    public void ClassicModel_MaskAndScale()
    {
        using var f = NetCdfFile.Open(Fixture("nc4_classic.nc"), maskAndScale: true);
        var t = f.Variables["temp"].GetData().ToArray();
        // netCDF4-python (float32 arithmetic): 270.1499938964844, 270.5199890136719, 270.8899841308594, 271.2599792480469
        double[] expected = [270.1499938964844, 270.5199890136719, 270.8899841308594, 271.2599792480469];
        Close(expected, t[..4], 1e-4, "temp");
        for (var i = 0; i < 23; i++) Close((i * 37 - 300) * 0.01 + 273.15, t[i], 1e-4, $"temp[{i}]");
        Assert.True(double.IsNaN(t[23]), "the fill value is masked");
    }

    [Fact]
    public void Enhanced_TypesStringsGroups()
    {
        using var f = NetCdfFile.Open(Fixture("nc4_enhanced.nc"));
        Assert.Equal("NETCDF4", f.DataModel);
        Assert.Equal("created for SciNet tests", f.Attributes["history"]);
        Assert.Equal(new[] { "a", "bb" }, (string[])f.Attributes["tags"]);
        Assert.Equal(NetCdfType.UShort, f.Variables["u16"].Type);
        Assert.Equal(new[] { 0.0, 1, 65535, 300, 7 }, f.Variables["u16"].GetData().ToArray());
        Assert.Equal(NetCdfType.Int64, f.Variables["i64"].Type);
        Assert.Equal(new[] { -1099511627776.0, 0, 1, 4503599627370496, -5 }, f.Variables["i64"].GetData().ToArray());
        Assert.Equal(NetCdfType.UInt64, f.Variables["u64"].Type);
        Assert.Equal(9223372036854775808.0, f.Variables["u64"].GetData().ToArray()[1]);
        Assert.Equal(NetCdfType.UByte, f.Variables["u8"].Type);
        Assert.Equal(new[] { 0.0, 255, 128, 1, 2 }, f.Variables["u8"].GetData().ToArray());
        Assert.Equal(NetCdfType.UInt, f.Variables["u32"].Type);
        Assert.Equal(4294967295.0, f.Variables["u32"].GetData().ToArray()[1]);

        var words = f.Variables["words"];
        Assert.Equal(NetCdfType.String, words.Type);
        Assert.Equal(["one", "two", "über", "", "five"], words.GetStrings());
        Assert.Throws<InvalidOperationException>(() => words.GetData());

        var series = f.Variables["series"];
        Assert.True(series.IsRecord);
        Assert.Equal(new[] { 3, 5 }, series.Shape);
        Assert.Equal(Enumerable.Range(0, 15).Select(i => i + 0.25).ToArray(), series.GetData().ToArray());
        Assert.Equal(3, f.RecordCount);

        var fc = f.Groups["forecast"];
        Assert.Equal("toy", fc.Attributes["model"]);
        Assert.Equal(["step"], fc.Dimensions.Keys);
        var wind = fc.Variables["wind"];
        Assert.Equal(["step", "x"], wind.Dimensions); // x lives in the parent group
        Assert.Equal(new[] { 0.0, 1.5, 3, 4.5, 6, -1e30, -1e30, -1e30, -1e30, -1e30 }, wind.GetData().ToArray());
        Assert.Equal(11.0, fc.Groups["inner"].Variables["k"].GetValue());
    }

    [Fact]
    public void Enhanced_FillValueMaskedInGroups()
    {
        using var f = NetCdfFile.Open(new MemoryStream(File.ReadAllBytes(Fixture("nc4_enhanced.nc"))), maskAndScale: true);
        var w = f.Groups["forecast"].Variables["wind"].GetData().ToArray();
        Assert.Equal(new[] { 0.0, 1.5, 3, 4.5, 6 }, w[..5]);
        Assert.All(w[5..], x => Assert.True(double.IsNaN(x)));
    }

    // ------------------------------------------------------------------ writing (NetCdfFile.Create(version: 4))

    private static byte[] Written(NetCdfFile f)
    {
        var ms = new MemoryStream();
        f.Save(ms);
        return ms.ToArray();
    }

    private static NetCdfFile Sample(bool maskAndScale = false)
    {
        var f = NetCdfFile.Create(version: 4, maskAndScale: maskAndScale);
        f.Attributes["title"] = "round trip";
        f.Attributes["ints"] = new[] { 1, 2, 3 };
        f.Attributes["one"] = 1.5f;
        f.Attributes["tags"] = new[] { "a", "bc" };
        f.CreateDimension("t", null);
        f.CreateDimension("y", 2);
        f.CreateDimension("len", 4);
        var t = f.CreateVariable("t", NetCdfType.Double, "t");
        t.Attributes["units"] = "days";
        t.SetData(new NdArray([0.0, 1, 2], 3));
        f.CreateVariable("b", NetCdfType.Byte, "t").SetData(new NdArray([-128.0, 127], 2)); // one record short: padded with fill
        var s = f.CreateVariable("s", NetCdfType.Short, "t", "y");
        s.Attributes["_FillValue"] = -1.0; // a double: converted to short, as netCDF requires
        s.SetData(new NdArray([1.0, -2, 3, -4, 5, -6], 3, 2));
        f.CreateVariable("grid", NetCdfType.Float, "y", "len").SetData(new NdArray([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], 2, 4));
        f.CreateVariable("names", NetCdfType.Char, "y", "len").SetStrings(["ab", "cdef"]);
        f.CreateVariable("k", NetCdfType.Int).SetData(new NdArray([42.0], 1));
        f.CreateVariable("ub", NetCdfType.UByte, "y").SetData(new NdArray([0.0, 255], 2));
        f.CreateVariable("us", NetCdfType.UShort, "y").SetData(new NdArray([0.0, 65535], 2));
        f.CreateVariable("ui", NetCdfType.UInt, "y").SetData(new NdArray([7.0, 4294967295], 2));
        f.CreateVariable("i8", NetCdfType.Int64, "y").SetData(new NdArray([-9007199254740992.0, 9007199254740992], 2));
        f.CreateVariable("u8", NetCdfType.UInt64, "y").SetData(new NdArray([0.0, 18446744073709549568], 2));
        f.CreateVariable("words", NetCdfType.String, "t").SetStrings(["un", "deux", "trois"]);
        f.CreateVariable("unset", NetCdfType.Short, "y");
        return f;
    }

    [Fact]
    public void Create4_WriteThenRead_RoundTrips()
    {
        byte[] bytes;
        using (var w = Sample()) bytes = Written(w);
        using var f = NetCdfFile.Open(new MemoryStream(bytes));
        Assert.Equal(4, f.Version);
        Assert.Equal("NETCDF4", f.DataModel);
        Assert.Equal(["t", "y", "len"], f.Dimensions.Keys); // _Netcdf4Dimid order
        Assert.Null(f.Dimensions["t"]);
        Assert.Equal(2, f.Dimensions["y"]);
        Assert.Equal(3, f.RecordCount);
        Assert.Equal(["t", "b", "s", "grid", "names", "k", "ub", "us", "ui", "i8", "u8", "words", "unset"], f.Variables.Keys);
        Assert.Equal("round trip", f.Attributes["title"]);
        Assert.Equal(new[] { 1, 2, 3 }, (int[])f.Attributes["ints"]);
        Assert.Equal(1.5f, f.Attributes["one"]);
        Assert.Equal(new[] { "a", "bc" }, (string[])f.Attributes["tags"]);

        var t = f.Variables["t"];
        Assert.True(t.IsRecord);
        Assert.Equal("days", t.Attributes["units"]);
        Assert.Equal(new[] { 0.0, 1, 2 }, t.GetData().ToArray());
        var b = f.Variables["b"];
        Assert.Equal(NetCdfType.Byte, b.Type);
        Assert.Equal(new[] { -128.0, 127, -127 }, b.GetData().ToArray()); // the missing record is the default byte fill
        var s = f.Variables["s"];
        Assert.Equal(["t", "y"], s.Dimensions);
        Assert.Equal((short)-1, s.Attributes["_FillValue"]);
        Assert.Equal(new[] { 1.0, -2, 3, -4, 5, -6 }, s.GetData().ToArray());
        Assert.Equal(new[] { 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4 }, f.Variables["grid"].GetData().ToArray());
        Assert.Equal(NetCdfType.Float, f.Variables["grid"].Type);
        Assert.Equal(["ab", "cdef"], f.Variables["names"].GetStrings());
        Assert.Equal(NetCdfType.Char, f.Variables["names"].Type);
        Assert.Empty(f.Variables["k"].Shape);
        Assert.Equal(42.0, f.Variables["k"].GetValue());
        Assert.Equal(NetCdfType.UByte, f.Variables["ub"].Type);
        Assert.Equal(new[] { 0.0, 255 }, f.Variables["ub"].GetData().ToArray());
        Assert.Equal(new[] { 0.0, 65535 }, f.Variables["us"].GetData().ToArray());
        Assert.Equal(new[] { 7.0, 4294967295 }, f.Variables["ui"].GetData().ToArray());
        Assert.Equal(NetCdfType.Int64, f.Variables["i8"].Type);
        Assert.Equal(new[] { -9007199254740992.0, 9007199254740992 }, f.Variables["i8"].GetData().ToArray());
        Assert.Equal(NetCdfType.UInt64, f.Variables["u8"].Type);
        Assert.Equal(new[] { 0.0, 18446744073709549568 }, f.Variables["u8"].GetData().ToArray());
        Assert.Equal(["un", "deux", "trois"], f.Variables["words"].GetStrings());
        Assert.Equal(new[] { -32767.0, -32767 }, f.Variables["unset"].GetData().ToArray());
    }

    [Fact]
    public void Create4_DimensionScalesAreNetCdfCs()
    {
        byte[] bytes;
        using (var w = Sample()) bytes = Written(w);
        using var h5 = Hdf5File.Open(new MemoryStream(bytes));
        Assert.Equal(["y", "len", "t", "b", "s", "grid", "names", "k", "ub", "us", "ui", "i8", "u8", "words", "unset"], h5.Root.Keys); // creation order: dimension-only scales first, as netCDF-C defines dimensions before variables
        var t = h5.Root.Dataset("t");
        var cls = t.Attributes["CLASS"];
        Assert.Equal("DIMENSION_SCALE", cls.Value);
        Assert.Equal(16, cls.DataType.Size); // NUL-terminated, as H5DSset_scale writes it (H5DSis_scale insists on NULLTERM)
        Assert.Equal(0, cls.DataType.StringPadding);
        Assert.Equal("t", t.Attributes["NAME"].Value);
        Assert.Equal(0, t.GetAttribute("_Netcdf4Dimid"));
        Assert.Equal(new long[] { -1 }, t.MaxShape);
        Assert.Equal(Hdf5Layout.Chunked, t.Layout);
        Assert.False(t.Attributes.ContainsKey("DIMENSION_LIST")); // a coordinate variable is its own scale

        var len = h5.Root.Dataset("len");
        Assert.Equal("This is a netCDF dimension but not a netCDF variable.         4", len.Attributes["NAME"].Value);
        Assert.Equal(2, len.GetAttribute("_Netcdf4Dimid"));
        Assert.Equal(new[] { 4 }, len.Shape);
        Assert.Equal(new[] { 0.0, 0, 0, 0 }, len.Read().ToArray()); // never allocated: reads as fill

        var s = h5.Root.Dataset("s");
        var dl = (Array[])s.GetAttribute("DIMENSION_LIST")!;
        Assert.Equal(2, dl.Length);
        Assert.Equal(t.Address, ((Hdf5Reference[])dl[0]).Single().Address);
        Assert.Equal(h5.Root.Dataset("y").Address, ((Hdf5Reference[])dl[1]).Single().Address);
        Assert.Equal(new long[] { -1, 2 }, s.MaxShape);
        Assert.Equal(new short[] { 1, -2, 3, -4, 5, -6 }, s.Read<short>());

        // REFERENCE_LIST on the record dimension: every (variable, axis) that uses it.
        var rl = t.Dataset_ReferenceList();
        Assert.Equal(["b", "s", "words"], rl.Select(x => h5.Dereference(x.Ref).Address).Select(a => h5.Root.Members.First(m => m.Address == a).Name));
        Assert.All(rl, x => Assert.Equal(0u, x.Axis));
    }

    [Fact]
    public void Create4_SixtyFourBitDefaultFillsAreExact()
    {
        // NC_FILL_INT64 / NC_FILL_UINT64 are not representable as doubles; unset elements get the exact values.
        byte[] bytes;
        using (var w = NetCdfFile.Create(version: 4))
        {
            w.CreateDimension("t", null);
            w.CreateVariable("a", NetCdfType.Int64, "t").SetData(new NdArray([5.0], 1));
            w.CreateVariable("b", NetCdfType.UInt64, "t").SetData(new NdArray([18446744073709549568, 1, 2], 3));
            bytes = Written(w);
        }
        using var h5 = Hdf5File.Open(new MemoryStream(bytes));
        Assert.Equal(new[] { 5L, -9223372036854775806L, -9223372036854775806L }, h5.Root.Dataset("a").Read<long>());
        Assert.Equal(new ulong[] { 18446744073709549568UL, 1, 2 }, h5.Root.Dataset("b").Read<ulong>()); // a real value is never mistaken for fill
    }

    [Fact]
    public void Create4_MaskAndScaleOnWrite()
    {
        byte[] bytes;
        using (var w = NetCdfFile.Create(version: 4, maskAndScale: true))
        {
            w.CreateDimension("x", 3);
            var v = w.CreateVariable("v", NetCdfType.Short, "x");
            v.Attributes["_FillValue"] = (short)-999;
            v.Attributes["scale_factor"] = 0.5;
            v.SetData(new NdArray([1.0, double.NaN, -2.5], 3));
            bytes = Written(w);
        }
        using (var raw = NetCdfFile.Open(new MemoryStream(bytes)))
            Assert.Equal(new[] { 2.0, -999, -5 }, raw.Variables["v"].GetData().ToArray());
        using var f = NetCdfFile.Open(new MemoryStream(bytes), maskAndScale: true);
        var d = f.Variables["v"].GetData().ToArray();
        Assert.Equal(1.0, d[0]);
        Assert.True(double.IsNaN(d[1]));
        Assert.Equal(-2.5, d[2]);
    }

    [Fact]
    public void Create4_Refusals()
    {
        Assert.Throws<ArgumentException>(() => NetCdfFile.Create(version: 3));
        using (var classic = NetCdfFile.Create(version: 2))
        {
            classic.CreateDimension("x", 2);
            Assert.Throws<ArgumentException>(() => classic.CreateVariable("u", NetCdfType.UInt64, "x"));
            Assert.Throws<ArgumentException>(() => classic.CreateVariable("s", NetCdfType.String, "x"));
        }
        using var f = NetCdfFile.Create(version: 4);
        f.CreateDimension("x", 2);
        var s = f.CreateVariable("s", NetCdfType.String, "x");
        Assert.Throws<InvalidOperationException>(() => s.SetData(new NdArray([1.0, 2], 2)));
        Assert.Throws<ArgumentException>(() => s.SetStrings(["only one"]));
        Assert.Throws<ArgumentException>(() => f.CreateVariable("u", NetCdfType.UByte, "x").SetData(new NdArray([256.0, 0], 2)));
        Assert.Throws<ArgumentException>(() => f.CreateVariable("q", NetCdfType.UInt64, "x").SetData(new NdArray([18446744073709551616.0, 0], 2)));
        // A 2-D variable named like a dimension: netCDF-C would rename the dimension's scale; this writer refuses.
        f.CreateDimension("y", 2);
        f.CreateVariable("x2", NetCdfType.Double, "x");
        f.CreateVariable("y", NetCdfType.Double, "x", "y");
        Assert.Throws<NotSupportedException>(() => f.Save(new MemoryStream()));
    }
}

internal static class NetCdf4TestExtensions
{
    /// <summary>A dimension scale's REFERENCE_LIST records: (dataset reference, axis).</summary>
    public static (Hdf5Reference Ref, uint Axis)[] Dataset_ReferenceList(this Hdf5Dataset scale)
    {
        var a = scale.Attributes["REFERENCE_LIST"];
        var records = (Dictionary<string, object?>[])a.Value!;
        return records.Select(r => ((Hdf5Reference)r["dataset"]!, (uint)r["dimension"]!)).ToArray();
    }
}
