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
}
