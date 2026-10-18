using System.Globalization;
using System.Numerics;
using System.Text;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;

// The C# half of tools/verify/hdf5_interop.py. It writes the files h5py / netCDF4-python will
// check, and reads back the files Python has modified or written; the Python side does the
// asserting. Kept out of the test project because the check needs a Python toolchain.

Console.OutputEncoding = Encoding.UTF8;
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: Hdf5Interop <write-h5|write-mat|write-nc|dump-h5|dump-mat|dump-nc> <path>");
    return 1;
}

switch (args[0])
{
    case "write-h5":
        Samples.Hdf5().Save(args[1]);
        break;
    case "write-mat":
        MatFile.Save(args[1], Samples.Mat(), format: "7.3");
        break;
    case "write-matz":
        MatFile.Save(args[1], Samples.Mat(), doCompression: true, format: "7.3");
        break;
    case "write-nc":
        Samples.NetCdf().Save(args[1]);
        break;
    case "dump-h5":
    {
        using var f = Hdf5File.Open(args[1]);
        DumpGroup(f.Root);
        break;
    }
    case "dump-mat":
        foreach (var (name, v) in MatFile.Load(args[1])) Console.WriteLine($"{name}={DumpMat(v)}");
        break;
    case "dump-nc":
    {
        using var f = NetCdfFile.Open(args[1]);
        Console.WriteLine($"dims={string.Join(",", f.Dimensions.Select(d => $"{d.Key}:{d.Value?.ToString() ?? "unlimited"}"))}");
        foreach (var (k, a) in f.Attributes) Console.WriteLine($"@{k}={Value(a)}");
        foreach (var v in f.Variables.Values)
        {
            var data = v.Type is NetCdfType.String or NetCdfType.Char ? string.Join("|", v.GetStrings()) : string.Join(",", v.GetData().ToArray().Select(Num));
            Console.WriteLine($"{v.Name}:{v.Type}({string.Join(",", v.Dimensions)})=[{data}]");
            foreach (var (k, a) in v.Attributes) Console.WriteLine($"  @{k}={Value(a)}");
        }
        break;
    }
    default:
        Console.Error.WriteLine($"unknown command {args[0]}");
        return 1;
}
return 0;

static string Num(double x) => x.ToString("R", CultureInfo.InvariantCulture);

static string Value(object? v) => v switch
{
    null => "null",
    string s => s,
    Hdf5Reference r => r.IsNull ? "ref(null)" : $"ref({r.Address})",
    Array a => "[" + string.Join(",", a.Cast<object?>().Select(Value)) + "]",
    IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
    _ => v.ToString() ?? "",
};

static void DumpGroup(Hdf5Group g)
{
    Console.WriteLine($"G {g.Path} [{string.Join(",", g.Keys)}]");
    foreach (var a in g.Attributes.Values) Console.WriteLine($"  @{a.Name}={Value(a.Value)}");
    foreach (var m in g.Members)
    {
        if (m is Hdf5Group sub) DumpGroup(sub);
        else if (m is Hdf5Dataset d)
        {
            string body;
            try
            {
                body = d.DataType.Class == Hdf5TypeClass.Compound
                    ? string.Join(";", d.ReadCompound().Select(kv => kv.Key + ":" + Value(kv.Value)))
                    : Value(d.ReadArray());
            }
            catch (Exception e)
            {
                body = "error " + e.Message;
            }
            Console.WriteLine($"D {d.Path} {d.DataType} ({string.Join(",", d.Shape)}) {d.Layout} {body}");
            foreach (var a in d.Attributes.Values) Console.WriteLine($"  @{a.Name}={Value(a.Value)}");
        }
    }
}

static string DumpMat(MatValue v) => v switch
{
    MatArray a => $"{a.ClassName}({string.Join("x", a.Dimensions)})[{string.Join(",", a.IsComplex ? a.ToComplex().ToArray().Select(z => $"{Num(z.Real)}{(z.Imaginary < 0 ? "" : "+")}{Num(z.Imaginary)}i") : a.Real.ToArray().Select(Num))}]",
    MatLogical l => $"logical({string.Join("x", l.Dimensions)})[{string.Join(",", l.ToBoolArray().Select(b => b ? 1 : 0))}]",
    MatChar c => $"char({string.Join("x", c.Dimensions)})[{string.Join("|", c.Strings)}]",
    MatSparse s => $"sparse({s.Matrix.Rows}x{s.Matrix.Columns},{(s.IsLogical ? "logical" : s.IsComplex ? "complex" : "double")})[{string.Join(",", s.Matrix.Data.ToArray().Select(Num))}]",
    MatCell c => $"cell({string.Join("x", c.Dimensions)}){{{string.Join(";", c.Select(DumpMat))}}}",
    MatStruct s => $"{s.ClassName}({string.Join("x", s.Dimensions)}){{{string.Join(";", Enumerable.Range(0, s.Count).SelectMany(e => s.FieldNames.Select(f => $"{e}.{f}={DumpMat(s[e, f])}")))}}}",
    _ => v.GetType().Name,
};

/// <summary>The reference content; hdf5_interop.py repeats the same formulas.</summary>
internal static class Samples
{
    public static Hdf5Writer Hdf5()
    {
        var w = new Hdf5Writer();
        var root = w.Root;
        var num = root.CreateGroup("num");
        var f8 = num.CreateDataset("f8", new NdArray(Enumerable.Range(0, 24).Select(i => i * 0.5 - 3).ToArray(), 2, 3, 4));
        num.CreateDataset("f4", new[] { 1.5f, -2.25f, 3.0f, 4.0e10f });
        num.CreateDataset("f2", new[] { (Half)0.5, (Half)(-1.25), (Half)65504 });
        num.CreateDataset("i1", new sbyte[] { -128, -1, 0, 1, 127 });
        num.CreateDataset("u1", new byte[] { 0, 1, 200, 255 });
        num.CreateDataset("i2", new short[,] { { -32768, -2 }, { 3, 32767 } });
        num.CreateDataset("u2", new ushort[] { 0, 1, 65535, 4660 });
        num.CreateDataset("i4", new[] { int.MinValue, 5, 7, int.MaxValue });
        num.CreateDataset("u4", new uint[] { 0, 4294967295, 12345 });
        num.CreateDataset("i8", new[] { -(1L << 62), -1, 1L << 53, 9 });
        num.CreateDataset("u8", new[] { 0UL, 1UL << 63, (1UL << 40) + 1, ulong.MaxValue });
        num.CreateDataset("bool", new[] { true, false, true, true });
        num.CreateDataset("c16", new Complex[,] { { new(1, 2), new(-3, 0.5) }, { new(0, -1), new(4.25, 0) } });
        num.CreateDataset("scalar", 2.5);
        num.CreateDataset("empty", new double[0, 3]);
        num.CreateDataset("reshaped", Enumerable.Range(0, 6).Select(i => (long)i).ToArray(), shape: [3, 2]);

        var str = root.CreateGroup("str");
        str.CreateDataset("vlen", new[] { "alpha", "", "日本語 €", "z" });
        str.CreateDataset("fixed", new[] { "ab", "cde", "" }, fixedLengthStrings: true);
        str.CreateDataset("scalar", "one string");

        var refs = root.CreateGroup("refs");
        refs.CreateDataset("objs", new Hdf5WriterNode[] { f8, str, root });

        var comp = root.CreateGroup("chunked");
        var big = new double[37 * 23];
        for (var k = 0; k < big.Length; k++) big[k] = k % 17 - 8;
        comp.CreateDataset("deflate", big, shape: [37, 23], compressionLevel: 6);
        comp.CreateDataset("ext", Enumerable.Range(0, 15).Select(i => i * 1.0).ToArray(), shape: [5, 3], maxShape: [-1, 3]);
        comp.CreateDataset("extz", new[] { 10, 20, 30, 40 }, compressionLevel: 4, maxShape: [-1]);
        comp.CreateDataset("ext0", new double[0, 2], maxShape: [-1, 2]);

        var many = root.CreateGroup("many");
        for (var i = 0; i < 20; i++) many.CreateDataset($"d{i:00}", (double)i);
        var attrs = many.CreateDataset("attrs", new[] { 1.0 });
        for (var i = 0; i < 12; i++) attrs.SetAttribute($"a{i:00}", i * 10);

        root.CreateGroup("a").CreateGroup("b").CreateGroup("c").CreateDataset("deep", new[] { 42 });
        root.CreateGroup("ünï");

        root.SetAttribute("title", "SciNet HDF5 writer");
        root.SetAttribute("utf", "héllo €");
        root.SetAttribute("fixed", "fixed text", fixedLengthStrings: true);
        root.SetAttribute("version", 3);
        root.SetAttribute("pi", Math.PI);
        root.SetAttribute("arr", new short[] { 1, -2, 3 });
        root.SetAttribute("mat", new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        root.SetAttribute("flag", true);
        root.SetAttribute("z", new Complex(1, -2));
        root.SetAttribute("names", new[] { "x", "yy", "zzz" });
        root.SetAttribute("ref", f8);
        f8.SetAttribute("units", "m/s");
        return w;
    }

    public static Dictionary<string, object?> Mat()
    {
        var spm = new Gravicode.Science.SciNet.Sparse.CscMatrix(
            new[] { 1.0, 4.0, -3.0, 2.5, 5.5 }, new[] { 0, 2, 1, 0, 3 }, new[] { 0, 1, 2, 3, 4, 5 }, 4, 5);
        return new()
        {
            ["dbl"] = new NdArray(Enumerable.Range(0, 12).Select(i => i * 1.5 - 2).ToArray(), 3, 4),
            ["nd"] = new NdArray(Enumerable.Range(0, 24).Select(i => (double)i).ToArray(), 2, 3, 4),
            ["sgl"] = new float[,] { { 1.5f, -2.25f, 3.0f }, { 4.0f, 5.5f, -6.75f } },
            ["i8"] = new sbyte[] { -128, 0, 127 },
            ["u16"] = new ushort[] { 0, 65535 },
            ["i32"] = new[] { int.MinValue, -1, int.MaxValue },
            ["u64"] = new[] { 0UL, 1UL << 63 },
            ["scalar"] = 3.5,
            ["cplx"] = new ComplexNdArray(new Complex[] { new(1, 2), new(-1, 0), new(0, 3), new(0.5, -0.25) }, 2, 2),
            ["lgc"] = new MatLogical(new[] { true, false, true, false, false, true }, 2, 3),
            ["str"] = "hello",
            ["uni"] = "héllo € 日本",
            ["sarr"] = new MatChar(new[] { "abc", "xyz" }),
            ["emptystr"] = "",
            ["empty"] = new double[0, 3],
            ["st"] = new Dictionary<string, object?> { ["a"] = 1.0, ["b"] = "text", ["c"] = new double[,] { { 1, 2 }, { 3, 4 } } },
            ["starr"] = new MatStruct([1, 2], ["x", "y"], [
                [MatValue.FromObject(1.0), MatValue.FromObject("first")],
                [MatValue.FromObject(new[] { 2.0, 3.0 }), MatValue.FromObject("second")],
            ]),
            ["cell"] = new MatCell(1.0, "two", new[] { 3.0, 4.0 }),
            ["nest"] = new MatCell(new Dictionary<string, object?> { ["q"] = new MatCell(6.0, -7.0) }, new MatCell("inner")),
            ["emptycell"] = new MatCell([0, 0], []),
            ["sp"] = new MatSparse(spm),
            ["splog"] = new MatSparse(new Gravicode.Science.SciNet.Sparse.CscMatrix(new[] { 1.0, 1.0, 1.0 }, new[] { 0, 2, 1 }, new[] { 0, 2, 3 }, 3, 2), isLogical: true),
            ["spc"] = new MatSparse(new Gravicode.Science.SciNet.Sparse.CscMatrix(new[] { 1.0, 3.0, 0.0 }, new[] { 0, 2, 1 }, new[] { 0, 2, 2, 3 }, 3, 3), new[] { 2.0, 0.0, -1.0 }),
            ["big"] = new NdArray(Enumerable.Range(0, 200 * 30).Select(i => Math.Sin(i * 0.01)).ToArray(), 200, 30),
        };
    }

    public static NetCdfFile NetCdf()
    {
        var f = NetCdfFile.Create(version: 4);
        f.CreateDimension("time", null);
        f.CreateDimension("lat", 3);
        f.CreateDimension("lon", 4);
        f.CreateDimension("strlen", 5);
        f.Attributes["title"] = "SciNet NetCDF-4 writer";
        f.Attributes["version"] = 3;
        f.Attributes["coeffs"] = new[] { 1.5, -2.0 };
        f.Attributes["tags"] = new[] { "a", "bc" };
        var time = f.CreateVariable("time", NetCdfType.Double, "time");
        time.Attributes["units"] = "hours since 2000-01-01";
        time.SetData(new NdArray(new[] { 0.0, 6.0 }, 2));
        var lat = f.CreateVariable("lat", NetCdfType.Float, "lat");
        lat.SetData(new NdArray(new[] { -10.0, 0, 10 }, 3));
        var temp = f.CreateVariable("temp", NetCdfType.Short, "time", "lat", "lon");
        temp.Attributes["_FillValue"] = (short)-999;
        temp.Attributes["scale_factor"] = 0.01f;
        temp.Attributes["long_name"] = "température";
        temp.SetData(new NdArray(Enumerable.Range(0, 24).Select(i => i == 23 ? -999.0 : i * 37 - 300).ToArray(), 2, 3, 4));
        f.CreateVariable("scalar", NetCdfType.Double).SetData(new NdArray(new[] { 42.5 }, 1));
        f.CreateVariable("name", NetCdfType.Char, "lat", "strlen").SetStrings(["south", "eq", "north"]);
        f.CreateVariable("flags", NetCdfType.Byte, "lon").SetData(new NdArray(new[] { -1.0, 0, 1, 127 }, 4));
        f.CreateVariable("ub", NetCdfType.UByte, "lon").SetData(new NdArray(new[] { 0.0, 1, 254, 255 }, 4));
        f.CreateVariable("u4", NetCdfType.UInt, "lat").SetData(new NdArray(new[] { 0.0, 7, 4294967295 }, 3));
        f.CreateVariable("i8", NetCdfType.Int64, "lat").SetData(new NdArray(new[] { -9007199254740992.0, 0, 9007199254740992 }, 3));
        var words = f.CreateVariable("words", NetCdfType.String, "lat");
        words.SetStrings(["un", "deux", "trois"]);
        f.CreateVariable("nodata", NetCdfType.Int, "lat", "lon"); // never set: all fill values
        return f;
    }
}
