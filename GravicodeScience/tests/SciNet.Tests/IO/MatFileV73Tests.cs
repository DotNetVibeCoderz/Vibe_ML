using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// MATLAB v7.3 (HDF5) loading. mat73_scinet.mat holds the same variables as scinet_v5.mat (written
/// by scipy savemat), laid out the way MATLAB lays out v7.3 files (see gen_hdf5.py: MATLAB is not
/// available to write it, and SciPy cannot), so every variable must load identically through both
/// readers. The real MATLAB-written v7.3 file is covered in MatFileTests.
/// </summary>
public class MatFileV73Tests
{
    private static readonly string V73 = Fixture("mat73_scinet.mat");

    internal static void Same(MatValue a, MatValue b, string path)
    {
        Assert.True(a.GetType() == b.GetType(), $"{path}: {b.GetType().Name}, expected {a.GetType().Name}");
        Assert.True(a.Dimensions.SequenceEqual(b.Dimensions), $"{path}: dims [{string.Join(",", b.Dimensions)}], expected [{string.Join(",", a.Dimensions)}]");
        Assert.True(a.ClassName == b.ClassName, $"{path}: class {b.ClassName}, expected {a.ClassName}");
        switch (a)
        {
            case MatArray x:
            {
                var y = (MatArray)b;
                Assert.Equal(x.Class, y.Class);
                Assert.True(x.Real.ToArray().SequenceEqual(y.Real.ToArray()), $"{path}: real parts differ");
                Assert.Equal(x.IsComplex, y.IsComplex);
                if (x.IsComplex) Assert.True(x.Imaginary!.ToArray().SequenceEqual(y.Imaginary!.ToArray()), $"{path}: imaginary parts differ");
                break;
            }
            case MatLogical x:
                Assert.True(x.ToBoolArray().SequenceEqual(((MatLogical)b).ToBoolArray()), $"{path}: logical values differ");
                break;
            case MatChar x:
                Assert.True(x.Chars.SequenceEqual(((MatChar)b).Chars), $"{path}: '{new string(((MatChar)b).Chars)}', expected '{new string(x.Chars)}'");
                break;
            case MatSparse x:
            {
                var y = (MatSparse)b;
                Assert.Equal(x.IsLogical, y.IsLogical);
                Assert.Equal(x.Matrix.Indptr.ToArray(), y.Matrix.Indptr.ToArray());
                Assert.Equal(x.Matrix.Indices.ToArray(), y.Matrix.Indices.ToArray());
                Assert.Equal(x.Matrix.Data.ToArray(), y.Matrix.Data.ToArray());
                Assert.Equal(x.Imaginary.ToArray(), y.Imaginary.ToArray());
                break;
            }
            case MatStruct x:
            {
                var y = (MatStruct)b;
                Assert.Equal(x.FieldNames, y.FieldNames);
                Assert.Equal(x.ObjectClassName, y.ObjectClassName);
                for (var e = 0; e < x.Count; e++)
                    foreach (var f in x.FieldNames)
                        Same(x[e, f], y[e, f], $"{path}({e}).{f}");
                break;
            }
            case MatCell x:
            {
                var y = (MatCell)b;
                for (var i = 0; i < x.Count; i++) Same(x[i], y[i], $"{path}{{{i}}}");
                break;
            }
        }
    }

    [Fact]
    public void LoadsExactlyWhatTheV5LoaderLoads()
    {
        var v5 = MatFile.Load(Fixture("scinet_v5.mat"));
        var v73 = MatFile.Load(V73);
        Assert.Equal((2, 0), MatFile.GetVersion(V73));
        var compared = 0;
        foreach (var (name, value) in v5)
        {
            if (name == "obj") continue; // old-style objects have no v7.3 equivalent here
            Assert.True(v73.ContainsKey(name), $"v7.3 file lacks '{name}'");
            Same(value, v73[name], name);
            compared++;
        }
        Assert.Equal(27, compared);
    }

    [Fact]
    public void ValuesAndLayoutConversion()
    {
        var d = MatFile.Load(V73);
        // dbl = arange(12).reshape(3, 4) * 1.5 - 2, stored transposed (column-major) in the file.
        var dbl = Assert.IsType<MatArray>(d["dbl"]);
        Shape([3, 4], dbl.Dimensions);
        Close(Enumerable.Range(0, 12).Select(i => i * 1.5 - 2).ToArray(), dbl.Real.ToArray(), 0, "dbl");
        var nd = Assert.IsType<MatArray>(d["nd"]);
        Shape([2, 3, 4], nd.Dimensions);
        Close(Enumerable.Range(0, 24).Select(i => (double)i).ToArray(), nd.Real.ToArray(), 0, "nd");
        var c = Assert.IsType<MatArray>(d["cplx"]);
        Close(new System.Numerics.Complex(0.5, -0.25), c.ToComplex().ToArray()[3], 0, "cplx(2,2)");
        Assert.Equal("héllo € 日本", d["uni"].Text);
        Assert.Equal(["abc", "xyz"], ((MatChar)d["sarr"]).Strings);
        Assert.Equal(MatClass.UInt64, ((MatArray)d["u64"]).Class);
        Assert.Equal(9223372036854775808.0, ((MatArray)d["u64"]).Real.ToArray()[1]);
        var st = Assert.IsType<MatStruct>(d["st"]);
        Assert.Equal("text", st["b"].Text);
        var starr = Assert.IsType<MatStruct>(d["starr"]);
        Shape([1, 2], starr.Dimensions);
        Assert.Equal("second", starr[1, "y"].Text);
        var cell = Assert.IsType<MatCell>(d["cell"]);
        Assert.Equal("two", cell[1].Text);
        var nest = Assert.IsType<MatCell>(d["nest"]);
        Assert.Equal(-7.0, ((MatCell)((MatStruct)nest[0])["q"])[1].Scalar);
        var sp = Assert.IsType<MatSparse>(d["spc"]);
        Assert.True(sp.IsComplex);
        Assert.True(Assert.IsType<MatSparse>(d["splog"]).IsLogical);
        Assert.True(d["empty"].IsEmpty);
        Assert.Equal("", d["emptystr"].Text);
    }

    [Fact]
    public void UndecodableVariablesAndGlobals()
    {
        var d = MatFile.Load(V73);
        var fh = Assert.IsType<MatUnsupported>(d["fh"]);
        Assert.Equal(MatClass.Function, fh.Class);
        var so = Assert.IsType<MatUnsupported>(d["strobj"]);
        Assert.Contains("string", so.Message);
        Assert.True(d["gl"].IsGlobal);
        Assert.False(d["dbl"].IsGlobal);
        Assert.DoesNotContain("#refs#", d.Keys);
    }

    private static void Who(MatVariableInfo info, string cls, params int[] shape)
    {
        Assert.Equal(cls, info.ClassName);
        Assert.Equal(shape, info.Shape);
    }

    [Fact]
    public void WhosmatAndOptions()
    {
        var w = MatFile.Whosmat(V73).ToDictionary(x => x.Name);
        Who(w["dbl"], "double", 3, 4);
        Who(w["sp"], "sparse", 4, 5);
        Who(w["lgc"], "logical", 2, 3);
        Who(w["starr"], "struct", 1, 2);
        Who(w["cell"], "cell", 1, 3);
        Who(w["emptystr"], "char", 0, 0);
        Who(w["sarr"], "char", 2, 3);

        var only = MatFile.Load(V73, variableNames: ["dbl", "cell"]);
        Assert.Equal(["cell", "dbl"], only.Keys); // v7.3 variables come in name order (MATLAB writes a v1 group B-tree)
        var sq = MatFile.Load(V73, squeezeMe: true);
        Assert.Empty(sq["scalar"].Dimensions);
        Assert.Equal(3.5, sq["scalar"].Scalar);
        var simple = MatFile.Load(V73, simplifyCells: true);
        Assert.IsType<MatCell>(simple["starr"]);

        using var s = File.OpenRead(V73);
        Assert.Equal(1.0, ((MatStruct)MatFile.Load(s)["st"])["a"].Scalar);
    }

    // ------------------------------------------------------------------ writing

    private static IReadOnlyDictionary<string, MatValue> SaveAndLoad(IReadOnlyDictionary<string, MatValue> vars, bool compress = false)
    {
        var path = TempFile(".mat");
        try
        {
            MatFile.Save(path, vars, doCompression: compress, format: "7.3");
            Assert.Equal((2, 0), MatFile.GetVersion(path));
            return MatFile.Load(path);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Save73_LoadsBackEveryVariableOfTheV5Fixture(bool compress)
    {
        // Every class SciPy's savemat produces (27 variables plus an old-style object), written as v7.3 and read back.
        var v5 = MatFile.Load(Fixture("scinet_v5.mat"));
        var back = SaveAndLoad(v5, compress);
        Assert.Equal(v5.Keys, back.Keys); // creation order is kept
        foreach (var (name, value) in v5) Same(value, back[name], name);
        Assert.Equal(28, back.Count);
    }

    [Fact]
    public void Save73_RoundTripsTheV73FixtureExceptWhatCannotBeDecoded()
    {
        var all = MatFile.Load(V73);
        Assert.Throws<NotSupportedException>(() => SaveAndLoad(all)); // function handle and string object
        var decodable = all.Where(kv => kv.Value is not MatUnsupported).ToDictionary(kv => kv.Key, kv => kv.Value);
        var back = SaveAndLoad(decodable);
        foreach (var (name, value) in decodable)
        {
            Same(value, back[name], name);
            Assert.Equal(value.IsGlobal, back[name].IsGlobal);
        }
        Assert.True(back["gl"].IsGlobal);
    }

    [Fact]
    public void Save73_LayoutIsMatlabs()
    {
        var path = TempFile(".mat");
        try
        {
            MatFile.Save(path, new Dictionary<string, object?>
            {
                ["a"] = new NdArray([1.0, 2, 3, 4, 5, 6], 2, 3),
                ["s"] = "hé",
                ["c"] = new object?[] { 1.0, "x" },
                ["e"] = new double[0, 4],
                ["big"] = new double[100, 100],
                ["i"] = new short[] { -1, 2 },
            }, doCompression: true, format: "7.3");
            var raw = File.ReadAllBytes(path);
            Assert.StartsWith("MATLAB 7.3 MAT-file, Platform: ", System.Text.Encoding.ASCII.GetString(raw, 0, 116));
            Assert.Contains("HDF5 schema 1.00 .", System.Text.Encoding.ASCII.GetString(raw, 0, 116));
            Assert.Equal(new byte[] { 0x00, 0x02, (byte)'I', (byte)'M' }, raw[124..128]);

            using var h5 = Hdf5File.Open(path);
            Assert.Equal(512, h5.UserBlockSize);
            Assert.Equal(["a", "s", "#refs#", "c", "e", "big", "i"], h5.Root.Keys); // #refs# appears when the first cell needs it
            var a = h5.Root.Dataset("a");
            Assert.Equal(new[] { 3, 2 }, a.Shape); // MATLAB 2 x 3, stored with the dimensions reversed
            Assert.Equal(new[] { 1.0, 4, 2, 5, 3, 6 }, a.Read().ToArray()); // column-major
            var cls = a.Attributes["MATLAB_class"];
            Assert.Equal("double", cls.Value);
            Assert.Equal(Hdf5TypeClass.String, cls.DataType.Class);
            Assert.Equal(6, cls.DataType.Size); // sized to the name, NUL-terminated padding, as MATLAB writes it
            Assert.Equal(0, cls.DataType.StringPadding);
            Assert.True(cls.IsScalar);
            var s = h5.Root.Dataset("s");
            Assert.Equal(new ushort[] { 'h', 'é' }, (ushort[])s.ReadArray());
            Assert.Equal(2, s.GetAttribute("MATLAB_int_decode"));
            var c = h5.Root.Dataset("c");
            Assert.Equal("cell", c.Attributes["MATLAB_class"].Value);
            Assert.All(c.ReadReferences(), r => Assert.Contains(h5.Dereference(r).Address, h5.Root.Group("#refs#").Members.Select(m => m.Address)));
            var e = h5.Root.Dataset("e");
            Assert.Equal(new ulong[] { 0, 4 }, (ulong[])e.ReadArray());
            Assert.Equal((byte)1, e.GetAttribute("MATLAB_empty"));
            Assert.Equal("deflate", Assert.Single(h5.Root.Dataset("big").Filters).Name); // 80 KB: compressed
            Assert.Empty(h5.Root.Dataset("a").Filters); // 48 bytes: a chunk index would cost more than it saves
            Assert.Equal("int16", h5.Root.Dataset("i").Attributes["MATLAB_class"].Value);

            var who = MatFile.Whosmat(path).ToDictionary(x => x.Name);
            Assert.Equal(new[] { 2, 3 }, who["a"].Shape);
            Assert.Equal("cell", who["c"].ClassName);
            Assert.Equal(new[] { 0, 4 }, who["e"].Shape);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Save73_RefusesValuesThatWouldNotSurvive()
    {
        var path = TempFile(".mat");
        try
        {
            Assert.Throws<ArgumentException>(() => MatFile.Save(path, new Dictionary<string, MatValue> { ["x"] = new MatArray(new NdArray([300.0], 1), MatClass.Int8) }, format: "7.3"));
            Assert.Throws<ArgumentException>(() => MatFile.Save(path, new Dictionary<string, object?> { [new string('n', 64)] = 1.0 }, format: "7.3"));
            Assert.Throws<ArgumentException>(() => MatFile.Save(path, new Dictionary<string, object?> { ["x"] = 1.0 }, format: "7"));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
