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

    private static void Same(MatValue a, MatValue b, string path)
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
        Assert.Throws<NotSupportedException>(() => MatFile.Save(new MemoryStream(), new Dictionary<string, object?> { ["x"] = 1.0 }, format: "7.3"));
    }
}
