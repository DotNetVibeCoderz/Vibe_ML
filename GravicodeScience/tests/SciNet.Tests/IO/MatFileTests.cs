using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Gravicode.Science.SciNet.Sparse;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// MAT-file tests. Fixtures under Fixtures/scinet_* were written by scipy 1.17.1 (savemat) from
/// gen_fixtures.py; Fixtures/scipy/* are MATLAB-written files copied from SciPy's own test suite
/// (scipy/io/matlab/tests/data, BSD-3-Clause) — v4 and v6 files from 32-bit SPARC Solaris
/// (big-endian), v7 from Linux. Expected values are SciPy 1.17.1's loadmat output.
/// Files written by MatFile.Save (v5 plain and compressed, v4; every numeric class, logical,
/// char incl. non-ASCII, complex, sparse incl. complex and logical, struct arrays, objects, 2-D
/// cells, n-D arrays) were read back by scipy 1.17.1 loadmat during development and matched
/// exactly; logical sparse comes back as uint8 there, as SciPy's own files do.
/// </summary>
public class MatFileTests
{
    private static string Scipy(string name) => Fixture(Path.Combine("scipy", name));

    [Theory]
    [InlineData("scinet_v5.mat")]
    [InlineData("scinet_v5z.mat")]
    public void ScipyWrittenV5_EveryClass(string file)
    {
        var d = MatFile.Load(Fixture(file));

        var dbl = (MatArray)d["dbl"];
        Assert.Equal("double", dbl.ClassName);
        Shape([3, 4], dbl.Dimensions);
        var expected = new double[12];
        for (var k = 0; k < 12; k++) expected[k] = k * 1.5 - 2;
        Close(expected, dbl.Real.ToArray(), 0, "dbl");

        var sgl = (MatArray)d["sgl"];
        Assert.Equal(MatClass.Single, sgl.Class);
        Close([1.5, -2.25, 3.0, 4.0, 5.5, -6.75], sgl.Real.ToArray(), 0, "sgl");

        void Ints(string name, MatClass cls, double[] values)
        {
            var a = (MatArray)d[name];
            Assert.Equal(cls, a.Class);
            Shape([2, 2], a.Dimensions, name);
            Close(values, a.Real.ToArray(), 0, name);
        }
        Ints("i8", MatClass.Int8, [-128, 127, 0, -1]);
        Ints("u8", MatClass.UInt8, [0, 255, 1, 128]);
        Ints("i16", MatClass.Int16, [-32768, 32767, 5, -5]);
        Ints("u16", MatClass.UInt16, [0, 65535, 7, 300]);
        Ints("i32", MatClass.Int32, [-2147483648, 2147483647, 11, -11]);
        Ints("u32", MatClass.UInt32, [0, 4294967295, 13, 99]);
        Ints("i64", MatClass.Int64, [-Math.Pow(2, 62), Math.Pow(2, 53), 17, -17]);
        Ints("u64", MatClass.UInt64, [0, Math.Pow(2, 63), 19, Math.Pow(2, 40)]);
        Assert.Equal("uint64", d["u64"].ClassName);

        var lgc = (MatLogical)d["lgc"];
        Assert.Equal("logical", lgc.ClassName);
        Assert.Equal([true, false, true, false, false, true], lgc.ToBoolArray());

        Assert.Equal("hello", d["s"].Text);
        Assert.Equal(["abc", "xyz"], ((MatChar)d["sarr"]).Strings);
        Shape([2, 3], d["sarr"].Dimensions);
        Assert.Equal("héllo € 日本", d["uni"].Text);
        Assert.Equal("", d["emptystr"].Text);

        var c = ((MatArray)d["cplx"]).ToComplex();
        Close(new Complex(1, 2), c[0, 0], 0);
        Close(new Complex(0, -3.5), c[0, 1], 0);
        Close(new Complex(4, 0), c[1, 0], 0);
        Close(new Complex(0.5, -0.25), c[1, 1], 0);

        var nd = d["nd"].ToNdArray();
        Shape([2, 3, 4], nd.Shape.ToArray());
        Close(Enumerable.Range(0, 24).Select(i => (double)i).ToArray(), nd.ToArray(), 0, "nd"); // numpy arange(24).reshape(2,3,4), same row-major order

        Assert.True(d["empty"].IsEmpty);
        Shape([0, 0], d["empty"].Dimensions);

        var sp = (MatSparse)d["sp"];
        Assert.False(sp.IsComplex);
        Assert.Equal(5, sp.Matrix.Nnz);
        Close(new NdArray([1.0, 0, 0, 2.5, 0, 0, 0, -3.0, 0, 0, 0, 4.0, 0, 0, 0, 0, 0, 0, 0, 5.5], 4, 5), sp.ToNdArray(), 0, "sp");

        var spc = ((MatSparse)d["spc"]).ToComplexDense();
        Close(new Complex(1, 2), spc[0, 0], 0);
        Close(new Complex(0, -1), spc[1, 2], 0);
        Close(new Complex(3, 0), spc[2, 0], 0);
        Close(Complex.Zero, spc[1, 1], 0);

        var splog = (MatSparse)d["splog"];
        Assert.True(splog.IsLogical);
        Close(new NdArray([1.0, 0, 0, 1, 1, 1], 3, 2), splog.ToNdArray(), 0, "splog");

        var st = (MatStruct)d["st"];
        Assert.Equal(["a", "b", "c"], st.FieldNames);
        Assert.Equal(1.0, st["a"].Scalar);
        Assert.Equal("text", st["b"].Text);
        Close(new NdArray([1.0, 2, 3, 4], 2, 2), st["c"].ToNdArray(), 0);

        var starr = (MatStruct)d["starr"];
        Shape([1, 2], starr.Dimensions);
        Assert.Equal(1.0, starr[0, "x"].Scalar);
        Assert.Equal("second", starr[1, "y"].Text);
        Close([2.0, 3.0], starr[1, "x"].ToNdArray().ToArray(), 0);

        var cell = (MatCell)d["cell"];
        Shape([1, 3], cell.Dimensions);
        Assert.Equal(1.0, cell[0].Scalar);
        Assert.Equal("two", cell[1].Text);
        Close([0.0, 1, 2], cell[2].ToNdArray().ToArray(), 0);

        var nest = (MatCell)d["nest"];
        Shape([2, 1], nest.Dimensions);
        var inner = (MatStruct)nest[0, 0];
        Close([5.0, 6.0], inner["p"].ToNdArray().ToArray(), 0);
        var q = (MatCell)inner["q"];
        Assert.Equal("deep", q[0].Text);
        Assert.Equal(MatClass.Int16, ((MatArray)q[1]).Class);
        Assert.Equal(-7.0, q[1].Scalar);
        Assert.Equal("deep", ((MatCell)nest[1, 0])[0].Text);

        Assert.Equal(3.5, d["scalar"].Scalar);
        Shape([1, 3], d["vec"].Dimensions); // scipy's default oned_as='row'

        var obj = (MatStruct)d["obj"];
        Assert.True(obj.IsObject);
        Assert.Equal("myclass", obj.ClassName);
        Assert.Equal(42.0, obj["val"].Scalar);
    }

    [Fact]
    public void Whosmat_MatchesScipy()
    {
        // scipy 1.17.1 whosmat('scinet_v5z.mat'), except that SciPy reports char arrays with
        // the string-collapsed shape ((1,) for 'hello'); SciNet reports MATLAB's dimensions.
        var w = MatFile.Whosmat(Fixture("scinet_v5z.mat"));
        var byName = w.ToDictionary(x => x.Name);
        Assert.Equal(28, w.Count);
        Assert.Equal("dbl", w[0].Name);
        Assert.Equal("double", byName["dbl"].ClassName);
        Assert.Equal(new[] { 3, 4 }, byName["dbl"].Shape);
        Assert.Equal("int8", byName["i8"].ClassName);
        Assert.Equal("logical", byName["lgc"].ClassName);
        Assert.Equal("char", byName["s"].ClassName);
        Assert.Equal(new[] { 1, 5 }, byName["s"].Shape);
        Assert.Equal("sparse", byName["sp"].ClassName);
        Assert.Equal("logical", byName["splog"].ClassName);
        Assert.Equal("struct", byName["starr"].ClassName);
        Assert.Equal(new[] { 1, 2 }, byName["starr"].Shape);
        Assert.Equal("cell", byName["nest"].ClassName);
        Assert.Equal("object", byName["obj"].ClassName);
        Assert.Equal(new[] { 2, 3, 4 }, byName["nd"].Shape);
    }

    [Fact]
    public void OnedAsColumn_FromScipy()
    {
        var d = MatFile.Load(Fixture("scinet_v5_col.mat"));
        Shape([3, 1], d["vec"].Dimensions);
    }

    [Fact]
    public void ScipyWrittenV4()
    {
        var d = MatFile.Load(Fixture("scinet_v4.mat"));
        Assert.Equal((0, 0), MatFile.GetVersion(Fixture("scinet_v4.mat")));
        var expected = new double[12];
        for (var k = 0; k < 12; k++) expected[k] = k * 1.5 - 2;
        Close(expected, d["dbl"].ToNdArray().ToArray(), 0);
        Assert.Equal("hello v4", d["s"].Text);
        var c = ((MatArray)d["cplx"]).ToComplex();
        Close(new Complex(0.5, -0.25), c[1, 1], 0);
        Close(new NdArray([1.0, 0, 0, 2.5, 0, 0, 0, -3.0, 0, 0, 0, 4.0, 0, 0, 0, 0, 0, 0, 0, 5.5], 4, 5), d["sp"].ToNdArray(), 0);
    }

    // ------------------------------------------------------------- MATLAB-written files

    [Theory]
    [InlineData("4.2c_SOL2")] // v4, big-endian
    [InlineData("6.1_SOL2")]  // v5, big-endian
    [InlineData("7.4_GLNX86")] // v7, little-endian, compressed
    public void MatlabWritten_NumericAndText(string suffix)
    {
        var d = MatFile.Load(Scipy($"testdouble_{suffix}.mat"));
        Close(Enumerable.Range(0, 9).Select(k => k * Math.PI / 4).ToArray(), d["testdouble"].ToNdArray().ToArray(), 1e-15, "testdouble");
        Assert.Equal("double", d["testdouble"].ClassName);

        var m = MatFile.Load(Scipy($"testmatrix_{suffix}.mat"))["testmatrix"];
        Assert.Equal("double", m.ClassName); // MATLAB stores it as miUINT8; the class is still double
        Close(new NdArray([1.0, 2, 3, 4, 5, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0], 3, 5), m.ToNdArray(), 0, "testmatrix");

        var c = ((MatArray)MatFile.Load(Scipy($"testcomplex_{suffix}.mat"))["testcomplex"]).ToComplex();
        for (var k = 0; k < 9; k++) Close(Complex.FromPolarCoordinates(1, k * Math.PI / 4), c[0, k], 1e-15, $"testcomplex[{k}]");

        Assert.Equal(-1.0, MatFile.Load(Scipy($"testminus_{suffix}.mat"))["testminus"].Scalar);
        Assert.Equal("r", MatFile.Load(Scipy($"testonechar_{suffix}.mat"))["testonechar"].Text);
        Assert.Equal("\"Do nine men interpret?\" \"Nine men,\" I nod.", MatFile.Load(Scipy($"teststring_{suffix}.mat"))["teststring"].Text);
        Assert.Equal(["one  ", "two  ", "three"], ((MatChar)MatFile.Load(Scipy($"teststringarray_{suffix}.mat"))["teststringarray"]).Strings);

        var multi = MatFile.Load(Scipy($"testmulti_{(suffix == "6.1_SOL2" ? "7.4_GLNX86" : suffix)}.mat"));
        Assert.Equal(["a", "theta"], multi.Keys);
    }

    [Theory]
    [InlineData("4.2c_SOL2")]
    [InlineData("6.1_SOL2")]
    [InlineData("7.4_GLNX86")]
    public void MatlabWritten_Sparse(string suffix)
    {
        var s = (MatSparse)MatFile.Load(Scipy($"testsparse_{suffix}.mat"))["testsparse"];
        var expected = new NdArray([1.0, 2, 3, 4, 5, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0], 3, 5);
        Close(expected, s.ToNdArray(), 0, "testsparse");
        Assert.Equal(7, s.Matrix.Nnz);

        var c = (MatSparse)MatFile.Load(Scipy($"testsparsecomplex_{suffix}.mat"))["testsparsecomplex"];
        Assert.True(c.IsComplex);
        var cd = c.ToComplexDense();
        Close(new Complex(1, 1), cd[0, 0], 0);
        Close(new Complex(5, 0), cd[0, 4], 0);
        Close(new Complex(3, 0), cd[2, 0], 0);
        Close(Complex.Zero, cd[2, 4], 0);
    }

    [Theory]
    [InlineData("6.1_SOL2")]
    [InlineData("7.4_GLNX86")]
    public void MatlabWritten_StructCellObject(string suffix)
    {
        var st = (MatStruct)MatFile.Load(Scipy($"teststruct_{suffix}.mat"))["teststruct"];
        Assert.Equal(["stringfield", "doublefield", "complexfield"], st.FieldNames);
        Assert.Equal("Rats live on no evil star.", st["stringfield"].Text);
        Close([Math.Sqrt(2), Math.E, Math.PI], st["doublefield"].ToNdArray().ToArray(), 1e-15);
        var cf = ((MatArray)st["complexfield"]).ToComplex();
        Close(new Complex(Math.E, Math.E), cf[0, 1], 1e-15);

        var arr = (MatStruct)MatFile.Load(Scipy($"teststructarr_{suffix}.mat"))["teststructarr"];
        Shape([1, 2], arr.Dimensions);
        Assert.Equal(1.0, arr[0, "one"].Scalar);
        Assert.Equal(2.0, arr[0, "two"].Scalar);
        Assert.Equal("number 1", arr[1, "one"].Text);
        Assert.Equal("number 2", arr[1, "two"].Text);

        var nest = (MatStruct)MatFile.Load(Scipy($"teststructnest_{suffix}.mat"))["teststructnest"];
        Assert.Equal("number 3", ((MatStruct)nest["two"])["three"].Text);

        var cell = (MatCell)MatFile.Load(Scipy($"testcell_{suffix}.mat"))["testcell"];
        Shape([1, 4], cell.Dimensions);
        Assert.Equal("This cell contains this string and 3 arrays of increasing length", cell[0].Text);
        Close([1.0, 2, 3], cell[3].ToNdArray().ToArray(), 0);

        var cn = (MatCell)MatFile.Load(Scipy($"testcellnest_{suffix}.mat"))["testcellnest"];
        var lvl2 = (MatCell)cn[1];
        var lvl3 = (MatCell)lvl2[2];
        Assert.Equal(5.0, lvl3[1].Scalar);

        var obj = (MatStruct)MatFile.Load(Scipy($"testobject_{suffix}.mat"))["testobject"];
        Assert.Equal("inline", obj.ObjectClassName);
        Assert.Equal(" x = INLINE_INPUTS_{1};", obj["inputExpr"].Text);
        Assert.Equal(1.0, obj["numArgs"].Scalar);

        var t3 = MatFile.Load(Scipy($"test3dmatrix_{suffix}.mat"))["test3dmatrix"].ToNdArray();
        Shape([2, 3, 4], t3.Shape.ToArray());
        // MATLAB: reshape(1:24, 2, 3, 4) — the (i,j,k) element is 1 + i + 2j + 6k.
        for (var i = 0; i < 2; i++)
            for (var j = 0; j < 3; j++)
                for (var k = 0; k < 4; k++)
                    Assert.Equal(1 + i + 2 * j + 6 * k, t3[i, j, k]);
    }

    [Fact]
    public void MatlabWritten_EdgeCases()
    {
        var empty = (MatCell)MatFile.Load(Scipy("testemptycell_7.4_GLNX86.mat"))["testemptycell"];
        Shape([1, 5], empty.Dimensions);
        Assert.True(empty[2].IsEmpty);
        Assert.Equal(3.0, empty[4].Scalar);

        Assert.Equal(1.0, ((MatCell)MatFile.Load(Scipy("testscalarcell_7.4_GLNX86.mat"))["testscalarcell"])[0].Scalar);
        var simple = (MatStruct)MatFile.Load(Scipy("testsimplecell.mat"))["s"];
        Assert.Equal(["a", "b", "c"], ((MatCell)simple["mycell"]).Select(v => v.Text));

        var es = (MatStruct)MatFile.Load(Scipy("test_empty_struct.mat"))["a"];
        Assert.Empty(es.FieldNames);

        var bools = (MatLogical)MatFile.Load(Scipy("testbool_8_WIN64.mat"))["testbools"];
        Assert.Equal([true, false], bools.ToBoolArray());
        Shape([2, 1], bools.Dimensions);

        var ls = (MatSparse)MatFile.Load(Scipy("logical_sparse.mat"))["sp_log_5_4"];
        Assert.True(ls.IsLogical);
        var lsd = ls.ToNdArray();
        Assert.Equal(5.0, lsd.ToArray().Sum());
        Assert.Equal(1.0, lsd[0, 0]);
        Assert.Equal(1.0, lsd[2, 2]);
        Assert.Equal(0.0, lsd[3, 3]);

        var sf = MatFile.Load(Scipy("testsparsefloat_7.4_GLNX86.mat"))["testsparsefloat"].ToNdArray();
        Close([1.0, 0, 2, 0, -3.5, 0], sf.ToArray(), 0);

        foreach (var f in new[] { "big_endian.mat", "little_endian.mat" })
        {
            var d = MatFile.Load(Scipy(f));
            Close([2.0, 3, 3, 4], d["floats"].ToNdArray().ToArray(), 0, f);
            Assert.Equal(MatClass.Single, ((MatArray)d["floats"]).Class);
            var strings = (MatCell)d["strings"];
            Assert.Equal("hello", strings[0].Text);
            Assert.Equal("world", strings[1].Text);
        }

        Assert.Contains("array_name", MatFile.Load(Scipy("miutf8_array_name.mat")).Keys);
        Assert.True(MatFile.Load(Scipy("one_by_zero_char.mat"))["var"].IsEmpty);
        Assert.Equal("", MatFile.Load(Scipy("single_empty_string.mat"))["a"].Text);

        var uni = MatFile.Load(Scipy("testunicode_7.4_GLNX86.mat"))["testunicode"].Text;
        Assert.StartsWith("Japanese: \nすべての人間", uni);

        Close([0.1, 1.2], MatFile.Load(Scipy("test_mat4_le_floats.mat"))["a"].ToNdArray().ToArray(), 1e-7);
    }

    [Fact]
    public void FunctionHandles_AreSkippedWithAReason()
    {
        var d = MatFile.Load(Scipy("some_functions.mat"));
        Assert.Equal(-3.9, d["a"].Scalar);
        Assert.Equal(52.0, d["b"].Scalar);
        var f = Assert.IsType<MatUnsupported>(d["sqr"]);
        Assert.Equal(MatClass.Function, f.Class);
        Assert.Contains("function handle", f.Message);
        Assert.IsType<MatUnsupported>(d["nCf"]);
        Assert.Contains("__function_workspace__", d.Keys);

        var w = MatFile.Whosmat(Scipy("some_functions.mat"));
        Assert.Equal("function", w.Single(x => x.Name == "parabola").ClassName);
        Assert.Equal(new[] { 1, 1408 }, w.Single(x => x.Name == "__function_workspace__").Shape);
    }

    [Fact]
    public void V73_IsRefusedWithAClearMessage()
    {
        var e = Assert.Throws<NotSupportedException>(() => MatFile.Load(Scipy("testhdf5_7.4_GLNX86.mat")));
        Assert.Contains("HDF5", e.Message);
        Assert.Equal((2, 0), MatFile.GetVersion(Scipy("testhdf5_7.4_GLNX86.mat")));
    }

    // ------------------------------------------------------------- options

    [Fact]
    public void VariableNames_SqueezeMe_SimplifyCells()
    {
        var only = MatFile.Load(Fixture("scinet_v5z.mat"), variableNames: ["dbl", "cell"]);
        Assert.Equal(["dbl", "cell"], only.Keys);

        var sq = MatFile.Load(Fixture("scinet_v5.mat"), squeezeMe: true);
        Assert.Empty(sq["scalar"].Dimensions);
        Assert.Equal(3.5, sq["scalar"].Scalar);
        Shape([3], sq["vec"].Dimensions);
        Shape([3, 4], sq["dbl"].Dimensions);
        Shape([3], sq["cell"].Dimensions);

        var simple = MatFile.Load(Fixture("scinet_v5.mat"), simplifyCells: true);
        var starr = Assert.IsType<MatCell>(simple["starr"]); // struct array → cell of scalar structs
        Assert.Equal(2, starr.Count);
        Assert.Equal("first", ((MatStruct)starr[0])["y"].Text);
        Assert.Equal(1.0, ((MatStruct)simple["st"])["a"].Scalar);
    }

    // ------------------------------------------------------------- writing

    private static Dictionary<string, object?> Sample() => new()
    {
        ["dbl"] = new NdArray(Enumerable.Range(0, 12).Select(i => i * 0.25 - 1).ToArray(), 3, 4),
        ["nd"] = new NdArray(Enumerable.Range(0, 120).Select(i => (double)i).ToArray(), 3, 4, 5, 2),
        ["vec"] = new[] { 1.0, 2.0, 3.0 },
        ["i16"] = new short[,] { { -5, 7 }, { 32767, -32768 } },
        ["u8"] = (byte)200,
        ["flag"] = new[] { true, false, true },
        ["text"] = "héllo \U0001F600",
        ["rows"] = new[] { "ab", "cde" },
        ["z"] = ComplexNdArray.FromParts(new NdArray([1.0, 2, 3, 4], 2, 2), new NdArray([0.5, -1, 0, 2], 2, 2)),
        ["sp"] = CscMatrix.FromDense(new NdArray([0, 1.5, 0, -2, 0, 0, 3, 0, 4.25], 3, 3)),
        ["st"] = new Dictionary<string, object?> { ["alpha"] = 1.0, ["beta"] = "two", ["inner"] = new Dictionary<string, object?> { ["x"] = 9 } },
        ["cell"] = new object?[] { 1.0, "s", new object?[] { 2.0, null } },
        ["empty"] = null,
    };

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SaveV5_RoundTrips(bool compress)
    {
        var path = TempFile(".mat");
        try
        {
            MatFile.Save(path, Sample(), doCompression: compress);
            var d = MatFile.Load(path);
            Close(Enumerable.Range(0, 12).Select(i => i * 0.25 - 1).ToArray(), d["dbl"].ToNdArray().ToArray(), 0);
            var nd = d["nd"].ToNdArray();
            Shape([3, 4, 5, 2], nd.Shape.ToArray());
            Close(Enumerable.Range(0, 120).Select(i => (double)i).ToArray(), nd.ToArray(), 0, "nd");
            Shape([1, 3], d["vec"].Dimensions);
            var i16 = (MatArray)d["i16"];
            Assert.Equal(MatClass.Int16, i16.Class);
            Close([-5.0, 7, 32767, -32768], i16.Real.ToArray(), 0);
            Assert.Equal("uint8", d["u8"].ClassName);
            Assert.Equal(200.0, d["u8"].Scalar);
            Assert.Equal([true, false, true], ((MatLogical)d["flag"]).ToBoolArray());
            Assert.Equal("héllo \U0001F600", d["text"].Text);
            Assert.Equal(["ab ", "cde"], ((MatChar)d["rows"]).Strings);
            var z = ((MatArray)d["z"]).ToComplex();
            Close(new Complex(2, -1), z[0, 1], 0);
            Close(new NdArray([0, 1.5, 0, -2, 0, 0, 3, 0, 4.25], 3, 3), d["sp"].ToNdArray(), 0);
            var st = (MatStruct)d["st"];
            Assert.Equal("two", st["beta"].Text);
            Assert.Equal(MatClass.Int32, ((MatArray)((MatStruct)st["inner"])["x"]).Class);
            var cell = (MatCell)d["cell"];
            Assert.Equal("s", cell[1].Text);
            Assert.True(((MatCell)cell[2])[1].IsEmpty);
            Assert.True(d["empty"].IsEmpty);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void SaveV5_RoundTripsLoadedValues_IncludingMatlabFiles()
    {
        // Every MATLAB-written v5/v7 fixture survives load → save → load unchanged.
        foreach (var f in Directory.GetFiles(Fixture("scipy"), "test*_7.4_GLNX86.mat"))
        {
            if (f.Contains("hdf5") || f.Contains("func")) continue;
            var a = MatFile.Load(f);
            var path = TempFile(".mat");
            try
            {
                MatFile.Save(path, a, doCompression: true);
                var b = MatFile.Load(path);
                Assert.Equal(a.Keys, b.Keys);
                foreach (var k in a.Keys) AssertSameValue(a[k], b[k], Path.GetFileName(f) + ":" + k);
            }
            finally { File.Delete(path); }
        }
    }

    private static void AssertSameValue(MatValue a, MatValue b, string what)
    {
        Assert.Equal(a.GetType(), b.GetType());
        Assert.Equal(a.ClassName, b.ClassName);
        Assert.Equal(a.Dimensions, b.Dimensions);
        switch (a)
        {
            case MatArray x:
                var y = (MatArray)b;
                Close(x.Real.ToArray(), y.Real.ToArray(), 0, what);
                Assert.Equal(x.IsComplex, y.IsComplex);
                if (x.IsComplex) Close(x.Imaginary!.ToArray(), y.Imaginary!.ToArray(), 0, what);
                break;
            case MatChar c: Assert.Equal(c.Chars.ToArray(), ((MatChar)b).Chars.ToArray()); break;
            case MatLogical l: Assert.Equal(l.ToBoolArray(), ((MatLogical)b).ToBoolArray()); break;
            case MatSparse s:
                var t = (MatSparse)b;
                Assert.True(s.Matrix.ValueEquals(t.Matrix), what);
                Close(s.Imaginary, t.Imaginary, 0, what);
                break;
            case MatCell cc:
                var dd = (MatCell)b;
                for (var i = 0; i < cc.Count; i++) AssertSameValue(cc[i], dd[i], $"{what}{{{i}}}");
                break;
            case MatStruct st:
                var su = (MatStruct)b;
                Assert.Equal(st.FieldNames, su.FieldNames);
                Assert.Equal(st.ObjectClassName, su.ObjectClassName);
                for (var e = 0; e < st.Count; e++)
                    foreach (var f in st.FieldNames) AssertSameValue(st[e, f], su[e, f], $"{what}({e}).{f}");
                break;
        }
    }

    [Fact]
    public void SaveV4_RoundTripsAndRefusesStructs()
    {
        var path = TempFile(".mat");
        try
        {
            var v = new Dictionary<string, object?>
            {
                ["dbl"] = new NdArray([1.0, 2, 3, 4, 5, 6], 2, 3),
                ["nd"] = new NdArray(Enumerable.Range(0, 24).Select(i => (double)i).ToArray(), 2, 3, 4),
                ["s"] = "v4 text",
                ["z"] = ComplexNdArray.FromParts(new NdArray([1.0, 2], 1, 2), new NdArray([3.0, -4], 1, 2)),
                ["sp"] = CscMatrix.FromDense(new NdArray([0, 1.5, 0, -2, 0, 0], 2, 3)),
            };
            MatFile.Save(path, v, format: "4");
            Assert.Equal((0, 0), MatFile.GetVersion(path));
            var d = MatFile.Load(path);
            Close([1.0, 2, 3, 4, 5, 6], d["dbl"].ToNdArray().ToArray(), 0);
            // v4 is 2-D: a 2x3x4 array comes back 2 x 12, in MATLAB's column-major element order.
            var nd = d["nd"].ToNdArray();
            Shape([2, 12], nd.Shape.ToArray());
            Assert.Equal(1 * 12 + 2 * 4 + 3, nd[1, 2 + 3 * 3]); // element (1,2,3) of the 2x3x4 array
            Assert.Equal("v4 text", d["s"].Text);
            Close(new Complex(2, -4), ((MatArray)d["z"]).ToComplex()[0, 1], 0);
            Close(new NdArray([0, 1.5, 0, -2, 0, 0], 2, 3), d["sp"].ToNdArray(), 0);

            Assert.Throws<NotSupportedException>(() => MatFile.Save(path, new Dictionary<string, object?> { ["c"] = new object?[] { 1.0 } }, format: "4"));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void SaveV4_ComplexSparse_RoundTrips()
    {
        var path = TempFile(".mat");
        try
        {
            var re = CscMatrix.FromDense(new NdArray([1.0, 0, 2, 0, 0, 3], 2, 3));
            var sp = new MatSparse(re, [0.5, -1, 2]);
            MatFile.Save(path, new Dictionary<string, MatValue> { ["zs"] = sp }, format: "4");
            var back = (MatSparse)MatFile.Load(path)["zs"];
            var dz = back.ToComplexDense();
            Close(new Complex(1, 0.5), dz[0, 0], 0);
            Close(new Complex(2, -1), dz[0, 2], 0);
            Close(Complex.Zero, dz[1, 0], 0);
            Close(new Complex(3, 2), dz[1, 2], 0);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Save_OnedAsColumn_AndIntegerRangeChecks()
    {
        var path = TempFile(".mat");
        try
        {
            MatFile.Save(path, new Dictionary<string, object?> { ["v"] = new[] { 1.0, 2.0 } }, onedAs: "column");
            Shape([2, 1], MatFile.Load(path)["v"].Dimensions);
            var bad = new MatArray(new NdArray([1.5], 1, 1), MatClass.Int32);
            Assert.Throws<ArgumentException>(() => MatFile.Save(path, new Dictionary<string, MatValue> { ["x"] = bad }));
            var overflow = new MatArray(new NdArray([300.0], 1, 1), MatClass.UInt8);
            Assert.Throws<ArgumentException>(() => MatFile.Save(path, new Dictionary<string, MatValue> { ["x"] = overflow }));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void LargeArray_TransposeIsExact()
    {
        // Above the parallel threshold of the blocked transpose, with ragged edge blocks.
        const int rows = 1031, cols = 1029;
        var data = new double[rows * cols];
        for (var i = 0; i < data.Length; i++) data[i] = i;
        var ms = new MemoryStream();
        MatFile.Save(ms, new Dictionary<string, object?> { ["big"] = new NdArray(data, rows, cols) });
        ms.Position = 0;
        var back = MatFile.Load(ms)["big"].ToNdArray();
        Close(data, back.ToArray(), 0, "big");
    }

    [Fact]
    public void Corrupt_Files_Throw()
    {
        var bytes = File.ReadAllBytes(Fixture("scinet_v5z.mat"));
        var truncated = bytes[..(bytes.Length / 2)];
        Assert.ThrowsAny<InvalidDataException>(() => MatFile.Load(new MemoryStream(truncated)));
        var garbage = (byte[])bytes.Clone();
        for (var i = 200; i < 260; i++) garbage[i] ^= 0x5A;
        Assert.ThrowsAny<InvalidDataException>(() => MatFile.Load(new MemoryStream(garbage)));
    }
}
