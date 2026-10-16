using System.Numerics;
using System.Text;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Gravicode.Science.SciNet.Sparse;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// Matrix Market tests. The mm_*.mtx fixtures were written by scipy 1.17.1 (mmwrite) and the
/// expected headers are its mminfo output. Files written here were read back by
/// scipy.io.mmread during development and matched exactly (general/symmetric/skew/hermitian,
/// coordinate and array).
/// </summary>
public class MatrixMarketTests
{
    private static readonly double[] A =
    [
        1.5, 0, 0, -2.0,
        0, 3.25, 0, 0,
        7.0, 0, 0, 1e-300,
        0, 0, 4.0, 0,
        0, 6.0, 0, 0,
    ];

    [Fact]
    public void MmInfo_MatchesScipy()
    {
        // scipy 1.17.1 mminfo on each fixture.
        Assert.Equal(new MatrixMarketInfo(5, 4, 7, "coordinate", "real", "general"), MatrixMarket.MmInfo(Fixture("mm_general.mtx")));
        Assert.Equal(new MatrixMarketInfo(3, 3, 5, "coordinate", "real", "symmetric"), MatrixMarket.MmInfo(Fixture("mm_symmetric.mtx")));
        Assert.Equal(new MatrixMarketInfo(3, 3, 3, "coordinate", "real", "skew-symmetric"), MatrixMarket.MmInfo(Fixture("mm_skew.mtx")));
        Assert.Equal(new MatrixMarketInfo(5, 4, 7, "coordinate", "pattern", "general"), MatrixMarket.MmInfo(Fixture("mm_pattern.mtx")));
        Assert.Equal(new MatrixMarketInfo(2, 3, 3, "coordinate", "integer", "general"), MatrixMarket.MmInfo(Fixture("mm_integer.mtx")));
        Assert.Equal(new MatrixMarketInfo(3, 3, 5, "coordinate", "complex", "hermitian"), MatrixMarket.MmInfo(Fixture("mm_hermitian.mtx")));
        Assert.Equal(new MatrixMarketInfo(2, 3, 6, "array", "real", "general"), MatrixMarket.MmInfo(Fixture("mm_array.mtx")));
        Assert.Equal(new MatrixMarketInfo(3, 3, 9, "array", "real", "symmetric"), MatrixMarket.MmInfo(Fixture("mm_array_sym.mtx")));
        Assert.Equal(new MatrixMarketInfo(2, 2, 4, "array", "complex", "general"), MatrixMarket.MmInfo(Fixture("mm_array_complex.mtx")));
    }

    [Fact]
    public void MmRead_ScipyFixtures()
    {
        var g = MatrixMarket.MmRead(Fixture("mm_general.mtx"));
        Assert.True(g.IsSparse);
        Close(new NdArray(A, 5, 4), g.ToDense(), 0, "general");
        Assert.Equal(7, g.Sparse!.Nnz);

        var s = MatrixMarket.MmRead(Fixture("mm_symmetric.mtx"));
        Close(new NdArray([4.0, 1, 0, 1, 5, 2, 0, 2, 6], 3, 3), s.ToDense(), 0, "symmetric");
        Assert.Equal(7, s.Sparse!.Nnz); // 5 stored + 2 mirrored, like scipy's coo

        var k = MatrixMarket.MmRead(Fixture("mm_skew.mtx"));
        Close(new NdArray([0, 2.0, -1, -2, 0, 3, 1, -3, 0], 3, 3), k.ToDense(), 0, "skew");

        var p = MatrixMarket.MmRead(Fixture("mm_pattern.mtx"));
        var pd = p.ToDense().ToArray();
        for (var i = 0; i < A.Length; i++) Assert.Equal(A[i] != 0 ? 1.0 : 0.0, pd[i]);

        var n = MatrixMarket.MmRead(Fixture("mm_integer.mtx"));
        Close(new NdArray([1.0, 0, -3, 0, 12345678901, 0], 2, 3), n.ToDense(), 0, "integer");

        var h = MatrixMarket.MmRead(Fixture("mm_hermitian.mtx"));
        Assert.True(h.IsComplex);
        var hd = Dense(h);
        Complex[,] H = { { 2, new(1, -1), 0 }, { new(1, 1), 3, new(0, 2) }, { 0, new(0, -2), 1 } };
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
                Close(H[i, j], hd[i, j], 0, $"H[{i},{j}]");

        Close(new NdArray([0, 0.25, 0.5, 0.75, 1.0, 1.25], 2, 3), MatrixMarket.MmRead(Fixture("mm_array.mtx")).Dense!, 0, "array");
        Close(new NdArray([4.0, 1, 0, 1, 5, 2, 0, 2, 6], 3, 3), MatrixMarket.MmRead(Fixture("mm_array_sym.mtx")).Dense!, 0, "array sym");
        var ac = MatrixMarket.MmRead(Fixture("mm_array_complex.mtx")).DenseComplex!;
        Close(new Complex(1, 1), ac[0, 0], 0);
        Close(new Complex(0, 3), ac[1, 0], 0);
        Close(new Complex(-4, -0.5), ac[1, 1], 0);
    }

    private static Complex[,] Dense(MatrixMarketData d)
    {
        var r = new Complex[d.Info.Rows, d.Info.Columns];
        var s = d.Sparse!;
        var im = d.SparseImaginary;
        for (var k = 0; k < s.Nnz; k++) r[s.Row[k], s.Col[k]] += new Complex(s.Data[k], im[k]);
        return r;
    }

    [Fact]
    public void Write_AutoDetectsSymmetry_LikeScipy()
    {
        var path = TempFile(".mtx");
        try
        {
            var sym = new NdArray([4.0, 1, 0, 1, 5, 2, 0, 2, 6], 3, 3);
            MatrixMarket.MmWrite(path, sym);
            Assert.Equal("symmetric", MatrixMarket.MmInfo(path).Symmetry);
            Close(sym, MatrixMarket.MmRead(path).Dense!, 0);

            MatrixMarket.MmWrite(path, CsrMatrix.FromDense(sym));
            var info = MatrixMarket.MmInfo(path);
            Assert.Equal(("coordinate", "symmetric", 5L), (info.Format, info.Symmetry, info.Entries));
            Close(sym, MatrixMarket.MmRead(path).ToDense(), 0);

            var skew = new NdArray([0, 2.0, -1, -2, 0, 3, 1, -3, 0], 3, 3);
            MatrixMarket.MmWrite(path, CsrMatrix.FromDense(skew));
            Assert.Equal("skew-symmetric", MatrixMarket.MmInfo(path).Symmetry);
            Close(skew, MatrixMarket.MmRead(path).ToDense(), 0);
            MatrixMarket.MmWrite(path, skew);
            Assert.Equal(3, File.ReadAllLines(path).Length - 2); // banner + size line + the strictly lower triangle
            Close(skew, MatrixMarket.MmRead(path).Dense!, 0);

            MatrixMarket.MmWrite(path, sym, symmetry: "general");
            Assert.Equal("general", MatrixMarket.MmInfo(path).Symmetry);

            var herm = ComplexNdArray.FromParts(new NdArray([2.0, 1, 1, 3], 2, 2), new NdArray([0, -1.0, 1, 0], 2, 2));
            MatrixMarket.MmWrite(path, herm);
            Assert.Equal("hermitian", MatrixMarket.MmInfo(path).Symmetry);
            Assert.True(herm.AllClose(MatrixMarket.MmRead(path).DenseComplex!, 0));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Write_RoundTripsExactly_AllFields()
    {
        var path = TempFile(".mtx");
        try
        {
            var rng = new Random(7);
            var dense = new double[40 * 30];
            for (var i = 0; i < dense.Length; i++) dense[i] = rng.NextDouble() < 0.2 ? (rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-300, 300)) : 0;
            var a = new NdArray(dense, 40, 30);
            MatrixMarket.MmWrite(path, CscMatrix.FromDense(a), comment: "line one\nline two");
            Assert.Equal("%line one", File.ReadLines(path).Skip(1).First());
            Close(a, MatrixMarket.MmRead(path).ToDense(), 0, "coordinate round trip");

            MatrixMarket.MmWrite(path, a);
            Close(a, MatrixMarket.MmRead(path).Dense!, 0, "array round trip");

            var ints = new NdArray([1.0, 0, -3, 0, 12345678901, 7], 2, 3);
            MatrixMarket.MmWrite(path, CsrMatrix.FromDense(ints), field: "integer");
            Assert.Equal("integer", MatrixMarket.MmInfo(path).Field);
            Assert.Contains("2 2 12345678901", File.ReadAllText(path));
            Close(ints, MatrixMarket.MmRead(path).ToDense(), 0);
            Assert.Throws<ArgumentException>(() => MatrixMarket.MmWrite(path, new NdArray([1.5], 1, 1), field: "integer"));

            MatrixMarket.MmWrite(path, CsrMatrix.FromDense(ints), field: "pattern");
            var pat = MatrixMarket.MmRead(path).ToDense().ToArray();
            Assert.Equal([1.0, 0, 1, 0, 1, 1], pat);

            MatrixMarket.MmWrite(path, new NdArray([Math.PI], 1, 1), precision: 3);
            Assert.Equal(3.14, MatrixMarket.MmRead(path).Dense![0, 0]);

            var re = CsrMatrix.FromDense(new NdArray([1.0, 0, 2, 0, 0, 3], 2, 3));
            MatrixMarket.MmWrite(path, re, [0.5, -1, 2]);
            var z = Dense(MatrixMarket.MmRead(path));
            Close(new Complex(2, -1), z[0, 2], 0);
            Close(new Complex(3, 2), z[1, 2], 0);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ParallelParse_MatchesSerialReference()
    {
        // ~2.5 MB body: above the parallel threshold, so chunk boundaries fall mid-file.
        const int n = 100_000;
        var rng = new Random(11);
        var sb = new StringBuilder("%%MatrixMarket matrix coordinate real general\n% big\n\n");
        sb.Append($"5000 4000 {n}\n");
        var rows = new int[n];
        var cols = new int[n];
        var vals = new double[n];
        for (var k = 0; k < n; k++)
        {
            rows[k] = rng.Next(5000);
            cols[k] = rng.Next(4000);
            vals[k] = Math.Round((rng.NextDouble() - 0.5) * 1e6, 3);
            sb.Append(rows[k] + 1).Append(' ').Append(cols[k] + 1).Append("   ").Append(vals[k].ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            sb.Append(k % 1000 == 0 ? "\r\n" : "\n"); // tolerate CRLF
        }
        var d = MatrixMarket.MmRead(new MemoryStream(Encoding.ASCII.GetBytes(sb.ToString())));
        var s = d.Sparse!;
        Assert.Equal(n, s.Nnz);
        Assert.Equal(rows, s.Row.ToArray());
        Assert.Equal(cols, s.Col.ToArray());
        Assert.Equal(vals, s.Data.ToArray());

        // and back out through the parallel writer
        var ms = new MemoryStream();
        MatrixMarket.MmWrite(ms, s, symmetry: "general");
        ms.Position = 0;
        var again = MatrixMarket.MmRead(ms).Sparse!;
        Assert.Equal(vals, again.Data.ToArray());
        Assert.Equal(rows, again.Row.ToArray());
    }

    [Fact]
    public void NumberParsing_IsCorrectlyRounded()
    {
        // The reader's fast decimal parser (Clinger + Eisel-Lemire, BCL fallback) must agree bit
        // for bit with double.Parse. Includes 17-19 digit mantissas, extreme exponents,
        // subnormals, and an exact halfway case with trailing zeros — which
        // System.Buffers.Text.Utf8Parser rounds the wrong way (…265.000 → …266, not …264).
        var rng = new Random(2024);
        var fmts = new[] { "R", "G17", "E16", "G15", "G6", "E3", "F10" };
        var tokens = new List<string>
        {
            "-17965106369622265.000", "9007199254740993", "2.2250738585072011e-308", "1.7976931348623157e308",
            "4.9e-324", "2.4703282292062327e-324", "1e23", "123456789012345678", "1234567890123456789",
            "0.000000000000000000000000000001", "1.00000000000000011102230246251565404236316680908203125", "-0", "7.2057594037927933e16",
        };
        for (var i = 0; i < 60000; i++)
        {
            double x = (i % 3) switch
            {
                0 => rng.NextDouble() * 2 - 1,
                1 => BitConverter.Int64BitsToDouble(rng.NextInt64()),
                _ => (rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-320, 308)),
            };
            if (double.IsFinite(x)) tokens.Add(x.ToString(fmts[i % fmts.Length], System.Globalization.CultureInfo.InvariantCulture));
        }
        for (var i = 0; i < 20000; i++)
        {
            var sb = new StringBuilder();
            var nd = rng.Next(1, 21);
            var dot = rng.Next(nd + 1);
            for (var k = 0; k < nd; k++) { if (k == dot) sb.Append('.'); sb.Append((char)('0' + rng.Next(10))); }
            if (rng.Next(2) == 0) sb.Append('e').Append(rng.Next(-340, 320));
            tokens.Add(sb.ToString());
        }
        var text = new StringBuilder($"%%MatrixMarket matrix array real general\n{tokens.Count} 1\n");
        foreach (var t in tokens) text.Append(t).Append('\n');
        var d = MatrixMarket.MmRead(new MemoryStream(Encoding.ASCII.GetBytes(text.ToString()))).Dense!.ToArray();
        for (var i = 0; i < tokens.Count; i++)
        {
            var expected = double.Parse(tokens[i], System.Globalization.CultureInfo.InvariantCulture);
            Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(d[i]), $"'{tokens[i]}': expected {expected:R}, got {d[i]:R}");
        }
    }

    [Fact]
    public void Malformed_Throws()
    {
        static MatrixMarketData Read(string text) => MatrixMarket.MmRead(new MemoryStream(Encoding.ASCII.GetBytes(text)));
        Assert.Throws<InvalidDataException>(() => Read("not a header\n1 1 1\n1 1 1\n"));
        Assert.Throws<InvalidDataException>(() => Read("%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n")); // too few
        Assert.Throws<InvalidDataException>(() => Read("%%MatrixMarket matrix coordinate real general\n2 2 1\n3 1 1\n")); // out of range
        Assert.Throws<InvalidDataException>(() => Read("%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 abc\n"));
        Assert.Throws<InvalidDataException>(() => Read("%%MatrixMarket matrix coordinate real symmetric\n2 3 0\n"));
        Assert.Throws<NotSupportedException>(() => Read("%%MatrixMarket vector coordinate real general\n2 1\n"));
        // Fortran D exponents and inf are accepted
        var ok = Read("%%MatrixMarket matrix array real general\n2 1\n1.5D+02\ninf\n").Dense!;
        Assert.Equal(150.0, ok[0, 0]);
        Assert.Equal(double.PositiveInfinity, ok[1, 0]);
    }
}
