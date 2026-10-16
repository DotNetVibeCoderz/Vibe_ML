using System.Text;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Gravicode.Science.SciNet.Sparse;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// Harwell-Boeing tests. hb_rua.rua was written by scipy 1.17.1 hb_write; hb_rsa.rsa and
/// hb_pua.pua are hand-written (SciPy cannot read them: it supports RUA only). Files written
/// here as RUA were read back by scipy.io.hb_read during development and matched exactly.
/// </summary>
public class HarwellBoeingTests
{
    private static readonly NdArray Hb = new([1.0, 0, 0, 2.5, 0, -3.75, 0, 0, 6.0, 0, 1e-10, 0, 0, 4.0e20, 0, 7.0], 4, 4);

    [Fact]
    public void ReadsScipyWrittenRua()
    {
        var m = HarwellBoeing.HbRead(Fixture("hb_rua.rua"), out var info);
        Close(Hb, m.ToDense(), 0, "rua");
        Assert.Equal("RUA", info.MatrixType);
        Assert.Equal("Default title", info.Title);
        Assert.Equal(new FortranFormat(40, 'I', 2, 0), info.PointerFormat);
        Assert.Equal(new FortranFormat(3, 'E', 25, 16), info.ValuesFormat); // SciPy declares E25.16 but writes 24-wide fields
        Assert.Equal(7, info.Nnz);
    }

    [Fact]
    public void ReadsSymmetricWithDExponents_AndPattern()
    {
        var s = HarwellBoeing.HbRead(Fixture("hb_rsa.rsa"));
        Close(new NdArray([4.0, 1, 0, 1, 5, -2, 0, -2, 6], 3, 3), s.ToDense(), 0, "rsa");

        var p = HarwellBoeing.HbRead(Fixture("hb_pua.pua"));
        Close(new NdArray([1.0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], 3, 4), p.ToDense(), 0, "pua");
    }

    [Fact]
    public void FixedWidthFields_PackedWithoutSpaces()
    {
        // Fields of 10 columns with no separating blanks, D exponents, and a three-digit
        // exponent written without its letter (Fortran's form for |exp| >= 100).
        var text =
            "Packed skew-symmetric                                                   RZA1    \n" +
            "             3             1             1             1\n" +
            "RZA                        3             3             3             0\n" +
            "(4I3)           (3I3)           (3D10.3)            \n" +
            "  1  3  4  4\n" +
            "  2  3  3\n" +
            "-0.200D+01 0.100D+01-0.300-100\n";
        var m = HarwellBoeing.HbRead(new MemoryStream(Encoding.ASCII.GetBytes(text)));
        var d = m.ToDense();
        Assert.Equal(-2.0, d[1, 0]);
        Assert.Equal(2.0, d[0, 1]);
        Assert.Equal(1.0, d[2, 0]);
        Assert.Equal(-1.0, d[0, 2]);
        Close(-0.3e-100, d[2, 1], 1e-115);
        Close(0.3e-100, d[1, 2], 1e-115);
    }

    [Theory]
    [InlineData("RUA")]
    [InlineData("RRA")]
    public void Write_RoundTripsExactly(string type)
    {
        var path = TempFile(".rb");
        try
        {
            var rng = new Random(3);
            var dense = new double[30 * 20];
            for (var i = 0; i < dense.Length; i++) dense[i] = rng.NextDouble() < 0.25 ? (rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-300, 300)) : 0;
            var a = CscMatrix.FromDense(new NdArray(dense, 30, 20));
            HarwellBoeing.HbWrite(path, a, title: "random", key: "R1", matrixType: type);
            var back = HarwellBoeing.HbRead(path, out var info);
            Assert.Equal(type, info.MatrixType);
            Assert.Equal("R1", info.Key);
            Close(dense, back.ToDense().ToArray(), 0, "values");
            // every value field is exactly as wide as declared
            var lines = File.ReadAllLines(path);
            foreach (var l in lines.Skip(4 + info.PointerLines + info.IndicesLines)) Assert.Equal(0, l.Length % 25);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Write_SymmetricPatternInteger()
    {
        var path = TempFile(".rb");
        try
        {
            var sym = CsrMatrix.FromDense(new NdArray([4.0, 1, 0, 1, 5, -2, 0, -2, 6], 3, 3));
            HarwellBoeing.HbWrite(path, sym, matrixType: "RSA");
            HarwellBoeing.HbRead(path, out var info);
            Assert.Equal(5, info.Nnz); // lower triangle only
            Close(sym.ToDense(), HarwellBoeing.HbRead(path).ToDense(), 0, "rsa");

            Assert.Throws<ArgumentException>(() => HarwellBoeing.HbWrite(path, CsrMatrix.FromDense(new NdArray([1.0, 2, 3, 4], 2, 2)), matrixType: "RSA"));

            var ints = CsrMatrix.FromDense(new NdArray([1.0, 0, -300, 7], 2, 2));
            HarwellBoeing.HbWrite(path, ints, matrixType: "IUA");
            Close(ints.ToDense(), HarwellBoeing.HbRead(path).ToDense(), 0, "iua");

            HarwellBoeing.HbWrite(path, ints, matrixType: "PUA");
            Close(new NdArray([1.0, 0, 1, 1], 2, 2), HarwellBoeing.HbRead(path).ToDense(), 0, "pua");
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Unsupported_AndMalformed()
    {
        static Stream S(string t) => new MemoryStream(Encoding.ASCII.GetBytes(t));
        var complex =
            "c                                                                       k       \n" +
            "             3             1             1             1\n" +
            "CUA                        1             1             1             0\n" +
            "(2I2)           (1I2)           (2E12.4)            \n";
        Assert.Throws<NotSupportedException>(() => HarwellBoeing.HbRead(S(complex)));
        var shortBody =
            "t                                                                       k       \n" +
            "             3             1             1             1\n" +
            "RUA                        2             2             2             0\n" +
            "(3I2)           (2I2)           (2E12.4)            \n" +
            " 1 2 3\n";
        Assert.Throws<InvalidDataException>(() => HarwellBoeing.HbRead(S(shortBody)));
        Assert.Throws<FormatException>(() => FortranFormat.Parse("(3A8)"));
        Assert.Equal(new FortranFormat(4, 'D', 20, 12), FortranFormat.Parse("(1P,4D20.12)"));
        Assert.Equal(new FortranFormat(1, 'I', 8, 0), FortranFormat.Parse("(I8)"));
    }
}
