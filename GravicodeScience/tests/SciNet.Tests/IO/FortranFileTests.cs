using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// Fortran unformatted-file tests. fortran_*.dat were written by scipy 1.17.1 FortranFile;
/// scipy/fortran-*.dat come from SciPy's test suite (BSD-3-Clause), fortran-3x3d-2i.dat
/// written by gfortran itself. Files written here were read back by scipy during development.
/// </summary>
public class FortranFileTests
{
    private static string Scipy(string name) => Fixture(Path.Combine("scipy", name));

    [Fact]
    public void ReadsScipyTestSuiteFiles()
    {
        // fortran-sf8-1x3x5.dat holds arange(15).reshape(1,3,5) as Fortran writes it (column-major).
        using (var f = new FortranFile(Scipy("fortran-sf8-1x3x5.dat")))
        {
            var a = f.ReadRecord().ReadArray<double>(1, 3, 5);
            Close(Enumerable.Range(0, 15).Select(i => (double)i).ToArray(), a.ToArray(), 0, "sf8");
            Assert.Throws<FortranEofException>(() => f.ReadRecord());
        }
        using (var f = new FortranFile(Scipy("fortran-si4-1x3x5.dat")))
        {
            var a = f.ReadRecord().ReadArray<int>(1, 3, 5);
            Close(Enumerable.Range(0, 15).Select(i => (double)i).ToArray(), a.ToArray(), 0, "si4");
        }
        using (var f = new FortranFile(Scipy("fortran-mixed.dat")))
        {
            // scipy: read_record('<i4,<f4,<i8,2<f8') → 1, 2.3, 4, [5.6, 7.8]
            var r = f.ReadRecord();
            Assert.Equal(1, r.Read<int>(1)[0]);
            Assert.Equal(2.3f, r.Read<float>(1)[0]);
            Assert.Equal(4L, r.Read<long>(1)[0]);
            Assert.Equal([5.6, 7.8], r.Read<double>(2));
            Assert.Equal(0, r.Remaining);
        }
        using (var f = new FortranFile(Scipy("fortran-3x3d-2i.dat")))
        {
            // gfortran: double precision :: a(3,3); integer :: b(2); write(1) a, b
            var r = f.ReadRecord();
            var a = r.ReadArray<double>(3, 3);
            Close(Enumerable.Range(0, 9).Select(i => (double)i).ToArray(), a.ToArray(), 0, "a");
            Assert.Equal([-1, -2], r.Read<int>(2));
        }
    }

    [Fact]
    public void ReadsScipyWrittenFiles()
    {
        using (var f = new FortranFile(Fixture("fortran_mixed4.dat")))
        {
            Assert.Equal([1, -2, 3, 2147483647], f.ReadInts());
            Assert.Equal([0.5, -1.25, 1e300], f.ReadReals());
            var r = f.ReadRecord();
            Assert.Equal([7, 8], r.Read<int>(2));
            Assert.Equal([2.5f, 3.5f, 4.5f], r.Read<float>());
            Assert.Equal([1L << 40, -5L], f.ReadRecord<long>());
            Assert.True(f.EndOfFile);
        }
        using (var f = new FortranFile(Fixture("fortran_h8.dat"), headerSize: 8))
        {
            Assert.Equal([0.0, 1, 2, 3, 4], f.ReadReals());
            Assert.Equal([42], f.ReadInts());
        }
        // Reading an 8-byte-header file with 4-byte markers is caught, not misread.
        using (var f = new FortranFile(Fixture("fortran_h8.dat")))
            Assert.Throws<FortranFormattingException>(() => f.ReadReals());
    }

    [Theory]
    [InlineData(4, false)]
    [InlineData(8, false)]
    [InlineData(4, true)]
    public void WriteThenRead_RoundTrips(int header, bool bigEndian)
    {
        var ms = new MemoryStream();
        using (var f = new FortranFile(ms, header, bigEndian, leaveOpen: true))
        {
            f.WriteRecord(new[] { 1, -2, 3 });
            f.WriteRecord(new[] { 7, 8 }, new[] { 2.5f, 3.5f }, new[] { 1e-300 });
            f.WriteRecord(new short[,] { { 1, 2 }, { 3, 4 } });
            f.WriteRecord(new NdArray([1.0, 2, 3, 4, 5, 6], 2, 3), fortranOrder: true);
        }
        Assert.Equal(4 * 2 * header + 12 + (8 + 8 + 8) + 8 + 48, ms.Length);
        ms.Position = 0;
        using var g = new FortranFile(ms, header, bigEndian);
        Assert.Equal([1, -2, 3], g.ReadInts());
        var r = g.ReadRecord();
        Assert.Equal([7, 8], r.Read<int>(2));
        Assert.Equal([2.5f, 3.5f], r.Read<float>(2));
        Assert.Equal([1e-300], r.Read<double>());
        Assert.Equal([(short)1, (short)2, (short)3, (short)4], g.ReadRecord<short>());
        var a = g.ReadRecord().ReadArray<double>(2, 3);
        Close([1.0, 2, 3, 4, 5, 6], a.ToArray(), 0);
        Assert.Throws<FortranEofException>(() => g.ReadRecord());
    }

    [Fact]
    public void Malformed_Throw()
    {
        // header says 8 bytes, footer says 4
        var bad = new byte[] { 8, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 0, 0, 0 };
        using (var f = new FortranFile(new MemoryStream(bad)))
            Assert.Throws<FortranFormattingException>(() => f.ReadRecord());
        // truncated payload
        var cut = new byte[] { 16, 0, 0, 0, 1, 2, 3, 4 };
        using (var f = new FortranFile(new MemoryStream(cut)))
            Assert.Throws<FortranFormattingException>(() => f.ReadRecord());
        // size not a multiple of the element type
        var odd = new byte[] { 6, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 0, 0, 0 };
        using (var f = new FortranFile(new MemoryStream(odd)))
            Assert.Throws<FortranFormattingException>(() => f.ReadInts());
    }
}
