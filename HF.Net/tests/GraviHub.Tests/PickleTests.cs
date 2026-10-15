using Gravicode.HFNet.GraviHub.Io;
using Xunit;

namespace Gravicode.HFNet.GraviHub.Tests;

/// <summary>
/// Data pickles of the kind Geneformer publishes - dicts of gene ids to NumPy scalars - read by the
/// same restricted reader the PyTorch checkpoints go through.
/// </summary>
/// <remarks>
/// <c>Fixtures/make_pickles.py</c> wrote the files with NumPy 2.4 (which names the module
/// <c>numpy._core</c>; NumPy 1.x, which wrote Geneformer's own files, says <c>numpy.core</c>). The
/// expected values are written out by hand from that script and are exact in their dtypes.
/// </remarks>
public sealed class PickleTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    public static TheoryData<string> Protocols => new() { "numpy-scalars-p2.pkl", "numpy-scalars-p4.pkl", "numpy-scalars-p5.pkl" };

    [Theory]
    [MemberData(nameof(Protocols))]
    public void NumPy_scalars_come_back_as_plain_values(string file)
    {
        var root = Assert.IsType<Dictionary<object, object?>>(PickleReader.LoadFile(Path.Combine(Fixtures, file)));

        Assert.Equal(1.5, root["ENSG00000000003"]);          // float32
        Assert.Equal(-2.25, root["ENSG00000000005"]);        // float64
        Assert.Equal(0.5, root["half"]);                     // float16
        Assert.Equal(7L, root["count"]);                     // int64
        Assert.Equal(-3L, root["small"]);                    // int8, sign kept
        Assert.Equal(65535L, root["u16"]);                   // uint16
        Assert.Equal(9223372036854775813UL, root["big"]);    // uint64 above long.MaxValue
        Assert.Equal(true, root["flag"]);                    // bool_
        Assert.Equal(3.5, root["big_endian"]);               // >f8, bytes turned
        Assert.Equal("abc", root["name"]);                   // str_, UTF-32 with padding removed
        Assert.Equal(0.125, root["plain"]);                  // a Python float beside them
    }

    [Fact]
    public void A_token_dictionary_of_plain_ints_reads_as_longs()
    {
        var root = Assert.IsType<Dictionary<object, object?>>(PickleReader.LoadFile(Path.Combine(Fixtures, "token-dictionary.pkl")));

        Assert.Equal(4, root.Count);
        Assert.Equal(0L, root["<pad>"]);
        Assert.Equal(3L, root["ENSG00000000005"]);
    }

    [Fact]
    public void Any_other_global_is_still_refused()
    {
        // datetime.date is harmless, but it is a callable outside the allow-list, and a reader that
        // admitted it would admit os.system by the same rule.
        var error = Assert.Throws<NotSupportedException>(() => PickleReader.LoadFile(Path.Combine(Fixtures, "not-data.pkl")));
        Assert.Contains("datetime.date", error.Message);
    }

    [Fact]
    public void A_dangerous_global_is_refused_before_anything_runs()
    {
        // Protocol 0: GLOBAL os.system, then ('echo',) and REDUCE - the classic payload.
        var payload = "cos\nsystem\n(S'echo pwned'\ntR."u8.ToArray();
        var error = Assert.Throws<NotSupportedException>(() => new PickleReader(new MemoryStream(payload)).Load());
        Assert.Contains("os.system", error.Message);
    }

    [Fact]
    public void An_unsupported_NumPy_kind_says_what_is_read()
    {
        // A complex64 scalar, written by hand: GLOBAL scalar, dtype('c8', False, True), eight bytes of data.
        var bytes = new List<byte> { 0x80, 0x02 };                                     // PROTO 2
        bytes.AddRange("cnumpy.core.multiarray\nscalar\ncnumpy\ndtype\n"u8.ToArray()); // two GLOBALs
        bytes.AddRange([(byte)'X', 2, 0, 0, 0, (byte)'c', (byte)'8']);                 // BINUNICODE 'c8'
        bytes.AddRange([0x89, 0x88, 0x87, (byte)'R']);                                 // False, True, TUPLE3, REDUCE
        bytes.AddRange([(byte)'C', 8, 0, 0, 0, 0, 0, 0, 0, 0]);                       // SHORT_BINBYTES, 8 zero bytes
        bytes.AddRange([0x86, (byte)'R', (byte)'.']);                                  // TUPLE2, REDUCE, STOP

        var error = Assert.Throws<NotSupportedException>(() => new PickleReader(new MemoryStream([.. bytes])).Load());
        Assert.Contains("c8", error.Message);
    }
}
