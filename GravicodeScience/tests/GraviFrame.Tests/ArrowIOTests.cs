using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Xunit;

namespace Gravicode.Science.Tests.GraviFrame;

/// <summary>
/// Tests for the Arrow IPC reader and writer.
/// </summary>
/// <remarks>
/// <para>
/// A round trip through my own code proves only that the two halves agree with each other, which
/// is worth very little for an interchange format — the whole point is that <em>other</em>
/// implementations can read it. So the pins here are structural: the file layout is checked against
/// the spec byte by byte, and the buffer conventions against what Arrow actually requires.
/// </para>
/// <para>
/// Cross-checking against pyarrow is done separately and is recorded in <c>docs/GraviFrame.md</c>;
/// it cannot run here without a Python dependency in the test project.
/// </para>
/// </remarks>
public class ArrowIOTests
{
    private static DataFrame Mixed() => new(
    [
        new NumericSeries("value", [1.5, -2.25, double.NaN, 1e300, 0.0]),
        new TextSeries("label", ["alpha", "", null, "ünïcødé", "last"]),
        new BooleanSeries("flag", [true, false, null, true, false]),
        new DateTimeSeries("when",
        [
            new DateTime(2024, 3, 1, 12, 30, 45),
            new DateTime(1999, 12, 31, 23, 59, 59),
            null,
            DateTime.UnixEpoch,
            new DateTime(2030, 6, 15),
        ]),
    ]);

    private static DataFrame RoundTrip(DataFrame frame)
    {
        using var stream = new MemoryStream();
        ArrowFile.Write(frame, stream);
        stream.Position = 0;
        return ArrowFile.Read(stream);
    }

    [Fact]
    public void EveryColumnTypeSurvivesTheRoundTrip()
    {
        var back = RoundTrip(Mixed());

        Assert.Equal(5, back.RowCount);
        Assert.Equal(["value", "label", "flag", "when"], back.ColumnNames);

        Assert.Equal(DataType.Numeric, back["value"].DataType);
        Assert.Equal(DataType.Text, back["label"].DataType);
        Assert.Equal(DataType.Boolean, back["flag"].DataType);
        Assert.Equal(DataType.DateTime, back["when"].DataType);

        Assert.Equal(1.5, back.Numeric("value")[0]);
        Assert.Equal(1e300, back.Numeric("value")[3]);
        Assert.Equal("alpha", ((TextSeries)back["label"])[0]);
        Assert.Equal(true, ((BooleanSeries)back["flag"])[0]);
        Assert.Equal(new DateTime(2024, 3, 1, 12, 30, 45), ((DateTimeSeries)back["when"])[0]);
    }

    [Fact]
    public void MissingValuesComeBackMissingInEveryColumn()
    {
        // Arrow records missing-ness in a bitmap rather than as a sentinel value, so this is
        // checking that the bitmap is written and read rather than that NaN survived.
        var back = RoundTrip(Mixed());

        foreach (var name in back.ColumnNames)
            Assert.True(back[name].IsMissing(2), $"'{name}' lost its missing value");
    }

    [Fact]
    public void AnEmptyStringIsNotTheSameAsAMissingOne()
    {
        // The distinction a naive writer loses: both are zero-length in the data buffer, and only
        // the validity bit tells them apart.
        var back = RoundTrip(Mixed());
        var label = (TextSeries)back["label"];

        Assert.False(back["label"].IsMissing(1));
        Assert.Equal("", label[1]);
        Assert.True(back["label"].IsMissing(2));
        Assert.Null(label[2]);
    }

    [Fact]
    public void MultiByteTextSurvives()
    {
        var frame = new DataFrame([new TextSeries("t", ["ünïcødé", "日本語", "🐱🐶", "", null])]);
        var back = (TextSeries)RoundTrip(frame)["t"];

        Assert.Equal("ünïcødé", back[0]);
        Assert.Equal("日本語", back[1]);
        Assert.Equal("🐱🐶", back[2]);
    }

    [Fact]
    public void CategoricalColumnsAreWrittenAsTheirStrings()
    {
        // A dictionary-encoded Arrow array is a separate message type with its own bookkeeping,
        // and writing one badly produces a file that loads with silently wrong values. Expanding
        // is the honest fallback, and the values must survive it.
        var frame = new DataFrame(
        [
            CategoricalSeries.FromValues("size", ["low", "high", null, "low"],
                categories: ["low", "medium", "high"], ordered: true),
        ]);

        var back = RoundTrip(frame);

        Assert.Equal(DataType.Text, back["size"].DataType);
        Assert.Equal("low", ((TextSeries)back["size"])[0]);
        Assert.Equal("high", ((TextSeries)back["size"])[1]);
        Assert.True(back["size"].IsMissing(2));
    }

    // ---------------------------------------------------------------- file layout

    private static byte[] Bytes(DataFrame frame)
    {
        using var stream = new MemoryStream();
        ArrowFile.Write(frame, stream);
        return stream.ToArray();
    }

    [Fact]
    public void TheFileIsBookendedByTheArrowMagic()
    {
        var bytes = Bytes(Mixed());
        var magic = "ARROW1"u8.ToArray();

        Assert.Equal(magic, bytes.Take(6));
        Assert.Equal(magic, bytes.Skip(bytes.Length - 6));
    }

    [Fact]
    public void TheTrailingFooterLengthLocatesTheFooter()
    {
        // What makes the format seekable: a reader jumps to the end and works backwards, so a
        // wrong length here means nothing can be found at all.
        var bytes = Bytes(Mixed());
        var footerLength = BitConverter.ToInt32(bytes, bytes.Length - 10);

        Assert.True(footerLength > 0, "the footer length was not written");
        Assert.True(footerLength < bytes.Length - 16, $"the footer length {footerLength} is impossible");

        var footerStart = bytes.Length - 10 - footerLength;
        Assert.True(footerStart > 0 && footerStart % 8 == 0,
            $"the footer starts at {footerStart}, which is not 8-aligned");
    }

    [Fact]
    public void EveryMessageCarriesTheContinuationMarker()
    {
        // The 0xFFFFFFFF prefix distinguishes the current encapsulated format from the pre-0.15
        // one. Omitting it produces a file that only old readers accept.
        var bytes = Bytes(Mixed());
        Assert.Equal(0xFFFFFFFF, BitConverter.ToUInt32(bytes, 8));
    }

    [Fact]
    public void MessageMetadataLengthsArePaddedToEightBytes()
    {
        var bytes = Bytes(Mixed());
        var schemaLength = BitConverter.ToInt32(bytes, 12);

        Assert.True(schemaLength % 8 == 0,
            $"the schema metadata is {schemaLength} bytes, which is not a multiple of 8");
    }

    [Fact]
    public void AllTablesAndTheRootAreFourByteAligned()
    {
        // The bug that cost the most to find: FlatBuffers tables begin with an int32 soffset, so a
        // table that starts two bytes out still decodes by hand and is rejected outright by Arrow's
        // verifier. Nothing about the failure points at alignment.
        var bytes = Bytes(Mixed());

        var footerLength = BitConverter.ToInt32(bytes, bytes.Length - 10);
        var footerStart = bytes.Length - 10 - footerLength;

        var root = BitConverter.ToInt32(bytes, footerStart);
        Assert.True(root % 4 == 0, $"the footer root table is at {root}, which is not 4-aligned");

        var schemaStart = 8 + 8;
        var schemaRoot = BitConverter.ToInt32(bytes, schemaStart);
        Assert.True(schemaRoot % 4 == 0,
            $"the schema root table is at {schemaRoot}, which is not 4-aligned");
    }

    [Fact]
    public void AnEmptyFrameStillProducesAReadableFile()
    {
        var frame = new DataFrame(
        [
            new NumericSeries("a", []),
            new TextSeries("b", []),
        ]);

        var back = RoundTrip(frame);

        Assert.Equal(0, back.RowCount);
        Assert.Equal(["a", "b"], back.ColumnNames);
    }

    [Fact]
    public void ASingleRowWorks()
    {
        var frame = new DataFrame([new NumericSeries("only", [42.0])]);
        var back = RoundTrip(frame);

        Assert.Equal(1, back.RowCount);
        Assert.Equal(42.0, back.Numeric("only")[0]);
    }

    [Fact]
    public void AWideFrameKeepsEveryColumnDistinct()
    {
        // Buffers are consumed positionally, so an off-by-one in the buffer index would shift every
        // column after the first and still produce a readable file.
        var columns = Enumerable.Range(0, 40)
            .Select(i => (Series)new NumericSeries($"c{i}", [i, i * 2.0]))
            .ToList();

        var back = RoundTrip(new DataFrame(columns));

        Assert.Equal(40, back.ColumnNames.Count);
        for (var i = 0; i < 40; i++)
        {
            Assert.Equal($"c{i}", back.ColumnNames[i]);
            Assert.Equal(i, back.Numeric($"c{i}")[0]);
            Assert.Equal(i * 2.0, back.Numeric($"c{i}")[1]);
        }
    }

    [Fact]
    public void ALongerFrameRoundTripsExactly()
    {
        var rng = new GraviNum.GraviRandom(11);
        var values = new double[5000];
        var text = new string?[5000];

        for (var i = 0; i < 5000; i++)
        {
            values[i] = i % 17 == 0 ? double.NaN : rng.Normal();
            text[i] = i % 13 == 0 ? null : $"row-{i}";
        }

        var back = RoundTrip(new DataFrame(
        [
            new NumericSeries("v", values),
            new TextSeries("t", text),
        ]));

        Assert.Equal(5000, back.RowCount);

        for (var i = 0; i < 5000; i++)
        {
            if (double.IsNaN(values[i])) Assert.True(back["v"].IsMissing(i));
            else Assert.Equal(values[i], back.Numeric("v")[i], 12);

            Assert.Equal(text[i], ((TextSeries)back["t"])[i]);
        }
    }

    [Fact]
    public void RoundTrippingThroughAFileOnDiskWorks()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gravi_{Guid.NewGuid():N}.arrow");
        try
        {
            ArrowFile.Write(Mixed(), path);
            var back = ArrowFile.Read(path);

            Assert.Equal(5, back.RowCount);
            Assert.Equal("alpha", ((TextSeries)back["label"])[0]);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void SomethingThatIsNotAnArrowFileIsRejected()
    {
        using var stream = new MemoryStream("not an arrow file at all, just text"u8.ToArray());
        Assert.Throws<InvalidDataException>(() => ArrowFile.Read(stream));
    }

    [Fact]
    public void ANonSeekableStreamIsRefused()
    {
        // The format is read back to front, so a forward-only stream cannot work. Saying so beats
        // reading half a file and failing somewhere confusing.
        using var inner = new MemoryStream(Bytes(Mixed()));
        using var forwardOnly = new ForwardOnlyStream(inner);

        Assert.Throws<ArgumentException>(() => ArrowFile.Read(forwardOnly));
    }

    private sealed class ForwardOnlyStream(Stream inner) : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();

        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
