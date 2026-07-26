using System.Text;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Datasets;

namespace BlazorML.Tests;

/// <summary>
/// Every dataset entering or leaving the workspace goes through here, so a round trip that loses
/// a value, a type or a gap corrupts everything downstream of it.
/// </summary>
public class SerializerTests
{
    private static TabularData Sample()
    {
        var table = TabularData.WithColumns("nama", "umur", "aktif", "catatan");
        table.AddRow("Ani", 34.5, true, "berisi, koma");
        table.AddRow("Budi", 28.0, false, "berisi \"kutip\"");
        table.AddRow("Citra", null, true, null);

        return table;
    }

    private static async Task<TabularData> RoundTripAsync(TabularData input, DatasetFormat format)
    {
        await using var written = await TabularSerializer.WriteAsync(input, format);
        written.Position = 0;

        return await TabularSerializer.ReadAsync(written, format);
    }

    [Theory]
    [InlineData(DatasetFormat.Csv)]
    [InlineData(DatasetFormat.Tsv)]
    [InlineData(DatasetFormat.Json)]
    [InlineData(DatasetFormat.Parquet)]
    public async Task A_round_trip_preserves_the_shape_and_the_values(DatasetFormat format)
    {
        var output = await RoundTripAsync(Sample(), format);

        Assert.Equal(3, output.RowCount);
        Assert.Equal(4, output.ColumnCount);
        Assert.Equal(["nama", "umur", "aktif", "catatan"], output.Columns.Select(c => c.Name));

        Assert.Equal("Ani", output.Text(0, "nama"));
        Assert.Equal(34.5, TabularData.ToNumber(output.Value(0, "umur")));
        Assert.Equal("berisi, koma", output.Text(0, "catatan"));
        Assert.Equal("berisi \"kutip\"", output.Text(1, "catatan"));
    }

    [Theory]
    [InlineData(DatasetFormat.Csv)]
    [InlineData(DatasetFormat.Json)]
    [InlineData(DatasetFormat.Parquet)]
    public async Task A_gap_survives_a_round_trip_as_a_gap(DatasetFormat format)
    {
        // A missing value that comes back as 0 or "" would quietly change what a model learns.
        var output = await RoundTripAsync(Sample(), format);

        Assert.True(TabularData.IsBlank(output.Value(2, "umur")));
        Assert.True(TabularData.IsBlank(output.Value(2, "catatan")));
    }

    [Fact]
    public async Task Parquet_keeps_column_types_rather_than_flattening_everything_to_text()
    {
        // This is the reason to support Parquet at all: unlike CSV it carries its own schema, so
        // a numeric column arrives numeric without needing to be re-inferred from the values.
        var output = await RoundTripAsync(Sample(), DatasetFormat.Parquet);

        Assert.Equal(ColumnDataType.Numeric, output.Column("umur")!.Value.DataType);
        Assert.Equal(ColumnDataType.Boolean, output.Column("aktif")!.Value.DataType);
        Assert.Equal(true, output.Value(0, "aktif"));
        Assert.Equal(false, output.Value(1, "aktif"));
    }

    [Fact]
    public async Task Parquet_is_substantially_smaller_than_the_same_data_as_csv()
    {
        var table = TabularData.WithColumns("id", "nilai", "kategori");

        for (var i = 0; i < 5000; i++)
        {
            table.AddRow(i, i * 1.5, i % 3 == 0 ? "alpha" : "beta");
        }

        await using var csv = await TabularSerializer.WriteAsync(table, DatasetFormat.Csv);
        await using var parquet = await TabularSerializer.WriteAsync(table, DatasetFormat.Parquet);

        Assert.True(parquet.Length < csv.Length,
            $"Parquet came out at {parquet.Length:N0} bytes against CSV's {csv.Length:N0}.");
    }

    [Fact]
    public async Task Parquet_reads_from_a_stream_that_cannot_seek()
    {
        // Uploads and HTTP responses are forward-only, and Parquet needs its footer, so the
        // reader has to buffer rather than throw.
        await using var written = await TabularSerializer.WriteAsync(Sample(), DatasetFormat.Parquet);
        await using var forwardOnly = new ForwardOnlyStream(((MemoryStream)written).ToArray());

        var output = await TabularSerializer.ReadAsync(forwardOnly, DatasetFormat.Parquet);

        Assert.Equal(3, output.RowCount);
    }

    [Fact]
    public async Task A_row_limit_returns_only_that_many_rows()
    {
        var table = TabularData.WithColumns("n");
        for (var i = 0; i < 100; i++)
        {
            table.AddRow(i);
        }

        foreach (var format in (DatasetFormat[])[DatasetFormat.Csv, DatasetFormat.Json, DatasetFormat.Parquet])
        {
            await using var written = await TabularSerializer.WriteAsync(table, format);
            written.Position = 0;

            var output = await TabularSerializer.ReadAsync(written, format, rowLimit: 10);

            Assert.Equal(10, output.RowCount);
        }
    }

    /// <summary>
    /// Counting the rows that came back does not show the reader stopped — trimming a fully
    /// parsed table would pass that just as well. This measures how much of the stream was
    /// actually consumed, which is the property the dataset preview depends on: showing the top
    /// of a large upload has to cost the top of it, not all of it.
    /// </summary>
    [Theory]
    [InlineData(DatasetFormat.Csv)]
    [InlineData(DatasetFormat.Tsv)]
    [InlineData(DatasetFormat.Json)]
    public async Task A_row_limit_leaves_most_of_a_large_file_unread(DatasetFormat format)
    {
        var table = TabularData.WithColumns("n", "catatan");

        for (var i = 0; i < 20_000; i++)
        {
            table.AddRow(i, $"baris ke-{i} dengan cukup teks supaya berkasnya besar");
        }

        await using var written = await TabularSerializer.WriteAsync(table, format);
        written.Position = 0;

        var counting = new CountingStream(written);
        var output = await TabularSerializer.ReadAsync(counting, format, rowLimit: 25);

        Assert.Equal(25, output.RowCount);

        // Generous: readers buffer, so the point is the order of magnitude, not a byte count.
        Assert.True(counting.BytesRead < written.Length / 4,
            $"Reading 25 of 20,000 rows consumed {counting.BytesRead:N0} of {written.Length:N0} bytes.");
    }

    /// <summary>Passes everything through, and remembers how much went past.</summary>
    private sealed class CountingStream(Stream inner) : Stream
    {
        public long BytesRead { get; private set; }

        public override int Read(byte[] buffer, int offset, int count)
        {
            var read = inner.Read(buffer, offset, count);
            BytesRead += read;

            return read;
        }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer,
            CancellationToken cancellationToken = default)
        {
            var read = await inner.ReadAsync(buffer, cancellationToken);
            BytesRead += read;

            return read;
        }

        public override bool CanRead => inner.CanRead;
        public override bool CanSeek => inner.CanSeek;
        public override bool CanWrite => false;
        public override long Length => inner.Length;

        public override long Position
        {
            get => inner.Position;
            set => inner.Position = value;
        }

        public override long Seek(long offset, SeekOrigin origin) => inner.Seek(offset, origin);
        public override void Flush() => inner.Flush();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    [Fact]
    public async Task An_empty_table_round_trips_without_throwing()
    {
        foreach (var format in Enum.GetValues<DatasetFormat>())
        {
            var output = await RoundTripAsync(new TabularData(), format);
            Assert.Equal(0, output.RowCount);
        }
    }

    [Theory]
    [InlineData("data.csv", DatasetFormat.Csv)]
    [InlineData("data.tsv", DatasetFormat.Tsv)]
    [InlineData("data.json", DatasetFormat.Json)]
    [InlineData("data.parquet", DatasetFormat.Parquet)]
    [InlineData("data.unknown", DatasetFormat.Csv)]
    public void GuessFormat_reads_the_extension(string fileName, DatasetFormat expected)
    {
        Assert.Equal(expected, TabularSerializer.GuessFormat(fileName));
    }

    [Fact]
    public async Task Json_accepts_a_wrapped_array_as_well_as_a_bare_one()
    {
        const string wrapped = """{ "data": [ { "a": 1, "b": "x" }, { "a": 2, "b": "y" } ] }""";

        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(wrapped));
        var output = await TabularSerializer.ReadAsync(stream, DatasetFormat.Json);

        Assert.Equal(2, output.RowCount);
        Assert.Equal("x", output.Text(0, "b"));
    }

    [Fact]
    public async Task Json_that_is_not_a_row_collection_is_refused_with_a_reason()
    {
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes("""{ "a": 1 }"""));

        var error = await Assert.ThrowsAsync<InvalidDataException>(
            () => TabularSerializer.ReadAsync(stream, DatasetFormat.Json));

        Assert.Contains("array of objects", error.Message);
    }

    /// <summary>A stream that refuses to seek, like an upload or an HTTP response body.</summary>
    private sealed class ForwardOnlyStream(byte[] data) : Stream
    {
        private readonly MemoryStream _inner = new(data);

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();

        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _inner.Dispose();
            }

            base.Dispose(disposing);
        }
    }
}
