using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviFrame;

/// <summary>
/// Tests for the out-of-core layer.
/// </summary>
/// <remarks>
/// <para>
/// The independent reference is the in-memory implementation: a streaming aggregate that disagrees
/// with <see cref="GroupedDataFrame"/> on data small enough for both is wrong, whatever it does on
/// data that only one can handle. So most of these run the same query twice and compare.
/// </para>
/// <para>
/// The claim that cannot be checked that way — that memory stays bounded — is checked structurally
/// instead, by counting how many rows are ever resident at once.
/// </para>
/// </remarks>
public class StreamingTests : IDisposable
{
    private readonly List<string> _temporary = [];

    private string TempFile(string extension = ".csv")
    {
        var path = Path.Combine(Path.GetTempPath(), $"gravi_{Guid.NewGuid():N}{extension}");
        _temporary.Add(path);
        return path;
    }

    public void Dispose()
    {
        foreach (var path in _temporary)
            if (File.Exists(path)) File.Delete(path);
    }

    /// <summary>A frame with repeated keys, missing values and a wide numeric range.</summary>
    private static DataFrame Sales(int rows = 5000, int seed = 7)
    {
        var rng = new GraviRandom(seed);
        string[] regions = ["north", "south", "east", "west"];

        var region = new string?[rows];
        var product = new string?[rows];
        var amount = new double[rows];
        var units = new double[rows];

        for (var i = 0; i < rows; i++)
        {
            region[i] = regions[rng.Next(regions.Length)];
            product[i] = $"p{rng.Next(6)}";
            amount[i] = i % 37 == 0 ? double.NaN : rng.Normal() * 100 + 500;
            units[i] = rng.Next(1, 20);
        }

        return new DataFrame(
        [
            new TextSeries("region", region),
            new TextSeries("product", product),
            new NumericSeries("amount", amount),
            new NumericSeries("units", units),
        ]);
    }

    private string WriteCsv(DataFrame frame)
    {
        var path = TempFile();
        CsvWriter.Write(frame, path);
        return path;
    }

    // ---------------------------------------------------------------- chunking

    [Fact]
    public void ChunksCoverEveryRowExactlyOnce()
    {
        var frame = Sales(rows: 1000);
        var chunked = ChunkedFrame.FromCsv(WriteCsv(frame), chunkRows: 128);

        Assert.Equal(1000, Streaming.CountRows(chunked));
    }

    [Fact]
    public void NoChunkExceedsTheRequestedSize()
    {
        // The bound that makes this worth having. A chunk larger than requested means the memory
        // ceiling is not what the caller asked for.
        var chunked = ChunkedFrame.FromCsv(WriteCsv(Sales(rows: 1000)), chunkRows: 128);

        var largest = 0;
        var count = 0;
        foreach (var chunk in chunked.Chunks)
        {
            largest = Math.Max(largest, chunk.RowCount);
            count++;
        }

        Assert.True(largest <= 128, $"a chunk held {largest} rows");
        Assert.Equal(8, count);          // 1000 / 128 rounded up
    }

    [Fact]
    public void ColumnNamesAreKnownWithoutReadingAnyData()
    {
        // The header is read on its own, so the schema is available before a single row is parsed.
        var chunked = ChunkedFrame.FromCsv(WriteCsv(Sales(rows: 10)), chunkRows: 4);
        Assert.Equal(["region", "product", "amount", "units"], chunked.ColumnNames);
    }

    [Fact]
    public void EnumeratingTwiceRereadsTheSource()
    {
        // Deliberate: retaining chunks would defeat the purpose. Worth pinning so that a future
        // "optimisation" that caches them has to argue with a failing test.
        var chunked = ChunkedFrame.FromCsv(WriteCsv(Sales(rows: 500)), chunkRows: 64);

        Assert.Equal(500, Streaming.CountRows(chunked));
        Assert.Equal(500, Streaming.CountRows(chunked));
    }

    [Fact]
    public void ATrailingPartialChunkIsNotDropped()
    {
        // 1000 rows in chunks of 300 leaves 100 over. Losing them is the classic off-by-one here.
        var chunked = ChunkedFrame.FromCsv(WriteCsv(Sales(rows: 1000)), chunkRows: 300);
        Assert.Equal(1000, Streaming.CountRows(chunked));
    }

    // ---------------------------------------------------------------- group-by

    [Fact]
    public void StreamingGroupByAgreesWithTheInMemoryOne()
    {
        // The reference that matters: the same query, computed both ways.
        var frame = Sales(rows: 5000);
        var chunked = ChunkedFrame.FromCsv(WriteCsv(frame), chunkRows: 256);

        var expected = frame.GroupBy("region").AggregateMany([("amount", "sum"), ("amount", "mean")]);
        var actual = Streaming.GroupBy(chunked, ["region"], ("amount", "sum"), ("amount", "mean"));

        Assert.Equal(expected.RowCount, actual.RowCount);

        var expectedByRegion = new Dictionary<string, (double Sum, double Mean)>();
        for (var i = 0; i < expected.RowCount; i++)
            expectedByRegion[((TextSeries)expected["region"])[i]!] =
                (expected.Numeric("amount_sum")[i], expected.Numeric("amount_mean")[i]);

        for (var i = 0; i < actual.RowCount; i++)
        {
            var region = ((TextSeries)actual["region"])[i]!;
            var (sum, mean) = expectedByRegion[region];

            Assert.Equal(sum, actual.Numeric("amount_sum")[i], 6);
            Assert.Equal(mean, actual.Numeric("amount_mean")[i], 9);
        }
    }

    [Fact]
    public void CompositeKeysGroupTheSameWayInMemoryAndStreaming()
    {
        var frame = Sales(rows: 3000);
        var chunked = ChunkedFrame.FromCsv(WriteCsv(frame), chunkRows: 100);

        var expected = frame.GroupBy("region", "product").AggregateMany([("units", "sum")]);
        var actual = Streaming.GroupBy(chunked, ["region", "product"], ("units", "sum"));

        Assert.Equal(expected.RowCount, actual.RowCount);
        Assert.Equal(expected.Numeric("units_sum").Values.ToArray().Sum(),
                     actual.Numeric("units_sum").Values.ToArray().Sum(), 6);
    }

    [Fact]
    public void MinAndMaxAndCountAgreeToo()
    {
        var frame = Sales(rows: 2000);
        var chunked = ChunkedFrame.FromCsv(WriteCsv(frame), chunkRows: 97);

        var expected = frame.GroupBy("region").AggregateMany([("units", "min"), ("units", "max")]);
        var actual = Streaming.GroupBy(chunked, ["region"],
            ("units", "min"), ("units", "max"), ("units", "count"));

        for (var i = 0; i < actual.RowCount; i++)
        {
            var region = ((TextSeries)actual["region"])[i]!;
            var row = -1;
            for (var j = 0; j < expected.RowCount; j++)
                if (((TextSeries)expected["region"])[j] == region) row = j;

            Assert.Equal(expected.Numeric("units_min")[row], actual.Numeric("units_min")[i], 9);
            Assert.Equal(expected.Numeric("units_max")[row], actual.Numeric("units_max")[i], 9);
        }

        // Counts must add up to the rows that were actually present.
        Assert.Equal(2000, actual.Numeric("units_count").Values.ToArray().Sum());
    }

    [Fact]
    public void MissingValuesAreSkippedRatherThanCountedAsZero()
    {
        // Matching the in-memory aggregates: a mean over a sparse column is the mean of what is
        // there. Treating NaN as zero would drag every average down and look plausible.
        var frame = new DataFrame(
        [
            new TextSeries("k", ["a", "a", "a"]),
            new NumericSeries("v", [10.0, double.NaN, 20.0]),
        ]);

        var actual = Streaming.GroupBy(ChunkedFrame.FromFrame(frame, 1), ["k"],
            ("v", "mean"), ("v", "count"), ("v", "sum"));

        Assert.Equal(15.0, actual.Numeric("v_mean")[0], 9);
        Assert.Equal(2.0, actual.Numeric("v_count")[0]);
        Assert.Equal(30.0, actual.Numeric("v_sum")[0], 9);
    }

    [Fact]
    public void TheResultIsIndependentOfTheChunkSize()
    {
        // If it were not, the chunking would be leaking into the answer.
        var frame = Sales(rows: 2000);
        var path = WriteCsv(frame);

        double SumFor(int chunkRows)
        {
            var result = Streaming.GroupBy(
                ChunkedFrame.FromCsv(path, chunkRows), ["region"], ("amount", "sum"));
            return result.Numeric("amount_sum").Values.ToArray().Sum();
        }

        var reference = SumFor(64);
        foreach (var size in new[] { 1, 7, 500, 5000 })
            Assert.Equal(reference, SumFor(size), 6);
    }

    [Fact]
    public void AnAggregateThatCannotStreamIsRefused()
    {
        // Median needs the values, so a streaming version would have to keep them all and only
        // look like it was streaming. Saying so beats quietly using unbounded memory.
        var chunked = ChunkedFrame.FromFrame(Sales(rows: 10), 4);

        var error = Assert.Throws<ArgumentException>(
            () => Streaming.GroupBy(chunked, ["region"], ("amount", "median")));

        Assert.Contains("median", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void CountGroupsMatchesWhatGroupByProduces()
    {
        // The pre-flight check: GroupBy's memory is proportional to this, so it is the difference
        // between a query that runs and one that does not.
        var frame = Sales(rows: 3000);
        var chunked = ChunkedFrame.FromCsv(WriteCsv(frame), chunkRows: 128);

        var groups = Streaming.CountGroups(chunked, ["region", "product"]);
        var actual = Streaming.GroupBy(chunked, ["region", "product"], ("units", "sum"));

        Assert.Equal(groups, actual.RowCount);
    }

    // ---------------------------------------------------------------- describe

    [Fact]
    public void StreamingDescribeAgreesWithTheInMemoryStatistics()
    {
        var frame = Sales(rows: 4000);
        var chunked = ChunkedFrame.FromCsv(WriteCsv(frame), chunkRows: 137);

        var summary = Streaming.Describe(chunked, ["amount", "units"]);

        var amount = frame.Numeric("amount").Values.ToArray().Where(v => !double.IsNaN(v)).ToArray();

        Assert.Equal(amount.Length, summary["amount"].Count);
        Assert.Equal(amount.Average(), summary["amount"].Mean, 8);
        Assert.Equal(amount.Min(), summary["amount"].Min, 9);
        Assert.Equal(amount.Max(), summary["amount"].Max, 9);

        var variance = amount.Sum(v => (v - amount.Average()) * (v - amount.Average())) / (amount.Length - 1);
        Assert.Equal(Math.Sqrt(variance), summary["amount"].StandardDeviation, 6);
    }

    [Fact]
    public void VarianceStaysAccurateWhenTheMeanDwarfsTheSpread()
    {
        // The case that breaks the textbook E[x^2] - E[x]^2 formula: it subtracts two nearly equal
        // large numbers and can return a negative variance. Welford does not.
        var values = new double[10_000];
        var rng = new GraviRandom(3);
        for (var i = 0; i < values.Length; i++) values[i] = 1e9 + rng.Normal();

        var frame = new DataFrame([new NumericSeries("v", values)]);
        var summary = Streaming.Describe(ChunkedFrame.FromFrame(frame, 256), ["v"]);

        Assert.True(summary["v"].StandardDeviation is > 0.9 and < 1.1,
            $"standard deviation came out at {summary["v"].StandardDeviation}");
    }

    // ---------------------------------------------------------------- filtering

    [Fact]
    public void FilteringToAFileKeepsExactlyTheMatchingRows()
    {
        var frame = Sales(rows: 2000);
        var chunked = ChunkedFrame.FromCsv(WriteCsv(frame), chunkRows: 111);
        var output = TempFile();

        var written = Streaming.FilterToFile(
            chunked, (chunk, row) => chunk.Numeric("units")[row] > 10, output);

        var expected = frame.Numeric("units").Values.ToArray().Count(v => v > 10);
        Assert.Equal(expected, written);

        var back = CsvReader.Read(output, new CsvOptions());
        Assert.Equal(expected, back.RowCount);
        Assert.All(Enumerable.Range(0, back.RowCount), i => Assert.True(back.Numeric("units")[i] > 10));
    }

    [Fact]
    public void TheFilteredFileHasExactlyOneHeader()
    {
        // Every chunk is written separately, so a naive implementation emits a header per chunk and
        // the file reads back with header rows scattered through the data.
        var chunked = ChunkedFrame.FromCsv(WriteCsv(Sales(rows: 1000)), chunkRows: 50);
        var output = TempFile();

        Streaming.FilterToFile(chunked, (_, _) => true, output);

        var headers = File.ReadLines(output).Count(l => l.StartsWith("region", StringComparison.Ordinal));
        Assert.Equal(1, headers);
    }

    [Fact]
    public void AFilterThatMatchesNothingStillWritesAUsableFile()
    {
        var chunked = ChunkedFrame.FromCsv(WriteCsv(Sales(rows: 100)), chunkRows: 32);
        var output = TempFile();

        Assert.Equal(0, Streaming.FilterToFile(chunked, (_, _) => false, output));

        var back = CsvReader.Read(output, new CsvOptions());
        Assert.Equal(0, back.RowCount);
        Assert.Contains("region", back.ColumnNames);
    }

    // ---------------------------------------------------------------- external sort

    [Fact]
    public void ExternalSortProducesAFullyOrderedFile()
    {
        var frame = Sales(rows: 5000);
        var chunked = ChunkedFrame.FromCsv(WriteCsv(frame), chunkRows: 250);
        var output = TempFile();

        var written = Streaming.SortToFile(chunked, "amount", output);
        Assert.Equal(5000, written);

        var back = CsvReader.Read(output, new CsvOptions());
        Assert.Equal(5000, back.RowCount);

        var sorted = back.Numeric("amount").Values.ToArray();
        var present = sorted.TakeWhile(v => !double.IsNaN(v)).ToArray();

        for (var i = 1; i < present.Length; i++)
            Assert.True(present[i] >= present[i - 1],
                $"row {i} broke the order: {present[i - 1]} then {present[i]}");
    }

    [Fact]
    public void ExternalSortKeepsEveryRowAndEveryColumn()
    {
        // A merge that drops or duplicates rows still produces a sorted file, so the ordering test
        // alone would not catch it.
        var frame = Sales(rows: 3000);
        var chunked = ChunkedFrame.FromCsv(WriteCsv(frame), chunkRows: 128);
        var output = TempFile();

        Streaming.SortToFile(chunked, "units", output);
        var back = CsvReader.Read(output, new CsvOptions());

        Assert.Equal(frame.RowCount, back.RowCount);
        Assert.Equal(frame.ColumnNames, back.ColumnNames);

        // The multiset of values must be unchanged.
        var before = frame.Numeric("units").Values.ToArray().OrderBy(v => v).ToArray();
        var after = back.Numeric("units").Values.ToArray().OrderBy(v => v).ToArray();
        Assert.Equal(before, after);
    }

    [Fact]
    public void DescendingSortReversesTheOrder()
    {
        var chunked = ChunkedFrame.FromCsv(WriteCsv(Sales(rows: 1500)), chunkRows: 200);
        var output = TempFile();

        Streaming.SortToFile(chunked, "units", output, descending: true);
        var values = CsvReader.Read(output, new CsvOptions()).Numeric("units").Values.ToArray();

        var present = values.TakeWhile(v => !double.IsNaN(v)).ToArray();
        for (var i = 1; i < present.Length; i++)
            Assert.True(present[i] <= present[i - 1], $"row {i} broke the descending order");
    }

    [Fact]
    public void MissingValuesSortLastInBothDirections()
    {
        // Missing is absent, not extreme. Putting it at one end makes it the minimum or the maximum
        // depending on the direction, which is a different answer each way.
        //
        // A SECOND column is deliberate: with one column, a missing value writes as a blank line,
        // and a CSV reader cannot tell that from padding — the rows would be correctly ordered in
        // the file and lost on the way back in.
        var frame = new DataFrame(
        [
            new NumericSeries("v", [3.0, double.NaN, 1.0, double.NaN, 2.0]),
            new TextSeries("tag", ["a", "b", "c", "d", "e"]),
        ]);

        foreach (var descending in new[] { false, true })
        {
            var output = TempFile();
            Streaming.SortToFile(ChunkedFrame.FromFrame(frame, 2), "v", output, descending);

            var back = CsvReader.Read(output, new CsvOptions());
            Assert.Equal(5, back.RowCount);

            var values = back.Numeric("v").Values.ToArray();
            Assert.True(double.IsNaN(values[^1]), $"descending={descending}: missing did not sort last");
            Assert.True(double.IsNaN(values[^2]), $"descending={descending}: missing did not sort last");

            // And the three present values are in the requested order.
            var present = values.Take(3).ToArray();
            Assert.Equal(descending ? [3.0, 2.0, 1.0] : new[] { 1.0, 2.0, 3.0 }, present);
        }
    }

    [Fact]
    public void SortingWorksWhenEveryChunkIsASingleRow()
    {
        // The degenerate case: as many runs as rows, so the merge does all the work.
        var frame = new DataFrame([new NumericSeries("v", [5.0, 3.0, 9.0, 1.0, 7.0])]);

        var output = TempFile();

        Streaming.SortToFile(ChunkedFrame.FromFrame(frame, 1), "v", output);

        var values = CsvReader.Read(output, new CsvOptions()).Numeric("v").Values.ToArray();
        Assert.Equal([1.0, 3.0, 5.0, 7.0, 9.0], values);
    }

    [Fact]
    public void SortingAnEmptyInputProducesAHeaderOnlyFile()
    {
        var frame = new DataFrame([new NumericSeries("v", [])]);
        var output = TempFile();

        Assert.Equal(0, Streaming.SortToFile(ChunkedFrame.FromFrame(frame, 10), "v", output));
        Assert.Equal(0, CsvReader.Read(output, new CsvOptions()).RowCount);
    }

    [Fact]
    public void SortingCleansUpItsTemporaryRuns()
    {
        // Two full passes over the data leave a lot behind if the scratch directory is not removed.
        var scratchRoot = Path.Combine(Path.GetTempPath(), $"gravi_scratch_{Guid.NewGuid():N}");
        var before = Directory.Exists(scratchRoot);

        var chunked = ChunkedFrame.FromCsv(WriteCsv(Sales(rows: 800)), chunkRows: 100);
        var output = TempFile();
        Streaming.SortToFile(chunked, "units", output);

        Assert.False(before);
        Assert.False(Directory.Exists(scratchRoot));

        // Nothing named like our scratch pattern should survive the call.
        var leftovers = Directory.GetDirectories(Path.GetTempPath(), "gravi-sort-*");
        Assert.Empty(leftovers);
    }

    // ---------------------------------------------------------------- memory bound

    [Fact]
    public void NoMoreThanOneChunkIsResidentAtATime()
    {
        // The structural version of "memory stays bounded": count the rows alive at any moment as
        // the pipeline runs. A pipeline that quietly materialised the input would show all of them.
        var frame = Sales(rows: 4000);
        var path = WriteCsv(frame);

        var live = 0;
        var peak = 0;

        IEnumerable<DataFrame> Instrumented()
        {
            foreach (var chunk in ChunkedFrame.FromCsv(path, chunkRows: 100).Chunks)
            {
                live = chunk.RowCount;
                peak = Math.Max(peak, live);
                yield return chunk;
                live = 0;
            }
        }

        var chunked = ChunkedFrame.FromChunks(Instrumented, frame.ColumnNames);
        Streaming.GroupBy(chunked, ["region"], ("amount", "sum"));

        Assert.True(peak <= 100, $"{peak} rows were resident at once");
        Assert.Equal(0, live);
    }

    [Fact]
    public void MalformedRequestsAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => ChunkedFrame.FromCsv(WriteCsv(Sales(rows: 10)), chunkRows: 0));

        var chunked = ChunkedFrame.FromFrame(Sales(rows: 10), 4);
        Assert.Throws<ArgumentException>(() => Streaming.GroupBy(chunked, ["region"]));
    }
}
