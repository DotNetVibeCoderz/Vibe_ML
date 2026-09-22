using Gravicode.HFNet.GraviDatasets;
using Gravicode.Science.GraviFrame;
using Xunit;

namespace Gravicode.HFNet.GraviDatasets.Tests;

/// <summary>Tests for dataset shaping, which run entirely on frames built in the test.</summary>
public sealed class DatasetTests
{
    private static Dataset Sample(int rows = 10)
    {
        var values = new double[rows];
        var labels = new string?[rows];

        for (var i = 0; i < rows; i++)
        {
            values[i] = i;
            labels[i] = i % 2 == 0 ? "even" : "odd";
        }

        return new Dataset(
            new DataFrame([new NumericSeries("value", values), new TextSeries("label", labels)]),
            "sample");
    }

    [Fact]
    public void ReportsItsShape()
    {
        var dataset = Sample();

        Assert.Equal(10, dataset.Count);
        Assert.Equal(["value", "label"], dataset.Columns);
    }

    [Fact]
    public void ShuffleIsReproducibleFromItsSeed()
    {
        var dataset = Sample(50);

        Assert.Equal(
            dataset.Shuffle(7).NumericColumn("value"),
            dataset.Shuffle(7).NumericColumn("value"));

        Assert.NotEqual(
            dataset.Shuffle(7).NumericColumn("value"),
            dataset.Shuffle(8).NumericColumn("value"));
    }

    [Fact]
    public void ShuffleIsAPermutationRatherThanAResample()
    {
        var shuffled = Sample(50).Shuffle(3).NumericColumn("value");

        Assert.Equal(50, shuffled.Distinct().Count());
        Assert.Equal(Enumerable.Range(0, 50).Select(i => (double)i).Order(), shuffled.Order());
    }

    [Fact]
    public void ShuffleLeavesTheOriginalAlone()
    {
        var dataset = Sample(20);
        var before = dataset.NumericColumn("value").ToArray();

        dataset.Shuffle(1);

        Assert.Equal(before, dataset.NumericColumn("value"));
    }

    [Fact]
    public void TrainTestSplitPartitionsWithoutOverlap()
    {
        var split = Sample(100).TrainTestSplit(testSize: 0.2, seed: 4);

        Assert.Equal(80, split.Train.Count);
        Assert.Equal(20, split.Test.Count);

        var train = split.Train.NumericColumn("value").ToHashSet();
        var test = split.Test.NumericColumn("value").ToHashSet();

        Assert.Empty(train.Intersect(test));
        Assert.Equal(100, train.Count + test.Count);
    }

    [Fact]
    public void TrainTestSplitShufflesByDefault()
    {
        // The rows here are sorted by label, so an unshuffled split would put every "even" row in
        // train. This is the scenario the default guards against.
        var split = Sample(100).TrainTestSplit(testSize: 0.3);
        var testValues = split.Test.NumericColumn("value");

        Assert.Contains(testValues, v => v % 2 == 0);
        Assert.Contains(testValues, v => v % 2 != 0);
    }

    [Fact]
    public void TrainTestSplitCanKeepTheOrder()
    {
        var split = Sample(10).TrainTestSplit(testSize: 0.2, shuffle: false);

        Assert.Equal([0, 1, 2, 3, 4, 5, 6, 7], split.Train.NumericColumn("value"));
        Assert.Equal([8, 9], split.Test.NumericColumn("value"));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(1.5)]
    [InlineData(-0.2)]
    public void TrainTestSplitRejectsAFractionOutsideTheOpenUnitInterval(double testSize)
        => Assert.Throws<ArgumentOutOfRangeException>(() => Sample().TrainTestSplit(testSize));

    [Fact]
    public void BatchesCoverEveryRowWhenTheLastIsShort()
    {
        var batches = Sample(10).Batches(3).ToList();

        Assert.Equal(4, batches.Count);
        Assert.Equal([3, 3, 3, 1], batches.Select(b => b.Count));
    }

    [Fact]
    public void BatchesCanDropTheShortTail()
    {
        var batches = Sample(10).Batches(3, dropLast: true).ToList();

        Assert.Equal(3, batches.Count);
        Assert.All(batches, b => Assert.Equal(3, b.Count));
    }

    [Fact]
    public void FilterKeepsMatchingRows()
    {
        var filtered = Sample(10).Filter(row => row.Number("value") >= 5);

        Assert.Equal(5, filtered.Count);
    }

    [Fact]
    public void MapAddsAComputedColumn()
    {
        var mapped = Sample(5).Map("doubled", i => i * 2);

        Assert.Contains("doubled", mapped.Columns);
        Assert.Equal([0, 2, 4, 6, 8], mapped.NumericColumn("doubled"));
    }

    [Fact]
    public void TextColumnReadsANumericColumnToo()
    {
        // Published datasets disagree on whether a label column is text or numeric, so the accessor
        // has to cope with either rather than throwing on the one it did not expect.
        var dataset = Sample(3);

        Assert.Equal(["even", "odd", "even"], dataset.TextColumn("label"));
        Assert.Equal(3, dataset.TextColumn("value").Count);
    }

    [Fact]
    public void RowsStreamInOrder()
    {
        var values = Sample(5).Rows().Select(r => r.Number("value")).ToList();

        Assert.Equal([0, 1, 2, 3, 4], values);
    }

    [Fact]
    public void MissingSplitNamesTheOnesThatExist()
    {
        var split = Sample(10).TrainTestSplit();

        var exception = Assert.Throws<KeyNotFoundException>(() => split["validation"]);

        Assert.Contains("train", exception.Message);
        Assert.Contains("test", exception.Message);
    }
}

/// <summary>Tests for reading data files.</summary>
public sealed class FileFormatTests : IDisposable
{
    private readonly string _directory = Path.Combine(
        Path.GetTempPath(), "hfnet-datasets", Guid.NewGuid().ToString("N"));

    public FileFormatTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    private string Write(string name, string content)
    {
        var path = Path.Combine(_directory, name);
        File.WriteAllText(path, content);
        return path;
    }

    [Fact]
    public void ReadsJsonLines()
    {
        var path = Write("data.jsonl",
            """
            {"text": "first", "score": 1.5}
            {"text": "second", "score": 2.5}
            """);

        var dataset = Dataset.FromFile(path);

        Assert.Equal(2, dataset.Count);
        Assert.Equal(["first", "second"], dataset.TextColumn("text"));
        Assert.Equal([1.5, 2.5], dataset.NumericColumn("score"));
    }

    [Fact]
    public void ReadsAJsonArrayFromAFileNamedJsonl()
    {
        // Repositories publish both spellings under both extensions, so the reader decides by
        // looking at the first character rather than by trusting the name.
        var path = Write("array.jsonl", """[{"a": 1}, {"a": 2}]""");

        Assert.Equal([1, 2], Dataset.FromFile(path).NumericColumn("a"));
    }

    [Fact]
    public void KeepsColumnOrderFromFirstAppearance()
    {
        var path = Write("order.jsonl",
            """
            {"zebra": 1, "apple": 2}
            {"apple": 3, "zebra": 4}
            """);

        Assert.Equal(["zebra", "apple"], Dataset.FromFile(path).Columns);
    }

    [Fact]
    public void FillsAMissingFieldRatherThanFailing()
    {
        var path = Write("ragged.jsonl",
            """
            {"a": 1, "b": "x"}
            {"a": 2}
            """);

        var dataset = Dataset.FromFile(path);

        Assert.Equal(2, dataset.Count);
        Assert.Equal([1, 2], dataset.NumericColumn("a"));
    }

    [Fact]
    public void ReadsCsv()
    {
        var path = Write("data.csv", "name,score\nalpha,1\nbeta,2\n");
        var dataset = Dataset.FromFile(path);

        Assert.Equal(2, dataset.Count);
        Assert.Equal([1, 2], dataset.NumericColumn("score"));
    }

    [Fact]
    public void ReadsTsv()
    {
        var path = Write("data.tsv", "name\tscore\nalpha\t1\nbeta\t2\n");
        var dataset = Dataset.FromFile(path);

        Assert.Equal(2, dataset.Count);
        Assert.Equal(["name", "score"], dataset.Columns);
    }

    [Fact]
    public void RefusesAnUnknownExtension()
    {
        var path = Write("data.xyz", "nothing");

        var exception = Assert.Throws<NotSupportedException>(() => Dataset.FromFile(path));
        Assert.Contains("CSV", exception.Message);
    }

    [Fact]
    public void RoundTripsThroughCsv()
    {
        var path = Write("source.csv", "a,b\n1,2\n3,4\n");
        var dataset = Dataset.FromFile(path);

        var destination = Path.Combine(_directory, "out.csv");
        dataset.WriteCsv(destination);

        var reread = Dataset.FromFile(destination);

        Assert.Equal(dataset.NumericColumn("a"), reread.NumericColumn("a"));
        Assert.Equal(dataset.NumericColumn("b"), reread.NumericColumn("b"));
    }
}

/// <summary>Tests for the built-in dataset registry.</summary>
public sealed class BuiltinDatasetTests
{
    [Fact]
    public void KnowsTheClassics()
    {
        Assert.Contains("titanic", BuiltinDatasets.Names);
        Assert.Contains("iris", BuiltinDatasets.Names);
    }

    [Fact]
    public void LoadsIrisWithoutANetwork()
    {
        // Skipped rather than failed when the repository's datasets directory is not reachable
        // from wherever the test runner put us.
        if (!BuiltinDatasets.TryResolve("iris", out _)) return;

        var iris = Dataset.Load("iris");

        Assert.Equal(150, iris.Count);
        Assert.Contains(iris.Columns, c => c.Contains("sepal", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void LoadsTitanicWithoutANetwork()
    {
        if (!BuiltinDatasets.TryResolve("titanic", out _)) return;

        var titanic = Dataset.Load("titanic");

        Assert.True(titanic.Count > 800, $"expected the full titanic set, got {titanic.Count} rows");
    }

    [Fact]
    public void AnUnknownBareNameIsNotResolvedLocally()
        => Assert.False(BuiltinDatasets.TryResolve("definitely-not-a-dataset", out _));
}
