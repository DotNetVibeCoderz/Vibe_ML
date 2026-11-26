using Gravicode.Science.DaskNet;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;

namespace Gravicode.Science.Tests.DaskNet;

public sealed class FileSourceTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("dasknet-files-").FullName;
    private readonly DataFrame _sales = SalesData.Make(3_000, seed: 9);

    public void Dispose()
    {
        try { Directory.Delete(_dir, true); } catch (IOException) { }
    }

    private string WriteCsvFiles(int files)
    {
        var folder = Path.Combine(_dir, "csv");
        Directory.CreateDirectory(folder);
        for (var f = 0; f < files; f++)
        {
            var (start, count) = (f * _sales.RowCount / files, _sales.RowCount / files);
            _sales.Rows(start, count).WriteCsv(Path.Combine(folder, $"sales-{f}.csv"));
        }
        return Path.Combine(folder, "sales-*.csv");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(4096)]
    [InlineData(777)]
    public void Csv_blocks_reassemble_the_files(long blockSize)
    {
        var pattern = WriteCsvFiles(3);
        var df = DaskFrame.ReadCsv(pattern, blockSize);
        if (blockSize > 0) Assert.True(df.PartitionCount > 3);
        else Assert.Equal(3, df.PartitionCount);

        var expected = DataFrame.Concat(GraphBuilders.ExpandPaths(pattern).Select(p => DataFrame.ReadCsv(p)).ToArray());
        var actual = df.Compute();
        Assert.Equal(expected.ColumnNames, actual.ColumnNames);
        Assert.Equal(expected.Columns.Select(c => c.DataType), actual.Columns.Select(c => c.DataType));
        Assert.Equal(SalesData.Rows(expected), SalesData.Rows(actual));
    }

    [Fact]
    public void Csv_projection_parses_only_selected_columns()
    {
        var df = DaskFrame.ReadCsv(WriteCsvFiles(2), 2048).Filter(Col("region").Eq("east")).Select("id", "price");
        Assert.Contains("columns=[id, region, price]", df.Explain());
        var result = df.Compute();
        Assert.Equal(new[] { "id", "price" }, result.ColumnNames);
        Assert.Equal(_sales.Text("region").Values.ToArray().Count(r => r == "east"), result.RowCount);
    }

    [Fact]
    public void Empty_blocks_keep_their_schema()
    {
        var df = DaskFrame.ReadCsv(WriteCsvFiles(2), 512).Filter(Col("id") < 0);
        var result = df.Compute();
        Assert.Equal(0, result.RowCount);
        Assert.Equal(_sales.ColumnCount, result.ColumnCount);
        Assert.Equal(0, df.GroupBy("region").Sum("price").Compute().RowCount);
    }

    [Fact]
    public void Csv_and_arrow_writers_round_trip()
    {
        var df = DaskFrame.FromFrame(_sales, 4).Filter(Col("promo"));
        var csvPaths = df.ToCsv(Path.Combine(_dir, "out-csv"));
        Assert.Equal(4, csvPaths.Length);
        Assert.Equal(df.Count().Compute(), DaskFrame.ReadCsv(Path.Combine(_dir, "out-csv", "*.csv"), 0).Count().Compute());

        var arrowPaths = df.ToArrow(Path.Combine(_dir, "out-arrow"));
        var back = DaskFrame.ReadArrow(Path.Combine(_dir, "out-arrow", "*.arrow"));
        Assert.Equal(arrowPaths.Length, back.PartitionCount);
        Assert.Equal(SalesData.Rows(df.Compute()), SalesData.Rows(back.Compute()));
    }

    [Fact]
    public void Checkpoint_resumes_from_completed_partitions()
    {
        var calls = 0;
        var source = DaskFrame.FromFrame(_sales, 6).MapPartitions(f => { Interlocked.Increment(ref calls); return f; }, _sales.ColumnNames);
        var dir = Path.Combine(_dir, "checkpoint");

        var first = source.Checkpoint(dir);
        Assert.Equal(6, calls);
        Assert.Equal(6, Directory.GetFiles(dir, "*.arrow").Length);

        // Simulate a crash that lost one partition: only that one is recomputed.
        File.Delete(Path.Combine(dir, "part-00002.arrow"));
        calls = 0;
        var second = source.Checkpoint(dir);
        Assert.Equal(1, calls);
        Assert.Equal(SalesData.Rows(first.Compute()), SalesData.Rows(second.Compute()));
        Assert.True(second.IsRemotable); // the lineage is now just Arrow files
    }

    [Fact]
    public void ReadFiles_takes_any_reader()
    {
        var folder = Path.Combine(_dir, "arrow-in");
        Directory.CreateDirectory(folder);
        ArrowFile.Write(_sales.Rows(0, 10), Path.Combine(folder, "a.arrow"));
        ArrowFile.Write(_sales.Rows(10, 5), Path.Combine(folder, "b.arrow"));
        var df = DaskFrame.ReadFiles(Path.Combine(folder, "*.arrow"), ArrowFile.Read);
        Assert.False(df.IsRemotable);
        Assert.Equal(15, df.Count().Compute());
    }
}
