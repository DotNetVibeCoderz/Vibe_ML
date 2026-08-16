using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;
using Gravicode.Science.GraviNum;

BenchmarkSwitcher.FromAssembly(typeof(CsvLoadBenchmark).Assembly).Run(args, DefaultConfig.Instance
    .AddJob(Job.ShortRun.WithWarmupCount(2).WithIterationCount(4))
    .WithOptions(ConfigOptions.DisableOptimizationsValidator));
return;

/// <summary>
/// Reading a CSV through a StreamReader against reading it through a memory-mapped view.
/// </summary>
/// <remarks>
/// The two paths cost about the same wall-clock time; the difference is peak memory. The streaming
/// reader is fine until the file no longer fits alongside the parsed frame, at which point the
/// mapped path is the only one that completes - the operating system pages the file in and out
/// instead of the runtime holding it all at once. <see cref="MemoryDiagnoserAttribute"/> is what
/// makes that visible here.
/// </remarks>
[MemoryDiagnoser]
public class CsvLoadBenchmark
{
    private string _path = "";

    /// <summary>Number of data rows in the generated file.</summary>
    [Params(10_000, 100_000, 500_000)]
    public int Rows { get; set; }

    /// <summary>Writes a temporary CSV of the requested size.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _path = Path.Combine(Path.GetTempPath(), $"graviframe-bench-{Rows}.csv");
        if (File.Exists(_path)) return;

        var rng = new GraviRandom(42);
        using var writer = new StreamWriter(_path);
        writer.WriteLine("id,date,category,value,quantity");

        var start = new DateTime(2020, 1, 1);
        for (var i = 0; i < Rows; i++)
            writer.WriteLine($"{i},{start.AddMinutes(i):yyyy-MM-dd},cat{i % 20},{rng.Normal(100, 25):F4},{rng.Next(1, 50)}");
    }

    /// <summary>Streaming read.</summary>
    [Benchmark(Baseline = true)]
    public int StreamReader() => DataFrame.ReadCsv(_path).RowCount;

    /// <summary>Memory-mapped read.</summary>
    [Benchmark]
    public int MemoryMapped() => DataFrame.ReadCsvMemoryMapped(_path).RowCount;

    /// <summary>Deletes the temporary file.</summary>
    [GlobalCleanup]
    public void Cleanup()
    {
        if (File.Exists(_path)) File.Delete(_path);
    }
}

/// <summary>CSV against Parquet, which is columnar on disk and therefore both smaller and typed.</summary>
[MemoryDiagnoser]
public class ParquetBenchmark
{
    private string _csvPath = "";
    private string _parquetPath = "";
    private DataFrame _frame = DataFrame.Empty();

    /// <summary>Number of data rows.</summary>
    [Params(50_000, 200_000)]
    public int Rows { get; set; }

    /// <summary>Builds a frame and writes it in both formats.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        var columns = new List<Series>
        {
            new NumericSeries("id", Enumerable.Range(0, Rows).Select(i => (double)i).ToArray()),
            new NumericSeries("value", Enumerable.Range(0, Rows).Select(_ => rng.Normal(100, 25)).ToArray()),
            new NumericSeries("quantity", Enumerable.Range(0, Rows).Select(_ => (double)rng.Next(1, 50)).ToArray()),
            new TextSeries("category", Enumerable.Range(0, Rows).Select(i => (string?)$"cat{i % 20}").ToArray()),
        };
        _frame = new DataFrame(columns);

        _csvPath = Path.Combine(Path.GetTempPath(), $"graviframe-bench-{Rows}.csv");
        _parquetPath = Path.Combine(Path.GetTempPath(), $"graviframe-bench-{Rows}.parquet");
        _frame.WriteCsv(_csvPath);
        ParquetIO.Write(_frame, _parquetPath);

        Console.WriteLine($"[setup] csv {new FileInfo(_csvPath).Length / 1024} KB, parquet {new FileInfo(_parquetPath).Length / 1024} KB");
    }

    /// <summary>Reading everything from CSV, including type inference.</summary>
    [Benchmark(Baseline = true)]
    public int ReadCsv() => DataFrame.ReadCsv(_csvPath).RowCount;

    /// <summary>Reading everything from Parquet, with types already recorded.</summary>
    [Benchmark]
    public int ReadParquet() => ParquetIO.Read(_parquetPath).RowCount;

    /// <summary>Reading a single column, where the columnar layout pays off most.</summary>
    [Benchmark]
    public int ReadParquetOneColumn() => ParquetIO.Read(_parquetPath, ["value"]).RowCount;

    /// <summary>Deletes the temporary files.</summary>
    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (var path in new[] { _csvPath, _parquetPath })
            if (File.Exists(path)) File.Delete(path);
    }
}

/// <summary>Group-by, join and rolling-window cost as the frame grows.</summary>
[MemoryDiagnoser]
public class TransformBenchmark
{
    private DataFrame _frame = DataFrame.Empty();
    private DataFrame _lookup = DataFrame.Empty();
    private NumericSeries _series = new("v", 1);

    /// <summary>Number of rows in the frame.</summary>
    [Params(50_000, 250_000)]
    public int Rows { get; set; }

    /// <summary>Builds the frame and a lookup table to join against.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        _frame = new DataFrame(
        [
            new NumericSeries("group", Enumerable.Range(0, Rows).Select(i => (double)(i % 500)).ToArray()),
            new NumericSeries("value", Enumerable.Range(0, Rows).Select(_ => rng.Normal(50, 10)).ToArray()),
            new TextSeries("label", Enumerable.Range(0, Rows).Select(i => (string?)$"g{i % 500}").ToArray()),
        ]);

        _lookup = new DataFrame(
        [
            new NumericSeries("group", Enumerable.Range(0, 500).Select(i => (double)i).ToArray()),
            new NumericSeries("weight", Enumerable.Range(0, 500).Select(_ => rng.NextDouble()).ToArray()),
        ]);

        _series = _frame.Numeric("value");
    }

    /// <summary>Mean per group over 500 groups.</summary>
    [Benchmark(Baseline = true)]
    public int GroupByMean() => _frame.GroupBy("group").Mean("value").RowCount;

    /// <summary>The same grouping with a composite key.</summary>
    [Benchmark]
    public int GroupByComposite() => _frame.GroupBy("group", "label").Sum("value").RowCount;

    /// <summary>Hash join against a 500-row lookup.</summary>
    [Benchmark]
    public int Join() => _frame.Join(_lookup, "group").RowCount;

    /// <summary>Sorting by a numeric column.</summary>
    [Benchmark]
    public int Sort() => _frame.SortBy("value").RowCount;

    /// <summary>Rolling mean, which uses an incremental accumulator.</summary>
    [Benchmark]
    public double RollingMean() => _series.Rolling(30).Mean()[Rows - 1];

    /// <summary>Rolling median, which has to re-sort each window.</summary>
    [Benchmark]
    public double RollingMedian() => _series.Rolling(30).Median()[Rows - 1];
}
