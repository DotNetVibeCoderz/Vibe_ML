using BenchmarkDotNet.Attributes;
using Gravicode.HFNet.GraviDatasets;

namespace Gravicode.HFNet.Benchmarks;

/// <summary>GraviDatasets: memory-mapped against full load, and CSV against Parquet.</summary>
/// <remarks>
/// On the 60 KB Titanic file the two loaders cost the same, so this generates a file large enough for
/// the difference to be about the loader: 200,000 rows of mixed numeric and text columns, about 16 MB.
/// </remarks>
[BenchmarkCategory("GraviDatasets")]
public class DatasetBenchmarks
{
    private string _csv = "";
    private string _parquet = "";

    [GlobalSetup]
    public void Setup()
    {
        var directory = Path.Combine(Path.GetTempPath(), "hfnet-bench");
        Directory.CreateDirectory(directory);
        _csv = Path.Combine(directory, "rows.csv");
        _parquet = Path.Combine(directory, "rows.parquet");

        if (!File.Exists(_csv))
        {
            var random = new Random(1);
            using var writer = new StreamWriter(_csv);
            writer.WriteLine("id,price,volume,ratio,label,city");
            string[] cities = ["Bandung", "Jakarta", "Surabaya", "Medan", "Bogor"];
            for (var i = 0; i < 200_000; i++)
            {
                writer.WriteLine($"{i},{random.NextDouble() * 1000:F4},{random.Next(1, 100000)},{random.NextDouble():F6},{random.Next(2)},{cities[i % cities.Length]}");
            }
        }

        if (!File.Exists(_parquet)) Dataset.FromFile(_csv).WriteParquet(_parquet);
    }

    [Benchmark(Baseline = true, Description = "CSV, full load (200k rows)")]
    public int FullLoad() => Dataset.FromFile(_csv).Count;

    [Benchmark(Description = "CSV, memory-mapped (200k rows)")]
    public int MemoryMapped() => Dataset.FromCsvMemoryMapped(_csv).Count;

    [Benchmark(Description = "Parquet (200k rows)")]
    public int Parquet() => Dataset.FromFile(_parquet).Count;

    [Benchmark(Description = "Built-in titanic, 891 rows")]
    public int Titanic() => Dataset.Load("titanic").Count;
}
