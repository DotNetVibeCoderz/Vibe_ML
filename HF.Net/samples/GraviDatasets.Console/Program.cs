using System.Diagnostics;
using Gravicode.HFNet.GraviDatasets;

// GraviDatasets sample. Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
Console.WriteLine("=== GraviDatasets ===\n");

// --- 1. The blueprint's example: a built-in dataset, no network needed ---------
var titanic = Dataset.Load("titanic");
Console.WriteLine(titanic);
Console.WriteLine($"  columns : {string.Join(", ", titanic.Columns)}");
Console.WriteLine($"  features: {string.Join(", ", titanic.Features.Take(5).Select(f => $"{f.Name}:{f.Type}"))} ...");
Console.WriteLine();

// --- 2. Filter, map, shuffle, split ---------------------------------------------
var adults = titanic.Filter(row => !row.IsMissing("age") && row.Number("age") >= 18);
Console.WriteLine($"adults: {adults.Count} of {titanic.Count}");

var fares = titanic.NumericColumn("fare");
var withBand = titanic.Map("fare_band", i => fares[i] switch { < 10 => 0, < 50 => 1, _ => 2 });
Console.WriteLine($"added fare_band: {string.Join(", ", withBand.Head(5).NumericColumn("fare_band"))} ...");

var splits = titanic.Shuffle(seed: 7).TrainTestSplit(testSize: 0.2, seed: 7);
Console.WriteLine($"split : {splits}");
Console.WriteLine();

// --- 3. Batches for a training loop ---------------------------------------------
var batches = splits.Train.Batches(128).ToList();
Console.WriteLine($"{batches.Count} batches of up to 128 rows; last has {batches[^1].Count}");
Console.WriteLine();

// --- 4. Memory-mapped against full load: the blueprint's benchmark ---------------
var path = Path.Combine(BuiltinDatasets.FindDatasetsDirectory()!, "titanic.csv");
var full = Time(() => Dataset.FromFile(path));
var mapped = Time(() => Dataset.FromCsvMemoryMapped(path));
Console.WriteLine($"full load {full.TotalMilliseconds:F2} ms, memory-mapped {mapped.TotalMilliseconds:F2} ms (best of 10)");
Console.WriteLine();

// --- 5. Parquet round trip -------------------------------------------------------
var parquet = Path.Combine(Path.GetTempPath(), "hfnet-titanic.parquet");
titanic.WriteParquet(parquet);
var back = Dataset.FromFile(parquet);
Console.WriteLine($"parquet round trip: {back.Count} rows, {back.Columns.Count} columns, {new FileInfo(parquet).Length / 1024} KB");
Console.WriteLine();

// --- 6. A dataset from the Hugging Face Hub -------------------------------------
try
{
    var iris = HubDatasets.Load("scikit-learn/iris");
    Console.WriteLine($"hub scikit-learn/iris: {iris}");
}
catch (Exception error)
{
    Console.WriteLine($"hub dataset skipped: {error.Message}");
}

static TimeSpan Time(Func<Dataset> load)
{
    load();
    var best = TimeSpan.MaxValue;
    for (var i = 0; i < 10; i++)
    {
        var watch = Stopwatch.StartNew();
        load();
        if (watch.Elapsed < best) best = watch.Elapsed;
    }

    return best;
}
