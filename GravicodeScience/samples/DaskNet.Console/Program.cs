using System.Diagnostics;
using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.DaskNet.Sql;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using static Gravicode.Science.DaskNet.Expr;

// `DaskNet.Console worker` turns this program into a cluster worker (token from DASKNET_TOKEN),
// so the sample's own [RemoteTask] methods are available without --preload.
if (args.Length > 0 && args[0] == "worker") return await Worker.RunCommandLineAsync(args[1..]);

Console.WriteLine(DaskNetInfo.Banner());
var datasets = Resolve("datasets") ?? throw new DirectoryNotFoundException("datasets directory not found.");
var watch = new Stopwatch();

// ---------------------------------------------------------------- delayed
Section("1. Delayed tasks and the task graph");

var a = Dask.Delayed(Pipeline.Load, Dask.Value(1.0));
var b = Dask.Delayed(Pipeline.Load, Dask.Value(2.0));
var total = Dask.Delayed(Pipeline.Combine, Dask.Delayed(Pipeline.Clean, a), Dask.Delayed(Pipeline.Clean, b));
Console.WriteLine($"  load -> clean -> combine = {total.Compute()}");
var graph = TaskGraph.FromCollections(total);
Console.WriteLine($"  {graph.Count} tasks; after linear fusion: {graph.Fuse([total.Key]).Count}");
Console.WriteLine("  Mermaid (fused, as it runs):");
foreach (var line in Dask.Graph(total).ToMermaid().Split('\n').Take(8)) Console.WriteLine("    " + line);
Console.WriteLine();

// ---------------------------------------------------------------- bag
Section("2. Bag: word frequencies over IMDB reviews");

var words = Bag.ReadText(Path.Combine(datasets, "imdb_reviews.csv"), blockSize: 4 * 1024)
    .FlatMap(line => line.ToLowerInvariant().Split([' ', ',', '.', '"', '!', '?'], StringSplitOptions.RemoveEmptyEntries))
    .Filter(w => w.Length > 5);
watch.Restart();
var top = words.Frequencies().Then(f => f.OrderByDescending(x => x.Count).Take(8).ToArray()).Compute();
watch.Stop();
Console.WriteLine($"  {words.PartitionCount} text blocks, {watch.Elapsed.TotalMilliseconds:F1} ms");
Console.WriteLine("  " + string.Join(", ", top.Select(t => $"{t.Item} {t.Count}")));
Console.WriteLine();

// ---------------------------------------------------------------- array
Section("3. Array: a 2000x2000 matrix in 500x500 chunks");

var x = DaskArray.RandomNormal([2000, 2000], [500, 500], seed: 7);
Console.WriteLine($"  {x}");
Console.WriteLine($"  mean {x.Mean().Compute():F5}, std {x.Std().Compute():F5} (standard normal: 0, 1)");
var gram = x.T.Dot(x) / 2000.0;
foreach (var scheduler in new IScheduler[] { new SynchronousScheduler(), new ThreadScheduler() })
{
    watch.Restart();
    var trace = gram.Compute(scheduler);
    watch.Stop();
    var diag = Enumerable.Range(0, 2000).Average(i => trace[i, i]);
    Console.WriteLine($"  X^T X / n on {scheduler.Name,-12}: {watch.Elapsed.TotalMilliseconds,7:F0} ms, mean diagonal {diag:F4} (expect ~1)");
}
Console.WriteLine("  (each 500-cube block product is already multi-threaded by GraviNum, so threads add little here)");
Console.WriteLine();

// ---------------------------------------------------------------- dataframe
Section("4. DataFrame: lazy CSV, optimised plan");

var prices = DaskFrame.ReadCsv(Path.Combine(datasets, "finance_timeseries.csv"), blockSize: 8 * 1024);
var monthly = prices
    .WithColumn("month", Col("date").Month())
    .Filter(Col("volume") > 500_000)
    .GroupBy("ticker", "month").Agg(("close", "mean"), ("volume", "sum"))
    .SortBy([("ticker", true), ("month", true)]);
Console.WriteLine($"  {prices}");
Console.WriteLine("  optimised plan (the filter is applied inside the CSV scan):");
foreach (var line in monthly.Explain().Split('\n')) Console.WriteLine("    " + line);
var result = monthly.Compute();
Console.WriteLine($"  {result.RowCount} (ticker, month) groups; first rows:");
Console.WriteLine(Indent(result.Head(4).ToString()));
Console.WriteLine();

// ---------------------------------------------------------------- sql
Section("5. SQL over registered frames");

var sql = new SqlContext().RegisterCsv("titanic", Path.Combine(datasets, "titanic.csv"));
const string query = """
    SELECT class, sex, COUNT(*) AS passengers, ROUND(AVG(survived) * 100, 1) AS survival_pct
    FROM titanic
    WHERE age IS NOT NULL
    GROUP BY class, sex
    ORDER BY survival_pct DESC
    """;
Console.WriteLine(Indent(sql.Execute(query).ToString()));
Console.WriteLine();

// ---------------------------------------------------------------- cluster
Section("6. A local cluster, and surviving a lost worker");

await using (var cluster = await LocalCluster.StartAsync(workers: 3, threadsPerWorker: 2))
{
    Console.WriteLine($"  workers: {string.Join(", ", cluster.Addresses)}");
    var onCluster = sql.Sql(query).Compute(cluster.Client);
    Console.WriteLine($"  the SQL query on the cluster: {onCluster.RowCount} rows, {cluster.Client.LastRun}");

    var leaves = Enumerable.Range(0, 24).Select(i => Dask.Delayed(Pipeline.Slow, Dask.Value((double)i))).ToList();
    while (leaves.Count > 1)
        leaves = leaves.Chunk(2).Select(p => p.Length == 2 ? Dask.Delayed(Pipeline.Combine, p[0], p[1]) : p[0]).ToList();
    var recomputed = 0;
    cluster.Client.OnEvent = e => { if (e.Kind == TaskEventKind.Recomputing) Interlocked.Increment(ref recomputed); };
    var run = Task.Run(() => leaves[0].Compute(cluster.Client));
    await Task.Delay(150);
    cluster.KillWorker(0);
    Console.WriteLine($"  killed worker 0 mid-run; sum 0..23 = {await run} (expect 276), {recomputed} task(s) recomputed from lineage");
}
Console.WriteLine();
Console.WriteLine(DaskNetInfo.Attribution);
return 0;

static void Section(string title)
    => Console.WriteLine($"--- {title} " + new string('-', Math.Max(0, 60 - title.Length)));

static string Indent(string text) => string.Join('\n', text.Split('\n').Select(l => "  " + l.TrimEnd('\r')));

static string? Resolve(string relative)
{
    foreach (var start in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
    {
        var directory = new DirectoryInfo(start);
        for (var depth = 0; directory is not null && depth < 12; depth++)
        {
            var candidate = Path.Combine(directory.FullName, relative);
            if (Directory.Exists(candidate)) return candidate;
            directory = directory.Parent;
        }
    }
    return null;
}

/// <summary>Task functions a worker may run: public, static, marked [RemoteTask].</summary>
public static class Pipeline
{
    /// <summary>Pretends to load a dataset.</summary>
    [RemoteTask] public static double Load(double id) => id * 10;

    /// <summary>Pretends to clean it.</summary>
    [RemoteTask] public static double Clean(double value) => value + 1;

    /// <summary>Combines two partial results.</summary>
    [RemoteTask] public static double Combine(double a, double b) => a + b;

    /// <summary>Takes 60 ms, so a run is long enough to lose a worker in the middle.</summary>
    [RemoteTask]
    public static double Slow(double value)
    {
        Thread.Sleep(60);
        return value;
    }
}
