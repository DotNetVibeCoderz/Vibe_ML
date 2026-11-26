using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;

namespace Gravicode.Science.Tests.DaskNet;

public sealed class WireCodecTests
{
    [Fact]
    public void Frames_round_trip_exactly()
    {
        var frame = new DataFrame([
            new NumericSeries("n", [1.5, double.NaN, -0.0, double.PositiveInfinity]),
            new TextSeries("t", ["a", null, "", "é"]),
            new BooleanSeries("b", new bool?[] { true, null, false, true }),
            new DateTimeSeries("d", new DateTime?[] { new DateTime(2024, 2, 29, 13, 0, 0, 123), null, DateTime.MinValue, DateTime.MaxValue }),
            CategoricalSeries.FromValues("c", ["x", "y", null, "x"]),
        ]);
        var back = (DataFrame)WireCodec.Deserialize(WireCodec.Serialize(frame))!;
        Assert.Equal(frame.ColumnNames, back.ColumnNames);
        Assert.Equal(frame.Columns.Select(c => c.DataType), back.Columns.Select(c => c.DataType));
        for (var c = 0; c < frame.ColumnCount; c++)
            for (var i = 0; i < frame.RowCount; i++)
                Assert.Equal(frame.Columns[c].GetValue(i), back.Columns[c].GetValue(i));
        Assert.True(double.IsNegative(((NumericSeries)back["n"])[2]));
    }

    [Fact]
    public void Values_and_calls_round_trip()
    {
        var values = new object?[]
        {
            null, true, 3, 4L, 2.5, "s", new[] { 1.0, 2.0 }, new[] { 1, 2 }, new[] { 1L }, new string?[] { "a", null },
            new[] { true }, new byte[] { 9 }, NdArray.Arange(6).Reshape(2, 3), new[] { new[] { 1.0 }, new[] { 2.0, 3.0 } },
            new TaskKey("k", 2), new TaskRef(new TaskKey("r")), new TaskRefList([new TaskKey("a", 0), new TaskKey("a", 1)]),
            new object?[] { 1, "x", null },
        };
        var back = (object?[])WireCodec.Deserialize(WireCodec.Serialize(values))!;
        Assert.Equal(values.Length, back.Length);
        Assert.Equal(new[] { 0.0, 1, 2, 3, 4, 5 }, ((NdArray)back[12]!).ToArray());
        Assert.Equal(new[] { 2, 3 }, ((NdArray)back[12]!).Shape.ToArray());
        Assert.Equal(values[14], back[14]);
        Assert.Equal(((TaskRefList)values[16]!).Keys, ((TaskRefList)back[16]!).Keys);

        var call = new RemoteCall(typeof(RemoteFunctions).GetMethod(nameof(RemoteFunctions.Add))!, new TaskRef(new TaskKey("x")), 2.0);
        var decoded = (RemoteCall)WireCodec.Deserialize(WireCodec.Serialize(call))!;
        Assert.Equal(call.Method, decoded.Method);
        Assert.Equal(5.0, decoded.Invoke(_ => 3.0));
    }

    [Fact]
    public void Unmarked_methods_cannot_be_named_on_the_wire()
    {
        var id = WireCodec.MethodId(typeof(File).GetMethod(nameof(File.Delete))!);
        Assert.Throws<ArgumentException>(() => WireCodec.ResolveMethod(id));
    }

    [Fact]
    public void Unknown_types_are_refused()
        => Assert.Throws<NotSupportedException>(() => WireCodec.Serialize(new Uri("http://example.com")));
}

public sealed class ClusterTests
{
    [Fact]
    public async Task Array_graphs_match_the_thread_scheduler()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 3, threadsPerWorker: 2);
        var a = DaskArray.RandomNormal([120, 80], [30, 20], seed: 5);
        var expr = (a.Dot(a.T) + 1.0).Sum(axis: 1);
        var onCluster = expr.Compute(cluster.Client);
        var onThreads = expr.Compute(new ThreadScheduler());
        Assert.Equal(onThreads.ToArray(), onCluster.ToArray());
        Assert.True(cluster.Client.LastRun!.TasksPerWorker.Count >= 2, "work should spread over workers");
        Assert.True(Math.Abs(a.Std().Compute(cluster.Client) - a.Std().Compute()) < 1e-15);
    }

    [Fact]
    public async Task Dataframe_plans_run_on_workers()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 2);
        var sales = SalesData.Make(1_500);
        var products = new DataFrame([
            new TextSeries("product", Enumerable.Range(0, 12).Select(i => (string?)$"p{i:D2}").ToArray()),
            new NumericSeries("weight", Enumerable.Range(0, 12).Select(i => i * 0.5).ToArray()),
        ]);
        var df = DaskFrame.FromFrame(sales, 6)
            .Filter(Col("price") > 20)
            .Join(DaskFrame.FromFrame(products, 2), "product", broadcast: false)   // a real shuffle across workers
            .WithColumn("load", Col("weight") * Col("qty"))
            .GroupBy("region").Agg(("load", "sum"), ("price", "mean"));

        Assert.True(df.IsRemotable);
        var expected = df.Compute(new ThreadScheduler());
        var actual = df.Compute(cluster.Client);
        Assert.Equal(SalesData.Rows(expected), SalesData.Rows(actual));
        Assert.Equal(SalesData.Rows(df.SortBy("load_sum").Compute()), SalesData.Rows(df.SortBy("load_sum").Compute(cluster.Client)));
    }

    [Fact]
    public async Task Delayed_method_groups_run_remotely()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 2);
        var x = Dask.Value(3.0);
        var y = Dask.Delayed(RemoteFunctions.Add, Dask.Delayed(RemoteFunctions.Square, x), Dask.Value(1.0));
        Assert.Equal(10.0, y.Compute(cluster.Client));
        using (Dask.UseScheduler(cluster.Client)) Assert.Equal(10.0, y.Compute());
    }

    [Fact]
    public async Task Local_delegates_are_refused_with_a_reason()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 1);
        var e = Assert.Throws<NotSupportedException>(() => Bag.Range(10).Map(x => x * 2).Compute(cluster.Client));
        Assert.Contains("RemoteTask", e.Message);
    }

    [Fact]
    public async Task Wrong_tokens_are_rejected()
    {
        await using var worker = await Worker.StartAsync(new WorkerOptions { Token = new string('a', 32), Threads = 1 });
        await Assert.ThrowsAsync<UnauthorizedAccessException>(() => Client.ConnectAsync([worker.Address], new string('b', 32)));
        Assert.Throws<ArgumentException>(() => Worker.StartAsync(new WorkerOptions { Token = "short" }).GetAwaiter().GetResult());
        await using var ok = await Client.ConnectAsync([worker.Address], new string('a', 32));
        Assert.Single(await ok.GetWorkerInfoAsync());
    }

    [Fact]
    public async Task Remote_failures_are_reported_and_retried()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 2, retry: new RetryPolicy(MaxRetries: 2));
        var bad = Dask.Delayed(RemoteFunctions.Fail, Dask.Value(1.0));
        var e = Assert.Throws<DaskTaskException>(() => bad.Compute(cluster.Client));
        var remote = Assert.IsType<RemoteTaskException>(e.InnerException);
        Assert.Equal(nameof(InvalidOperationException), remote.RemoteType);
        Assert.Equal(2, cluster.Client.LastRun!.Retries);
    }

    [Fact]
    public async Task Results_are_released_after_the_run()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 2);
        DaskArray.Ones([100, 100], 10).Sum().Compute(cluster.Client);
        Assert.All(cluster.InProcessWorkers, w => Assert.Equal(0, w.StoredKeys));
    }

    [Fact]
    public async Task A_lost_worker_is_recovered_by_recomputation()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 3, threadsPerWorker: 2);
        var (total, events) = await RunAndKill(cluster, victim: "worker-0", kill: () => cluster.KillWorker(0), gateMilliseconds: 30_000);
        Assert.Equal(Enumerable.Range(0, 12).Sum(), total);
        Assert.Equal(2, cluster.Client.Workers.Count);
        Assert.Contains(events, e => e.Kind == TaskEventKind.Recomputing);
    }

    /// <summary>
    /// Kills <paramref name="victim"/> the moment one of its leaf results exists. Every leaf is
    /// consumed together with a "gate" task, so no leaf result can have been released yet:
    /// recovery has to recompute it, whatever the machine's load. (A wall-clock kill passed
    /// alone and failed under the full suite, when no leaf had finished in time.) The gate opens
    /// only after the kill on in-process workers; a fixed 600 ms gate once closed before a
    /// starved thread pool ran the kill, leaving nothing to recompute. Child processes cannot
    /// see the event and wait <paramref name="gateMilliseconds"/>.
    /// </summary>
    internal static async Task<(double Total, List<TaskEvent> Events)> RunAndKill(LocalCluster cluster, string victim, Action kill, int gateMilliseconds)
    {
        var events = new List<TaskEvent>();
        var leafDone = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        cluster.Client.OnEvent = e =>
        {
            lock (events) events.Add(e);
            if (e.Kind == TaskEventKind.Finished && e.Label.Contains(nameof(RemoteFunctions.Slow)) && e.Worker == victim && !e.Key.Name.StartsWith("gate", StringComparison.Ordinal))
                leafDone.TrySetResult();
        };

        var gateId = Guid.NewGuid().ToString("N");
        var gate = Dask.Delayed(RemoteFunctions.Gate, Dask.Value(0.0), Dask.Value(gateId), Dask.Value(gateMilliseconds), name: "gate");
        var leaves = Enumerable.Range(0, 12)
            .Select(i => Dask.Delayed(RemoteFunctions.Add, Dask.Delayed(RemoteFunctions.Slow, Dask.Value((double)i), Dask.Value(30)), gate))
            .ToList();
        while (leaves.Count > 1)
            leaves = leaves.Chunk(2).Select(p => p.Length == 2 ? Dask.Delayed(RemoteFunctions.Add, p[0], p[1]) : p[0]).ToList();

        var run = Task.Run(() => leaves[0].Compute(cluster.Client));
        var first = await Task.WhenAny(leafDone.Task, run, Task.Delay(TimeSpan.FromSeconds(30)));
        Assert.True(first == leafDone.Task, "the victim never finished a leaf before the run ended");
        kill();
        RemoteFunctions.Release(gateId);
        var total = await run;
        lock (events) return (total, events.ToList());
    }
}

public sealed class ProcessClusterTests
{
    private static string Host => Path.Combine(AppContext.BaseDirectory, "DaskNet.Worker.dll");

    [Fact]
    public async Task Process_workers_run_preloaded_tasks_in_other_processes()
    {
        await using var cluster = await LocalCluster.StartProcessesAsync(Host, workers: 2, threadsPerWorker: 1,
            preload: [typeof(RemoteFunctions).Assembly.Location]);
        var info = await cluster.Client.GetWorkerInfoAsync();
        Assert.Equal(2, info.Select(i => i.ProcessId).Distinct().Count());
        Assert.DoesNotContain(Environment.ProcessId, info.Select(i => i.ProcessId));

        var pids = Dask.Compute(Enumerable.Range(0, 8).Select(i =>
            Dask.Delayed(RemoteFunctions.ProcessId, Dask.Delayed(RemoteFunctions.Slow, Dask.Value((double)i), Dask.Value(50)))), cluster.Client);
        Assert.All(pids, p => Assert.Contains(p, info.Select(i => i.ProcessId)));

        var a = DaskArray.Random([60, 60], [20, 20]);
        Assert.Equal(a.Sum().Compute(), a.Sum().Compute(cluster.Client));

        // Large blocks cross between the processes through shared memory, bit for bit.
        var big = DaskArray.Random([1200, 1200], [600, 600], seed: 11);
        Assert.Equal(big.Compute().ToArray(), big.Compute(cluster.Client).ToArray());
        Assert.Equal((big * 3).Sum(axis: 1).Compute().ToArray(), (big * 3).Sum(axis: 1).Compute(cluster.Client).ToArray());
        var after = await cluster.Client.GetWorkerInfoAsync();
        Assert.True(after.Sum(i => i.SharedMemoryBytesSent) >= 4L * 600 * 600 * 8, "blocks did not go through shared memory");
    }

    [Fact]
    public async Task A_killed_process_is_recovered()
    {
        await using var cluster = await LocalCluster.StartProcessesAsync(Host, workers: 3, threadsPerWorker: 2,
            preload: [typeof(RemoteFunctions).Assembly.Location]);
        var (total, events) = await ClusterTests.RunAndKill(cluster, victim: "process-1", kill: () => cluster.KillWorker(1), gateMilliseconds: 1_500);
        Assert.Equal(Enumerable.Range(0, 12).Sum(), total);
        Assert.Contains(events, e => e.Kind == TaskEventKind.Recomputing);
    }
}
