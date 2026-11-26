using Gravicode.Science.DaskNet;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

public sealed class GraphAndSchedulerTests
{
    public static TheoryData<string> Schedulers => ["sync", "threads"];

    private static IScheduler Make(string name) => name == "sync" ? new SynchronousScheduler() : new ThreadScheduler(4);

    [Theory]
    [MemberData(nameof(Schedulers))]
    public void Diamond_computes_the_shared_ancestor_once(string scheduler)
    {
        var calls = 0;
        var root = Dask.Delayed(() => { Interlocked.Increment(ref calls); return 10; });
        var left = root.Then(x => x + 1);
        var right = root.Then(x => x * 2);
        var total = Dask.Delayed((a, b) => a + b, left, right);

        Assert.Equal(31, total.Compute(Make(scheduler)));
        Assert.Equal(1, calls);
    }

    [Fact]
    public void Compute_of_two_values_shares_one_graph()
    {
        var calls = 0;
        var root = Dask.Delayed(() => { Interlocked.Increment(ref calls); return 3.0; });
        var (a, b) = Dask.Compute(root.Then(x => x * x), root.Then(x => x + 1));
        Assert.Equal((9.0, 4.0), (a, b));
        Assert.Equal(1, calls);
    }

    [Fact]
    public void Cull_keeps_only_what_outputs_need()
    {
        var a = Dask.Value(1);
        var b = a.Then(x => x + 1);
        var unused = a.Then(x => x * 100);
        var graph = TaskGraph.FromCollections(b, unused).Cull(b.OutputKeys);
        Assert.Equal(2, graph.Count);
        Assert.False(graph.Contains(unused.Key));
    }

    [Fact]
    public void Fuse_collapses_linear_chains_but_not_fan_in()
    {
        var bag = Bag.FromSequence(Enumerable.Range(0, 100), 4).Map(x => x + 1).Filter(x => x % 2 == 0).Map(x => x * 3);
        var unfused = TaskGraph.FromCollections(bag).Cull(bag.OutputKeys);
        var fused = unfused.Fuse(bag.OutputKeys);
        Assert.Equal(16, unfused.Count);
        Assert.Equal(4, fused.Count);

        // A reduction's fan-in must stay parallel: its leaves each have one reader, but the
        // reader has several dependencies.
        var count = bag.Count();
        var g = Dask.Graph(count);
        Assert.Equal(5, g.Count); // 4 fused partition chains, the tree root fused with its finaliser
        Assert.Equal(Enumerable.Range(0, 100).Select(x => x + 1).Where(x => x % 2 == 0).Select(x => x * 3).ToList(),
            bag.Compute());
    }

    [Fact]
    public void Fused_remote_calls_stay_remotable()
    {
        var x = DaskArray.Arange(100, 25);
        var y = (x * 2.0 + 1.0).Exp().Log();
        var graph = Dask.Graph(y);
        Assert.True(graph.IsRemotable);
        Assert.Equal(4, graph.Count);
        var result = y.Compute();
        for (var i = 0; i < 100; i++) Assert.True(Math.Abs(result.At(i) - (2.0 * i + 1)) < 1e-9);
    }

    [Fact]
    public void Cycles_are_reported()
    {
        var a = new TaskKey("a");
        var b = new TaskKey("b");
        var graph = new TaskGraph([TaskSpec.Create(a, [b], v => v[0]), TaskSpec.Create(b, [a], v => v[0])]);
        var e = Assert.Throws<InvalidOperationException>(graph.Validate);
        Assert.Contains("cycle", e.Message);
    }

    [Fact]
    public void Missing_dependencies_are_reported()
    {
        var graph = new TaskGraph([TaskSpec.Create(new TaskKey("a"), [new TaskKey("ghost")], v => v[0])]);
        Assert.Throws<InvalidOperationException>(graph.Validate);
    }

    [Theory]
    [MemberData(nameof(Schedulers))]
    public void A_failing_task_names_its_key(string scheduler)
    {
        var bad = Dask.Value(1).Then<int>(_ => throw new InvalidDataException("boom"), "explode");
        var e = Assert.Throws<DaskTaskException>(() => bad.Compute(Make(scheduler)));
        Assert.EndsWith("explode", e.Label);
        Assert.IsType<InvalidDataException>(e.InnerException);
    }

    [Fact]
    public void Retries_recover_transient_failures()
    {
        var attempts = 0;
        var flaky = Dask.Delayed(() =>
        {
            if (Interlocked.Increment(ref attempts) < 3) throw new IOException("transient");
            return 42;
        });
        var scheduler = new ThreadScheduler(2, new RetryPolicy(MaxRetries: 3));
        Assert.Equal(42, flaky.Compute(scheduler));
        Assert.Equal(3, attempts);
        Assert.Equal(2, scheduler.LastRun!.Retries);
    }

    [Fact]
    public void Retries_respect_the_filter()
    {
        var attempts = 0;
        var flaky = Dask.Delayed<int>(() => { attempts++; throw new ArgumentException("permanent"); });
        var scheduler = new SynchronousScheduler(new RetryPolicy(5, ShouldRetry: e => e is IOException));
        Assert.Throws<DaskTaskException>(() => flaky.Compute(scheduler));
        Assert.Equal(1, attempts);
    }

    [Fact]
    public async Task Cancellation_stops_the_run()
    {
        using var cts = new CancellationTokenSource();
        var started = new TaskCompletionSource();
        var slow = Bag.Range(64, 64).Map(x =>
        {
            started.TrySetResult();
            Thread.Sleep(50);
            return x;
        });
        var run = Task.Run(() => slow.Compute(new ThreadScheduler(2), cts.Token));
        await started.Task;
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => run);
    }

    [Fact]
    public void Intermediate_results_are_released_as_the_reduction_proceeds()
    {
        // 200 partitions summed: depth-first ordering plus release-on-last-reader keeps only a
        // handful alive at once, not all 200.
        var bag = Bag.Range(20_000, 200).Map(x => (double)x);
        var scheduler = new SynchronousScheduler();
        Dask.FuseGraphs = true;
        var total = bag.Sum(x => x).Compute(scheduler);
        Assert.Equal(19_999.0 * 20_000 / 2, total);
        Assert.True(scheduler.LastRun!.PeakResultsHeld < 40, $"peak {scheduler.LastRun.PeakResultsHeld}");
    }

    [Fact]
    public void Events_report_every_task()
    {
        var events = new List<TaskEvent>();
        var scheduler = new SynchronousScheduler { OnEvent = e => { lock (events) events.Add(e); } };
        Bag.Range(10, 5).Count().Compute(scheduler);
        var finished = events.Count(e => e.Kind == TaskEventKind.Finished);
        Assert.Equal(scheduler.LastRun!.TasksTotal, finished);
        Assert.Equal(events.Count(e => e.Kind == TaskEventKind.Started), finished);
    }

    [Fact]
    public void Scoped_scheduler_applies_and_restores()
    {
        var sync = new SynchronousScheduler();
        using (Dask.UseScheduler(sync))
        {
            Assert.Same(sync, Dask.CurrentScheduler);
            Dask.Value(5).Then(x => x + 1).Compute();
            Assert.NotNull(sync.LastRun);
        }
        Assert.NotSame(sync, Dask.CurrentScheduler);
    }

    [Fact]
    public void Visualisation_lists_every_task_and_edge()
    {
        var total = Dask.Delayed((a, b) => a + b, Dask.Value(1, "one"), Dask.Value(2, "two"));
        var dot = total.Visualize();
        Assert.StartsWith("digraph", dot);
        Assert.Equal(2, dot.Split("->").Length - 1);
        var mermaid = Dask.Graph(total).ToMermaid();
        Assert.StartsWith("flowchart", mermaid);
        var high = Bag.Range(100, 10).Map(x => x * 2).Count().Visualize(highLevel: true);
        Assert.Contains("10 tasks", high);
    }

    [Fact]
    public void Remote_calls_refuse_unmarked_methods()
    {
        var method = typeof(Math).GetMethod(nameof(Math.Sqrt))!;
        Assert.Throws<ArgumentException>(() => new RemoteCall(method, 4.0));
    }

    [Fact]
    public void Static_method_groups_with_the_attribute_become_remotable()
    {
        var a = Dask.Value(3.0);
        var remote = Dask.Delayed(RemoteFunctions.Square, a);
        var local = Dask.Delayed(x => x * x, a);
        Assert.True(Dask.Graph(remote).IsRemotable);
        Assert.False(Dask.Graph(local).IsRemotable);
        Assert.Equal(9.0, remote.Compute());
    }

    [Fact]
    public void All_gathers_in_order()
    {
        var items = Enumerable.Range(0, 20).Select(i => Dask.Delayed(() => i * i)).ToArray();
        Assert.Equal(Enumerable.Range(0, 20).Select(i => i * i), Dask.All(items).Compute());
    }

    [Fact]
    public void Text_blocks_tile_a_file_exactly()
    {
        var path = Path.GetTempFileName();
        try
        {
            var lines = Enumerable.Range(0, 997).Select(i => $"line {i} {new string('x', i % 37)} é").ToArray();
            File.WriteAllText(path, string.Join("\r\n", lines) + "\r\n");
            var length = new FileInfo(path).Length;
            foreach (var block in new long[] { 1, 7, 64, 1000, length })
            {
                var read = GraphBuilders.ByteRanges(length, block)
                    .SelectMany(r => TextBlocks.ReadLines(path, r.Start, r.End)).ToArray();
                Assert.Equal(lines, read);
            }
        }
        finally { File.Delete(path); }
    }
}

public static class RemoteFunctions
{
    [RemoteTask]
    public static double Square(double x) => x * x;

    [RemoteTask]
    public static double Add(double x, double y) => x + y;

    [RemoteTask]
    public static double Slow(double x, int milliseconds)
    {
        Thread.Sleep(milliseconds);
        return x;
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, ManualResetEventSlim> Gates = new();

    /// <summary>
    /// Blocks until <see cref="Release"/> opens the named gate, or <paramref name="maxMilliseconds"/>
    /// pass. In-process workers see the test's gate; a child-process worker cannot, and waits the
    /// whole time.
    /// </summary>
    [RemoteTask]
    public static double Gate(double x, string id, int maxMilliseconds)
    {
        Gates.GetOrAdd(id, _ => new ManualResetEventSlim()).Wait(maxMilliseconds);
        return x;
    }

    public static void Release(string id) => Gates.GetOrAdd(id, _ => new ManualResetEventSlim()).Set();

    [RemoteTask]
    public static int ProcessId(double _) => Environment.ProcessId;

    [RemoteTask]
    public static double Fail(double _) => throw new InvalidOperationException("remote failure");

    [RemoteTask]
    public static string Shout(string text) => text.ToUpperInvariant();
}
