using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using static Gravicode.Science.DaskNet.Expr;

BenchmarkSwitcher.FromAssembly(typeof(SchedulerOverheadBenchmark).Assembly).Run(args, DefaultConfig.Instance
    .AddJob(Job.ShortRun.WithWarmupCount(3).WithIterationCount(5))
    .WithOptions(ConfigOptions.DisableOptimizationsValidator));
return;

/// <summary>
/// What a task costs: a graph of many trivial tasks, so the time measured is the scheduler's.
/// </summary>
/// <remarks>
/// Dask's rule of thumb is ~1 ms of work per task; this is the number behind that rule here.
/// Divide by <see cref="Tasks"/> for the per-task overhead of each scheduler.
/// </remarks>
[MemoryDiagnoser]
public class SchedulerOverheadBenchmark
{
    private Bag<int> _bag = null!;

    /// <summary>Partitions, hence map tasks (plus a tree of combine tasks).</summary>
    [Params(100, 1000)]
    public int Tasks { get; set; }

    /// <summary>Builds the graph once; only execution is timed.</summary>
    [GlobalSetup]
    public void Setup() => _bag = Bag.Range(Tasks * 10, Tasks).Map(x => x + 1);

    /// <summary>One thread, no queueing.</summary>
    [Benchmark(Baseline = true)]
    public double Synchronous() => _bag.Sum(x => x).Compute(new SynchronousScheduler());

    /// <summary>The default thread pool scheduler.</summary>
    [Benchmark]
    public double Threads() => _bag.Sum(x => x).Compute(new ThreadScheduler());
}

/// <summary>
/// A chunked reduction against the same reduction on one in-memory array: what chunking costs
/// when the data fits, and what the thread scheduler wins back.
/// </summary>
[MemoryDiagnoser]
public class ArrayReductionBenchmark
{
    private NdArray _whole = null!;
    private DaskArray _chunked = null!;

    /// <summary>Square side.</summary>
    [Params(2000, 4000)]
    public int Size { get; set; }

    /// <summary>Generates the matrix once, and persists the chunked copy so neither side pays for generation.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _chunked = DaskArray.RandomNormal([Size, Size], [Size / 4, Size / 4], seed: 1).Persist();
        _whole = _chunked.Compute();
    }

    /// <summary>GraviNum on the whole array.</summary>
    [Benchmark(Baseline = true)]
    public double GraviNumStd() => Statistics.Std(_whole);

    /// <summary>16 chunks, one thread.</summary>
    [Benchmark]
    public double ChunkedSync() => _chunked.Std().Compute(new SynchronousScheduler());

    /// <summary>16 chunks on the thread pool.</summary>
    [Benchmark]
    public double ChunkedThreads() => _chunked.Std().Compute(new ThreadScheduler());
}

/// <summary>
/// A filter-and-group-by over partitions against GraviFrame on the whole frame.
/// </summary>
[MemoryDiagnoser]
public class FrameGroupByBenchmark
{
    private DataFrame _frame = null!;
    private DaskFrame _partitioned = null!;

    /// <summary>Row count.</summary>
    [Params(200_000, 1_000_000)]
    public int Rows { get; set; }

    /// <summary>A frame of 50 keys with two numeric columns.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(5);
        _frame = new DataFrame([
            new TextSeries("key", Enumerable.Range(0, Rows).Select(_ => (string?)$"k{rng.Next(50):D2}").ToArray()),
            new NumericSeries("x", Enumerable.Range(0, Rows).Select(_ => rng.NextDouble()).ToArray()),
            new NumericSeries("y", Enumerable.Range(0, Rows).Select(_ => rng.NextDouble() * 100).ToArray()),
        ]);
        _partitioned = DaskFrame.FromFrame(_frame, Environment.ProcessorCount * 2).Persist();
    }

    /// <summary>GraviFrame: filter, then group.</summary>
    [Benchmark(Baseline = true)]
    public DataFrame GraviFrame()
        => _frame.FilterBy("y", y => y > 50).GroupBy("key").Mean("x", "y");

    /// <summary>Partitioned, on the thread pool.</summary>
    [Benchmark]
    public DataFrame DaskThreads()
        => _partitioned.Filter(Col("y") > 50).GroupBy("key").Mean("x", "y").Compute(new ThreadScheduler());
}

/// <summary>
/// The same reduction through an in-process cluster: every task and result crosses loopback TCP
/// and the wire codec, so this is the floor of what distribution costs.
/// </summary>
public class ClusterRoundTripBenchmark
{
    private LocalCluster _cluster = null!;
    private DaskArray _array = null!;

    /// <summary>Starts three workers.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _cluster = LocalCluster.StartAsync(workers: 3, threadsPerWorker: 2).GetAwaiter().GetResult();
        _array = DaskArray.Ones([1000, 1000], [250, 250]);
    }

    /// <summary>Stops the workers.</summary>
    [GlobalCleanup]
    public void Cleanup() => _cluster.DisposeAsync().AsTask().GetAwaiter().GetResult();

    /// <summary>Thread pool.</summary>
    [Benchmark(Baseline = true)]
    public double Threads() => (_array * 2.0).Sum().Compute(new ThreadScheduler());

    /// <summary>Three workers over loopback.</summary>
    [Benchmark]
    public double Cluster() => (_array * 2.0).Sum().Compute(_cluster.Client);
}
