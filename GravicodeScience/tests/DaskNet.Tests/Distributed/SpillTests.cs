using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

public sealed class SpillTests
{
    private static string NewDirectory() => Path.Combine(Path.GetTempPath(), "dasknet-spill-test-" + Guid.NewGuid().ToString("N"));

    private static NdArray Block(int seed) => NdArray.Arange(1000) + seed;   // 8 KB estimated

    [Fact]
    public void The_store_spills_least_recently_used_first_and_reads_back_exactly()
    {
        var dir = NewDirectory();
        using (var store = new WorkerStore(memoryLimit: 50_000, target: 0.6, spillDirectory: dir, name: "t"))
        {
            var keys = Enumerable.Range(0, 10).Select(i => new TaskKey("block", i)).ToArray();
            for (var i = 0; i < 10; i++)
            {
                store.Set(keys[i], Block(i));
                if (i == 2) Assert.True(store.TryGet(keys[0], out _));   // block 0 is now more recent than 1 and 2
            }
            var stats = store.Statistics();
            Assert.Equal(10, stats.Keys);
            Assert.True(stats.MemoryBytes <= 30_000, $"memory {stats.MemoryBytes} is over the 30 KB target");
            Assert.Equal(10 - stats.SpilledKeys, stats.Keys - stats.SpilledKeys);
            Assert.True(stats.SpilledKeys >= 6);
            Assert.Equal(stats.SpilledKeys, Directory.GetFiles(dir, "*.bin", SearchOption.AllDirectories).Length);

            // Least recently used went first: 1 and 2 before 0, which was touched after them.
            var spilledFirst = store.Statistics();
            Assert.True(store.TryGetSerialized(keys[1], out var bytes1) && bytes1.Length > 0);

            // Every value comes back exactly, whether it was on disk or not.
            for (var i = 0; i < 10; i++)
            {
                Assert.True(store.TryGet(keys[i], out var value));
                Assert.Equal(Block(i).ToArray(), ((NdArray)value!).ToArray());
            }
            Assert.True(store.Statistics().Unspills >= spilledFirst.SpilledKeys);
            Assert.True(store.Statistics().MemoryBytes <= 30_000);

            // Releasing a spilled result deletes its file.
            foreach (var k in keys) store.Remove(k);
            Assert.Equal(new StoreStatistics(0, 0, 0, 0, store.Statistics().Spills, store.Statistics().Unspills), store.Statistics());
            Assert.Empty(Directory.GetFiles(dir, "*.bin", SearchOption.AllDirectories));
        }
        Assert.Empty(Directory.Exists(dir) ? Directory.GetDirectories(dir) : []);
        if (Directory.Exists(dir)) Directory.Delete(dir, true);
    }

    [Fact]
    public void Least_recently_used_order_decides_what_goes()
    {
        var dir = NewDirectory();
        try
        {
            using var store = new WorkerStore(memoryLimit: 30_000, target: 1.0, spillDirectory: dir, name: "t");
            var a = new TaskKey("a", 0);
            var b = new TaskKey("b", 0);
            var c = new TaskKey("c", 0);
            var d = new TaskKey("d", 0);
            store.Set(a, Block(0));
            store.Set(b, Block(1));
            store.Set(c, Block(2));
            store.TryGet(a, out _);    // order now b, c, a
            store.Set(d, Block(3));     // over 30 KB: b is the one to go
            Assert.Equal(1, store.Statistics().SpilledKeys);
            var onDisk = Directory.GetFiles(dir, "*.bin", SearchOption.AllDirectories);
            Assert.Single(onDisk);
            Assert.Contains("b", Path.GetFileName(onDisk[0]));
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
    }

    [Fact]
    public void A_result_larger_than_the_target_is_read_through_and_unencodable_ones_stay()
    {
        var dir = NewDirectory();
        try
        {
            using var store = new WorkerStore(memoryLimit: 10_000, target: 0.5, spillDirectory: dir, name: "t");
            var big = new TaskKey("big", 0);
            store.Set(big, Block(7));   // 8 KB against a 5 KB target: spilled at once
            Assert.Equal(1, store.Statistics().SpilledKeys);
            Assert.True(store.TryGet(big, out var value));
            Assert.Equal(Block(7).ToArray(), ((NdArray)value!).ToArray());
            Assert.Equal(1, store.Statistics().SpilledKeys);   // still on disk, not evicting everything else to fit

            var odd = new TaskKey("odd", 0);
            var unencodable = new object[] { new Uri("https://example.org"), Enumerable.Range(0, 2000).Select(i => (double)i).ToArray() };
            store.Set(odd, unencodable);
            Assert.True(store.TryGet(odd, out var same));
            Assert.Same(unencodable, same);
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
    }

    [Fact]
    public void Sizes_are_estimated_from_content()
    {
        Assert.InRange(SizeOf.Estimate(NdArray.Zeros(1000)), 8000, 8100);
        Assert.InRange(SizeOf.Estimate(new double[500]), 4000, 4100);
        var frame = new DataFrame([new NumericSeries("x", new double[100]), new TextSeries("s", Enumerable.Repeat((string?)"abcd", 100).ToArray())]);
        Assert.InRange(SizeOf.Estimate(frame), 800 + 100 * (8 + 32 + 8), 800 + 100 * (8 + 32 + 8) + 200);
        var list = Enumerable.Range(0, 100_000).ToList();
        Assert.InRange(SizeOf.Estimate(list), 100_000 * 40, 100_000 * 41);   // sampled, then scaled
        Assert.Equal(4L << 30, Worker.ParseBytes("4GB"));
        Assert.Equal(1536L << 20, Worker.ParseBytes("1.5 GiB"));
        Assert.Equal(12345, Worker.ParseBytes("12345"));
    }

    /// <summary>
    /// A cluster under a memory limit far below its data still gives the exact answer. Centring
    /// columns holds every block until all column means are known, so blocks are spilled while the
    /// means are computed and read back when the centred blocks are made.
    /// </summary>
    [Fact]
    public async Task A_worker_under_its_limit_spills_and_still_gives_the_same_answer()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 1, threadsPerWorker: 1, memoryLimit: 1L << 20);
        var x = DaskArray.RandomNormal([4000, 250], [200, 250], seed: 5);   // 20 blocks of 400 KB, 8 MB in all
        var centred = x - x.Mean(axis: 0);
        var expected = centred.Compute();
        var actual = centred.Compute(cluster.Client);
        Assert.Equal(expected.ToArray(), actual.ToArray());

        var worker = cluster.InProcessWorkers[0];
        var memory = worker.Memory;
        Assert.True(memory.Spills > 0, "nothing was spilled");
        Assert.True(memory.Unspills > 0, "nothing was read back");
        var info = (await cluster.Client.GetWorkerInfoAsync()).Single();
        Assert.True(info.MemoryBytes <= 1L << 20, $"{info.MemoryBytes} bytes in memory");
        Assert.Equal(memory.SpilledKeys, info.SpilledKeys);
    }

    [Fact]
    public async Task Spill_files_are_removed_when_the_worker_stops()
    {
        var dir = NewDirectory();
        try
        {
            await using (var worker = await Worker.StartAsync(new WorkerOptions
                         { Token = new string('t', 32), Threads = 1, MemoryLimit = 100_000, SpillDirectory = dir }))
            {
                await using var client = await Client.ConnectAsync([worker.Address], new string('t', 32));
                var x = DaskArray.Random([400, 100], [50, 100], seed: 1);
                var y = x - x.Mean(axis: 0);
                Assert.Equal(y.Compute().ToArray(), y.Compute(client).ToArray());
                Assert.True(worker.Memory.Spills > 0);
                // Results are released once the client has them, and their files with them.
                Assert.Equal(0, worker.Memory.SpilledKeys);
                Assert.Empty(Directory.GetFiles(dir, "*.bin", SearchOption.AllDirectories));
            }
            Assert.Empty(Directory.GetFiles(dir, "*", SearchOption.AllDirectories));
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
    }

    [Fact]
    public void Bad_limits_are_refused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Worker.StartAsync(new WorkerOptions { Token = new string('t', 32), MemoryLimit = 0 }).GetAwaiter().GetResult());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Worker.StartAsync(new WorkerOptions { Token = new string('t', 32), MemoryLimit = 1000, MemoryTarget = 1.5 }).GetAwaiter().GetResult());
    }
}
