using System.Diagnostics;
using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

/// <summary>Large results cross between processes on one machine through shared memory.</summary>
public sealed class SharedMemoryTests(ITestOutputHelper output)
{
    private const string Token = "shared-memory-test-token";

    private static async Task<(Worker[] Workers, Client Client)> Start(int count, long? threshold)
    {
        var workers = new Worker[count];
        for (var i = 0; i < count; i++)
            workers[i] = await Worker.StartAsync(new WorkerOptions { Token = Token, Threads = 1, SharedMemoryThreshold = threshold });
        return (workers, await Client.ConnectAsync(workers.Select(w => w.Address), Token));
    }

    private static async Task Stop(Worker[] workers, Client client)
    {
        await client.DisposeAsync();
        foreach (var w in workers) await w.DisposeAsync();
    }

    [Fact]
    public async Task Shared_memory_results_are_bit_identical_and_released()
    {
        var a = DaskArray.Random([1200, 1200], [600, 600], seed: 7);   // four 2.9 MB blocks
        var expectedBlocks = a.Compute().ToArray();
        var expectedSum = (a * 2 + 1).Sum(axis: 0).Compute().ToArray();

        var (workers, client) = await Start(2, threshold: 1 << 20);
        try
        {
            Assert.Equal(expectedBlocks, a.Compute(client).ToArray());          // gathered by the client
            Assert.Equal(expectedSum, (a * 2 + 1).Sum(axis: 0).Compute(client).ToArray());
            Assert.True(workers.Sum(w => w.SharedMemoryBytesSent) >= 4L * 600 * 600 * 8, "the blocks did not go through shared memory");
            Assert.All(workers, w => Assert.Equal(0, w.LiveSharedSegments));    // every reader said it was done

            // The client can opt out; then every block crosses the socket.
            client.SharedMemory = false;
            var before = workers.Sum(w => w.SharedMemoryBytesSent);
            Assert.Equal(expectedBlocks, a.Compute(client).ToArray());
            Assert.Equal(before, workers.Sum(w => w.SharedMemoryBytesSent));
        }
        finally { await Stop(workers, client); }
    }

    [Fact]
    public async Task A_view_reads_the_segment_in_place_until_disposed()
    {
        var a = DaskArray.Random([1200, 1200], [600, 600], seed: 11);
        var block = a.ToDelayed()[0];                                      // one 600x600 block, 2.9 MB
        var expected = block.Compute().ToArray();
        var (workers, client) = await Start(2, threshold: 1 << 20);
        try
        {
            var view = await client.ComputeViewAsync(block).WaitAsync(TimeSpan.FromSeconds(30));
            Assert.True(view.IsShared);
            Assert.Equal([600, 600], view.Shape);
            Assert.Equal(expected, view.Span.ToArray());
            Assert.Equal(expected[12345], view[12345]);
            Assert.Equal(expected, view.Memory.ToArray());
            Assert.Equal(expected, view.ToNdArray().ToArray());
            // The segment stays lent while the view lives, and goes back when it is disposed.
            Assert.Equal(1, workers.Sum(w => w.LiveSharedSegments));
            var memory = view.Memory;
            view.Dispose();
            view.Dispose();                                                // idempotent
            Assert.Throws<ObjectDisposedException>(() => view.Span.Length);
            Assert.Throws<ObjectDisposedException>(() => memory.Span.Length);
            await WaitFor(() => workers.Sum(w => w.LiveSharedSegments) == 0);

            // Several live views are distinct segments, not one reused buffer.
            var views = await Task.WhenAll(a.ToDelayed().Select(d => client.ComputeViewAsync(d))).WaitAsync(TimeSpan.FromSeconds(30));
            var blocks = a.ToDelayed().Select(d => d.Compute().ToArray()).ToArray();
            for (var i = 0; i < views.Length; i++) Assert.Equal(blocks[i], views[i].Span.ToArray());
            Assert.Equal(4, workers.Sum(w => w.LiveSharedSegments));
            foreach (var v in views) v.Dispose();
            await WaitFor(() => workers.Sum(w => w.LiveSharedSegments) == 0);

            // With shared memory off the view is over a copy.
            client.SharedMemory = false;
            using var copy = await client.ComputeViewAsync(block).WaitAsync(TimeSpan.FromSeconds(30));
            Assert.False(copy.IsShared);
            Assert.Equal(expected, copy.Span.ToArray());
        }
        finally { await Stop(workers, client); }
    }

    [Fact]
    public async Task Small_or_non_array_results_view_a_copy_or_refuse()
    {
        var small = DaskArray.Random([100, 100], [100, 100], seed: 2).ToDelayed()[0];   // 80 KB, under the threshold
        var (workers, client) = await Start(1, threshold: 1 << 20);
        try
        {
            using var view = await client.ComputeViewAsync(small);
            Assert.False(view.IsShared);
            Assert.Equal(small.Compute().ToArray(), view.Span.ToArray());
            Assert.Equal(0, workers[0].SharedMemoryBytesSent);
        }
        finally { await Stop(workers, client); }
    }

    private static async Task WaitFor(Func<bool> condition)
    {
        // SharedDone is sent without waiting for its reply; give it a moment to land.
        var clock = Stopwatch.StartNew();
        while (!condition())
        {
            if (clock.Elapsed > TimeSpan.FromSeconds(10)) Assert.Fail("condition not reached in 10 s");
            await Task.Delay(10);
        }
    }

    [Fact]
    public async Task Small_results_and_disabled_workers_use_the_socket()
    {
        var a = DaskArray.Random([400, 400], [100, 100], seed: 3);           // 80 KB blocks
        var (workers, client) = await Start(2, threshold: 1 << 20);
        try
        {
            Assert.Equal(a.Compute().ToArray(), a.Compute(client).ToArray());
            Assert.Equal(0, workers.Sum(w => w.SharedMemoryBytesSent));
            Assert.True(workers.Sum(w => w.SocketBytesSent) > 0);
        }
        finally { await Stop(workers, client); }

        var big = DaskArray.Random([1000, 1000], [500, 500], seed: 4);
        (workers, client) = await Start(2, threshold: null);
        try
        {
            Assert.Equal(big.Compute().ToArray(), big.Compute(client).ToArray());
            Assert.Equal(0, workers.Sum(w => w.SharedMemoryBytesSent));
        }
        finally { await Stop(workers, client); }
    }

    [Fact]
    public void A_segment_round_trips_every_value_the_codec_knows()
    {
        var frame = SalesData.Make(5_000);
        object?[] values = [frame, Enumerable.Range(0, 300_000).Select(i => i * 0.5).ToArray(), "text", null];
        foreach (var value in values)
        {
            var bytes = WireCodec.Serialize(value);
            Assert.Equal(bytes.LongLength, WireCodec.Measure(value));
            using var segments = new SharedSegments();
            var (id, mapping) = segments.Lend(bytes.LongLength);
            SharedMemory.Write(mapping, value, bytes.LongLength);
            Assert.Equal(bytes, WireCodec.Serialize(SharedMemory.Read(mapping.Locator, bytes.LongLength)));
            segments.Release(id);
            // Reused from the pool, rewritten with the encoded bytes this time.
            var (_, again) = segments.Lend(bytes.LongLength);
            Assert.Same(mapping, again);
            SharedMemory.Write(again, bytes);
            Assert.Equal(bytes, WireCodec.Serialize(SharedMemory.Read(again.Locator, bytes.LongLength)));
        }
    }

    /// <summary>Not a pass/fail test: the A/B behind the documented figure. One worker holds the
    /// value; only the fetch is timed, alternating paths in one process, best of many.</summary>
    [Theory]
    [InlineData(64)]
    [InlineData(8)]
    public async Task Measure_fetch_through_shared_memory_and_socket(int mebibytes)
    {
        var worker = await Worker.StartAsync(new WorkerOptions { Token = Token, Threads = 1 });
        try
        {
            var key = new TaskKey("held", 0);
            var value = new NdArray(Enumerable.Range(0, mebibytes * 131_072).Select(i => i * 0.25).ToArray(), mebibytes * 128, 1024);
            worker.Put(key, value);
            await using var rpc = await RpcClient.ConnectAsync(worker.Address, Token);
            async Task<double> Best(bool shared)
            {
                var best = double.MaxValue;
                for (var i = 0; i < 8; i++)
                {
                    var clock = Stopwatch.StartNew();
                    var (found, _) = await SharedMemory.GetAsync(rpc, key, shared, null, default);
                    best = Math.Min(best, clock.Elapsed.TotalMilliseconds);
                    Assert.True(found);
                }
                return best;
            }
            double socket = double.MaxValue, shm = double.MaxValue;
            for (var round = 0; round < 4; round++)
            {
                socket = Math.Min(socket, await Best(false));
                shm = Math.Min(shm, await Best(true));
            }
            output.WriteLine($"{mebibytes} MiB NdArray fetch: socket {socket:F1} ms, shared memory {shm:F1} ms ({socket / shm:F2}x)");
            var (_, back) = await SharedMemory.GetAsync(rpc, key, true, null, default);
            Assert.Equal(value.ToArray(), ((NdArray)back!).ToArray());
            Assert.Equal(0, worker.LiveSharedSegments);
        }
        finally { await worker.DisposeAsync(); }
    }
}
