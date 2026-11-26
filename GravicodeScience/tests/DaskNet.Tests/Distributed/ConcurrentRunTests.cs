using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

/// <summary>Several runs on one client at once share the workers' slots.</summary>
public sealed class ConcurrentRunTests
{
    private const string Token = "concurrent-runs-token-0123";

    [Fact]
    public async Task Concurrent_runs_on_a_saturated_cluster_never_stall()
    {
        // One thread per worker: two runs fill the cluster and the rest wait for a slot. A slot
        // freed by one run used to wake only that run, so the waiting ones slept forever.
        var workers = new Worker[2];
        for (var i = 0; i < workers.Length; i++)
            workers[i] = await Worker.StartAsync(new WorkerOptions { Token = Token, Threads = 1 });
        var client = await Client.ConnectAsync(workers.Select(w => w.Address), Token);
        try
        {
            var a = DaskArray.Random([1200, 1200], [300, 300], seed: 5);
            var parts = a.ToDelayed().ToArray();
            var expected = parts.Select(d => d.Compute().ToArray()).ToArray();
            var results = await Task.WhenAll(parts.Select(d => client.ExecuteAsync(Dask.Graph(d), d.OutputKeys)))
                .WaitAsync(TimeSpan.FromSeconds(60));
            for (var i = 0; i < parts.Length; i++)
                Assert.Equal(expected[i], ((Gravicode.Science.GraviNum.NdArray)results[i][parts[i].OutputKeys[0]]!).ToArray());

            // Views too, each holding its segment until disposed.
            var views = await Task.WhenAll(parts.Select(d => client.ComputeViewAsync(d))).WaitAsync(TimeSpan.FromSeconds(60));
            for (var i = 0; i < views.Length; i++) Assert.Equal(expected[i], views[i].Span.ToArray());
            foreach (var v in views) v.Dispose();
        }
        finally
        {
            await client.DisposeAsync();
            foreach (var w in workers) await w.DisposeAsync();
        }
    }
}
