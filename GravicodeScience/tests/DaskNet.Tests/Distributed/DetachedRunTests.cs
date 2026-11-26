using System.Diagnostics;
using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Gravicode.Science.GraviFrame;
using Xunit;
using static Gravicode.Science.DaskNet.Expr;

namespace Gravicode.Science.Tests.DaskNet;

/// <summary>Runs scheduled by a worker (the submitting client may go away), and speculative
/// re-execution of stragglers.</summary>
public sealed class DetachedRunTests
{
    private static Delayed<double> Sum(int n, int milliseconds)
    {
        var leaves = Enumerable.Range(0, n)
            .Select(i => Dask.Delayed(RemoteFunctions.Slow, Dask.Value((double)i), Dask.Value(milliseconds))).ToList();
        while (leaves.Count > 1)
            leaves = leaves.Chunk(2).Select(p => p.Length == 2 ? Dask.Delayed(RemoteFunctions.Add, p[0], p[1]) : p[0]).ToList();
        return leaves[0];
    }

    private static async Task<RunStatus> WaitFor(Client client, RunHandle run, RunState state)
    {
        var clock = Stopwatch.StartNew();
        while (true)
        {
            var status = await client.StatusAsync(run);
            if (status.State == state) return status;
            Assert.True(status.State == RunState.Running, $"run ended as {status.State}: {status.Error}");
            Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), "run never reached " + state);
            await Task.Delay(20);
        }
    }

    [Fact]
    public async Task A_run_outlives_the_client_that_submitted_it()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 3, threadsPerWorker: 2);
        var submitter = await Client.ConnectAsync(cluster.Addresses, cluster.Token);
        var run = await submitter.SubmitAsync(Sum(24, 40));
        var saved = run.ToString();
        await submitter.DisposeAsync();                       // the submitting process "dies"

        await using var other = await Client.ConnectAsync(cluster.Addresses, cluster.Token);
        var handle = RunHandle.Parse(saved);
        Assert.Equal(run.Id, handle.Id);
        Assert.Equal(run.Outputs, handle.Outputs);
        var status = await WaitFor(other, handle, RunState.Finished);
        Assert.Equal(status.Total, status.Completed);
        Assert.Equal(Enumerable.Range(0, 24).Sum(), await other.ResultAsync<double>(handle));

        // Results are held until forgotten, then the run is unknown.
        Assert.Equal(Enumerable.Range(0, 24).Sum(), await other.ResultAsync<double>(handle));
        await other.ForgetAsync(handle);
        await Assert.ThrowsAsync<KeyNotFoundException>(() => other.StatusAsync(handle));
    }

    [Fact]
    public async Task Submitted_frames_failures_and_cancellation()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 2, threadsPerWorker: 2);
        var client = cluster.Client;

        var sales = DaskFrame.FromFrame(SalesData.Make(3_000), 6);
        var report = sales.Filter(Col("qty") > 3).GroupBy("region").Agg(("qty", "sum"), ("price", "mean"));
        var frameRun = await client.SubmitAsync(report);
        Assert.Equal(SalesData.Rows(report.Compute(), sorted: true), SalesData.Rows(await client.FrameResultAsync(frameRun), sorted: true));

        var failing = Dask.Delayed(RemoteFunctions.Add, Dask.Delayed(RemoteFunctions.Fail, Dask.Value(1.0)), Dask.Value(2.0));
        var failed = await client.SubmitAsync(failing);
        var status = await WaitFor(client, failed, RunState.Failed);
        Assert.Contains("remote failure", status.Error);
        await Assert.ThrowsAsync<DaskTaskException>(() => client.ResultAsync<double>(failed));

        var slow = await client.SubmitAsync(Sum(40, 300));
        await client.CancelAsync(slow);
        await WaitFor(client, slow, RunState.Cancelled);
        await Assert.ThrowsAsync<OperationCanceledException>(() => client.ResultAsync<double>(slow));

        // Lambdas cannot leave the process.
        await Assert.ThrowsAsync<NotSupportedException>(() => client.SubmitAsync(Dask.Delayed((double x) => x + 1, Dask.Value(1.0))));
    }

    [Fact]
    public async Task A_straggler_is_overtaken_by_a_speculative_attempt()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 3, threadsPerWorker: 1);
        var client = cluster.Client;
        client.Speculation = new SpeculationPolicy(Multiplier: 3, Minimum: TimeSpan.FromMilliseconds(200), MinimumSamples: 3);
        var events = new List<TaskEvent>();
        client.OnEvent = e => { lock (events) events.Add(e); };

        var id = Guid.NewGuid().ToString("N");
        var tasks = Enumerable.Range(0, 9).Select(i => Dask.Delayed(RemoteFunctions.SlowFirstTime,
            Dask.Value((double)i), Dask.Value(i == 4 ? id : Guid.NewGuid().ToString("N")), Dask.Value(i == 4 ? 6_000 : 60), Dask.Value(60)));
        var clock = Stopwatch.StartNew();
        var values = Dask.Compute(tasks, client);
        var elapsed = clock.Elapsed;

        Assert.Equal(Enumerable.Range(0, 9).Select(i => (double)i), values);
        Assert.True(elapsed < TimeSpan.FromSeconds(4), $"took {elapsed}; the 6 s first attempt was waited for");
        Assert.True(client.LastRun!.Speculated >= 1);
        Assert.True(client.LastRun.SpeculationWins >= 1);
        Assert.Equal(2, RemoteFunctions.AttemptsOf(id));
        lock (events) Assert.Contains(events, e => e.Kind == TaskEventKind.Speculating);

        // Off by default: the same shape waits for the straggler.
        client.Speculation = null;
        var id2 = Guid.NewGuid().ToString("N");
        clock.Restart();
        Dask.Compute([Dask.Delayed(RemoteFunctions.SlowFirstTime, Dask.Value(1.0), Dask.Value(id2), Dask.Value(700), Dask.Value(10))], client);
        Assert.True(clock.Elapsed >= TimeSpan.FromMilliseconds(650));
        Assert.Equal(0, client.LastRun!.Speculated);
    }

    [Fact]
    public async Task Speculation_never_changes_answers()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 3, threadsPerWorker: 2);
        cluster.Client.Speculation = new SpeculationPolicy(Multiplier: 1.0, Minimum: TimeSpan.FromMilliseconds(1), MinimumSamples: 1);
        var a = DaskArray.Random([300, 300], [50, 50]);
        var expected = (a * 2 + 1).Sum(axis: 0).Compute();
        for (var i = 0; i < 3; i++)
            Assert.Equal(expected.ToArray(), (a * 2 + 1).Sum(axis: 0).Compute(cluster.Client).ToArray());
        var frame = DaskFrame.FromFrame(SalesData.Make(2_000), 8);
        var grouped = frame.GroupBy("region").Agg(("qty", "sum"));
        Assert.Equal(SalesData.Rows(grouped.Compute(), sorted: true), SalesData.Rows(grouped.Compute(cluster.Client), sorted: true));
    }
}
