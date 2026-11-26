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

    private static Worker WorkerAt(LocalCluster cluster, string address)
        => cluster.InProcessWorkers.Single(w => w.Address.Equals(address, StringComparison.OrdinalIgnoreCase));

    private static void Kill(LocalCluster cluster, string address)
        => cluster.KillWorker(cluster.Addresses.ToList().FindIndex(a => a.Equals(address, StringComparison.OrdinalIgnoreCase)));

    [Fact]
    public async Task The_standby_takes_over_when_the_scheduler_dies_mid_run()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 3, threadsPerWorker: 2);
        var client = cluster.Client;
        var run = await client.SubmitAsync(Sum(30, 60));
        Assert.NotNull(run.Standby);
        Assert.NotEqual(run.Scheduler, run.Standby);
        Assert.Equal(run.Standby, RunHandle.Parse(run.ToString()).Standby);

        // Kill on an event, not a clock: the scheduler has made progress the standby can see.
        var clock = Stopwatch.StartNew();
        while ((await client.StatusAsync(run)).Completed < 3)
        {
            Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), "the run made no progress");
            await Task.Delay(10);
        }
        var standby = WorkerAt(cluster, run.Standby!);
        Assert.Equal(0, standby.Takeovers);
        Kill(cluster, run.Scheduler);

        Assert.Equal(Enumerable.Range(0, 30).Sum(), await client.ResultAsync<double>(run));
        Assert.Equal(1, standby.Takeovers);
        var status = await client.StatusAsync(run);
        Assert.Equal(RunState.Finished, status.State);
        Assert.Equal(status.Total, status.Completed);
        await client.ForgetAsync(run);
        await Assert.ThrowsAsync<KeyNotFoundException>(() => client.StatusAsync(run));
    }

    [Fact]
    public async Task A_finished_run_is_served_from_the_standbys_copies_without_recomputing()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 4, threadsPerWorker: 2);
        var client = cluster.Client;
        client.RunStandbys = 2;
        var sales = DaskFrame.FromFrame(SalesData.Make(2_000), 4);
        var report = sales.GroupBy("region").Agg(("qty", "sum"));
        var run = await client.SubmitAsync(report);
        Assert.Equal(2, run.Standbys.Count);
        Assert.Equal(3, run.Standbys.Append(run.Scheduler).Distinct(StringComparer.OrdinalIgnoreCase).Count());
        var expected = SalesData.Rows(report.Compute(), sorted: true);
        Assert.Equal(expected, SalesData.Rows(await client.FrameResultAsync(run), sorted: true));
        var tasksRun = cluster.InProcessWorkers.Sum(w => w.TasksRun);

        // Finished means copied: the scheduler, then the first standby, can die and the next holder
        // serves the same outputs without running a task.
        Kill(cluster, run.Scheduler);
        Assert.Equal(expected, SalesData.Rows(await client.FrameResultAsync(run), sorted: true));
        var first = WorkerAt(cluster, run.Standbys[0]);
        Assert.Equal(1, first.Takeovers);
        Assert.Equal(1, first.TakeoversFromReplica);

        Kill(cluster, run.Standbys[0]);
        Assert.Equal(expected, SalesData.Rows(await client.FrameResultAsync(run), sorted: true));
        var second = WorkerAt(cluster, run.Standbys[1]);
        Assert.Equal(1, second.Takeovers);
        Assert.Equal(1, second.TakeoversFromReplica);
        Assert.Equal(tasksRun, cluster.InProcessWorkers.Sum(w => w.TasksRun));

        await client.ForgetAsync(run);
        await Assert.ThrowsAsync<KeyNotFoundException>(() => client.StatusAsync(run));
    }

    [Fact]
    public async Task With_two_standbys_a_run_survives_losing_its_scheduler_and_first_standby_mid_run()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 4, threadsPerWorker: 2);
        var client = cluster.Client;
        Assert.Throws<ArgumentOutOfRangeException>(() => client.RunStandbys = -1);
        client.RunStandbys = 2;
        var run = await client.SubmitAsync(Sum(30, 60));
        Assert.Equal(2, run.Standbys.Count);
        Assert.Equal(run.Standbys, RunHandle.Parse(run.ToString()).Standbys);

        var clock = Stopwatch.StartNew();
        while ((await client.StatusAsync(run)).Completed < 3)
        {
            Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), "the run made no progress");
            await Task.Delay(10);
        }
        var last = WorkerAt(cluster, run.Standbys[1]);
        Kill(cluster, run.Scheduler);
        Kill(cluster, run.Standbys[0]);

        Assert.Equal(Enumerable.Range(0, 30).Sum(), await client.ResultAsync<double>(run));
        Assert.Equal(1, last.Takeovers);
        Assert.Equal(0, last.TakeoversFromReplica);           // it died mid-run: nothing to copy yet
        Assert.Equal(1, cluster.InProcessWorkers.Sum(w => w.Takeovers));
        await client.ForgetAsync(run);
    }

    [Fact]
    public void Handles_carry_their_succession_and_read_the_older_form()
    {
        var keys = new[] { new TaskKey("sum", 0) };
        var run = new RunHandle("r", "tcp://a:1", keys) { Standbys = ["tcp://b:2", "tcp://c:3"] };
        var back = RunHandle.Parse(run.ToString());
        Assert.Equal(run.Standbys, back.Standbys);
        Assert.Equal("tcp://b:2", back.Standby);

        // A 2.4 handle has a single "standby".
        var old = RunHandle.Parse("""{"id":"r","scheduler":"tcp://a:1","standby":"tcp://b:2","outputs":[{"name":"sum","index":0}]}""");
        Assert.Equal(["tcp://b:2"], old.Standbys);
        Assert.Empty(RunHandle.Parse("""{"id":"r","scheduler":"tcp://a:1","standby":null,"outputs":[]}""").Standbys);
        Assert.Equal(["tcp://z:9"], (run with { Standby = "tcp://z:9" }).Standbys);
    }

    [Fact]
    public async Task Forgetting_or_cancelling_never_wakes_the_standby()
    {
        await using var cluster = await LocalCluster.StartAsync(workers: 2, threadsPerWorker: 2);
        var client = cluster.Client;
        var done = await client.SubmitAsync(Sum(4, 5));
        Assert.Equal(6, await client.ResultAsync<double>(done));
        await client.ForgetAsync(done);

        var slow = await client.SubmitAsync(Sum(40, 300));
        await client.CancelAsync(slow);
        await WaitFor(client, slow, RunState.Cancelled);

        await Task.Delay(Worker.StandbyPoll * (Worker.StandbyStrikes + 3));
        Assert.All(cluster.InProcessWorkers, w => Assert.Equal(0, w.Takeovers));
        await Assert.ThrowsAsync<KeyNotFoundException>(() => client.StatusAsync(done));

        // One worker: no standby, nothing changes.
        await using var single = await LocalCluster.StartAsync(workers: 1, threadsPerWorker: 2);
        var alone = await single.Client.SubmitAsync(Sum(4, 5));
        Assert.Null(alone.Standby);
        Assert.Equal(6, await single.Client.ResultAsync<double>(alone));
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
