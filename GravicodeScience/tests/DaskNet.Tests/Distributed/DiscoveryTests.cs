using Gravicode.Science.DaskNet;
using Gravicode.Science.DaskNet.Distributed;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

public sealed class DiscoveryTests
{
    private static string Token => new('d', 32);

    private static Task<Worker> StartWorker(string name) => Worker.StartAsync(new WorkerOptions { Token = Token, Threads = 2, Name = name });

    [Fact]
    public async Task Aspire_variables_become_addresses()
    {
        var variables = new Dictionary<string, string>
        {
            ["services__dask-workers__tcp__0"] = "tcp://localhost:51001",
            ["services__dask-workers__tcp__1"] = "tcp://10.0.0.7:51002",
            ["services__dask-workers__dashboard__0"] = "http://localhost:8787",
            ["SERVICES__DASK-WORKERS__TCP__2"] = "tcp://[::1]:51003",          // Windows keys are case-insensitive
            ["services__other__tcp__0"] = "tcp://elsewhere:1",
            ["services__dask-workers__tcp__notanumber"] = "tcp://ignored:1",
        };
        var tcp = await WorkerDiscovery.Aspire("dask-workers", "tcp", variables).DiscoverAsync();
        Assert.Equal(["localhost:51001", "10.0.0.7:51002", "[::1]:51003"], tcp);
        var all = await WorkerDiscovery.Aspire("dask-workers", variables: variables).DiscoverAsync();
        Assert.Equal(4, all.Count);
        Assert.Contains("localhost:8787", all);

        // One resource per worker, matched by a wildcard; "dask-worker" must not match "dask-workers".
        var perWorker = new Dictionary<string, string>
        {
            ["services__dask-worker-1__tcp__0"] = "tcp://localhost:7001",
            ["services__dask-worker-0__tcp__0"] = "tcp://localhost:7000",
            ["services__dask-workers__tcp__0"] = "tcp://localhost:9999",
        };
        Assert.Equal(["localhost:7000", "localhost:7001"], await WorkerDiscovery.Aspire("dask-worker-*", "tcp", perWorker).DiscoverAsync());
        Assert.Empty(await WorkerDiscovery.Aspire("dask-worker", "tcp", perWorker).DiscoverAsync());
    }

    [Fact]
    public async Task Static_environment_and_combined_sources()
    {
        var name = "DASKNET_TEST_WORKERS_" + Guid.NewGuid().ToString("N");
        Environment.SetEnvironmentVariable(name, "a:1, b:2;c:3");
        try
        {
            Assert.Equal(["a:1", "b:2", "c:3"], await WorkerDiscovery.EnvironmentVariable(name).DiscoverAsync());
            var failing = WorkerDiscovery.From(_ => throw new InvalidOperationException("registry down"), "broken registry");
            var combined = WorkerDiscovery.Combine(WorkerDiscovery.Static("a:1", "d:4"), WorkerDiscovery.EnvironmentVariable(name), failing);
            Assert.Equal(["a:1", "d:4", "b:2", "c:3"], await combined.DiscoverAsync());
            Assert.Contains("broken registry", combined.Describe());
        }
        finally { Environment.SetEnvironmentVariable(name, null); }
    }

    /// <summary>
    /// Kubernetes discovery is DNS: a headless Service resolves to every pod. <c>localhost</c>
    /// resolves to 127.0.0.1 and usually ::1, and the worker listens only on the first, so this
    /// also checks that an address that does not answer is skipped rather than fatal.
    /// </summary>
    [Fact]
    public async Task Dns_discovery_connects_to_what_answers()
    {
        await using var worker = await StartWorker("dns-worker");
        var port = WireConnection.ParseAddress(worker.Address).Port;
        var found = await WorkerDiscovery.Dns("localhost", port).DiscoverAsync();
        Assert.Contains($"127.0.0.1:{port}", found);

        await using var client = await Client.ConnectAsync(WorkerDiscovery.Dns("localhost", port), Token);
        Assert.Equal([$"127.0.0.1:{port}"], client.Workers);
        Assert.Equal(9.0, Dask.Delayed(RemoteFunctions.Square, Dask.Value(3.0)).Compute(client));
    }

    [Fact]
    public async Task Nothing_reachable_is_an_error_that_names_the_source()
    {
        var e = await Assert.ThrowsAsync<InvalidOperationException>(() => Client.ConnectAsync(WorkerDiscovery.Static(), Token));
        Assert.Contains("found no workers", e.Message);
        var closed = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            Client.ConnectAsync(WorkerDiscovery.Static("127.0.0.1:1"), Token));
        Assert.Contains("127.0.0.1:1", closed.Message);

        // A wrong token is not "unreachable": it fails loudly instead of being skipped.
        await using var worker = await StartWorker("w");
        await Assert.ThrowsAsync<UnauthorizedAccessException>(() =>
            Client.ConnectAsync(WorkerDiscovery.Static(worker.Address), new string('x', 32)));
    }

    [Fact]
    public async Task Workers_found_later_join_and_take_work()
    {
        await using var first = await StartWorker("first");
        var addresses = new List<string> { first.Address };
        var discovery = WorkerDiscovery.From(_ => { lock (addresses) return Task.FromResult<IReadOnlyList<string>>(addresses.ToArray()); });
        await using var client = await Client.ConnectAsync(discovery, Token);
        Assert.Single(client.Workers);

        await using var second = await StartWorker("second");
        lock (addresses) addresses.Add(second.Address);
        Assert.Equal(1, await client.RefreshAsync());
        Assert.Equal(0, await client.RefreshAsync());   // already connected
        Assert.Equal(2, client.Workers.Count);

        var x = DaskArray.Random([400, 400], [50, 50], seed: 2);
        Assert.Equal(x.Sum().Compute(), x.Sum().Compute(client));
        Assert.Equal(2, client.LastRun!.TasksPerWorker.Count);
    }

    /// <summary>A worker that appears while a run is going is used by that run.</summary>
    [Fact]
    public async Task Background_refresh_adds_a_worker_in_the_middle_of_a_run()
    {
        await using var first = await StartWorker("first");
        var addresses = new List<string> { first.Address };
        var discovery = WorkerDiscovery.From(_ => { lock (addresses) return Task.FromResult<IReadOnlyList<string>>(addresses.ToArray()); });
        await using var client = await Client.ConnectAsync(discovery, Token, refreshEvery: TimeSpan.FromMilliseconds(50));

        Worker? second = null;
        var events = new List<TaskEvent>();
        client.OnEvent = e =>
        {
            lock (events)
            {
                events.Add(e);
                // On the first finished task, bring up a second worker and publish it.
                if (second is null && e.Kind == TaskEventKind.Finished)
                {
                    second = StartWorker("late").GetAwaiter().GetResult();
                    lock (addresses) addresses.Add(second.Address);
                }
            }
        };
        try
        {
            var leaves = Enumerable.Range(0, 40).Select(i => Dask.Delayed(RemoteFunctions.Slow, Dask.Value((double)i), Dask.Value(40))).ToArray();
            var total = leaves.Skip(1).Aggregate(leaves[0], (acc, d) => Dask.Delayed(RemoteFunctions.Add, acc, d));
            Assert.Equal(Enumerable.Range(0, 40).Sum(), total.Compute(client));
            Assert.NotNull(second);
            Assert.True(client.LastRun!.TasksPerWorker.ContainsKey("late"), $"ran on: {string.Join(", ", client.LastRun.TasksPerWorker.Keys)}");
        }
        finally
        {
            if (second is not null) await second.DisposeAsync();
        }
    }
}
