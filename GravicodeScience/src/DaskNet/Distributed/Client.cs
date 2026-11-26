using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>What a worker reported about itself.</summary>
/// <param name="Address">Where it listens.</param>
/// <param name="Name">Its name.</param>
/// <param name="Threads">Tasks it runs at once.</param>
/// <param name="StoredKeys">Results it holds.</param>
/// <param name="TasksRun">Tasks it has run.</param>
/// <param name="ProcessId">Its operating-system process id.</param>
/// <param name="MemoryBytes">Estimated bytes of results in memory.</param>
/// <param name="SpilledKeys">Results spilled to disk.</param>
/// <param name="SpilledBytes">Bytes spilled to disk.</param>
public sealed record WorkerInfo(string Address, string Name, int Threads, int StoredKeys, long TasksRun, int ProcessId,
    long MemoryBytes, int SpilledKeys, long SpilledBytes)
{
    /// <summary>Encoded bytes of results it handed to readers on its machine through shared memory.</summary>
    public long SharedMemoryBytesSent { get; init; }

    /// <summary>Encoded bytes of results it sent to readers over the socket.</summary>
    public long SocketBytesSent { get; init; }
}

/// <summary>
/// A scheduler that runs task graphs on a cluster of <see cref="Worker"/>s (Dask's
/// <c>distributed.Client</c>). Pass it wherever an <see cref="IScheduler"/> goes, or scope it
/// with <see cref="Dask.UseScheduler"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Placement.</b> A ready task goes to the live worker that already holds the most of its
/// inputs, ties broken by the fewest tasks in flight, so data moves only when it must. Each worker
/// gets at most as many tasks as it has threads.
/// </para>
/// <para>
/// <b>Recovery.</b> When a worker drops, everything it held is gone. The client re-derives which
/// tasks must run from what the live workers still hold - walking back from the outputs and
/// stopping at data that survived - and runs exactly those again (Dask's lineage recomputation).
/// Tasks in flight elsewhere that needed lost data fail their fetch and are rescheduled; they are
/// not counted as task failures. A run fails only when a task's own code throws more often than
/// <see cref="Retry"/> allows, or no worker is left.
/// </para>
/// <para>
/// Only graphs whose every task has a <see cref="RemoteCall"/> form can run here - see
/// <see cref="RemoteTaskAttribute"/>. Array graphs and DataFrame plans without
/// <c>MapPartitions</c> qualify; bags of lambdas do not.
/// </para>
/// </remarks>
public sealed partial class Client : IScheduler, IAsyncDisposable
{
    // Replaced, never mutated: a run reads it while discovery adds workers. Lost workers stay in
    // it (closed), because a run's bookkeeping refers to their handles.
    private WorkerHandle[] _workers = [];

    /// <summary>The runs executing on this client right now; they share the workers' slots.</summary>
    private readonly ConcurrentDictionary<ClusterRun, byte> _activeRuns = new();

    /// <summary>Wakes every active run so one blocked on a full cluster dispatches again.</summary>
    private void SlotFreed()
    {
        foreach (var run in _activeRuns.Keys) run.Nudge();
    }
    private readonly string _token;
    private readonly TlsOptions? _tls;
    private readonly SemaphoreSlim _joining = new(1, 1);
    private IWorkerDiscovery? _discovery;
    private CancellationTokenSource? _refreshStop;
    private Task? _refreshLoop;

    private Client(string token, TlsOptions? tls)
    {
        _token = token;
        _tls = tls;
    }

    /// <summary>Retries for tasks whose own code fails (not for lost workers, which are always recovered).</summary>
    public RetryPolicy Retry { get; set; } = RetryPolicy.None;

    /// <summary>Receives an event per task state change.</summary>
    public Action<TaskEvent>? OnEvent { get; set; }

    /// <summary>Speculative re-execution of stragglers on idle workers; off (null) by default.</summary>
    public SpeculationPolicy? Speculation { get; set; }

    /// <summary>Asks workers on this machine for large results through shared memory (their
    /// <see cref="WorkerOptions.SharedMemoryThreshold"/> decides what is large); on by default.</summary>
    public bool SharedMemory { get; set; } = true;

    /// <summary>Statistics of the most recent run.</summary>
    public RunStatistics? LastRun { get; private set; }

    /// <inheritdoc />
    public string Name => "cluster";

    /// <summary>Addresses of the workers still connected.</summary>
    public IReadOnlyList<string> Workers => _workers.Where(w => !w.Rpc.IsClosed).Select(w => w.Address).ToArray();

    /// <summary>Connects to running workers over plain TCP.</summary>
    public static Task<Client> ConnectAsync(IEnumerable<string> addresses, string token, CancellationToken cancellationToken = default)
        => ConnectAsync(addresses, token, tls: null, cancellationToken);

    /// <summary>Connects to running workers, over TLS when <paramref name="tls"/> is given.</summary>
    public static async Task<Client> ConnectAsync(IEnumerable<string> addresses, string token, TlsOptions? tls,
        CancellationToken cancellationToken = default)
    {
        var client = new Client(token, tls);
        foreach (var address in addresses)
        {
            var rpc = await RpcClient.ConnectAsync(address, token, tls, cancellationToken).ConfigureAwait(false);
            var info = await Ping(rpc, cancellationToken).ConfigureAwait(false);
            client._workers = [.. client._workers, new WorkerHandle(address, rpc, info.Threads, info.Name)];
        }
        if (client._workers.Length == 0) throw new ArgumentException("No worker addresses given.");
        return client;
    }

    /// <summary>
    /// Connects to the workers <paramref name="discovery"/> finds, skipping any that do not answer
    /// (a pod still starting). With <paramref name="refreshEvery"/>, discovery runs again in the
    /// background and newly found workers join - between runs or in the middle of one - so the
    /// cluster can grow while it is used. Workers that leave are handled as lost workers always are.
    /// </summary>
    /// <exception cref="InvalidOperationException">No discovered worker could be reached.</exception>
    public static async Task<Client> ConnectAsync(IWorkerDiscovery discovery, string token, TlsOptions? tls = null,
        TimeSpan? refreshEvery = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(discovery);
        var client = new Client(token, tls) { _discovery = discovery };
        var (added, tried, failures) = await client.JoinAsync(cancellationToken).ConfigureAwait(false);
        if (added == 0)
        {
            await client.DisposeAsync().ConfigureAwait(false);
            throw new InvalidOperationException(tried.Count == 0
                ? $"{discovery.Describe()} found no workers."
                : $"No worker found by {discovery.Describe()} answered: {string.Join("; ", failures)}");
        }
        if (refreshEvery is { } every)
        {
            if (every <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(refreshEvery));
            client._refreshStop = new CancellationTokenSource();
            client._refreshLoop = client.RefreshLoopAsync(every, client._refreshStop.Token);
        }
        return client;
    }

    /// <summary>
    /// Runs discovery once and connects any worker not already connected (including one that
    /// came back at an address that was lost). Returns how many joined. Only for clients made
    /// with an <see cref="IWorkerDiscovery"/>.
    /// </summary>
    public async Task<int> RefreshAsync(CancellationToken cancellationToken = default)
    {
        if (_discovery is null) throw new InvalidOperationException("This client was given fixed addresses; connect with an IWorkerDiscovery to refresh.");
        return (await JoinAsync(cancellationToken).ConfigureAwait(false)).Added;
    }

    private async Task<(int Added, IReadOnlyList<string> Tried, List<string> Failures)> JoinAsync(CancellationToken cancellationToken)
    {
        var found = await _discovery!.DiscoverAsync(cancellationToken).ConfigureAwait(false);
        await _joining.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var live = _workers.Where(w => w.Alive).Select(w => w.Address).ToHashSet(StringComparer.OrdinalIgnoreCase);
            var fresh = found.Where(a => !live.Contains(a)).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
            var failures = new List<string>();
            var joined = await Task.WhenAll(fresh.Select(async address =>
            {
                try
                {
                    var rpc = await RpcClient.ConnectAsync(address, _token, _tls, cancellationToken).ConfigureAwait(false);
                    var info = await Ping(rpc, cancellationToken).ConfigureAwait(false);
                    return new WorkerHandle(address, rpc, info.Threads, info.Name);
                }
                catch (Exception e) when (e is not OperationCanceledException and not UnauthorizedAccessException
                                              and not System.Security.Authentication.AuthenticationException)
                {
                    lock (failures) failures.Add($"{address}: {e.Message}");
                    return null;
                }
            })).ConfigureAwait(false);
            var added = joined.OfType<WorkerHandle>().ToArray();
            if (added.Length > 0) _workers = [.. _workers, .. added];
            return (added.Length, found, failures);
        }
        finally { _joining.Release(); }
    }

    private async Task RefreshLoopAsync(TimeSpan every, CancellationToken stop)
    {
        while (!stop.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(every, stop).ConfigureAwait(false);
                await JoinAsync(stop).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (stop.IsCancellationRequested) { return; }
            catch (Exception) { /* discovery or a connection failed this time; try again next time */ }
        }
    }

    /// <summary>Asks every connected worker for its state.</summary>
    public async Task<IReadOnlyList<WorkerInfo>> GetWorkerInfoAsync(CancellationToken cancellationToken = default)
    {
        var live = _workers.Where(w => !w.Rpc.IsClosed).ToArray();
        return await Task.WhenAll(live.Select(w => Ping(w.Rpc, cancellationToken))).ConfigureAwait(false);
    }

    private static async Task<WorkerInfo> Ping(RpcClient rpc, CancellationToken cancellationToken)
    {
        var reply = await rpc.RequestAsync(MessageKind.Ping, cancellationToken: cancellationToken).ConfigureAwait(false);
        using var r = reply.Reader();
        return new WorkerInfo(rpc.Address, r.ReadString(), r.ReadInt32(), r.ReadInt32(), r.ReadInt64(), r.ReadInt32(),
            r.ReadInt64(), r.ReadInt32(), r.ReadInt64()) { SharedMemoryBytesSent = r.ReadInt64(), SocketBytesSent = r.ReadInt64() };
    }

    /// <inheritdoc />
    public async Task<IReadOnlyDictionary<TaskKey, object?>> ExecuteAsync(TaskGraph graph, IReadOnlyList<TaskKey> outputs,
        CancellationToken cancellationToken = default)
    {
        var local = graph.Tasks.Values.Where(t => !t.IsRemotable).Select(t => t.Label).Distinct().Take(5).ToArray();
        if (local.Length > 0)
            throw new NotSupportedException(
                $"Tasks [{string.Join(", ", local)}] wrap local delegates and cannot run on a cluster. " +
                "Use [RemoteTask] static methods, or a ThreadScheduler.");

        var run = new ClusterRun(this, graph.Cull(outputs), outputs.Distinct().ToArray(), cancellationToken);
        try { return await run.ExecuteAsync().ConfigureAwait(false); }
        finally { LastRun = run.Statistics; }
    }

    /// <summary>
    /// Computes an array and returns it as a view rather than a copy: from a worker on this
    /// machine, the view's span is that worker's shared-memory segment, so a large result is never
    /// copied into this process. Dispose the view to release the segment.
    /// </summary>
    /// <remarks>Results from other machines, or smaller than the worker's
    /// <see cref="WorkerOptions.SharedMemoryThreshold"/>, come back as a view over a decoded copy
    /// (<see cref="SharedArrayView.IsShared"/> false).</remarks>
    public Task<SharedArrayView> ComputeViewAsync(Delayed<NdArray> value, CancellationToken cancellationToken = default)
        => ViewAsync(value, cancellationToken);

    /// <inheritdoc cref="ComputeViewAsync(Delayed{NdArray}, CancellationToken)"/>
    public Task<SharedArrayView> ComputeViewAsync(Delayed<double[]> value, CancellationToken cancellationToken = default)
        => ViewAsync(value, cancellationToken);

    private async Task<SharedArrayView> ViewAsync(IDaskCollection value, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(value);
        var key = value.OutputKeys[0];
        var graph = Dask.Graph(value);
        var local = graph.Tasks.Values.Where(t => !t.IsRemotable).Select(t => t.Label).Distinct().Take(5).ToArray();
        if (local.Length > 0)
            throw new NotSupportedException($"Tasks [{string.Join(", ", local)}] wrap local delegates and cannot run on a cluster.");
        var run = new ClusterRun(this, graph.Cull([key]), [key], cancellationToken) { Views = true };
        try { return (SharedArrayView)(await run.ExecuteAsync().ConfigureAwait(false))[key]!; }
        finally { LastRun = run.Statistics; }
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_refreshStop is not null)
        {
            await _refreshStop.CancelAsync().ConfigureAwait(false);
            try { await _refreshLoop!.ConfigureAwait(false); } catch (OperationCanceledException) { }
            _refreshStop.Dispose();
            _refreshStop = null;
        }
        await _joining.WaitAsync().ConfigureAwait(false);
        try
        {
            foreach (var w in _workers) await w.Rpc.DisposeAsync().ConfigureAwait(false);
            _workers = [];
        }
        finally { _joining.Release(); }
    }

    internal sealed class WorkerHandle(string address, RpcClient rpc, int threads, string name)
    {
        public string Address { get; } = address;
        public RpcClient Rpc { get; } = rpc;
        public int Threads { get; } = threads;
        public string Name { get; } = name;
        private int _inFlight;

        /// <summary>Tasks sent to this worker and not yet answered, by every run of the client.</summary>
        public int InFlight => Volatile.Read(ref _inFlight);

        public void Started() => Interlocked.Increment(ref _inFlight);

        /// <summary>One attempt answered or abandoned. Other runs of the client may be waiting
        /// for this very slot, and they only look again when an event of theirs arrives, so each
        /// is woken.</summary>
        public void Ended(Client client)
        {
            Interlocked.Decrement(ref _inFlight);
            client.SlotFreed();
        }
        public bool Alive => !Rpc.IsClosed;
        /// <summary>Set once a shared-memory read from this worker failed (not the same machine after all).</summary>
        public volatile bool NoSharedMemory;
    }

    /// <summary>One <c>ExecuteAsync</c>: the state machine, driven from a single loop so it needs no locks.</summary>
    private sealed class ClusterRun
    {
        private readonly Client _client;
        private readonly TaskGraph _graph;
        private readonly TaskKey[] _outputs;
        private readonly HashSet<TaskKey> _outputSet;
        private readonly CancellationToken _cancel;
        private readonly IReadOnlyDictionary<TaskKey, List<TaskKey>> _dependents;
        private readonly IReadOnlyDictionary<TaskKey, int> _priority;
        private readonly Dictionary<TaskKey, HashSet<WorkerHandle>> _holders = [];
        private readonly Dictionary<TaskKey, WorkerHandle> _inFlight = [];
        private readonly Dictionary<TaskKey, int> _remaining = [];
        private readonly Dictionary<TaskKey, int> _readers = [];
        private readonly Dictionary<TaskKey, int> _attempts = [];
        private readonly Dictionary<TaskKey, Dictionary<TaskKey, WorkerHandle>> _sources = [];
        private readonly PriorityQueue<TaskKey, int> _ready = new();
        private readonly Channel<Func<Task>> _events = Channel.CreateUnbounded<Func<Task>>();
        private int _completed;

        // Speculation: when each attempt started, how long finished tasks took per label, the
        // second attempt of a straggler, and attempts that lost and still occupy a thread.
        private readonly Dictionary<TaskKey, long> _startedAt = [];
        private readonly Dictionary<string, List<double>> _durations = [];
        private readonly Dictionary<TaskKey, WorkerHandle> _copies = [];
        private readonly HashSet<(TaskKey, WorkerHandle)> _losers = [];

        public ClusterRun(Client client, TaskGraph graph, TaskKey[] outputs, CancellationToken cancel)
        {
            _client = client;
            _graph = graph;
            _outputs = outputs;
            _outputSet = outputs.ToHashSet();
            _cancel = cancel;
            _dependents = graph.Dependents();
            _priority = graph.Priorities(outputs);
            Statistics.TasksTotal = graph.Count;
        }

        public RunStatistics Statistics { get; } = new();

        /// <summary>Gather outputs as <see cref="SharedArrayView"/>s instead of values.</summary>
        public bool Views { get; init; }

        private readonly HashSet<WorkerHandle> _watched = [];

        /// <summary>Hooks the loss of every live worker not hooked yet - those present at the start,
        /// and any that discovery adds while the run is going.</summary>
        private void WatchNewWorkers()
        {
            foreach (var w in _client._workers)
            {
                if (!w.Alive || !_watched.Add(w)) continue;
                var worker = w;
                _ = worker.Rpc.Closed.ContinueWith(_ => _events.Writer.TryWrite(() => { OnWorkerLost(worker); return Task.CompletedTask; }),
                    TaskScheduler.Default);
            }
        }

        public async Task<IReadOnlyDictionary<TaskKey, object?>> ExecuteAsync()
        {
            var clock = Stopwatch.StartNew();
            _graph.Validate();
            WatchNewWorkers();
            using var stopTicks = new CancellationTokenSource();
            if (_client.Speculation is { } policy) _ = TickAsync(policy, stopTicks.Token);

            _client._activeRuns[this] = 0;
            try
            {
                Replan();
                while (!AllOutputsAvailable())
                {
                    _cancel.ThrowIfCancellationRequested();
                    WatchNewWorkers();
                    if (!_client._workers.Any(w => w.Alive))
                        throw new WorkerLostException("all", new InvalidOperationException("Every worker is gone."));
                    Dispatch();
                    if (_inFlight.Count == 0 && _ready.Count == 0)
                    {
                        if (AllOutputsAvailable()) break;
                        throw new InvalidOperationException("The cluster scheduler stalled with nothing ready; this is a bug.");
                    }
                    var next = await _events.Reader.ReadAsync(_cancel).ConfigureAwait(false);
                    await next().ConfigureAwait(false);
                }

                var results = new Dictionary<TaskKey, object?>();
                foreach (var key in _outputs) results[key] = await GatherAsync(key).ConfigureAwait(false);
                return results;
            }
            finally
            {
                _client._activeRuns.TryRemove(this, out _);
                await stopTicks.CancelAsync().ConfigureAwait(false);
                await ReleaseAllAsync().ConfigureAwait(false);
                Statistics.Elapsed = clock.Elapsed;
            }
        }

        /// <summary>An empty event: the loop runs <c>Dispatch</c> again after it.</summary>
        public void Nudge() => _events.Writer.TryWrite(static () => Task.CompletedTask);

        /// <summary>Wakes the loop now and then so stragglers are noticed while nothing else happens.</summary>
        private async Task TickAsync(SpeculationPolicy policy, CancellationToken stop)
        {
            var every = TimeSpan.FromMilliseconds(Math.Clamp(policy.EffectiveMinimum.TotalMilliseconds / 4, 20, 500));
            try
            {
                while (!stop.IsCancellationRequested)
                {
                    await Task.Delay(every, stop).ConfigureAwait(false);
                    _events.Writer.TryWrite(() => { Speculate(policy); return Task.CompletedTask; });
                }
            }
            catch (OperationCanceledException) { }
        }

        /// <summary>Gives each straggler, at most once, a second attempt on an idle worker - only
        /// when nothing is waiting to run, so speculation never delays real work.</summary>
        private void Speculate(SpeculationPolicy policy)
        {
            if (_ready.Count > 0) return;
            var now = Stopwatch.GetTimestamp();
            foreach (var (key, worker) in _inFlight.ToArray())
            {
                if (_copies.ContainsKey(key) || !_startedAt.TryGetValue(key, out var started)) continue;
                var task = _graph[key];
                if (!_durations.TryGetValue(task.Label, out var seen) || seen.Count < policy.MinimumSamples) continue;
                var sorted = seen.Order().ToArray();
                var median = sorted[sorted.Length / 2];
                var threshold = Math.Max(policy.EffectiveMinimum.TotalMilliseconds, policy.Multiplier * median);
                if (Stopwatch.GetElapsedTime(started, now).TotalMilliseconds < threshold) continue;
                if (!task.Dependencies.All(Available)) continue;
                var idle = _client._workers.Where(w => w.Alive && w != worker && w.InFlight < w.Threads).OrderBy(w => w.InFlight).FirstOrDefault();
                if (idle is null) return;
                _copies[key] = idle;
                Statistics.Speculated++;
                _client.OnEvent?.Invoke(new TaskEvent(TaskEventKind.Speculating, key, task.Label, _completed, _graph.Count, idle.Name));
                Request(task, idle);
            }
        }

        private bool Available(TaskKey key) => _holders.TryGetValue(key, out var h) && h.Any(w => w.Alive);

        private bool AllOutputsAvailable() => _outputs.All(Available);

        /// <summary>
        /// Recomputes what must run from what is held now: walk back from the outputs, stopping
        /// at data some live worker has. Called at the start and after every loss.
        /// </summary>
        private void Replan()
        {
            // Losses are reported here, not in OnWorkerLost: a handle is marked dead as soon as its
            // connection drops, but the loss event is queued, and a Dispatch that runs first
            // replans and drops the dead holders - the loss handler then found nothing to report.
            foreach (var (key, holders) in _holders.ToArray())
            {
                var dead = holders.FirstOrDefault(w => !w.Alive);
                if (dead is null && holders.Count > 0) continue;
                holders.RemoveWhere(w => !w.Alive);
                if (holders.Count > 0) continue;
                _holders.Remove(key);
                if (dead is not null)
                    _client.OnEvent?.Invoke(new TaskEvent(TaskEventKind.Recomputing, key, _graph.Contains(key) ? _graph[key].Label : "?", _completed, _graph.Count, dead.Name));
            }

            var needed = new HashSet<TaskKey>();
            var stack = new Stack<TaskKey>(_outputs);
            while (stack.Count > 0)
            {
                var key = stack.Pop();
                if (Available(key) || !needed.Add(key)) continue;
                foreach (var d in _graph[key].Dependencies) stack.Push(d);
            }

            _remaining.Clear();
            _ready.Clear();
            _readers.Clear();
            foreach (var key in needed)
            {
                foreach (var d in _graph[key].Dependencies.Distinct())
                    _readers[d] = _readers.GetValueOrDefault(d) + 1;
                if (_inFlight.ContainsKey(key)) continue;
                var missing = _graph[key].Dependencies.Distinct().Count(d => !Available(d));
                _remaining[key] = missing;
                if (missing == 0) _ready.Enqueue(key, _priority[key]);
            }
        }

        private void Dispatch()
        {
            while (_ready.Count > 0)
            {
                var candidates = _client._workers.Where(w => w.Alive && w.InFlight < w.Threads).ToArray();
                if (candidates.Length == 0) return;
                var key = _ready.Dequeue();
                if (!_remaining.Remove(key)) continue; // stale entry from before a replan
                var task = _graph[key];
                if (!task.Dependencies.All(Available))
                {
                    // A holder dropped and the loss event is still queued: start over from what is held.
                    Replan();
                    continue;
                }
                var worker = candidates
                    .OrderByDescending(w => task.Dependencies.Count(d => _holders.TryGetValue(d, out var h) && h.Contains(w)))
                    .ThenBy(w => w.InFlight)
                    .First();
                Send(task, worker);
            }
        }

        private void Send(TaskSpec task, WorkerHandle worker)
        {
            _inFlight[task.Key] = worker;
            _startedAt[task.Key] = Stopwatch.GetTimestamp();
            _attempts[task.Key] = _attempts.GetValueOrDefault(task.Key) + 1;
            _client.OnEvent?.Invoke(new TaskEvent(TaskEventKind.Started, task.Key, task.Label, _completed, _graph.Count, worker.Name));
            Request(task, worker);
        }

        /// <summary>Sends one attempt of <paramref name="task"/> to <paramref name="worker"/>.</summary>
        private void Request(TaskSpec task, WorkerHandle worker)
        {
            worker.Started();
            Statistics.TasksRun++;
            Statistics.TasksPerWorker[worker.Name] = Statistics.TasksPerWorker.GetValueOrDefault(worker.Name) + 1;

            var sources = new Dictionary<TaskKey, WorkerHandle>();
            foreach (var d in task.Dependencies)
            {
                var holders = _holders[d];
                sources[d] = holders.Contains(worker) && worker.Alive ? worker : holders.First(h => h.Alive);
            }
            _sources[task.Key] = sources;
            var deps = sources.Select(p => (p.Key, p.Value.Address)).ToArray();

            var request = worker.Rpc.RequestAsync(MessageKind.Run, w =>
            {
                WireCodec.WriteKey(w, task.Key);
                WireCodec.Write(w, task.Remote);
                w.Write(deps.Length);
                foreach (var (d, address) in deps)
                {
                    WireCodec.WriteKey(w, d);
                    w.Write(address);
                }
            }, _cancel);

            _ = request.ContinueWith(t => _events.Writer.TryWrite(() => OnReplyAsync(task, worker, t)), TaskScheduler.Default);
        }

        private async Task OnReplyAsync(TaskSpec task, WorkerHandle worker, Task<Message> reply)
        {
            if (_losers.Remove((task.Key, worker)))
            {
                // The other attempt already won; this one only held a thread. Drop its copy.
                worker.Ended(_client);
                if (reply.IsCompletedSuccessfully && reply.Result.Kind == MessageKind.Done && worker.Alive) _ = ReleaseAsync(worker, [task.Key]);
                return;
            }
            var original = _inFlight.TryGetValue(task.Key, out var assigned) && assigned == worker;
            var copy = _copies.TryGetValue(task.Key, out var copier) && copier == worker;
            if (!original && !copy) return; // superseded
            var other = original ? (_copies.TryGetValue(task.Key, out var c) ? c : null) : assigned;
            var succeeded = reply.IsCompletedSuccessfully && reply.Result.Kind == MessageKind.Done;
            if (other is not null && !succeeded && !reply.IsCanceled)
            {
                // One of two attempts failed while the other still runs: let the other decide.
                worker.Ended(_client);
                if (copy) _copies.Remove(task.Key);
                else
                {
                    _inFlight[task.Key] = other;
                    _copies.Remove(task.Key);
                }
                return;
            }
            _inFlight.Remove(task.Key);
            _copies.Remove(task.Key);
            worker.Ended(_client);
            if (other is not null && succeeded)
            {
                _losers.Add((task.Key, other));
                if (copy) Statistics.SpeculationWins++;
            }

            if (reply.IsCanceled) throw new OperationCanceledException(_cancel);
            if (reply.IsFaulted)
            {
                // The connection dropped; the loss handler replans.
                if (reply.Exception!.InnerException is WorkerLostException) { OnWorkerLost(worker); return; }
                throw reply.Exception.InnerException!;
            }

            var message = reply.Result;
            using var r = message.Reader();
            if (message.Kind == MessageKind.Done)
            {
                var key = WireCodec.ReadKey(r);
                var elapsed = TimeSpan.FromMilliseconds(r.ReadDouble());
                var fetched = r.ReadInt32();
                for (var i = 0; i < fetched; i++) Hold(WireCodec.ReadKey(r), worker);
                Hold(key, worker);
                if (!_durations.TryGetValue(task.Label, out var durations)) _durations[task.Label] = durations = [];
                durations.Add(elapsed.TotalMilliseconds);
                _completed++;
                DaskTelemetry.TasksCompleted.Add(1);
                DaskTelemetry.TaskDuration.Record(elapsed.TotalMilliseconds);
                _client.OnEvent?.Invoke(new TaskEvent(TaskEventKind.Finished, key, task.Label, _completed, _graph.Count, worker.Name, elapsed));
                OnAvailable(key);
                await ReleaseUnreadAsync(task).ConfigureAwait(false);
                return;
            }

            var type = r.ReadString();
            var text = r.ReadString();
            var missingDependency = r.ReadBoolean();
            if (missingDependency)
            {
                // A fetch failed: the holder is gone or about to be noticed as gone. Forget that
                // copy and work out again what has to run.
                var lost = WireCodec.ReadKey(r);
                if (_sources.TryGetValue(task.Key, out var told) && told.TryGetValue(lost, out var source)
                    && _holders.TryGetValue(lost, out var holders))
                {
                    holders.Remove(source);
                    if (holders.Count == 0) _holders.Remove(lost);
                }
                _client.OnEvent?.Invoke(new TaskEvent(TaskEventKind.Recomputing, lost, _graph.Contains(lost) ? _graph[lost].Label : "?", _completed, _graph.Count, worker.Name));
                Replan();
                return;
            }

            var error = new RemoteTaskException(type, text, worker.Address);
            if (_client.Retry.Allows(_attempts[task.Key], error))
            {
                Statistics.Retries++;
                DaskTelemetry.TasksRetried.Add(1);
                _client.OnEvent?.Invoke(new TaskEvent(TaskEventKind.Retrying, task.Key, task.Label, _completed, _graph.Count, worker.Name, Error: error));
                var delay = _client.Retry.DelayFor(_attempts[task.Key]);
                if (delay > TimeSpan.Zero) await Task.Delay(delay, _cancel).ConfigureAwait(false);
                _remaining[task.Key] = 0;
                _ready.Enqueue(task.Key, _priority[task.Key]);
                return;
            }

            DaskTelemetry.TasksFailed.Add(1);
            _client.OnEvent?.Invoke(new TaskEvent(TaskEventKind.Failed, task.Key, task.Label, _completed, _graph.Count, worker.Name, Error: error));
            throw new DaskTaskException(task.Key, task.Label, error);
        }

        private void Hold(TaskKey key, WorkerHandle worker)
        {
            if (!_holders.TryGetValue(key, out var set)) _holders[key] = set = [];
            set.Add(worker);
        }

        private void OnAvailable(TaskKey key)
        {
            if (!_dependents.TryGetValue(key, out var readers)) return;
            foreach (var t in readers.Distinct())
                if (_remaining.TryGetValue(t, out var n) && n > 0)
                {
                    _remaining[t] = n - 1;
                    if (n - 1 == 0) _ready.Enqueue(t, _priority[t]);
                }
        }

        /// <summary>Frees inputs that no remaining task reads.</summary>
        private async Task ReleaseUnreadAsync(TaskSpec finished)
        {
            foreach (var d in finished.Dependencies.Distinct())
            {
                if (!_readers.TryGetValue(d, out var n)) continue;
                _readers[d] = --n;
                if (n > 0 || _outputSet.Contains(d) || !_holders.Remove(d, out var holders)) continue;
                foreach (var h in holders.Where(h => h.Alive))
                    _ = ReleaseAsync(h, [d]);
            }
            await Task.CompletedTask.ConfigureAwait(false);
        }

        private void OnWorkerLost(WorkerHandle worker)
        {
            var orphaned = _inFlight.Where(p => p.Value == worker).Select(p => p.Key).ToArray();
            foreach (var key in orphaned)
            {
                worker.Ended(_client);
                // A straggler's second attempt elsewhere becomes the attempt.
                if (_copies.Remove(key, out var copy)) _inFlight[key] = copy;
                else _inFlight.Remove(key);
            }
            foreach (var key in _copies.Where(p => p.Value == worker).Select(p => p.Key).ToArray())
            {
                _copies.Remove(key);
                worker.Ended(_client);
            }
            _losers.RemoveWhere(l => l.Item2 == worker);
            Replan();   // reports the keys only this worker held
        }

        private async Task<object?> GatherAsync(TaskKey key)
        {
            while (true)
            {
                if (!Available(key))
                {
                    // Lost between finishing and being gathered (its holder may already be marked
                    // dead before its loss event runs): run what is needed again.
                    Replan();
                    while (!Available(key))
                    {
                        if (!_client._workers.Any(w => w.Alive))
                            throw new WorkerLostException("all", new InvalidOperationException("Every worker is gone."));
                        Dispatch();
                        var next = await _events.Reader.ReadAsync(_cancel).ConfigureAwait(false);
                        await next().ConfigureAwait(false);
                    }
                }
                var holder = _holders[key].First(h => h.Alive);
                try
                {
                    var shared = _client.SharedMemory && !holder.NoSharedMemory;
                    if (Views)
                    {
                        var (seen, view) = await Distributed.SharedMemory.GetViewAsync(holder.Rpc, key, shared, () => holder.NoSharedMemory = true, _cancel)
                            .ConfigureAwait(false);
                        if (seen) return view;
                        _holders[key].Remove(holder);
                        continue;
                    }
                    var (found, value) = await Distributed.SharedMemory.GetAsync(holder.Rpc, key, shared,
                        () => holder.NoSharedMemory = true, _cancel).ConfigureAwait(false);
                    if (found) return value;
                    _holders[key].Remove(holder);
                }
                catch (WorkerLostException)
                {
                    _holders[key].Remove(holder);
                }
            }
        }

        private async Task ReleaseAllAsync()
        {
            var byWorker = _holders.SelectMany(p => p.Value.Select(w => (w, p.Key))).GroupBy(p => p.w);
            foreach (var group in byWorker)
                if (group.Key.Alive)
                    await ReleaseAsync(group.Key, group.Select(p => p.Key).ToArray()).ConfigureAwait(false);
            _holders.Clear();
        }

        private static async Task ReleaseAsync(WorkerHandle worker, TaskKey[] keys)
        {
            try
            {
                await worker.Rpc.RequestAsync(MessageKind.Release, w =>
                {
                    w.Write(keys.Length);
                    foreach (var k in keys) WireCodec.WriteKey(w, k);
                }).ConfigureAwait(false);
            }
            catch (WorkerLostException) { }
        }
    }
}

/// <summary>A task's own code threw on a worker.</summary>
public sealed class RemoteTaskException(string type, string message, string worker)
    : Exception($"{type} on {worker}: {message}")
{
    /// <summary>The exception type's name on the worker.</summary>
    public string RemoteType { get; } = type;

    /// <summary>The worker that ran the task.</summary>
    public string Worker { get; } = worker;
}
