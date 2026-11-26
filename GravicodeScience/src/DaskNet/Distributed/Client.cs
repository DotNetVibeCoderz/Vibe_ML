using System.Diagnostics;
using System.Threading.Channels;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>What a worker reported about itself.</summary>
/// <param name="Address">Where it listens.</param>
/// <param name="Name">Its name.</param>
/// <param name="Threads">Tasks it runs at once.</param>
/// <param name="StoredKeys">Results it holds.</param>
/// <param name="TasksRun">Tasks it has run.</param>
/// <param name="ProcessId">Its operating-system process id.</param>
public sealed record WorkerInfo(string Address, string Name, int Threads, int StoredKeys, long TasksRun, int ProcessId);

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
public sealed class Client : IScheduler, IAsyncDisposable
{
    private readonly List<WorkerHandle> _workers = [];
    private readonly string _token;

    private Client(string token) => _token = token;

    /// <summary>Retries for tasks whose own code fails (not for lost workers, which are always recovered).</summary>
    public RetryPolicy Retry { get; set; } = RetryPolicy.None;

    /// <summary>Receives an event per task state change.</summary>
    public Action<TaskEvent>? OnEvent { get; set; }

    /// <summary>Statistics of the most recent run.</summary>
    public RunStatistics? LastRun { get; private set; }

    /// <inheritdoc />
    public string Name => "cluster";

    /// <summary>Addresses of the workers still connected.</summary>
    public IReadOnlyList<string> Workers => _workers.Where(w => !w.Rpc.IsClosed).Select(w => w.Address).ToArray();

    /// <summary>Connects to running workers.</summary>
    public static async Task<Client> ConnectAsync(IEnumerable<string> addresses, string token, CancellationToken cancellationToken = default)
    {
        var client = new Client(token);
        foreach (var address in addresses)
        {
            var rpc = await RpcClient.ConnectAsync(address, token, cancellationToken).ConfigureAwait(false);
            var info = await Ping(rpc, cancellationToken).ConfigureAwait(false);
            client._workers.Add(new WorkerHandle(address, rpc, info.Threads, info.Name));
        }
        if (client._workers.Count == 0) throw new ArgumentException("No worker addresses given.");
        return client;
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
        return new WorkerInfo(rpc.Address, r.ReadString(), r.ReadInt32(), r.ReadInt32(), r.ReadInt64(), r.ReadInt32());
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

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        foreach (var w in _workers) await w.Rpc.DisposeAsync().ConfigureAwait(false);
        _workers.Clear();
    }

    internal sealed class WorkerHandle(string address, RpcClient rpc, int threads, string name)
    {
        public string Address { get; } = address;
        public RpcClient Rpc { get; } = rpc;
        public int Threads { get; } = threads;
        public string Name { get; } = name;
        public int InFlight { get; set; }
        public bool Alive => !Rpc.IsClosed;
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

        public async Task<IReadOnlyDictionary<TaskKey, object?>> ExecuteAsync()
        {
            var clock = Stopwatch.StartNew();
            _graph.Validate();
            foreach (var w in _client._workers.Where(w => w.Alive))
            {
                var worker = w;
                _ = worker.Rpc.Closed.ContinueWith(_ => _events.Writer.TryWrite(() => { OnWorkerLost(worker); return Task.CompletedTask; }),
                    TaskScheduler.Default);
            }

            try
            {
                Replan();
                while (!AllOutputsAvailable())
                {
                    _cancel.ThrowIfCancellationRequested();
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
                await ReleaseAllAsync().ConfigureAwait(false);
                Statistics.Elapsed = clock.Elapsed;
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
            foreach (var (key, holders) in _holders.ToArray())
            {
                holders.RemoveWhere(w => !w.Alive);
                if (holders.Count == 0) _holders.Remove(key);
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
            worker.InFlight++;
            _inFlight[task.Key] = worker;
            _attempts[task.Key] = _attempts.GetValueOrDefault(task.Key) + 1;
            Statistics.TasksRun++;
            Statistics.TasksPerWorker[worker.Name] = Statistics.TasksPerWorker.GetValueOrDefault(worker.Name) + 1;
            _client.OnEvent?.Invoke(new TaskEvent(TaskEventKind.Started, task.Key, task.Label, _completed, _graph.Count, worker.Name));

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
            if (!_inFlight.TryGetValue(task.Key, out var assigned) || assigned != worker) return; // superseded
            _inFlight.Remove(task.Key);
            worker.InFlight--;

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
                _inFlight.Remove(key);
                worker.InFlight--;
            }
            var lostKeys = _holders.Where(p => p.Value.Contains(worker) && p.Value.All(h => h == worker || !h.Alive)).Select(p => p.Key).ToArray();
            foreach (var key in lostKeys)
                _client.OnEvent?.Invoke(new TaskEvent(TaskEventKind.Recomputing, key, _graph.Contains(key) ? _graph[key].Label : "?", _completed, _graph.Count, worker.Name));
            Replan();
        }

        private async Task<object?> GatherAsync(TaskKey key)
        {
            while (true)
            {
                var holder = _holders[key].First(h => h.Alive);
                try
                {
                    var reply = await holder.Rpc.RequestAsync(MessageKind.Get, w => WireCodec.WriteKey(w, key), _cancel).ConfigureAwait(false);
                    using var r = reply.Reader();
                    if (reply.Kind == MessageKind.Data) return WireCodec.Deserialize(r.ReadBytes(r.ReadInt32()));
                    _holders[key].Remove(holder);
                }
                catch (WorkerLostException)
                {
                    _holders[key].Remove(holder);
                }

                if (!Available(key))
                {
                    // Lost between finishing and being gathered: run what is needed again.
                    OnWorkerLost(holder);
                    while (!Available(key))
                    {
                        Dispatch();
                        var next = await _events.Reader.ReadAsync(_cancel).ConfigureAwait(false);
                        await next().ConfigureAwait(false);
                    }
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
