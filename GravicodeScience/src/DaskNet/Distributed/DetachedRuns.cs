using System.Collections.Concurrent;
using System.Text.Json;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>Where a detached run is.</summary>
public enum RunState
{
    /// <summary>Tasks are still running.</summary>
    Running,
    /// <summary>Every output is computed and held by the scheduling worker.</summary>
    Finished,
    /// <summary>A task failed beyond the retry policy, or no worker was left.</summary>
    Failed,
    /// <summary>Cancelled by a client.</summary>
    Cancelled,
}

/// <summary>Progress of a detached run.</summary>
/// <param name="State">Where it is.</param>
/// <param name="Completed">Tasks finished so far (recomputed tasks count again).</param>
/// <param name="Total">Tasks in the graph.</param>
/// <param name="Error">Why it failed, when it did.</param>
public sealed record RunStatus(RunState State, int Completed, int Total, string? Error);

/// <summary>
/// A run submitted with <see cref="Client.SubmitAsync(IDaskCollection, CancellationToken)"/>: which
/// worker schedules it and which results to ask for. It is plain data - <see cref="ToString"/>
/// and <see cref="Parse"/> carry it to another process, which can reattach to the run.
/// </summary>
/// <param name="Id">The run's id on its scheduling worker.</param>
/// <param name="Scheduler">The address of the worker that schedules the run and holds its results.</param>
/// <param name="Outputs">The result keys, in collection order.</param>
public sealed record RunHandle(string Id, string Scheduler, IReadOnlyList<TaskKey> Outputs)
{
    /// <summary>
    /// The workers that hold a copy of the run's graph, in order of succession: the first takes
    /// the run over if <see cref="Scheduler"/> dies, the second if both are gone, and so on.
    /// Empty when the cluster had only one worker. Set the count with
    /// <see cref="Client.RunStandbys"/>.
    /// </summary>
    public IReadOnlyList<string> Standbys { get; init; } = [];

    /// <summary>The first of <see cref="Standbys"/>, or null.</summary>
    public string? Standby
    {
        get => Standbys.Count > 0 ? Standbys[0] : null;
        init => Standbys = value is null ? [] : [value];
    }

    /// <summary>A JSON form of the handle, for <see cref="Parse"/>.</summary>
    public override string ToString() => JsonSerializer.Serialize(new
    {
        id = Id,
        scheduler = Scheduler,
        standby = Standby,                              // what 2.4 readers look for
        standbys = Standbys,
        outputs = Outputs.Select(k => new { name = k.Name, index = k.Index }),
    });

    /// <summary>Reads a handle written by <see cref="ToString"/>.</summary>
    public static RunHandle Parse(string text)
    {
        using var doc = JsonDocument.Parse(text);
        var root = doc.RootElement;
        return new RunHandle(root.GetProperty("id").GetString()!, root.GetProperty("scheduler").GetString()!,
            root.GetProperty("outputs").EnumerateArray()
                .Select(o => new TaskKey(o.GetProperty("name").GetString()!, o.GetProperty("index").GetInt32())).ToArray())
        {
            Standbys = root.TryGetProperty("standbys", out var list) && list.ValueKind == JsonValueKind.Array
                ? list.EnumerateArray().Select(s => s.GetString()!).ToArray()
                : root.TryGetProperty("standby", out var standby) && standby.ValueKind == JsonValueKind.String ? [standby.GetString()!] : [],
        };
    }
}

public sealed partial class Client
{
    /// <summary>
    /// Hands a whole graph to one worker, which schedules it on the cluster from then on (Dask's
    /// <c>client.compute(..., fire_and_forget)</c> with the scheduler inside the cluster). This
    /// client may disconnect or die; the run goes on, its results stay on the scheduling worker,
    /// and any client connected to that worker can follow it with <see cref="StatusAsync"/> and
    /// collect it with <see cref="ResultsAsync"/>.
    /// </summary>
    /// <remarks>
    /// The scheduling worker is the connected live worker with the fewest tasks in flight. It
    /// connects to every worker this client knows, itself included, and recovers lost workers as
    /// a client does. <see cref="RunStandbys"/> more workers, the <see cref="RunHandle.Standbys"/>,
    /// keep a copy of the graph and watch their predecessors in the succession. When the run
    /// finishes, its outputs are copied to every standby before it reports
    /// <see cref="RunState.Finished"/>, so a scheduler that dies afterwards costs nothing: the
    /// next standby serves the copies. One that dies mid-run is replaced by the next live standby,
    /// which runs the graph again from its inputs. Every call here falls back along the
    /// succession; losing every holder loses the run. Results are held until
    /// <see cref="ForgetAsync"/>.
    /// </remarks>
    public async Task<RunHandle> SubmitAsync(IDaskCollection collection, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(collection);
        return await SubmitAsync(TaskGraph.FromCollections(collection), collection.OutputKeys, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Submits a frame's partitions; collect them with <see cref="FrameResultAsync"/>.</summary>
    public Task<RunHandle> SubmitAsync(DaskFrame frame, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frame);
        var parts = frame.Lower(this);
        return SubmitAsync(TaskGraph.FromLayers([parts.Layer]), parts.Keys, cancellationToken);
    }

    /// <summary>Submits <paramref name="outputs"/> of <paramref name="graph"/>.</summary>
    public async Task<RunHandle> SubmitAsync(TaskGraph graph, IReadOnlyList<TaskKey> outputs, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var culled = graph.Cull(outputs);
        var local = culled.Tasks.Values.Where(t => !t.IsRemotable).Select(t => t.Label).Distinct().Take(5).ToArray();
        if (local.Length > 0)
            throw new NotSupportedException($"Tasks [{string.Join(", ", local)}] wrap local delegates and cannot be submitted to a cluster.");

        var live = _workers.Where(w => w.Alive).ToArray();
        var ranked = live.OrderBy(w => w.InFlight).ToArray();
        var scheduler = ranked.FirstOrDefault() ?? throw new InvalidOperationException("No worker is connected.");
        var id = Guid.NewGuid().ToString("N");
        void Body(BinaryWriter w, string[]? predecessors)
        {
            w.Write(id);
            w.Write(live.Length);
            foreach (var h in live) w.Write(h.Address);
            w.Write(Retry.MaxRetries);
            w.Write(culled.Tasks.Count);
            foreach (var task in culled.Tasks.Values)
            {
                WireCodec.WriteKey(w, task.Key);
                w.Write(task.Label);
                WireCodec.Write(w, task.Remote);
            }
            w.Write(outputs.Count);
            foreach (var k in outputs) WireCodec.WriteKey(w, k);
            if (predecessors is null) return;
            w.Write(predecessors.Length);
            foreach (var p in predecessors) w.Write(p);
        }

        var reply = await scheduler.Rpc.RequestAsync(MessageKind.Submit, w => Body(w, null), cancellationToken).ConfigureAwait(false);
        ThrowIfError(reply);
        var succession = new List<string> { scheduler.Address };
        var holders = new List<WorkerHandle> { scheduler };
        foreach (var candidate in ranked.Skip(1))
        {
            if (holders.Count > RunStandbys) break;
            var predecessors = succession.ToArray();
            try
            {
                var standby = await candidate.Rpc.RequestAsync(MessageKind.Standby, w => Body(w, predecessors), cancellationToken).ConfigureAwait(false);
                ThrowIfError(standby);
                succession.Add(candidate.Address);
                holders.Add(candidate);
            }
            catch (WorkerLostException)
            {
                // try the next one; the run itself is already scheduled
            }
        }
        if (holders.Count > 1)
        {
            // Every holder learns the whole succession, so whichever finishes the run knows whom
            // to copy its outputs to. A run that already finished copies them before replying.
            var all = succession.ToArray();
            foreach (var holder in holders)
                try
                {
                    ThrowIfError(await holder.Rpc.RequestAsync(MessageKind.Replicas, w =>
                    {
                        w.Write(id);
                        w.Write(all.Length);
                        foreach (var a in all) w.Write(a);
                    }, cancellationToken).ConfigureAwait(false));
                }
                catch (WorkerLostException)
                {
                    // its successors will notice
                }
        }
        return new RunHandle(id, scheduler.Address, outputs.ToArray()) { Standbys = succession.Skip(1).ToArray() };
    }

    /// <summary>
    /// How many standbys <see cref="SubmitAsync(TaskGraph, IReadOnlyList{TaskKey}, CancellationToken)"/>
    /// asks for (default 1; fewer when the cluster is smaller). Each one survives one more lost
    /// holder, and each receives a copy of the finished outputs.
    /// </summary>
    public int RunStandbys
    {
        get => _runStandbys;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            _runStandbys = value;
        }
    }

    private int _runStandbys = 1;

    /// <summary>Where a submitted run is.</summary>
    /// <exception cref="KeyNotFoundException">The scheduling worker knows no such run (it was forgotten, or the worker restarted).</exception>
    public async Task<RunStatus> StatusAsync(RunHandle run, CancellationToken cancellationToken = default)
        => (await StatusFromAsync(run, cancellationToken).ConfigureAwait(false)).Status;

    /// <summary>The run's status and the worker that gave it: the scheduler while it is reachable
    /// and knows the run, the standby otherwise.</summary>
    private async Task<(RunStatus Status, WorkerHandle Holder)> StatusFromAsync(RunHandle run, CancellationToken cancellationToken)
    {
        Exception? last = null;
        foreach (var holder in HoldersOf(run))
        {
            try
            {
                var reply = await holder.Rpc.RequestAsync(MessageKind.Status, w => w.Write(run.Id), cancellationToken).ConfigureAwait(false);
                ThrowIfError(reply);
                using var r = reply.Reader();
                var state = (RunState)r.ReadByte();
                var completed = r.ReadInt32();
                var total = r.ReadInt32();
                var error = r.ReadBoolean() ? r.ReadString() : null;
                return (new RunStatus(state, completed, total, error), holder);
            }
            catch (Exception e) when (e is WorkerLostException or KeyNotFoundException)
            {
                last = e;     // a dead scheduler, or one restarted without the run: ask the standby
            }
        }
        throw last ?? NotConnected(run);
    }

    /// <summary>Waits for a submitted run and returns its outputs, in order.</summary>
    /// <exception cref="DaskTaskException">The run failed; the message is the scheduling worker's.</exception>
    /// <exception cref="OperationCanceledException">The run was cancelled.</exception>
    public async Task<IReadOnlyList<object?>> ResultsAsync(RunHandle run, TimeSpan? pollEvery = null, CancellationToken cancellationToken = default)
    {
        var every = pollEvery ?? TimeSpan.FromMilliseconds(50);
        while (true)
        {
            var (status, holder) = await StatusFromAsync(run, cancellationToken).ConfigureAwait(false);
            if (status.State == RunState.Failed)
                throw new DaskTaskException(new TaskKey(run.Id), "detached-run", new InvalidOperationException(status.Error));
            if (status.State == RunState.Cancelled) throw new OperationCanceledException($"Run {run.Id} was cancelled.");
            if (status.State == RunState.Finished)
            {
                try
                {
                    var results = new object?[run.Outputs.Count];
                    for (var i = 0; i < results.Length; i++)
                    {
                        var key = run.Outputs[i];
                        var reply = await holder.Rpc.RequestAsync(MessageKind.Get, w => WireCodec.WriteKey(w, key), cancellationToken).ConfigureAwait(false);
                        ThrowIfError(reply);
                        using var r = reply.Reader();
                        results[i] = WireCodec.Deserialize(r.ReadBytes(r.ReadInt32()));
                    }
                    return results;
                }
                catch (WorkerLostException)
                {
                    // the holder died between reporting and sending: ask along the succession
                    // again (which throws once nobody is left)
                }
            }
            await Task.Delay(every, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Waits for a submitted <see cref="Delayed{T}"/> and returns its value.</summary>
    public async Task<T> ResultAsync<T>(RunHandle run, CancellationToken cancellationToken = default)
        => (T)(await ResultsAsync(run, cancellationToken: cancellationToken).ConfigureAwait(false))[0]!;

    /// <summary>Waits for a submitted frame and concatenates its partitions.</summary>
    public async Task<DataFrame> FrameResultAsync(RunHandle run, CancellationToken cancellationToken = default)
        => FrameKernels.ConcatFrames((await ResultsAsync(run, cancellationToken: cancellationToken).ConfigureAwait(false)).Cast<DataFrame>().ToArray());

    /// <summary>Stops a submitted run; tasks already running finish, nothing new starts.</summary>
    public Task CancelAsync(RunHandle run, CancellationToken cancellationToken = default)
        => TellBothAsync(run, MessageKind.Cancel, cancellationToken);

    /// <summary>Drops a run's results and record from its scheduling worker and its standby
    /// (cancelling it if it is still running).</summary>
    public Task ForgetAsync(RunHandle run, CancellationToken cancellationToken = default)
        => TellBothAsync(run, MessageKind.Forget, cancellationToken);

    /// <summary>
    /// Sends a cancel or forget along the succession backwards - last standby first, scheduler
    /// last: told the other way round, a standby would see its predecessor forget the run, take
    /// it for a restart and run the graph again. Succeeds if any holder knew the run.
    /// </summary>
    private async Task TellBothAsync(RunHandle run, MessageKind kind, CancellationToken cancellationToken)
    {
        Exception? last = null;
        var told = false;
        foreach (var holder in Enumerable.Reverse(HoldersOf(run)))
        {
            try
            {
                ThrowIfError(await holder.Rpc.RequestAsync(kind, w => w.Write(run.Id), cancellationToken).ConfigureAwait(false));
                told = true;
            }
            catch (Exception e) when (e is WorkerLostException or KeyNotFoundException)
            {
                last = e;
            }
        }
        if (!told) throw last ?? NotConnected(run);
    }

    /// <summary>The connected, live workers that may know the run: scheduler first, then the standbys in order.</summary>
    private List<WorkerHandle> HoldersOf(RunHandle run)
    {
        var holders = new List<WorkerHandle>(1 + run.Standbys.Count);
        foreach (var address in run.Standbys.Prepend(run.Scheduler))
            if (_workers.FirstOrDefault(w => w.Alive && w.Address.Equals(address, StringComparison.OrdinalIgnoreCase)) is { } h)
                holders.Add(h);
        return holders;
    }

    private static WorkerLostException NotConnected(RunHandle run)
        => new(run.Scheduler, new InvalidOperationException(
            $"This client is not connected to {run.Scheduler} (the worker scheduling run {run.Id})"
            + (run.Standbys.Count == 0 ? "." : $" or to any of its standbys ({string.Join(", ", run.Standbys)}).")));

    private static void ThrowIfError(Message reply)
    {
        if (reply.Kind != MessageKind.Error) return;
        using var r = reply.Reader();
        var type = r.ReadString();
        var text = r.ReadString();
        throw type == nameof(KeyNotFoundException) ? new KeyNotFoundException(text) : new InvalidOperationException($"{type}: {text}");
    }
}

/// <summary>A run a worker schedules for a client (see <see cref="Client.SubmitAsync(IDaskCollection, CancellationToken)"/>).</summary>
internal sealed class DetachedRun
{
    public required TaskKey[] Outputs { get; init; }
    public required int Total { get; init; }
    /// <summary>The whole succession (scheduler, then standbys), once the client sends it.</summary>
    public string[] Succession = [];
    /// <summary>Set under the run's lock when this holder has the outputs it must copy onwards.</summary>
    public bool ResultsReady;
    /// <summary>A standby that received every output from the holder that finished the run.</summary>
    public volatile bool Replicated;
    public CancellationTokenSource Cancel { get; } = new();
    public volatile int Completed;
    public volatile RunState State = RunState.Running;
    public string? Error;
    public Task? Execution;
}

public sealed partial class Worker
{
    private readonly ConcurrentDictionary<string, DetachedRun> _runs = new(StringComparer.Ordinal);

    private async Task SubmitAsync(WireConnection connection, Message message)
    {
        string id;
        string[] addresses;
        int attempts;
        TaskSpec[] tasks;
        TaskKey[] outputs;
        string[]? predecessors;
        try
        {
            using var r = message.Reader();
            id = r.ReadString();
            addresses = Enumerable.Range(0, r.ReadInt32()).Select(_ => r.ReadString()).ToArray();
            attempts = r.ReadInt32();
            tasks = Enumerable.Range(0, r.ReadInt32()).Select(_ =>
            {
                var key = WireCodec.ReadKey(r);
                var label = r.ReadString();
                return TaskSpec.FromRemote(key, (RemoteCall)WireCodec.Read(r)!, label);
            }).ToArray();
            outputs = Enumerable.Range(0, r.ReadInt32()).Select(_ => WireCodec.ReadKey(r)).ToArray();
            predecessors = message.Kind == MessageKind.Standby ? Enumerable.Range(0, r.ReadInt32()).Select(_ => r.ReadString()).ToArray() : null;
        }
        catch (Exception e)
        {
            await SendErrorAsync(connection, message.Id, e, null).ConfigureAwait(false);
            return;
        }

        var run = new DetachedRun { Outputs = outputs, Total = tasks.Length };
        if (!_runs.TryAdd(id, run))
        {
            await SendErrorAsync(connection, message.Id, new InvalidOperationException($"Run {id} already exists."), null).ConfigureAwait(false);
            return;
        }
        var graph = new TaskGraph(tasks);
        if (!addresses.Contains(Address, StringComparer.OrdinalIgnoreCase)) addresses = [Address, .. addresses];
        run.Execution = predecessors is null
            ? Task.Run(() => ScheduleAsync(run, id, graph, addresses, attempts))
            : Task.Run(() => StandByAsync(run, id, predecessors, graph, addresses, attempts));
        await connection.SendAsync(MessageKind.Ok, message.Id).ConfigureAwait(false);
    }

    /// <summary>Runs a submitted graph as a client of the cluster, keeps its outputs here and
    /// copies them to the holders after this one before reporting it finished.</summary>
    private async Task ScheduleAsync(DetachedRun run, string id, TaskGraph graph, string[] addresses, int attempts)
    {
        try
        {
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(run.Cancel.Token, _stop.Token);
            await using (var client = await ConnectReachableAsync(addresses, linked.Token).ConfigureAwait(false))
            {
                client.Retry = attempts > 0 ? new RetryPolicy(attempts) : RetryPolicy.None;
                client.OnEvent = e => { if (e.Kind == TaskEventKind.Finished) Interlocked.Increment(ref run.Completed); };
                var results = await client.ExecuteAsync(graph, run.Outputs, linked.Token).ConfigureAwait(false);
                foreach (var (key, value) in results) _data.Set(key, value);
            }
            await FinishAsync(run, id).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (run.Cancel.IsCancellationRequested)
        {
            run.State = RunState.Cancelled;
        }
        catch (Exception e)
        {
            run.Error = e is DaskTaskException d ? d.Message : $"{e.GetType().Name}: {e.Message}";
            run.State = RunState.Failed;
        }
    }

    /// <summary>How often a standby asks its scheduler for the run, and how many failed asks in a
    /// row it takes as the scheduler's death.</summary>
    internal static TimeSpan StandbyPoll { get; set; } = TimeSpan.FromMilliseconds(200);
    internal const int StandbyStrikes = 3;

    /// <summary>Marks the run finished once its outputs are on every later holder. The flag is
    /// set under the run's lock so a succession arriving at the same moment copies them instead
    /// (<see cref="ReplicasAsync"/>); copying twice is harmless.</summary>
    private async Task FinishAsync(DetachedRun run, string id)
    {
        string[] targets;
        lock (run)
        {
            run.ResultsReady = true;
            targets = Successors(run);
        }
        await ReplicateAsync(run, id, targets).ConfigureAwait(false);
        run.Completed = Math.Max(run.Completed, run.Total);
        run.State = RunState.Finished;
    }

    /// <summary>The holders after this worker in the run's succession.</summary>
    private string[] Successors(DetachedRun run)
    {
        var at = Array.FindIndex(run.Succession, a => a.Equals(Address, StringComparison.OrdinalIgnoreCase));
        return at < 0 ? [] : run.Succession[(at + 1)..];
    }

    /// <summary>Copies the run's outputs to <paramref name="targets"/>; a holder that cannot be
    /// reached is skipped - it is dead, and its own successors have the copy.</summary>
    private async Task ReplicateAsync(DetachedRun run, string id, string[] targets)
    {
        if (targets.Length == 0) return;
        var payload = new List<(TaskKey Key, byte[] Bytes)>(run.Outputs.Length);
        foreach (var key in run.Outputs)
        {
            if (!_data.TryGetSerialized(key, out var bytes)) return;        // forgotten meanwhile
            payload.Add((key, bytes));
        }
        foreach (var target in targets)
        {
            try
            {
                await using var rpc = await RpcClient.ConnectAsync(target, _options.Token, ClientTls(), _stop.Token).ConfigureAwait(false);
                await rpc.RequestAsync(MessageKind.Replica, w =>
                {
                    w.Write(id);
                    w.Write(payload.Count);
                    foreach (var (key, bytes) in payload)
                    {
                        WireCodec.WriteKey(w, key);
                        w.Write(bytes.Length);
                        w.Write(bytes);
                    }
                }, _stop.Token).ConfigureAwait(false);
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                // unreachable: the next one still gets its copy
            }
        }
    }

    /// <summary>The client telling a holder the whole succession; a run already finished here
    /// copies its outputs onwards before replying.</summary>
    private async Task ReplicasAsync(WireConnection connection, Message message)
    {
        using var r = message.Reader();
        var id = r.ReadString();
        var succession = Enumerable.Range(0, r.ReadInt32()).Select(_ => r.ReadString()).ToArray();
        if (!_runs.TryGetValue(id, out var run))
        {
            await SendErrorAsync(connection, message.Id, new KeyNotFoundException($"{Name} knows no run {id}."), null).ConfigureAwait(false);
            return;
        }
        string[]? targets = null;
        lock (run)
        {
            run.Succession = succession;
            if (run.ResultsReady) targets = Successors(run);
        }
        if (targets is not null) await ReplicateAsync(run, id, targets).ConfigureAwait(false);
        await connection.SendAsync(MessageKind.Ok, message.Id).ConfigureAwait(false);
    }

    /// <summary>A finished run's outputs, copied here by the holder that finished it.</summary>
    private async Task ReplicaAsync(WireConnection connection, Message message)
    {
        using var r = message.Reader();
        var id = r.ReadString();
        if (!_runs.TryGetValue(id, out var run))
        {
            await SendErrorAsync(connection, message.Id, new KeyNotFoundException($"{Name} knows no run {id}."), null).ConfigureAwait(false);
            return;
        }
        var count = r.ReadInt32();
        for (var i = 0; i < count; i++)
        {
            var key = WireCodec.ReadKey(r);
            _data.Set(key, WireCodec.Deserialize(r.ReadBytes(r.ReadInt32())));
        }
        run.Replicated = run.Outputs.All(_data.Contains);
        await connection.SendAsync(MessageKind.Ok, message.Id).ConfigureAwait(false);
    }

    /// <summary>
    /// Holds a run's graph and watches the holders before this one in the succession, asking the
    /// first that has not been given up for. While one answers, this reports its progress; once
    /// it stops answering (or says it does not know the run - it restarted) the next is watched,
    /// and when every predecessor is gone this worker takes the run over. If the run had
    /// finished and its outputs were copied here, that is all a takeover is; otherwise the graph
    /// runs again, on every worker but the dead ones.
    /// </summary>
    private async Task StandByAsync(DetachedRun run, string id, string[] predecessors, TaskGraph graph, string[] addresses, int attempts)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(run.Cancel.Token, _stop.Token);
        var dead = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        RpcClient? rpc = null;
        string? watched = null;
        var strikes = 0;
        try
        {
            while (predecessors.FirstOrDefault(p => !dead.Contains(p)) is { } target)
            {
                if (!target.Equals(watched, StringComparison.OrdinalIgnoreCase))
                {
                    if (rpc is not null) await rpc.DisposeAsync().ConfigureAwait(false);
                    rpc = null;
                    watched = target;
                    strikes = 0;
                }
                try
                {
                    rpc ??= await RpcClient.ConnectAsync(target, _options.Token, ClientTls(), linked.Token).ConfigureAwait(false);
                    var reply = await rpc.RequestAsync(MessageKind.Status, w => w.Write(id), linked.Token).ConfigureAwait(false);
                    if (reply.Kind == MessageKind.Error)
                    {
                        using var e = reply.Reader();
                        if (e.ReadString() == nameof(KeyNotFoundException)) strikes = StandbyStrikes;   // restarted without the run
                        else strikes++;
                    }
                    else
                    {
                        using var r = reply.Reader();
                        var state = (RunState)r.ReadByte();
                        run.Completed = r.ReadInt32();
                        r.ReadInt32();
                        var error = r.ReadBoolean() ? r.ReadString() : null;
                        strikes = 0;
                        if (state is RunState.Failed or RunState.Cancelled)
                        {
                            run.Error = error;
                            run.State = state;          // nothing to take over
                            return;
                        }
                    }
                }
                catch (OperationCanceledException) when (linked.IsCancellationRequested)
                {
                    throw;
                }
                catch (Exception)
                {
                    strikes++;
                    if (rpc is not null) await rpc.DisposeAsync().ConfigureAwait(false);
                    rpc = null;
                }
                if (strikes >= StandbyStrikes)
                {
                    dead.Add(target);                   // ask the next predecessor at once
                    continue;
                }
                await Task.Delay(StandbyPoll, linked.Token).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException) when (run.Cancel.IsCancellationRequested)
        {
            run.State = RunState.Cancelled;
            return;
        }
        catch (OperationCanceledException)
        {
            return;                                     // this worker is shutting down
        }
        finally
        {
            if (rpc is not null) await rpc.DisposeAsync().ConfigureAwait(false);
        }

        Interlocked.Increment(ref _takeovers);
        if (run.Replicated && run.Outputs.All(_data.Contains))
        {
            Interlocked.Increment(ref _replicaTakeovers);
            await FinishAsync(run, id).ConfigureAwait(false);     // pass the copy on down the succession
            return;
        }
        run.Completed = 0;
        await ScheduleAsync(run, id, graph, addresses.Where(a => !dead.Contains(a)).ToArray(), attempts).ConfigureAwait(false);
    }

    private int _takeovers;
    private int _replicaTakeovers;

    /// <summary>Runs this worker took over from a dead scheduler, as a standby.</summary>
    public int Takeovers => _takeovers;

    /// <summary>The part of <see cref="Takeovers"/> served from copied outputs, with nothing recomputed.</summary>
    public int TakeoversFromReplica => _replicaTakeovers;

    /// <summary>A client of every address that answers; a dead worker in the list is skipped
    /// rather than failing the run, while a wrong token or TLS failure still throws.</summary>
    private async Task<Client> ConnectReachableAsync(string[] addresses, CancellationToken cancellationToken)
    {
        var reachable = new List<string>();
        foreach (var address in addresses)
        {
            try
            {
                await using var probe = await RpcClient.ConnectAsync(address, _options.Token, ClientTls(), cancellationToken).ConfigureAwait(false);
                reachable.Add(address);
            }
            catch (Exception e) when (e is System.Net.Sockets.SocketException or IOException { InnerException: System.Net.Sockets.SocketException })
            {
                // gone; the run goes on without it
            }
        }
        if (reachable.Count == 0) throw new InvalidOperationException("No worker of the run could be reached.");
        return await Client.ConnectAsync(reachable, _options.Token, ClientTls(), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>The client side of this worker's TLS settings: it presents its own certificate and
    /// trusts what it trusts as a server.</summary>
    private TlsOptions? ClientTls() => _options.Tls;

    private async Task StatusAsync(WireConnection connection, Message message)
    {
        using var r = message.Reader();
        var id = r.ReadString();
        if (!_runs.TryGetValue(id, out var run))
        {
            await SendErrorAsync(connection, message.Id, new KeyNotFoundException($"{Name} knows no run {id}."), null).ConfigureAwait(false);
            return;
        }
        await connection.SendAsync(MessageKind.RunStatus, message.Id, w =>
        {
            w.Write((byte)run.State);
            w.Write(run.Completed);
            w.Write(run.Total);
            w.Write(run.Error is not null);
            if (run.Error is { } e) w.Write(e);
        }).ConfigureAwait(false);
    }

    private async Task CancelOrForgetAsync(WireConnection connection, Message message, bool forget)
    {
        using var r = message.Reader();
        var id = r.ReadString();
        if (!_runs.TryGetValue(id, out var run))
        {
            await SendErrorAsync(connection, message.Id, new KeyNotFoundException($"{Name} knows no run {id}."), null).ConfigureAwait(false);
            return;
        }
        await run.Cancel.CancelAsync().ConfigureAwait(false);
        if (forget)
        {
            if (run.Execution is { } execution)
                try { await execution.ConfigureAwait(false); } catch { /* reported through State */ }
            foreach (var key in run.Outputs) _data.Remove(key);
            _runs.TryRemove(id, out _);
        }
        await connection.SendAsync(MessageKind.Ok, message.Id).ConfigureAwait(false);
    }
}
