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
    /// <summary>A JSON form of the handle, for <see cref="Parse"/>.</summary>
    public override string ToString() => JsonSerializer.Serialize(new
    {
        id = Id,
        scheduler = Scheduler,
        outputs = Outputs.Select(k => new { name = k.Name, index = k.Index }),
    });

    /// <summary>Reads a handle written by <see cref="ToString"/>.</summary>
    public static RunHandle Parse(string text)
    {
        using var doc = JsonDocument.Parse(text);
        var root = doc.RootElement;
        return new RunHandle(root.GetProperty("id").GetString()!, root.GetProperty("scheduler").GetString()!,
            root.GetProperty("outputs").EnumerateArray()
                .Select(o => new TaskKey(o.GetProperty("name").GetString()!, o.GetProperty("index").GetInt32())).ToArray());
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
    /// a client does. Lose the scheduling worker and the run is lost - the single point of failure
    /// moves from the client into the cluster, it does not disappear. Results are held until
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
        var scheduler = live.OrderBy(w => w.InFlight).FirstOrDefault() ?? throw new InvalidOperationException("No worker is connected.");
        var id = Guid.NewGuid().ToString("N");
        var reply = await scheduler.Rpc.RequestAsync(MessageKind.Submit, w =>
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
        }, cancellationToken).ConfigureAwait(false);
        ThrowIfError(reply);
        return new RunHandle(id, scheduler.Address, outputs.ToArray());
    }

    /// <summary>Where a submitted run is.</summary>
    /// <exception cref="KeyNotFoundException">The scheduling worker knows no such run (it was forgotten, or the worker restarted).</exception>
    public async Task<RunStatus> StatusAsync(RunHandle run, CancellationToken cancellationToken = default)
    {
        var reply = await SchedulerOf(run).Rpc.RequestAsync(MessageKind.Status, w => w.Write(run.Id), cancellationToken).ConfigureAwait(false);
        ThrowIfError(reply);
        using var r = reply.Reader();
        var state = (RunState)r.ReadByte();
        var completed = r.ReadInt32();
        var total = r.ReadInt32();
        var error = r.ReadBoolean() ? r.ReadString() : null;
        return new RunStatus(state, completed, total, error);
    }

    /// <summary>Waits for a submitted run and returns its outputs, in order.</summary>
    /// <exception cref="DaskTaskException">The run failed; the message is the scheduling worker's.</exception>
    /// <exception cref="OperationCanceledException">The run was cancelled.</exception>
    public async Task<IReadOnlyList<object?>> ResultsAsync(RunHandle run, TimeSpan? pollEvery = null, CancellationToken cancellationToken = default)
    {
        var every = pollEvery ?? TimeSpan.FromMilliseconds(50);
        while (true)
        {
            var status = await StatusAsync(run, cancellationToken).ConfigureAwait(false);
            if (status.State == RunState.Finished) break;
            if (status.State == RunState.Failed)
                throw new DaskTaskException(new TaskKey(run.Id), "detached-run", new InvalidOperationException(status.Error));
            if (status.State == RunState.Cancelled) throw new OperationCanceledException($"Run {run.Id} was cancelled.");
            await Task.Delay(every, cancellationToken).ConfigureAwait(false);
        }
        var scheduler = SchedulerOf(run);
        var results = new object?[run.Outputs.Count];
        for (var i = 0; i < results.Length; i++)
        {
            var key = run.Outputs[i];
            var reply = await scheduler.Rpc.RequestAsync(MessageKind.Get, w => WireCodec.WriteKey(w, key), cancellationToken).ConfigureAwait(false);
            ThrowIfError(reply);
            using var r = reply.Reader();
            results[i] = WireCodec.Deserialize(r.ReadBytes(r.ReadInt32()));
        }
        return results;
    }

    /// <summary>Waits for a submitted <see cref="Delayed{T}"/> and returns its value.</summary>
    public async Task<T> ResultAsync<T>(RunHandle run, CancellationToken cancellationToken = default)
        => (T)(await ResultsAsync(run, cancellationToken: cancellationToken).ConfigureAwait(false))[0]!;

    /// <summary>Waits for a submitted frame and concatenates its partitions.</summary>
    public async Task<DataFrame> FrameResultAsync(RunHandle run, CancellationToken cancellationToken = default)
        => FrameKernels.ConcatFrames((await ResultsAsync(run, cancellationToken: cancellationToken).ConfigureAwait(false)).Cast<DataFrame>().ToArray());

    /// <summary>Stops a submitted run; tasks already running finish, nothing new starts.</summary>
    public async Task CancelAsync(RunHandle run, CancellationToken cancellationToken = default)
        => ThrowIfError(await SchedulerOf(run).Rpc.RequestAsync(MessageKind.Cancel, w => w.Write(run.Id), cancellationToken).ConfigureAwait(false));

    /// <summary>Drops a run's results and record from its scheduling worker (cancelling it if it is still running).</summary>
    public async Task ForgetAsync(RunHandle run, CancellationToken cancellationToken = default)
        => ThrowIfError(await SchedulerOf(run).Rpc.RequestAsync(MessageKind.Forget, w => w.Write(run.Id), cancellationToken).ConfigureAwait(false));

    private WorkerHandle SchedulerOf(RunHandle run)
        => _workers.FirstOrDefault(w => w.Alive && w.Address.Equals(run.Scheduler, StringComparison.OrdinalIgnoreCase))
           ?? throw new WorkerLostException(run.Scheduler, new InvalidOperationException(
               $"This client is not connected to {run.Scheduler}, the worker scheduling run {run.Id}."));

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
        run.Execution = Task.Run(() => ScheduleAsync(run, graph, addresses, attempts));
        await connection.SendAsync(MessageKind.Ok, message.Id).ConfigureAwait(false);
    }

    /// <summary>Runs a submitted graph as a client of the cluster, then keeps its outputs here.</summary>
    private async Task ScheduleAsync(DetachedRun run, TaskGraph graph, string[] addresses, int attempts)
    {
        try
        {
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(run.Cancel.Token, _stop.Token);
            await using var client = await Client.ConnectAsync(addresses, _options.Token, ClientTls(), linked.Token).ConfigureAwait(false);
            client.Retry = attempts > 0 ? new RetryPolicy(attempts) : RetryPolicy.None;
            client.OnEvent = e => { if (e.Kind == TaskEventKind.Finished) Interlocked.Increment(ref run.Completed); };
            var results = await client.ExecuteAsync(graph, run.Outputs, linked.Token).ConfigureAwait(false);
            foreach (var (key, value) in results) _data.Set(key, value);
            run.State = RunState.Finished;
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
