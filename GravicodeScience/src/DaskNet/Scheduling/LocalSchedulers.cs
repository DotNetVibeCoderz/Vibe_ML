using System.Diagnostics;
using System.Threading.Channels;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// Runs tasks on the thread pool, at most <see cref="MaxWorkers"/> at a time. Dask's
/// <c>scheduler="threads"</c>, and the default.
/// </summary>
/// <remarks>
/// <para>
/// Threads share memory, so results pass between tasks by reference with no serialisation - the
/// right choice whenever the work is in managed code that releases no lock, which in .NET (unlike
/// CPython) is all of it. Use a cluster for isolation or for more than one machine, not for speed.
/// </para>
/// <para>
/// Ready tasks run in depth-first priority order (see <see cref="TaskGraph.Priorities"/>) and an
/// intermediate result is dropped the moment its last reader finishes, so a reduction over a
/// dataset larger than memory holds only a few partitions at once.
/// </para>
/// </remarks>
public sealed class ThreadScheduler(int? maxWorkers = null, RetryPolicy? retry = null) : IScheduler
{
    /// <summary>How many tasks may run at once.</summary>
    public int MaxWorkers { get; } = maxWorkers is > 0 ? maxWorkers.Value
        : maxWorkers is null ? Environment.ProcessorCount
        : throw new ArgumentOutOfRangeException(nameof(maxWorkers));

    /// <summary>Retries for failing tasks.</summary>
    public RetryPolicy Retry { get; } = retry ?? RetryPolicy.None;

    /// <summary>Receives an event per task state change; may be called from several threads.</summary>
    public Action<TaskEvent>? OnEvent { get; set; }

    /// <summary>Statistics of the most recent run.</summary>
    public RunStatistics? LastRun { get; private set; }

    /// <inheritdoc />
    public string Name => "threads";

    /// <inheritdoc />
    public async Task<IReadOnlyDictionary<TaskKey, object?>> ExecuteAsync(TaskGraph graph, IReadOnlyList<TaskKey> outputs,
        CancellationToken cancellationToken = default)
    {
        var engine = new LocalEngine(graph, outputs, MaxWorkers, inline: false, Retry, OnEvent);
        try { return await engine.RunAsync(cancellationToken).ConfigureAwait(false); }
        finally { LastRun = engine.Statistics; }
    }
}

/// <summary>
/// Runs every task on the calling thread, one at a time, in priority order. Dask's
/// <c>scheduler="sync"</c>: for debugging, where a stack trace should lead straight to the task
/// and a breakpoint should not be hit by eight threads at once.
/// </summary>
public sealed class SynchronousScheduler(RetryPolicy? retry = null) : IScheduler
{
    /// <summary>Retries for failing tasks.</summary>
    public RetryPolicy Retry { get; } = retry ?? RetryPolicy.None;

    /// <summary>Receives an event per task state change.</summary>
    public Action<TaskEvent>? OnEvent { get; set; }

    /// <summary>Statistics of the most recent run.</summary>
    public RunStatistics? LastRun { get; private set; }

    /// <inheritdoc />
    public string Name => "sync";

    /// <inheritdoc />
    public async Task<IReadOnlyDictionary<TaskKey, object?>> ExecuteAsync(TaskGraph graph, IReadOnlyList<TaskKey> outputs,
        CancellationToken cancellationToken = default)
    {
        var engine = new LocalEngine(graph, outputs, 1, inline: true, Retry, OnEvent);
        try { return await engine.RunAsync(cancellationToken).ConfigureAwait(false); }
        finally { LastRun = engine.Statistics; }
    }
}

/// <summary>The dependency-counting loop both local schedulers share.</summary>
internal sealed class LocalEngine
{
    private readonly TaskGraph _graph;
    private readonly IReadOnlyList<TaskKey> _outputs;
    private readonly int _maxWorkers;
    private readonly bool _inline;
    private readonly RetryPolicy _retry;
    private readonly Action<TaskEvent>? _onEvent;
    private readonly Dictionary<TaskKey, object?> _results = [];
    private readonly Lock _gate = new();
    private int _completed;

    public LocalEngine(TaskGraph graph, IReadOnlyList<TaskKey> outputs, int maxWorkers, bool inline,
        RetryPolicy retry, Action<TaskEvent>? onEvent)
    {
        _graph = graph;
        _outputs = outputs.Distinct().ToArray();
        _maxWorkers = maxWorkers;
        _inline = inline;
        _retry = retry;
        _onEvent = onEvent;
    }

    public RunStatistics Statistics { get; } = new();

    public async Task<IReadOnlyDictionary<TaskKey, object?>> RunAsync(CancellationToken cancellationToken)
    {
        var clock = Stopwatch.StartNew();
        var graph = _graph.Cull(_outputs);
        graph.Validate();
        Statistics.TasksTotal = graph.Count;

        var dependents = graph.Dependents();
        var remaining = graph.Tasks.Values.ToDictionary(t => t.Key, t => t.Dependencies.Distinct().Count());
        var readers = dependents.ToDictionary(p => p.Key, p => p.Value.Distinct().Count());
        var outputs = _outputs.ToHashSet();
        var priority = graph.Priorities(_outputs);
        var ready = new PriorityQueue<TaskKey, int>();
        foreach (var (key, count) in remaining) if (count == 0) ready.Enqueue(key, priority[key]);

        var completions = Channel.CreateUnbounded<(TaskKey Key, object? Value, Exception? Error)>();
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var running = 0;
        Exception? failure = null;

        while (_completed < graph.Count)
        {
            while (failure is null && running < _maxWorkers && ready.TryDequeue(out var key, out _))
            {
                var task = graph[key];
                var arguments = Arguments(task);
                running++;
                if (_inline)
                {
                    var outcome = await RunWithRetriesAsync(task, arguments, graph.Count, linked.Token).ConfigureAwait(false);
                    completions.Writer.TryWrite((key, outcome.Value, outcome.Error));
                }
                else
                {
                    _ = Task.Run(async () =>
                    {
                        var outcome = await RunWithRetriesAsync(task, arguments, graph.Count, linked.Token).ConfigureAwait(false);
                        completions.Writer.TryWrite((key, outcome.Value, outcome.Error));
                    }, CancellationToken.None);
                }
            }

            if (running == 0) break;

            var (done, value, error) = await completions.Reader.ReadAsync(CancellationToken.None).ConfigureAwait(false);
            running--;

            if (error is not null)
            {
                failure ??= error is OperationCanceledException && cancellationToken.IsCancellationRequested
                    ? error
                    : new DaskTaskException(done, graph[done].Label, error);
                linked.Cancel();
                continue;
            }
            if (failure is not null) continue;

            lock (_gate)
            {
                _results[done] = value;
                _completed++;
                if (_results.Count > Statistics.PeakResultsHeld) Statistics.PeakResultsHeld = _results.Count;

                // Release inputs nobody else will read.
                foreach (var d in graph[done].Dependencies.Distinct())
                    if (--readers[d] == 0 && !outputs.Contains(d)) _results.Remove(d);
            }

            foreach (var next in dependents[done].Distinct())
                if (--remaining[next] == 0) ready.Enqueue(next, priority[next]);
        }

        Statistics.Elapsed = clock.Elapsed;
        if (failure is not null)
        {
            if (failure is OperationCanceledException) throw failure;
            throw failure;
        }
        cancellationToken.ThrowIfCancellationRequested();

        lock (_gate) return _outputs.ToDictionary(k => k, k => _results[k]);
    }

    private object?[] Arguments(TaskSpec task)
    {
        lock (_gate)
        {
            var values = new object?[task.Dependencies.Count];
            for (var i = 0; i < values.Length; i++) values[i] = _results[task.Dependencies[i]];
            return values;
        }
    }

    private async Task<(object? Value, Exception? Error)> RunWithRetriesAsync(TaskSpec task, object?[] arguments,
        int total, CancellationToken token)
    {
        for (var attempt = 1; ; attempt++)
        {
            if (token.IsCancellationRequested) return (null, new OperationCanceledException(token));

            using var activity = DaskTelemetry.Source.StartActivity(task.Label);
            activity?.SetTag("dask.key", task.Key.ToString());
            activity?.SetTag("dask.attempt", attempt);
            _onEvent?.Invoke(new TaskEvent(TaskEventKind.Started, task.Key, task.Label, Volatile.Read(ref _completed), total));
            var clock = Stopwatch.StartNew();
            lock (_gate) Statistics.TasksRun++;

            try
            {
                var value = task.Function(arguments);
                var elapsed = clock.Elapsed;
                DaskTelemetry.TasksCompleted.Add(1);
                DaskTelemetry.TaskDuration.Record(elapsed.TotalMilliseconds);
                _onEvent?.Invoke(new TaskEvent(TaskEventKind.Finished, task.Key, task.Label,
                    Volatile.Read(ref _completed) + 1, total, Elapsed: elapsed));
                return (value, null);
            }
            catch (Exception e)
            {
                activity?.SetStatus(ActivityStatusCode.Error, e.Message);
                if (_retry.Allows(attempt, e) && !token.IsCancellationRequested)
                {
                    lock (_gate) Statistics.Retries++;
                    DaskTelemetry.TasksRetried.Add(1);
                    _onEvent?.Invoke(new TaskEvent(TaskEventKind.Retrying, task.Key, task.Label,
                        Volatile.Read(ref _completed), total, Error: e));
                    var delay = _retry.DelayFor(attempt);
                    if (delay > TimeSpan.Zero)
                    {
                        try { await Task.Delay(delay, token).ConfigureAwait(false); }
                        catch (OperationCanceledException oce) { return (null, oce); }
                    }
                    continue;
                }

                DaskTelemetry.TasksFailed.Add(1);
                _onEvent?.Invoke(new TaskEvent(TaskEventKind.Failed, task.Key, task.Label,
                    Volatile.Read(ref _completed), total, Error: e));
                return (null, e);
            }
        }
    }
}
