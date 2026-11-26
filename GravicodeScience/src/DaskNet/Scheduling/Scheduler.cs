using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace Gravicode.Science.DaskNet;

/// <summary>Runs a task graph and returns the results of the requested keys.</summary>
public interface IScheduler
{
    /// <summary>A short name for logs (<c>"threads"</c>, <c>"sync"</c>, <c>"cluster"</c>).</summary>
    string Name { get; }

    /// <summary>
    /// Computes <paramref name="outputs"/> from <paramref name="graph"/>. Intermediate results
    /// are released as soon as nothing else needs them.
    /// </summary>
    Task<IReadOnlyDictionary<TaskKey, object?>> ExecuteAsync(TaskGraph graph, IReadOnlyList<TaskKey> outputs,
        CancellationToken cancellationToken = default);
}

/// <summary>How often, and after how long, a failed task is tried again.</summary>
/// <param name="MaxRetries">Attempts after the first; 0 means a failure is final.</param>
/// <param name="Delay">Wait before the first retry; doubled each time.</param>
/// <param name="ShouldRetry">Which exceptions are worth retrying; <c>null</c> retries everything
/// except cancellation.</param>
public sealed record RetryPolicy(int MaxRetries = 0, TimeSpan Delay = default, Func<Exception, bool>? ShouldRetry = null)
{
    /// <summary>No retries.</summary>
    public static RetryPolicy None { get; } = new();

    /// <summary>Whether attempt <paramref name="attempt"/> (1-based) may be followed by another.</summary>
    public bool Allows(int attempt, Exception error)
        => attempt <= MaxRetries && error is not OperationCanceledException && (ShouldRetry?.Invoke(error) ?? true);

    /// <summary>The wait before retry number <paramref name="attempt"/> (1-based).</summary>
    public TimeSpan DelayFor(int attempt)
        => Delay <= TimeSpan.Zero ? TimeSpan.Zero : TimeSpan.FromTicks(Delay.Ticks * (1L << Math.Min(attempt - 1, 16)));
}

/// <summary>What happened to a task.</summary>
public enum TaskEventKind
{
    /// <summary>The task started running.</summary>
    Started,
    /// <summary>The task finished.</summary>
    Finished,
    /// <summary>The task failed and will be tried again.</summary>
    Retrying,
    /// <summary>The task failed for good.</summary>
    Failed,
    /// <summary>A result held by a lost worker is being recomputed.</summary>
    Recomputing,
    /// <summary>A straggling task got a second attempt on another worker (<see cref="SpeculationPolicy"/>).</summary>
    Speculating,
}

/// <summary>A progress event from a scheduler.</summary>
/// <param name="Kind">What happened.</param>
/// <param name="Key">The task.</param>
/// <param name="Label">The task's operation name.</param>
/// <param name="Completed">Tasks finished so far in this run.</param>
/// <param name="Total">Tasks in this run.</param>
/// <param name="Worker">Where it ran, for cluster schedulers.</param>
/// <param name="Elapsed">Run time, for <see cref="TaskEventKind.Finished"/>.</param>
/// <param name="Error">The failure, for <see cref="TaskEventKind.Retrying"/> and <see cref="TaskEventKind.Failed"/>.</param>
public sealed record TaskEvent(TaskEventKind Kind, TaskKey Key, string Label, int Completed, int Total,
    string? Worker = null, TimeSpan Elapsed = default, Exception? Error = null);

/// <summary>A task failed after its retries.</summary>
public sealed class DaskTaskException(TaskKey key, string label, Exception inner)
    : Exception($"Task {key} ({label}) failed: {inner.Message}", inner)
{
    /// <summary>The failing task.</summary>
    public TaskKey Key { get; } = key;

    /// <summary>The failing task's operation name.</summary>
    public string Label { get; } = label;
}

/// <summary>What one scheduler run did.</summary>
/// <summary>
/// When a cluster run gives a straggling task a second attempt on an idle worker (speculative
/// execution - the cluster's form of work stealing). Whichever attempt finishes first wins; the
/// other is ignored and its result released.
/// </summary>
/// <param name="Multiplier">A task straggles once it has run this many times the median duration of
/// finished tasks with the same label.</param>
/// <param name="Minimum">...and at least this long, so short tasks are never duplicated.</param>
/// <param name="MinimumSamples">Finished tasks with that label needed before the median is trusted.</param>
/// <remarks>Only for tasks without side effects: both attempts may run to completion.</remarks>
public sealed record SpeculationPolicy(double Multiplier = 3.0, TimeSpan Minimum = default, int MinimumSamples = 3)
{
    /// <summary>The threshold actually used: <see cref="Minimum"/>, or one second when unset.</summary>
    public TimeSpan EffectiveMinimum => Minimum == default ? TimeSpan.FromSeconds(1) : Minimum;
}

public sealed class RunStatistics
{
    /// <summary>Tasks executed (retries and recomputations included).</summary>
    public int TasksRun { get; internal set; }

    /// <summary>Tasks in the graph.</summary>
    public int TasksTotal { get; internal set; }

    /// <summary>Retries after failures.</summary>
    public int Retries { get; internal set; }

    /// <summary>Straggling tasks given a second, speculative attempt on an idle worker.</summary>
    public int Speculated { get; internal set; }

    /// <summary>Speculative attempts that finished first.</summary>
    public int SpeculationWins { get; internal set; }

    /// <summary>The most results held in memory at once.</summary>
    public int PeakResultsHeld { get; internal set; }

    /// <summary>Wall-clock time.</summary>
    public TimeSpan Elapsed { get; internal set; }

    /// <summary>Tasks run per worker, for cluster schedulers.</summary>
    public Dictionary<string, int> TasksPerWorker { get; } = [];

    /// <inheritdoc />
    public override string ToString()
        => $"{TasksRun}/{TasksTotal} tasks, {Retries} retries, peak {PeakResultsHeld} results held, {Elapsed.TotalMilliseconds:F1} ms";
}

/// <summary>
/// OpenTelemetry-compatible instrumentation: one <see cref="Activity"/> per task and counters on
/// a <see cref="Meter"/>, both named <c>Gravicode.Science.DaskNet</c>.
/// </summary>
/// <remarks>
/// Built on <c>System.Diagnostics</c> only, so the library takes no OpenTelemetry dependency; an
/// application that registers the source and meter with its OpenTelemetry SDK exports to Jaeger,
/// Zipkin, Prometheus or Aspire with no further code. Nothing is recorded while nobody listens.
/// </remarks>
public static class DaskTelemetry
{
    /// <summary>The source and meter name.</summary>
    public const string Name = "Gravicode.Science.DaskNet";

    /// <summary>The activity source tasks are traced on.</summary>
    public static ActivitySource Source { get; } = new(Name, "1.0");

    /// <summary>The meter tasks are counted on.</summary>
    public static Meter Meter { get; } = new(Name, "1.0");

    internal static readonly Counter<long> TasksCompleted = Meter.CreateCounter<long>("dask.tasks.completed", "{task}");
    internal static readonly Counter<long> TasksFailed = Meter.CreateCounter<long>("dask.tasks.failed", "{task}");
    internal static readonly Counter<long> TasksRetried = Meter.CreateCounter<long>("dask.tasks.retried", "{task}");
    internal static readonly Histogram<double> TaskDuration = Meter.CreateHistogram<double>("dask.task.duration", "ms");
}
