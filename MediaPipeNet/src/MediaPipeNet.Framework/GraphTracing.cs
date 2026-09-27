using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.Json;

namespace MediaPipeNet.Framework;

/// <summary>What a <see cref="GraphTraceEvent"/> measured.</summary>
public enum GraphTraceKind
{
    /// <summary>The node's <see cref="ICalculatorNode.OpenAsync"/>.</summary>
    Open,
    /// <summary>One <see cref="ICalculatorNode.ProcessAsync"/> call.</summary>
    Process,
    /// <summary>The node's <see cref="ICalculatorNode.CloseAsync"/>.</summary>
    Close,
}

/// <summary>One recorded node execution.</summary>
/// <param name="Node">Node name.</param>
/// <param name="Kind">Open, process or close.</param>
/// <param name="Timestamp">Packet timestamp processed (for <see cref="GraphTraceKind.Process"/>).</param>
/// <param name="StartMicroseconds">Start time in microseconds since the graph was built.</param>
/// <param name="DurationMicroseconds">Duration in microseconds.</param>
/// <param name="ThreadId">Managed thread that ran the node.</param>
public readonly record struct GraphTraceEvent(string Node, GraphTraceKind Kind, long Timestamp, double StartMicroseconds, double DurationMicroseconds, int ThreadId);

/// <summary>Collects <see cref="GraphTraceEvent"/>s (bounded, thread-safe) and exports them in Chrome trace format.</summary>
internal sealed class GraphTracer(int capacity)
{
    private readonly ConcurrentQueue<GraphTraceEvent> _events = new();
    private readonly long _origin = Stopwatch.GetTimestamp();
    private int _count;

    public void Record(string node, GraphTraceKind kind, long timestamp, long start)
    {
        long end = Stopwatch.GetTimestamp();
        double startUs = (start - _origin) * 1e6 / Stopwatch.Frequency;
        double durUs = (end - start) * 1e6 / Stopwatch.Frequency;
        _events.Enqueue(new GraphTraceEvent(node, kind, timestamp, startUs, durUs, Environment.CurrentManagedThreadId));
        if (Interlocked.Increment(ref _count) > capacity && _events.TryDequeue(out _)) Interlocked.Decrement(ref _count);
    }

    public IReadOnlyList<GraphTraceEvent> Snapshot() => [.. _events];

    /// <summary>
    /// Writes the events as a Chrome trace (<c>{"traceEvents": [...]}</c>, complete "X" events, one row per
    /// thread) loadable in <c>chrome://tracing</c>, <c>edge://tracing</c> or https://ui.perfetto.dev.
    /// </summary>
    public static void WriteChromeTrace(IEnumerable<GraphTraceEvent> events, Stream stream)
    {
        using var w = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = false });
        w.WriteStartObject();
        w.WriteString("displayTimeUnit", "ms");
        w.WriteStartArray("traceEvents");
        foreach (var e in events)
        {
            w.WriteStartObject();
            w.WriteString("name", e.Node);
            w.WriteString("cat", e.Kind.ToString().ToLowerInvariant());
            w.WriteString("ph", "X");
            w.WriteNumber("ts", Math.Round(e.StartMicroseconds, 3));
            w.WriteNumber("dur", Math.Round(e.DurationMicroseconds, 3));
            w.WriteNumber("pid", 1);
            w.WriteNumber("tid", e.ThreadId);
            w.WriteStartObject("args");
            if (e.Kind == GraphTraceKind.Process) w.WriteNumber("timestamp_us", e.Timestamp);
            w.WriteEndObject();
            w.WriteEndObject();
        }
        w.WriteEndArray();
        w.WriteEndObject();
    }
}

/// <summary>A fixed set of dedicated threads running the nodes assigned to it.</summary>
internal sealed class GraphExecutor : IDisposable
{
    private readonly BlockingCollection<Action> _work = [];
    private readonly Thread[] _threads;

    public GraphExecutor(string name, int threads)
    {
        Name = name;
        _threads = new Thread[threads];
        for (int i = 0; i < threads; i++)
        {
            _threads[i] = new Thread(Run) { IsBackground = true, Name = $"mpnet-executor-{name}-{i}" };
            _threads[i].Start();
        }
    }

    public string Name { get; }

    public void Post(Action action)
    {
        if (!_work.IsAddingCompleted) _work.Add(action);
    }

    private void Run()
    {
        foreach (var action in _work.GetConsumingEnumerable())
        {
            // Nodes are synchronous in practice; run the whole node step on this thread.
            var previous = SynchronizationContext.Current;
            SynchronizationContext.SetSynchronizationContext(null);
            try { action(); }
            finally { SynchronizationContext.SetSynchronizationContext(previous); }
        }
    }

    public void Dispose()
    {
        _work.CompleteAdding();
        foreach (var t in _threads)
            if (t != Thread.CurrentThread) t.Join(TimeSpan.FromSeconds(5));
        _work.Dispose();
    }
}
