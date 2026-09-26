using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace MediaPipeNet.Diagnostics;

/// <summary>
/// The <see cref="Meter"/> and <see cref="ActivitySource"/> through which MediaPipe.NET publishes
/// metrics and traces. Both are named <c>MediaPipeNet</c>, so an OpenTelemetry pipeline can
/// subscribe with <c>AddMeter("MediaPipeNet")</c> / <c>AddSource("MediaPipeNet")</c>.
/// </summary>
/// <remarks>
/// Instruments:
/// <list type="bullet">
/// <item><c>mediapipenet.inference.duration</c> (histogram, ms) — one ONNX Runtime run, tagged <c>model</c> and <c>provider</c>.</item>
/// <item><c>mediapipenet.task.duration</c> (histogram, ms) — one task invocation end to end, tagged <c>task</c>.</item>
/// <item><c>mediapipenet.frames.processed</c> (counter) — frames processed by tasks, tagged <c>task</c>.</item>
/// <item><c>mediapipenet.frames.dropped</c> (counter) — live-stream frames dropped because the task was busy.</item>
/// <item><c>mediapipenet.graph.packets</c> (counter) — packets emitted by graph nodes, tagged <c>node</c>.</item>
/// </list>
/// </remarks>
public static class MediaPipeTelemetry
{
    /// <summary>The meter and activity source name.</summary>
    public const string Name = "MediaPipeNet";

    /// <summary>The library version reported by the meter.</summary>
    public static readonly string Version = typeof(MediaPipeTelemetry).Assembly.GetName().Version?.ToString() ?? "0.0.0";

    /// <summary>The meter used by all MediaPipe.NET instruments.</summary>
    public static Meter Meter { get; } = new(Name, Version);

    /// <summary>The activity source used for tracing task and graph execution.</summary>
    public static ActivitySource ActivitySource { get; } = new(Name, Version);

    /// <summary>Duration of a single model inference in milliseconds.</summary>
    public static Histogram<double> InferenceDuration { get; } =
        Meter.CreateHistogram<double>("mediapipenet.inference.duration", "ms", "Duration of one ONNX Runtime inference.");

    /// <summary>Duration of a single task invocation in milliseconds.</summary>
    public static Histogram<double> TaskDuration { get; } =
        Meter.CreateHistogram<double>("mediapipenet.task.duration", "ms", "End-to-end duration of one task invocation.");

    /// <summary>Number of frames processed by tasks.</summary>
    public static Counter<long> FramesProcessed { get; } =
        Meter.CreateCounter<long>("mediapipenet.frames.processed", "{frame}", "Frames processed by tasks.");

    /// <summary>Number of live-stream frames dropped because the pipeline was busy.</summary>
    public static Counter<long> FramesDropped { get; } =
        Meter.CreateCounter<long>("mediapipenet.frames.dropped", "{frame}", "Live-stream frames dropped by flow limiting.");

    /// <summary>Number of packets emitted by graph nodes.</summary>
    public static Counter<long> GraphPackets { get; } =
        Meter.CreateCounter<long>("mediapipenet.graph.packets", "{packet}", "Packets emitted by calculator graph nodes.");
}
