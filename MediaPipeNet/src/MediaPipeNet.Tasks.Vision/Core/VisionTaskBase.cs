using System.Diagnostics;
using MediaPipeNet.Diagnostics;
using MediaPipeNet.Framework;
using MediaPipeNet.Imaging;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>
/// Common machinery of every vision task: running-mode checks, monotonic timestamps, telemetry and
/// the live-stream pipeline. Live-stream mode runs the task inside a small <see cref="CalculatorGraph"/>
/// with in-flight limiting, so frames submitted while the task is busy are dropped instead of queued.
/// </summary>
/// <typeparam name="TResult">The task result type.</typeparam>
public abstract class VisionTaskBase<TResult> : IDisposable where TResult : class
{
    private readonly Lock _sequentialGate = new();
    private readonly Action<TResult, MPImage, long>? _callback;
    private readonly int _maxInFlight;
    private readonly KeyValuePair<string, object?>[] _tags;
    private CalculatorGraph? _graph;
    private long _lastTimestampMs = long.MinValue;
    private long _droppedFrames;
    private bool _disposed;

    /// <summary>Initializes the base.</summary>
    protected VisionTaskBase(string taskName, RunningMode runningMode, BaseOptions baseOptions,
        Action<TResult, MPImage, long>? resultCallback, int maxInFlightFrames)
    {
        ArgumentNullException.ThrowIfNull(baseOptions);
        TaskName = taskName;
        RunningMode = runningMode;
        _callback = resultCallback;
        _maxInFlight = Math.Max(1, maxInFlightFrames);
        Logger = baseOptions.LoggerFactory?.CreateLogger("MediaPipeNet.Tasks." + taskName) ?? NullLogger.Instance;
        _tags = [new("task", taskName)];
        if (runningMode == RunningMode.LiveStream && resultCallback is null)
            throw new ArgumentException("A ResultCallback is required in LiveStream mode.", nameof(resultCallback));
        if (runningMode != RunningMode.LiveStream && resultCallback is not null)
            throw new ArgumentException("ResultCallback is only allowed in LiveStream mode.", nameof(resultCallback));
    }

    /// <summary>Task name used in logs and metrics.</summary>
    public string TaskName { get; }

    /// <summary>The running mode the task was created with.</summary>
    public RunningMode RunningMode { get; }

    /// <summary>Live-stream frames dropped because the task was busy.</summary>
    public long DroppedFrames => Interlocked.Read(ref _droppedFrames);

    /// <summary>Task logger.</summary>
    protected ILogger Logger { get; }

    /// <summary>
    /// Runs the task on one image. <paramref name="tracking"/> is true in video and live-stream
    /// modes, where state from previous frames (ROIs, filters) may be reused.
    /// </summary>
    protected abstract TResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs);

    /// <summary>Must be called at the end of the derived constructor (starts the live-stream graph when needed).</summary>
    protected void CompleteInitialization()
    {
        if (RunningMode != RunningMode.LiveStream) return;
        var builder = new GraphBuilder { Options = new GraphOptions { MaxInFlight = _maxInFlight, Logger = Logger } };
        builder.AddInputStream<LiveFrame>("frames")
            .AddNode(TaskName, new LiveTaskNode(this)).In("IN", "frames").Out("OUT", "results").Graph
            .AddOutputStream("results");
        _graph = builder.Build();
        _graph.ObserveOutputStream<LiveOutput>("results", p =>
        {
            try { _callback!(p.Value.Result, p.Value.Frame, (long)p.Timestamp.Milliseconds); }
            catch (Exception e) { Logger.LogError(e, "{Task}: result callback threw", TaskName); }
            finally { p.Value.Frame.Dispose(); }
        });
        _graph.StartAsync().GetAwaiter().GetResult();
    }

    /// <summary>Image mode entry point.</summary>
    protected TResult RunImage(MPImage image, ImageProcessingOptions? options)
    {
        EnsureMode(RunningMode.Image, image);
        return Measure(image, options, tracking: false, 0);
    }

    /// <summary>Video mode entry point (sequential, monotonic timestamps).</summary>
    protected TResult RunVideo(MPImage image, long timestampMs, ImageProcessingOptions? options)
    {
        EnsureMode(RunningMode.Video, image);
        lock (_sequentialGate)
        {
            CheckTimestamp(timestampMs);
            return Measure(image, options, tracking: true, timestampMs);
        }
    }

    /// <summary>Live-stream entry point; returns false when the frame was dropped.</summary>
    protected bool RunLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? options)
    {
        EnsureMode(RunningMode.LiveStream, image);
        lock (_sequentialGate)
        {
            CheckTimestamp(timestampMs);
            if (_graph!.InFlightCount >= _maxInFlight)
            {
                Interlocked.Increment(ref _droppedFrames);
                MediaPipeTelemetry.FramesDropped.Add(1, _tags);
                return false;
            }
            var clone = image.Clone();
            if (_graph.AddPacket("frames", Packet.Create(new LiveFrame(clone, options), timestampMs))) return true;
            clone.Dispose();
            Interlocked.Increment(ref _droppedFrames);
            return false;
        }
    }

    /// <summary>Clears tracking state (previous ROIs, filters). Call when a video jumps.</summary>
    public virtual void ResetTracking() { }

    private TResult Measure(MPImage image, ImageProcessingOptions? options, bool tracking, long ts)
    {
        long start = Stopwatch.GetTimestamp();
        var result = Process(image, options, tracking, ts);
        MediaPipeTelemetry.TaskDuration.Record(Stopwatch.GetElapsedTime(start).TotalMilliseconds, _tags);
        MediaPipeTelemetry.FramesProcessed.Add(1, _tags);
        return result;
    }

    private void CheckTimestamp(long timestampMs)
    {
        if (timestampMs <= _lastTimestampMs)
            throw new ArgumentException($"Timestamps must increase monotonically: {timestampMs} ms after {_lastTimestampMs} ms.", nameof(timestampMs));
        _lastTimestampMs = timestampMs;
    }

    private void EnsureMode(RunningMode expected, MPImage image)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(image);
        if (RunningMode != expected)
            throw new InvalidOperationException($"{TaskName} was created in {RunningMode} mode; this method requires {expected} mode.");
    }

    /// <summary>Releases models and the live-stream graph.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_graph is not null)
        {
            _graph.DisposeAsync().AsTask().GetAwaiter().GetResult();
            _graph = null;
        }
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>Releases task resources.</summary>
    protected virtual void Dispose(bool disposing) { }

    private sealed record LiveFrame(MPImage Frame, ImageProcessingOptions? Options);

    private sealed record LiveOutput(TResult Result, MPImage Frame);

    private sealed class LiveTaskNode(VisionTaskBase<TResult> task) : CalculatorNode
    {
        public override void GetContract(CalculatorContract contract) => contract.AddInput<LiveFrame>("IN").AddOutput<LiveOutput>("OUT");

        protected override void Process(CalculatorContext context)
        {
            var input = context.GetInput<LiveFrame>("IN");
            TResult result;
            try
            {
                result = task.Measure(input.Frame, input.Options, tracking: true, (long)context.InputTimestamp.Milliseconds);
            }
            catch
            {
                input.Frame.Dispose();
                throw;
            }
            context.Send("OUT", new LiveOutput(result, input.Frame));
        }
    }
}
