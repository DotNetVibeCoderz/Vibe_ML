using System.Diagnostics;
using MediaPipeNet.Diagnostics;
using MediaPipeNet.Imaging;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="LiveStreamProcessor{TResult}"/>.</summary>
public sealed record LiveStreamProcessorOptions
{
    /// <summary>
    /// Drop frames that arrive while the previous one is still being processed (always processing the
    /// newest frame). Defaults to the source's <see cref="IFrameSource.IsLive"/>; when false every
    /// frame is processed and capture waits for processing.
    /// </summary>
    public bool? DropFramesWhenBusy { get; init; }

    /// <summary>Mirror frames horizontally before processing (selfie view). Default false.</summary>
    public bool MirrorFrames { get; init; }
}

/// <summary>Live throughput statistics.</summary>
/// <param name="CaptureFps">Frames per second delivered by the source.</param>
/// <param name="ProcessingFps">Frames per second processed.</param>
/// <param name="LastLatencyMs">Processing time of the last frame.</param>
/// <param name="FramesCaptured">Frames read from the source.</param>
/// <param name="FramesProcessed">Frames processed.</param>
/// <param name="FramesDropped">Frames skipped because processing was busy.</param>
public sealed record LiveStreamStats(double CaptureFps, double ProcessingFps, double LastLatencyMs, long FramesCaptured, long FramesProcessed, long FramesDropped);

/// <summary>A processed frame with its result.</summary>
/// <typeparam name="TResult">Result type.</typeparam>
/// <param name="Frame">The processed frame — only valid during the event handler; clone it to keep it.</param>
/// <param name="Result">The task result.</param>
/// <param name="TimestampMs">Frame timestamp.</param>
/// <param name="Latency">Processing time.</param>
/// <param name="Stats">Current statistics.</param>
public sealed record LiveFrameResult<TResult>(MPImage Frame, TResult Result, long TimestampMs, TimeSpan Latency, LiveStreamStats Stats);

/// <summary>
/// Pumps frames from an <see cref="IFrameSource"/> (webcam, video, images) through a processing
/// function on a background thread. With live sources, frames that arrive while the previous frame
/// is still being processed replace the pending frame (latest-frame-wins), so latency never builds
/// up when inference is slower than the camera. Frame buffers are reused — no per-frame allocations.
/// </summary>
/// <example>
/// <code>
/// using var hands = HandLandmarker.Create(new() { RunningMode = RunningMode.Video });
/// await using var processor = new LiveStreamProcessor&lt;HandLandmarkResult&gt;(camera, (frame, ts) =&gt; hands.DetectForVideo(frame, ts));
/// processor.ResultReady += (_, r) =&gt; Console.WriteLine($"{r.Result.Hands.Count} hands, {r.Stats.ProcessingFps:F1} fps");
/// await processor.RunAsync(cancellationToken);
/// </code>
/// </example>
/// <typeparam name="TResult">Result type.</typeparam>
public sealed class LiveStreamProcessor<TResult> : IAsyncDisposable
{
    private readonly IFrameSource _source;
    private readonly Func<MPImage, long, TResult> _process;
    private readonly bool _drop;
    private readonly bool _mirror;
    private readonly Lock _gate = new();
    private readonly SemaphoreSlim _frameReady = new(0, 1);
    private readonly SemaphoreSlim _slotFree = new(1, 1);
    private readonly FrameRateCounter _captureRate = new();
    private readonly FrameRateCounter _processRate = new();
    private MPImage? _pending;
    private MPImage? _working;
    private long _pendingTimestamp;
    private bool _hasPending;
    private bool _captureDone;
    private long _captured, _processed, _dropped;
    private double _lastLatencyMs;

    /// <summary>Creates the processor.</summary>
    /// <param name="source">Frame source.</param>
    /// <param name="process">Processing function (frame, timestamp in ms) — e.g. a task's <c>DetectForVideo</c>.</param>
    /// <param name="options">Options.</param>
    public LiveStreamProcessor(IFrameSource source, Func<MPImage, long, TResult> process, LiveStreamProcessorOptions? options = null)
    {
        _source = source ?? throw new ArgumentNullException(nameof(source));
        _process = process ?? throw new ArgumentNullException(nameof(process));
        options ??= new LiveStreamProcessorOptions();
        _drop = options.DropFramesWhenBusy ?? source.IsLive;
        _mirror = options.MirrorFrames;
    }

    /// <summary>Raised on the processing thread for every processed frame.</summary>
    public event EventHandler<LiveFrameResult<TResult>>? ResultReady;

    /// <summary>Raised when processing a frame throws; processing continues with the next frame.</summary>
    public event EventHandler<Exception>? ProcessingFailed;

    /// <summary>Current statistics.</summary>
    public LiveStreamStats Stats => new(_captureRate.Rate, _processRate.Rate, Volatile.Read(ref _lastLatencyMs),
        Interlocked.Read(ref _captured), Interlocked.Read(ref _processed), Interlocked.Read(ref _dropped));

    /// <summary>Runs until the source ends or <paramref name="cancellationToken"/> fires.</summary>
    public async Task RunAsync(CancellationToken cancellationToken = default)
    {
        var processing = Task.Run(() => ProcessLoopAsync(cancellationToken), CancellationToken.None);
        try
        {
            await foreach (var frame in _source.ReadFramesAsync(cancellationToken).ConfigureAwait(false))
            {
                if (!_drop) await _slotFree.WaitAsync(cancellationToken).ConfigureAwait(false);
                _captureRate.Tick();
                Interlocked.Increment(ref _captured);
                lock (_gate)
                {
                    if (_hasPending)
                    {
                        Interlocked.Increment(ref _dropped);
                        MediaPipeTelemetry.FramesDropped.Add(1);
                    }
                    if (_mirror)
                    {
                        using var flipped = frame.Image.FlipHorizontal();
                        CopyInto(ref _pending, flipped);
                    }
                    else
                    {
                        CopyInto(ref _pending, frame.Image);
                    }
                    _pendingTimestamp = frame.TimestampMs;
                    if (!_hasPending)
                    {
                        _hasPending = true;
                        _frameReady.Release();
                    }
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { }
        finally
        {
            lock (_gate)
            {
                _captureDone = true;
                if (!_hasPending && _frameReady.CurrentCount == 0) _frameReady.Release();
            }
        }
        await processing.ConfigureAwait(false);
    }

    private async Task ProcessLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try { await _frameReady.WaitAsync(ct).ConfigureAwait(false); }
            catch (OperationCanceledException) { return; }

            long ts;
            lock (_gate)
            {
                if (!_hasPending)
                {
                    if (_captureDone) return;
                    continue;
                }
                (_pending, _working) = (_working, _pending);
                _hasPending = false;
                ts = _pendingTimestamp;
            }
            if (!_drop) _slotFree.Release();

            long start = Stopwatch.GetTimestamp();
            try
            {
                var result = _process(_working!, ts);
                var latency = Stopwatch.GetElapsedTime(start);
                Volatile.Write(ref _lastLatencyMs, latency.TotalMilliseconds);
                _processRate.Tick();
                Interlocked.Increment(ref _processed);
                ResultReady?.Invoke(this, new LiveFrameResult<TResult>(_working!, result, ts, latency, Stats));
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                ProcessingFailed?.Invoke(this, e);
            }
            lock (_gate)
            {
                if (_captureDone && !_hasPending) return;
            }
        }
    }

    private static void CopyInto(ref MPImage? target, MPImage source)
    {
        if (target is null) target = source.Clone();
        else target.CopyFrom(source);
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        await _source.DisposeAsync().ConfigureAwait(false);
        _pending?.Dispose();
        _working?.Dispose();
        _frameReady.Dispose();
        _slotFree.Dispose();
    }
}
