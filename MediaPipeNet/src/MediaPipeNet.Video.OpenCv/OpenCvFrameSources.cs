using System.Diagnostics;
using System.Runtime.CompilerServices;
using MediaPipeNet.Imaging;
using OpenCvSharp;

namespace MediaPipeNet.Video.OpenCv;

/// <summary>Base class of OpenCV <see cref="VideoCapture"/>-backed sources.</summary>
public abstract class OpenCvFrameSource : IFrameSource
{
    private readonly VideoCapture _capture;
    private bool _disposed;

    /// <summary>Wraps an opened capture.</summary>
    protected OpenCvFrameSource(VideoCapture capture, string name)
    {
        _capture = capture ?? throw new ArgumentNullException(nameof(capture));
        if (!_capture.IsOpened()) throw new InvalidOperationException($"Could not open video source '{name}'.");
        Name = name;
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public ImageSize? FrameSize => _capture.FrameWidth > 0 ? new ImageSize(_capture.FrameWidth, _capture.FrameHeight) : null;

    /// <inheritdoc />
    public double? FrameRate => _capture.Fps > 0 ? _capture.Fps : null;

    /// <inheritdoc />
    public abstract bool IsLive { get; }

    /// <summary>The underlying OpenCV capture (e.g. to set exposure).</summary>
    protected VideoCapture Capture => _capture;

    /// <summary>Timestamp of the frame just read.</summary>
    protected abstract long GetTimestampMs(long index, Stopwatch clock);

    /// <inheritdoc />
    public async IAsyncEnumerable<VideoFrame> ReadFramesAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var mat = new Mat();
        MPImage? frame = null;
        var clock = Stopwatch.StartNew();
        long index = 0, lastTs = -1;
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                // VideoCapture.Read blocks until a frame is available; keep it off the caller's thread.
                bool ok = await Task.Run(() => _capture.Read(mat), cancellationToken).ConfigureAwait(false);
                if (!ok || mat.Empty())
                {
                    if (IsLive) continue;
                    yield break;
                }
                Fill(ref frame, mat);
                long ts = Math.Max(GetTimestampMs(index, clock), lastTs + 1);
                lastTs = ts;
                yield return new VideoFrame(frame!, ts, index++);
            }
        }
        finally
        {
            frame?.Dispose();
        }
    }

    private static unsafe void Fill(ref MPImage? frame, Mat mat)
    {
        var format = mat.Channels() switch
        {
            1 => PixelFormat.Gray8,
            3 => PixelFormat.Bgr24,
            4 => PixelFormat.Bgra32,
            var c => throw new NotSupportedException($"Unsupported channel count {c}."),
        };
        int stride = (int)mat.Step();
        var span = new ReadOnlySpan<byte>((void*)mat.Data, stride * mat.Rows);
        if (frame is null) frame = MPImage.FromPixelData(span, mat.Cols, mat.Rows, format, stride);
        else frame.CopyFrom(span, mat.Cols, mat.Rows, format, stride);
    }

    /// <inheritdoc />
    public ValueTask DisposeAsync()
    {
        if (!_disposed)
        {
            _disposed = true;
            _capture.Release();
            _capture.Dispose();
        }
        GC.SuppressFinalize(this);
        return ValueTask.CompletedTask;
    }
}

/// <summary>A camera (webcam) source. Frames are timestamped with the wall clock.</summary>
public sealed class WebcamFrameSource : OpenCvFrameSource
{
    /// <summary>Opens camera <paramref name="deviceIndex"/>, optionally requesting a resolution and frame rate.</summary>
    public WebcamFrameSource(int deviceIndex = 0, int width = 640, int height = 480, double fps = 30)
        : base(Open(deviceIndex, width, height, fps), $"Camera {deviceIndex}")
    {
        DeviceIndex = deviceIndex;
    }

    /// <summary>The camera index.</summary>
    public int DeviceIndex { get; }

    /// <inheritdoc />
    public override bool IsLive => true;

    /// <inheritdoc />
    protected override long GetTimestampMs(long index, Stopwatch clock) => clock.ElapsedMilliseconds;

    /// <summary>Returns the indices of cameras that can be opened (probes 0..<paramref name="maxDevices"/>-1).</summary>
    public static IReadOnlyList<int> ListCameras(int maxDevices = 5)
    {
        var found = new List<int>();
        for (int i = 0; i < maxDevices; i++)
        {
            using var cap = new VideoCapture(i, OperatingSystem.IsWindows() ? VideoCaptureAPIs.DSHOW : VideoCaptureAPIs.ANY);
            if (cap.IsOpened()) found.Add(i);
        }
        return found;
    }

    private static VideoCapture Open(int index, int width, int height, double fps)
    {
        var cap = new VideoCapture(index, OperatingSystem.IsWindows() ? VideoCaptureAPIs.DSHOW : VideoCaptureAPIs.ANY);
        if (cap.IsOpened())
        {
            cap.Set(VideoCaptureProperties.FrameWidth, width);
            cap.Set(VideoCaptureProperties.FrameHeight, height);
            cap.Set(VideoCaptureProperties.Fps, fps);
        }
        return cap;
    }
}

/// <summary>A video file source. Frames are timestamped from the container's frame rate.</summary>
public sealed class VideoFileFrameSource : OpenCvFrameSource
{
    /// <summary>Opens a video file (any format OpenCV/FFmpeg can decode).</summary>
    public VideoFileFrameSource(string path) : base(new VideoCapture(path), Path.GetFileName(path)) { }

    /// <inheritdoc />
    public override bool IsLive => false;

    /// <summary>Total frames reported by the container (may be approximate).</summary>
    public int FrameCount => Capture.FrameCount;

    /// <inheritdoc />
    protected override long GetTimestampMs(long index, Stopwatch clock) =>
        (long)Math.Round(index * 1000.0 / (FrameRate ?? 30.0));
}
