using System.Runtime.CompilerServices;

namespace MediaPipeNet.Imaging;

/// <summary>A decoded frame produced by an <see cref="IFrameSource"/>.</summary>
/// <param name="Image">
/// The pixels. Sources may reuse one buffer across frames: the image is only guaranteed valid until
/// the enumeration advances. Call <see cref="MPImage.Clone"/> to keep it longer.
/// </param>
/// <param name="TimestampMs">Presentation timestamp in milliseconds (monotonically increasing).</param>
/// <param name="Index">Zero-based frame index.</param>
public sealed record VideoFrame(MPImage Image, long TimestampMs, long Index);

/// <summary>
/// A source of video frames: image files, image sequences, video files or cameras. Consumers
/// enumerate <see cref="ReadFramesAsync"/>; live sources never complete on their own.
/// </summary>
public interface IFrameSource : IAsyncDisposable
{
    /// <summary>Human-readable description of the source.</summary>
    string Name { get; }

    /// <summary>Frame size when known in advance.</summary>
    ImageSize? FrameSize { get; }

    /// <summary>Nominal frame rate when known.</summary>
    double? FrameRate { get; }

    /// <summary>True for real-time sources (cameras) that produce frames regardless of the consumer's pace.</summary>
    bool IsLive { get; }

    /// <summary>Enumerates frames until the source ends or <paramref name="cancellationToken"/> fires.</summary>
    IAsyncEnumerable<VideoFrame> ReadFramesAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Plays a list of still images as a video: each file becomes one frame, timestamped at
/// <see cref="FrameRate"/>. Optional looping makes it behave like an endless stream.
/// </summary>
public sealed class ImageFileFrameSource : IFrameSource
{
    private readonly IReadOnlyList<string> _paths;
    private readonly bool _loop;

    /// <summary>Creates a source over the given image files.</summary>
    /// <param name="paths">Image files, played in order.</param>
    /// <param name="frameRate">Rate used to compute timestamps.</param>
    /// <param name="loop">Restart from the first image after the last one.</param>
    public ImageFileFrameSource(IEnumerable<string> paths, double frameRate = 30, bool loop = false)
    {
        ArgumentNullException.ThrowIfNull(paths);
        _paths = paths.ToArray();
        if (_paths.Count == 0) throw new ArgumentException("At least one image path is required.", nameof(paths));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(frameRate);
        FrameRate = frameRate;
        _loop = loop;
    }

    /// <summary>Creates a source over a single image file.</summary>
    public ImageFileFrameSource(string path, double frameRate = 30, bool loop = false) : this([path], frameRate, loop) { }

    /// <summary>Creates a source over every image in a directory, sorted by file name.</summary>
    public static ImageFileFrameSource FromDirectory(string directory, double frameRate = 30, bool loop = false)
    {
        string[] extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"];
        var files = Directory.EnumerateFiles(directory)
            .Where(f => extensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase))
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        return new ImageFileFrameSource(files, frameRate, loop);
    }

    /// <inheritdoc />
    public string Name => _paths.Count == 1 ? Path.GetFileName(_paths[0]) : $"{_paths.Count} images";

    /// <inheritdoc />
    public ImageSize? FrameSize => null;

    /// <inheritdoc />
    public double? FrameRate { get; }

    /// <inheritdoc />
    public bool IsLive => false;

    /// <inheritdoc />
    public async IAsyncEnumerable<VideoFrame> ReadFramesAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        long index = 0;
        double frameMs = 1000.0 / FrameRate!.Value;
        do
        {
            foreach (var path in _paths)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var frame = await MPImage.LoadAsync(path, cancellationToken).ConfigureAwait(false);
                yield return new VideoFrame(frame, (long)Math.Round(index * frameMs), index);
                index++;
            }
        } while (_loop);
    }

    /// <inheritdoc />
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

/// <summary>
/// Plays in-memory frames (e.g. frames already decoded by another library). The frames are not
/// disposed by the source.
/// </summary>
public sealed class MemoryFrameSource : IFrameSource
{
    private readonly IReadOnlyList<MPImage> _frames;
    private readonly bool _pace;

    /// <summary>Creates the source.</summary>
    /// <param name="frames">Frames to play.</param>
    /// <param name="frameRate">Rate used for timestamps (and pacing when <paramref name="realTime"/> is set).</param>
    /// <param name="realTime">Emit frames at <paramref name="frameRate"/> in wall-clock time, simulating a camera.</param>
    public MemoryFrameSource(IReadOnlyList<MPImage> frames, double frameRate = 30, bool realTime = false)
    {
        _frames = frames ?? throw new ArgumentNullException(nameof(frames));
        FrameRate = frameRate;
        _pace = realTime;
    }

    /// <inheritdoc />
    public string Name => $"{_frames.Count} in-memory frames";

    /// <inheritdoc />
    public ImageSize? FrameSize => _frames.Count > 0 ? _frames[0].Size : null;

    /// <inheritdoc />
    public double? FrameRate { get; }

    /// <inheritdoc />
    public bool IsLive => _pace;

    /// <inheritdoc />
    public async IAsyncEnumerable<VideoFrame> ReadFramesAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        double frameMs = 1000.0 / FrameRate!.Value;
        var start = System.Diagnostics.Stopwatch.GetTimestamp();
        for (int i = 0; i < _frames.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            long ts = (long)Math.Round(i * frameMs);
            if (_pace)
            {
                var wait = TimeSpan.FromMilliseconds(ts) - System.Diagnostics.Stopwatch.GetElapsedTime(start);
                if (wait > TimeSpan.Zero) await Task.Delay(wait, cancellationToken).ConfigureAwait(false);
            }
            yield return new VideoFrame(_frames[i], ts, i);
        }
    }

    /// <inheritdoc />
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
