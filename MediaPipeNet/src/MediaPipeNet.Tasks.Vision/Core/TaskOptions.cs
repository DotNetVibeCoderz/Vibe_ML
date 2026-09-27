using MediaPipeNet.Imaging;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Per-call pre-processing: an optional region of interest and a rotation.</summary>
/// <param name="RegionOfInterest">Only this part of the image is processed (normalized coordinates).</param>
/// <param name="RotationDegrees">Clockwise rotation of the image content, a multiple of 90.</param>
public sealed record ImageProcessingOptions(NormalizedRect? RegionOfInterest = null, int RotationDegrees = 0)
{
    internal NormalizedRect ToRoi()
    {
        if (RotationDegrees % 90 != 0) throw new ArgumentException("RotationDegrees must be a multiple of 90.");
        var roi = RegionOfInterest ?? NormalizedRect.FullImage;
        return roi with { Rotation = Angles.DegreesToRadians(-RotationDegrees) };
    }
}

/// <summary>Options common to every vision task.</summary>
/// <typeparam name="TResult">Result type delivered to <see cref="ResultCallback"/>.</typeparam>
public abstract record VisionTaskOptions<TResult>
{
    /// <summary>Model and runtime options.</summary>
    public BaseOptions BaseOptions { get; init; } = BaseOptions.Default;

    /// <summary>Running mode (image, video or live stream).</summary>
    public RunningMode RunningMode { get; init; } = RunningMode.Image;

    /// <summary>
    /// Receives results in <see cref="RunningMode.LiveStream"/> mode: the result, the processed frame
    /// (valid only during the call) and the frame timestamp in milliseconds.
    /// </summary>
    public Action<TResult, MPImage, long>? ResultCallback { get; init; }

    /// <summary>
    /// Maximum frames being processed at once in live-stream mode; newer frames are dropped while the
    /// task is busy, keeping latency low. Default 1.
    /// </summary>
    public int MaxInFlightFrames { get; init; } = 1;
}
