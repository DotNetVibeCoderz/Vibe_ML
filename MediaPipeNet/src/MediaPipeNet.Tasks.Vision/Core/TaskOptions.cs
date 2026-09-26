using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using Microsoft.Extensions.Logging;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options shared by every task: where models come from and how they run.</summary>
public sealed record BaseOptions
{
    /// <summary>Default options (default <see cref="ModelStore"/>, automatic execution provider).</summary>
    public static BaseOptions Default { get; } = new();

    /// <summary>Model store used to resolve model files; <see cref="ModelStore.Default"/> when null.</summary>
    public ModelStore? ModelStore { get; init; }

    /// <summary>A directory searched first for model files (e.g. a folder you ship yourself).</summary>
    public string? ModelDirectory { get; init; }

    /// <summary>Explicit model file per model id (see <see cref="ModelCatalog"/>), overriding every other source.</summary>
    public IReadOnlyDictionary<string, string>? ModelPaths { get; init; }

    /// <summary>Execution provider and threading.</summary>
    public InferenceOptions Inference { get; init; } = InferenceOptions.Default;

    /// <summary>Logger factory for task diagnostics.</summary>
    public ILoggerFactory? LoggerFactory { get; init; }
}

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
