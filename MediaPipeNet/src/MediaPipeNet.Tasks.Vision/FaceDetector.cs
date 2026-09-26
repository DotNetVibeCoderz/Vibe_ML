using MediaPipeNet.Imaging;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="FaceDetector"/>.</summary>
public sealed record FaceDetectorOptions : VisionTaskOptions<FaceDetectionResult>
{
    /// <summary>Minimum confidence for a face to be reported. Default 0.5.</summary>
    public float MinDetectionConfidence { get; init; } = 0.5f;

    /// <summary>IoU above which overlapping detections are merged. Default 0.3.</summary>
    public float MinSuppressionThreshold { get; init; } = 0.3f;

    /// <summary>Maximum faces to return (-1 = all).</summary>
    public int MaxResults { get; init; } = -1;
}

/// <summary>
/// Detects faces with BlazeFace (short range): a bounding box and 6 keypoints per face
/// (eyes, nose tip, mouth, ear tragions). Works best for faces within ~2 m of the camera.
/// </summary>
/// <example>
/// <code>
/// using var detector = FaceDetector.Create();
/// using var image = MPImage.Load("portrait.jpg");
/// foreach (var face in detector.Detect(image).Detections)
///     Console.WriteLine($"{face.BoundingBox} {face.Score:P0}");
/// </code>
/// </example>
public sealed class FaceDetector : VisionTaskBase<FaceDetectionResult>
{
    private static readonly string[] s_keypointNames = ["rightEye", "leftEye", "noseTip", "mouthCenter", "rightEarTragion", "leftEarTragion"];
    private readonly SsdDetector _detector;

    private FaceDetector(FaceDetectorOptions options, SsdDetector detector)
        : base(nameof(FaceDetector), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _detector = detector;
        CompleteInitialization();
    }

    /// <summary>The options the detector was created with.</summary>
    public FaceDetectorOptions Options { get; }

    /// <summary>Creates a detector (resolving the model synchronously; may download it).</summary>
    public static FaceDetector Create(FaceDetectorOptions? options = null)
    {
        options ??= new FaceDetectorOptions();
        return new FaceDetector(options, CreateDetector(options, ModelLoader.Load(options.BaseOptions, ModelCatalog.FaceDetectionShortRange)));
    }

    /// <summary>Creates a detector, downloading the model asynchronously when needed.</summary>
    public static async Task<FaceDetector> CreateAsync(FaceDetectorOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new FaceDetectorOptions();
        var model = await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.FaceDetectionShortRange, cancellationToken).ConfigureAwait(false);
        return new FaceDetector(options, CreateDetector(options, model));
    }

    private static SsdDetector CreateDetector(FaceDetectorOptions o, Inference.OnnxModel model) =>
        new(model, SsdDetectorSpec.FaceShortRange with { NmsThreshold = o.MinSuppressionThreshold });

    /// <summary>Detects faces in a still image.</summary>
    public FaceDetectionResult Detect(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Detects faces in a still image on a worker thread.</summary>
    public Task<FaceDetectionResult> DetectAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Detect(image, processingOptions), cancellationToken);

    /// <summary>Detects faces in a video frame (timestamps must increase).</summary>
    public FaceDetectionResult DetectForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunVideo(image, timestampMs, processingOptions);

    /// <summary>Submits a live-stream frame; the result arrives via <see cref="VisionTaskOptions{TResult}.ResultCallback"/>. Returns false if dropped.</summary>
    public bool DetectLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunLiveStream(image, timestampMs, processingOptions);

    /// <inheritdoc />
    protected override FaceDetectionResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        var raw = DetectRaw(image, options?.ToRoi() ?? NormalizedRect.FullImage, Options.MinDetectionConfidence, Options.MaxResults);
        return new FaceDetectionResult(raw.Select(d => d.ToDetection(image.Width, image.Height, null, s_keypointNames)).ToArray());
    }

    internal List<RawDetection> DetectRaw(MPImage image, in NormalizedRect roi, float minScore, int maxResults) =>
        _detector.Detect(image, roi, minScore, maxResults);

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing) _detector.Dispose();
    }
}
