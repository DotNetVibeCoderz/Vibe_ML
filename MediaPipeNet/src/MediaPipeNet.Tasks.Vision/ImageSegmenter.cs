using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="ImageSegmenter"/>.</summary>
public sealed record ImageSegmenterOptions : VisionTaskOptions<SegmentationResult>
{
    /// <summary>
    /// Temporal smoothing factor in video/live-stream mode: the new mask is blended with the previous
    /// one (0 = no smoothing, 0.9 = heavy). Default 0.3.
    /// </summary>
    public float TemporalSmoothing { get; init; } = 0.3f;
}

/// <summary>
/// Selfie segmentation: a per-pixel foreground (person) probability for the whole image, suitable
/// for background blur/replacement. Uses the MediaPipe selfie segmenter (256×256).
/// </summary>
/// <example>
/// <code>
/// using var segmenter = ImageSegmenter.Create();
/// var mask = segmenter.Segment(image).ConfidenceMask;
/// Console.WriteLine($"Person covers {mask.Coverage():P0} of the image");
/// </code>
/// </example>
public sealed class ImageSegmenter : VisionTaskBase<SegmentationResult>
{
    private readonly OnnxModel _model;
    private float[]? _previous;

    private ImageSegmenter(ImageSegmenterOptions options, OnnxModel model)
        : base(nameof(ImageSegmenter), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _model = model;
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public ImageSegmenterOptions Options { get; }

    /// <summary>Creates the task (resolving the model synchronously).</summary>
    public static ImageSegmenter Create(ImageSegmenterOptions? options = null)
    {
        options ??= new ImageSegmenterOptions();
        return new ImageSegmenter(options, ModelLoader.Load(options.BaseOptions, ModelCatalog.SelfieSegmenter));
    }

    /// <summary>Creates the task, downloading the model asynchronously when needed.</summary>
    public static async Task<ImageSegmenter> CreateAsync(ImageSegmenterOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new ImageSegmenterOptions();
        return new ImageSegmenter(options, await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.SelfieSegmenter, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Segments a still image.</summary>
    public SegmentationResult Segment(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Segments a still image on a worker thread.</summary>
    public Task<SegmentationResult> SegmentAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Segment(image, processingOptions), cancellationToken);

    /// <summary>Segments a video frame (with temporal smoothing).</summary>
    public SegmentationResult SegmentForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunVideo(image, timestampMs, processingOptions);

    /// <summary>Submits a live-stream frame. Returns false if it was dropped.</summary>
    public bool SegmentLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunLiveStream(image, timestampMs, processingOptions);

    /// <inheritdoc />
    public override void ResetTracking() => _previous = null;

    /// <inheritdoc />
    protected override SegmentationResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        var spec = _model.Inputs[0];
        int th = spec.Shape[1], tw = spec.Shape[2];
        var probs = new float[tw * th];
        TensorMapping mapping;
        using (var ctx = _model.RentContext())
        {
            mapping = ImageToTensor.Convert(image, options?.ToRoi() ?? NormalizedRect.FullImage,
                new ImageToTensorOptions(tw, th, 0f, 1f, KeepAspectRatio: false, BorderMode.Replicate), ctx.GetInput(0));
            ctx.Run();
            ctx.GetOutput(0).CopyTo(probs);
        }
        if (tracking && Options.TemporalSmoothing > 0)
        {
            float a = Math.Clamp(Options.TemporalSmoothing, 0f, 0.99f);
            if (_previous is { Length: var n } prev && n == probs.Length)
                for (int i = 0; i < probs.Length; i++) probs[i] = prev[i] * a + probs[i] * (1 - a);
            _previous = probs;
        }
        var data = new float[image.Width * image.Height];
        TensorWarp.ProjectToImage(probs, tw, th, mapping, data, image.Width, image.Height);
        return new SegmentationResult(new SegmentationMask(image.Width, image.Height, data));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing) _model.Dispose();
    }
}
