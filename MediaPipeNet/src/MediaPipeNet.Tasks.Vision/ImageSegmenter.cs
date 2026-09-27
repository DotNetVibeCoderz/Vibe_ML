using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Segmentation model used by <see cref="ImageSegmenter"/>.</summary>
public enum SegmenterModel
{
    /// <summary>Selfie segmenter (256×256): one person-probability mask. Fastest.</summary>
    Selfie,
    /// <summary>Selfie multiclass (256×256): background, hair, body-skin, face-skin, clothes, others.</summary>
    SelfieMulticlass,
    /// <summary>Hair segmenter (512×512): background, hair.</summary>
    Hair,
    /// <summary>DeepLab v3 (257×257): the 21 PASCAL VOC classes (person, cat, dog, car, ...).</summary>
    DeepLabV3,
}

/// <summary>Options of <see cref="ImageSegmenter"/>.</summary>
public sealed record ImageSegmenterOptions : VisionTaskOptions<SegmentationResult>
{
    /// <summary>The segmentation model. Default <see cref="SegmenterModel.Selfie"/>.</summary>
    public SegmenterModel Model { get; init; } = SegmenterModel.Selfie;

    /// <summary>Output one probability mask per category. Default true.</summary>
    public bool OutputConfidenceMasks { get; init; } = true;

    /// <summary>
    /// Output a category mask: the most likely category index per pixel (for single-mask models, 0 where
    /// the probability exceeds 0.5 and <see cref="CategoryMask.Unlabeled"/> elsewhere). Default false.
    /// </summary>
    public bool OutputCategoryMask { get; init; }

    /// <summary>
    /// Temporal smoothing in video/live-stream mode: how much of the previous confidence masks is blended
    /// into uncertain pixels (MediaPipe's segmentation smoothing; 0 = off). Default 0.3.
    /// </summary>
    public float TemporalSmoothing { get; init; } = 0.3f;
}

/// <summary>
/// Image segmentation: per-pixel category probabilities (confidence masks) and/or the most likely
/// category per pixel (category mask). Supports the selfie, selfie multiclass, hair and DeepLab v3
/// MediaPipe models.
/// </summary>
/// <example>
/// <code>
/// using var segmenter = ImageSegmenter.Create(new() { Model = SegmenterModel.SelfieMulticlass, OutputCategoryMask = true });
/// var result = segmenter.Segment(image);
/// Console.WriteLine($"Hair covers {result.GetConfidenceMask("hair")!.Coverage():P0} of the image");
/// </code>
/// </example>
public sealed class ImageSegmenter : VisionTaskBase<SegmentationResult>
{
    private readonly OnnxModel _model;
    private readonly SegmenterSpec _spec;
    private readonly SegmentationSmoother _smoother;

    private ImageSegmenter(ImageSegmenterOptions options, OnnxModel model)
        : base(nameof(ImageSegmenter), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        if (!options.OutputConfidenceMasks && !options.OutputCategoryMask)
            throw new ArgumentException("At least one of OutputConfidenceMasks and OutputCategoryMask must be enabled.", nameof(options));
        Options = options;
        _model = model;
        _spec = SegmenterSpec.For(options.Model);
        Labels = _spec.Labels;
        _smoother = new SegmentationSmoother(options.TemporalSmoothing);
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public ImageSegmenterOptions Options { get; }

    /// <summary>Category names, in confidence-mask / category-index order.</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>Creates the task (resolving the model synchronously).</summary>
    public static ImageSegmenter Create(ImageSegmenterOptions? options = null)
    {
        options ??= new ImageSegmenterOptions();
        return new ImageSegmenter(options, ModelLoader.Load(options.BaseOptions, SegmenterSpec.For(options.Model).Model));
    }

    /// <summary>Creates the task, downloading the model asynchronously when needed.</summary>
    public static async Task<ImageSegmenter> CreateAsync(ImageSegmenterOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new ImageSegmenterOptions();
        var model = await ModelLoader.LoadAsync(options.BaseOptions, SegmenterSpec.For(options.Model).Model, cancellationToken).ConfigureAwait(false);
        return new ImageSegmenter(options, model);
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
    public override void ResetTracking()
    {
        _smoother.Reset();
    }

    /// <inheritdoc />
    protected override SegmentationResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        var input = _model.Inputs[0];
        int th = input.Shape[1], tw = input.Shape[2], inChannels = input.Shape[3];
        int channels = _model.Outputs[0].Shape[^1];
        var logits = new float[tw * th * channels];
        TensorMapping mapping;
        using (var ctx = _model.RentContext())
        {
            var tensorOptions = new ImageToTensorOptions(tw, th, _spec.RangeMin, _spec.RangeMax, KeepAspectRatio: false, BorderMode.Replicate, Antialias: false);
            var roi = options?.ToRoi() ?? NormalizedRect.FullImage;
            if (inChannels == 3)
            {
                mapping = ImageToTensor.Convert(image, roi, tensorOptions, ctx.GetInput(0));
            }
            else
            {
                // RGBA models (hair segmentation) expect an empty alpha channel (MediaPipe's SetAlphaCalculator, alpha 0).
                var rgb = new float[tw * th * 3];
                mapping = ImageToTensor.Convert(image, roi, tensorOptions, rgb);
                var dst = ctx.GetInput(0);
                float alpha = _spec.RangeMin;
                for (int i = 0, n = tw * th; i < n; i++)
                {
                    dst[4 * i] = rgb[3 * i];
                    dst[4 * i + 1] = rgb[3 * i + 1];
                    dst[4 * i + 2] = rgb[3 * i + 2];
                    dst[4 * i + 3] = alpha;
                }
            }
            ctx.Run();
            ctx.GetOutput(0).CopyTo(logits);
        }

        // Activation and temporal smoothing at tensor resolution, then all channels projected in one pass.
        if (_spec.Softmax) Softmax(logits, channels);
        if (tracking && Options.TemporalSmoothing > 0) _smoother.Apply(logits);
        int w = image.Width, h = image.Height;
        var planes = new float[channels][];
        for (int c = 0; c < channels; c++) planes[c] = new float[w * h];
        TensorWarp.ProjectChannelsToImage(logits, tw, th, channels, mapping, planes, w, h);
        var masks = new SegmentationMask[channels];
        for (int c = 0; c < channels; c++) masks[c] = new SegmentationMask(w, h, planes[c]);
        var category = Options.OutputCategoryMask ? CategoryMask.FromConfidenceMasks(masks) : null;
        return new SegmentationResult(Options.OutputConfidenceMasks ? masks : [], category, Labels);
    }

    private static void Softmax(float[] values, int channels)
    {
        for (int i = 0; i < values.Length; i += channels)
        {
            var px = values.AsSpan(i, channels);
            float max = float.MinValue, sum = 0;
            foreach (var v in px) max = MathF.Max(max, v);
            for (int c = 0; c < channels; c++) sum += px[c] = MathF.Exp(px[c] - max);
            for (int c = 0; c < channels; c++) px[c] /= sum;
        }
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing) _model.Dispose();
    }

    private sealed record SegmenterSpec(ModelDescriptor Model, float RangeMin, float RangeMax, bool Softmax, IReadOnlyList<string> Labels)
    {
        public static SegmenterSpec For(SegmenterModel model) => model switch
        {
            SegmenterModel.Selfie => new(ModelCatalog.SelfieSegmenter, 0f, 1f, false, ["person"]),
            SegmenterModel.SelfieMulticlass => new(ModelCatalog.SelfieMulticlass, -1f, 1f, true, Processing.Labels.SelfieMulticlass),
            SegmenterModel.Hair => new(ModelCatalog.HairSegmenter, 0f, 1f, true, Processing.Labels.Hair),
            SegmenterModel.DeepLabV3 => new(ModelCatalog.DeepLabV3, -1f, 1f, true, Processing.Labels.PascalVoc),
            _ => throw new ArgumentOutOfRangeException(nameof(model)),
        };
    }
}
