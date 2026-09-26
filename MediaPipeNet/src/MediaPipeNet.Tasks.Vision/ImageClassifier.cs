using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="ImageClassifier"/>.</summary>
public sealed record ImageClassifierOptions : VisionTaskOptions<ClassificationResult>
{
    /// <summary>Number of top categories returned. Default 5.</summary>
    public int MaxResults { get; init; } = 5;

    /// <summary>Minimum score for a category to be returned. Default 0.</summary>
    public float ScoreThreshold { get; init; }

    /// <summary>If set, only these labels are considered.</summary>
    public IReadOnlySet<string>? CategoryAllowlist { get; init; }

    /// <summary>If set, these labels are never returned.</summary>
    public IReadOnlySet<string>? CategoryDenylist { get; init; }
}

/// <summary>Classifies the whole image into the 1000 ImageNet classes with EfficientNet-Lite0.</summary>
public sealed class ImageClassifier : VisionTaskBase<ClassificationResult>
{
    private readonly OnnxModel _model;

    private ImageClassifier(ImageClassifierOptions options, OnnxModel model)
        : base(nameof(ImageClassifier), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _model = model;
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public ImageClassifierOptions Options { get; }

    /// <summary>Creates the task (resolving the model synchronously).</summary>
    public static ImageClassifier Create(ImageClassifierOptions? options = null)
    {
        options ??= new ImageClassifierOptions();
        return new ImageClassifier(options, ModelLoader.Load(options.BaseOptions, ModelCatalog.EfficientNetLite0));
    }

    /// <summary>Creates the task, downloading the model asynchronously when needed.</summary>
    public static async Task<ImageClassifier> CreateAsync(ImageClassifierOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new ImageClassifierOptions();
        return new ImageClassifier(options, await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.EfficientNetLite0, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Classifies a still image.</summary>
    public ClassificationResult Classify(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Classifies a still image on a worker thread.</summary>
    public Task<ClassificationResult> ClassifyAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Classify(image, processingOptions), cancellationToken);

    /// <summary>Classifies a video frame.</summary>
    public ClassificationResult ClassifyForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunVideo(image, timestampMs, processingOptions);

    /// <summary>Submits a live-stream frame. Returns false if it was dropped.</summary>
    public bool ClassifyLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunLiveStream(image, timestampMs, processingOptions);

    /// <inheritdoc />
    protected override ClassificationResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        var spec = _model.Inputs[0];
        var labels = Labels.ImageNet;
        var categories = new List<Category>();
        using (var ctx = _model.RentContext())
        {
            ImageToTensor.Convert(image, options?.ToRoi() ?? NormalizedRect.FullImage,
                new ImageToTensorOptions(spec.Shape[2], spec.Shape[1], -1f, 1f), ctx.GetInput(0));
            ctx.Run();
            var scores = ctx.GetOutput(0);
            for (int i = 0; i < scores.Length && i < labels.Count; i++)
            {
                if (scores[i] < Options.ScoreThreshold) continue;
                var label = labels[i];
                if (Options.CategoryAllowlist is { } allow && !allow.Contains(label)) continue;
                if (Options.CategoryDenylist is { } deny && deny.Contains(label)) continue;
                categories.Add(new Category(i, scores[i], label));
            }
        }
        categories.Sort((a, b) => b.Score.CompareTo(a.Score));
        if (Options.MaxResults > 0 && categories.Count > Options.MaxResults) categories.RemoveRange(Options.MaxResults, categories.Count - Options.MaxResults);
        return new ClassificationResult(categories);
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing) _model.Dispose();
    }
}
