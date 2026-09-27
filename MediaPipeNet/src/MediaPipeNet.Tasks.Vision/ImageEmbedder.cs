using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Serialization;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="ImageEmbedder"/>.</summary>
public sealed record ImageEmbedderOptions : VisionTaskOptions<ImageEmbeddingResult>
{
    /// <summary>L2-normalize the embedding (recommended for cosine similarity / nearest-neighbour search). Default false, as in MediaPipe.</summary>
    public bool L2Normalize { get; init; }

    /// <summary>Output a scalar-quantized (int8) embedding instead of floats. Default false.</summary>
    public bool Quantize { get; init; }
}

/// <summary>Result of <see cref="ImageEmbedder"/>.</summary>
/// <param name="Embeddings">One embedding per model head (MobileNet V3 has one, 1024-D).</param>
public sealed record ImageEmbeddingResult(IReadOnlyList<Embedding> Embeddings)
{
    /// <summary>The first (usually only) embedding.</summary>
    public Embedding Embedding => Embeddings[0];

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>
/// Image embedding with MobileNet V3 (small): a 1024-D feature vector per image for similarity search,
/// clustering or de-duplication. Compare vectors with <see cref="CosineSimilarity"/>.
/// </summary>
/// <example>
/// <code>
/// using var embedder = ImageEmbedder.Create(new() { L2Normalize = true });
/// var a = embedder.Embed(image1).Embedding;
/// var b = embedder.Embed(image2).Embedding;
/// Console.WriteLine($"similarity {ImageEmbedder.CosineSimilarity(a, b):F3}");
/// </code>
/// </example>
public sealed class ImageEmbedder : VisionTaskBase<ImageEmbeddingResult>
{
    private readonly OnnxModel _model;

    private ImageEmbedder(ImageEmbedderOptions options, OnnxModel model)
        : base(nameof(ImageEmbedder), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _model = model;
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public ImageEmbedderOptions Options { get; }

    /// <summary>Dimension of the produced embeddings.</summary>
    public int Dimension => _model.Outputs[0].Shape[^1];

    /// <summary>Creates the task (resolving the model synchronously).</summary>
    public static ImageEmbedder Create(ImageEmbedderOptions? options = null)
    {
        options ??= new ImageEmbedderOptions();
        return new ImageEmbedder(options, ModelLoader.Load(options.BaseOptions, ModelCatalog.MobileNetV3SmallEmbedder));
    }

    /// <summary>Creates the task, downloading the model asynchronously when needed.</summary>
    public static async Task<ImageEmbedder> CreateAsync(ImageEmbedderOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new ImageEmbedderOptions();
        return new ImageEmbedder(options, await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.MobileNetV3SmallEmbedder, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Cosine similarity of two embeddings, in [−1, 1].</summary>
    public static double CosineSimilarity(Embedding a, Embedding b) => Embedding.CosineSimilarity(a, b);

    /// <summary>Embeds a still image.</summary>
    public ImageEmbeddingResult Embed(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Embeds a still image on a worker thread.</summary>
    public Task<ImageEmbeddingResult> EmbedAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Embed(image, processingOptions), cancellationToken);

    /// <summary>Embeds a video frame.</summary>
    public ImageEmbeddingResult EmbedForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunVideo(image, timestampMs, processingOptions);

    /// <summary>Submits a live-stream frame. Returns false if it was dropped.</summary>
    public bool EmbedLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunLiveStream(image, timestampMs, processingOptions);

    /// <inheritdoc />
    protected override ImageEmbeddingResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        var spec = _model.Inputs[0];
        using var ctx = _model.RentContext();
        ImageToTensor.Convert(image, options?.ToRoi() ?? NormalizedRect.FullImage,
            new ImageToTensorOptions(spec.Shape[2], spec.Shape[1], 0f, 1f, KeepAspectRatio: false, BorderMode.Replicate, Antialias: false), ctx.GetInput(0));
        ctx.Run();
        return new ImageEmbeddingResult([Embedding.FromTensor(ctx.GetOutput(0), Options.L2Normalize, Options.Quantize, 0, "feature")]);
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing) _model.Dispose();
    }
}
