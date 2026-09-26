using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="GestureRecognizer"/>.</summary>
public sealed record GestureRecognizerOptions : VisionTaskOptions<GestureRecognitionResult>
{
    /// <summary>Hand tracking options (running mode and callback are taken from this object).</summary>
    public HandLandmarkerOptions Hands { get; init; } = new();

    /// <summary>Gestures scoring below this are reported as <c>None</c>. Default 0.</summary>
    public float MinGestureScore { get; init; }

    /// <summary>Maximum gestures listed per hand (-1 = all 8). Default 1.</summary>
    public int MaxResults { get; init; } = 1;

    /// <summary>If set, only these gesture names are reported.</summary>
    public IReadOnlySet<string>? CategoryAllowlist { get; init; }
}

/// <summary>
/// Recognizes hand gestures: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up,
/// Victory and ILoveYou. Hand landmarks are normalized (aspect ratio, rotation, object scale),
/// embedded with the gesture embedder and classified by the canned gesture classifier.
/// </summary>
public sealed class GestureRecognizer : VisionTaskBase<GestureRecognitionResult>
{
    private readonly HandLandmarker _hands;
    private readonly OnnxModel _embedder;
    private readonly OnnxModel _classifier;
    private readonly int _handInput, _handednessInput, _worldInput;

    private GestureRecognizer(GestureRecognizerOptions options, HandLandmarker hands, OnnxModel embedder, OnnxModel classifier)
        : base(nameof(GestureRecognizer), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _hands = hands;
        _embedder = embedder;
        _classifier = classifier;
        _handInput = embedder.GetInputIndex("hand");
        _handednessInput = embedder.GetInputIndex("handedness");
        _worldInput = embedder.GetInputIndex("world_hand");
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public GestureRecognizerOptions Options { get; }

    /// <summary>Creates the task (resolving models synchronously).</summary>
    public static GestureRecognizer Create(GestureRecognizerOptions? options = null) =>
        CreateAsync(options).ConfigureAwait(false).GetAwaiter().GetResult();

    /// <summary>Creates the task, downloading models asynchronously when needed.</summary>
    public static async Task<GestureRecognizer> CreateAsync(GestureRecognizerOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new GestureRecognizerOptions();
        var handOptions = options.Hands with { BaseOptions = options.BaseOptions, RunningMode = RunningMode.Image, ResultCallback = null };
        var hands = await HandLandmarker.CreateAsync(handOptions, cancellationToken).ConfigureAwait(false);
        var embedder = await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.GestureEmbedder, cancellationToken).ConfigureAwait(false);
        var classifier = await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.CannedGestureClassifier, cancellationToken).ConfigureAwait(false);
        return new GestureRecognizer(options, hands, embedder, classifier);
    }

    /// <summary>Recognizes gestures in a still image.</summary>
    public GestureRecognitionResult Recognize(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Recognizes gestures in a still image on a worker thread.</summary>
    public Task<GestureRecognitionResult> RecognizeAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Recognize(image, processingOptions), cancellationToken);

    /// <summary>Recognizes gestures in a video frame.</summary>
    public GestureRecognitionResult RecognizeForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunVideo(image, timestampMs, processingOptions);

    /// <summary>Submits a live-stream frame. Returns false if it was dropped.</summary>
    public bool RecognizeLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunLiveStream(image, timestampMs, processingOptions);

    /// <inheritdoc />
    public override void ResetTracking() => _hands.ResetTracking();

    /// <inheritdoc />
    protected override GestureRecognitionResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        var hands = _hands.Compute(image, options, tracking, timestampMs);
        if (hands.Hands.Count == 0) return GestureRecognitionResult.Empty;
        var result = new RecognizedHand[hands.Hands.Count];
        for (int i = 0; i < result.Length; i++)
            result[i] = new RecognizedHand(hands.Hands[i], Classify(hands.Hands[i], image.Width, image.Height));
        return new GestureRecognitionResult(result);
    }

    /// <summary>Classifies the gesture of already-computed hand landmarks.</summary>
    public IReadOnlyList<Category> Classify(HandLandmarks hand, int imageWidth, int imageHeight)
    {
        ArgumentNullException.ThrowIfNull(hand);
        Span<float> embedding = stackalloc float[128];
        using (var ctx = _embedder.RentContext())
        {
            WriteHandMatrix(hand, imageWidth, imageHeight, ctx.GetInput(_handInput));
            WriteWorldMatrix(hand, ctx.GetInput(_worldInput));
            ctx.GetInput(_handednessInput)[0] = hand.Handedness.CategoryName == "Right" ? hand.Handedness.Score : 1f - hand.Handedness.Score;
            ctx.Run();
            ctx.GetOutput(0).CopyTo(embedding);
        }

        var labels = Labels.Gestures;
        var categories = new List<Category>(labels.Count);
        using (var ctx = _classifier.RentContext())
        {
            embedding.CopyTo(ctx.GetInput(0));
            ctx.Run();
            var scores = ctx.GetOutput(0);
            for (int i = 0; i < labels.Count; i++)
            {
                if (Options.CategoryAllowlist is { } allow && !allow.Contains(labels[i])) continue;
                categories.Add(new Category(i, scores[i], labels[i]));
            }
        }
        categories.Sort((a, b) => b.Score.CompareTo(a.Score));
        if (categories.Count == 0 || categories[0].Score < Options.MinGestureScore)
            return [new Category(0, categories.Count > 0 ? categories[0].Score : 0f, labels[0])];
        return Options.MaxResults > 0 ? categories.Take(Options.MaxResults).ToArray() : categories;
    }

    // MediaPipe's LandmarksToMatrixCalculator: aspect-ratio normalization and object normalization around the wrist.
    // The gesture graph passes the image rect (rotation 0), so landmarks stay in image orientation —
    // verified against MediaPipe's own scores on the fixture images.
    private static void WriteHandMatrix(HandLandmarks hand, int w, int h, Span<float> dst)
    {
        float maxDim = Math.Max(w, h), sx = w / maxDim, sy = h / maxDim;
        Span<float> xs = stackalloc float[HandLandmarker.LandmarkCount];
        Span<float> ys = stackalloc float[HandLandmarker.LandmarkCount];
        Span<float> zs = stackalloc float[HandLandmarker.LandmarkCount];
        float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, maxY = float.MinValue;
        for (int i = 0; i < xs.Length; i++)
        {
            var l = hand.Landmarks[i];
            float x = (l.X - 0.5f) * sx, y = (l.Y - 0.5f) * sy;
            xs[i] = x + 0.5f;
            ys[i] = y + 0.5f;
            zs[i] = l.Z * sx;
            minX = MathF.Min(minX, xs[i]); maxX = MathF.Max(maxX, xs[i]);
            minY = MathF.Min(minY, ys[i]); maxY = MathF.Max(maxY, ys[i]);
        }
        float scale = MathF.Max(maxX - minX, maxY - minY);
        if (scale <= 0) scale = 1;
        for (int i = 0; i < xs.Length; i++)
        {
            dst[3 * i] = (xs[i] - xs[0]) / scale;
            dst[3 * i + 1] = (ys[i] - ys[0]) / scale;
            dst[3 * i + 2] = (zs[i] - zs[0]) / scale;
        }
    }

    private static void WriteWorldMatrix(HandLandmarks hand, Span<float> dst)
    {
        for (int i = 0; i < HandLandmarker.LandmarkCount; i++)
        {
            var l = hand.WorldLandmarks[i];
            dst[3 * i] = l.X;
            dst[3 * i + 1] = l.Y;
            dst[3 * i + 2] = l.Z;
        }
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (!disposing) return;
        _hands.Dispose();
        _embedder.Dispose();
        _classifier.Dispose();
    }
}
