using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>The user's hint of what to segment: a point on the object or a scribble over it.</summary>
public sealed record RegionOfInterest
{
    private RegionOfInterest(IReadOnlyList<NormalizedKeypoint> points, bool isScribble)
    {
        Points = points;
        IsScribble = isScribble;
    }

    /// <summary>The point(s), normalized to the image.</summary>
    public IReadOnlyList<NormalizedKeypoint> Points { get; }

    /// <summary>True for a scribble (connected points), false for a single keypoint.</summary>
    public bool IsScribble { get; }

    /// <summary>A single point on the object.</summary>
    public static RegionOfInterest FromKeypoint(float x, float y) => new([new NormalizedKeypoint(x, y)], false);

    /// <summary>A scribble over the object (at least one point).</summary>
    public static RegionOfInterest FromScribble(IReadOnlyList<NormalizedKeypoint> points)
    {
        ArgumentNullException.ThrowIfNull(points);
        if (points.Count == 0) throw new ArgumentException("A scribble needs at least one point.", nameof(points));
        return new(points.ToArray(), true);
    }
}

/// <summary>Options of <see cref="InteractiveSegmenter"/>.</summary>
public sealed record InteractiveSegmenterOptions
{
    /// <summary>Model and runtime options.</summary>
    public BaseOptions BaseOptions { get; init; } = BaseOptions.Default;

    /// <summary>Output the object's probability mask. Default true.</summary>
    public bool OutputConfidenceMasks { get; init; } = true;

    /// <summary>Output a category mask (0 = object, <see cref="CategoryMask.Unlabeled"/> = background). Default false.</summary>
    public bool OutputCategoryMask { get; init; }
}

/// <summary>
/// Interactive segmentation with MagicTouch: segments the object under a user-supplied point or scribble
/// ("tap to select"). Like MediaPipe's, it runs on still images only.
/// </summary>
/// <example>
/// <code>
/// using var segmenter = InteractiveSegmenter.Create();
/// var result = segmenter.Segment(image, RegionOfInterest.FromKeypoint(0.62f, 0.55f));
/// var cutout = result.ConfidenceMask; // probability that each pixel belongs to the selected object
/// </code>
/// </example>
public sealed class InteractiveSegmenter : VisionTaskBase<SegmentationResult>
{
    private const int ModelInputSize = 512;
    [ThreadStatic] private static RegionOfInterest? t_roi;
    private readonly OnnxModel _model;

    private InteractiveSegmenter(InteractiveSegmenterOptions options, OnnxModel model)
        : base(nameof(InteractiveSegmenter), RunningMode.Image, options.BaseOptions, null, 1)
    {
        if (!options.OutputConfidenceMasks && !options.OutputCategoryMask)
            throw new ArgumentException("At least one of OutputConfidenceMasks and OutputCategoryMask must be enabled.", nameof(options));
        Options = options;
        _model = model;
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public InteractiveSegmenterOptions Options { get; }

    /// <summary>Creates the task (resolving the model synchronously).</summary>
    public static InteractiveSegmenter Create(InteractiveSegmenterOptions? options = null)
    {
        options ??= new InteractiveSegmenterOptions();
        return new InteractiveSegmenter(options, ModelLoader.Load(options.BaseOptions, ModelCatalog.MagicTouch));
    }

    /// <summary>Creates the task, downloading the model asynchronously when needed.</summary>
    public static async Task<InteractiveSegmenter> CreateAsync(InteractiveSegmenterOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new InteractiveSegmenterOptions();
        return new InteractiveSegmenter(options, await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.MagicTouch, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Segments the object indicated by <paramref name="roi"/>.</summary>
    public SegmentationResult Segment(MPImage image, RegionOfInterest roi, ImageProcessingOptions? processingOptions = null)
    {
        ArgumentNullException.ThrowIfNull(roi);
        t_roi = roi;
        try
        {
            return RunImage(image, processingOptions);
        }
        finally
        {
            t_roi = null;
        }
    }

    /// <summary>Segments the object indicated by <paramref name="roi"/> on a worker thread.</summary>
    public Task<SegmentationResult> SegmentAsync(MPImage image, RegionOfInterest roi, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Segment(image, roi, processingOptions), cancellationToken);

    /// <inheritdoc />
    protected override SegmentationResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        var roi = t_roi ?? throw new InvalidOperationException("Call Segment(image, roi).");
        var input = _model.Inputs[0];
        int th = input.Shape[1], tw = input.Shape[2];
        var probs = new float[tw * th];
        TensorMapping mapping;
        using (var ctx = _model.RentContext())
        {
            var rgb = new float[tw * th * 3];
            mapping = ImageToTensor.Convert(image, options?.ToRoi() ?? NormalizedRect.FullImage,
                new ImageToTensorOptions(tw, th, 0f, 1f, KeepAspectRatio: false, BorderMode.Replicate, Antialias: false), rgb);
            var hint = RenderHint(roi, image.Width, image.Height, mapping);
            var dst = ctx.GetInput(0);
            for (int i = 0, n = tw * th; i < n; i++)
            {
                dst[4 * i] = rgb[3 * i];
                dst[4 * i + 1] = rgb[3 * i + 1];
                dst[4 * i + 2] = rgb[3 * i + 2];
                dst[4 * i + 3] = hint[i];
            }
            ctx.Run();
            ctx.GetOutput(0).CopyTo(probs);
        }
        var data = new float[image.Width * image.Height];
        TensorWarp.ProjectToImage(probs, tw, th, mapping, data, image.Width, image.Height);
        var mask = new SegmentationMask(image.Width, image.Height, data);
        return new SegmentationResult(Options.OutputConfidenceMasks ? [mask] : [],
            Options.OutputCategoryMask ? CategoryMask.FromConfidenceMasks([mask]) : null, ["object"]);
    }

    /// <summary>
    /// Renders the hint channel in tensor space: MediaPipe draws the point or scribble on an image-sized
    /// canvas with a thickness of max(width/512, height/512, 1) pixels, i.e. about one tensor pixel.
    /// </summary>
    private static float[] RenderHint(RegionOfInterest roi, int imageWidth, int imageHeight, TensorMapping mapping)
    {
        int tw = mapping.TensorWidth, th = mapping.TensorHeight;
        var hint = new float[tw * th];
        float thickness = MathF.Max(MathF.Max(imageWidth / (float)ModelInputSize, imageHeight / (float)ModelInputSize), 1f);
        float radius = MathF.Max(1f, thickness * MathF.Max((float)tw / imageWidth, (float)th / imageHeight));
        var pts = roi.Points.Select(p =>
        {
            var (u, v) = mapping.ImageToTensor(p.X, p.Y);
            return (X: u * tw, Y: v * th);
        }).ToArray();
        if (!roi.IsScribble || pts.Length == 1)
        {
            DrawSegment(hint, tw, th, pts[0], pts[0], radius);
        }
        else
        {
            for (int i = 1; i < pts.Length; i++) DrawSegment(hint, tw, th, pts[i - 1], pts[i], radius);
        }
        return hint;
    }

    private static void DrawSegment(float[] canvas, int w, int h, (float X, float Y) a, (float X, float Y) b, float radius)
    {
        int x0 = Math.Max(0, (int)MathF.Floor(MathF.Min(a.X, b.X) - radius - 1)), x1 = Math.Min(w - 1, (int)MathF.Ceiling(MathF.Max(a.X, b.X) + radius + 1));
        int y0 = Math.Max(0, (int)MathF.Floor(MathF.Min(a.Y, b.Y) - radius - 1)), y1 = Math.Min(h - 1, (int)MathF.Ceiling(MathF.Max(a.Y, b.Y) + radius + 1));
        float dx = b.X - a.X, dy = b.Y - a.Y, len2 = dx * dx + dy * dy;
        for (int y = y0; y <= y1; y++)
        {
            for (int x = x0; x <= x1; x++)
            {
                float px = x + 0.5f, py = y + 0.5f;
                float t = len2 > 0 ? Math.Clamp(((px - a.X) * dx + (py - a.Y) * dy) / len2, 0f, 1f) : 0f;
                float ex = px - (a.X + t * dx), ey = py - (a.Y + t * dy);
                if (ex * ex + ey * ey <= radius * radius) canvas[y * w + x] = 1f;
            }
        }
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing) _model.Dispose();
    }
}
