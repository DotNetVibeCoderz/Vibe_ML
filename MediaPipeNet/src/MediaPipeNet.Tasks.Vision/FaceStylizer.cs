using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using System.Text.Json.Serialization;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Serialization;
using SixLabors.ImageSharp.PixelFormats;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="FaceStylizer"/>.</summary>
public sealed record FaceStylizerOptions : VisionTaskOptions<FaceStylizerResult>
{
    /// <summary>Stylization model. Default: <see cref="ModelCatalog.FaceStylizerColorSketch"/>.</summary>
    public ModelDescriptor? Model { get; init; }

    /// <summary>Minimum face detection confidence. Default 0.5.</summary>
    public float MinFaceDetectionConfidence { get; init; } = 0.5f;

    /// <summary>Minimum face presence confidence of the face mesh. Default 0.5.</summary>
    public float MinFacePresenceConfidence { get; init; } = 0.5f;

    /// <summary>Also return the aligned face crop that was fed to the model. Default false.</summary>
    public bool OutputFaceAlignment { get; init; }
}

/// <summary>Result of <see cref="FaceStylizer"/>. Dispose it to return the images' pixel buffers.</summary>
/// <param name="StylizedImage">The stylized face (the model's resolution, 256×256), or null when no face was found.</param>
/// <param name="FaceAlignment">The aligned face crop the model saw, when <see cref="FaceStylizerOptions.OutputFaceAlignment"/> is set.</param>
/// <param name="FaceRect">The rotated face rectangle that was cropped, in normalized image coordinates.</param>
/// <param name="Face">The face landmarks the alignment was computed from.</param>
public sealed record FaceStylizerResult(
    [property: JsonIgnore] MPImage? StylizedImage,
    [property: JsonIgnore] MPImage? FaceAlignment,
    NormalizedRect? FaceRect,
    FaceLandmarks? Face) : IDisposable
{
    /// <summary>A result with no face.</summary>
    public static FaceStylizerResult Empty => new(null, null, null, null);

    /// <summary>True when a face was found and stylized.</summary>
    public bool HasFace => StylizedImage is not null;

    /// <summary>Serializes the face rectangle and landmarks to JSON (the images are left out).</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);

    /// <summary>
    /// Pastes the stylized face back into (a copy of) the original image, undoing the alignment crop and
    /// feathering the square's border over <paramref name="feather"/> of its size. Returns null when no face was found.
    /// </summary>
    public MPImage? Composite(MPImage original, float feather = 0.08f)
    {
        ArgumentNullException.ThrowIfNull(original);
        if (StylizedImage is not { } styled || FaceRect is not { } rect) return null;
        var output = original.Clone();
        var dst = output.GetPixelSpan();
        int w = original.Width, h = original.Height;
        float cx = rect.XCenter * w, cy = rect.YCenter * h, rw = rect.Width * w, rh = rect.Height * h;
        float cos = MathF.Cos(rect.Rotation), sin = MathF.Sin(rect.Rotation);
        float reach = 0.5f * MathF.Sqrt(rw * rw + rh * rh);
        int x0 = Math.Max(0, (int)(cx - reach)), x1 = Math.Min(w - 1, (int)(cx + reach) + 1);
        int y0 = Math.Max(0, (int)(cy - reach)), y1 = Math.Min(h - 1, (int)(cy + reach) + 1);
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
            {
                float dx = x + 0.5f - cx, dy = y + 0.5f - cy;
                float u = (dx * cos + dy * sin) / rw + 0.5f, v = (-dx * sin + dy * cos) / rh + 0.5f;
                if (u < 0 || u >= 1 || v < 0 || v >= 1) continue;
                float edge = MathF.Min(MathF.Min(u, 1 - u), MathF.Min(v, 1 - v));
                float alpha = feather > 0 ? Math.Clamp(edge / feather, 0f, 1f) : 1f;
                var s = Sample(styled, u * styled.Width - 0.5f, v * styled.Height - 0.5f);
                ref var d = ref dst[y * w + x];
                d = new Rgba32(
                    (byte)(d.R + (s.X - d.R) * alpha + 0.5f),
                    (byte)(d.G + (s.Y - d.G) * alpha + 0.5f),
                    (byte)(d.B + (s.Z - d.B) * alpha + 0.5f),
                    d.A);
            }
        return output;
    }

    private static System.Numerics.Vector3 Sample(MPImage image, float x, float y)
    {
        int ix = (int)MathF.Floor(x), iy = (int)MathF.Floor(y);
        float fx = x - ix, fy = y - iy;
        System.Numerics.Vector3 P(int px, int py)
        {
            var p = image[Math.Clamp(px, 0, image.Width - 1), Math.Clamp(py, 0, image.Height - 1)];
            return new(p.R, p.G, p.B);
        }
        var top = System.Numerics.Vector3.Lerp(P(ix, iy), P(ix + 1, iy), fx);
        var bottom = System.Numerics.Vector3.Lerp(P(ix, iy + 1), P(ix + 1, iy + 1), fx);
        return System.Numerics.Vector3.Lerp(top, bottom, fy);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        StylizedImage?.Dispose();
        FaceAlignment?.Dispose();
    }
}

/// <summary>
/// Face stylization (MediaPipe's <c>FaceStylizer</c>): finds the most prominent face with the face landmarker,
/// aligns it exactly as MediaPipe does (eyes and mouth → rotated square crop) and runs a generator that
/// redraws it in a style, e.g. a color sketch.
/// </summary>
/// <remarks>
/// The generator injects Gaussian noise, like the original model, so two runs on the same image differ slightly
/// (as they do in MediaPipe).
/// </remarks>
/// <example>
/// <code>
/// using var stylizer = FaceStylizer.Create();
/// using var result = stylizer.Stylize(MPImage.Load("portrait.jpg"));
/// result.StylizedImage?.SaveAsPng("sketch.png");
/// </code>
/// </example>
public sealed class FaceStylizer : VisionTaskBase<FaceStylizerResult>
{
    // face_stylizer_graph.cc: LandmarksToDetectionCalculator selects left eye (33, 133), right eye (263, 362) and mouth (61, 291).
    private const int LeftEyeA = 33, LeftEyeB = 133, RightEyeA = 263, RightEyeB = 362, MouthA = 61, MouthB = 291;

    private readonly FaceLandmarker _landmarker;
    private readonly OnnxModel _model;

    private FaceStylizer(FaceStylizerOptions options, FaceLandmarker landmarker, OnnxModel model)
        : base(nameof(FaceStylizer), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _landmarker = landmarker;
        _model = model;
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public FaceStylizerOptions Options { get; }

    /// <summary>Creates the task (resolving models synchronously).</summary>
    public static FaceStylizer Create(FaceStylizerOptions? options = null) =>
        CreateAsync(options).ConfigureAwait(false).GetAwaiter().GetResult();

    /// <summary>Creates the task, downloading models asynchronously when needed.</summary>
    public static async Task<FaceStylizer> CreateAsync(FaceStylizerOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new FaceStylizerOptions();
        var landmarker = FaceLandmarker.CreateAsync(new FaceLandmarkerOptions
        {
            BaseOptions = options.BaseOptions, NumFaces = 1, SmoothLandmarks = false, LandmarksModel = ModelCatalog.FaceLandmarksDetector192,
            MinFaceDetectionConfidence = options.MinFaceDetectionConfidence, MinFacePresenceConfidence = options.MinFacePresenceConfidence,
        }, cancellationToken);
        var model = ModelLoader.LoadAsync(options.BaseOptions, options.Model ?? ModelCatalog.FaceStylizerColorSketch, cancellationToken).AsTask();
        await Task.WhenAll(landmarker, model).ConfigureAwait(false);
        return new FaceStylizer(options, landmarker.Result, model.Result);
    }

    /// <summary>Stylizes the most prominent face of a still image.</summary>
    public FaceStylizerResult Stylize(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Stylizes the most prominent face of a still image on a worker thread.</summary>
    public Task<FaceStylizerResult> StylizeAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Stylize(image, processingOptions), cancellationToken);

    /// <summary>Stylizes the most prominent face of a video frame.</summary>
    public FaceStylizerResult StylizeForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunVideo(image, timestampMs, processingOptions);

    /// <summary>Submits a live-stream frame. Returns false if it was dropped.</summary>
    public bool StylizeLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunLiveStream(image, timestampMs, processingOptions);

    /// <summary>
    /// The face crop MediaPipe's <c>FaceToRectCalculator</c> derives from face landmarks: centred between the eyes,
    /// shifted 10 % towards the mouth, sized max(3.6 × eye-to-mouth, 4 × eye-to-eye) and rotated upright.
    /// </summary>
    public static NormalizedRect ComputeFaceRect(IReadOnlyList<NormalizedLandmark> landmarks, int imageWidth, int imageHeight)
    {
        ArgumentNullException.ThrowIfNull(landmarks);
        (float X, float Y) P(int a, int b) =>
            ((landmarks[a].X + landmarks[b].X) * 0.5f * imageWidth, (landmarks[a].Y + landmarks[b].Y) * 0.5f * imageHeight);
        var left = P(LeftEyeA, LeftEyeB);
        var right = P(RightEyeA, RightEyeB);
        var mouth = P(MouthA, MouthB);
        float ecx = (left.X + right.X) * 0.5f, ecy = (left.Y + right.Y) * 0.5f;
        float eeX = right.X - left.X, eeY = right.Y - left.Y;
        float emX = mouth.X - ecx, emY = mouth.Y - ecy;
        float cx = MathF.Round(ecx + emX * 0.1f), cy = MathF.Round(ecy + emY * 0.1f);
        float size = MathF.Round(MathF.Max(MathF.Sqrt(emX * emX + emY * emY) * 3.6f, MathF.Sqrt(eeX * eeX + eeY * eeY) * 4.0f));
        float dirX = eeX + emY, dirY = eeY - emX;
        float rotation = Angles.NormalizeRadians(MathF.Atan2(dirY, dirX));
        return new NormalizedRect(cx / imageWidth, cy / imageHeight, size / imageWidth, size / imageHeight, rotation);
    }

    /// <inheritdoc />
    protected override FaceStylizerResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        var faces = _landmarker.Compute(image, options, tracking, timestampMs);
        if (faces.Faces.Count == 0) return FaceStylizerResult.Empty;
        var face = faces.Faces[0];
        var rect = ComputeFaceRect(face.Landmarks, image.Width, image.Height);

        var spec = _model.Inputs[0];
        int width = spec.Shape[2], height = spec.Shape[1];
        using var ctx = _model.RentContext();
        var input = ctx.GetInput(0);
        ImageToTensor.Convert(image, rect, new ImageToTensorOptions(width, height, -1f, 1f, KeepAspectRatio: true, BorderMode.Zero, Antialias: false), input);
        var alignment = Options.OutputFaceAlignment ? ToImage(input, width, height, -1f, 1f) : null;
        ctx.Run();
        var output = ctx.GetOutput(0);
        var outShape = _model.Outputs[0].Shape;
        return new FaceStylizerResult(ToImage(output, outShape[2], outShape[1], 0f, 1f), alignment, rect, face);
    }

    // TensorsToImageCalculator: maps [min, max] floats to 8-bit RGB, clamping out-of-range values.
    private static MPImage ToImage(ReadOnlySpan<float> tensor, int width, int height, float min, float max)
    {
        var image = MPImage.Create(width, height);
        var pixels = image.GetPixelSpan();
        float scale = 255f / (max - min);
        static byte Channel(float v) => (byte)Math.Clamp(MathF.Round(v), 0f, 255f);
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = new Rgba32(
                Channel((tensor[3 * i] - min) * scale),
                Channel((tensor[3 * i + 1] - min) * scale),
                Channel((tensor[3 * i + 2] - min) * scale),
                255);
        }
        return image;
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _landmarker.Dispose();
            _model.Dispose();
        }
    }
}
