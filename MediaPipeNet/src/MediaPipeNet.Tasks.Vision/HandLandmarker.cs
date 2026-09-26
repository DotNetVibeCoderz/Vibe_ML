using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="HandLandmarker"/>.</summary>
public sealed record HandLandmarkerOptions : VisionTaskOptions<HandLandmarkResult>
{
    /// <summary>Maximum number of hands. Default 2.</summary>
    public int NumHands { get; init; } = 2;

    /// <summary>Minimum palm detector confidence. Default 0.5.</summary>
    public float MinHandDetectionConfidence { get; init; } = 0.5f;

    /// <summary>Minimum hand presence score for a hand to be reported. Default 0.5.</summary>
    public float MinHandPresenceConfidence { get; init; } = 0.5f;

    /// <summary>Minimum presence to keep tracking a hand without re-detection (video/live). Default 0.5.</summary>
    public float MinTrackingConfidence { get; init; } = 0.5f;
}

/// <summary>
/// Hand tracking: 21 3-D landmarks, world landmarks and handedness per hand. Runs the BlazePalm
/// detector, derives a rotated hand ROI from the palm keypoints and runs the hand landmark model on it.
/// In video/live-stream mode hands are tracked from their previous landmarks and the palm detector
/// only runs while fewer than <see cref="HandLandmarkerOptions.NumHands"/> hands are tracked.
/// </summary>
public sealed class HandLandmarker : VisionTaskBase<HandLandmarkResult>
{
    /// <summary>Landmarks per hand.</summary>
    public const int LandmarkCount = 21;

    private readonly SsdDetector _palm;
    private readonly OnnxModel _landmarks;
    private readonly int _screenOutput, _presenceOutput, _handednessOutput, _worldOutput;
    private readonly List<NormalizedRect> _tracked = [];

    private HandLandmarker(HandLandmarkerOptions options, OnnxModel palm, OnnxModel landmarks)
        : base(nameof(HandLandmarker), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _palm = new SsdDetector(palm, SsdDetectorSpec.Palm);
        _landmarks = landmarks;
        _screenOutput = landmarks.GetOutputIndex("Identity");
        _presenceOutput = landmarks.GetOutputIndex("Identity_1");
        _handednessOutput = landmarks.GetOutputIndex("Identity_2");
        _worldOutput = landmarks.GetOutputIndex("Identity_3");
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public HandLandmarkerOptions Options { get; }

    /// <summary>Creates the task (resolving models synchronously).</summary>
    public static HandLandmarker Create(HandLandmarkerOptions? options = null) =>
        CreateAsync(options).ConfigureAwait(false).GetAwaiter().GetResult();

    /// <summary>Creates the task, downloading models asynchronously when needed.</summary>
    public static async Task<HandLandmarker> CreateAsync(HandLandmarkerOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new HandLandmarkerOptions();
        var palm = await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.PalmDetection, cancellationToken).ConfigureAwait(false);
        var lm = await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.HandLandmarksDetector, cancellationToken).ConfigureAwait(false);
        return new HandLandmarker(options, palm, lm);
    }

    /// <summary>Finds hands in a still image.</summary>
    public HandLandmarkResult Detect(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Finds hands in a still image on a worker thread.</summary>
    public Task<HandLandmarkResult> DetectAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Detect(image, processingOptions), cancellationToken);

    /// <summary>Finds hands in a video frame, tracking them across frames.</summary>
    public HandLandmarkResult DetectForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunVideo(image, timestampMs, processingOptions);

    /// <summary>Submits a live-stream frame. Returns false if it was dropped.</summary>
    public bool DetectLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunLiveStream(image, timestampMs, processingOptions);

    /// <inheritdoc />
    public override void ResetTracking() => _tracked.Clear();

    internal HandLandmarkResult Compute(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs) =>
        Process(image, options, tracking, timestampMs);

    /// <inheritdoc />
    protected override HandLandmarkResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        int w = image.Width, h = image.Height;
        var rois = new List<NormalizedRect>(tracking ? _tracked : []);
        if (rois.Count < Options.NumHands)
        {
            var palms = _palm.Detect(image, options?.ToRoi() ?? NormalizedRect.FullImage, Options.MinHandDetectionConfidence, Options.NumHands);
            foreach (var d in palms)
            {
                if (rois.Count >= Options.NumHands) break;
                var roi = PalmToRoi(d, w, h);
                if (rois.TrueForAll(r => RoiCalculator.Overlap(r, roi) < 0.5f)) rois.Add(roi);
            }
        }

        _tracked.Clear();
        var hands = new List<HandLandmarks>(rois.Count);
        foreach (var roi in rois)
        {
            var hand = RunLandmarks(image, roi);
            if (hand.PresenceScore < Options.MinHandPresenceConfidence) continue;
            // Two ROIs can converge on the same hand while tracking; keep the first.
            var next = RoiCalculator.FromHandLandmarks(hand.Landmarks, w, h);
            if (_tracked.Exists(t => RoiCalculator.Overlap(t, next) > 0.5f)) continue;
            hands.Add(hand);
            if (tracking && hand.PresenceScore >= Options.MinTrackingConfidence) _tracked.Add(next);
        }
        return hands.Count == 0 ? HandLandmarkResult.Empty : new HandLandmarkResult(hands);
    }

    /// <summary>Converts a palm detection to the rotated hand ROI (MediaPipe's palm_detection_detection_to_roi).</summary>
    internal static NormalizedRect PalmToRoi(in RawDetection palm, int w, int h) =>
        RoiCalculator.Transform(RoiCalculator.FromDetection(palm, w, h, 0, 2, MathF.PI / 2), w, h, 2.6f, 2.6f, 0f, -0.5f, squareLong: true);

    /// <summary>Runs the hand landmark model on an ROI.</summary>
    internal HandLandmarks RunLandmarks(MPImage image, in NormalizedRect roi)
    {
        var spec = _landmarks.Inputs[0];
        int size = spec.Shape[2];
        using var ctx = _landmarks.RentContext();
        var mapping = ImageToTensor.Convert(image, roi, new ImageToTensorOptions(size, spec.Shape[1], 0f, 1f, KeepAspectRatio: true, BorderMode.Replicate), ctx.GetInput(0));
        ctx.Run();
        var screen = ctx.GetOutput(_screenOutput);
        var world = ctx.GetOutput(_worldOutput);
        float presence = ctx.GetOutput(_presenceOutput)[0];
        float left = ctx.GetOutput(_handednessOutput)[0];

        var landmarks = new NormalizedLandmark[LandmarkCount];
        var worldLandmarks = new Landmark[LandmarkCount];
        float cos = MathF.Cos(roi.Rotation), sin = MathF.Sin(roi.Rotation);
        for (int i = 0; i < LandmarkCount; i++)
        {
            var (x, y) = mapping.TensorToImage(screen[3 * i] / size, screen[3 * i + 1] / size);
            landmarks[i] = new NormalizedLandmark(x, y, mapping.ScaleZ(screen[3 * i + 2] / size));
            float wx = world[3 * i], wy = world[3 * i + 1];
            worldLandmarks[i] = new Landmark(cos * wx - sin * wy, sin * wx + cos * wy, world[3 * i + 2]);
        }
        // The handedness output is the probability of a right hand (MediaPipe labels: 0 = Left, 1 = Right).
        var handedness = left > 0.5f ? new Category(1, left, "Right") : new Category(0, 1f - left, "Left");
        return new HandLandmarks(handedness, landmarks, worldLandmarks, presence, roi);
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (!disposing) return;
        _palm.Dispose();
        _landmarks.Dispose();
    }
}
