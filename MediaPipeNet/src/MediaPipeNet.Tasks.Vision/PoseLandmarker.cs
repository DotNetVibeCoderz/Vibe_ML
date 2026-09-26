using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>BlazePose landmark model variant.</summary>
public enum PoseModel
{
    /// <summary>Fastest (GHUM Lite, 5.5 MB).</summary>
    Lite,
    /// <summary>More accurate (GHUM Full, 12.8 MB).</summary>
    Full,
}

/// <summary>Options of <see cref="PoseLandmarker"/>.</summary>
public sealed record PoseLandmarkerOptions : VisionTaskOptions<PoseLandmarkResult>
{
    /// <summary>Landmark model variant. Default <see cref="PoseModel.Lite"/>.</summary>
    public PoseModel Model { get; init; } = PoseModel.Lite;

    /// <summary>Maximum number of people. Default 1.</summary>
    public int NumPoses { get; init; } = 1;

    /// <summary>Minimum person detector confidence. Default 0.5.</summary>
    public float MinPoseDetectionConfidence { get; init; } = 0.5f;

    /// <summary>Minimum pose presence score for a pose to be reported. Default 0.5.</summary>
    public float MinPosePresenceConfidence { get; init; } = 0.5f;

    /// <summary>Minimum presence to keep tracking without re-detection (video/live). Default 0.5.</summary>
    public float MinTrackingConfidence { get; init; } = 0.5f;

    /// <summary>Also output a person segmentation mask for the whole image. Default false.</summary>
    public bool OutputSegmentationMasks { get; init; }

    /// <summary>Smooth landmarks over time in video/live-stream mode (single pose). Default true.</summary>
    public bool SmoothLandmarks { get; init; } = true;
}

/// <summary>
/// Body pose estimation with BlazePose GHUM: 33 landmarks with visibility and presence, 33 world
/// landmarks in meters, and an optional person segmentation mask. In video/live-stream mode the
/// person is tracked from the previous frame's landmarks and the detector only runs when tracking is lost.
/// </summary>
public sealed class PoseLandmarker : VisionTaskBase<PoseLandmarkResult>
{
    /// <summary>Landmarks returned per pose.</summary>
    public const int LandmarkCount = 33;

    private const int ModelLandmarks = 39;
    private readonly SsdDetector _detector;
    private readonly OnnxModel _landmarks;
    private readonly int _screenOutput, _presenceOutput, _segmentationOutput, _worldOutput;
    private readonly List<NormalizedRect> _tracked = [];
    private LandmarkSmoother? _smoother;

    private PoseLandmarker(PoseLandmarkerOptions options, OnnxModel detector, OnnxModel landmarks)
        : base(nameof(PoseLandmarker), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _detector = new SsdDetector(detector, SsdDetectorSpec.Pose);
        _landmarks = landmarks;
        _screenOutput = landmarks.GetOutputIndex("Identity");
        _presenceOutput = landmarks.GetOutputIndex("Identity_1");
        _segmentationOutput = landmarks.GetOutputIndex("Identity_2");
        _worldOutput = landmarks.GetOutputIndex("Identity_4");
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public PoseLandmarkerOptions Options { get; }

    /// <summary>Creates the task (resolving models synchronously).</summary>
    public static PoseLandmarker Create(PoseLandmarkerOptions? options = null) =>
        CreateAsync(options).ConfigureAwait(false).GetAwaiter().GetResult();

    /// <summary>Creates the task, downloading models asynchronously when needed.</summary>
    public static async Task<PoseLandmarker> CreateAsync(PoseLandmarkerOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new PoseLandmarkerOptions();
        var det = await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.PoseDetection, cancellationToken).ConfigureAwait(false);
        var lm = await ModelLoader.LoadAsync(options.BaseOptions,
            options.Model == PoseModel.Full ? ModelCatalog.PoseLandmarksFull : ModelCatalog.PoseLandmarksLite, cancellationToken).ConfigureAwait(false);
        return new PoseLandmarker(options, det, lm);
    }

    /// <summary>Estimates poses in a still image.</summary>
    public PoseLandmarkResult Detect(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Estimates poses in a still image on a worker thread.</summary>
    public Task<PoseLandmarkResult> DetectAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Detect(image, processingOptions), cancellationToken);

    /// <summary>Estimates poses in a video frame, tracking across frames.</summary>
    public PoseLandmarkResult DetectForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunVideo(image, timestampMs, processingOptions);

    /// <summary>Submits a live-stream frame. Returns false if it was dropped.</summary>
    public bool DetectLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunLiveStream(image, timestampMs, processingOptions);

    /// <inheritdoc />
    public override void ResetTracking()
    {
        _tracked.Clear();
        _smoother?.Reset();
    }

    internal PoseLandmarkResult Compute(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs) =>
        Process(image, options, tracking, timestampMs);

    /// <inheritdoc />
    protected override PoseLandmarkResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        int w = image.Width, h = image.Height;
        var rois = new List<NormalizedRect>(tracking ? _tracked : []);
        if (rois.Count < Options.NumPoses)
        {
            var detections = _detector.Detect(image, options?.ToRoi() ?? NormalizedRect.FullImage, Options.MinPoseDetectionConfidence, Options.NumPoses);
            foreach (var d in detections)
            {
                if (rois.Count >= Options.NumPoses) break;
                var (x0, y0) = d.Keypoint(0);
                var (x1, y1) = d.Keypoint(1);
                var roi = RoiCalculator.Transform(RoiCalculator.FromAlignmentPoints(x0, y0, x1, y1, w, h, MathF.PI / 2), w, h, 1.25f, 1.25f, squareLong: true);
                if (rois.TrueForAll(r => RoiCalculator.Overlap(r, roi) < 0.5f)) rois.Add(roi);
            }
        }

        _tracked.Clear();
        var poses = new List<PoseLandmarks>(rois.Count);
        foreach (var roi in rois)
        {
            var (pose, next) = RunLandmarks(image, roi);
            if (pose.PresenceScore < Options.MinPosePresenceConfidence) continue;
            if (tracking && Options.SmoothLandmarks && Options.NumPoses == 1)
            {
                var arr = pose.Landmarks.ToArray();
                _smoother ??= new LandmarkSmoother(LandmarkCount);
                _smoother.Apply(arr, timestampMs, w, h, roi.Width * w);
                pose = pose with { Landmarks = arr };
            }
            poses.Add(pose);
            if (tracking && pose.PresenceScore >= Options.MinTrackingConfidence) _tracked.Add(next);
        }
        if (poses.Count == 0) _smoother?.Reset();
        return poses.Count == 0 ? PoseLandmarkResult.Empty : new PoseLandmarkResult(poses);
    }

    private (PoseLandmarks Pose, NormalizedRect NextRoi) RunLandmarks(MPImage image, in NormalizedRect roi)
    {
        int w = image.Width, h = image.Height;
        var spec = _landmarks.Inputs[0];
        int size = spec.Shape[2];
        using var ctx = _landmarks.RentContext();
        var mapping = ImageToTensor.Convert(image, roi, new ImageToTensorOptions(size, spec.Shape[1], 0f, 1f, KeepAspectRatio: true, BorderMode.Replicate), ctx.GetInput(0));
        ctx.Run();
        var raw = ctx.GetOutput(_screenOutput);
        var world = ctx.GetOutput(_worldOutput);
        float presence = ctx.GetOutput(_presenceOutput)[0]; // already a probability

        var all = new NormalizedLandmark[ModelLandmarks];
        for (int i = 0; i < ModelLandmarks; i++)
        {
            var (x, y) = mapping.TensorToImage(raw[5 * i] / size, raw[5 * i + 1] / size);
            all[i] = new NormalizedLandmark(x, y, mapping.ScaleZ(raw[5 * i + 2] / size),
                DetectionDecoder.Sigmoid(raw[5 * i + 3]), DetectionDecoder.Sigmoid(raw[5 * i + 4]));
        }
        var worldLandmarks = new Landmark[LandmarkCount];
        float cos = MathF.Cos(roi.Rotation), sin = MathF.Sin(roi.Rotation);
        for (int i = 0; i < LandmarkCount; i++)
        {
            float wx = world[3 * i], wy = world[3 * i + 1];
            worldLandmarks[i] = new Landmark(cos * wx - sin * wy, sin * wx + cos * wy, world[3 * i + 2], all[i].Visibility, all[i].Presence);
        }

        SegmentationMask? mask = null;
        if (Options.OutputSegmentationMasks)
        {
            var seg = ctx.GetOutput(_segmentationOutput);
            var probs = new float[seg.Length];
            for (int i = 0; i < probs.Length; i++) probs[i] = DetectionDecoder.Sigmoid(seg[i]);
            var data = new float[w * h];
            TensorWarp.ProjectToImage(probs, size, spec.Shape[1], mapping, data, w, h);
            mask = new SegmentationMask(w, h, data);
        }

        // Auxiliary landmarks 33 (hip center) and 34 (full-body scale) give the next-frame ROI.
        var next = RoiCalculator.Transform(
            RoiCalculator.FromAlignmentPoints(all[33].X, all[33].Y, all[34].X, all[34].Y, w, h, MathF.PI / 2), w, h, 1.25f, 1.25f, squareLong: true);
        return (new PoseLandmarks(all[..LandmarkCount], worldLandmarks, presence, roi, mask), next);
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (!disposing) return;
        _detector.Dispose();
        _landmarks.Dispose();
    }
}
