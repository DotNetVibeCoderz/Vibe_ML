using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="FaceLandmarker"/>.</summary>
public sealed record FaceLandmarkerOptions : VisionTaskOptions<FaceLandmarkResult>
{
    /// <summary>Maximum number of faces. Default 1.</summary>
    public int NumFaces { get; init; } = 1;

    /// <summary>Minimum face detector confidence. Default 0.5.</summary>
    public float MinFaceDetectionConfidence { get; init; } = 0.5f;

    /// <summary>Minimum face presence score for a face to be reported. Default 0.5.</summary>
    public float MinFacePresenceConfidence { get; init; } = 0.5f;

    /// <summary>Minimum presence to keep tracking a face without re-detection (video/live). Default 0.5.</summary>
    public float MinTrackingConfidence { get; init; } = 0.5f;

    /// <summary>Also compute the 52 blendshape scores. Default false.</summary>
    public bool OutputFaceBlendshapes { get; init; }

    /// <summary>Smooth landmarks over time in video/live-stream mode (single face). Default true.</summary>
    public bool SmoothLandmarks { get; init; } = true;
}

/// <summary>
/// Face mesh: 478 3-D landmarks per face (468 mesh points plus 10 iris points) and optionally 52
/// ARKit-compatible blendshapes. Runs BlazeFace to find faces, then the Face Mesh V2 model on a
/// rotated crop; in video/live-stream mode faces are tracked from frame to frame and the detector
/// only runs when a face is lost.
/// </summary>
public sealed class FaceLandmarker : VisionTaskBase<FaceLandmarkResult>
{
    /// <summary>Number of landmarks produced per face.</summary>
    public const int LandmarkCount = 478;

    /// <summary>The 52 blendshape names, in model output order.</summary>
    public static IReadOnlyList<string> BlendshapeNames { get; } =
    [
        "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
        "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft",
        "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
        "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
        "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
        "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
        "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
        "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
    ];

    // Landmarks fed to the blendshape model (face_blendshapes_graph.cc).
    private static readonly int[] s_blendshapeLandmarks =
    [
        0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
        81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155,
        157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269,
        270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332,
        334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397,
        398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
    ];

    private readonly SsdDetector _detector;
    private readonly OnnxModel _landmarkModel;
    private readonly OnnxModel? _blendshapeModel;
    private readonly int _landmarksOutput;
    private readonly int _presenceOutput;
    private readonly List<NormalizedRect> _tracked = [];
    private LandmarkSmoother? _smoother;

    private FaceLandmarker(FaceLandmarkerOptions options, OnnxModel detector, OnnxModel landmarks, OnnxModel? blendshapes)
        : base(nameof(FaceLandmarker), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _detector = new SsdDetector(detector, SsdDetectorSpec.FaceShortRange);
        _landmarkModel = landmarks;
        _blendshapeModel = blendshapes;
        _landmarksOutput = FindOutput(landmarks, LandmarkCount * 3);
        _presenceOutput = landmarks.GetOutputIndex("Identity_1");
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public FaceLandmarkerOptions Options { get; }

    /// <summary>Creates the task (resolving models synchronously).</summary>
    public static FaceLandmarker Create(FaceLandmarkerOptions? options = null) =>
        CreateAsync(options).ConfigureAwait(false).GetAwaiter().GetResult();

    /// <summary>Creates the task, downloading models asynchronously when needed.</summary>
    public static async Task<FaceLandmarker> CreateAsync(FaceLandmarkerOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new FaceLandmarkerOptions();
        var b = options.BaseOptions;
        var det = await ModelLoader.LoadAsync(b, ModelCatalog.FaceDetectionShortRange, cancellationToken).ConfigureAwait(false);
        var lm = await ModelLoader.LoadAsync(b, ModelCatalog.FaceLandmarksDetector, cancellationToken).ConfigureAwait(false);
        var bs = options.OutputFaceBlendshapes
            ? await ModelLoader.LoadAsync(b, ModelCatalog.FaceBlendshapes, cancellationToken).ConfigureAwait(false)
            : null;
        return new FaceLandmarker(options, det, lm, bs);
    }

    /// <summary>Finds face landmarks in a still image.</summary>
    public FaceLandmarkResult Detect(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Finds face landmarks in a still image on a worker thread.</summary>
    public Task<FaceLandmarkResult> DetectAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Detect(image, processingOptions), cancellationToken);

    /// <summary>Finds face landmarks in a video frame, tracking faces across frames.</summary>
    public FaceLandmarkResult DetectForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
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

    /// <inheritdoc />
    protected override FaceLandmarkResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        int w = image.Width, h = image.Height;
        var rois = new List<NormalizedRect>(tracking ? _tracked : []);
        if (rois.Count < Options.NumFaces)
        {
            var detections = _detector.Detect(image, options?.ToRoi() ?? NormalizedRect.FullImage, Options.MinFaceDetectionConfidence, Options.NumFaces);
            foreach (var d in detections)
            {
                if (rois.Count >= Options.NumFaces) break;
                var roi = RoiCalculator.Transform(RoiCalculator.FromDetection(d, w, h, 0, 1, 0f), w, h, 1.5f, 1.5f, squareLong: true);
                if (rois.TrueForAll(r => RoiCalculator.Overlap(r, roi) < 0.5f)) rois.Add(roi);
            }
        }

        var faces = new List<FaceLandmarks>(rois.Count);
        _tracked.Clear();
        foreach (var roi in rois)
        {
            var (landmarks, presence) = RunLandmarks(image, roi);
            if (presence < Options.MinFacePresenceConfidence) continue;
            if (tracking && Options.SmoothLandmarks && Options.NumFaces == 1)
            {
                _smoother ??= new LandmarkSmoother(LandmarkCount);
                _smoother.Apply(landmarks, timestampMs, w, h, roi.Width * w);
            }
            var blendshapes = _blendshapeModel is null ? null : RunBlendshapes(landmarks, w, h);
            faces.Add(new FaceLandmarks(landmarks, blendshapes, presence, roi));
            if (tracking && presence >= Options.MinTrackingConfidence)
                _tracked.Add(RoiCalculator.Transform(RoiCalculator.FromLandmarkBounds(landmarks, w, h, 33, 263, 0f), w, h, 1.5f, 1.5f, squareLong: true));
        }
        if (faces.Count == 0) _smoother?.Reset();
        return faces.Count == 0 ? FaceLandmarkResult.Empty : new FaceLandmarkResult(faces);
    }

    internal FaceLandmarkResult Compute(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs) =>
        Process(image, options, tracking, timestampMs);

    /// <summary>Runs the face mesh on an externally supplied ROI (used by the holistic pipeline).</summary>
    internal FaceLandmarks? ComputeOnRoi(MPImage image, in NormalizedRect roi)
    {
        var (landmarks, presence) = RunLandmarks(image, roi);
        if (presence < Options.MinFacePresenceConfidence) return null;
        return new FaceLandmarks(landmarks, _blendshapeModel is null ? null : RunBlendshapes(landmarks, image.Width, image.Height), presence, roi);
    }

    private (NormalizedLandmark[] Landmarks, float Presence) RunLandmarks(MPImage image, in NormalizedRect roi)
    {
        var spec = _landmarkModel.Inputs[0];
        int size = spec.Shape[2];
        using var ctx = _landmarkModel.RentContext();
        var mapping = ImageToTensor.Convert(image, roi, new ImageToTensorOptions(size, spec.Shape[1], 0f, 1f, KeepAspectRatio: true, BorderMode.Replicate), ctx.GetInput(0));
        ctx.Run();
        var raw = ctx.GetOutput(_landmarksOutput);
        float presence = DetectionDecoder.Sigmoid(ctx.GetOutput(_presenceOutput)[0]);
        var result = new NormalizedLandmark[LandmarkCount];
        for (int i = 0; i < LandmarkCount; i++)
        {
            var (x, y) = mapping.TensorToImage(raw[3 * i] / size, raw[3 * i + 1] / size);
            result[i] = new NormalizedLandmark(x, y, mapping.ScaleZ(raw[3 * i + 2] / size));
        }
        return (result, presence);
    }

    private Category[] RunBlendshapes(NormalizedLandmark[] landmarks, int imageWidth, int imageHeight)
    {
        using var ctx = _blendshapeModel!.RentContext();
        var input = ctx.GetInput(0);
        for (int i = 0; i < s_blendshapeLandmarks.Length; i++)
        {
            var l = landmarks[s_blendshapeLandmarks[i]];
            input[2 * i] = l.X * imageWidth;
            input[2 * i + 1] = l.Y * imageHeight;
        }
        ctx.Run();
        var scores = ctx.GetOutput(0);
        var result = new Category[BlendshapeNames.Count];
        for (int i = 0; i < result.Length; i++) result[i] = new Category(i, scores[i], BlendshapeNames[i]);
        return result;
    }

    private static int FindOutput(OnnxModel model, int elements)
    {
        for (int i = 0; i < model.Outputs.Count; i++)
            if (model.Outputs[i].ElementCount == elements) return i;
        throw new MediaPipeException($"Model '{model.Name}' has no output with {elements} values.");
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (!disposing) return;
        _detector.Dispose();
        _landmarkModel.Dispose();
        _blendshapeModel?.Dispose();
    }
}
