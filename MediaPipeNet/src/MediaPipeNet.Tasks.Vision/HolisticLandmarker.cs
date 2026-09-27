using MediaPipeNet.Imaging;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Tasks.Vision.Processing;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="HolisticLandmarker"/>.</summary>
public sealed record HolisticLandmarkerOptions : VisionTaskOptions<HolisticResult>
{
    /// <summary>Pose landmark model variant. Default <see cref="PoseModel.Lite"/>.</summary>
    public PoseModel PoseModel { get; init; } = PoseModel.Lite;

    /// <summary>Minimum detector confidence for the body and face. Default 0.5.</summary>
    public float MinDetectionConfidence { get; init; } = 0.5f;

    /// <summary>Minimum presence score for each part. Default 0.5.</summary>
    public float MinPresenceConfidence { get; init; } = 0.5f;

    /// <summary>Also compute face blendshapes. Default false.</summary>
    public bool OutputFaceBlendshapes { get; init; }

    /// <summary>Also compute the person segmentation mask. Default false.</summary>
    public bool OutputSegmentationMask { get; init; }
}

/// <summary>
/// Holistic tracking of one person — 33 body landmarks, 478 face landmarks and 21 landmarks for each
/// hand — following MediaPipe's holistic graph: the pose is found first; the face ROI comes from the
/// pose's face landmarks (refined by the face detector) and each hand ROI from the pose's wrist, index
/// and pinky landmarks (refined by the hand ROI refinement model). In video/live-stream mode every part
/// keeps its previous ROI while it agrees with the new one (MediaPipe's ROI tracking), and hand world
/// landmarks are aligned to the pose's world wrists. Unlike MediaPipe, a face too small for the face
/// detector is still tracked from the pose's face landmarks when the face mesh reports it present.
/// </summary>
public sealed class HolisticLandmarker : VisionTaskBase<HolisticResult>
{
    private const float PalmVisibilityThreshold = 0.1f;

    private readonly PoseLandmarker _pose;
    private readonly FaceLandmarker _face;
    private readonly HandLandmarker _hands;
    private readonly OnnxModel _handRoiRefinement;
    private readonly PartTracker _faceTracker = new(), _leftTracker = new(), _rightTracker = new();

    private HolisticLandmarker(HolisticLandmarkerOptions options, PoseLandmarker pose, FaceLandmarker face, HandLandmarker hands, OnnxModel handRoiRefinement)
        : base(nameof(HolisticLandmarker), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _pose = pose;
        _face = face;
        _hands = hands;
        _handRoiRefinement = handRoiRefinement;
        CompleteInitialization();
    }

    /// <summary>The options the task was created with.</summary>
    public HolisticLandmarkerOptions Options { get; }

    /// <summary>Creates the task (resolving models synchronously).</summary>
    public static HolisticLandmarker Create(HolisticLandmarkerOptions? options = null) =>
        CreateAsync(options).ConfigureAwait(false).GetAwaiter().GetResult();

    /// <summary>Creates the task, downloading models asynchronously when needed.</summary>
    public static async Task<HolisticLandmarker> CreateAsync(HolisticLandmarkerOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new HolisticLandmarkerOptions();
        var b = options.BaseOptions;
        var pose = PoseLandmarker.CreateAsync(new PoseLandmarkerOptions
        {
            BaseOptions = b, Model = options.PoseModel, OutputSegmentationMasks = options.OutputSegmentationMask,
            MinPoseDetectionConfidence = options.MinDetectionConfidence, MinPosePresenceConfidence = options.MinPresenceConfidence,
        }, cancellationToken);
        var face = FaceLandmarker.CreateAsync(new FaceLandmarkerOptions
        {
            BaseOptions = b, OutputFaceBlendshapes = options.OutputFaceBlendshapes,
            MinFaceDetectionConfidence = options.MinDetectionConfidence, MinFacePresenceConfidence = options.MinPresenceConfidence,
        }, cancellationToken);
        var hands = HandLandmarker.CreateAsync(new HandLandmarkerOptions
        {
            BaseOptions = b, NumHands = 2, MinHandPresenceConfidence = options.MinPresenceConfidence,
        }, cancellationToken);
        var refinement = ModelLoader.LoadAsync(b, ModelCatalog.HandRoiRefinement, cancellationToken).AsTask();
        await Task.WhenAll(pose, face, hands, refinement).ConfigureAwait(false);
        return new HolisticLandmarker(options, pose.Result, face.Result, hands.Result, refinement.Result);
    }

    /// <summary>Tracks a person in a still image.</summary>
    public HolisticResult Detect(MPImage image, ImageProcessingOptions? processingOptions = null) => RunImage(image, processingOptions);

    /// <summary>Tracks a person in a still image on a worker thread.</summary>
    public Task<HolisticResult> DetectAsync(MPImage image, ImageProcessingOptions? processingOptions = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Detect(image, processingOptions), cancellationToken);

    /// <summary>Tracks a person in a video frame.</summary>
    public HolisticResult DetectForVideo(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunVideo(image, timestampMs, processingOptions);

    /// <summary>Submits a live-stream frame. Returns false if it was dropped.</summary>
    public bool DetectLiveStream(MPImage image, long timestampMs, ImageProcessingOptions? processingOptions = null) =>
        RunLiveStream(image, timestampMs, processingOptions);

    /// <inheritdoc />
    public override void ResetTracking()
    {
        _pose.ResetTracking();
        _faceTracker.Reset();
        _leftTracker.Reset();
        _rightTracker.Reset();
    }

    /// <inheritdoc />
    protected override HolisticResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        var pose = _pose.Compute(image, options, tracking, timestampMs);
        if (pose.Poses.Count == 0)
        {
            if (tracking) ResetParts();
            return HolisticResult.Empty;
        }
        var body = pose.Poses[0];
        FaceLandmarks? face = null;
        HandLandmarks? left = null, right = null;
        Parallel.Invoke(
            () => face = TrackFace(image, body, tracking),
            () => left = TrackHand(image, body, PoseLandmark.LeftWrist, PoseLandmark.LeftPinky, PoseLandmark.LeftIndex, _leftTracker, tracking),
            () => right = TrackHand(image, body, PoseLandmark.RightWrist, PoseLandmark.RightPinky, PoseLandmark.RightIndex, _rightTracker, tracking));
        return new HolisticResult(body, face, left, right);
    }

    private void ResetParts()
    {
        _faceTracker.Reset();
        _leftTracker.Reset();
        _rightTracker.Reset();
    }

    private FaceLandmarks? TrackFace(MPImage image, PoseLandmarks body, bool tracking)
    {
        int w = image.Width, h = image.Height;
        // ROI from the pose's face landmarks (0-10): rotated right eye → left eye, scaled 3× and squared.
        var poseFace = body.Landmarks.Take(11).ToArray();
        var roiFromPose = RoiCalculator.Transform(
            RoiCalculator.FromLandmarkBounds(poseFace, w, h, (int)PoseLandmark.RightEye, (int)PoseLandmark.LeftEye, 0f), w, h, 3f, 3f, squareLong: true);
        var detections = _face.DetectFaces(image, roiFromPose, Options.MinDetectionConfidence);
        var poseRoi = RoiCalculator.Transform(
            RoiCalculator.FromLandmarkBounds(poseFace, w, h, (int)PoseLandmark.RightEye, (int)PoseLandmark.LeftEye, 0f), w, h, 2f, 2f, squareLong: true);
        var recrop = detections.Count > 0
            ? RoiCalculator.Transform(RoiCalculator.FromKeypointBounds(detections[0], w, h, 0, 1, 0f), w, h, 2f, 2f, squareLong: true)
            : poseRoi;
        var roi = tracking ? RoiTracking.Select(_faceTracker.Landmarks, _faceTracker.Rect, recrop, w, h, RoiTrackingRequirements.Face) : recrop;
        var face = _face.ComputeOnRoi(image, roi);
        // Small faces (a few dozen pixels) defeat the face detector: it misses them or places its keypoints
        // poorly. Unlike MediaPipe, fall back to the pose's face landmarks (2× their bounds), still gated by
        // the face mesh's presence score.
        if (face is null && roi != poseRoi) face = _face.ComputeOnRoi(image, poseRoi);
        if (!tracking) return face;
        if (face is null)
        {
            _faceTracker.Reset();
            return null;
        }
        // Next-frame ROI from the landmarks: bounds rotated along the eye corners (33 → 263), scaled 1.5×.
        _faceTracker.Landmarks = face.Landmarks;
        _faceTracker.Rect = RoiCalculator.Transform(RoiCalculator.FromLandmarkBounds(face.Landmarks, w, h, 33, 263, 0f), w, h, 1.5f, 1.5f);
        return face;
    }

    private HandLandmarks? TrackHand(MPImage image, PoseLandmarks body, PoseLandmark wristIndex, PoseLandmark pinkyIndex, PoseLandmark indexIndex,
        PartTracker tracker, bool tracking)
    {
        int w = image.Width, h = image.Height;
        var wrist = body[wristIndex];
        if ((wrist.Visibility ?? 1f) <= PalmVisibilityThreshold)
        {
            if (tracking) tracker.Reset();
            return null;
        }
        var roiFromPose = RoiCalculator.Transform(RoiCalculator.FromPosePalm(wrist, body[pinkyIndex], body[indexIndex], w, h), w, h, 2.7f, 2.7f, 0f, -0.1f, squareLong: true);
        var recrop = RefineHandRoi(image, roiFromPose);
        var roi = tracking ? RoiTracking.Select(tracker.Landmarks, tracker.Rect, recrop, w, h, RoiTrackingRequirements.Hand) : recrop;
        var hand = _hands.RunLandmarks(image, roi);
        if (hand.PresenceScore < Options.MinPresenceConfidence)
        {
            if (tracking) tracker.Reset();
            return null;
        }
        // Align the hand's world landmarks to the pose's world wrist.
        var poseWrist = body.WorldLandmarks[(int)wristIndex];
        var handWrist = hand.WorldLandmarks[0];
        var world = hand.WorldLandmarks.Select(l => l with
        {
            X = l.X - handWrist.X + poseWrist.X,
            Y = l.Y - handWrist.Y + poseWrist.Y,
            Z = l.Z - handWrist.Z + poseWrist.Z,
        }).ToArray();
        hand = hand with { WorldLandmarks = world };
        if (tracking)
        {
            tracker.Landmarks = hand.Landmarks;
            tracker.Rect = RoiCalculator.FromHandLandmarks(hand.Landmarks, w, h);
        }
        return hand;
    }

    /// <summary>
    /// MediaPipe's <c>HandRoiRefinementGraph</c>: the refinement model predicts two points (wrist and middle
    /// finger) inside the rough pose-derived crop; the refined ROI is centered on the first, twice their
    /// distance in size, rotated −90° along them, shifted up by 10% and squared.
    /// </summary>
    private NormalizedRect RefineHandRoi(MPImage image, in NormalizedRect roi)
    {
        var spec = _handRoiRefinement.Inputs[0];
        int th = spec.Shape[1], tw = spec.Shape[2];
        using var ctx = _handRoiRefinement.RentContext();
        var mapping = ImageToTensor.Convert(image, roi, new ImageToTensorOptions(tw, th, 0f, 1f, KeepAspectRatio: true, BorderMode.Replicate), ctx.GetInput(0));
        ctx.Run();
        var points = ctx.GetOutput(0);
        var (x0, y0) = mapping.TensorToImage(points[0] / tw, points[1] / th);
        var (x1, y1) = mapping.TensorToImage(points[2] / tw, points[3] / th);
        int w = image.Width, h = image.Height;
        var rect = RoiCalculator.FromAlignmentPoints(x0, y0, x1, y1, w, h, -MathF.PI / 2);
        return RoiCalculator.Transform(rect, w, h, 1f, 1f, 0f, -0.1f, squareLong: true);
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (!disposing) return;
        _pose.Dispose();
        _face.Dispose();
        _hands.Dispose();
        _handRoiRefinement.Dispose();
    }

    private sealed class PartTracker
    {
        public IReadOnlyList<NormalizedLandmark>? Landmarks { get; set; }

        public NormalizedRect? Rect { get; set; }

        public void Reset()
        {
            Landmarks = null;
            Rect = null;
        }
    }
}
