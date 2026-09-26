using MediaPipeNet.Imaging;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Options of <see cref="HolisticLandmarker"/>.</summary>
public sealed record HolisticLandmarkerOptions : VisionTaskOptions<HolisticResult>
{
    /// <summary>Pose landmark model variant. Default <see cref="PoseModel.Lite"/>.</summary>
    public PoseModel PoseModel { get; init; } = PoseModel.Lite;

    /// <summary>Minimum detector confidence for the body, face and hands. Default 0.5.</summary>
    public float MinDetectionConfidence { get; init; } = 0.5f;

    /// <summary>Minimum presence score for each part. Default 0.5.</summary>
    public float MinPresenceConfidence { get; init; } = 0.5f;

    /// <summary>Also compute face blendshapes. Default false.</summary>
    public bool OutputFaceBlendshapes { get; init; }

    /// <summary>Also compute the person segmentation mask. Default false.</summary>
    public bool OutputSegmentationMask { get; init; }
}

/// <summary>
/// Holistic tracking of one person: 33 body landmarks, 478 face landmarks and 21 landmarks for each
/// hand. The pose, face and hand pipelines run in parallel; hands are assigned to the person's left
/// and right wrist from the pose.
/// </summary>
public sealed class HolisticLandmarker : VisionTaskBase<HolisticResult>
{
    private readonly PoseLandmarker _pose;
    private readonly FaceLandmarker _face;
    private readonly HandLandmarker _hands;

    private HolisticLandmarker(HolisticLandmarkerOptions options, PoseLandmarker pose, FaceLandmarker face, HandLandmarker hands)
        : base(nameof(HolisticLandmarker), options.RunningMode, options.BaseOptions, options.ResultCallback, options.MaxInFlightFrames)
    {
        Options = options;
        _pose = pose;
        _face = face;
        _hands = hands;
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
            BaseOptions = b, NumHands = 2,
            MinHandDetectionConfidence = options.MinDetectionConfidence, MinHandPresenceConfidence = options.MinPresenceConfidence,
        }, cancellationToken);
        await Task.WhenAll(pose, face, hands).ConfigureAwait(false);
        return new HolisticLandmarker(options, pose.Result, face.Result, hands.Result);
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
        _face.ResetTracking();
        _hands.ResetTracking();
    }

    /// <inheritdoc />
    protected override HolisticResult Process(MPImage image, ImageProcessingOptions? options, bool tracking, long timestampMs)
    {
        PoseLandmarkResult pose = PoseLandmarkResult.Empty;
        FaceLandmarkResult face = FaceLandmarkResult.Empty;
        HandLandmarkResult hands = HandLandmarkResult.Empty;
        Parallel.Invoke(
            () => pose = _pose.Compute(image, options, tracking, timestampMs),
            () => face = _face.Compute(image, options, tracking, timestampMs),
            () => hands = _hands.Compute(image, options, tracking, timestampMs));

        var body = pose.Poses.Count > 0 ? pose.Poses[0] : null;
        HandLandmarks? left = null, right = null;
        if (body is not null)
        {
            var lw = body[PoseLandmark.LeftWrist];
            var rw = body[PoseLandmark.RightWrist];
            float maxDist = MathF.Max(body.Roi.Width, body.Roi.Height) * 0.25f;
            foreach (var hand in hands.Hands.OrderByDescending(h => h.PresenceScore))
            {
                var wrist = hand.Landmarks[0];
                float dl = Distance(wrist, lw), dr = Distance(wrist, rw);
                if (dl <= dr && dl < maxDist && left is null) left = hand;
                else if (dr < maxDist && right is null) right = hand;
                else if (dl < maxDist && left is null) left = hand;
            }
        }
        else if (hands.Hands.Count > 0)
        {
            // Without a body, fall back to the model's handedness (which assumes a mirrored image).
            left = hands.Hands.FirstOrDefault(h => !h.IsLeft);
            right = hands.Hands.FirstOrDefault(h => h.IsLeft);
        }
        var faceResult = face.Faces.Count > 0 ? face.Faces[0] : null;
        if (faceResult is null && body is not null)
        {
            // Small or profile faces escape the short-range face detector: derive the ROI from the pose's
            // face landmarks (0-10), rotated so the eyes are level, as MediaPipe Holistic does.
            var faceLandmarks = body.Landmarks.Take(11).ToArray();
            var roi = Processing.RoiCalculator.FromLandmarkBounds(faceLandmarks, image.Width, image.Height, (int)PoseLandmark.RightEye, (int)PoseLandmark.LeftEye, 0f);
            roi = Processing.RoiCalculator.Transform(roi, image.Width, image.Height, 2.0f, 2.0f, squareLong: true);
            faceResult = _face.ComputeOnRoi(image, roi);
        }
        return new HolisticResult(body, faceResult, left, right);
    }

    private static float Distance(NormalizedLandmark a, NormalizedLandmark b) =>
        MathF.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (!disposing) return;
        _pose.Dispose();
        _face.Dispose();
        _hands.Dispose();
    }
}
