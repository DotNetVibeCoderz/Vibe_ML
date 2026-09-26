using MediaPipeNet.Imaging;
using MediaPipeNet.Serialization;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Visualization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace MediaPipeNet.Cli;

/// <summary>A task as seen by the CLI: run it, summarize and render its result.</summary>
internal interface ICliTask : IDisposable
{
    object Run(MPImage image);
    object RunVideo(MPImage image, long timestampMs);
    string Summarize(object result);
    void Render(Image<Rgba32> image, object result);
    string ToJson(object result) => MediaPipeJson.Serialize(result, indented: true);
}

internal sealed class CliTask<TTask, TResult>(
    TTask task,
    Func<TTask, MPImage, TResult> run,
    Func<TTask, MPImage, long, TResult> runVideo,
    Func<TResult, string> summarize,
    Action<Image<Rgba32>, TResult> render) : ICliTask where TTask : IDisposable
{
    public object Run(MPImage image) => run(task, image)!;
    public object RunVideo(MPImage image, long timestampMs) => runVideo(task, image, timestampMs)!;
    public string Summarize(object result) => summarize((TResult)result);
    public void Render(Image<Rgba32> image, object result) => render(image, (TResult)result);
    public void Dispose() => task.Dispose();
}

internal static class CliTaskFactory
{
    public static readonly IReadOnlyDictionary<string, string> Descriptions = new Dictionary<string, string>
    {
        ["faces"] = "Face detection (BlazeFace): boxes + 6 keypoints",
        ["face-mesh"] = "Face mesh: 478 landmarks + 52 blendshapes",
        ["hands"] = "Hand landmarks: 21 points per hand + handedness",
        ["gestures"] = "Gesture recognition (thumbs up, victory, ...)",
        ["pose"] = "Pose landmarks: 33 body points (+ mask)",
        ["holistic"] = "Pose + face + both hands",
        ["segment"] = "Selfie segmentation mask",
        ["objects"] = "Object detection (80 COCO classes)",
        ["classify"] = "Image classification (1000 ImageNet classes)",
    };

    public static ICliTask Create(string name, BaseOptions b, RunningMode mode) => name switch
    {
        "faces" => new CliTask<FaceDetector, FaceDetectionResult>(
            FaceDetector.Create(new() { BaseOptions = b, RunningMode = mode }), (t, i) => t.Detect(i), (t, i, ts) => t.DetectForVideo(i, ts),
            r => $"{r.Detections.Count} face(s): " + string.Join(", ", r.Detections.Select(d => $"{d.BoundingBox.Width:F0}x{d.BoundingBox.Height:F0}@({d.BoundingBox.X:F0},{d.BoundingBox.Y:F0}) {d.Score:P0}")),
            ResultRenderer.Render),
        "face-mesh" => new CliTask<FaceLandmarker, FaceLandmarkResult>(
            FaceLandmarker.Create(new() { BaseOptions = b, RunningMode = mode, OutputFaceBlendshapes = true }), (t, i) => t.Detect(i), (t, i, ts) => t.DetectForVideo(i, ts),
            r => $"{r.Faces.Count} face(s)" + string.Concat(r.Faces.Select(f => "; top blendshapes: " + string.Join(", ", f.Blendshapes!.Where(c => c.CategoryName != "_neutral").OrderByDescending(c => c.Score).Take(3)))),
            ResultRenderer.Render),
        "hands" => new CliTask<HandLandmarker, HandLandmarkResult>(
            HandLandmarker.Create(new() { BaseOptions = b, RunningMode = mode }), (t, i) => t.Detect(i), (t, i, ts) => t.DetectForVideo(i, ts),
            r => $"{r.Hands.Count} hand(s): " + string.Join(", ", r.Hands.Select(h => h.Handedness.ToString())),
            ResultRenderer.Render),
        "gestures" => new CliTask<GestureRecognizer, GestureRecognitionResult>(
            GestureRecognizer.Create(new() { BaseOptions = b, RunningMode = mode }), (t, i) => t.Recognize(i), (t, i, ts) => t.RecognizeForVideo(i, ts),
            r => $"{r.Hands.Count} hand(s): " + string.Join(", ", r.Hands.Select(h => h.TopGesture.ToString())),
            ResultRenderer.Render),
        "pose" => new CliTask<PoseLandmarker, PoseLandmarkResult>(
            PoseLandmarker.Create(new() { BaseOptions = b, RunningMode = mode, OutputSegmentationMasks = true }), (t, i) => t.Detect(i), (t, i, ts) => t.DetectForVideo(i, ts),
            r => $"{r.Poses.Count} pose(s)" + string.Concat(r.Poses.Select(p => $"; presence {p.PresenceScore:P0}, nose ({p[PoseLandmark.Nose].X:F3}, {p[PoseLandmark.Nose].Y:F3})")),
            ResultRenderer.Render),
        "holistic" => new CliTask<HolisticLandmarker, HolisticResult>(
            HolisticLandmarker.Create(new() { BaseOptions = b, RunningMode = mode }), (t, i) => t.Detect(i), (t, i, ts) => t.DetectForVideo(i, ts),
            r => $"pose: {r.Pose is not null}, face: {r.Face is not null}, left hand: {r.LeftHand is not null}, right hand: {r.RightHand is not null}",
            ResultRenderer.Render),
        "segment" => new CliTask<ImageSegmenter, SegmentationResult>(
            ImageSegmenter.Create(new() { BaseOptions = b, RunningMode = mode }), (t, i) => t.Segment(i), (t, i, ts) => t.SegmentForVideo(i, ts),
            r => $"foreground covers {r.ConfidenceMask.Coverage():P1} of the image",
            ResultRenderer.Render),
        "objects" => new CliTask<ObjectDetector, ObjectDetectionResult>(
            ObjectDetector.Create(new() { BaseOptions = b, RunningMode = mode }), (t, i) => t.Detect(i), (t, i, ts) => t.DetectForVideo(i, ts),
            r => $"{r.Detections.Count} object(s): " + string.Join(", ", r.Detections.Select(d => d.TopCategory.ToString())),
            ResultRenderer.Render),
        "classify" => new CliTask<ImageClassifier, ClassificationResult>(
            ImageClassifier.Create(new() { BaseOptions = b, RunningMode = mode }), (t, i) => t.Classify(i), (t, i, ts) => t.ClassifyForVideo(i, ts),
            r => string.Join(", ", r.Categories),
            (_, _) => { }),
        _ => throw new ArgumentException($"Unknown task '{name}'. Available: {string.Join(", ", Descriptions.Keys)}."),
    };
}
