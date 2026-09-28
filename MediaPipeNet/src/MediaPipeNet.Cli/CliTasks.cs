using MediaPipeNet.Imaging;
using MediaPipeNet.Serialization;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Visualization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using MediaPipeNet.Tasks;

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
        ["faces"] = "Face detection (BlazeFace short range): boxes + 6 keypoints",
        ["faces-full"] = "Face detection (BlazeFace full range, up to ~5 m)",
        ["face-mesh"] = "Face mesh: 478 landmarks + 52 blendshapes",
        ["hands"] = "Hand landmarks: 21 points per hand + handedness",
        ["gestures"] = "Gesture recognition (thumbs up, victory, ...)",
        ["pose"] = "Pose landmarks: 33 body points (+ mask)",
        ["holistic"] = "Pose + face + both hands",
        ["segment"] = "Selfie segmentation mask",
        ["segment-multiclass"] = "Hair / body skin / face skin / clothes / background",
        ["segment-hair"] = "Hair segmentation",
        ["segment-deeplab"] = "DeepLab v3: 21 PASCAL VOC classes",
        ["embed"] = "Image embedding (MobileNet V3, 1024-D)",
        ["stylize"] = "Face stylization (color sketch), pasted back onto the face",
        ["objects"] = "Object detection (80 COCO classes)",
        ["classify"] = "Image classification (1000 ImageNet classes)",
    };

    public static ICliTask Create(string name, BaseOptions b, RunningMode mode) => name switch
    {
        "faces" => new CliTask<FaceDetector, FaceDetectionResult>(
            FaceDetector.Create(new() { BaseOptions = b, RunningMode = mode }), (t, i) => t.Detect(i), (t, i, ts) => t.DetectForVideo(i, ts),
            r => $"{r.Detections.Count} face(s): " + string.Join(", ", r.Detections.Select(d => $"{d.BoundingBox.Width:F0}x{d.BoundingBox.Height:F0}@({d.BoundingBox.X:F0},{d.BoundingBox.Y:F0}) {d.Score:P0}")),
            ResultRenderer.Render),
        "faces-full" => new CliTask<FaceDetector, FaceDetectionResult>(
            FaceDetector.Create(new() { BaseOptions = b, RunningMode = mode, Model = FaceDetectorModel.FullRange }), (t, i) => t.Detect(i), (t, i, ts) => t.DetectForVideo(i, ts),
            r => $"{r.Detections.Count} face(s): " + string.Join(", ", r.Detections.Select(d => $"{d.BoundingBox.Width:F0}x{d.BoundingBox.Height:F0}@({d.BoundingBox.X:F0},{d.BoundingBox.Y:F0}) {d.Score:P0}")),
            ResultRenderer.Render),
        "segment-multiclass" or "segment-hair" or "segment-deeplab" => new CliTask<ImageSegmenter, SegmentationResult>(
            ImageSegmenter.Create(new()
            {
                BaseOptions = b, RunningMode = mode, OutputCategoryMask = true,
                Model = name switch { "segment-hair" => SegmenterModel.Hair, "segment-deeplab" => SegmenterModel.DeepLabV3, _ => SegmenterModel.SelfieMulticlass },
            }), (t, i) => t.Segment(i), (t, i, ts) => t.SegmentForVideo(i, ts),
            r => string.Join(", ", r.CategoryMask!.Histogram().Where(kv => kv.Value >= 0.005f)
                .Select(kv => $"{(kv.Key < r.Labels.Count ? r.Labels[kv.Key] : "unlabeled")} {kv.Value:P1}")),
            ResultRenderer.Render),
        "embed" => new CliTask<ImageEmbedder, ImageEmbeddingResult>(
            ImageEmbedder.Create(new() { BaseOptions = b, RunningMode = mode, L2Normalize = true }), (t, i) => t.Embed(i), (t, i, ts) => t.EmbedForVideo(i, ts),
            r => $"{r.Embedding.Dimension}-D embedding: [{string.Join(", ", r.Embedding.Values.Take(6).Select(v => v.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)))}, ...]",
            (_, _) => { }),
        "stylize" => new CliTask<FaceStylizer, FaceStylizerResult>(
            FaceStylizer.Create(new() { BaseOptions = b, RunningMode = mode }), (t, i) => t.Stylize(i), (t, i, ts) => t.StylizeForVideo(i, ts),
            r => r.FaceRect is { } f
                ? $"face at ({f.XCenter:F3}, {f.YCenter:F3}), size {f.Width:F3}, rotation {f.Rotation:F2} rad; stylized {r.StylizedImage!.Width}x{r.StylizedImage.Height}"
                : "no face",
            (canvas, r) =>
            {
                using var original = MPImage.FromImage(canvas);
                using var composite = r.Composite(original);
                if (composite is null) return;
                using var pasted = composite.ToImage();
                canvas.Mutate(c => c.DrawImage(pasted, 1f));
            }),
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
