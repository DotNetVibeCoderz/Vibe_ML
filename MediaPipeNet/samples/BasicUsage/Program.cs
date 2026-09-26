// MediaPipe.NET — BasicUsage sample
// Created by Gravicode Studios, led by Kang Fadhil.
//
// Runs every vision task on the bundled sample images, prints the results, and writes annotated
// images (plus JSON) to ./output. Models are copied next to the executable by the
// Gravicode.MediaPipeNet.Models.All package, so everything works offline.

using System.Diagnostics;
using MediaPipeNet;
using MediaPipeNet.Imaging;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Visualization;
using SixLabors.ImageSharp;

var images = Path.Combine(AppContext.BaseDirectory, "images");
var output = Directory.CreateDirectory(Path.Combine(AppContext.BaseDirectory, "output")).FullName;

Console.WriteLine("MediaPipe.NET — basic usage (Gravicode Studios / Kang Fadhil)\n");

// 1. Face detection -----------------------------------------------------------------------------
using (var detector = FaceDetector.Create())
using (var image = MPImage.Load(Path.Combine(images, "portrait.jpg")))
{
    var result = Timed("Face detection", () => detector.Detect(image));
    foreach (var face in result.Detections)
        Console.WriteLine($"   face at {face.BoundingBox.X:F0},{face.BoundingBox.Y:F0} size {face.BoundingBox.Width:F0}px, score {face.Score:P0}");
    Save(image, output, "faces.png", img => ResultRenderer.Render(img, result));
    File.WriteAllText(Path.Combine(output, "faces.json"), result.ToJson(indented: true));
}

// 2. Face mesh + blendshapes --------------------------------------------------------------------
using (var landmarker = FaceLandmarker.Create(new FaceLandmarkerOptions { OutputFaceBlendshapes = true }))
using (var image = MPImage.Load(Path.Combine(images, "portrait.jpg")))
{
    var result = Timed("Face mesh", () => landmarker.Detect(image));
    var face = result.Faces[0];
    Console.WriteLine($"   {face.Landmarks.Count} landmarks, smile {face.GetBlendshape("mouthSmileLeft"):P0}, jaw open {face.GetBlendshape("jawOpen"):P0}");
    Save(image, output, "face-mesh.png", img => ResultRenderer.Render(img, result));
}

// 3. Hands and gestures -------------------------------------------------------------------------
using (var gestures = GestureRecognizer.Create())
{
    foreach (var name in new[] { "thumb_up.jpg", "victory.jpg", "pointing_up.jpg" })
    {
        using var image = MPImage.Load(Path.Combine(images, name));
        var result = Timed($"Gestures ({name})", () => gestures.Recognize(image));
        foreach (var hand in result.Hands)
            Console.WriteLine($"   {hand.Hand.Handedness.CategoryName} hand: {hand.TopGesture}, index tip at ({hand.Hand[HandLandmark.IndexFingerTip].X:F2}, {hand.Hand[HandLandmark.IndexFingerTip].Y:F2})");
        Save(image, output, "gesture-" + Path.ChangeExtension(name, ".png"), img => ResultRenderer.Render(img, result));
    }
}

// 4. Pose with segmentation mask -----------------------------------------------------------------
using (var pose = PoseLandmarker.Create(new PoseLandmarkerOptions { OutputSegmentationMasks = true }))
using (var image = MPImage.Load(Path.Combine(images, "pose.jpg")))
{
    var result = Timed("Pose", () => pose.Detect(image));
    var p = result.Poses[0];
    Console.WriteLine($"   left wrist ({p[PoseLandmark.LeftWrist].X:F2}, {p[PoseLandmark.LeftWrist].Y:F2}), visibility {p[PoseLandmark.LeftWrist].Visibility:P0}");
    Console.WriteLine($"   person covers {p.SegmentationMask!.Coverage():P1} of the image");
    Save(image, output, "pose.png", img => ResultRenderer.Render(img, result));
}

// 5. Selfie segmentation: blur the background ----------------------------------------------------
using (var segmenter = ImageSegmenter.Create())
using (var image = MPImage.Load(Path.Combine(images, "portrait.jpg")))
{
    var result = Timed("Selfie segmentation", () => segmenter.Segment(image));
    Save(image, output, "portrait-blur.png", img => SegmentationMaskOverlay.BlurBackground(img, result.ConfidenceMask, 14));
}

// 6. Objects and classification ------------------------------------------------------------------
using (var objects = ObjectDetector.Create(new ObjectDetectorOptions { ScoreThreshold = 0.4f }))
using (var image = MPImage.Load(Path.Combine(images, "cats_and_dogs.jpg")))
{
    var result = Timed("Object detection", () => objects.Detect(image));
    Console.WriteLine("   " + string.Join(", ", result.Detections.Select(d => d.TopCategory)));
    Save(image, output, "objects.png", img => ResultRenderer.Render(img, result));
}

using (var classifier = ImageClassifier.Create(new ImageClassifierOptions { MaxResults = 3 }))
using (var image = MPImage.Load(Path.Combine(images, "burger.jpg")))
{
    var result = await Timed("Image classification (async)", () => classifier.ClassifyAsync(image));
    Console.WriteLine("   " + string.Join(", ", result.Categories));
}

// 7. Video mode: tracking across frames -----------------------------------------------------------
using (var hands = HandLandmarker.Create(new HandLandmarkerOptions { RunningMode = RunningMode.Video }))
using (var image = MPImage.Load(Path.Combine(images, "victory.jpg")))
{
    var sw = Stopwatch.StartNew();
    for (int frame = 0; frame < 30; frame++) hands.DetectForVideo(image, frame * 33L);
    Console.WriteLine($"\nVideo mode: 30 frames in {sw.ElapsedMilliseconds} ms ({30_000.0 / sw.ElapsedMilliseconds:F1} FPS) — the palm detector only runs on the first frame");
}

Console.WriteLine($"\nAnnotated images written to {output}");

static T Timed<T>(string label, Func<T> action)
{
    var sw = Stopwatch.StartNew();
    var result = action();
    if (result is Task task) task.GetAwaiter().GetResult();
    Console.WriteLine($"{label} — {sw.Elapsed.TotalMilliseconds:F1} ms");
    return result;
}

static void Save(MPImage image, string folder, string name, Action<SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>> draw)
{
    using var canvas = image.ToImage();
    draw(canvas);
    canvas.Save(Path.Combine(folder, name));
}
