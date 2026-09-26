using System.Globalization;
using MediaPipeNet.Gallery.Controls;
using MediaPipeNet.Imaging;
using MediaPipeNet.Serialization;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Visualization;

namespace MediaPipeNet.Gallery.Services;

/// <summary>The nine vision tasks shown in the Gallery.</summary>
public static class TaskCatalog
{
    private static string P(float v) => v.ToString("P0", CultureInfo.CurrentCulture);
    private static string N(float v) => v.ToString("0.000", CultureInfo.InvariantCulture);
    private static string Json(object o) => MediaPipeJson.Serialize(o, indented: true);

    private static readonly OptionSpec Confidence = new("confidence", "Min detection confidence", "Confidence deteksi minimum", OptionKind.Slider, 0.5, 0.1, 0.95);

    public static IReadOnlyList<GalleryTask> All { get; } =
    [
        new GalleryTask
        {
            Id = "faces", Category = "detect",
            TitleEn = "Face detection", TitleId = "Deteksi wajah",
            BlurbEn = "BlazeFace finds faces and six keypoints — eyes, nose tip, mouth and ear tragions. Tuned for faces within about two meters.",
            BlurbId = "BlazeFace menemukan wajah beserta enam keypoint — mata, ujung hidung, mulut, dan tragus telinga. Optimal untuk wajah dalam jarak sekitar dua meter.",
            ModelLabel = "blaze_face_short_range · 128²",
            Samples = ["portrait.jpg", "pose.jpg", "victory.jpg"],
            Options = [Confidence, new("nms", "Merge overlap (IoU)", "Gabungkan tumpang-tindih (IoU)", OptionKind.Slider, 0.3, 0.1, 0.9)],
            Create = (o, mode, b) => FaceDetector.Create(new() { BaseOptions = b, RunningMode = mode, MinDetectionConfidence = o.F("confidence"), MinSuppressionThreshold = o.F("nms") }),
            Run = (t, img, ts, o) =>
            {
                var d = (FaceDetector)t;
                var r = ts is { } v ? d.DetectForVideo(img, v) : d.Detect(img);
                var overlay = new Overlay();
                var rows = new List<ResultRow>();
                int i = 1;
                foreach (var f in r.Detections)
                {
                    var box = f.BoundingBox.Scale(1f / img.Width, 1f / img.Height);
                    overlay.Boxes.Add(OverlayBox.FromRect(box, $"face {P(f.Score)}", Overlay.Amber));
                    foreach (var k in f.Keypoints) overlay.Points.Add(new OverlayPoint(k.X, k.Y, Overlay.Mint, 4));
                    rows.Add(new ResultRow($"Face {i++}", $"{f.BoundingBox.Width:0}×{f.BoundingBox.Height:0} px", f.Score));
                }
                return new GalleryOutput(overlay, rows, r.ToJson(true), $"{r.Detections.Count} face(s)");
            },
            Code = o => $$"""
                using MediaPipeNet.Imaging;
                using MediaPipeNet.Tasks.Vision;

                using var detector = FaceDetector.Create(new FaceDetectorOptions
                {
                    MinDetectionConfidence = {{o.Inv("confidence")}}f,
                    MinSuppressionThreshold = {{o.Inv("nms")}}f,
                });

                using var image = MPImage.Load("portrait.jpg");
                FaceDetectionResult result = detector.Detect(image);

                foreach (var face in result.Detections)
                {
                    Console.WriteLine($"{face.BoundingBox} score {face.Score:P0}");
                    var rightEye = face.Keypoints[(int)FaceKeypoint.RightEye];
                }
                """,
        },
        new GalleryTask
        {
            Id = "face-mesh", Category = "landmarks",
            TitleEn = "Face mesh", TitleId = "Face mesh",
            BlurbEn = "478 three-dimensional landmarks including both irises, plus 52 blendshape scores that describe the expression — ready to drive an avatar.",
            BlurbId = "478 landmark tiga dimensi termasuk kedua iris, ditambah 52 skor blendshape yang menggambarkan ekspresi — siap menggerakkan avatar.",
            ModelLabel = "face_landmarks_detector · 256² + blendshapes",
            Samples = ["portrait.jpg"],
            Options = [Confidence, new("mesh", "Show every mesh point", "Tampilkan semua titik mesh", OptionKind.Toggle, 1)],
            Create = (o, mode, b) => FaceLandmarker.Create(new() { BaseOptions = b, RunningMode = mode, OutputFaceBlendshapes = true, MinFaceDetectionConfidence = o.F("confidence") }),
            Run = (t, img, ts, o) =>
            {
                var d = (FaceLandmarker)t;
                var r = ts is { } v ? d.DetectForVideo(img, v) : d.Detect(img);
                var overlay = new Overlay();
                var rows = new List<ResultRow>();
                foreach (var f in r.Faces)
                {
                    overlay.AddSkeleton(f.Landmarks, Connections.FaceContours, Overlay.Mint, Overlay.Paper, 1.6, 1.2, points: o.B("mesh"));
                    overlay.AddRoi(f.Roi, img.Width, img.Height, Overlay.Cobalt);
                    foreach (var c in f.Blendshapes!.Where(c => c.CategoryName != "_neutral").OrderByDescending(c => c.Score).Take(12))
                        rows.Add(new ResultRow(c.CategoryName!, P(c.Score), c.Score));
                }
                return new GalleryOutput(overlay, rows, r.ToJson(true), $"{r.Faces.Count} face(s) · 478 landmarks");
            },
            Code = o => $$"""
                using var landmarker = FaceLandmarker.Create(new FaceLandmarkerOptions
                {
                    OutputFaceBlendshapes = true,
                    MinFaceDetectionConfidence = {{o.Inv("confidence")}}f,
                });

                using var image = MPImage.Load("portrait.jpg");
                var face = landmarker.Detect(image).Faces[0];

                NormalizedLandmark noseTip = face.Landmarks[1];
                float smile = face.GetBlendshape("mouthSmileLeft");
                float blink = face.GetBlendshape("eyeBlinkRight");
                Console.WriteLine($"smile {smile:P0}, blink {blink:P0}");
                """,
        },
        new GalleryTask
        {
            Id = "hands", Category = "landmarks",
            TitleEn = "Hand landmarks", TitleId = "Landmark tangan",
            BlurbEn = "21 landmarks per hand in image and world coordinates, with left/right handedness. The palm detector finds hands; the landmark model follows them.",
            BlurbId = "21 landmark per tangan dalam koordinat gambar dan dunia, lengkap dengan handedness kiri/kanan. Detektor telapak menemukan tangan; model landmark mengikutinya.",
            ModelLabel = "palm_detection 192² → hand_landmarks 224²",
            Samples = ["victory.jpg", "thumb_up.jpg", "pointing_up.jpg", "pose.jpg"],
            Options = [Confidence, new("hands", "Maximum hands", "Jumlah tangan maksimum", OptionKind.Slider, 2, 1, 4, 1, "0")],
            Create = (o, mode, b) => HandLandmarker.Create(new() { BaseOptions = b, RunningMode = mode, NumHands = o.I("hands"), MinHandDetectionConfidence = o.F("confidence") }),
            Run = (t, img, ts, o) =>
            {
                var d = (HandLandmarker)t;
                var r = ts is { } v ? d.DetectForVideo(img, v) : d.Detect(img);
                var overlay = new Overlay();
                var rows = new List<ResultRow>();
                foreach (var h in r.Hands)
                {
                    overlay.AddSkeleton(h.Landmarks, Connections.Hand, h.IsLeft ? Overlay.Coral : Overlay.Mint, Overlay.Paper, 2.6, 3.6);
                    overlay.AddRoi(h.Roi, img.Width, img.Height, Overlay.Cobalt);
                    rows.Add(new ResultRow(h.Handedness.CategoryName!, P(h.Handedness.Score), h.Handedness.Score));
                    var tip = h[HandLandmark.IndexFingerTip];
                    rows.Add(new ResultRow("  index tip", $"{N(tip.X)}, {N(tip.Y)}"));
                }
                return new GalleryOutput(overlay, rows, r.ToJson(true), $"{r.Hands.Count} hand(s)");
            },
            Code = o => $$"""
                using var hands = HandLandmarker.Create(new HandLandmarkerOptions
                {
                    NumHands = {{o.I("hands")}},
                    MinHandDetectionConfidence = {{o.Inv("confidence")}}f,
                });

                using var image = MPImage.Load("victory.jpg");
                foreach (var hand in hands.Detect(image).Hands)
                {
                    var tip = hand[HandLandmark.IndexFingerTip];
                    Console.WriteLine($"{hand.Handedness.CategoryName}: index tip at ({tip.X:F2}, {tip.Y:F2})");
                }
                """,
        },
        new GalleryTask
        {
            Id = "gestures", Category = "landmarks",
            TitleEn = "Gesture recognition", TitleId = "Pengenalan gestur",
            BlurbEn = "Recognizes eight hand gestures — closed fist, open palm, pointing up, thumbs down, thumbs up, victory and “I love you” — from the hand landmarks.",
            BlurbId = "Mengenali delapan gestur tangan — kepalan, telapak terbuka, menunjuk ke atas, jempol ke bawah, jempol ke atas, victory, dan “I love you” — dari landmark tangan.",
            ModelLabel = "hand landmarks → gesture_embedder → classifier",
            Samples = ["thumb_up.jpg", "victory.jpg", "pointing_up.jpg"],
            Options = [Confidence],
            Create = (o, mode, b) => GestureRecognizer.Create(new() { BaseOptions = b, RunningMode = mode, MaxResults = 3, Hands = new() { MinHandDetectionConfidence = o.F("confidence") } }),
            Run = (t, img, ts, o) =>
            {
                var d = (GestureRecognizer)t;
                var r = ts is { } v ? d.RecognizeForVideo(img, v) : d.Recognize(img);
                var overlay = new Overlay();
                var rows = new List<ResultRow>();
                foreach (var h in r.Hands)
                {
                    overlay.AddSkeleton(h.Hand.Landmarks, Connections.Hand, Overlay.Mint, Overlay.Paper, 2.6, 3.6);
                    var xs = h.Hand.Landmarks.Select(l => l.X).ToArray();
                    var ys = h.Hand.Landmarks.Select(l => l.Y).ToArray();
                    overlay.Boxes.Add(OverlayBox.FromRect(RectF.FromLtrb(xs.Min() - 0.02f, ys.Min() - 0.02f, xs.Max() + 0.02f, ys.Max() + 0.02f),
                        $"{h.TopGesture.CategoryName} {P(h.TopGesture.Score)}", Overlay.Amber));
                    foreach (var g in h.Gestures) rows.Add(new ResultRow(g.CategoryName!, P(g.Score), g.Score));
                }
                return new GalleryOutput(overlay, rows, r.ToJson(true), string.Join(", ", r.Hands.Select(h => h.TopGesture.CategoryName)));
            },
            Code = o => $$"""
                using var recognizer = GestureRecognizer.Create(new GestureRecognizerOptions
                {
                    Hands = new() { MinHandDetectionConfidence = {{o.Inv("confidence")}}f },
                });

                using var image = MPImage.Load("thumb_up.jpg");
                foreach (var hand in recognizer.Recognize(image).Hands)
                    Console.WriteLine($"{hand.Hand.Handedness.CategoryName} hand: {hand.TopGesture.CategoryName}");
                """,
        },
        new GalleryTask
        {
            Id = "pose", Category = "landmarks",
            TitleEn = "Pose landmarks", TitleId = "Landmark pose",
            BlurbEn = "BlazePose GHUM tracks 33 body landmarks with visibility, 3-D world coordinates in meters and an optional person mask.",
            BlurbId = "BlazePose GHUM melacak 33 landmark tubuh beserta visibilitas, koordinat dunia 3-D dalam meter, dan mask orang opsional.",
            ModelLabel = "pose_detection 224² → pose_landmarks 256²",
            Samples = ["pose.jpg", "portrait.jpg"],
            Options = [Confidence, new("full", "Use the Full model", "Gunakan model Full", OptionKind.Toggle, 0), new("mask", "Show person mask", "Tampilkan mask orang", OptionKind.Toggle, 1)],
            Create = (o, mode, b) => PoseLandmarker.Create(new()
            {
                BaseOptions = b, RunningMode = mode, MinPoseDetectionConfidence = o.F("confidence"),
                Model = o.B("full") ? PoseModel.Full : PoseModel.Lite, OutputSegmentationMasks = o.B("mask"),
            }),
            Run = (t, img, ts, o) =>
            {
                var d = (PoseLandmarker)t;
                var r = ts is { } v ? d.DetectForVideo(img, v) : d.Detect(img);
                var overlay = new Overlay();
                var rows = new List<ResultRow>();
                foreach (var p in r.Poses)
                {
                    if (p.SegmentationMask is { } m) overlay.Mask = new OverlayMask(m.Data, m.Width, m.Height, Overlay.Violet, 0.45f);
                    overlay.AddSkeleton(p.Landmarks, Connections.Pose, Overlay.Mint, Overlay.Paper, 3, 4);
                    foreach (var lm in new[] { PoseLandmark.Nose, PoseLandmark.LeftWrist, PoseLandmark.RightWrist, PoseLandmark.LeftAnkle, PoseLandmark.RightAnkle })
                        rows.Add(new ResultRow(lm.ToString(), $"vis {P(p[lm].Visibility ?? 0)}", p[lm].Visibility));
                }
                return new GalleryOutput(overlay, rows, Json(new { poses = r.Poses.Select(p => new { p.Landmarks, p.WorldLandmarks, p.PresenceScore }) }), $"{r.Poses.Count} pose(s)");
            },
            Code = o => $$"""
                using var pose = PoseLandmarker.Create(new PoseLandmarkerOptions
                {
                    Model = PoseModel.{{(o.B("full") ? "Full" : "Lite")}},
                    OutputSegmentationMasks = {{(o.B("mask") ? "true" : "false")}},
                    MinPoseDetectionConfidence = {{o.Inv("confidence")}}f,
                });

                using var image = MPImage.Load("pose.jpg");
                var person = pose.Detect(image).Poses[0];
                var wrist = person[PoseLandmark.LeftWrist];
                Console.WriteLine($"left wrist ({wrist.X:F2}, {wrist.Y:F2}), visible {wrist.Visibility:P0}");
                Console.WriteLine($"world: {person.WorldLandmarks[(int)PoseLandmark.LeftWrist]} m");
                """,
        },
        new GalleryTask
        {
            Id = "holistic", Category = "landmarks",
            TitleEn = "Holistic", TitleId = "Holistic",
            BlurbEn = "Body, face mesh and both hands of one person in a single call. The three pipelines run in parallel; hands are matched to the body's wrists.",
            BlurbId = "Tubuh, face mesh, dan kedua tangan satu orang dalam satu panggilan. Tiga pipeline berjalan paralel; tangan dicocokkan ke pergelangan tubuh.",
            ModelLabel = "pose + face mesh + hands",
            Samples = ["pose.jpg", "portrait.jpg"],
            Options = [Confidence],
            Create = (o, mode, b) => HolisticLandmarker.Create(new() { BaseOptions = b, RunningMode = mode, MinDetectionConfidence = o.F("confidence") }),
            Run = (t, img, ts, o) =>
            {
                var d = (HolisticLandmarker)t;
                var r = ts is { } v ? d.DetectForVideo(img, v) : d.Detect(img);
                var overlay = new Overlay();
                if (r.Pose is { } p) overlay.AddSkeleton(p.Landmarks, Connections.Pose, Overlay.Paper, Overlay.Paper, 2.4, 3);
                if (r.Face is { } f) overlay.AddSkeleton(f.Landmarks, Connections.FaceContours, Overlay.Mint, Overlay.Mint, 1.4, 1, points: false);
                if (r.LeftHand is { } lh) overlay.AddSkeleton(lh.Landmarks, Connections.Hand, Overlay.Coral, Overlay.Paper, 2.2, 2.6);
                if (r.RightHand is { } rh) overlay.AddSkeleton(rh.Landmarks, Connections.Hand, Overlay.Cobalt, Overlay.Paper, 2.2, 2.6);
                var rows = new List<ResultRow>
                {
                    new("Pose", r.Pose is null ? "—" : "33 landmarks", r.Pose?.PresenceScore),
                    new("Face", r.Face is null ? "—" : "478 landmarks", r.Face?.PresenceScore),
                    new(Loc.Pick("Left hand", "Tangan kiri"), r.LeftHand is null ? "—" : "21 landmarks", r.LeftHand?.PresenceScore),
                    new(Loc.Pick("Right hand", "Tangan kanan"), r.RightHand is null ? "—" : "21 landmarks", r.RightHand?.PresenceScore),
                };
                return new GalleryOutput(overlay, rows, Json(new { pose = r.Pose?.Landmarks, face = r.Face?.Landmarks.Count, left = r.LeftHand?.Landmarks, right = r.RightHand?.Landmarks }), "holistic");
            },
            Code = o => $$"""
                using var holistic = HolisticLandmarker.Create(new HolisticLandmarkerOptions
                {
                    MinDetectionConfidence = {{o.Inv("confidence")}}f,
                });

                using var image = MPImage.Load("pose.jpg");
                HolisticResult person = holistic.Detect(image);
                Console.WriteLine($"pose {person.Pose is not null}, face {person.Face is not null}");
                Console.WriteLine($"left hand {person.LeftHand is not null}, right hand {person.RightHand is not null}");
                """,
        },
        new GalleryTask
        {
            Id = "segment", Category = "understand",
            TitleEn = "Selfie segmentation", TitleId = "Segmentasi selfie",
            BlurbEn = "A per-pixel person mask for background blur or replacement — the effect behind video-call backgrounds.",
            BlurbId = "Mask orang per piksel untuk blur atau penggantian latar — efek di balik latar video call.",
            ModelLabel = "selfie_segmenter · 256²",
            Samples = ["portrait.jpg", "pose.jpg"],
            Options = [new("effect", "Effect", "Efek", OptionKind.Choice, 0, Choices: ["Mask", "Blur background", "Studio backdrop"])],
            Create = (o, mode, b) => ImageSegmenter.Create(new() { BaseOptions = b, RunningMode = mode }),
            Run = (t, img, ts, o) =>
            {
                var d = (ImageSegmenter)t;
                var r = ts is { } v ? d.SegmentForVideo(img, v) : d.Segment(img);
                var mask = r.ConfidenceMask;
                var overlay = new Overlay();
                MPImage? composed = null;
                switch (o.I("effect"))
                {
                    case 0:
                        overlay.Mask = new OverlayMask(mask.Data, mask.Width, mask.Height, Overlay.Violet, 0.55f);
                        break;
                    default:
                        using (var canvas = img.ToImage())
                        {
                            if (o.I("effect") == 1) SegmentationMaskOverlay.BlurBackground(canvas, mask, 18);
                            else SegmentationMaskOverlay.ReplaceBackground(canvas, mask, SixLabors.ImageSharp.Color.ParseHex("#2B59FF"));
                            composed = MPImage.FromImage(canvas);
                        }
                        break;
                }
                var rows = new List<ResultRow> { new(Loc.Pick("Person coverage", "Cakupan orang"), P(mask.Coverage()), mask.Coverage()), new("Mask", $"{mask.Width}×{mask.Height}") };
                return new GalleryOutput(overlay, rows, $"{{ \"width\": {mask.Width}, \"height\": {mask.Height}, \"coverage\": {mask.Coverage().ToString(CultureInfo.InvariantCulture)} }}", "mask", composed);
            },
            Code = o => """
                using MediaPipeNet.Visualization;

                using var segmenter = ImageSegmenter.Create();
                using var image = MPImage.Load("portrait.jpg");
                SegmentationMask mask = segmenter.Segment(image).ConfidenceMask;

                using var canvas = image.ToImage();          // SixLabors.ImageSharp image
                SegmentationMaskOverlay.BlurBackground(canvas, mask, sigma: 18);
                canvas.SaveAsPng("portrait-blur.png");
                """,
        },
        new GalleryTask
        {
            Id = "objects", Category = "detect",
            TitleEn = "Object detection", TitleId = "Deteksi objek",
            BlurbEn = "EfficientDet-Lite0 locates and labels 80 everyday object classes — people, vehicles, animals, food, furniture.",
            BlurbId = "EfficientDet-Lite0 menemukan dan melabeli 80 kelas objek sehari-hari — orang, kendaraan, hewan, makanan, furnitur.",
            ModelLabel = "efficientdet_lite0 · 320² · COCO",
            Samples = ["cats_and_dogs.jpg", "portrait.jpg", "burger.jpg"],
            Options = [new("score", "Score threshold", "Ambang skor", OptionKind.Slider, 0.3, 0.05, 0.9), new("max", "Maximum results", "Hasil maksimum", OptionKind.Slider, 10, 1, 25, 1, "0")],
            Create = (o, mode, b) => ObjectDetector.Create(new() { BaseOptions = b, RunningMode = mode, ScoreThreshold = o.F("score"), MaxResults = o.I("max") }),
            Run = (t, img, ts, o) =>
            {
                var d = (ObjectDetector)t;
                var r = ts is { } v ? d.DetectForVideo(img, v) : d.Detect(img);
                var overlay = new Overlay();
                var rows = new List<ResultRow>();
                foreach (var det in r.Detections)
                {
                    overlay.Boxes.Add(OverlayBox.FromRect(det.BoundingBox.Scale(1f / img.Width, 1f / img.Height), $"{det.TopCategory.CategoryName} {P(det.Score)}", Overlay.Amber));
                    rows.Add(new ResultRow(det.TopCategory.CategoryName!, P(det.Score), det.Score));
                }
                return new GalleryOutput(overlay, rows, r.ToJson(true), $"{r.Detections.Count} object(s)");
            },
            Code = o => $$"""
                using var detector = ObjectDetector.Create(new ObjectDetectorOptions
                {
                    ScoreThreshold = {{o.Inv("score")}}f,
                    MaxResults = {{o.I("max")}},
                    // CategoryAllowlist = new HashSet<string> { "person", "dog" },
                });

                using var image = MPImage.Load("cats_and_dogs.jpg");
                foreach (var obj in detector.Detect(image).Detections)
                    Console.WriteLine($"{obj.TopCategory.CategoryName} {obj.Score:P0} at {obj.BoundingBox}");
                """,
        },
        new GalleryTask
        {
            Id = "classify", Category = "understand",
            TitleEn = "Image classification", TitleId = "Klasifikasi gambar",
            BlurbEn = "EfficientNet-Lite0 names what the whole picture shows, choosing among 1,000 ImageNet categories.",
            BlurbId = "EfficientNet-Lite0 menyebutkan isi keseluruhan gambar dari 1.000 kategori ImageNet.",
            ModelLabel = "efficientnet_lite0 · 224² · ImageNet",
            Samples = ["burger.jpg", "cats_and_dogs.jpg", "pose.jpg"],
            Options = [new("max", "Top results", "Hasil teratas", OptionKind.Slider, 5, 1, 10, 1, "0")],
            Create = (o, mode, b) => ImageClassifier.Create(new() { BaseOptions = b, RunningMode = mode, MaxResults = o.I("max") }),
            Run = (t, img, ts, o) =>
            {
                var d = (ImageClassifier)t;
                var r = ts is { } v ? d.ClassifyForVideo(img, v) : d.Classify(img);
                var overlay = new Overlay();
                var rows = r.Categories.Select(c => new ResultRow(c.CategoryName!, P(c.Score), c.Score)).ToList();
                if (r.Categories.Count > 0)
                    overlay.Boxes.Add(OverlayBox.FromRect(new RectF(0.02f, 0.02f, 0.96f, 0.96f), $"{r.Categories[0].CategoryName} {P(r.Categories[0].Score)}", Overlay.Amber, dashed: true));
                return new GalleryOutput(overlay, rows, r.ToJson(true), r.Categories.FirstOrDefault()?.CategoryName ?? "");
            },
            Code = o => $$"""
                using var classifier = ImageClassifier.Create(new ImageClassifierOptions { MaxResults = {{o.I("max")}} });

                using var image = MPImage.Load("burger.jpg");
                ClassificationResult result = await classifier.ClassifyAsync(image);
                foreach (var category in result.Categories)
                    Console.WriteLine($"{category.CategoryName}: {category.Score:P1}");
                """,
        },
    ];

    public static GalleryTask Get(string id) => All.First(t => t.Id == id);

    /// <summary>The path of a bundled sample image.</summary>
    public static string SamplePath(string name) => Path.Combine(AppContext.BaseDirectory, "images", name);
}
