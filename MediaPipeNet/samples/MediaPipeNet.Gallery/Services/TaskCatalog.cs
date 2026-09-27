using System.Globalization;
using MediaPipeNet.Gallery.Controls;
using MediaPipeNet.Imaging;
using MediaPipeNet.Serialization;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Visualization;

namespace MediaPipeNet.Gallery.Services;

/// <summary>The vision tasks shown in the Gallery.</summary>
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
            BlurbEn = "BlazeFace finds faces and six keypoints — eyes, nose tip, mouth and ear tragions. The short-range model suits selfies; the full-range model finds faces up to about five meters away.",
            BlurbId = "BlazeFace menemukan wajah beserta enam keypoint — mata, ujung hidung, mulut, dan tragus telinga. Model short-range cocok untuk selfie; model full-range menemukan wajah hingga sekitar lima meter.",
            ModelLabel = "blaze_face_short_range 128² · full_range 192²",
            Samples = ["portrait.jpg", "pose.jpg", "victory.jpg"],
            Options = [new("model", "Model", "Model", OptionKind.Choice, 0, Choices: ["Short range", "Full range"]), Confidence, new("nms", "Merge overlap (IoU)", "Gabungkan tumpang-tindih (IoU)", OptionKind.Slider, 0.3, 0.1, 0.9)],
            Create = (o, mode, b) => FaceDetector.Create(new()
            {
                BaseOptions = b, RunningMode = mode, MinDetectionConfidence = o.F("confidence"), MinSuppressionThreshold = o.F("nms"),
                Model = o.I("model") == 1 ? FaceDetectorModel.FullRange : FaceDetectorModel.ShortRange,
            }),
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
                    Model = FaceDetectorModel.{{(o.I("model") == 1 ? "FullRange" : "ShortRange")}},
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
            Create = (o, mode, b) => FaceLandmarker.Create(new()
            {
                BaseOptions = b, RunningMode = mode, OutputFaceBlendshapes = true, OutputFacialTransformationMatrixes = true,
                MinFaceDetectionConfidence = o.F("confidence"),
            }),
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
                    if (f.FacialTransformationMatrix is { } m)
                    {
                        // Head pose from the rotation part of the facial transformation matrix (degrees).
                        double s = Math.Sqrt(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]);
                        double pitch = Math.Asin(Math.Clamp(-m[9] / s, -1, 1)) * 180 / Math.PI;
                        double yaw = Math.Atan2(m[8], m[10]) * 180 / Math.PI;
                        double roll = Math.Atan2(m[1], m[5]) * 180 / Math.PI;
                        rows.Add(new ResultRow(Loc.Pick("Head pose (yaw / pitch / roll)", "Pose kepala (yaw / pitch / roll)"), $"{yaw:0}° / {pitch:0}° / {roll:0}°"));
                        rows.Add(new ResultRow(Loc.Pick("Distance to camera", "Jarak ke kamera"), $"{-m[11]:0} cm"));
                    }
                    foreach (var c in f.Blendshapes!.Where(c => c.CategoryName != "_neutral").OrderByDescending(c => c.Score).Take(10))
                        rows.Add(new ResultRow(c.CategoryName!, P(c.Score), c.Score));
                }
                return new GalleryOutput(overlay, rows, r.ToJson(true), $"{r.Faces.Count} face(s) · 478 landmarks");
            },
            Code = o => $$"""
                using var landmarker = FaceLandmarker.Create(new FaceLandmarkerOptions
                {
                    OutputFaceBlendshapes = true,
                    OutputFacialTransformationMatrixes = true,   // head pose for AR
                    MinFaceDetectionConfidence = {{o.Inv("confidence")}}f,
                });

                using var image = MPImage.Load("portrait.jpg");
                var face = landmarker.Detect(image).Faces[0];

                NormalizedLandmark noseTip = face.Landmarks[1];
                float smile = face.GetBlendshape("mouthSmileLeft");
                float blink = face.GetBlendshape("eyeBlinkRight");
                Console.WriteLine($"smile {smile:P0}, blink {blink:P0}");

                // 4×4 pose of the canonical face in camera space (translation in cm).
                System.Numerics.Matrix4x4 pose = face.GetTransformMatrix()!.Value;
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
            TitleEn = "Image segmentation", TitleId = "Segmentasi gambar",
            BlurbEn = "Per-pixel masks: a person mask for background blur, six selfie classes (hair, skin, clothes…), hair only, or DeepLab's 21 PASCAL classes.",
            BlurbId = "Mask per piksel: mask orang untuk blur latar, enam kelas selfie (rambut, kulit, pakaian…), rambut saja, atau 21 kelas PASCAL dari DeepLab.",
            ModelLabel = "selfie · multiclass · hair · deeplab_v3",
            Samples = ["portrait.jpg", "pose.jpg", "cats_and_dogs.jpg"],
            Options =
            [
                new("model", "Model", "Model", OptionKind.Choice, 0, Choices: ["Selfie", "Selfie multiclass", "Hair", "DeepLab v3"]),
                new("effect", "Effect", "Efek", OptionKind.Choice, 0, Choices: ["Mask", "Blur background", "Studio backdrop"]),
            ],
            Create = (o, mode, b) => ImageSegmenter.Create(new()
            {
                BaseOptions = b, RunningMode = mode, OutputCategoryMask = true,
                Model = o.I("model") switch { 1 => SegmenterModel.SelfieMulticlass, 2 => SegmenterModel.Hair, 3 => SegmenterModel.DeepLabV3, _ => SegmenterModel.Selfie },
            }),
            Run = (t, img, ts, o) =>
            {
                var d = (ImageSegmenter)t;
                var r = ts is { } v ? d.SegmentForVideo(img, v) : d.Segment(img);
                var mask = r.ConfidenceMask;
                var overlay = new Overlay();
                MPImage? composed = null;
                using (var canvas = img.ToImage())
                {
                    switch (o.I("effect"))
                    {
                        case 0 when r.ConfidenceMasks.Count == 1:
                            overlay.Mask = new OverlayMask(mask.Data, mask.Width, mask.Height, Overlay.Violet, 0.55f);
                            break;
                        case 0:
                            SegmentationMaskOverlay.OverlayCategories(canvas, r.CategoryMask!);
                            composed = MPImage.FromImage(canvas);
                            break;
                        case 1:
                            SegmentationMaskOverlay.BlurBackground(canvas, mask, 18);
                            composed = MPImage.FromImage(canvas);
                            break;
                        default:
                            SegmentationMaskOverlay.ReplaceBackground(canvas, mask, SixLabors.ImageSharp.Color.ParseHex("#2B59FF"));
                            composed = MPImage.FromImage(canvas);
                            break;
                    }
                }
                var rows = new List<ResultRow>();
                if (r.ConfidenceMasks.Count == 1)
                {
                    rows.Add(new(Loc.Pick("Person coverage", "Cakupan orang"), P(mask.Coverage()), mask.Coverage()));
                }
                else
                {
                    foreach (var (category, fraction) in r.CategoryMask!.Histogram().OrderByDescending(kv => kv.Value))
                        if (fraction >= 0.002f && category < r.Labels.Count) rows.Add(new(r.Labels[category], P(fraction), fraction));
                }
                rows.Add(new("Mask", $"{mask.Width}×{mask.Height} · {r.ConfidenceMasks.Count} ch"));
                var json = $"{{ \"labels\": [{string.Join(", ", r.Labels.Select(l => $"\"{l}\""))}], \"width\": {mask.Width}, \"height\": {mask.Height} }}";
                return new GalleryOutput(overlay, rows, json, r.ConfidenceMasks.Count == 1 ? "mask" : $"{rows.Count - 1} categories", composed);
            },
            Code = o => $$"""
                using MediaPipeNet.Visualization;

                using var segmenter = ImageSegmenter.Create(new ImageSegmenterOptions
                {
                    Model = SegmenterModel.{{(o.I("model") switch { 1 => "SelfieMulticlass", 2 => "Hair", 3 => "DeepLabV3", _ => "Selfie" })}},
                    OutputCategoryMask = true,
                });
                using var image = MPImage.Load("portrait.jpg");
                SegmentationResult result = segmenter.Segment(image);

                // One probability mask per category, plus the argmax category per pixel.
                foreach (var (index, fraction) in result.CategoryMask!.Histogram())
                    Console.WriteLine($"{result.Labels[index]}: {fraction:P1}");

                using var canvas = image.ToImage();
                SegmentationMaskOverlay.BlurBackground(canvas, result.ConfidenceMask, sigma: 18);
                """,
        },
        new GalleryTask
        {
            Id = "interactive", Category = "understand",
            TitleEn = "Interactive segmentation", TitleId = "Segmentasi interaktif",
            BlurbEn = "MagicTouch cuts out whatever object sits under a point of interest — tap-to-select for photo editors. Move the point with the sliders.",
            BlurbId = "MagicTouch memotong objek apa pun di bawah titik yang dipilih — tap-untuk-memilih ala editor foto. Geser titiknya dengan slider.",
            ModelLabel = "magic_touch · 512² RGB + point",
            Samples = ["cats_and_dogs.jpg", "burger.jpg", "portrait.jpg"],
            SupportsLive = false,
            Options =
            [
                new("x", "Point X", "Titik X", OptionKind.Slider, 0.62, 0.02, 0.98, 0.01),
                new("y", "Point Y", "Titik Y", OptionKind.Slider, 0.5, 0.02, 0.98, 0.01),
                new("cutout", "Cut out (hide background)", "Potong (sembunyikan latar)", OptionKind.Toggle, 0),
            ],
            Create = (o, mode, b) => InteractiveSegmenter.Create(new() { BaseOptions = b }),
            Run = (t, img, ts, o) =>
            {
                var d = (InteractiveSegmenter)t;
                var r = d.Segment(img, RegionOfInterest.FromKeypoint(o.F("x"), o.F("y")));
                var mask = r.ConfidenceMask;
                var overlay = new Overlay();
                overlay.Points.Add(new OverlayPoint(o.F("x"), o.F("y"), Overlay.Coral, 7));
                MPImage? composed = null;
                if (o.B("cutout"))
                {
                    using var canvas = img.ToImage();
                    SegmentationMaskOverlay.ReplaceBackground(canvas, mask, SixLabors.ImageSharp.Color.ParseHex("#F4F1EA"));
                    composed = MPImage.FromImage(canvas);
                }
                else
                {
                    overlay.Mask = new OverlayMask(mask.Data, mask.Width, mask.Height, Overlay.Mint, 0.55f);
                }
                var rows = new List<ResultRow>
                {
                    new(Loc.Pick("Selected area", "Area terpilih"), P(mask.Coverage()), mask.Coverage()),
                    new(Loc.Pick("Point", "Titik"), $"({o.Inv("x")}, {o.Inv("y")})"),
                };
                return new GalleryOutput(overlay, rows, $"{{ \"coverage\": {mask.Coverage().ToString(CultureInfo.InvariantCulture)} }}", Loc.Pick("object selected", "objek terpilih"), composed);
            },
            Code = o => $$"""
                using var segmenter = InteractiveSegmenter.Create();
                using var image = MPImage.Load("cats_and_dogs.jpg");

                // The object under the point (normalized coordinates) — or pass a scribble.
                var roi = RegionOfInterest.FromKeypoint({{o.Inv("x")}}f, {{o.Inv("y")}}f);
                SegmentationMask cutout = segmenter.Segment(image, roi).ConfidenceMask;
                Console.WriteLine($"selected {cutout.Coverage():P0} of the image");
                """,
        },
        new GalleryTask
        {
            Id = "embed", Category = "understand",
            TitleEn = "Image embedding", TitleId = "Embedding gambar",
            BlurbEn = "MobileNet V3 turns an image into a 1,024-number fingerprint. Similar pictures get similar vectors — the basis of visual search and de-duplication.",
            BlurbId = "MobileNet V3 mengubah gambar menjadi sidik jari 1.024 angka. Gambar serupa menghasilkan vektor serupa — dasar pencarian visual dan de-duplikasi.",
            ModelLabel = "mobilenet_v3_small · 224² · 1024-D",
            Samples = ["burger.jpg", "burger_crop.jpg", "cat.jpg", "cats_and_dogs.jpg", "portrait.jpg"],
            Options = [],
            SupportsLive = false,
            Create = (o, mode, b) => ImageEmbedder.Create(new() { BaseOptions = b, RunningMode = mode, L2Normalize = true }),
            Run = (t, img, ts, o) =>
            {
                var d = (ImageEmbedder)t;
                var r = ts is { } v ? d.EmbedForVideo(img, v) : d.Embed(img);
                var rows = new List<ResultRow>();
                foreach (var name in new[] { "burger.jpg", "burger_crop.jpg", "cat.jpg", "cats_and_dogs.jpg", "portrait.jpg" })
                {
                    double similarity = ImageEmbedder.CosineSimilarity(r.Embedding, ReferenceEmbedding(d, name));
                    rows.Add(new(Loc.Pick("Similarity to ", "Kemiripan dengan ") + name, similarity.ToString("0.000", CultureInfo.InvariantCulture), Math.Max(0, similarity)));
                }
                rows.Sort((a, b) => b.Fraction!.Value.CompareTo(a.Fraction!.Value));
                return new GalleryOutput(new Overlay(), rows, r.ToJson(true), $"{r.Embedding.Dimension}-D");
            },
            Code = o => """
                using var embedder = ImageEmbedder.Create(new ImageEmbedderOptions { L2Normalize = true });

                using var a = MPImage.Load("burger.jpg");
                using var b = MPImage.Load("burger_crop.jpg");
                Embedding ea = embedder.Embed(a).Embedding;   // 1024 floats
                Embedding eb = embedder.Embed(b).Embedding;

                double similarity = ImageEmbedder.CosineSimilarity(ea, eb);   // ≈ 0.92
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

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<ImageEmbedder, Dictionary<string, Embedding>> s_references = [];

    /// <summary>Embedding of a bundled sample, computed once per embedder instance.</summary>
    private static Embedding ReferenceEmbedding(ImageEmbedder embedder, string sample)
    {
        var cache = s_references.GetOrCreateValue(embedder);
        lock (cache)
        {
            if (cache.TryGetValue(sample, out var e)) return e;
            using var image = MPImage.Load(SamplePath(sample));
            return cache[sample] = embedder.Embed(image).Embedding;
        }
    }

    /// <summary>The path of a bundled sample image.</summary>
    public static string SamplePath(string name) => Path.Combine(AppContext.BaseDirectory, "images", name);
}
