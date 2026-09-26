namespace MediaPipeNet.Inference.Models;

/// <summary>
/// The registry of pretrained models shipped with MediaPipe.NET. Every model was converted from the
/// official Google MediaPipe TFLite release with <c>tools/model-conversion/convert_models.py</c> and
/// cross-validated against the TFLite interpreter (max |Δ| ≈ 1e-4).
/// </summary>
/// <remarks>Generated from <c>models/onnx/manifest.json</c>.</remarks>
public static class ModelCatalog
{
    /// <summary>NuGet package holding the face models.</summary>
    public const string FacePackage = "Gravicode.MediaPipeNet.Models.Face";
    /// <summary>NuGet package holding the hand and gesture models.</summary>
    public const string HandPackage = "Gravicode.MediaPipeNet.Models.Hand";
    /// <summary>NuGet package holding the pose models.</summary>
    public const string PosePackage = "Gravicode.MediaPipeNet.Models.Pose";
    /// <summary>NuGet package holding the segmentation models.</summary>
    public const string SegmentationPackage = "Gravicode.MediaPipeNet.Models.Segmentation";
    /// <summary>NuGet package holding the object detection models.</summary>
    public const string ObjectDetectionPackage = "Gravicode.MediaPipeNet.Models.ObjectDetection";
    /// <summary>NuGet package holding the image classification models.</summary>
    public const string ImageClassificationPackage = "Gravicode.MediaPipeNet.Models.ImageClassification";

    private const string Mp = "https://storage.googleapis.com/mediapipe-models";

    /// <summary>BlazeFace short-range face detector (128×128).</summary>
    public static ModelDescriptor FaceDetectionShortRange { get; } = new(
        "face_detection_short_range", "face_detection_short_range.onnx",
        "0f9ebac6ae85a09f5e37d46c3e018494932838b9aeb9e9bf2e6743d846764d3a", 418_458, FacePackage,
        "BlazeFace (short range)", $"{Mp}/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite");

    /// <summary>Face mesh landmark model: 478 landmarks including irises (256×256).</summary>
    public static ModelDescriptor FaceLandmarksDetector { get; } = new(
        "face_landmarks_detector", "face_landmarks_detector.onnx",
        "f5140965778a7d7bb5e843ca10b344369ef0edc4c2c8e092385d2fe729f55281", 4_921_009, FacePackage,
        "Face Mesh V2 (478 landmarks)", $"{Mp}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task#face_landmarks_detector.tflite");

    /// <summary>Face blendshapes model: 52 ARKit-style expression coefficients.</summary>
    public static ModelDescriptor FaceBlendshapes { get; } = new(
        "face_blendshapes", "face_blendshapes.onnx",
        "de2b4ea38439a88d7f2b48d25794d742cee048413045a14f94813e6fec23e38a", 1_883_812, FacePackage,
        "Face Blendshapes", $"{Mp}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task#face_blendshapes.tflite");

    /// <summary>Palm detector (192×192).</summary>
    public static ModelDescriptor PalmDetection { get; } = new(
        "palm_detection", "palm_detection.onnx",
        "a07968a28db24b4818d8b85464f3ff258b527bc08fdee613555a2100d0f32231", 4_589_700, HandPackage,
        "Palm Detection (full)", $"{Mp}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task#hand_detector.tflite");

    /// <summary>Hand landmark model: 21 3-D landmarks and handedness (224×224).</summary>
    public static ModelDescriptor HandLandmarksDetector { get; } = new(
        "hand_landmarks_detector", "hand_landmarks_detector.onnx",
        "01df77bedaedfaa204cd186f89d2dd3988ea960d4335c1880a0d8824671e7fca", 10_903_815, HandPackage,
        "Hand Landmarks (full)", $"{Mp}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task#hand_landmarks_detector.tflite");

    /// <summary>Gesture embedder: hand landmarks to a 128-D embedding.</summary>
    public static ModelDescriptor GestureEmbedder { get; } = new(
        "gesture_embedder", "gesture_embedder.onnx",
        "1b6226028ea6f383047cd7df567439dd04eb43735618a6db0487abc78978ed37", 546_063, HandPackage,
        "Gesture Embedder", $"{Mp}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task#gesture_embedder.tflite");

    /// <summary>Canned gesture classifier: 8 gestures (None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou).</summary>
    public static ModelDescriptor CannedGestureClassifier { get; } = new(
        "canned_gesture_classifier", "canned_gesture_classifier.onnx",
        "d18989ddd043b2267a130c6f2889ebd681652e64829216d0b4069440ad6a3312", 6_388, HandPackage,
        "Canned Gesture Classifier", $"{Mp}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task#canned_gesture_classifier.tflite");

    /// <summary>BlazePose person detector (224×224).</summary>
    public static ModelDescriptor PoseDetection { get; } = new(
        "pose_detection", "pose_detection.onnx",
        "5ab3870ca30b51b32b65d7c59322aed35b67f4dd5fcb02521966a85844f0880c", 11_925_657, PosePackage,
        "BlazePose Detector", "https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite");

    /// <summary>BlazePose GHUM Lite landmark model: 33 landmarks + segmentation (256×256).</summary>
    public static ModelDescriptor PoseLandmarksLite { get; } = new(
        "pose_landmarks_detector_lite", "pose_landmarks_detector_lite.onnx",
        "690f9fd395eba776a4ed8434fa65d8b9c8196488054b702274982620d291c1db", 5_533_837, PosePackage,
        "BlazePose GHUM Lite", $"{Mp}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task#pose_landmarks_detector.tflite");

    /// <summary>BlazePose GHUM Full landmark model: 33 landmarks + segmentation (256×256).</summary>
    public static ModelDescriptor PoseLandmarksFull { get; } = new(
        "pose_landmarks_detector_full", "pose_landmarks_detector_full.onnx",
        "5437a4336695a81cb27c9619ff7bd609b7ed7ebddb3267850940cc90b75296b9", 12_765_338, PosePackage,
        "BlazePose GHUM Full", $"{Mp}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task#pose_landmarks_detector.tflite");

    /// <summary>Selfie segmenter (256×256, square).</summary>
    public static ModelDescriptor SelfieSegmenter { get; } = new(
        "selfie_segmenter", "selfie_segmenter.onnx",
        "d9877981bfcfa556d846d4e5b757750e1c4ca04d694e64adcd9562c8dfdaa165", 450_202, SegmentationPackage,
        "Selfie Segmenter", $"{Mp}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite");

    /// <summary>EfficientDet-Lite0 object detector, 90 COCO classes (320×320).</summary>
    public static ModelDescriptor EfficientDetLite0 { get; } = new(
        "efficientdet_lite0", "efficientdet_lite0.onnx",
        "e15e6a31c697d933700986b59725dc45f2b273474850c38b69949faef9bba748", 13_455_619, ObjectDetectionPackage,
        "EfficientDet-Lite0", $"{Mp}/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite");

    /// <summary>EfficientNet-Lite0 image classifier, 1000 ImageNet classes (224×224).</summary>
    public static ModelDescriptor EfficientNetLite0 { get; } = new(
        "efficientnet_lite0", "efficientnet_lite0.onnx",
        "fbcd11eee78c348a7d3ee62d62db2301255db9ed81288fceb7c2b25bb7e9bbf1", 18_599_569, ImageClassificationPackage,
        "EfficientNet-Lite0", $"{Mp}/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite");

    /// <summary>Every model in the catalog.</summary>
    public static IReadOnlyList<ModelDescriptor> All { get; } =
    [
        FaceDetectionShortRange, FaceLandmarksDetector, FaceBlendshapes,
        PalmDetection, HandLandmarksDetector, GestureEmbedder, CannedGestureClassifier,
        PoseDetection, PoseLandmarksLite, PoseLandmarksFull,
        SelfieSegmenter, EfficientDetLite0, EfficientNetLite0,
    ];

    /// <summary>Finds a model by id or file name.</summary>
    public static ModelDescriptor? Find(string idOrFileName) =>
        All.FirstOrDefault(m => m.Id.Equals(idOrFileName, StringComparison.OrdinalIgnoreCase)
                                || m.FileName.Equals(idOrFileName, StringComparison.OrdinalIgnoreCase));
}
