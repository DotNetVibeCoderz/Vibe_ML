namespace MediaPipeNet.Inference.Models;

/// <summary>
/// The registry of pretrained models shipped with MediaPipe.NET. Every model was converted from the
/// official Google MediaPipe TFLite release with <c>tools/model-conversion/convert_models.py</c> and
/// cross-validated against the TFLite interpreter (max |Δ| ≈ 1e-4).
/// </summary>
/// <remarks>Generated from <c>models/onnx/manifest.json</c>.</remarks>
public static partial class ModelCatalog
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
    /// <summary>NuGet package holding the image embedding models.</summary>
    public const string ImageEmbeddingPackage = "Gravicode.MediaPipeNet.Models.ImageEmbedding";
    /// <summary>NuGet package holding the audio models.</summary>
    public const string AudioPackage = "Gravicode.MediaPipeNet.Models.Audio";
    /// <summary>NuGet package holding the text models.</summary>
    public const string TextPackage = "Gravicode.MediaPipeNet.Models.Text";
    /// <summary>NuGet package holding the face stylizer model.</summary>
    public const string FaceStylizerPackage = "Gravicode.MediaPipeNet.Models.FaceStylizer";

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

    /// <summary>BlazeFace full-range face detector (192×192): faces up to ~5 m from the camera.</summary>
    public static ModelDescriptor FaceDetectionFullRange { get; } = new(
        "face_detection_full_range", "face_detection_full_range.onnx",
        "30843582dfac582284b32bd100d80a6bb4f5225bce4a50cfdab0080145b22879", 2_076_730, FacePackage,
        "BlazeFace (full range)", $"{Mp}/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite");

    /// <summary>Hand ROI refinement (256×256): re-crops a hand around the pose model's palm landmarks (holistic).</summary>
    public static ModelDescriptor HandRoiRefinement { get; } = new(
        "hand_roi_refinement", "hand_roi_refinement.onnx",
        "5a277b78674e20426caf746ed8a6284f02b702b3fff9b62a651e648748624e83", 129_266, HandPackage,
        "Hand ROI Refinement", $"{Mp}/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task#hand_roi_refinement.tflite");

    /// <summary>Selfie multiclass segmenter (256×256): background, hair, body skin, face skin, clothes, others.</summary>
    public static ModelDescriptor SelfieMulticlass { get; } = new(
        "selfie_multiclass", "selfie_multiclass.onnx",
        "220c6cdd795b44a6c90f539950a0374b3467886065b8af8ac934a4cc72644d63", 16_454_487, SegmentationPackage,
        "Selfie Multiclass Segmenter", $"{Mp}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite");

    /// <summary>Hair segmenter (512×512).</summary>
    public static ModelDescriptor HairSegmenter { get; } = new(
        "hair_segmenter", "hair_segmenter.onnx",
        "e378ed4107fc0f76eae003917e47b8e028c286b09850339cdd7901b2d23422c3", 771_295, SegmentationPackage,
        "Hair Segmenter", $"{Mp}/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite");

    /// <summary>DeepLab v3 (257×257): 21 PASCAL VOC classes.</summary>
    public static ModelDescriptor DeepLabV3 { get; } = new(
        "deeplab_v3", "deeplab_v3.onnx",
        "2a6c4360d502edcd9e4c7bf5882cc835295e0e82885f556a3e383eaf24d5cac2", 2_782_445, SegmentationPackage,
        "DeepLab v3", $"{Mp}/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite");

    /// <summary>MagicTouch interactive segmenter (512×512): segments the object under a point of interest.</summary>
    public static ModelDescriptor MagicTouch { get; } = new(
        "magic_touch", "magic_touch.onnx",
        "acbd5aca6e65b56aa759c4bdb31fc0665be5858d13283768ff31f9d02c4455d1", 12_388_061, SegmentationPackage,
        "MagicTouch", $"{Mp}/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite");

    /// <summary>Face mesh landmark model (192×192, 478 landmarks, no attention): the mesh bundled with MediaPipe's face stylizer.</summary>
    public static ModelDescriptor FaceLandmarksDetector192 { get; } = new(
        "face_landmarks_detector_192", "face_landmarks_detector_192.onnx",
        "99f180b280cfa556646a4af64f45eae4c891e9a177701ab23d49289c9fdee11b", 1_153_831, FaceStylizerPackage,
        "Face Mesh 192 (stylizer)", $"{Mp}/face_stylizer/blaze_face_stylizer/float32/latest/face_stylizer_color_sketch.task");

    /// <summary>
    /// BlazeFaceStylizer "color sketch" generator (256×256 aligned face in, stylized face out). Converted with
    /// its training-mode batch norms lowered to per-instance normalization and its noise injection kept.
    /// </summary>
    public static ModelDescriptor FaceStylizerColorSketch { get; } = new(
        "face_stylizer_color_sketch", "face_stylizer_color_sketch.onnx",
        "f854f242ca2b187d57743b55b30404be5f5af81d6bcff1a0f599d62870086776", 28_237_573, FaceStylizerPackage,
        "Face Stylizer (color sketch)", $"{Mp}/face_stylizer/blaze_face_stylizer/float32/latest/face_stylizer_color_sketch.task");

    /// <summary>MobileNet V3 small image embedder (224×224, 1024-D).</summary>
    public static ModelDescriptor MobileNetV3SmallEmbedder { get; } = new(
        "mobilenet_v3_small_embedder", "mobilenet_v3_small_embedder.onnx",
        "30c919adcdb1057c086444c7cdbede62db283ceee0667a442d12bf0163f52683", 4_189_825, ImageEmbeddingPackage,
        "MobileNet V3 Small (embedder)", $"{Mp}/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite");

    /// <summary>YAMNet audio event classifier: 521 AudioSet classes from 0.975 s of 16 kHz mono audio.</summary>
    public static ModelDescriptor YamNet { get; } = new(
        "yamnet", "yamnet.onnx",
        "2b7cfdfb9d4ed54a456165202d82019b53d98a8fb4737a61dde35b5eb9c1dbc8", 5_077_494, AudioPackage,
        "YAMNet", $"{Mp}/audio_classifier/yamnet/float32/latest/yamnet.tflite");

    /// <summary>MobileBERT sentiment classifier (SST-2: negative / positive, 128 tokens).</summary>
    public static ModelDescriptor BertClassifier { get; } = new(
        "bert_classifier", "bert_classifier.onnx",
        "57a062d6eb4887c3c6d47f0a5fcad6ad53b4ac9e5d771126f55a21229665a8d8", 25_479_396, TextPackage,
        "BERT Classifier (SST-2)", $"{Mp}/text_classifier/bert_classifier/float32/latest/bert_classifier.tflite");

    /// <summary>Average word-embedding sentiment classifier (SST-2, 256 tokens; tiny and fast).</summary>
    public static ModelDescriptor AverageWordClassifier { get; } = new(
        "average_word_classifier", "average_word_classifier.onnx",
        "fe2c2d0f8e1a5ca038b64ed2385197628d724f03b80869bc348bba450de65231", 642_925, TextPackage,
        "Average Word Classifier (SST-2)", $"{Mp}/text_classifier/average_word_classifier/float32/latest/average_word_classifier.tflite");

    /// <summary>MobileBERT sentence embedder (512-D, 128 tokens).</summary>
    public static ModelDescriptor BertEmbedder { get; } = new(
        "bert_embedder", "bert_embedder.onnx",
        "2ae36698213c8648ebb595e4399ae15553532e58d8f8aa151abade49c5a1ced9", 26_622_911, TextPackage,
        "BERT Embedder", $"{Mp}/text_embedder/bert_embedder/float32/latest/bert_embedder.tflite");

    /// <summary>Language detector: 110 languages from character n-grams.</summary>
    public static ModelDescriptor LanguageDetector { get; } = new(
        "language_detector", "language_detector.onnx",
        "b80eef5271030261bd5637365595ed54bce3b5edece46d27c9f9d112692fef36", 3_833_619, TextPackage,
        "Language Detector", $"{Mp}/language_detector/language_detector/float32/latest/language_detector.tflite");

    /// <summary>Every model in the catalog.</summary>
    public static IReadOnlyList<ModelDescriptor> All { get; } =
    [
        FaceDetectionShortRange, FaceLandmarksDetector, FaceBlendshapes,
        PalmDetection, HandLandmarksDetector, GestureEmbedder, CannedGestureClassifier,
        PoseDetection, PoseLandmarksLite, PoseLandmarksFull,
        SelfieSegmenter, EfficientDetLite0, EfficientNetLite0,
        FaceDetectionFullRange, HandRoiRefinement,
        SelfieMulticlass, HairSegmenter, DeepLabV3, MagicTouch, MobileNetV3SmallEmbedder, FaceStylizerColorSketch, FaceLandmarksDetector192,
        YamNet, BertClassifier, AverageWordClassifier, BertEmbedder, LanguageDetector,
    ];

    /// <summary>Finds a model by id or file name.</summary>
    public static ModelDescriptor? Find(string idOrFileName) =>
        All.FirstOrDefault(m => m.Id.Equals(idOrFileName, StringComparison.OrdinalIgnoreCase)
                                || m.FileName.Equals(idOrFileName, StringComparison.OrdinalIgnoreCase));
}
