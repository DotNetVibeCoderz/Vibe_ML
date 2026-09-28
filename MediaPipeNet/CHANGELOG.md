# Changelog

All notable changes to MediaPipe.NET. Versioning follows [SemVer](https://semver.org).

## 1.0.0 — 2026-09-28

The first stable release. The public API is recorded in `PublicAPI.Shipped.txt` for every library and from here on
changes only compatibly within 1.x (SemVer). Upgrading from 0.3.0 needs no code changes. Created by Gravicode Studios,
led by Kang Fadhil.

### Added
- `FaceStylizer` (MediaPipe's color-sketch face stylizer): face mesh → MediaPipe's `FaceToRectCalculator` alignment →
  generator → 256×256 image; `FaceStylizerResult.Composite(image)` pastes the result back onto the photo;
  `OutputFaceAlignment`; `FaceStylizer.ComputeFaceRect`. Golden-tested against MediaPipe 0.10.21 (mean |Δ| 2.4 / 255
  at 32×32 against the average of 16 MediaPipe runs). DI (`AddFaceStylizer`), CLI (`stylize`), Gallery page.
- Models: `face_stylizer_color_sketch` (converted: resource variables frozen, training-mode `FusedBatchNormV3`
  lowered to per-instance normalization, noise kept) and `face_landmarks_detector_192` (the stylizer's face mesh) in
  the new `Gravicode.MediaPipeNet.Models.FaceStylizer` package; FP16 and INT8 variants of both.
- `FaceLandmarkerOptions.LandmarksModel` to choose the face mesh model.
- `ExecutionProvider.Nnapi` (Android) — `Auto` now tries CUDA → DirectML → CoreML → NNAPI → CPU; CLI `--provider nnapi`.
- `samples/MediaPipeNet.Maui`: .NET MAUI app for Android (and iOS on a Mac) with bundled INT8 models.
- `benchmarks/MediaPipeNet.GpuBenchmark`: CPU vs DirectML (fp32, fp16, I/O binding) latency table that checks every
  configuration's result against the CPU's.
- Documentation site on GitHub Pages (`.github/workflows/mediapipenet-docs.yml`).
- Tools: `convert_models.py` freezes resource variables and lowers `FusedBatchNormV3`, and prunes dead
  initializers; `generate_golden_stylizer.py`; `evaluate_static_int8.py`.

### Fixed
- `InferenceOptions.UseIoBinding` with a GPU provider (DirectML, CUDA, CoreML, NNAPI) returned results for stale
  inputs: ONNX Runtime copies a bound CPU input to the device when it is bound, so the inputs are now re-bound before
  every run on those providers.

### Evaluated, not shipped
- Static (calibrated) INT8 for the models without an INT8 variant: below the 0.99 cosine bar on every model
  (MediaPipe's models were not trained for quantization).
- In-browser Blazor WebAssembly inference: needs an asynchronous backend under the tasks; planned as an additive 1.x
  feature. Server-side inference works today.

## 0.3.0 — 2026-09-28

Delivers the whole 0.2 and 0.3 roadmap in one release (there is no separate 0.2.0). Created by Gravicode Studios,
led by Kang Fadhil.

### Breaking
- `BaseOptions` and `ModelLoader` moved to the new `Gravicode.MediaPipeNet.Tasks.Core` package, namespace
  `MediaPipeNet.Tasks` — add `using MediaPipeNet.Tasks;` where you name these types.
- `SegmentationResult` is no longer a positional record: it has `ConfidenceMasks`, `CategoryMask` and `Labels`;
  `ConfidenceMask` remains as the foreground mask, `new SegmentationResult(mask)` still works.
- `ImageSegmenter` and the object/segmentation tasks no longer anti-alias their input (matching MediaPipe's CPU
  preprocessing); results move slightly closer to MediaPipe.

### Added — vision
- `FaceDetector`: full-range BlazeFace (`FaceDetectorModel.FullRange`, faces up to ~5 m).
- `FaceLandmarker`: facial transformation matrix (`OutputFacialTransformationMatrixes`, port of MediaPipe's geometry
  pipeline and weighted Procrustes solver), `GetTransformMatrix()`.
- `PoseLandmarker`: heatmap landmark refinement (2.7× closer to MediaPipe), segmentation-mask smoothing.
- `HolisticLandmarker`: MediaPipe's holistic pipeline — pose-derived hand ROIs refined by the hand ROI refinement
  model, face ROI from the pose plus face detector, ROI tracking between frames, hand world landmarks aligned to the
  pose.
- `ImageSegmenter`: selfie multiclass, hair and DeepLab v3 models, category masks, MediaPipe's segmentation smoothing.
- New tasks: `InteractiveSegmenter` (MagicTouch), `ImageEmbedder` (MobileNet V3 small).
- `ProcessBatch` / `ProcessBatchAsync` on every vision task; image-mode calls made fully thread-safe.
- Custom models: `ModelPath` / `Labels` (`ImageClassifier`, `ObjectDetector` of any input size, `AudioClassifier`,
  `TextClassifier` + `VocabularyPath`), `GestureRecognizer.ClassifierModelPath`; `convert_models.py --custom` for
  `.tflite` files and `.task` bundles.

### Added — audio and text
- `Gravicode.MediaPipeNet.Tasks.Audio`: `AudioClassifier` (YAMNet, clip and stream modes), `VoiceActivityDetector`,
  `AudioData` (WAV loader, band-limited resampler).
- `Gravicode.MediaPipeNet.Tasks.Text`: `TextClassifier` (MobileBERT and average-word), `TextEmbedder`,
  `LanguageDetector` (110 languages), MediaPipe-exact `BertTokenizer`, `RegexTokenizer`, `NGramHasher`.
- `Embedding` with cosine similarity and scalar quantization (Core).

### Added — runtime, models and tooling
- 12 new models (25 in total) in the new `Models.ImageEmbedding`, `Models.Audio`, `Models.Text` packages and the
  existing face/hand/segmentation packages; `Models.Quantized` with 14 FP16 and 9 weight-only INT8 variants, selected
  with `InferenceOptions.Precision`, each validated against its float model.
- `InferenceOptions.UseIoBinding`, int32/int64 tensors, `OnnxModel.RunDynamic` for variable shapes.
- Graph API: back edges and `PreviousLoopbackNode<T>`, dedicated executors, Chrome-trace export, subgraphs (code and
  `.pbtxt` via `CalculatorRegistry.RegisterSubgraph`), `ImageEmbedderCalculator`.
- DI registrations and CLI commands (`faces-full`, `segment-*`, `embed`, `audio`, `text`, `--precision`) for the new
  tasks; Gallery pages for interactive segmentation, embeddings, audio and text, and a precision setting.
- API review: PublicApiAnalyzers baselines for every library (unrecorded API changes fail the build).
- Conversion: lowering of `MaxPoolingWithArgmax2D` / `MaxUnpooling2D`, removal of tf2onnx's dynamic-quantization
  emulation, a hand-built language detector; `quantize_models.py`, `generate_quantized_catalog.py`,
  `extract_face_geometry.py`, `generate_golden_v2.py`.

### Fixed
- ONNX Runtime's QDQ fusion quantized MobileBERT's activations on the fly (≈ 7 % embedding error); disabled via
  `session.disable_quant_qdq`.
- `HandLandmarker` image mode did not de-duplicate converging ROIs; tracking state was mutated in image mode.

## 0.1.0 — 2026-09-26

First preview. Created by Gravicode Studios, led by Kang Fadhil.

### Added
- Tasks: `FaceDetector`, `FaceLandmarker` (478 landmarks + 52 blendshapes), `HandLandmarker`, `GestureRecognizer`,
  `PoseLandmarker` (Lite/Full, segmentation mask), `HolisticLandmarker`, `ImageSegmenter` (selfie), `ObjectDetector`
  (EfficientDet-Lite0), `ImageClassifier` (EfficientNet-Lite0); image, video and live-stream modes.
- Graph API: `CalculatorGraph`, `ICalculatorNode`, `Packet<T>`, synchronized/immediate input policies, timestamp
  bounds, `MaxInFlight` flow limiting, bounded queues, side packets, pollers, `.pbtxt` configs, Mermaid export.
- `MPImage`, anti-aliased rotated-ROI `ImageToTensor`, `TensorMapping`, `TensorWarp`, frame sources.
- `OnnxModel` with pooled preallocated I/O; execution providers CPU, DirectML, CUDA, CoreML with automatic fallback.
- Model catalog of 13 ONNX models, `ModelStore` (directory, bundled, cache, HTTP, embedded, NuGet providers,
  SHA-256 verification), `Gravicode.MediaPipeNet.Models.*` content packages.
- `MediaPipeNet.Visualization`, `MediaPipeNet.Video.OpenCv`, `Gravicode.MediaPipeNet.Extensions.DI`, `mediapipenet-cli`.
- Telemetry: meter/activity source `Gravicode.MediaPipeNet`.
- Samples (BasicUsage, GraphApiDemo, Avalonia Gallery), notebook, benchmarks, bilingual documentation.
