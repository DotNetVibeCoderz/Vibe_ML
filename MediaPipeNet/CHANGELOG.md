# Changelog

All notable changes to MediaPipe.NET. Versioning follows [SemVer](https://semver.org).

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
