# Arsitektur

> 🇬🇧 [Read in English](../en/architecture.md)

## Lapisan

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Tasks API   MediaPipeNet.Tasks.Vision                                         │
│   FaceDetector · FaceLandmarker · HandLandmarker · GestureRecognizer ·        │
│   PoseLandmarker · HolisticLandmarker · ImageSegmenter · InteractiveSegmenter │
│   ImageEmbedder · ObjectDetector · ImageClassifier · LiveStreamProcessor<T>   │
│             MediaPipeNet.Tasks.Audio   AudioClassifier · VoiceActivityDetector│
│             MediaPipeNet.Tasks.Text    TextClassifier · TextEmbedder ·        │
│                                        LanguageDetector · tokenizer           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Fondasi task  MediaPipeNet.Tasks (Tasks.Core)                                 │
│   BaseOptions · ModelLoader (varian presisi) · ClassifierOptions ·            │
│   CustomModels                                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ Graph API   MediaPipeNet.Framework                                            │
│   CalculatorGraph · ICalculatorNode · Packet<T> · GraphConfig (.pbtxt) ·      │
│   back edge · executor · subgraph · Chrome trace                             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Inferensi   MediaPipeNet.Inference                                            │
│   OnnxModel / InferenceContext (IoBinding, tensor int, RunDynamic) ·          │
│   ExecutionProviderSelector · ModelStore · ModelCatalog (+ FP16/INT8)         │
├──────────────────────────────────────────────────────────────────────────────┤
│ Imaging & tensor   MediaPipeNet.Imaging                                       │
│   MPImage · ImageToTensor · TensorMapping · TensorWarp · IFrameSource          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Core        MediaPipeNet.Core                                                 │
│   Timestamp · NormalizedLandmark · Detection · Embedding · telemetri          │
└──────────────────────────────────────────────────────────────────────────────┘
  Tambahan: Visualization (ImageSharp.Drawing) · Video.OpenCv · Extensions.DI · Cli
  Meta package: MediaPipeNet (CPU) · MediaPipeNet.DirectML · MediaPipeNet.Cuda
```

Task audio dan teks hanya bergantung pada `Tasks.Core` dan `Inference` — tanpa stack imaging, tanpa ImageSharp.

Dependensi hanya mengarah ke bawah. `Inference` hanya mereferensikan API *managed* ONNX Runtime; varian runtime
native ditentukan oleh meta package yang direferensikan aplikasi, sehingga build CPU, DirectML, dan CUDA tidak
pernah bentrok.

## Satu task, langkah demi langkah (HandLandmarker)

```mermaid
flowchart LR
    A[MPImage] --> B[ImageToTensor<br/>letterbox 192² · rentang 0..1]
    B --> C[OnnxModel<br/>palm_detection]
    C --> D[DetectionDecoder<br/>2016 anchor SSD]
    D --> E[Weighted NMS]
    E --> F[RoiCalculator<br/>rotasi pergelangan→MCP tengah · ×2,6]
    F --> G[ImageToTensor<br/>ROI berotasi 224²]
    G --> H[OnnxModel<br/>hand_landmarks_detector]
    H --> I[TensorMapping.TensorToImage<br/>+ rotasi world]
    I --> J[HandLandmarkResult]
    I -. mode video .-> K[RoiCalculator.FromHandLandmarks<br/>ROI frame berikutnya]
    K -.-> G
```

Setiap kotak adalah kelas publik yang bisa dipakai ulang di `MediaPipeNet.Tasks.Vision.Processing` /
`MediaPipeNet.Imaging`, di-port dari kalkulator MediaPipe dengan fungsi yang sama:

| Kalkulator MediaPipe | MediaPipe.NET |
|---|---|
| `ImageToTensorCalculator` | `ImageToTensor.Convert` → `TensorMapping` |
| `SsdAnchorsCalculator` | `SsdAnchors.Generate` (+ `GenerateEfficientDet`) |
| `TensorsToDetectionsCalculator` | `DetectionDecoder.Decode` |
| `NonMaxSuppressionCalculator` | `NonMaxSuppression.Weighted` / `.Hard` |
| `DetectionLetterboxRemovalCalculator`, `LandmarkProjectionCalculator` | `RawDetection.MapToImage`, `TensorMapping.TensorToImage` |
| `DetectionsToRectsCalculator`, `AlignmentPointsRectsCalculator`, `RectTransformationCalculator` | `RoiCalculator.FromDetection`, `.FromAlignmentPoints`, `.Transform` |
| `HandLandmarksToRectCalculator` | `RoiCalculator.FromHandLandmarks` |
| `LandmarksSmoothingCalculator` (One-Euro) | `LandmarkSmoother`, `OneEuroFilter` |
| `WorldLandmarkProjectionCalculator` | rotasi landmark world sebesar sudut ROI |
| `LandmarksToMatrixCalculator` | persiapan input `GestureRecognizer` |
| `FlowLimiterCalculator` | `GraphOptions.MaxInFlight` |

`TensorMapping` adalah abstraksi kuncinya: ia mencatat ROI berotasi (termasuk padding letterbox) yang di-sampling,
sehingga setiap output (keypoint, landmark, mask) dapat dipetakan kembali ke koordinat gambar dengan satu panggilan —
di ruang piksel, agar rotasi tetap benar pada gambar yang tidak persegi.

## Running mode

`VisionTaskBase<TResult>` mengimplementasikan ketiga mode sekali untuk semua task:

- **Image** — `Process` dengan `tracking: false`; tanpa state; thread-safe lewat context inferensi yang di-pool.
- **Video** — `Process` dengan `tracking: true` di bawah lock; timestamp monoton diwajibkan.
- **LiveStream** — task berjalan di dalam `CalculatorGraph` satu-node dengan `MaxInFlight`; frame hanya di-clone bila
  diterima; hasil dikirim ke `ResultCallback`.

## Graph engine

`CalculatorGraph` menyimpan antrean input dan timestamp bound untuk setiap input node. Keputusan penjadwalan diambil
di bawah satu lock graph (murah — graph vision hanya membawa puluhan packet per detik), sedangkan kode node berjalan
di luar lock pada thread pool. Node dapat berjalan ketika timestamp kandidat terkecilnya sudah "settled" di setiap
input (`DefaultInputStreamHandler` MediaPipe); setelah `Process`, output yang tidak mengirim memajukan bound-nya
sehingga hilir tidak pernah macet. Output graph dapat diamati (callback) atau di-poll (`Channel`). Lihat
[Graph API](graph-api.md).

## Struktur solusi

```
MediaPipeNet.slnx
├── src/        proyek library + meta package + CLI
├── models/     onnx/ + paket konten Gravicode.MediaPipeNet.Models.*
├── samples/    BasicUsage · GraphApiDemo · MediaPipeNet.Gallery (Avalonia 11)
├── tests/      Core.Tests · Framework.Tests · Tasks.Tests (+ assets/images, assets/golden)
├── benchmarks/ MediaPipeNet.Benchmarks
├── notebooks/  MediaPipeNet_QuickStart.ipynb
├── tools/      model-conversion · golden · notebook · packaging
└── docs/       en · id · images
```

Pengaturan build tersentral di `Directory.Build.props` (net10.0, nullable, analyzer, metadata paket) dan
`Directory.Packages.props` (versi paket terpusat; ONNX Runtime dikunci ke satu versi untuk CPU/DirectML/CUDA).

## Kalkulator MediaPipe yang di-port (0.2 / 0.3)

| MediaPipe | MediaPipe.NET |
|---|---|
| `GeometryPipeline` + `ProcrustesSolver` (geometri wajah) | `Processing.FaceGeometry` (mesh kanonis diekstrak dari `face_landmarker.task` ke `Resources/face_geometry.bin`) |
| `RefineLandmarksFromHeatmapCalculator` | `Processing.HeatmapRefinement` |
| `SegmentationSmoothingCalculator` | `Processing.SegmentationSmoother` |
| `RoiTrackingCalculator` | `Processing.RoiTracking` |
| `HandDetectionsFromPoseToRectsCalculator`, `HandRoiRefinementGraph`, `AlignHandToPoseInWorldCalculator` | `RoiCalculator.FromPosePalm`, `HolisticLandmarker` |
| `TensorsToSegmentationCalculator` (softmax, category mask) | `ImageSegmenter`, `CategoryMask.FromConfidenceMasks` |
| `TensorsToEmbeddingsCalculator` | `Embedding.FromTensor` (normalisasi L2, kuantisasi skalar) |
| `AudioToTensorCalculator` (jendela, resampling) | `AudioData.Resample`, `AudioStreamBuffer` |
| `BertPreprocessorCalculator`, `RegexPreprocessorCalculator` | `Tokenizers.BertTokenizer`, `Tokenizers.RegexTokenizer` |
| `NGramHash` (custom op TFLite) | `Tokenizers.NGramHasher` |
| `PreviousLoopbackCalculator` | `Nodes.PreviousLoopbackNode<T>` |

