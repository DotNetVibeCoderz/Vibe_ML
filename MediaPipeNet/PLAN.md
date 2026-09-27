# MediaPipe.NET — PLAN / Roadmap

> Created by Gravicode Studios, led by Kang Fadhil · Dibuat oleh Gravicode Studios, dipimpin Kang Fadhil
> Tracking of what is done lives in [Progress.md](Progress.md). The original design is in [DESIGN.md](DESIGN.md).

Legend / Keterangan: ✅ done/selesai · 🟡 partial/sebagian · ⬜ planned/direncanakan · ⛔ not feasible now/belum memungkinkan

## Milestones (from DESIGN.md) / Milestone

| Phase | Scope (EN) | Cakupan (ID) | Status |
|---|---|---|---|
| **M0 — Foundation** | Core, Imaging, ONNX Runtime wrapper, model conversion pipeline | Core, Imaging, wrapper ONNX Runtime, pipeline konversi model | ✅ 0.1.0 |
| **M1 — Face & Hand** | FaceDetector, FaceLandmarker, HandLandmarker + sample + notebook | FaceDetector, FaceLandmarker, HandLandmarker + sampel + notebook | ✅ |
| **M2 — Pose & Segmentation** | PoseLandmarker, SelfieSegmenter (`ImageSegmenter`), HolisticLandmarker | PoseLandmarker, SelfieSegmenter, HolisticLandmarker | ✅ |
| **M3 — Public Graph API** | Customizable CalculatorGraph, .pbtxt, docs | CalculatorGraph kustom, .pbtxt, dokumentasi | ✅ |
| **M4 — Objects & Classification** | ObjectDetector, ImageClassifier, GestureRecognizer | ObjectDetector, ImageClassifier, GestureRecognizer | ✅ |
| **M5 — Tooling & DX** | CLI, DI, benchmarks, docs site (mkdocs), Gallery | CLI, DI, benchmark, situs dokumentasi, Gallery | ✅ |
| **M6 — v1.0 stabilization** | Cross-platform CI runs, API review, license audit, publish | CI lintas platform, review API, audit lisensi, publikasi | 🟡 CI green on 3 OSes, 0.1.0 and 0.3.0 published, **API review done** (PublicApiAnalyzers baselines); 1.0 criteria below remain |

## 0.2.0 scope — delivered in 0.3.0 / cakupan 0.2.0 — dirilis di 0.3.0

| Item | EN | ID |
|---|---|---|
| ✅ Publish | 0.1.0 on nuget.org as `Gravicode.MediaPipeNet.*`. | 0.1.0 terbit di nuget.org. |
| ✅ CI green on Linux/macOS | `mediapipenet-ci.yml` passes on Windows, Ubuntu and macOS. | Matriks CI lulus di ketiga OS. |
| ✅ Face detector full-range | `FaceDetectorModel.FullRange` (192×192, 2304 anchors), golden-tested. | BlazeFace full-range, diuji golden. |
| ✅ Facial transformation matrix | Port of MediaPipe's geometry pipeline + weighted Procrustes; within 2° / 1.5 cm of MediaPipe. | Matriks transformasi wajah 4×4, selisih 2° / 1,5 cm. |
| ✅ Pose landmark refinement | Heatmap refinement: pose error 0.0065 → 0.0024. | Penyempurnaan dari heatmap: error 0,0065 → 0,0024. |
| ✅ Segmentation temporal filter | MediaPipe's segmentation smoothing for pose masks and `ImageSegmenter`. | Smoothing temporal mask. |
| ✅ Hand re-crop for holistic | Holistic's `hand_roi_refinement` model + pose-derived ROIs + ROI tracking + world alignment. | Model hand ROI refinement di holistic. |
| ✅ Model quantization | `Models.Quantized`: 14 FP16 + 9 INT8 (weight-only) variants, each validated; `InferenceOptions.Precision`. | Varian FP16/INT8 tervalidasi. |
| ✅ Batching | `ProcessBatch` / `ProcessBatchAsync` (+ `ClassifyBatch`, `EmbedBatch`): 2× offline throughput. | API batch, throughput 2×. |
| ✅ IoBinding | `InferenceOptions.UseIoBinding` (per pooled context); tested, benchmarked (no CPU gain, for GPUs). | IoBinding untuk GPU. |

## 0.3 scope — delivered in 0.3.0 / cakupan 0.3 — dirilis di 0.3.0

- ✅ **More vision tasks**: `InteractiveSegmenter` (MagicTouch), multiclass selfie / hair / DeepLab v3 segmentation with
  category masks, `ImageEmbedder`, holistic's dedicated hand ROI refinement model.
- ⛔ **Face stylizer**: not convertible (resource variables, Flex `FusedBatchNormV3`, random ops) and absent from the
  MediaPipe Python package — see [docs/en/platforms.md](docs/en/platforms.md#face-stylizer).
- ✅ **MediaPipeNet.Tasks.Audio**: `AudioClassifier` (YAMNet, clip + stream), `VoiceActivityDetector`, WAV/resampling.
- ✅ **MediaPipeNet.Tasks.Text**: `TextClassifier` (BERT + average word), `TextEmbedder`, `LanguageDetector` (110 languages).
- ✅ **Custom model support**: `convert_models.py --custom` (`.tflite` and `.task`), `ModelPath` / `Labels` /
  `ClassifierModelPath` / `VocabularyPath` for Model Maker classifiers, detectors, gestures, audio and text.
- ✅ **Graph API**: back edges + `PreviousLoopbackNode<T>`, per-node executors, Chrome trace export, subgraphs (code and
  `.pbtxt`).
- 🟡 **Blazor WebAssembly**: evaluated — server-side inference works today; in-browser needs an onnxruntime-web backend
  (after 1.0). See [platforms](docs/en/platforms.md).
- 🟡 **Mobile (.NET MAUI)**: evaluated — compiles and the steps are documented; NNAPI provider option and a MAUI sample
  still to do.

## Next: toward 1.0 / berikutnya: menuju 1.0

| Item | EN | ID |
|---|---|---|
| ⬜ MAUI sample + NNAPI | `ExecutionProvider.Nnapi`, Android/iOS sample, model assets. | Sampel MAUI + provider NNAPI. |
| ⬜ Static INT8 (QDQ) | Calibrated activation quantization for the CNNs that reject weight-only/dynamic INT8. | Kuantisasi INT8 statis terkalibrasi. |
| ⬜ GPU benchmarks | Publish DirectML/CUDA numbers (FP16, IoBinding) on discrete GPUs. | Benchmark GPU diskret. |
| ⬜ onnxruntime-web backend | `IInferenceBackend` abstraction for Blazor WASM. | Backend onnxruntime-web untuk Blazor WASM. |
| ⬜ Face stylizer | Revisit when a convertible model is published. | Tinjau ulang bila model yang dapat dikonversi tersedia. |

## v1.0 exit criteria / Kriteria rilis 1.0

1. ✅ API review done (PublicApiAnalyzers baselines; RS0016/RS0017 are build errors); breaking changes only in a major
   version afterwards (SemVer, NFR-4).
2. ✅ CI green on win-x64, linux-x64, osx-arm64; golden tests within tolerance on all three.
3. 🟡 Packages published with symbols (done); documentation site online (EN/ID) — mkdocs config ready, hosting pending.
4. ✅ License audit: model attribution in every model package, third-party notices, ImageSharp licensing guidance.
5. 🟡 Performance: NFR-1 re-measured on the reference 8-core machine and published (measured on a 4-core 2017 CPU so far).
