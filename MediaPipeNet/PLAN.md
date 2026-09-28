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
| **M6 — v1.0 stabilization** | Cross-platform CI runs, API review, license audit, publish | CI lintas platform, review API, audit lisensi, publikasi | ✅ 1.0.0 — all exit criteria below met |

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
- ⛔ → ✅ **Face stylizer**: judged not convertible in 0.3; converted in 1.0 (see below).
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

## 1.0 scope — delivered in 1.0.0 / cakupan 1.0 — dirilis di 1.0.0

| Item | EN | ID |
|---|---|---|
| ✅ Face stylizer | Converted to ONNX (variables frozen, training-mode batch norm lowered, noise kept); `FaceStylizer` with MediaPipe's alignment, golden-tested vs MediaPipe 0.10.21; FP16/INT8. No TFLite fallback needed. | Dikonversi ke ONNX; `FaceStylizer` dengan penyelarasan MediaPipe, diuji golden; FP16/INT8. Tidak perlu fallback TFLite. |
| ✅ MAUI sample + NNAPI | `ExecutionProvider.Nnapi`; `samples/MediaPipeNet.Maui` (Android build verified; iOS needs a Mac); bundled INT8 models. Not yet run on a device. | Provider NNAPI + sampel MAUI (build Android terverifikasi; belum dijalankan di perangkat). |
| ✅ GPU benchmarks | `benchmarks/MediaPipeNet.GpuBenchmark`; DirectML fp32/fp16/IoBinding numbers on the available (integrated) GPU published; found and fixed stale inputs with IoBinding on GPU providers. Discrete-GPU numbers: community. | Benchmark DirectML (GPU terintegrasi) dipublikasikan; bug IoBinding di provider GPU diperbaiki. |
| ⛔ Static INT8 (QDQ) | Evaluated (`evaluate_static_int8.py`): no model reaches cosine 0.99 — needs quantization-aware training. | Dievaluasi: tidak ada yang memenuhi batas — perlu QAT. |
| ✅ Docs site | `mediapipenet-docs.yml` → https://dotnetvibecoderz.github.io/Vibe_ML/mediapipenet/ | Situs dokumentasi di GitHub Pages. |

## After 1.0 / setelah 1.0 (additive, 1.x)

| Item | EN | ID |
|---|---|---|
| ⬜ onnxruntime-web backend | Async inference backend under the tasks for in-browser Blazor WASM (new interface next to `OnnxModel`). | Backend asinkron untuk Blazor WASM di browser. |
| ⬜ Mobile in CI | MAUI workloads + an Android emulator job; device-tested NNAPI numbers. | Mobile di CI + angka NNAPI di perangkat. |
| ⬜ More stylizer styles | Convert other BlazeFaceStylizer styles as Google publishes them (same pipeline). | Gaya stylizer lain. |
| ⬜ Discrete-GPU numbers | DirectML/CUDA on discrete GPUs. | Angka GPU diskret. |

## v1.0 exit criteria / Kriteria rilis 1.0

1. ✅ API review done (PublicApiAnalyzers baselines; RS0016/RS0017 are build errors); breaking changes only in a major
   version afterwards (SemVer, NFR-4).
2. ✅ CI green on win-x64, linux-x64, osx-arm64; golden tests within tolerance on all three.
3. ✅ Packages published with symbols; documentation site online (EN/ID) on GitHub Pages.
4. ✅ License audit: model attribution in every model package, third-party notices, ImageSharp licensing guidance.
5. ✅ Performance: NFR-1 (< 50 ms face detection at 640×480 on an 8-core CPU) met with a 6× margin on a *slower*
   4-core 2017 CPU (8.1 ms); no 8-core machine was available, and a faster CPU can only lower the number.
