# MediaPipe.NET — PLAN / Roadmap

> Created by Gravicode Studios, led by Kang Fadhil · Dibuat oleh Gravicode Studios, dipimpin Kang Fadhil
> Tracking of what is done lives in [Progress.md](Progress.md). The original design is in [DESIGN.md](DESIGN.md).

Legend / Keterangan: ✅ done/selesai · 🟡 partial/sebagian · ⬜ planned/direncanakan

## Milestones (from DESIGN.md) / Milestone

| Phase | Scope (EN) | Cakupan (ID) | Status |
|---|---|---|---|
| **M0 — Foundation** | Core, Imaging, ONNX Runtime wrapper, model conversion pipeline | Core, Imaging, wrapper ONNX Runtime, pipeline konversi model | ✅ 0.1.0 |
| **M1 — Face & Hand** | FaceDetector, FaceLandmarker, HandLandmarker + sample + notebook | FaceDetector, FaceLandmarker, HandLandmarker + sampel + notebook | ✅ |
| **M2 — Pose & Segmentation** | PoseLandmarker, SelfieSegmenter (`ImageSegmenter`), HolisticLandmarker | PoseLandmarker, SelfieSegmenter, HolisticLandmarker | ✅ |
| **M3 — Public Graph API** | Customizable CalculatorGraph, .pbtxt, docs | CalculatorGraph kustom, .pbtxt, dokumentasi | ✅ |
| **M4 — Objects & Classification** | ObjectDetector, ImageClassifier, GestureRecognizer | ObjectDetector, ImageClassifier, GestureRecognizer | ✅ |
| **M5 — Tooling & DX** | CLI, DI, benchmarks, docs site (mkdocs), Gallery | CLI, DI, benchmark, situs dokumentasi, Gallery | ✅ |
| **M6 — v1.0 stabilization** | Cross-platform CI runs, API review, license audit, publish | CI lintas platform, review API, audit lisensi, publikasi | 🟡 CI defined, Windows verified; publish pending |

## Next: 0.2.0 (short term / jangka pendek)

| Item | EN | ID |
|---|---|---|
| ⬜ Publish | Push 0.1.0 packages to nuget.org (enables `NuGetModelProvider` downloads). | Publikasikan paket ke nuget.org (mengaktifkan unduhan model otomatis). |
| ⬜ CI green on Linux/macOS | Run the `ci.yml` matrix; fix platform differences (OpenCV runtime on macOS). | Jalankan matriks CI; perbaiki perbedaan platform. |
| ⬜ Face detector full-range | Add BlazeFace full-range (sparse model, 192×192) for faces farther away. | Tambah BlazeFace full-range untuk wajah jauh. |
| ⬜ Facial transformation matrix | `FaceLandmarker.OutputFacialTransformationMatrixes` (geometry pipeline metadata). | Matriks transformasi wajah 4×4. |
| ⬜ Pose landmark refinement | Heatmap-based refinement (output `Identity_3`) like MediaPipe's `RefineLandmarksFromHeatmap`. | Penyempurnaan landmark dari heatmap. |
| ⬜ Segmentation temporal filter | MediaPipe's segmentation smoothing for pose masks. | Smoothing temporal mask pose. |
| ⬜ Hand re-crop for holistic | `hand_recrop.tflite` for more accurate hand ROIs from the pose. | Model hand re-crop untuk holistic. |
| ⬜ Model quantization | FP16 / INT8 ONNX variants (`ModelCatalog` variants), measured with the benchmark suite. | Varian model FP16/INT8. |
| ⬜ Batching | `DetectBatch(IReadOnlyList<MPImage>)` for offline workloads (dynamic batch dimension). | API batch untuk beban offline. |
| ⬜ IoBinding / GPU I/O | Keep tensors on the device for CUDA/DirectML to cut transfer overhead. | IoBinding untuk mengurangi overhead transfer GPU. |

## Later: 0.3+ (medium term / jangka menengah)

- ⬜ **More vision tasks**: interactive segmenter (MagicTouch), multi-class selfie / hair segmentation, image embedder,
  face stylizer, holistic's dedicated models.
- ⬜ **MediaPipeNet.Tasks.Audio** (roadmap in DESIGN): audio classifier (YAMNet), voice activity detection.
- ⬜ **MediaPipeNet.Tasks.Text**: text classifier, text embedder, language detector.
- ⬜ **Custom model support**: load MediaPipe `.task` bundles directly (on-the-fly conversion or an ONNX sidecar),
  Model Maker-trained gesture/object models.
- ⬜ **Graph API**: back edges (loops) for flow-limiter topologies, per-node executors, graph profiler/visualizer
  export (Chrome trace), subgraphs.
- ⬜ **Blazor WebAssembly** target (onnxruntime-web interop) — out of scope for v1 per DESIGN.
- ⬜ **Mobile**: .NET MAUI sample (Android/iOS) using CoreML/NNAPI providers.

## v1.0 exit criteria / Kriteria rilis 1.0

1. API review done; breaking changes only in a major version afterwards (SemVer, NFR-4).
2. CI green on win-x64, linux-x64, osx-arm64; golden tests within tolerance on all three.
3. Packages published with symbols; documentation site online (EN/ID).
4. License audit: model attribution in every model package (done), third-party notices (done), ImageSharp licensing
   guidance (done).
5. Performance: NFR-1 re-measured on the reference 8-core machine and published.
