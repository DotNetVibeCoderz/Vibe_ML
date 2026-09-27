# MediaPipe.NET — Progress

> Development tracking · Pelacakan pengembangan — Gravicode Studios, led by Kang Fadhil
> Roadmap: [PLAN.md](PLAN.md)

**Current version:** `0.3.0` · **Status:** 0.1.0 and 0.3.0 released on nuget.org as `Gravicode.MediaPipeNet.*`; the whole 0.2 and 0.3 roadmap is delivered; CI green on Windows, Linux and macOS.

## Requirements status / Status requirement

### Functional

| ID | Requirement | Status | Where / Di mana |
|---|---|---|---|
| FR-1 | Load ONNX models converted from MediaPipe and run inference | ✅ | `tools/model-conversion`, `OnnxModel` — 25 models (+ 23 FP16/INT8 variants), validated vs TFLite and MediaPipe |
| FR-2 | Task API: face detection, face mesh, hands, pose, selfie segmentation, object detection | ✅ + extra | 16 tasks: + gestures, holistic, image classification, interactive segmentation, image embedding, audio classification, VAD, text classification, text embedding, language detection |
| FR-3 | Inputs: image files, byte[]/Stream, `Image<Rgba32>`, video via `IFrameSource` | ✅ | `MPImage`, `ImageFileFrameSource`, `MemoryFrameSource`, `WebcamFrameSource`, `VideoFileFrameSource` |
| FR-4 | Graph API: `ICalculatorNode` + `Packet<T>` streams | ✅ | `MediaPipeNet.Framework` |
| FR-5 | Streaming with timestamps and packet dropping | ✅ | Timestamp bounds, `MaxInFlight`, queue drop, `LiveStreamProcessor` |
| FR-6 | Strongly-typed, JSON-serializable results | ✅ | `*Result` records, `MediaPipeJson` |
| FR-7 | Visualization utilities | ✅ | `MediaPipeNet.Visualization` |
| FR-8 | CPU fallback, automatic execution provider | ✅ | `ExecutionProviderSelector` (CUDA → DirectML → CoreML → CPU) |
| FR-9 | Model management: download, cache, NuGet model packages | ✅ | `ModelStore`, `Gravicode.MediaPipeNet.Models.*` (downloads activate once published) |
| FR-10 | Sync and async APIs | ✅ | `Detect`/`DetectAsync`, `CreateAsync` |

### Non-functional

| ID | Requirement | Status | Evidence / Bukti |
|---|---|---|---|
| NFR-1 | Face detection < 50 ms @ 640×480 on modern 8-core CPU | ✅ | 8.1 ms on a 2017 4-core i7-8650U (BenchmarkDotNet) |
| NFR-2 | Windows / Ubuntu / macOS without code changes | ✅ | CI green on windows-latest, ubuntu-latest, macos-latest (all tests incl. golden) |
| NFR-3 | ≥ 70 % coverage for Core and Tasks | ✅ | Core 99 %, Imaging 96 %, Inference 91 %, Framework 94 %, Tasks.Core 97 %, Tasks.Vision 91 %, Tasks.Audio 83 %, Tasks.Text 89 % |
| NFR-4 | Semantic versioning | ✅ | `VersionPrefix`/`VersionSuffix`; public API tracked by PublicApiAnalyzers (API review, M6) |
| NFR-5 | `Span<T>`/`Memory<T>`, low GC pressure | ✅ | Pooled frames, pooled I/O contexts; 3–18 KB per inference |
| NFR-6 | `ILogger` + metrics via `System.Diagnostics.Metrics` | ✅ | Meter/ActivitySource `Gravicode.MediaPipeNet` |
| NFR-7 | Model license attribution | ✅ | `NOTICE`, `ModelDescriptor.Attribution`, package descriptions |
| NFR-8 | XML docs on public API, bilingual documentation | ✅ | `GenerateDocumentationFile`, `docs/en`, `docs/id` |

## Deliverables / Hasil kerja

- [x] Solution `MediaPipeNet.slnx` — 15 source projects, 11 model packages, 3 samples, 3 test projects, benchmarks
- [x] Model conversion + validation tooling (`tools/model-conversion`)
- [x] Golden references from official MediaPipe Python (`tools/golden`, `tests/assets/golden`)
- [x] 159 tests (unit, golden cross-validation v1 + v2, precision variants, custom models, streaming)
- [x] Samples: `BasicUsage`, `GraphApiDemo`, `MediaPipeNet.Gallery` (Avalonia; ID/EN; light/dark; settings; code per case)
- [x] CLI `mediapipenet-cli` (dotnet tool, installed and smoke-tested from the local feed)
- [x] Polyglot notebook `notebooks/MediaPipeNet_QuickStart.ipynb`
- [x] Documentation EN/ID (16 pages each), README EN/ID, screenshots (23 + examples)
- [x] NuGet packages build (`dotnet pack`) and consume correctly (verified in a fresh project)
- [x] GitHub Actions (Vibe_ML root): `mediapipenet-ci.yml` (win/linux/macOS build, test, coverage, pack) and
      `mediapipenet-release.yml` (tag `mediapipenet-v*` → test, pack, push to nuget.org, GitHub Release)
- [x] Moved into the Vibe_ML monorepo (`MediaPipeNet/`); NuGet ids `Gravicode.MediaPipeNet.*`
- [x] Published 0.1.0 to nuget.org via `mediapipenet-release.yml` (19 packages + symbols), GitHub Release `mediapipenet-v0.1.0`
- [x] CI green on Windows, Linux and macOS
- [x] 0.3.0: all 0.2 and 0.3 roadmap items (see [PLAN.md](PLAN.md)), released via tag `mediapipenet-v0.3.0`

## Log

### 2026-09-28 — 0.3.0 (0.2 + 0.3 roadmap)

- **Models**: 12 new conversions (full-range BlazeFace, holistic hand ROI refinement, selfie multiclass, hair, DeepLab v3,
  MagicTouch, MobileNet V3 embedder, YAMNet, MobileBERT classifier/embedder, average-word classifier, language detector).
  New lowering of `MaxPoolingWithArgmax2D`/`MaxUnpooling2D`; the language detector rebuilt by hand (C# n-gram hashing +
  dense `KmeansEmbeddingLookup`). Face stylizer not convertible (documented).
- **Golden v2** from MediaPipe Python 1.0.1 (`generate_golden_v2.py`) for every new feature.
- **Findings while validating**: tf2onnx emulates TFLite's hybrid `asymmetric_quantize_inputs` with uint8 per-tensor
  `DynamicQuantizeLinear` on weights and bias (stripped), and ONNX Runtime's QDQ fusion quantized MobileBERT's
  activations (disabled with `session.disable_quant_qdq`) — found by a tensor-by-tensor diff against the TFLite
  interpreter; MobileBERT now matches float TFLite at cosine 1.0000. DeepLab is on a dog/horse decision boundary for the
  fixture and is sensitive to anti-aliasing: segmenters now sample like MediaPipe's CPU path. Full-range faces of ~10
  tensor pixels jump between neighbouring anchors (verified in Python).
- **Vision**: facial transformation matrix (geometry pipeline + Procrustes, SVD by one-sided Jacobi), pose heatmap
  refinement (2.7× closer to MediaPipe), segmentation smoothing, holistic rewritten after MediaPipe's graph (hand ROI
  refinement, ROI tracking, world alignment; face fallback kept for tiny faces), generalized `ImageSegmenter`,
  `InteractiveSegmenter`, `ImageEmbedder`, batch API, image-mode thread-safety fixes.
- **Audio / Text**: new packages; MediaPipe-exact BERT/regex tokenizers and n-gram hashing; language probabilities
  within 0.001 of MediaPipe.
- **Runtime**: int tensors, IoBinding, `RunDynamic`, precision variants with validated FP16/INT8 files
  (`quantize_models.py`; dynamic activation INT8 rejected — destroys landmark CNNs), `Tasks.Core` layer.
- **Graph API**: back edges, executors, Chrome trace, subgraphs.
- **API review**: PublicApiAnalyzers baselines for 11 libraries; `tools/update_public_api.py`.
- **Custom models**: `--custom` conversion and `ModelPath`/`Labels` on five tasks.
- **Gallery**: interactive segmentation, embeddings, audio (waveform + VAD) and text pages, model/precision options,
  head pose; screenshots regenerated. CLI: `faces-full`, `segment-*`, `embed`, `audio`, `text`, `--precision`.
- Evaluated MAUI and Blazor WASM (documented in `docs/*/platform*.md`).

### 2026-09-26 — 0.1.0

- Converted 13 MediaPipe models (TFLite → ONNX): custom `Convolution2DTransposeBias` lowered to `ConvTranspose`;
  BlazePose sparse weights densified (fixed: tensors must be read after the first `invoke`); XNNPACK delegate
  disabled during conversion.
- Implemented Core, Imaging (anti-aliased rotated-ROI image-to-tensor), Inference (pooled I/O, provider selection,
  model store with SHA-256 and NuGet download), Framework (graph engine, `.pbtxt`), 9 vision tasks, visualization,
  video sources, DI, CLI.
- Cross-validation against MediaPipe Python found and fixed: EfficientDet anchor scale (3), double sigmoid on
  detector scores, gesture input normalization (no ROI rotation), pose presence already a probability.
- Graph tests found a `Send<T>` overload recursion (renamed the untyped overload `SendPacket`).
- DirectML: serialized `Run` and session creation (device hang with concurrent sessions); Gallery defaults to CPU
  (faster than DirectML on integrated GPUs).
- ImageSharp pinned to 3.1.x (4.x requires a license key).
- Renamed `ImageFrame` → `MPImage` to avoid a clash with `SixLabors.ImageSharp.ImageFrame`.
- Gallery: "optics lab" design; screenshot automation (`--screenshots`), fixed shell rebuild on language/theme change.
