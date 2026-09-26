# MediaPipe.NET — Progress

> Development tracking · Pelacakan pengembangan — Gravicode Studios, led by Kang Fadhil
> Roadmap: [PLAN.md](PLAN.md)

**Current version:** `0.1.0` · **Status:** feature-complete preview; all tests green on Windows 11 / .NET 10.0.12.

## Requirements status / Status requirement

### Functional

| ID | Requirement | Status | Where / Di mana |
|---|---|---|---|
| FR-1 | Load ONNX models converted from MediaPipe and run inference | ✅ | `tools/model-conversion`, `OnnxModel` — 13 models, validated vs TFLite (max Δ ≈ 1e-4) |
| FR-2 | Task API: face detection, face mesh, hands, pose, selfie segmentation, object detection | ✅ + extra | 9 tasks: + gestures, holistic, image classification |
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
| NFR-2 | Windows / Ubuntu / macOS without code changes | 🟡 | Portable code, CI matrix defined; verified on Windows 11 |
| NFR-3 | ≥ 70 % coverage for Core and Tasks | ✅ | Core 94 %, Imaging 94 %, Inference 91 %, Framework 94 %, Tasks.Vision 92 % |
| NFR-4 | Semantic versioning | ✅ | `VersionPrefix`/`VersionSuffix`, preview tag |
| NFR-5 | `Span<T>`/`Memory<T>`, low GC pressure | ✅ | Pooled frames, pooled I/O contexts; 3–18 KB per inference |
| NFR-6 | `ILogger` + metrics via `System.Diagnostics.Metrics` | ✅ | Meter/ActivitySource `Gravicode.MediaPipeNet` |
| NFR-7 | Model license attribution | ✅ | `NOTICE`, `ModelDescriptor.Attribution`, package descriptions |
| NFR-8 | XML docs on public API, bilingual documentation | ✅ | `GenerateDocumentationFile`, `docs/en`, `docs/id` |

## Deliverables / Hasil kerja

- [x] Solution `MediaPipeNet.slnx` — 12 source projects, 7 model packages, 3 samples, 3 test projects, benchmarks
- [x] Model conversion + validation tooling (`tools/model-conversion`)
- [x] Golden references from official MediaPipe Python (`tools/golden`, `tests/assets/golden`)
- [x] 104 tests (unit, golden cross-validation, streaming)
- [x] Samples: `BasicUsage`, `GraphApiDemo`, `MediaPipeNet.Gallery` (Avalonia; ID/EN; light/dark; settings; code per case)
- [x] CLI `mediapipenet-cli` (dotnet tool, installed and smoke-tested from the local feed)
- [x] Polyglot notebook `notebooks/MediaPipeNet_QuickStart.ipynb`
- [x] Documentation EN/ID (13 pages each), README EN/ID, screenshots (19)
- [x] NuGet packages build (`dotnet pack`) and consume correctly (verified in a fresh project)
- [x] GitHub Actions (Vibe_ML root): `mediapipenet-ci.yml` (win/linux/macOS build, test, coverage, pack) and
      `mediapipenet-release.yml` (tag `mediapipenet-v*` → test, pack, push to nuget.org, GitHub Release)
- [x] Moved into the Vibe_ML monorepo (`MediaPipeNet/`); NuGet ids `Gravicode.MediaPipeNet.*`
- [ ] Publish 0.1.0 to nuget.org (release workflow)
- [ ] CI runs on Linux/macOS

## Log

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
