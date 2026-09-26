# Testing & validation

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/pengujian.md)

```bash
dotnet test MediaPipeNet.slnx -c Release
dotnet test MediaPipeNet.slnx -c Release --collect:"XPlat Code Coverage"
```

## Test projects (104 tests)

| Project | Covers |
|---|---|
| `Gravicode.MediaPipeNet.Core.Tests` (44) | Timestamps, geometry, JSON, telemetry; `MPImage` formats/strides/pooling; image-to-tensor (normalization, letterbox, rotation, replicate border, anti-aliasing) and mapping round-trips; mask projection; frame sources; ONNX sessions, pooled contexts and concurrency; provider selection and fallback; model catalog, checksum verification, NuGet/HTTP/embedded providers (with a stubbed HTTP handler). |
| `Gravicode.MediaPipeNet.Framework.Tests` (23) | Packets; validation (duplicate producers, unknown ports/streams, type mismatch, cycles, missing side packets); ordering; input synchronization and bound propagation; flow limiting and queue dropping; side packets; pollers; failures and cancellation; immediate policy; `.pbtxt` parsing, round-trip, errors; registry. |
| `Gravicode.MediaPipeNet.Tasks.Tests` (37) | **Golden cross-validation** of all nine tasks against MediaPipe Python; running-mode rules; live-stream callbacks and dropping; video tracking; async APIs, ROI/rotation, allow/deny lists; holistic hand assignment; JSON round-trips; DI registration; `LiveStreamProcessor`; vision calculators in a `.pbtxt` graph; visualization; anchors, decoding, NMS, ROI math, One-Euro filter, labels. |

## Coverage (line)

| Module | Line | Branch |
|---|---:|---:|
| MediaPipeNet.Core | 94 % | 69 % |
| MediaPipeNet.Imaging | 94 % | 89 % |
| MediaPipeNet.Inference | 91 % | 70 % |
| MediaPipeNet.Framework | 94 % | 85 % |
| MediaPipeNet.Tasks.Vision | 92 % | 78 % |
| MediaPipeNet.Visualization | 99 % | 78 % |
| MediaPipeNet.Extensions.DI | 84 % | 75 % |

NFR-3 (≥ 70 % for core modules) is met.

## The three levels of validation

1. **Model conversion** — `tools/model-conversion/validate_models.py` feeds identical random inputs to each
   original TFLite graph and its ONNX conversion: max |Δ| ≈ 1e-4 (`models/onnx/validation.json`).
2. **Cross-validation with MediaPipe Python** — `tools/golden/generate_golden.py` runs the official
   `mediapipe` 1.0.1 package on the fixture images and stores the results in
   `tests/assets/golden/mediapipe_python_reference.json`. `GoldenTests` compare MediaPipe.NET against it with
   tolerances (box IoU > 0.9, landmark mean error < 0.006–0.02, identical gesture/object/class labels, mask mean
   within 0.01).
3. **Streaming behavior** — tests feed frame sequences to verify timestamp monotonicity, tracking and packet
   dropping in live-stream mode.

These references caught real porting bugs during development: EfficientDet's anchor scale (3, not 4), a double
sigmoid on detector scores, the missing rotation-free normalization of gesture inputs, and an uninitialized
densified tensor in the pose detector conversion.

## Regenerating the golden data

```bash
python -m venv C:\mpp && C:\mpp\Scripts\pip install mediapipe
C:\mpp\Scripts\python tools/golden/generate_golden.py --models artifacts/models/_work
```

## Benchmarks

`benchmarks/MediaPipeNet.Benchmarks` (BenchmarkDotNet) — see [Performance](performance.md). CI runs it on
release tags.
