# Testing & validation

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/pengujian.md)

```bash
dotnet test MediaPipeNet.slnx -c Release
dotnet test MediaPipeNet.slnx -c Release --collect:"XPlat Code Coverage"
```

## Test projects (159 tests)

| Project | Covers |
|---|---|
| `MediaPipeNet.Core.Tests` (44) | Timestamps, geometry, JSON, telemetry; `MPImage` formats/strides/pooling; image-to-tensor (normalization, letterbox, rotation, replicate border, anti-aliasing) and mapping round-trips; mask projection; frame sources; ONNX sessions, pooled contexts and concurrency; provider selection and fallback; model catalog (all 25 files verified by SHA-256), checksum verification, NuGet/HTTP/embedded providers. |
| `MediaPipeNet.Framework.Tests` (31) | Packets; validation; ordering; input synchronization and bound propagation; flow limiting and queue dropping; side packets; pollers; failures and cancellation; immediate policy; `.pbtxt` parsing; **back edges and loopback, executors, Chrome tracing, subgraphs (code and .pbtxt)**. |
| `MediaPipeNet.Tasks.Tests` (84) | **Golden cross-validation** of every task against MediaPipe Python (v1 and v2 references); precision variants; custom models; batches, I/O binding and thread safety; audio/text tokenizers, WAV loading and resampling; heatmap refinement, segmentation smoothing, ROI tracking; running-mode rules, live stream, tracking, DI, visualization, graph calculators. |

## Coverage (line)

| Module | Line | Branch |
|---|---:|---:|
| MediaPipeNet.Core | 99 % | 76 % |
| MediaPipeNet.Imaging | 96 % | 86 % |
| MediaPipeNet.Inference | 91 % | 71 % |
| MediaPipeNet.Framework | 94 % | 84 % |
| MediaPipeNet.Tasks.Core | 97 % | 62 % |
| MediaPipeNet.Tasks.Vision | 91 % | 77 % |
| MediaPipeNet.Tasks.Audio | 83 % | 62 % |
| MediaPipeNet.Tasks.Text | 89 % | 75 % |
| MediaPipeNet.Visualization | 81 % | 67 % |
| MediaPipeNet.Extensions.DI | 100 % | 88 % |

NFR-3 (≥ 70 % for core modules) is met.

## The three levels of validation

1. **Model conversion** — `tools/model-conversion/validate_models.py` feeds identical random inputs to each
   original TFLite graph and its ONNX conversion: max |Δ| ≈ 1e-4 (`models/onnx/validation.json`).
2. **Cross-validation with MediaPipe Python** — `tools/golden/generate_golden.py` and `generate_golden_v2.py` run the
   official `mediapipe` 1.0.1 package on the fixture images, audio and sentences and store the results in
   `tests/assets/golden/mediapipe_python_reference.json` and `…_v2.json` (full-range faces, the facial transformation
   matrix, holistic, image embeddings, multiclass/hair/DeepLab segmentation, audio classification, text
   classification and embedding, language detection). `GoldenTests` compare MediaPipe.NET against it with
   tolerances (box IoU > 0.9, landmark mean error < 0.006–0.02, identical gesture/object/class labels, mask mean
   within 0.01).
3. **Streaming behavior** — tests feed frame sequences to verify timestamp monotonicity, tracking and packet
   dropping in live-stream mode.

The v2 references pinned down: DeepLab's sensitivity to resampling (MediaPipe's CPU preprocessing does not
anti-alias, so the segmenters do not either), the hair segmenter's empty alpha channel, and — through a
tensor-by-tensor comparison with the TFLite interpreter — tf2onnx's dynamic-quantization emulation and ONNX Runtime's
QDQ fusion that had shifted MobileBERT's embeddings. Earlier references caught: EfficientDet's anchor scale (3, not 4), a double
sigmoid on detector scores, the missing rotation-free normalization of gesture inputs, and an uninitialized
densified tensor in the pose detector conversion.

## Regenerating the golden data

```bash
python -m venv C:\mpp && C:\mpp\Scripts\pip install mediapipe
C:\mpp\Scripts\python tools/golden/generate_golden.py --models artifacts/models/_work
C:\mpp\Scripts\python tools/golden/generate_golden_v2.py --models artifacts/models/_work
```

Sections the Python package cannot run (the interactive segmenter and face stylizer are missing from 1.0.1) are
listed under `"unavailable"`; the interactive segmenter is validated behaviorally instead (the mask must cover the
object under the point and little else).

## Benchmarks

`benchmarks/MediaPipeNet.Benchmarks` (BenchmarkDotNet) — see [Performance](performance.md). CI runs it on
release tags.
