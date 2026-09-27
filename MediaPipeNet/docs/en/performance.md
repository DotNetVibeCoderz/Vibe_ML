# Performance & GPUs

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/performa.md)

## Measured numbers

BenchmarkDotNet 0.15.8, .NET 10.0.12, Windows 11, **Intel Core i7-8650U** (2017 ultrabook, 4 cores / 8 threads),
CPU provider, 640×480 input, end-to-end (image-to-tensor, inference, decoding):

| Task | Mean | StdDev | Allocated per call |
|---|---:|---:|---:|
| FaceDetector | **8.06 ms** | 0.14 ms | 3.3 KB |
| ImageSegmenter (selfie) | 9.58 ms | 0.19 ms | 2.9 MB ¹ |
| ImageClassifier | 10.50 ms | 0.25 ms | 58 KB |
| FaceLandmarker | 25.08 ms | 0.39 ms | 18.7 KB |
| ObjectDetector | 30.56 ms | 3.69 ms | 17.2 KB |
| HandLandmarker (image) | 31.44 ms | 1.13 ms | 11.1 KB |
| GestureRecognizer | 31.77 ms | 0.70 ms | 12.3 KB |
| PoseLandmarker (lite) | 36.90 ms | 0.39 ms | 10.7 KB |
| ImageToTensor 720p → 256² letterbox (anti-aliased) | 6.54 ms | | 3.2 KB |
| ImageToTensor 720p rotated ROI → 256² | 1.69 ms | | 3.1 KB |

¹ The output mask itself (640×480 floats).

**NFR-1** (face detection < 50 ms at 640×480 on a modern 8-core CPU): met with a 6× margin on a 4-core 2017 CPU.

### 0.2 / 0.3 tasks (same machine, BenchmarkDotNet short run)

| Task | Mean | Allocated per call |
|---|---:|---:|
| ImageEmbedder (MobileNet V3, 820×1024 photo) | 3.6 ms | 7 KB |
| ImageSegmenter, selfie multiclass + category mask (820×1024) | 104 ms | 22 MB ¹ |
| AudioClassifier, 4.3 s clip (5 YAMNet windows) | 14.0 ms | 256 KB |
| TextClassifier, MobileBERT | 56 ms | 4 KB |
| TextClassifier, average word | 0.024 ms | 2.5 KB |
| TextEmbedder, MobileBERT | 56 ms | 6 KB |
| LanguageDetector | 0.062 ms | 12 KB |

¹ Six full-resolution probability masks plus the category mask; all channels are projected in a single pass
(`TensorWarp.ProjectChannelsToImage`), which made this 1.6× faster than per-channel projection.

### Runtime features (FaceLandmarker on the portrait, CPU)

| Configuration | Mean | Ratio |
|---|---:|---:|
| float32 | 20.0 ms | 1.00 |
| float32 + `UseIoBinding` | 21.0 ms | 1.05 |
| `ModelPrecision.Float16` | 23.7 ms | 1.19 |
| `ModelPrecision.Int8` (weight-only) | 22.8 ms | 1.15 |
| 8 images, sequential loop | 24.3 ms / image | |
| 8 images, `ProcessBatch` | **12.3 ms / image** | 2× throughput |

On a CPU, I/O binding and FP16 do not pay off (they target GPUs, where they avoid host↔device copies and use
half-precision units); INT8 variants trade ~15 % speed for a 3–4× smaller download. `ProcessBatch` doubles offline
throughput on 4 cores.

Run them yourself: `dotnet run -c Release --project benchmarks/MediaPipeNet.Benchmarks -- --filter "*"`, or the
Gallery's *Benchmark* page, or `mediapipenet-cli benchmark faces image.jpg`.

## Why it is fast

- **One session per model**, created once with full graph optimizations (`ORT_ENABLE_ALL`).
- **Preallocated, pooled I/O** — `OnnxModel.RentContext()` returns a context whose input/output tensors are pinned
  `OrtValue`s over reusable `float[]` buffers; steady-state inference allocates nothing on the managed heap.
  Concurrent callers get separate contexts, so image-mode tasks are thread-safe without locks.
- **Single-pass image-to-tensor** — crop, rotate, resize, letterbox and normalize in one unsafe loop, parallelized
  over rows for large tensors, with n×n supersampling on strong downscales (matches MediaPipe more closely than
  plain bilinear).
- **Decode only what passes** — SSD decoding skips anchors below the score threshold before touching box data.
- **Tracking** — in video mode detectors run only when needed.
- **Pooled frames** — `MPImage` rents its pixels from `ArrayPool`; `CopyFrom` refills a frame without allocating;
  `LiveStreamProcessor` double-buffers frames.

## Tuning

| Knob | Effect |
|---|---|
| `RunningMode.Video` / `LiveStream` | Tracking skips detectors: largest win for video. |
| `HandLandmarkerOptions.NumHands = 1` | In video mode the palm detector keeps running while fewer than `NumHands` hands are tracked. |
| `PoseModel.Lite` vs `Full` | Lite is ~2× faster. |
| `InferenceOptions.IntraOpThreads` | Threads per operator. Lower it when running several tasks in parallel (e.g. 2 each). |
| `FaceLandmarkerOptions.OutputFaceBlendshapes = false` | Skips the blendshape model. |
| Input size | Resize huge photos before processing; the models see 128–320 px anyway. |
| `ProcessBatch(images)` / `ClassifyBatch` / `EmbedBatch` | Parallel offline processing on one model instance. |
| `InferenceOptions.Precision` | `Float16` for GPUs, `Int8` for small downloads (see [Models](models.md#precision-variants-fp16-and-int8)). |
| `InferenceOptions.UseIoBinding` | Binds each pooled context's buffers once (ONNX Runtime I/O binding); useful with GPU providers. |
| `ImageSegmenterOptions.OutputConfidenceMasks = false` | Only the category mask when that is all you need. |

## GPUs

| Provider | Package | Platforms |
|---|---|---|
| `Cpu` | `Gravicode.MediaPipeNet` | everywhere |
| `CoreML` | `Gravicode.MediaPipeNet` | macOS / Apple silicon |
| `DirectML` | `Gravicode.MediaPipeNet.DirectML` | Windows, any DirectX 12 GPU |
| `Cuda` | `Gravicode.MediaPipeNet.Cuda` | Windows/Linux, NVIDIA + CUDA 12 + cuDNN 9 |

`ExecutionProvider.Auto` tries CUDA → DirectML → CoreML → CPU, skipping providers the loaded runtime lacks or that
fail to initialize (`FallbackToCpu`). The provider actually chosen is exposed as `OnnxModel.Provider` and tagged on
the `mediapipenet.inference.duration` metric.

Notes from testing:

- These MediaPipe models are small; on an **integrated** GPU (Intel UHD 620) DirectML was slower than the CPU
  (face detection 16.6 ms vs 8.1 ms) because of transfer overhead. Discrete GPUs and batch/background workloads
  benefit more. Measure with the Gallery's Benchmark page.
- DirectML does not support concurrent `Run` calls or concurrent session creation on one device; MediaPipe.NET
  serializes both automatically when the DirectML provider is active.
- The Gallery defaults to the CPU provider; switch provider and precision in *Settings*.
- `UseIoBinding` creates one `OrtIoBinding` per pooled context and reuses it for every run; results are identical
  (`BatchAndBindingTests`).
