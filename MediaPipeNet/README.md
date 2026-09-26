<p align="center">
  <img src="build/icon-128.png" width="96" alt="MediaPipe.NET logo" />
</p>

<h1 align="center">MediaPipe.NET</h1>

<p align="center">
  <b>Google MediaPipe's vision tasks, native in .NET 10.</b><br/>
  Faces · face mesh & blendshapes · hands · gestures · pose · holistic · segmentation · objects · classification<br/>
  <i>Created by <b>Gravicode Studios</b>, led by <b>Kang Fadhil</b></i>
</p>

<p align="center">
  <a href="README.id.md">🇮🇩 Baca dalam Bahasa Indonesia</a> ·
  <a href="docs/en/getting-started.md">Documentation</a> ·
  <a href="docs/en/tasks.md">Tasks</a> ·
  <a href="docs/en/graph-api.md">Graph API</a> ·
  <a href="PLAN.md">Roadmap</a>
</p>

---

![MediaPipe.Net Gallery — overview](docs/images/gallery-home.png)

MediaPipe.NET is a port of [Google MediaPipe](https://ai.google.dev/edge/mediapipe) for the .NET ecosystem. It
runs MediaPipe's own models — converted from the official TFLite releases to ONNX and **cross-validated against
the official MediaPipe Python package** — on ONNX Runtime, with the pre- and post-processing (anchors, rotated
regions of interest, landmark projection, tracking, smoothing) re-implemented in C#. No Python, no C++ interop,
no separate process.

```csharp
using MediaPipeNet.Imaging;
using MediaPipeNet.Tasks.Vision;

using var gestures = GestureRecognizer.Create();
using var image = MPImage.Load("hand.jpg");

foreach (var hand in gestures.Recognize(image).Hands)
    Console.WriteLine($"{hand.Hand.Handedness.CategoryName} hand: {hand.TopGesture}");   // Right hand: Thumb_Up (74 %)
```

## Highlights

- **Nine tasks** with the same shape as MediaPipe Tasks — `FaceDetector`, `FaceLandmarker` (478 landmarks + 52
  blendshapes), `HandLandmarker`, `GestureRecognizer`, `PoseLandmarker` (+ segmentation mask),
  `HolisticLandmarker`, `ImageSegmenter`, `ObjectDetector`, `ImageClassifier`.
- **Three running modes** — `Image`, `Video` (tracking and One-Euro smoothing between frames) and `LiveStream`
  (asynchronous, frames dropped while busy so latency never builds up).
- **Graph API** — compose `ICalculatorNode`s connected by timestamped `Packet<T>` streams, with MediaPipe's input
  synchronization and timestamp-bound semantics, pipelined parallel execution, flow limiting, side packets and a
  `.pbtxt` config parser.
- **Fast and frugal** — 8 ms face detection at 640×480 on a 2017 4-core laptop CPU; pooled, preallocated
  tensors mean a steady-state inference allocates ~3–18 KB.
- **CPU everywhere, GPU when present** — CPU (Windows / Linux / macOS, x64 / ARM64), DirectML (any DX12 GPU),
  CUDA, CoreML; `ExecutionProvider.Auto` picks the best and falls back to CPU.
- **Models handled for you** — `Gravicode.MediaPipeNet.Models.*` packages copy models next to your app; otherwise they are
  downloaded from nuget.org on first use and verified by SHA-256.
- **Strongly-typed, JSON-ready results**, visualization helpers, `IFrameSource` for webcams and video files,
  `LiveStreamProcessor<T>`, DI for ASP.NET Core, `ILogger` and `System.Diagnostics.Metrics` telemetry, a CLI,
  a Polyglot notebook and an Avalonia Gallery app.

## Install

```bash
dotnet add package Gravicode.MediaPipeNet               # tasks + CPU runtime (Windows, Linux, macOS)
dotnet add package Gravicode.MediaPipeNet.Models.All    # optional: bundle every model (≈75 MB) for offline use
```

| Package | What it contains |
|---|---|
| `Gravicode.MediaPipeNet` | All tasks + ONNX Runtime **CPU** (and CoreML on macOS). Start here. |
| `Gravicode.MediaPipeNet.DirectML` / `Gravicode.MediaPipeNet.Cuda` | All tasks + the DirectML or CUDA runtime (use instead of `Gravicode.MediaPipeNet`). |
| `Gravicode.MediaPipeNet.Models.Face` · `.Hand` · `.Pose` · `.Segmentation` · `.ObjectDetection` · `.ImageClassification` · `.All` | The ONNX models, copied to `bin/…/models`. |
| `Gravicode.MediaPipeNet.Visualization` | Draw landmarks, boxes and masks on ImageSharp images. |
| `Gravicode.MediaPipeNet.Video.OpenCv` | `WebcamFrameSource`, `VideoFileFrameSource`. |
| `Gravicode.MediaPipeNet.Extensions.DI` | `services.AddMediaPipeNet().AddFaceDetector()…` |
| `Gravicode.MediaPipeNet.Cli` | `dotnet tool install -g Gravicode.MediaPipeNet.Cli` → `mediapipenet-cli` |
| `Gravicode.MediaPipeNet.Core` · `.Imaging` · `.Inference` · `.Framework` · `.Tasks.Vision` | The layers, for advanced use. |

## A tour

<table>
<tr>
<td width="50%"><img src="docs/images/gallery-face-mesh.png" alt="Face mesh" /><br/><b>Face mesh</b> — 478 landmarks + blendshapes</td>
<td width="50%"><img src="docs/images/gallery-hands.png" alt="Hands" /><br/><b>Hand landmarks</b> — 21 points, rotated ROI</td>
</tr>
<tr>
<td><img src="docs/images/gallery-pose.png" alt="Pose" /><br/><b>Pose</b> — 33 landmarks + person mask</td>
<td><img src="docs/images/gallery-objects.png" alt="Objects" /><br/><b>Object detection</b> — EfficientDet-Lite0</td>
</tr>
<tr>
<td><img src="docs/images/gallery-graph.png" alt="Graph API" /><br/><b>Graph API</b> — parallel nodes, custom calculator</td>
<td><img src="docs/images/gallery-hands-dark-id.png" alt="Dark theme, Indonesian" /><br/><b>Gallery</b> — dark theme, Bahasa Indonesia</td>
</tr>
</table>

## More examples

**Video with tracking** — the detector only runs when a hand is lost:

```csharp
using var hands = HandLandmarker.Create(new HandLandmarkerOptions { RunningMode = RunningMode.Video, NumHands = 1 });
await using var camera = new WebcamFrameSource(0);                       // MediaPipeNet.Video.OpenCv
await using var live = new LiveStreamProcessor<HandLandmarkResult>(camera, (frame, ts) => hands.DetectForVideo(frame, ts));
live.ResultReady += (_, r) => Console.WriteLine($"{r.Result.Hands.Count} hand(s) · {r.Stats.ProcessingFps:F0} fps");
await live.RunAsync(cancellationToken);
```

**Portrait mode** — blur the background with the selfie segmenter:

```csharp
using var segmenter = ImageSegmenter.Create();
var mask = segmenter.Segment(image).ConfidenceMask;
using var canvas = image.ToImage();
SegmentationMaskOverlay.BlurBackground(canvas, mask, sigma: 18);        // MediaPipeNet.Visualization
canvas.SaveAsPng("portrait.png");
```

**A graph from MediaPipe-style config:**

```csharp
var graph = GraphConfig.ParsePbtxt("""
    input_stream: "image"
    output_stream: "objects"
    node {
      calculator: "ObjectDetectorCalculator"
      input_stream: "IMAGE:image"
      output_stream: "RESULT:objects"
      options { score_threshold: 0.4 }
    }
    """).Build(CalculatorRegistry.Default.AddVisionCalculators());
```

**ASP.NET Core:**

```csharp
builder.Services.AddMediaPipeNet().AddFaceDetector().AddObjectDetector(o => o with { ScoreThreshold = 0.5f });
app.MapPost("/faces", async (IFormFile file, FaceDetector detector) =>
{
    using var image = await MPImage.LoadAsync(file.OpenReadStream());
    return detector.Detect(image);                                       // serialized as JSON
});
```

## Accuracy: cross-validated against MediaPipe

Every converted model is compared with the TFLite interpreter on random input (max |Δ| ≈ 1e-4), and every task
is compared end-to-end with the **official MediaPipe Python package 1.0.1** on fixture images
([`tests/assets/golden`](tests/assets/golden)):

| Task | MediaPipe (Python) | MediaPipe.NET |
|---|---|---|
| Face detection (portrait) | box 234 px, score 0.922 | box IoU > 0.9, score 0.928 |
| Face mesh | 478 landmarks | mean error < 0.006 (normalized) · smile 0.96 vs 0.96 |
| Gestures | Thumb_Up · Victory · Pointing_Up | Thumb_Up · Victory · Pointing_Up |
| Selfie segmentation | mean 0.5072 | mean 0.5080 |
| Object detection | dog 0.73 · cat 0.70 · dog 0.68 · cat 0.65 | same 4 objects, box IoU > 0.85 |
| Image classification | cheeseburger 0.889 | cheeseburger 0.848 |

## Performance

BenchmarkDotNet, Intel Core i7-8650U (2017, 4 cores), CPU provider, 640×480 input, end to end:

| Task | Mean | Allocated / call |
|---|---:|---:|
| FaceDetector | **8.1 ms** | 3 KB |
| ImageSegmenter (selfie) | 9.6 ms | 2.9 MB (the output mask) |
| ImageClassifier | 10.5 ms | 58 KB |
| FaceLandmarker | 25.1 ms | 19 KB |
| ObjectDetector | 30.6 ms | 17 KB |
| HandLandmarker / GestureRecognizer | 31.4 / 31.8 ms | 11 / 12 KB |
| PoseLandmarker (lite) | 36.9 ms | 11 KB |

NFR-1 asks for face detection under 50 ms at 640×480 on a modern 8-core CPU; MediaPipe.NET needs 8 ms on a
2017 4-core laptop. See [docs/en/performance.md](docs/en/performance.md).

## Repository

```
src/          Core · Imaging · Inference · Framework · Tasks.Vision · Visualization · Video.OpenCv · Extensions.DI · Cli · meta packages
models/       onnx/ (the 13 converted models) + Gravicode.MediaPipeNet.Models.* package projects
samples/      BasicUsage · GraphApiDemo · MediaPipeNet.Gallery (Avalonia)
tests/        Core.Tests · Framework.Tests · Tasks.Tests (golden cross-validation) — 104 tests
benchmarks/   BenchmarkDotNet suite
notebooks/    MediaPipeNet_QuickStart.ipynb (Polyglot Notebooks)
tools/        model-conversion (TFLite → ONNX + validation) · golden (MediaPipe Python references)
docs/         en/ and id/ documentation, images/
```

Build and test:

```bash
dotnet build MediaPipeNet.slnx -c Release
dotnet test MediaPipeNet.slnx -c Release
dotnet run --project samples/MediaPipeNet.Gallery
```

## Documentation

| English | Bahasa Indonesia |
|---|---|
| [Getting started](docs/en/getting-started.md) | [Memulai](docs/id/memulai.md) |
| [Tasks](docs/en/tasks.md) | [Task](docs/id/task.md) |
| [Running modes & live video](docs/en/video-and-live-stream.md) | [Mode & video langsung](docs/id/video-dan-live-stream.md) |
| [Graph API](docs/en/graph-api.md) | [Graph API](docs/id/graph-api.md) |
| [Models](docs/en/models.md) | [Model](docs/id/model.md) |
| [Performance & GPUs](docs/en/performance.md) | [Performa & GPU](docs/id/performa.md) |
| [Architecture](docs/en/architecture.md) | [Arsitektur](docs/id/arsitektur.md) |
| [Integration (DI, ASP.NET, telemetry)](docs/en/integration.md) | [Integrasi](docs/id/integrasi.md) |
| [CLI](docs/en/cli.md) | [CLI](docs/id/cli.md) |
| [Gallery app](docs/en/gallery.md) | [Aplikasi Gallery](docs/id/gallery.md) |
| [Testing & validation](docs/en/testing.md) | [Pengujian & validasi](docs/id/pengujian.md) |
| [API reference](docs/en/api-reference.md) | [Referensi API](docs/id/referensi-api.md) |
| [Troubleshooting](docs/en/troubleshooting.md) | [Pemecahan masalah](docs/id/pemecahan-masalah.md) |

## License

MediaPipe.NET is licensed under the [Apache License 2.0](LICENSE). The model weights are © Google LLC, Apache-2.0,
converted from the official MediaPipe releases — see [NOTICE](NOTICE) for attribution and third-party licenses
(ImageSharp is under the Six Labors Split License).

---

<p align="center">Created by <b>Gravicode Studios</b> · led by <b>Kang Fadhil</b></p>
