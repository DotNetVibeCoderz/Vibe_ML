"""Writes a PACKAGE.md (the NuGet readme) into every shippable project."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

QUICK = """```csharp
using MediaPipeNet.Imaging;
using MediaPipeNet.Tasks.Vision;

using var hands = HandLandmarker.Create();
using var image = MPImage.Load("hand.jpg");
foreach (var hand in hands.Detect(image).Hands)
    Console.WriteLine($"{hand.Handedness.CategoryName}: index tip {hand[HandLandmark.IndexFingerTip]}");
```"""

FOOTER = """
---
**MediaPipe.NET** — a native .NET 10 port of Google MediaPipe's vision tasks on ONNX Runtime.
Created by **Gravicode Studios**, led by **Kang Fadhil**. Library: Apache-2.0. Model weights: © Google LLC, Apache-2.0.
Documentation (English & Bahasa Indonesia): see the `docs/` folder of the repository.
"""

PACKAGES = {
    "src/MediaPipeNet": ("MediaPipeNet", "The recommended entry point: all vision tasks plus the **CPU** ONNX Runtime (and CoreML on macOS). Works on Windows, Linux and macOS, x64 and ARM64.\n\nModels are resolved from the `Gravicode.MediaPipeNet.Models.*` packages, a local folder, or downloaded on first use from nuget.org (SHA-256 verified).", QUICK),
    "src/MediaPipeNet.DirectML": ("MediaPipeNet.DirectML", "All vision tasks plus the **DirectML** ONNX Runtime: GPU acceleration on any DirectX 12 GPU (NVIDIA, AMD, Intel) on Windows. Reference this *instead of* `Gravicode.MediaPipeNet`, then pick the provider:\n\n```csharp\nvar options = new BaseOptions { Inference = new() { Provider = ExecutionProvider.DirectML } };\nusing var faces = FaceDetector.Create(new() { BaseOptions = options });\n```", ""),
    "src/MediaPipeNet.Cuda": ("MediaPipeNet.Cuda", "All vision tasks plus the **CUDA** ONNX Runtime for NVIDIA GPUs (Windows/Linux, CUDA 12 + cuDNN 9). Reference this *instead of* `Gravicode.MediaPipeNet`. `ExecutionProvider.Auto` picks CUDA first and falls back to CPU.", ""),
    "src/MediaPipeNet.Core": ("MediaPipeNet.Core", "Core primitives shared by every MediaPipe.NET package: `Timestamp`, `NormalizedLandmark`, `Landmark`, `Detection`, `Category`, `RectF`, `NormalizedRect`, telemetry (`System.Diagnostics.Metrics` meter `Gravicode.MediaPipeNet`) and JSON settings.", ""),
    "src/MediaPipeNet.Imaging": ("MediaPipeNet.Imaging", "`MPImage` (pooled RGBA frames from files, streams, byte arrays, ImageSharp images or raw BGR/RGBA buffers), rotated-ROI **image-to-tensor** conversion with letterboxing, normalization and anti-aliasing, mask projection, and the `IFrameSource` abstraction for video.", ""),
    "src/MediaPipeNet.Inference": ("MediaPipeNet.Inference", "ONNX Runtime wrapper with preallocated, pooled I/O buffers (zero allocations per inference, thread-safe), automatic execution-provider selection (CUDA → DirectML → CoreML → CPU) and model management: `ModelCatalog`, `ModelStore`, checksum verification, cache, NuGet/HTTP/embedded providers.", ""),
    "src/MediaPipeNet.Framework": ("MediaPipeNet.Framework", "The graph engine: `CalculatorGraph`, `ICalculatorNode`, `Packet<T>`, MediaPipe-style input synchronization and timestamp bounds, pipelined parallel scheduling, flow limiting for live streams, side packets and a `.pbtxt` config parser.\n\n```csharp\nvar graph = new GraphBuilder()\n    .AddInputStream<int>(\"in\")\n    .AddNode(\"square\", new LambdaNode<int, int>(x => x * x)).In(\"IN\", \"in\").Out(\"OUT\", \"out\").Graph\n    .AddOutputStream(\"out\")\n    .Build();\n```", ""),
    "src/MediaPipeNet.Tasks.Vision": ("MediaPipeNet.Tasks.Vision", "The vision tasks: `FaceDetector`, `FaceLandmarker` (478 landmarks + 52 blendshapes), `HandLandmarker`, `GestureRecognizer`, `PoseLandmarker`, `HolisticLandmarker`, `ImageSegmenter`, `ObjectDetector`, `ImageClassifier` — with IMAGE / VIDEO / LIVE_STREAM modes, tracking, smoothing, `LiveStreamProcessor<T>` and graph calculators. Needs a native ONNX Runtime: prefer the `Gravicode.MediaPipeNet` package.", QUICK),
    "src/MediaPipeNet.Visualization": ("MediaPipeNet.Visualization", "Drawing helpers on ImageSharp images: `LandmarkDrawer`, `BoundingBoxDrawer`, `SegmentationMaskOverlay` (tint, blur background, replace background) and `ResultRenderer.Render(image, result)` for every result type.", ""),
    "src/MediaPipeNet.Video.OpenCv": ("MediaPipeNet.Video.OpenCv", "`WebcamFrameSource` and `VideoFileFrameSource` (OpenCvSharp) implementing `IFrameSource`, with frame-buffer reuse. Add an OpenCvSharp runtime package for your OS (e.g. `OpenCvSharp4.runtime.win`).\n\n```csharp\nawait using var camera = new WebcamFrameSource(0);\nusing var hands = HandLandmarker.Create(new() { RunningMode = RunningMode.Video });\nawait using var live = new LiveStreamProcessor<HandLandmarkResult>(camera, (f, ts) => hands.DetectForVideo(f, ts));\nlive.ResultReady += (_, r) => Console.WriteLine($\"{r.Result.Hands.Count} hands, {r.Stats.ProcessingFps:F1} fps\");\nawait live.RunAsync();\n```", ""),
    "src/MediaPipeNet.Extensions.DI": ("MediaPipeNet.Extensions.DI", "Dependency-injection registration for ASP.NET Core and worker services:\n\n```csharp\nbuilder.Services.AddMediaPipeNet(o => o.ModelDirectory = \"models\")\n    .AddFaceDetector()\n    .AddHandLandmarker(o => o with { NumHands = 2 });\n```\n\nTasks are registered as thread-safe singletons (image mode) and log through `ILogger`.", ""),
    "src/MediaPipeNet.Cli": ("MediaPipeNet.Cli", "`mediapipenet-cli` — run any task from the command line, benchmark it, process a webcam or video, and manage models.\n\n```\ndotnet tool install -g Gravicode.MediaPipeNet.Cli\nmediapipenet-cli gestures hand.jpg --output annotated.png --json result.json\nmediapipenet-cli benchmark faces portrait.jpg --iterations 100\nmediapipenet-cli video hands --camera 0\nmediapipenet-cli models download\n```", ""),
    "models/MediaPipeNet.Models.Face": ("MediaPipeNet.Models.Face", "Face models (BlazeFace short range, Face Mesh V2 with irises, face blendshapes) as ONNX. Copied to your output folder under `models/`, where MediaPipe.NET finds them automatically.", ""),
    "models/MediaPipeNet.Models.Hand": ("MediaPipeNet.Models.Hand", "Hand models (BlazePalm, hand landmarks, gesture embedder, canned gesture classifier) as ONNX, copied to `models/` in your output folder.", ""),
    "models/MediaPipeNet.Models.Pose": ("MediaPipeNet.Models.Pose", "Pose models (BlazePose detector, GHUM Lite and Full landmark models) as ONNX, copied to `models/` in your output folder.", ""),
    "models/MediaPipeNet.Models.Segmentation": ("MediaPipeNet.Models.Segmentation", "The selfie segmentation model as ONNX, copied to `models/` in your output folder.", ""),
    "models/MediaPipeNet.Models.ObjectDetection": ("MediaPipeNet.Models.ObjectDetection", "EfficientDet-Lite0 (COCO) as ONNX, copied to `models/` in your output folder.", ""),
    "models/MediaPipeNet.Models.ImageClassification": ("MediaPipeNet.Models.ImageClassification", "EfficientNet-Lite0 (ImageNet) as ONNX, copied to `models/` in your output folder.", ""),
    "models/MediaPipeNet.Models.All": ("MediaPipeNet.Models.All", "Every MediaPipe.NET model package in one reference (≈75 MB): apps run fully offline.", ""),
}

for folder, (name, body, snippet) in PACKAGES.items():
    text = f"# Gravicode.{name}\n\n{body}\n"
    if snippet:
        text += f"\n## Quick start\n\n{snippet}\n"
    text += FOOTER
    (ROOT / folder / "PACKAGE.md").write_text(text, encoding="utf-8")
    print("wrote", folder)
