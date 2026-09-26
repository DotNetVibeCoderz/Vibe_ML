# Getting started

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/memulai.md) · MediaPipe.NET — created by Gravicode Studios, led by Kang Fadhil

## Requirements

- .NET 10 SDK
- Windows 10/11, Ubuntu 22.04+ or macOS 13+ (x64 or ARM64)
- No GPU required. Optional: any DirectX 12 GPU (DirectML) or an NVIDIA GPU with CUDA 12 + cuDNN 9.

## 1. Create a project and add the packages

```bash
dotnet new console -n Hello.MediaPipe
cd Hello.MediaPipe
dotnet add package Gravicode.MediaPipeNet
dotnet add package Gravicode.MediaPipeNet.Models.Face   # the face models, copied to bin/…/models
```

`Gravicode.MediaPipeNet` contains every task and the CPU build of ONNX Runtime. The `Gravicode.MediaPipeNet.Models.*` packages are
optional: without them, a task downloads the model package it needs from nuget.org on first use and caches it in
`%LOCALAPPDATA%/MediaPipeNet/models` (`~/.local/share/MediaPipeNet/models` on Linux/macOS). Every file is verified
against its SHA-256 before it is used.

## 2. Detect faces

```csharp
using MediaPipeNet.Imaging;
using MediaPipeNet.Tasks.Vision;

using var detector = FaceDetector.Create();              // loads the model once
using var image = MPImage.Load("people.jpg");            // JPEG, PNG, BMP, GIF, WebP, TIFF; EXIF orientation applied

FaceDetectionResult result = detector.Detect(image);
foreach (var face in result.Detections)
{
    Console.WriteLine($"face at {face.BoundingBox} (score {face.Score:P0})");
    var rightEye = face.Keypoints[(int)FaceKeypoint.RightEye];   // normalized [0,1] coordinates
}

Console.WriteLine(result.ToJson(indented: true));         // every result is JSON-serializable
```

Coordinates follow MediaPipe's conventions: bounding boxes are in **pixels**, keypoints and landmarks are
**normalized** to `[0, 1]` of the image width and height (`landmark.ToPixel(width, height)` converts).

## 3. Where images come from

`MPImage` is the input of every task — MediaPipe.NET's `mp.Image`. It holds RGBA pixels in a pooled buffer:

```csharp
using var a = MPImage.Load("photo.jpg");                               // file
using var b = await MPImage.LoadAsync(stream);                         // stream (upload, HTTP body, ...)
using var c = MPImage.Load(bytes);                                     // encoded bytes
using var d = MPImage.FromImage(imageSharpImage);                      // SixLabors.ImageSharp image
using var e = MPImage.FromPixelData(bgr, width, height, PixelFormat.Bgr24, stride);   // raw OpenCV/camera buffer
```

Dispose images when done (the pixel buffer returns to the pool). For video, reuse one image with
`image.CopyFrom(bytes, width, height, PixelFormat.Bgr24)` to avoid per-frame allocations.

## 4. Options

Every task takes an options record. Unset properties keep MediaPipe's defaults:

```csharp
using var hands = HandLandmarker.Create(new HandLandmarkerOptions
{
    NumHands = 2,
    MinHandDetectionConfidence = 0.6f,
    BaseOptions = new BaseOptions
    {
        Inference = new InferenceOptions { Provider = ExecutionProvider.Auto, IntraOpThreads = 4 },
        ModelDirectory = "my-models",                 // optional: look here first
    },
});
```

## 5. Async creation and async calls

`Create` resolves models synchronously (and may block while downloading). In UI apps and servers prefer:

```csharp
using var pose = await PoseLandmarker.CreateAsync(new() { OutputSegmentationMasks = true }, cancellationToken);
PoseLandmarkResult result = await pose.DetectAsync(image, cancellationToken: cancellationToken);
```

Image-mode tasks are thread-safe: one instance can serve many concurrent requests.

## 6. Draw the result (optional)

```bash
dotnet add package Gravicode.MediaPipeNet.Visualization
```

```csharp
using MediaPipeNet.Visualization;

using var canvas = image.ToImage();          // SixLabors.ImageSharp Image<Rgba32>
ResultRenderer.Render(canvas, result);        // overloads for every result type
canvas.SaveAsPng("annotated.png");
```

## Next steps

- [Tasks](tasks.md) — every task, its options and its result.
- [Running modes & live video](video-and-live-stream.md) — webcams, video files, tracking.
- [Graph API](graph-api.md) — build your own pipelines.
- [Samples](../../samples) — `BasicUsage`, `GraphApiDemo` and the Avalonia `MediaPipeNet.Gallery`.
