# Gravicode.MediaPipeNet.Video.OpenCv

`WebcamFrameSource` and `VideoFileFrameSource` (OpenCvSharp) implementing `IFrameSource`, with frame-buffer reuse. Add an OpenCvSharp runtime package for your OS (e.g. `OpenCvSharp4.runtime.win`).

```csharp
await using var camera = new WebcamFrameSource(0);
using var hands = HandLandmarker.Create(new() { RunningMode = RunningMode.Video });
await using var live = new LiveStreamProcessor<HandLandmarkResult>(camera, (f, ts) => hands.DetectForVideo(f, ts));
live.ResultReady += (_, r) => Console.WriteLine($"{r.Result.Hands.Count} hands, {r.Stats.ProcessingFps:F1} fps");
await live.RunAsync();
```

---
**MediaPipe.NET** — a native .NET 10 port of Google MediaPipe's vision tasks on ONNX Runtime.
Created by **Gravicode Studios**, led by **Kang Fadhil**. Library: Apache-2.0. Model weights: © Google LLC, Apache-2.0.
Documentation (English & Bahasa Indonesia): see the `docs/` folder of the repository.
