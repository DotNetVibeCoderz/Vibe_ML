# Gravicode.MediaPipeNet

The recommended entry point: all vision tasks plus the **CPU** ONNX Runtime (and CoreML on macOS). Works on Windows, Linux and macOS, x64 and ARM64.

Models are resolved from the `Gravicode.MediaPipeNet.Models.*` packages, a local folder, or downloaded on first use from nuget.org (SHA-256 verified).

## Quick start

```csharp
using MediaPipeNet.Imaging;
using MediaPipeNet.Tasks.Vision;

using var hands = HandLandmarker.Create();
using var image = MPImage.Load("hand.jpg");
foreach (var hand in hands.Detect(image).Hands)
    Console.WriteLine($"{hand.Handedness.CategoryName}: index tip {hand[HandLandmark.IndexFingerTip]}");
```

---
**MediaPipe.NET** — a native .NET 10 port of Google MediaPipe's vision tasks on ONNX Runtime.
Created by **Gravicode Studios**, led by **Kang Fadhil**. Library: Apache-2.0. Model weights: © Google LLC, Apache-2.0.
Documentation (English & Bahasa Indonesia): see the `docs/` folder of the repository.
