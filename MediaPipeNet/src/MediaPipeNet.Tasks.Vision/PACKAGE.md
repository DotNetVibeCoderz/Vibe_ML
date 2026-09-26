# Gravicode.MediaPipeNet.Tasks.Vision

The vision tasks: `FaceDetector`, `FaceLandmarker` (478 landmarks + 52 blendshapes), `HandLandmarker`, `GestureRecognizer`, `PoseLandmarker`, `HolisticLandmarker`, `ImageSegmenter`, `ObjectDetector`, `ImageClassifier` — with IMAGE / VIDEO / LIVE_STREAM modes, tracking, smoothing, `LiveStreamProcessor<T>` and graph calculators. Needs a native ONNX Runtime: prefer the `Gravicode.MediaPipeNet` package.

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
