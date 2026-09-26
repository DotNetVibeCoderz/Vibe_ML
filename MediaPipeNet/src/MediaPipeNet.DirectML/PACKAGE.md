# Gravicode.MediaPipeNet.DirectML

All vision tasks plus the **DirectML** ONNX Runtime: GPU acceleration on any DirectX 12 GPU (NVIDIA, AMD, Intel) on Windows. Reference this *instead of* `Gravicode.MediaPipeNet`, then pick the provider:

```csharp
var options = new BaseOptions { Inference = new() { Provider = ExecutionProvider.DirectML } };
using var faces = FaceDetector.Create(new() { BaseOptions = options });
```

---
**MediaPipe.NET** — a native .NET 10 port of Google MediaPipe's vision tasks on ONNX Runtime.
Created by **Gravicode Studios**, led by **Kang Fadhil**. Library: Apache-2.0. Model weights: © Google LLC, Apache-2.0.
Documentation (English & Bahasa Indonesia): see the `docs/` folder of the repository.
