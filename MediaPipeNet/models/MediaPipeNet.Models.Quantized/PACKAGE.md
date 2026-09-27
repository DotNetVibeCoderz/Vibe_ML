# Gravicode.MediaPipeNet.Models.Quantized

Reduced-precision variants of the MediaPipe.NET models, copied to `models/` in your output folder:

- **FP16** — float16 weights and activations (inputs/outputs stay float32): half the size, fastest on GPUs
  (DirectML, CUDA). 14 models.
- **INT8** — weight-only int8 (per-channel), computed in float: about a quarter of the size with
  near-identical results. 9 models.

Every variant was validated against its float32 model (cosine similarity ≥ 0.999 for FP16, ≥ 0.99 for INT8).
Select them with `InferenceOptions.Precision`; models without a variant use float32:

```csharp
var options = new BaseOptions { Inference = new InferenceOptions { Precision = ModelPrecision.Float16 } };
using var landmarker = FaceLandmarker.Create(new() { BaseOptions = options });
```

---
**MediaPipe.NET** — a native .NET 10 port of Google MediaPipe's tasks on ONNX Runtime.
Created by **Gravicode Studios**, led by **Kang Fadhil**. Library: Apache-2.0. Model weights: © Google LLC, Apache-2.0.
Documentation (English & Bahasa Indonesia): see the `docs/` folder of the repository.
