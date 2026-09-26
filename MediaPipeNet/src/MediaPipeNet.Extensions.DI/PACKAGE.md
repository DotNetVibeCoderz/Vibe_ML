# Gravicode.MediaPipeNet.Extensions.DI

Dependency-injection registration for ASP.NET Core and worker services:

```csharp
builder.Services.AddMediaPipeNet(o => o.ModelDirectory = "models")
    .AddFaceDetector()
    .AddHandLandmarker(o => o with { NumHands = 2 });
```

Tasks are registered as thread-safe singletons (image mode) and log through `ILogger`.

---
**MediaPipe.NET** — a native .NET 10 port of Google MediaPipe's vision tasks on ONNX Runtime.
Created by **Gravicode Studios**, led by **Kang Fadhil**. Library: Apache-2.0. Model weights: © Google LLC, Apache-2.0.
Documentation (English & Bahasa Indonesia): see the `docs/` folder of the repository.
