# Gravicode.MediaPipeNet.Cli

`mediapipenet-cli` — run any task from the command line, benchmark it, process a webcam or video, and manage models.

```
dotnet tool install -g Gravicode.MediaPipeNet.Cli
mediapipenet-cli gestures hand.jpg --output annotated.png --json result.json
mediapipenet-cli benchmark faces portrait.jpg --iterations 100
mediapipenet-cli video hands --camera 0
mediapipenet-cli models download
```

---
**MediaPipe.NET** — a native .NET 10 port of Google MediaPipe's vision tasks on ONNX Runtime.
Created by **Gravicode Studios**, led by **Kang Fadhil**. Library: Apache-2.0. Model weights: © Google LLC, Apache-2.0.
Documentation (English & Bahasa Indonesia): see the `docs/` folder of the repository.
