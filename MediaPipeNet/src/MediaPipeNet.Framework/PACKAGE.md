# Gravicode.MediaPipeNet.Framework

The graph engine: `CalculatorGraph`, `ICalculatorNode`, `Packet<T>`, MediaPipe-style input synchronization and timestamp bounds, pipelined parallel scheduling, flow limiting for live streams, side packets and a `.pbtxt` config parser.

```csharp
var graph = new GraphBuilder()
    .AddInputStream<int>("in")
    .AddNode("square", new LambdaNode<int, int>(x => x * x)).In("IN", "in").Out("OUT", "out").Graph
    .AddOutputStream("out")
    .Build();
```

---
**MediaPipe.NET** — a native .NET 10 port of Google MediaPipe's vision tasks on ONNX Runtime.
Created by **Gravicode Studios**, led by **Kang Fadhil**. Library: Apache-2.0. Model weights: © Google LLC, Apache-2.0.
Documentation (English & Bahasa Indonesia): see the `docs/` folder of the repository.
