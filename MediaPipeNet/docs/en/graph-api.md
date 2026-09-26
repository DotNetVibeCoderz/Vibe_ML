# Graph API

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/graph-api.md)

`MediaPipeNet.Framework` is the .NET counterpart of MediaPipe's calculator framework. A **graph** is a set of
**nodes** (calculators) connected by **streams** of timestamped **packets**.

![Graph API page of the Gallery](../images/gallery-graph.png)

## Concepts

| MediaPipe | MediaPipe.NET | |
|---|---|---|
| `mediapipe::Packet` | `Packet<T>` | Immutable value + `Timestamp` (microseconds). |
| `CalculatorBase` | `ICalculatorNode` / `CalculatorNode` | `GetContract`, `OpenAsync`, `ProcessAsync`, `CloseAsync`. |
| `CalculatorContract` | `CalculatorContract` | Tagged, typed inputs/outputs and side packets. |
| `CalculatorGraph` | `CalculatorGraph` | Built with `GraphBuilder` or from `GraphConfig`. |
| Input side packets | `AddSidePacket<T>` + `StartAsync(sidePackets)` | Configuration injected once. |
| `DefaultInputStreamHandler` | `InputPolicy.Synchronized` | One call per timestamp once every input is settled. |
| `ImmediateInputStreamHandler` | `InputPolicy.Immediate` | Every packet processed on arrival (a timestamp may arrive once per input: emit once). |
| `FlowLimiterCalculator` | `GraphOptions.MaxInFlight` | Drops new input while the graph is busy. |
| `CalculatorGraphConfig` (.pbtxt) | `GraphConfig.ParsePbtxt` | Same text syntax (subset). |

## Semantics

- Each stream has exactly one producer; timestamps on a stream must strictly increase.
- A synchronized node runs for timestamp **T** once every input either delivered its packet for T or advanced its
  **timestamp bound** past T. Inputs without a packet at T appear as absent (`HasInput` is false).
- After processing T, outputs that emitted nothing automatically advance their bound to T+1, so downstream nodes
  never wait for packets that will not come (MediaPipe's `SetOffset(0)`).
- A node never runs concurrently with itself, but different nodes run in parallel on the thread pool: frames are
  **pipelined** through the graph.
- Graph validation at `Build()`: unknown ports or streams, type mismatches, missing required inputs, two producers
  for one stream and **cycles** (topological sort) are rejected with `GraphValidationException`.

## Building a graph in code

```csharp
using MediaPipeNet.Framework;
using MediaPipeNet.Framework.Nodes;
using MediaPipeNet.Tasks.Vision.Graph;

var builder = new GraphBuilder { Options = new GraphOptions { MaxInFlight = 2 } };
builder.AddInputStream<MPImage>("image");

builder.AddNode("faces", new VisionTaskNode<FaceDetector, FaceDetectionResult>(
        ct => FaceDetector.CreateAsync(new() { RunningMode = RunningMode.Video }, ct),
        (task, img, ts) => task.DetectForVideo(img, ts)))
    .In("IMAGE", "image").Out("RESULT", "faces");

builder.AddNode("hands", new VisionTaskNode<HandLandmarker, HandLandmarkResult>(
        ct => HandLandmarker.CreateAsync(new() { RunningMode = RunningMode.Video }, ct),
        (task, img, ts) => task.DetectForVideo(img, ts)))
    .In("IMAGE", "image").Out("RESULT", "hands");

builder.AddNode("summary", new CombineNode<FaceDetectionResult, HandLandmarkResult, string>(
        (f, h) => $"{f?.Detections.Count ?? 0} faces, {h?.Hands.Count ?? 0} hands"))
    .In("A", "faces").In("B", "hands").Out("OUT", "summary");

builder.AddOutputStream("summary");

await using var graph = builder.Build();
graph.ObserveOutputStream<string>("summary", p => Console.WriteLine($"{p.Timestamp}: {p.Value}"));
await graph.StartAsync();
graph.AddPacket("image", image, timestampMs: 0);
await graph.CloseAsync();                 // closes inputs and waits for every node to finish
Console.WriteLine(graph.ToMermaid());     // diagram for docs
```

Pull-based consumption instead of callbacks:

```csharp
ChannelReader<Packet<string>> reader = graph.CreateOutputStreamPoller<string>("summary", capacity: 8);
await foreach (var packet in reader.ReadAllAsync()) { … }
```

## Writing a calculator

```csharp
public sealed class ThresholdNode(float minScore) : CalculatorNode
{
    public override void GetContract(CalculatorContract contract) => contract
        .AddInput<ObjectDetectionResult>("DETECTIONS")
        .AddOutput<ObjectDetectionResult>("FILTERED")
        .AddInputSidePacket<string>("LABEL", optional: true);

    private string? _label;

    protected override void Open(CalculatorContext ctx) => ctx.TryGetSidePacket("LABEL", out _label);

    protected override void Process(CalculatorContext ctx)
    {
        var input = ctx.GetInput<ObjectDetectionResult>("DETECTIONS");
        var kept = input.Detections.Where(d => d.Score >= minScore && (_label is null || d.TopCategory.CategoryName == _label)).ToList();
        if (kept.Count > 0) ctx.Send("FILTERED", new ObjectDetectionResult(kept));   // no send → bound advances
    }
}
```

Implement `ICalculatorNode` directly for asynchronous work (`ProcessAsync` returns `ValueTask`). Useful context
members: `InputTimestamp`, `HasInput`, `TryGetInput`, `Send(tag, value[, timestamp])`, `SendPacket`,
`SetNextTimestampBound`, `GetSidePacket`, `Logger`, `CancellationToken`.

## Built-in nodes

| Node | Ports | |
|---|---|---|
| `PassThroughNode<T>` | IN → OUT | |
| `LambdaNode<TIn,TOut>` / `AsyncLambdaNode` | IN → OUT | Return null to emit nothing. |
| `CombineNode<TA,TB,TOut>` | A, B → OUT | Synchronized join; absent inputs are `default`. |
| `SinkNode<T>` | IN | Callback per packet. |
| `PacketThinnerNode` | IN → OUT | At most one packet per period. |
| `PacketCounterNode` | IN → COUNT | |
| `VisionTaskNode<TTask,TResult>` | IMAGE → RESULT | Any task, created at open (video mode). |
| `InferenceNode` | IMAGE → TENSORS | Any ONNX image model (resize/letterbox/normalize). |

## Graphs from `.pbtxt`

```csharp
const string Config = """
    input_stream: "image"
    output_stream: "faces"
    output_stream: "hands"
    max_in_flight: 1

    node {
      calculator: "FaceDetectorCalculator"
      input_stream: "IMAGE:image"
      output_stream: "RESULT:faces"
      options { min_detection_confidence: 0.6 }
    }
    node {
      calculator: "HandLandmarkerCalculator"
      input_stream: "IMAGE:image"
      output_stream: "RESULT:hands"
      options { num_hands: 2 }
    }
    """;

var registry = new CalculatorRegistry().AddVisionCalculators();     // + built-ins via CalculatorRegistry.Default
await using var graph = GraphConfig.ParsePbtxt(Config).Build(registry);
```

Registered vision calculators: `FaceDetectorCalculator`, `FaceLandmarkerCalculator`, `HandLandmarkerCalculator`,
`GestureRecognizerCalculator`, `PoseLandmarkerCalculator`, `HolisticLandmarkerCalculator`,
`ImageSegmenterCalculator`, `ObjectDetectorCalculator`, `ImageClassifierCalculator`. Register your own with
`registry.Register("MyCalculator", nodeConfig => new MyNode(nodeConfig.GetOption("threshold", 0.5f)))`.
Extension option blocks such as `[mediapipe.FooOptions.ext] { … }` are accepted (the type name is ignored).
`GraphConfig.ToPbtxt()` renders a config back to text.

## Live streams

- `GraphOptions.MaxInFlight = n`: `AddPacket` returns `false` (and advances the input bound) while `n` timestamps
  are still in flight — MediaPipe's flow limiter.
- `GraphInputOptions(MaxQueueSize, QueueOverflowPolicy.DropOldest)` bounds the queue at each consumer instead.
- `graph.DroppedPackets` counts both; the `mediapipenet.frames.dropped` metric is published too.

## Errors and lifecycle

`StartAsync` opens nodes in topological order. A node exception fails the graph: `Failed` is raised,
`WaitUntilDoneAsync` throws `MediaPipeException`, further `AddPacket` calls throw. `Cancel()` stops immediately;
`CloseAsync()` drains and closes normally; `WaitUntilIdleAsync()` waits until nothing is runnable.
