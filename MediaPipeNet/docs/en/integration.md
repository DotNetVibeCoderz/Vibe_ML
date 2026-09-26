# Integration: DI, ASP.NET Core, logging and metrics

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/integrasi.md)

## Dependency injection

```bash
dotnet add package Gravicode.MediaPipeNet.Extensions.DI
```

```csharp
using MediaPipeNet.Extensions.DependencyInjection;

builder.Services
    .AddMediaPipeNet(o =>
    {
        o.ModelDirectory = "/app/models";              // optional
        o.AllowModelDownload = false;                   // containers: ship models, never download
        o.Inference = new InferenceOptions { Provider = ExecutionProvider.Cpu, IntraOpThreads = 2 };
    })
    .AddFaceDetector(o => o with { MinDetectionConfidence = 0.6f })
    .AddFaceLandmarker(o => o with { OutputFaceBlendshapes = true })
    .AddHandLandmarker()
    .AddGestureRecognizer()
    .AddPoseLandmarker()
    .AddHolisticLandmarker()
    .AddImageSegmenter()
    .AddObjectDetector()
    .AddImageClassifier();
```

Tasks are registered as **singletons in image mode** — thread-safe, created on first resolution, disposed with the
container. `ModelStore` and `BaseOptions` are registered too (wired to the container's `ILoggerFactory`).

## Minimal API example

```csharp
app.MapPost("/api/objects", async (IFormFile file, ObjectDetector detector, CancellationToken ct) =>
{
    await using var stream = file.OpenReadStream();
    using var image = await MPImage.LoadAsync(stream, ct);
    return Results.Ok(await detector.DetectAsync(image, cancellationToken: ct));
});
```

Results serialize with ASP.NET's default System.Text.Json settings; for MediaPipe.NET's exact format (camelCase,
nulls skipped, masks as base64) use `result.ToJson()` or `MediaPipeJson.Options`.

## Worker service processing a camera

```csharp
public sealed class CameraWorker(HandLandmarker hands, ILogger<CameraWorker> log) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        await using var camera = new WebcamFrameSource(0);
        // Video mode needs its own (stateful) instance; the DI singleton is image mode.
        using var tracker = await HandLandmarker.CreateAsync(new() { RunningMode = RunningMode.Video }, ct);
        await using var live = new LiveStreamProcessor<HandLandmarkResult>(camera, (f, ts) => tracker.DetectForVideo(f, ts));
        live.ResultReady += (_, r) => log.LogInformation("{Hands} hands at {Fps:F0} fps", r.Result.Hands.Count, r.Stats.ProcessingFps);
        await live.RunAsync(ct);
    }
}
```

## Logging

Pass an `ILoggerFactory` through `BaseOptions.LoggerFactory` (DI does it for you). MediaPipe.NET logs execution
provider selection and fallbacks, model resolution and download problems, graph failures and callback exceptions.

## Metrics and tracing (OpenTelemetry)

Meter and ActivitySource are both named **`Gravicode.MediaPipeNet`**:

| Instrument | Type | Tags |
|---|---|---|
| `mediapipenet.inference.duration` | histogram (ms) | `model`, `provider` |
| `mediapipenet.task.duration` | histogram (ms) | `task` |
| `mediapipenet.frames.processed` | counter | `task` |
| `mediapipenet.frames.dropped` | counter | `task` (live stream) |
| `mediapipenet.graph.packets` | counter | `node` |

```csharp
builder.Services.AddOpenTelemetry()
    .WithMetrics(m => m.AddMeter("MediaPipeNet").AddPrometheusExporter())
    .WithTracing(t => t.AddSource("MediaPipeNet"));   // one activity per graph node invocation
```

Quick look without OpenTelemetry: `dotnet-counters monitor --counters MediaPipeNet -n YourApp`.

## Deployment notes

- Ship models with `Gravicode.MediaPipeNet.Models.*` (they land in `bin/…/models`) or pre-fetch them into
  `MEDIAPIPENET_MODELS` during the image build (`mediapipenet-cli models download`).
- Linux containers: the CPU runtime needs glibc (Debian/Ubuntu base images); Alpine is not supported by ONNX Runtime.
- `IntraOpThreads` × number of concurrently running models should not exceed the core count on busy servers.
