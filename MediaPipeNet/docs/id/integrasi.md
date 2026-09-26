# Integrasi: DI, ASP.NET Core, logging, dan metrik

> 🇬🇧 [Read in English](../en/integration.md)

## Dependency injection

```bash
dotnet add package Gravicode.MediaPipeNet.Extensions.DI
```

```csharp
using MediaPipeNet.Extensions.DependencyInjection;

builder.Services
    .AddMediaPipeNet(o =>
    {
        o.ModelDirectory = "/app/models";              // opsional
        o.AllowModelDownload = false;                   // container: bawa model sendiri, jangan unduh
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

Task didaftarkan sebagai **singleton dalam mode image** — thread-safe, dibuat saat pertama di-resolve, di-dispose
bersama container. `ModelStore` dan `BaseOptions` juga didaftarkan (terhubung ke `ILoggerFactory` container).

## Contoh Minimal API

```csharp
app.MapPost("/api/objects", async (IFormFile file, ObjectDetector detector, CancellationToken ct) =>
{
    await using var stream = file.OpenReadStream();
    using var image = await MPImage.LoadAsync(stream, ct);
    return Results.Ok(await detector.DetectAsync(image, cancellationToken: ct));
});
```

Hasil diserialisasi dengan pengaturan System.Text.Json bawaan ASP.NET; untuk format persis MediaPipe.NET
(camelCase, null dilewati, mask sebagai base64) gunakan `result.ToJson()` atau `MediaPipeJson.Options`.

## Worker service yang memproses kamera

```csharp
public sealed class CameraWorker(HandLandmarker hands, ILogger<CameraWorker> log) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        await using var camera = new WebcamFrameSource(0);
        // Mode video butuh instance sendiri (ber-state); singleton dari DI berada di mode image.
        using var tracker = await HandLandmarker.CreateAsync(new() { RunningMode = RunningMode.Video }, ct);
        await using var live = new LiveStreamProcessor<HandLandmarkResult>(camera, (f, ts) => tracker.DetectForVideo(f, ts));
        live.ResultReady += (_, r) => log.LogInformation("{Hands} tangan pada {Fps:F0} fps", r.Result.Hands.Count, r.Stats.ProcessingFps);
        await live.RunAsync(ct);
    }
}
```

## Logging

Berikan `ILoggerFactory` melalui `BaseOptions.LoggerFactory` (DI melakukannya otomatis). MediaPipe.NET mencatat
pemilihan dan fallback execution provider, masalah resolusi dan unduhan model, kegagalan graph, serta exception di
callback.

## Metrik dan tracing (OpenTelemetry)

Meter dan ActivitySource sama-sama bernama **`Gravicode.MediaPipeNet`**:

| Instrumen | Tipe | Tag |
|---|---|---|
| `mediapipenet.inference.duration` | histogram (ms) | `model`, `provider` |
| `mediapipenet.task.duration` | histogram (ms) | `task` |
| `mediapipenet.frames.processed` | counter | `task` |
| `mediapipenet.frames.dropped` | counter | `task` (live stream) |
| `mediapipenet.graph.packets` | counter | `node` |

```csharp
builder.Services.AddOpenTelemetry()
    .WithMetrics(m => m.AddMeter("MediaPipeNet").AddPrometheusExporter())
    .WithTracing(t => t.AddSource("MediaPipeNet"));   // satu activity per pemanggilan node graph
```

Cara cepat tanpa OpenTelemetry: `dotnet-counters monitor --counters MediaPipeNet -n AplikasiAnda`.

## Catatan deployment

- Sertakan model dengan `Gravicode.MediaPipeNet.Models.*` (masuk ke `bin/…/models`) atau unduh lebih dulu ke
  `MEDIAPIPENET_MODELS` saat build image (`mediapipenet-cli models download`).
- Container Linux: runtime CPU butuh glibc (image dasar Debian/Ubuntu); Alpine tidak didukung ONNX Runtime.
- `IntraOpThreads` × jumlah model yang berjalan bersamaan sebaiknya tidak melebihi jumlah core pada server sibuk.
