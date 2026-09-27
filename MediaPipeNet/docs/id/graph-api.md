# Graph API

> 🇬🇧 [Read in English](../en/graph-api.md)

`MediaPipeNet.Framework` adalah padanan .NET dari framework kalkulator MediaPipe. Sebuah **graph** adalah kumpulan
**node** (kalkulator) yang terhubung oleh **stream** berisi **packet** ber-timestamp.

![Halaman Graph API di Gallery](../images/gallery-graph.png)

## Konsep

| MediaPipe | MediaPipe.NET | |
|---|---|---|
| `mediapipe::Packet` | `Packet<T>` | Nilai immutable + `Timestamp` (mikrodetik). |
| `CalculatorBase` | `ICalculatorNode` / `CalculatorNode` | `GetContract`, `OpenAsync`, `ProcessAsync`, `CloseAsync`. |
| `CalculatorContract` | `CalculatorContract` | Input/output bertag dan bertipe, serta side packet. |
| `CalculatorGraph` | `CalculatorGraph` | Dibangun dengan `GraphBuilder` atau dari `GraphConfig`. |
| Input side packet | `AddSidePacket<T>` + `StartAsync(sidePackets)` | Konfigurasi yang disuntikkan sekali. |
| `DefaultInputStreamHandler` | `InputPolicy.Synchronized` | Satu panggilan per timestamp setelah semua input "settled". |
| `ImmediateInputStreamHandler` | `InputPolicy.Immediate` | Setiap packet diproses saat tiba (satu timestamp bisa datang sekali per input: kirim sekali saja). |
| `FlowLimiterCalculator` | `GraphOptions.MaxInFlight` | Membuang input baru selama graph sibuk. |
| `back_edge: true` | `In(tag, stream, backEdge: true)` | Loop, mis. mengumpankan hasil sebelumnya kembali. |
| `PreviousLoopbackCalculator` | `PreviousLoopbackNode<T>` | Mengirim nilai loop sebelumnya pada timestamp saat ini. |
| `executor { … }` / `executor:` | `AddExecutor(name, threads)` + `OnExecutor(name)` | Thread khusus untuk node tertentu. |
| Subgraph | `AddSubgraph(…)` / `CalculatorRegistry.RegisterSubgraph` | Blok yang dapat dipakai ulang, di-inline saat build. |
| Profiler / tracer | `GraphOptions.EnableTracing` + `WriteChromeTrace` | Timeline untuk `chrome://tracing` / Perfetto. |
| `CalculatorGraphConfig` (.pbtxt) | `GraphConfig.ParsePbtxt` | Sintaks teks yang sama (subset). |

## Semantik

- Setiap stream punya tepat satu produsen; timestamp di satu stream harus naik ketat.
- Node tersinkronisasi berjalan untuk timestamp **T** setelah setiap input mengirim packet untuk T atau memajukan
  **timestamp bound**-nya melewati T. Input tanpa packet di T dianggap absen (`HasInput` bernilai false).
- Setelah memproses T, output yang tidak mengirim apa-apa otomatis memajukan bound-nya ke T+1, sehingga node
  hilir tidak pernah menunggu packet yang tidak akan datang (`SetOffset(0)` di MediaPipe).
- Satu node tidak pernah berjalan bersamaan dengan dirinya sendiri, tetapi node yang berbeda berjalan paralel di
  thread pool: frame **di-pipeline** melalui graph.
- Validasi saat `Build()`: port atau stream tak dikenal, tipe tidak cocok, input wajib tidak terhubung, dua produsen
  untuk satu stream, dan **siklus** (topological sort) ditolak dengan `GraphValidationException`.

## Membangun graph lewat kode

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
        (f, h) => $"{f?.Detections.Count ?? 0} wajah, {h?.Hands.Count ?? 0} tangan"))
    .In("A", "faces").In("B", "hands").Out("OUT", "summary");

builder.AddOutputStream("summary");

await using var graph = builder.Build();
graph.ObserveOutputStream<string>("summary", p => Console.WriteLine($"{p.Timestamp}: {p.Value}"));
await graph.StartAsync();
graph.AddPacket("image", image, timestampMs: 0);
await graph.CloseAsync();                 // tutup input dan tunggu semua node selesai
Console.WriteLine(graph.ToMermaid());     // diagram untuk dokumentasi
```

Konsumsi berbasis pull sebagai ganti callback:

```csharp
ChannelReader<Packet<string>> reader = graph.CreateOutputStreamPoller<string>("summary", capacity: 8);
await foreach (var packet in reader.ReadAllAsync()) { … }
```

## Menulis kalkulator

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
        if (kept.Count > 0) ctx.Send("FILTERED", new ObjectDetectionResult(kept));   // tidak mengirim → bound maju
    }
}
```

Implementasikan `ICalculatorNode` langsung untuk pekerjaan asinkron (`ProcessAsync` mengembalikan `ValueTask`).
Anggota context yang berguna: `InputTimestamp`, `HasInput`, `TryGetInput`, `Send(tag, nilai[, timestamp])`,
`SendPacket`, `SetNextTimestampBound`, `GetSidePacket`, `Logger`, `CancellationToken`.

## Node bawaan

| Node | Port | |
|---|---|---|
| `PassThroughNode<T>` | IN → OUT | |
| `LambdaNode<TIn,TOut>` / `AsyncLambdaNode` | IN → OUT | Kembalikan null untuk tidak mengirim apa pun. |
| `CombineNode<TA,TB,TOut>` | A, B → OUT | Join tersinkronisasi; input absen bernilai `default`. |
| `SinkNode<T>` | IN | Callback per packet. |
| `PacketThinnerNode` | IN → OUT | Maksimal satu packet per periode. |
| `PacketCounterNode` | IN → COUNT | |
| `VisionTaskNode<TTask,TResult>` | IMAGE → RESULT | Task apa pun, dibuat saat open (mode video). |
| `InferenceNode` | IMAGE → TENSORS | Model gambar ONNX apa pun (resize/letterbox/normalisasi). |

## Graph dari `.pbtxt`

Parser memahami field di atas ditambah `input_stream_info { tag_index: "LOOP" back_edge: true }`, `executor: "name"`
per node, `executor { name: "gpu" … num_threads: 1 }` tingkat graph, dan `profiler_config { trace_enabled: true }`.
Subgraph didaftarkan sebagai config: node yang calculator-nya adalah subgraph terdaftar diganti dengan node-node
subgraph tersebut, entri `TAG:stream`-nya dicocokkan berdasarkan tag (atau posisi bila tanpa tag):

```csharp
var registry = CalculatorRegistry.Default.AddVisionCalculators()
    .RegisterSubgraph("FaceAndHands", GraphConfig.ParsePbtxt("""
        input_stream: "IMAGE:image"
        output_stream: "FACES:faces"
        node { calculator: "FaceDetectorCalculator" input_stream: "IMAGE:image" output_stream: "RESULT:faces" }
        """));
```


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

var registry = new CalculatorRegistry().AddVisionCalculators();
await using var graph = GraphConfig.ParsePbtxt(Config).Build(registry);
```

Kalkulator vision terdaftar: `FaceDetectorCalculator`, `FaceLandmarkerCalculator`, `HandLandmarkerCalculator`,
`GestureRecognizerCalculator`, `PoseLandmarkerCalculator`, `HolisticLandmarkerCalculator`,
`ImageSegmenterCalculator`, `ObjectDetectorCalculator`, `ImageClassifierCalculator`. Daftarkan milik Anda dengan
`registry.Register("KalkulatorSaya", nodeConfig => new NodeSaya(nodeConfig.GetOption("threshold", 0.5f)))`.
Blok opsi ekstensi seperti `[mediapipe.FooOptions.ext] { … }` diterima (nama tipenya diabaikan).
`GraphConfig.ToPbtxt()` mengubah konfigurasi kembali menjadi teks.

## Loop (back edge)

Graph boleh mengandung siklus bila input yang menutupnya ditandai sebagai **back edge**, seperti `back_edge: true`
di MediaPipe. Back edge diabaikan oleh deteksi siklus, tidak pernah menahan node-nya, dan mengirim paket terbaru pada
atau sebelum timestamp yang sedang diproses (pada node `Synchronized`). Pasangan klasiknya `PreviousLoopbackNode<T>`:
untuk setiap paket `MAIN` ia mengirim nilai `LOOP` terakhir di `PREV_LOOP`.

```csharp
// Jumlah berjalan: sum[t] = in[t] + sum[t-1]
var b = new GraphBuilder().AddInputStream<int>("in").AddOutputStream("sum");
b.AddNode("prev", new PreviousLoopbackNode<int>())
    .In("MAIN", "in").In("LOOP", "sum", backEdge: true).Out("PREV_LOOP", "prev_sum");
b.AddNode("add", new CombineNode<int, int, int>((x, prev) => x + prev))
    .In("A", "in").In("B", "prev_sum").Out("OUT", "sum");
```

Karena frame di-pipeline, "terakhir" berarti nilai loop terbaru yang sudah tiba — nilai frame sebelumnya bila frame
diproses satu per satu (mis. dengan `MaxInFlight = 1`). Tanpa tanda back edge, builder menolak siklus dan
menjelaskannya.

## Executor

Secara default setiap node berjalan di thread pool .NET. `AddExecutor(name, threads)` mendeklarasikan sekumpulan
thread khusus dan `OnExecutor(name)` menempatkan node di sana — setara `executor` / `ThreadPoolExecutor` MediaPipe.
Pakai executor satu thread untuk menserialkan node yang berbagi sumber daya tidak thread-safe (konteks GPU, kamera,
library native), atau pool terpisah agar node berat tidak menghambat yang lain.

```csharp
var b = new GraphBuilder().AddInputStream<MPImage>("frames").AddExecutor("gpu", threads: 1);
b.AddNode("faces", faceNode).In("IN", "frames").Out("OUT", "faces").OnExecutor("gpu");
```

## Tracing (Chrome trace / Perfetto)

Dengan `GraphOptions.EnableTracing` graph merekam setiap open, process (beserta timestamp paketnya), dan close node,
lengkap dengan waktu mulai, durasi, dan thread. `GetTraceEvents()` mengembalikannya dan `WriteChromeTrace(stream)`
menulis JSON trace-event Chrome yang ditampilkan `chrome://tracing`, `edge://tracing`, dan https://ui.perfetto.dev
sebagai timeline — satu baris per thread, sehingga pipelining dan bottleneck terlihat. Buffer menyimpan
`MaxTraceEvents` (100.000) event terbaru. Setiap pemanggilan process juga menjadi `Activity` pada
`MediaPipeTelemetry.ActivitySource` untuk OpenTelemetry.

```csharp
var graph = new GraphBuilder { Options = new GraphOptions { EnableTracing = true } } /* … */ .Build();
// … jalankan …
await using var file = File.Create("graph-trace.json");
graph.WriteChromeTrace(file);
```

## Subgraph

`AddSubgraph(name, subgraphBuilder, inputs, outputs)` meng-inline `GraphBuilder` lain sebagai blok yang dapat
dipakai ulang: node-nya menjadi `{name}/{node}`, input stream-nya disambungkan ke stream luar di `inputs`, stream
yang tercantum di `outputs` diganti nama menjadi stream luar, dan stream internal lainnya menjadi `{name}/{stream}`.
Buat builder subgraph baru untuk setiap pemakaian (instance node menyimpan state).

```csharp
static GraphBuilder Affine(int skala, int offset)
{
    var g = new GraphBuilder().AddInputStream<int>("x").AddOutputStream("y");
    g.AddNode("scale", new LambdaNode<int, int>(v => v * skala)).In("IN", "x").Out("OUT", "scaled");
    g.AddNode("offset", new LambdaNode<int, int>(v => v + offset)).In("IN", "scaled").Out("OUT", "y");
    return g;
}
builder.AddSubgraph("first", Affine(2, 1), new Dictionary<string, string> { ["x"] = "in" }, new Dictionary<string, string> { ["y"] = "mid" });
builder.AddSubgraph("second", Affine(10, 0), new Dictionary<string, string> { ["x"] = "mid" }, new Dictionary<string, string> { ["y"] = "out" });
```

## Live stream

- `GraphOptions.MaxInFlight = n`: `AddPacket` mengembalikan `false` (dan memajukan bound input) selama masih ada `n`
  timestamp dalam proses — flow limiter MediaPipe.
- `GraphInputOptions(MaxQueueSize, QueueOverflowPolicy.DropOldest)` membatasi antrean di setiap konsumen.
- `graph.DroppedPackets` menghitung keduanya; metrik `mediapipenet.frames.dropped` juga dipublikasikan.

## Error dan siklus hidup

`StartAsync` membuka node dalam urutan topologis. Exception dari node membuat graph gagal: event `Failed` dipicu,
`WaitUntilDoneAsync` melempar `MediaPipeException`, dan `AddPacket` berikutnya melempar. `Cancel()` berhenti
seketika; `CloseAsync()` menghabiskan antrean lalu menutup; `WaitUntilIdleAsync()` menunggu sampai tidak ada node
yang bisa berjalan.
