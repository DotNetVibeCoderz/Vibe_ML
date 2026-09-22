# GraviOptimum

**Inferensi sadar-perangkat-keras lewat ONNX Runtime, dan kuantisasi yang melaporkan ongkosnya.**

Padanan `optimum`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviOptimum;
```

## Mengapa ini ada

Encoder terkelola di [GraviTransformers](GraviTransformers.md) bertipe `double` sepenuhnya. Ia ada
supaya model bisa dimuat, diperiksa dan dipahami dalam .NET murni. Permintaan produksi sebaiknya lewat
ekspor ONNX yang berjalan di kernel presisi tunggal yang ditulis untuk perangkat kerasnya.

```csharp
using var model = Optimum.Optimize("hf-internal-testing/tiny-random-BertModel", target: "auto");
using var tokenizer = HfTokenizer.FromPretrained("hf-internal-testing/tiny-random-BertModel");

var feeds = Optimum.BuildEncoderInputs(tokenizer, "HF.Net runs ONNX on .NET.", model.Session);

foreach (var (name, tensor) in model.Run(feeds))
    Console.WriteLine($"{name}: [{string.Join(" x ", tensor.Shape.ToArray())}]");

Console.WriteLine(model.Measure(feeds, iterations: 50));
// Cpu: best 0.67 ms, median 1.04 ms (50 runs)
```

## Execution provider

```csharp
Optimum.Optimize(id, target: "CPU");       // atau CUDA, DirectML, oneDNN, auto
Optimum.OptimizeFile("model.onnx", "cuda");
Optimum.ParseTarget("gpu");                // -> ExecutionTarget.Cuda
```

**Provider yang disebut namanya tetapi tidak tersedia akan melempar exception.** Perilaku bawaan ONNX
Runtime sendiri adalah diam-diam jatuh ke CPU, dan persis begitulah sebuah penerapan "CUDA" ternyata
sudah berjalan di CPU berbulan-bulan. Gunakan `"auto"` bila fallback memang diinginkan; ia mencoba
CUDA, lalu DirectML, lalu CPU.

```csharp
session.RequestedTarget;      // yang diminta
session.ActualTarget;         // yang benar-benar terdaftar
session.RanOnRequestedTarget;
```

Paket dasar `Microsoft.ML.OnnxRuntime` hanya memuat **provider CPU**. CUDA memerlukan
`Microsoft.ML.OnnxRuntime.Gpu`; DirectML memerlukan `Microsoft.ML.OnnxRuntime.DirectML`.

## Membandingkan provider

```csharp
foreach (var report in Optimum.Compare("model.onnx", feeds))
    Console.WriteLine(report);
```

Membandingkan adalah satu-satunya cara mengetahui apakah sebuah akselerator sepadan dengan kerumitan
penerapannya. Untuk encoder kecil pada urutan pendek, provider CPU sering menang, karena biaya
transfer dan peluncuran kernel menjadi porsi kerja yang lebih besar daripada aritmetikanya.

## Pengukuran

```csharp
session.Measure(feeds, iterations: 20, warmup: 5);   // -> LatencyReport
report.Best; report.Median; report.PeakThroughput;
```

**Pemanasan bukan pilihan.** Panggilan pertama membangun rencana eksekusi dan mengalokasikan arena,
dan pada provider GPU ia juga mengompilasi kernel. Mengukurnya berarti melaporkan biaya persiapan
seolah-olah itu biaya inferensi — begitulah ekspor ONNX dilaporkan lebih lambat daripada
sebenarnya.

Waktu terbaik dilaporkan berdampingan dengan median dengan alasan yang sama seperti harness benchmark:
pada laptop yang mengalami throttling, rata-rata justru mengukur suhu ruangan.

## Kuantisasi

```csharp
var report = Optimum.Quantize("model.safetensors", "model-bf16.safetensors",
                              QuantizationLevel.BFloat16);

Console.WriteLine(report);
// 31.9 MB -> 16.0 MB (50 %), max error 1.56E-002, mean 6.79E-005
```

**Galatnya diukur dengan membaca kembali apa yang ditulis**, bukan diramalkan dari formatnya. Hanya
begitulah angkanya mencerminkan pembulatan yang benar-benar terjadi.

### bfloat16 atau float16?

Keduanya enam belas bit dan membelanjakannya berbeda:

| | Eksponen | Mantissa | Konsekuensi |
|---|---|---|---|
| bfloat16 | 8 bit, sama seperti float32 | 7 bit | Jangkauan penuh; langkah lebih kasar |
| float16 | 5 bit | 10 bit | Langkah lebih halus; underflow di bawah ~6e-5 |

**bfloat16 biasanya pilihan yang tepat** meski langkahnya lebih kasar: ia mempertahankan jangkauan
eksponen float32, sehingga bobot di ujung bawah distribusi tetap terwakili. float16 diam-diam
kehilangan ekor distribusi bobot yang berekor panjang.

### Buffer indeks

Sebuah checkpoint tidak seluruhnya berisi bobot. Checkpoint BERT membawa buffer integer `position_ids`
bernilai sampai 511, dan bfloat16 punya delapan bit mantissa — jadi mengkuantisasinya bersama yang
lain **membulatkan 511 menjadi 512** dan melaporkan galat maksimum 1,0 yang sama sekali tidak ada
hubungannya dengan model.

Tensor yang di sumbernya tersimpan sebagai integer dipertahankan presisi penuhnya secara otomatis.
Untuk sumber yang sudah terlanjur dilebarkan ke float, sebutkan namanya:

```csharp
Optimum.Quantize(source, destination, QuantizationLevel.BFloat16,
                 keepFullPrecision: new HashSet<string> { "position_ids" });
```

### Int8

**Ditolak, bukan dihampiri.** Int8 simetris memerlukan skala per-tensor yang disimpan berdampingan
dengan bobot, dan format safetensors tidak punya tempat untuk itu. Kuantisasi graf ONNX-nya saja,
dengan perkakas kuantisasi milik onnxruntime, lalu muat hasilnya di sini.

## Menyusun masukan

```csharp
Optimum.BuildEncoderInputs(tokenizer, text, session);
```

Graf hasil ekspor berbeda dalam hal apakah ia menerima `token_type_ids`, jadi masukan disusun dari
**input yang dideklarasikan session**, bukan dari daftar tetap. Menyodorkan input yang tidak
dideklarasikan graf adalah kesalahan di ORT, demikian pula menghilangkan yang dideklarasikannya.

## Konversi tipe

Semua isi tumpukan Gravicode bertipe `double`; hampir setiap graf ONNX menginginkan `float` untuk
aktivasi dan `int64` untuk id token. Konversi terjadi di batas, dikendalikan oleh tipe elemen yang
dideklarasikan graf — menyodorkan tensor double ke input float melempar exception di dalam runtime
dengan pesan yang tidak menyebut nama tensor maupun pemanggilnya.

## Lihat juga

[GraviTransformers](GraviTransformers.md) · [GraviDiffusers](GraviDiffusers.md) · [GraviAccelerate](GraviAccelerate.md)
