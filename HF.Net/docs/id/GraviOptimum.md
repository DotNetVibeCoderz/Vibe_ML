# GraviOptimum

**Inferensi sadar-perangkat-keras lewat ONNX Runtime, dan kuantisasi yang melaporkan ongkosnya.**

Padanan `optimum`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviOptimum;
```

## Mengapa ini ada

Encoder terkelola di [GraviTransformers](GraviTransformers.md) menghitung dalam `double` dan sepakat
dengan torch hingga sekitar 1e-13. Ia ada supaya model bisa dimuat, diperiksa dan dipahami dalam
.NET murni. Permintaan produksi sebaiknya lewat ekspor ONNX yang berjalan di kernel presisi tunggal
yang ditulis untuk perangkat kerasnya. Pada `bert-base-uncased`, satu kalimat 12 token memakan
**31,5 ms** lewat jalur ini, melawan 37,6 ms di torch dan 48,1 ms secara terkelola. Lihat
[benchmark](benchmarks.md).

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

## Presisi setengah

Masukan dan keluaran float16 dan bfloat16 dikonversi di perbatasan seperti tipe lain. Kebanyakan
ekspor Stable Diffusion memakai float16 di seluruhnya, termasuk timestep, dan tidak perlu diatur apa
pun.

## Membaca dan menambal berkas model

```csharp
var file = OnnxModelFile.Read("unet/model.onnx");
file.Nodes;          // nama, tipe op, masukan, keluaran, nama atribut
file.Initializers;   // nama, bentuk, tipe data - termasuk berkas data eksternal
file.ReadFloats(file.Initializers["onnx::MatMul_2567"]);

var (bytes, changed) = OnnxModelFile.SetFloatAttribute("vae_encoder/model.onnx", "RandomNormalLike", "scale", 0f);
```

Pembaca protobuf streaming, tanpa ketergantungan pada paket `onnx`: ia membaca struktur graf dan nilai
initializer mana pun tanpa memuat seluruh berkas. `SetFloatAttribute` menambah atau mengubah satu
atribut pada setiap node bertipe tertentu dan mengembalikan model yang sudah ditambal - begitulah
sampling tanpa seed milik encoder VAE diubah menjadi reratanya.

## Mengganti bobot tanpa menulis ulang berkas

```csharp
OnnxSession.Open(path, target, new Dictionary<string, InitializerOverride>
{
    ["onnx::MatMul_2567"] = new InitializerOverride(values, shape, AsHalf: true),
});
```

Melalui `SessionOptions.AddInitializer` milik ONNX Runtime: sesi memakai nilai yang diberikan sebagai
pengganti nilai tersimpan. Begitulah LoRA digabungkan ke UNet difusi.

## Model dari cache Hugging Face milik Python

Cache Python menyimpan setiap berkas sekali di `blobs/` dan menautkannya dari folder snapshot. ONNX
Runtime yang lebih baru menelusuri berkas data eksternal model (`weights.pb`, `model.onnx_data`) ke
jalur aslinya dan menolaknya karena keluar dari folder model. Karena itu folder yang berisi tautan
dibuka melalui folder kembaran berisi hard link ke berkas yang sama - nama kedua, bukan salinan kedua -
sehingga model yang diunduh Python bisa dibuka apa adanya.

## Kernel yang tidak dimiliki CPU

Ekspor yang dioptimalkan untuk GPU memakai operator gabungan `com.microsoft` milik ONNX Runtime
(`NhwcConv`, `GroupNorm`, `BiasSplitGelu`), yang tidak punya kernel CPU. Membukanya di CPU ditolak
dengan menyebut nama operatornya dan apa yang harus dilakukan - jalankan dengan
`ExecutionTarget.Cuda`, atau pakai ekspor yang tidak dioptimalkan.

## Lihat juga

[GraviTransformers](GraviTransformers.md) · [GraviDiffusers](GraviDiffusers.md) · [GraviAccelerate](GraviAccelerate.md)
