# Instalasi

*[English](../installation.md)*

## Kebutuhan

| | |
|---|---|
| **.NET SDK** | 10.0 atau lebih baru |
| **Sistem operasi** | Windows, Linux, atau macOS |
| **RAM** | Minimal 4 GB; 8 GB untuk menjalankan benchmark |
| **GPU** *(opsional)* | Perangkat CUDA atau OpenCL yang mendukung presisi ganda |

Periksa SDK Anda:

```bash
dotnet --version   # diharapkan 10.0.x
```

## Membangun dari sumber

```bash
git clone https://github.com/DotNetVibeCoderz/Vibe_ML.git
cd Vibe_ML/GravicodeScience
dotnet build Gravicode.Science.sln -c Release
dotnet test
```

Seluruh rangkaian tes (1.049 tes) selesai dalam sekitar satu setengah menit.

## Menambahkan library ke proyek Anda

Setiap library berdiri sendiri. Ambil hanya yang Anda perlukan — semuanya bergantung pada
`GraviNum`, selebihnya tidak wajib.

```bash
dotnet add package Gravicode.Science.GraviNum
dotnet add package Gravicode.Science.GraviFrame
dotnet add package Gravicode.Science.GraviLearn
dotnet add package Gravicode.Science.GraviText
dotnet add package Gravicode.Science.GraviGraph
dotnet add package Gravicode.Science.GraviProb
```

Jika bekerja dari hasil clone, rujuk proyeknya langsung:

```xml
<ItemGroup>
  <ProjectReference Include="../GravicodeScience/src/GraviNum/GraviNum.csproj" />
  <ProjectReference Include="../GravicodeScience/src/GraviFrame/GraviFrame.csproj" />
</ItemGroup>
```

Setiap paket membawa dokumentasi XML-nya, sehingga komentar di repositori ini sampai kepada Anda
melalui IntelliSense, serta paket simbol `.snupkg`, sehingga debugger bisa masuk ke dalam kode
pustakanya. Lihat [publishing.md](publishing.md) untuk cara paketnya dibangun dan dirilis.

### Ketergantungan antar library

```
GraviNum ──┬── GraviFrame ── GraviLearn ── GraviText ── GraviGraph
           ├── GraviProb
           └── (semuanya)
```

`GraviNum` tidak pernah merujuk apa pun di atasnya. `GraviText` bergantung pada `GraviLearn`
karena pipeline tugasnya melatih classifier sungguhan alih-alih menulis ulang, dan `GraviGraph`
bergantung pada `GraviText` karena node2vec adalah random walk yang diumpankan ke skip-gram.

## Dukungan GPU

Backend GPU dibangun di atas [ILGPU](https://ilgpu.net) dan **sepenuhnya opsional**. Tidak ada yang
perlu dikonfigurasi: saat pertama kali diminta, library memeriksa perangkat yang tersedia, dan
jika tidak ada yang bisa dipakai, eksekusi diam-diam berlanjut di CPU.

```csharp
using Gravicode.Science.GraviNum.Compute;

Console.WriteLine(Compute.DescribeDevices());
// CPU: CPU (SIMD x4, 8 threads); GPU: OpenCL - Intel(R) UHD Graphics 620 (6502 MB)

Console.WriteLine(Compute.IsGpuAvailable);
```

Memilih backend secara eksplisit:

```csharp
var product = Compute.Dot(a, b, Compute.Gpu);   // paksa GPU
var onCpu   = Compute.Dot(a, b, Compute.Cpu);   // paksa CPU
var best    = Compute.Dot(a, b);                // biarkan ukuran yang menentukan
```

**GPU tidak otomatis lebih cepat.** Setiap panggilan menyalin kedua operand melewati bus dan
mengembalikan hasilnya; biaya itu tetap sementara beban aritmetika bertambah seiring ukuran
masalah. Di bawah sekitar 250.000 elemen, CPU menang. Lihat [benchmarks.md](benchmarks.md) untuk
titik silang yang terukur.

### Kebutuhan driver

| Backend | Membutuhkan |
|---|---|
| CUDA | Driver NVIDIA 450 atau lebih baru |
| OpenCL | Runtime OpenCL dari vendor (Intel, AMD, atau NVIDIA) |
| CPU accelerator | Tidak ada — fallback perangkat lunak ILGPU, hanya berguna untuk menguji kernel |

Jika `Compute.IsGpuAvailable` bernilai `false`, penyebab paling umum adalah runtime OpenCL yang
belum terpasang atau perangkat tanpa dukungan float64. Keduanya bukan kondisi error: jalur CPU
lengkap fiturnya dan itulah yang diuji oleh rangkaian tes.

## Memverifikasi instalasi

```csharp
using Gravicode.Science.GraviNum;

Console.WriteLine(GraviInfo.Banner("GraviNum"));
Console.WriteLine(GraviInfo.HardwareReport());

var a = NdArray.Arange(6).Reshape(2, 3);
Console.WriteLine(a.Dot(a.T));
```

Atau jalankan salah satu dari enam contoh, yang mencetak laporan yang sama sebelum bekerja:

```bash
dotnet run --project samples/GraviNum.Console
dotnet run --project samples/GraviFrame.Console
dotnet run --project samples/GraviLearn.Console
dotnet run --project samples/GraviText.Console
dotnet run --project samples/GraviGraph.Console
dotnet run --project samples/GraviProb.Console
```

## Notebook

Notebook di [`notebooks/`](../../notebooks) memerlukan ekstensi
[Polyglot Notebooks](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.dotnet-interactive-vscode)
untuk VS Code. Notebook merujuk ke assembly hasil build, jadi build Release dulu:

```bash
dotnet build Gravicode.Science.sln -c Release
```

## Benchmark

BenchmarkDotNet mewajibkan build Release dan akan menolak berjalan tanpanya:

```bash
dotnet run --project benchmarks/GraviNum.Benchmark -c Release
dotnet run --project benchmarks/GraviNum.Benchmark -c Release -- --filter "*MatrixProduct*"
```

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
