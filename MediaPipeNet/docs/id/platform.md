# Platform: desktop, server, mobile, dan browser

> 🇬🇧 [Read in English](../en/platforms.md)

## Didukung dan diuji

| Platform | Status | Execution provider |
|---|---|---|
| Windows x64 | ✅ diuji di CI (semua tes, termasuk validasi silang golden) | CPU, DirectML, CUDA |
| Linux x64 | ✅ diuji di CI | CPU, CUDA |
| macOS arm64 | ✅ diuji di CI | CPU, CoreML |
| Windows ARM64, Linux ARM64, macOS x64 | 🟡 seharusnya berjalan (ONNX Runtime menyediakannya), belum di CI | CPU |

Aplikasi konsol, ASP.NET Core, worker service, aplikasi desktop WPF/WinForms/Avalonia, dan Polyglot notebook
memakai paket yang sama.

## .NET MAUI (Android / iOS) — sudah dievaluasi, belum didukung

Library menargetkan `net10.0` tanpa kode khusus platform, sehingga bisa dikompilasi ke aplikasi MAUI. Yang masih
diperlukan agar mobile didukung resmi:

1. **Runtime native**: referensikan `Microsoft.ML.OnnxRuntime` (menyediakan aset `.aar` Android dan `.xcframework`
   iOS) sebagai pengganti `Gravicode.MediaPipeNet`, yang membawa runtime CPU desktop.
2. **Provider**: `ExecutionProvider.CoreML` berjalan di iOS; NNAPI (Android) belum tersedia di `InferenceOptions` —
   `ExecutionProvider.Cpu` berjalan di mana saja.
3. **Model**: paket model menyalin file ke folder output, yang tidak ada di dalam APK/IPA. Sertakan file `.onnx`
   sebagai `MauiAsset` lalu sajikan dengan `EmbeddedResourceModelProvider`, atau salin ke
   `FileSystem.AppDataDirectory` dan arahkan `BaseOptions.ModelDirectory` ke sana.
4. **Frame kamera**: bungkus buffer RGBA/NV21 kamera platform dengan `MPImage.FromPixelData`.
5. **Ukuran**: pilih varian `ModelPrecision.Int8` / `Float16` dan hanya model yang dibutuhkan.

Langkah-langkah ini diperkirakan berhasil tetapi belum dicakup CI; sampel MAUI dan opsi provider NNAPI ada di
roadmap.

## Blazor WebAssembly — sudah dievaluasi, di luar cakupan

API managed ONNX Runtime memanggil library native lewat P/Invoke yang tidak tersedia di browser, sehingga task tidak
bisa berjalan di Blazor WebAssembly apa adanya. Opsi yang realistis:

- **Inferensi di server** (Blazor Server, atau API ASP.NET Core yang dipanggil klien WASM) — sudah bisa hari ini
  dengan ekstensi DI; lihat [Integrasi](integrasi.md).
- **onnxruntime-web** lewat JS interop, memakai ulang pre/post-processing MediaPipe.NET (C# murni yang bisa
  dikompilasi ke WASM) — mungkin dilakukan tetapi butuh abstraksi `IInferenceBackend` di atas `OnnxModel`; dijadwalkan
  setelah 1.0.

## Face stylizer

Face stylizer MediaPipe (`face_stylizer_color_sketch.task`) tidak di-port: graph generatornya menginisialisasi
resource variable lewat `CALL_ONCE`/`VAR_HANDLE`/`ASSIGN_VARIABLE`, memakai op Flex TensorFlow `FusedBatchNormV3`,
dan mengambil noise `RANDOM_STANDARD_NORMAL` — semuanya tidak dapat dikonversi tf2onnx — dan paket resmi MediaPipe
Python 1.0.1 pun tidak menyertakan task ini, sehingga tidak ada referensi untuk validasi. Akan dipertimbangkan lagi
ketika Google menerbitkan model yang dapat dikonversi.
