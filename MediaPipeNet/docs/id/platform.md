# Platform: desktop, server, mobile, dan browser

> 🇬🇧 [Read in English](../en/platforms.md)

## Didukung dan diuji

| Platform | Status | Execution provider |
|---|---|---|
| Windows x64 | ✅ diuji di CI (semua tes, termasuk validasi silang golden) | CPU, DirectML, CUDA |
| Linux x64 | ✅ diuji di CI | CPU, CUDA |
| macOS arm64 | ✅ diuji di CI | CPU, CoreML |
| Android (.NET MAUI) | 🟡 aplikasi sampel dapat di-build (`samples/MediaPipeNet.Maui`); belum diuji di perangkat lewat CI | CPU, NNAPI |
| iOS (.NET MAUI) | 🟡 sampel yang sama, perlu Mac untuk build | CPU, CoreML |
| Windows ARM64, Linux ARM64, macOS x64 | 🟡 seharusnya berjalan (ONNX Runtime menyediakannya), belum di CI | CPU |

Aplikasi konsol, ASP.NET Core, worker service, aplikasi desktop WPF/WinForms/Avalonia, dan Polyglot notebook
memakai paket yang sama.

## .NET MAUI (Android / iOS)

Library menargetkan `net10.0` tanpa kode khusus platform, sehingga berjalan di dalam aplikasi MAUI.
[`samples/MediaPipeNet.Maui`](https://github.com/DotNetVibeCoderz/Vibe_ML/tree/main/MediaPipeNet/samples/MediaPipeNet.Maui) adalah aplikasi lengkap — pilih foto, jalankan face
stylizer, face mesh, atau deteksi objek, pilih execution provider, lihat hasil dan latensinya:

```bash
dotnet workload install maui-android              # sekali saja
dotnet build samples/MediaPipeNet.Maui -f net10.0-android
dotnet build samples/MediaPipeNet.Maui -t:Run -f net10.0-android   # perangkat atau emulator terhubung
```

Yang ditunjukkan sampel, dan yang dibutuhkan aplikasi Anda:

1. **Runtime native** — referensikan paket task plus `Microsoft.ML.OnnxRuntime`, yang membawa `.aar` Android (dengan
   NNAPI) dan `.xcframework` iOS (dengan CoreML). Jangan referensikan `Gravicode.MediaPipeNet`, yang membawa native
   desktop.
2. **Provider** — `ExecutionProvider.Nnapi` (Android 8.1+, NPU/GPU/DSP, presisi float32 penuh) dan
   `ExecutionProvider.CoreML` (iOS). `Auto` mencoba CUDA → DirectML → CoreML → NNAPI → CPU; operator yang tidak bisa
   dijalankan provider mobile jatuh ke CPU. `ExecutionProviderSelector.GetAvailableProviders()` menampilkan apa yang
   tersedia di perangkat.
3. **Model** — paket model menyalin file ke folder output, yang tidak ada di dalam APK/IPA. Sertakan file `.onnx`
   sebagai `MauiAsset`, salin ke `FileSystem.AppDataDirectory` saat pertama kali dijalankan, dan arahkan
   `BaseOptions.ModelPaths` (id katalog → file) ke sana — lihat `ModelInstaller.cs`. Sampel menyertakan varian INT8
   (seperempat ukuran): sekitar 14 MB untuk lima model.
4. **Frame kamera** — bungkus buffer RGBA kamera platform dengan `MPImage.FromPixelData`.

Sampel di-build di Windows (Android) tetapi belum dijalankan proyek ini di perangkat atau emulator, dan build mobile
belum masuk CI (image CI tidak punya workload MAUI) — anggap mobile sebagai preview di 1.0.

## Blazor WebAssembly — sisi server sekarang, di browser setelah 1.0

API managed ONNX Runtime memanggil library native lewat P/Invoke yang tidak ada di browser, sehingga task tidak bisa
berjalan di dalam Blazor WebAssembly apa adanya. Pilihannya:

- **Inferensi di server** (Blazor Server, atau API ASP.NET Core yang dipanggil klien WASM) — sudah berfungsi dengan
  ekstensi DI; lihat [Integrasi](integrasi.md).
- **onnxruntime-web** lewat JS interop, memakai ulang pra/pasca-pemrosesan MediaPipe.NET (C# biasa yang bisa
  dikompilasi ke WASM). Ini memerlukan backend inferensi asinkron di bawah setiap task (inferensi di browser hanya
  bisa `await`, sedangkan task saat ini sinkron). Merancang abstraksi itu terburu-buru akan membekukan API yang salah
  di 1.0, jadi direncanakan sebagai fitur tambahan 1.x (interface backend baru di samping `OnnxModel`, tanpa breaking
  change).

## Face stylizer

Didukung sejak 1.0 sebagai `FaceStylizer` — generator dikonversi ke ONNX setelah resource variable-nya dibekukan dan
batch norm mode training-nya diturunkan; lihat [Task](task.md#facestylizer) dan [Model](model.md#face-stylizer).
Tidak perlu runtime TFLite.
