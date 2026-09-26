# Pemecahan masalah

> 🇬🇧 [Read in English](../en/troubleshooting.md)

**`ModelNotFoundException: Model 'x' could not be resolved`**
Pesan error mencantumkan setiap lokasi yang sudah dicoba. Tambahkan paket `Gravicode.MediaPipeNet.Models.*` untuk task
tersebut, isi `MEDIAPIPENET_MODELS` dengan folder berisi file `.onnx`, atur `BaseOptions.ModelDirectory`, atau
izinkan unduhan (`ModelStore.CreateDefault(allowDownload: true)`, default). Di balik proxy, atur
`HttpClient.DefaultProxy`.

**"checksum mismatch"**
File bukan yang diharapkan katalog (unduhan tidak lengkap, konversi berbeda). Hapus file dan biarkan store
mengunduh ulang, atau arahkan `BaseOptions.ModelPaths[id]` ke file Anda (path eksplisit tidak diverifikasi).

**`Could not create an ONNX Runtime session … Make sure a native ONNX Runtime package … is referenced`**
Anda hanya mereferensikan paket lapisan (`MediaPipeNet.Tasks.Vision`) tanpa runtime native. Referensikan
`Gravicode.MediaPipeNet` (CPU), `Gravicode.MediaPipeNet.DirectML`, atau `Gravicode.MediaPipeNet.Cuda` — cukup salah satu.

**Provider GPU tidak dipakai**
Periksa `OnnxModel.Provider`/readout di Gallery dan log: `Auto` diam-diam kembali ke CPU bila library CUDA tidak ada
atau DirectML tidak tersedia. `mediapipenet-cli info` menampilkan provider yang tersedia di runtime yang dimuat.
Set `FallbackToCpu = false` agar mendapat exception.

**`InvalidOperationException: … was created in Image mode; this method requires Video mode`**
Buat task dengan `RunningMode` yang sesuai. Mode video membutuhkan `timestampMs` yang naik ketat.

**Handedness terbalik**
Handedness MediaPipe mengasumsikan gambar tercermin (selfie). Untuk kamera belakang atau foto tanpa cermin, tukar
nilainya, atau cerminkan frame (`LiveStreamProcessorOptions.MirrorFrames = true`, `MPImage.FlipHorizontal()`).

**Wajah tidak terdeteksi pada foto seluruh badan**
`FaceDetector` memakai model short-range (wajah dalam ~2 m, lebih besar dari ~15 % frame). Crop dengan
`ImageProcessingOptions.RegionOfInterest`, atau gunakan `HolisticLandmarker` yang menurunkan area wajah dari pose.

**Webcam tidak terbuka**
Coba indeks lain (`WebcamFrameSource.ListCameras()`), tutup aplikasi lain yang memakai kamera, dan pastikan paket
runtime OpenCvSharp untuk OS Anda direferensikan.

**CPU tinggi saat menjalankan beberapa task**
Setiap sesi memakai semua core fisik secara default; atur `InferenceOptions.IntraOpThreads` (mis. 2) per task.

**Linux: `DllNotFoundException: onnxruntime`**
Gunakan distribusi berbasis glibc (Debian/Ubuntu); Alpine/musl tidak didukung ONNX Runtime.

**Lisensi ImageSharp**
MediaPipe.NET memakai SixLabors.ImageSharp 3.1 (Six Labors Split License: ketentuan Apache-2.0 untuk open source dan
organisasi di bawah ambang pendapatan). Pengguna komersial di atas ambang membutuhkan lisensi Six Labors.
