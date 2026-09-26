# Memulai

> 🇬🇧 [Read in English](../en/getting-started.md) · MediaPipe.NET — dibuat oleh Gravicode Studios, dipimpin Kang Fadhil

## Kebutuhan

- .NET 10 SDK
- Windows 10/11, Ubuntu 22.04+, atau macOS 13+ (x64 atau ARM64)
- Tidak wajib GPU. Opsional: GPU DirectX 12 apa pun (DirectML) atau GPU NVIDIA dengan CUDA 12 + cuDNN 9.

## 1. Buat proyek dan tambahkan paket

```bash
dotnet new console -n Halo.MediaPipe
cd Halo.MediaPipe
dotnet add package Gravicode.MediaPipeNet
dotnet add package Gravicode.MediaPipeNet.Models.Face   # model wajah, disalin ke bin/…/models
```

`Gravicode.MediaPipeNet` berisi semua task dan ONNX Runtime versi CPU. Paket `Gravicode.MediaPipeNet.Models.*` bersifat opsional: tanpa
paket itu, task mengunduh paket model yang dibutuhkan dari nuget.org saat pertama kali dipakai lalu menyimpannya di
`%LOCALAPPDATA%/MediaPipeNet/models` (`~/.local/share/MediaPipeNet/models` di Linux/macOS). Setiap file diverifikasi
SHA-256-nya sebelum digunakan.

## 2. Deteksi wajah

```csharp
using MediaPipeNet.Imaging;
using MediaPipeNet.Tasks.Vision;

using var detector = FaceDetector.Create();              // model dimuat sekali
using var image = MPImage.Load("orang.jpg");             // JPEG, PNG, BMP, GIF, WebP, TIFF; orientasi EXIF diterapkan

FaceDetectionResult result = detector.Detect(image);
foreach (var face in result.Detections)
{
    Console.WriteLine($"wajah di {face.BoundingBox} (skor {face.Score:P0})");
    var mataKanan = face.Keypoints[(int)FaceKeypoint.RightEye];   // koordinat ternormalisasi [0,1]
}

Console.WriteLine(result.ToJson(indented: true));         // setiap hasil bisa diserialisasi ke JSON
```

Koordinat mengikuti konvensi MediaPipe: bounding box dalam **piksel**, keypoint dan landmark **ternormalisasi** ke
`[0, 1]` terhadap lebar dan tinggi gambar (`landmark.ToPixel(lebar, tinggi)` untuk konversi).

## 3. Sumber gambar

`MPImage` adalah input setiap task — padanan `mp.Image` di MediaPipe. Piksel RGBA disimpan di buffer yang di-pool:

```csharp
using var a = MPImage.Load("foto.jpg");                                // file
using var b = await MPImage.LoadAsync(stream);                         // stream (upload, body HTTP, ...)
using var c = MPImage.Load(bytes);                                     // byte terenkode
using var d = MPImage.FromImage(gambarImageSharp);                     // gambar SixLabors.ImageSharp
using var e = MPImage.FromPixelData(bgr, lebar, tinggi, PixelFormat.Bgr24, stride);   // buffer mentah OpenCV/kamera
```

Dispose gambar setelah selesai (buffer piksel kembali ke pool). Untuk video, pakai ulang satu gambar dengan
`image.CopyFrom(bytes, lebar, tinggi, PixelFormat.Bgr24)` agar tidak ada alokasi per frame.

## 4. Opsi

Setiap task menerima record opsi. Properti yang tidak diisi memakai default MediaPipe:

```csharp
using var hands = HandLandmarker.Create(new HandLandmarkerOptions
{
    NumHands = 2,
    MinHandDetectionConfidence = 0.6f,
    BaseOptions = new BaseOptions
    {
        Inference = new InferenceOptions { Provider = ExecutionProvider.Auto, IntraOpThreads = 4 },
        ModelDirectory = "model-saya",                // opsional: dicari lebih dulu
    },
});
```

## 5. Pembuatan dan pemanggilan asinkron

`Create` me-resolve model secara sinkron (dan bisa memblokir saat mengunduh). Di aplikasi UI dan server gunakan:

```csharp
using var pose = await PoseLandmarker.CreateAsync(new() { OutputSegmentationMasks = true }, cancellationToken);
PoseLandmarkResult result = await pose.DetectAsync(image, cancellationToken: cancellationToken);
```

Task dalam mode image bersifat thread-safe: satu instance dapat melayani banyak request bersamaan.

## 6. Gambar hasilnya (opsional)

```bash
dotnet add package Gravicode.MediaPipeNet.Visualization
```

```csharp
using MediaPipeNet.Visualization;

using var canvas = image.ToImage();          // Image<Rgba32> dari SixLabors.ImageSharp
ResultRenderer.Render(canvas, result);        // tersedia untuk setiap tipe hasil
canvas.SaveAsPng("beranotasi.png");
```

## Langkah berikutnya

- [Task](task.md) — setiap task, opsi, dan hasilnya.
- [Mode & video langsung](video-dan-live-stream.md) — webcam, file video, tracking.
- [Graph API](graph-api.md) — membangun pipeline sendiri.
- [Sampel](../../samples) — `BasicUsage`, `GraphApiDemo`, dan `MediaPipeNet.Gallery` (Avalonia).
