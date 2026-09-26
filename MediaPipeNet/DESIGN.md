# MediaPipe.NET â€” Dokumen Desain / Design Document

**Versi / Version:** 0.1.0 (design) — implementation 0.1.0
**Target Framework:** .NET 10 (C# 13)
**Lisensi / License:** Apache License 2.0 (mengikuti lisensi asli Google MediaPipe)
**Status:** Implemented in 0.1.0 — see the appendix / Diimplementasikan di 0.1.0 — lihat lampiran

---

## Daftar Isi / Table of Contents

1. [Bagian I â€” Bahasa Indonesia](#bagian-i--bahasa-indonesia)
   - Ringkasan
   - Latar Belakang & Tujuan
   - Requirement
   - Daftar Fitur
   - Arsitektur
   - Struktur Repositori
   - Strategi Pengujian
   - Pertimbangan Performa
   - Roadmap
2. [Part II â€” English](#part-ii--english)
   - Overview
   - Background & Goals
   - Requirements
   - Feature List
   - Architecture
   - Repository Structure
   - Testing Strategy
   - Performance Considerations
   - Roadmap

---

# BAGIAN I â€” BAHASA INDONESIA

## 1. Ringkasan

**MediaPipe.NET** adalah proyek port/rewrite dari [Google MediaPipe](https://ai.google.dev/edge/mediapipe) ke ekosistem **.NET 10**. Tujuannya adalah menyediakan pustaka (library) native C#/.NET yang memungkinkan pengembang .NET membangun aplikasi *computer vision* dan *perception* (deteksi wajah, pelacakan tangan, estimasi pose, segmentasi, dsb.) tanpa harus melakukan interop langsung ke pustaka C++ Google MediaPipe atau bergantung pada binding Python.

MediaPipe.NET mengadopsi dua lapisan API yang sama seperti MediaPipe asli:

- **Task API (high-level)** â€” API sederhana berorientasi tugas (`FaceDetector`, `HandLandmarker`, `PoseLandmarker`, dst.) yang cukup dipanggil dengan beberapa baris kode.
- **Graph API (low-level)** â€” Model *Calculator Graph* yang dapat dikomposisi (graph node-based, streaming, packet-based) untuk kasus penggunaan lanjutan (pipeline kustom, multi-model, sinkronisasi multi-stream).

## 2. Latar Belakang & Tujuan

### 2.1 Latar Belakang
- MediaPipe asli ditulis dalam C++ dengan binding resmi untuk Python, JavaScript (Web/WASM), Android (Kotlin/Java), dan iOS (Swift/Obj-C). **Tidak ada binding resmi untuk .NET/C#.**
- Pengembang .NET (termasuk tim di Gravicode Studios) sering membutuhkan kemampuan computer vision (misalnya untuk integrasi perangkat medis, sistem monitoring, atau aplikasi klien) namun harus menulis P/Invoke manual atau menggunakan proses terpisah (Python subprocess) yang tidak efisien dan sulit di-maintain.
- Model-model MediaPipe (BlazeFace, BlazePose, Hand Landmark, Selfie Segmentation, dll.) sudah tersedia dalam format **TFLite**, yang dapat dikonversi ke **ONNX** dan dijalankan secara efisien menggunakan **ONNX Runtime**, yang punya binding resmi dan matang untuk .NET (`Microsoft.ML.OnnxRuntime`).

### 2.2 Tujuan
1. Menyediakan pustaka .NET native (tanpa dependensi Python) untuk menjalankan model-model vision MediaPipe.
2. Mendukung eksekusi di CPU maupun GPU (via `DirectML`, `CUDA`, atau `CoreML` melalui backend ONNX Runtime yang sesuai platform).
3. Menyediakan abstraksi graph/pipeline mirip MediaPipe asli agar pengguna yang familiar dengan MediaPipe C++/Python dapat bermigrasi konsep dengan mudah.
4. Cross-platform: Windows, Linux, dan macOS (x64 & ARM64), termasuk dukungan untuk *headless server* (tanpa GPU/tanpa layar) untuk skenario backend/API.
5. Mudah dipasang sebagai paket NuGet, dengan model-model pretrained yang diunduh otomatis atau disertakan sebagai *content package* terpisah.
6. Dokumentasi dan sample lengkap (README dwibahasa, dokumentasi API, contoh kode, dan notebook interaktif).

### 2.3 Non-Tujuan (Out of Scope untuk v1)
- Tidak mem-port ulang seluruh 100% *calculator* dari MediaPipe C++ (ratusan calculator internal Google). Fokus awal pada *task/solution* yang paling umum dipakai.
- Tidak menyediakan training pipeline (MediaPipe.NET adalah *inference-only* library pada v1).
- Tidak menyasar target WebAssembly/Blazor WASM pada rilis awal (dapat menjadi target masa depan).

## 3. Requirement

### 3.1 Functional Requirements (FR)

| ID | Requirement |
|----|-------------|
| FR-1 | Pustaka harus dapat memuat model ONNX hasil konversi dari model MediaPipe (TFLite â†’ ONNX) dan menjalankan inferensi. |
| FR-2 | Pustaka harus menyediakan Task API tingkat tinggi minimal untuk: Face Detection, Face Mesh (landmark wajah), Hand Landmark Detection, Pose Landmark Detection, Selfie Segmentation, Object Detection. |
| FR-3 | Pustaka harus mendukung input berupa: file gambar (jpg/png/bmp), byte array/`Stream`, `Image<Rgba32>` (ImageSharp), dan frame video (webcam/file) via abstraksi `IFrameSource`. |
| FR-4 | Pustaka harus menyediakan Graph API yang memungkinkan pengguna menyusun pipeline dari node-node (`ICalculatorNode`) yang terhubung melalui *input/output stream* bertipe `Packet<T>`. |
| FR-5 | Pustaka harus mendukung eksekusi *streaming* (video real-time) dengan overhead rendah, termasuk mekanisme *timestamp* dan *packet dropping* seperti MediaPipe asli. |
| FR-6 | Pustaka harus dapat mengekspor hasil deteksi/landmark sebagai struktur data terketik kuat (strongly-typed), bukan array mentah, agar mudah dipakai dan di-*serialize* (JSON). |
| FR-7 | Pustaka harus menyediakan utilitas visualisasi (menggambar landmark, bounding box, mask segmentasi) di atas gambar untuk keperluan debug/demo. |
| FR-8 | Pustaka harus dapat berjalan tanpa GPU (fallback CPU) dan otomatis mendeteksi *execution provider* terbaik yang tersedia. |
| FR-9 | Pustaka harus menyediakan mekanisme manajemen model (download otomatis dari model registry / cache lokal / model yang disertakan dalam paket NuGet terpisah `Gravicode.MediaPipeNet.Models.*`). |
| FR-10 | API publik harus mendukung penggunaan asinkron (`async/await`) dan sinkron. |

### 3.2 Non-Functional Requirements (NFR)

| ID | Requirement |
|----|-------------|
| NFR-1 | **Performa**: latensi inferensi single-frame untuk Face Detection pada CPU modern (x64, 8 core) harus < 50ms per frame pada resolusi 640Ã—480. |
| NFR-2 | **Portabilitas**: harus berjalan di .NET 10 pada Windows 10/11, Ubuntu 22.04+, dan macOS 13+ tanpa modifikasi kode. |
| NFR-3 | **Keandalan**: seluruh public API harus memiliki unit test dengan cakupan (coverage) minimal 70% untuk modul inti (Core, Tasks). |
| NFR-4 | **Kompatibilitas biner**: mengikuti semantic versioning; breaking change hanya pada major version. |
| NFR-5 | **Keamanan memori**: penggunaan `Span<T>`/`Memory<T>` untuk manipulasi buffer gambar guna meminimalkan alokasi dan *GC pressure* pada pipeline real-time. |
| NFR-6 | **Observability**: mendukung `ILogger` (Microsoft.Extensions.Logging) dan metrik dasar (FPS, waktu inferensi) yang dapat diekspos ke `System.Diagnostics.Metrics`. |
| NFR-7 | **Lisensi & kepatuhan**: model pretrained wajib mencantumkan atribusi lisensi asal (Apache-2.0 dari Google MediaPipe), tidak boleh menyertakan model dengan lisensi non-komersial tanpa disclaimer. |
| NFR-8 | **Dokumentasi**: seluruh public API harus memiliki XML doc comment (untuk IntelliSense) dan didokumentasikan dalam dua bahasa (ID/EN) pada situs dokumentasi. |

## 4. Daftar Fitur

### 4.1 Fitur Inti (Core)
- **Graph Engine** â€” model kalkulator/graph mirip MediaPipe (`CalculatorGraph`, `CalculatorNode`, `Packet<T>`, `Timestamp`).
- **Tensor & Image Abstraction** â€” wrapper di atas `DenseTensor<float>` (ONNX Runtime) dan `Image<Rgba32>` (ImageSharp) dengan konversi otomatis (resize, normalize, letterbox/pad, colorspace conversion RGB/BGR).
- **Model Loader & Registry** â€” pemuatan model ONNX dari path lokal, embedded resource, atau unduhan otomatis (dengan checksum verification) via `IModelProvider`.
- **Execution Provider Selector** â€” deteksi otomatis backend terbaik: CUDA â†’ DirectML â†’ CoreML â†’ CPU (fallback), dapat dikonfigurasi manual.
- **Frame Source Abstraction** â€” `IFrameSource` dengan implementasi bawaan: `ImageFileFrameSource`, `WebcamFrameSource` (via `OpenCvSharp.VideoCapture` opsional), `VideoFileFrameSource`.

### 4.2 Solutions / Tasks (High-Level API)
| Task | Model dasar (mengacu MediaPipe asli) | Output |
|------|----------------------------------------|--------|
| `FaceDetector` | BlazeFace (short/full range) | Bounding box wajah + 6 keypoint |
| `FaceLandmarker` (Face Mesh) | Face Mesh 468 landmark (+ iris opsional) | 468/478 landmark 3D wajah, blendshapes |
| `HandLandmarker` | Palm Detection + Hand Landmark | 21 landmark 3D per tangan, handedness |
| `PoseLandmarker` | BlazePose (Lite/Full/Heavy) | 33 landmark 3D tubuh + segmentation mask opsional |
| `HolisticLandmarker` | Gabungan Face + Hand + Pose | Landmark gabungan wajah, tangan kiri/kanan, pose |
| `SelfieSegmenter` | Selfie Segmentation (general/landscape) | Mask segmentasi biner/alpha |
| `ObjectDetector` | EfficientDet-Lite | Bounding box + label + skor untuk objek umum (COCO) |
| `ImageClassifier` | MobileNet/EfficientNet | Top-K label klasifikasi gambar |
| `GestureRecognizer` | Hand Landmark + gesture classifier | Label gestur tangan (thumbs up, victory, dst.) |

### 4.3 Fitur Pendukung
- **Visualization Utilities** â€” `LandmarkDrawer`, `BoundingBoxDrawer`, `SegmentationMaskOverlay` untuk menggambar hasil di atas `Image<Rgba32>`.
- **Serialization** â€” hasil deteksi (`FaceDetectionResult`, `HandLandmarkResult`, dst.) mendukung `System.Text.Json` serialization out-of-the-box.
- **Streaming Pipeline Helper** â€” `LiveStreamProcessor<TResult>` untuk memudahkan pemrosesan webcam/video real-time dengan *frame skipping* otomatis saat inferensi lebih lambat dari frame rate sumber.
- **Dependency Injection Integration** â€” extension method `IServiceCollection.AddMediaPipeNet(...)` untuk registrasi task sebagai service (cocok untuk ASP.NET Core / Worker Service).
- **Metrics & Logging** â€” integrasi `ILogger<T>` dan meter OpenTelemetry-compatible.
- **CLI Tool (opsional)** â€” `mediapipenet-cli` (dotnet tool) untuk menjalankan task dari command line tanpa menulis kode, cocok untuk pengujian cepat.

## 5. Arsitektur

### 5.1 Lapisan (Layers)

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Tasks API (high-level)                                  â”‚
â”‚  MediaPipeNet.Tasks.Vision.*  (FaceDetector, dll.)        â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Graph API (mid-level)                                    â”‚
â”‚  MediaPipeNet.Framework  (CalculatorGraph, Packet<T>)      â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Inference Layer                                           â”‚
â”‚  MediaPipeNet.Inference  (ONNX Runtime wrapper, EP select) â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Imaging & Tensor Layer                                     â”‚
â”‚  MediaPipeNet.Imaging  (ImageSharp interop, tensor utils)   â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Core / Primitives                                          â”‚
â”‚  MediaPipeNet.Core  (Timestamp, Rect, Point3D, Options)     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 5.2 Diagram Graph Engine (mermaid)

```mermaid
flowchart LR
    A[IFrameSource] -->|Packet<Image>| B(CalculatorNode: PreProcess)
    B -->|Packet<Tensor>| C(CalculatorNode: Inference - ONNX)
    C -->|Packet<Tensor>| D(CalculatorNode: PostProcess/Decode)
    D -->|Packet<LandmarkList>| E(CalculatorNode: Tracking/Smoothing)
    E -->|Packet<Result>| F[Output Sink / Callback]
```

### 5.3 Konsep Kunci
- **Packet<T>** â€” pembungkus data ber-*timestamp* yang mengalir antar node, meniru `mediapipe::Packet`.
- **CalculatorNode** â€” unit pemrosesan dengan `Process(CalculatorContext ctx)`, memiliki *input stream* dan *output stream* bernama.
- **CalculatorGraph** â€” kumpulan node + edge (stream) yang divalidasi saat build (topological sort, deteksi siklus).
- **SidePacket** â€” data konfigurasi statis yang di-inject sekali di awal graph (mis. path model, threshold).
- **Task** â€” pembungkus siap pakai di atas `CalculatorGraph` dengan graph pre-built yang sudah dioptimalkan untuk skenario umum (single image / video / live stream), menyediakan 3 mode seperti MediaPipe asli: `IMAGE`, `VIDEO`, `LIVE_STREAM`.

### 5.4 Struktur Proyek (Solution Layout)

```
MediaPipeNet.sln
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ MediaPipeNet.Core/            # primitives: Timestamp, Rect, NormalizedLandmark, Options
â”‚   â”œâ”€â”€ MediaPipeNet.Imaging/         # ImageSharp <-> Tensor conversion, resize/pad/normalize
â”‚   â”œâ”€â”€ MediaPipeNet.Inference/       # ONNX Runtime wrapper, execution provider selection
â”‚   â”œâ”€â”€ MediaPipeNet.Framework/       # Graph engine: CalculatorGraph, Packet<T>, CalculatorNode
â”‚   â”œâ”€â”€ MediaPipeNet.Tasks.Vision/    # FaceDetector, HandLandmarker, PoseLandmarker, dst.
â”‚   â”œâ”€â”€ MediaPipeNet.Tasks.Audio/     # (roadmap) audio classification, VAD
â”‚   â”œâ”€â”€ MediaPipeNet.Extensions.DI/   # DI integration untuk ASP.NET Core/Worker
â”‚   â””â”€â”€ MediaPipeNet.Cli/             # dotnet tool CLI
â”œâ”€â”€ models/
â”‚   â””â”€â”€ Gravicode.MediaPipeNet.Models.*/        # paket NuGet terpisah per model (agar core package tetap ringan)
â”œâ”€â”€ samples/
â”‚   â”œâ”€â”€ BasicUsage/                   # console app: task API dasar
â”‚   â””â”€â”€ GraphApiDemo/                 # console app: custom graph
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ MediaPipeNet_QuickStart.ipynb # Polyglot Notebook (.NET Interactive)
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ MediaPipeNet.Core.Tests/
â”‚   â”œâ”€â”€ MediaPipeNet.Framework.Tests/
â”‚   â””â”€â”€ MediaPipeNet.Tasks.Tests/
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ id/                           # dokumentasi Bahasa Indonesia
â”‚   â””â”€â”€ en/                           # English documentation
â”œâ”€â”€ README.md
â””â”€â”€ DESIGN.md
```

### 5.5 Teknologi & Dependensi
| Komponen | Pilihan |
|----------|---------|
| Runtime target | .NET 10 (LTS berikutnya setelah .NET 8; C# 13) |
| Inferensi | `Microsoft.ML.OnnxRuntime` (+ `.Gpu`, `.DirectML` sesuai platform) |
| Imaging | `SixLabors.ImageSharp` (cross-platform, tanpa dependensi native System.Drawing) |
| Video capture (opsional) | `OpenCvSharp4` (untuk webcam/video, opt-in via package terpisah) |
| Testing | `xUnit` + `FluentAssertions` |
| Benchmark | `BenchmarkDotNet` |
| Notebook | Polyglot Notebooks (`.NET Interactive`) â€” kernel C# |
| CI/CD | GitHub Actions (build matrix: win-x64, linux-x64, osx-arm64) |
| Dokumentasi | DocFX atau Markdown statis + mkdocs (dwibahasa via i18n folder) |

## 6. Strategi Pengujian
1. **Unit test** murni logika (Core, Framework graph validation, tensor conversion) â€” tanpa model, cepat dijalankan di CI.
2. **Golden-image test** â€” menjalankan task pada sekumpulan gambar tetap (fixture) dan membandingkan landmark/bounding box terhadap hasil referensi (dengan toleransi epsilon), untuk mendeteksi regresi numerik akibat perubahan versi ONNX Runtime.
3. **Cross-validation dengan MediaPipe Python** â€” sebagai bagian dari proses port awal, hasil MediaPipe.NET dibandingkan terhadap output MediaPipe Python resmi pada dataset sampel untuk memvalidasi akurasi konversi model.
4. **Performance benchmark** â€” menggunakan BenchmarkDotNet untuk mengukur FPS/latensi per task, dijalankan otomatis di CI pada setiap rilis minor.
5. **Integration test streaming** â€” mensimulasikan input video untuk memverifikasi *timestamp monotonicity* dan *packet dropping* pada mode `LIVE_STREAM`.

## 7. Pertimbangan Performa
- Gunakan `ArrayPool<byte>`/`Memory<T>` untuk buffer gambar agar meminimalkan tekanan GC pada loop video real-time.
- Reuse `InferenceSession` (ONNX Runtime) â€” sesi model dibuat sekali dan dipakai berulang, bukan per-frame.
- Preallocate tensor input/output berdasarkan ukuran model tetap.
- Opsi *model quantization* (INT8/FP16) untuk skenario CPU-bound atau edge device.
- Dukungan *batching* opsional untuk skenario non-real-time (memproses banyak gambar sekaligus).

## 8. Roadmap

| Fase | Cakupan | Estimasi |
|------|---------|----------|
| **M0 â€” Fondasi** | `MediaPipeNet.Core`, `MediaPipeNet.Imaging`, wrapper ONNX Runtime dasar | 4 minggu |
| **M1 â€” Face & Hand** | `FaceDetector`, `FaceLandmarker`, `HandLandmarker` + sample + notebook | 6 minggu |
| **M2 â€” Pose & Segmentation** | `PoseLandmarker`, `SelfieSegmenter`, `HolisticLandmarker` | 6 minggu |
| **M3 â€” Graph API Publik** | `CalculatorGraph` dapat dikustomisasi penuh oleh pengguna, dokumentasi lanjutan | 4 minggu |
| **M4 â€” Object Detection & Classification** | `ObjectDetector`, `ImageClassifier`, `GestureRecognizer` | 4 minggu |
| **M5 â€” Tooling & DX** | CLI tool, DI integration, benchmark publik, dokumentasi situs (mkdocs) | 3 minggu |
| **M6 â€” Stabilisasi v1.0** | Hardening, cross-platform testing, audit performa & lisensi | 3 minggu |

---

# PART II â€” ENGLISH

## 1. Overview

**MediaPipe.NET** is a port/rewrite of [Google MediaPipe](https://ai.google.dev/edge/mediapipe) for the **.NET 10** ecosystem. Its goal is to provide a native C#/.NET library that lets .NET developers build computer vision and perception applications (face detection, hand tracking, pose estimation, segmentation, etc.) without direct interop into Google's C++ MediaPipe library or reliance on Python bindings.

MediaPipe.NET adopts the same two API layers as the original MediaPipe:

- **Task API (high-level)** â€” a simple, task-oriented API (`FaceDetector`, `HandLandmarker`, `PoseLandmarker`, etc.) callable in a few lines of code.
- **Graph API (low-level)** â€” a composable *Calculator Graph* model (node-based, streaming, packet-based) for advanced use cases (custom pipelines, multi-model, multi-stream synchronization).

## 2. Background & Goals

### 2.1 Background
- Original MediaPipe is written in C++ with official bindings for Python, JavaScript (Web/WASM), Android (Kotlin/Java), and iOS (Swift/Obj-C). **There is no official .NET/C# binding.**
- .NET developers (including the team at Gravicode Studios) often need computer vision capability (e.g., for medical device integration, monitoring systems, or client-facing apps) but end up writing manual P/Invoke or spawning separate Python processes, which is inefficient and hard to maintain.
- MediaPipe's models (BlazeFace, BlazePose, Hand Landmark, Selfie Segmentation, etc.) already ship as **TFLite**, which can be converted to **ONNX** and run efficiently via **ONNX Runtime**, which has mature, official .NET bindings (`Microsoft.ML.OnnxRuntime`).

### 2.2 Goals
1. Provide a native .NET library (no Python dependency) to run MediaPipe's vision models.
2. Support both CPU and GPU execution (via `DirectML`, `CUDA`, or `CoreML` through the appropriate ONNX Runtime backend per platform).
3. Provide a graph/pipeline abstraction similar to the original MediaPipe so users familiar with MediaPipe C++/Python can migrate concepts easily.
4. Cross-platform: Windows, Linux, and macOS (x64 & ARM64), including headless server scenarios (no GPU/no display) for backend/API use cases.
5. Easy to install as a NuGet package, with pretrained models either auto-downloaded or shipped as separate content packages.
6. Complete documentation and samples (bilingual README, API docs, code samples, interactive notebook).

### 2.3 Non-Goals (Out of Scope for v1)
- Not porting 100% of the C++ calculators (hundreds of internal Google calculators). Initial focus is on the most commonly used tasks/solutions.
- No training pipeline (MediaPipe.NET is inference-only in v1).
- No WebAssembly/Blazor WASM target in the initial release (potential future target).

## 3. Requirements

### 3.1 Functional Requirements (FR)

| ID | Requirement |
|----|-------------|
| FR-1 | The library must be able to load ONNX models converted from MediaPipe models (TFLite â†’ ONNX) and run inference. |
| FR-2 | The library must provide a high-level Task API for at least: Face Detection, Face Mesh (facial landmarks), Hand Landmark Detection, Pose Landmark Detection, Selfie Segmentation, Object Detection. |
| FR-3 | The library must support input as: image files (jpg/png/bmp), byte array/`Stream`, `Image<Rgba32>` (ImageSharp), and video frames (webcam/file) via an `IFrameSource` abstraction. |
| FR-4 | The library must provide a Graph API allowing users to compose pipelines out of nodes (`ICalculatorNode`) connected via typed `Packet<T>` input/output streams. |
| FR-5 | The library must support streaming (real-time video) execution with low overhead, including timestamp and packet-dropping mechanisms similar to the original MediaPipe. |
| FR-6 | The library must export detection/landmark results as strongly-typed data structures rather than raw arrays, for ease of use and JSON serialization. |
| FR-7 | The library must provide visualization utilities (drawing landmarks, bounding boxes, segmentation masks) on top of images for debugging/demo purposes. |
| FR-8 | The library must run without a GPU (CPU fallback) and automatically detect the best available execution provider. |
| FR-9 | The library must provide model management (auto-download from a model registry / local cache / models shipped in separate `Gravicode.MediaPipeNet.Models.*` NuGet packages). |
| FR-10 | The public API must support both asynchronous (`async/await`) and synchronous usage. |

### 3.2 Non-Functional Requirements (NFR)

| ID | Requirement |
|----|-------------|
| NFR-1 | **Performance**: single-frame inference latency for Face Detection on a modern x64 8-core CPU must be < 50ms per frame at 640Ã—480. |
| NFR-2 | **Portability**: must run on .NET 10 on Windows 10/11, Ubuntu 22.04+, and macOS 13+ without code changes. |
| NFR-3 | **Reliability**: all public APIs must have unit tests with at least 70% coverage for core modules (Core, Tasks). |
| NFR-4 | **Binary compatibility**: follows semantic versioning; breaking changes only on major version bumps. |
| NFR-5 | **Memory safety**: use `Span<T>`/`Memory<T>` for image buffer manipulation to minimize allocations and GC pressure in real-time pipelines. |
| NFR-6 | **Observability**: supports `ILogger` (Microsoft.Extensions.Logging) and basic metrics (FPS, inference time) exposable via `System.Diagnostics.Metrics`. |
| NFR-7 | **Licensing & compliance**: pretrained models must retain attribution to the original license (Apache-2.0 from Google MediaPipe); no non-commercial-licensed models without a disclaimer. |
| NFR-8 | **Documentation**: all public APIs must have XML doc comments (for IntelliSense) and be documented bilingually (ID/EN) on the documentation site. |

## 4. Feature List

### 4.1 Core Features
- **Graph Engine** â€” a calculator/graph model similar to MediaPipe (`CalculatorGraph`, `CalculatorNode`, `Packet<T>`, `Timestamp`).
- **Tensor & Image Abstraction** â€” wrapper over `DenseTensor<float>` (ONNX Runtime) and `Image<Rgba32>` (ImageSharp) with automatic conversion (resize, normalize, letterbox/pad, RGB/BGR colorspace conversion).
- **Model Loader & Registry** â€” loading ONNX models from a local path, embedded resource, or auto-download (with checksum verification) via `IModelProvider`.
- **Execution Provider Selector** â€” automatic detection of the best backend: CUDA â†’ DirectML â†’ CoreML â†’ CPU (fallback), manually configurable.
- **Frame Source Abstraction** â€” `IFrameSource` with built-in implementations: `ImageFileFrameSource`, `WebcamFrameSource` (via optional `OpenCvSharp.VideoCapture`), `VideoFileFrameSource`.

### 4.2 Solutions / Tasks (High-Level API)
| Task | Underlying model (per original MediaPipe) | Output |
|------|--------------------------------------------|--------|
| `FaceDetector` | BlazeFace (short/full range) | Face bounding box + 6 keypoints |
| `FaceLandmarker` (Face Mesh) | Face Mesh 468 landmarks (+ optional iris) | 468/478 3D face landmarks, blendshapes |
| `HandLandmarker` | Palm Detection + Hand Landmark | 21 3D landmarks per hand, handedness |
| `PoseLandmarker` | BlazePose (Lite/Full/Heavy) | 33 3D body landmarks + optional segmentation mask |
| `HolisticLandmarker` | Combined Face + Hand + Pose | Combined face, left/right hand, and pose landmarks |
| `SelfieSegmenter` | Selfie Segmentation (general/landscape) | Binary/alpha segmentation mask |
| `ObjectDetector` | EfficientDet-Lite | Bounding box + label + score for common (COCO) objects |
| `ImageClassifier` | MobileNet/EfficientNet | Top-K image classification labels |
| `GestureRecognizer` | Hand Landmark + gesture classifier | Hand gesture labels (thumbs up, victory, etc.) |

### 4.3 Supporting Features
- **Visualization Utilities** â€” `LandmarkDrawer`, `BoundingBoxDrawer`, `SegmentationMaskOverlay` for drawing results on top of `Image<Rgba32>`.
- **Serialization** â€” detection results (`FaceDetectionResult`, `HandLandmarkResult`, etc.) support `System.Text.Json` serialization out of the box.
- **Streaming Pipeline Helper** â€” `LiveStreamProcessor<TResult>` for easy real-time webcam/video processing with automatic frame skipping when inference is slower than the source frame rate.
- **Dependency Injection Integration** â€” `IServiceCollection.AddMediaPipeNet(...)` extension method to register tasks as services (suited for ASP.NET Core / Worker Service).
- **Metrics & Logging** â€” `ILogger<T>` integration and an OpenTelemetry-compatible meter.
- **CLI Tool (optional)** â€” `mediapipenet-cli` (dotnet tool) to run a task from the command line without writing code, useful for quick testing.

## 5. Architecture

### 5.1 Layers

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Tasks API (high-level)                                  â”‚
â”‚  MediaPipeNet.Tasks.Vision.*  (FaceDetector, etc.)        â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Graph API (mid-level)                                    â”‚
â”‚  MediaPipeNet.Framework  (CalculatorGraph, Packet<T>)      â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Inference Layer                                           â”‚
â”‚  MediaPipeNet.Inference  (ONNX Runtime wrapper, EP select) â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Imaging & Tensor Layer                                     â”‚
â”‚  MediaPipeNet.Imaging  (ImageSharp interop, tensor utils)   â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Core / Primitives                                          â”‚
â”‚  MediaPipeNet.Core  (Timestamp, Rect, Point3D, Options)     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 5.2 Graph Engine Diagram (mermaid)

```mermaid
flowchart LR
    A[IFrameSource] -->|Packet<Image>| B(CalculatorNode: PreProcess)
    B -->|Packet<Tensor>| C(CalculatorNode: Inference - ONNX)
    C -->|Packet<Tensor>| D(CalculatorNode: PostProcess/Decode)
    D -->|Packet<LandmarkList>| E(CalculatorNode: Tracking/Smoothing)
    E -->|Packet<Result>| F[Output Sink / Callback]
```

### 5.3 Key Concepts
- **Packet<T>** â€” a timestamped data wrapper flowing between nodes, mirroring `mediapipe::Packet`.
- **CalculatorNode** â€” a processing unit with `Process(CalculatorContext ctx)`, with named input and output streams.
- **CalculatorGraph** â€” a set of nodes + edges (streams) validated at build time (topological sort, cycle detection).
- **SidePacket** â€” static configuration data injected once at graph startup (e.g., model path, threshold).
- **Task** â€” a ready-to-use wrapper over a `CalculatorGraph` with a pre-built, optimized graph for common scenarios (single image / video / live stream), offering the same 3 modes as the original MediaPipe: `IMAGE`, `VIDEO`, `LIVE_STREAM`.

### 5.4 Repository / Solution Layout

```
MediaPipeNet.sln
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ MediaPipeNet.Core/            # primitives: Timestamp, Rect, NormalizedLandmark, Options
â”‚   â”œâ”€â”€ MediaPipeNet.Imaging/         # ImageSharp <-> Tensor conversion, resize/pad/normalize
â”‚   â”œâ”€â”€ MediaPipeNet.Inference/       # ONNX Runtime wrapper, execution provider selection
â”‚   â”œâ”€â”€ MediaPipeNet.Framework/       # Graph engine: CalculatorGraph, Packet<T>, CalculatorNode
â”‚   â”œâ”€â”€ MediaPipeNet.Tasks.Vision/    # FaceDetector, HandLandmarker, PoseLandmarker, etc.
â”‚   â”œâ”€â”€ MediaPipeNet.Tasks.Audio/     # (roadmap) audio classification, VAD
â”‚   â”œâ”€â”€ MediaPipeNet.Extensions.DI/   # DI integration for ASP.NET Core/Worker
â”‚   â””â”€â”€ MediaPipeNet.Cli/             # dotnet tool CLI
â”œâ”€â”€ models/
â”‚   â””â”€â”€ Gravicode.MediaPipeNet.Models.*/        # separate NuGet package per model (keeps core package lightweight)
â”œâ”€â”€ samples/
â”‚   â”œâ”€â”€ BasicUsage/                   # console app: basic Task API
â”‚   â””â”€â”€ GraphApiDemo/                 # console app: custom graph
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ MediaPipeNet_QuickStart.ipynb # Polyglot Notebook (.NET Interactive)
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ MediaPipeNet.Core.Tests/
â”‚   â”œâ”€â”€ MediaPipeNet.Framework.Tests/
â”‚   â””â”€â”€ MediaPipeNet.Tasks.Tests/
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ id/                           # Indonesian documentation
â”‚   â””â”€â”€ en/                           # English documentation
â”œâ”€â”€ README.md
â””â”€â”€ DESIGN.md
```

### 5.5 Technology & Dependencies
| Component | Choice |
|-----------|--------|
| Target runtime | .NET 10 (next LTS after .NET 8; C# 13) |
| Inference | `Microsoft.ML.OnnxRuntime` (+ `.Gpu`, `.DirectML` per platform) |
| Imaging | `SixLabors.ImageSharp` (cross-platform, no native System.Drawing dependency) |
| Video capture (optional) | `OpenCvSharp4` (for webcam/video, opt-in via separate package) |
| Testing | `xUnit` + `FluentAssertions` |
| Benchmarking | `BenchmarkDotNet` |
| Notebook | Polyglot Notebooks (`.NET Interactive`) â€” C# kernel |
| CI/CD | GitHub Actions (build matrix: win-x64, linux-x64, osx-arm64) |
| Documentation | DocFX or static Markdown + mkdocs (bilingual via i18n folders) |

## 6. Testing Strategy
1. **Unit tests** for pure logic (Core, Framework graph validation, tensor conversion) â€” model-free, fast to run in CI.
2. **Golden-image tests** â€” run tasks against a fixed set of fixture images and compare landmarks/bounding boxes against reference outputs (with epsilon tolerance) to catch numerical regressions from ONNX Runtime version changes.
3. **Cross-validation against MediaPipe Python** â€” as part of the initial porting process, MediaPipe.NET outputs are compared against official MediaPipe Python outputs on a sample dataset to validate conversion accuracy.
4. **Performance benchmarking** â€” using BenchmarkDotNet to measure FPS/latency per task, run automatically in CI on every minor release.
5. **Streaming integration tests** â€” simulate video input to verify timestamp monotonicity and packet dropping in `LIVE_STREAM` mode.

## 7. Performance Considerations
- Use `ArrayPool<byte>`/`Memory<T>` for image buffers to minimize GC pressure in real-time video loops.
- Reuse the ONNX Runtime `InferenceSession` â€” created once and reused, not per frame.
- Preallocate input/output tensors based on fixed model shapes.
- Optional model quantization (INT8/FP16) for CPU-bound or edge-device scenarios.
- Optional batching support for non-real-time scenarios (processing many images at once).

## 8. Roadmap

| Phase | Scope | Estimate |
|-------|-------|----------|
| **M0 â€” Foundation** | `MediaPipeNet.Core`, `MediaPipeNet.Imaging`, basic ONNX Runtime wrapper | 4 weeks |
| **M1 â€” Face & Hand** | `FaceDetector`, `FaceLandmarker`, `HandLandmarker` + sample + notebook | 6 weeks |
| **M2 â€” Pose & Segmentation** | `PoseLandmarker`, `SelfieSegmenter`, `HolisticLandmarker` | 6 weeks |
| **M3 â€” Public Graph API** | Fully user-customizable `CalculatorGraph`, extended docs | 4 weeks |
| **M4 â€” Object Detection & Classification** | `ObjectDetector`, `ImageClassifier`, `GestureRecognizer` | 4 weeks |
| **M5 â€” Tooling & DX** | CLI tool, DI integration, public benchmarks, docs site (mkdocs) | 3 weeks |
| **M6 â€” v1.0 Stabilization** | Hardening, cross-platform testing, performance & license audit | 3 weeks |

---

# Appendix — Implementation decisions / Lampiran — Keputusan implementasi

Created by Gravicode Studios, led by Kang Fadhil. / Dibuat oleh Gravicode Studios, dipimpin Kang Fadhil.

| Topic | Decision (EN) | Keputusan (ID) |
|---|---|---|
| Language | .NET 10 with `LangVersion=latest` (C# 14 features such as primary constructors and collection expressions). | .NET 10 dengan `LangVersion=latest` (fitur C# 14). |
| Image type | `MPImage` (pooled RGBA, mirrors `mp.Image`) instead of a type named `ImageFrame`, which clashes with `SixLabors.ImageSharp.ImageFrame`. `Image<Rgba32>` is accepted through `MPImage.FromImage`. | `MPImage` (RGBA di-pool) menggantikan nama `ImageFrame` yang bentrok dengan ImageSharp. |
| ImageSharp | Pinned to 3.1.x: 4.x requires a license key at build time. | Dikunci ke 3.1.x karena 4.x mewajibkan license key. |
| ONNX Runtime | `Inference` references only `Microsoft.ML.OnnxRuntime.Managed`; meta packages `Gravicode.MediaPipeNet` (CPU), `.DirectML`, `.Cuda` bring the native flavor. Version pinned to 1.24.4 (last DirectML release). | `Inference` hanya mereferensikan API managed; meta package membawa runtime native. Versi 1.24.4. |
| Result naming | `FaceDetectionResult`, `FaceLandmarkResult`, `HandLandmarkResult`, `GestureRecognitionResult`, `PoseLandmarkResult`, `HolisticResult`, `SegmentationResult`, `ObjectDetectionResult`, `ClassificationResult` — per-item records (`HandLandmarks`, `FaceLandmarks`…) instead of parallel lists. | Record per item, bukan list paralel. |
| Selfie segmenter | Implemented as `ImageSegmenter` (MediaPipe Tasks name). | Diimplementasikan sebagai `ImageSegmenter`. |
| Tasks on the graph | Live-stream mode runs each task inside a `CalculatorGraph` (flow limiting); image/video modes call the same processing stages directly for minimal overhead. Tasks are also available as graph nodes (`VisionTaskNode`, `.pbtxt` calculators). | Mode live-stream berjalan di atas `CalculatorGraph`; mode image/video memanggil tahap pemrosesan langsung. |
| Models | Converted with tf2onnx; custom op and sparse-weight handling documented in `docs/*/models.md`. Hosted as NuGet content packages, which also serve as the download registry (nuget.org flat container). | Model di-host sebagai paket NuGet yang sekaligus menjadi registry unduhan. |
| Validation | Golden tests against the official MediaPipe Python package in addition to TFLite-vs-ONNX checks. | Test golden terhadap paket resmi MediaPipe Python. |
| Video capture | `MediaPipeNet.Video.OpenCv` (OpenCvSharp4 4.13), opt-in package. | Paket opsional berbasis OpenCvSharp. |
| Tests | xUnit 2.9 + FluentAssertions 7.2 (the last Apache-2.0 release; 8.x is commercial). | FluentAssertions 7.2 (Apache-2.0). |
| Gallery | Avalonia 11.3; bilingual (EN/ID), light/dark, settings page, code + JSON per task, screenshot automation. | Avalonia 11.3; dwibahasa, terang/gelap, pengaturan, kode + JSON per task. |
| Audio | `MediaPipeNet.Tasks.Audio` remains on the roadmap (PLAN.md). | Tetap di roadmap. |
