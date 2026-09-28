# Performa & GPU

> 🇬🇧 [Read in English](../en/performance.md)

## Hasil pengukuran

BenchmarkDotNet 0.15.8, .NET 10.0.12, Windows 11, **Intel Core i7-8650U** (ultrabook 2017, 4 core / 8 thread),
provider CPU, input 640×480, end-to-end (image-to-tensor, inferensi, decoding):

| Task | Rata-rata | StdDev | Alokasi per panggilan |
|---|---:|---:|---:|
| FaceDetector | **8,06 ms** | 0,14 ms | 3,3 KB |
| ImageSegmenter (selfie) | 9,58 ms | 0,19 ms | 2,9 MB ¹ |
| ImageClassifier | 10,50 ms | 0,25 ms | 58 KB |
| FaceLandmarker | 25,08 ms | 0,39 ms | 18,7 KB |
| ObjectDetector | 30,56 ms | 3,69 ms | 17,2 KB |
| HandLandmarker (image) | 31,44 ms | 1,13 ms | 11,1 KB |
| GestureRecognizer | 31,77 ms | 0,70 ms | 12,3 KB |
| PoseLandmarker (lite) | 36,90 ms | 0,39 ms | 10,7 KB |
| ImageToTensor 720p → 256² letterbox (anti-aliasing) | 6,54 ms | | 3,2 KB |
| ImageToTensor 720p ROI berotasi → 256² | 1,69 ms | | 3,1 KB |

¹ Mask output itu sendiri (640×480 float).

**NFR-1** (deteksi wajah < 50 ms pada 640×480 di CPU 8-core modern): terpenuhi dengan margin 6× di CPU 4-core 2017.

### Task 0.2 / 0.3 (mesin yang sama, BenchmarkDotNet short run)

| Task | Rata-rata | Alokasi per panggilan |
|---|---:|---:|
| ImageEmbedder (MobileNet V3, foto 820×1024) | 3,6 ms | 7 KB |
| ImageSegmenter, selfie multiclass + category mask (820×1024) | 104 ms | 22 MB ¹ |
| AudioClassifier, klip 4,3 s (5 jendela YAMNet) | 14,0 ms | 256 KB |
| TextClassifier, MobileBERT | 56 ms | 4 KB |
| TextClassifier, average word | 0,024 ms | 2,5 KB |
| TextEmbedder, MobileBERT | 56 ms | 6 KB |
| LanguageDetector | 0,062 ms | 12 KB |

¹ Enam mask probabilitas resolusi penuh ditambah category mask; semua kanal diproyeksikan dalam satu pass
(`TensorWarp.ProjectChannelsToImage`), 1,6× lebih cepat daripada proyeksi per kanal.

### Fitur runtime (FaceLandmarker pada portrait, CPU)

| Konfigurasi | Rata-rata | Rasio |
|---|---:|---:|
| float32 | 20,0 ms | 1,00 |
| float32 + `UseIoBinding` | 21,0 ms | 1,05 |
| `ModelPrecision.Float16` | 23,7 ms | 1,19 |
| `ModelPrecision.Int8` (bobot saja) | 22,8 ms | 1,15 |
| 8 gambar, loop berurutan | 24,3 ms / gambar | |
| 8 gambar, `ProcessBatch` | **12,3 ms / gambar** | 2× throughput |

Di CPU, I/O binding dan FP16 tidak menguntungkan (keduanya ditujukan untuk GPU, di mana menghindari salinan
host↔device dan memakai unit half-precision); varian INT8 menukar ~15 % kecepatan dengan unduhan 3–4× lebih kecil.
`ProcessBatch` menggandakan throughput offline di 4 core.

Jalankan sendiri: `dotnet run -c Release --project benchmarks/MediaPipeNet.Benchmarks -- --filter "*"`, atau halaman
*Benchmark* di Gallery, atau `mediapipenet-cli benchmark faces gambar.jpg`.

## Mengapa cepat

- **Satu sesi per model**, dibuat sekali dengan optimasi graph penuh (`ORT_ENABLE_ALL`).
- **I/O pra-alokasi yang di-pool** — `OnnxModel.RentContext()` mengembalikan context yang tensor input/output-nya
  berupa `OrtValue` ter-pin di atas buffer `float[]` yang dipakai ulang; inferensi steady-state tidak mengalokasikan
  apa pun di managed heap. Pemanggil konkuren mendapat context terpisah, sehingga task mode image thread-safe tanpa
  lock.
- **Image-to-tensor sekali jalan** — crop, rotasi, resize, letterbox, dan normalisasi dalam satu loop unsafe,
  diparalelkan per baris untuk tensor besar, dengan supersampling n×n saat downscale besar (lebih mendekati
  MediaPipe dibanding bilinear biasa).
- **Hanya decode yang lolos** — decoding SSD melewati anchor di bawah ambang skor sebelum menyentuh data kotak.
- **Tracking** — di mode video detektor hanya berjalan bila perlu.
- **Frame di-pool** — `MPImage` menyewa piksel dari `ArrayPool`; `CopyFrom` mengisi ulang frame tanpa alokasi;
  `LiveStreamProcessor` memakai double-buffer.

## Tuning

| Pengaturan | Efek |
|---|---|
| `RunningMode.Video` / `LiveStream` | Tracking melewati detektor: keuntungan terbesar untuk video. |
| `HandLandmarkerOptions.NumHands = 1` | Di mode video detektor telapak tetap berjalan selama tangan yang dilacak kurang dari `NumHands`. |
| `PoseModel.Lite` vs `Full` | Lite ~2× lebih cepat. |
| `InferenceOptions.IntraOpThreads` | Thread per operator. Turunkan bila beberapa task berjalan paralel (mis. 2 per task). |
| `FaceLandmarkerOptions.OutputFaceBlendshapes = false` | Melewati model blendshape. |
| Ukuran input | Perkecil foto yang sangat besar; model hanya melihat 128–320 px. |
| `ProcessBatch(images)` / `ClassifyBatch` / `EmbedBatch` | Pemrosesan offline paralel dengan satu instance model. |
| `InferenceOptions.Precision` | `Float16` untuk GPU, `Int8` untuk unduhan kecil (lihat [Model](model.md#varian-presisi-fp16-dan-int8)). |
| `InferenceOptions.UseIoBinding` | Mengikat buffer tiap konteks pool sekali (I/O binding ONNX Runtime); berguna dengan provider GPU. |
| `ImageSegmenterOptions.OutputConfidenceMasks = false` | Hanya category mask bila hanya itu yang dibutuhkan. |

## GPU

| Provider | Paket | Platform |
|---|---|---|
| `Cpu` | `Gravicode.MediaPipeNet` | semua |
| `CoreML` | `Gravicode.MediaPipeNet` | macOS / Apple silicon |
| `DirectML` | `Gravicode.MediaPipeNet.DirectML` | Windows, GPU DirectX 12 apa pun |
| `Cuda` | `Gravicode.MediaPipeNet.Cuda` | Windows/Linux, NVIDIA + CUDA 12 + cuDNN 9 |
| `Nnapi` | `Microsoft.ML.OnnxRuntime` di aplikasi MAUI | Android 8.1+ (lihat [Platform](platform.md)) |

`ExecutionProvider.Auto` mencoba CUDA → DirectML → CoreML → NNAPI → CPU, melewati provider yang tidak ada di runtime yang
dimuat atau gagal diinisialisasi (`FallbackToCpu`). Provider yang benar-benar dipakai tersedia di
`OnnxModel.Provider` dan menjadi tag metrik `mediapipenet.inference.duration`.

### Angka DirectML (GPU terintegrasi)

`benchmarks/MediaPipeNet.GpuBenchmark` (median 30 kali proses, mode image, termasuk pra/pasca-pemrosesan, hasil setiap
konfigurasi dicek terhadap hasil CPU) pada satu-satunya GPU yang tersedia bagi proyek, **Intel UHD 620** (terintegrasi,
memori bersama) di samping i7-8650U, ONNX Runtime 1.24.4, dalam ms:

| Task | CPU fp32 | DirectML fp32 | DirectML fp16 | DirectML fp32 + IoBinding |
|---|---:|---:|---:|---:|
| FaceDetector | 8,3 | 9,0 | 8,5 | 8,6 |
| FaceLandmarker | 22,0 | 26,4 | 25,0 | 26,3 |
| HandLandmarker | 26,7 | 40,6 | 39,9 | 40,5 |
| PoseLandmarker (full) | 54,3 | 60,2 | 54,0 | 58,6 |
| ImageSegmenter (multiclass) | 103,5 | 105,9 | 89,6 | 101,7 |
| ObjectDetector | 45,5 | 67,3 | 61,7 | 67,3 |
| ImageClassifier | 7,4 | 18,8 | 17,5 | 18,3 |
| InteractiveSegmenter | 80,4 | 105,3 | 85,3 | 98,1 |
| FaceStylizer | 455,5 | 468,5 | 414,5 | 465,9 |

- Di GPU terintegrasi DirectML tidak mengalahkan CPU modern untuk model sekecil ini: setiap panggilan membayar salinan
  host↔device dan dispatch kernel, dan UHD 620 berbagi bandwidth memori dengan CPU. FP16 sedikit membantu (−5 sampai
  −15 %), paling terasa di model besar (segmentasi, stylizer, MagicTouch). I/O binding tidak menghemat apa pun di sini
  karena input tetap datang dari memori CPU setiap frame.
- GPU diskrit (dan CUDA) adalah tempat paket GPU memberi manfaat — terutama stylizer (≈ 1 GFLOP per panggilan),
  segmentasi, dan beban batch. Proyek tidak punya GPU diskrit untuk diukur; jalankan
  `dotnet run -c Release --project benchmarks/MediaPipeNet.GpuBenchmark` di GPU Anda (silakan kirim angkanya lewat issue).

Catatan dari pengujian:

- DirectML tidak mendukung panggilan `Run` konkuren maupun pembuatan sesi konkuren pada satu device; MediaPipe.NET
  menserialisasi keduanya otomatis saat provider DirectML aktif.
- Gallery memakai provider CPU secara default; ubah provider dan presisi di *Pengaturan*.
- `UseIoBinding` membuat satu `OrtIoBinding` per konteks pool dan memakainya ulang di setiap run; hasilnya identik
  (`BatchAndBindingTests`). ONNX Runtime menyalin input CPU yang di-bind ke device *saat di-bind*, sehingga pada provider
  GPU MediaPipe.NET mem-bind ulang input sebelum setiap run (diperbaiki di 1.0 — 0.3 mengembalikan hasil basi dengan
  DirectML + `UseIoBinding`, yang tertangkap oleh pengecekan hasil di benchmark GPU).
