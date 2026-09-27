# Model

> 🇬🇧 [Read in English](../en/models.md)

## Katalog

| Id | Ukuran | Paket | Dikonversi dari |
|---|---:|---|---|
| `face_detection_short_range` | 0,4 MB | Models.Face | `blaze_face_short_range.tflite` |
| `face_landmarks_detector` | 4,9 MB | Models.Face | `face_landmarker.task` |
| `face_blendshapes` | 1,9 MB | Models.Face | `face_landmarker.task` |
| `palm_detection` | 4,6 MB | Models.Hand | `hand_landmarker.task` |
| `hand_landmarks_detector` | 10,9 MB | Models.Hand | `hand_landmarker.task` |
| `gesture_embedder` | 0,5 MB | Models.Hand | `gesture_recognizer.task` |
| `canned_gesture_classifier` | 6 KB | Models.Hand | `gesture_recognizer.task` |
| `pose_detection` | 11,9 MB | Models.Pose | `pose_detection.tflite` (bobot sparse dipadatkan) |
| `pose_landmarks_detector_lite` | 5,5 MB | Models.Pose | `pose_landmarker_lite.task` |
| `pose_landmarks_detector_full` | 12,8 MB | Models.Pose | `pose_landmarker_full.task` |
| `selfie_segmenter` | 0,5 MB | Models.Segmentation | `selfie_segmenter.tflite` |
| `efficientdet_lite0` | 13,5 MB | Models.ObjectDetection | `efficientdet_lite0.tflite` (float32) |
| `efficientnet_lite0` | 18,6 MB | Models.ImageClassification | `efficientnet_lite0.tflite` (float32) |
| `face_detection_full_range` | 2,1 MB | Models.Face | `blaze_face_full_range.tflite` |
| `hand_roi_refinement` | 0,1 MB | Models.Hand | `holistic_landmarker.task` |
| `selfie_multiclass` | 16,5 MB | Models.Segmentation | `selfie_multiclass_256x256.tflite` |
| `hair_segmenter` | 0,8 MB | Models.Segmentation | `hair_segmenter.tflite` (custom op diturunkan) |
| `deeplab_v3` | 2,8 MB | Models.Segmentation | `deeplab_v3.tflite` |
| `magic_touch` | 12,4 MB | Models.Segmentation | `magic_touch.tflite` |
| `mobilenet_v3_small_embedder` | 4,2 MB | Models.ImageEmbedding | `mobilenet_v3_small.tflite` |
| `yamnet` | 5,1 MB | Models.Audio | `yamnet.tflite` (bobot int8) |
| `bert_classifier` | 25,5 MB | Models.Text | `bert_classifier.tflite` (bobot int8) |
| `average_word_classifier` | 0,6 MB | Models.Text | `average_word_classifier.tflite` |
| `bert_embedder` | 26,6 MB | Models.Text | `bert_embedder.tflite` (bobot int8) |
| `language_detector` | 3,8 MB | Models.Text | `language_detector.tflite` (dibangun ulang, lihat di bawah) |

`ModelCatalog.All` berisi semuanya lengkap dengan SHA-256, ukuran, paket, dan URL sumber; `descriptor.Attribution`
memberikan teks lisensi yang wajib dicantumkan. Bobot model © Google LLC, Apache-2.0.

## Varian presisi: FP16 dan INT8

`Gravicode.MediaPipeNet.Models.Quantized` menambahkan salinan berpresisi rendah, dipilih lewat
`InferenceOptions.Precision`:

```csharp
var options = new BaseOptions { Inference = new InferenceOptions { Precision = ModelPrecision.Float16 } };
using var landmarker = FaceLandmarker.Create(new() { BaseOptions = options });
```

| Presisi | Yang berubah | Ukuran | Cocok untuk | Model |
|---|---|---|---|---|
| `Float32` (default) | model referensi | 100 % | semua | ke-25 model |
| `Float16` | bobot dan aktivasi float16 (input/output tetap float32) | ~50 % | GPU (DirectML, CUDA) | 14 |
| `Int8` | bobot int8 per kanal saja, dihitung dalam float | ~27–32 % | unduhan / aplikasi lebih kecil | 9 |

`tools/model-conversion/quantize_models.py` membuat varian dan **hanya menyimpan yang tetap setia**: setiap varian
dijalankan berdampingan dengan model float32-nya dan harus mencapai cosine similarity ≥ 0,999 (FP16) atau ≥ 0,99
(INT8) di setiap output. Ditolak: INT8 untuk model tangan, landmark pose, EfficientNet, dan MagicTouch (akurasinya
turun), serta kedua varian untuk model teks dan YAMNet (sudah menyimpan bobot int8). Kuantisasi aktivasi dinamis juga
dievaluasi dan merusak CNN landmark, sehingga tidak dipakai. `PrecisionTests` menguji varian end-to-end (face mesh
dalam selisih 0,002 / 0,004, objek yang sama terdeteksi, < 0,5 % piksel segmentasi berubah).

Model tanpa varian yang diminta memakai float32; demikian pula varian yang tidak ditemukan (dengan peringatan).
`ModelCatalog.GetVariant(model, precision)` dan `ModelLoader.SelectVariant(options, model)` menunjukkan file mana
yang dipakai. Di CPU, FP16 biasanya *lebih lambat* (sedikit kernel float16 native) — pakai di GPU; bobot INT8
berjalan secepat float.

## Cara task menemukan model

1. `BaseOptions.ModelPaths[id]` — file eksplisit.
2. `BaseOptions.ModelDirectory` — folder Anda.
3. `ModelStore` (`BaseOptions.ModelStore` atau `ModelStore.Default`), yang bertanya ke provider secara berurutan:
   1. direktori pada variabel lingkungan `MEDIAPIPENET_MODELS`;
   2. `<app>/models` — diisi oleh paket `Gravicode.MediaPipeNet.Models.*` (`BundledModelProvider`);
   3. cache pengguna `%LOCALAPPDATA%/MediaPipeNet/models`;
   4. **nuget.org** — paket `Gravicode.MediaPipeNet.Models.*` yang sesuai diunduh dan modelnya diekstrak ke cache
      (`NuGetModelProvider`).

Setiap file yang ditemukan dicek terhadap SHA-256 di katalog (sekali per proses). Unduhan yang rusak dihapus dan
provider berikutnya dicoba; jika semua gagal, `ModelNotFoundException` menjelaskan apa saja yang sudah dicoba.

```csharp
// Hanya offline, dengan folder sendiri lebih dulu:
var store = ModelStore.CreateDefault(modelDirectory: "D:/models", allowDownload: false);
var options = new BaseOptions { ModelStore = store };

// Unduh semuanya saat startup (mis. saat build image container):
await ModelStore.Default.EnsureModelsAsync(ModelCatalog.All, new Progress<ModelDownloadProgress>(p =>
    Console.WriteLine($"{p.Model.Id}: {p.Stage} {p.Fraction:P0}")));
```

Provider lain: `HttpModelProvider(baseUrl, cache)` (mirror sendiri), `EmbeddedResourceModelProvider(assembly, cache)`
(model di-embed ke assembly Anda), atau implementasikan `IModelProvider`.

CLI melakukan hal yang sama: `mediapipenet-cli models list | download | verify`.

## Cara model dikonversi

`tools/model-conversion/convert_models.py` mereproduksi setiap file:

1. Mengunduh model TFLite resmi dan bundel `.task` (file zip) dari `storage.googleapis.com/mediapipe-models`.
2. Mengonversi setiap graph dengan **tf2onnx** (opset 13; 14 untuk HardSwish).
3. Menangani kekhususan MediaPipe:
   - **Bobot sparse** — detektor BlazePose menyimpan bobot sebagai tensor sparse yang diekspansi oleh op `DENSIFY`.
     Skrip mengevaluasinya dengan interpreter TFLite lalu menulis konstanta padat.
   - **Custom op** — `Convolution2DTransposeBias` (selfie dan hair segmenter) menjadi `ConvTranspose` ONNX;
     `MaxPoolingWithArgmax2D` / `MaxUnpooling2D` (hair segmenter) menjadi `MaxPool` dengan indeks / `MaxUnpool` —
     masing-masing dengan transpose NHWC↔NCHW, parameternya dibaca dari custom options TFLite.
   - **Layer hibrida int8** — untuk op `FULLY_CONNECTED` MobileBERT dengan `asymmetric_quantize_inputs`, tf2onnx
     menyisipkan pasangan `DynamicQuantizeLinear` pada input, bobot, *dan bias* (uint8, per tensor) yang jauh lebih
     kasar daripada TFLite dan menurunkan cosine similarity hingga 7 %; skrip menghapusnya (layer berjalan dalam
     float, penyimpanan bobot int8 tetap).
   - **Language detector** — op string `NGramHash`-nya tidak punya padanan ONNX, sehingga model dibangun ulang:
     hashing berjalan di C# dan `KmeansEmbeddingLookup` (embedding terkuantisasi produk) diekspansi menjadi tabel
     padat dengan `Gather` + `ReduceMean`.
   - Delegate XNNPACK interpreter TFLite dimatikan selama konversi (crash pada beberapa graph di Windows).
4. Menulis `manifest.json` (input, output, SHA-256) dan `failed.json` (graph yang tidak bisa dikonversi — hanya
   face stylizer, lihat [Platform](platform.md#face-stylizer)).

### Bobot int8 dan ONNX Runtime

Optimizer graph ONNX Runtime memfusikan `DequantizeLinear(bobot int8) → MatMul` menjadi kernel yang juga
mengkuantisasi *aktivasi* secara dinamis, sehingga embedding MobileBERT berubah beberapa persen. Karena itu
MediaPipe.NET menyetel opsi sesi `session.disable_quant_qdq`: bobot int8 didekuantisasi sekali dan jaringan
berjalan dalam float — dengan itu MobileBERT ONNX identik dengan model TFLite (yang didekuantisasi) hingga cosine
similarity 1,0000.

`tools/model-conversion/validate_models.py` lalu memberi input acak yang sama ke graph TFLite **asli** dan file
ONNX: selisih maks ≈ 1e-4 untuk model float (`validation.json`); model int8 (YAMNet, MobileBERT) berbeda sebesar
pembulatan kernel int8 TFLite, dan model yang graph TFLite-nya butuh custom op MediaPipe divalidasi end-to-end
terhadap MediaPipe. Label map dan kosakata tokenizer di metadata TFLite ikut diekstrak (`{id}.labels.txt`,
`{id}.vocab.txt`).

```bash
python -m venv C:\mpv
C:\mpv\Scripts\pip install tensorflow-cpu==2.17.1 tf2onnx==1.16.1 protobuf==3.20.3 "numpy<2" onnx==1.16.2 onnxruntime
C:\mpv\Scripts\python tools/model-conversion/convert_models.py --out artifacts/models
C:\mpv\Scripts\python tools/model-conversion/validate_models.py --models artifacts/models
C:\mpv\Scripts\python tools/model-conversion/quantize_models.py               # varian FP16 / INT8
python tools/model-conversion/generate_quantized_catalog.py                     # ModelCatalog.Quantized.g.cs
python tools/model-conversion/extract_face_geometry.py                          # wajah kanonis untuk matriks transformasi
```

## Memakai model ONNX sendiri

Untuk model MediaPipe Model Maker (classifier, detektor, gestur, audio, teks) lihat [Model kustom](model-kustom.md):
konversi dengan `convert_models.py --custom` lalu isi `ModelPath`.

- Jalankan model gambar apa pun di dalam graph dengan `InferenceNode(modelPath, rangeMin, rangeMax, keepAspectRatio)`.
- Pakai blok penyusunnya langsung: `OnnxModel.Load`, `ImageToTensor.Convert` (ROI berotasi, letterbox, normalisasi),
  `SsdAnchors` + `DetectionDecoder` + `NonMaxSuppression` untuk detektor gaya SSD, `TensorWarp` untuk mask.
- Ganti model bawaan dengan versi fine-tuned bersignature sama melalui `BaseOptions.ModelPaths` (checksum tidak
  diwajibkan untuk path eksplisit).
