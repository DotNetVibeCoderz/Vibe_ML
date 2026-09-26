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

`ModelCatalog.All` berisi semuanya lengkap dengan SHA-256, ukuran, paket, dan URL sumber; `descriptor.Attribution`
memberikan teks lisensi yang wajib dicantumkan. Bobot model © Google LLC, Apache-2.0.

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
   - **Custom op `Convolution2DTransposeBias`** (selfie segmenter) ditulis ulang menjadi `ConvTranspose` ONNX dengan
     transpose NHWC↔NCHW.
   - Delegate XNNPACK interpreter TFLite dimatikan selama konversi (crash pada beberapa graph di Windows).
4. Menulis `manifest.json` (input, output, SHA-256).

`tools/model-conversion/validate_models.py` lalu memberi input acak yang sama ke graph TFLite **asli** dan file
ONNX: selisih maks ≈ 1e-4 untuk setiap model (`validation.json`). Label map di metadata TFLite (COCO, ImageNet,
gestur, handedness) ikut diekstrak.

```bash
python -m venv C:\mpv
C:\mpv\Scripts\pip install tensorflow-cpu==2.17.1 tf2onnx==1.16.1 protobuf==3.20.3 "numpy<2" onnx==1.16.2 onnxruntime
C:\mpv\Scripts\python tools/model-conversion/convert_models.py --out artifacts/models
C:\mpv\Scripts\python tools/model-conversion/validate_models.py --models artifacts/models
```

## Memakai model ONNX sendiri

- Jalankan model gambar apa pun di dalam graph dengan `InferenceNode(modelPath, rangeMin, rangeMax, keepAspectRatio)`.
- Pakai blok penyusunnya langsung: `OnnxModel.Load`, `ImageToTensor.Convert` (ROI berotasi, letterbox, normalisasi),
  `SsdAnchors` + `DetectionDecoder` + `NonMaxSuppression` untuk detektor gaya SSD, `TensorWarp` untuk mask.
- Ganti model bawaan dengan versi fine-tuned bersignature sama melalui `BaseOptions.ModelPaths` (checksum tidak
  diwajibkan untuk path eksplisit).
