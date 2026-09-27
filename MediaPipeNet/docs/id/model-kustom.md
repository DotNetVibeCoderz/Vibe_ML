# Model kustom

> 🇬🇧 [Read in English](../en/custom-models.md)

MediaPipe.NET menjalankan model milik Anda sendiri selain model katalog — umumnya model yang dilatih dengan
[MediaPipe Model Maker](https://ai.google.dev/edge/mediapipe/solutions/model_maker) (image classifier, object
detector, gesture recognizer, audio classifier, text classifier), yang mengekspor file TFLite dengan arsitektur
sama seperti model bawaan MediaPipe.

## 1. Konversi model

```bash
python -m venv C:\mpv
C:\mpv\Scripts\pip install tensorflow-cpu==2.17.1 tf2onnx==1.16.1 protobuf==3.20.3 "numpy<2" onnx==1.16.2 onnxruntime
C:\mpv\Scripts\python tools/model-conversion/convert_models.py --custom makanan_saya.tflite --custom-id makanan_saya --out model_saya
```

`--custom` menjalankan pipeline yang sama dengan model katalog — memadatkan bobot sparse, menurunkan custom op
TFLite MediaPipe (`Convolution2DTransposeBias`, `MaxPoolingWithArgmax2D`, `MaxUnpooling2D`), menghapus emulasi
kuantisasi dinamis tf2onnx yang terlalu kasar, mencoba ulang dengan opset lebih baru bila perlu — lalu menulis:

| File | |
|---|---|
| `makanan_saya.onnx` | modelnya |
| `makanan_saya.labels.txt` | nama kategori dari metadata TFLite, bila ada |
| `makanan_saya.vocab.txt` | kosakata tokenizer (model teks) |

Bundel `.task` (misalnya gesture recognizer dari Model Maker) dibongkar dan setiap graph di dalamnya dikonversi
sebagai `{id}_{graph}` — misalnya `gesture_saya_custom_gesture_classifier.onnx`.

Konverter mencetak input dan output model; cocokkan dengan task yang ingin dipakai.

## 2. Pakai modelnya

Task yang mendukung model kustom menerima `ModelPath`; labelnya diambil dari opsi `Labels`, jika tidak ada dari
`{model}.labels.txt` di sebelah model, jika tidak ada juga dari label bawaan.

| Task | Opsi | Model yang diharapkan |
|---|---|---|
| `ImageClassifier` | `ModelPath`, `Labels` | classifier gambar dengan input float di [-1, 1] (EfficientNet-Lite, Model Maker) |
| `ObjectDetector` | `ModelPath`, `Labels` | EfficientDet-Lite dengan **ukuran input berapa pun** (Lite0–Lite4); anchor diturunkan dari inputnya |
| `GestureRecognizer` | `ClassifierModelPath`, `Labels` | classifier gesture yang menerima embedding gesture 128-D (`custom_gesture_classifier` dari Model Maker) |
| `AudioClassifier` | `ModelPath`, `Labels` | input 15.600 sampel audio 16 kHz, output skor (model Model Maker berbasis YAMNet) |
| `TextClassifier` | `ModelPath`, `Labels`, `VocabularyPath` | classifier BERT atau average-word (atur `Model` sesuai); kosakata default dari `{model}.vocab.txt` |

```csharp
using var classifier = ImageClassifier.Create(new()
{
    ModelPath = "model_saya/makanan_saya.onnx",     // label dari model_saya/makanan_saya.labels.txt
    MaxResults = 3,
});

using var detector = ObjectDetector.Create(new()
{
    ModelPath = "model_saya/figurin_android.onnx",
    Labels = ["background", "android", "cupcake"],  // label eksplisit mengalahkan file
});

using var gestures = GestureRecognizer.Create(new()
{
    ClassifierModelPath = "model_saya/gesture_saya_custom_gesture_classifier.onnx",
});
```

Mengganti file model katalog secara utuh juga bisa lewat `BaseOptions.ModelPaths`
(`{ ["palm_detection"] = "palm_saya.onnx" }`) — file tersebut harus mempertahankan input dan output aslinya.

## Tips

- Model kustom dimuat apa adanya (tanpa checksum); simpan di samping aplikasi atau di folder yang Anda kendalikan.
- Varian presisi (`InferenceOptions.Precision`) hanya berlaku untuk model katalog; untuk memperkecil model kustom,
  jalankan `tools/model-conversion/quantize_models.py` padanya (ubah `CANDIDATES`) atau gunakan tool ONNX Runtime.
- Validasi model hasil konversi terhadap TFLite aslinya dengan `tools/model-conversion/validate_models.py`.
