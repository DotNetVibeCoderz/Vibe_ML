# Custom models

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/model-kustom.md)

MediaPipe.NET runs your own models as well as the catalog ones — typically models trained with
[MediaPipe Model Maker](https://ai.google.dev/edge/mediapipe/solutions/model_maker) (image classifier, object
detector, gesture recognizer, audio classifier, text classifier), which export TFLite files with the same
architecture as MediaPipe's stock models.

## 1. Convert the model

```bash
python -m venv C:\mpv
C:\mpv\Scripts\pip install tensorflow-cpu==2.17.1 tf2onnx==1.16.1 protobuf==3.20.3 "numpy<2" onnx==1.16.2 onnxruntime
C:\mpv\Scripts\python tools/model-conversion/convert_models.py --custom my_food.tflite --custom-id my_food --out my_models
```

`--custom` runs the same pipeline as the catalog models — densifying sparse weights, lowering MediaPipe's custom
TFLite ops (`Convolution2DTransposeBias`, `MaxPoolingWithArgmax2D`, `MaxUnpooling2D`), removing tf2onnx's coarse
dynamic-quantization emulation, retrying with a newer opset when needed — and writes:

| File | |
|---|---|
| `my_food.onnx` | the model |
| `my_food.labels.txt` | category names from the TFLite metadata, when present |
| `my_food.vocab.txt` | tokenizer vocabulary (text models) |

A `.task` bundle (for example a Model Maker gesture recognizer) is unpacked and every graph inside is converted as
`{id}_{graph}` — e.g. `my_gestures_custom_gesture_classifier.onnx`.

The converter prints the model's inputs and outputs; check them against the task you want to use.

## 2. Use it

Tasks that accept custom models take `ModelPath`; their labels come from the `Labels` option, else from the
`{model}.labels.txt` next to the model, else the built-in labels.

| Task | Options | Expected model |
|---|---|---|
| `ImageClassifier` | `ModelPath`, `Labels` | image classifier with a float input in [-1, 1] (EfficientNet-Lite, Model Maker) |
| `ObjectDetector` | `ModelPath`, `Labels` | EfficientDet-Lite of **any input size** (Lite0–Lite4); anchors are derived from the input |
| `GestureRecognizer` | `ClassifierModelPath`, `Labels` | a gesture classifier taking the 128-D gesture embedding (Model Maker's `custom_gesture_classifier`) |
| `AudioClassifier` | `ModelPath`, `Labels` | 15 600 samples of 16 kHz audio in, scores out (YAMNet-based Model Maker models) |
| `TextClassifier` | `ModelPath`, `Labels`, `VocabularyPath` | a BERT or average-word classifier (set `Model` accordingly); the vocabulary defaults to `{model}.vocab.txt` |

```csharp
using var classifier = ImageClassifier.Create(new()
{
    ModelPath = "my_models/my_food.onnx",        // labels from my_models/my_food.labels.txt
    MaxResults = 3,
});

using var detector = ObjectDetector.Create(new()
{
    ModelPath = "my_models/android_figurines.onnx",
    Labels = ["background", "android", "cupcake"],   // explicit labels win over the file
});

using var gestures = GestureRecognizer.Create(new()
{
    ClassifierModelPath = "my_models/my_gestures_custom_gesture_classifier.onnx",
});
```

Replacing a catalog model file wholesale is also possible with `BaseOptions.ModelPaths`
(`{ ["palm_detection"] = "my_palm.onnx" }`) — the file must keep the original's inputs and outputs.

## Tips

- Custom models are loaded as they are (no checksum); keep them next to your application or in a folder you
  control.
- Precision variants (`InferenceOptions.Precision`) apply to catalog models only; to shrink a custom model, run
  `tools/model-conversion/quantize_models.py` on it (edit `CANDIDATES`) or use ONNX Runtime's tooling.
- Validate a converted model against its TFLite original with `tools/model-conversion/validate_models.py`.
