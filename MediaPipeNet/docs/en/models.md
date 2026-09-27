# Models

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/model.md)

## The catalog

| Id | Size | Package | Converted from |
|---|---:|---|---|
| `face_detection_short_range` | 0.4 MB | Models.Face | `blaze_face_short_range.tflite` |
| `face_landmarks_detector` | 4.9 MB | Models.Face | `face_landmarker.task` |
| `face_blendshapes` | 1.9 MB | Models.Face | `face_landmarker.task` |
| `palm_detection` | 4.6 MB | Models.Hand | `hand_landmarker.task` |
| `hand_landmarks_detector` | 10.9 MB | Models.Hand | `hand_landmarker.task` |
| `gesture_embedder` | 0.5 MB | Models.Hand | `gesture_recognizer.task` |
| `canned_gesture_classifier` | 6 KB | Models.Hand | `gesture_recognizer.task` |
| `pose_detection` | 11.9 MB | Models.Pose | `pose_detection.tflite` (sparse weights densified) |
| `pose_landmarks_detector_lite` | 5.5 MB | Models.Pose | `pose_landmarker_lite.task` |
| `pose_landmarks_detector_full` | 12.8 MB | Models.Pose | `pose_landmarker_full.task` |
| `selfie_segmenter` | 0.5 MB | Models.Segmentation | `selfie_segmenter.tflite` |
| `efficientdet_lite0` | 13.5 MB | Models.ObjectDetection | `efficientdet_lite0.tflite` (float32) |
| `efficientnet_lite0` | 18.6 MB | Models.ImageClassification | `efficientnet_lite0.tflite` (float32) |
| `face_detection_full_range` | 2.1 MB | Models.Face | `blaze_face_full_range.tflite` |
| `hand_roi_refinement` | 0.1 MB | Models.Hand | `holistic_landmarker.task` |
| `selfie_multiclass` | 16.5 MB | Models.Segmentation | `selfie_multiclass_256x256.tflite` |
| `hair_segmenter` | 0.8 MB | Models.Segmentation | `hair_segmenter.tflite` (custom ops lowered) |
| `deeplab_v3` | 2.8 MB | Models.Segmentation | `deeplab_v3.tflite` |
| `magic_touch` | 12.4 MB | Models.Segmentation | `magic_touch.tflite` |
| `mobilenet_v3_small_embedder` | 4.2 MB | Models.ImageEmbedding | `mobilenet_v3_small.tflite` |
| `yamnet` | 5.1 MB | Models.Audio | `yamnet.tflite` (int8 weights) |
| `bert_classifier` | 25.5 MB | Models.Text | `bert_classifier.tflite` (int8 weights) |
| `average_word_classifier` | 0.6 MB | Models.Text | `average_word_classifier.tflite` |
| `bert_embedder` | 26.6 MB | Models.Text | `bert_embedder.tflite` (int8 weights) |
| `language_detector` | 3.8 MB | Models.Text | `language_detector.tflite` (rebuilt, see below) |

`ModelCatalog.All` lists them with SHA-256, size, package and source URL; `descriptor.Attribution` gives the
required license text. Model weights are © Google LLC, Apache-2.0.

## Precision variants: FP16 and INT8

`Gravicode.MediaPipeNet.Models.Quantized` adds reduced-precision copies, selected with `InferenceOptions.Precision`:

```csharp
var options = new BaseOptions { Inference = new InferenceOptions { Precision = ModelPrecision.Float16 } };
using var landmarker = FaceLandmarker.Create(new() { BaseOptions = options });
```

| Precision | What changes | Size | Best for | Models |
|---|---|---|---|---|
| `Float32` (default) | the reference models | 100 % | everything | all 25 |
| `Float16` | weights and activations in float16 (inputs/outputs stay float32) | ~50 % | GPUs (DirectML, CUDA) | 14 |
| `Int8` | weight-only int8, per channel, computed in float | ~27–32 % | smaller downloads / apps | 9 |

`tools/model-conversion/quantize_models.py` creates the variants and **keeps only those that stay faithful**: every
variant runs next to its float32 model and must reach a cosine similarity of ≥ 0.999 (FP16) or ≥ 0.99 (INT8) on
every output. Rejected: INT8 for the hand, pose-landmark, EfficientNet and MagicTouch models (their accuracy drops),
and both variants for the text models and YAMNet (they already store int8 weights). Dynamic activation
quantization was also evaluated and destroys the landmark CNNs, so it is not used. `PrecisionTests` checks the
variants end to end (face mesh within 0.002 / 0.004, same objects detected, < 0.5 % of segmentation pixels change).

A model without the requested variant uses float32; so does a variant that cannot be found (with a warning).
`ModelCatalog.GetVariant(model, precision)` and `ModelLoader.SelectVariant(options, model)` tell which file is used.
On a CPU, FP16 is usually *slower* (few native float16 kernels) — use it on GPUs; INT8 weights run at float speed.

## How a task finds its model

1. `BaseOptions.ModelPaths[id]` — an explicit file.
2. `BaseOptions.ModelDirectory` — your folder.
3. The `ModelStore` (`BaseOptions.ModelStore` or `ModelStore.Default`), which asks its providers in order:
   1. the directory in the `MEDIAPIPENET_MODELS` environment variable;
   2. `<app>/models` — filled by the `Gravicode.MediaPipeNet.Models.*` packages (`BundledModelProvider`);
   3. the user cache `%LOCALAPPDATA%/MediaPipeNet/models`;
   4. **nuget.org** — the matching `Gravicode.MediaPipeNet.Models.*` package is downloaded and its models extracted into the
      cache (`NuGetModelProvider`).

Every resolved file is checked against the catalog's SHA-256 (once per process). A corrupt download is deleted and
the next provider is tried; if nothing works, `ModelNotFoundException` explains what was tried.

```csharp
// Offline only, with your own folder first:
var store = ModelStore.CreateDefault(modelDirectory: "D:/models", allowDownload: false);
var options = new BaseOptions { ModelStore = store };

// Pre-fetch everything at startup (e.g. in a container image build step):
await ModelStore.Default.EnsureModelsAsync(ModelCatalog.All, new Progress<ModelDownloadProgress>(p =>
    Console.WriteLine($"{p.Model.Id}: {p.Stage} {p.Fraction:P0}")));
```

Other providers: `HttpModelProvider(baseUrl, cache)` (your own mirror), `EmbeddedResourceModelProvider(assembly,
cache)` (models embedded in your assembly), or implement `IModelProvider`.

The CLI does the same: `mediapipenet-cli models list | download | verify`.

## How the models were converted

`tools/model-conversion/convert_models.py` reproduces every file:

1. Downloads the official TFLite models and `.task` bundles (zip files) from `storage.googleapis.com/mediapipe-models`.
2. Converts each graph with **tf2onnx** (opset 13; 14 for HardSwish).
3. Handles MediaPipe specifics:
   - **Sparse weights** — BlazePose's detector stores weights as sparse tensors expanded by `DENSIFY` ops. The
     script evaluates them with the TFLite interpreter and writes dense constants.
   - **Custom ops** — `Convolution2DTransposeBias` (selfie and hair segmenters) becomes ONNX `ConvTranspose`;
     `MaxPoolingWithArgmax2D` / `MaxUnpooling2D` (hair segmenter) become `MaxPool` with indices / `MaxUnpool` —
     each with NHWC↔NCHW transposes, their parameters read from the TFLite custom options.
   - **Hybrid int8 layers** — for MobileBERT's `FULLY_CONNECTED` ops with `asymmetric_quantize_inputs`, tf2onnx
     inserts `DynamicQuantizeLinear` pairs on the input, the weights *and the bias* (uint8, per tensor), which is far
     coarser than TFLite and cost up to 7 % cosine similarity; the script removes them (the layer runs in float,
     the int8 weight storage stays).
   - **Language detector** — its `NGramHash` string op has no ONNX equivalent, so the model is rebuilt by hand:
     the hashing runs in C# and `KmeansEmbeddingLookup` (product-quantized embeddings) is expanded into dense
     tables with `Gather` + `ReduceMean`.
   - The TFLite interpreter's XNNPACK delegate is disabled during conversion (it crashes on some graphs on Windows).
4. Writes `manifest.json` (inputs, outputs, SHA-256) and `failed.json` (graphs that cannot be converted — only the
   face stylizer, see [Platforms](platforms.md#face-stylizer)).

### Int8 weights and ONNX Runtime

ONNX Runtime's graph optimizer fuses `DequantizeLinear(int8 weights) → MatMul` into a kernel that also quantizes
the *activations* on the fly, which changed MobileBERT's embeddings by several percent. MediaPipe.NET therefore sets
the session option `session.disable_quant_qdq`: int8 weights are dequantized once and the network runs in float —
with it, the ONNX MobileBERT matches the (dequantized) TFLite model to a cosine similarity of 1.0000.

`tools/model-conversion/validate_models.py` then feeds identical random inputs to the **original** TFLite graph and
the ONNX file: max |Δ| ≈ 1e-4 for the float models (`validation.json`); the int8 models (YAMNet, MobileBERT) differ
by the TFLite int8 kernels' rounding, and the models whose TFLite graph needs MediaPipe's custom ops are validated
end to end against MediaPipe instead. The label maps and tokenizer vocabularies embedded in TFLite metadata are
extracted alongside (`{id}.labels.txt`, `{id}.vocab.txt`).

```bash
python -m venv C:\mpv
C:\mpv\Scripts\pip install tensorflow-cpu==2.17.1 tf2onnx==1.16.1 protobuf==3.20.3 "numpy<2" onnx==1.16.2 onnxruntime
C:\mpv\Scripts\python tools/model-conversion/convert_models.py --out artifacts/models
C:\mpv\Scripts\python tools/model-conversion/validate_models.py --models artifacts/models
C:\mpv\Scripts\python tools/model-conversion/quantize_models.py               # FP16 / INT8 variants
python tools/model-conversion/generate_quantized_catalog.py                     # ModelCatalog.Quantized.g.cs
python tools/model-conversion/extract_face_geometry.py                          # canonical face for the transform matrix
```

## Using your own ONNX models

For MediaPipe Model Maker models (classifiers, detectors, gestures, audio, text) see
[Custom models](custom-models.md): convert with `convert_models.py --custom` and pass `ModelPath`.

- Run any image model inside a graph with `InferenceNode(modelPath, rangeMin, rangeMax, keepAspectRatio)`.
- Use the building blocks directly: `OnnxModel.Load`, `ImageToTensor.Convert` (rotated ROI, letterbox, normalize),
  `SsdAnchors` + `DetectionDecoder` + `NonMaxSuppression` for SSD-style detectors, `TensorWarp` for masks.
- Replace a built-in model with a fine-tuned one of the same signature through `BaseOptions.ModelPaths`
  (checksums are not enforced for explicit paths).
