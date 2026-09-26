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

`ModelCatalog.All` lists them with SHA-256, size, package and source URL; `descriptor.Attribution` gives the
required license text. Model weights are © Google LLC, Apache-2.0.

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
   - **Custom op `Convolution2DTransposeBias`** (selfie segmenter) is rewritten as ONNX `ConvTranspose` with
     NHWC↔NCHW transposes.
   - The TFLite interpreter's XNNPACK delegate is disabled during conversion (it crashes on some graphs on Windows).
4. Writes `manifest.json` (inputs, outputs, SHA-256).

`tools/model-conversion/validate_models.py` then feeds identical random inputs to the **original** TFLite graph and
the ONNX file: max |Δ| ≈ 1e-4 for every model (`validation.json`). The label maps embedded in TFLite metadata
(COCO, ImageNet, gestures, handedness) are extracted alongside.

```bash
python -m venv C:\mpv
C:\mpv\Scripts\pip install tensorflow-cpu==2.17.1 tf2onnx==1.16.1 protobuf==3.20.3 "numpy<2" onnx==1.16.2 onnxruntime
C:\mpv\Scripts\python tools/model-conversion/convert_models.py --out artifacts/models
C:\mpv\Scripts\python tools/model-conversion/validate_models.py --models artifacts/models
```

## Using your own ONNX models

- Run any image model inside a graph with `InferenceNode(modelPath, rangeMin, rangeMax, keepAspectRatio)`.
- Use the building blocks directly: `OnnxModel.Load`, `ImageToTensor.Convert` (rotated ROI, letterbox, normalize),
  `SsdAnchors` + `DetectionDecoder` + `NonMaxSuppression` for SSD-style detectors, `TensorWarp` for masks.
- Replace a built-in model with a fine-tuned one of the same signature through `BaseOptions.ModelPaths`
  (checksums are not enforced for explicit paths).
