# MediaPipe.Net Gallery

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/gallery.md)

A cross-platform Avalonia desktop app (Windows, Linux, macOS) that showcases every task: try it on the bundled
samples or your own images and webcam, tune the options, read the results, and copy the C# that produced them.

```bash
dotnet run --project samples/MediaPipeNet.Gallery -c Release
```

The app bundles every model (`Gravicode.MediaPipeNet.Models.All` and `.Models.Quantized`), so it runs offline. On Windows it references the DirectML
build of ONNX Runtime; the default provider is CPU and can be changed in *Settings*.

## Pages

| Page | What it shows |
|---|---|
| Overview | A live holistic result on the stage and a card per task. |
| Face detection · Object detection | Boxes, keypoints and labels; short- or full-range face model. |
| Face mesh · Hand landmarks · Gesture recognition · Pose landmarks · Holistic | Landmark overlays with the rotated ROI (dashed) the model ran on; head pose (yaw/pitch/roll, distance) from the facial transformation matrix. |
| Image segmentation · Interactive segmentation · Image embedding · Image classification | Selfie / multiclass / hair / DeepLab masks with blur, backdrop or colored categories; MagicTouch cut-out at a movable point; similarity to the sample images; top-K labels. |
| Audio classification | Waveform with detected speech segments, YAMNet events per window, the bundled clip or synthetic tones/noise, or any WAV. |
| Text understanding | Sentiment (MobileBERT and average word), language (110), and sentence similarity for any text. |
| Live camera | Any task on your webcam in video mode: FPS, latency, dropped frames. |
| Graph API | A schematic of a custom graph (parallel face + hand nodes, a join and a custom pixelation node), its output, and an editable `.pbtxt` config you can run. |
| Benchmark | Per-task latency on your machine at 640×480, with the NFR-1 target line. |
| Models | Catalog status, SHA-256 verification, download of missing models. |
| Settings | Execution provider, CPU threads, model precision (FP32 / FP16 / INT8), language (English / Bahasa Indonesia), theme (light / dark), extra model folder, landmark points. |
| About | Credits and licenses. |

Every task page has three tabs: **Preview** (the stage with overlays and an instrument readout: latency ·
provider · image size · summary), **C# code** (a snippet that follows the current option values, with a copy
button) and **JSON** (the serialized result).

## Screenshots

| | |
|---|---|
| ![Overview](../images/gallery-home.png) | ![Face detection](../images/gallery-faces.png) |
| ![Face mesh](../images/gallery-face-mesh.png) | ![Hands](../images/gallery-hands.png) |
| ![Gestures](../images/gallery-gestures.png) | ![Pose](../images/gallery-pose.png) |
| ![Holistic](../images/gallery-holistic.png) | ![Segmentation](../images/gallery-segment.png) |
| ![Interactive segmentation](../images/gallery-interactive.png) | ![Image embedding](../images/gallery-embed.png) |
| ![Audio classification](../images/gallery-audio.png) | ![Text understanding](../images/gallery-text.png) |
| ![Objects](../images/gallery-objects.png) | ![Classification](../images/gallery-classify.png) |
| ![Graph API](../images/gallery-graph.png) | ![Benchmark](../images/gallery-benchmark.png) |
| ![Models](../images/gallery-models.png) | ![Settings](../images/gallery-settings.png) |
| ![Live camera](../images/gallery-live.png) | ![About](../images/gallery-about.png) |
| ![Overview, dark, Indonesian](../images/gallery-home-dark-id.png) | ![Settings, dark, Indonesian](../images/gallery-settings-dark-id.png) |

## Design

"Optics lab": light chrome around a dark viewfinder stage framed by registration marks, with a monospace readout
strip. Cobalt is the only interface accent; coral and mint are reserved for what the models see. Type: *Unbounded*
for page titles, *Inter* for text, *JetBrains Mono* for data and code (fonts under the SIL Open Font License,
bundled in `Assets/Fonts`).

## Regenerating the screenshots

```bash
dotnet run --project samples/MediaPipeNet.Gallery -- --screenshots docs/images
```

The app visits every page, waits for its results, renders the window to PNG, repeats three pages in the dark theme
and Indonesian, and exits. `--lang id` and `--theme Dark` start the interactive app in those modes without
changing saved settings.

## Code map

| Folder | |
|---|---|
| `Services/TaskCatalog.cs` | The vision task definitions: options, overlay rendering, result rows, code snippet. |
| `Views/AudioTextViews.cs` | The audio (waveform, events, VAD) and text pages. |
| `Controls/StageView.cs` | The viewfinder control (image fit, vector overlay, registration marks, readout). |
| `Controls/GraphDiagram.cs` | Layered schematic of any `CalculatorGraph` (uses `GetEdges()`). |
| `Views/*` | Pages. `MainWindow.cs` hosts navigation and the screenshot mode. |
| `Services/Loc.cs` | English / Indonesian strings. |
| `Theme/Tokens.axaml`, `Theme/Styles.axaml` | Design tokens (light/dark) and styles. |
