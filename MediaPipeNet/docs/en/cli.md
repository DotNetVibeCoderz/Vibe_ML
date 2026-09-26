# CLI — `mediapipenet-cli`

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/cli.md)

```bash
dotnet tool install -g Gravicode.MediaPipeNet.Cli
mediapipenet-cli help
```

## Commands

| Command | |
|---|---|
| `mediapipenet-cli <task> <image> [--output out.png] [--json out.json \| --json-stdout]` | Run a task on an image. |
| `mediapipenet-cli video <task> [--input file.mp4 \| --camera 0] [--frames N]` | Video/webcam in video mode with live FPS, latency and drop statistics. |
| `mediapipenet-cli benchmark <task> <image> [--iterations 50]` | Mean, p50, p95, min latency and FPS. |
| `mediapipenet-cli models list \| download [ids…] \| verify` | Model status, download, SHA-256 verification. |
| `mediapipenet-cli info` | Version, ONNX Runtime, available execution providers, model cache. |

Tasks: `faces`, `face-mesh`, `hands`, `gestures`, `pose`, `holistic`, `segment`, `objects`, `classify`.

Common options: `--provider auto|cpu|directml|cuda|coreml`, `--threads N`, `--models-dir DIR`, `--no-download`.

## Examples

```text
$ mediapipenet-cli gestures thumb_up.jpg --output annotated.png
gestures: 1 hand(s): Thumb_Up (67.9 %)  [55.4 ms]
Annotated image written to annotated.png

$ mediapipenet-cli objects cats_and_dogs.jpg --json objects.json
objects: 4 object(s): dog (71.4 %), cat (68.8 %), dog (65.5 %), cat (64.5 %)  [87.6 ms]
JSON written to objects.json

$ mediapipenet-cli benchmark faces portrait.jpg --iterations 100
faces on portrait.jpg (820x1024), 100 runs:
  mean 9.37 ms | p50 8.61 ms | p95 14.37 ms | min 7.20 ms | 106.7 FPS

$ mediapipenet-cli models verify
  face_detection_short_range     OK
  …
```

The first call of a run includes model loading; `benchmark` warms up first. The Windows build of the tool carries
the OpenCV runtime for `video`; on Linux/macOS install OpenCV for the `video` command.
