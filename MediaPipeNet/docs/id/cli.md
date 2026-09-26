# CLI — `mediapipenet-cli`

> 🇬🇧 [Read in English](../en/cli.md)

```bash
dotnet tool install -g Gravicode.MediaPipeNet.Cli
mediapipenet-cli help
```

## Perintah

| Perintah | |
|---|---|
| `mediapipenet-cli <task> <gambar> [--output hasil.png] [--json hasil.json \| --json-stdout]` | Jalankan task pada gambar. |
| `mediapipenet-cli video <task> [--input file.mp4 \| --camera 0] [--frames N]` | Video/webcam dalam mode video dengan statistik FPS, latensi, dan frame yang dibuang. |
| `mediapipenet-cli benchmark <task> <gambar> [--iterations 50]` | Latensi rata-rata, p50, p95, minimum, dan FPS. |
| `mediapipenet-cli models list \| download [id…] \| verify` | Status model, unduh, verifikasi SHA-256. |
| `mediapipenet-cli info` | Versi, ONNX Runtime, execution provider yang tersedia, cache model. |

Task: `faces`, `face-mesh`, `hands`, `gestures`, `pose`, `holistic`, `segment`, `objects`, `classify`.

Opsi umum: `--provider auto|cpu|directml|cuda|coreml`, `--threads N`, `--models-dir DIR`, `--no-download`.

## Contoh

```text
$ mediapipenet-cli gestures thumb_up.jpg --output beranotasi.png
gestures: 1 hand(s): Thumb_Up (67.9 %)  [55.4 ms]
Annotated image written to beranotasi.png

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

Panggilan pertama dalam satu run termasuk waktu memuat model; `benchmark` melakukan warm-up lebih dulu. Build tool
untuk Windows membawa runtime OpenCV untuk perintah `video`; di Linux/macOS pasang OpenCV untuk perintah tersebut.
