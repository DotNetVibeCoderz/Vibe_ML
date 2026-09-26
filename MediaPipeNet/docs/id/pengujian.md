# Pengujian & validasi

> 🇬🇧 [Read in English](../en/testing.md)

```bash
dotnet test MediaPipeNet.slnx -c Release
dotnet test MediaPipeNet.slnx -c Release --collect:"XPlat Code Coverage"
```

## Proyek test (104 test)

| Proyek | Cakupan |
|---|---|
| `Gravicode.MediaPipeNet.Core.Tests` (44) | Timestamp, geometri, JSON, telemetri; format/stride/pooling `MPImage`; image-to-tensor (normalisasi, letterbox, rotasi, border replicate, anti-aliasing) dan round-trip mapping; proyeksi mask; frame source; sesi ONNX, context yang di-pool dan konkurensi; pemilihan provider dan fallback; katalog model, verifikasi checksum, provider NuGet/HTTP/embedded (dengan HTTP handler tiruan). |
| `Gravicode.MediaPipeNet.Framework.Tests` (23) | Packet; validasi (produsen ganda, port/stream tak dikenal, tipe tidak cocok, siklus, side packet hilang); urutan; sinkronisasi input dan propagasi bound; flow limiting dan pembuangan antrean; side packet; poller; kegagalan dan pembatalan; kebijakan immediate; parsing `.pbtxt`, round-trip, error; registry. |
| `Gravicode.MediaPipeNet.Tasks.Tests` (37) | **Validasi silang golden** kesembilan task terhadap MediaPipe Python; aturan running mode; callback dan pembuangan live-stream; tracking video; API async, ROI/rotasi, allow/deny list; penempatan tangan holistic; round-trip JSON; registrasi DI; `LiveStreamProcessor`; kalkulator vision di graph `.pbtxt`; visualisasi; anchor, decoding, NMS, matematika ROI, filter One-Euro, label. |

## Cakupan (baris)

| Modul | Baris | Branch |
|---|---:|---:|
| MediaPipeNet.Core | 94 % | 69 % |
| MediaPipeNet.Imaging | 94 % | 89 % |
| MediaPipeNet.Inference | 91 % | 70 % |
| MediaPipeNet.Framework | 94 % | 85 % |
| MediaPipeNet.Tasks.Vision | 92 % | 78 % |
| MediaPipeNet.Visualization | 99 % | 78 % |
| MediaPipeNet.Extensions.DI | 84 % | 75 % |

NFR-3 (≥ 70 % untuk modul inti) terpenuhi.

## Tiga tingkat validasi

1. **Konversi model** — `tools/model-conversion/validate_models.py` memberi input acak yang sama ke setiap graph
   TFLite asli dan hasil konversi ONNX-nya: selisih maks ≈ 1e-4 (`models/onnx/validation.json`).
2. **Validasi silang dengan MediaPipe Python** — `tools/golden/generate_golden.py` menjalankan paket resmi
   `mediapipe` 1.0.1 pada gambar fixture dan menyimpan hasilnya di
   `tests/assets/golden/mediapipe_python_reference.json`. `GoldenTests` membandingkan MediaPipe.NET dengan toleransi
   (IoU kotak > 0,9, error rata-rata landmark < 0,006–0,02, label gestur/objek/kelas identik, rata-rata mask dalam
   0,01).
3. **Perilaku streaming** — test memberi urutan frame untuk memverifikasi timestamp monoton, tracking, dan pembuangan
   packet di mode live-stream.

Referensi ini menangkap bug porting nyata selama pengembangan: skala anchor EfficientDet (3, bukan 4), sigmoid ganda
pada skor detektor, normalisasi input gestur yang ternyata tanpa rotasi, dan tensor densify yang belum
terinisialisasi pada konversi detektor pose.

## Membuat ulang data golden

```bash
python -m venv C:\mpp && C:\mpp\Scripts\pip install mediapipe
C:\mpp\Scripts\python tools/golden/generate_golden.py --models artifacts/models/_work
```

## Benchmark

`benchmarks/MediaPipeNet.Benchmarks` (BenchmarkDotNet) — lihat [Performa](performa.md). CI menjalankannya pada tag
rilis.
