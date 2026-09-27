# Pengujian & validasi

> 🇬🇧 [Read in English](../en/testing.md)

```bash
dotnet test MediaPipeNet.slnx -c Release
dotnet test MediaPipeNet.slnx -c Release --collect:"XPlat Code Coverage"
```

## Proyek tes (159 tes)

| Proyek | Cakupan |
|---|---|
| `MediaPipeNet.Core.Tests` (44) | Timestamp, geometri, JSON, telemetri; format/stride/pooling `MPImage`; image-to-tensor (normalisasi, letterbox, rotasi, border replicate, anti-aliasing) dan round-trip mapping; proyeksi mask; frame source; sesi ONNX, konteks pool dan konkurensi; pemilihan provider dan fallback; katalog model (ke-25 file diverifikasi SHA-256), verifikasi checksum, provider NuGet/HTTP/embedded. |
| `MediaPipeNet.Framework.Tests` (31) | Packet; validasi; urutan; sinkronisasi input dan propagasi bound; flow limiting dan pembuangan antrean; side packet; poller; kegagalan dan pembatalan; policy immediate; parsing `.pbtxt`; **back edge dan loopback, executor, Chrome tracing, subgraph (kode dan .pbtxt)**. |
| `MediaPipeNet.Tasks.Tests` (84) | **Validasi silang golden** setiap task terhadap MediaPipe Python (referensi v1 dan v2); varian presisi; model kustom; batch, I/O binding, dan thread safety; tokenizer audio/teks, pemuatan WAV dan resampling; penyempurnaan heatmap, segmentation smoothing, ROI tracking; aturan running mode, live stream, tracking, DI, visualisasi, kalkulator graph. |

## Cakupan (baris)

| Modul | Baris | Branch |
|---|---:|---:|
| MediaPipeNet.Core | 99 % | 76 % |
| MediaPipeNet.Imaging | 96 % | 86 % |
| MediaPipeNet.Inference | 91 % | 71 % |
| MediaPipeNet.Framework | 94 % | 84 % |
| MediaPipeNet.Tasks.Core | 97 % | 62 % |
| MediaPipeNet.Tasks.Vision | 91 % | 77 % |
| MediaPipeNet.Tasks.Audio | 83 % | 62 % |
| MediaPipeNet.Tasks.Text | 89 % | 75 % |
| MediaPipeNet.Visualization | 81 % | 67 % |
| MediaPipeNet.Extensions.DI | 100 % | 88 % |

NFR-3 (≥ 70 % untuk modul inti) terpenuhi.

## Tiga tingkat validasi

1. **Konversi model** — `tools/model-conversion/validate_models.py` memberi input acak yang sama ke setiap graph
   TFLite asli dan hasil konversi ONNX-nya: selisih maks ≈ 1e-4 (`models/onnx/validation.json`).
2. **Validasi silang dengan MediaPipe Python** — `tools/golden/generate_golden.py` dan `generate_golden_v2.py`
   menjalankan paket resmi `mediapipe` 1.0.1 pada gambar, audio, dan kalimat fixture lalu menyimpan hasilnya di
   `tests/assets/golden/mediapipe_python_reference.json` dan `…_v2.json` (wajah full-range, matriks transformasi
   wajah, holistic, embedding gambar, segmentasi multikelas/rambut/DeepLab, klasifikasi audio, klasifikasi dan
   embedding teks, deteksi bahasa). `GoldenTests` membandingkan MediaPipe.NET dengan toleransi
   (IoU kotak > 0,9, error rata-rata landmark < 0,006–0,02, label gestur/objek/kelas identik, rata-rata mask dalam
   0,01).
3. **Perilaku streaming** — test memberi urutan frame untuk memverifikasi timestamp monoton, tracking, dan pembuangan
   packet di mode live-stream.

Referensi v2 memastikan: sensitivitas DeepLab terhadap resampling (preprocessing CPU MediaPipe tidak memakai
anti-aliasing, maka segmenter pun tidak), kanal alpha kosong pada hair segmenter, dan — lewat perbandingan tensor
demi tensor dengan interpreter TFLite — emulasi kuantisasi dinamis tf2onnx serta fusi QDQ ONNX Runtime yang
menggeser embedding MobileBERT. Referensi sebelumnya menangkap: skala anchor EfficientDet (3, bukan 4), sigmoid ganda
pada skor detektor, normalisasi input gestur yang ternyata tanpa rotasi, dan tensor densify yang belum
terinisialisasi pada konversi detektor pose.

## Membuat ulang data golden

```bash
python -m venv C:\mpp && C:\mpp\Scripts\pip install mediapipe
C:\mpp\Scripts\python tools/golden/generate_golden.py --models artifacts/models/_work
C:\mpp\Scripts\python tools/golden/generate_golden_v2.py --models artifacts/models/_work
```

Bagian yang tidak dapat dijalankan paket Python (interactive segmenter dan face stylizer tidak ada di 1.0.1) dicatat
di `"unavailable"`; interactive segmenter divalidasi secara perilaku (mask harus menutupi objek di bawah titik dan
sedikit di luar itu).

## Benchmark

`benchmarks/MediaPipeNet.Benchmarks` (BenchmarkDotNet) — lihat [Performa](performa.md). CI menjalankannya pada tag
rilis.
