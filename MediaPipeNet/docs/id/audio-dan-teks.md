# Task audio & teks

> 🇬🇧 [Read in English](../en/audio-and-text.md)

`Gravicode.MediaPipeNet.Tasks.Audio` dan `Gravicode.MediaPipeNet.Tasks.Text` adalah port task audio dan teks
MediaPipe. Keduanya hanya bergantung pada `Tasks.Core` dan `Inference` (tanpa stack pengolahan gambar), serta
berbagi `BaseOptions`, model store, execution provider, dan varian presisi dengan task vision.

```bash
dotnet add package Gravicode.MediaPipeNet.Tasks.Audio      # + Gravicode.MediaPipeNet.Models.Audio (opsional, offline)
dotnet add package Gravicode.MediaPipeNet.Tasks.Text       # + Gravicode.MediaPipeNet.Models.Text  (opsional, offline)
```

(Paket native ONNX Runtime tetap diperlukan: referensikan `Gravicode.MediaPipeNet`, `.DirectML`, `.Cuda`, atau
`Microsoft.ML.OnnxRuntime` langsung.)

![Klasifikasi audio](../images/gallery-audio.png)

## Input audio: `AudioData`

`AudioData` adalah audio mono float dengan sample rate yang diketahui. `AudioData.LoadWav(path | stream | bytes)`
membaca RIFF/WAVE (PCM 8/16/24/32-bit, IEEE float 32/64-bit, `WAVE_FORMAT_EXTENSIBLE`, jumlah kanal berapa pun —
di-downmix dengan rata-rata). `FromPcm16` / `FromInterleaved` membungkus buffer dari API mikrofon. `Resample(rate)`
adalah resampler band-limited (sinc berjendela Hann, anti-aliasing); task otomatis me-resample ke 16 kHz.

## AudioClassifier

YAMNet: 521 kelas AudioSet (ucapan, musik, instrumen, hewan, kendaraan, alarm, …) untuk setiap jendela 0,975 detik
(15.600 sampel pada 16 kHz).

| Opsi | Default | |
|---|---|---|
| `RunningMode` | `AudioClips` | `AudioClips`: `Classify(clip)` mengembalikan satu hasil per jendela (jendela terakhir diisi nol, seperti MediaPipe). `AudioStream`: `ClassifyStream(chunk, timestampMs)` menerima potongan berukuran bebas dan memanggil `ResultCallback` untuk setiap jendela yang lengkap. |
| `Classifier` | 5 teratas | `ClassifierOptions`: `MaxResults`, `ScoreThreshold`, `CategoryAllowlist`, `CategoryDenylist`. |
| `ModelPath` / `Labels` | null | Model audio kustom — lihat [Model kustom](model-kustom.md). |

```csharp
using var classifier = AudioClassifier.Create(new() { Classifier = new() { MaxResults = 3 } });
foreach (var window in classifier.Classify(AudioData.LoadWav("speech.wav")))
    Console.WriteLine($"{window.TimestampMs} ms: {string.Join(", ", window.Categories)}");
// 0 ms: Speech (91.8 %), Inside, small room (5.9 %), ...

// Streaming dari mikrofon:
using var live = AudioClassifier.Create(new()
{
    RunningMode = AudioRunningMode.AudioStream,
    ResultCallback = r => Console.WriteLine($"{r.TimestampMs} ms {r.TopCategory}"),
});
live.ClassifyStream(AudioData.FromPcm16(buffer, 48_000), timestampMs);
```

Akurasi: pada fixture ucapan MediaPipe, kelima jendela beserta kategori teratasnya sama dengan paket Python resmi;
skor berbeda hingga 0,07 karena YAMNet Google berjalan dengan kernel int8 di TFLite sedangkan model ONNX menghitung
dalam float.

## VoiceActivityDetector

Dibangun di atas YAMNet: sebuah jendela dianggap ucapan bila skor tertinggi dari kelas-kelas ucapan (Speech,
Conversation, Narration, Child speech, Babbling, Speech synthesizer, Shout, Yell, Whispering) mencapai `Threshold`
(0,5). Jendela saling tumpang-tindih (`HopMs`, default 487 ms) agar pewaktuan lebih halus; jendela ucapan yang
berdekatan (jeda < `MinSilenceMs`, 500) digabung menjadi `SpeechSegment`, dan segmen yang lebih pendek dari
`MinSpeechMs` (250) dibuang.

```csharp
using var vad = VoiceActivityDetector.Create();
var result = vad.Detect(AudioData.LoadWav("rapat.wav"));
foreach (var s in result.Segments) Console.WriteLine($"bicara {s.StartMs}–{s.EndMs} ms (p {s.MeanProbability:P0})");
Console.WriteLine($"{result.SpeechRatio:P0} rekaman berisi ucapan");
```

Pada mode `AudioStream`, `DetectStream(chunk, ts)` melaporkan setiap `VoiceActivityFrame` lewat `FrameCallback`;
`VoiceActivityDetector.Merge(frames, …)` mengubah frame yang terkumpul menjadi segmen.

![Pemahaman teks](../images/gallery-text.png)

## TextClassifier

Sentimen (SST-2) dengan salah satu model:

| `Model` | Ukuran | Kecepatan (CPU) | Label |
|---|---|---|---|
| `Bert` (default) | 26 MB | ~60 ms | negative, positive |
| `AverageWord` | 0,6 MB | ~0,04 ms | 0 (negatif), 1 (positif) |

```csharp
using var classifier = TextClassifier.Create();
Console.WriteLine(classifier.Classify("It's beautiful outside.").TopCategory);   // positive (99.9 %)
var banyak = classifier.ClassifyBatch(ulasan);                                     // paralel
```

Kedua tokenizer mereproduksi milik MediaPipe: preprocessor BERT mengecilkan huruf ASCII, memecah pada spasi, tanda
baca, dan ideograf CJK, lalu menjalankan WordPiece greedy dengan lanjutan `##` dan `[UNK]` (`BertTokenizer`); model
average-word memecah dengan regex `[^\w\']+` memakai `\w` ASCII ala RE2, tanpa pengecilan huruf, dan menambahkan
`<START>` / `<PAD>` / `<UNKNOWN>` (`RegexTokenizer`). Skor cocok dengan paket resmi dalam selisih 0,01.

## TextEmbedder

Embedding kalimat MobileBERT (512-D). Opsi `L2Normalize`, `Quantize`. `TextEmbedder.CosineSimilarity(a, b)`,
`EmbedBatch(texts)`.

```csharp
using var embedder = TextEmbedder.Create(new() { L2Normalize = true });
double s = TextEmbedder.CosineSimilarity(embedder.Embed("I love sunny days.").Embedding,
                                         embedder.Embed("Sunny weather makes me happy.").Embedding);   // ≈ 0,97
```

Model MobileBERT menyimpan bobot int8. MediaPipe menjalankannya dengan aktivasi yang dikuantisasi dinamis;
MediaPipe.NET menghitung jaringan yang sama dalam float (lihat [Model](model.md#bobot-int8-dan-onnx-runtime)), sehingga
kemiripan berbeda sekitar 0,02–0,03 dari MediaPipe namun identik dengan model TFLite float.

## LanguageDetector

110 bahasa dari n-gram karakter. Hashing n-gram yang di model TFLite dikerjakan custom op kini berjalan di C#
(`NGramHasher`: pengecilan huruf, non-huruf → spasi, penanda `^…$`, 1- hingga 4-gram di-hash dengan MurmurHash
64-bit), sedangkan lookup embedding dan classifier berjalan di ONNX Runtime. Probabilitas cocok dengan MediaPipe
dalam selisih 0,001 untuk kelima kalimat uji.

```csharp
using var detector = LanguageDetector.Create();
foreach (var p in detector.Detect("Selamat pagi, apa kabar hari ini?").Predictions)
    Console.WriteLine($"{p.LanguageCode} {p.Probability:P1}");    // id 97.6 %, ms 2.2 %, …
```

## Command line

```bash
mediapipenet-cli audio speech.wav [--vad] [--top 3]
mediapipenet-cli text classify "It's beautiful outside."
mediapipenet-cli text language "Selamat pagi" "Guten Morgen"
mediapipenet-cli text embed "I love sunny days." "Sunny weather makes me happy."
```
