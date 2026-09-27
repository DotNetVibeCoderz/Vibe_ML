# Audio & text tasks

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/audio-dan-teks.md)

`Gravicode.MediaPipeNet.Tasks.Audio` and `Gravicode.MediaPipeNet.Tasks.Text` port MediaPipe's audio and text tasks.
They depend only on `Tasks.Core` and `Inference` (no imaging stack), and share `BaseOptions`, the model store,
execution providers and precision variants with the vision tasks.

```bash
dotnet add package Gravicode.MediaPipeNet.Tasks.Audio      # + Gravicode.MediaPipeNet.Models.Audio (optional, offline)
dotnet add package Gravicode.MediaPipeNet.Tasks.Text       # + Gravicode.MediaPipeNet.Models.Text  (optional, offline)
```

(An ONNX Runtime native package is still needed: reference `Gravicode.MediaPipeNet`, `.DirectML` or `.Cuda`, or
`Microsoft.ML.OnnxRuntime` directly.)

![Audio classification](../images/gallery-audio.png)

## Audio input: `AudioData`

`AudioData` is mono float audio at a known sample rate. `AudioData.LoadWav(path | stream | bytes)` reads RIFF/WAVE
(PCM 8/16/24/32-bit, IEEE float 32/64-bit, `WAVE_FORMAT_EXTENSIBLE`, any number of channels — down-mixed by
averaging). `FromPcm16` / `FromInterleaved` wrap buffers from a microphone API. `Resample(rate)` is a band-limited
(Hann-windowed sinc, anti-aliased) resampler; the tasks resample to 16 kHz automatically.

## AudioClassifier

YAMNet: 521 AudioSet classes (speech, music, instruments, animals, vehicles, alarms, …) for every 0.975 s window
(15 600 samples at 16 kHz).

| Option | Default | |
|---|---|---|
| `RunningMode` | `AudioClips` | `AudioClips`: `Classify(clip)` returns one result per window (the last window zero-padded, as MediaPipe). `AudioStream`: `ClassifyStream(chunk, timestampMs)` accepts chunks of any size and calls `ResultCallback` for every completed window. |
| `Classifier` | top 5 | `ClassifierOptions`: `MaxResults`, `ScoreThreshold`, `CategoryAllowlist`, `CategoryDenylist`. |
| `ModelPath` / `Labels` | null | A custom audio model — see [Custom models](custom-models.md). |

```csharp
using var classifier = AudioClassifier.Create(new() { Classifier = new() { MaxResults = 3 } });
foreach (var window in classifier.Classify(AudioData.LoadWav("speech.wav")))
    Console.WriteLine($"{window.TimestampMs} ms: {string.Join(", ", window.Categories)}");
// 0 ms: Speech (91.8 %), Inside, small room (5.9 %), ...

// Streaming from a microphone:
using var live = AudioClassifier.Create(new()
{
    RunningMode = AudioRunningMode.AudioStream,
    ResultCallback = r => Console.WriteLine($"{r.TimestampMs} ms {r.TopCategory}"),
});
live.ClassifyStream(AudioData.FromPcm16(buffer, 48_000), timestampMs);
```

Accuracy: on MediaPipe's speech fixture the five windows and their top categories match the official Python
package; scores differ by up to 0.07 because Google's YAMNet runs with int8 kernels in TFLite while the ONNX model
computes in float.

## VoiceActivityDetector

Built on YAMNet: a window is speech when the highest score among the speech classes (Speech, Conversation,
Narration, Child speech, Babbling, Speech synthesizer, Shout, Yell, Whispering) reaches `Threshold` (0.5).
Windows overlap (`HopMs`, default 487 ms) for finer timing; neighbouring speech windows closer than
`MinSilenceMs` (500) merge into `SpeechSegment`s, and segments shorter than `MinSpeechMs` (250) are dropped.

```csharp
using var vad = VoiceActivityDetector.Create();
var result = vad.Detect(AudioData.LoadWav("meeting.wav"));
foreach (var s in result.Segments) Console.WriteLine($"speech {s.StartMs}–{s.EndMs} ms (p {s.MeanProbability:P0})");
Console.WriteLine($"{result.SpeechRatio:P0} of the recording contains speech");
```

In `AudioStream` mode, `DetectStream(chunk, ts)` reports every analysed `VoiceActivityFrame` through
`FrameCallback`; `VoiceActivityDetector.Merge(frames, …)` turns collected frames into segments.

![Text understanding](../images/gallery-text.png)

## TextClassifier

Sentiment (SST-2) with either model:

| `Model` | Size | Speed (CPU) | Labels |
|---|---|---|---|
| `Bert` (default) | 26 MB | ~60 ms | negative, positive |
| `AverageWord` | 0.6 MB | ~0.04 ms | 0 (negative), 1 (positive) |

```csharp
using var classifier = TextClassifier.Create();
Console.WriteLine(classifier.Classify("It's beautiful outside.").TopCategory);   // positive (99.9 %)
var many = classifier.ClassifyBatch(reviews);                                      // parallel
```

Both tokenizers reproduce MediaPipe's: the BERT preprocessor lower-cases ASCII, splits on whitespace, punctuation
and CJK ideographs and runs greedy WordPiece with `##` continuations and `[UNK]` (`BertTokenizer`); the average-word
model splits on the regex `[^\w\']+` with RE2's ASCII `\w`, without lower-casing, and pads with `<START>` / `<PAD>` /
`<UNKNOWN>` (`RegexTokenizer`). Scores match the official package to within 0.01.

## TextEmbedder

MobileBERT sentence embeddings (512-D). Options `L2Normalize`, `Quantize`. `TextEmbedder.CosineSimilarity(a, b)`,
`EmbedBatch(texts)`.

```csharp
using var embedder = TextEmbedder.Create(new() { L2Normalize = true });
double s = TextEmbedder.CosineSimilarity(embedder.Embed("I love sunny days.").Embedding,
                                         embedder.Embed("Sunny weather makes me happy.").Embedding);   // ≈ 0.97
```

The MobileBERT models store int8 weights. MediaPipe runs them with dynamically quantized activations; MediaPipe.NET
computes the same network in float (see [Models](models.md#int8-weights-and-onnx-runtime)), so similarities
differ from MediaPipe's by about 0.02–0.03 while matching the float TFLite model exactly.

## LanguageDetector

110 languages from character n-grams. The n-gram hashing that the TFLite model does with a custom op runs in C#
(`NGramHasher`: lower-casing, non-letters → spaces, `^…$` markers, 1- to 4-grams hashed with 64-bit MurmurHash),
and the embedding lookup and classifier run in ONNX Runtime. Probabilities match MediaPipe to within 0.001 on all
five test sentences.

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
