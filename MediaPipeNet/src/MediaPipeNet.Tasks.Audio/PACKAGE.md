# Gravicode.MediaPipeNet.Tasks.Audio

Audio tasks of MediaPipe.NET, ported from Google MediaPipe Tasks and running on ONNX Runtime:

- **AudioClassifier** — YAMNet, 521 AudioSet classes per 0.975 s window (speech, music, animals, alarms, ...).
- **VoiceActivityDetector** — speech/non-speech frames and merged speech segments, built on YAMNet's speech classes.

Both support an audio-clips mode and an audio-stream mode (chunks of any size, callback per window).
`AudioData` loads WAV files (PCM 8/16/24/32-bit, float) and resamples to 16 kHz.

```csharp
using MediaPipeNet.Tasks.Audio;

using var classifier = AudioClassifier.Create();
foreach (var window in classifier.Classify(AudioData.LoadWav("speech.wav")))
    Console.WriteLine($"{window.TimestampMs} ms: {window.TopCategory}");
```

The YAMNet model comes from `Gravicode.MediaPipeNet.Models.Audio` (installed or downloaded on first use).

---
**MediaPipe.NET** — a native .NET 10 port of Google MediaPipe's tasks on ONNX Runtime.
Created by **Gravicode Studios**, led by **Kang Fadhil**. Library: Apache-2.0. Model weights: © Google LLC, Apache-2.0.
Documentation (English & Bahasa Indonesia): see the `docs/` folder of the repository.
