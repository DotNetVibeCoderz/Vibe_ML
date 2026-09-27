# Gravicode.MediaPipeNet.Tasks.Text

Text tasks of MediaPipe.NET, ported from Google MediaPipe Tasks and running on ONNX Runtime:

- **TextClassifier** — sentiment with MobileBERT (SST-2) or the tiny average word-embedding model.
- **TextEmbedder** — 512-D MobileBERT sentence embeddings with cosine similarity.
- **LanguageDetector** — 110 languages from character n-grams.

The tokenizers (BERT WordPiece, regex, n-gram hashing) reproduce MediaPipe's exactly.

```csharp
using MediaPipeNet.Tasks.Text;

using var classifier = TextClassifier.Create();
Console.WriteLine(classifier.Classify("It's beautiful outside.").TopCategory); // positive

using var detector = LanguageDetector.Create();
Console.WriteLine(detector.Detect("Selamat pagi, apa kabar?").TopLanguage);   // id
```

Models come from `Gravicode.MediaPipeNet.Models.Text` (installed or downloaded on first use).

---
**MediaPipe.NET** — a native .NET 10 port of Google MediaPipe's tasks on ONNX Runtime.
Created by **Gravicode Studios**, led by **Kang Fadhil**. Library: Apache-2.0. Model weights: © Google LLC, Apache-2.0.
Documentation (English & Bahasa Indonesia): see the `docs/` folder of the repository.
