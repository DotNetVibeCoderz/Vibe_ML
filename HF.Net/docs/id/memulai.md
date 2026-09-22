# Memulai

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

## Kebutuhan

- .NET 10 SDK
- Koneksi internet untuk apa pun yang menyentuh Hub

## Build

```bash
git clone <repositori ini>
cd HF.Net
dotnet build HF.Net.sln -c Release
dotnet test
```

Seluruh 178 test berjalan tanpa jaringan.

## Model pertama Anda

```bash
dotnet run --project samples/GraviTransformers.Console -- bert-base-uncased
```

Jalan pertama mengunduh sekitar 440 MB. Jalan berikutnya hanya memvalidasi ulang cache dengan satu
permintaan HTTP.

```csharp
using Gravicode.HFNet.GraviTransformers;

using var model = TransformerModel.Load("bert-base-uncased");

foreach (var fill in model.FillMask("The capital of France is [MASK].", topK: 3))
    Console.WriteLine($"{fill.Token,-10} {fill.Score:P2}");
```

```
paris      41.53 %
lille       7.16 %
lyon        6.31 %
```

`Load` mengerjakan empat hal yang semuanya harus sepakat: membaca `config.json`, mengunduh bobot,
memetakan nama parameter checkpoint ke encoder, dan mengambil tokenizer yang cocok. Kalau salah satu
keliru, model tetap berjalan — jadi hasil fill-mask di atas itulah ujian yang sebenarnya, bukan
ketiadaan exception.

## Token

Set `HF_TOKEN` sebelum menjalankan apa pun yang menyentuh Hub:

```bash
export HF_TOKEN=hf_...          # bash
$env:HF_TOKEN = "hf_..."        # PowerShell
```

Token diperlukan untuk repositori privat dan gated. Tanpa token Anda dibatasi sebagai klien anonim,
dan batasnya cukup rendah untuk memutus serangkaian unduhan.

## Ke mana unduhan disimpan

Ke cache yang sama dengan yang dipakai perkakas Python, diselesaikan dalam urutan ini:

1. `HF_HUB_CACHE`
2. `HF_HOME`, ditambah `/hub`
3. `%LOCALAPPDATA%/huggingface/hub` di Windows, `~/.cache/huggingface/hub` di sistem lain

Berbagi cache berarti mesin dengan kedua toolchain hanya menyimpan satu salinan tiap checkpoint.

```csharp
Console.WriteLine(Hub.Cache.Root);
Console.WriteLine($"{Hub.Cache.SizeInBytes() / (1024.0 * 1024):N0} MB");
```

Tata letaknya berupa pohon direktori biasa yang bisa dibaca —
`models/bert-base-uncased/main/model.safetensors` — bukan susunan blob-dan-symlink seperti klien
Python. Symlink memerlukan hak elevasi atau Developer Mode di Windows, dan cache yang tidak bisa
diperiksa lewat file browser adalah cache yang tidak bisa dibersihkan ketika terjadi masalah.

## Klasifikasi

Checkpoint yang sudah di-fine-tune membawa head dan nama labelnya sendiri:

```csharp
using var model = TransformerModel.Load("distilbert-base-uncased-finetuned-sst-2-english");

Console.WriteLine(string.Join(", ", model.Labels));          // NEGATIVE, POSITIVE

var best = model.Predict("I absolutely loved this film.", topK: 1)[0];
Console.WriteLine($"{best.Label} {best.Score:P2}");          // POSITIVE 99.99 %
```

Checkpoint *dasar* tidak punya head klasifikasi, dan `Predict` mengatakannya alih-alih mengarang.

## Embedding dan kemiripan

```csharp
using var model = TransformerModel.Load("bert-base-uncased");

Console.WriteLine(model.Similarity("the cat sat on the mat",
                                   "the dog sat on the rug"));       // 0.8949
Console.WriteLine(model.Similarity("the cat sat on the mat",
                                   "quarterly earnings beat expectations"));  // 0.4936
```

`Embed` melakukan mean-pooling atas token, bukan mengambil `[CLS]`. Pada model yang belum di-fine-tune
untuk kemiripan kalimat, `[CLS]` nyaris konstan sehingga semua pasangan kalimat tampak mirip.

## Ketika ia menolak

```csharp
TransformerModel.Load("gpt2");
// NotSupportedException: 'gpt2' adalah model 'gpt2'. GraviTransformers menjalankan encoder
// keluarga BERT (...); arsitektur decoder-only dan encoder-decoder memerlukan causal masking
// dan cross-attention yang tidak dimiliki encoder ini.
```

Ini disengaja. Mengisi encoder dari bobot decoder menghasilkan model yang berjalan mulus dan
mengembalikan omong kosong.

## Selanjutnya

- [GraviTransformers](GraviTransformers.md) untuk kemampuan encoder lainnya
- [GraviOptimum](GraviOptimum.md) ketika Anda butuh throughput
- [HFAppGen](HFAppGen.md) agar aplikasinya dituliskan untuk Anda
