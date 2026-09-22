# Getting started

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

## Requirements

- .NET 10 SDK
- An internet connection for anything that touches the Hub

## Build

```bash
git clone <this repository>
cd HF.Net
dotnet build HF.Net.sln -c Release
dotnet test
```

All 178 tests run offline.

## Your first model

```bash
dotnet run --project samples/GraviTransformers.Console -- bert-base-uncased
```

The first run downloads about 440 MB. Later runs revalidate the cache with a single HTTP request.

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

`Load` is doing four things that all have to agree: reading `config.json`, downloading the weights,
mapping the checkpoint's parameter names onto the encoder, and fetching the matching tokenizer. If
any of them is wrong the model still runs — so the fill-mask result above is the real test, not the
absence of an exception.

## A token

Set `HF_TOKEN` before running anything that touches the Hub:

```bash
export HF_TOKEN=hf_...          # bash
$env:HF_TOKEN = "hf_..."        # PowerShell
```

You need one for private and gated repositories. Without one you are rate limited as an anonymous
client, which is low enough to interrupt a batch of downloads.

## Where downloads go

The same cache the Python tooling uses, resolved in this order:

1. `HF_HUB_CACHE`
2. `HF_HOME`, plus `/hub`
3. `%LOCALAPPDATA%/huggingface/hub` on Windows, `~/.cache/huggingface/hub` elsewhere

Sharing the cache means a machine with both toolchains keeps one copy of each checkpoint.

```csharp
Console.WriteLine(Hub.Cache.Root);
Console.WriteLine($"{Hub.Cache.SizeInBytes() / (1024.0 * 1024):N0} MB");
```

The layout is a plain readable directory tree — `models/bert-base-uncased/main/model.safetensors` —
rather than the blob-and-symlink arrangement the Python client uses. Symlinks need elevation or
Developer Mode on Windows, and a cache you cannot inspect in a file browser is one you cannot clear
when something goes wrong.

## Classification

A fine-tuned checkpoint carries its own head and its own label names:

```csharp
using var model = TransformerModel.Load("distilbert-base-uncased-finetuned-sst-2-english");

Console.WriteLine(string.Join(", ", model.Labels));          // NEGATIVE, POSITIVE

var best = model.Predict("I absolutely loved this film.", topK: 1)[0];
Console.WriteLine($"{best.Label} {best.Score:P2}");          // POSITIVE 99.99 %
```

A *base* checkpoint has no classification head, and `Predict` says so rather than inventing one.

## Embeddings and similarity

```csharp
using var model = TransformerModel.Load("bert-base-uncased");

Console.WriteLine(model.Similarity("the cat sat on the mat",
                                   "the dog sat on the rug"));       // 0.8949
Console.WriteLine(model.Similarity("the cat sat on the mat",
                                   "quarterly earnings beat expectations"));  // 0.4936
```

`Embed` mean-pools over the tokens rather than taking `[CLS]`. On a model that has not been
fine-tuned for sentence similarity, `[CLS]` is close to constant and makes every pair look alike.

## When it says no

```csharp
TransformerModel.Load("gpt2");
// NotSupportedException: 'gpt2' is a 'gpt2' model. GraviTransformers runs BERT-family
// encoders (bert, camembert, deberta, distilbert, electra, mpnet, roberta, xlm-roberta);
// decoder-only and encoder-decoder architectures need causal masking and cross-attention,
// which this encoder does not have.
```

This is deliberate. Filling an encoder from a decoder's weights produces a model that runs and
returns nonsense.

## Next

- [GraviTransformers](GraviTransformers.md) for what else the encoder can do
- [GraviOptimum](GraviOptimum.md) when you need throughput
- [HFAppGen](HFAppGen.md) to have the application written for you
