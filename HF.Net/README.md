# HF.Net

**A Hugging Face style machine learning ecosystem for .NET.**

Load a real model from the Hugging Face Hub, tokenize exactly the way it was trained, and run it —
all from C#, with no Python in the loop.

```csharp
using var model = TransformerModel.Load("bert-base-uncased");

foreach (var fill in model.FillMask("The capital of France is [MASK].", topK: 3))
    Console.WriteLine($"{fill.Token,-10} {fill.Score:P2}");

// paris      41.53 %
// lille       7.16 %
// lyon        6.31 %
```

Built on [Gravicode.Science](https://github.com/DotNetVibeCoderz/Vibe_ML/tree/main/GravicodeScience)
— `GraviNum` (arrays), `GraviFrame` (dataframes) and `GraviLearn` (classical ML).

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

**Documentation:** [English](docs/) · [Bahasa Indonesia](docs/id/)

---

## The libraries

Each mirrors a package in the Python Hugging Face stack.

| Library | Mirrors | What it does |
|---|---|---|
| **GraviHub** | `huggingface_hub` | Hub download and upload, a readable local cache, and readers for the formats the Hub serves: **safetensors** and **`pytorch_model.bin`** |
| **GraviTokenizers** | `tokenizers` | WordPiece, byte-level BPE, Unigram and CLIP, loading `tokenizer.json` — with character offsets that point back into the original text |
| **GraviDatasets** | `datasets` | CSV, Parquet, JSON and JSON Lines, Hub datasets, splits, streaming and memory-mapped reads |
| **GraviTransformers** | `transformers` | Pretrained BERT-family encoders, GPT-2, ViT and CLIP: classification, fill-mask, named entities, question answering, embeddings, text generation, image and zero-shot classification — and streaming for models larger than memory |
| **GraviPEFT** | `peft` | LoRA and prefix tuning in the Hugging Face PEFT format — train, apply, merge, save, load |
| **GraviAccelerate** | `accelerate` | Device selection across CPU SIMD and ILGPU, sharding, weighted gradient averaging, honest throughput measurement |
| **GraviOptimum** | `optimum` | ONNX Runtime inference with an explicitly chosen execution provider, and weight quantisation that reports its measured error |
| **GraviDiffusers** | `diffusers` | DDPM, DDIM and Euler schedulers, and Stable Diffusion over ONNX — text to image, image to image, inpainting, LoRA — reproducing diffusers' own pipelines |

```
Gravicode.Science ──► GraviHub ──┬──► GraviTokenizers ──┐
  GraviNum                       ├──► GraviDatasets     ├──► GraviTransformers ──┬──► GraviPEFT
  GraviFrame                     └──► GraviOptimum ─────┘                        └──► GraviDiffusers
  GraviLearn                          GraviAccelerate
```

## Install

```bash
dotnet add package Gravicode.HFNet.GraviTransformers
```

Each library is a separate package; add only what you need, and the rest comes transitively.

| | |
|---|---|
| `Gravicode.HFNet.GraviHub` | `Gravicode.HFNet.GraviPEFT` |
| `Gravicode.HFNet.GraviTokenizers` | `Gravicode.HFNet.GraviAccelerate` |
| `Gravicode.HFNet.GraviDatasets` | `Gravicode.HFNet.GraviOptimum` |
| `Gravicode.HFNet.GraviTransformers` | `Gravicode.HFNet.GraviDiffusers` |

Or build from source:

```bash
git clone <this repository>
cd HF.Net
dotnet build HF.Net.sln -c Release
dotnet run --project samples/GraviTransformers.Console -- bert-base-uncased
```

Set `HF_TOKEN` for private or gated repositories, and for a much higher anonymous rate limit.
Downloads land in the same cache the Python tooling uses (`HF_HUB_CACHE`, then `HF_HOME`), so the
two share a machine without duplicating a single checkpoint.

## What it can do today

**Reads both weight formats the Hub serves.** safetensors is memory-mapped and read lazily, so
listing four hundred tensors costs a header read rather than a multi-gigabyte load. The many
repositories that only ever published `pytorch_model.bin` work too — the pickle is interpreted
against an allow-list of tensor constructors, so nothing in the file is executed.

**Tokenizes identically to the reference implementation.** `bert-base-uncased` on
*"Hello, world! Tokenizers are unbelievable."* produces
`[CLS] hello , world ! token ##izer ##s are unbelievable . [SEP]` with ids
`101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102` — the same ids Python gives.

**Runs real models.** Verified against `bert-base-uncased` (fill-mask),
`distilbert-base-uncased-finetuned-sst-2-english` (classification: 99.99% POSITIVE on
*"I absolutely loved this film."*) and `prajjwal1/bert-tiny` (a pickle-only checkpoint).

**Sees, as well as reads.** `google/vit-base-patch16-224` on the canonical two-cats image answers
**Egyptian cat 0.937441702420** - torch's probability, to all twelve digits. The image is resized with
PIL's own algorithm, reproduced to the byte, because that is what the model was trained on.
`openai/clip-vit-base-patch32` classifies among labels it has never seen - *a bee 76.7%, a flower
21.0%* - with logits within 8e-14 of torch's.

**Writes.** `gpt2` continues *"The lighthouse keeper opened the door and"* with exactly the words
transformers' `generate` produces, through a key/value cache, at about 22 tokens a second on a laptop
CPU.

**Draws.** Stable Diffusion 1.5 from its ONNX export, with the same seed and settings as diffusers'
own pipeline, gives the same picture - to the pixel on float32 exports, to 55 dB on float16 ones -
and image to image, inpainting, prompt emphasis and LoRA work the same way.

![A red lighthouse, drawn by HF.Net](docs/screenshots/diffusion-lighthouse.png)

**Answers with spans of your text, not with new text.** `dslim/bert-base-NER` on
*"Kang Fadhil founded Gravicode Studios in Bandung"* returns `PER Kang Fadhil` (98.8%),
`ORG Gravicode Studios` (99.5%) and `LOC Bandung` (99.7%) with exact character offsets, and
`distilbert-base-cased-distilled-squad` extracts its answer from the passage it was given. Sentence
pairs are exact: segment 0 is folded into the word embeddings and segment 1 is carried as a
difference applied per position.

## What it cannot do yet

Stated plainly, because a library that fails quietly is worse than one that says no:

- **Rotary-position decoders** — Llama, Mistral, GPT-NeoX — are **refused**, not half-loaded. GPT-2
  and its family run; those have a different block. Export them to ONNX and use GraviOptimum.
- **LoRA training runs on the CPU, bound by the linear kernel.** `PeftModel.Train` fits adapters and a
  sequence classification head exactly, which is checked against numerical gradients and against
  PEFT in Python. It suits hundreds of examples. For tens of thousands, train with PEFT in Python
  and serve the adapter here. Sequence classification, token classification and extractive question answering heads train;
  prefix tuning trains sequence classification.
- **Windowed and convolutional vision backbones** — Swin, ConvNeXt — are refused by name. ViT, DeiT
  and CLIP run.
- **Diffusion needs an ONNX export**, not the PyTorch weights, and exports built from ONNX Runtime's
  fused GPU kernels (`onnxruntime/sd-turbo`) need CUDA.
- **Uploads are capped at 10 MB.** Real weights need Git LFS, which GraviHub does not implement.

## How it compares to Python

Measured against the reference implementation on the same machine in the same session — full
method and caveats in **[docs/benchmarks.md](docs/benchmarks.md)**.

| | Python | HF.Net | |
|---|---:|---:|---|
| Tokenize 1,000 documents | 20.6 ms | **9.6 ms** | **2.15x faster** |
| Open a 420 MB checkpoint, list 206 tensors | 0.70 ms | 0.71 ms | level |
| Read one 30,522 × 768 tensor | 0.74 ms | 185 ms | 251x slower |
| bert-base forward pass, 1 document | 50.1 ms | 59.5 ms | 1.19x slower |
| bert-base, float32 linear layers | 50.1 ms | **45.0 ms** | **1.11x faster** |
| ViT-base forward pass, 1 image | 345 ms | 913 ms | 2.65x slower |
| **bert-base through ONNX Runtime, from .NET** | 50.1 ms | **28.9 ms** | **1.73x faster** |

**Tokenization is faster than the Rust `tokenizers` crate, and the ids are identical.** Reading a
tensor is slower because every value is widened to `double` — paid once at load time. Managed
inference is slower than torch in double precision and level with it in float32, and it exists so a
model can be **loaded, inspected and understood** in pure .NET. When you need throughput, export to ONNX and run it through
`GraviOptimum`, which is faster than torch.

**And they agree — to the last digit that means anything.** Against torch in float64, on the same
inputs, `bert-base-uncased`'s hidden states agree to about 1e-13 and ViT's top-five probabilities to
5.7e-16:

```
The capital of France is [MASK].

        torch (float64)       hf.net
  1     paris   0.4167877541  paris   0.4167877541
  2     lille   0.0714164028  lille   0.0714164028
  3     lyon    0.0633924121  lyon    0.0633924121
```

## HF Gallery

`samples/HFGallery` is a desktop application that runs fifteen HF.Net use cases against real models and
shows the answer next to the code that produced it. Nothing in it is mocked.

![HF Gallery — image classification](docs/screenshots/hfgallery-image.png)

![HF Gallery — named entities](docs/screenshots/hfgallery-entities.png)

Entities come back as spans of the original text, taken from the tokenizer's character offsets, so
the casing and the punctuation survive and an off-by-one is visible rather than plausible.

![HF Gallery — inside a checkpoint](docs/screenshots/hfgallery-checkpoint.png)

`bert-base-uncased` is 420 MB, and the treemap says where those bytes are: feed-forward 216 MB,
attention 108 MB, embeddings 91 MB. Producing it costs a header parse, not a 420 MB load.

![HF Gallery — embedding map](docs/screenshots/hfgallery-embedding-map.png)

Eight sentences from three topics, embedded and projected onto two principal components. The groups
separate without being told to.

```bash
dotnet run --project samples/HFGallery
dotnet run --project samples/HFGallery -- --list            # the catalog
dotnet run --project samples/HFGallery -- --run Named       # one case, headless
dotnet run --project samples/HFGallery -- --open Question   # open on a case and run it
```

The charts are drawn directly into a `DrawingContext` — no charting library — and the palettes were
searched in OKLCH and checked with a validator rather than chosen by eye. See
[docs/hf-gallery.md](docs/hf-gallery.md) for the other six cases and what the validator turned up.

## HFAppGen

`tools/HFAppGen` is an Avalonia IDE whose assistant — **Jack, the Code Bender** — builds HF.Net
applications from a prompt. It writes the files, runs the build, and fixes what the compiler says.
Supports OpenAI, Azure OpenAI, Claude, Gemini and Ollama through Semantic Kernel; everything is
configured in `app.config` and editable from the UI.

![HFAppGen](docs/screenshots/hfappgen-main.png)

The banded rule under the toolbar is the **offset rail**: eight spans, one per HF.Net library, each
as wide as that library's real share of the source.

![New project](docs/screenshots/hfappgen-new-project.png)

```bash
dotnet run --project tools/HFAppGen
dotnet run --project tools/HFAppGen -- --selftest   # one headless round trip
```

See [docs/HFAppGen.md](docs/HFAppGen.md).

## Repository layout

```
src/            one class library per Gravi* project
samples/        a Gravi*.Console app per library, plus HFGallery — fifteen use cases in an Avalonia window
tests/          Gravi*.Tests — 414 tests, no network required
benchmarks/     HFNet.Benchmarks (BenchmarkDotNet, every library) and comparison/ (against Python)
notebooks/      .NET Interactive notebooks: 01 getting started, 02 performance, 03 GPT-2 and CLIP,
                04 LoRA and prefix tuning, 05 Stable Diffusion
datasets/       titanic.csv, iris.csv, imdb_reviews.csv, finance_timeseries.csv
docs/           one page per library, plus id/ mirroring every page
tools/HFAppGen/ the Avalonia app generator
```

## Commands

```bash
dotnet build HF.Net.sln -c Release
dotnet test                                                   # every test
dotnet test tests/GraviHub.Tests                              # one project
dotnet test tests/GraviHub.Tests --filter "FullyQualifiedName~SafeTensors"
dotnet run --project samples/GraviHub.Console
dotnet run --project samples/HFGallery                        # the use case gallery
dotnet run --project tools/HFAppGen                           # the IDE
dotnet pack HF.Net.sln -c Release                             # -> artifacts/packages
dotnet format
```

## Requirements

- .NET 10 SDK
- An internet connection for anything that touches the Hub (the built-in datasets and every test
  work offline)

## Licence

MIT.
