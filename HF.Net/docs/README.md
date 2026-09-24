# HF.Net documentation

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

**Bahasa Indonesia:** [docs/id/](id/)

## Start here

- [Getting started](getting-started.md) — install, first model, first prediction
- [HF Gallery](hf-gallery.md) — nine use cases running against real models, in one window
- [HFAppGen](HFAppGen.md) — the IDE that writes HF.Net applications for you
- [Benchmarks](benchmarks.md) — HF.Net measured against the Python reference
- Notebooks: [01 getting started](../notebooks/01-hugging-face-from-dotnet.ipynb) · [02 performance](../notebooks/02-where-the-time-goes.ipynb)

## The libraries

| Page | Library | Mirrors |
|---|---|---|
| [GraviHub](GraviHub.md) | Hub client, safetensors, PyTorch checkpoints | `huggingface_hub` |
| [GraviTokenizers](GraviTokenizers.md) | WordPiece, BPE, Unigram, `tokenizer.json` | `tokenizers` |
| [GraviDatasets](GraviDatasets.md) | Files, Hub datasets, splits, streaming | `datasets` |
| [GraviTransformers](GraviTransformers.md) | Pretrained encoders and task heads | `transformers` |
| [GraviPEFT](GraviPEFT.md) | LoRA adapters | `peft` |
| [GraviAccelerate](GraviAccelerate.md) | Devices, sharding, measurement | `accelerate` |
| [GraviOptimum](GraviOptimum.md) | ONNX Runtime, quantisation | `optimum` |
| [GraviDiffusers](GraviDiffusers.md) | Schedulers, Stable Diffusion | `diffusers` |

## Conventions used throughout

**Everything is `double`.** `NdArray` holds `double`, so a checkpoint stored as F16 or F32 is
widened on read. This costs memory and buys uniformity with the rest of the Gravicode stack; when
throughput matters, [GraviOptimum](GraviOptimum.md) is the answer, not a different array type.

**Refusals are explicit.** Where HF.Net cannot do something it says so and names the reason — an
unsupported architecture, a missing file format, an operation that would need a gradient it cannot
take. It does not half-load a model and let you discover the problem in the output.

**Offsets are real.** Tokenizer offsets index the original, untouched string, so a span can always
be returned as a substring rather than as token indices.
