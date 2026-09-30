# HF.Net documentation

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

**Bahasa Indonesia:** [docs/id/](id/)

## Start here

- [Getting started](getting-started.md) — install, first model, first prediction
- [HF Gallery](hf-gallery.md) — fifteen use cases running against real models, in one window
- [HFAppGen](HFAppGen.md) — the IDE that writes HF.Net applications for you
- [Benchmarks](benchmarks.md) — HF.Net measured against the Python reference
- Notebooks: [01 getting started](../notebooks/01-hugging-face-from-dotnet.ipynb) · [02 performance](../notebooks/02-where-the-time-goes.ipynb) · [03 GPT-2 and CLIP](../notebooks/03-generate-and-see.ipynb) · [04 LoRA and prefix tuning](../notebooks/04-train-an-adapter.ipynb) · [05 Stable Diffusion](../notebooks/05-stable-diffusion.ipynb)

## The libraries

| Page | Library | Mirrors |
|---|---|---|
| [GraviHub](GraviHub.md) | Hub client, safetensors, PyTorch checkpoints | `huggingface_hub` |
| [GraviTokenizers](GraviTokenizers.md) | WordPiece, BPE, Unigram, `tokenizer.json` | `tokenizers` |
| [GraviDatasets](GraviDatasets.md) | Files, Hub datasets, splits, streaming | `datasets` |
| [GraviTransformers](GraviTransformers.md) | Encoders and task heads; GPT-2, Llama, Mistral, Qwen, Pythia; ViT, CLIP | `transformers` |
| [GraviPEFT](GraviPEFT.md) | LoRA and prefix tuning | `peft` |
| [GraviAccelerate](GraviAccelerate.md) | Devices, sharding, measurement | `accelerate` |
| [GraviOptimum](GraviOptimum.md) | ONNX Runtime, quantisation | `optimum` |
| [GraviDiffusers](GraviDiffusers.md) | Schedulers, Stable Diffusion, img2img, inpainting, LoRA | `diffusers` |

## Conventions used throughout

**Everything is `double`.** `NdArray` holds `double`, so a checkpoint stored as F16 or F32 is
widened on read. This costs memory and buys uniformity with the rest of the Gravicode stack; when
throughput matters, [GraviOptimum](GraviOptimum.md) is the answer, not a different array type.
Inside the inference kernels weights are float32, and `ComputeOptions.LinearLayers = Precision.Single`
runs the linear layers in float32 too, for about twice their speed.

**Refusals are explicit.** Where HF.Net cannot do something it says so and names the reason — an
unsupported architecture, a missing file format, an operation that would need a gradient it cannot
take. It does not half-load a model and let you discover the problem in the output.

**Offsets are real.** Tokenizer offsets index the original, untouched string, so a span can always
be returned as a substring rather than as token indices.
