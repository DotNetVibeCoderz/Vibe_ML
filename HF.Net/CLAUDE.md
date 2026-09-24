# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

v0.2.0 is published: **26 projects, 250 tests passing**, the whole solution builds clean, and the
eight libraries are published on nuget.org as `Gravicode.HFNet.*`. `requirements.md` remains the
specification of record; [Progress.md](Progress.md) says what exists and [PLAN.md](PLAN.md) says
where it is going.

Target framework is **.NET 10**. The solution file is `HF.Net.sln` (classic format — `dotnet new sln`
defaults to `.slnx` on .NET 10, so pass `--format sln` if regenerating).

**This lives at `C:\experiment\VibeCoding\Vibe_ML\HF.Net`**, one project inside the Vibe_ML
monorepo alongside GravicodeScience. That placement is what puts the CI workflows at the repository
root rather than here — see below.

## Things that will bite you

- **The foundation's namespaces are `Gravicode.Science.*`, not `GraviNum`.** HF.Net's own are
  `Gravicode.HFNet.*`. Both define a `TransformerModel` and a `Dataset`, so files that touch both
  need a `using` alias — see `CheckpointLoader.cs` and `Accelerator.cs`.
- **`GraviRandom` has `Normal()`, not `NextGaussian()`.**
- **`CsvOptions.Delimiter` is a `string`, not a `char`.**
- **`bert-base-uncased` names its layer norms `gamma`/`beta`**, not `weight`/`bias` — a TensorFlow
  inheritance. Both spellings are handled in `CheckpointLoader` *and* in the masked-LM head; a
  loader that knows only the modern one fails on the most-downloaded model on the Hub.
- **Token type embeddings are folded into the word embeddings.** Exact for a single sequence. A pair
  also needs segment 1, which is carried as `LoadReport.SegmentDelta` — the *difference* between the
  two segment vectors — and added per position. Both halves are exact; `SupportsPairs` is true.
- **A token classifier and a sequence classifier both store `classifier.weight` at `[classes,
  hidden]`.** The token head must be claimed first or every NER checkpoint loads as a sentence
  classifier with several hundred nonsense classes.
- **ViT is pre-norm; BERT is post-norm.** Same parameter shapes, so each loads the other's weights
  in silence and returns confident nonsense. That is why `GraviTransformers.Vision` has its own
  block rather than reusing the foundation's. The test that catches it: zero every projection — a
  pre-norm block is then the identity, a post-norm block returns `LayerNorm(input)`.
- **Inference runs on `Kernels.cs`, not on the foundation's layers.** Both encoders, text and
  vision, go through the same `Linear` / `Attention` / `Norm` / `EncoderBlock`. The foundation's
  `MultiHeadAttention` is a naive indexer loop about 20x slower than the kernel. `NormOrder` is a
  required argument with no default, because pre-norm and post-norm load each other's weights in
  silence.
- **Weights are float32 and activations double.** That is lossless, because every checkpoint stores
  F32 or narrower, and it halves the bytes a short input streams. The float32 copy must come from
  the *stored* tensor. The MLM decoder used to be tied to `Encoder.TokenEmbeddings`, which has
  segment 0 folded in; that double sum is not a float32 value, and rounding it cost fill-mask five
  orders of magnitude of agreement with torch.
- **The text encoder's inference weights are a copy.** `TransformerModel.Encoder` is the model of
  record, but `Hidden`/`Forward`/`FillMask` run on a `CompiledEncoder` built from it. Anything that
  changes the encoder's parameters must call `WeightsChanged()`, or predictions silently stay the
  same. `PeftModel.Merge` does.
- **Weight matrices stay in the checkpoint's `(outputs, inputs)` order.** The linear kernel is
  4 input rows x 2 weight rows per pass, and it depends on both being contiguous. Transposing
  measured five times slower.
- **Hot loops need `[MethodImpl(AggressiveOptimization)]`.** A loop-heavy method called only a
  handful of times runs as tier-0 code, and one forward pass is exactly that: the linear layer
  measured **12x slower** before tiering caught up. Benchmark kernels with a warm-up of 30+ calls,
  or with `DOTNET_TieredCompilation=0`, or you are timing the wrong code.
- **Do not use `MathUtil.Erf` in a hot loop.** Its Maclaurin series takes up to 50 terms and loses
  about 5e-14 near x = 3. `Activation.Erf` combines a table stepped out from 0 with a 9-term
  Hermite/Taylor correction. It is 3x faster and within 5e-16 of a correctly rounded erf.
- **`BatchOptions.Default` must not be written `new()`.** It is a record struct, so the
  parameterless constructor zeroes the fields rather than running the primary constructor's
  defaults — the result means "no padding", and the first ragged batch throws about rectangularity.
- **safetensors offsets are relative to the data block, not the file.** Reading them as absolute
  gives tensors shifted by the header length and full of plausible numbers.
- **Read the declared header length before allocating.** The reader used to copy a fixed 256 MB
  prefix to parse a ~20 KB header, which made listing tensors take 429 ms against the reference's
  0.65 ms. It is 0.71 ms now. Do not reintroduce a fixed-size prefix read.
- **`SafeTensorsReader`'s constructor must dispose its mapping on failure**, or a malformed header
  leaves the file locked for the life of the process.
- **`JsonElement` is only valid while its `JsonDocument` lives.** `JsonLines` clones every value;
  storing them by reference threw `ObjectDisposedException` on every JSON Lines dataset.
- **Quantisation must not touch integer index buffers.** bfloat16 has eight mantissa bits, so
  `position_ids` value 511 becomes 512 and the reported error is 1.0 for reasons unrelated to the
  weights.
- **Decoder-only models are refused, not half-loaded.** Filling an encoder from a decoder's weights
  produces a model that runs and returns nonsense.
- **`hidden_act: "gelu"` is the exact erf GELU; the foundation's `Activations.Gelu` is the tanh
  approximation.** They differ by up to 4e-4, and on ViT that moved probabilities in the third
  decimal place, which is easy to mistake for resampler noise. On BERT hidden states it was 2.8e-2.
  Both encoders now resolve the activation from the config (`Activation.For`), and both match torch
  in float64: ViT to ten digits of a probability, BERT hidden states to about 1e-13. The
  foundation's own layers still use the tanh form, so do not route inference back through them.
- **Position interpolation must be torch's bicubic exactly**: `a = -0.75` (not `-0.5`), half-pixel
  source coordinate *not* clamped at zero, edge taps clamped. Each wrong choice moves values by
  hundredths and still gives confident answers.
- **The managed encoder is accurate first and fast second.** Its linear kernel runs at roughly the
  foundation's packed `MatMul` rate (about 13 GMAC/s here). It beats `MatMul` below about 200 rows
  and loses by 1.3x at 577 (ViT at 384 px). For throughput the answer is an ONNX export through
  GraviOptimum — see [docs/benchmarks.md](docs/benchmarks.md).
- **The GPU is not a win here.** Everything is double precision, which consumer and integrated GPUs
  run at a fraction of their single-precision rate; the foundation measured its ILGPU path 5–8x
  *slower* than the CPU.
- **Heredocs with large C# or Markdown content fail in this environment.** Use the Write tool for
  anything substantial.

## CI

**Workflows live at the *repository* root, not here.** HF.Net is a subdirectory of the Vibe_ML
monorepo and GitHub reads `.github/workflows/` only from the root, so the files are at
`Vibe_ML/.github/workflows/hf-net-ci.yml` and `hf-net-release.yml`, with `working-directory: HF.Net`
on every step and path filters on `HF.Net/**`.

This is worth knowing because it is easy to get wrong and silent when you do: GravicodeScience keeps
its own `ci.yml` and `release.yml` inside `GravicodeScience/.github/workflows/`, where GitHub never
reads them — so that project has had no CI running at all.

- `hf-net-ci.yml` — build and test on Ubuntu and Windows, validate the notebooks, check every
  `docs/` page has a `docs/id/` counterpart and every referenced screenshot exists, then pack and
  upload the packages as artifacts.
- `hf-net-release.yml` — fires on a `hf-net-v*` tag, takes the version from the tag so the two
  cannot disagree, runs the tests as a gate, and pushes **GraviHub first** because everything
  depends on it. Needs a `NUGET_API_KEY` repository secret.

Publishing is never done from CI on a push to main: a version on nuget.org can be unlisted but
never replaced.

### HFAppGen specifics

- **Anthropic has no official Semantic Kernel connector**; Claude is served by a hand-written
  `IChatCompletionService`. Google and Ollama connectors are alpha-only packages.
- **`gpt-5` / `o1` / `o3` / `o4` reject `max_tokens`** and require `max_completion_tokens`, and
  reject a non-default temperature. `AssistantService.UsesCompletionTokenLimit` routes around it.
- **Avalonia bindings must not cast in the path.**
  `{Binding $parent[Window].((vm:Shell)DataContext).X}` compiles and then throws at startup.
- **AvaloniaEdit's `.xshd` definitions are written for a white page.** `Services/SyntaxTheme.cs`
  remaps their named colours; it mutates the process-wide `HighlightingManager.Instance`, so it must
  be re-run on `ActualThemeVariantChanged`.
- **The AvaloniaEdit package is `Avalonia.AvaloniaEdit`, the assembly is `AvaloniaEdit`.**
- **The assistant will invent a NuGet package for HF.Net if it is not told otherwise**, hit NU1101,
  and then write a fake shim so the build goes green. `HFNetProjectReferences` and an explicit
  prohibition in the system prompt exist because of that.

## Architecture

Eight libraries mirroring the Python Hugging Face stack, built on Gravicode.Science.

| Project | Python analogue | Core responsibility |
|---|---|---|
| `GraviHub` | `huggingface_hub` | Hub transfer, cache, **safetensors** and **PyTorch pickle** readers |
| `GraviTokenizers` | `tokenizers` | WordPiece, BPE, Unigram, `tokenizer.json`, character offsets |
| `GraviDatasets` | `datasets` | Files, Hub datasets, splits, streaming, memory mapping |
| `GraviTransformers` | `transformers` | Pretrained BERT-family encoders, ViT vision encoders, and task heads |
| `GraviPEFT` | `peft` | LoRA adapters in the PEFT format |
| `GraviAccelerate` | `accelerate` | Device selection, sharding, measurement |
| `GraviOptimum` | `optimum` | ONNX Runtime, quantisation |
| `GraviDiffusers` | `diffusers` | DDPM/DDIM/Euler schedulers, Stable Diffusion over ONNX |

```
GraviHub ──┬── GraviTokenizers ──┐
           ├── GraviDatasets     ├── GraviTransformers ──┬── GraviPEFT
           └── GraviOptimum ─────┘                       └── GraviDiffusers
                GraviAccelerate
```

**The foundation is consumed as NuGet packages, not cross-repository ProjectReferences.** A
ProjectReference into a sibling checkout packs into a .nupkg with no dependency recorded at all.

Adding a library means adding all five artifacts (src, sample, tests, benchmark, docs) plus
`docs/<Name>.md` **and** `docs/id/<Name>.md` — the spec treats them as one deliverable.

`samples/HFGallery` is an Avalonia app holding one case per capability, each running against a real
model. A case is a `GalleryCase` subclass registered in `Catalog.All` that returns a `CaseResult`;
whichever fields it fills in get drawn. `--run <case>` runs one headless, which is the quickest way
to check a model-facing change end to end without opening a window. Its charts are drawn by hand in
`Controls/Charts.cs` — there is no charting library, and `Palette.Resource` must pass
`ActualThemeVariant` or every token defined under `ThemeDictionaries` silently resolves to the
fallback grey.

## Testing conventions

Pin against something independently known — the reference implementation's own output, a
closed-form answer, or a naive version written inside the test. Examples already in the suite:
`bert-base-uncased` tokenizing to ids `101 7592 ... 102`; DDIM with a perfect noise oracle
recovering `x₀` to 1e-9; bf16 bit patterns computed by hand from the format definition.

A test that only checks HF.Net against itself passes just as happily when both sides are wrong.

Stochastic results use explicit tolerances (`Assert.True(Math.Abs(a - b) < tol, message)`) rather
than `Assert.Equal(a, b, decimals)`, which rounds and fails spuriously.

## Commands

```powershell
dotnet build HF.Net.sln -c Release
dotnet test                                                     # all 250
dotnet test tests/GraviHub.Tests
dotnet test tests/GraviHub.Tests --filter "FullyQualifiedName~SafeTensors"
dotnet run --project samples/GraviTransformers.Console -- bert-base-uncased
dotnet run --project samples/HFGallery                          # the use case gallery
dotnet run --project samples/HFGallery -- --run Named           # one case, headless
dotnet run --project tools/HFAppGen                             # the IDE
dotnet run --project tools/HFAppGen -- --selftest               # headless LLM check
dotnet pack HF.Net.sln -c Release                               # -> artifacts/packages
```

Comparison benchmark against Python:

```bash
cd benchmarks/comparison
python python/bench.py --out python/python.json
dotnet run -c Release --project HFNet.Comparison -- dotnet.json
python report.py
```

## Credentials

All outside the repo and gitignored. **`HFToken.txt` is not a bare token** — it reads
`Hugging-Face Access Token: hf_...`, so extract with `grep -oE 'hf_[A-Za-z0-9]+'`. Reading the whole
file into `HF_TOKEN` yields an anonymous client that still works on public repos, which makes the
mistake easy to miss.

- `C:\Users\mifma\Documents\CodeSandbox\HFToken.txt` — Hugging Face
- `C:\Users\mifma\Documents\CodeSandbox\testkey.txt` — LLM keys for HFAppGen
- `C:\Users\mifma\Documents\CodeSandbox\PackageCredentials.txt` — NuGet API key

## Conventions from the spec

- **Bilingual docs.** Every page under `docs/` has a counterpart under `docs/id/`. A page in one
  language only is an unfinished page.
- **Attribution.** Docs and the application carry: "Dibuat oleh Gravicode Studios dipimpin oleh
  Kang Fadhil".
- **API shape follows the Python original but stays idiomatic C#** — PascalCase, named arguments
  where Python uses keywords, and the worked examples in `requirements.md`
  (`TransformerModel.Load("bert-base")`, `Dataset.Load("titanic")`, `PEFT.ApplyLoRA(model)`) are
  the signatures, not suggestions.
- **Say no clearly.** Every refusal names what is unsupported and what to do instead.
