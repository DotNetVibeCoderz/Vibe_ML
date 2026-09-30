# Progress

Development tracking for HF.Net. `requirements.md` is the specification of record and
[PLAN.md](PLAN.md) holds the roadmap; this file says what exists today.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

---

## v0.7.0 — current

**27 projects, 440 tests passing, whole solution builds clean with no warnings.**

Verified against real Hugging Face models rather than fixtures: `bert-base-uncased`,
`distilbert-base-uncased-finetuned-sst-2-english`, `dslim/bert-base-NER`,
`distilbert-base-cased-distilled-squad`, `google/vit-base-patch16-224`, `prajjwal1/bert-tiny`, `gpt2`,
`openai/clip-vit-base-patch32`, `bert-large-uncased`, `nmkd/stable-diffusion-1.5-onnx-fp16`,
`hf-internal-testing/tiny-random-BertModel`.

### Libraries

| Library | State | Tests | Notes |
|---|---|---|---|
| GraviHub | **Complete** | 27 | Hub client, cache, safetensors, PyTorch pickle reader |
| GraviTokenizers | **Complete** | 57 | WordPiece, BPE, Unigram, CLIP, `tokenizer.json`, offsets; matches Python on BERT, GPT-2, CLIP and XLM-R |
| GraviDatasets | **Complete** | 30 | Files, Hub datasets, splits, streaming |
| GraviTransformers | **Complete** | 147 | BERT-family encoders, GPT-2/Llama/Mistral/Qwen/NeoX decoders with a KV cache, ViT, CLIP; streaming for models larger than memory; opt-in float32 |
| GraviPEFT | **Complete** | 69 | LoRA and prefix tuning, trained here and loaded in Python and the reverse |
| GraviAccelerate | **Core complete** | 14 | Device selection, sharding, weighted averaging, measurement |
| GraviOptimum | **Complete** | 25 | ONNX Runtime, float16, weight overrides, model-file reading, quantisation with measured error |
| GraviDiffusers | **Complete** | 71 | Schedulers from the repo's config; text to image, img2img, inpainting, emphasis, LoRA, pixel-identical to diffusers |

### Verified behaviour

These are the checks that establish the stack is actually correct, not merely running.

- **Tokenization matches the reference implementation exactly.** `bert-base-uncased` on
  *"Hello, world! Tokenizers are unbelievable."* gives ids
  `101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102`; `gpt2` on *"Hello world"* gives
  `15496 995`. Per-subword offsets recover `Token` / `izer` / `s` from the original casing.
- **Fill-mask on `bert-base-uncased`** answers *"The capital of France is [MASK]."* with
  **paris 41.5%**, then lille, lyon, marseille — all French places. This is the sharpest available
  check: a transposed weight or a shifted position embedding still produces plausible vectors but
  not this answer.
- **Classification on DistilBERT SST-2** gives POSITIVE 99.99% / NEGATIVE 99.98% on the canonical
  pair, matching the reference model.
- **Named entities on `dslim/bert-base-NER`** tag *"Kang Fadhil founded Gravicode Studios in
  Bandung… a Microsoft event"* as `PER Kang Fadhil` 98.8%, `ORG Gravicode Studios` 99.5%,
  `LOC Bandung` 99.7%, `ORG Microsoft` 99.9%, each with exact character offsets into the input.
- **`google/vit-base-patch16-224`** answers **bee 94.48%** on the Hub's own sample photograph,
  against torch's 94.38%, with the same top-five ranking. **Given identical pixels, the top-five
  probabilities match torch to ten decimal places** at 160, 224 and 384 px, so the remaining
  residual on a photograph is the image resampler alone. Until the GELU fix below, part of it was
  the model.
- **Question answering on `distilbert-base-cased-distilled-squad`** extracts *"safetensors and
  PyTorch checkpoints"* and *"Gravicode Studios"* from a passage about HF.Net. This is also the
  check on sentence pairs: segment 1 is carried as a difference and applied per position, so both
  segments are reproduced exactly rather than approximately.
- **DDIM with a perfect noise oracle recovers `x₀` to 1e-9**, which pins every coefficient in the
  reverse step.
- **HFAppGen end to end**: a prompt produced a project that referenced the real libraries, built
  clean, and ran — downloading DistilBERT and classifying three sentences correctly.
- **`bert-base-uncased` agrees with torch in float64 to about 1e-13** on hidden states, for a single
  sentence and for a pair (segment 1). Fill-mask probabilities agree to ten decimal places.
- **LoRA training is exact.** Every adapter gradient agrees with central differences to 1e-6 on a
  two-layer model with every bias and norm perturbed, all six projections adapted, with and without
  dropout. A deliberately wrong backward pass fails by 2x. On `bert-base-uncased`, 32 sentences
  that differ mostly by a negation train from loss 0.70 to 0.003 in about a minute. Predictions
  from the adapters in the loop and from the merged weights agree to 1e-11. **An adapter trained
  here and loaded into PEFT 0.21 in Python** gives the same hidden states as HF.Net's merged model
  to 7e-7, the float32 rounding of the merged weights, while the adapter moves them by up to 4.
- **A classifier trained here predicts the same thing in Python.** On BERT the head is
  `BertForSequenceClassification`'s own (pretrained pooler, trained `classifier`) and is saved as
  PEFT saves a `SEQ_CLS` head. Loaded into `AutoModelForSequenceClassification` with PEFT 0.21,
  its class probabilities match HF.Net's to 5e-9, the F32 the file stores, with no warning.
  Reloaded here with `PEFT.LoadAdapter`, it matches to the same 5e-9.
- **Named entities and question answering train, and agree with Python.** A token classifier trained
  on 16 tagged sentences gives Transformers' own pipeline the same spans and labels as
  `FindEntities`, scores to 5e-11. A QA head trained on 24 SQuAD-style examples gives the same
  answers as `AutoModelForQuestionAnswering`, scores to 3e-10. Both are right on names and places
  they were never shown.
- **Stable Diffusion reproduces diffusers.** On tiny exported models the pipeline gives diffusers'
  own ONNX pipelines' images to the pixel - text to image with DDIM and with Euler, image to image,
  and inpainting with a 9-channel UNet - with the NumPy-seeded noise reproduced bit for bit. A LoRA
  merged into the ONNX weights gives the image of the model it was fused into in PyTorch, bar three
  values in 12,288 off by one level. On `nmkd/stable-diffusion-1.5-onnx-fp16`, a 512x512 image at 25
  steps agrees with diffusers to 54.7 dB PSNR.
- **GPT-2 generates what transformers generates.** `gpt2`'s greedy continuation is word for word
  transformers' `generate`, and on a random checkpoint the logits agree with torch in float64 to 1e-11.
- **CLIP and ViT are exact from an image file.** With PIL's resize reproduced to the byte,
  `google/vit-base-patch16-224` gives torch's probabilities to twelve digits from a PNG, and
  `openai/clip-vit-base-patch32`'s logits agree to 8e-14.
- **Prefix tuning round-trips with PEFT.** An adapter PEFT 0.21 saved gives its logits here to 1e-10;
  one trained here gives the same logits in Python to 1e-9. The prefix gradient agrees with
  numerical differentiation to 1e-6.
- **`bert-large-uncased` streams in 0.45 GB** of private memory against 5.9 GB loaded whole, with the
  same embedding to the last bit.
- **Measured against the Python reference** on the same machine in the same session. Tokenization is
  **2.25x faster than the Rust `tokenizers` crate** with byte-identical ids. bert-base takes 48 ms
  managed against torch's 38 ms, and **31 ms through ONNX Runtime from .NET, faster than torch**.
  ViT-base takes 948 ms against 214 ms. See [docs/benchmarks.md](docs/benchmarks.md).

### Found and fixed during development

Each of these was caught by a test or by a live run, not by reading the code.

- The safetensors reader allocated and copied a fixed **256 MB prefix** to parse a header of about
  twenty kilobytes, making "open and list tensors" take 429 ms against the reference's 0.65 ms.
  Reading the declared length first and then exactly that many bytes brought it to **0.71 ms — a
  600x improvement**. Only the comparison against Python made it visible; on its own it looked fast.
- `bert-base-uncased` stores its layer norms as **`gamma`/`beta`**, not `weight`/`bias` — a
  TensorFlow inheritance. The loader now accepts both spellings; without it, the most-downloaded
  model on the Hub failed to load.
- `SafeTensorsReader` **leaked its memory mapping** when the header was malformed, leaving the file
  locked for the life of the process.
- `JsonLines` stored `JsonElement` values **past the lifetime of their `JsonDocument`**, so every
  JSON Lines dataset threw `ObjectDisposedException`.
- `BatchOptions.Default` was written `new()`, which on a record struct **zeroes the fields** rather
  than running the primary constructor's defaults — so the default meant "no padding".
- Quantising a checkpoint **corrupted its integer index buffers**: bfloat16 has eight mantissa bits,
  so `position_ids` value 511 became 512. Integer tensors are now kept at full precision.
- The vision encoder ran the **tanh approximation of GELU** where ViT's config says `gelu`, which
  transformers reads as the exact erf form. The top five still ranked correctly and agreed to the
  third decimal place, so it passed for resampler noise. It came out only when both sides were fed
  identical pixels. The text encoder had the same problem, and there it left bert-base's hidden
  states **2.8e-2** away from torch. Both encoders now follow `hidden_act`.
- **The foundation's attention was a sequential indexer loop**, about 20x slower than a vectorised
  one, and it was two thirds of a ViT forward pass. Text and vision inference now run on HF.Net's
  own kernels. bert-base went from 300–600 ms to 111 ms and ViT-base from about 12 s to 2 s.
- **A loop-heavy method called a handful of times runs as tier-0 code.** The linear kernel measured
  12x slower before the JIT caught up, and one forward pass is exactly a handful of calls. The hot
  kernels are now `AggressiveOptimization`.
- **The tied MLM decoder was rounded.** It was taken from `Encoder.TokenEmbeddings`, which has
  segment 0 folded in, and that double sum is not a float32 value. Fill-mask agreed with torch to
  5e-9 instead of ten digits until it read the stored word embeddings.
- **The benchmark's ONNX row was measured on a tiny test model** and so could not be compared with
  anything. It now exports bert-base itself, and the production path turns out to beat torch.
- **`ApplyLoRA` named its adapters `layer.0.query`.** Saved, they were valid safetensors in the
  PEFT layout, and PEFT in Python placed none of them. It matches tensors to modules by name and
  skips, without an error, any it cannot place. Adapters now take the checkpoint's own module paths
  (`bert.encoder.layer.0.attention.self.query`), and the round trip was checked in Python.
- **PEFT 0.21 drops `token_type_ids` in `PeftModelForQuestionAnswering.forward`.** Found when
  HF.Net's QA predictions disagreed with Python's by up to 0.6 in a score, although the encoder and
  the head matched part for part. Through the wrapper every passage is segment 0. HF.Net follows
  Transformers' own `BertForQuestionAnswering`; in Python, `merge_and_unload()` gives the same thing.
- **Entities came back in pieces.** "Kartini" decoded as "Ka", "rti", "ni", because each
  continuation piece carried the `B-` its word's first piece was trained on. Entities are now decoded
  a word at a time, Transformers' `aggregation_strategy="first"`.
- **Adapter initialisation was seeded from `HashCode.Combine`**, which .NET randomises per process,
  so "the same configuration always initialises the same way" was false: two runs of the same
  training gave two loss curves. It is an FNV-1a hash now.
- **Stable Diffusion's prompts were tokenized as GPT-2's.** The CLIP vocabulary went through the GPT-2
  loader: plausible ids, and every prompt encoded wrongly. Six tokenizer differences from Python came
  out of the same investigation - `Split`'s `invert` and `behavior`, the normalizer running after the
  byte-level mapping, a missing model `type` read as WordPiece, BERT's CJK handling, SentencePiece's
  `Precompiled` normalizer being skipped, and Unigram not fusing unknown pieces.
- **The diffusion pipeline assumed a VAE factor of 8** and so returned a 16x16 image for a 64x64
  request on a model whose VAE downsamples by 2 - as diffusers' own ONNX pipelines still do. The
  factor now comes from the VAE's config.
- **The VAE encoder is not deterministic as exported**: an unseeded `RandomNormalLike` samples inside
  the graph, so image to image could not be reproduced from a seed, in Python either. The node's
  scale is set to zero in a copy beside the file, which gives the mean.
- **A protobuf length read with `Position += ReadVarint()`** took `Position` before the call advanced
  it, and the ONNX reader desynchronised with "wire type 6".
- **PEFT prefix tuning shifts BERT's positions.** BERT numbers positions from `past_key_values`'
  length, so with ten virtual tokens the first word is at position 10. Without that, logits were 3.4
  away from PEFT's.
- **ONNX Runtime refuses external data behind a symbolic link.** Python's Hub cache links snapshot
  files into `blobs/`, and a newer runtime resolves `weights.pb` there and calls it an escape from the
  model's folder. Such folders are now opened through hard links.
- **The GELU epilogue ran on one thread**: a quarter of a ViT block's time went on `erf` calls after
  the product had been spread across four cores. With that parallel and attention scoring four keys
  per pass, ViT-base went from 1,034 ms to 774 ms with the same result to nine digits.
- HFAppGen's assistant, asked to build a project, **invented a NuGet package** for HF.Net, hit
  NU1101, and wrote a fake shim to get a green build. It now has an `HFNetProjectReferences` tool
  and a system prompt that forbids shims outright.

### Application

**HFAppGen** (`tools/HFAppGen`) — Avalonia IDE, ported from ScienceAppGen and retargeted.

- Code explorer, editor with line numbers and syntax highlighting, chat panel, logs, status bar
- Menu and toolbar: New Project, Open, Close, Go To Line, Format, Build, Run, Deploy, Exit
- New Project offers Blank or one of **12 templates**, including two .NET Interactive notebooks
- Chat panel: resizable, hideable, image attachments, Ctrl+Enter, clear thread, model picker
- LLM providers: OpenAI, Azure OpenAI, Claude, Gemini, Ollama — all configured in `app.config`
- Kernel functions: project and file manipulation, build, `HFNetReference`, `HFNetExample`,
  `HFNetProjectReferences`, `SearchInternet` (Tavily), `ScrapeWebPage`, `MathCalculation`, date/time
- `--selftest` runs one headless round trip, so the LLM path is checkable from a terminal
- **Retheme**: new identity built on the *offset rail* — eight spans whose widths are each library's
  real share of the source. Amber for identity and action, cyan for measured quantities only.

### HF Gallery

`samples/HFGallery` — an Avalonia application holding fifteen use cases, each running against a real
model and shown next to the code that produced it.

- Image classification, CLIP zero-shot classification, GPT-2 story writing, sentiment, fill-mask, named entities, question answering, semantic search,
  an embedding map, LoRA training for a classifier, for names and for answers, the tokenizer, a checkpoint's byte layout, and the diffusion noise schedules
- Charts drawn straight into a `DrawingContext`: bars, scatter, lines, treemap, plus a span view
  built from text inlines so wrapping and selection come from the text stack
- `--list`, `--run <case>` (headless), `--open <case>`, `--light`, and `--capture <png>`, which
  renders the window itself. A screen grab needs the window in front, and Windows may refuse that
  to a process in the background, so the grab silently gets whatever else is on the desktop
- Palettes searched in OKLCH and checked with the dataviz validator, which is what established that
  **the eight-band library ramp fails as a categorical palette** — its adjacent pairs measure ΔE 6.8
  under normal vision against a floor of 15. Correct for the offset rail, where adjacency carries
  meaning; wrong the moment the colours have to say which thing this is. The set the gallery uses
  measures ΔE 9.4 under deuteranopia and ΔE 15.8 under normal vision, all pairs, both surfaces.

### Documentation

Complete and bilingual. Every page under `docs/` has a counterpart under `docs/id/`:

- `README`, `getting-started`, `benchmarks`, `HFAppGen`, `hf-gallery`, and one page per library
- Three screenshots of HFAppGen and fifteen of HF Gallery, captured from the real windows, and a
  Stable Diffusion image generated by HF.Net
- Five notebooks: getting started, performance, GPT-2 and CLIP, LoRA and prefix tuning, Stable Diffusion

### Continuous integration

At `Vibe_ML/.github/workflows/` — the repository root, because GitHub reads workflows only from
there and HF.Net is a subdirectory.

- **`hf-net-ci.yml`** — build and test on Ubuntu and Windows, validate the notebooks, check that
  every `docs/` page has a `docs/id/` counterpart and every referenced screenshot exists, then pack.
- **`hf-net-release.yml`** — tag-driven (`hf-net-v*`), version read from the tag, tests as the gate,
  GraviHub pushed first. Needs a `NUGET_API_KEY` secret.

> Found while setting this up: **GravicodeScience's own workflows have never run.** They sit in
> `GravicodeScience/.github/workflows/`, which GitHub does not read. Its `CLAUDE.md` even says they
> "have to be copied up after cloning"; nobody did.

### Not done yet

Carried into [PLAN.md](PLAN.md):

- Compiling the notebooks' C# in CI; today CI checks their structure only
- Holding a tied embedding table once rather than twice
- Checking TinyLlama and Llama 3.2 against transformers. SmolLM2-135M, Pythia-160m, Qwen2.5-0.5B and
  Qwen3-0.6B give the same prompt ids and the same 30 greedy tokens; the run exhausted the machine's
  memory before the last two and has to be redone one model per process

---

## Log

**2026-10-01** — Rotary-position decoders and the foundation's biased attention. `CausalLanguageModel`
runs Llama, Mistral, Qwen2, Qwen3 and GPT-NeoX beside GPT-2, each matching transformers in float64 to
1e-10 with token-for-token generation, on a growable KV cache with grouped-query attention and sliding
windows. The tokenizers behind them match Python on twelve families, encode and decode: SentencePiece
BPE with byte fallback, the Rust crate's decoder chain, added tokens matched in normalized text,
`Digits` that no longer deleted digits, and Unicode normalization from generated tables, since
`String.Normalize` is a no-op under InvariantGlobalization. GravicodeScience's tape gained attention
biases and the exact GELU, and its CI now runs from the repository root. 440 tests.

**2026-10-01** — Everything the v0.2-v0.5 plans left open. Stable Diffusion rebuilt to reproduce
diffusers' pipelines to the pixel, with image to image, inpainting, AUTOMATIC1111 emphasis and LoRA,
and checked on SD 1.5. CLIP and GPT-2, both on the causal attention the spec's GPT needed. PEFT prefix
tuning, round-tripping with Python. A streaming encoder that runs `bert-large-uncased` in 0.45 GB.
PIL's resize reproduced to the byte, making ViT and CLIP exact from an image file. An opt-in float32
GEMM, bit-identical to an in-order float32 sum, and a parallel GELU. Tokenizers now match Python on
BERT, GPT-2, CLIP and XLM-R. A sample for every library, a BenchmarkDotNet suite, three new
notebooks. 414 tests.

**2026-09-30** — v0.6.0, a register-blocked GEMM. Weights move into float32 panels of twelve outputs, and a
4x12 kernel holds its block of the result in AVX registers, widening each panel slice once per call.
The linear layers went from about 12 GMAC/s to 35-54 at 80 rows. bert-base went from 111 ms to 48 ms
(torch 38 ms), ViT-base from 1,964 ms to 948 ms, and a LoRA training step 2.2x faster, all with
unchanged results. The first attempt to use the dot-product kernel better found that `Span`
slicing in its inner loop cost 1.7x on its own. 308 tests.

**2026-09-28** — v0.5.0; v0.3 complete. Training steps pack each micro-batch end to end: the linear layers
see all its rows at once, attention stays inside each example, and nothing is padded. That was
1.25x on bert-base, and it leaves training bound by the linear kernel. Question answering splits a
long passage into overlapping windows (`DocStride`) in training and in `Answer`. 304 tests.

**2026-09-28** — v0.4.0. Named entities and question answering train. `TrainTokenClassifier` takes words
and one tag per word, trains the first piece of each word, and `FindEntities` decodes a word at a
time. `TrainQuestionAnswering` takes SQuAD-style examples and trains `qa_outputs` on sentence pairs,
which the training encoder now supports. Both heads save in PEFT's layout (`TOKEN_CLS`,
`QUESTION_ANS`) and agree with Python. Checking the QA one found that PEFT's QA wrapper drops the
segment ids. Two more gallery cases. 297 tests.

**2026-09-28** — Trained heads save and load. On BERT, `Train` now fits
`BertForSequenceClassification`'s own head (the frozen pretrained pooler, then `classifier`), and
`SaveAdapter` writes it the way PEFT writes `SEQ_CLS`. Python loads it and predicts the same thing to
5e-9. `PEFT.LoadAdapter` takes a local directory, and it restores the classifier, including one
trained in Python. Label names go in `hfnet_head.json`, because PEFT answers an unknown
`adapter_config.json` key with advice to upgrade. 283 tests.

**2026-09-25** — v0.3.0, LoRA training. `PeftModel.Train` fits adapters and a mean-pooled classification
head with AdamW, the Hugging Face linear schedule and gradient clipping. It uses a hand-written
backward pass over the inference kernels, because the foundation's autodiff encoder has no Q/K/V
biases. The gradients are checked against central differences to 1e-6. Adapters now carry the
checkpoint's module paths, so an adapter trained here loads in Python PEFT and agrees with it to
7e-7. HF Gallery gained an eleventh case and `--capture`. 277 tests.

**2026-09-24** — Performance. Text and vision inference moved onto shared kernels: a vectorised
attention, a 4x2 float32-weight linear kernel, a table-plus-Taylor erf, and `AggressiveOptimization`
on the hot loops. bert-base went from 300–600 ms to 111 ms (torch: 36 ms), and ViT-base from 12 s
to 2 s. The text encoder picked up the exact GELU on the way and now matches torch in float64 to
1e-13. The comparison benchmark gained a real bert-base ONNX row: 23.8 ms from .NET, faster than
torch. 250 tests.

**2026-09-24** — Vision models at any resolution. Position embeddings are interpolated with torch's
own bicubic, pinned to 1e-12. Checking against torch at 384 px found that the vision block used
the tanh GELU instead of the exact one; with that fixed, ViT matches torch to ten decimal places at
160, 224 and 384 px. 229 tests.

**2026-09-24** — Vision. ViT and DeiT load and classify; a pre-norm encoder block written here
because the foundation's is post-norm, and the two accept each other's parameters in silence. The
feed-forward pair and the patch projection keep the checkpoint's own memory layout and are
vectorised. 212 tests.

**2026-09-24** — Named entities and question answering, with sentence pairs made exact via a
per-position segment delta. HF Gallery added: nine use cases in an Avalonia window, custom-drawn
charts, a validated palette. 191 tests.

**2026-09-22** — Comparison benchmark against Python, which found a 600x bug in the safetensors
header read. Bilingual docs completed, screenshots captured, solution moved into the Vibe_ML
monorepo, packages published to nuget.org.

**2026-09-22** — v0.1.0. Eight libraries, 178 tests, HFAppGen ported and retargeted, verified end to
end against live models. Solution builds clean.

**2026-09-21** — Repository initialised from `requirements.md`. Foundation identified:
GravicodeScience, consumed as NuGet packages rather than cross-repository project references.
