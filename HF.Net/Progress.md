# Progress

Development tracking for HF.Net. `requirements.md` is the specification of record and
[PLAN.md](PLAN.md) holds the roadmap; this file says what exists today.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

---

## v0.1.0 — current

**26 projects, 212 tests passing, whole solution builds clean with no warnings.**

Verified against real Hugging Face models rather than fixtures: `bert-base-uncased`,
`distilbert-base-uncased-finetuned-sst-2-english`, `dslim/bert-base-NER`,
`distilbert-base-cased-distilled-squad`, `google/vit-base-patch16-224`, `prajjwal1/bert-tiny`, `gpt2`,
`hf-internal-testing/tiny-random-BertModel`.

### Libraries

| Library | State | Tests | Notes |
|---|---|---|---|
| GraviHub | **Complete** | 27 | Hub client, cache, safetensors, PyTorch pickle reader |
| GraviTokenizers | **Complete** | 29 | WordPiece, BPE, Unigram, `tokenizer.json`, offsets |
| GraviDatasets | **Complete** | 30 | Files, Hub datasets, splits, streaming |
| GraviTransformers | **Core complete** | 53 | BERT-family encoders and ViT; classification, fill-mask, named entities, question answering, embeddings, image classification |
| GraviPEFT | **Partial** | 15 | LoRA apply/merge/save/load; adapter training not implemented |
| GraviAccelerate | **Core complete** | 14 | Device selection, sharding, weighted averaging, measurement |
| GraviOptimum | **Core complete** | 22 | ONNX Runtime, provider choice, quantisation with measured error |
| GraviDiffusers | **Core complete** | 22 | DDPM/DDIM/Euler; SD pipeline needs an ONNX export to exercise |

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
- **`google/vit-base-patch16-224`** answers **bee 94.46%** on the Hub's own sample photograph and
  **Egyptian cat 93.81%** on the canonical two-cats image, against torch's 94.38% and 93.74%. Same
  ranking, five for five, within 0.08 of a percentage point; the residual is the resampler, not the
  model.
- **Question answering on `distilbert-base-cased-distilled-squad`** extracts *"safetensors and
  PyTorch checkpoints"* and *"Gravicode Studios"* from a passage about HF.Net. This is also the
  check on sentence pairs: segment 1 is carried as a difference and applied per position, so both
  segments are reproduced exactly rather than approximately.
- **DDIM with a perfect noise oracle recovers `x₀` to 1e-9**, which pins every coefficient in the
  reverse step.
- **HFAppGen end to end**: a prompt produced a project that referenced the real libraries, built
  clean, and ran — downloading DistilBERT and classifying three sentences correctly.
- **Measured against the Python reference** on the same machine in the same session. Tokenization is
  **1.66x faster than the Rust `tokenizers` crate** with byte-identical ids; fill-mask agrees to a
  tenth of a percentage point across the top five. See [docs/benchmarks.md](docs/benchmarks.md).

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

`samples/HFGallery` — an Avalonia application holding ten use cases, each running against a real
model and shown next to the code that produced it.

- Image classification, sentiment, fill-mask, named entities, question answering, semantic search,
  an embedding map, the tokenizer, a checkpoint's byte layout, and the diffusion noise schedules
- Charts drawn straight into a `DrawingContext`: bars, scatter, lines, treemap, plus a span view
  built from text inlines so wrapping and selection come from the text stack
- `--list`, `--run <case>` (headless), `--open <case>`, `--light`
- Palettes searched in OKLCH and checked with the dataviz validator, which is what established that
  **the eight-band library ramp fails as a categorical palette** — its adjacent pairs measure ΔE 6.8
  under normal vision against a floor of 15. Correct for the offset rail, where adjacency carries
  meaning; wrong the moment the colours have to say which thing this is. The set the gallery uses
  measures ΔE 9.4 under deuteranopia and ΔE 15.8 under normal vision, all pairs, both surfaces.

### Documentation

Complete and bilingual. Every page under `docs/` has a counterpart under `docs/id/`:

- `README`, `getting-started`, `benchmarks`, `HFAppGen`, `hf-gallery`, and one page per library
- Three screenshots of HFAppGen and ten of HF Gallery, captured from the real windows

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

- Per-library BenchmarkDotNet suites (the comparison benchmark is done; the per-library harnesses
  are thin)
- Notebooks under `notebooks/` beyond the two shipped as HFAppGen templates
- CLIP (its text tower is causal, which this encoder is not)
- LoRA adapter training (the foundation's tape omits attention biases)

---

## Log

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
