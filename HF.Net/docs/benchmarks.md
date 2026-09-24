# Benchmarks: HF.Net against the Python reference

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

The point of this page is not to win. It is to know where HF.Net stands against the implementation
everyone else uses, so that the docs can tell you which tool to reach for and be right.

Both halves measure **the same inputs on the same machine in the same session**, and both report the
**best** of several timed runs after a warm-up. On a laptop that throttles, the mean measures the
thermal state of the room.

| | |
|---|---|
| Machine | Windows 11 (10.0.26200), 8 logical cores |
| Python | 3.12.10 — transformers 5.17.0, tokenizers 0.23.2, torch 2.14.0+cpu (4 threads) |
| .NET | 10.0.12, Release |

Reproduce with the commands at the bottom. Raw output lives in
[`benchmarks/comparison/report.md`](../benchmarks/comparison/report.md).

---

## Tokenization — HF.Net is faster

`bert-base-uncased`. The Python side is the Rust `tokenizers` crate behind a thin binding, which is
the fastest thing Python has; GraviTokenizers is managed C#.

| Measure | Python (Rust) | HF.Net (C#) | |
|---|---:|---:|---|
| One document | 0.03 ms | 0.01 ms | **2.8x faster** |
| 1,000 documents | 18.13 ms | 10.94 ms | **1.66x faster** |

**55,144 docs/s** against **91,423 docs/s**.

The result is less surprising than it looks. Both implementations do the same greedy longest-match
walk; the Rust crate pays a Python-boundary crossing per call, and the managed version caches
segmentation per word — real text repeats words heavily, and the merge loop is the expensive part.

### And the ids are identical

The speed only matters if the output is the same. It is:

```
input   Hello, world! Tokenizers are unbelievable.
python  101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102
hf.net  101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102
```

## Reading safetensors — level

`model.safetensors` for `bert-base-uncased`: 420 MB, 206 tensors.

| Measure | Python | HF.Net | |
|---|---:|---:|---|
| Open and list every tensor | 0.65 ms | 0.71 ms | 1.09x slower |
| Read one 30,522 × 768 tensor | 1.14 ms | 196.96 ms | 173x slower |

Listing is level, because both read only the header.

**Reading a tensor is not, and the reason is structural rather than fixable.** PyTorch hands back a
zero-copy view over the F32 bytes as they lie on disk. HF.Net widens every value to `double`,
because that is what `NdArray` holds — 23.4 million values, 187 MB materialised. The cost is real
and it is the price of uniformity with the rest of the Gravicode stack.

> This table is also where a **600x performance bug** was found. The reader was allocating and
> copying a fixed 256 MB prefix to parse a header of about twenty kilobytes, which made listing
> tensors take 429 ms. Reading the declared header length first and then exactly that many bytes
> brought it to 0.71 ms. Nothing but a comparison against a reference implementation would have
> made that visible — it looked fast enough on its own.

## Encoder inference — Python wins, by a lot

One forward pass over 12 tokens.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| bert-base-uncased, 1 document | 50.8 ms | 300–600 ms | **6–12x slower** |
| bert-base-uncased, 8 documents | 261.6 ms | 1,736 ms | 6.6x slower |
| bert-tiny, 1 document | 1.5 ms | 2.1 ms | 1.4x slower |

The bert-base single-document figure moved between 304 ms and 598 ms across two runs an hour apart
on the same build — thermal throttling, and a good illustration of why a single number quoted
without its conditions is worth very little. The range is given rather than a false precision.

**This gap is expected and not a defect.** The managed encoder is `double` end to end and written to
be read; torch dispatches to hand-tuned single-precision kernels with fused attention.
GraviTransformers exists so a model can be **loaded, inspected and understood** in pure .NET.

Note that the gap narrows sharply on the small model — 1.4x on bert-tiny against 6–12x on bert-base.
Most of what torch wins is in the large matrix multiplies, not in the framework.

### Vision

One image at 224x224, which is 197 positions rather than 12.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| vit-base-patch16-224, 1 image | 441 ms | 12.0 s | 26x slower |

Two thirds of the managed figure is attention, which is the foundation's. The feed-forward pair and
the patch projection are HF.Net's, and both were made faster by changing their memory layout rather
than their arithmetic: they keep their weights in the checkpoint's own `(outputs, inputs)` order so
each dot product walks contiguous memory and can be vectorised.

That direction is counter-intuitive enough to be worth stating plainly. Transposing them into the
`(inputs, outputs)` order the obvious loop wants, and parallelising across rows, measured **five
times slower than the sequential version it replaced** — at 3072 columns every step of the inner
loop is a fresh cache line, and a strided operand cannot be loaded into a vector register at all.

### The production path

The same work through ONNX Runtime, via GraviOptimum, on a small test model:

**0.67 ms** best, 0.85 ms median.

Not comparable with the table above — a different model — but it shows the shape of the answer: the
same class of single-precision kernels torch uses, reached from .NET. **When you need throughput,
export to ONNX.** See [GraviOptimum](GraviOptimum.md).

## Do they agree? — yes

Speed is the easy half. This is the half that decides whether any of it is usable.

**`The capital of France is [MASK].`**

| Rank | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | paris | 41.68% | paris | 41.53% |
| 2 | lille | 7.14% | lille | 7.16% |
| 3 | lyon | 6.34% | lyon | 6.31% |
| 4 | marseille | 4.44% | marseille | 4.46% |
| 5 | tours | 3.03% | tours | 3.02% |

**`He was a [MASK] player in the national team.`**

| Rank | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | regular | 54.89% | regular | 55.00% |
| 2 | key | 19.47% | key | 19.39% |
| 3 | former | 5.82% | former | 5.83% |
| 4 | capped | 2.16% | capped | 2.14% |
| 5 | prominent | 1.36% | prominent | 1.36% |

**`google/vit-base-patch16-224`**, on the Hub's own sample photograph and on the canonical two-cats
image. The small residual is the resampler — PIL's bilinear against ImageSharp's — not the model.

| Image | Python | | HF.Net | |
|---|---|---:|---|---:|
| bee.jpg | bee | 94.38% | bee | 94.46% |
| | pot, flowerpot | 1.36% | pot, flowerpot | 1.32% |
| cats.jpg | Egyptian cat | 93.74% | Egyptian cat | 93.81% |
| | tabby, tabby cat | 3.84% | tabby, tabby cat | 3.80% |

Same order, same five candidates, probabilities agreeing to about a tenth of a percentage point.
The residual difference is `double` against `float` arithmetic, which is HF.Net being *more*
precise rather than less.

## Two things Python could not open

Both turned up while writing this benchmark, and both are cases HF.Net handles without being asked:

- **`prajjwal1/bert-tiny` has no `tokenizer.json`.** `transformers` 5.x refuses to build a fast
  tokenizer from `vocab.txt` without `sentencepiece` installed. HF.Net falls back to `vocab.txt`
  on its own.
- **`prajjwal1/bert-tiny`'s `config.json` has no `model_type` key.** `AutoModel` refuses it
  outright; the architecture has to be named explicitly. HF.Net defaults a missing `model_type`
  to `bert` and loads it.

Neither is a performance result. Both are the kind of thing that decides whether a library opens the
model you actually have.

## What to take from this

| If you are | Use |
|---|---|
| Tokenizing large volumes of text | **HF.Net** — faster, identical output |
| Loading and inspecting a checkpoint | **HF.Net** — level on headers, and it reads formats Python's fast path refuses |
| Running an encoder in production | **ONNX through GraviOptimum**, not the managed path |
| Running an encoder to understand it | **HF.Net managed** — slower, and readable |
| Training | **Python.** HF.Net does not backpropagate into a pretrained encoder; see [PLAN.md](../PLAN.md) |

## Reproducing this

```bash
cd benchmarks/comparison
pip install tokenizers transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu

python python/bench.py --out python/python.json
dotnet run -c Release --project HFNet.Comparison -- dotnet.json
python report.py
```

Run both halves in the same session. Do not compare numbers taken on different days — on a
throttling laptop that difference exceeds most of what is being measured.
