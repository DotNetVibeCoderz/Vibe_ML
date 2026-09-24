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
| One document | 0.03 ms | 0.01 ms | **2.6x faster** |
| 1,000 documents | 15.52 ms | 8.70 ms | **1.78x faster** |

**64,441 docs/s** against **114,887 docs/s**.

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

## Reading safetensors — listing is level, reading is not

`model.safetensors` for `bert-base-uncased`: 420 MB, 206 tensors.

| Measure | Python | HF.Net | |
|---|---:|---:|---|
| Open and list every tensor | 0.49 ms | 0.79 ms | 1.6x slower |
| Read one 30,522 × 768 tensor | 0.49 ms | 149 ms | 306x slower |

Listing is level, because both read only the header.

**Reading a tensor is not, and the reason is structural.** PyTorch hands back a zero-copy view over
the F32 bytes as they lie on disk. HF.Net widens every value to `double`, because that is what
`NdArray` holds — 23.4 million values, 187 MB materialised. It is paid once per tensor at load
time, not per forward pass.

> This table is also where a **600x performance bug** was found. The reader was allocating and
> copying a fixed 256 MB prefix to parse a header of about twenty kilobytes, which made listing
> tensors take 429 ms. Reading the declared header length first and then exactly that many bytes
> brought it under a millisecond. Nothing but a comparison against a reference implementation would
> have made that visible — it looked fast enough on its own.

## Encoder inference

One forward pass over 12 tokens.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| bert-base-uncased, 1 document | 36.4 ms | 111 ms | 3.1x slower |
| bert-base-uncased, 8 documents | 161 ms | 1,104 ms | 6.9x slower |
| bert-tiny, 1 document | 1.17 ms | 0.95 ms | **1.22x faster** |

The previous version of this page recorded **300–600 ms** for one bert-base document and 1,736 ms
for eight. What changed is described below; what did not change is the precision. The managed
encoder agrees with torch **in float64 to about 1e-13** on bert-base's hidden states, which is the
number that says the speed was not bought with correctness.

### Where the time went, and what was done about it

Measured per encoder block before touching anything:

| | 12 tokens | 197 positions (ViT) | 577 positions (ViT at 384 px) |
|---|---:|---:|---:|
| Foundation attention, inner loop | ~2 ms | **~298 ms** | **~2,760 ms** |
| Four Q/K/V/O projections | 4.2 ms | 38 ms | 76 ms |
| Feed-forward pair | 9.6–15 ms | 91–165 ms | 178–618 ms |

Four changes, each measured before it was kept:

1. **Attention.** The foundation's `MultiHeadAttention` is a sequential indexer loop. HF.Net's own
   copies each head's keys and values into contiguous blocks, makes every score one vectorised dot
   product, and splits the work by head and block of queries. The inner loop is **about 20x
   faster**, with results equal to 1e-15.
2. **One linear kernel for everything.** It computes four input rows against two weight rows per
   pass, over tiles of outputs and rows run in parallel. For a 12-token input it is **4.4x faster**
   than the foundation's packed `MatMul`, and 1.35x faster at 197 rows. At 577 rows it is still
   1.3x slower, which is the next item on the list.
3. **Weights in float32, activations in double.** Every checkpoint stores F32 or narrower, so
   float32 holds the weights exactly while halving the bytes streamed per forward pass.
   Activations and every sum stay `double`, which is why the 1e-13 agreement survived.
4. **The JIT.** A loop-heavy method called a handful of times runs as unoptimised tier-0 code, and
   one forward pass is exactly a handful of times: the linear kernel measured **12x slower** before
   tiering caught up. The hot methods are marked `AggressiveOptimization`.

Two smaller ones fell out of profiling. The exact GELU spent 40% of a feed-forward layer inside
the foundation's 50-term erf series; a table-plus-Taylor erf is 3x faster and closer to a correctly
rounded erf. The ViT patch embedding was reading pixels through an indexer that allocated per call:
74 ms, now 8.

### What is left

The single-document gap is the arithmetic rate. The kernel runs at about 13 GMAC/s on this machine,
roughly what the foundation's packed `MatMul` manages; torch's float32 MKL kernels do about four
times that. Eight documents at once is almost pure arithmetic, so that row moved least (1.6x). The
two known steps are a packed GEMM with a wider register tile and, as an opt-in, float32
activations. Both are in [PLAN.md](../PLAN.md).

### Vision

`google/vit-base-patch16-224` on one 224 × 224 image. The image is a formula both halves compute
rather than a photograph, so no resampler sits between them.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| vit-base-patch16-224, 1 image | 226 ms | 1,964 ms | 8.7x slower |

The previous version of this page recorded **12 seconds**. At 384 px, which needs the position
embeddings interpolated, it is 6.8 s against 45.8 s before.

### The production path — faster than torch

The same `bert-base-uncased` checkpoint, exported to ONNX by the Python half and run from .NET
through GraviOptimum on ONNX Runtime's CPU provider:

| Path | One document, 12 tokens | against torch |
|---|---:|---|
| torch (Python) | 36.4 ms | — |
| HF.Net managed | 111 ms | 3.1x slower |
| **HF.Net through ONNX Runtime** | **23.8 ms** | **1.53x faster** |

The ONNX hidden states differ from the managed ones by at most **4.7e-6**, the size of float32
arithmetic. The previous version of this page measured this path on a tiny test model because no
export of the real one was at hand; the benchmark now exports it itself.

**When you need throughput, this is the answer**, and it is not a compromise: it is faster than the
reference implementation, from .NET. See [GraviOptimum](GraviOptimum.md).

## Do they agree? — to the last digit that means anything

Speed is the easy half. This is the half that decides whether any of it is usable.

**`The capital of France is [MASK].`**

| Rank | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | paris | 41.6790% | paris | 41.6788% |
| 2 | lille | 7.1416% | lille | 7.1416% |
| 3 | lyon | 6.3393% | lyon | 6.3392% |
| 4 | marseille | 4.4448% | marseille | 4.4447% |
| 5 | tours | 3.0297% | tours | 3.0297% |

**`He was a [MASK] player in the national team.`**

| Rank | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | regular | 54.8921% | regular | 54.8917% |
| 2 | key | 19.4748% | key | 19.4747% |
| 3 | former | 5.8151% | former | 5.8151% |
| 4 | capped | 2.1575% | capped | 2.1575% |
| 5 | prominent | 1.3643% | prominent | 1.3643% |

The Python column is the `pipeline`, which runs torch in float32; the differences in the fourth
decimal of a percentage are float32's. Against torch in **float64**, HF.Net's fill-mask
probabilities agree to ten decimal places and bert-base's hidden states to about 1e-13.

**`google/vit-base-patch16-224`**, torch in float64 against HF.Net on the same pixels:

| Rank | torch | | HF.Net | |
|---|---|---:|---|---:|
| 1 | binder, ring-binder | 0.1186940263 | binder, ring-binder | 0.1186940263 |
| 2 | coil, spiral, volute, whorl, helix | 0.0446382711 | coil, spiral, volute, whorl, helix | 0.0446382711 |
| 3 | screen, CRT screen | 0.0433549419 | screen, CRT screen | 0.0433549419 |
| 4 | television, television system | 0.0315982656 | television, television system | 0.0315982656 |
| 5 | rubber eraser, rubber, pencil eraser | 0.0241216848 | rubber eraser, rubber, pencil eraser | 0.0241216848 |

Largest difference: **1.3e-15**.

> **What the precision check found.** An earlier version of this page said the managed encoder
> agreed with torch "to about a tenth of a percentage point" and put the residual down to
> `double` against `float`. That was wrong. Both encoders ran the **tanh approximation** of GELU,
> and every one of these checkpoints asks for the exact, erf-based one. It left bert-base's hidden
> states **2.8e-2** from torch. It was found by feeding both sides identical inputs in the same
> precision, which removes every other explanation.

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
| Running an encoder in production | **ONNX through GraviOptimum** — faster than torch, from .NET |
| Running an encoder to understand it, or checking an export | **HF.Net managed** — 3x torch on a sentence, and exact to 1e-13 in float64 |
| Training LoRA on hundreds of examples | **HF.Net** — `PeftModel.Train`, exact, and the adapter loads in Python PEFT; see [GraviPEFT](GraviPEFT.md) |
| Training at scale | **Python.** HF.Net trains on the CPU one sequence at a time; train there and serve the adapter here |

## Reproducing this

```bash
cd benchmarks/comparison
pip install tokenizers transformers onnx
pip install torch --index-url https://download.pytorch.org/whl/cpu

python python/bench.py --out python/python.json     # also exports onnx/bert-base-uncased.onnx
dotnet run -c Release --project HFNet.Comparison -- dotnet.json
python report.py
```

Run both halves in the same session. Do not compare numbers taken on different days — on a
throttling laptop that difference exceeds most of what is being measured.
