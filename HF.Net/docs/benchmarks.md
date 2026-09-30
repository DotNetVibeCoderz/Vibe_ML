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
| One document | 0.04 ms | 0.01 ms | **3.1x faster** |
| 1,000 documents | 20.60 ms | 9.60 ms | **2.15x faster** |

**48,553 docs/s** against **104,154 docs/s**.

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
| Open and list every tensor | 0.70 ms | 0.71 ms | level |
| Read one 30,522 × 768 tensor | 0.74 ms | 185 ms | 251x slower |

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
| bert-base-uncased, 1 document | 50.1 ms | 59.5 ms | 1.19x slower |
| bert-base-uncased, 8 documents | 263 ms | 428 ms | 1.63x slower |
| bert-tiny, 1 document | 1.75 ms | 1.05 ms | **1.66x faster** |

Earlier versions of this page recorded **300–600 ms** for one bert-base document, then 111 ms.
What changed is described below; what did not change is the precision. The managed
encoder agrees with torch **in float64 to about 1e-13** on bert-base's hidden states, which is the
number that says the speed was not bought with correctness.


With `ComputeOptions.LinearLayers = Precision.Single` - the linear layers in float32, as torch runs
them - the managed encoder overtakes torch on one sentence:

| Model | Python (torch) | HF.Net, float32 linear | |
|---|---:|---:|---|
| bert-base-uncased, 1 document | 50.1 ms | 45.0 ms | **1.11x faster** |
| bert-base-uncased, 8 documents | 263 ms | 287 ms | 1.09x slower |
| bert-tiny, 1 document | 1.75 ms | 1.37 ms | **1.28x faster** |

The hidden states then move by at most 9.9e-6 from the double-precision ones.

Every torch time on this page is slower than in earlier versions (50 ms here against 37.6 before):
the laptop ran warmer. Both halves ran back to back in the same session, so the ratios compare; the
absolute numbers from different days do not.

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
2. **One linear kernel for everything: a register-blocked GEMM.** The weights sit in panels of
   twelve outputs, and the inner kernel holds a 4-row by 12-output block of the result in twelve AVX
   registers: one broadcast input, three weight vectors, twelve fused multiply-adds per step. The
   dot-product kernel it replaced needed a load for every multiply-add. The float32 weights are
   widened into a per-thread buffer 256 inputs at a time and reused by every group of four rows.
   Widening inside the inner loop instead was 40% slower, because the conversion competes with the
   multiply-adds for the same port. Single-threaded it runs at 17.6 GMAC/s against the old kernel's
   5.6; across the cores, 35-54 GMAC/s at 80 rows and about 40 at 577, against about 12 before.
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

The GEMM took one bert-base document from 111 ms to 48 ms and eight from 1,104 ms to 386 ms, with
the same results: fill-mask still agrees with torch to the fourth decimal of a percentage, and ViT's
top five to 6e-16. What remains of the gap is precision. torch runs float32 and does twice the
multiply-adds per instruction; HF.Net keeps every activation and sum in `double`, which is what the
1e-13 agreement rests on. The opt-in float32 mode, above, is the answer to that. On ViT, the
feed-forward's exact GELU and attention over 197 positions are now a larger share of the time.

### Vision

`google/vit-base-patch16-224` on one 224 × 224 image. The image is a formula both halves compute
rather than a photograph, so no resampler sits between them.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| vit-base-patch16-224, 1 image | 345 ms | 913 ms | 2.65x slower |
| vit-base-patch16-224, float32 linear | 345 ms | 682 ms | 1.98x slower |

Earlier versions of this page recorded **12 seconds**, then 1,964 ms. At 384 px, which needs the
position embeddings interpolated, it was 6.8 s with the previous linear kernel, against 45.8 s
before that.

### The production path — faster than torch

The same `bert-base-uncased` checkpoint, exported to ONNX by the Python half and run from .NET
through GraviOptimum on ONNX Runtime's CPU provider:

| Path | One document, 12 tokens | against torch |
|---|---:|---|
| torch (Python) | 50.1 ms | — |
| HF.Net managed | 59.5 ms | 1.19x slower |
| **HF.Net through ONNX Runtime** | **28.9 ms** | **1.73x faster** |

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

Largest difference: **5.7e-16**.

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

## The per-library suites

`benchmarks/HFNet.Benchmarks` is one BenchmarkDotNet project covering every library, run on the same
laptop: Intel Core i7-8650U, 4 cores and 8 threads, .NET 10.0.12. The numbers below are means of eight
iterations after three warm-ups; run it with `dotnet run -c Release --project benchmarks/HFNet.Benchmarks`
or one suite with `-- --filter *Tokenizer*`.

**GraviTransformers.** One `[128, 768] x [768, 3072]` product - bert-base's first feed-forward layer at
128 tokens - then whole models, in both precisions:

| | double | float32 linear |
|---|---:|---:|
| HF.Net GEMM, the product | 9.2 ms | 6.5 ms |
| Foundation `MatMul` (CPU), the product | 68.7 ms | |
| Foundation ILGPU, the product | 155 ms | |
| bert-base forward pass, 8 tokens | 44 ms | 33 ms |
| bert-base, 32 tokens | 108 ms | 79 ms |
| bert-base, 128 tokens | 434 ms | 326 ms |
| bert-base, 512 tokens | 2,271 ms | 1,891 ms |
| vit-base-patch16-224, one image | 743 ms | 551 ms |
| clip-vit-base-patch32, one image against five labels | 231 ms | 159 ms |
| gpt2, 32 new tokens with the KV cache | 1,480 ms | 1,258 ms |

The GPU row is the foundation's double-precision path on an integrated GPU, which runs double at a
fraction of its float32 rate - slower than the CPU, as the foundation found. At 512 tokens float32 gains
only 17%, because attention, which grows with the square of the length and stays double, is most of
the cost there.

**GraviTokenizers.** 1,000 documents of a multilingual corpus:

| Tokenizer | one thread | `EncodeBatch` |
|---|---:|---:|
| bert-base-multilingual-cased | 5.1 ms | 3.4 ms |
| gpt2 | 9.4 ms | 6.6 ms |
| xlm-roberta-base | 29.3 ms | 11.2 ms |

**GraviDatasets.** 200,000 generated rows: a full CSV load 356 ms, the same file memory-mapped
349 ms, Parquet 56 ms. Memory-mapping saves memory, not time, on a file that fits in RAM; Parquet is
six times faster to read.

**GraviPEFT.** One LoRA epoch of 32 sentences in batches of 8: 63 ms on bert-tiny, 3.3 s on bert-base.

**GraviAccelerate.** A data-parallel gradient over 400,000 rows: 45 ms on one worker, 26 on two, 24 on
four, 16 on eight - 2.8x, on four physical cores.

**GraviOptimum.** bert-base on one sentence: 63 ms managed, 37 ms through ONNX Runtime.

**GraviDiffusers.** The tiny test pipeline, 10 steps at 64x64 with guidance: 455 ms. 25 DDIM steps on a
64x64 Stable Diffusion latent, scheduler only: 16 ms; 25 Euler steps: 27 ms.

**GraviHub.** Opening bert-base's `model.safetensors` and listing its 206 tensors: 0.45 ms. Downloading
bert-tiny's 17.7 MB pickle into a fresh cache: 2.7 s.

## What to take from this

| If you are | Use |
|---|---|
| Tokenizing large volumes of text | **HF.Net** — faster, identical output |
| Loading and inspecting a checkpoint | **HF.Net** — level on headers, and it reads formats Python's fast path refuses |
| Running an encoder in production | **ONNX through GraviOptimum** — faster than torch, from .NET |
| Running an encoder to understand it, or checking an export | **HF.Net managed** — within 1.2x of torch on a sentence and exact to 1e-13 in float64; faster than torch with float32 linear layers |
| Training LoRA on hundreds of examples | **HF.Net** — `PeftModel.Train`, exact, and the adapter loads in Python PEFT; see [GraviPEFT](GraviPEFT.md) |
| Training at scale | **Python.** HF.Net trains on the CPU, a few seconds per epoch of a few dozen sentences; train there and serve the adapter here |

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
