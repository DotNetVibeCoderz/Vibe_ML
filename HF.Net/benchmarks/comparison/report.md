# HF.Net versus the Python reference implementation

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

Both halves measure the same inputs on the same machine in the same session, and both
report the **best** of several timed runs after a warm-up. On a laptop that throttles, the
mean measures the thermal state of the room.

| | |
|---|---|
| Machine | Windows-11-10.0.26200-SP0 |
| CPU | 8 logical cores |
| Python | Python 3.12.10, transformers 5.17.0, tokenizers 0.23.2, torch 2.14.0+cpu |
| .NET | .NET 10.0.12 (Release) |
| torch threads | 4 |

## Tokenization

`bert-base-uncased`. The Python tokenizer is the Rust `tokenizers` crate behind a thin
binding; GraviTokenizers is managed C#.

| Measure | Python | HF.Net | |
|---|---:|---:|---|
| One document | 0.03 ms | 0.01 ms | **2.81x faster** |
| 1,000 documents | 18.13 ms | 10.94 ms | **1.66x faster** |

Throughput: **55,144 docs/s** (Python) against **91,423 docs/s** (HF.Net).

**Ids identical to the reference: yes.**

```
input   Hello, world! Tokenizers are unbelievable.
python  101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102
hf.net  101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102
```

## Reading safetensors

`model.safetensors` for `bert-base-uncased` — 420.0 MB, 206 tensors.

| Measure | Python | HF.Net | |
|---|---:|---:|---|
| Open and list tensors | 0.65 ms | 0.71 ms | 1.09x slower |
| Read one 30522x768 tensor | 1.14 ms | 196.96 ms | 172.59x slower |

Reading one tensor costs HF.Net more because every value is widened to `double` on the
way out, where PyTorch hands back the F32 buffer as it lies on disk. Listing the
tensors touches only the header in both.

## Encoder inference

One forward pass over 12 tokens.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| bert-base-uncased, 1 document | 50.83 ms | 597.94 ms | 11.76x slower |
| bert-base-uncased, 8 documents | 261.62 ms | 1,736.40 ms | 6.64x slower |
| bert-tiny, 1 document | 1.47 ms | 2.09 ms | 1.42x slower |

**This is the gap, and it is the expected one.** The managed encoder is `double` end to
end and written for clarity; torch dispatches to hand-tuned single-precision kernels
with fused attention. GraviTransformers exists so a model can be loaded, inspected and
understood in pure .NET — for throughput the answer is an ONNX export.

### The same work through ONNX Runtime

`Cpu` provider, via GraviOptimum:

- **0.67 ms** best, 0.85 ms median

Measured on a small test model rather than on bert-base, so it is not comparable with
the table above. It is here to show the shape of the production path: the same
single-precision kernels torch uses, reached from .NET.

## Do they agree?

Speed is the easy half. This is the half that matters: the same prompt, the same
checkpoint, and whether HF.Net's encoder reaches the same conclusions.

**`The capital of France is [MASK].`**

| Rank | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | paris | 41.68% | paris | 41.53% |
| 2 | lille | 7.14% | lille | 7.16% |
| 3 | lyon | 6.34% | lyon | 6.31% |
| 4 | marseille | 4.44% | marseille | 4.46% |
| 5 | tours | 3.03% | tours | 3.02% |

Top prediction matches: **yes**. Overlap in the top five: **5/5**.

**`He was a [MASK] player in the national team.`**

| Rank | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | regular | 54.89% | regular | 55.00% |
| 2 | key | 19.47% | key | 19.39% |
| 3 | former | 5.82% | former | 5.83% |
| 4 | capped | 2.16% | capped | 2.14% |
| 5 | prominent | 1.36% | prominent | 1.36% |

Top prediction matches: **yes**. Overlap in the top five: **5/5**.

## Reproducing this

```bash
cd benchmarks/comparison
pip install tokenizers transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
python python/bench.py --out python/python.json
dotnet run -c Release --project HFNet.Comparison -- dotnet.json
python report.py
```

Run them in the same session, and do not compare numbers taken on different days — on a
throttling laptop that difference exceeds most of what is measured here.
