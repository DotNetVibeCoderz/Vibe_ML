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
| One document | 0.03 ms | 0.01 ms | **2.62x faster** |
| 1,000 documents | 15.52 ms | 8.70 ms | **1.78x faster** |

Throughput: **64,441 docs/s** (Python) against **114,887 docs/s** (HF.Net).

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
| Open and list tensors | 0.49 ms | 0.79 ms | 1.60x slower |
| Read one 30522x768 tensor | 0.49 ms | 149.25 ms | 306.53x slower |

Reading one tensor costs HF.Net more because every value is widened to `double` on the
way out, where PyTorch hands back the F32 buffer as it lies on disk. Listing the
tensors touches only the header in both.

## Encoder inference

One forward pass over 12 tokens.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| bert-base-uncased, 1 document | 36.44 ms | 111.28 ms | 3.05x slower |
| bert-base-uncased, 8 documents | 161.10 ms | 1,104.40 ms | 6.86x slower |
| bert-tiny, 1 document | 1.17 ms | 0.95 ms | **1.22x faster** |

The managed encoder computes in `double` with float32 weights - which is exact, since
every checkpoint stores float32 or narrower - and agrees with torch in float64 to about
1e-13. torch runs float32 through hand-tuned kernels. For throughput the answer is the
ONNX row below.

### The same model through ONNX Runtime

`bert-base-uncased`, exported from the same checkpoint by the Python half and run from
.NET through GraviOptimum on the `Cpu` provider.

| Path | Time | against torch |
|---|---:|---|
| torch (Python) | 36.44 ms | — |
| HF.Net managed | 111.28 ms | 3.05x slower |
| HF.Net through ONNX Runtime | 23.78 ms | **1.53x faster** |

Largest difference between the ONNX and managed hidden states: **4.7e-06** -
float32 arithmetic, since the managed encoder matches torch in float64 to about 1e-13.

## Vision

`google/vit-base-patch16-224` on one 224x224 image. The image is a formula both halves
compute, not a photograph, so no resampler sits between them.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| vit-base-patch16-224, 1 image | 226.22 ms | 1,963.85 ms | 8.68x slower |

| Rank | torch (float64) | | HF.Net | |
|---|---|---:|---|---:|
| 1 | binder, ring-binder | 0.1186940263 | binder, ring-binder | 0.1186940263 |
| 2 | coil, spiral, volute, whorl, helix | 0.0446382711 | coil, spiral, volute, whorl, helix | 0.0446382711 |
| 3 | screen, CRT screen | 0.0433549419 | screen, CRT screen | 0.0433549419 |
| 4 | television, television system | 0.0315982656 | television, television system | 0.0315982656 |
| 5 | rubber eraser, rubber, pencil eraser | 0.0241216848 | rubber eraser, rubber, pencil eraser | 0.0241216848 |

Largest difference in the top five: **1.3e-15**.

## Do they agree?

Speed is the easy half. This is the half that matters: the same prompt, the same
checkpoint, and whether HF.Net's encoder reaches the same conclusions.

**`The capital of France is [MASK].`**

| Rank | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | paris | 41.6790% | paris | 41.6788% |
| 2 | lille | 7.1416% | lille | 7.1416% |
| 3 | lyon | 6.3393% | lyon | 6.3392% |
| 4 | marseille | 4.4448% | marseille | 4.4447% |
| 5 | tours | 3.0297% | tours | 3.0297% |

Top prediction matches: **yes**. Overlap in the top five: **5/5**.

**`He was a [MASK] player in the national team.`**

| Rank | Python | | HF.Net | |
|---|---|---:|---|---:|
| 1 | regular | 54.8921% | regular | 54.8917% |
| 2 | key | 19.4748% | key | 19.4747% |
| 3 | former | 5.8151% | former | 5.8151% |
| 4 | capped | 2.1575% | capped | 2.1575% |
| 5 | prominent | 1.3643% | prominent | 1.3643% |

Top prediction matches: **yes**. Overlap in the top five: **5/5**.

## Reproducing this

```bash
cd benchmarks/comparison
pip install tokenizers transformers onnx
pip install torch --index-url https://download.pytorch.org/whl/cpu
python python/bench.py --out python/python.json     # also exports onnx/bert-base-uncased.onnx
dotnet run -c Release --project HFNet.Comparison -- dotnet.json
python report.py
```

Run them in the same session, and do not compare numbers taken on different days — on a
throttling laptop that difference exceeds most of what is measured here.
