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
| One document | 0.04 ms | 0.01 ms | **3.11x faster** |
| 1,000 documents | 20.60 ms | 9.60 ms | **2.15x faster** |

Throughput: **48,553 docs/s** (Python) against **104,154 docs/s** (HF.Net).

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
| Open and list tensors | 0.70 ms | 0.71 ms | 1.01x slower |
| Read one 30522x768 tensor | 0.74 ms | 185.18 ms | 251.19x slower |

Reading one tensor costs HF.Net more because every value is widened to `double` on the
way out, where PyTorch hands back the F32 buffer as it lies on disk. Listing the
tensors touches only the header in both.

## Encoder inference

One forward pass over 12 tokens.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| bert-base-uncased, 1 document | 50.06 ms | 59.49 ms | 1.19x slower |
| bert-base-uncased, 8 documents | 263.15 ms | 427.73 ms | 1.63x slower |
| bert-tiny, 1 document | 1.75 ms | 1.05 ms | **1.66x faster** |

With `ComputeOptions.LinearLayers = Precision.Single`, the linear layers in float32 as torch runs them:

| Model | Python (torch) | HF.Net (float32 linear) | |
|---|---:|---:|---|
| bert-base-uncased, 1 document | 50.06 ms | 45.00 ms | **1.11x faster** |
| bert-base-uncased, 8 documents | 263.15 ms | 286.67 ms | 1.09x slower |
| bert-tiny, 1 document | 1.75 ms | 1.37 ms | **1.28x faster** |

On bert-base the hidden states move by at most **9.9e-06** from the double-precision ones.

The managed encoder computes in `double` with float32 weights - which is exact, since
every checkpoint stores float32 or narrower - and agrees with torch in float64 to about
1e-13. torch runs float32 through hand-tuned kernels. For throughput the answer is the
ONNX row below.

### The same model through ONNX Runtime

`bert-base-uncased`, exported from the same checkpoint by the Python half and run from
.NET through GraviOptimum on the `Cpu` provider.

| Path | Time | against torch |
|---|---:|---|
| torch (Python) | 50.06 ms | — |
| HF.Net managed | 59.49 ms | 1.19x slower |
| HF.Net through ONNX Runtime | 28.92 ms | **1.73x faster** |

Largest difference between the ONNX and managed hidden states: **4.7e-06** -
float32 arithmetic, since the managed encoder matches torch in float64 to about 1e-13.

## Vision

`google/vit-base-patch16-224` on one 224x224 image. The image is a formula both halves
compute, not a photograph, so no resampler sits between them.

| Model | Python (torch) | HF.Net (managed) | |
|---|---:|---:|---|
| vit-base-patch16-224, 1 image | 344.51 ms | 912.65 ms | 2.65x slower |
| vit-base-patch16-224, float32 linear | 344.51 ms | 682.24 ms | 1.98x slower |

| Rank | torch (float64) | | HF.Net | |
|---|---|---:|---|---:|
| 1 | binder, ring-binder | 0.1186940263 | binder, ring-binder | 0.1186940263 |
| 2 | coil, spiral, volute, whorl, helix | 0.0446382711 | coil, spiral, volute, whorl, helix | 0.0446382711 |
| 3 | screen, CRT screen | 0.0433549419 | screen, CRT screen | 0.0433549419 |
| 4 | television, television system | 0.0315982656 | television, television system | 0.0315982656 |
| 5 | rubber eraser, rubber, pencil eraser | 0.0241216848 | rubber eraser, rubber, pencil eraser | 0.0241216848 |

Largest difference in the top five: **5.7e-16**.

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
