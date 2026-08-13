# PLAN — Gravicode.Science roadmap

Status of the current release is in [Progress.md](Progress.md). This file is about direction.

---

## Where the project is

**v0.1.0 — feature complete against the original specification.** All six libraries, their
samples, notebooks, benchmarks and tests exist and run. 400 tests pass. Documentation is
published in English and Bahasa Indonesia.

Every algorithm listed in `requirements.md` is implemented rather than stubbed. Three things are
deliberately *not* claimed, and are documented as such wherever they appear:

1. **No pretrained transformer weights.** `TransformerModel`'s forward pass is complete and
   correct, but a fresh model is randomly initialised. Semantics require `LoadWeights`.
2. **Named entity recognition is rule-based**, not a trained sequence model.
3. **GPU dispatch is opt-in**, because on integrated hardware the float64 GPU path measured
   5–8× slower than the CPU.

---

## v0.2 — Precision and interoperability

The theme is making the numeric core competitive with native libraries and easy to plug into
existing .NET ML work.

### Better decomposition algorithms — ✅ done

The cross-stack comparison made this the clearest item on the roadmap, and most of it turned out
not to need a native dependency at all: the Jacobi SVD and eigen solvers were losing to LAPACK on
*algorithm*, not on Fortran. Both now reduce the matrix once with Householder reflections and then
run a shifted QR/QL iteration on the condensed form.

| | v0.1 | v0.2 | Gain | Gap to NumPy |
|---|---:|---:|---|---|
| symmetric eigen, 256×256 | 2,740 ms | 116 ms | **23.6×** | 139× → **5.9×** |
| SVD, 256×256 | 1,795 ms | 328 ms | **5.5×** | 66× → **12.1×** |
| PCA to 5 components | 375 ms | 63.9 ms | **5.9×** | 64× → **10.8×** |

PCA gained beyond its SVD because `SvdRightVectors` skips the left factor, which PCA never reads
and which is most of the work on a 20 000-row matrix.

`SvdJacobi` and `SymmetricEigenJacobi` remain as the independent reference the tests pin against.

### CPU kernel work — ✅ done

Prompted by a review of `rekomendasi.md`. Three of its suggestions were already in place
(`Vector<T>`, TPL, ILGPU); the useful part of the exercise was measuring the rest rather than
adopting them on reputation.

| Change | Result |
|---|---|
| Element-wise parallel path stopped copying its operands | **3.06×**, now ahead of NumPy |
| `Unary` gained a parallel path (it had none) | `Exp`/`Log` now scale with cores |
| `MathUtil.Tanh` via `Math.Exp` | **1.6×**, agrees to 2.2e-16 |
| `LinAlg.Dot` keeps its C tile in registers across a `k` slice | **1.7×** at 128, 1.05–1.4× from 512 up |

**`TensorPrimitives` was measured and rejected.** For `double` its `Exp` came in at 0.93× and `Log`
at 0.57× against plain `Math` — .NET's scalar intrinsics already beat the vectorised polynomials,
and single precision was no better (`Exp` 0.83×). Only `Tanh` (2.2×) and `Cosh` (2.5×) gained, and
most of the `Tanh` win was available for free by routing through `Math.Exp`. A NuGet dependency for
one hyperbolic function is not a trade worth making in a foundation library.

**Hardware intrinsics were not adopted either**, and the reason is the same measurement: element-wise
work is bound by memory bandwidth, not instruction width. `Vector<T>` already maps to AVX2, and the
element-wise fix bought 2.9 → 10.2 GB/s by moving *less* data, which no amount of wider registers
would have done.

### BLAS/LAPACK interop — still quantified, now smaller

What is left is genuinely the native-code gap, and it is one order of magnitude rather than two:

| | Gravicode.Science | NumPy | Gap |
|---|---:|---:|---|
| matrix inverse, 256×256 | 147 ms | 9.9 ms | 15× |
| QR, 256×256 | 154 ms | 10.4 ms | 15× |
| LU, 256×256 | 41 ms | 3.3 ms | 13× |
| SVD, 256×256 | 308 ms | 27.1 ms | 11× |
| matmul, 1024×1024 | 93 ms | 27.8 ms | 3.4× |

The plan is a native backend behind the existing `IComputeBackend` interface, so it becomes a
fourth dispatch target rather than a rewrite:

- P/Invoke bindings for `dgemm`, `dgesv`, `dgeqrf`, `dgesvd`, `dsyev`
- Runtime probing with a silent fall back to the managed path, exactly as the GPU backend does
- The managed implementations stay as the reference the tests compare against

**Open question before starting**: this ships no native binary, so it only helps users who already
have OpenBLAS or MKL on the machine. Deciding what to do when they don't — bundle per-RID native
assets, or leave it opt-in and document it — is the real design work, not the P/Invoke.

### A packed matrix product
Independent of native interop, the remaining 3.4× on `matmul` is reachable in managed code: pack A
and B into tile-contiguous buffers and block on three levels, the way OpenBLAS does. The v0.2
kernel took the cheap part of that (register accumulation and a `k` slice) and got 1.05–1.4×;
the rest needs the packing, which is a few hundred lines of careful work.

### Single precision
Every array is currently `double`. A `float` path would roughly double SIMD throughput and, more
importantly, make the GPU backend genuinely worthwhile — the float64 penalty is the single reason
the GPU loses today. This means a generic `NdArray<T>` over `INumber<T>`.

This is the largest single change on the roadmap: `NdArray` is the type every other library is
written against, so making it generic touches all six. It is also the item where the payoff is
least certain on a machine like the reference one — the measured `TensorPrimitives` figures showed
single precision buying nothing for transcendentals, and the element-wise path is bandwidth-bound,
where halving the element size genuinely does help. Worth prototyping on one library before
committing the whole stack to it.

### ONNX Runtime and ML.NET
- Export a fitted `Pipeline` to ONNX so a model trained here can be served anywhere
- Import an ONNX model as an `IEstimator`, which would also give `TransformerModel` real weights
- `IDataView` adapters both ways for `DataFrame`

### Arrow
Zero-copy interchange with pandas, Polars and Spark via the Arrow memory format. `DataFrame` is
already columnar, so this is mostly a matter of matching buffer layouts.

---

## v0.3 — Scale

### Out-of-core dataframes
`MemoryMappedArray` handles arrays larger than RAM; `DataFrame` does not yet. The plan is chunked
columns with a streaming group-by and sort, so a frame can exceed memory the way the array already
can.

### Automatic differentiation — ✅ done
`GraviNum.Autodiff` is a reverse-mode tape over `NdArray`: `Tensor.Parameter`, the usual
elementwise and linear-algebra operations, `Backward()`, and a `GradientCheck` helper that pins any
new gradient against central finite differences.

What it has **not** yet done is replace the hand-derived gradients in the GNN layers — the other
half of the original item, now unblocked but unstarted. See *GNN layers on the tape* below.

### Better MCMC — ✅ done
`SampleHMC` and `SampleNUTS`, both with dual-averaging step-size adaptation, working in
unconstrained space with the support transform built on the tape so the chain rule through it is
never hand-derived. Pinned against the exact Beta-binomial posterior, like the existing samplers.

**The measured result is not the one the roadmap assumed.** On a 20-parameter model NUTS returns
draws that are 100% effective with R-hat 1.0000, while random-walk Metropolis returns R-hat 1.773 —
chains that have not converged and draws that are not from the posterior. But on a *one*-parameter
model the random walk is 4× faster per effective sample, and reverse-mode differentiation of a
2-parameter log posterior measured **11× slower** than central differences, because the tape
allocates a node per operation while finite differences just call a cheap scalar function `2d`
times. The crossover is near 50 parameters. Variational inference therefore picks its gradient
method by dimension rather than always paying for the tape.

### GNN layers on the tape
GCN, GraphSAGE and GAT still carry hand-derived backward passes, including through the attention
softmax. They work and they are tested, so this is not urgent — but a fourth architecture would
mean a fourth derivation, which is exactly the cost the tape exists to remove. The same applies to
`TransformerModel` in GraviText.

The tape's per-node allocation is the thing to watch here: a GNN forward pass is a handful of large
matrix operations rather than thousands of scalar ones, which is the regime where reverse mode is
unambiguously the right choice — the opposite of the low-dimensional log posteriors above.

### Distributed training
Data-parallel training across processes for the tree ensembles and GNNs, which are the two places
where a single machine runs out first.

---

## v0.4 — Breadth

### GraviNum
Einstein summation, FFT, more `Slice` ergonomics, complex number support.

### GraviFrame
Window functions with partitioning, `asof` joins for time series alignment, categorical column type
with dictionary encoding, SQL and Excel readers.

### GraviLearn
Calibration curves, permutation importance, SHAP-style explanations, isotonic regression,
one-class SVM, HDBSCAN, imbalanced-data resampling.

### GraviText
BPE and SentencePiece tokenizers, a decoder stack for generation, sequence labelling with a CRF
head, a real trained NER model to replace the rule-based one.

### GraviGraph
Heterogeneous graphs, temporal graphs, edge features, graph-level pooling and classification,
neighbourhood sampling so GraphSAGE can train on graphs that do not fit in memory.

### GraviProb
Dirichlet and multivariate distributions, Gaussian processes, state-space models, model comparison
via WAIC and LOO.

---

## Continuous

These are not versioned; they run alongside everything above.

- **Test coverage.** 458 tests today. Every bug found gets a regression test — that is how the
  memory-mapped CSV page-padding bug and the directed-graph connectivity bug are now covered.
  Where a fast path replaces a simple one, the simple one stays as the reference it is checked
  against, as `SvdJacobi` and `SymmetricEigenJacobi` now do.
- **Documentation stays bilingual.** Every English page has an Indonesian counterpart, and they
  are updated together.
- **Benchmarks stay honest.** Where a result contradicts the obvious assumption — as the GPU
  numbers do — it gets documented, not buried.
- **Public API stability.** Breaking changes wait for a minor version and are listed in the
  release notes.

---

## Non-goals

Worth stating so the scope stays legible:

- **Not a deep learning framework.** No competition with TensorFlow or PyTorch. The transformer
  and GNN layers exist to make those architectures usable from .NET, not to train foundation
  models.
- **Not a distributed compute engine.** No competition with Spark or Dask.
- **Not a plotting library.** Visualisation stays behind ScottPlot and OxyPlot hooks.

---

## How to contribute

1. Open an issue describing the use case before writing a large feature.
2. Every new algorithm needs tests that pin it against a known-correct reference — a closed-form
   answer, a published figure, or a naive implementation in the test itself.
3. Documentation lands in both languages in the same change.
4. Performance claims need a benchmark in `benchmarks/`.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
