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

### BLAS/LAPACK interop — now quantified

The cross-stack comparison put numbers on this, and they are the strongest argument in the
roadmap. Against NumPy on the same machine:

| | Gravicode.Science | NumPy | Gap |
|---|---:|---:|---|
| symmetric eigen, 256×256 | 2,740 ms | 19.7 ms | **139×** |
| SVD, 256×256 | 1,795 ms | 27.1 ms | **66×** |
| PCA to 5 components (SVD underneath) | 375 ms | 5.9 ms | **64×** |
| LU, 256×256 | 49.4 ms | 3.3 ms | 15× |
| matmul, 1024×1024 | 170 ms | 27.8 ms | 6× |

The Jacobi SVD and eigen solvers were chosen for robustness and zero dependencies, and they cost
one to two orders of magnitude against LAPACK's divide-and-conquer routines. Nothing else on this
roadmap would improve the library as much.

The plan is a native backend behind the existing `IComputeBackend` interface, so it becomes a
fourth dispatch target rather than a rewrite:

- P/Invoke bindings for `dgemm`, `dgesv`, `dgeqrf`, `dgesvd`, `dsyev`
- Runtime probing with a silent fall back to the managed path, exactly as the GPU backend does
- The managed implementations stay as the reference the tests compare against

### Single precision
Every array is currently `double`. A `float` path would roughly double SIMD throughput and, more
importantly, make the GPU backend genuinely worthwhile — the float64 penalty is the single reason
the GPU loses today. This means a generic `NdArray<T>` over `INumber<T>`, which is a large change
and is why it is not in v0.1.

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

### Automatic differentiation
The GNN layers carry hand-derived gradients. That was the right call for three known
architectures, but it does not extend. A small reverse-mode tape over `NdArray` would let
GraviGraph and GraviText express new layers without deriving backward passes by hand, and would
replace the finite-difference gradients in variational inference.

### Better MCMC
Random-walk Metropolis mixes slowly on correlated posteriors. Hamiltonian Monte Carlo and NUTS
need gradients of the log posterior — so they depend on the autodiff work above, and would land
together with it.

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

- **Test coverage.** 400 tests today. Every bug found gets a regression test — that is how the
  memory-mapped CSV page-padding bug and the directed-graph connectivity bug are now covered.
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
