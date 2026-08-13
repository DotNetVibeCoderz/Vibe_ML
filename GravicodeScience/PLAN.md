?# PLAN — Gravicode.Science roadmap

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

### BLAS/LAPACK interop — ✅ done, and optional

**The open question is answered**: leave it optional. `Compute.NativeBlas` probes for a library the
machine already has, and keeps silently to the managed kernels otherwise — the same shape as the
GPU backend. Nothing is bundled, because a tuned BLAS is a large platform-specific binary and every
user would pay for it.

`NativeLapack` now binds the factorisations too — `dgesv`, `dgeqrf`/`dorgqr`, `dgesvd`, `dsyev` —
through the **LAPACKE** interface, which takes a layout argument so row-major matrices pass straight
through. The Fortran entry points would need a transpose in and another out on every call.

| | Managed | Native | | vs NumPy, before → after |
|---|---:|---:|---|---|
| solve, 512×512, many RHS | 887 ms | 17.4 ms | **51×** | — |
| LU, 256×256 | 77.9 ms | 4.5 ms | **17×** | 21.6× → **1.4×** |
| QR, 256×256 | 151 ms | 10.7 ms | **14×** | 15× → **1.0×** |
| Cholesky, 256×256 | 14.4 ms | 1.6 ms | **9×** | 14.4× → **1.3×** |
| symmetric eigen, 256×256 | 120 ms | 45.6 ms | 2.6× | 5.6× → **2.3×** |
| matrix inverse, 256×256 | 135 ms | 58.0 ms | 2.3× | 15× → **5.9×** |
| SVD, 256×256 | 331 ms | 176 ms | 1.9× | 11× → **6.5×** |

With a BLAS present the 512-cube product is **faster than NumPy**, and LU, QR and Cholesky land
within 40% of it — which follows once you notice both stacks are then calling the same OpenBLAS,
and what is left is marshalling.

Three conventions had to be converted rather than assumed: LAPACK sorts eigenvalues ascending,
returns `V^T` from `dgesvd`, and — the one that nearly slipped through — with a row-major layout
already presents eigenvector *j* in column *j*, so transposing it as the Fortran convention suggests
breaks `A V = V Λ` while leaving the eigenvalues perfectly correct. The `A V = V Λ` check caught it.

`dgetrf` and `dpotrf` followed, so `Lu` and `Cholesky` are covered too — **LU 77.9 → 4.5 ms** and
**Cholesky 14.4 → 1.6 ms** at 256, both now within 40% of NumPy. Every factorisation in the library
now has a native path.

The pivot conversion was the reason those two waited, and it earned the caution: LAPACK reports a
*sequence of row swaps*, not a finished permutation. Reading one as the other produces a perfectly
valid-looking L and U that reconstructs the wrong matrix, so the tests check `P A = L U`, that the
permutation is a bijection, and that the determinant sign agrees — the last catching a sign
convention error the first two would miss.

### The benchmark now depends on what is installed

Worth stating as a methodology point, because it is the biggest threat to reproducibility here: the
published comparison figures are the **managed** path, since a clean machine finds no BLAS. On a
machine with OpenBLAS the same harness measures something quite different — parity with NumPy on
matmul, 1.3× on QR. Both are real; quoting either without saying which would be the error.
`docs/benchmarks.md` now reports both configurations side by side.

For `dgemm` specifically, measured against the OpenBLAS that ships inside numpy: **3.6× at 256,
4.8× at 512, 4.0× at 1024**, agreeing with the managed kernel to 1e-13. Two details that would
otherwise have been silent corruption:
ILP64 builds export `cblas_dgemm64_` and take 64-bit dimensions, so the width is detected rather
than assumed; and numpy/scipy rename every symbol with a `scipy_` prefix, which on a data-science
machine is usually the only BLAS present.

### A packed matrix product — ✅ done

`PackedMatMul` packs both operands into tile-contiguous buffers, parallel over row blocks with a
shared packed B panel. **1.65× at 256, 1.96× at 1024, 2.08× at 2048**, against the kernel it
replaced. Below about eight million multiply-adds packing costs more than it saves, so the
threshold sits above that, conservatively.

Two things worth recording. The first attempt was *slower everywhere* because it was
single-threaded while the kernel it was compared against ran on four cores — the packing was fine,
the parallelism had been dropped. And an early measurement showed a 192-cube taking eight times
longer than a 224-cube, which is impossible: tiered JIT had not yet recompiled the kernel, and two
warm-up calls are not enough when promotion happens after about thirty.

### The generic core — ✅ built, and it costs nothing

`NdArray<T>` over `IFloatingPointIeee754<T>`, with `UFunc<T>` carrying the element-wise kernels and
the register-blocked matrix product. One implementation serves both widths, because `Vector<T>` is
itself generic.

The prototype below said single precision was worth having. The question this had to answer was
different and more important: **does genericising slow down the `double` path**, which is what all
six libraries use today?

| | `NdArray` | `NdArray<double>` | `NdArray<float>` | Generic overhead | Float gain |
|---|---:|---:|---:|---:|---:|
| element-wise add, 1M | 2.56 ms | 2.56 ms | 1.19 ms | 1.00× | **2.16×** |
| element-wise add, 10M | 32.7 ms | 31.5 ms | 16.6 ms | 0.96× | **1.89×** |
| 512-cube product | 15.1 ms | 13.7 ms | 7.80 ms | 0.91× | **1.75×** |
| 1024-cube product | 89.0 ms | 92.6 ms | 47.1 ms | 1.04× | **1.96×** |

**No.** The overhead column sits inside measurement noise. That is the result that makes the full
migration viable — had the generic double path been even 20% slower, replacing `NdArray` with
`NdArray<double>` would have been a regression for every existing user in exchange for a `float`
path most of them would not use.

**Deliberately additive.** The six libraries still speak `NdArray`; `NdArrayConvert` bridges at the
boundary. Rewriting them so `NdArray` *is* `NdArray<double>` touches every file and all 559 tests,
and is the one remaining large change — but it is now a decision with numbers rather than a hope.

Two differences from the `double` `UFunc`, both deliberate: the generic kernels do not broadcast
and say so; and `Sum` is pairwise, which matters far more in `float` — a naive running total over a
million values of `0.1f` drifts visibly.

### Single precision — ✅ prototyped, and it is worth doing

Following this file's own advice to prototype before committing. `Single.SingleKernels` has `float`
versions of the element-wise and matrix-product kernels — deliberately not an `NdArray`-shaped
surface, because a half-finished second numeric stack would be worse than none.

| | double | float | |
|---|---:|---:|---|
| element-wise add, 1M | 2.24 ms | 1.07 ms | **2.09×** |
| element-wise add, 10M | 24.9 ms | 11.7 ms | **2.13×** |
| 1024-cube product | 94.9 ms | 52.2 ms | 1.82× |

The two gains have different causes and correctly differ: twice the SIMD lanes helps compute-bound
work, half the bytes helps bandwidth-bound work, and element-wise arithmetic — being purely
bandwidth-bound — lands at a clean 2.1×. Accuracy cost on a 1000-cube product: **1.3e-6** relative.

So the answer to "is the generic `NdArray<T>` rewrite worth it" was **yes** — and that core is now
built, above. `SingleKernels` stays as the hand-written baseline the generic version was measured
against; the two agree, which is what makes the generic figures trustworthy.

### ONNX — ✅ weight import done

`Io.OnnxReader` reads the initializers out of an ONNX file: float32, float64, float16 and the
integer types, widened to `double`. `TransformerModel.LoadOnnxWeights` uses it, which is the route
to a transformer whose embeddings mean something — the library's oldest documented limitation.

Dependency-free on purpose. ONNX Runtime is a large native package and this needs a few hundred
lines of protobuf wire format, so pulling it in to read some arrays would be the same bad trade as
`TensorPrimitives`. Verified against files written by the official Python `onnx` library, not
against a fixture built to match the reader.

A tensor whose type will not decode, or whose data does not match its declared shape, is skipped
rather than guessed at.

### ONNX export — ✅ done

`OnnxExport.Save` writes a fitted `Pipeline` as ONNX, so a model trained here can be served
anywhere. Scalers, PCA and linear models all export; each is an affine map, which is why the whole
pipeline collapses into `Sub`, `Div`, `MatMul`, `Add` and `ArgMax`. Core operators are used rather
than `ai.onnx.ml` because every runtime implements them.

Verified against the real tooling rather than against this library's own reader: `onnx.checker`
confirms the graph is valid, and Python's **onnxruntime** reproduces the .NET predictions —
150/150 labels on two Iris pipelines, 5.9e-07 maximum difference on a regression one.

Two mistakes it was worth designing against, because both produce a model that loads cleanly and
predicts wrongly:

- Writing the batch dimension as a **number** rather than a symbol, which pins the model to the
  training set's row count and makes it useless for the single row a serving endpoint sends.
- Storing PCA components untransposed. The graph multiplies rows of `x` by them, so the constant
  has to be the transpose of `ComponentVectors`.

Trees, forests and kNN are **refused**, not approximated. `OnnxExport.Supports` reports this before
anything is written.

**Still open**: the reader remains a weight reader, not a runtime — it cannot execute an arbitrary
ONNX graph. The `IDataView` adapters for ML.NET are untouched.

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

The other half of the original item — replacing the hand-derived gradients in the GNN layers — is
now done for all three. See *GNN layers on the tape* below.

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

### GNN layers on the tape — ✅ done

All three architectures now express only their forward pass; `Backward()` supplies the rest. The
graph-shaped tape operations this needed — `SparseMatMul`, `Gather`, `SegmentSum`, `ConcatColumns`,
`LeakyRelu` and a masked `SoftmaxCrossEntropy` — are in `GraviNum.Autodiff`, and the layer helpers
are public in `GnnTape` so a fourth architecture can be written without touching the library.

GAT was the one this was really for. Its attention needs a softmax over each node's incident edges,
and that was the hardest backward pass in the library to derive by hand. It turned out not to need
a bespoke kernel at all: `Gather` and `SegmentSum` are adjoints of one another, and segment softmax
*composes* from them plus `Exp`, inheriting their already-verified gradients rather than requiring
the softmax Jacobian to be worked out again. Two new primitives covered a case that looked like it
would need several.

**It immediately found a real bug.** The hand-derived GCN backward pass omitted the dropout mask on
the hidden layer, making the gradient roughly **40% wrong** whenever dropout was active. It had
been that way since the layer was written, and the model still trained to a plausible 71% on Cora —
which is precisely why it survived review. Corrected, Cora sits at 69.3%; the old number came from
a broken gradient acting as an accidental regulariser.

Measured cost, hand-written against tape, one epoch, interleaved in one process:

| Shape | hand-written | tape | |
|---|---:|---:|---|
| 500 nodes × 100 features | 2.90 ms | 5.61 ms | 1.93× |
| 2,708 × 1,433 (Cora) | 55.74 ms | 70.48 ms | **1.26×** |
| 5,000 × 500 | 48.43 ms | 70.09 ms | 1.45× |

So the tape costs about a quarter on the realistic shape and less as the matrices grow — the
opposite of its behaviour on the scalar log posteriors above, and the reason this was the right
place to apply it. GraphSAGE actually got *faster* end to end (26.9 s → 19.8 s on Cora), because
expressing mean aggregation as a sparse matrix replaced two hand-written gather/scatter loops.

On Cora at 60 epochs the three now sit at GCN 69.3%, GraphSAGE 70.3%, **GAT 72.1%** — attention
ahead of a fixed degree normalisation, which is the expected ordering and more convincing observed
after the move than before it.

### The transformer on the tape — ✅ done, and the premise was wrong

This roadmap said `TransformerModel` "still carries hand-derived gradients". It does not — it never
carried any. `LayerNorm`, `DenseLayer`, `MultiHeadAttention`, `TransformerEncoderLayer` and
`TransformerModel` all define `Forward` and nothing else. There was no backward pass to replace,
which is why a fresh model stayed randomly initialised for ever: nothing could move a weight.

So the work was not a refactor but a capability that did not exist. `TransformerTape` is the same
architecture on the tape, and `TransformerClassifier` trains it end to end on labelled text.

It needed **one** new tape primitive, `SliceColumns` — the inverse of `ConcatColumns`, for cutting
a wide projection into per-head slices. Everything else composed from what the graph networks
already needed:

| Layer | Built from |
|---|---|
| Layer normalisation | `Sum(axis)`, `Reshape`, `Sqrt`, broadcast arithmetic |
| Row softmax | `Exp`, `Sum(axis)`, a detached row maximum |
| GELU | `Tanh`, `Pow` |
| Multi-head attention | `MatMul`, `Transpose`, `SliceColumns`, `ConcatColumns` |
| Embedding lookup | `Gather` — so a repeated token accumulates both occurrences' gradient |

The layer-norm and attention-softmax backward passes are the two most commonly got wrong by hand;
neither was written down at all. Every layer is checked against central finite differences, and the
tape version reproduces the forward-only implementation to 1e-10, so the architecture is provably
unchanged.

**Scope is unchanged.** This does not make the library a deep learning framework and does not
produce pretrained weights — see *Non-goals*. A model trained here learns from the corpus it is
given, and for a small labelled set TF-IDF plus a linear model remains the better baseline. What
changed is that the architecture is trainable at all.

### Distributed training
Data-parallel training across processes for the tree ensembles and GNNs, which are the two places
where a single machine runs out first.

---

## v0.4 — Breadth

### GraviNum
Einstein summation, more `Slice` ergonomics, complex number support.

**FFT — ✅ done.** `Signal.Fft` transforms **any length**: radix-2 Cooley-Tukey for powers of two,
Bluestein's algorithm for everything else, so a prime length is still `O(n log n)`. That second
path is the part worth having — padding to a power of two changes the spectrum, smearing each peak
across neighbouring bins, so a library that padded silently would answer a question nobody asked.

Also `ForwardReal` (the `n/2+1` distinct bins of a real signal), `FrequencyBins`, and an
FFT-based `Convolve` that pads to `n + m - 1` so the result is linear rather than circular.

Verified against **NumPy's `rfft`** — pocketfft, a separate implementation — to 5e-14 relative at
lengths 64, 100, 101, 360 and 1531. At 4096 it beats the direct `O(n²)` DFT by 3,536×.

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

- **Test coverage.** 589 tests today. Every bug found gets a regression test — that is how the
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

