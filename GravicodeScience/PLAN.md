?# PLAN — Gravicode.Science roadmap

Status of the current release is in [Progress.md](Progress.md). This file is about direction.

---

## Where the project is

**v0.1.0 — feature complete against the original specification.** All six libraries, their
samples, notebooks, benchmarks and tests exist and run. 962 tests pass. Documentation is
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

### Arrow — done in v0.5
Zero-copy interchange with pandas, Polars and Spark via the Arrow memory format. `DataFrame` is
already columnar, so this is mostly a matter of matching buffer layouts.

---

## v0.3 — Scale

### Out-of-core dataframes — done in v0.5
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

### Distributed training — done in v0.5 (forests; GNNs still open)
Data-parallel training across processes for the tree ensembles and GNNs, which are the two places
where a single machine runs out first.

---

## v0.4 — Breadth — ✅ complete

Every item below is implemented, tested and documented in both languages. The suite went from 589
to **962 tests**.

### GraviNum — ✅ done

**Einstein summation.** `Einsum.Evaluate` covers matrix products, transposes, traces, diagonals,
axis reductions, outer products, Frobenius products and batched contractions in one notation, with
NumPy's implicit-output rule. A nested-loop evaluator rather than an optimising one — for two
operands, which is nearly every real use, there is no ordering choice to make.

**Complex arrays.** `ComplexNdArray` over a flat interleaved `Complex` buffer, so the layout matches
FFTW and NumPy and goes to `Fft` without a repack. `ConjugateTranspose` and the Hermitian `Inner`
are provided alongside their plain counterparts precisely because the two are easy to confuse and
the wrong one produces plausible-looking nonsense. Separable 2-D transforms. Unlike `NdArray`,
reshape and transpose copy: complex arithmetic does six flops per element, so the copy no longer
dominates.

**Slice ergonomics.** `SliceOps` adds ellipsis slicing, `TakeAlong` on an arbitrary axis, `AxisAt`
and `AxisRange` views, three-argument `Select`, masked writes, `IndicesWhere`, `FilterRows` and
`Clip`, plus named shorthands. `NdArray.Assign` already handled broadcast assignment through a view
and was left alone.

**FFT — done earlier.** `Signal.Fft` transforms **any length**: radix-2 Cooley-Tukey for powers of
two, Bluestein's algorithm for everything else. Verified against NumPy's `rfft` — pocketfft, a
separate implementation — to 5e-14 relative. At 4096 it beats the direct DFT by 3,536×.

### GraviFrame — ✅ done

**Window functions.** `Rank`, `CumulativeSum`, `RollingMean`, `Lag` and `Lead`, all partitioned. The
tests check partition boundaries specifically, because that is where the leak lives: without
partitioning, `Lag` makes each group's first row reach into the previous group's last, which quietly
inflates a model's score and is invisible in the output. `RollingMean` leaves incomplete windows
missing rather than averaging over what is there.

**As-of join.** Backward-only by design — matching the *nearest* row in either direction is
look-ahead, and is how a backtest ends up predicting the past. Sorts the right frame itself rather
than demanding sorted input, and supports a staleness tolerance.

**Categorical columns.** `CategoricalSeries` with dictionary encoding, ordered categories,
one-hot with an optional dropped baseline, and category reordering that remaps codes rather than
relabelling them. The categories are part of the column's *type*: `Take` keeps every category, and
writing an unlisted value throws rather than widening the dictionary.

**SQL.** `SqlReader`/`SqlWriter` against `System.Data.Common`, so any ADO.NET provider works with no
package dependency. Parameterised throughout; table and column names are checked as plain
identifiers since they cannot be parameterised. Tested end to end against SQLite, and `FromReader`
additionally against a `DataTable` — a second, independent `IDataReader`.

**Excel.** `ExcelReader`/`ExcelWriter` for `.xlsx` with no spreadsheet library, since the format is
a zip of XML the BCL already reads. Handles the three things that bite a first-time reader: absent
cells are gaps not blanks, dates are numbers marked only by a style, and the 1900 leap-year bug.
Tested against workbooks assembled by hand from the spec, not only round trips.

### GraviLearn — ✅ done

**Permutation importance** with per-feature standard deviations, and a documented warning that
correlated features share the blame.

**Shapley values**, exact by subset enumeration below twenty features and Monte Carlo above. Pinned
against the closed form for linear models — `wᵢ(xᵢ − E[xᵢ])` — and against the three axioms:
efficiency, symmetry, and the dummy property.

**Calibration.** Reliability curves, expected calibration error and the Brier score, plus
`IsotonicRegression` by pool-adjacent-violators. The backward-cascading merge is the whole algorithm
and is tested against a brute-force search over monotone fits.

**Imbalanced data.** Over- and under-sampling, SMOTE, and class weights. Tests verify that synthetic
points stay inside the minority region and that features and targets survive the shuffle aligned.

**One-class SVM** by SMO on the nu-formulation, checked directly against Schölkopf's nu-property at
four values. Found and documented two real behaviours: the default tolerance had to drop to 1e-6 for
the property to hold, and an isolated *training* point lands on the boundary because it appears in
its own decision function.

**HDBSCAN.** Core distances, mutual reachability, MST, condensed tree, stability selection. The
central test runs DBSCAN across a sweep of `eps` and asserts that **none** of them recovers a
varying-density dataset that HDBSCAN gets right — so the comparison is demonstrated rather than
asserted.

### GraviText — ✅ done

**BPE**, trained from a corpus, with the classic `low`/`lower`/`newest`/`widest` example as the
reference for the first merges. Merges are applied by rank rather than position, and saved rather
than the vocabulary, because the order *is* the model.

**Unigram (SentencePiece)** — a genuinely different algorithm: EM with iterative pruning, Viterbi
segmentation, and sampling for subword regularisation. Whitespace is encoded rather than split on,
which makes it exactly reversible and lets it handle text with no spaces at all.

**Linear-chain CRF.** Viterbi decoding, exact forward partition, forward-backward marginals, and
gradient training on observed-minus-expected counts. The partition function and the marginals are
checked against **brute-force enumeration** of every labelling on short sequences. BIO constraints
are enforced structurally, including the entity-type match that a prefix-only check misses.

**Trained NER.** Feature-based tagger — word shape, affixes, capitalisation, neighbours — feeding a
CRF. **97.9% entity F1 on a held-out split** of the new `ner_conll.txt` corpus. Tested for the claim
that matters: names appearing nowhere in the corpus are still recognised, from shape and context.

**Decoder stack.** Causal self-attention, a language-model head, and greedy/top-k/nucleus sampling
with a repetition penalty. The central test changes a later token and asserts no earlier hidden
state moved — the one property that cannot be seen by looking at generated text.

### GraviGraph — ✅ done

**Heterogeneous graphs** with per-type features, per-relation edge lists, edge feature matrices, and
`RelationalConvolution` (R-GCN). Normalisation is per relation, so three `bought` edges are not
drowned out by a thousand `viewed` ones.

**Temporal graphs** with time-respecting reachability, snapshots, windows, time-decayed features and
a `TemporalEfficiency` measure of how much a static view overstates. The reachability test is the
argument for the whole type: statically 0 reaches 3, temporally it cannot.

**Graph-level pooling and classification.** Mean, sum, max, mean-max and attention pooling, all
permutation-invariant — tested by shuffling the node order. `GraphClassifier` separates cycles from
complete graphs and generalises to sizes it never saw.

**Neighbourhood sampling.** GraphSAGE-style bounded fan-out per hop, with the block built outward
and reversed. Tested on a hub with two thousand neighbours: the block stays under six nodes.

### GraviProb — ✅ done

**Multivariate normal** through a single Cholesky factor for density, determinant and sampling, with
closed-form conditioning. **Dirichlet** with conjugate updating and Beta marginals, and
**Multinomial** sampled as a chain of binomials so the counts sum exactly. Each is pinned against
the scalar distribution it reduces to.

**Gaussian processes** with RBF, Matérn (ν = 1/2, 3/2, 5/2), periodic and sum kernels; posterior
sampling, credible intervals, log marginal likelihood and grid-search hyperparameter selection. The
posterior is checked against the closed-form Gaussian conditional computed independently through
`MultivariateNormal`.

**Kalman filter and RTS smoother**, with local-level and local-linear-trend constructors, forecasting
and simulation. The scalar steady-state gain is pinned against the closed-form root of the Riccati
equation. Uses the Joseph form, because the short update drifts into an asymmetric covariance over a
long series and then diverges with no warning.

**WAIC and PSIS-LOO** with Pareto-k diagnostics and paired-difference standard errors. Both are
pinned on the degenerate case where they must reduce exactly to the log likelihood, and on
hand-computed values — the `lppd` is a log-of-mean, and writing it as a mean-of-logs gives a number
in the right range that is not WAIC.

---

## v0.5 — Consolidation

v0.4 finished the breadth work, and three items from earlier milestones are still open. They are
carried here rather than left orphaned in a version that is otherwise complete, because each one is
a real piece of work and none of them was dropped for a reason.

### Migrating the six libraries onto `NdArray<T>` — not started, and the estimate was wrong

This was described as "a scheduling decision rather than a risk". Measuring it says otherwise, and
the correction matters more than the original claim did.

**Surface coverage is 22%, not near-complete.** Across `NdArray`, `LinAlg`, `Decomposition`,
`Statistics` and `UFunc` the `double` path exposes 153 public members; `NdArray<T>` and `UFunc<T>`
expose 35, of which 33 overlap. Missing from the generic side: every decomposition, every statistic,
broadcasting, concatenation, slicing helpers, and even the arithmetic operators — `NdArray<T>` has
no `operator +`. So the migration is not "change the type parameter"; it is reimplementing most of
GraviNum generically first.

**The performance claim was measured on the wrong sample.** Re-measured, interleaved in one process,
best-of-25 after a 50-call warm-up:

| operation | n | double | generic | ratio |
|---|---:|---:|---:|---|
| `Add` | 100k | 0.244 ms | 0.211 ms | **0.86×** |
| `Add` | 1M | 2.019 ms | 2.034 ms | **1.01×** |
| `Exp` | 100k | 0.361 ms | 0.436 ms | **1.21×** |
| `Exp` | 1M | 1.906 ms | 2.232 ms | **1.17×** |

Simple element-wise work is free, sometimes faster. **Transcendentals are consistently 17–21%
slower** — `T.Exp` goes through the generic math interface and does not inline the way `Math.Exp`
does. The original "0.91–1.04×, inside noise" came from the arithmetic kernels alone; it is not a
statement about the whole surface, and `Exp`/`Log` are hot in exactly the places that matter
(softmax, logistic loss, every likelihood in GraviProb).

**Scale.** 1,846 `NdArray` references across the six libraries — GraviLearn 624, GraviProb 223,
GraviGraph 164, GraviText 115, GraviFrame 19, and 701 inside GraviNum itself.

So the honest ordering is: grow `NdArray<T>` to parity with the `double` path first, decide what to
do about the transcendental gap, and only then migrate callers. Attempting the migration against a
22% surface would mean writing both paths at once, and the half-migrated state would be worse than
either end of it. This is not deferred for lack of time; it is deferred because the preceding work
has not been done.

`Single.SingleKernels` stays regardless — it is the hand-written baseline the generic version was
measured against, and deleting the comparand would make the next measurement meaningless.

### Arrow interchange — ✅ done

`ArrowFile.Read`/`Write` handle the Arrow IPC file format, verified **in both directions against
pyarrow**: 21 checks in `tools/verify/arrow_interop.py` covering every integer width and sign, all
four timestamp units, null handling, the empty-string-versus-null distinction, multi-byte text, and
a round trip through pandas.

Arrow's metadata is FlatBuffers rather than Protobuf, so none of the ONNX writer's machinery
transferred and `Io/FlatBuffers.cs` is a small encoder and decoder written for it. Three things cost
real time and are now documented where they bite:

- **A FlatBuffers table must start 4-byte aligned**, because it begins with an `int32` soffset. A
  table whose last field was a `short` finishes two bytes out. Every offset still resolves, every
  structural decode by hand still succeeds, and Arrow's verifier rejects the file with nothing to
  say about why. This was the expensive one.
- **`Finish` aligns the whole buffer to the largest element written**, not to 4. Missing that
  leaves an 8-byte-struct buffer 4-aligned.
- **An empty vector is not an absent one.** Arrow's C++ reader walks `field->children()` without a
  null check, so a `Field` that omits the vector fails verification.

Not done, and stated rather than implied: one record batch per file, no dictionary-encoded arrays on
the wire, no compression, no nested types. A `CategoricalSeries` is written as its expanded strings
— dictionary batches are a separate message type whose bookkeeping is easy to get wrong in a way
that loads with silently wrong values.

The reads are also not zero-copy: buffers are decoded into `Series` arrays rather than pointed at.
Genuine zero-copy needs `NdArray` to be constructible over borrowed memory, which is the
`NdArray<T>` migration below.

### Out-of-core dataframes — ✅ done

`ChunkedFrame` reads a source in bounded pieces; `Streaming` provides the single-pass operations
over it — `GroupBy`, `Describe`, `Filter`, `FilterToFile` and `SortToFile`. Each is pinned against
the in-memory implementation on data small enough for both, because a streaming aggregate that
disagrees with `GroupedDataFrame` is wrong whatever it does on data only one of them can handle.

What this is **not** is a lazy query engine: no optimiser, no predicate pushdown, no plan. A chunked
frame is an `IEnumerable<DataFrame>` and the operations are hand-written passes. That keeps the cost
model obvious, which matters more here than cleverness — the reason to reach for this is that the
data does not fit, and a surprise materialisation defeats the whole point.

Three decisions worth recording:

- **Column types are inferred once from a sample and pinned for every chunk.** Inferring per chunk
  is simpler and is a trap: a column that parses as numeric early and turns textual later comes back
  with different types in different chunks, so the same query gives different answers at different
  chunk sizes. The chunk size is a memory knob, not part of the question.
- **`GroupBy`'s memory is proportional to the number of distinct groups, not the input.** That is
  the whole trick and the whole limitation. `CountGroups` exists so the ceiling can be checked
  before the query is run.
- **Median is refused rather than offered.** It needs the values, so a streaming version would keep
  them all and only look like it was streaming.

`SortToFile` is the classic two-phase external merge sort — sort each chunk into a run, then merge
by repeatedly taking the smallest head, so the merge holds one row per run. It needs disk roughly
equal to the input and makes two full passes, so it should not be reached for when the data fits.

### Distributed training — ✅ the coordination layer is done

`GraviLearn.Distributed` provides the parts that are the same whatever is being trained:
`DataParallel.Partition`, `DataParallel.AverageGradients`, a `ParameterServer` for synchronous
rounds, and two transports behind one interface.

**`DistributedForest` is bit-identical to single-process training.** Not "equivalent" —
identical. `RandomForestClassifier` seeds tree `t` from `seed + t * 7919`, a function of the global
index alone, so a worker handed trees 40–79 grows exactly the trees a single process would have
grown at those positions. Tested across 1, 2, 3, 5, 8 and 24 workers, comparing predicted
*probabilities* rather than labels, because equal labels could survive a small difference and equal
probabilities could not.

**`FileTransport` genuinely crosses a process boundary**, verified by spawning real worker
processes rather than threads — `tools/verify/DistributedInterop`. With 2, 4 and 7 workers (the
last giving deliberately uneven shards of 143×6 + 142), the aggregate matches the single-process
answer to ~1e-17, which is the few ulps that regrouping the additions costs. The in-process tests
could never have established this: threads share a heap, so a non-atomic handoff would still look
correct.

Two decisions that are correctness conditions rather than polish, both pinned by tests:

- **Gradients are weighted by sample count, not averaged plainly.** A plain average of per-worker
  means equals the global mean only when every shard is the same size, and `Partition` produces
  uneven shards whenever the count does not divide. Unweighted, the small shards are silently
  over-weighted and the model trains to something slightly wrong that no shape or convergence check
  would catch.
- **Results are collected in worker order, not arrival order.** Floating-point addition is not
  associative, so summing as results arrive makes the answer depend on scheduling and two runs of
  the same job differ in the last bits.

The transport is deliberately the least clever thing that works: a worker writes its payload and
renames it into place, the collector polls for finished names. No broker, no ports, no protocol —
and it runs across machines sharing a filesystem. The rename is what makes it safe, since writing
directly to the final name lets a collector read a half-written file.

**Not done: distributed GNN training.** The pieces are in place — the v0.4 neighbourhood sampler
produces bounded independent per-batch computation graphs, which is exactly a worker's unit of work,
and `ParameterServer` averages what comes back. What is missing is the loop that drives them, and
adding it without a multi-machine setup to test against would be writing code that has never run in
the configuration it exists for.

### Pretrained weights — ✅ the loader is done

`TransformerCheckpoint.Load` fills **every** parameter from an ONNX checkpoint: embeddings, the
embedding layer norm, and per layer the four attention projections, both feed-forward layers and
both layer norms.

This closes a gap that was worse than it looked. `LoadOnnxWeights` already existed and loaded the
embedding *tables* — enough to look up a word vector, and not enough to run the model. Every
attention and feed-forward weight stayed randomly initialised, so a model that reported
`HasPretrainedWeights == true` still produced noise shaped like a sentence.

**Verified against an independent NumPy implementation of the same encoder**, agreeing to
**2.6e-07** — float32 precision, which is what float32 initializers widened to float64 should give.
That checks the whole stack at once: embeddings, both layer norms, all four projections, the
residuals, GELU and the attention softmax. A load that is *nearly* right agrees on nothing.
See `tools/verify/checkpoint_interop.py`.

Three things the design has to refuse rather than absorb, each pinned by a test:

- **A wrong transpose.** PyTorch's `nn.Linear` stores (out, in) and `DenseLayer` stores (in, out).
  A BERT attention projection is 768×768 — square, so both readings are consistent and the mistake
  loads silently. The feed-forward weight is 768×3072 and is not, so that is what the orientation
  is checked against before anything is written.
- **A partial checkpoint.** Strict mode throws; lenient mode loads and reports exactly what is
  missing. Either way `HasPretrainedWeights` stays false, because a model with three of twelve
  layers loaded produces output that is neither the checkpoint's nor a random model's, and nothing
  downstream could tell.
- **A mismatched architecture.** Shapes are checked rather than trusted.

Naming is data, not inference: `CheckpointNames` carries the templates, with `HuggingFaceBert`,
`Unprefixed` and `Reprefixed` covering the common conventions, and `Inspect` lists what a file
actually holds.

What remains is not code: **no weights are shipped**. Licensing and size keep a real checkpoint out
of the repository, so the loader is verified against a synthetic one and a user brings their own
export. The tokenizer half is already in place — `BpeTokenizer.Load` and `UnigramTokenizer.Load`
read the formats published tokenizers ship in.

### Sparse paths — ✅ done, and the premise was half wrong

The claim here was that "nothing above `GraviNum` uses `SparseMatrix`". That was wrong, and I wrote
it: `GraviGraph`, `GraviText` and the autodiff tape all used it already, and the GCN had been
propagating through sparse SpMM since v0.3. Measured on Cora at 64 features:

| | time | vs dense |
|---|---:|---|
| sparse SpMM | 0.30 ms | **93.6×** |
| dense GEMM | 28.15 ms | — |

Identical to 1e-10. So the graph side needed nothing.

The real gap was one layer up: **no `GraviLearn` model could train on a sparse matrix**, and
`TfidfVectorizer` had no sparse output at all — only `CountVectorizer` did. That is the memory wall
for text, because a realistic vocabulary densifies into something that does not fit.

`SparseLogisticRegression` now trains directly on CSR, and `TfidfVectorizer.TransformSparse` feeds
it. On 4,000 documents over an 8,000-word vocabulary at 0.37% density:

| | value |
|---|---|
| matrix, dense | 244.1 MB |
| matrix, CSR | 1.4 MB (**176×**) |
| fit, dense | 12,802 ms |
| fit, sparse | 67 ms (**191×**) |
| accuracy | 99.52% either way |
| largest coefficient difference | 6.9e-18 |

The same model, not an approximation — which is the claim worth making, because a sparse optimiser
that reached a *different* answer would be a different algorithm rather than a faster one.

Two things the implementation had to get right and which are pinned by tests. The L2 penalty is
applied to the accumulated gradient rather than by decaying every weight each step: decaying is
`O(features)` per update and puts the dense cost straight back in. And the weight vector stays
dense, so this bounds the **feature count**, not the document count — 30,000 doubles is nothing,
but it is worth knowing which axis is free.

Quantisation stays speculative until there is a pretrained model to quantise.

---

## Continuous

These are not versioned; they run alongside everything above.

- **Test coverage.** 962 tests today. Every bug found gets a regression test — that is how the
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

