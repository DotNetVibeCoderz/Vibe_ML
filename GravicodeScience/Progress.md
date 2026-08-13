# Progress — Gravicode.Science

**Release**: v0.2.0-dev · **Target framework**: .NET 10 · **Tests**: 589 passing, 0 failing

Roadmap: [PLAN.md](PLAN.md)

---

## Summary

| Area | Status | Notes |
|---|---|---|
| Solution and build | ✅ Complete | 24 projects, central package management, Release build clean |
| GraviNum | ✅ Complete | 236 tests |
| GraviFrame | ✅ Complete | 56 tests |
| GraviLearn | ✅ Complete | 76 tests |
| GraviText | ✅ Complete | 88 tests |
| GraviGraph | ✅ Complete | 62 tests |
| GraviProb | ✅ Complete | 71 tests |
| Sample apps | ✅ Complete | 6 apps, all run end to end |
| Notebooks | ✅ Complete | 6 notebooks, JSON validated |
| Benchmarks | ✅ Complete | 6 suites; GraviNum measured and published |
| Datasets | ✅ Complete | 4 real, 3 generated |
| Documentation | ✅ Complete | 10 pages × 2 languages |
| Screenshots | ✅ Complete | 6, rendered by the samples |

---

## Libraries

### GraviNum — 236 tests

- [x] `NdArray` with shape, strides and offset; views for reshape, transpose, slice
- [x] Slicing with index, range, step and reverse selectors
- [x] NumPy broadcasting with clear errors on incompatible shapes
- [x] Ufuncs with SIMD, threaded and strided-broadcast dispatch
- [x] `LinAlg`: dot, solve, inverse, pseudo-inverse, least squares, rank, condition, norms, Kronecker
- [x] Decompositions: LU, QR (Householder), Cholesky, SVD (Golub–Kahan bidiagonal QR), symmetric eigen (Householder tridiagonal + QL), general eigenvalues (Hessenberg + shifted QR)
- [x] Partial SVD: `SvdRightVectors` and `SingularValues` skip the factors a caller does not need
- [x] `GraviRandom`: xoshiro256++, 12 distributions with exact sampling algorithms
- [x] `Statistics`: moments, quantiles, axis reductions, correlation, covariance, histogram
- [x] `SparseMatrix` CSR with builder, transpose, SpMV and SpMM
- [x] IO: CSV, JSON, binary `.gnb`, memory-mapped arrays
- [x] Compute backends: CPU (SIMD + TPL), GPU (ILGPU), and an optional native BLAS/LAPACK when the machine has one
- [x] Native `dgesv`, `dgeqrf`/`dorgqr`, `dgesvd`, `dsyev`, `dgetrf`, `dpotrf` — every factorisation has a native path
- [x] Packed cache-blocked matrix product, for machines without a BLAS
- [x] `Io.OnnxReader` — dependency-free ONNX weight import
- [x] `Single.SingleKernels` — single-precision prototype for the two hot kernels
- [x] `Generic.NdArray<T>` and `UFunc<T>` over `IFloatingPointIeee754<T>` — one implementation, both widths
- [x] `NdArrayConvert` bridges the generic core to the `double` `NdArray` the six libraries use
- [x] `Signal.Fft` — radix-2 and Bluestein, so any length is `O(n log n)`; real transform, frequency bins, FFT convolution
- [x] Reverse-mode autodiff: `Tensor` tape, broadcasting-aware gradients, `GradientCheck`
- [x] Graph-shaped tape ops: `SparseMatMul`, `Gather`, `SegmentSum`, `ConcatColumns`, `LeakyRelu`, masked `SoftmaxCrossEntropy`

**Performance note.** `LinAlg.Dot` was rewritten during development from a per-row `Axpy` helper
to four-row register blocking with an inlined SIMD loop: **0.62 → 24.5 GFLOP/s** at 512×512, with
identical results.

**v0.2 element-wise and matmul.** The parallel element-wise path stopped copying its operands
(**3.06×**, and now ahead of NumPy); `Unary` gained a parallel path; `MathUtil.Tanh` goes through
`Math.Exp` (**1.6×**); and `LinAlg.Dot` holds its C tile in registers across a slice of `k`
(**1.7×** at 128, 1.05–1.4× from 512 up).

**v0.2 decompositions.** SVD and symmetric eigen moved off Jacobi onto Householder reduction plus
a shifted QR/QL iteration: symmetric eigen **2,740 → 116 ms** (23.6×) and SVD **1,795 → 328 ms**
(5.5×) at 256×256, closing the gap to NumPy from 139× and 66× to 5.9× and 12.1×. PCA fell
**375 → 63.9 ms**, partly from the new SVD and partly from no longer accumulating the left factor
it never reads. `SvdJacobi` and `SymmetricEigenJacobi` stay as the reference the tests check
against — two unrelated routes to the same factorisation.

### GraviFrame — 56 tests

- [x] Typed columns: numeric, text, boolean, timestamp
- [x] CSV reader with type inference, quoting, configurable missing tokens
- [x] Memory-mapped CSV reader
- [x] Parquet read and write, including column subsets
- [x] Filter, sort (single and multi-key), select, drop, rename, sample
- [x] Missing values: drop, fill, forward/backward fill, interpolate
- [x] GroupBy with composite keys and 9 named aggregates plus custom reducers
- [x] Pivot, melt, one-hot, pivot table
- [x] Inner, left, right and outer joins; multi-key joins; concat
- [x] Time series: shift, diff, percent change, rolling, expanding, EMA
- [x] Calendar resampling at six frequencies
- [x] Describe, correlation matrix, GraviNum interop

### GraviLearn — 76 tests

- [x] Scalers: standard, min-max, robust, normalizer
- [x] Imputation, label and one-hot encoding, polynomial features
- [x] PCA, LDA, t-SNE
- [x] Linear: OLS, ridge, lasso (coordinate descent), logistic (one-vs-rest), linear SVM
- [x] Trees: CART classifier and regressor with feature importances and text rendering
- [x] Ensembles: random forest with out-of-bag scoring, gradient boosting with Friedman's Newton step
- [x] kNN classifier and regressor, four distance metrics
- [x] Gaussian and multinomial naive Bayes
- [x] Clustering: k-means++ with restarts, DBSCAN, agglomerative, Gaussian mixture with BIC
- [x] Metrics: 20+ classification, regression and clustering measures, classification report
- [x] `Pipeline` with leakage-safe cross-validation
- [x] Train/test split, k-fold, stratified k-fold, grid search
- [x] Dataset loaders and generators
- [x] JSON model persistence
- [x] `OnnxExport` — writes a fitted affine pipeline as ONNX, verified against Python's onnxruntime

### GraviText — 88 tests

- [x] Whitespace, regex, character and WordPiece tokenizers; sentence splitter
- [x] WordPiece vocabulary training
- [x] Accent folding that works under `InvariantGlobalization`
- [x] Bilingual stop words; Porter and Nazief-Adriani stemmers; lemmatizer; n-grams
- [x] Language detection
- [x] Count and TF-IDF vectorizers with document-frequency bounds, sparse output
- [x] Similarity: cosine, Jaccard, Levenshtein
- [x] Word2Vec skip-gram with negative sampling and subsampling
- [x] GloVe
- [x] Common-component removal for embeddings
- [x] Transformer: multi-head attention, layer norm, GELU feed-forward, residuals, encoder stack
- [x] `TransformerTape` — the same encoder on the autodiff tape, every layer gradient-checked
- [x] `TransformerClassifier` — trains the encoder end to end on labelled text
- [x] Task pipelines: sentiment (lexicon and supervised), classification, NER, TextRank summariser, keywords

⚠️ **No pretrained weights.** Documented at the top of [GraviText.md](docs/GraviText.md) and printed
by the sample at runtime. The architecture can now be *trained* on your own labelled text via
`TransformerClassifier`; that is not the same as shipping BERT, and TF-IDF plus a linear model
remains the better baseline on a small dataset.
⚠️ **NER is rule-based.** Documented in the same places.

### GraviGraph — 62 tests

- [x] Adjacency-list graph, directed and undirected, with node features and labels
- [x] Sparse and dense adjacency, normalised propagation matrix, Laplacian
- [x] Subgraph, `AsUndirected`, four generators
- [x] JSON format with sparse node features; edge-list reader
- [x] BFS, DFS, hop distances, Dijkstra
- [x] PageRank with dangling-mass redistribution; personalised PageRank
- [x] Degree, closeness, betweenness (Brandes) and eigenvector centrality
- [x] Weak and strong connected components, triangles, clustering coefficient, topological sort
- [x] Label propagation and modularity
- [x] Random walks: uniform and node2vec-biased
- [x] DeepWalk and node2vec
- [x] GCN, GraphSAGE and GAT all trained on the autodiff tape — forward pass only, no hand-derived gradients
- [x] GAT's attention softmax composed from `Gather`/`SegmentSum`, which are adjoints of each other
- [x] `GnnTape` layer helpers and `TapeAdam`, public so a new architecture needs no library change

### GraviProb — 71 tests

- [x] 12 distributions with log densities, sampling, moments and support bounds
- [x] `BayesianModel` with a fluent API and latent-variable references in likelihoods
- [x] Metropolis-Hastings with warmup-only adaptation to 0.234 acceptance
- [x] Metropolis within Gibbs
- [x] Hamiltonian Monte Carlo and NUTS, with dual-averaging step-size adaptation
- [x] Differentiable log densities, and `LogPosteriorGradient` from one tape pass
- [x] Parallel chains
- [x] Mean-field variational inference with support transforms and log Jacobians
- [x] Posterior trace: HDI, credible intervals, R-hat, effective sample size, summary table
- [x] Prior and posterior predictive checks
- [x] Bayesian networks with exact enumeration and ancestral sampling
- [x] Hidden Markov models: forward, Viterbi, forward-backward, Baum-Welch
- [x] Conjugate Bayesian linear regression with predictive intervals

---

## Bugs found and fixed during development

Each is now covered by a regression test.

| Bug | Cause | Fix |
|---|---|---|
| **Decision-tree split was O(n²)** | `FindBestSplit` materialised `sorted[..k]` / `sorted[k..]` and rebuilt a count dictionary at every candidate split point | Sweep the sorted order with running class counts; **83 min → 1.9 s** on the 20k-sample forest, and it now beats scikit-learn |
| k-means 20× slower than scikit-learn | The distance loop went through `NdArray`'s two-index accessor, paying stride maths and bounds checks per feature | Run Lloyd's loop on the raw contiguous buffers; 9.9 s → 2.5 s |
| A comparison benchmark measured nothing | The scalar log-density result was discarded, so the JIT deleted the loop (0.45 ms for 1M logarithms) | Route values through a non-inlined sink; real figure 2.08 ms |
| Memory-mapped CSV read one extra row | `CreateViewStream(0, 0, …)` rounds the mapping to the page size; trailing NUL bytes decoded as a line | Pass the exact file length |
| Cora reported 1,550 components, largest 77 nodes | `ConnectedComponents` followed out-edges only on a directed graph | Weak connectivity; added `StronglyConnectedComponents` |
| Node embeddings put same-topic papers *further* apart | Random walks on a directed citation graph strand after one step | Added `Graph.AsUndirected()` |
| Every word similarity ≈ 1.00 | Trained vectors share a dominant common direction | Added `RemoveCommonComponent()` |
| Variational inference returned θ = 11.3 for a Beta parameter | The Gaussian family was fitted on the constrained scale | Support transforms with log Jacobians |
| `AggregateException` from a misspecified model | `Parallel.For` wrapped the real exception | Probe for a feasible start before going parallel |
| `GroupBy.Aggregate` with a custom result name threw | The output name was used to look up the source column | Separate source and output names |
| Gradient boosting classifier stalled at 92.5% | Leaf values used the gradient without the curvature | Friedman's Newton step |
| Matrix multiply ran at 0.62 GFLOP/s | Per-row helper call defeated inlining | Four-row register blocking |
| `StripAccents` did nothing | `InvariantGlobalization` disables `String.Normalize` | Explicit folding table |
| Erf inaccurate beyond ~1e-5 | Wrong partial denominators in the continued fraction | Corrected, with a wider series crossover |
| **New QL eigen returned unsorted eigenvalues** | The iteration deflates blocks as they converge, which has nothing to do with magnitude; the wrapper assumed ascending order | Sort explicitly, carrying the eigenvector columns along. Caught by checking against the Jacobi reference — `A V = V Λ` still passed, so only the cross-check found it |
| **Element-wise parallel path copied its own inputs** | `RunBinaryContiguous` called `ToArray()` on both operands and allocated a third array for the result, because a `Span<T>` cannot cross a lambda closure — three extra passes over memory per operation | Pin the buffers and hand the workers pointers. **3.06× faster** on a 1M add (7.23 → 2.36 ms), bit-identical results, and element-wise arithmetic moved from *Python 2–3×* to *.NET 1.8×*. `Unary` had no parallel path at all and gained one |
| **The GCN's hand-derived gradient was ~40% wrong** | The backward pass omitted the dropout mask on the hidden layer, so with dropout active it differed from the true gradient by 4.06e-01 relative. It had been wrong since the layer was written, and the model still trained to a plausible 71% on Cora — which is why review never caught it | Put GCN and GraphSAGE on the autodiff tape: the forward pass is written once and `Backward()` derives the rest. Pinned by `GnnGradientTests`, which checks both against central finite differences with and without dropout. Corrected accuracy is 69.3% |
| **Gradient-based sampling was unusably slow on real datasets** | The tape built a few nodes *per observation*, so a 200-row model put thousands of nodes on the graph for every gradient — and a gradient is evaluated at every leapfrog step of every iteration of every chain | Vectorise each likelihood over the whole dataset, broadcasting the scalar parameters against the data vector. Graph size no longer depends on the data. The GraviProb suite fell from **34 s to 5 s** |

---

## Verified reference results

| Check | Expected | Achieved |
|---|---|---|
| PCA on Iris, first two components | 92.46% / 5.31% | matches |
| Digits, scaler → PCA(30) → kNN(3) | ~98% | 97.96% |
| Titanic random forest | 0.78–0.83 published | above 0.75 required, passes |
| Cora largest weak component | 2,485 nodes | 2,485 |
| Cora GCN vs 30.2% baseline | ~81% published | 69.3% with the corrected gradient (was 71.2% with the buggy one) |
| Cora GraphSAGE / GAT, same split | — | 70.3% / **72.1%** — attention ahead, as expected |
| GCN, GraphSAGE and GAT gradients vs finite differences | agreement to ~1e-6 | passes, including through the attention softmax |
| GAT attention rows sum to 1 | exactly 1 | within 3.3e-16 |
| Transformer tape vs the forward-only encoder | identical architecture | agrees to 1e-10 |
| Transformer layer gradients vs finite differences | agreement to ~1e-6 | passes, layer norm and attention softmax included |
| Native BLAS product vs the managed kernel | identical to rounding | agrees to 1e-13 |
| Native LAPACK factorisations vs their defining property | `A=QR`, `A V = V Λ`, `A x = b` | all agree to ~1e-14 |
| Packed product vs a naive triple loop | exact | agrees to 5e-14 |
| ONNX reader vs files written by the official Python library | exact | all 5 initializer types read correctly |
| ONNX export vs Python onnxruntime | same predictions | 150/150 labels; 5.9e-07 on regression |
| Single-precision 1000-cube product vs double | float32 accuracy | 1.3e-6 relative |
| `NdArray<double>` kernels vs the non-generic `NdArray` | bit-identical | element-wise exact, product to 1e-9 |
| Genericising the `double` path | no slowdown | 0.91–1.04×, inside measurement noise |
| FFT vs NumPy `rfft` (pocketfft) | same spectrum | 5e-14 relative, at prime and composite lengths |
| FFT vs the direct `O(n²)` DFT | same spectrum | agrees; 3,536× faster at n=4096 |
| Coin posterior mean vs exact conjugate | 0.623762 | within 0.002 |
| HMC and NUTS vs the same exact posterior | 0.623762 | within 0.01, mean and sd |
| Autodiff gradients vs central differences | agreement to ~1e-6 | passes on 14 functions |
| Log posterior gradient at the Beta(8,4) mode θ=0.7 | exactly 0 | < 1e-8 |
| Baum-Welch vs generating model | should approach | slightly exceeds, as EM allows |
| LU, QR, SVD, Cholesky reconstruction | exact to 1e-8 | passes |

---

## Known limitations

1. **No pretrained transformer weights.** The architecture is correct; the weights are not
   provided.
2. **NER is rule-based**, so it misses entities outside its gazetteers and trigger patterns.
3. **GPU is opt-in and float64-limited.** On integrated hardware it is slower than the CPU.
4. **Cora GCN reaches 69%, not the published 81%.** No learning-rate schedule, no early stopping,
   and 60 epochs. The gradient itself is now verified against finite differences, so the gap is
   training procedure rather than correctness.
5. **DBSCAN and agglomerative clustering have no out-of-sample prediction.** This is inherent to
   the algorithms; both throw rather than inventing an answer.
6. **Betweenness centrality is `O(VE)`** even with Brandes' algorithm.
7. **Reverse-mode autodiff is slower than finite differences below ~50 parameters.** The tape
   allocates a node per operation, so on a small log posterior it loses to `2d` cheap scalar
   evaluations; it wins by growing dimension, not by being faster per call. Variational inference
   picks its gradient method accordingly.
8. **A likelihood built through `DistributionSpec.From` has no differentiable form**, because the
   resolver is an opaque function of doubles. Such models report `IsDifferentiable == false` and
   the gradient samplers refuse rather than guessing. Use `FromTensor` or the named factories.
9. **`TransformerModel` computes a forward pass only and cannot be trained.** It never carried
   gradients — earlier notes here claiming otherwise were wrong. Use `TransformerClassifier` for
   the trainable version; `TransformerModel` remains the right choice for running a reference
   architecture with loaded weights.

---

## Next

See [PLAN.md](PLAN.md). Every factorisation now has a native path when a LAPACK is present, ONNX
works in both directions, and the generic `NdArray<T>` core is built and measured — genericising
costs nothing, so migrating the six libraries onto it is now a scheduling decision rather than a
gamble. That migration, and the v0.4 breadth items, are what remain.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
