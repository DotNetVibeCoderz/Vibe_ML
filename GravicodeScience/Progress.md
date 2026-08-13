# Progress — Gravicode.Science

**Release**: v0.1.0 · **Target framework**: .NET 10 · **Tests**: 400 passing, 0 failing

Roadmap: [PLAN.md](PLAN.md)

---

## Summary

| Area | Status | Notes |
|---|---|---|
| Solution and build | ✅ Complete | 24 projects, central package management, Release build clean |
| GraviNum | ✅ Complete | 95 tests |
| GraviFrame | ✅ Complete | 56 tests |
| GraviLearn | ✅ Complete | 70 tests |
| GraviText | ✅ Complete | 74 tests |
| GraviGraph | ✅ Complete | 53 tests |
| GraviProb | ✅ Complete | 52 tests |
| Sample apps | ✅ Complete | 6 apps, all run end to end |
| Notebooks | ✅ Complete | 6 notebooks, JSON validated |
| Benchmarks | ✅ Complete | 6 suites; GraviNum measured and published |
| Datasets | ✅ Complete | 4 real, 3 generated |
| Documentation | ✅ Complete | 10 pages × 2 languages |
| Screenshots | ✅ Complete | 6, rendered by the samples |

---

## Libraries

### GraviNum — 95 tests

- [x] `NdArray` with shape, strides and offset; views for reshape, transpose, slice
- [x] Slicing with index, range, step and reverse selectors
- [x] NumPy broadcasting with clear errors on incompatible shapes
- [x] Ufuncs with SIMD, threaded and strided-broadcast dispatch
- [x] `LinAlg`: dot, solve, inverse, pseudo-inverse, least squares, rank, condition, norms, Kronecker
- [x] Decompositions: LU, QR (Householder), Cholesky, SVD (one-sided Jacobi), symmetric eigen (Jacobi), general eigenvalues (Hessenberg + shifted QR)
- [x] `GraviRandom`: xoshiro256++, 12 distributions with exact sampling algorithms
- [x] `Statistics`: moments, quantiles, axis reductions, correlation, covariance, histogram
- [x] `SparseMatrix` CSR with builder, transpose, SpMV and SpMM
- [x] IO: CSV, JSON, binary `.gnb`, memory-mapped arrays
- [x] Compute backends: CPU (SIMD + TPL) and GPU (ILGPU)

**Performance note.** `LinAlg.Dot` was rewritten during development from a per-row `Axpy` helper
to four-row register blocking with an inlined SIMD loop: **0.62 → 24.5 GFLOP/s** at 512×512, with
identical results.

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

### GraviLearn — 70 tests

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

### GraviText — 74 tests

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
- [x] Task pipelines: sentiment (lexicon and supervised), classification, NER, TextRank summariser, keywords

⚠️ **No pretrained weights.** Documented at the top of [GraviText.md](docs/GraviText.md) and printed
by the sample at runtime.
⚠️ **NER is rule-based.** Documented in the same places.

### GraviGraph — 53 tests

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
- [x] GCN, GraphSAGE and GAT — all fully trained with hand-derived gradients, GAT including through the attention softmax

### GraviProb — 52 tests

- [x] 12 distributions with log densities, sampling, moments and support bounds
- [x] `BayesianModel` with a fluent API and latent-variable references in likelihoods
- [x] Metropolis-Hastings with warmup-only adaptation to 0.234 acceptance
- [x] Metropolis within Gibbs
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

---

## Verified reference results

| Check | Expected | Achieved |
|---|---|---|
| PCA on Iris, first two components | 92.46% / 5.31% | matches |
| Digits, scaler → PCA(30) → kNN(3) | ~98% | 97.96% |
| Titanic random forest | 0.78–0.83 published | above 0.75 required, passes |
| Cora largest weak component | 2,485 nodes | 2,485 |
| Cora GCN vs 30.2% baseline | ~81% published | 71.2% |
| Coin posterior mean vs exact conjugate | 0.623762 | within 0.002 |
| Baum-Welch vs generating model | should approach | slightly exceeds, as EM allows |
| LU, QR, SVD, Cholesky reconstruction | exact to 1e-8 | passes |

---

## Known limitations

1. **No pretrained transformer weights.** The architecture is correct; the weights are not
   provided.
2. **NER is rule-based**, so it misses entities outside its gazetteers and trigger patterns.
3. **GPU is opt-in and float64-limited.** On integrated hardware it is slower than the CPU.
4. **Cora GCN reaches 71%, not the published 81%.** No learning-rate schedule, no early stopping,
   and 60 epochs.
5. **DBSCAN and agglomerative clustering have no out-of-sample prediction.** This is inherent to
   the algorithms; both throw rather than inventing an answer.
6. **Betweenness centrality is `O(VE)`** even with Brandes' algorithm.
7. **Variational inference uses finite-difference gradients**, which limits it to a modest number
   of parameters.

---

## Next

See [PLAN.md](PLAN.md). The nearest items are BLAS/LAPACK interop, a single-precision path, and
ONNX import — the last of which would also give `TransformerModel` real weights.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
