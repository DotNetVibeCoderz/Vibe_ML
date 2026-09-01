# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

v0.1.0 is implemented and complete against `requirements.md`: 24 projects, ~15.5k lines of library
code, **1,049 tests passing**, six samples that run end to end, six notebooks, six benchmark suites,
and bilingual docs. `requirements.md` remains the specification of record; [Progress.md](Progress.md)
tracks what is done and [PLAN.md](PLAN.md) tracks direction.

Target framework is **.NET 10** (SDK 10.0.400 installed locally). The solution file is
`Gravicode.Science.sln` (classic format — `dotnet new sln` defaults to `.slnx` on .NET 10, so pass
`--format sln` if regenerating).

`tools/ScienceAppGen` is an Avalonia IDE whose assistant ("Jack, the Code Bender") generates
applications via Semantic Kernel function calling. See [docs/ScienceAppGen.md](docs/ScienceAppGen.md).

### Things that will bite you

- **`Compute.AutomaticGpuDispatch` is off by default and should stay that way** unless benchmarked.
  Everything is float64; on integrated GPUs the ILGPU path measured 5–8× *slower* than the CPU.
- **`LinAlg.Dot` uses unsafe four-row register blocking, plus a `k` slice.** The tile of C lives in
  registers across the slice; slicing `k` is what keeps the strided walk through B in cache.
  Register accumulation *without* the slice is slower at 1024+. Column panelling (blocking `j`) was
  measured and is worse at every size — don't re-try it.
- **Element-wise parallel paths pin and pass pointers, never `ToArray()`.** A `Span<T>` cannot be
  captured by a lambda; copying to satisfy that cost three extra passes over memory and made `Add`
  3× slower than it needed to be. `Unary` is parallel too — it wasn't, and that was free money.
- **`MathUtil.Tanh` goes through `Math.Exp` on purpose.** `Math.Tanh` measured ~1.6× slower; the
  sign split keeps the exponent negative so nothing overflows. Agrees with `Math.Tanh` to 2.2e-16.
- **`TensorPrimitives` was measured and rejected.** For `double`, its `Exp` is 0.93× and `Log`
  0.57× against plain `Math` — .NET's scalar intrinsics already win. Only `Tanh`/`Cosh` gained, and
  not enough to justify the dependency. Don't add `System.Numerics.Tensors` without re-measuring.
- **This machine is a 4-core 15W laptop that throttles.** The same build measured 78 ms and 168 ms
  for a 1024-cube matmul within an hour. Never compare timings across separate runs — run both
  variants alternately in one process and take the best of many.
- **`Svd`/`SymmetricEigen` are Householder reduction + shifted QR/QL** (`DecompositionKernels.cs`),
  not the Jacobi methods that share the file. `SvdJacobi`/`SymmetricEigenJacobi` are kept
  deliberately: they reach the same factorisation by a different route, so the tests use them as
  the reference. Don't delete them as dead code.
- **`TridiagonalQl` returns eigenvalues in no particular order** — blocks deflate as they
  converge. `SymmetricEigen` sorts descending and carries the eigenvector columns with them.
  `A V = V Λ` passes either way, so only the cross-check against Jacobi catches a broken sort.
- **`SvdRightVectors`/`SingularValues` skip factors the caller doesn't need.** For a tall matrix
  accumulating `U` is most of the runtime, and PCA never reads it. A *wide* matrix is transposed
  first, which swaps which factor gets skipped — see the branch in `SvdRightVectors`.
- **`NdArray` slicing/reshape/transpose return views over a shared buffer.** Mutating one view
  mutates the others. `Copy()` is the escape hatch.
- **Libraries build with `InvariantGlobalization`**, so `String.Normalize` silently no-ops —
  `TextNormalizer.StripAccents` uses an explicit folding table for this reason.
- **`ConnectedComponents` is *weak* connectivity on directed graphs.** This is deliberate; see the
  Cora numbers in `docs/GraviGraph.md`.
- Parquet is pinned to **Parquet.Net 5.6.1**; 6.x replaced the `DataColumn` API with a lower-level
  `Memory<T>` surface.
- **`DecisionTree.FindBestSplit` sweeps running class counts, deliberately.** Materialising
  `sorted[..k]`/`sorted[k..]` per split point makes it O(n²) — that version ran 83 minutes without
  finishing on 20k samples.
- **`KMeans` runs Lloyd's loop on raw `double[]` buffers**, not through the `NdArray` indexer.
  The indexer's stride maths and bounds checks cost 4× in that loop.
- **The autodiff tape is slower than finite differences below ~50 parameters.** It allocates a node
  per operation; `2d` cheap scalar evaluations beat that on a small log posterior. It wins by
  growing dimension, not per call — `MeanFieldVariational` switches at 32 parameters for this
  reason. Don't "optimise" that back into always using the tape.
- **Likelihoods are vectorised over the whole dataset on purpose.** `DistributionSpec.FromTensor`
  takes the data as one `NdArray` and returns the summed log density. A per-observation version put
  thousands of nodes on the graph per gradient and made the GraviProb suite 7× slower.
- **All three GNN architectures are on the autodiff tape.** Converting the GCN found that its
  hand-derived backward pass had been omitting the dropout mask — a ~40% wrong gradient that still
  trained to a plausible 71% on Cora. Cora is 69.3% now, with the *correct* gradient; don't "fix"
  that number by tuning.
- **`NativeBlas` ships nothing and probes for what's installed.** ILP64 builds export
  `cblas_dgemm64_` and take 64-bit dimensions — using the wrong signature silently corrupts memory,
  so the width is detected. numpy/scipy prefix every symbol with `scipy_`, and on a data-science
  machine that's usually the only BLAS present. Test with and without `GRAVICODE_BLAS` set.
- **`NativeLapack` binds LAPACKE, not Fortran** — the layout argument is what avoids a transpose in
  and out on every call. Three conventions are converted, not assumed: LAPACK sorts eigenvalues
  ascending, returns `V^T`, and in row-major already puts eigenvector *j* in column *j* — so
  transposing it (as Fortran habits suggest) breaks `A V = V Λ` while leaving eigenvalues correct.
- **`OnnxExport` only handles affine steps, and refuses the rest.** Trees/forests/kNN throw rather
  than exporting an approximation. Two things that silently produce a wrong-but-loadable model:
  the batch dimension must be *symbolic*, and PCA components must be stored *transposed*.
  Re-verify with `onnx.checker` + Python onnxruntime after any change to the writer.
- **`dgetrf` reports row *swaps*, not a permutation.** At step `i`, row `i` was exchanged with row
  `ipiv[i]` (1-based); `LuResult.Pivot` is the finished permutation, so the swaps are replayed.
  Reading one as the other gives a valid-looking L and U that reconstructs the wrong matrix — the
  determinant-sign test is what catches it, `P A = L U` alone does not.
- **`dpotrf` leaves the other triangle holding the input**, so it must be zeroed; a positive `info`
  means not positive definite and is turned into the same exception the managed path throws.
- **Benchmark numbers depend on whether a BLAS is installed.** Published figures are the managed
  path. Always say which configuration a number came from; `docs/benchmarks.md` reports both.
- **`PackedMatMul` must stay parallel.** The first version was single-threaded and lost everywhere;
  packing was never the problem. Its threshold (8M multiply-adds) is measured — below it packing
  costs more than it saves.
- **Benchmark warm-up must exceed ~30 calls.** Tiered JIT recompiles a hot method around there, and
  two warm-up iterations produced a 192-cube measuring 8× slower than a 224-cube. Any new kernel
  measurement needs a real warm-up loop.
- **`Single.SingleKernels` is the hand-written baseline, `Generic.NdArray<T>` is the real answer.**
  Keep `SingleKernels` — it's what the generic version was measured against. Don't grow it.
- **Genericising the `double` path costs nothing** (0.91–1.04×, inside noise), which is the whole
  argument for eventually making `NdArray` *be* `NdArray<double>`. Re-measure before assuming that
  still holds after a kernel change.
- **`UFunc<T>` deliberately does not broadcast** and throws instead — the `double` `UFunc` does.
  Its `Sum` is pairwise, which matters far more in `float`: a naive total over 1M × `0.1f` drifts.
- **`Fft` handles any length on purpose** — Bluestein for non-powers-of-two. Don't "simplify" it to
  radix-2 plus zero-padding: padding changes the spectrum, smearing peaks across bins. Its
  Bluestein path uses `t² mod 2n`, not `t²`, or the phase drifts as n grows.
- **`Fft.Convolve` pads to `n + m - 1`** so the convolution is linear; without that the tail wraps
  around and corrupts the start. `DirectDft` is the O(n²) reference the tests check against — keep it.
- **`ComplexNdArray` reshape/transpose *copy*, unlike `NdArray`'s views.** Deliberate: complex
  arithmetic is six flops per element, so the copy stopped being what dominates, and a plain
  contiguous buffer keeps every kernel simple. Its layout is interleaved re/im — the same as FFTW
  and NumPy — which is what lets the buffer go to `Fft` without a repack.
- **`ConjugateTranspose`, not `Transpose`, is what complex formulas mean.** `AᴴA` is positive
  semi-definite with a real diagonal; `AᵀA` is neither, and gives complex "variances" that pass
  every shape check. Both exist because they're easy to confuse. Same for `Inner`, which conjugates
  its *first* operand — without that the "norm" of `[i]` is -1.
- **`SliceOps.Assign(double)` is the only Assign there.** `NdArray.Assign(NdArray)` already existed
  with broadcasting; I duplicated it as an extension and the extension was silently shadowed, so
  every test exercised the original. Check for an existing member before adding an extension.
- **`OneClassSvm`'s tolerance changes the answer, not just the runtime.** At the old 1e-3 default
  the nu-property measurably failed (13.5% outliers at nu=0.05). It's 1e-6 now. Also: an isolated
  *training* point appears in its own decision function, so when ρ < 1/(νn) it scores exactly zero
  and lands on the boundary. Never judge that model by scoring its training set.
- **`DecisionTree`-based tests need real held-out data for `PermutationImportance`.** On the
  training set it measures memorisation; the tests use it that way only where the dataset is a
  deterministic function of one feature.
- **HDBSCAN's condense step must cascade backwards.** So must isotonic regression's PAVA merge.
  Both look right on smooth data and fail on the ragged data they exist for.
- **The CRF's partition function and marginals are pinned against brute-force enumeration**, not
  against each other. A forward algorithm that is subtly wrong still decodes plausible sequences.
- **BPE merges are applied by *rank*, not position.** Left-to-right lets an early low-rank merge
  consume a symbol a higher-rank merge needed. `Save` writes merges rather than the vocabulary
  because the order *is* the model — a token set alone cannot tokenize.
- **`UnigramTokenizer` never prunes single characters**, or text containing them becomes
  unsegmentable. Its EM accumulates expected counts over *all* segmentations via forward-backward;
  using the Viterbi path alone makes rare pieces look worse than they are.
- **`TransformerDecoder` has no KV cache and generation is quadratic**, knowingly. The causality
  test changes a later token and asserts no earlier hidden state moved — that's the only property
  that can't be seen by reading generated text.
- **`TrainedNer` hits ~98% F1 because the corpus is generated.** That says it learned the templates,
  not that it would reach 98% on newswire. Don't quote it as a benchmark.
- **`KalmanFilter` uses the Joseph form covariance update.** Algebraically identical to `P − KHP`
  and numerically much better: the short form drifts into an asymmetric or negative-definite
  covariance over a long series and then diverges silently. Its `Simulate` handles an all-zero Q or
  R specially — zero noise is a valid model and has no Cholesky factor.
- **`GaussianProcess` centres its targets.** A GP has a zero prior mean, so without centring a
  series sitting at 1000 gets predictions dragged towards zero between observations. Noise is not
  optional either: it's what keeps the covariance invertible with near-duplicate inputs.
- **`ModelComparison`'s lppd is a log-of-mean, not a mean-of-logs.** Getting it backwards gives a
  number in the right range that is not WAIC. Both criteria are pinned on the degenerate zero-spread
  case where they must equal the plain log likelihood exactly.
- **`Graph`'s generators are static methods on `Graph` itself** (`Graph.Cycle`, `Graph.Random`),
  not a separate `GraphGenerators` class, and `Random` takes a `seed` int rather than a `GraviRandom`.
- **A SAGE layer takes twice its feature width**, because `NeighborSampler.Aggregate` concatenates
  self and neighbourhood before projecting. Keeping them separate rather than folding self into the
  mean is the difference between SAGE and a plain GCN.
- **`RelationalConvolution` normalises per relation, not overall.** A node with a thousand `viewed`
  edges and three `bought` ones would otherwise lose the purchases entirely, and purchases are the
  signal. Its per-type self-loop is what stops an isolated node's representation being exactly zero.
- **`TransformerModel` is forward-only and always was** — it never had hand-derived gradients to
  replace, whatever older notes claimed. `TransformerTape` + `TransformerClassifier` are the
  trainable version; leave `TransformerModel` alone as the inference path for loaded weights.
- **`Gather` and `SegmentSum` are adjoints** — gathering forwards is summing backwards. That is why
  `GnnTape.SegmentSoftmax` is *composed* from them rather than being its own kernel, and why GAT's
  attention needed no derivation. Don't replace it with a bespoke op; it would need its own
  softmax-Jacobian proof.
- **`GnnTape` and `TapeAdam` are public on purpose** — the point of the tape is that a fourth
  architecture needs no library change.
- **The tape costs ~1.26× per GCN epoch at Cora's shape** and less as matrices grow. That is the
  opposite of its behaviour on scalar log posteriors, and why GNNs were the right place for it.
- **`DistributionSpec.From` produces a model that cannot be differentiated** — the resolver is an
  opaque `Func` of doubles. `IsDifferentiable` reports it and HMC/NUTS throw rather than guess.
- **`TridiagonalQl` and the NUTS tree both look right when they are wrong.** For the eigen sort,
  `A V = V Λ` passes with the values in any order; for a sampler, plausible-looking draws can come
  from the wrong distribution. Both are pinned against independent references (Jacobi, and the
  exact Beta-binomial posterior) for that reason.

- **Notebook plot cells must use `GetPngHtml`, not `GetImageHtml`.** The latter is
  `[Obsolete(error: true)]` in ScottPlot 5.1.59, so a notebook using it fails rather than warns.
  All six notebooks were on the old call until this was caught by compiling their cells.
- **Notebook cells are compile-checked, not just JSON-validated.** Extract every code cell, strip
  `#r` directives, concatenate, and build against the libraries — that is what found the ScottPlot
  break and several wrong method names. .NET Interactive allows redeclaring a variable across
  cells, so the check is stricter than the notebook; rename rather than shadow.
- **The template self-test claim is real**: every `TemplateService` template is generated and built.
  `FindLibrarySource()` walks up from `AppContext.BaseDirectory` *and* `Environment.CurrentDirectory`,
  so a generator run from outside the repository silently emits empty `ProjectReference` paths.

- **Arrow metadata is FlatBuffers, not Protobuf** — the ONNX writer's machinery does not transfer.
  Three things cost real time in `Io/FlatBuffers.cs`, all of which produce files that decode by hand
  and fail Arrow's verifier with no explanation: a table must start **4-byte aligned** (it begins
  with an int32 soffset, so a table whose last field was a `short` finishes two bytes out);
  `Finish` must align to the largest element written, not to 4; and an **empty vector is not an
  absent one** — Arrow's C++ reader walks `field->children()` without a null check.
- **`ChunkedFrame` pins column types from one sample.** Inferring per chunk means the same query
  gives different answers at different chunk sizes, because a column that parses numeric early and
  turns textual later comes back differently typed in different chunks.
- **`Streaming.GroupBy` is bounded by distinct groups, not rows.** `CountGroups` exists to check
  that ceiling first. Median is refused deliberately — it would have to keep every value and only
  look like it was streaming.
- **`SortToFile` writes CSV, so a single-column frame with missing values writes blank lines**, and
  a CSV reader cannot tell those from padding. The rows are in the file and correctly ordered; the
  round trip loses them. Two or more columns survive.
- **`DataParallel.AverageGradients` must be weighted by sample count.** A plain average of
  per-worker means equals the global mean only for equal shards, and `Partition` produces uneven
  ones whenever the count does not divide. Unweighted, the model trains to something slightly wrong
  that no shape or convergence check catches.
- **Transports collect in worker order, not arrival order** — floating-point addition is not
  associative, so arrival order makes the answer depend on scheduling.
- **`DistributedForest` is bit-identical to single-process** because tree `t` is seeded from
  `seed + t * 7919`. If that seeding ever changes, the identity claim goes with it.
- **`SparseLogisticRegression` applies L2 to the accumulated gradient**, not by decaying weights per
  step — decaying is O(features) per update and reintroduces the dense cost. Its weight vector is
  dense, so it bounds the feature count, not the document count. On separable data an unpenalised
  fit never converges (the MLE is at infinity); that is not a bug.
- **`TransformerCheckpoint` checks the transpose against a NON-square matrix.** BERT attention
  projections are 768×768 and accept either reading, so the feed-forward weight (768×3072) is what
  catches a wrong convention. A partial load never sets `HasPretrainedWeights`.
- **The generic core covers 22% of the double path**, not "nearly all" — `NdArray<T>` has no
  arithmetic operators, no decompositions, no statistics. And genericising is free for `Add`
  (0.86–1.01×) but costs **17–21% on `Exp`**, which is hot in every softmax and likelihood. Do not
  repeat the claim that migrating is only a scheduling decision.

### ScienceAppGen specifics

- **Anthropic has no official Semantic Kernel connector** — `AnthropicChatCompletionService` is
  hand-written against the Messages API. Google and Ollama connectors are alpha-only packages.
- **gpt-5 / o1 / o3 / o4 reject `max_tokens`** and require `max_completion_tokens`; SK 1.79 still
  emits the legacy field, so `AssistantService.UsesCompletionTokenLimit` routes around it via
  `ExtraBody`. Those models also reject a non-default temperature.
- **Avalonia bindings must not cast in the path.** `{Binding $parent[Window].((vm:Shell)DataContext).X}`
  compiles and then throws at startup. Use `{Binding $parent[Window].DataContext.X}`.
- **AvaloniaEdit's bundled `.xshd` definitions are written for a white page** — `MethodCall` is
  MidnightBlue, `NumberLiteral` DarkBlue. `Services/SyntaxTheme.cs` remaps their *named colours*
  onto the `Code*` tokens in `Themes/Tokens.axaml`. It mutates `HighlightingManager.Instance`,
  which is process-wide, so it must be re-run on `ActualThemeVariantChanged`. Don't fork the
  `.xshd` files; the remap keeps unmapped languages working.
- **Selection and current-line come from AvaloniaEdit, not the control template**, so they ignore
  the theme dictionaries and are set explicitly in `EditorView.ApplyTheme`.
- The AvaloniaEdit package is `Avalonia.AvaloniaEdit` but the **assembly** is `AvaloniaEdit`, so the
  style include is `avares://AvaloniaEdit/Themes/Fluent/AvaloniaEdit.xaml`.

## Packaging and CI

The six libraries under `src/` pack as `Gravicode.Science.*`; everything else is
`IsPackable=false`. Metadata lives once in `Directory.Build.props`, not in six `.csproj` files.

- **`PublishRepositoryUrl` must stay unset.** `PackageProjectUrl` and `RepositoryUrl` both point at
  `https://github.com/DotNetVibeCoderz/Vibe_ML/tree/main/GravicodeScience` — the subdirectory, not
  the repository root, because that is where the project is. Setting `PublishRepositoryUrl` lets
  SourceLink overwrite `RepositoryUrl` with the git origin and silently replaces that deep link.
- **`GenerateDocumentationFile` is on for packable projects**, which turned 19 latent XML-doc
  defects into warnings — a positional record documenting some components but not all (CS1573), a
  `<param>` naming a property rather than the constructor parameter (CS1572), and a `cref` to a
  type in a different namespace (CS1574). All are fixed; `CS1591` stays silenced.
- **The version comes from the tag.** `VersionPrefix` in `Directory.Build.props` is the local
  default; `release.yml` passes `-p:Version=` parsed from a `gravicode-science-v*` tag, so the tag
  and the package version cannot disagree.
- **`GraviNum` is pushed to NuGet first.** Every other package depends on it, and a dependency that
  is not yet indexed leaves the dependents unrestorable for a few minutes.
- **Workflows live at the *repository* root, not here.** GravicodeScience is a subdirectory of the
  Vibe_ML monorepo and GitHub reads `.github/workflows/` only from the root. The files in
  `.github/workflows/` are written for that layout — `working-directory: GravicodeScience` on every
  step, triggers filtered on `GravicodeScience/**` — and have to be copied up after cloning.
- **`tools/verify/notebook_cells.py` is the CI job that catches what nothing else can.** A notebook
  is JSON; it validates whether or not the C# inside compiles. Run it after touching any notebook.
- **`testkey.txt` is gitignored and holds live keys.** Check it is not in git history before the
  repository is made public.

## Architecture

`Gravicode.Science` is a data-science/AI ecosystem for .NET: six libraries that mirror the Python stack, each with a matching sample console app, notebook, benchmark project, and test project.

| Project | Python analogue | Core responsibility |
|---|---|---|
| `GraviNum` | NumPy | `NdArray` n-dim arrays, ufuncs, `LinAlg` (dot/decomposition), RNG, statistics, sparse arrays |
| `GraviFrame` | pandas | `DataFrame`/`Series`, GroupBy, pivot, join, time-series (rolling/resample), IO |
| `GraviLearn` | scikit-learn | Preprocessing, supervised/unsupervised algorithms, `Pipeline` API, `Metrics` |
| `GraviText` | HuggingFace Transformers | Tokenization, embeddings, `TransformerModel`, sentiment/NER/classification |
| `GraviGraph` | PyTorch Geometric / DGL | Graph structures, PageRank/centrality, node2vec, GCN/GAT |
| `GraviProb` | PyMC / Stan | Distributions, `BayesianModel`, MCMC & variational inference |

**The dependency direction matters:** `GraviNum` is the foundation — every other library builds on its array/tensor and linear-algebra layer. Never introduce a dependency from `GraviNum` back up the stack. The actual graph as built:

```
GraviNum ──┬── GraviFrame ── GraviLearn ── GraviText ── GraviGraph
           └── GraviProb
```

`GraviText` references `GraviLearn` because its task pipelines train real classifiers rather than reimplementing them; `GraviGraph` references `GraviText` because node2vec is random walks fed to skip-gram.

**Each library has a "Performance Layer" as an explicit architectural concern**, not an afterthought. The spec expects, per library: `System.Numerics.Vector<T>` SIMD paths, a GPU backend via ILGPU (CUDA/OpenCL), TPL multi-threading, and memory mapping for datasets larger than RAM. Compute kernels should therefore be written so a scalar reference implementation, a SIMD path, and a GPU path can coexist behind one API — benchmarks in the spec are all framed as "CPU vs GPU" comparisons, so both paths need to remain runnable.

Interop targets that constrain public API design: ML.NET, ONNX Runtime, and BLAS/LAPACK native bindings; visualization hooks go to ScottPlot/OxyPlot.

## Repository layout

```
src/            one class library per Gravi* project
samples/        Gravi*.Console apps (one per library)
notebooks/      .NET Interactive .ipynb, one per library
benchmarks/     Gravi*.Benchmark projects (CPU vs GPU comparisons)
datasets/       iris.csv, titanic.csv, mnist_subset/, imdb_reviews.csv, cora_graph.json,
                bayesian_coin.csv, finance_timeseries.csv
tests/          Gravi*.Tests
docs/           installation, getting_started, one page per library, benchmarks, datasets,
                screenshots/ (rendered by the samples), and id/ mirroring every page
Gravicode.Science.sln
```

Adding a library means adding all five artifacts (src, sample, notebook, benchmark, tests) plus its `docs/<Name>.md` **and** `docs/id/<Name>.md` — the spec treats them as one deliverable.

## Testing conventions

New algorithms are pinned against something independently known — a closed-form answer, a published figure, or a naive implementation written inside the test. Examples already in the suite: PCA on Iris must give 92.46%/5.31%; Cora's largest weakly connected component must be 2485; MCMC on the coin model must match the exact Beta posterior. Stochastic results use explicit tolerances (`Assert.True(Math.Abs(a - b) < tol, message)`) rather than `Assert.Equal(a, b, decimals)`, which rounds and fails spuriously.

## Commands

```powershell
dotnet build Gravicode.Science.sln -c Release
dotnet test                                    # whole solution
dotnet test tests/GraviNum.Tests               # one project
dotnet test tests/GraviNum.Tests --filter "FullyQualifiedName~LinAlg"   # one test / class
dotnet run --project samples/GraviNum.Console
dotnet run --project benchmarks/GraviNum.Benchmark -c Release           # BenchmarkDotNet needs Release
dotnet format                                  # if a format/lint step is added
```

## Conventions from the spec

- **Bilingual docs.** Every document under `docs/` is expected in both English and Bahasa Indonesia.
- **Attribution.** Docs and applications must carry: "Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil".
- **API shape follows the Python original but stays idiomatic C#** — PascalCase methods, named arguments where the Python call uses keywords (`new PCA(components: 10)`, `.Rolling(window: 7)`), fluent chaining for pipelines and model building. `requirements.md` contains worked API examples for `GraviFrame`, `GraviLearn`, `GraviText`, `GraviGraph`, and `GraviProb`; match those signatures rather than inventing new ones.
