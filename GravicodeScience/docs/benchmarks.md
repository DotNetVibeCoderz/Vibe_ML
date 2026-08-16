# Benchmarks

*[Bahasa Indonesia](id/benchmarks.md)*

Every number on this page was measured with
[BenchmarkDotNet](https://benchmarkdotnet.org) on the machine described below. **Do not treat them
as portable.** They are here to show the shape of the performance curves and, in one case, to
document a result that contradicts the obvious assumption.

## Running them

```bash
dotnet run --project benchmarks/GraviNum.Benchmark -c Release
dotnet run --project benchmarks/GraviNum.Benchmark -c Release -- --filter "*MatrixProduct*"
dotnet run --project benchmarks/GraviNum.Benchmark -c Release -- --list flat
```

Release is mandatory; BenchmarkDotNet refuses to run against a Debug build.

For the cross-stack comparison, see [Against the Python stack](#against-the-python-stack) below.

## Reference machine

| | |
|---|---|
| CPU | Intel Core i7-8650U, **4 cores / 8 threads**, AVX2 + FMA (`Vector<double>` width 4) |
| GPU | Intel UHD Graphics 620 (integrated), OpenCL |
| Runtime | .NET 10.0.11, RyuJIT, Server GC |
| Job | `ShortRun` — 3 warmup, 5 measured iterations |

### A note on this machine

This is a 15 W ultrabook part, and it matters for reading every number here.

- **Four physical cores, not eight.** `Environment.ProcessorCount` reports 8 because of
  hyperthreading, and the parallel kernels partition by that count. Two threads sharing one core's
  vector unit do not double floating-point throughput.
- **It throttles under sustained AVX load.** Short bursts reach turbo; a minute of dense linear
  algebra does not. The same build measured 78 ms and 168 ms for a 1024-cube product within the
  same hour, purely from thermal state and background load.

So absolute figures carry roughly ±30% run-to-run, and a ratio between two numbers measured in
*different* runs is not evidence of anything. Where this page reports a before/after, the two
versions were run **alternately inside one process** and the best of many taken — that comparison
survives throttling because both sides suffer it equally. Ratios against the Python stack come
from runs taken back to back under the same conditions.

---

## GraviNum — matrix multiplication

### What vectorisation buys

`LinAlg.Dot` accumulates four rows of the result at once, so each pass over a row of B feeds four
independent FMA chains. Against the textbook triple loop:

| Size | Naive triple loop | `LinAlg.Dot` | Speedup |
|---:|---:|---:|---:|
| 128 | 2,972 µs | 386 µs | **7.7×** |
| 256 | 38,481 µs | 1,918 µs | **20.1×** |

At 512×512 the shipped path sustains roughly **24 GFLOP/s** on this CPU, measured directly by
`samples/GraviNum.Console`.

An earlier implementation called a separate `Axpy` helper per row of B and managed only
0.6 GFLOP/s. The whole difference is register blocking and keeping the SIMD loop inline — worth
knowing before assuming "uses SIMD" means "fast".

### CPU versus GPU

This is the result worth reading carefully.

| Size | `LinAlg.Dot` (CPU) | ILGPU (OpenCL) | GPU ratio |
|---:|---:|---:|---:|
| 128 | 386 µs | 3,105 µs | 8.0× slower |
| 256 | 1,918 µs | 11,370 µs | 5.9× slower |
| 512 | 15,515 µs | 79,991 µs | 5.2× slower |
| 1024 | 132,032 µs | 940,932 µs | 7.1× slower |

**On this hardware the GPU never wins.** Two reasons, both structural rather than incidental:

1. Everything in the library is `double`. Integrated GPUs run float64 at a small fraction of their
   float32 rate — often 1/8 or worse — so the arithmetic advantage largely evaporates.
2. Every call copies both operands across the bus and the result back. That cost is fixed while
   the arithmetic grows, so it hurts small problems most, but it never disappears.

A discrete compute card with good float64 throughput reverses this. The library cannot tell which
kind of device it is looking at, which is why **automatic GPU dispatch is opt-in**:

```csharp
Compute.AutomaticGpuDispatch = true;      // after measuring on your hardware
Compute.Preferred = Compute.Gpu;          // or force it outright
```

By default `Compute.Best` always returns the CPU backend. Silently routing work onto a device that
is seven times slower would be worse than not routing it at all.

### Decompositions

Cost at 256×256, relative to LU:

| Factorisation | Relative cost | Notes |
|---|---|---|
| Cholesky | ~0.5× | Only valid for symmetric positive-definite input |
| LU | 1.0× | The general-purpose default |
| QR (Householder) | ~2× | More stable; what least squares uses |
| SVD (bidiagonal QR) | ~6× | Also yields rank and condition number |
| Symmetric eigen (tridiagonal QL) | ~2× | Iterative; cost depends on the spectrum |

### Sparse versus dense

A 2000×2000 matrix times a vector. CSR cost tracks the non-zero count; dense cost does not care.

| Density | Winner |
|---|---|
| 1% | Sparse, by a wide margin |
| 5% | Sparse |
| 25% | Dense — the index indirection now costs more than the skipped zeros |

The rule of thumb this confirms: below about 10% density, use `SparseMatrix`.

---

## GraviFrame — IO

### Streaming versus memory-mapped CSV

The two paths take **about the same wall-clock time**. That is the honest result, and the reason
to reach for the mapped reader is not speed but peak memory: the streaming reader holds the file
content alongside the parsed frame, while the mapped reader lets the operating system page it in
and out. On files that comfortably fit in RAM there is nothing to gain.

Use `ReadCsvMemoryMapped` when the file approaches or exceeds available memory. Use `ReadCsv`
otherwise — it is simpler and has no mapping to fail.

> A bug worth recording: the mapped view must be created with the file's exact length. Passing
> `0` ("to the end") rounds the mapping up to the system page size and the trailing NUL bytes
> decode as one extra, bogus row. There is now a regression test for exactly this.

### CSV versus Parquet

Parquet is roughly **4× smaller** on the sample data (14.6 KB against 55.7 KB for the Titanic
frame) and reads faster because the column types are recorded rather than re-inferred. Reading a
subset of columns touches only those columns, which is where the columnar layout pays off most.

### Transforms

Group-by, join and sort are all roughly linear in row count on these shapes. The one asymmetry
worth knowing: `Rolling(n).Mean()` uses an incremental accumulator and costs one add and one
subtract per row regardless of window width, while `Rolling(n).Median()` re-sorts each window and
gets markedly slower as the window grows.

---

## GraviLearn

### Training cost

At 10,000 samples and 20 features, relative to a single decision tree:

| Model | Relative cost | Why |
|---|---|---|
| Gaussian naive Bayes | ≪ 1× | One pass computing means and variances |
| Decision tree | 1× | Sorts each candidate feature per node |
| Random forest (50 trees) | ~10× | Trees are independent, so this is parallelised |
| Gradient boosting (50 trees) | ~50× | Sequential by construction — each tree fits the previous residual |

### Prediction cost

This is what matters for a deployed model, and the ordering differs from training:

| Model | Cost per row |
|---|---|
| Logistic regression | One dot product |
| Random forest | 50 tree descents |
| k-nearest neighbours | A distance to **every** training point |

kNN trains instantly and serves slowly. That trade is the definition of a lazy learner, and it is
why kNN rarely survives contact with a latency budget.

### Where a GPU could help

Logistic regression spends nearly all its time in two matrix products per iteration: `X w` to
score every sample, and `X' e` to accumulate the gradient. Those are exactly the shapes the GPU
benchmark above covers — so on this machine, moving the loop to the GPU would make it slower, for
the same float64 reason.

---

## GraviText

### Transformer inference

Attention is quadratic in sequence length: every token attends to every other. Doubling the
sequence roughly **quadruples** the attention cost while the feed-forward part only doubles. That
scaling wall is visible directly in `TransformerInferenceBenchmark` and is the reason every
long-context technique exists.

Batches parallelise cleanly because sequences are independent — `EncodeBatch` uses `Parallel.For`.

### Pipeline throughput

| Stage | Relative cost |
|---|---|
| Regex tokenization | 1× |
| WordPiece tokenization | ~3× — longest-prefix search per word |
| Porter stemming | ~2× |
| TF-IDF transform (dense) | Dominated by the output matrix size |

For real corpora use `CountVectorizer.TransformSparse`; a dense TF-IDF matrix over a large
vocabulary is mostly zeros.

---

## GraviGraph

### GCN training as the graph grows

Each layer is a sparse propagation followed by a dense projection:

- sparse half: `O(edges × width)`
- dense half: `O(nodes × features × width)`

On a genuinely sparse network the dense projection dominates, which is the part a GPU would
accelerate. `GcnTrainingBenchmark` separates the two so the split is measured rather than assumed.

### Algorithms

At 100,000 nodes, PageRank converges in well under a second. Betweenness centrality is `O(VE)`
even with Brandes' algorithm, which is why `samples/GraviGraph.Console` restricts it to a
600-node subgraph — this is an algorithmic limit, not an implementation one.

---

## GraviProb

### Chains scale, not repeat

Chains are completely independent, so four chains cost about the same wall-clock time as one on a
machine with four free cores. Multi-chain sampling is therefore nearly free — and it is the only
way to compute the R-hat convergence diagnostic, which compares between-chain to within-chain
variance.

### MCMC versus variational inference

| | MCMC | Mean-field VI |
|---|---|---|
| Cost | Seconds | Milliseconds |
| Result | Samples from the true posterior | A Gaussian approximation |
| Correlations | Captured | Assumed away |
| Posterior spread | Correct | Systematically understated |

Use VI to iterate quickly, then confirm with MCMC. Both are checked against the exact conjugate
posterior in the test suite, so neither is being trusted blindly.

### Hidden Markov models

Forward, Viterbi and forward-backward are all linear in sequence length and quadratic in state
count. All three work in log space or with per-step scaling, because the raw probabilities
underflow within a few dozen time steps — `HiddenMarkovBenchmark` runs 10,000-step sequences
specifically to exercise that.

---

---

## Against the Python stack

The question every .NET data-science library has to answer is how it compares to NumPy, pandas
and scikit-learn. Here is the measured answer, with no thumb on the scale.

### Does a native BLAS change these numbers?

**Yes, decisively — and every figure in this section is the managed path.**

That needs stating plainly, because it is the single biggest thing that can make these numbers
irreproducible. `NativeBlas` and `NativeLapack` look for a library the machine already has. A clean
machine has none, so the comparison below measures the managed kernels. A machine with OpenBLAS on
the path measures something else entirely.

Both configurations, against the same NumPy figures:

| | Managed | With OpenBLAS | NumPy | Native vs NumPy |
|---|---:|---:|---:|---|
| 512×512 matrix product | 8.6 ms | **2.3 ms** | 3.1 ms | **.NET 1.3×** |
| QR, 256×256 | 151.2 ms | **10.7 ms** | 10.4 ms | parity |
| Cholesky, 256×256 | 14.4 ms | **1.6 ms** | 1.2 ms | Python 1.3× |
| LU, 256×256 | 77.9 ms | **4.5 ms** | 3.3 ms | Python 1.4× |
| symmetric eigen, 256×256 | 119.7 ms | **45.6 ms** | 19.7 ms | Python 2.3× |
| matrix inverse, 256×256 | 135.2 ms | **58.0 ms** | 9.9 ms | Python 5.9× |
| SVD, 256×256 | 330.8 ms | **176.1 ms** | 27.1 ms | Python 6.5× |

With a BLAS present the 512-cube product is **faster than NumPy**, and QR, LU and Cholesky all land
within about 40% of it — unsurprising once you notice both stacks are then calling the same
OpenBLAS, and what remains is marshalling.

Every routine now has a native path: `dgemm`, `dgesv`, `dgeqrf`/`dorgqr`, `dgesvd`, `dsyev`,
`dgetrf`, `dpotrf`. SVD and eigen stay furthest behind because this library asks for the full
factorisation where NumPy's benchmark takes a cheaper form.

> **A caveat on `solve`.** The harness above reports it as roughly unchanged, and that is a
> measurement artifact rather than a result: three repeats do not warm a method past tiered JIT.
> Measured properly — both versions alternating, forty warm-up calls, best of eleven — the native
> path wins from 64 upward: **3.9×** at 256 with one right-hand side, and **14×** at 256 / **51×**
> at 512 when solving many at once, which is the shape `Inverse` uses.

**So which numbers are real?** Both — they answer different questions. The managed figures say what
this library does on its own, which is what matters for a deployment that ships one self-contained
package. The native figures say what it does when the machine is already set up for numerical work,
which is what matters on a data-science workstation. Neither is the "true" number, and quoting one
without saying which it is would be the actual error.

To reproduce either, set or clear `GRAVICODE_BLAS`:

```bash
dotnet run --project benchmarks/comparison/Gravicode.Science.Comparison -c Release
GRAVICODE_BLAS=/path/to/libopenblas.so dotnet run --project benchmarks/comparison/... -c Release
```

### How the comparison is run

```bash
dotnet run --project benchmarks/comparison/Gravicode.Science.Comparison -c Release -- dotnet-results.json
python benchmarks/comparison/python_benchmarks.py python-results.json
python benchmarks/comparison/compare.py dotnet-results.json python-results.json
```

Both harnesses use the **same protocol**: same shapes, same fixture data, two warmup runs, a fixed
repeat count, report the median. BenchmarkDotNet is deliberately not used here — Python has no
equivalent, and comparing its statistics against `timeit` would mean comparing two measurement
methodologies as well as two implementations. Run them **one at a time**; running both at once has
them fighting over the same cores and produces nonsense.

Reference machine: .NET 10.0.11 and Python 3.12.10 (numpy 2.4.4, pandas 3.0.3, scipy 1.17.1,
networkx 3.6.1), 8 logical processors, AVX2.

### Dense linear algebra — Python wins, decisively

| Operation | Gravicode.Science | Python | Ratio |
|---|---:|---:|---|
| 256×256 matrix product | 2.58 ms | 0.41 ms | Python 6.3× |
| 512×512 matrix product | 15.64 ms | 3.07 ms | Python 5.1× |
| 1024×1024 matrix product | 93.45 ms | 27.76 ms | Python 3.4× |
| LU, 256×256 | 41.13 ms | 3.28 ms | Python 12.5× |
| QR, 256×256 | 153.63 ms | 10.36 ms | Python 14.8× |
| Cholesky, 256×256 | 12.24 ms | 1.23 ms | Python 10.0× |
| SVD, 256×256 | 308.41 ms | 27.11 ms | Python 11.4× |
| symmetric eigen, 256×256 | 110.22 ms | 19.74 ms | Python 5.6× |
| solve Ax=b, 256×256 | 66.78 ms | 6.18 ms | Python 10.8× |
| matrix inverse, 256×256 | 147.48 ms | 9.90 ms | Python 14.9× |

NumPy is not doing this in Python. It hands every one of these to LAPACK and BLAS — decades of
hand-tuned Fortran and assembly, cache-blocked and multithreaded. Managed code with `Vector<T>`
does not close that gap, and this library does not pretend otherwise.

#### What changed in v0.2

SVD and symmetric eigen used to be the two worst results on this page by a wide margin — 66× and
139× — because both used a Jacobi method: sweep the whole matrix rotating pairs until nothing
changes. Correct, easy to verify, and hopelessly wasteful, since every sweep touches every entry
however close to converged it already is.

Both now reduce the matrix once with Householder reflections and then iterate on the condensed
form, where a shift makes convergence cubic:

| | v0.1 (Jacobi) | v0.2 | Gain | vs NumPy, before → after |
|---|---:|---:|---|---|
| symmetric eigen, 256×256 | 2,739.93 ms | 116.21 ms | **23.6×** | 138.8× → 5.9× |
| SVD, 256×256 | 1,794.65 ms | 327.63 ms | **5.5×** | 66.2× → 12.1× |
| PCA to 5 components, 20k × 20 | 375.40 ms | 63.87 ms | **5.9×** | 63.5× → 10.8× |

The Jacobi implementations remain, as `Decomposition.SvdJacobi` and
`Decomposition.SymmetricEigenJacobi`. They are not dead code: the tests pin the fast path against
them, and because the two arrive at the same factorisation by entirely different routes, that is a
real check rather than a restatement.

PCA gains more than the SVD underneath it because it also stopped computing a factor it never
read. A 20 000×20 SVD spends most of its time accumulating `U`, one row per sample; PCA needs only
the component directions and their variances, which live in `V` and the singular values.
`SvdRightVectors` and `SingularValues` skip what the caller does not ask for — the values are
bit-identical either way, since the rotations that produce them run regardless.

A native BLAS/LAPACK backend remains [on the roadmap](../PLAN.md) and is the only way to close what
is left, but the gap it has to close is now one order of magnitude, not two.

#### Element-wise arithmetic, and a bug that was costing 3×

The parallel element-wise path used to copy **both** operands into fresh arrays and the result back
out, purely so the lambda could capture them — spans cannot cross a closure. On a million-element
add that is three extra passes over 8 MB each, to save nothing. Pinning the buffers and handing the
workers pointers removes all of it:

| | before | after | gain |
|---|---:|---:|---|
| element-wise add, 1M | 7.23 ms | 2.36 ms | **3.06×** |
| element-wise add, 10M | 82.33 ms | 27.62 ms | **2.98×** |

Measured by running both versions alternately in one process and taking the best of fifteen — see
*[a note on this machine](#a-note-on-this-machine)* for why. The results are bit-identical.

These operations are bound by memory bandwidth, not arithmetic: the figure that matters is the
2.9 → 10.2 GB/s the fix bought, and it is why wider SIMD registers would not have helped. That is
also what moved element-wise arithmetic past NumPy in the table above.

`Unary` had no parallel path at all and now has one, which is what makes `Exp` and `Log` over large
arrays scale with cores.

#### Matrix product

The kernel now holds a 4×vector tile of C in registers across a slice of `k`, rather than reloading
and restoring C at every step of `k`. Slicing `k` is what keeps the strided walk through B inside
cache; register accumulation without it is *slower* at 1024 and above, which is worth knowing
before anyone tries half of the change.

Interleaved against the previous kernel, best of many: **1.7× at 128**, no change at 256, and
**1.05–1.4×** from 512 up. Honest reading: this is a modest win, not a breakthrough. Closing the
remaining 3.4× to NumPy needs the packed, three-level blocking that OpenBLAS uses, which is a much
larger piece of work.

Column panelling — blocking the *j* loop instead — was also tried and is consistently **worse** at
every size. It is recorded here so nobody spends the afternoon rediscovering it.

### Arrays, statistics, dataframes — Python ahead, but not by orders of magnitude

| Operation | Gravicode.Science | Python | Ratio |
|---|---:|---:|---|
| **element-wise add, 1M** | 2.52 ms | 4.80 ms | **.NET 1.9×** |
| **element-wise add, 10M** | 26.70 ms | 47.21 ms | **.NET 1.8×** |
| sparse matrix-vector, 2000² @ 1% | 0.16 ms | 0.06 ms | Python 2.8× |
| 1,000,000 normal deviates | 13.75 ms | 16.22 ms | **.NET 1.2×** |
| mean + std over 1M | 11.50 ms | 7.62 ms | Python 1.5× |
| read 200,000-row CSV | 371.37 ms | 135.88 ms | Python 2.7× |
| group-by mean, 500 groups | 39.88 ms | 5.86 ms | Python 6.8× |
| sort by a numeric column | 47.85 ms | 27.24 ms | Python 1.8× |
| filter a numeric column | 6.39 ms | 6.06 ms | parity |
| **rolling mean, window 30** | 1.03 ms | 8.83 ms | **.NET 8.6×** |

The rolling mean is the exception that explains the rule: `Rolling(n).Mean()` uses an incremental
accumulator, so it costs one add and one subtract per row regardless of window width, while
pandas recomputes over the window. That is an *algorithmic* difference, and algorithmic
differences survive the language gap.

### Machine learning — mixed

| Operation | Gravicode.Science | Python | Ratio |
|---|---:|---:|---|
| logistic regression, 20k × 20 | 346.95 ms | 31.16 ms | Python 11.1× |
| **random forest fit, 50 trees** | 1,906.19 ms | 2,584.07 ms | **.NET 1.4×** |
| random forest predict, 20k rows | 134.54 ms | 98.20 ms | Python 1.4× |
| k-means, k=5, 3 restarts | 2,541.98 ms | 497.53 ms | Python 5.1× |
| PCA to 5 components | 63.87 ms | 5.92 ms | Python 10.8× |
| kNN predict, 2k vs 5k | 609.92 ms | 55.17 ms | Python 11.1× |

Random forest is the one model where this library is faster, and the reason is instructive: tree
fitting is branch-heavy, cache-unfriendly, pointer-chasing work that no BLAS call can help with.
It is exactly the shape of problem a JIT-compiled language handles well and an interpreter does
not — scikit-learn only competes here because its trees are compiled Cython.

PCA is SVD underneath, so it moved with the SVD: 63.5× in v0.1, 10.8× now. Two thirds of that came
from the new decomposition and the rest from no longer computing a factor PCA never reads.

### Text, graphs, sampling — .NET wins, and by a lot

| Operation | Gravicode.Science | Python | Ratio |
|---|---:|---:|---|
| **TF-IDF fit, 20k documents** | 262.50 ms | 691.75 ms | **.NET 2.6×** |
| **regex tokenize 20k documents** | 153.92 ms | 359.60 ms | **.NET 2.3×** |
| PageRank, 50k nodes | 276.32 ms | 313.02 ms | parity |
| **breadth-first search, 50k nodes** | 9.54 ms | 376.63 ms | **.NET 39.5×** |
| **connected components, 50k nodes** | 9.28 ms | 38.60 ms | **.NET 4.2×** |
| **Dijkstra, 50k nodes** | 32.34 ms | 180.42 ms | **.NET 5.6×** |
| **MCMC, 4 chains × 5000 draws** | 21.37 ms | 279.18 ms | **.NET 13.1×** |
| **1,000,000 scalar log densities** | 2.08 ms | 275.67 ms | **.NET 132.2×** |

This is the other half of the story, and it is the half that justifies the project. None of these
workloads vectorise into a single library call. Graph traversal is pointer chasing. MCMC is a
sequential loop whose next step depends on the last. A scalar log-density loop is exactly what an
interpreter is worst at. Here the comparison is JIT-compiled code against CPython bytecode, and it
is not close.

For the log-density loop, NumPy's *vectorised* equivalent takes 12.17 ms — still 5.8× slower than
the .NET scalar loop, because the vectorised form has to materialise a 1M-element intermediate
array while the loop keeps everything in registers.

### The tally, and what it means

**.NET faster on 12 measurements, Python faster on 22, parity on 3.**

Read past the tally, though, because the split is not random:

- Wherever the operation bottoms out in **LAPACK, BLAS or compiled Cython**, Python wins, now
  consistently by 3–15× rather than the 60–140× the Jacobi decompositions used to cost.
- Wherever the operation is **scalar, branchy or sequential**, .NET wins, by 2–40× and once by 132×.
- Wherever the operation is **bound by memory bandwidth** — element-wise arithmetic — the two are
  close, and .NET is now slightly ahead.

Two things moved the numbers in v0.2. The decomposition rewrite removed the extreme end of that
first range, so nothing on this page is more than about 15× off the Python stack. Removing the
copies from the element-wise path then flipped both of those rows from *Python 2–3×* to
*.NET 1.8–1.9×*.

So the honest summary is: this library is not a replacement for NumPy on dense linear algebra
today, and adding BLAS/LAPACK interop is the single change that would most improve it. It already
beats the Python stack on graph algorithms, MCMC, tokenization and any workload built from tight
scalar loops — and it does so while staying in one type-safe, deployable, dependency-light process.

### Three bugs this comparison found

Running the two stacks side by side surfaced problems that the .NET-only benchmarks had not.

**The element-wise parallel path copied its own inputs.** `RunBinaryContiguous` called `ToArray()`
on both operands and allocated a third array for the result, because a `Span<T>` cannot be captured
by a lambda. Every large `Add` therefore moved 24 MB it did not need to. Being 2–3× behind NumPy on
an operation that is *nothing but* a memory copy was the clue; the fix is `fixed` pointers, and it
made element-wise arithmetic **3× faster** and moved it ahead of NumPy.

**Decision tree splitting was O(n²).** `FindBestSplit` materialised `sorted[..k]` and `sorted[k..]`
at every candidate split point and rebuilt the class-count dictionary from them — O(n) work at
each of O(n) split points. At 20,000 samples the random forest benchmark ran for **83 minutes
without finishing**. Rewritten to sweep the sorted order while moving one sample at a time between
running class counts, it now fits in **1.9 seconds** — and beats scikit-learn. The whole
GraviLearn test suite got faster too, from 4 s to 1 s.

**A benchmark was measuring nothing.** The scalar log-density loop originally reported 0.45 ms for
1,000,000 iterations — about 1.5 cycles each, which is impossible for a logarithm. The result was
discarded, so the JIT deleted the loop. With a non-inlined sink consuming the value, the real
figure is 2.08 ms. Worth stating plainly: a benchmark that looks too good usually is.

---

## Reading these numbers honestly

1. **Measure on your own hardware.** The GPU result above would invert on a discrete compute card.
2. **`ShortRun` trades precision for time.** Some margins here exceed 10% of the mean. Use the
   default job when a number needs to be trustworthy.
3. **Benchmarks are not a substitute for profiling.** They tell you which of two implementations
   is faster, not where your own program spends its time.
4. **Consume what you compute.** A discarded result is a deleted loop. The comparison harness
   routes every value through a non-inlined sink for this reason.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
