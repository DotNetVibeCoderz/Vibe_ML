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

## Reference machine

| | |
|---|---|
| CPU | x86-64-v3, 8 logical processors, AVX2 + FMA (`Vector<double>` width 4) |
| GPU | Intel UHD Graphics 620 (integrated), OpenCL |
| Runtime | .NET 10.0.11, RyuJIT, Server GC |
| Job | `ShortRun` — 3 warmup, 5 measured iterations |

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
| SVD (one-sided Jacobi) | ~8× | Also yields rank and condition number |
| Symmetric eigen (Jacobi) | ~6× | Iterative; cost depends on the spectrum |

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

## Reading these numbers honestly

1. **Measure on your own hardware.** The GPU result above would invert on a discrete compute card.
2. **`ShortRun` trades precision for time.** Some margins here exceed 10% of the mean. Use the
   default job when a number needs to be trustworthy.
3. **Benchmarks are not a substitute for profiling.** They tell you which of two implementations
   is faster, not where your own program spends its time.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
