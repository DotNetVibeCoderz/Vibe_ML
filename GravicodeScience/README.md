# Gravicode.Science

A data science and AI ecosystem for **.NET 10** — six libraries that mirror the Python stack,
each with a runnable sample, an interactive notebook, benchmarks and tests.

[![.NET 10](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com)
[![tests](https://img.shields.io/badge/tests-400%20passing-brightgreen)](tests)
[![licence](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

| Library | Python analogue | Focus |
|---|---|---|
| [**GraviNum**](docs/GraviNum.md) | NumPy | N-dimensional arrays, linear algebra, random, statistics |
| [**GraviFrame**](docs/GraviFrame.md) | pandas | DataFrames, group-by, pivot, joins, time series |
| [**GraviLearn**](docs/GraviLearn.md) | scikit-learn | Preprocessing, supervised & unsupervised ML, pipelines, metrics |
| [**GraviText**](docs/GraviText.md) | HuggingFace Transformers | Tokenization, embeddings, transformers, NLP tasks |
| [**GraviGraph**](docs/GraviGraph.md) | PyTorch Geometric / DGL | Graph structures, embeddings, GNNs |
| [**GraviProb**](docs/GraviProb.md) | PyMC / Stan | Distributions, Bayesian inference, probabilistic models |

## Quick start

```bash
git clone https://github.com/gravicode/Gravicode.Science.git
cd Gravicode.Science
dotnet build Gravicode.Science.sln -c Release
dotnet test

dotnet run --project samples/GraviLearn.Console
```

```csharp
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviLearn.ModelSelection;
using Gravicode.Science.GraviLearn.Preprocessing;
using Gravicode.Science.GraviLearn.Trees;

var iris = Datasets.LoadIris();
var split = Selection.Split(iris.Features, iris.Target, testSize: 0.3, stratify: true);

var pipeline = new Pipeline()
    .Add(new StandardScaler())
    .Add(new PCA(components: 3))
    .Add(new RandomForestClassifier(nTrees: 100));

pipeline.Fit(split.TrainX, split.TrainY);
Console.WriteLine(Metrics.ClassificationReport(split.TestY, pipeline.Predict(split.TestX), iris.LabelNames));
```

## What is in the box

```
src/          six class libraries
samples/      six console apps, each printing real results
notebooks/    six .NET Interactive notebooks with charts
benchmarks/   six BenchmarkDotNet suites
datasets/     Iris, Titanic, MNIST digits, Cora, plus generated data
tests/        400 tests
docs/         English, with Bahasa Indonesia in docs/id/
```

## Design

**`GraviNum` is the foundation.** Every other library builds on its array and linear-algebra
layer, and nothing above it is ever referenced back down.

```
GraviNum ──┬── GraviFrame ── GraviLearn ── GraviText ── GraviGraph
           └── GraviProb
```

**Views, not copies.** Reshaping, transposing and slicing an `NdArray` produce views over one
shared buffer. Only `Copy()` moves data.

**Three-tier dispatch.** Element-wise work automatically picks a `Vector<double>` SIMD loop, a
threaded version of it, or a strided broadcast walk, based on shape and size. Callers never choose.

**Honest performance.** `LinAlg.Dot` sustains ~24 GFLOP/s at 512×512 on a mid-range laptop CPU,
about 20× the textbook triple loop. GPU support exists through ILGPU but is **opt-in**: everything
here is float64, and on integrated hardware the GPU measured 5–8× *slower*. See
[benchmarks.md](docs/benchmarks.md).

## Verified against reference implementations

The test suite pins results that are independently known, so a regression shows up as a test
failure rather than as a plausible-looking number:

| | Result |
|---|---|
| PCA on Iris | 92.46% / 5.31% variance on the first two components |
| Digits, `StandardScaler → PCA(30) → kNN(3)` | 97.96% test accuracy |
| Cora largest weakly connected component | 2,485 of 2,708 nodes |
| Coin-flip posterior, MCMC vs exact conjugate | agrees to ~0.002 |
| Titanic random forest | above the 0.78 published baseline |

## Documentation

| | English | Bahasa Indonesia |
|---|---|---|
| Installation | [installation.md](docs/installation.md) | [id/installation.md](docs/id/installation.md) |
| Getting started | [getting_started.md](docs/getting_started.md) | [id/getting_started.md](docs/id/getting_started.md) |
| Benchmarks | [benchmarks.md](docs/benchmarks.md) | [id/benchmarks.md](docs/id/benchmarks.md) |
| Datasets | [datasets.md](docs/datasets.md) | [id/datasets.md](docs/id/datasets.md) |

Per-library guides sit alongside them in [`docs/`](docs) and [`docs/id/`](docs/id).

Roadmap: [PLAN.md](PLAN.md) · Status: [Progress.md](Progress.md)

## Samples

Each prints a hardware report, then does real work on real data:

```bash
dotnet run --project samples/GraviNum.Console     # matrix ops, decompositions, GFLOP/s
dotnet run --project samples/GraviFrame.Console   # Titanic: group-by, pivot, joins, time series
dotnet run --project samples/GraviLearn.Console   # Iris: forests, cross-validation, grid search
dotnet run --project samples/GraviText.Console    # bilingual sentiment, embeddings, transformer
dotnet run --project samples/GraviGraph.Console   # Cora: PageRank, centrality, GCN
dotnet run --project samples/GraviProb.Console    # Bayesian coin toss, HMM, Bayesian network
```

They also render the charts in [`docs/screenshots/`](docs/screenshots).

## Requirements

.NET SDK 10.0 or later. Windows, Linux or macOS. A GPU is optional and never required.

## Licence

MIT.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
