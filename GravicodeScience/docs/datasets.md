# Datasets

*[Bahasa Indonesia](id/datasets.md)*

Everything in [`datasets/`](../datasets) is committed to the repository, so the samples, notebooks
and tests run with no download step. Total size is about 750 KB.

| File | Rows | Source | Used by |
|---|---:|---|---|
| `iris.csv` | 150 | Real — Fisher (1936) | GraviLearn |
| `titanic.csv` | 891 | Real — Titanic passenger manifest | GraviFrame, GraviLearn |
| `mnist_subset/digits.csv` | 1,797 | Real — UCI optical digits | GraviLearn |
| `cora_graph.json` | 2,708 nodes | Real — Cora citation network | GraviGraph |
| `imdb_reviews.csv` | 80 | **Synthetic** — written for this repository | GraviText |
| `bayesian_coin.csv` | 200 | **Synthetic** — generated, bias 0.62 | GraviProb |
| `finance_timeseries.csv` | 1,116 | **Synthetic** — geometric random walk | GraviFrame |

The synthetic files are marked as such deliberately. Three of them stand in for datasets that are
either too large to commit or encumbered; the substitution is noted below in each case.

---

## iris.csv

Fisher's iris measurements: 150 flowers, four measurements each, three species evenly split.

```
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
```

Loaded by `Datasets.LoadIris()`. Two of the three species are not linearly separable, which is why
it remains a useful classification test rather than a trivial one. Reference results the test
suite pins: PCA explains **92.46%** of variance on the first component and **5.31%** on the
second; a random forest reaches about 95% test accuracy.

**Source**: [seaborn-data](https://github.com/mwaskom/seaborn-data). Public domain.

---

## titanic.csv

The Titanic passenger manifest: 891 passengers with survival, class, sex, age, fare and port of
embarkation.

Loaded by `Datasets.LoadTitanic()`, which encodes `sex` and `embarked` and imputes the missing
ages and fares with the column median.

Its real value here is that it is **messy**: `age` is missing for 177 passengers (19.9%) and
`deck` for 688 (77.2%). That makes it the right dataset for demonstrating missing-value handling
rather than a cleaned-up toy. Published baseline accuracy is 0.78–0.83; the test suite requires
above 0.75.

**Source**: [seaborn-data](https://github.com/mwaskom/seaborn-data). Public domain.

---

## mnist_subset/digits.csv

The UCI optical-digits dataset: 1,797 handwritten digits as 8×8 greyscale images, flattened to 64
pixel columns plus a label.

```
pixel0,pixel1,...,pixel63,label
```

Loaded by `Datasets.LoadDigits()`. This is the same subset scikit-learn ships as `load_digits`,
not the full 70,000-image MNIST — small enough to commit and to run in a test, while still being
a real ten-class problem. A `StandardScaler → PCA(30) → kNN(3)` pipeline reaches **97.96%** test
accuracy, matching the reference implementation.

**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits),
via scikit-learn. Public domain (CC0).

---

## cora_graph.json

The Cora citation network: 2,708 machine-learning papers, 5,429 citations, each paper described
by a 1,433-word binary bag of words and labelled with one of seven topics.

```json
{
  "directed": true,
  "featureDimension": 1433,
  "classes": ["Neural_Networks", "Rule_Learning", ...],
  "nodes": [{"id": 0, "paperId": "31336", "label": 0, "features": [8, 14, 251, ...]}],
  "edges": [[163, 0], [402, 0], ...]
}
```

Node features are stored as the **indices of the non-zero entries** rather than as full vectors.
Cora's features average about eighteen non-zeros out of 1,433, so the sparse form is roughly
eighty times smaller and loads correspondingly faster.

Loaded by `Graph.Load()`. Reference figures the test suite pins:

- largest weakly connected component: **2,485 nodes** (91.8%)
- majority class baseline: **30.2%**
- published GCN accuracy: ~81%; this implementation reaches ~71% at 60 epochs

Edges are stored as `citing → cited`. Because citations point one way, `ConnectedComponents`
reports *weak* connectivity — see [GraviGraph.md](GraviGraph.md).

**Source**: [LINQS](https://linqs.soe.ucsc.edu/data), converted from `cora.content` and
`cora.cites`. Free for research use.

---

## imdb_reviews.csv

**Synthetic.** 80 short film reviews labelled positive or negative, 50 English and 30 Bahasa
Indonesia.

```
review,sentiment,language
"An absolute masterpiece from start to finish; ...",positive,en
"Film ini bagus sekali, ceritanya menarik ...",positive,id
```

The real IMDB dataset is 50,000 reviews and about 80 MB — too large to commit and encumbered by
its own licence. These reviews were written for this repository to give the sentiment sample a
bilingual corpus that is small enough to train in under a second.

**What this means for results.** 80 short documents is a demo-sized corpus. The supervised
classifier genuinely separates the two classes, but the Word2Vec neighbours in
`samples/GraviText.Console` are noisy, and the sample says so. Real embeddings need millions of
tokens. Swap in your own corpus to see the difference:

```csharp
var reviews = DataFrame.ReadCsv("your-corpus.csv");
```

---

## bayesian_coin.csv

**Synthetic.** 200 coin flips generated with a true bias of 0.62, split into four sessions of 50.

```
flip_id,outcome,session
1,1,session1
```

The file contains **125 heads in 200 flips**. The point of a generated file is that the true
parameter is known, so `samples/GraviProb.Console` can check the sampled posterior against the
exact conjugate answer — the Beta(1,1) prior with a binomial likelihood gives Beta(126, 76), mean
0.623762. The MCMC estimate agrees to within about 0.002.

---

## finance_timeseries.csv

**Synthetic.** Two tickers over three years of trading days as geometric random walks with a
weekly seasonal component.

```
date,ticker,close,volume
2022-01-03,GRVC,120.44,847231
```

Real market data cannot be redistributed. This file is generated so the time-series features have
something with genuine structure to find — trend, weekly seasonality and volatility clustering —
without a licensing problem. It is the input for rolling windows, percentage change, calendar
resampling and the memory-mapped-versus-streaming comparison.

---

## Regenerating the synthetic files

The generators live in the repository history; the parameters are documented above and the seeds
are fixed, so the files are reproducible. The `Datasets` class also exposes generators that need
no files at all, which is what the unit tests use:

```csharp
Datasets.MakeBlobs(samples: 300, features: 2, centers: 3, spread: 1.0, seed: 42);
Datasets.MakeMoons(samples: 200, noise: 0.1, seed: 42);      // not linearly separable
Datasets.MakeRegression(samples: 200, features: 5, noise: 0.5, seed: 42);

Graph.Random(nodes: 100, p: 0.05, seed: 42);
Graph.ScaleFree(nodes: 1000, edgesPerNode: 3, seed: 42);     // power-law degrees
Graph.Communities(communities: 3, sizePerCommunity: 50, seed: 42);
```

## Adding your own data

`Datasets.FindDatasetDirectory()` walks up from the running assembly looking for a `datasets`
folder, so samples, notebooks and tests all find data without hard-coded paths. Drop a file in
`datasets/` and read it directly:

```csharp
var frame = DataFrame.ReadCsv(Datasets.ResolvePath("my-data.csv"));
```

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
