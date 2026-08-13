# Getting started

*[Bahasa Indonesia](id/getting_started.md)*

Six libraries, one example each. Every snippet below runs as written once the solution is built.

- [GraviNum — arrays and linear algebra](#gravinum)
- [GraviFrame — dataframes](#graviframe)
- [GraviLearn — machine learning](#gravilearn)
- [GraviText — natural language](#gravitext)
- [GraviGraph — graphs](#gravigraph)
- [GraviProb — Bayesian inference](#graviprob)

---

## GraviNum

```csharp
using Gravicode.Science.GraviNum;

// Reshaping and transposing are views: no data is copied.
var a = NdArray.Arange(12).Reshape(3, 4);
Console.WriteLine(a[1, 2]);        // 6
Console.WriteLine(a.T[2, 1]);      // 6 - the same element

// Broadcasting stretches a length-4 vector across all three rows.
var shifted = a + NdArray.Arange(4);

// Linear algebra
var m = NdArray.FromArray(new double[,] { { 4, 7 }, { 2, 6 } });
Console.WriteLine(LinAlg.Determinant(m));           // 10
Console.WriteLine(LinAlg.Inverse(m));
Console.WriteLine(LinAlg.Solve(m, NdArray.FromValues([1.0, 2.0])));

// Decompositions
var svd = Decomposition.Svd(m);
var (values, vectors) = Decomposition.SymmetricEigen(m + m.T);

// Random numbers and statistics
var rng = new GraviRandom(seed: 42);
var samples = rng.Normal(mean: 100, stdDev: 15, 10_000);
Console.WriteLine(Statistics.Mean(samples));
Console.WriteLine(Statistics.Percentile(samples, 95));
```

[Full guide →](GraviNum.md)

---

## GraviFrame

```csharp
using Gravicode.Science.GraviFrame;

var df = DataFrame.ReadCsv("datasets/titanic.csv");

// Column types are inferred: numeric, text, boolean or timestamp.
Console.WriteLine(df.Info());

// Missing values are explicit and easy to handle.
var clean = df.WithColumn(df.Numeric("age").FillMissingWithMedian().Rename("age_filled"));

// Grouping, pivoting and joining
var survival = clean.GroupBy("pclass", "sex").Mean("survived");
var wide = clean.Pivot("pclass", "sex", "survived");

// Time series
var prices = DataFrame.ReadCsv("datasets/finance_timeseries.csv").SortBy("date");
var close = prices.Numeric("close");
var withAverages = prices
    .WithColumn(close.Rolling(window: 7).Mean().Rename("ma7"))
    .WithColumn(close.PercentChange().Rename("daily_return"));

var monthly = Resampling.Resample(prices, "date", ResampleFrequency.Monthly, "mean");
```

[Full guide →](GraviFrame.md)

---

## GraviLearn

```csharp
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviLearn.ModelSelection;
using Gravicode.Science.GraviLearn.Preprocessing;
using Gravicode.Science.GraviLearn.Trees;

var iris = Datasets.LoadIris();
var split = Selection.Split(iris.Features, iris.Target, testSize: 0.3, stratify: true);

// A pipeline keeps preprocessing and the model together, which is what stops
// cross-validation leaking test statistics into training.
var pipeline = new Pipeline()
    .Add(new StandardScaler())
    .Add(new PCA(components: 3))
    .Add(new RandomForestClassifier(nTrees: 100));

pipeline.Fit(split.TrainX, split.TrainY);
var predictions = pipeline.Predict(split.TestX);

Console.WriteLine(Metrics.ClassificationReport(split.TestY, predictions, iris.LabelNames));

// Cross-validation refits every step on each fold.
var score = Selection.CrossValidatePipeline(
    () => new Pipeline().Add(new StandardScaler()).Add(new RandomForestClassifier(nTrees: 100)),
    iris.Features, iris.Target, folds: 5, stratified: true);
Console.WriteLine(score);   // 0.9533 +/- 0.0340 over 5 folds
```

[Full guide →](GraviLearn.md)

---

## GraviText

```csharp
using Gravicode.Science.GraviText.Embeddings;
using Gravicode.Science.GraviText.Linguistics;
using Gravicode.Science.GraviText.Tasks;
using Gravicode.Science.GraviText.Tokenization;

// Stemming in both shipped languages
Console.WriteLine(PorterStemmer.Stem("connecting"));       // connect
Console.WriteLine(IndonesianStemmer.Stem("berlari"));      // lari

// Sentiment with no training data at all
var analyzer = new SentimentAnalyzer();
Console.WriteLine(analyzer.Analyze("produknya bagus dan sangat memuaskan"));   // positive (…)
Console.WriteLine(analyzer.Analyze("this is not good"));                       // negative (…)

// Or train on labelled examples, which is materially better
analyzer.Train(documents, labels);

// Word vectors
var embeddings = new Word2Vec(dimensions: 100, epochs: 10)
    .Train(documents)
    .RemoveCommonComponent();

foreach (var (word, score) in embeddings.MostSimilar("excellent", top: 5))
    Console.WriteLine($"{word} {score:F3}");
```

[Full guide →](GraviText.md)

---

## GraviGraph

```csharp
using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Algorithms;
using Gravicode.Science.GraviGraph.Neural;

var graph = Graph.Load("datasets/cora_graph.json");

// Ranking and structure
var rank = GraphAlgorithms.PageRank(graph);
var (components, componentOf) = GraphAlgorithms.ConnectedComponents(graph);
var communities = GraphAlgorithms.LabelPropagation(graph);

// Semi-supervised node classification: only 140 of 2708 papers are labelled.
var train = new GraviNum.GraviRandom(42).Permutation(graph.NodeCount).Take(140).ToArray();

var gcn = new GraphConvolutionalNetwork(hiddenSize: 16, epochs: 60)
    .Train(graph, train, features: graph.NodeFeatures);

Console.WriteLine(gcn.Score(graph, test));
```

[Full guide →](GraviGraph.md)

---

## GraviProb

```csharp
using Gravicode.Science.GraviProb;

// 125 heads in 200 flips: what is the coin's bias?
var model = new BayesianModel()
    .AddDistribution("theta", Distribution.Beta(1, 1))
    .AddObservation("data", DistributionSpec.Binomial(200, "theta"), 125);

var posterior = model.SampleMCMC(iterations: 20_000, chains: 4);
Console.WriteLine(posterior.Summary());

var (low, high) = posterior.HighestDensityInterval("theta", 0.95);
Console.WriteLine($"95% HDI: [{low:F4}, {high:F4}]");

// The answer is not a number but a distribution, which is the whole point.
var aboveHalf = posterior["theta"].ToArray().Count(v => v > 0.5) / (double)posterior.TotalDraws;
Console.WriteLine($"P(biased toward heads) = {aboveHalf:P1}");
```

[Full guide →](GraviProb.md)

---

## Using several libraries together

The libraries are designed to hand data to each other without conversion glue:

```csharp
// CSV -> dataframe -> matrix -> model
var frame = DataFrame.ReadCsv("data.csv");
var features = frame.ToNdArray("age", "income", "score");   // GraviFrame -> GraviNum
var labels = frame.Numeric("churn").ToNdArray();

var model = new RandomForestClassifier(nTrees: 200);
model.Fit(features, labels);                                 // GraviNum -> GraviLearn

// Text -> features -> the same model family
var tfidf = new TfidfVectorizer().FitTransform(reviews);     // GraviText -> GraviNum
```

## Where to go next

| | |
|---|---|
| **Runnable programs** | [`samples/`](../samples) — six console apps, one per library |
| **Interactive** | [`notebooks/`](../notebooks) — six .NET Interactive notebooks with charts |
| **Performance** | [benchmarks.md](benchmarks.md) — measured CPU and GPU numbers |
| **Data** | [datasets.md](datasets.md) — what ships in `datasets/` and where it came from |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
