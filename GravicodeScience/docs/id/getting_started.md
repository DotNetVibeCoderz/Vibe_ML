# Memulai

*[English](../getting_started.md)*

Enam library, satu contoh untuk masing-masing. Semua potongan kode di bawah berjalan apa adanya
setelah solusi di-build.

- [GraviNum — array dan aljabar linear](#gravinum)
- [GraviFrame — dataframe](#graviframe)
- [GraviLearn — machine learning](#gravilearn)
- [GraviText — pemrosesan bahasa](#gravitext)
- [GraviGraph — graf](#gravigraph)
- [GraviProb — inferensi Bayesian](#graviprob)

---

## GraviNum

```csharp
using Gravicode.Science.GraviNum;

// Reshape dan transpose menghasilkan view: tidak ada data yang disalin.
var a = NdArray.Arange(12).Reshape(3, 4);
Console.WriteLine(a[1, 2]);        // 6
Console.WriteLine(a.T[2, 1]);      // 6 - elemen yang sama

// Broadcasting merentangkan vektor panjang 4 ke seluruh tiga baris.
var shifted = a + NdArray.Arange(4);

// Aljabar linear
var m = NdArray.FromArray(new double[,] { { 4, 7 }, { 2, 6 } });
Console.WriteLine(LinAlg.Determinant(m));           // 10
Console.WriteLine(LinAlg.Inverse(m));
Console.WriteLine(LinAlg.Solve(m, NdArray.FromValues([1.0, 2.0])));

// Dekomposisi
var svd = Decomposition.Svd(m);
var (values, vectors) = Decomposition.SymmetricEigen(m + m.T);

// Bilangan acak dan statistik
var rng = new GraviRandom(seed: 42);
var samples = rng.Normal(mean: 100, stdDev: 15, 10_000);
Console.WriteLine(Statistics.Mean(samples));
Console.WriteLine(Statistics.Percentile(samples, 95));
```

[Panduan lengkap →](GraviNum.md)

---

## GraviFrame

```csharp
using Gravicode.Science.GraviFrame;

var df = DataFrame.ReadCsv("datasets/titanic.csv");

// Tipe kolom disimpulkan otomatis: numerik, teks, boolean, atau timestamp.
Console.WriteLine(df.Info());

// Nilai kosong bersifat eksplisit dan mudah ditangani.
var clean = df.WithColumn(df.Numeric("age").FillMissingWithMedian().Rename("age_filled"));

// Pengelompokan, pivot, dan join
var survival = clean.GroupBy("pclass", "sex").Mean("survived");
var wide = clean.Pivot("pclass", "sex", "survived");

// Deret waktu
var prices = DataFrame.ReadCsv("datasets/finance_timeseries.csv").SortBy("date");
var close = prices.Numeric("close");
var withAverages = prices
    .WithColumn(close.Rolling(window: 7).Mean().Rename("ma7"))
    .WithColumn(close.PercentChange().Rename("daily_return"));

var monthly = Resampling.Resample(prices, "date", ResampleFrequency.Monthly, "mean");
```

[Panduan lengkap →](GraviFrame.md)

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

// Pipeline menyatukan prapemrosesan dan model, sehingga cross-validation tidak
// membocorkan statistik data uji ke dalam pelatihan.
var pipeline = new Pipeline()
    .Add(new StandardScaler())
    .Add(new PCA(components: 3))
    .Add(new RandomForestClassifier(nTrees: 100));

pipeline.Fit(split.TrainX, split.TrainY);
var predictions = pipeline.Predict(split.TestX);

Console.WriteLine(Metrics.ClassificationReport(split.TestY, predictions, iris.LabelNames));

// Cross-validation melatih ulang setiap langkah pada tiap fold.
var score = Selection.CrossValidatePipeline(
    () => new Pipeline().Add(new StandardScaler()).Add(new RandomForestClassifier(nTrees: 100)),
    iris.Features, iris.Target, folds: 5, stratified: true);
Console.WriteLine(score);   // 0.9533 +/- 0.0340 over 5 folds
```

[Panduan lengkap →](GraviLearn.md)

---

## GraviText

```csharp
using Gravicode.Science.GraviText.Embeddings;
using Gravicode.Science.GraviText.Linguistics;
using Gravicode.Science.GraviText.Tasks;
using Gravicode.Science.GraviText.Tokenization;

// Stemming untuk kedua bahasa yang disertakan
Console.WriteLine(PorterStemmer.Stem("connecting"));       // connect
Console.WriteLine(IndonesianStemmer.Stem("berlari"));      // lari

// Analisis sentimen tanpa data latih sama sekali
var analyzer = new SentimentAnalyzer();
Console.WriteLine(analyzer.Analyze("produknya bagus dan sangat memuaskan"));   // positive (…)
Console.WriteLine(analyzer.Analyze("this is not good"));                       // negative (…)

// Atau latih dengan contoh berlabel, yang hasilnya jauh lebih baik
analyzer.Train(documents, labels);

// Vektor kata
var embeddings = new Word2Vec(dimensions: 100, epochs: 10)
    .Train(documents)
    .RemoveCommonComponent();

foreach (var (word, score) in embeddings.MostSimilar("excellent", top: 5))
    Console.WriteLine($"{word} {score:F3}");
```

[Panduan lengkap →](GraviText.md)

---

## GraviGraph

```csharp
using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Algorithms;
using Gravicode.Science.GraviGraph.Neural;

var graph = Graph.Load("datasets/cora_graph.json");

// Peringkat dan struktur
var rank = GraphAlgorithms.PageRank(graph);
var (components, componentOf) = GraphAlgorithms.ConnectedComponents(graph);
var communities = GraphAlgorithms.LabelPropagation(graph);

// Klasifikasi node semi-supervised: hanya 140 dari 2708 paper yang berlabel.
var train = new GraviNum.GraviRandom(42).Permutation(graph.NodeCount).Take(140).ToArray();

var gcn = new GraphConvolutionalNetwork(hiddenSize: 16, epochs: 60)
    .Train(graph, train, features: graph.NodeFeatures);

Console.WriteLine(gcn.Score(graph, test));
```

[Panduan lengkap →](GraviGraph.md)

---

## GraviProb

```csharp
using Gravicode.Science.GraviProb;

// 125 sisi angka dari 200 lemparan: berapa bias koinnya?
var model = new BayesianModel()
    .AddDistribution("theta", Distribution.Beta(1, 1))
    .AddObservation("data", DistributionSpec.Binomial(200, "theta"), 125);

var posterior = model.SampleMCMC(iterations: 20_000, chains: 4);
Console.WriteLine(posterior.Summary());

var (low, high) = posterior.HighestDensityInterval("theta", 0.95);
Console.WriteLine($"HDI 95%: [{low:F4}, {high:F4}]");

// Jawabannya bukan sebuah angka melainkan sebuah distribusi — itulah intinya.
var aboveHalf = posterior["theta"].ToArray().Count(v => v > 0.5) / (double)posterior.TotalDraws;
Console.WriteLine($"P(condong ke angka) = {aboveHalf:P1}");
```

[Panduan lengkap →](GraviProb.md)

---

## Menggunakan beberapa library bersamaan

Library dirancang agar bisa saling mengoper data tanpa kode perekat:

```csharp
// CSV -> dataframe -> matriks -> model
var frame = DataFrame.ReadCsv("data.csv");
var features = frame.ToNdArray("age", "income", "score");   // GraviFrame -> GraviNum
var labels = frame.Numeric("churn").ToNdArray();

var model = new RandomForestClassifier(nTrees: 200);
model.Fit(features, labels);                                 // GraviNum -> GraviLearn

// Teks -> fitur -> keluarga model yang sama
var tfidf = new TfidfVectorizer().FitTransform(reviews);     // GraviText -> GraviNum
```

## Langkah berikutnya

| | |
|---|---|
| **Program siap jalan** | [`samples/`](../../samples) — enam aplikasi konsol, satu per library |
| **Interaktif** | [`notebooks/`](../../notebooks) — enam notebook .NET Interactive dengan grafik |
| **Performa** | [benchmarks.md](benchmarks.md) — angka CPU dan GPU hasil pengukuran |
| **Data** | [datasets.md](datasets.md) — isi `datasets/` dan asal-usulnya |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
