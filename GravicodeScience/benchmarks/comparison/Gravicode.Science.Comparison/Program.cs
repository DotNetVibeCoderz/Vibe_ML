using System.Diagnostics;
using System.Numerics;
using System.Text.Json;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Algorithms;
using Gravicode.Science.GraviLearn.Clustering;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.Neighbors;
using Gravicode.Science.GraviLearn.Trees;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Vectorization;
using Gravicode.Science.GraviProb;

// ---------------------------------------------------------------------------
// The .NET half of the cross-stack comparison.
//
// This deliberately does NOT use BenchmarkDotNet. The Python side cannot run it,
// and comparing BenchmarkDotNet's statistics against Python's timeit would be
// comparing two different measurement protocols. Instead both sides use the same
// simple harness - fixed warmup, fixed repeat count, report the median - so the
// only difference between the two numbers is the code being measured.
// ---------------------------------------------------------------------------

var outputPath = args.Length > 0
    ? args[0]
    : Path.Combine(AppContext.BaseDirectory, "dotnet-results.json");

var results = new List<Measurement>();
var rng = new GraviRandom(42);

Console.WriteLine("Gravicode.Science comparison harness (.NET)");
Console.WriteLine($"  runtime : .NET {Environment.Version}");
Console.WriteLine($"  cpu     : {Environment.ProcessorCount} logical processors, SIMD width {Vector<double>.Count}");
Console.WriteLine();

// ---------------------------------------------------------------- GraviNum

foreach (var size in new[] { 256, 512, 1024 })
{
    var a = rng.StandardNormal(size, size);
    var b = rng.StandardNormal(size, size);
    Measure($"matmul_{size}", "GraviNum", $"{size}x{size} matrix product", () => LinAlg.Dot(a, b));
}

foreach (var length in new[] { 1_000_000, 10_000_000 })
{
    var a = rng.StandardNormal(length);
    var b = rng.StandardNormal(length);
    Measure($"elementwise_add_{length}", "GraviNum", $"element-wise add, {length:N0} elements",
        () => UFunc.Add(a, b));
}

{
    var m = rng.StandardNormal(256, 256) + NdArray.Eye(256) * 256;
    var spd = LinAlg.Dot(m, m.T) + NdArray.Eye(256) * 256;
    var rhs = rng.StandardNormal(256);

    Measure("lu_256", "GraviNum", "LU factorisation, 256x256", () => Decomposition.Lu(m));
    Measure("qr_256", "GraviNum", "QR factorisation, 256x256", () => Decomposition.Qr(m));
    Measure("cholesky_256", "GraviNum", "Cholesky factorisation, 256x256", () => Decomposition.Cholesky(spd));
    Measure("svd_256", "GraviNum", "SVD, 256x256", () => Decomposition.Svd(m), repeats: 3);
    Measure("eigh_256", "GraviNum", "symmetric eigen, 256x256", () => Decomposition.SymmetricEigen(spd), repeats: 3);
    Measure("solve_256", "GraviNum", "solve Ax=b, 256x256", () => LinAlg.Solve(m, rhs));
    Measure("inverse_256", "GraviNum", "matrix inverse, 256x256", () => LinAlg.Inverse(m));
}

{
    // 2000x2000 at 1% density, the regime where CSR is clearly the right choice.
    const int size = 2000;
    var dense = NdArray.Zeros(size, size);
    var nonZeros = (int)(size * (long)size * 0.01);
    for (var k = 0; k < nonZeros; k++) dense[rng.Next(size), rng.Next(size)] = rng.Normal();

    var sparse = SparseMatrix.FromDense(dense);
    var vector = rng.StandardNormal(size);

    Measure("sparse_matvec_2000", "GraviNum", "sparse matrix-vector, 2000x2000 @ 1%",
        () => sparse.Multiply(vector));
    Measure("dense_matvec_2000", "GraviNum", "dense matrix-vector, 2000x2000",
        () => LinAlg.Dot(dense, vector));
}

{
    Measure("random_normal_1m", "GraviNum", "1,000,000 normal deviates",
        () => new GraviRandom(1).StandardNormal(1_000_000));

    var data = rng.StandardNormal(1_000_000);
    Measure("statistics_1m", "GraviNum", "mean + std over 1,000,000",
        () => Sink.Consume(Statistics.Mean(data) + Statistics.Std(data)));
}

// ---------------------------------------------------------------- GraviFrame

var csvPath = Path.Combine(Path.GetTempPath(), "gravicode-comparison.csv");
WriteComparisonCsv(csvPath, 200_000);
Console.WriteLine($"  fixture : {csvPath} ({new FileInfo(csvPath).Length / 1024 / 1024.0:F1} MB)");
Console.WriteLine();

Measure("csv_read_200k", "GraviFrame", "read 200,000-row CSV", () => DataFrame.ReadCsv(csvPath), repeats: 3);

{
    var frame = DataFrame.ReadCsv(csvPath);
    Measure("groupby_mean_200k", "GraviFrame", "group-by mean, 500 groups",
        () => frame.GroupBy("group").Mean("value"));
    Measure("groupby_composite_200k", "GraviFrame", "group-by two keys",
        () => frame.GroupBy("group", "label").Sum("value"));
    Measure("sort_200k", "GraviFrame", "sort by a numeric column", () => frame.SortBy("value"), repeats: 3);
    Measure("rolling_mean_200k", "GraviFrame", "rolling mean, window 30",
        () => frame.Numeric("value").Rolling(30).Mean());
    Measure("filter_200k", "GraviFrame", "filter a numeric column",
        () => frame.FilterBy("value", v => v > 100));
    Measure("describe_200k", "GraviFrame", "describe all numeric columns", () => frame.Describe(), repeats: 3);
}

// ---------------------------------------------------------------- GraviLearn

{
    var features = rng.StandardNormal(20_000, 20);
    var labels = NdArray.Zeros(20_000);
    var truth = rng.StandardNormal(20);
    for (var i = 0; i < 20_000; i++)
    {
        var score = 0.0;
        for (var j = 0; j < 20; j++) score += features[i, j] * truth.At(j);
        labels.SetAt(i, score > 0 ? 1 : 0);
    }

    Measure("logistic_fit_20k", "GraviLearn", "logistic regression, 20k x 20, 100 iterations", () =>
    {
        var model = new LogisticRegression(learningRate: 0.1, maxIterations: 100);
        model.Fit(features, labels);
    }, repeats: 3);

    Measure("randomforest_fit_20k", "GraviLearn", "random forest fit, 50 trees, depth 8", () =>
    {
        var model = new RandomForestClassifier(nTrees: 50, maxDepth: 8, seed: 42);
        model.Fit(features, labels);
    }, repeats: 3);

    var forest = new RandomForestClassifier(nTrees: 50, maxDepth: 8, seed: 42);
    forest.Fit(features, labels);
    Measure("randomforest_predict_20k", "GraviLearn", "random forest predict, 20k rows",
        () => forest.Predict(features), repeats: 3);

    Measure("kmeans_20k", "GraviLearn", "k-means, k=5, 3 restarts", () =>
    {
        var model = new KMeans(clusters: 5, restarts: 3, seed: 42);
        model.Fit(features);
    }, repeats: 3);

    Measure("pca_20k", "GraviLearn", "PCA to 5 components, 20k x 20",
        () => new PrincipalComponentAnalysis(5).FitTransform(features), repeats: 3);

    var knnTrain = rng.StandardNormal(5_000, 20);
    var knnLabels = NdArray.Zeros(5_000);
    for (var i = 0; i < 5_000; i++) knnLabels.SetAt(i, i % 3);
    var knn = new KNearestNeighborsClassifier(k: 5);
    knn.Fit(knnTrain, knnLabels);
    var knnQuery = rng.StandardNormal(2_000, 20);
    Measure("knn_predict_2k", "GraviLearn", "kNN predict, 2k queries against 5k points",
        () => knn.Predict(knnQuery), repeats: 3);
}

// ---------------------------------------------------------------- GraviText

{
    var documents = BuildCorpus(20_000, 40);
    Measure("tfidf_20k", "GraviText", "TF-IDF fit+transform, 20k documents", () =>
    {
        var vectorizer = new TfidfVectorizer(new VectorizerOptions { MinDocumentFrequency = 2 });
        vectorizer.Fit(documents);
        _ = vectorizer.FeatureCount;
    }, repeats: 3);

    var tokenizer = new Gravicode.Science.GraviText.Tokenization.RegexTokenizer();
    Measure("tokenize_20k", "GraviText", "regex tokenize 20k documents", () =>
    {
        var total = 0;
        foreach (var document in documents) total += tokenizer.Tokenize(document).Count;
        _ = total;
    }, repeats: 3);
}

// ---------------------------------------------------------------- GraviGraph

{
    var graph = Graph.ScaleFree(50_000, edgesPerNode: 3, seed: 42);
    Measure("pagerank_50k", "GraviGraph", "PageRank, 50k nodes",
        () => GraphAlgorithms.PageRank(graph), repeats: 3);
    Measure("bfs_50k", "GraviGraph", "breadth-first search, 50k nodes",
        () => GraphAlgorithms.BreadthFirstSearch(graph, 0), repeats: 3);
    Measure("components_50k", "GraviGraph", "connected components, 50k nodes",
        () => GraphAlgorithms.ConnectedComponents(graph), repeats: 3);
    Measure("dijkstra_50k", "GraviGraph", "Dijkstra from one source, 50k nodes",
        () => GraphAlgorithms.ShortestPaths(graph, 0), repeats: 3);
}

// ---------------------------------------------------------------- GraviProb

{
    var data = Enumerable.Range(0, 500).Select(_ => rng.Normal(5.0, 2.0)).ToArray();
    var model = new BayesianModel()
        .AddDistribution("mu", Distribution.Normal(0, 10))
        .AddDistribution("sigma", Distribution.HalfNormal(5))
        .AddObservation("y", DistributionSpec.Normal("mu", "sigma"), data);

    Measure("mcmc_4x5000", "GraviProb", "Metropolis-Hastings, 4 chains x 5000 draws",
        () => model.SampleMCMC(iterations: 5_000, chains: 4, warmup: 2_500, seed: 42), repeats: 3);

    var normal = Distribution.Normal(0, 1);
    Measure("logpdf_1m", "GraviProb", "1,000,000 normal log densities", () =>
    {
        var total = 0.0;
        for (var i = 0; i < 1_000_000; i++) total += normal.LogDensity(i * 1e-6);
        Sink.Consume(total);
    });
}

// ---------------------------------------------------------------- output

var payload = new Payload(
    "dotnet",
    $".NET {Environment.Version}",
    Environment.ProcessorCount,
    Vector<double>.Count,
    DateTime.UtcNow.ToString("O"),
    results);

File.WriteAllText(outputPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));

Console.WriteLine();
Console.WriteLine($"Wrote {results.Count} measurements to {outputPath}");
Console.WriteLine($"  (sink checksum {Sink.Total:E3} - printed so the JIT cannot elide the work)");
File.Delete(csvPath);
return;

// ---------------------------------------------------------------- helpers

void Measure(string id, string library, string operation, Action action, int repeats = 7, int warmup = 2)
{
    for (var i = 0; i < warmup; i++) action();

    var timings = new double[repeats];
    for (var i = 0; i < repeats; i++)
    {
        var watch = Stopwatch.StartNew();
        action();
        watch.Stop();
        timings[i] = watch.Elapsed.TotalMilliseconds;
    }

    Array.Sort(timings);
    var median = timings[repeats / 2];
    results.Add(new Measurement(id, library, operation, median, timings[0], timings[^1], repeats));
    Console.WriteLine($"  {library,-12}{id,-28}{median,10:F3} ms");
}

static void WriteComparisonCsv(string path, int rows)
{
    if (File.Exists(path)) File.Delete(path);

    var rng = new GraviRandom(7);
    using var writer = new StreamWriter(path);
    writer.WriteLine("id,group,label,value,quantity");
    for (var i = 0; i < rows; i++)
        writer.WriteLine($"{i},{i % 500},cat{i % 20},{rng.Normal(100, 25):F4},{rng.Next(1, 50)}");
}

static string[] BuildCorpus(int count, int wordsPerDocument)
{
    string[] vocabulary =
    [
        "the", "model", "learns", "from", "data", "and", "produces", "an", "embedding", "vector",
        "graph", "network", "attention", "layer", "token", "sentence", "document", "corpus",
        "bagus", "sangat", "menarik", "hasil", "jaringan", "kata", "kalimat", "analisis",
    ];

    var rng = new GraviRandom(11);
    var documents = new string[count];
    for (var i = 0; i < count; i++)
    {
        var words = new string[wordsPerDocument];
        for (var w = 0; w < wordsPerDocument; w++) words[w] = vocabulary[rng.Next(vocabulary.Length)];
        documents[i] = string.Join(' ', words);
    }
    return documents;
}

/// <summary>
/// Keeps computed values alive so the JIT cannot delete the loop that produced them.
/// </summary>
/// <remarks>
/// Without this, a benchmark whose result is discarded gets eliminated as dead code. The scalar
/// log-density loop originally "ran" 1,000,000 iterations in 0.45 ms - about 1.5 cycles each,
/// which is impossible for a logarithm. Writing to a non-inlined static sink forces the work to
/// actually happen.
/// </remarks>
internal static class Sink
{
    private static double _value;

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    public static void Consume(double value) => _value += value;

    /// <summary>Read at the end so the accumulated total is observably used.</summary>
    public static double Total => _value;
}

internal sealed record Measurement(
    string Id, string Library, string Operation,
    double MedianMs, double MinMs, double MaxMs, int Repeats);

internal sealed record Payload(
    string Stack, string Runtime, int Processors, int SimdWidth, string TimestampUtc,
    IReadOnlyList<Measurement> Results);
