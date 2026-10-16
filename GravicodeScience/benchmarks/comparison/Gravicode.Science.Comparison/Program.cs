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
using SciNet = Gravicode.Science.SciNet;

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

// ---------------------------------------------------------------- SciNet (vs SciPy)
// Inputs come from the same deterministic formula on both sides (U below), not from either
// stack's RNG, so the two halves solve the same problems. Every SciNet workload gets 40 warm-up
// calls on this side: tiered JIT recompiles a hot method after ~30 calls, and with 5 the Lorenz
// solve measured 19 ms against 6.5 ms at steady state. Python has no JIT, so its warm-up only
// warms caches and stays short.
{
    double[] Uniform(int n, int salt) => Enumerable.Range(0, n).Select(i => U(i, salt)).ToArray();
    NdArray Points(int n, int d, int salt) => new(Uniform(n * d, salt), n, d);

    var erfInput = new NdArray(Uniform(1_000_000, 1).Select(v => 8 * v - 4).ToArray(), 1_000_000);
    Measure("scinet_erf_1m", "SciNet", "erf over 1,000,000 values", () => Sink.Consume(SciNet.Special.Erf(erfInput).At(0)), warmup: 40);

    var complexSignal = Uniform(65536, 2).Select((v, i) => new Complex(v, U(i, 3))).ToArray();
    Measure("scinet_fft_65536", "SciNet", "complex FFT, 65,536 points", () => Sink.Consume(SciNet.Fft.Fft1D(complexSignal)[1].Real), warmup: 40);
    var realSignal = Uniform(1 << 20, 4);
    Measure("scinet_rfft_1m", "SciNet", "real FFT, 2^20 points", () => Sink.Consume(SciNet.Fft.Rfft(realSignal)[1].Real), warmup: 40);
    var dctSignal = Uniform(65536, 5);
    Measure("scinet_dct_65536", "SciNet", "DCT-II, 65,536 points", () => Sink.Consume(SciNet.Fft.Dct(dctSignal)[0]), warmup: 40);

    Measure("scinet_quad", "SciNet", "quad of log(x)/sqrt(x) on [0,1]",
        () => Sink.Consume(SciNet.Integrate.Quad(x => Math.Log(x) / Math.Sqrt(x), 0, 1).Value), warmup: 40);

    Measure("scinet_ivp_lorenz", "SciNet", "solve_ivp Lorenz 0..100, RK45, rtol 1e-9", () =>
    {
        var r = SciNet.Integrate.SolveIvp(static (double t, ReadOnlySpan<double> y, Span<double> dy) =>
        {
            dy[0] = 10 * (y[1] - y[0]);
            dy[1] = y[0] * (28 - y[2]) - y[1];
            dy[2] = y[0] * y[1] - 8.0 / 3.0 * y[2];
        }, 0, 100, [1.0, 1.0, 1.0], SciNet.OdeMethod.Rk45, rtol: 1e-9, atol: 1e-9);
        Sink.Consume(r.T.Length);
    }, repeats: 5, warmup: 40);
    Measure("scinet_ivp_vdp", "SciNet", "solve_ivp Van der Pol mu=1000, BDF", () =>
    {
        var r = SciNet.Integrate.SolveIvp(static (double t, ReadOnlySpan<double> y, Span<double> dy) =>
        {
            dy[0] = y[1];
            dy[1] = 1000 * (1 - y[0] * y[0]) * y[1] - y[0];
        }, 0, 3000, [2.0, 0.0], SciNet.OdeMethod.Bdf);
        Sink.Consume(r.T.Length);
    }, repeats: 5, warmup: 40);

    var rosenStart = Enumerable.Repeat(-1.0, 100).ToArray();
    Measure("scinet_bfgs_rosen100", "SciNet", "minimize BFGS, Rosenbrock-100 with gradient", () =>
        Sink.Consume(SciNet.Optimize.Minimize(SciNet.Optimize.Rosen, rosenStart,
            SciNet.MinimizeMethod.Bfgs, jac: SciNet.Optimize.RosenDer).Fun), repeats: 5, warmup: 40);
    var fitX = Enumerable.Range(0, 10_000).Select(i => i * 0.001).ToArray();
    var fitY = fitX.Select((x, i) => 2.5 * Math.Exp(-1.3 * x) + 0.5 + 0.05 * (U(i, 6) - 0.5)).ToArray();
    Measure("scinet_curve_fit_10k", "SciNet", "curve_fit exponential decay, 10,000 points", () =>
        Sink.Consume(SciNet.Optimize.CurveFit(static (double[] x, double[] p) =>
        {
            var y = new double[x.Length];
            for (var i = 0; i < x.Length; i++) y[i] = p[0] * Math.Exp(-p[1] * x[i]) + p[2];
            return y;
        }, fitX, fitY, p0: [1.0, 1.0, 1.0]).Popt[0]), repeats: 5, warmup: 40);
    var assignment = Points(500, 500, 7);
    Measure("scinet_lsa_500", "SciNet", "linear_sum_assignment 500x500",
        () => Sink.Consume(SciNet.Optimize.LinearSumAssignment(assignment).Item2[0]), repeats: 5, warmup: 40);

    var clouds = Points(2000, 64, 8);
    Measure("scinet_cdist_2000x64", "SciNet", "cdist 2000x2000 in 64-D, euclidean",
        () => Sink.Consume(SciNet.Spatial.Distance.Cdist(clouds, clouds).At(1)), repeats: 5, warmup: 40);
    var cloud3 = Points(100_000, 3, 9);
    Measure("scinet_kdtree_100k", "SciNet", "KD-tree build + 5-NN query, 100k points in 3-D", () =>
    {
        var tree = new SciNet.Spatial.KDTree(cloud3);
        Sink.Consume(tree.Query(cloud3, k: 5).Distances.At(1));
    }, repeats: 5, warmup: 40);
    var cloud2 = Points(100_000, 2, 10);
    Measure("scinet_delaunay_100k", "SciNet", "Delaunay triangulation, 100k points",
        () => Sink.Consume(new SciNet.Spatial.Delaunay(cloud2).Simplices.Length), repeats: 5, warmup: 40);
    var linkageData = Points(2000, 8, 11);
    Measure("scinet_linkage_ward_2000", "SciNet", "hierarchical clustering, Ward, 2000 points",
        () => Sink.Consume(SciNet.Cluster.Linkage(linkageData, SciNet.LinkageMethod.Ward).At(0)), repeats: 5, warmup: 40);

    var knots = Enumerable.Range(0, 1000).Select(i => i * 0.01).ToArray();
    var spline = new SciNet.CubicSpline(knots, knots.Select(Math.Sin).ToArray());
    var splineQueries = Uniform(1_000_000, 12).Select(v => v * 9.99).ToArray();
    Measure("scinet_spline_eval_1m", "SciNet", "cubic spline, evaluate 1,000,000 points",
        () => Sink.Consume(spline.Evaluate(splineQueries)[0]), warmup: 40);
    var gridAxis = Enumerable.Range(0, 100).Select(i => i / 99.0).ToArray();
    var gridValues = new NdArray(Uniform(100 * 100 * 100, 13), 100, 100, 100);
    var regular = new SciNet.RegularGridInterpolator([gridAxis, gridAxis, gridAxis], gridValues);
    var gridQueries = Points(1_000_000, 3, 14);
    Measure("scinet_regular_grid_1m", "SciNet", "RegularGridInterpolator linear, 1,000,000 points in 3-D",
        () => Sink.Consume(regular.Evaluate(gridQueries).At(0)), repeats: 5, warmup: 40);

    double[][] Tridiagonal(int n) =>
        [Enumerable.Repeat(-1.0, n - 1).ToArray(), Enumerable.Repeat(2.0, n).ToArray(), Enumerable.Repeat(-1.0, n - 1).ToArray()];
    var second = SciNet.Sparse.SparseMatrices.Diags(Tridiagonal(200), [-1, 0, 1]);
    var poisson = SciNet.Sparse.SparseMatrices.Kronsum(second, second);
    var ones = Enumerable.Repeat(1.0, 40_000).ToArray();
    Measure("scinet_cg_poisson_200", "SciNet", "conjugate gradient, 2-D Poisson 200x200, rtol 1e-8",
        () => Sink.Consume(SciNet.Sparse.SparseLinalg.Cg(poisson, ones, rtol: 1e-8, maxiter: 5000).X[0]), repeats: 5, warmup: 40);
    var small = SciNet.Sparse.SparseMatrices.Diags(Tridiagonal(100), [-1, 0, 1]);
    var poissonSmall = SciNet.Sparse.SparseMatrices.Kronsum(small, small);
    Measure("scinet_splu_poisson_100", "SciNet", "sparse LU factor + solve, 2-D Poisson 100x100",
        () => Sink.Consume(SciNet.Sparse.SparseLinalg.Splu(poissonSmall).Solve(ones.AsSpan(0, 10_000))[0]), repeats: 5, warmup: 40);
    var expmInput = new NdArray(Uniform(200 * 200, 15).Select(v => v - 0.5).ToArray(), 200, 200);
    Measure("scinet_expm_200", "SciNet", "matrix exponential, 200x200",
        () => Sink.Consume(SciNet.Linalg.Expm(expmInput).At(0)), repeats: 5, warmup: 40);

    var image = new NdArray(Uniform(2048 * 2048, 16), 2048, 2048);
    Measure("scinet_gaussian_2048", "SciNet", "gaussian_filter sigma 3, 2048x2048",
        () => Sink.Consume(SciNet.NdImage.GaussianFilter(image, 3.0).At(0)), repeats: 5, warmup: 40);
    var image1k = new NdArray(Uniform(1024 * 1024, 17), 1024, 1024);
    Measure("scinet_median_1024", "SciNet", "median_filter 5x5, 1024x1024",
        () => Sink.Consume(SciNet.NdImage.MedianFilter(image1k, 5).At(0)), repeats: 5, warmup: 40);

    var samples = Uniform(1_000_000, 18);
    var butter = SciNet.Signal.Butter(4, 0.1);
    Measure("scinet_lfilter_1m", "SciNet", "lfilter Butterworth order 4, 1,000,000 samples",
        () => Sink.Consume(SciNet.Signal.Lfilter(butter.B, butter.A, samples)[0]), warmup: 40);
    var sos = SciNet.Signal.ButterSos(8, 0.1);
    Measure("scinet_sosfiltfilt_1m", "SciNet", "sosfiltfilt Butterworth order 8, 1,000,000 samples",
        () => Sink.Consume(SciNet.Signal.Sosfiltfilt(sos, samples)[0]), warmup: 40);
    var longRecording = Uniform(10_000_000, 19);
    Measure("scinet_welch_10m", "SciNet", "welch PSD, 10,000,000 samples, nperseg 4096",
        () => Sink.Consume(SciNet.Signal.Welch(longRecording, nperseg: 4096).Pxx[1]), repeats: 5, warmup: 40);
    var convKernel = Uniform(10_000, 20);
    Measure("scinet_fftconvolve_1m", "SciNet", "fftconvolve 1,000,000 x 10,000",
        () => Sink.Consume(SciNet.Signal.Fftconvolve(samples, convKernel)[0]), repeats: 5, warmup: 40);
    Measure("scinet_find_peaks_1m", "SciNet", "find_peaks with prominence, 1,000,000 samples",
        () => Sink.Consume(SciNet.Signal.FindPeaks(samples, prominence: 0.5).Peaks.Length), repeats: 5, warmup: 40);
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

/// <summary>
/// The deterministic input both halves of the SciNet comparison use: a hash of (index, salt) mapped
/// to [0, 1). Same formula as <c>u()</c> in python_benchmarks.py.
/// </summary>
static double U(int i, int salt)
{
    var v = Math.Sin(i * 12.9898 + salt * 78.233) * 43758.5453;
    return v - Math.Floor(v);
}


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
