using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Clustering;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.Neighbors;
using Gravicode.Science.GraviLearn.Preprocessing;
using Gravicode.Science.GraviLearn.Trees;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Compute;

BenchmarkSwitcher.FromAssembly(typeof(LogisticRegressionBenchmark).Assembly).Run(args, DefaultConfig.Instance
    .AddJob(Job.ShortRun.WithWarmupCount(2).WithIterationCount(4))
    .WithOptions(ConfigOptions.DisableOptimizationsValidator));
return;

/// <summary>
/// Logistic regression training cost, and where the GPU could take over.
/// </summary>
/// <remarks>
/// Batch gradient descent spends nearly all its time in two matrix products per iteration:
/// <c>X w</c> to score every sample and <c>X' e</c> to accumulate the gradient. Those are exactly
/// the operations a GPU is built for, so this benchmark measures both the full CPU training loop
/// and the isolated per-iteration products on each backend - which is what determines whether
/// moving the loop to the GPU would pay for the transfers.
/// </remarks>
[MemoryDiagnoser]
public class LogisticRegressionBenchmark
{
    private NdArray _x = NdArray.Zeros(1, 1);
    private NdArray _y = NdArray.Zeros(1);
    private double[] _xFlat = [];
    private double[] _wFlat = [];
    private IComputeBackend? _gpu;

    /// <summary>Number of training samples.</summary>
    [Params(5_000, 50_000)]
    public int Samples { get; set; }

    /// <summary>Number of features.</summary>
    [Params(32, 256)]
    public int Features { get; set; }

    /// <summary>Generates a linearly separable problem.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        _x = rng.StandardNormal(Samples, Features);
        _y = NdArray.Zeros(Samples);

        var truth = rng.StandardNormal(Features);
        for (var i = 0; i < Samples; i++)
        {
            var score = 0.0;
            for (var j = 0; j < Features; j++) score += _x[i, j] * truth.At(j);
            _y.SetAt(i, score > 0 ? 1 : 0);
        }

        _xFlat = _x.ToArray();
        _wFlat = truth.ToArray();
        _gpu = Compute.Gpu;
        Console.WriteLine($"[setup] {Compute.DescribeDevices()}");
    }

    /// <summary>Full training on the CPU, 50 gradient descent iterations.</summary>
    [Benchmark(Baseline = true)]
    public double TrainCpu()
    {
        var model = new LogisticRegression(learningRate: 0.1, maxIterations: 50);
        model.Fit(_x, _y);
        return model.Intercept;
    }

    /// <summary>One iteration's scoring product on the CPU backend.</summary>
    [Benchmark]
    public double ScoringProductCpu()
        => Compute.Cpu.MatMul(_xFlat, _wFlat, Samples, Features, 1)[0];

    /// <summary>The same product on the GPU, transfers included.</summary>
    [Benchmark]
    public double ScoringProductGpu()
    {
        if (_gpu is null) return ScoringProductCpu();
        return _gpu.MatMul(_xFlat, _wFlat, Samples, Features, 1)[0];
    }

    /// <summary>Releases the accelerator.</summary>
    [GlobalCleanup]
    public void Cleanup() => Compute.ReleaseGpu();
}

/// <summary>Training cost across the model families, at a fixed problem size.</summary>
[MemoryDiagnoser]
public class ModelTrainingBenchmark
{
    private NdArray _x = NdArray.Zeros(1, 1);
    private NdArray _y = NdArray.Zeros(1);

    /// <summary>Number of training samples.</summary>
    [Params(2_000, 10_000)]
    public int Samples { get; set; }

    /// <summary>Generates a three-class problem with 20 features.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var data = Datasets.MakeBlobs(Samples, features: 20, centers: 3, spread: 2.0, seed: 42);
        _x = data.Features;
        _y = data.Target;
    }

    /// <summary>A single decision tree.</summary>
    [Benchmark(Baseline = true)]
    public bool DecisionTree()
    {
        var model = new DecisionTree(maxDepth: 8);
        model.Fit(_x, _y);
        return model.IsFitted;
    }

    /// <summary>A 50-tree forest, fitted in parallel.</summary>
    [Benchmark]
    public bool RandomForest()
    {
        var model = new RandomForestClassifier(nTrees: 50, maxDepth: 8, seed: 42);
        model.Fit(_x, _y);
        return model.IsFitted;
    }

    /// <summary>Gradient boosting, which is sequential by construction.</summary>
    [Benchmark]
    public bool GradientBoosting()
    {
        var model = new GradientBoostingClassifier(nTrees: 50, maxDepth: 3);
        model.Fit(_x, _y);
        return model.IsFitted;
    }

    /// <summary>Gaussian naive Bayes: a single pass over the data.</summary>
    [Benchmark]
    public bool NaiveBayes()
    {
        var model = new GaussianNaiveBayes();
        model.Fit(_x, _y);
        return model.IsFitted;
    }

    /// <summary>k-means with three restarts.</summary>
    [Benchmark]
    public double KMeans()
    {
        var model = new KMeans(clusters: 3, restarts: 3, seed: 42);
        model.Fit(_x);
        return model.Inertia;
    }

    /// <summary>Principal components, dominated by the SVD.</summary>
    [Benchmark]
    public double Pca() => new PrincipalComponentAnalysis(5).FitTransform(_x)[0, 0];
}

/// <summary>
/// Prediction throughput, which for a deployed model matters more than training time.
/// </summary>
[MemoryDiagnoser]
public class PredictionBenchmark
{
    private NdArray _test = NdArray.Zeros(1, 1);
    private RandomForestClassifier _forest = new();
    private LogisticRegression _logistic = new();
    private KNearestNeighborsClassifier _knn = new();
    private Pipeline _pipeline = new();

    /// <summary>Number of rows scored per call.</summary>
    [Params(1_000, 10_000)]
    public int Rows { get; set; }

    /// <summary>Trains each model once, outside the measured region.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var train = Datasets.MakeBlobs(5_000, features: 20, centers: 3, spread: 2.0, seed: 42);
        _test = Datasets.MakeBlobs(Rows, features: 20, centers: 3, spread: 2.0, seed: 7).Features;

        _forest = new RandomForestClassifier(nTrees: 50, maxDepth: 8, seed: 42);
        _forest.Fit(train.Features, train.Target);

        _logistic = new LogisticRegression(learningRate: 0.1, maxIterations: 200);
        _logistic.Fit(train.Features, train.Target);

        _knn = new KNearestNeighborsClassifier(k: 5);
        _knn.Fit(train.Features, train.Target);

        _pipeline = new Pipeline()
            .Add(new StandardScaler())
            .Add(new PCA(components: 8))
            .Add(new RandomForestClassifier(nTrees: 50, seed: 42));
        _pipeline.Fit(train.Features, train.Target);
    }

    /// <summary>Logistic regression: one matrix product.</summary>
    [Benchmark(Baseline = true)]
    public double Logistic() => _logistic.Predict(_test).At(0);

    /// <summary>Random forest: 50 tree descents per row.</summary>
    [Benchmark]
    public double RandomForest() => _forest.Predict(_test).At(0);

    /// <summary>kNN: a distance to every training point, which is why lazy learners are slow to serve.</summary>
    [Benchmark]
    public double KNearestNeighbors() => _knn.Predict(_test).At(0);

    /// <summary>The full pipeline, transformations included.</summary>
    [Benchmark]
    public double Pipeline() => _pipeline.Predict(_test).At(0);
}
