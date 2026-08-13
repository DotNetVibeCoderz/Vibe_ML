using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Gravicode.Science.GraviProb.Models;

BenchmarkSwitcher.FromAssembly(typeof(McmcChainBenchmark).Assembly).Run(args, DefaultConfig.Instance
    .AddJob(Job.ShortRun.WithWarmupCount(1).WithIterationCount(3))
    .WithOptions(ConfigOptions.DisableOptimizationsValidator));
return;

/// <summary>
/// How MCMC scales as chains are added.
/// </summary>
/// <remarks>
/// Chains are completely independent, so running four of them should cost about the same wall
/// clock as one on a machine with four free cores - the work is spread rather than repeated.
/// That is the scaling this benchmark checks, and it is why multi-chain sampling is nearly free
/// while also being the only way to compute the R-hat convergence diagnostic.
/// </remarks>
[MemoryDiagnoser]
public class McmcChainBenchmark
{
    private BayesianModel _model = new();

    /// <summary>Number of independent chains.</summary>
    [Params(1, 2, 4, 8)]
    public int Chains { get; set; }

    /// <summary>Draws per chain.</summary>
    [Params(10_000)]
    public int Iterations { get; set; }

    /// <summary>Builds a two-parameter normal model over 500 observations.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        var data = Enumerable.Range(0, 500).Select(_ => rng.Normal(5.0, 2.0)).ToArray();

        _model = new BayesianModel()
            .AddDistribution("mu", Distribution.Normal(0, 10))
            .AddDistribution("sigma", Distribution.HalfNormal(5))
            .AddObservation("y", DistributionSpec.Normal("mu", "sigma"), data);

        Console.WriteLine($"[setup] {Environment.ProcessorCount} logical processors available");
    }

    /// <summary>Random-walk Metropolis-Hastings.</summary>
    [Benchmark(Baseline = true)]
    public double MetropolisHastings()
        => _model.SampleMCMC(Iterations, Chains, Iterations / 2, seed: 42).Mean("mu");

    /// <summary>Metropolis within Gibbs, which proposes one coordinate at a time.</summary>
    [Benchmark]
    public double MetropolisWithinGibbs()
        => _model.SampleGibbs(Iterations, Chains, Iterations / 2, seed: 42).Mean("mu");
}

/// <summary>
/// Sampling cost against variational inference, the trade-off between exactness and speed.
/// </summary>
[MemoryDiagnoser]
public class InferenceMethodBenchmark
{
    private BayesianModel _model = new();

    /// <summary>Number of observations in the likelihood.</summary>
    [Params(100, 1_000, 10_000)]
    public int Observations { get; set; }

    /// <summary>Builds a Poisson rate model.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        var data = Enumerable.Range(0, Observations).Select(_ => (double)rng.Poisson(4.5)).ToArray();

        _model = new BayesianModel()
            .AddDistribution("lambda", Distribution.Gamma(1, 0.1))
            .AddObservation("counts", DistributionSpec.Poisson("lambda"), data);
    }

    /// <summary>MCMC with four chains.</summary>
    [Benchmark(Baseline = true)]
    public double Mcmc() => _model.SampleMCMC(5_000, chains: 4, warmup: 2_500, seed: 42).Mean("lambda");

    /// <summary>Mean-field variational inference, which optimises instead of sampling.</summary>
    [Benchmark]
    public double Variational() => _model.FitVariational(iterations: 500, seed: 42).Means["lambda"];

    /// <summary>One log-posterior evaluation, the inner loop both methods share.</summary>
    [Benchmark]
    public double SingleLogPosterior()
        => _model.LogPosterior(new Dictionary<string, double> { ["lambda"] = 4.5 });
}

/// <summary>Distribution sampling and density throughput.</summary>
[MemoryDiagnoser]
public class DistributionBenchmark
{
    private readonly GraviRandom _rng = new(42);
    private Distribution _distribution = Distribution.Normal();

    /// <summary>Number of draws per call.</summary>
    [Params(100_000)]
    public int Draws { get; set; }

    /// <summary>The distribution family under test.</summary>
    [Params("Normal", "Gamma", "Beta", "Poisson", "Binomial")]
    public string Family { get; set; } = "Normal";

    /// <summary>Selects the distribution.</summary>
    [GlobalSetup]
    public void Setup() => _distribution = Family switch
    {
        "Gamma" => Distribution.Gamma(2.5, 1.5),
        "Beta" => Distribution.Beta(2, 5),
        "Poisson" => Distribution.Poisson(40.0),
        "Binomial" => Distribution.Binomial(200, 0.3),
        _ => Distribution.Normal(0, 1),
    };

    /// <summary>Drawing samples.</summary>
    [Benchmark(Baseline = true)]
    public double Sample() => _distribution.Sample(_rng, Draws).At(0);

    /// <summary>Evaluating the log density.</summary>
    [Benchmark]
    public double LogDensity()
    {
        var total = 0.0;
        for (var i = 0; i < Draws; i++) total += _distribution.LogDensity(1.0 + i % 10);
        return total;
    }
}

/// <summary>Hidden Markov model algorithms as the sequence grows.</summary>
[MemoryDiagnoser]
public class HiddenMarkovBenchmark
{
    private HiddenMarkovModel _model = null!;
    private int[] _observations = [];
    private List<IReadOnlyList<int>> _sequences = [];

    /// <summary>Length of each observation sequence.</summary>
    [Params(1_000, 10_000)]
    public int Length { get; set; }

    /// <summary>Builds a model and generates sequences from it.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _model = new HiddenMarkovModel(
            [0.6, 0.4],
            new[,] { { 0.7, 0.3 }, { 0.4, 0.6 } },
            new[,] { { 0.1, 0.4, 0.5 }, { 0.6, 0.3, 0.1 } });

        var rng = new GraviRandom(42);
        _observations = _model.Generate(Length, rng).Observations;
        _sequences = Enumerable.Range(0, 5)
            .Select(_ => (IReadOnlyList<int>)_model.Generate(Length / 5, rng).Observations)
            .ToList();
    }

    /// <summary>The forward recursion, with per-step scaling.</summary>
    [Benchmark(Baseline = true)]
    public double Forward() => _model.LogLikelihood(_observations);

    /// <summary>Viterbi decoding in log space.</summary>
    [Benchmark]
    public int Viterbi() => _model.Viterbi(_observations)[0];

    /// <summary>Forward-backward posteriors.</summary>
    [Benchmark]
    public double StatePosteriors() => _model.StatePosteriors(_observations)[0, 0];

    /// <summary>Ten Baum-Welch iterations over five sequences.</summary>
    [Benchmark]
    public double BaumWelch()
        => HiddenMarkovModel.Random(2, 3, seed: 7).Fit(_sequences, iterations: 10);
}

/// <summary>Conjugate Bayesian regression, where the posterior is closed form.</summary>
[MemoryDiagnoser]
public class BayesianRegressionBenchmark
{
    private NdArray _x = NdArray.Zeros(1, 1);
    private NdArray _y = NdArray.Zeros(1);

    /// <summary>Number of observations.</summary>
    [Params(1_000, 10_000)]
    public int Samples { get; set; }

    /// <summary>Number of predictors.</summary>
    [Params(10, 50)]
    public int Features { get; set; }

    /// <summary>Generates a linear problem.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        _x = rng.StandardNormal(Samples, Features);
        _y = NdArray.Zeros(Samples);

        var truth = rng.StandardNormal(Features);
        for (var i = 0; i < Samples; i++)
        {
            var value = 0.0;
            for (var j = 0; j < Features; j++) value += _x[i, j] * truth.At(j);
            _y.SetAt(i, value + rng.Normal(0, 0.5));
        }
    }

    /// <summary>Fitting the conjugate posterior: one Gram matrix and one inverse.</summary>
    [Benchmark(Baseline = true)]
    public double Fit() => new BayesianLinearRegression().Fit(_x, _y).CoefficientMeans.At(0);

    /// <summary>Prediction with per-row uncertainty, which needs a quadratic form per row.</summary>
    [Benchmark]
    public double PredictWithUncertainty()
        => new BayesianLinearRegression().Fit(_x, _y).PredictWithUncertainty(_x).StandardDeviation.At(0);
}
