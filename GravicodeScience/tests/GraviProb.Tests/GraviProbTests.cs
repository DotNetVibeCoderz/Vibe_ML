using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Gravicode.Science.GraviProb.Models;
using Xunit;

namespace Gravicode.Science.Tests.GraviProb;

public class DistributionTests
{
    [Fact]
    public void Normal_DensityMatchesTheClosedForm()
    {
        var normal = Distribution.Normal(0, 1);
        Assert.Equal(1.0 / Math.Sqrt(2 * Math.PI), normal.Density(0), 12);
        Assert.Equal(0.0, normal.Mean, 12);
        Assert.Equal(1.0, normal.Variance, 12);
        Assert.Equal(0.5, normal.Cdf(0), 10);
    }

    [Fact]
    public void Normal_SamplesReproduceItsMoments()
    {
        var samples = Distribution.Normal(3.0, 2.0).Sample(new GraviRandom(1), 50_000);
        Assert.Equal(3.0, Statistics.Mean(samples), 1);
        Assert.Equal(2.0, Statistics.Std(samples), 1);
    }

    [Fact]
    public void Beta_MomentsAndSupportAreCorrect()
    {
        var beta = Distribution.Beta(2, 5);
        Assert.Equal(2.0 / 7.0, beta.Mean, 12);
        Assert.False(beta.Supports(0.0));
        Assert.False(beta.Supports(1.0));
        Assert.True(beta.Supports(0.5));
        Assert.Equal(double.NegativeInfinity, beta.LogDensity(1.5));
    }

    [Fact]
    public void Beta_DensityIntegratesToOne()
    {
        // Trapezoidal integration over the unit interval.
        var beta = Distribution.Beta(2, 3);
        var total = 0.0;
        const int steps = 20_000;
        for (var i = 1; i < steps; i++) total += beta.Density((double)i / steps) / steps;
        Assert.Equal(1.0, total, 3);
    }

    [Fact]
    public void Binomial_MassSumsToOneAndMatchesKnownValues()
    {
        var binomial = Distribution.Binomial(10, 0.3);
        var total = 0.0;
        for (var k = 0; k <= 10; k++) total += binomial.Density(k);
        Assert.Equal(1.0, total, 10);
        Assert.Equal(3.0, binomial.Mean, 12);
        Assert.Equal(Math.Pow(0.7, 10), binomial.Density(0), 12);
    }

    [Fact]
    public void Poisson_MassSumsToOneAndHasMeanEqualToVariance()
    {
        var poisson = Distribution.Poisson(4.0);
        var total = 0.0;
        for (var k = 0; k <= 60; k++) total += poisson.Density(k);
        Assert.Equal(1.0, total, 8);
        Assert.Equal(poisson.Mean, poisson.Variance, 12);
    }

    [Fact]
    public void Gamma_And_Exponential_Agree_ForShapeOne()
    {
        var gamma = Distribution.Gamma(1.0, 2.0);
        var exponential = Distribution.Exponential(2.0);
        foreach (var x in new[] { 0.1, 0.5, 1.0, 3.0 })
            Assert.Equal(exponential.LogDensity(x), gamma.LogDensity(x), 10);
    }

    [Fact]
    public void Gamma_CdfMatchesTheExponentialCase()
    {
        var gamma = Distribution.Gamma(1.0, 1.5);
        Assert.Equal(1 - Math.Exp(-1.5 * 2.0), gamma.Cdf(2.0), 9);
    }

    [Fact]
    public void HalfNormal_HasNoMassBelowZero()
    {
        var halfNormal = Distribution.HalfNormal(1.0);
        Assert.Equal(double.NegativeInfinity, halfNormal.LogDensity(-0.5));
        Assert.True(halfNormal.Density(0.5) > 0);
        Assert.Equal(Math.Sqrt(2 / Math.PI), halfNormal.Mean, 10);
    }

    [Fact]
    public void Categorical_NormalisesWeights()
    {
        var categorical = new Categorical([2.0, 6.0, 2.0]);
        Assert.Equal(0.6, categorical.Probabilities[1], 12);

        var draws = categorical.Sample(new GraviRandom(3), 20_000);
        var middle = draws.ToArray().Count(v => v == 1) / 20_000.0;
        Assert.Equal(0.6, middle, 1);
    }

    [Fact]
    public void LogLikelihood_SumsOverTheDataset()
    {
        var normal = Distribution.Normal(0, 1);
        double[] data = [-1.0, 0.0, 1.0];
        Assert.Equal(data.Sum(normal.LogDensity), normal.LogLikelihood(data), 12);
    }
}

public class BayesianModelTests
{
    [Fact]
    public void Model_ExposesItsStructure()
    {
        var model = new BayesianModel()
            .AddDistribution("theta", Distribution.Beta(1, 1))
            .AddObservation("data", DistributionSpec.Binomial(10, "theta"), 7.0);

        Assert.Equal(1, model.ParameterCount);
        Assert.Equal(["theta"], model.ParameterNames);
        Assert.Equal(1, model.ObservationCount);
    }

    [Fact]
    public void Model_RejectsADuplicateVariable()
    {
        var model = new BayesianModel().AddDistribution("theta", Distribution.Beta(1, 1));
        Assert.Throws<ArgumentException>(() => model.AddDistribution("theta", Distribution.Beta(2, 2)));
    }

    [Fact]
    public void Model_RejectsALikelihoodWithNoPrior()
    {
        var model = new BayesianModel();
        var ex = Assert.Throws<ArgumentException>(
            () => model.AddObservation("data", DistributionSpec.Binomial(10, "theta"), 7.0));
        Assert.Contains("AddDistribution", ex.Message);
    }

    [Fact]
    public void LogPosterior_CombinesPriorAndLikelihood()
    {
        var model = new BayesianModel()
            .AddDistribution("theta", Distribution.Beta(2, 2))
            .AddObservation("data", DistributionSpec.Binomial(10, "theta"), 7.0);

        var values = new Dictionary<string, double> { ["theta"] = 0.6 };
        var expected = Distribution.Beta(2, 2).LogDensity(0.6)
            + Distribution.Binomial(10, 0.6).LogDensity(7.0);

        Assert.Equal(expected, model.LogPosterior(values), 10);
    }

    [Fact]
    public void LogPosterior_IsNegativeInfinityOutsideTheSupport()
    {
        var model = new BayesianModel().AddDistribution("theta", Distribution.Beta(1, 1));
        Assert.Equal(double.NegativeInfinity,
            model.LogPosterior(new Dictionary<string, double> { ["theta"] = 1.5 }));
    }

    [Fact]
    public void PriorPredictive_GeneratesPlausibleData()
    {
        var model = new BayesianModel()
            .AddDistribution("theta", Distribution.Beta(1, 1))
            .AddObservation("data", DistributionSpec.Binomial(10, "theta"), 7.0);

        var predictive = model.PriorPredictive(2000, seed: 5)["data"];
        // With a uniform prior on theta the counts should be roughly uniform on 0..10, mean 5.
        Assert.Equal(5.0, Statistics.Mean(predictive), 0);
        Assert.True(Statistics.Min(predictive) >= 0);
        Assert.True(Statistics.Max(predictive) <= 10);
    }
}

public class InferenceTests
{
    /// <summary>
    /// The coin-toss model from the specification. Beta is conjugate to the binomial, so the exact
    /// posterior is available and the samplers can be checked against ground truth rather than
    /// against each other.
    /// </summary>
    private static (BayesianModel Model, Beta Exact) CoinModel(int successes = 7, int trials = 10)
    {
        var model = new BayesianModel()
            .AddDistribution("theta", Distribution.Beta(1, 1))
            .AddObservation("data", DistributionSpec.Binomial(trials, "theta"), successes);

        return (model, Distribution.Beta(1, 1).PosteriorAfter(successes, trials - successes));
    }

    [Fact]
    public void MetropolisHastings_RecoversTheConjugatePosterior()
    {
        var (model, exact) = CoinModel();
        var posterior = model.SampleMCMC(iterations: 20_000, chains: 4, seed: 7);

        // MCMC is stochastic, so it is checked against the exact posterior with an explicit
        // tolerance rather than by rounding to a fixed number of decimals.
        Assert.True(Math.Abs(posterior.Mean("theta") - exact.Mean) < 0.01,
            $"posterior mean {posterior.Mean("theta"):F4} vs exact {exact.Mean:F4}");
        Assert.True(Math.Abs(posterior.StandardDeviation("theta") - exact.StandardDeviation) < 0.01,
            $"posterior sd {posterior.StandardDeviation("theta"):F4} vs exact {exact.StandardDeviation:F4}");
    }

    [Fact]
    public void MetropolisHastings_ChainsConverge()
    {
        var (model, _) = CoinModel();
        var posterior = model.SampleMCMC(iterations: 20_000, chains: 4, seed: 11);

        Assert.True(posterior.RHat("theta") < 1.05, $"r_hat = {posterior.RHat("theta"):F4}");
        Assert.True(posterior.EffectiveSampleSize("theta") > 200);
        Assert.InRange(posterior.AcceptanceRate, 0.1, 0.7);
    }

    [Fact]
    public void MetropolisHastings_CredibleIntervalCoversTheTruth()
    {
        var (model, exact) = CoinModel();
        var posterior = model.SampleMCMC(iterations: 20_000, chains: 4, seed: 13);

        var (low, high) = posterior.CredibleInterval("theta", 0.95);
        Assert.True(low < exact.Mean && exact.Mean < high);
        Assert.True(high - low < 0.6);

        var (hdiLow, hdiHigh) = posterior.HighestDensityInterval("theta", 0.95);
        // The HDI is never wider than the equal-tailed interval at the same mass.
        Assert.True(hdiHigh - hdiLow <= high - low + 1e-9);
    }

    [Fact]
    public void Gibbs_AlsoRecoversTheConjugatePosterior()
    {
        var (model, exact) = CoinModel();
        var posterior = model.SampleGibbs(iterations: 20_000, chains: 4, seed: 17);

        Assert.True(Math.Abs(posterior.Mean("theta") - exact.Mean) < 0.01,
            $"posterior mean {posterior.Mean("theta"):F4} vs exact {exact.Mean:F4}");
        Assert.True(posterior.RHat("theta") < 1.05);
    }

    [Fact]
    public void Mcmc_RecoversTwoParametersOfANormalModel()
    {
        // Data generated with a known mean and scale; the sampler must find both.
        var rng = new GraviRandom(19);
        var data = Enumerable.Range(0, 200).Select(_ => rng.Normal(5.0, 2.0)).ToArray();

        var model = new BayesianModel()
            .AddDistribution("mu", Distribution.Normal(0, 10))
            .AddDistribution("sigma", Distribution.HalfNormal(5))
            .AddObservation("y", DistributionSpec.Normal("mu", "sigma"), data);

        var posterior = model.SampleMCMC(iterations: 20_000, chains: 4, seed: 19);

        // The posterior concentrates on the statistics of the sample that was actually drawn,
        // not on the generating parameters, so those are what it is compared against.
        var sample = NdArray.FromValues(data);
        Assert.True(Math.Abs(posterior.Mean("mu") - Statistics.Mean(sample)) < 0.1,
            $"mu {posterior.Mean("mu"):F3} vs sample mean {Statistics.Mean(sample):F3}");
        Assert.True(Math.Abs(posterior.Mean("sigma") - Statistics.Std(sample)) < 0.15,
            $"sigma {posterior.Mean("sigma"):F3} vs sample sd {Statistics.Std(sample):F3}");

        Assert.True(posterior.RHat("mu") < 1.1);
        Assert.True(posterior.RHat("sigma") < 1.1);
    }

    [Fact]
    public void Mcmc_RecoversAPoissonRate()
    {
        var rng = new GraviRandom(23);
        var data = Enumerable.Range(0, 300).Select(_ => (double)rng.Poisson(4.5)).ToArray();

        var model = new BayesianModel()
            .AddDistribution("lambda", Distribution.Gamma(1, 0.1))
            .AddObservation("counts", DistributionSpec.Poisson("lambda"), data);

        var posterior = model.SampleMCMC(iterations: 15_000, chains: 4, seed: 23);

        // Gamma is conjugate to the Poisson likelihood, so the exact posterior mean is
        // (alpha + sum y) / (rate + n) and the sampler can be checked against it directly.
        var exactMean = (1.0 + data.Sum()) / (0.1 + data.Length);
        Assert.True(Math.Abs(posterior.Mean("lambda") - exactMean) < 0.05,
            $"lambda {posterior.Mean("lambda"):F4} vs exact {exactMean:F4}");
    }

    [Fact]
    public void Trace_ExposesChainsSeparately()
    {
        var (model, _) = CoinModel();
        var posterior = model.SampleMCMC(iterations: 4000, chains: 3, warmup: 2000, seed: 29);

        Assert.Equal(3, posterior.ChainCount);
        Assert.Equal(2000, posterior.DrawsPerChain);
        Assert.Equal(6000, posterior.TotalDraws);
        Assert.Equal(2000, posterior.Chain("theta", 0).Size);
    }

    [Fact]
    public void Thinning_ReducesTheRetainedDraws()
    {
        var (model, _) = CoinModel();
        var posterior = model.SampleMCMC(iterations: 4000, chains: 2, warmup: 2000, thin: 4, seed: 31);
        Assert.Equal(500, posterior.DrawsPerChain);
    }

    [Fact]
    public void Summary_ListsEveryParameter()
    {
        var (model, _) = CoinModel();
        var summary = model.SampleMCMC(iterations: 4000, chains: 2, seed: 33).Summary();

        Assert.Contains("theta", summary);
        Assert.Contains("r_hat", summary);
        Assert.Contains("ess", summary);
    }

    [Fact]
    public void PosteriorPredictive_ReproducesTheObservedScale()
    {
        var (model, _) = CoinModel();
        var posterior = model.SampleMCMC(iterations: 10_000, chains: 2, seed: 37);

        var replicated = posterior.PosteriorPredictive(
            (values, rng) => rng.Binomial(10, values["theta"]), draws: 4000, seed: 37);

        // Observed 7 of 10, so replicated counts should centre near 7.
        Assert.Equal(7.0, Statistics.Mean(replicated), 0);
    }

    [Fact]
    public void VariationalInference_FindsTheSamePosteriorMean()
    {
        var (model, exact) = CoinModel();
        var fit = model.FitVariational(iterations: 1500, learningRate: 0.05, seed: 41);

        // The fit happens on a logit scale, so the reported mean must land back inside (0, 1)
        // and near the exact posterior mean.
        Assert.InRange(fit.Means["theta"], 0.0, 1.0);
        Assert.True(Math.Abs(fit.Means["theta"] - exact.Mean) < 0.08,
            $"variational mean {fit.Means["theta"]:F4} vs exact {exact.Mean:F4}");
        Assert.True(fit.StandardDeviations["theta"] > 0);
        Assert.Contains("theta", fit.Summary());
    }

    [Fact]
    public void Inference_FailsClearlyWhenThePriorExcludesTheData()
    {
        // A Beta prior cannot produce a probability above 1, so a 20-of-10 observation is impossible.
        var model = new BayesianModel()
            .AddDistribution("theta", Distribution.Beta(1, 1))
            .AddObservation("data", DistributionSpec.Binomial(10, "theta"), 20.0);

        var ex = Assert.Throws<InvalidOperationException>(() => model.SampleMCMC(iterations: 100, chains: 1));
        Assert.Contains("finite posterior", ex.Message);
    }

    [Fact]
    public void Mcmc_RejectsAModelWithNoParameters()
    {
        Assert.Throws<ArgumentException>(() => new BayesianModel().SampleMCMC(iterations: 10));
    }
}

public class BayesianNetworkTests
{
    /// <summary>The classic sprinkler network: rain and sprinkler both make the grass wet.</summary>
    private static BayesianNetwork Sprinkler() => new BayesianNetwork()
        .AddVariable("rain", 0.8, 0.2)
        .AddVariable("sprinkler", 2, ["rain"], [[0.6, 0.4], [0.99, 0.01]])
        .AddVariable("wet", 2, ["rain", "sprinkler"],
        [
            [1.0, 0.0],     // no rain, no sprinkler
            [0.1, 0.9],     // no rain, sprinkler
            [0.2, 0.8],     // rain, no sprinkler
            [0.01, 0.99],   // rain and sprinkler
        ]);

    [Fact]
    public void Network_ReportsItsStructure()
    {
        var network = Sprinkler();
        Assert.Equal(["rain", "sprinkler", "wet"], network.Variables);
        Assert.Equal(2, network.StateCount("wet"));
    }

    [Fact]
    public void Network_RejectsARowThatDoesNotSumToOne()
    {
        var network = new BayesianNetwork().AddVariable("a", 0.5, 0.5);
        Assert.Throws<ArgumentException>(
            () => network.AddVariable("b", 2, ["a"], [[0.5, 0.7], [0.5, 0.5]]));
    }

    [Fact]
    public void Network_RejectsAWrongSizedTable()
    {
        var network = new BayesianNetwork().AddVariable("a", 0.5, 0.5);
        Assert.Throws<ArgumentException>(() => network.AddVariable("b", 2, ["a"], [[0.5, 0.5]]));
    }

    [Fact]
    public void MarginalOfARootMatchesItsPrior()
    {
        var marginal = Sprinkler().Infer("rain");
        Assert.Equal(0.8, marginal[0], 10);
        Assert.Equal(0.2, marginal[1], 10);
    }

    [Fact]
    public void Marginal_MatchesAHandComputedEnumeration()
    {
        // P(wet) = sum over rain, sprinkler of P(r) P(s|r) P(wet|r,s)
        var expected =
            0.8 * 0.6 * 0.0 + 0.8 * 0.4 * 0.9 +
            0.2 * 0.99 * 0.8 + 0.2 * 0.01 * 0.99;

        Assert.Equal(expected, Sprinkler().Infer("wet")[1], 10);
    }

    [Fact]
    public void Evidence_ChangesThePosteriorInTheExpectedDirection()
    {
        var network = Sprinkler();
        var prior = network.Infer("rain")[1];
        var posterior = network.Infer("rain", new Dictionary<string, int> { ["wet"] = 1 })[1];

        // Wet grass is evidence for rain.
        Assert.True(posterior > prior, $"{posterior:F4} should exceed {prior:F4}");
    }

    [Fact]
    public void ExplainingAway_ReducesTheProbabilityOfTheOtherCause()
    {
        var network = Sprinkler();
        var wetOnly = network.Infer("rain", new Dictionary<string, int> { ["wet"] = 1 })[1];
        var wetAndSprinkler = network.Infer("rain",
            new Dictionary<string, int> { ["wet"] = 1, ["sprinkler"] = 1 })[1];

        // Once the sprinkler explains the wet grass, rain becomes less necessary.
        Assert.True(wetAndSprinkler < wetOnly);
    }

    [Fact]
    public void Sampling_ApproximatesTheExactMarginal()
    {
        var network = Sprinkler();
        var rng = new GraviRandom(43);
        var wet = 0;
        const int draws = 40_000;
        for (var i = 0; i < draws; i++) if (network.Sample(rng)["wet"] == 1) wet++;

        Assert.Equal(network.Infer("wet")[1], (double)wet / draws, 2);
    }

    [Fact]
    public void ImpossibleEvidence_IsReportedClearly()
    {
        var network = new BayesianNetwork()
            .AddVariable("a", 1.0, 0.0)
            .AddVariable("b", 2, ["a"], [[1.0, 0.0], [0.0, 1.0]]);

        Assert.Throws<InvalidOperationException>(
            () => network.Infer("a", new Dictionary<string, int> { ["b"] = 1 }));
    }
}

public class HiddenMarkovModelTests
{
    /// <summary>A two-state weather model emitting three activity symbols.</summary>
    private static HiddenMarkovModel Weather() => new(
        [0.6, 0.4],
        new[,] { { 0.7, 0.3 }, { 0.4, 0.6 } },
        new[,] { { 0.1, 0.4, 0.5 }, { 0.6, 0.3, 0.1 } });

    [Fact]
    public void LogLikelihood_MatchesADirectEnumeration()
    {
        var model = Weather();
        int[] observations = [0, 1, 2];

        // Enumerate every hidden path and sum the joint probabilities.
        var total = 0.0;
        for (var a = 0; a < 2; a++)
            for (var b = 0; b < 2; b++)
                for (var c = 0; c < 2; c++)
                    total += model.InitialProbabilities[a] * model.Emissions[a, observations[0]]
                        * model.Transitions[a, b] * model.Emissions[b, observations[1]]
                        * model.Transitions[b, c] * model.Emissions[c, observations[2]];

        Assert.Equal(Math.Log(total), model.LogLikelihood(observations), 10);
    }

    [Fact]
    public void Viterbi_ReturnsAPathOfTheRightLengthAndValidStates()
    {
        var path = Weather().Viterbi([0, 1, 2, 2, 1, 0]);
        Assert.Equal(6, path.Length);
        Assert.All(path, s => Assert.InRange(s, 0, 1));
    }

    [Fact]
    public void Viterbi_FollowsAnUnambiguousEmission()
    {
        // State 0 emits symbol 0 with certainty, state 1 emits symbol 1 with certainty.
        var model = new HiddenMarkovModel(
            [0.5, 0.5],
            new[,] { { 0.5, 0.5 }, { 0.5, 0.5 } },
            new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } });

        Assert.Equal([0, 1, 0, 1], model.Viterbi([0, 1, 0, 1]));
    }

    [Fact]
    public void StatePosteriors_SumToOneAtEachStep()
    {
        var gamma = Weather().StatePosteriors([0, 1, 2, 0]);
        for (var t = 0; t < 4; t++)
        {
            var total = gamma[t, 0] + gamma[t, 1];
            Assert.Equal(1.0, total, 9);
        }
    }

    [Fact]
    public void LongSequences_DoNotUnderflow()
    {
        var rng = new GraviRandom(47);
        var (_, observations) = Weather().Generate(2000, rng);

        var likelihood = Weather().LogLikelihood(observations);
        Assert.True(double.IsFinite(likelihood));
        Assert.True(likelihood < 0);
    }

    [Fact]
    public void BaumWelch_IncreasesTheLikelihood()
    {
        var truth = Weather();
        var rng = new GraviRandom(53);
        var sequences = Enumerable.Range(0, 30)
            .Select(_ => (IReadOnlyList<int>)truth.Generate(60, rng).Observations)
            .ToList();

        var model = HiddenMarkovModel.Random(2, 3, seed: 53);
        var before = sequences.Sum(model.LogLikelihood);
        model.Fit(sequences, iterations: 40);
        var after = sequences.Sum(model.LogLikelihood);

        Assert.True(after > before, $"{before:F2} -> {after:F2}");
    }

    [Fact]
    public void BaumWelch_ApproachesTheGeneratingModelsLikelihood()
    {
        var truth = Weather();
        var rng = new GraviRandom(59);
        var sequences = Enumerable.Range(0, 40)
            .Select(_ => (IReadOnlyList<int>)truth.Generate(80, rng).Observations)
            .ToList();

        var model = HiddenMarkovModel.Random(2, 3, seed: 59);
        model.Fit(sequences, iterations: 80);

        var fitted = sequences.Sum(model.LogLikelihood);
        var reference = sequences.Sum(truth.LogLikelihood);

        // EM finds a local optimum, so it should land close to - and may even exceed - the truth.
        Assert.True(fitted > reference * 1.05, $"fitted {fitted:F1} vs truth {reference:F1}");
    }

    [Fact]
    public void Rows_StayNormalisedAfterFitting()
    {
        var rng = new GraviRandom(61);
        var sequences = Enumerable.Range(0, 10)
            .Select(_ => (IReadOnlyList<int>)Weather().Generate(40, rng).Observations)
            .ToList();

        var model = HiddenMarkovModel.Random(2, 3, seed: 61);
        model.Fit(sequences, iterations: 20);

        Assert.Equal(1.0, model.InitialProbabilities.Sum(), 8);
        for (var i = 0; i < 2; i++)
        {
            var transitionRow = 0.0;
            var emissionRow = 0.0;
            for (var j = 0; j < 2; j++) transitionRow += model.Transitions[i, j];
            for (var k = 0; k < 3; k++) emissionRow += model.Emissions[i, k];
            Assert.Equal(1.0, transitionRow, 8);
            Assert.Equal(1.0, emissionRow, 8);
        }
    }
}

public class BayesianRegressionTests
{
    [Fact]
    public void Regression_RecoversKnownCoefficients()
    {
        var rng = new GraviRandom(67);
        var x = rng.StandardNormal(200, 2);
        var y = NdArray.Zeros(200);
        for (var i = 0; i < 200; i++) y.SetAt(i, 1.5 + 2.0 * x[i, 0] - 3.0 * x[i, 1] + rng.Normal(0, 0.2));

        var model = new BayesianLinearRegression(priorPrecision: 1e-3, noisePrecision: 25.0).Fit(x, y);

        Assert.Equal(1.5, model.CoefficientMeans.At(0), 1);
        Assert.Equal(2.0, model.CoefficientMeans.At(1), 1);
        Assert.Equal(-3.0, model.CoefficientMeans.At(2), 1);
    }

    [Fact]
    public void PredictionIntervals_CoverMostObservations()
    {
        var rng = new GraviRandom(71);
        var x = rng.StandardNormal(300, 1);
        var y = NdArray.Zeros(300);
        for (var i = 0; i < 300; i++) y.SetAt(i, 2.0 * x[i, 0] + rng.Normal(0, 0.5));

        var model = new BayesianLinearRegression(noisePrecision: 4.0).Fit(x, y);
        var (lower, upper) = model.PredictInterval(x, 0.95);

        var covered = 0;
        for (var i = 0; i < 300; i++) if (y.At(i) >= lower.At(i) && y.At(i) <= upper.At(i)) covered++;
        Assert.True(covered / 300.0 > 0.9, $"coverage = {covered / 300.0:P1}");
    }

    [Fact]
    public void UncertaintyIsLargerWhereThereIsLessData()
    {
        var rng = new GraviRandom(73);
        var x = rng.Normal(0.0, 1.0, 200, 1);
        var y = NdArray.Zeros(200);
        for (var i = 0; i < 200; i++) y.SetAt(i, 3.0 * x[i, 0] + rng.Normal(0, 0.3));

        var model = new BayesianLinearRegression(noisePrecision: 10.0).Fit(x, y);
        var query = NdArray.FromArray(new double[,] { { 0.0 }, { 8.0 } });
        var (_, deviation) = model.PredictWithUncertainty(query);

        // Far outside the observed range the model should be markedly less certain.
        Assert.True(deviation.At(1) > deviation.At(0));
    }

    [Fact]
    public void SampledCoefficients_CentreOnThePosteriorMean()
    {
        var rng = new GraviRandom(79);
        var x = rng.StandardNormal(150, 1);
        var y = NdArray.Zeros(150);
        for (var i = 0; i < 150; i++) y.SetAt(i, 1.0 + 4.0 * x[i, 0] + rng.Normal(0, 0.4));

        var model = new BayesianLinearRegression(noisePrecision: 6.0).Fit(x, y);
        var draws = model.SampleCoefficients(4000, seed: 79);

        Assert.Equal(model.CoefficientMeans.At(1), Statistics.Mean(draws.Column(1).Copy()), 1);
    }

    [Fact]
    public void Regression_MustBeFittedFirst()
    {
        Assert.Throws<InvalidOperationException>(() => new BayesianLinearRegression().Predict(NdArray.Ones(2, 1)));
    }
}
