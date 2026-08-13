using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;
using Gravicode.Science.GraviProb;
using Xunit;

namespace GraviProb.Tests;

/// <summary>
/// Tests for the differentiable log posterior and the samplers built on it.
/// </summary>
/// <remarks>
/// The gradients are checked against central finite differences, which share no code with the
/// tape. The samplers are checked against the exact conjugate posterior, exactly as the
/// Metropolis-Hastings tests are — a sampler that agrees with another sampler has proved nothing.
/// </remarks>
public class GradientInferenceTests
{
    /// <summary>The coin-toss model, whose Beta-binomial conjugacy gives an exact posterior.</summary>
    private static (BayesianModel Model, Beta Exact) CoinModel(int successes = 7, int trials = 10)
    {
        var model = new BayesianModel()
            .AddDistribution("theta", Distribution.Beta(1, 1))
            .AddObservation("data", DistributionSpec.Binomial(trials, "theta"), successes);

        return (model, Distribution.Beta(1, 1).PosteriorAfter(successes, trials - successes));
    }

    // ---------------------------------------------------------------- densities and gradients

    [Theory]
    [InlineData(0.3)]
    [InlineData(1.7)]
    [InlineData(-2.4)]
    public void TensorLogDensity_AgreesWithTheScalarOne(double x)
    {
        // The two forms are written separately, so this checks the tape version transcribes the
        // same formula rather than a plausible-looking neighbour of it.
        Distribution[] distributions =
        [
            Distribution.Normal(0.5, 2.0),
            Distribution.StudentT(4, 0.0, 1.0),
        ];

        foreach (var distribution in distributions)
        {
            var expected = distribution.LogDensity(x);
            var actual = distribution.LogDensity(Tensor.Constant(x)).Item;
            Assert.True(Math.Abs(expected - actual) < 1e-10,
                $"{distribution.Name} at {x}: scalar {expected:G12} vs tensor {actual:G12}");
        }
    }

    [Theory]
    [InlineData(0.4)]
    [InlineData(2.5)]
    public void TensorLogDensity_AgreesWithTheScalarOneOnPositiveSupport(double x)
    {
        Distribution[] distributions =
        [
            Distribution.Gamma(2.0, 1.5),
            Distribution.Exponential(0.8),
            Distribution.LogNormal(0.0, 1.0),
            Distribution.HalfNormal(1.5),
        ];

        foreach (var distribution in distributions)
        {
            var expected = distribution.LogDensity(x);
            var actual = distribution.LogDensity(Tensor.Constant(x)).Item;
            Assert.True(Math.Abs(expected - actual) < 1e-10,
                $"{distribution.Name} at {x}: scalar {expected:G12} vs tensor {actual:G12}");
        }
    }

    [Fact]
    public void BetaLogDensity_AgreesWithTheScalarOne()
    {
        var beta = Distribution.Beta(2, 5);
        foreach (var x in new[] { 0.1, 0.35, 0.9 })
            Assert.Equal(beta.LogDensity(x), beta.LogDensity(Tensor.Constant(x)).Item, 10);
    }

    [Theory]
    [InlineData(0.25)]
    [InlineData(0.5)]
    [InlineData(0.8)]
    public void PosteriorGradient_MatchesFiniteDifferences(double theta)
    {
        var (model, _) = CoinModel();
        var (_, gradient) = model.LogPosteriorGradient(Point(theta));

        const double step = 1e-6;
        var numeric = (model.LogPosterior(Point(theta + step)) - model.LogPosterior(Point(theta - step)))
                      / (2 * step);

        Assert.True(Math.Abs(gradient[0] - numeric) < 1e-4,
            $"autodiff {gradient[0]:G8} vs finite difference {numeric:G8}");

        static Dictionary<string, double> Point(double value) => new(StringComparer.Ordinal) { ["theta"] = value };
    }

    [Fact]
    public void PosteriorGradient_IsZeroAtTheKnownMode()
    {
        // Beta(1,1) prior with 7 successes in 10 gives a Beta(8,4) posterior, whose mode is
        // (8-1)/(8+4-2) = 0.7. The gradient of the log posterior vanishes exactly there.
        var (model, _) = CoinModel();
        var (_, gradient) = model.LogPosteriorGradient(
            new Dictionary<string, double>(StringComparer.Ordinal) { ["theta"] = 0.7 });

        Assert.True(Math.Abs(gradient[0]) < 1e-8, $"gradient at the mode was {gradient[0]:G8}");
    }

    [Fact]
    public void PosteriorGradient_ReportsMultipleParametersInDeclarationOrder()
    {
        var rng = new GraviRandom(5);
        var data = Distribution.Normal(3.0, 2.0).Sample(rng, 40).ToArray();

        var model = new BayesianModel()
            .AddDistribution("mu", Distribution.Normal(0, 10))
            .AddDistribution("sigma", Distribution.HalfNormal(5))
            .AddObservation("y", DistributionSpec.Normal("mu", "sigma"), data);

        var at = new Dictionary<string, double>(StringComparer.Ordinal) { ["mu"] = 2.5, ["sigma"] = 1.8 };
        var (_, gradient) = model.LogPosteriorGradient(at);

        Assert.Equal(["mu", "sigma"], model.ParameterNames);
        Assert.Equal(2, gradient.Length);

        const double step = 1e-6;
        for (var i = 0; i < 2; i++)
        {
            var name = model.ParameterNames[i];
            var forward = new Dictionary<string, double>(at, StringComparer.Ordinal);
            var backward = new Dictionary<string, double>(at, StringComparer.Ordinal);
            forward[name] += step;
            backward[name] -= step;

            var numeric = (model.LogPosterior(forward) - model.LogPosterior(backward)) / (2 * step);
            Assert.True(Math.Abs(gradient[i] - numeric) < 1e-3,
                $"d/d{name}: autodiff {gradient[i]:G8} vs finite difference {numeric:G8}");
        }
    }

    [Fact]
    public void OutsideTheSupportTheDensityIsRejectedRatherThanReturningNaN()
    {
        var (model, _) = CoinModel();
        var (density, gradient) = model.LogPosteriorGradient(
            new Dictionary<string, double>(StringComparer.Ordinal) { ["theta"] = 1.5 });

        Assert.True(double.IsNegativeInfinity(density));
        Assert.False(double.IsNaN(gradient[0]));
    }

    // ---------------------------------------------------------------- the samplers

    [Fact]
    public void Hmc_RecoversTheConjugatePosterior()
    {
        var (model, exact) = CoinModel();
        var posterior = model.SampleHMC(iterations: 2000, chains: 4, seed: 7);

        Assert.True(Math.Abs(posterior.Mean("theta") - exact.Mean) < 0.01,
            $"posterior mean {posterior.Mean("theta"):F4} vs exact {exact.Mean:F4}");
        Assert.True(Math.Abs(posterior.StandardDeviation("theta") - exact.StandardDeviation) < 0.01,
            $"posterior sd {posterior.StandardDeviation("theta"):F4} vs exact {exact.StandardDeviation:F4}");
    }

    [Fact]
    public void Nuts_RecoversTheConjugatePosterior()
    {
        var (model, exact) = CoinModel();
        var posterior = model.SampleNUTS(iterations: 2000, chains: 4, seed: 7);

        Assert.True(Math.Abs(posterior.Mean("theta") - exact.Mean) < 0.01,
            $"posterior mean {posterior.Mean("theta"):F4} vs exact {exact.Mean:F4}");
        Assert.True(Math.Abs(posterior.StandardDeviation("theta") - exact.StandardDeviation) < 0.01,
            $"posterior sd {posterior.StandardDeviation("theta"):F4} vs exact {exact.StandardDeviation:F4}");
    }

    [Fact]
    public void Nuts_ChainsConvergeAndMixWell()
    {
        var (model, _) = CoinModel();
        var posterior = model.SampleNUTS(iterations: 2000, chains: 4, seed: 11);

        Assert.True(posterior.RHat("theta") < 1.05, $"r_hat = {posterior.RHat("theta"):F4}");

        // The point of a gradient sampler is that draws are far less correlated. Random-walk
        // Metropolis needs 20,000 iterations to clear 200 effective samples on this model;
        // NUTS should clear it from a tenth as many.
        Assert.True(posterior.EffectiveSampleSize("theta") > 400,
            $"effective sample size {posterior.EffectiveSampleSize("theta"):F0}");
    }

    [Fact]
    public void Nuts_IsMoreEfficientPerIterationThanRandomWalk()
    {
        var (model, _) = CoinModel();

        var nuts = model.SampleNUTS(iterations: 2000, chains: 4, seed: 3);
        var randomWalk = model.SampleMCMC(iterations: 2000, chains: 4, seed: 3);

        var nutsEfficiency = nuts.EffectiveSampleSize("theta") / nuts.TotalDraws;
        var walkEfficiency = randomWalk.EffectiveSampleSize("theta") / randomWalk.TotalDraws;

        Assert.True(nutsEfficiency > walkEfficiency,
            $"NUTS {nutsEfficiency:P1} of draws effective vs random walk {walkEfficiency:P1}");
    }

    [Fact]
    public void Nuts_FitsATwoParameterNormalModel()
    {
        // A correlated two-parameter posterior, where a random walk starts to struggle.
        var rng = new GraviRandom(19);
        var data = Distribution.Normal(5.0, 2.0).Sample(rng, 200).ToArray();

        var model = new BayesianModel()
            .AddDistribution("mu", Distribution.Normal(0, 10))
            .AddDistribution("sigma", Distribution.HalfNormal(5))
            .AddObservation("y", DistributionSpec.Normal("mu", "sigma"), data);

        var posterior = model.SampleNUTS(iterations: 1500, chains: 4, seed: 23);

        // With 200 observations the posterior concentrates near the sample moments.
        var sampleMean = Statistics.Mean(NdArray.FromValues(data));
        var sampleSd = Statistics.Std(NdArray.FromValues(data));

        Assert.True(Math.Abs(posterior.Mean("mu") - sampleMean) < 0.1,
            $"mu {posterior.Mean("mu"):F3} vs sample mean {sampleMean:F3}");
        Assert.True(Math.Abs(posterior.Mean("sigma") - sampleSd) < 0.2,
            $"sigma {posterior.Mean("sigma"):F3} vs sample sd {sampleSd:F3}");
        Assert.True(posterior.RHat("mu") < 1.05);
        Assert.True(posterior.RHat("sigma") < 1.05);
    }

    [Fact]
    public void SamplersStayInsideTheSupport()
    {
        // theta lives on (0,1). The unconstrained transform is what keeps every draw there;
        // without it a Hamiltonian trajectory would routinely step outside.
        var (model, _) = CoinModel();
        var posterior = model.SampleNUTS(iterations: 800, chains: 2, seed: 31);

        foreach (var draw in posterior["theta"].ToArray())
            Assert.InRange(draw, 0.0, 1.0);
    }

    [Fact]
    public void AModelWithoutGradientsSaysSoRatherThanFailingLater()
    {
        var model = new BayesianModel()
            .AddDistribution("p", Distribution.Beta(1, 1))
            .AddObservation("data",
                DistributionSpec.From(values => new Binomial(10, values["p"]), "p"), 7.0);

        Assert.False(model.IsDifferentiable);
        Assert.Throws<NotSupportedException>(() => model.SampleNUTS(iterations: 10, chains: 1));
        Assert.Throws<NotSupportedException>(() => model.SampleHMC(iterations: 10, chains: 1));

        // The gradient-free samplers still work on it.
        Assert.True(model.SampleMCMC(iterations: 500, chains: 1).TotalDraws > 0);
    }
}
