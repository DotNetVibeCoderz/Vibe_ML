using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Xunit;

namespace Gravicode.Science.Tests.GraviProb;

/// <summary>
/// Tests for the Kalman filter and smoother.
/// </summary>
/// <remarks>
/// A Kalman filter that is subtly wrong still produces a smooth, plausible-looking series — which
/// is why the pins here are the properties that distinguish the exact recursion from a merely
/// reasonable one: it beats the observations it was given, the smoother beats the filter, the
/// covariance stays positive definite over a long run, and the scalar case matches the closed-form
/// steady-state gain computed independently.
/// </remarks>
public class StateSpaceTests
{
    /// <summary>A noisy walk around a constant level.</summary>
    private static (NdArray Truth, NdArray Observations) NoisyLevel(
        double level = 5.0, double noise = 1.0, int steps = 200, int seed = 3)
    {
        var rng = new GraviRandom(seed);
        var truth = NdArray.Zeros(steps, 1);
        var observed = NdArray.Zeros(steps, 1);

        for (var t = 0; t < steps; t++)
        {
            truth[t, 0] = level;
            observed[t, 0] = level + rng.Normal() * noise;
        }

        return (truth, observed);
    }

    private static double RootMeanSquare(IReadOnlyList<double> estimates, NdArray truth)
    {
        var total = 0.0;
        for (var t = 0; t < estimates.Count; t++)
        {
            var error = estimates[t] - truth[t, 0];
            total += error * error;
        }
        return Math.Sqrt(total / estimates.Count);
    }

    [Fact]
    public void TheFilterIsMoreAccurateThanTheObservationsItWasGiven()
    {
        // The whole point. If filtering does not beat the raw measurements, it is doing nothing.
        var (truth, observed) = NoisyLevel();
        var filter = KalmanFilter.LocalLevel(processVariance: 1e-4, observationVariance: 1.0);

        var result = filter.Filter(observed);
        var estimates = result.Filtered.Select(s => s.Mean.At(0)).ToArray();

        var rawError = 0.0;
        for (var t = 0; t < observed.Shape[0]; t++)
        {
            var error = observed[t, 0] - truth[t, 0];
            rawError += error * error;
        }
        rawError = Math.Sqrt(rawError / observed.Shape[0]);

        var filteredError = RootMeanSquare(estimates, truth);

        Assert.True(filteredError < rawError / 2,
            $"filtering gave {filteredError:F4} against the raw observations' {rawError:F4}");
    }

    [Fact]
    public void TheSmootherIsAtLeastAsAccurateAsTheFilter()
    {
        // Smoothing uses the whole series, so it cannot do worse. It is most visibly better at the
        // start, where the filter had almost nothing to go on.
        var (truth, observed) = NoisyLevel();
        var filter = KalmanFilter.LocalLevel(1e-4, 1.0);

        var filtered = filter.Filter(observed).Filtered.Select(s => s.Mean.At(0)).ToArray();
        var smoothed = filter.Smooth(observed).Select(s => s.Mean.At(0)).ToArray();

        Assert.True(RootMeanSquare(smoothed, truth) <= RootMeanSquare(filtered, truth),
            "smoothing should never be worse than filtering");

        // And the early steps are where the gap is.
        Assert.True(Math.Abs(smoothed[0] - truth[0, 0]) < Math.Abs(filtered[0] - truth[0, 0]));
    }

    [Fact]
    public void SmoothingNeverIncreasesUncertainty()
    {
        var (_, observed) = NoisyLevel(steps: 50);
        var filter = KalmanFilter.LocalLevel(0.01, 1.0);

        var filtered = filter.Filter(observed).Filtered;
        var smoothed = filter.Smooth(observed);

        for (var t = 0; t < filtered.Count; t++)
            Assert.True(smoothed[t].Covariance[0, 0] <= filtered[t].Covariance[0, 0] + 1e-12,
                $"at step {t} smoothing raised the variance from " +
                $"{filtered[t].Covariance[0, 0]:E4} to {smoothed[t].Covariance[0, 0]:E4}");
    }

    [Fact]
    public void TheScalarSteadyStateGainMatchesTheClosedForm()
    {
        // For a local level model the filter converges to a fixed gain, and that gain solves a
        // scalar Riccati equation with a closed-form root. Computing it independently here is the
        // strongest available check that the recursion is the exact one.
        const double q = 0.01;
        const double r = 1.0;

        var filter = KalmanFilter.LocalLevel(q, r);
        var (_, observed) = NoisyLevel(steps: 500, noise: 1.0, seed: 11);

        var covariance = filter.Filter(observed).Filtered[^1].Covariance[0, 0];

        // Steady state: P = (P + Q)R / (P + Q + R). Solving gives P = (−Q + sqrt(Q² + 4QR)) / 2.
        var expected = (-q + Math.Sqrt(q * q + 4 * q * r)) / 2;

        Assert.True(Math.Abs(covariance - expected) < 1e-6,
            $"the filter settled at {covariance:F8} against the closed-form {expected:F8}");
    }

    [Fact]
    public void TheCovarianceStaysSymmetricAndPositiveOverALongRun()
    {
        // What the Joseph form buys. The short update can drift into an asymmetric or negative
        // covariance over hundreds of steps, and the filter then diverges with no warning.
        var (_, observed) = NoisyLevel(steps: 1000, seed: 17);
        var filter = KalmanFilter.LocalLinearTrend(1e-5, 1e-7, 1.0);

        foreach (var state in filter.Filter(observed).Filtered)
        {
            Assert.Equal(state.Covariance[0, 1], state.Covariance[1, 0], 12);

            for (var i = 0; i < 2; i++)
                Assert.True(state.Covariance[i, i] > 0,
                    $"variance {i} went to {state.Covariance[i, i]:E4}");

            // Positive definite, not merely positive on the diagonal.
            var determinant = state.Covariance[0, 0] * state.Covariance[1, 1]
                              - state.Covariance[0, 1] * state.Covariance[1, 0];
            Assert.True(determinant > 0, $"the covariance determinant went to {determinant:E4}");
        }
    }

    [Fact]
    public void ALinearTrendIsTrackedAndExtrapolated()
    {
        // The slope is never observed — it is inferred entirely from how the level moves, which is
        // what makes it a genuine latent variable.
        var rng = new GraviRandom(23);
        const double slope = 0.5;

        var observed = NdArray.Zeros(100, 1);
        for (var t = 0; t < 100; t++) observed[t, 0] = 10 + slope * t + rng.Normal() * 0.5;

        var filter = KalmanFilter.LocalLinearTrend(1e-4, 1e-6, 0.25);
        var filtered = filter.Filter(observed).Filtered;

        Assert.True(Math.Abs(filtered[^1].Mean.At(1) - slope) < 0.05,
            $"the recovered slope was {filtered[^1].Mean.At(1):F4} against {slope}");

        // And the forecast continues the trend rather than flattening out.
        var forecast = filter.Forecast(observed, horizon: 10);
        var predicted = filter.Observe(forecast);

        Assert.True(Math.Abs(predicted[9, 0] - (10 + slope * 109)) < 2.0,
            $"the ten-step forecast was {predicted[9, 0]:F3} against the truth's {10 + slope * 109:F3}");
    }

    [Fact]
    public void ForecastUncertaintyGrowsWithTheHorizon()
    {
        // No observations arrive, so only the predict step runs and Q accumulates. A forecast whose
        // uncertainty does not grow is not a forecast.
        var (_, observed) = NoisyLevel(steps: 100);
        var filter = KalmanFilter.LocalLevel(0.01, 1.0);

        var forecast = filter.Forecast(observed, horizon: 20);

        for (var h = 1; h < 20; h++)
            Assert.True(forecast[h].Covariance[0, 0] > forecast[h - 1].Covariance[0, 0],
                $"uncertainty did not grow between horizons {h - 1} and {h}");
    }

    [Fact]
    public void TheNoiseRatioDecidesHowHardTheFilterSmooths()
    {
        // Only the ratio matters, which is why a filter can be tuned with one number. A large Q
        // relative to R means "the state moves faster than the sensor lies", so track closely.
        var (_, observed) = NoisyLevel(steps: 100, seed: 29);

        var trusting = KalmanFilter.LocalLevel(1.0, 0.01).Filter(observed).Filtered;
        var sceptical = KalmanFilter.LocalLevel(1e-6, 1.0).Filter(observed).Filtered;

        double Roughness(IReadOnlyList<StateEstimate> states)
        {
            var total = 0.0;
            for (var t = 1; t < states.Count; t++)
                total += Math.Abs(states[t].Mean.At(0) - states[t - 1].Mean.At(0));
            return total;
        }

        Assert.True(Roughness(trusting) > 10 * Roughness(sceptical),
            "a high process-noise filter should chase the measurements far more closely");
    }

    [Fact]
    public void TheFilterRecoversStatesItCouldNotSee()
    {
        // Simulate from the model, then check the filter against a truth it was never shown.
        var filter = KalmanFilter.LocalLevel(0.05, 1.0);
        var (states, observations) = filter.Simulate(300, new GraviRandom(31));

        var filtered = filter.Filter(observations).Filtered.Select(s => s.Mean.At(0)).ToArray();

        var filteredError = RootMeanSquare(filtered, states);

        var rawError = 0.0;
        for (var t = 0; t < 300; t++)
        {
            var error = observations[t, 0] - states[t, 0];
            rawError += error * error;
        }
        rawError = Math.Sqrt(rawError / 300);

        Assert.True(filteredError < rawError,
            $"the filter's error {filteredError:F4} was not below the observations' {rawError:F4}");
    }

    [Fact]
    public void TheReportedUncertaintyIsHonest()
    {
        // A filter that is confidently wrong is worse than one that is uncertain. Standardised
        // errors should have roughly unit variance if the covariance means what it says.
        var filter = KalmanFilter.LocalLevel(0.05, 1.0);
        var (states, observations) = filter.Simulate(2000, new GraviRandom(37));

        var filtered = filter.Filter(observations).Filtered;

        var total = 0.0;
        var counted = 0;

        for (var t = 100; t < 2000; t++)      // skip the burn-in from the diffuse prior
        {
            var error = filtered[t].Mean.At(0) - states[t, 0];
            var standardised = error / Math.Sqrt(filtered[t].Covariance[0, 0]);
            total += standardised * standardised;
            counted++;
        }

        var variance = total / counted;
        Assert.True(variance is > 0.7 and < 1.4,
            $"standardised errors had variance {variance:F3}, so the reported uncertainty is off");
    }

    [Fact]
    public void TheLogLikelihoodPeaksAtTheTrueNoiseRatio()
    {
        // What makes parameter fitting possible: the series decomposes into independent one-step
        // prediction errors, so the likelihood is a real likelihood and has a maximum in the right
        // place.
        var truth = KalmanFilter.LocalLevel(0.05, 1.0);
        var (_, observations) = truth.Simulate(1000, new GraviRandom(41));

        double Score(double q) => KalmanFilter.LocalLevel(q, 1.0).Filter(observations).LogLikelihood;

        var correct = Score(0.05);
        Assert.True(correct > Score(1e-6), "a far too small process noise should score worse");
        Assert.True(correct > Score(10.0), "a far too large process noise should score worse");
    }

    [Fact]
    public void SimulationReproducesTheModelsOwnStatistics()
    {
        var filter = KalmanFilter.LocalLevel(0.0, 4.0);      // a fixed state, noisy observations
        var (states, observations) = filter.Simulate(20000, new GraviRandom(43));

        // With zero process noise the state never moves from its zero start.
        for (var t = 0; t < 100; t++) Assert.Equal(0.0, states[t, 0], 10);

        var variance = 0.0;
        for (var t = 0; t < 20000; t++) variance += observations[t, 0] * observations[t, 0];
        variance /= 20000;

        Assert.True(Math.Abs(variance - 4.0) < 0.15,
            $"observation variance came out at {variance:F4} against the model's 4.0");
    }

    [Fact]
    public void MalformedInputsAreRejected()
    {
        var filter = KalmanFilter.LocalLevel(0.1, 1.0);

        // The wrong observation width.
        Assert.Throws<ArgumentException>(() => filter.Filter(NdArray.Zeros(10, 3)));

        // A rank 1 series, which is ambiguous between one long observation and many short ones.
        Assert.Throws<ArgumentException>(() => filter.Filter(NdArray.Zeros(10)));

        Assert.Throws<ArgumentOutOfRangeException>(() => filter.Forecast(NdArray.Zeros(5, 1), 0));

        // A non-square transition matrix.
        Assert.Throws<ArgumentException>(() => new KalmanFilter(
            NdArray.Zeros(2, 3), NdArray.Zeros(1, 3), NdArray.Zeros(2, 2), NdArray.Zeros(1, 1)));
    }
}
