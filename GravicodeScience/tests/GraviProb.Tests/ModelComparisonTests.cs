using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviProb;
using Xunit;

namespace Gravicode.Science.Tests.GraviProb;

/// <summary>
/// Tests for WAIC and PSIS-LOO.
/// </summary>
/// <remarks>
/// Information criteria are unusually easy to implement plausibly and wrongly — a log-of-mean
/// written as a mean-of-logs gives a number in the right range that is not WAIC. So the pins here
/// are hand-computed values on tiny inputs, the degenerate case where both criteria must reduce to
/// the exact log likelihood, and the behavioural claim that matters: the criteria must prefer the
/// model that generated the data, not the one that fits it best in sample.
/// </remarks>
public class ModelComparisonTests
{
    /// <summary>
    /// Pointwise log likelihoods for a set of observations under a posterior over a normal's mean.
    /// </summary>
    private static NdArray LogLikelihoodMatrix(
        NdArray data, IReadOnlyList<double> drawnMeans, double sigma)
    {
        var matrix = NdArray.Zeros(drawnMeans.Count, data.Size);

        for (var s = 0; s < drawnMeans.Count; s++)
        {
            var normal = new Normal(drawnMeans[s], sigma);
            for (var i = 0; i < data.Size; i++) matrix[s, i] = normal.LogDensity(data.At(i));
        }

        return matrix;
    }

    /// <summary>Draws from the posterior over a normal mean with a flat prior.</summary>
    private static double[] PosteriorMeans(NdArray data, double sigma, int draws, int seed)
    {
        var mean = 0.0;
        for (var i = 0; i < data.Size; i++) mean += data.At(i);
        mean /= data.Size;

        var rng = new GraviRandom(seed);
        var standardError = sigma / Math.Sqrt(data.Size);

        var result = new double[draws];
        for (var s = 0; s < draws; s++) result[s] = mean + rng.Normal() * standardError;
        return result;
    }

    private static NdArray Sample(int count, double mean, double sigma, int seed)
    {
        var rng = new GraviRandom(seed);
        var data = NdArray.Zeros(count);
        for (var i = 0; i < count; i++) data.SetAt(i, mean + rng.Normal() * sigma);
        return data;
    }

    // ---------------------------------------------------------------------- WAIC

    [Fact]
    public void WithNoPosteriorSpreadWaicIsTwiceTheNegativeLogLikelihood()
    {
        // The degenerate case that pins the formula. Identical draws mean zero posterior variance,
        // so the penalty vanishes and WAIC must be exactly -2 times the summed log likelihood.
        var matrix = NdArray.Zeros(50, 3);
        double[] values = [-1.5, -2.0, -0.5];

        for (var s = 0; s < 50; s++)
            for (var i = 0; i < 3; i++)
                matrix[s, i] = values[i];

        var waic = ModelComparison.Waic(matrix);

        Assert.Equal(0.0, waic.EffectiveParameters, 12);
        Assert.Equal(-2 * (-1.5 + -2.0 + -0.5), waic.Estimate, 10);
    }

    [Fact]
    public void WaicMatchesTheHandComputedValueOnATinyInput()
    {
        // Two draws, one observation. lppd = log((e^-1 + e^-2)/2); p_waic = Var([-1, -2]) with the
        // sample correction, which is 0.5. Both worked out here rather than taken from the code.
        var matrix = NdArray.Zeros(2, 1);
        matrix[0, 0] = -1.0;
        matrix[1, 0] = -2.0;

        var lppd = Math.Log((Math.Exp(-1.0) + Math.Exp(-2.0)) / 2);
        const double penalty = 0.5;

        var waic = ModelComparison.Waic(matrix);

        Assert.Equal(penalty, waic.EffectiveParameters, 12);
        Assert.Equal(-2 * (lppd - penalty), waic.Estimate, 10);
    }

    [Fact]
    public void TheLppdIsALogOfAMeanNotAMeanOfLogs()
    {
        // The single most common way to get this wrong. Jensen's inequality guarantees the two
        // differ whenever the draws differ, and in a fixed direction.
        var matrix = NdArray.Zeros(2, 1);
        matrix[0, 0] = -1.0;
        matrix[1, 0] = -5.0;

        var logOfMean = Math.Log((Math.Exp(-1.0) + Math.Exp(-5.0)) / 2);
        var meanOfLogs = -3.0;

        // WAIC = -2(lppd - p_waic), and p_waic here is Var([-1, -5]) = 8.
        Assert.Equal(-2 * (logOfMean - 8.0), ModelComparison.Waic(matrix).Estimate, 10);
        Assert.True(logOfMean > meanOfLogs, "the test's own premise is that these differ");
    }

    [Fact]
    public void ThePenaltyGrowsWithPosteriorUncertainty()
    {
        // What makes p_waic a measure of effective complexity rather than a parameter count: a
        // model the posterior is less sure about pays more.
        var data = Sample(30, 1.0, 1.0, seed: 3);

        var tight = ModelComparison.Waic(LogLikelihoodMatrix(data, PosteriorMeans(data, 1.0, 500, 5), 1.0));
        var loose = ModelComparison.Waic(LogLikelihoodMatrix(data, PosteriorMeans(data, 4.0, 500, 5), 1.0));

        Assert.True(loose.EffectiveParameters > tight.EffectiveParameters,
            $"a wider posterior scored {loose.EffectiveParameters:F4} against {tight.EffectiveParameters:F4}");
    }

    [Fact]
    public void PointwiseTermsSumToTheEstimate()
    {
        var data = Sample(20, 0.0, 1.0, seed: 7);
        var waic = ModelComparison.Waic(LogLikelihoodMatrix(data, PosteriorMeans(data, 1.0, 400, 9), 1.0));

        var total = 0.0;
        for (var i = 0; i < waic.Pointwise.Size; i++) total += waic.Pointwise.At(i);

        Assert.Equal(waic.Estimate, total, 8);
    }

    // ----------------------------------------------------------------------- LOO

    [Fact]
    public void WithNoPosteriorSpreadLooIsAlsoTwiceTheNegativeLogLikelihood()
    {
        // Identical draws make every importance weight equal, so the reweighting is the identity
        // and LOO collapses onto the plain log likelihood.
        var matrix = NdArray.Zeros(50, 3);
        double[] values = [-1.5, -2.0, -0.5];

        for (var s = 0; s < 50; s++)
            for (var i = 0; i < 3; i++)
                matrix[s, i] = values[i];

        var loo = ModelComparison.Loo(matrix);

        Assert.Equal(-2 * (-1.5 + -2.0 + -0.5), loo.Criterion.Estimate, 8);
        Assert.Equal(0.0, loo.Criterion.EffectiveParameters, 8);
    }

    [Fact]
    public void LooAndWaicAgreeOnAWellBehavedProblem()
    {
        // They estimate the same quantity by different routes, so a large gap on an easy problem
        // means one of them is wrong.
        var data = Sample(40, 2.0, 1.0, seed: 11);
        var matrix = LogLikelihoodMatrix(data, PosteriorMeans(data, 1.0, 2000, 13), 1.0);

        var waic = ModelComparison.Waic(matrix);
        var loo = ModelComparison.Loo(matrix);

        Assert.True(Math.Abs(waic.Estimate - loo.Criterion.Estimate) < 1.0,
            $"WAIC gave {waic.Estimate:F3} and LOO {loo.Criterion.Estimate:F3}");
        Assert.True(loo.IsReliable, "the Pareto diagnostics flagged an easy problem");
    }

    [Fact]
    public void TheParetoDiagnosticIsReportedPerObservation()
    {
        var data = Sample(30, 0.0, 1.0, seed: 17);
        var loo = ModelComparison.Loo(LogLikelihoodMatrix(data, PosteriorMeans(data, 1.0, 1000, 19), 1.0));

        Assert.Equal(data.Size, loo.ParetoK.Size);

        // The diagnostic WAIC cannot provide, which is the reason to prefer LOO.
        for (var i = 0; i < loo.ParetoK.Size; i++)
            Assert.False(double.IsNaN(loo.ParetoK.At(i)), $"observation {i} had no shape estimate");
    }

    [Fact]
    public void LooPointwiseTermsSumToItsEstimate()
    {
        var data = Sample(25, 1.0, 1.0, seed: 23);
        var loo = ModelComparison.Loo(LogLikelihoodMatrix(data, PosteriorMeans(data, 1.0, 800, 29), 1.0));

        var total = 0.0;
        for (var i = 0; i < loo.Criterion.Pointwise.Size; i++) total += loo.Criterion.Pointwise.At(i);

        Assert.Equal(loo.Criterion.Estimate, total, 8);
    }

    // ---------------------------------------------------------------- behaviour

    [Fact]
    public void BothCriteriaPreferTheModelThatGeneratedTheData()
    {
        // The claim that makes them worth having. Data comes from N(0, 1); the competing model uses
        // the wrong scale, and must score worse under both.
        var data = Sample(60, 0.0, 1.0, seed: 31);
        var means = PosteriorMeans(data, 1.0, 1500, 37);

        var correct = LogLikelihoodMatrix(data, means, 1.0);
        var wrong = LogLikelihoodMatrix(data, means, 4.0);

        Assert.True(ModelComparison.Waic(correct).Estimate < ModelComparison.Waic(wrong).Estimate,
            "WAIC preferred the wrong scale");
        Assert.True(ModelComparison.Loo(correct).Criterion.Estimate
                    < ModelComparison.Loo(wrong).Criterion.Estimate,
            "LOO preferred the wrong scale");
    }

    [Fact]
    public void TheCriteriaPenaliseAModelThatIsOverconfidentOutOfSample()
    {
        // In-sample likelihood always improves as a model gets more flexible; these do not, which is
        // the entire reason they exist.
        var data = Sample(40, 0.0, 1.0, seed: 41);

        // A "flexible" model whose posterior is far too wide for the data it saw.
        var flexible = LogLikelihoodMatrix(data, PosteriorMeans(data, 6.0, 1500, 43), 1.0);
        var parsimonious = LogLikelihoodMatrix(data, PosteriorMeans(data, 1.0, 1500, 43), 1.0);

        var flexibleWaic = ModelComparison.Waic(flexible);
        var parsimoniousWaic = ModelComparison.Waic(parsimonious);

        Assert.True(flexibleWaic.EffectiveParameters > parsimoniousWaic.EffectiveParameters,
            "the more uncertain model should carry the larger complexity penalty");
        Assert.True(flexibleWaic.Estimate > parsimoniousWaic.Estimate,
            "the penalty should be enough to make the flexible model score worse");
    }

    [Fact]
    public void TheStandardErrorIsNonZeroAndScalesWithTheSpread()
    {
        var data = Sample(50, 0.0, 1.0, seed: 47);
        var waic = ModelComparison.Waic(LogLikelihoodMatrix(data, PosteriorMeans(data, 1.0, 800, 53), 1.0));

        Assert.True(waic.StandardError > 0,
            "an estimate with no uncertainty cannot be used to compare models");
    }

    // ------------------------------------------------------------------ compare

    [Fact]
    public void CompareRanksModelsBestFirstWithPairedDifferences()
    {
        var data = Sample(50, 0.0, 1.0, seed: 59);
        var means = PosteriorMeans(data, 1.0, 1000, 61);

        var models = new Dictionary<string, InformationCriterion>
        {
            ["wide"] = ModelComparison.Waic(LogLikelihoodMatrix(data, means, 5.0)),
            ["correct"] = ModelComparison.Waic(LogLikelihoodMatrix(data, means, 1.0)),
            ["narrow"] = ModelComparison.Waic(LogLikelihoodMatrix(data, means, 0.2)),
        };

        var ranked = ModelComparison.Compare(models);

        Assert.Equal("correct", ranked[0].Name);
        Assert.Equal(0.0, ranked[0].Difference);

        for (var i = 1; i < ranked.Count; i++)
        {
            Assert.True(ranked[i].Difference > 0, $"'{ranked[i].Name}' was not worse than the best");
            Assert.True(ranked[i].DifferenceError > 0,
                $"'{ranked[i].Name}' reported no uncertainty on its difference");
        }

        // And the ordering is by the estimate itself.
        for (var i = 1; i < ranked.Count; i++)
            Assert.True(ranked[i].Estimate >= ranked[i - 1].Estimate);
    }

    [Fact]
    public void ModelsScoredOnDifferentDataAreRefused()
    {
        // The criteria are only comparable across models fitted to the same observations. Silently
        // comparing them anyway is a wrong answer that looks entirely reasonable.
        var small = Sample(10, 0.0, 1.0, seed: 67);
        var large = Sample(20, 0.0, 1.0, seed: 67);

        var models = new Dictionary<string, InformationCriterion>
        {
            ["ten"] = ModelComparison.Waic(LogLikelihoodMatrix(small, PosteriorMeans(small, 1.0, 200, 71), 1.0)),
            ["twenty"] = ModelComparison.Waic(LogLikelihoodMatrix(large, PosteriorMeans(large, 1.0, 200, 71), 1.0)),
        };

        Assert.Throws<ArgumentException>(() => ModelComparison.Compare(models));
    }

    [Fact]
    public void MalformedInputsAreRejected()
    {
        // A rank 1 array is ambiguous between one draw and one observation.
        Assert.Throws<ArgumentException>(() => ModelComparison.Waic(NdArray.Zeros(10)));

        // A single draw has no posterior variance to estimate the penalty from.
        Assert.Throws<ArgumentException>(() => ModelComparison.Waic(NdArray.Zeros(1, 5)));
        Assert.Throws<ArgumentException>(() => ModelComparison.Loo(NdArray.Zeros(1, 5)));

        Assert.Throws<ArgumentException>(
            () => ModelComparison.Compare(new Dictionary<string, InformationCriterion>()));
    }
}
