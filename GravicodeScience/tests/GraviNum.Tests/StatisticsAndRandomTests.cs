using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviNum;

public class StatisticsTests
{
    private static readonly NdArray Sample = NdArray.FromValues([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);

    [Fact]
    public void BasicMoments_MatchHandComputedValues()
    {
        Assert.Equal(40.0, Statistics.Sum(Sample), 12);
        Assert.Equal(5.0, Statistics.Mean(Sample), 12);
        Assert.Equal(4.0, Statistics.Var(Sample), 12);
        Assert.Equal(2.0, Statistics.Std(Sample), 12);
        Assert.Equal(2.0, Statistics.Min(Sample), 12);
        Assert.Equal(9.0, Statistics.Max(Sample), 12);
    }

    [Fact]
    public void SampleVariance_UsesTheDdofCorrection()
    {
        // n = 8, so the unbiased estimate is 32 / 7 rather than 32 / 8.
        Assert.Equal(32.0 / 7.0, Statistics.Var(Sample, ddof: 1), 12);
    }

    [Fact]
    public void MedianAndPercentiles_InterpolateLinearly()
    {
        Assert.Equal(4.5, Statistics.Median(Sample), 12);
        Assert.Equal(2.0, Statistics.Percentile(Sample, 0), 12);
        Assert.Equal(9.0, Statistics.Percentile(Sample, 100), 12);
    }

    [Fact]
    public void ArgMinAndArgMax_ReturnFlatIndices()
    {
        Assert.Equal(0, Statistics.ArgMin(Sample));
        Assert.Equal(7, Statistics.ArgMax(Sample));
    }

    [Fact]
    public void AxisReductions_CollapseTheRequestedAxis()
    {
        var a = NdArray.Arange(6).Reshape(2, 3);

        var columnSums = Statistics.Sum(a, axis: 0);
        Assert.Equal(3, columnSums.Size);
        Assert.Equal(new[] { 3.0, 5.0, 7.0 }, columnSums.ToArray());

        var rowSums = Statistics.Sum(a, axis: 1);
        Assert.Equal(2, rowSums.Size);
        Assert.Equal(new[] { 3.0, 12.0 }, rowSums.ToArray());

        var rowMeans = Statistics.Mean(a, axis: 1);
        Assert.Equal(new[] { 1.0, 4.0 }, rowMeans.ToArray());
    }

    [Fact]
    public void AxisReductions_WorkOnHigherRankArrays()
    {
        var a = NdArray.Arange(24).Reshape(2, 3, 4);
        var reduced = Statistics.Sum(a, axis: 1);

        Assert.Equal(2, reduced.Rank);
        Assert.Equal(2, reduced.Shape[0]);
        Assert.Equal(4, reduced.Shape[1]);
        // Element [0,0] sums indices (0,0,0), (0,1,0), (0,2,0) = 0 + 4 + 8.
        Assert.Equal(12.0, reduced[0, 0], 12);
    }

    [Fact]
    public void Correlation_IsOneForAPerfectLinearRelationship()
    {
        var x = NdArray.Arange(10);
        var y = x * 3.0 + 5.0;
        Assert.Equal(1.0, Statistics.Correlation(x, y), 9);

        var inverse = x * -2.0;
        Assert.Equal(-1.0, Statistics.Correlation(x, inverse), 9);
    }

    [Fact]
    public void SpearmanCorrelation_IsOneForAnyMonotonicRelationship()
    {
        var x = NdArray.Arange(1, 11);
        var y = x.Map(v => Math.Exp(v));
        Assert.Equal(1.0, Statistics.SpearmanCorrelation(x, y), 9);
    }

    [Fact]
    public void CorrelationMatrix_HasUnitDiagonalAndIsSymmetric()
    {
        var rng = new GraviRandom(31);
        var data = rng.StandardNormal(200, 4);
        var corr = Statistics.CorrelationMatrix(data);

        for (var i = 0; i < 4; i++) Assert.Equal(1.0, corr[i, i], 9);
        for (var i = 0; i < 4; i++)
            for (var j = 0; j < 4; j++)
                Assert.Equal(corr[i, j], corr[j, i], 12);
    }

    [Fact]
    public void CumulativeSum_And_Diff_AreInverses()
    {
        var a = NdArray.FromValues([1.0, 3.0, 6.0, 10.0]);
        var diff = Statistics.Diff(a);
        Assert.Equal(new[] { 2.0, 3.0, 4.0 }, diff.ToArray());

        var cumulative = Statistics.CumulativeSum(NdArray.FromValues([1.0, 2.0, 3.0]));
        Assert.Equal(new[] { 1.0, 3.0, 6.0 }, cumulative.ToArray());
    }

    [Fact]
    public void Standardize_ProducesZeroMeanUnitVariance()
    {
        var rng = new GraviRandom(41);
        var z = Statistics.Standardize(rng.Normal(10.0, 3.0, 500));
        Assert.Equal(0.0, Statistics.Mean(z), 9);
        Assert.Equal(1.0, Statistics.Std(z), 9);
    }

    [Fact]
    public void Histogram_CountsEveryElementExactlyOnce()
    {
        var rng = new GraviRandom(43);
        var data = rng.Random(1000);
        var (edges, counts) = Statistics.Histogram(data, bins: 10);

        Assert.Equal(11, edges.Length);
        Assert.Equal(1000, counts.Sum());
    }

    [Fact]
    public void Describe_ReportsTheStandardSummaryKeys()
    {
        var summary = Statistics.Describe(Sample);
        Assert.Equal(8.0, summary["count"]);
        Assert.Equal(5.0, summary["mean"], 12);
        Assert.Equal(4.5, summary["50%"], 12);
    }
}

public class GraviRandomTests
{
    [Fact]
    public void SameSeed_ProducesTheSameSequence()
    {
        var a = new GraviRandom(1234);
        var b = new GraviRandom(1234);
        for (var i = 0; i < 100; i++) Assert.Equal(a.NextDouble(), b.NextDouble());
    }

    [Fact]
    public void DifferentSeeds_ProduceDifferentSequences()
    {
        var a = new GraviRandom(1);
        var b = new GraviRandom(2);
        Assert.NotEqual(a.NextDouble(), b.NextDouble());
    }

    [Fact]
    public void UniformSamples_StayInRangeAndCentreCorrectly()
    {
        var rng = new GraviRandom(7);
        var values = rng.Uniform(-2.0, 5.0, 20_000);

        Assert.True(Statistics.Min(values) >= -2.0);
        Assert.True(Statistics.Max(values) < 5.0);
        Assert.Equal(1.5, Statistics.Mean(values), 1);
    }

    [Fact]
    public void NormalSamples_MatchTheRequestedMomentsWithinSamplingError()
    {
        var rng = new GraviRandom(11);
        var values = rng.Normal(3.0, 2.0, 50_000);

        Assert.Equal(3.0, Statistics.Mean(values), 1);
        Assert.Equal(2.0, Statistics.Std(values), 1);
    }

    [Fact]
    public void BinomialSamples_MatchTheAnalyticMean()
    {
        var rng = new GraviRandom(13);

        // Below 30 trials the direct Bernoulli path runs.
        var small = rng.Binomial(20, 0.3, 20_000);
        Assert.Equal(6.0, Statistics.Mean(small), 1);

        // Above it the recursive beta-splitting path runs; the mean must still be n*p.
        var large = rng.Binomial(200, 0.25, 20_000);
        Assert.Equal(50.0, Statistics.Mean(large), 0);
        Assert.True(Statistics.Min(large) >= 0);
        Assert.True(Statistics.Max(large) <= 200);
    }

    [Fact]
    public void PoissonSamples_HaveMeanAndVarianceEqualToLambda()
    {
        var rng = new GraviRandom(17);

        var knuth = rng.Poisson(4.0, 20_000);
        Assert.Equal(4.0, Statistics.Mean(knuth), 1);
        Assert.Equal(4.0, Statistics.Var(knuth), 0);

        // Lambda above 30 switches to transformed rejection.
        var ptrs = rng.Poisson(50.0, 20_000);
        Assert.Equal(50.0, Statistics.Mean(ptrs), 0);
    }

    [Fact]
    public void GammaSamples_MatchShapeTimesScale()
    {
        var rng = new GraviRandom(19);
        var values = rng.Gamma(2.0, 3.0, 40_000);
        Assert.Equal(6.0, Statistics.Mean(values), 0);

        // Shapes below one take the boosting branch.
        var small = rng.Gamma(0.5, 1.0, 40_000);
        Assert.Equal(0.5, Statistics.Mean(small), 1);
    }

    [Fact]
    public void BetaSamples_MatchTheAnalyticMean()
    {
        var rng = new GraviRandom(23);
        var values = rng.Beta(2.0, 5.0, 40_000);
        Assert.Equal(2.0 / 7.0, Statistics.Mean(values), 2);
    }

    [Fact]
    public void Permutation_IsABijection()
    {
        var rng = new GraviRandom(29);
        var permutation = rng.Permutation(500);
        Assert.Equal(500, permutation.Distinct().Count());
        Assert.Equal(0, permutation.Min());
        Assert.Equal(499, permutation.Max());
    }

    [Fact]
    public void ChoiceWithoutReplacement_ReturnsDistinctIndices()
    {
        var rng = new GraviRandom(31);
        var picked = rng.Choice(100, 20, replace: false);
        Assert.Equal(20, picked.Distinct().Count());
    }

    [Fact]
    public void MultivariateNormal_ReproducesTheRequestedCovariance()
    {
        var rng = new GraviRandom(37);
        var mean = NdArray.FromValues([1.0, -2.0]);
        var cov = NdArray.FromArray(new double[,] { { 2.0, 0.6 }, { 0.6, 1.0 } });

        var samples = rng.MultivariateNormal(mean, cov, 40_000);
        var empirical = Statistics.CovarianceMatrix(samples);

        Assert.Equal(1.0, Statistics.Mean(samples.Column(0).Copy()), 1);
        Assert.Equal(-2.0, Statistics.Mean(samples.Column(1).Copy()), 1);
        Assert.Equal(2.0, empirical[0, 0], 1);
        Assert.Equal(0.6, empirical[0, 1], 1);
    }
}

public class MathUtilTests
{
    [Fact]
    public void LogGamma_MatchesKnownFactorials()
    {
        Assert.Equal(0.0, MathUtil.LogGamma(1.0), 10);
        Assert.Equal(Math.Log(24.0), MathUtil.LogGamma(5.0), 10);
        Assert.Equal(0.5 * Math.Log(Math.PI), MathUtil.LogGamma(0.5), 10);
    }

    [Fact]
    public void Erf_MatchesReferenceValues()
    {
        Assert.Equal(0.0, MathUtil.Erf(0.0), 12);
        Assert.Equal(0.8427007929497149, MathUtil.Erf(1.0), 10);
        Assert.Equal(-0.8427007929497149, MathUtil.Erf(-1.0), 10);
        Assert.Equal(0.9953222650189527, MathUtil.Erf(2.0), 10);
    }

    [Fact]
    public void ErfInv_InvertsErf()
    {
        foreach (var x in new[] { -0.9, -0.3, 0.0, 0.25, 0.75, 0.99 })
            Assert.Equal(x, MathUtil.Erf(MathUtil.ErfInv(x)), 10);
    }

    [Fact]
    public void NormalCdf_MatchesTheStandardTable()
    {
        Assert.Equal(0.5, MathUtil.NormalCdf(0.0), 10);
        Assert.Equal(0.8413447460685429, MathUtil.NormalCdf(1.0), 9);
        Assert.Equal(0.9772498680518208, MathUtil.NormalCdf(2.0), 9);
    }

    [Fact]
    public void NormalQuantile_InvertsTheCdf()
    {
        foreach (var p in new[] { 0.025, 0.1, 0.5, 0.9, 0.975 })
            Assert.Equal(p, MathUtil.NormalCdf(MathUtil.NormalQuantile(p)), 8);
    }

    [Fact]
    public void GammaP_MatchesTheExponentialCdf()
    {
        // P(1, x) is exactly 1 - exp(-x).
        foreach (var x in new[] { 0.5, 1.0, 3.0 })
            Assert.Equal(1.0 - Math.Exp(-x), MathUtil.GammaP(1.0, x), 10);
    }

    [Fact]
    public void BetaInc_IsSymmetricAndBounded()
    {
        Assert.Equal(0.0, MathUtil.BetaInc(2, 3, 0.0), 12);
        Assert.Equal(1.0, MathUtil.BetaInc(2, 3, 1.0), 12);
        // I_x(a,b) = 1 - I_(1-x)(b,a)
        Assert.Equal(MathUtil.BetaInc(2, 5, 0.3), 1.0 - MathUtil.BetaInc(5, 2, 0.7), 10);
    }

    [Fact]
    public void LogSumExp_IsStableForLargeMagnitudes()
    {
        double[] values = [1000.0, 1000.0];
        Assert.Equal(1000.0 + Math.Log(2), MathUtil.LogSumExp(values), 10);
    }

    [Fact]
    public void Softmax_SumsToOne()
    {
        var probabilities = MathUtil.Softmax([1.0, 2.0, 3.0]);
        Assert.Equal(1.0, probabilities.Sum(), 12);
        Assert.True(probabilities[2] > probabilities[1]);
    }
}
