using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsDistAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Multivariate distributions against scipy 1.17.1 and closed forms.</summary>
public class StatsMultivariateTests
{
    private static readonly double[] Mean3 = [1, -0.5, 2];
    private static NdArray Cov3() => new([2, 0.3, 0.5, 0.3, 1, -0.2, 0.5, -0.2, 1.5], 3, 3);

    [Fact]
    public void MultivariateNormalLogPdfAndEntropyMatchScipy()
    {
        var d = Stats.MultivariateNormal(Mean3, Cov3());
        // scipy 1.17.1 multivariate_normal.logpdf / entropy
        Close(-3.4624610501495985, d.LogPdf([1.5, 0, 2.5]), 1e-13, "logpdf");
        Close(-25.63973377742233, d.LogPdf([-3, 2, 6]), 1e-13, "logpdf");
        Close(-109.4599357976243, d.LogPdf([10, -8, -4]), 1e-13, "logpdf");
        Close(4.709935797624346, d.Entropy(), 1e-13, "entropy");
        var rows = d.LogPdf(new NdArray([1.5, 0, 2.5, -3, 2, 6], 2, 3));
        Close(-25.63973377742233, rows[1], 1e-13, "row logpdf");
    }

    [Fact]
    public void MultivariateNormalCdfMatchesScipyToQmcAccuracy()
    {
        var d = Stats.MultivariateNormal(Mean3, Cov3());
        // scipy 1.17.1 (mvndst, abseps 1e-5): 0.32562530362592823; with lower_limit 0.07477488177423618
        Near(0.32562530362592823, d.Cdf([1.5, 0, 2.5], seed: 1), 3e-5, "cdf");
        Near(0.07477488177423618, d.Cdf([1.5, 0, 2.5], lowerLimit: [0, -1, 1], seed: 1), 3e-5, "cdf with lower limit");
    }

    [Fact]
    public void MultivariateNormalOrthantProbabilityIsExact()
    {
        // Equicorrelated ρ = ½ orthant probabilities: 1/8 + 3/(4π)·asin ρ = ¼ in 3-D, and Plackett's
        // 1/4 + asin(ρ)/(2π) = 1/3 in 2-D.
        var c3 = new NdArray([1, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1], 3, 3);
        Near(0.25, Stats.MultivariateNormal([0, 0, 0], c3).Cdf([0, 0, 0], seed: 3), 2e-5, "3-D orthant");
        var c2 = new NdArray([1, 0.5, 0.5, 1], 2, 2);
        Near(1.0 / 3.0, Stats.MultivariateNormal([0, 0], c2).Cdf([0, 0], seed: 3), 2e-5, "2-D orthant");
        // Independent components: the CDF factorises.
        var id = new NdArray([1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 4, 4);
        var expected = Special.Ndtr(0.3) * Special.Ndtr(1.0 / 2) * Special.Ndtr(-0.4) * Special.Ndtr(1.1);
        Near(expected, Stats.MultivariateNormal([0, 0, 0, 0], id).Cdf([0.3, 1, -0.4, 1.1], seed: 5), 1e-5, "factorised");
    }

    [Fact]
    public void SingularMultivariateNormalLivesOnItsSubspace()
    {
        var d = Stats.MultivariateNormal([0, 0], new NdArray([1, 1, 1, 1], 2, 2), allowSingular: true);
        // scipy 1.17.1: logpdf([.5,.5]) = -1.3905121234846454, off the line -inf, entropy 1.7655121234846454
        Close(-1.3905121234846454, d.LogPdf([0.5, 0.5]), 1e-13, "on support");
        Assert.True(double.IsNegativeInfinity(d.LogPdf([0.5, 0.7])));
        Close(1.7655121234846454, d.Entropy(), 1e-13, "entropy");
        Assert.Equal(1, d.Rank);
        Assert.Throws<ArgumentException>(() => Stats.MultivariateNormal([0, 0], new NdArray([1, 1, 1, 1], 2, 2)));
    }

    [Fact]
    public void MultivariateNormalSamplesHaveTheRightCovariance()
    {
        var d = Stats.MultivariateNormal(Mean3, Cov3());
        var x = d.Rvs(200_000, seed: 11).AsContiguous().ToArray();
        var n = x.Length / 3;
        var m = new double[3];
        for (var i = 0; i < n; i++) for (var j = 0; j < 3; j++) m[j] += x[i * 3 + j] / n;
        for (var j = 0; j < 3; j++) Near(Mean3[j], m[j], 0.015, $"mean {j}");
        var c = Cov3().AsContiguous().ToArray();
        for (var a = 0; a < 3; a++)
            for (var b = 0; b < 3; b++)
            {
                var s = 0.0;
                for (var i = 0; i < n; i++) s += (x[i * 3 + a] - m[a]) * (x[i * 3 + b] - m[b]);
                Near(c[a * 3 + b], s / (n - 1), 0.02, $"cov {a},{b}");
            }
    }

    [Fact]
    public void MultivariateTMatchesScipy()
    {
        var t = Stats.MultivariateT([0.5, 1], new NdArray([2, 0.4, 0.4, 1], 2, 2), df: 4.5);
        // scipy 1.17.1 multivariate_t
        Close(-2.7981494892759198, t.LogPdf([1, 2]), 1e-13, "logpdf");
        Close(-11.637163109933649, t.LogPdf([-6, 7]), 1e-13, "logpdf");
        Close(3.587204296664236, t.Entropy(), 1e-12, "entropy");
    }

    [Fact]
    public void DirichletMatchesScipy()
    {
        var d = Stats.Dirichlet([2.5, 1.2, 3.0]);
        // scipy 1.17.1 dirichlet
        Close(1.6178367187469211, d.LogPdf([0.3, 0.2, 0.5]), 1e-13, "logpdf");
        Close(1.6178367187469211, d.LogPdf([0.3, 0.2]), 1e-13, "logpdf, K-1 components");
        Close(-1.2196485754805693, d.Entropy(), 1e-13, "entropy");
        double[] mean = [0.3731343283582089, 0.1791044776119403, 0.44776119402985076];
        double[] var = [0.03037728589076328, 0.01909429398847978, 0.0321131307988069];
        for (var i = 0; i < 3; i++) { Close(mean[i], d.Mean()[i], 1e-14, "mean"); Close(var[i], d.Var()[i], 1e-14, "var"); }
        var s = d.Rvs(5, seed: 1).AsContiguous().ToArray();
        for (var i = 0; i < 5; i++) Near(1.0, s[i * 3] + s[i * 3 + 1] + s[i * 3 + 2], 1e-14, "simplex");
    }

    [Fact]
    public void MultinomialMatchesScipy()
    {
        var m = Stats.Multinomial(10, [0.2, 0.5, 0.3]);
        // scipy 1.17.1 multinomial
        Close(-2.4645159601402664, m.LogPmf([2, 5, 3]), 1e-13, "logpmf");
        Close(-12.03972804325936, m.LogPmf([0, 0, 10]), 1e-13, "logpmf");
        Assert.True(double.IsNegativeInfinity(m.LogPmf([2, 5, 4])));
        Close(3.3412185163945054, m.Entropy(), 1e-12, "entropy");
        var draws = m.Rvs(1000, seed: 2).AsContiguous().ToArray();
        for (var i = 0; i < 1000; i++) Assert.Equal(10.0, draws[i * 3] + draws[i * 3 + 1] + draws[i * 3 + 2]);
    }

    [Fact]
    public void WishartAndInverseWishartMatchScipy()
    {
        var s = new NdArray([2, 0.5, 0.5, 1], 2, 2);
        var x = new NdArray([3, 1, 1, 2.5], 2, 2);
        // scipy 1.17.1 wishart / invwishart
        Close(-5.850021009134311, Stats.Wishart(5, s).LogPdf(x), 1e-13, "wishart logpdf");
        Close(7.649972061236879, Stats.Wishart(5, s).Entropy(), 1e-13, "wishart entropy");
        Close(-10.949414492426692, Stats.InvWishart(5, s).LogPdf(x), 1e-13, "invwishart logpdf");

        // Sample means: df·S for the Wishart, S/(df − p − 1) for the inverse.
        var w = Stats.Wishart(5, s).Rvs(20000, seed: 4);
        var iw = Stats.InvWishart(8, s).Rvs(20000, seed: 4);
        var sa = s.AsContiguous().ToArray();
        for (var k = 0; k < 4; k++)
        {
            Near(5 * sa[k], w.Average(m => m.AsContiguous().ToArray()[k]), 0.1, "wishart mean");
            Near(sa[k] / 5, iw.Average(m => m.AsContiguous().ToArray()[k]), 0.01, "invwishart mean");
        }
    }

    [Fact]
    public void OrthoGroupsAreOrthogonalWithTheRightDeterminant()
    {
        for (var seed = 0; seed < 5; seed++)
        {
            var q = Stats.SpecialOrthoGroup(5, seed).AsContiguous().ToArray();
            var o = Stats.OrthoGroup(4, seed).AsContiguous().ToArray();
            CheckOrthogonal(q, 5);
            CheckOrthogonal(o, 4);
            Near(1.0, Linalg.Det(new NdArray(q, 5, 5)), 1e-12, "det");
        }
        // Haar measure: E[Q₀₀] = 0 and E[Q₀₀²] = 1/n. Without the sign correction the first is not zero.
        double m1 = 0, m2 = 0;
        const int reps = 4000;
        var rng = new GraviRandom(9);
        for (var i = 0; i < reps; i++)
        {
            var q = RandomMatrixGroups.OrthoGroup(3, rng).AsContiguous().ToArray();
            m1 += q[0] / reps;
            m2 += q[0] * q[0] / reps;
        }
        Near(0.0, m1, 0.03, "E[Q00]");
        Near(1.0 / 3, m2, 0.02, "E[Q00^2]");
    }

    private static void CheckOrthogonal(double[] q, int n)
    {
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                var s = 0.0;
                for (var k = 0; k < n; k++) s += q[k * n + i] * q[k * n + j];
                Near(i == j ? 1 : 0, s, 1e-13, "QᵀQ");
            }
    }
}
