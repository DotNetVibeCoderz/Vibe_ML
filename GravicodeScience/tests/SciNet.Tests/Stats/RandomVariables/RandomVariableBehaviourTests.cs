using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsDistAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>A distribution with nothing but a density: everything else must come from the fall-backs.</summary>
internal sealed class PdfOnlyExponential(double rate) : ContinuousRandomVariable
{
    protected internal override (double Lower, double Upper) SupportCore() => (0.0, double.PositiveInfinity);
    protected override double PdfFormula(double x) => rate * Math.Exp(-rate * x);
}

/// <summary>A discrete distribution with only a mass function (geometric on 1, 2, â€¦).</summary>
internal sealed class PmfOnlyGeometric(double p) : DiscreteRandomVariable
{
    protected internal override (double Lower, double Upper) SupportCore() => (1.0, double.PositiveInfinity);
    protected override double PmfFormula(double x) => Math.Pow(1 - p, x - 1) * p;
}

/// <summary>Behaviour of the random-variable framework that a table of values cannot show.</summary>
public class RandomVariableBehaviourTests
{
    [Fact]
    public void EveryMethodOverrideAgreesWithTheFormula()
    {
        var x = new Normal(1.0, 2.0);
        foreach (var t in new[] { -3.0, 0.5, 4.0 })
        {
            Close(x.Cdf(t), x.Cdf(t, RandomVariableMethod.Quadrature), 1e-11, "cdf quadrature");
            Close(x.Cdf(t), x.Cdf(t, RandomVariableMethod.Complement), 1e-14, "cdf complement");
            Close(x.Cdf(t), x.Cdf(t, RandomVariableMethod.LogExp), 1e-14, "cdf log/exp");
            Close(x.LogCcdf(t), x.LogCcdf(t, RandomVariableMethod.Quadrature), 1e-10, "logccdf quadrature");
            Close(x.Pdf(t), x.Pdf(t, RandomVariableMethod.LogExp), 1e-14, "pdf log/exp");
        }
        foreach (var p in new[] { 0.01, 0.5, 0.93 })
        {
            Close(x.Icdf(p), x.Icdf(p, RandomVariableMethod.Inversion), 1e-12, "icdf inversion");
            Close(x.Iccdf(p), x.Iccdf(p, RandomVariableMethod.Complement), 1e-13, "iccdf complement");
        }
        Close(x.Entropy(), x.Entropy(RandomVariableMethod.Quadrature), 1e-11, "entropy quadrature");
        Close(x.Median(), x.Median(RandomVariableMethod.Icdf), 1e-14, "median icdf");
        Near(x.Mode(), x.Mode(RandomVariableMethod.Optimization), 1e-6, "mode optimisation");
        Close(x.Variance(), x.Variance(RandomVariableMethod.Quadrature), 1e-11, "variance quadrature");
        Close(x.Moment(4, MomentKind.Central), x.Moment(4, MomentKind.Central, RandomVariableMethod.QuadratureIcdf), 1e-6, "central 4 by icdf");
        Assert.Throws<ArgumentException>(() => x.Pdf(0.0, RandomVariableMethod.Inversion));
        Assert.Throws<NotSupportedException>(() => new Uniform(0, 1).Moment(3, MomentKind.Central, RandomVariableMethod.Formula));
    }

    [Fact]
    public void ArgumentsOutsideTheSupportFollowScipysReplacements()
    {
        var u = new Uniform(1.0, 3.0);
        Assert.Equal(0.5, u.Pdf(1.0));            // inclusive ends: the density is evaluated there
        Assert.Equal(0.0, u.Pdf(0.999));
        Assert.Equal(0.0, u.Cdf(1.0));
        Assert.Equal(1.0, u.Cdf(3.0));
        Assert.Equal(double.NegativeInfinity, u.LogCdf(0.0));
        Assert.Equal(1.0, u.Icdf(0.0) - 0.0);
        Assert.Equal(3.0, u.Icdf(1.0));
        Assert.Equal(3.0, u.Iccdf(0.0));
        Assert.True(double.IsNaN(u.Icdf(1.5)));
        Assert.Equal(1.0, u.ILogCdf(double.NegativeInfinity));
        Assert.True(double.IsNaN(u.ILogCdf(0.1)));
        // Two-argument CDF: swapped arguments give the negative mass; out-of-support ends are clipped.
        Close(-0.25, u.Cdf(2.5, 2.0), 1e-15);
        Close(1.0, u.Cdf(-5, 5), 1e-15);
        Assert.Equal(Math.PI, u.LogCdf(2.5, 2.0).Imaginary);
        Close(1.25, u.Ccdf(2.5, 2.0), 1e-15); // 2 - (mass outside [2, 2.5]), scipy's convention
        // Discrete: zero mass between the integers, and no two-argument CDF.
        var b = new Binomial(10, 0.4);
        Assert.Equal(0.0, b.Pmf(2.5));
        Assert.Equal(double.PositiveInfinity, b.Pdf(3));
        Assert.Throws<NotSupportedException>(() => b.Cdf(1, 3));
    }

    [Fact]
    public void ArithmeticFoldsShiftsAndScales()
    {
        var y = (2.0 * new StandardNormal() + 1.0) * 3.0 - 4.0;
        var s = Assert.IsType<ShiftedScaledDistribution>(y);
        Assert.IsType<StandardNormal>(s.Inner);
        Assert.Equal(-1.0, s.Loc);
        Assert.Equal(6.0, s.Scale);
        Assert.Equal("6.0*StandardNormal() - 1.0", y.ToString());
        Close(36.0, y.Variance(), 1e-15);
        // Reflection swaps the tails.
        var n = -new Normal(2.0, 1.0);
        Close(new Normal(-2.0, 1.0).Cdf(-1.5), n.Cdf(-1.5), 1e-15);
        Close(new Normal(-2.0, 1.0).Icdf(0.2), n.Icdf(0.2), 1e-14);
        Assert.Throws<NotSupportedException>(() => 1.0 / new StandardNormal());
        Assert.Throws<NotSupportedException>(() => new StandardNormal().Pow(0));
        Assert.Throws<NotSupportedException>(() => new StandardNormal().Log());
        Assert.Throws<NotSupportedException>(() => ContinuousRandomVariable.Pow(1.0, new StandardNormal()));
        Assert.Throws<ArgumentException>(() => new StandardNormal() * 0.0);
        Assert.Throws<ArgumentException>(() => Stats.Truncate(new Uniform(0, 1), 2, 3));
        Assert.Throws<ArgumentException>(() => Stats.OrderStatistic(new Uniform(0, 1), 5, 4));
    }

    [Fact]
    public void ExpOfANormalIsTheLogNormal()
    {
        var y = new Normal(0.3, 0.7).Exp();
        var ln = Stats.LogNorm(0.7, scale: Math.Exp(0.3));
        foreach (var t in new[] { 0.05, 0.8, 1.9, 7.0 })
        {
            Close(ln.Pdf(t), y.Pdf(t), 1e-13, "pdf");
            Close(ln.Sf(t), y.Ccdf(t), 1e-13, "sf");
        }
        Close(ln.Mean(), y.Mean(), 1e-10, "mean by quadrature");
        Close(ln.Var(), y.Variance(), 1e-9, "variance by quadrature");
        Close(ln.Entropy(), y.Entropy(), 1e-10, "entropy by quadrature");
    }

    [Fact]
    public void OrderStatisticOfUniformIsBeta()
    {
        // The r-th of n uniforms is Beta(r, n - r + 1).
        var x = Stats.OrderStatistic(new Uniform(0, 1), r: 2, n: 6);
        var beta = Stats.Beta(2, 5);
        foreach (var t in new[] { 0.01, 0.2, 0.55, 0.97 })
        {
            Close(beta.Pdf(t), x.Pdf(t), 1e-13, "pdf");
            Close(beta.Cdf(t), x.Cdf(t), 1e-13, "cdf");
        }
        Close(beta.Mean(), x.Mean(), 1e-11, "mean");
        Close(beta.Var(), x.Variance(), 1e-10, "variance");
        Near(1.0 / 5.0, x.Mode(), 1e-7, "mode");
    }

    [Fact]
    public void ADensityAloneIsEnough()
    {
        var x = new PdfOnlyExponential(2.0);
        var e = Stats.Expon(scale: 0.5);
        foreach (var t in new[] { 0.01, 0.7, 5.0 })
        {
            Close(e.Cdf(t), x.Cdf(t), 1e-11, "cdf by quadrature");
            Close(e.LogSf(t), x.LogCcdf(t), 1e-10, "logccdf by log-space quadrature");
        }
        Close(e.Ppf(0.3), x.Icdf(0.3), 1e-10, "icdf by inversion");
        Close(e.Entropy(), x.Entropy(), 1e-10, "entropy");
        Close(0.5, x.Mean(), 1e-11, "mean");
        Close(2.0, x.Skewness(), 1e-9, "skewness");
        Near(0.0, x.Mode(), 1e-12, "mode at the end of the support");
        Close(Math.Log(2) / 2, x.Median(), 1e-10, "median");

        var g = new PmfOnlyGeometric(0.3);
        var geo = Stats.Geom(0.3);
        foreach (var k in new[] { 1.0, 2.0, 7.0, 7.5, 30.0 })
        {
            Close(geo.Cdf(k), g.Cdf(k), 1e-13, "discrete cdf by summation");
            Close(geo.Sf(k), g.Ccdf(k), 1e-12, "discrete ccdf by summation");
        }
        foreach (var p in new[] { 0.05, 0.3, 0.5, 0.99 }) Assert.Equal(geo.Ppf(p), g.Icdf(p));
        Assert.Equal(1.0, g.Mode());
        Close(geo.Mean(), g.Mean(), 1e-12, "discrete mean");
        Close(geo.Entropy(), g.Entropy(), 1e-12, "discrete entropy");
    }

    [Fact]
    public void SamplesHaveTheRightDistribution()
    {
        var rng = new GraviRandom(11);
        var cases = new (string Name, IUnivariateDistribution D)[]
        {
            ("Normal", new Normal(1, 2)),
            ("Logistic (inverse transform)", 3.0 * new Logistic() + 1.0),
            ("truncated", Stats.Truncate(new StandardNormal(), -0.5, 1.5)),
            ("folded", Stats.Abs(new Normal(0.5, 1))),
            ("exp", Stats.Exp(new StandardNormal())),
            ("order statistic", Stats.OrderStatistic(new StandardNormal(), 3, 5)),
            ("mixture", new Mixture([new Normal(-2, 0.5), new Uniform(0, 3)], [0.4, 0.6])),
            ("gamma via make_distribution", Stats.MakeDistribution(Stats.Gamma(2.5))),
        };
        foreach (var (name, d) in cases)
        {
            var draws = d.Rvs(4000, rng);
            var ks = Stats.Kstest(draws, d);
            Assert.True(ks.Pvalue > 1e-3, $"{name}: KS p = {ks.Pvalue}");
        }
        var b = new Binomial(30, 0.25).Sample(20000, rng);
        Near(7.5, b.Average(), 0.06, "binomial sample mean");
        Assert.All(b, v => Assert.Equal(Math.Floor(v), v));
    }

    [Fact]
    public void QmcSamplingIsLowDiscrepancy()
    {
        var x = new StandardNormal();
        var s = x.Sample([256], new SobolEngine(1, scramble: false));
        Assert.Equal([256], s.Shape);
        // An unscrambled Sobol' sequence of 256 points puts exactly one point in each 1/256 cell,
        // so the sample's CDF values are stratified.
        var u = s.ToArray().Select(v => x.Cdf(v)).Order().ToArray();
        for (var i = 0; i < u.Length; i++) Assert.InRange(u[i], i / 256.0 - 1e-12, (i + 1) / 256.0 + 1e-12);
        var two = x.Sample([64, 3], new SobolEngine(3, seed: 1));
        Assert.Equal([64, 3], two.Shape);
        Assert.Throws<ArgumentException>(() => x.Sample([64, 3], new SobolEngine(2)));
    }

    [Fact]
    public void KurtosisConventionsAndInterfaceView()
    {
        var x = new Logistic();
        Close(4.2, x.Kurtosis(), 1e-14);
        Close(1.2, x.Kurtosis(KurtosisConvention.Excess), 1e-13);
        IUnivariateDistribution view = x;
        Close(1.2, view.Kurtosis(), 1e-13, "the interface reports excess kurtosis like rv_continuous");
        Close(Math.PI * Math.PI / 3, view.Var(), 1e-14);
        var (lo, hi) = view.Interval(0.9);
        Close(x.Icdf(0.05), lo, 1e-15);
        Close(x.Iccdf(0.05), hi, 1e-15);
    }

    [Fact]
    public void MakeDistributionKeepsLocAndScale()
    {
        var x = Stats.MakeDistribution(Stats.Gamma(2.5, loc: 1.0, scale: 3.0));
        var same = 3.0 * Stats.MakeDistribution(Stats.Gamma(2.5)) + 1.0;
        foreach (var t in new[] { 1.5, 4.0, 20.0 }) Close(same.Pdf(t), x.Pdf(t), 1e-14);
        Close(same.Moment(5, MomentKind.Central), x.Moment(5, MomentKind.Central), 1e-9);
        Assert.Equal("Gamma", x.Name);
        Assert.Equal("Poisson", Stats.MakeDistribution(Stats.Poisson(2)).Name);
    }

    [Fact]
    public void TruncatedTailQuantilesKeepAllTheirDigits()
    {
        // Far-tail truncation: scipy's icdf keeps nine digits here; inverting through the survival
        // function keeps them all (reference: mpmath, 40 digits).
        var x = Stats.Truncate(new StandardNormal(), lb: 8.0);
        Close(8.0, x.Icdf(1e-300 + 1e-17), 1e-15);
        var y = Stats.Truncate(new StandardNormal(), ub: -8.0);
        Close(-x.Icdf(0.3), y.Iccdf(0.3), 1e-15, "the mirror image");
        Close(8.0438016529280543, x.Icdf(0.3), 1e-14, "mpmath");
    }
}
