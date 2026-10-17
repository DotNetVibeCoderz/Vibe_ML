using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsExtraAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>CensoredData, ecdf (Kaplan–Meier), logrank and censored maximum likelihood. Values from scipy 1.17.1.</summary>
public class StatsSurvivalTests
{
    private static readonly CensoredData Fanbelts = new(uncensored: [77, 47, 81, 56, 80], right: [62, 60, 43, 71, 37]);

    [Fact]
    public void CensoredDataNormalisesIntervalRows()
    {
        var d = CensoredData.IntervalCensored([1, double.NegativeInfinity, 3, 4], [1, 2, double.PositiveInfinity, 5]);
        Assert.Equal([1.0], d.Uncensored);
        Assert.Equal([2.0], d.Left);
        Assert.Equal([3.0], d.Right);
        Assert.Equal(1, d.Interval.GetLength(0));
        Assert.Equal(4, d.Count);
        Assert.Equal(3, d.NumCensored());
        var rc = CensoredData.RightCensored([1, 2, 3], [false, true, false]);
        Assert.Equal([1.0, 3.0], rc.Uncensored);
        Assert.Equal([2.0], rc.Right);
        Assert.Throws<ArgumentException>(() => CensoredData.IntervalCensored([2], [1]));
        Assert.Throws<ArgumentException>(() => new CensoredData(uncensored: [double.NaN]));
    }

    [Fact]
    public void KaplanMeierMatchesScipy()
    {
        var r = Stats.Ecdf(Fanbelts);
        Assert.Equal([37.0, 43, 47, 56, 60, 62, 71, 77, 80, 81], r.Sf.Quantiles);
        All([1.0, 1.0, 0.875, 0.75, 0.75, 0.75, 0.75, 0.5, 0.25, 0.0], r.Sf.Probabilities, 1e-15, 1e-15, "km sf");
        var (lo, hi) = r.Sf.ConfidenceInterval();
        All([1.0, 1.0, 0.6458276962323382, 0.44994302022779314, 0.44994302022779314, 0.44994302022779314, 0.44994302022779314, 0.052701464070710746, 0.0, double.NaN], lo.Probabilities, 1e-13, 1e-15, "linear low");
        All([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9472985359292893, 0.6623888737682102, double.NaN], hi.Probabilities, 1e-13, 1e-15, "linear high");
        (lo, hi) = r.Sf.ConfidenceInterval(method: "log-log");
        All([double.NaN, double.NaN, 0.38700001403202516, 0.314807113705519, 0.314807113705519, 0.314807113705519, 0.314807113705519, 0.0804882114850773, 0.010499589866805993, double.NaN], lo.Probabilities, 1e-12, 1e-15, "loglog low");
        All([double.NaN, double.NaN, 0.981392965878966, 0.9308983170906275, 0.9308983170906275, 0.9308983170906275, 0.9308983170906275, 0.8263946341076416, 0.6558775085110888, double.NaN], hi.Probabilities, 1e-12, 1e-15, "loglog high");
        (lo, hi) = r.Cdf.ConfidenceInterval(0.9, "log-log");
        All([double.NaN, double.NaN, 0.025416930990815656, 0.08565391295001379, 0.08565391295001379, 0.08565391295001379, 0.08565391295001379, 0.20915534149096793, 0.3999274786948481, double.NaN], lo.Probabilities, 1e-12, 1e-15, "cdf loglog low");
        All([double.NaN, double.NaN, 0.49971244101072854, 0.6031620562921582, 0.6031620562921582, 0.6031620562921582, 0.6031620562921582, 0.8709425974514649, 0.9767873713880767, double.NaN], hi.Probabilities, 1e-12, 1e-15, "cdf loglog high");
        Assert.Throws<InvalidOperationException>(() => lo.ConfidenceInterval());
    }

    [Fact]
    public void UncensoredEcdfStepsAndEvaluates()
    {
        var r = Stats.Ecdf([6.23, 5.58, 7.06, 6.42, 5.20, 5.58]);
        Assert.Equal([5.2, 5.58, 6.23, 6.42, 7.06], r.Cdf.Quantiles);
        All([1 / 6.0, 0.5, 2 / 3.0, 5 / 6.0, 1.0], r.Cdf.Probabilities, 1e-15, 1e-15, "cdf");
        All([0.0, 0.5, 2 / 3.0, 1.0], r.Cdf.Evaluate([5, 5.58, 6.3, 8]), 1e-15, 1e-15, "evaluate");
        Near(1.0, r.Sf.Evaluate(-1e300), 0, 0, "sf left");
        var (lo, hi) = r.Cdf.ConfidenceInterval();
        All([0.0, 0.09992402697039082, 0.2894714219746094, 0.5351343093804739, double.NaN], lo.Probabilities, 1e-12, 1e-15, "low");
        All([0.4648656906195262, 0.9000759730296092, 1.0, 1.0, double.NaN], hi.Probabilities, 1e-12, 1e-15, "high");
        Assert.Throws<NotSupportedException>(() => Stats.Ecdf(new CensoredData(uncensored: [1, 2], left: [0.5])));
    }

    [Fact]
    public void LogrankMatchesScipy()
    {
        // Bland & Altman (2004) gliomas example from scipy's docstring.
        var x = new CensoredData(uncensored: [6, 13, 21, 30, 37, 38, 49, 50, 63, 79, 86, 98, 202, 219], right: [31, 47, 80, 82, 82, 149]);
        var y = new CensoredData(uncensored: [10, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 24, 25, 28, 30, 33, 35, 37, 40, 40, 46, 48, 76, 81, 82, 91, 112, 181], right: [34, 40, 70]);
        var p = new[] { 0.006181578637461755, 0.0030907893187308777, 0.9969092106812691 };
        var alts = new[] { Alternative.TwoSided, Alternative.Less, Alternative.Greater };
        for (var i = 0; i < 3; i++)
        {
            var r = Stats.Logrank(x, y, alts[i]);
            Near(-2.7379909000095908, r.Statistic, 1e-13, 0, "logrank stat");
            Near(p[i], r.Pvalue, 1e-11, 0, $"logrank p {alts[i]}");
        }
        var u = Stats.Logrank([1, 2, 3, 4, 5.5], [2.5, 6, 7, 8, 9]);
        Near(2.332588819771318, u.Statistic, 1e-13, 0, "uncensored stat");
        Near(0.01966973629197448, u.Pvalue, 1e-11, 0, "uncensored p");
    }

    [Fact]
    public void CensoredMaximumLikelihoodMatchesScipy()
    {
        var d = new CensoredData(uncensored: [1.2, 2.3, 0.7, 3.1, 1.9, 2.6, 0.4], right: [3.0, 3.0, 2.5], left: [0.3], interval: new double[,] { { 1, 1.5 }, { 2, 4 } });
        // Direct sum of log pdf / log sf / log cdf / log ΔF at N(1.5, 1.2).
        Near(20.866283940825525, Stats.Norm(1.5, 1.2).NegativeLogLikelihood(d), 1e-12, 0, "nnlf");
        // scipy 1.17.1 dist.fit(CensoredData) with fmin tightened to xtol 1e-12 (its default 1e-4 stops early).
        var n = Stats.Norm().FitTo(d);
        Near(2.07893026137051, n.Loc, 1e-6, 0, "norm loc");
        Near(1.3546164949732167, n.Scale, 1e-6, 0, "norm scale");
        var e = Stats.Expon().FitTo(d, floc: 0);
        Near(2.4956495523452795, e.Scale, 1e-6, 0, "expon scale");
        var w = Stats.WeibullMin(1).FitTo(d, floc: 0);
        Near(1.394718273866409, w.Shapes[0], 1e-6, 0, "weibull c");
        Near(2.460624879091939, w.Scale, 1e-6, 0, "weibull scale");
        var g = Stats.Gamma(1).FitTo(d, floc: 0);
        Near(1.4967067635214533, g.Shapes[0], 1e-6, 0, "gamma a");
        Near(1.5629753025575988, g.Scale, 1e-6, 0, "gamma scale");
        // No censored values: identical to the plain (closed-form) fit.
        var plain = Stats.Norm().FitTo([1.0, 2.0, 4.0]);
        var viaCensored = Stats.Norm().FitTo(new CensoredData(uncensored: [1.0, 2.0, 4.0]));
        Assert.Equal(plain.Loc, viaCensored.Loc);
        Assert.Equal(plain.Scale, viaCensored.Scale);
    }
}
