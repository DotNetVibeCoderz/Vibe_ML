using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Box–Cox/Yeo–Johnson, circular statistics, Mood's median test and small morestats helpers against scipy 1.17.1.</summary>
public class StatsMorestatsTests
{
    private static readonly double[] X = [0.8, 1.3, 2.7, 3.1, 0.45, 5.6, 2.2, 1.9, 7.4, 0.95, 3.3, 4.1, 1.15, 2.05, 6.2];
    private static readonly double[] Z = [-2.1, 0.4, 1.7, -0.3, 3.9, 0.0, -1.2, 2.6, 5.1, -0.8, 1.1, 0.25];

    private static void Close(double e, double a, double tol, string w)
        => Assert.True(Math.Abs(e - a) <= tol * Math.Max(1, Math.Abs(e)), $"{w}: expected {e:R}, got {a:R}");

    [Fact]
    public void Boxcox_MatchesScipy()
    {
        // λ comes from Brent's method in both libraries (scipy's tol 1.48e-8), so λ agrees to ~1e-8.
        var (y, l) = Stats.Boxcox(X);
        Close(0.1664279161325853, l, 1e-7, "lambda");
        Close(-0.21905089482644735, y[0], 1e-7, "y0");
        Close(2.1318709385188273, y[^1], 1e-7, "y-1");
        var (_, _, ci) = Stats.BoxcoxInterval(X, alpha: 0.05);
        Close(-0.4758465069650796, ci.Lower, 1e-7, "ci lo");
        Close(0.8512810865413347, ci.Upper, 1e-7, "ci hi");
        Close(0.1789116044139873, Stats.BoxcoxNormmax(X), 1e-7, "pearsonr");
        Close(0.1664279161325853, Stats.BoxcoxNormmax(X, method: BoxcoxNormmaxMethod.Mle), 1e-7, "mle");
        var all = Stats.BoxcoxNormmaxAll(X);
        Close(0.1789116044139873, all.Pearsonr, 1e-7, "all p");
        Close(-7.960287508865742, Stats.BoxcoxLlf(0.3, X), 1e-13, "llf 0.3");
        Close(-8.007404996372014, Stats.BoxcoxLlf(0, X), 1e-13, "llf 0");
        Close(-75.06644363703752, Stats.BoxcoxLlf(-4.5, X), 1e-12, "llf -4.5");
        Assert.Throws<ArgumentException>(() => Stats.Boxcox([1.0, -2.0, 3.0]));
    }

    [Fact]
    public void BoxcoxNormmax_ConstrainsToYmax()
    {
        // With a small ymax the optimum is replaced by the λ at which the extreme datum maps to ymax.
        // scipy 1.17.1: boxcox_normmax(x, method='mle', ymax=2.1) = 0.047636450256621554.
        var l = Stats.BoxcoxNormmax(X, method: BoxcoxNormmaxMethod.Mle, ymax: 2.1);
        Close(0.047636450256621554, l, 1e-10, "lambda");
        Close(2.1, Special.Boxcox(7.4, l), 1e-12, "constrained");
    }

    [Fact]
    public void Yeojohnson_MatchesScipy()
    {
        var (y, l) = Stats.Yeojohnson(Z);
        Close(0.6371742375297597, l, 1e-7, "lambda");
        Close(-2.695487535244656, y[0], 1e-7, "y0");
        Close(2.750900891897068, y[4], 1e-7, "y4");
        Close(-7.706193516894919, Stats.YeojohnsonLlf(0.7, Z), 1e-13, "llf");
        Close(-16.6573598686829, Stats.YeojohnsonLlf(2, Z), 1e-13, "llf 2");
        Close(0.6371742399498834, Stats.YeojohnsonNormmax(Z, [-2, 2]), 1e-7, "brack");
        Close(-0.17371413349456563, Stats.YeojohnsonNormmax(X), 1e-7, "positive data");
        Assert.Equal(1.0, Stats.YeojohnsonNormmax([0.0, 0.0]));
    }

    [Fact]
    public void CircularStatistics_MatchScipy()
    {
        double[] ang = [0.1, 6.2, 0.3, 5.9, 1.2, 0.05, 6.0];
        Close(0.10309152400590599, Stats.Circmean(ang), 1e-13, "circmean");
        Close(0.11018441267513057, Stats.Circvar(ang), 1e-13, "circvar");
        Close(0.4831998406019636, Stats.Circstd(ang), 1e-13, "circstd");
        Close(0.4831998406019636, Stats.Circstd(ang, normalize: true), 1e-13, "circstd normalized");
        double[] deg = [350, 10, 20, 355, 340, 30];
        Close(4.134520862853874, Stats.Circmean(deg, 360), 1e-12, "deg mean");
        Close(0.045646175965197044, Stats.Circvar(deg, 360), 1e-12, "deg var");
        Close(17.51429765749215, Stats.Circstd(deg, 360), 1e-12, "deg std");
        Close(-180.0, Stats.Circmean([-170, 170, 180], 180, -180), 1e-12, "wrap");
    }

    [Fact]
    public void MedianTest_MatchesScipy()
    {
        double[] a = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49];
        double[] b = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99];
        double[] c = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84];
        var r = Stats.MedianTest([a, b, c]);
        Close(4.141505553270259, r.Statistic, 1e-12, "stat");
        Close(0.12609082774093244, r.Pvalue, 1e-12, "p");
        Assert.Equal(34.0, r.Median);
        Assert.Equal(new[,] { { 5, 10, 7 }, { 11, 5, 10 } }, r.Table);
        var ab = Stats.MedianTest([a, b, c], MedianTestTies.Above, lambda: 0); // lambda_='log-likelihood'
        Close(5.678406103713711, ab.Statistic, 1e-12, "above stat");
        Close(0.05847224674927279, ab.Pvalue, 1e-12, "above p");
        var ig = Stats.MedianTest([a, b], MedianTestTies.Ignore);
        Close(3.3482142857142856, ig.Statistic, 1e-12, "ignore stat");
        Close(0.06727796053834906, ig.Pvalue, 1e-12, "ignore p");
        Assert.Equal(new[,] { { 5, 10 }, { 11, 4 } }, ig.Table);
        var nc = Stats.MedianTest([a, b], correction: false);
        Close(3.888454861111112, nc.Statistic, 1e-12, "nocorr stat");
        Close(0.04861913422927604, nc.Pvalue, 1e-12, "nocorr p");
    }

    [Fact]
    public void SmallHelpers_MatchScipy()
    {
        Close(2.239443655490851, Stats.Gstd(X), 1e-13, "gstd");
        double[] k = [0.8874999999999998, 4.493693181818182, 7.019350568181816, -0.15688144886364097];
        for (var n = 1; n <= 4; n++) Assert.True(Math.Abs(k[n - 1] - Stats.Kstat(Z, n)) < 1e-12, $"kstat {n}");
        Close(0.37447443181818185, Stats.Kstatvar(Z, 1), 1e-13, "kstatvar 1");
        Close(3.095596063834367, Stats.Kstatvar(Z, 2), 1e-12, "kstatvar 2");
        Close(0.9107142857142857, Stats.Tiecorrect([1, 2.5, 2.5, 5, 5, 5, 7]), 1e-15, "tiecorrect");

        var (m, v, s) = Stats.BayesMvs(X, 0.9);
        Close(2.88, m.Statistic, 1e-13, "mean");
        Close(1.9217899342685516, m.Minmax.Lower, 1e-10, "mean lo");
        Close(3.8382100657314475, m.Minmax.Upper, 1e-10, "mean hi");
        Close(5.1795, v.Statistic, 1e-12, "var");
        Close(2.624215649613822, v.Minmax.Lower, 1e-10, "var lo");
        Close(9.459364917858963, v.Minmax.Upper, 1e-10, "var hi");
        Close(2.228981702917002, s.Statistic, 1e-12, "std");
        Close(1.6199431007334244, s.Minmax.Lower, 1e-10, "std lo");
        Close(3.075608056605874, s.Minmax.Upper, 1e-10, "std hi");

        var pp = Stats.Probplot(X);
        Close(-1.6937289945417815, pp.Osm[0], 1e-13, "osm0");
        Assert.Equal(0.0, pp.Osm[7], 1e-15);
        Close(2.142772964852169, pp.Slope, 1e-12, "slope");
        Close(2.8800000000000003, pp.Intercept, 1e-12, "intercept");
        Close(0.9535800824111831, pp.R, 1e-12, "r");
        var pe = Stats.Probplot(X, new ExponentialDistribution());
        Close(0.04620981203732974, pe.Osm[0], 1e-13, "expon osm0");
        Close(2.4089380960041407, pe.Slope, 1e-12, "expon slope");
        Close(0.9889163499700967, pe.R, 1e-12, "expon r");

        var d = Stats.DirectionalStats(new NdArray([1, 0, 0, 0.8, 0.6, 0, 0.9, 0, 0.3, 2, 0.5, 0.5], 4, 3));
        double[] md = [0.9651120095928093, 0.21848786716752253, 0.14429781994304702];
        for (var i = 0; i < 3; i++) Close(md[i], d.MeanDirection[i], 1e-13, $"dir[{i}]");
        Close(0.9562341735831409, d.MeanResultantLength, 1e-13, "mrl");
    }

    [Fact]
    public void Anderson_WeibullMinMatchesScipy()
    {
        // scipy 1.17.1: anderson(x, 'weibull_min') on weibull_min.rvs(2.5, loc=1, scale=3, size=60,
        // random_state=default_rng(1)) rounded to 6 decimals.
        double[] x =
        [
            3.626316, 5.65867, 2.425623, 5.636284, 3.023674, 3.362749, 4.759945, 3.320635, 3.740542, 1.717214, 4.432637, 3.705707,
            3.079595, 4.577762, 2.996384, 3.45244, 2.381554, 3.302451, 2.659176, 2.863827, 4.420159, 2.923258, 3.5467, 6.196867,
            5.813618, 4.321902, 3.715068, 2.911807, 2.494395, 5.953938, 3.63907, 2.298045, 3.971976, 4.52745, 3.938257, 5.322907,
            1.831123, 3.67679, 3.469785, 2.00142, 4.030189, 4.890211, 3.874656, 2.856455, 4.821874, 3.619339, 3.623517, 4.430717,
            2.441628, 4.720459, 4.172225, 4.571973, 2.615263, 4.639735, 2.614164, 2.119523, 4.904604, 4.938914, 5.030236, 3.507162,
        ];
        var r = Stats.Anderson(x, "weibull_min");
        Close(0.20419350325291674, r.Statistic, 1e-6, "A2");
        Assert.Equal(new[] { 0.329, 0.454, 0.536, 0.603, 0.717, 0.831, 0.983, 1.099 }, r.CriticalValues);
        Assert.Equal(new[] { 0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995 }, r.SignificanceLevel);
        Close(2.4206494717241727, r.FitShape!.Value, 1e-6, "shape");
        Close(1.3318254209902816, r.FitLocation, 1e-6, "loc");
        Close(2.7494791409488712, r.FitScale, 1e-6, "scale");
        Assert.Equal(0.5, r.Pvalue);
    }
}
