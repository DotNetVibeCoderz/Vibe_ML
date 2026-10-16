using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsTestsAssert;
using R = Gravicode.Science.Tests.SciNet.StatsTestsRef;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Descriptive statistics against scipy 1.17.1.</summary>
public class StatsDescriptiveTests
{
    private const double Rt = 1e-12;

    [Fact]
    public void MomentsMatchScipy()
    {
        for (var o = 0; o <= 5; o++) Close($"moment/{o}", Rt, 1e-14, Stats.Moment(R.SK, o));
        Close("moment_center", Rt, 1e-14, Stats.Moment(R.SK, 3, 2.0));
        foreach (var b in new[] { true, false })
        {
            var bn = b ? "True" : "False";
            Close($"skew/{bn}", Rt, 1e-14, Stats.Skew(R.SK, b));
            foreach (var f in new[] { true, false })
                Close($"kurtosis/{(f ? "True" : "False")}/{bn}", Rt, 1e-14, Stats.Kurtosis(R.SK, f, b));
        }
        var d = Stats.Describe(R.SK);
        Close("describe", Rt, 1e-14, d.Nobs, d.Min, d.Max, d.Mean, d.Variance, d.Skewness, d.Kurtosis);
        d = Stats.Describe(R.SK, ddof: 0, bias: false);
        Close("describe_unbiased", Rt, 1e-14, d.Variance, d.Skewness, d.Kurtosis);
        Close("variation", Rt, 1e-14, Stats.Variation(R.SK), Stats.Variation(R.SK, 1));
        Assert.True(double.IsNaN(Stats.Skew([2.0, 2.0, 2.0])), "constant data has undefined skew");
    }

    [Fact]
    public void MeansMatchScipy()
    {
        Close("gmean", Rt, 0, Stats.Gmean(R.POS), Stats.Gmean(R.POS, R.WTS));
        Close("hmean", Rt, 0, Stats.Hmean(R.POS), Stats.Hmean(R.POS, R.WTS));
        foreach (var p in new[] { -2.0, 0.5, 3.0 })
            Close($"pmean/{p.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture)}", Rt, 0, Stats.Pmean(R.POS, p), Stats.Pmean(R.POS, p, R.WTS));
        Close("tmean", Rt, 0, Stats.Tmean(R.SK, (1.0, 6.0)), Stats.Tmean(R.SK, (1.0, 6.0), (false, true)));
        Close("tvar", Rt, 0, Stats.Tvar(R.SK, (1.0, 6.0)), Stats.Tstd(R.SK, (1.0, 6.0)), Stats.Tsem(R.SK, (1.0, 6.0)));
        Close("tminmax", 0, 0, Stats.Tmin(R.SK, 1.0), Stats.Tmin(R.SK, 1.0, false), Stats.Tmax(R.SK, 6.0), Stats.Tmax(R.SK));
        Close("trim_mean", Rt, 0, Stats.TrimMean(R.SK, 0.1), Stats.TrimMean(R.SK, 0.25));
        Close("trimboth", 0, 0, Stats.Trimboth(R.SK, 0.1));
        Close("trim1", 0, 0, [.. Stats.Trim1(R.SK, 0.1), .. Stats.Trim1(R.SK, 0.1, "left")]);
    }

    [Fact]
    public void ScalesAndStandardisationMatchScipy()
    {
        Close("sem", Rt, 0, Stats.Sem(R.SK), Stats.Sem(R.SK, 0));
        Close("zscore", Rt, 1e-14, Stats.Zscore(R.X));
        Close("zscore1", Rt, 1e-14, Stats.Zscore(R.X, 1));
        Close("zmap", Rt, 1e-14, Stats.Zmap(R.Y[..5], R.X));
        Close("gzscore", Rt, 1e-14, Stats.Gzscore(R.POS[..10]));
        Close("iqr", Rt, 0, Stats.Iqr(R.SK), Stats.Iqr(R.SK, (10, 80)), Stats.Iqr(R.SK, scale: Stats.NormalIqrScale));
        Close("mad", Rt, 0, Stats.MedianAbsDeviation(R.SK), Stats.MedianAbsDeviation(R.SK, Stats.NormalMadScale));
        var mo = Stats.Mode(R.T1);
        Close("mode", 0, 0, mo.Mode, mo.Count);
    }

    [Fact]
    public void RanksAndPercentilesMatchScipy()
    {
        foreach (var m in new[] { "average", "min", "max", "dense", "ordinal" })
            Close($"rankdata/{m}", 0, 0, Stats.Rankdata(R.T1, m));
        foreach (var k in new[] { "rank", "weak", "strict", "mean" })
            Close($"percentileofscore/{k}", 1e-14, 0, Stats.Percentileofscore(R.T1, 5.0, k), Stats.Percentileofscore(R.T1, 5.5, k));
        foreach (var im in new[] { "fraction", "lower", "higher" })
            Close($"scoreatpercentile/{im}", 1e-14, 0, Stats.Scoreatpercentile(R.SK, 37, null, im), Stats.Scoreatpercentile(R.SK, 50, (1, 5), im));
        double[] ps = [0.0, 0.05, 0.25, 0.5, 0.62, 0.9, 1.0];
        foreach (var q in new[] { "inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull",
                     "linear", "median_unbiased", "normal_unbiased", "harrell-davis", "round_outward", "round_inward", "round_nearest" })
            Close($"quantile/{q}", 1e-12, 1e-14, Stats.Quantile(R.SK, ps, q));
    }

    [Fact]
    public void HistogramsAndBinnedStatisticsMatchScipy()
    {
        var c = Stats.Cumfreq(R.SK, 8);
        Close("cumfreq", 1e-13, 1e-14, [.. c.Frequency, c.Lowerlimit, c.Binsize, c.Extrapoints]);
        var r = Stats.Relfreq(R.SK, 6, (0, 10));
        Close("relfreq", 1e-13, 1e-14, [.. r.Frequency, r.Lowerlimit, r.Binsize, r.Extrapoints]);
        var w = Stats.Cumfreq(R.SK, 5, weights: [.. R.POS, .. R.POS[..10]]);
        Close("cumfreq_w", 1e-12, 1e-12, w.Frequency);
        foreach (var st in new[] { "mean", "std", "median", "count", "sum", "min", "max" })
        {
            var b = Stats.BinnedStatistic(R.CX, R.CY, st, 6);
            Close($"binned/{st}", 1e-12, 1e-14, [.. Flat(b.Statistic), .. b.BinEdges[0], .. b.Binnumber.Select(v => (double)v)]);
        }
        var e = Stats.BinnedStatistic(R.CX, R.CY, "mean", [-2.0, -1, 0, 0.5, 2]);
        Close("binned_edges", 1e-12, 1e-14, [.. Flat(e.Statistic), .. e.Binnumber.Select(v => (double)v)]);
        var cu = Stats.BinnedStatistic(R.CX, R.CY, v => v.Max() - v.Min(), 5, (-1.5, 1.5));
        Close("binned_custom", 1e-12, 1e-14, [.. Flat(cu.Statistic), .. cu.Binnumber.Select(v => (double)v)]);
        var mat = R.MAT;
        var third = Enumerable.Range(0, 40).Select(i => mat[i * 3 + 2]).ToArray();
        var b2 = Stats.BinnedStatistic2d(R.CX, R.CY, third, "mean", (3, 4));
        Close("binned2d", 1e-12, 1e-14, [.. Flat(b2.Statistic), .. b2.Binnumber.Select(v => (double)v)]);
        var bd = Stats.BinnedStatisticDd([R.CX, R.CY, third], R.CY, "count", [2, 3, 2]);
        Close("binneddd", 0, 0, [.. Flat(bd.Statistic), .. bd.Binnumber.Select(v => (double)v)]);
    }

    [Fact]
    public void DistancesAndEntropyMatchScipy()
    {
        Close("wasserstein", 1e-12, 1e-15, Stats.WassersteinDistance(R.X, R.Y), Stats.WassersteinDistance(R.X[..10], R.Y[..10], R.WTS[..10], R.POS[..10]));
        Close("energy", 1e-12, 1e-15, Stats.EnergyDistance(R.X, R.Y), Stats.EnergyDistance(R.X[..10], R.Y[..10], R.WTS[..10], R.POS[..10]));
        Close("entropy", 1e-13, 1e-15, Stats.Entropy(R.PK), Stats.Entropy(R.PK, logBase: 2), Stats.Entropy(R.PK, R.QK), Stats.Entropy([1.0, 2, 3, 0]));
        foreach (var m in new[] { "vasicek", "van es", "ebrahimi", "correa", "auto" })
            Close($"diffent/{m}", 1e-12, 1e-14,
                Stats.DifferentialEntropy(R.SK, method: m), Stats.DifferentialEntropy(R.SK[..9], method: m), Stats.DifferentialEntropy(R.SK, 3, 2, m));
    }

    [Fact]
    public void MiscellaneousMatchScipy()
    {
        var s = Stats.Sigmaclip([.. R.X, 8.0, -9.0, 15.0], 3, 3);
        Close("sigmaclip", 1e-13, 1e-14, [.. StatsSorted(s.Clipped), s.Lower, s.Upper]);
        var o = Stats.Obrientransform(R.X[..8], R.Y[..9]);
        Close("obrien", 1e-12, 1e-14, [.. o[0], .. o[1]]);
        Close("lmoment", 1e-12, 1e-14, [.. Stats.Lmoment(R.SK), .. Stats.Lmoment(R.SK, standardize: false), .. Stats.Lmoment(R.SK, 6)]);
        Close("expectile", 1e-11, 1e-14, Stats.Expectile(R.SK, 0.2), Stats.Expectile(R.SK, 0.5), Stats.Expectile(R.SK, 0.9), Stats.Expectile(R.SK[..50], 0.7, R.WTS));
    }

    [Fact]
    public void AxisOverloadsMatchScipy()
    {
        var m = new NdArray(R.M2, 5, 7);
        Close("axis/skew0", 1e-12, 1e-14, Flat(Stats.Skew(m, 0)));
        Close("axis/skew1", 1e-12, 1e-14, Flat(Stats.Kurtosis(m, 1, bias: false)));
        Close("axis/zscore1", 1e-12, 1e-14, Flat(Stats.Zscore(m, 1)));
        Close("axis/sem0", 1e-12, 1e-14, Flat(Stats.Sem(m, 0)));
        Close("axis/rank1", 0, 0, Flat(Stats.Rankdata(m, 1)));
        Close("axis/iqr1", 1e-12, 1e-14, Flat(Stats.Iqr(m, 1)));
        var abs = new NdArray(R.M2.Select(Math.Abs).ToArray(), 5, 7);
        Close("axis/entropy0", 1e-12, 1e-14, Flat(Stats.Entropy(abs, 0)));
    }

    [Fact]
    public void AxisReductionRunsInParallelAndAgreesWithLoop()
    {
        var rng = new GraviRandom(7);
        var big = rng.Normal(0, 1, 300, 400);
        var viaAxis = Flat(Stats.Skew(big, 1));
        var data = Flat(big);
        for (var i = 0; i < 300; i++)
            Near(Stats.Skew(data.AsSpan(i * 400, 400).ToArray()), viaAxis[i], 0, 0, $"row {i}");
    }

    private static double[] StatsSorted(double[] a)
    {
        var c = (double[])a.Clone();
        Array.Sort(c);
        return c;
    }
}
