using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsTestsAssert;
using R = Gravicode.Science.Tests.SciNet.StatsTestsRef;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>KS, Cramér–von Mises, Anderson–Darling and normality tests against scipy 1.17.1.</summary>
public class StatsGoodnessOfFitTests
{
    private static readonly NormalDistribution Ref = Stats.Norm(0.1, 1.2);

    [Fact]
    public void KsOneSampleMatchesScipy()
    {
        foreach (var (name, alt) in Alternatives)
            foreach (var m in alt == Alternative.TwoSided ? new[] { "exact", "approx", "asymp" } : new[] { "exact" })
            {
                var r = Stats.Ks1samp(R.X, Ref, alt, m);
                Close($"ks_1samp/{m}/{name}", 1e-9, 1e-14, r.Statistic, r.Pvalue, r.StatisticLocation, r.StatisticSign);
            }
    }

    [Fact]
    public void KsTwoSampleMatchesScipy()
    {
        foreach (var (name, alt) in Alternatives)
        {
            foreach (var m in new[] { "exact", "asymp" })
            {
                var r = Stats.Ks2samp(R.X, R.Y, alt, m);
                Close($"ks_2samp/{m}/{name}", 1e-9, 1e-14, r.Statistic, r.Pvalue, r.StatisticLocation, r.StatisticSign);
                r = Stats.Ks2samp(R.X[..20], R.Z, alt, m);
                Close($"ks_2samp_eq/{m}/{name}", 1e-9, 1e-14, r.Statistic, r.Pvalue, r.StatisticLocation, r.StatisticSign);
                r = Stats.Ks2samp(R.L1, R.L2, alt, m);
                // KSTwo(171).Sf agrees with scipy to ~2e-7 relative at this size.
                Close($"ks_2samp_large/{m}/{name}", 1e-6, 1e-14, r.Statistic, r.Pvalue, r.StatisticLocation, r.StatisticSign);
            }
            var t = Stats.Kstest(R.T1, R.T2, alt);
            Close($"ks_2samp_ties/{name}", 1e-9, 1e-14, t.Statistic, t.Pvalue, t.StatisticLocation, t.StatisticSign);
        }
    }

    [Fact]
    public void CramerVonMisesMatchesScipy()
    {
        var r = Stats.CramerVonMises(R.X, Ref);
        Close("cvm", 1e-8, 1e-12, r.Statistic, r.Pvalue);
        r = Stats.CramerVonMises(R.L1, Stats.Norm());
        Close("cvm_large", 1e-8, 1e-12, r.Statistic, r.Pvalue);
        r = Stats.CramerVonMises(R.E1, Stats.Expon(scale: 2));
        Close("cvm_expon", 1e-8, 1e-12, r.Statistic, r.Pvalue);
        r = Stats.CramerVonMises2Samp(R.S1, R.S2);
        Close("cvm2_exact", 1e-10, 1e-14, r.Statistic, r.Pvalue);
        r = Stats.CramerVonMises2Samp(R.X[..15], R.Z[..12]);
        Close("cvm2_exact2", 1e-10, 1e-14, r.Statistic, r.Pvalue);
        r = Stats.CramerVonMises2Samp(R.X, R.Y);
        Close("cvm2_asym", 1e-8, 1e-12, r.Statistic, r.Pvalue);
        r = Stats.CramerVonMises2Samp(R.T1, R.T2, "asymptotic");
        Close("cvm2_ties", 1e-8, 1e-12, r.Statistic, r.Pvalue);
    }

    [Fact]
    public void AndersonMatchesScipy()
    {
        foreach (var (dist, data) in new[] { ("norm", R.SK), ("expon", R.E1), ("gumbel_r", R.SK), ("gumbel_l", R.X) })
        {
            var r = Stats.Anderson(data, dist);
            var expected = StatsTestsRef.Expected[$"anderson/{dist}"];
            var k = r.CriticalValues.Length;
            Near(expected[0], r.Statistic, 1e-6, 1e-9, $"{dist} A2");
            for (var i = 0; i < k; i++)
            {
                Near(expected[1 + i], r.CriticalValues[i], 0, 1e-12, $"{dist} crit");
                Near(expected[1 + k + i], r.SignificanceLevel[i], 0, 1e-12, $"{dist} sig");
            }
            Near(expected[^3], r.FitLocation, 1e-6, 1e-9, $"{dist} loc");
            Near(expected[^2], r.FitScale, 1e-6, 1e-9, $"{dist} scale");
            Near(expected[^1], r.Pvalue, 1e-5, 1e-9, $"{dist} p");
        }
        // scipy's logistic fit stops at fsolve's xtol = 1e-5; ours is solved to convergence.
        var lg = Stats.Anderson(R.L2[..80], "logistic");
        var le = StatsTestsRef.Expected["anderson/logistic"];
        Near(le[0], lg.Statistic, 1e-4, 0, "logistic A2");
    }

    [Fact]
    public void AndersonKsampMatchesScipy()
    {
        foreach (var v in new[] { "midrank", "right", "continuous" })
        {
            var r = Stats.AndersonKsamp([R.X, R.Y, R.Z], v);
            Close($"anderson_ksamp/{v}", 1e-9, 1e-12, r.Statistic, r.Pvalue);
            r = Stats.AndersonKsamp([R.T1, R.T2, R.T3], v);
            Close($"anderson_ksamp_ties/{v}", 1e-9, 1e-12, r.Statistic, r.Pvalue);
        }
        var c = Stats.AndersonKsamp([R.X, R.Y]);
        Close("anderson_ksamp_crit2", 1e-9, 1e-12, [c.Statistic, .. c.CriticalValues, c.Pvalue]);
    }

    [Fact]
    public void ShapiroMatchesScipy()
    {
        foreach (var (name, data) in new[] { ("x", R.X), ("y", R.Y), ("s1", R.S1), ("skewed", R.SK), ("l1", R.L1), ("e1", R.E1), ("t1", R.T1) })
        {
            var r = Stats.Shapiro(data);
            Close($"shapiro/{name}", 1e-6, 1e-9, r.Statistic, r.Pvalue);
        }
        foreach (var n in new[] { 3, 4, 5, 6, 11, 12 })
        {
            var r = Stats.Shapiro(R.Y[..n]);
            Close($"shapiro_n/{n}", 1e-6, 1e-9, r.Statistic, r.Pvalue);
        }
    }

    [Fact]
    public void MomentNormalityTestsMatchScipy()
    {
        foreach (var (name, alt) in Alternatives)
        {
            var s = Stats.Skewtest(R.SK, alt);
            Close($"skewtest/{name}", 1e-10, 1e-14, s.Statistic, s.Pvalue);
            var k = Stats.Kurtosistest(R.L2, alt);
            Close($"kurtosistest/{name}", 1e-10, 1e-14, k.Statistic, k.Pvalue);
        }
        var n = Stats.Normaltest(R.SK);
        Close("normaltest", 1e-10, 1e-14, n.Statistic, n.Pvalue);
        n = Stats.Normaltest(R.X);
        Close("normaltest_x", 1e-10, 1e-14, n.Statistic, n.Pvalue);
        var j = Stats.JarqueBera(R.SK);
        Close("jarque_bera", 1e-10, 1e-14, j.Statistic, j.Pvalue);
    }

    [Fact]
    public void EppsSingletonMatchesScipy()
    {
        var r = Stats.EppsSingleton2samp(R.X, R.Y);
        Close("epps", 1e-8, 1e-12, r.Statistic, r.Pvalue);
        r = Stats.EppsSingleton2samp(R.L1, R.L2, [0.3, 0.6, 0.9]);
        Close("epps_large", 1e-8, 1e-12, r.Statistic, r.Pvalue);
    }
}
