using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsTestsAssert;
using R = Gravicode.Science.Tests.SciNet.StatsTestsRef;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Scale tests, multiple comparisons and correlation against scipy 1.17.1.</summary>
public class StatsScaleAndCorrelationTests
{
    [Fact]
    public void VarianceTestsMatchScipy()
    {
        foreach (var c in new[] { "median", "mean", "trimmed" })
        {
            var l = Stats.Levene([R.X, R.Y, R.Z], c, 0.1);
            Close($"levene/{c}", 1e-10, 1e-14, l.Statistic, l.Pvalue);
            var f = Stats.Fligner([R.X, R.Y, R.Z], c, 0.1);
            Close($"fligner/{c}", 1e-10, 1e-14, f.Statistic, f.Pvalue);
        }
        var r = Stats.Levene(R.T1, R.T2, R.T3);
        Close("levene_ties", 1e-10, 1e-14, r.Statistic, r.Pvalue);
        r = Stats.Bartlett(R.X, R.Y, R.Z);
        Close("bartlett", 1e-10, 1e-14, r.Statistic, r.Pvalue);
        r = Stats.Fligner(R.T1, R.T2, R.T3);
        Close("fligner_ties", 1e-10, 1e-14, r.Statistic, r.Pvalue);
    }

    [Fact]
    public void MoodAndAnsariMatchScipy()
    {
        foreach (var (name, alt) in Alternatives)
        {
            var r = Stats.Mood(R.X, R.Y, alt);
            Close($"mood/{name}", 1e-10, 1e-14, r.Statistic, r.Pvalue);
            r = Stats.Mood(R.T1, R.T2, alt);
            Close($"mood_ties/{name}", 1e-10, 1e-14, r.Statistic, r.Pvalue);
            // scipy's exact Ansari–Bradley frequencies (gscale) are float32, so its p-values are only
            // good to ~3e-8 relative; ours are exact (see AnsariExactAgreesWithEnumeration).
            r = Stats.Ansari(R.X, R.Y, alt);
            Close($"ansari/{name}", 1e-7, 1e-14, r.Statistic, r.Pvalue);
            r = Stats.Ansari(R.S1, R.S2, alt);
            Close($"ansari_small/{name}", 1e-7, 1e-14, r.Statistic, r.Pvalue);
            r = Stats.Ansari(R.T1, R.T2, alt);
            Close($"ansari_ties/{name}", 1e-10, 1e-14, r.Statistic, r.Pvalue);
            r = Stats.Ansari(R.L1[..60], R.L2[..70], alt);
            Close($"ansari_large/{name}", 1e-10, 1e-14, r.Statistic, r.Pvalue);
        }
    }

    [Fact]
    public void AnsariExactAgreesWithEnumeration()
    {
        // All C(9, 4) = 126 ways to place the test sample among N = 9 ranks.
        double[] x = [0.3, -1.2, 2.5, 0.9];
        double[] y = [0.1, -0.4, 1.7, -2.2, 0.6];
        var res = Stats.Ansari(x, y, Alternative.Less);
        var scores = Enumerable.Range(1, 9).Select(i => Math.Min(i, 10 - i)).ToArray();
        int ge = 0, total = 0;
        for (var mask = 0; mask < 1 << 9; mask++)
        {
            if (System.Numerics.BitOperations.PopCount((uint)mask) != 4) continue;
            var s = 0;
            for (var i = 0; i < 9; i++) if (((mask >> i) & 1) == 1) s += scores[i];
            total++;
            if (s >= res.Statistic) ge++;
        }
        Near((double)ge / total, res.Pvalue, 1e-14, 0, "P(AB >= ab)");
    }

    [Fact]
    public void TukeyHsdMatchesScipy()
    {
        var r = Stats.TukeyHsd(R.X, R.Y, R.Z);
        var (lo, hi) = r.ConfidenceInterval();
        Close("tukey", 1e-7, 1e-10, [.. Flat(r.Statistic), .. Flat(r.Pvalue), .. Flat(lo), .. Flat(hi)]);
        r = Stats.TukeyHsd(R.X[..20], R.Y[..20], R.Z);
        Close("tukey_equal_n", 1e-7, 1e-10, [.. Flat(r.Statistic), .. Flat(r.Pvalue)]);
        r = Stats.TukeyHsd([R.X, R.Y, R.Z], equalVar: false);
        (lo, hi) = r.ConfidenceInterval(0.9);
        Close("games_howell", 1e-7, 1e-10, [.. Flat(r.Statistic), .. Flat(r.Pvalue), .. Flat(lo), .. Flat(hi)]);
    }

    [Fact]
    public void DunnettMatchesScipy()
    {
        foreach (var (name, alt) in Alternatives)
        {
            var r = Stats.Dunnett([R.X, R.Y], R.Z, alt);
            var e = StatsTestsRef.Expected[$"dunnett/{name}"];
            Near(e[0], r.Statistic[0], 1e-12, 1e-14, "stat 0");
            Near(e[1], r.Statistic[1], 1e-12, 1e-14, "stat 1");
            // scipy's p-values come from randomised Genz integration (~1e-4); ours are deterministic.
            Near(e[2], r.Pvalue[0], 2e-3, 2e-4, $"{name} p 0");
            Near(e[3], r.Pvalue[1], 2e-3, 2e-4, $"{name} p 1");
            var (lo, hi) = r.ConfidenceInterval();
            for (var i = 0; i < 2; i++)
            {
                Near(e[4 + i], lo[i], 5e-3, 5e-3, $"{name} low {i}");
                Near(e[6 + i], hi[i], 5e-3, 5e-3, $"{name} high {i}");
            }
        }
        // With one treatment Dunnett reduces to the pooled two-sample t-test (df = n1 + n0 − 2).
        var one = Stats.Dunnett([R.X], R.Z);
        var t = Stats.TtestInd(R.X, R.Z);
        Near(t.Statistic, one.Statistic[0], 1e-12, 0, "t statistic");
        Near(t.Pvalue, one.Pvalue[0], 1e-8, 1e-12, "t p-value");
    }

    [Fact]
    public void FdrAndCombinationMatchScipy()
    {
        Close("fdr/bh", 1e-14, 0, Stats.FalseDiscoveryControl(R.PS));
        Close("fdr/by", 1e-14, 0, Stats.FalseDiscoveryControl(R.PS, "by"));
        foreach (var m in new[] { "fisher", "pearson", "tippett", "stouffer", "mudholkar_george" })
        {
            var c = Stats.CombinePvalues(R.CP, m);
            Close($"combine/{m}", 1e-12, 1e-15, c.Statistic, c.Pvalue);
        }
        var w = Stats.CombinePvalues(R.CP, "stouffer", R.CW);
        Close("combine/stouffer_w", 1e-12, 1e-15, w.Statistic, w.Pvalue);
    }

    [Fact]
    public void PageTrendMatchesScipy()
    {
        var pg = new NdArray(R.PG, 6, 4);
        var r = Stats.PageTrendTest(pg);
        Assert.Equal("exact", r.Method);
        Close("page/exact", 1e-12, 1e-15, r.Statistic, r.Pvalue);
        r = Stats.PageTrendTest(pg, method: "asymptotic");
        Close("page/asym", 1e-12, 1e-15, r.Statistic, r.Pvalue);
        r = Stats.PageTrendTest(new NdArray(R.PG2, 15, 5));
        Assert.Equal("asymptotic", r.Method);
        Close("page/auto_asym", 1e-12, 1e-15, r.Statistic, r.Pvalue);
        r = Stats.PageTrendTest(pg, predictedRanks: [2, 1, 3, 4]);
        Close("page/pred", 1e-12, 1e-15, r.Statistic, r.Pvalue);
    }

    [Fact]
    public void CorrelationMatchesScipy()
    {
        var t1 = R.T1[..12];
        foreach (var (name, alt) in Alternatives)
        {
            var p = Stats.Pearsonr(R.CX, R.CY, alt);
            var ci = p.ConfidenceInterval(0.9);
            Close($"pearsonr/{name}", 1e-11, 1e-15, p.Statistic, p.Pvalue, ci.Low, ci.High);
            var s = Stats.Spearmanr(R.CX, R.CY, alt);
            Close($"spearmanr/{name}", 1e-11, 1e-15, s.Statistic, s.Pvalue);
            s = Stats.Spearmanr(t1, R.T3, alt);
            Close($"spearmanr_ties/{name}", 1e-11, 1e-15, s.Statistic, s.Pvalue);
            var k = Stats.Kendalltau(R.CX, R.CY, alternative: alt);
            Close($"kendall/{name}", 1e-11, 1e-15, k.Statistic, k.Pvalue);
            k = Stats.Kendalltau(R.CX[..20], R.CY[..20], alternative: alt);
            Close($"kendall_exact/{name}", 1e-11, 1e-15, k.Statistic, k.Pvalue);
            k = Stats.Kendalltau(t1, R.T3, alternative: alt);
            Close($"kendall_ties/{name}", 1e-11, 1e-15, k.Statistic, k.Pvalue);
            k = Stats.Kendalltau(t1, R.T3, variant: "c", alternative: alt);
            Close($"kendall_c/{name}", 1e-11, 1e-15, k.Statistic, k.Pvalue);
            var lr = Stats.Linregress(R.CX, R.CY, alt);
            Close($"linregress/{name}", 1e-11, 1e-15, lr.Slope, lr.Intercept, lr.Rvalue, lr.Pvalue, lr.Stderr, lr.InterceptStderr);
            var sd = Stats.Somersd(t1, R.T3, alt);
            Close($"somersd/{name}", 1e-11, 1e-15, sd.Statistic, sd.Pvalue);
        }
        var e = Stats.Kendalltau(R.CX, R.CY, method: "exact");
        Close("kendall_exact40", 1e-11, 1e-15, e.Statistic, e.Pvalue);
        var m = Stats.Spearmanr(new NdArray(R.MAT, 40, 3));
        Close("spearman_matrix", 1e-11, 1e-15, [.. Flat(m.Statistic), .. Flat(m.Pvalue)]);
        var pb = Stats.Pointbiserialr(R.BX, R.BY);
        Close("pointbiserial", 1e-11, 1e-15, pb.Statistic, pb.Pvalue);
        var st = Stats.Somersd(new NdArray(R.SDT, 3, 3));
        Close("somersd_table", 1e-11, 1e-15, st.Statistic, st.Pvalue);
    }

    [Fact]
    public void RobustRegressionMatchesScipy()
    {
        foreach (var m in new[] { "separate", "joint" })
        {
            var t = Stats.Theilslopes(R.CY, R.CX, method: m);
            Close($"theil/{m}", 1e-12, 1e-15, t.Slope, t.Intercept, t.LowSlope, t.HighSlope);
            t = Stats.Theilslopes(R.T3, R.T1[..12], 0.9, m);
            Close($"theil_ties/{m}", 1e-12, 1e-15, t.Slope, t.Intercept, t.LowSlope, t.HighSlope);
        }
        var n = Stats.Theilslopes(R.CY);
        Close("theil_nox", 1e-12, 1e-15, n.Slope, n.Intercept, n.LowSlope, n.HighSlope);
        foreach (var m in new[] { "hierarchical", "separate" })
        {
            var s = Stats.Siegelslopes(R.CY, R.CX, m);
            Close($"siegel/{m}", 1e-12, 1e-15, s.Slope, s.Intercept);
        }
        var xi = Stats.Chatterjeexi(R.CX, R.CY);
        Close("xi", 1e-12, 1e-15, xi.Statistic, xi.Pvalue);
        xi = Stats.Chatterjeexi(R.CX, R.CY, yContinuous: true);
        Close("xi_cont", 1e-12, 1e-15, xi.Statistic, xi.Pvalue);
        xi = Stats.Chatterjeexi(R.CX[..12], R.T3);
        Close("xi_ties", 1e-12, 1e-15, xi.Statistic, xi.Pvalue);
    }

    [Fact]
    public void WeightedTauMatchesScipy()
    {
        var t1 = R.T1[..12];
        Close("wtau", 1e-12, 1e-15,
            Stats.Weightedtau(R.CX, R.CY).Statistic, Stats.Weightedtau(R.CX, R.CY, rank: false).Statistic,
            Stats.Weightedtau(R.CX, R.CY, additive: false).Statistic, Stats.Weightedtau(R.CX, R.CY, weigher: r => 1.0 / ((r + 1.0) * (r + 1.0))).Statistic,
            Stats.Weightedtau(t1, R.T3).Statistic, Stats.Weightedtau(t1, R.T3, rank: false).Statistic,
            Stats.Weightedtau(R.CX[..10], R.CY[..10], [3, 1, 0, 2, 9, 8, 7, 6, 5, 4]).Statistic);
    }

    [Fact]
    public void KendallInversionCountAgreesWithNaivePairs()
    {
        var x = R.CX[..25];
        var y = R.CY[..25];
        long con = 0, dis = 0;
        for (var i = 0; i < x.Length; i++)
            for (var j = i + 1; j < x.Length; j++)
            {
                var s = Math.Sign(x[i] - x[j]) * Math.Sign(y[i] - y[j]);
                if (s > 0) con++; else if (s < 0) dis++;
            }
        var n = x.Length;
        var tau = (con - dis) / (n * (n - 1) / 2.0);
        Near(tau, Stats.Kendalltau(x, y).Statistic, 1e-14, 1e-15, "tau (no ties)");
    }
}
