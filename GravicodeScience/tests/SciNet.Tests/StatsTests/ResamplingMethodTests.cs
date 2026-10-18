using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// PermutationMethod / MonteCarloMethod / BootstrapMethod wired into the tests that accept them in
/// scipy 1.17. Permutation cases are small enough to be enumerated exactly, so their p-values are
/// deterministic and pinned to scipy (gen_resampling_methods.py).
/// </summary>
public class ResamplingMethodTests
{
    private static readonly double[] X = [1.2, -0.4, 2.2, 0.9, 3.1, -1.0];
    private static readonly double[] Y = [0.7, -0.1, 1.9, 1.5, 2.0, 0.2];
    private static readonly double[] A = [4.1, 5.3, 2.2, 6.0, 3.9];
    private static readonly double[] B = [6.6, 7.1, 5.2, 8.0];
    private static readonly PermutationMethod Pm = new();

    private static void Same(double expected, double actual, string what)
        => Assert.True(Math.Abs(expected - actual) <= 1e-12 * Math.Max(1, Math.Abs(expected)), $"{what}: expected {expected:R}, got {actual:R}");

    [Fact]
    public void ExactPermutationPvaluesMatchScipy()
    {
        var r = Stats.Pearsonr(X, Y, Pm);
        Same(0.9023634042614117, r.Statistic, "pearsonr r");
        Same(0.022222222222222223, r.Pvalue, "pearsonr two-sided");
        Same(0.9902777777777778, Stats.Pearsonr(X, Y, Pm, Alternative.Less).Pvalue, "pearsonr less");
        Same(0.011111111111111112, Stats.Pearsonr(X, Y, Pm, Alternative.Greater).Pvalue, "pearsonr greater");
        var s = Stats.Spearmanrho(X, Y, Pm);
        Same(0.8857142857142855, s.Statistic, "spearmanrho");
        Same(0.03333333333333333, s.Pvalue, "spearmanrho p");
        var t = Stats.TtestInd(A, B, Pm);
        Same(-2.6936428842766436, t.Statistic, "ttest_ind t");
        Same(0.047619047619047616, t.Pvalue, "ttest_ind two-sided");
        Same(0.9841269841269841, Stats.TtestInd(A, B, Pm, alternative: Alternative.Greater).Pvalue, "ttest_ind greater");
        var w = Stats.TtestInd(A, B, Pm, equalVar: false);
        Same(-2.76841696654944, w.Statistic, "welch t");
        Same(0.047619047619047616, w.Pvalue, "welch p");
        var u = Stats.MannWhitneyU(A, B, Pm);
        Same(2.0, u.Statistic, "U");
        Same(0.06349206349206349, u.Pvalue, "mannwhitneyu");
        Same(0.031746031746031744, Stats.MannWhitneyU(A, B, Pm, alternative: Alternative.Less).Pvalue, "mannwhitneyu less");
        double[] d = [0.5, -1.25, 2.0, 0.75, -0.25, 1.5, 3.0, 0.0];
        var wp = Stats.Wilcoxon(d, null, Pm, zeroMethod: "pratt");
        Same(7.0, wp.Statistic, "wilcoxon pratt T");
        Same(0.171875, wp.Pvalue, "wilcoxon pratt");
        Same(0.078125, Stats.Wilcoxon(d.Where(v => v != 0).ToArray(), null, Pm, alternative: Alternative.Greater).Pvalue, "wilcoxon greater");
        var xi = Stats.Chatterjeexi(X, Y, Pm);
        Same(0.3142857142857143, xi.Statistic, "xi");
        Same(0.08055555555555556, xi.Pvalue, "chatterjeexi");
        var ak = Stats.AndersonKsamp([A, B, [3.0, 4.5]], Pm);
        Same(1.5549598025436782, ak.Statistic, "anderson_ksamp");
        Same(0.09581529581529581, ak.Pvalue, "anderson_ksamp p");
        var c2 = Stats.Chi2Contingency(new NdArray([3.0, 1, 1, 2], 2, 2), Pm);
        Same(1.2152777777777777, c2.Statistic, "chi2 2x2");
        Same(0.4857142857142857, c2.Pvalue, "chi2 2x2 p");
        Assert.True(double.IsNaN(c2.Dof));
        var c3 = Stats.Chi2Contingency(new NdArray([2.0, 0, 1, 0, 2, 1], 2, 3), Pm);
        Same(4.0, c3.Statistic, "chi2 2x3");
        Same(0.6, c3.Pvalue, "chi2 2x3 p");
        var bws = Stats.BwsTest(A, B, Pm);
        Same(2.320863425925926, bws.Statistic, "bws");
        Same(0.05555555555555555, bws.Pvalue, "bws p");
    }

    [Fact]
    public void MonteCarloPvaluesAgreeWithTheParametricOnes()
    {
        // Under normality the Monte Carlo null of t (or r) is the exact one, so the p-values must agree
        // to Monte Carlo error: sd = sqrt(p(1-p)/B).
        var rng = new GraviRandom(7);
        var a = Enumerable.Range(0, 12).Select(_ => rng.Normal()).ToArray();
        var b = Enumerable.Range(0, 15).Select(_ => rng.Normal(0.6, 1)).ToArray();
        var mc = new MonteCarloMethod(NResamples: 20000) { Seed = 3 };
        var t = Stats.TtestInd(a, b);
        var tm = Stats.TtestInd(a, b, mc);
        Assert.Equal(t.Statistic, tm.Statistic);
        Assert.True(Math.Abs(t.Pvalue - tm.Pvalue) < 4 * Math.Sqrt(t.Pvalue * (1 - t.Pvalue) / 20000) + 1e-4, $"{t.Pvalue} vs {tm.Pvalue}");
        var x = a[..12];
        var y = x.Select(v => 0.3 * v + rng.Normal()).ToArray();
        var r = Stats.Pearsonr(x, y);
        var rm = Stats.Pearsonr(x, y, mc);
        Assert.True(Math.Abs(r.Pvalue - rm.Pvalue) < 4 * Math.Sqrt(r.Pvalue * (1 - r.Pvalue) / 20000) + 1e-4, $"{r.Pvalue} vs {rm.Pvalue}");
        // Same seed, same answer; a custom null generator is honoured.
        Assert.Equal(rm.Pvalue, Stats.Pearsonr(x, y, mc).Pvalue);
        var uniformNull = mc with { Rvs = [(n, g) => Enumerable.Range(0, n).Select(_ => g.NextDouble()).ToArray()] };
        Assert.InRange(Stats.Pearsonr(x, y, uniformNull).Pvalue, 0, 1);
        // Monte Carlo tables with the observed margins approximate the permutation p-value.
        var tab = new NdArray([3.0, 1, 1, 2], 2, 2);
        var c = Stats.Chi2Contingency(tab, new MonteCarloMethod(NResamples: 20000) { Seed = 5 });
        Assert.True(Math.Abs(c.Pvalue - 0.4857142857142857) < 0.02, c.Pvalue.ToString());
        Assert.Throws<ArgumentException>(() => Stats.Chi2Contingency(tab, uniformNull));
    }

    [Fact]
    public void AndersonWithMonteCarloMethod()
    {
        var rng = new GraviRandom(21);
        var normal = Enumerable.Range(0, 60).Select(_ => rng.Normal(2, 3)).ToArray();
        var expo = Enumerable.Range(0, 60).Select(_ => rng.Exponential(1.0)).ToArray();
        var m = new MonteCarloMethod(NResamples: 499) { Seed = 1 };
        var ok = Stats.Anderson(normal, "norm", m);
        Assert.Equal(Stats.Anderson(normal).Statistic, ok.Statistic);
        Assert.True(ok.Pvalue > 0.05, ok.Pvalue.ToString());
        var bad = Stats.Anderson(expo, "norm", m);
        Assert.True(bad.Pvalue < 0.01, bad.Pvalue.ToString());
        Assert.True(Stats.Anderson(expo, "expon", m).Pvalue > 0.05);
    }

    [Fact]
    public void BootstrapConfidenceIntervalForPearsonR()
    {
        var rng = new GraviRandom(5);
        var x = Enumerable.Range(0, 80).Select(_ => rng.Normal()).ToArray();
        var y = x.Select(v => 0.6 * v + 0.8 * rng.Normal()).ToArray();
        var r = Stats.Pearsonr(x, y);
        var fisher = r.ConfidenceInterval(0.9);
        var boot = r.ConfidenceInterval(0.9, new BootstrapMethod(NResamples: 4000) { Seed = 2 });
        Assert.InRange(r.Statistic, boot.Low, boot.High);
        // For bivariate normal data the BCa interval and Fisher's agree to sampling error.
        Assert.True(Math.Abs(boot.Low - fisher.Low) < 0.06 && Math.Abs(boot.High - fisher.High) < 0.06, $"{boot} vs {fisher}");
        var oneSided = Stats.Pearsonr(x, y, Alternative.Greater).ConfidenceInterval(0.9, new BootstrapMethod(NResamples: 2000, Method: "percentile") { Seed = 2 });
        Assert.Equal(1.0, oneSided.High);
        Assert.Equal(fisher, r.ConfidenceInterval(0.9, null));
        Assert.Throws<ArgumentException>(() => Stats.TtestInd(A, B, new MonteCarloMethod(NResamples: 0)));
    }
}
