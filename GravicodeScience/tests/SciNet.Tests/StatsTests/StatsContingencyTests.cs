using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsTestsAssert;
using R = Gravicode.Science.Tests.SciNet.StatsTestsRef;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Count tests, contingency tables and exact tests against scipy 1.17.1.</summary>
public class StatsContingencyTests
{
    private static readonly string[] Lambdas = ["pearson", "log-likelihood", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"];

    [Fact]
    public void ChisquareAndPowerDivergenceMatchScipy()
    {
        var r = Stats.Chisquare(R.OBS);
        Close("chisquare", 1e-12, 1e-14, r.Statistic, r.Pvalue);
        r = Stats.Chisquare(R.OBS, R.EXPV, ddof: 1);
        Close("chisquare_exp", 1e-12, 1e-14, r.Statistic, r.Pvalue);
        foreach (var lam in Lambdas)
        {
            r = Stats.PowerDivergence(R.OBS, R.EXPV, 0, lam);
            Close($"power_divergence/{lam}", 1e-12, 1e-14, r.Statistic, r.Pvalue);
        }
        Assert.Throws<ArgumentException>(() => Stats.Chisquare(R.OBS, [1.0, 1, 1, 1, 1, 1]));
    }

    [Fact]
    public void Chi2ContingencyMatchesScipy()
    {
        var tab = new NdArray(R.TAB, 3, 3);
        var r = Stats.Chi2Contingency(tab);
        Close("chi2c", 1e-12, 1e-12, [r.Statistic, r.Pvalue, r.Dof, .. Flat(r.ExpectedFreq)]);
        var g = Stats.Chi2Contingency(tab, lambda: 0.0);
        Close("chi2c_g", 1e-12, 1e-14, g.Statistic, g.Pvalue);
        var t2 = new NdArray(R.TAB2, 2, 2);
        var c = Stats.Chi2Contingency(t2);
        Close("chi2c_2x2", 1e-12, 1e-14, c.Statistic, c.Pvalue, c.Dof);
        c = Stats.Chi2Contingency(t2, correction: false);
        Close("chi2c_2x2_nc", 1e-12, 1e-14, c.Statistic, c.Pvalue);
        var t3 = Stats.Chi2Contingency(new NdArray(R.TAB3, 2, 2, 2));
        Close("chi2c_3d", 1e-12, 1e-12, [t3.Statistic, t3.Pvalue, t3.Dof, .. Flat(t3.ExpectedFreq)]);
        foreach (var m in new[] { "cramer", "tschuprow", "pearson" })
            Close($"association/{m}", 1e-12, 1e-14, Stats.Association(tab, m));
        var mg = Stats.Margins(tab);
        Close("margins", 0, 1e-12, [.. mg[0], .. mg[1]]);
        var (elements, count) = Stats.Crosstab(R.CA, R.CB);
        Assert.Equal([1.0, 2, 3], elements[0]);
        Close("crosstab", 0, 0, Flat(count));
    }

    [Fact]
    public void FisherExactMatchesScipy()
    {
        foreach (var t in new[] { "f1", "f2", "f3", "f4", "f5" })
        {
            var table = new NdArray(t switch
            {
                "f1" => R.FT_f1, "f2" => R.FT_f2, "f3" => R.FT_f3, "f4" => R.FT_f4, _ => R.FT_f5,
            }, 2, 2);
            foreach (var (name, alt) in Alternatives)
            {
                var r = Stats.FisherExact(table, alt);
                Close($"fisher/{t}/{name}", 1e-11, 1e-14, r.Statistic, r.Pvalue);
            }
        }
    }

    [Fact]
    public void UnconditionalExactTestsMatchScipy()
    {
        foreach (var t in new[] { "f1", "f2" })
        {
            var table = new NdArray(t == "f1" ? R.FT_f1 : R.FT_f2, 2, 2);
            foreach (var (name, alt) in Alternatives)
            {
                var b = Stats.BarnardExact(table, alt);
                Close($"barnard/{t}/{name}", 1e-6, 1e-10, b.Statistic, b.Pvalue);
                b = Stats.BarnardExact(table, alt, pooled: false);
                Close($"barnard_unpooled/{t}/{name}", 1e-6, 1e-10, b.Statistic, b.Pvalue);
                var bo = Stats.BoschlooExact(table, alt);
                Close($"boschloo/{t}/{name}", 1e-6, 1e-10, bo.Statistic, bo.Pvalue);
            }
        }
    }

    [Fact]
    public void BinomtestMatchesScipy()
    {
        foreach (var (k, n, p) in new[] { (7, 20, 0.5), (3, 25, 0.3), (18, 20, 0.6), (50, 120, 0.4), (0, 10, 0.2), (10, 10, 0.7) })
            foreach (var (name, alt) in Alternatives)
            {
                var r = Stats.Binomtest(k, n, p, alt);
                var ci = r.ProportionCi(0.95);
                var cw = r.ProportionCi(0.9, "wilson");
                var cc = r.ProportionCi(0.99, "wilsoncc");
                var key = $"binomtest/{k}/{n}/{p.ToString(System.Globalization.CultureInfo.InvariantCulture)}/{name}";
                Close(key, 1e-9, 1e-13, r.Statistic, r.Pvalue, ci.Low, ci.High, cw.Low, cw.High, cc.Low, cc.High);
            }
    }

    [Fact]
    public void PoissonMeansTestMatchesScipy()
    {
        foreach (var (name, alt) in Alternatives)
        {
            var r = Stats.PoissonMeansTest(10, 3.0, 4, 2.5, alternative: alt);
            Close($"poisson_means/{name}", 1e-9, 1e-13, r.Statistic, r.Pvalue);
            r = Stats.PoissonMeansTest(25, 10.0, 12, 8.0, diff: 0.2, alternative: alt);
            Close($"poisson_means_diff/{name}", 1e-9, 1e-13, r.Statistic, r.Pvalue);
        }
    }
}
