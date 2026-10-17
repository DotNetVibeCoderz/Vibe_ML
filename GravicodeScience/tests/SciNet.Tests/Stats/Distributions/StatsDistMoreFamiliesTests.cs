using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsDistAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// The families added to match scipy 1.17.1 completely: the generated reference table
/// (StatsDistReference2.Generated.cs, from gen_cont2.py / gen_disc2.py), identities, and samplers.
/// </summary>
public class StatsDistMoreFamiliesTests
{
    public static TheoryData<int> ContinuousIndices()
    {
        var d = new TheoryData<int>();
        for (var i = 0; i < StatsDistReference.Continuous2.Length; i++) d.Add(i);
        return d;
    }

    public static TheoryData<int> DiscreteIndices()
    {
        var d = new TheoryData<int>();
        for (var i = 0; i < StatsDistReference.Discrete2.Length; i++) d.Add(i);
        return d;
    }

    [Theory]
    [MemberData(nameof(ContinuousIndices))]
    public void ContinuousMatchesScipy(int index)
    {
        var c = StatsDistReference.Continuous2[index];
        var d = c.Make();
        var errors = new List<string>();
        Collect(errors, "pdf", c.X, c.Pdf, d.Pdf, c.Tol);
        Collect(errors, "logpdf", c.X, c.LogPdf, d.LogPdf, c.Tol);
        Collect(errors, "cdf", c.X, c.Cdf, d.Cdf, c.Tol);
        Collect(errors, "sf", c.X, c.Sf, d.Sf, c.Tol);
        Collect(errors, "logcdf", c.X, c.LogCdf, d.LogCdf, c.Tol);
        Collect(errors, "logsf", c.X, c.LogSf, d.LogSf, c.Tol);
        Collect(errors, "ppf", c.Q, c.Ppf, d.Ppf, c.Tol, 1e-13);
        Collect(errors, "isf", c.Q, c.Isf, d.Isf, c.Tol, 1e-13);
        Collect(errors, "mean", c.Mean, d.Mean, c.MomentTol);
        Collect(errors, "var", c.Var, d.Var, c.MomentTol);
        Collect(errors, "skew", c.Skew, d.Skew, c.MomentTol);
        Collect(errors, "kurtosis", c.Kurt, d.Kurtosis, c.MomentTol);
        Collect(errors, "entropy", c.Entropy, d.Entropy, c.EntropyTol);
        Collect(errors, "median", [0.5], [c.Median], _ => d.Median(), c.Tol, 1e-13);
        Assert.True(errors.Count == 0, c.Label + ":\n  " + string.Join("\n  ", errors));
    }

    [Theory]
    [MemberData(nameof(DiscreteIndices))]
    public void DiscreteMatchesScipy(int index)
    {
        var c = StatsDistReference.Discrete2[index];
        var d = c.Make();
        var errors = new List<string>();
        Collect(errors, "pmf", c.K, c.Pmf, d.Pmf, c.Tol);
        Collect(errors, "logpmf", c.K, c.LogPmf, d.LogPmf, c.Tol);
        Collect(errors, "cdf", c.K, c.Cdf, d.Cdf, c.Tol);
        Collect(errors, "sf", c.K, c.Sf, d.Sf, c.Tol);
        Collect(errors, "logcdf", c.K, c.LogCdf, d.LogCdf, c.Tol);
        Collect(errors, "logsf", c.K, c.LogSf, d.LogSf, c.Tol);
        Collect(errors, "ppf", c.Q, c.Ppf, d.Ppf, 0.0);
        Collect(errors, "isf", c.Q, c.Isf, d.Isf, 0.0);
        Collect(errors, "mean", c.Mean, d.Mean, c.MomentTol);
        Collect(errors, "var", c.Var, d.Var, c.MomentTol);
        Collect(errors, "skew", c.Skew, d.Skew, c.MomentTol);
        Collect(errors, "kurtosis", c.Kurt, d.Kurtosis, c.MomentTol);
        Collect(errors, "entropy", c.Entropy, d.Entropy, c.EntropyTol);
        Collect(errors, "median", c.Median, d.Median, 0.0);
        Assert.True(errors.Count == 0, c.Label + ":\n  " + string.Join("\n  ", errors));
    }

    private static bool IsSlow(string label) => label.Contains("Landau") || label.Contains("KSOne");

    [Theory]
    [MemberData(nameof(ContinuousIndices))]
    public void ContinuousIdentitiesHold(int index)
    {
        var c = StatsDistReference.Continuous2[index];
        var d = c.Make();
        var errors = new List<string>();
        foreach (var x in c.X)
        {
            var cdf = d.Cdf(x);
            var sf = d.Sf(x);
            if (Math.Abs(cdf + sf - 1) > 1e-12) errors.Add($"cdf+sf at {x}: {cdf + sf:R}");
            if (cdf > 1e-250 && cdf <= 0.5 && d.Pdf(x) > 0)
            {
                var back = d.Ppf(cdf);
                if (Math.Abs(back - x) > 1e-8 * Math.Max(1, Math.Abs(x))) errors.Add($"ppf(cdf({x})) = {back:R}");
            }
            if (sf > 1e-250 && sf < 0.5 && d.Pdf(x) > 0)
            {
                var back = d.Isf(sf);
                if (Math.Abs(back - x) > 1e-8 * Math.Max(1, Math.Abs(x))) errors.Add($"isf(sf({x})) = {back:R}");
            }
        }
        if (!IsSlow(c.Label))
        {
            var (lo, hi) = d.Support();
            var med = d.Median();
            var total = Integrate.Quad(d.Pdf, lo, med, 1e-12, 1e-10, limit: 200).Value + Integrate.Quad(d.Pdf, med, hi, 1e-12, 1e-10, limit: 200).Value;
            if (Math.Abs(total - 1) > 1e-7) errors.Add($"pdf integrates to {total:R}");
        }
        Assert.True(errors.Count == 0, c.Label + ":\n  " + string.Join("\n  ", errors));
    }

    [Theory]
    [MemberData(nameof(DiscreteIndices))]
    public void DiscreteIdentitiesHold(int index)
    {
        var c = StatsDistReference.Discrete2[index];
        var d = c.Make();
        var errors = new List<string>();
        var lo = d.Ppf(1e-13);
        // Heavy tails (betanbinom with a = 4.5 reaches 1e-13 near k = 9000) are checked over the first 600 points.
        var hi = Math.Min(d.Isf(1e-13), lo + 600);
        var sum = 0.0;
        for (var k = lo; k <= hi; k++)
        {
            sum += d.Pmf(k);
            var cdf = d.Cdf(k);
            var sf = d.Sf(k);
            if (Math.Abs(cdf + sf - 1) > 1e-13) errors.Add($"cdf+sf at {k}: {cdf + sf:R}");
            if (cdf > 1e-12 && d.Pmf(k) > 0 && d.Ppf(cdf) != k) errors.Add($"ppf(cdf({k})) = {d.Ppf(cdf)}");
            if (sf > 1e-12 && d.Pmf(k + 1) > 0 && d.Isf(sf) != k) errors.Add($"isf(sf({k})) = {d.Isf(sf)}");
        }
        if (hi < lo + 600 && Math.Abs(sum - 1) > 1e-11) errors.Add($"pmf sums to {sum:R}");
        Assert.True(errors.Count == 0, c.Label + ":\n  " + string.Join("\n  ", errors));
    }

    public static TheoryData<string> SamplerNames() =>
    [
        "alpha", "argus-small", "argus-mid", "argus-big", "betaprime", "burr", "fisk", "crystalball", "dgamma", "dpareto_lognorm",
        "exponpow", "exponweib", "fatiguelife", "foldcauchy", "gausshyper", "genexpon", "genhalflogistic", "genhyperbolic",
        "geninvgauss", "geninvgauss-small", "geninvgauss-neg", "gennorm", "gibrat", "gompertz", "halfgennorm", "halflogistic",
        "hypsecant", "invweibull", "irwinhall", "jf_skew_t", "kappa3", "kappa4", "ksone", "landau", "laplace_asymmetric",
        "levy_l", "levy_stable", "levy_stable-a1", "levy_stable-s0", "loggamma", "mielke", "moyal", "norminvgauss",
        "powerlognorm", "powernorm", "rdist", "recipinvgauss", "rel_breitwigner", "skewcauchy", "trapezoid", "truncpareto",
        "truncweibull_min", "vonmises_line", "wrapcauchy",
    ];

    private static ContinuousDistribution Sampler(string name) => name switch
    {
        "alpha" => Stats.Alpha(2.5), "argus-small" => Stats.Argus(0.3), "argus-mid" => Stats.Argus(1.2), "argus-big" => Stats.Argus(3),
        "betaprime" => Stats.BetaPrime(3, 4), "burr" => Stats.Burr(3, 2), "fisk" => Stats.Fisk(2.5), "crystalball" => Stats.CrystalBall(1.5, 3),
        "dgamma" => Stats.DGamma(1.5), "dpareto_lognorm" => Stats.DParetoLogNorm(0.5, 0.8, 2.5, 1.5), "exponpow" => Stats.ExponPow(2),
        "exponweib" => Stats.ExponWeib(2, 1.5), "fatiguelife" => Stats.FatigueLife(0.8), "foldcauchy" => Stats.FoldCauchy(2),
        "gausshyper" => Stats.GaussHyper(2, 3, 1.5, 2), "genexpon" => Stats.GenExpon(1, 2, 3), "genhalflogistic" => Stats.GenHalfLogistic(0.6),
        "genhyperbolic" => Stats.GenHyperbolic(0.5, 1.5, -0.5), "geninvgauss" => Stats.GenInvGauss(2.3, 1.5),
        "geninvgauss-small" => Stats.GenInvGauss(0.3, 0.2), "geninvgauss-neg" => Stats.GenInvGauss(-0.6, 0.8), "gennorm" => Stats.GenNorm(1.3),
        "gibrat" => Stats.Gibrat(), "gompertz" => Stats.Gompertz(0.9), "halfgennorm" => Stats.HalfGenNorm(0.7), "halflogistic" => Stats.HalfLogistic(),
        "hypsecant" => Stats.HypSecant(), "invweibull" => Stats.InvWeibull(3), "irwinhall" => Stats.IrwinHall(4), "jf_skew_t" => Stats.JFSkewT(3, 1.5),
        "kappa3" => Stats.Kappa3(1.5), "kappa4" => Stats.Kappa4(0.3, -0.2), "ksone" => Stats.KSOne(6), "landau" => Stats.Landau(),
        "laplace_asymmetric" => Stats.LaplaceAsymmetric(1.8), "levy_l" => Stats.LevyL(), "levy_stable" => Stats.LevyStable(1.6, -0.4),
        "levy_stable-a1" => Stats.LevyStable(1, 0.5, 0, 2), "levy_stable-s0" => Stats.LevyStable(0.8, 0.6, 1, 1, LevyStableParameterization.S0),
        "loggamma" => Stats.LogGamma(0.7), "mielke" => Stats.Mielke(2, 3), "moyal" => Stats.Moyal(), "norminvgauss" => Stats.NormInvGauss(1.5, -0.6),
        "powerlognorm" => Stats.PowerLogNorm(2, 0.5), "powernorm" => Stats.PowerNorm(3), "rdist" => Stats.RDist(3.5), "recipinvgauss" => Stats.RecipInvGauss(0.7),
        "rel_breitwigner" => Stats.RelBreitWigner(2), "skewcauchy" => Stats.SkewCauchy(0.4), "trapezoid" => Stats.Trapezoid(0.3, 0.6),
        "truncpareto" => Stats.TruncPareto(1.5, 4), "truncweibull_min" => Stats.TruncWeibullMin(2, 0.5, 2), "vonmises_line" => Stats.VonMisesLine(2),
        "wrapcauchy" => Stats.WrapCauchy(0.6),
        _ => throw new ArgumentException(name),
    };

    [Theory]
    [MemberData(nameof(SamplerNames))]
    public void ContinuousSamplesFollowTheirCdf(string name)
    {
        // Kolmogorov–Smirnov against the family's own CDF: D·√n above 1.95 has probability 0.001.
        // The integral-based CDFs (stable, Landau, GIG, …) are evaluated at a subsample of the sorted draws.
        var d = Sampler(name);
        var slow = name.StartsWith("levy_stable") || name is "landau" or "genhyperbolic" or "norminvgauss" or "gausshyper" or "ksone"
                   || name.StartsWith("geninvgauss");
        var n = slow ? 4_000 : 40_000;
        var x = d.Rvs(n, seed: 11);
        Array.Sort(x);
        var dmax = 0.0;
        var step = slow ? 4 : 1;
        for (var i = 0; i < n; i += step)
        {
            var f = d.Cdf(x[i]);
            dmax = Math.Max(dmax, Math.Max((i + 1.0) / n - f, f - (double)i / n));
        }
        Assert.True(dmax * Math.Sqrt(n) < 1.95, $"{name}: KS statistic {dmax * Math.Sqrt(n):F3}");
    }

    public static TheoryData<string> DiscreteSamplerNames() => ["betanbinom", "fisher", "wallenius", "poisson_binom"];

    private static DiscreteDistribution DiscreteSampler(string name) => name switch
    {
        "betanbinom" => Stats.BetaNBinom(5, 9.3, 1), "fisher" => Stats.NcHypergeomFisher(140, 80, 60, 2.5),
        "wallenius" => Stats.NcHypergeomWallenius(140, 80, 60, 2.5), "poisson_binom" => Stats.PoissonBinom([0.1, 0.5, 0.9, 0.3, 0.7, 0.25]),
        _ => throw new ArgumentException(name),
    };

    [Theory]
    [MemberData(nameof(DiscreteSamplerNames))]
    public void DiscreteSampleFrequenciesMatchThePmf(string name)
    {
        var d = DiscreteSampler(name);
        const int n = 60_000;
        var counts = new Dictionary<double, int>();
        foreach (var v in d.Rvs(n, seed: 5)) counts[v] = counts.GetValueOrDefault(v) + 1;
        for (var k = d.Ppf(1e-6); k <= d.Isf(1e-6); k++)
        {
            var p = d.Pmf(k);
            if (p < 1e-3) continue;
            var f = counts.GetValueOrDefault(k) / (double)n;
            Assert.True(Math.Abs(f - p) < 5 * Math.Sqrt(p * (1 - p) / n) + 1e-12, $"{name}: P({k}) = {p:F5}, sample {f:F5}");
        }
    }

    [Fact]
    public void LevyStableReducesToItsClosedForms()
    {
        // α = 2 is N(0, 2γ²); α = 1, β = 0 is Cauchy; α = ½, β = 1 is Lévy shifted by ζ = −1 (S0) / unshifted (S1).
        foreach (var x in new[] { -7.0, -1.3, 0.0, 0.4, 3.0 })
        {
            Close(Stats.Norm(0, Math.Sqrt(2)).Pdf(x), Stats.LevyStable(2, 0.3).Pdf(x), 1e-14, "alpha 2 pdf");
            Close(Stats.Cauchy().Cdf(x), Stats.LevyStable(1, 0).Cdf(x), 1e-14, "cauchy cdf");
        }
        foreach (var x in new[] { 0.05, 1.0, 20.0 })
        {
            Close(Stats.Levy().Pdf(x), Stats.LevyStable(0.5, 1).Pdf(x), 1e-13, "levy pdf");
            Close(Stats.Levy().Sf(x), Stats.LevyStable(0.5, 1).Sf(x), 1e-13, "levy sf");
        }
        // The general Nolan integral against those special cases' neighbours: continuous in α.
        Close(Stats.LevyStable(2, 0).Pdf(0.7), Stats.LevyStable(1.9999999, 0).Pdf(0.7), 1e-6, "continuity at alpha 2");
        Close(Stats.Levy().Cdf(3), Stats.LevyStable(0.5, 0.9999999).Cdf(3), 1e-6, "continuity at beta 1");
    }

    [Fact]
    public void LevyStableParameterisationsDifferByZeta()
    {
        // S1 at x equals S0 at x + ζ (α ≠ 1), ζ = −β tan(πα/2); the S0 mean is ζ for α > 1.
        double a = 1.4, b = -0.6;
        var zeta = -b * Math.Tan(Math.PI * a / 2);
        var s0 = Stats.LevyStable(a, b, parameterization: LevyStableParameterization.S0);
        var s1 = Stats.LevyStable(a, b);
        foreach (var x in new[] { -3.0, 0.2, 5.0 }) Close(s0.Pdf(x + zeta), s1.Pdf(x), 1e-12, $"pdf {x}");
        Close(zeta, s0.Mean(), 1e-14, "S0 mean");
        Assert.Equal(0.0, s1.Mean());
        // Landau is the α = 1, β = 1 stable law in S0 with unit scale.
        Close(Stats.LevyStable(1, 1, parameterization: LevyStableParameterization.S0).Pdf(2.5), Stats.Landau().Pdf(2.5), 1e-14, "landau");
        // α = 1 in S1 shifts the location by 2βγ log γ/π, as scipy.
        var shifted = Stats.LevyStable(1, 0.7, 0, 2);
        Close(2 * 0.7 * 2 * Math.Log(2) / Math.PI, shifted.Loc, 1e-14, "S1 alpha-1 location");
        Assert.Equal(0.0, shifted.NominalLoc);
    }

    [Fact]
    public void LandauTailsAreAccurate()
    {
        // Right tail 2/(πx)·(1 + O(log x / x)); the doubly exponential left tail (exp(−e^(−πx/2−1))) stays positive until it underflows.
        var l = Stats.Landau();
        var x = 1e8;
        Assert.True(Math.Abs(l.Sf(x) * Math.PI * x / 2 - 1) < 1e-6, $"sf {l.Sf(x)}");
        Assert.True(l.Cdf(-3.5) > 0 && l.Cdf(-3.5) < 1e-20, $"cdf(-3.5) {l.Cdf(-3.5)}");
        Close(l.Cdf(-1.5), 1 - l.Sf(-1.5), 1e-12, "cdf+sf");
    }

    [Fact]
    public void SpearmanrhoMatchesSpearmanrAndScipy()
    {
        double[] x = [1, 2, 3, 4, 5, 7, 6, 9, 8, 10], y = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9];
        var r = Stats.Spearmanrho(x, y);
        // scipy 1.17.1: spearmanrho(x, y) = (0.8666666666666663, 0.0011735381801554782)
        Close(0.8666666666666663, r.Statistic, 1e-13, "rho");
        Close(0.0011735381801554782, r.Pvalue, 1e-9, "p");
        Close(0.9994132309099223, Stats.Spearmanrho(x, y, Alternative.Less).Pvalue, 1e-12, "less");
    }

    [Fact]
    public void NumericalFitThatLeavesDataOutsideTheSupportThrowsFitError()
    {
        // Everything fixed: nothing to optimise, and −0.5 lies below an exponential's support.
        var ex = Assert.Throws<FitError>(() => Stats.Expon().FitNumerically([-0.5, 1, 2], floc: 0, fscale: 1));
        Assert.IsAssignableFrom<InvalidOperationException>(ex);
    }

    [Fact]
    public void MomentsThatDoNotExistAreReportedHonestly()
    {
        // scipy evaluates Γ(1 − n/c) at negative arguments for invweibull(0.8) and reports a mean of −4.9.
        var w = Stats.InvWeibull(0.8);
        Assert.True(double.IsPositiveInfinity(w.Mean()));
        Assert.True(double.IsPositiveInfinity(w.Var()));
        var w3 = Stats.InvWeibull(3.5);
        Close(Special.Gamma(1 - 1 / 3.5), w3.Mean(), 1e-14, "mean");
        Assert.True(double.IsNaN(w3.Kurtosis()));
        Assert.True(double.IsNegativeInfinity(Stats.LevyL().Mean()));
    }

    [Fact]
    public void LevyStableAlphaOneS1MatchesScipysUnfrozenCall()
    {
        // scipy 1.17.1 levy_stable.pdf(x, 1, 0.7, scale=2) (the call form; its frozen object drops the
        // 2βγ log γ/π shift that S1 needs at α = 1, so it is not a usable reference there).
        var d = Stats.LevyStable(1, 0.7, 0, 2);
        Close(0.12897263934885508, d.Pdf(1.0), 1e-8, "pdf(1)");
        Close(0.12371775317643505, d.Pdf(-1.0), 1e-8, "pdf(-1)");
        Close(0.6582082543242085, d.Cdf(3.0), 1e-8, "cdf(3)");
    }

    [Fact]
    public void ScipyInaccurateTails()
    {
        // Points left out of the generated table because scipy 1.17.1 is wrong there (1 − CDF in a tail,
        // exp underflowing before log, 1 − q inside a quantile). Expected values are closed forms or
        // 25-digit mpmath.
        Close(1 / (1 + Math.Pow(1000, 3.09)), Stats.Fisk(3.09).Sf(1000), 1e-13, "fisk sf");
        Close(-Special.Expm1(-Math.Exp(-30)), Stats.Kappa4(0, 0).Sf(30), 1e-13, "gumbel-kappa4 sf");
        Close(Math.Log(10.58) - 11.58 * Math.Log(0.5) - Math.Pow(0.5, -10.58), Stats.InvWeibull(10.58).LogPdf(0.5), 1e-13, "invweibull logpdf");
        Close(-Math.Log(Math.PI) - 798 + 2 * Math.Log(2), Stats.HypSecant(1, 0.5).LogPdf(400), 1e-13, "hypsecant logpdf");
        var t = Math.Pow(50, -10.5);
        Close(Math.Log(4.3 * t) - 5.3 * t / 2, Stats.Burr(10.5, 4.3).LogSf(50), 1e-13, "burr logsf");
        var u = 1 - 0.99999;
        Close(u * u / (1.6 * 0.2), Stats.Trapezoid(0.2, 0.8).Sf(0.99999), 1e-12, "trapezoid sf");
        Close(2.25 / (Math.PI * 1e9), Stats.SkewCauchy(0.5).Sf(1e9), 1e-12, "skewcauchy sf");
        var y = Math.Pow(1 - 0.2 * 4.99, 5);
        Close(-Special.Expm1(-2 * Special.Log1p(0.5 * y)), Stats.Kappa4(-0.5, 0.2).Sf(4.99), 1e-11, "kappa4 sf");
        Close(3.657920025754286287e-13, Stats.Landau().Cdf(-3), 1e-11, "landau cdf (mpmath; Boost agrees)");
        Close(36.75444828002595, Stats.GenInvGauss(2.3, 1.5).Ppf(0.9999999999), 1e-10, "geninvgauss ppf (mpmath; scipy 4e-4 off)");
        // Moments scipy integrates numerically; mpmath quadrature agrees with SciNet to 1e-15.
        Close(5.455399540097439, Stats.Gompertz(80).Kurtosis(), 1e-9, "gompertz kurtosis");
        Close(-0.4202485172399629, Stats.ExponPow(2.7).Kurtosis(), 1e-9, "exponpow kurtosis");
    }
}
