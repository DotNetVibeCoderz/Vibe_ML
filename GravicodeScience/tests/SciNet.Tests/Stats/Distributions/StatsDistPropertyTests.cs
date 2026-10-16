using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsDistAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// Identities every distribution must satisfy whatever scipy says (Ppf ∘ Cdf, Cdf + Sf = 1, the
/// density integrating to one), tail values pinned against 40-digit mpmath where scipy itself is
/// inaccurate, and sampler checks.
/// </summary>
public class StatsDistPropertyTests
{
    private static bool IsSlow(string label) => label.Contains("Studentized") || label.Contains("KSTwoDistribution");

    [Theory]
    [MemberData(nameof(StatsDistReferenceTests.ContinuousIndices), MemberType = typeof(StatsDistReferenceTests))]
    public void ContinuousIdentitiesHold(int index)
    {
        var c = StatsDistReference.Continuous[index];
        var d = c.Make();
        var errors = new List<string>();
        foreach (var x in c.X)
        {
            var cdf = d.Cdf(x);
            var sf = d.Sf(x);
            if (Math.Abs(cdf + sf - 1) > 1e-13) errors.Add($"cdf+sf at {x}: {cdf + sf:R}");
            // Invert on the side where the probability is small, where it carries full precision.
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
            var mean = d.Mean();
            if (double.IsFinite(mean) && double.IsFinite(d.Var()))
            {
                var e = d.Expect();
                if (Math.Abs(e - mean) > 1e-6 * Math.Max(1, d.Std())) errors.Add($"expect(x) = {e:R}, mean {mean:R}");
            }
        }
        Assert.True(errors.Count == 0, c.Label + ":\n  " + string.Join("\n  ", errors));
    }

    [Theory]
    [MemberData(nameof(StatsDistReferenceTests.DiscreteIndices), MemberType = typeof(StatsDistReferenceTests))]
    public void DiscreteIdentitiesHold(int index)
    {
        var c = StatsDistReference.Discrete[index];
        var d = c.Make();
        var errors = new List<string>();
        var lo = d.Ppf(1e-13);
        // Heavy tails (Zipf) reach 1e-13 only after ~1e7 points; check the first 5000 and skip the sum.
        var hi = Math.Min(d.Isf(1e-13), lo + 5000);
        var truncated = hi < d.Isf(1e-13);
        var sum = 0.0;
        for (var k = lo; k <= hi; k++)
        {
            sum += d.Pmf(k);
            var cdf = d.Cdf(k);
            var sf = d.Sf(k);
            if (Math.Abs(cdf + sf - 1) > 1e-13) errors.Add($"cdf+sf at {k}: {cdf + sf:R}");
            // scipy's semantics: Ppf(q) is the smallest k with Cdf(k) ≥ q, so Ppf(Cdf(k)) = k exactly.
            if (cdf > 1e-12 && d.Pmf(k) > 0 && d.Ppf(cdf) != k) errors.Add($"ppf(cdf({k})) = {d.Ppf(cdf)}");
            if (sf > 1e-12 && d.Pmf(k + 1) > 0 && d.Isf(sf) != k) errors.Add($"isf(sf({k})) = {d.Isf(sf)}");
            // Any q strictly between Cdf(k) and Cdf(k + 1) maps to k + 1.
            var next = d.Cdf(k + 1);
            var mid = cdf + 0.5 * (next - cdf);
            if (cdf > 1e-12 && mid > cdf && mid < next && d.Ppf(mid) != k + 1) errors.Add($"ppf between cdf({k}) and cdf({k + 1}) = {d.Ppf(mid)}");
        }
        if (!truncated && Math.Abs(sum - 1) > 1e-11) errors.Add($"pmf sums to {sum:R}");
        Assert.True(errors.Count == 0, c.Label + ":\n  " + string.Join("\n  ", errors));
    }

    [Fact]
    public void TailValuesWhereScipyIsInaccurateMatchMpmath()
    {
        // Each expected value is a 40-digit mpmath root of the exact CDF/SF; scipy 1.17.1 is off in
        // the 4th–5th digit for these (it complements, or rounds 1 − q, inside its quantiles).
        Close(35.96944060997121, Stats.Pearson3(0.8, 1, 2).Isf(1e-15), 1e-12, "pearson3 isf");
        Close(-22.73643988627036, Stats.Pearson3(-1.2).Ppf(1e-15), 1e-12, "pearson3 ppf");
        Close(1184.444406826866, Stats.F(5, 12).Isf(1e-15), 1e-12, "f isf");
        Close(15700159456.45515, Stats.F(1.5, 3).Isf(1e-15), 1e-11, "f isf heavy");
        Close(8.16655890078333, Stats.TruncNorm(1, double.PositiveInfinity).Isf(1e-15), 1e-12, "truncnorm isf");
        Close(-38.60078200730818, Stats.TruncNorm(-40, -38).Isf(0.9999999999), 1e-12, "truncnorm far tail");
        Close(7.594411951464345e-10, Stats.SkewNorm(-2).Sf(2.5), 1e-9, "skewnorm sf");
        Close(105.28299585139872, Stats.ExponNorm(1.5, 1, 2).Isf(1e-15), 1e-12, "exponnorm isf");
        Close(19.782690652341994, Stats.FoldNorm(1.95, 0, 2).Isf(1e-15), 1e-12, "foldnorm isf");
        Close(0.00089032265440267, Stats.KSTwoBign().Entropy(), 1e-9, "kstwobign entropy");
        // Closed forms: halfnorm ppf(q) → √(2π)·q·scale/2… exactly scale·√2·erfinv(q) ≈ √(2π)·q for scale 2;
        // beta(2,5) cdf ≈ 15x² near 0, so ppf(1e-300) = √(1e-300/15).
        Close(Math.Sqrt(2 * Math.PI) * 1e-15, Stats.HalfNorm(0, 2).Ppf(1e-15), 1e-12, "halfnorm tiny q");
        Close(Math.Sqrt(1e-300 / 15), Stats.Beta(2, 5).Ppf(1e-300), 1e-10, "beta tiny q");
        // Student t far tail: the quantile must invert the CDF at q = 1e-300 (scipy returns +inf).
        var t = Stats.T(3.5, 1, 2);
        var x = t.Ppf(1e-300);
        Assert.True(x < -1e85, $"t ppf {x}");
        Close(1e-300, t.Cdf(x), 1e-10, "t cdf(ppf)");
        // log tails stay finite where the probability itself underflows.
        Close(-1250.0 - 0.5 * Math.Log(2 * Math.PI * 2500) + Math.Log(1 - 1.0 / 2500 + 3.0 / (2500.0 * 2500)), Stats.Norm().LogSf(50), 1e-10, "norm logsf(50)");
    }

    [Fact]
    public void FarTailTruncatedNormalMomentsMatchMpmath()
    {
        // 60-digit mpmath quadrature; raw-moment recursions lose 6 digits of the variance here.
        var d = Stats.TruncNorm(-40, -38);
        Close(-38.02627946657587, d.Mean(), 1e-14, "mean");
        Close(0.000689659753466258869, d.Var(), 1e-9, "var");
        Close(-1.99587763355426040, d.Skew(), 1e-7, "skew");
        Close(5.96709446781893103, d.Kurtosis(), 1e-6, "kurtosis");
    }

    public static TheoryData<string> SamplerNames() =>
    [
        "norm", "expon", "gamma0.4", "gamma3", "beta", "betasmall", "t", "chi2", "f", "lognorm", "weibull", "laplace", "logistic",
        "gumbel", "rayleigh", "maxwell", "invgauss", "vonmises", "skewnorm", "truncnorm", "genextreme", "genpareto", "ncx2",
        "nct", "rice", "pareto", "exponnorm", "johnsonsu", "levy", "cauchy", "semicircular", "arcsine",
    ];

    private static ContinuousDistribution Sampler(string name) => name switch
    {
        "norm" => Stats.Norm(1, 2), "expon" => Stats.Expon(0.5, 2), "gamma0.4" => Stats.Gamma(0.4), "gamma3" => Stats.Gamma(3, 0, 2),
        "beta" => Stats.Beta(2, 5), "betasmall" => Stats.Beta(0.3, 0.4), "t" => Stats.T(4.5), "chi2" => Stats.Chi2(3),
        "f" => Stats.F(4, 9), "lognorm" => Stats.LogNorm(0.6), "weibull" => Stats.WeibullMin(1.7), "laplace" => Stats.Laplace(),
        "logistic" => Stats.Logistic(), "gumbel" => Stats.GumbelR(), "rayleigh" => Stats.Rayleigh(), "maxwell" => Stats.Maxwell(),
        "invgauss" => Stats.InvGauss(0.6), "vonmises" => Stats.VonMises(3), "skewnorm" => Stats.SkewNorm(3), "truncnorm" => Stats.TruncNorm(-1, 2),
        "genextreme" => Stats.GenExtreme(-0.2), "genpareto" => Stats.GenPareto(0.2), "ncx2" => Stats.Ncx2(3, 2), "nct" => Stats.Nct(6, 1),
        "rice" => Stats.Rice(1.5), "pareto" => Stats.Pareto(3), "exponnorm" => Stats.ExponNorm(1.2), "johnsonsu" => Stats.JohnsonSU(0.5, 1.5),
        "levy" => Stats.Levy(), "cauchy" => Stats.Cauchy(), "semicircular" => Stats.Semicircular(), "arcsine" => Stats.Arcsine(),
        _ => throw new ArgumentException(name),
    };

    [Theory]
    [MemberData(nameof(SamplerNames))]
    public void ContinuousSamplesFollowTheirCdf(string name)
    {
        // Kolmogorov–Smirnov against the distribution's own CDF: D·√n above 1.95 has probability
        // 0.001 under the null, so a correct sampler fails this with a fixed seed essentially never.
        var d = Sampler(name);
        const int n = 40_000;
        var x = d.Rvs(n, seed: 7);
        Array.Sort(x);
        var dmax = 0.0;
        for (var i = 0; i < n; i++)
        {
            var f = d.Cdf(x[i]);
            dmax = Math.Max(dmax, Math.Max((i + 1.0) / n - f, f - (double)i / n));
        }
        Assert.True(dmax * Math.Sqrt(n) < 1.95, $"{name}: KS statistic {dmax * Math.Sqrt(n):F3}");
    }

    public static TheoryData<string> DiscreteSamplerNames() =>
        ["binom", "binombig", "poisson", "poissonbig", "geom", "nbinom", "hypergeom", "zipf", "logser", "betabinom", "skellam", "dlaplace", "planck", "yulesimon", "bernoulli", "randint"];

    private static DiscreteDistribution DiscreteSampler(string name) => name switch
    {
        "binom" => Stats.Binom(30, 0.3), "binombig" => Stats.Binom(5000, 0.4), "poisson" => Stats.Poisson(4.2), "poissonbig" => Stats.Poisson(350),
        "geom" => Stats.Geom(0.15), "nbinom" => Stats.NBinom(3.5, 0.4), "hypergeom" => Stats.Hypergeom(60, 20, 15), "zipf" => Stats.Zipf(2.5),
        "logser" => Stats.Logser(0.8), "betabinom" => Stats.BetaBinom(20, 2, 3), "skellam" => Stats.Skellam(4, 2.5), "dlaplace" => Stats.DLaplace(0.6),
        "planck" => Stats.Planck(0.4), "yulesimon" => Stats.YuleSimon(3.5), "bernoulli" => Stats.Bernoulli(0.37), "randint" => Stats.RandInt(-2, 9),
        _ => throw new ArgumentException(name),
    };

    [Theory]
    [MemberData(nameof(DiscreteSamplerNames))]
    public void DiscreteSampleFrequenciesMatchThePmf(string name)
    {
        // Chi-squared-style check per support point: each empirical frequency within 5 binomial
        // standard errors of the mass, over every point with mass above 1e-3.
        var d = DiscreteSampler(name);
        const int n = 60_000;
        var counts = new Dictionary<double, int>();
        foreach (var v in d.Rvs(n, seed: 3)) counts[v] = counts.GetValueOrDefault(v) + 1;
        var lo = d.Ppf(1e-6);
        var hi = d.Isf(1e-6);
        for (var k = lo; k <= hi; k++)
        {
            var p = d.Pmf(k);
            if (p < 1e-3) continue;
            var f = counts.GetValueOrDefault(k) / (double)n;
            var se = Math.Sqrt(p * (1 - p) / n);
            Assert.True(Math.Abs(f - p) < 5 * se + 1e-12, $"{name}: P({k}) = {p:F5}, sample {f:F5}");
        }
    }

    [Fact]
    public void NormalSamplerHasExactTailMass()
    {
        // The ziggurat's tail branch must reproduce P(|Z| > 3.65) = 2.6e-4 (its base-strip boundary).
        var x = Stats.Norm().Rvs(2_000_000, seed: 21);
        var tail = x.Count(v => Math.Abs(v) > 3.6541528853610088) / (double)x.Length;
        var p = 2 * Special.Ndtr(-3.6541528853610088);
        Assert.True(Math.Abs(tail - p) < 5 * Math.Sqrt(p / x.Length), $"tail {tail:E3} vs {p:E3}");
        var beyond5 = x.Count(v => v > 5) / (double)x.Length;
        Assert.True(beyond5 < 10 * Special.Ndtr(-5), "far tail");
    }

    [Fact]
    public void VectorisedOverloadsMatchScalarsIncludingTheParallelPath()
    {
        var d = Stats.Gamma(2.5, 1, 3);
        var x = Enumerable.Range(0, 10_000).Select(i => i * 0.01).ToArray();
        var pdf = d.Pdf(x);
        var cdf = d.Cdf(x);
        var q = Enumerable.Range(1, 9_999).Select(i => i / 10_000.0).ToArray();
        var ppf = d.Ppf(q);
        for (var i = 0; i < x.Length; i += 37)
        {
            Assert.Equal(d.Pdf(x[i]), pdf[i]);
            Assert.Equal(d.Cdf(x[i]), cdf[i]);
        }
        for (var i = 0; i < q.Length; i += 41) Assert.Equal(d.Ppf(q[i]), ppf[i]);
        var nd = d.LogPdf(new NdArray(x[..12], 3, 4));
        Assert.Equal([3, 4], nd.Shape.ToArray());
        Assert.Equal(d.LogPdf(x[7]), nd.AsContiguous().ToArray()[7]);
        var pois = Stats.Poisson(3).Pmf(new[] { 0.0, 1, 2.5, 3 });
        Assert.Equal(0.0, pois[2]);
    }

    [Fact]
    public void ExpectIntervalMomentAndStatsAreConsistent()
    {
        var n = Stats.Norm(1.5, 2);
        Close(2 * 2 + 1.5 * 1.5, n.Expect(x => x * x), 1e-10, "E[X²]");
        Close(2 * 2 + 1.5 * 1.5, n.Moment(2), 1e-14, "moment 2");
        Close(1.5 * 1.5 * 1.5 * 1.5 + 6 * 1.5 * 1.5 * 4 + 3 * 16, n.Moment(4), 1e-13, "moment 4");
        Close(Math.Pow(1.5, 5) + 10 * Math.Pow(1.5, 3) * 4 + 15 * 1.5 * 16, n.Moment(5), 1e-12, "moment 5 (via Munp)");
        // E[X | X > 1.5] = μ + σ·φ(0)/(1 − Φ(0)) = 1.5 + 2·√(2/π).
        Close(1.5 + 2 * Math.Sqrt(2 / Math.PI), n.Expect(lb: 1.5, conditional: true), 1e-10, "conditional mean");
        var (lo, hi) = n.Interval(0.95);
        Close(1.5 - 2 * 1.959963984540054, lo, 1e-14, "interval lo");
        Close(1.5 + 2 * 1.959963984540054, hi, 1e-14, "interval hi");
        var p = Stats.Poisson(2.5);
        Close(2.5 * 2.5 + 2.5, p.Expect(k => k * k), 1e-12, "poisson E[K²]");
        Close(2.5 * 2.5 + 2.5, p.Moment(2), 1e-14, "poisson moment 2");
        var s = Stats.Gamma(2).Stats();
        Assert.Equal(new DistributionMoments(2, 2, 2 / Math.Sqrt(2), 3), s);
    }

    [Fact]
    public void DiscretePpfEdgeSemanticsFollowScipy()
    {
        var b = Stats.Binom(10, 0.3);
        Assert.Equal(-1, b.Ppf(0));      // one below the support
        Assert.Equal(10, b.Ppf(1));
        Assert.Equal(10, b.Isf(0));
        Assert.Equal(-1, b.Isf(1));
        Assert.True(double.IsNaN(b.Ppf(1.5)));
        Assert.Equal(0, b.Ppf(b.Cdf(0)));             // exactly at a CDF value → that k
        Assert.Equal(1, b.Ppf(b.Cdf(0) * (1 + 1e-12)));
        var sh = Stats.Poisson(2, loc: 5);
        Assert.Equal(5 + Stats.Poisson(2).Ppf(0.7), sh.Ppf(0.7));
        Assert.Equal(0.0, sh.Pmf(6.5));
    }

    [Fact]
    public void InvalidParametersThrow()
    {
        Assert.Throws<ArgumentException>(() => Stats.Norm(0, -1));
        Assert.Throws<ArgumentException>(() => Stats.Gamma(0));
        Assert.Throws<ArgumentException>(() => Stats.Beta(1, double.NaN));
        Assert.Throws<ArgumentException>(() => Stats.TruncNorm(2, 1));
        Assert.Throws<ArgumentException>(() => Stats.Binom(2.5, 0.3));
        Assert.Throws<ArgumentException>(() => Stats.Hypergeom(10, 11, 3));
        Assert.True(double.IsNaN(Stats.Norm().Ppf(-0.1)));
        Assert.True(double.IsNaN(Stats.Norm().Cdf(double.NaN)));
        Assert.Equal(0.0, Stats.Expon().Pdf(-1));
        Assert.Equal(1.0, Stats.Expon().Pdf(0));    // closed support, as scipy
    }
}
