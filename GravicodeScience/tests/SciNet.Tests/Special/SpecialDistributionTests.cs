using Gravicode.Science.SciNet;
using Xunit;
using R = Gravicode.Science.Tests.SciNet.SpecialReference;

namespace Gravicode.Science.Tests.SciNet;

public class SpecialDistributionTests
{
    [Fact] public void Stdtr_MatchesReference() => SpecialAssert.Table(R.Stdtr, a => Special.Stdtr(a[0], a[1]), 5e-14, "Stdtr");
    [Fact] public void Stdtrit_MatchesReference() => SpecialAssert.Table(R.Stdtrit, a => Special.Stdtrit(a[0], a[1]), 3e-15, "Stdtrit");
    [Fact] public void Chdtr_MatchesReference() => SpecialAssert.Table(R.Chdtr, a => Special.Chdtr(a[0], a[1]), 1e-15, "Chdtr");
    [Fact] public void Chdtrc_MatchesReference() => SpecialAssert.Table(R.Chdtrc, a => Special.Chdtrc(a[0], a[1]), 1e-15, "Chdtrc");
    [Fact] public void Chdtri_MatchesReference() => SpecialAssert.Table(R.Chdtri, a => Special.Chdtri(a[0], a[1]), 5e-15, "Chdtri");
    [Fact] public void Fdtr_MatchesReference() => SpecialAssert.Table(R.Fdtr, a => Special.Fdtr(a[0], a[1], a[2]), 5e-15, "Fdtr");
    [Fact] public void Fdtrc_MatchesReference() => SpecialAssert.Table(R.Fdtrc, a => Special.Fdtrc(a[0], a[1], a[2]), 5e-15, "Fdtrc");
    [Fact] public void Fdtri_MatchesReference() => SpecialAssert.Table(R.Fdtri, a => Special.Fdtri(a[0], a[1], a[2]), 5e-15, "Fdtri");
    [Fact] public void Bdtr_MatchesReference() => SpecialAssert.Table(R.Bdtr, a => Special.Bdtr(a[0], a[1], a[2]), 5e-14, "Bdtr");
    [Fact] public void Bdtrc_MatchesReference() => SpecialAssert.Table(R.Bdtrc, a => Special.Bdtrc(a[0], a[1], a[2]), 5e-14, "Bdtrc");
    [Fact] public void Bdtri_MatchesReference() => SpecialAssert.Table(R.Bdtri, a => Special.Bdtri(a[0], a[1], a[2]), 1e-14, "Bdtri");
    [Fact] public void Pdtr_MatchesReference() => SpecialAssert.Table(R.Pdtr, a => Special.Pdtr(a[0], a[1]), 1e-15, "Pdtr");
    [Fact] public void Pdtrc_MatchesReference() => SpecialAssert.Table(R.Pdtrc, a => Special.Pdtrc(a[0], a[1]), 1e-15, "Pdtrc");
    [Fact] public void Pdtri_MatchesReference() => SpecialAssert.Table(R.Pdtri, a => Special.Pdtri(a[0], a[1]), 1e-15, "Pdtri");
    [Fact] public void Nbdtr_MatchesReference() => SpecialAssert.Table(R.Nbdtr, a => Special.Nbdtr(a[0], a[1], a[2]), 4e-15, "Nbdtr");
    [Fact] public void Nbdtrc_MatchesReference() => SpecialAssert.Table(R.Nbdtrc, a => Special.Nbdtrc(a[0], a[1], a[2]), 4e-15, "Nbdtrc");
    [Fact] public void Nbdtri_MatchesReference() => SpecialAssert.Table(R.Nbdtri, a => Special.Nbdtri(a[0], a[1], a[2]), 2e-15, "Nbdtri");
    [Fact] public void Gdtr_MatchesReference() => SpecialAssert.Table(R.Gdtr, a => Special.Gdtr(a[0], a[1], a[2]), 2e-14, "Gdtr");
    [Fact] public void Gdtrc_MatchesReference() => SpecialAssert.Table(R.Gdtrc, a => Special.Gdtrc(a[0], a[1], a[2]), 1e-14, "Gdtrc");
    [Fact] public void Kolmogorov_MatchesReference() => SpecialAssert.Table(R.Kolmogorov, a => Special.Kolmogorov(a[0]), 1e-14, "Kolmogorov");
    [Fact] public void KolmogorovInverse_MatchesReference() => SpecialAssert.Table(R.KolmogorovInverse, a => Special.KolmogorovInverse(a[0]), 5e-15, "KolmogorovInverse");
    [Fact] public void Smirnov_MatchesReference() => SpecialAssert.Table(R.Smirnov, a => Special.Smirnov((int)a[0], a[1]), 5e-14, "Smirnov");
    [Fact] public void SmirnovInverse_MatchesReference() => SpecialAssert.Table(R.SmirnovInverse, a => Special.SmirnovInverse((int)a[0], a[1]), 2e-13, "SmirnovInverse");

    [Fact]
    public void Stdtr_ClosedFormsForOneAndTwoDegreesOfFreedom()
    {
        foreach (var t in new[] { -50.0, -1.3 })
        {
            // Lower-tail forms without the cancellation of 1/2 + (...)
            SpecialAssert.Close(-Math.Atan(1 / t) / Math.PI, Special.Stdtr(1, t), 2e-15, $"Cauchy CDF at {t}");
            var r = Math.Sqrt(2 + t * t);
            SpecialAssert.Close(1 / (r * (r - t)), Special.Stdtr(2, t), 2e-15, $"t(2) CDF at {t}");
        }
        foreach (var t in new[] { 0.2, 7.0 })
        {
            SpecialAssert.Close(0.5 + Math.Atan(t) / Math.PI, Special.Stdtr(1, t), 2e-15, $"Cauchy CDF at {t}");
            SpecialAssert.Close(0.5 + t / (2 * Math.Sqrt(2 + t * t)), Special.Stdtr(2, t), 2e-15, $"t(2) CDF at {t}");
        }
        foreach (var p in new[] { 1e-8, 0.1, 0.5, 0.8, 0.999 })
            foreach (var df in new[] { 1.0, 2.0, 3.5, 25.0 })
                SpecialAssert.Close(p, Special.Stdtr(df, Special.Stdtrit(df, p)), 1e-13, $"stdtr(stdtrit({df}, {p}))");
        SpecialAssert.Close(Special.Ndtr(1.7), Special.Stdtr(1e12, 1.7), 1e-12, "normal limit");
    }

    [Fact]
    public void ChiSquare_ClosedFormAndInverse()
    {
        foreach (var x in new[] { 0.01, 1.0, 5.0, 40.0 })
            SpecialAssert.Close(-Special.Expm1(-x / 2), Special.Chdtr(2, x), 1e-15, $"χ²(2) at {x}");
        foreach (var v in new[] { 1.0, 7.0, 120.0 })
            SpecialAssert.Close(0.05, Special.Chdtrc(v, Special.Chdtri(v, 0.05)), 1e-14, $"chdtri round trip, v={v}");
    }

    [Fact]
    public void Binomial_And_Poisson_MatchNaiveSums()
    {
        // Naive pmf sums are exact enough for small n to pin the incomplete-beta mapping.
        foreach (var (n, p) in new[] { (12, 0.3), (20, 0.85) })
        {
            double Pmf(int j) => (double)Special.CombExact(n, j) * Math.Pow(p, j) * Math.Pow(1 - p, n - j);
            for (var k = 0; k < n; k++)
            {
                double cdf = 0, sf = 0;
                for (var j = 0; j <= k; j++) cdf += Pmf(j);
                for (var j = k + 1; j <= n; j++) sf += Pmf(j);
                SpecialAssert.Close(cdf, Special.Bdtr(k, n, p), 1e-13, $"bdtr({k}, {n}, {p})");
                SpecialAssert.Close(sf, Special.Bdtrc(k, n, p), 1e-13, $"bdtrc({k}, {n}, {p})");
            }
        }
        double pc = 0, term = Math.Exp(-3.7);
        for (var k = 0; k < 15; k++)
        {
            if (k > 0) term *= 3.7 / k;
            pc += term;
            SpecialAssert.Close(pc, Special.Pdtr(k, 3.7), 1e-14, $"pdtr({k}, 3.7)");
        }
        Assert.Equal(0.0, Special.Bdtr(-1, 10, 0.5));
        Assert.Equal(1.0, Special.Bdtr(10, 10, 0.5));
        SpecialAssert.Close(0.3, Special.Bdtr(4, 15, Special.Bdtri(4, 15, 0.3)), 1e-14, "bdtri round trip");
        SpecialAssert.Close(0.6, Special.Pdtr(7, Special.Pdtri(7, 0.6)), 1e-14, "pdtri round trip");
        SpecialAssert.Close(0.25, Special.Nbdtr(3, 5, Special.Nbdtri(3, 5, 0.25)), 1e-14, "nbdtri round trip");
    }

    [Fact]
    public void FDistribution_InverseRoundTrips()
    {
        foreach (var (a, b) in new[] { (1.0, 1.0), (4.0, 17.0), (60.0, 3.5) })
            foreach (var p in new[] { 1e-6, 0.3, 0.95 })
                SpecialAssert.Close(p, Special.Fdtr(a, b, Special.Fdtri(a, b, p)), 1e-13, $"fdtr(fdtri({a},{b},{p}))");
    }

    [Fact]
    public void Kolmogorov_InverseRoundTripsAndLimits()
    {
        foreach (var p in new[] { 1e-10, 0.01, 0.3, 0.7, 0.999 })
            SpecialAssert.Close(p, Special.Kolmogorov(Special.KolmogorovInverse(p)), 1e-13, $"kolmogorov(kolmogi({p}))");
        Assert.Equal(1.0, Special.Kolmogorov(0));
        Assert.Equal(0.0, Special.Kolmogorov(double.PositiveInfinity));
        Assert.Equal(1.0 - 0.3, Special.Smirnov(1, 0.3));
        foreach (var p in new[] { 0.05, 0.5 })
            SpecialAssert.Close(p, Special.Smirnov(30, Special.SmirnovInverse(30, p)), 1e-12, $"smirnov(smirnovi(30, {p}))");
    }
}
