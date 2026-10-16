using System.Numerics;
using Gravicode.Science.SciNet;
using Xunit;
using R = Gravicode.Science.Tests.SciNet.SpecialReference;

namespace Gravicode.Science.Tests.SciNet;

public class SpecialErfTests
{
    [Fact]
    public void Erf_MatchesReference() => SpecialAssert.Table(R.Erf, a => Special.Erf(a[0]), 1e-15, "Erf");

    [Fact]
    public void Erfc_MatchesReference_DownToTheSubnormalTail() => SpecialAssert.Table(R.Erfc, a => Special.Erfc(a[0]), 1e-15, "Erfc");

    [Fact]
    public void Erfcx_MatchesReference() => SpecialAssert.Table(R.Erfcx, a => Special.Erfcx(a[0]), 1e-15, "Erfcx");

    [Fact]
    public void Erfi_MatchesReference() => SpecialAssert.Table(R.Erfi, a => Special.Erfi(a[0]), 4e-15, "Erfi");

    [Fact]
    public void Dawsn_MatchesReference() => SpecialAssert.Table(R.Dawsn, a => Special.Dawsn(a[0]), 4e-15, "Dawsn");

    [Fact]
    public void ErfInv_MatchesReference() => SpecialAssert.Table(R.ErfInv, a => Special.ErfInv(a[0]), 1e-15, "ErfInv");

    [Fact]
    public void ErfcInv_MatchesReference() => SpecialAssert.Table(R.ErfcInv, a => Special.ErfcInv(a[0]), 1e-15, "ErfcInv");

    [Fact]
    public void Ndtr_MatchesReference_FarIntoTheLowerTail() => SpecialAssert.Table(R.Ndtr, a => Special.Ndtr(a[0]), 2e-15, "Ndtr");

    [Fact]
    public void LogNdtr_MatchesReference_InBothTails() => SpecialAssert.Table(R.LogNdtr, a => Special.LogNdtr(a[0]), 1.5e-13, "LogNdtr");

    [Fact]
    public void Ndtri_MatchesReference() => SpecialAssert.Table(R.Ndtri, a => Special.Ndtri(a[0]), 1e-15, "Ndtri");

    [Fact]
    public void Fresnel_MatchesReference()
        => SpecialAssert.Table(R.Fresnel, 2, a => { var (s, c) = Special.Fresnel(a[0]); return [s, c]; }, 4e-15, "Fresnel");

    [Fact]
    public void OwensT_MatchesReference() => SpecialAssert.Table(R.OwensT, a => Special.OwensT(a[0], a[1]), 1e-14, "OwensT");

    [Fact]
    public void Wofz_MatchesReference()
        => SpecialAssert.Table(R.Wofz, 2, a => { var w = Special.Wofz(new Complex(a[0], a[1])); return [w.Real, w.Imaginary]; }, 4e-15, "Wofz");

    [Fact]
    public void VoigtProfile_MatchesReference() => SpecialAssert.Table(R.VoigtProfile, a => Special.VoigtProfile(a[0], a[1], a[2]), 4e-15, "VoigtProfile");

    [Fact]
    public void ErfAndErfc_AreComplementary_AndRoundTrip()
    {
        for (var x = -3.0; x <= 3.0; x += 0.173)
        {
            Assert.True(Math.Abs(Special.Erf(x) + Special.Erfc(x) - 1) < 3e-16, $"erf + erfc at {x}");
            // erfinv is ill-conditioned near ±1 (dx = dy e^(x²) √π/2), so the far side uses erfcinv.
            if (Math.Abs(x) < 1.5) SpecialAssert.Close(x, Special.ErfInv(Special.Erf(x)), 4e-15, $"erfinv(erf({x}))");
            else if (x > 0) SpecialAssert.Close(x, Special.ErfcInv(Special.Erfc(x)), 4e-15, $"erfcinv(erfc({x}))");
            SpecialAssert.Close(x, Special.Ndtri(Special.Ndtr(x)), 2e-14, $"ndtri(ndtr({x}))");
            Assert.True(Math.Abs(Special.Ndtr(x) + Special.Ndtr(-x) - 1) < 3e-16, $"Φ(x) + Φ(-x) at {x}");
        }
        foreach (var x in new[] { 0.6, 3.0, 9.0, 25.0 })
            SpecialAssert.Close(Special.Erfc(x), Math.Exp(-x * x) * Special.Erfcx(x), 1e-13, $"erfcx consistency at {x}");
    }

    [Fact]
    public void Erf_OddAndLimits()
    {
        Assert.Equal(-Special.Erf(0.7), Special.Erf(-0.7));
        Assert.Equal(1.0, Special.Erf(double.PositiveInfinity));
        Assert.Equal(2.0, Special.Erfc(double.NegativeInfinity));
        Assert.Equal(double.PositiveInfinity, Special.ErfInv(1));
        Assert.True(double.IsNaN(Special.ErfInv(1.5)));
        Assert.Equal(double.NegativeInfinity, Special.Ndtri(0));
        Assert.Equal(double.NegativeInfinity, Special.LogNdtr(double.NegativeInfinity));
    }

    [Fact]
    public void LogNdtr_StaysFiniteWhereNdtrUnderflows()
    {
        // log Φ(x) ≈ -x²/2 - log(-x √(2π)) for x → -∞; at x = -1e5 Φ itself is 0.
        Assert.Equal(0.0, Special.Ndtr(-1e5));
        var x = -1e5;
        SpecialAssert.Close(-0.5 * x * x - Math.Log(-x * Math.Sqrt(2 * Math.PI)), Special.LogNdtr(x), 1e-12, "asymptotic log Φ");
    }

    [Fact]
    public void Fresnel_IsOddAndTendsToOneHalf()
    {
        var (s, c) = Special.Fresnel(-1.3);
        var (s2, c2) = Special.Fresnel(1.3);
        Assert.Equal(-s2, s);
        Assert.Equal(-c2, c);
        var (sl, cl) = Special.Fresnel(1e9);
        Assert.True(Math.Abs(sl - 0.5) < 1e-9 && Math.Abs(cl - 0.5) < 1e-9);
    }

    [Fact]
    public void OwensT_ClosedForms()
    {
        foreach (var a in new[] { 0.3, 1.0, 4.0 })
            SpecialAssert.Close(Math.Atan(a) / (2 * Math.PI), Special.OwensT(0, a), 1e-15, $"T(0, {a})");
        foreach (var h in new[] { 0.2, 1.1, 3.0 })
        {
            // T(h, 1) = Φ(h)(1 - Φ(h))/2
            var p = Special.Ndtr(h);
            SpecialAssert.Close(0.5 * p * (1 - p), Special.OwensT(h, 1), 2e-14, $"T({h}, 1)");
        }
    }

    [Fact]
    public void Wofz_RealAxisAndSymmetry()
    {
        foreach (var x in new[] { 0.3, 2.0, 7.5 })
        {
            var w = Special.Wofz(new Complex(x, 0));
            SpecialAssert.Close(Math.Exp(-x * x), w.Real, 1e-15, $"Re w({x})");
            SpecialAssert.Close(2 / Math.Sqrt(Math.PI) * Special.Dawsn(x), w.Imaginary, 1e-15, $"Im w({x})");
        }
        var z = new Complex(1.3, 0.8);
        var a1 = Special.Wofz(-Complex.Conjugate(z));
        var a2 = Complex.Conjugate(Special.Wofz(z));
        Assert.True(Complex.Abs(a1 - a2) < 1e-15 * Complex.Abs(a2));
    }

    [Fact]
    public void NormalCdf_HasTheLimitsAtInfinity()
    {
        // Ndtr(-inf) used to be NaN: the double-double split computed inf - inf. Any probability
        // over a box with an open side goes through it, so the limits are pinned explicitly.
        Assert.Equal(0.0, Special.Ndtr(double.NegativeInfinity));
        Assert.Equal(1.0, Special.Ndtr(double.PositiveInfinity));
        Assert.Equal(double.NegativeInfinity, Special.LogNdtr(double.NegativeInfinity));
        Assert.Equal(0.0, Special.LogNdtr(double.PositiveInfinity));
        Assert.Equal(-1.0, Special.Erf(double.NegativeInfinity));
        Assert.Equal(2.0, Special.Erfc(double.NegativeInfinity));
        Assert.Equal(0.0, Special.Erfc(double.PositiveInfinity));
    }
}
