using Gravicode.Science.SciNet;
using Xunit;
using R = Gravicode.Science.Tests.SciNet.SpecialReference;

namespace Gravicode.Science.Tests.SciNet;

public class SpecialIncompleteTests
{
    [Fact]
    public void GammaInc_MatchesReference_IncludingTheUniformAsymptoticRegion()
        => SpecialAssert.Table(R.GammaInc, a => Special.GammaInc(a[0], a[1]), 3e-15, "GammaInc");

    [Fact]
    public void GammaIncC_MatchesReference() => SpecialAssert.Table(R.GammaIncC, a => Special.GammaIncC(a[0], a[1]), 3e-15, "GammaIncC");

    [Fact]
    public void GammaIncInv_MatchesReference_DownTo1e300()
        => SpecialAssert.Table(R.GammaIncInv, a => Special.GammaIncInv(a[0], a[1]), 3e-14, "GammaIncInv");

    [Fact]
    public void GammaIncCInv_MatchesReference() => SpecialAssert.Table(R.GammaIncCInv, a => Special.GammaIncCInv(a[0], a[1]), 3e-15, "GammaIncCInv");

    [Fact]
    public void BetaInc_MatchesReference() => SpecialAssert.Table(R.BetaInc, a => Special.BetaInc(a[0], a[1], a[2]), 5e-15, "BetaInc");

    [Fact]
    public void BetaIncC_MatchesReference() => SpecialAssert.Table(R.BetaIncC, a => Special.BetaIncC(a[0], a[1], a[2]), 5e-15, "BetaIncC");

    [Fact]
    public void BetaIncInv_MatchesReference() => SpecialAssert.Table(R.BetaIncInv, a => Special.BetaIncInv(a[0], a[1], a[2]), 1e-13, "BetaIncInv");

    [Fact]
    public void GammaInc_ComplementsSumToOne_AndClosedForms()
    {
        foreach (var a in new[] { 0.3, 1.0, 4.5, 60.0, 900.0 })
            foreach (var f in new[] { 0.2, 0.9, 1.0, 1.3, 4.0 })
            {
                var x = a * f;
                Assert.True(Math.Abs(Special.GammaInc(a, x) + Special.GammaIncC(a, x) - 1) < 1e-15, $"P + Q at ({a}, {x})");
            }
        foreach (var x in new[] { 1e-6, 0.5, 3.0, 30.0 })
        {
            SpecialAssert.Close(-Special.Expm1(-x), Special.GammaInc(1, x), 3e-15, $"P(1, {x})");
            SpecialAssert.Close(Special.Erf(Math.Sqrt(x)), Special.GammaInc(0.5, x), 2e-15, $"P(1/2, {x}) = erf √x");
        }
        Assert.Equal(1.0, Special.GammaInc(0, 2.0));
        Assert.True(double.IsNaN(Special.GammaInc(-1, 2.0)));
    }

    [Fact]
    public void GammaIncInverses_RoundTrip()
    {
        foreach (var a in new[] { 0.05, 0.7, 3.0, 45.0, 2000.0 })
            foreach (var p in new[] { 1e-200, 1e-9, 0.02, 0.5, 0.97 })
            {
                if (a < 0.5 && p < 1e-100) continue; // x = (p Γ(a+1))^(1/a) underflows: 1e-4000
                // The round trip is limited by the conditioning of P in x: one ulp of x moves log P
                // by |a - x|·1e-16 (a = 2000 at 1e-200: ~3e-13).
                var x = Special.GammaIncInv(a, p);
                var tol = 5e-14 + 2e-15 * Math.Abs(a - x);
                SpecialAssert.Close(p, Special.GammaInc(a, x), tol, $"P(a, P⁻¹({a}, {p}))");
                var xc = Special.GammaIncCInv(a, p);
                tol = 5e-14 + 2e-15 * Math.Abs(a - xc);
                SpecialAssert.Close(p, Special.GammaIncC(a, xc), tol, $"Q(a, Q⁻¹({a}, {p}))");
            }
    }

    [Fact]
    public void BetaInc_SymmetryAndClosedForms()
    {
        foreach (var (a, b, x) in new[] { (0.5, 2.5, 0.3), (7.0, 3.0, 0.8), (40.0, 41.0, 0.49) })
        {
            SpecialAssert.Close(Special.BetaInc(a, b, x), 1 - Special.BetaInc(b, a, 1 - x), 1e-14, "I_x(a,b) = 1 - I_(1-x)(b,a)");
            Assert.True(Math.Abs(Special.BetaInc(a, b, x) + Special.BetaIncC(a, b, x) - 1) < 1e-15);
        }
        foreach (var (a, x) in new[] { (0.3, 0.2), (2.0, 0.6), (17.0, 0.95) })
            SpecialAssert.Close(Math.Pow(x, a), Special.BetaInc(a, 1, x), 2e-15, $"I_x({a}, 1) = x^a");
        Assert.Equal(1.0, Special.BetaInc(0, 2, 0.5));
        Assert.Equal(0.0, Special.BetaInc(2, 0, 0.5));
    }

    [Fact]
    public void BetaIncInverses_RoundTrip()
    {
        foreach (var (a, b) in new[] { (0.2, 0.3), (2.0, 5.0), (60.0, 7.0), (500.0, 800.0) })
            foreach (var p in new[] { 1e-30, 0.01, 0.5, 0.9, 1 - 1e-10 })
            {
                // With b = 0.3, I = 1 - 1e-10 needs 1 - x ~ 1e-33: x correctly rounds to 1.
                if (p > 1 - 1e-6 && b < 1) { Assert.Equal(1.0, Special.BetaIncInv(a, b, p)); continue; }
                var x = Special.BetaIncInv(a, b, p);
                var back = p > 0.5 ? 1 - Special.BetaIncC(a, b, x) : Special.BetaInc(a, b, x);
                SpecialAssert.Close(p, back, 2e-13, $"I(I⁻¹({a},{b},{p}))");
                // 1 - I = 1e-30 with b = 0.3 needs 1 - x ~ 1e-100, which a double x cannot hold.
                if (p < 1e-12 && b < 1) continue;
                var xc = Special.BetaIncCInv(a, b, p);
                // When the answer sits near 1, an ulp of x is a large relative change in 1 - x.
                var tol = 2e-13 + 2.2e-16 * b / (1 - xc);
                SpecialAssert.Close(p, Special.BetaIncC(a, b, xc), tol, $"Ic(Ic⁻¹({a},{b},{p}))");
            }
    }
}
