using System.Diagnostics;
using Gravicode.Science.SciNet;
using Xunit;
using R = Gravicode.Science.Tests.SciNet.SpecialReference;

namespace Gravicode.Science.Tests.SciNet;

public class SpecialOrthogonalTests
{
    [Fact] public void EvalLegendre_MatchesReference() => SpecialAssert.Table(R.EvalLegendre, a => Special.EvalLegendre((int)a[0], a[1]), 2e-14, "EvalLegendre");
    [Fact] public void EvalHermite_MatchesReference() => SpecialAssert.Table(R.EvalHermite, a => Special.EvalHermite((int)a[0], a[1]), 5e-15, "EvalHermite");
    [Fact] public void EvalHermiteE_MatchesReference() => SpecialAssert.Table(R.EvalHermiteE, a => Special.EvalHermiteE((int)a[0], a[1]), 2e-15, "EvalHermiteE");
    [Fact] public void EvalLaguerre_MatchesReference() => SpecialAssert.Table(R.EvalLaguerre, a => Special.EvalLaguerre((int)a[0], a[1]), 4e-14, "EvalLaguerre");
    [Fact] public void EvalGenLaguerre_MatchesReference() => SpecialAssert.Table(R.EvalGenLaguerre, a => Special.EvalGenLaguerre((int)a[0], a[1], a[2]), 4e-14, "EvalGenLaguerre");
    [Fact] public void EvalChebyT_MatchesReference() => SpecialAssert.Table(R.EvalChebyT, a => Special.EvalChebyT((int)a[0], a[1]), 4e-15, "EvalChebyT");
    [Fact] public void EvalChebyU_MatchesReference() => SpecialAssert.Table(R.EvalChebyU, a => Special.EvalChebyU((int)a[0], a[1]), 2e-15, "EvalChebyU");
    [Fact] public void EvalGegenbauer_MatchesReference() => SpecialAssert.Table(R.EvalGegenbauer, a => Special.EvalGegenbauer((int)a[0], a[1], a[2]), 2e-14, "EvalGegenbauer");
    [Fact] public void EvalJacobi_MatchesReference() => SpecialAssert.Table(R.EvalJacobi, a => Special.EvalJacobi((int)a[0], a[1], a[2], a[3]), 4e-15, "EvalJacobi");
    [Fact] public void Lpmv_MatchesReference() => SpecialAssert.Table(R.Lpmv, a => Special.Lpmv(a[0], a[1], a[2]), 2e-14, "Lpmv");

    [Fact]
    public void SphHarmY_MatchesReference()
        => SpecialAssert.Table(R.SphHarmY, 2, a => { var y = Special.SphHarmY((int)a[0], (int)a[1], a[2], a[3]); return [y.Real, y.Imaginary]; }, 5e-15, "SphHarmY");

    [Fact]
    public void SphHarmY_IsOrthonormal()
    {
        // ∫∫ Y_n^m conj(Y_n'^m') sinθ dθ dφ = δ: Gauss–Legendre in cos θ, trapezoid in φ (exact for trig polynomials).
        var rule = Special.RootsLegendre(24);
        var pairs = new[] { (2, 1), (3, 1), (5, -2), (4, 0) };
        foreach (var (n1, m1) in pairs)
            foreach (var (n2, m2) in pairs)
            {
                System.Numerics.Complex s = 0;
                const int nphi = 32;
                for (var i = 0; i < rule.Count; i++)
                {
                    var theta = Math.Acos(rule.Nodes[i]);
                    for (var j = 0; j < nphi; j++)
                    {
                        var phi = 2 * Math.PI * j / nphi;
                        s += rule.Weights[i] * (2 * Math.PI / nphi) * Special.SphHarmY(n1, m1, theta, phi) * System.Numerics.Complex.Conjugate(Special.SphHarmY(n2, m2, theta, phi));
                    }
                }
                var expected = n1 == n2 && m1 == m2 ? 1.0 : 0.0;
                Assert.True(System.Numerics.Complex.Abs(s - expected) < 1e-13, $"<Y{n1}{m1}, Y{n2}{m2}> = {s}");
            }
        SpecialAssert.Close(0.5 / Math.Sqrt(Math.PI), Special.SphHarmY(0, 0, 1.0, 2.0).Real, 1e-16, "Y00");
    }

    [Fact]
    public void GaussRules_IntegratePolynomialsExactly()
    {
        // Legendre: ∫_-1^1 x^(2k) dx = 2/(2k+1), exact up to degree 2n - 1.
        var leg = Special.RootsLegendre(12);
        for (var k = 0; k < 12; k++)
            SpecialAssert.Close(2.0 / (2 * k + 1), leg.Integrate(x => Math.Pow(x, 2 * k)), 4e-15, $"Legendre x^{2 * k}");
        // Hermite: ∫ x^(2k) e^(-x²) = Γ(k + 1/2).
        var her = Special.RootsHermite(15);
        for (var k = 0; k < 15; k++)
            SpecialAssert.Close(Special.Gamma(k + 0.5), her.Integrate(x => Math.Pow(x, 2 * k)), 2e-14, $"Hermite x^{2 * k}");
        // Generalised Laguerre: ∫ x^k x^α e^-x = Γ(k + α + 1).
        var lag = Special.RootsGenLaguerre(10, 1.5);
        for (var k = 0; k < 20; k++)
            SpecialAssert.Close(Special.Gamma(k + 2.5), lag.Integrate(x => Math.Pow(x, k)), 2e-13, $"GenLaguerre x^{k}");
        // Jacobi mass: ∫ (1-x)^a (1+x)^b = 2^(a+b+1) B(a+1, b+1).
        var jac = Special.RootsJacobi(9, 0.3, -0.6);
        SpecialAssert.Close(Math.Pow(2, 0.3 - 0.6 + 1) * Special.Beta(1.3, 0.4), jac.Weights.Sum(), 1e-14, "Jacobi mass");
        SpecialAssert.Close(jac.Mu, jac.Weights.Sum(), 1e-14, "Jacobi Mu");
        SpecialAssert.Close(Math.PI, Special.RootsChebyT(7).Weights.Sum(), 1e-15, "ChebyT mass");
        SpecialAssert.Close(Math.PI / 2, Special.RootsChebyU(7).Weights.Sum(), 1e-15, "ChebyU mass");
        SpecialAssert.Close(Math.Sqrt(2 * Math.PI), Special.RootsHermiteE(11).Weights.Sum(), 1e-14, "HermiteE mass");
        // Gegenbauer: (1 - x²)^(α - 1/2), mass √π Γ(α + 1/2)/Γ(α + 1).
        var geg = Special.RootsGegenbauer(8, 1.75);
        SpecialAssert.Close(Math.Sqrt(Math.PI) * Special.Gamma(2.25) / Special.Gamma(2.75), geg.Weights.Sum(), 1e-14, "Gegenbauer mass");
    }

    [Fact]
    public void RootsLegendre_SmallRulesMatchClosedForms()
    {
        var r = Special.RootsLegendre(3);
        SpecialAssert.Close(-Math.Sqrt(0.6), r.Nodes[0], 1e-16, "n=3 node");
        Assert.Equal(0.0, r.Nodes[1]);
        SpecialAssert.Close(5.0 / 9, r.Weights[0], 2e-16, "n=3 weight");
        SpecialAssert.Close(8.0 / 9, r.Weights[1], 5e-16, "n=3 centre weight");
        var (x, w) = Special.RootsLegendre(1);
        Assert.Equal(0.0, x[0]);
        Assert.Equal(2.0, w[0]);
    }

    [Fact]
    public void RootsLegendre_Thousand_IsFastAndAccurate()
    {
        var sw = Stopwatch.StartNew();
        var r = Special.RootsLegendre(1000);
        sw.Stop();
        // Nodes: scipy.special.roots_legendre(1000), SciPy 1.17.1. Weights: mpmath at 30 digits,
        // 2/((1 - x²) P'_n(x)²), because SciPy's large-n weights are off (w[0] by 1.8e-8).
        SpecialAssert.Close(-0.9999971112980756, r.Nodes[0], 2e-16, "x[0]");
        SpecialAssert.Close(7.4133384164320715e-06, r.Weights[0], 1e-10, "w[0]");
        SpecialAssert.Close(-0.0015700104800832153, r.Nodes[499], 1e-13, "x[499]");
        SpecialAssert.Close(0.003140018380182868, r.Weights[499], 1e-14, "w[499]");
        SpecialAssert.Close(-0.7057176251892954, r.Nodes[250], 1e-15, "x[250]");
        SpecialAssert.Close(0.0022246841786682929, r.Weights[250], 1e-14, "w[250]"); // mpmath
        SpecialAssert.Close(2.0, r.Weights.Sum(), 1e-14, "mass");
        for (var i = 1; i < r.Count; i++) Assert.True(r.Nodes[i] > r.Nodes[i - 1]);
        Assert.True(sw.Elapsed.TotalSeconds < 5, $"took {sw.Elapsed}");
    }

    [Fact]
    public void RootsHermite_LargeOrderWeightsDoNotOverflow()
    {
        var r = Special.RootsHermite(200);
        // scipy.special.roots_hermite(200), SciPy 1.17.1
        SpecialAssert.Close(-19.33924866791141, r.Nodes[0], 1e-14, "x[0]");
        SpecialAssert.Close(2.229093496280268e-163, r.Weights[0], 1e-11, "w[0]");
        SpecialAssert.Near(0.07844190391741279, r.Nodes[100], 1e-14);
        SpecialAssert.Close(0.15592224233010094, r.Weights[100], 1e-13, "w[100]");
        SpecialAssert.Close(Math.Sqrt(Math.PI), r.Weights.Sum(), 1e-14, "mass");
    }

    [Fact]
    public void RootsFunctions_RejectBadArguments()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Special.RootsLegendre(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Special.RootsGenLaguerre(4, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Special.RootsJacobi(4, -1.5, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Special.RootsGegenbauer(4, -0.5));
    }
}
