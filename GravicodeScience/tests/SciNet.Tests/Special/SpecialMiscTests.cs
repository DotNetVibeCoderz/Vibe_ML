using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using R = Gravicode.Science.Tests.SciNet.SpecialReference;

namespace Gravicode.Science.Tests.SciNet;

public class SpecialMiscTests
{
    [Fact] public void Hyp0f1_MatchesReference() => SpecialAssert.Table(R.Hyp0f1, a => Special.Hyp0f1(a[0], a[1]), 4e-14, "Hyp0f1");
    [Fact] public void Hyp1f1_MatchesReference() => SpecialAssert.Table(R.Hyp1f1, a => Special.Hyp1f1(a[0], a[1], a[2]), 5e-15, "Hyp1f1");

    // The polynomial case (-3, 2.5, 1.5, 0.95) sums terms of order one to -0.004625: 7e-14 relative.
    [Fact] public void Hyp2f1_MatchesReference() => SpecialAssert.Table(R.Hyp2f1, a => Special.Hyp2f1(a[0], a[1], a[2], a[3]), 2e-13, "Hyp2f1");

    [Fact] public void Exp1_MatchesReference() => SpecialAssert.Table(R.Exp1, a => Special.Exp1(a[0]), 4e-15, "Exp1");
    [Fact] public void Expi_MatchesReference_ThroughItsZero() => SpecialAssert.Table(R.Expi, a => Special.Expi(a[0]), 2e-15, "Expi");
    [Fact] public void Expn_MatchesReference() => SpecialAssert.Table(R.Expn, a => Special.Expn((int)a[0], a[1]), 1e-14, "Expn");

    [Fact]
    public void Sici_MatchesReference()
        => SpecialAssert.Table(R.Sici, 2, a => { var (s, c) = Special.Sici(a[0]); return [s, c]; }, 1e-14, "Sici");

    [Fact]
    public void Shichi_MatchesReference()
        => SpecialAssert.Table(R.Shichi, 2, a => { var (s, c) = Special.Shichi(a[0]); return [s, c]; }, 2e-15, "Shichi");

    [Fact] public void Spence_MatchesReference() => SpecialAssert.Table(R.Spence, a => Special.Spence(a[0]), 2e-15, "Spence");
    [Fact] public void LambertW_MatchesReference() => SpecialAssert.Table(R.LambertW, a => Special.LambertW(a[0], (int)a[1]), 3e-15, "LambertW");
    [Fact] public void EllipK_MatchesReference() => SpecialAssert.Table(R.EllipK, a => Special.EllipK(a[0]), 2e-15, "EllipK");
    [Fact] public void EllipKm1_MatchesReference() => SpecialAssert.Table(R.EllipKm1, a => Special.EllipKm1(a[0]), 2e-15, "EllipKm1");
    [Fact] public void EllipE_MatchesReference() => SpecialAssert.Table(R.EllipE, a => Special.EllipE(a[0]), 3e-15, "EllipE");
    [Fact] public void EllipKinc_MatchesReference() => SpecialAssert.Table(R.EllipKinc, a => Special.EllipKinc(a[0], a[1]), 2e-15, "EllipKinc");
    [Fact] public void EllipEinc_MatchesReference() => SpecialAssert.Table(R.EllipEinc, a => Special.EllipEinc(a[0], a[1]), 2e-15, "EllipEinc");

    [Fact]
    public void EllipJ_MatchesReference()
        => SpecialAssert.Table(R.EllipJ, 4, a => { var r = Special.EllipJ(a[0], a[1]); return [r.Sn, r.Cn, r.Dn, r.Ph]; }, 2e-15, "EllipJ");

    [Fact] public void EllipRF_MatchesReference() => SpecialAssert.Table(R.EllipRF, a => Special.EllipRF(a[0], a[1], a[2]), 2e-15, "EllipRF");
    [Fact] public void EllipRD_MatchesReference() => SpecialAssert.Table(R.EllipRD, a => Special.EllipRD(a[0], a[1], a[2]), 2e-15, "EllipRD");
    [Fact] public void EllipRJ_MatchesReference_IncludingPrincipalValues() => SpecialAssert.Table(R.EllipRJ, a => Special.EllipRJ(a[0], a[1], a[2], a[3]), 2e-15, "EllipRJ");
    [Fact] public void EllipRC_MatchesReference() => SpecialAssert.Table(R.EllipRC, a => Special.EllipRC(a[0], a[1]), 2e-15, "EllipRC");
    [Fact] public void EllipRG_MatchesReference() => SpecialAssert.Table(R.EllipRG, a => Special.EllipRG(a[0], a[1], a[2]), 2e-15, "EllipRG");

    [Fact]
    public void Hypergeometric_ClosedForms()
    {
        foreach (var z in new[] { -5.0, -0.6, 0.3, 0.7, 0.99 })
        {
            SpecialAssert.Close(-Math.Log(1 - z) / z, Special.Hyp2f1(1, 1, 2, z), 2e-15, $"2F1(1,1;2;{z})");
            SpecialAssert.Close(Math.Pow(1 - z, -0.75), Special.Hyp2f1(0.75, 2.5, 2.5, z), 2e-15, $"2F1(a,b;b;{z})");
        }
        // Gauss's theorem at z = 1.
        SpecialAssert.Close(Special.Gamma(4) * Special.Gamma(1.5) / (Special.Gamma(3.5) * Special.Gamma(2)), Special.Hyp2f1(0.5, 2, 4, 1), 2e-15, "2F1 at 1");
        foreach (var x in new[] { -30.0, -1.0, 2.0, 50.0 })
            SpecialAssert.Close(Math.Exp(x), Special.Hyp1f1(1.7, 1.7, x), 1e-15, $"1F1(a;a;{x})");
        SpecialAssert.Close(Math.Cosh(2 * Math.Sqrt(3.0)), Special.Hyp0f1(0.5, 3), 2e-15, "0F1(;1/2;x) = cosh 2√x");
        SpecialAssert.Close(Math.Cos(2 * Math.Sqrt(30.0)), Special.Hyp0f1(0.5, -30), 1e-13, "0F1(;1/2;-x) = cos 2√x");
        Assert.True(double.IsNaN(Special.Hyp2f1(0.5, 0.5, 1.5, 2)));
    }

    [Fact]
    public void ExponentialIntegrals_Identities()
    {
        foreach (var x in new[] { 0.1, 1.0, 5.0 })
        {
            // E_(n+1)(x) = (e^-x - x E_n(x))/n
            for (var n = 1; n < 5; n++)
                SpecialAssert.Close((Math.Exp(-x) - x * Special.Expn(n, x)) / n, Special.Expn(n + 1, x), 1e-14, $"E{n + 1}({x})");
            var (shi, chi) = Special.Shichi(x);
            SpecialAssert.Close(Special.Expi(x), shi + chi, 1e-14, $"Shi + Chi = Ei at {x}");
        }
        Assert.Equal(double.PositiveInfinity, Special.Exp1(0));
        Assert.True(double.IsNaN(Special.Exp1(-1)));
        SpecialAssert.Close(-Special.Exp1(2.5), Special.Expi(-2.5), 1e-16, "Ei(-x) = -E1(x)");
        var (si, ci) = Special.Sici(1e8);
        Assert.True(Math.Abs(si - Math.PI / 2) < 1e-7 && Math.Abs(ci) < 1e-7);
    }

    [Fact]
    public void LambertW_InvertsWeW()
    {
        foreach (var x in new[] { -0.3678, -0.2, 1e-12, 0.5, 3.0, 1e10 })
        {
            var w = Special.LambertW(x);
            SpecialAssert.Close(x, w * Math.Exp(w), 1e-14, $"W0({x})");
        }
        foreach (var x in new[] { -0.3678, -0.1, -1e-8 })
        {
            var w = Special.LambertW(x, -1);
            Assert.True(w <= -1);
            SpecialAssert.Close(x, w * Math.Exp(w), 1e-14, $"W-1({x})");
        }
        SpecialAssert.Near(-1.0, Special.LambertW(-1 / Math.E), 1e-15);
        Assert.True(double.IsNaN(Special.LambertW(-0.5)));
        Assert.True(double.IsNaN(Special.LambertW(0.5, -1)));
        foreach (var (z, k) in new[] { (new Complex(1, 1), 0), (new Complex(-2, 0.5), 0), (new Complex(3, -4), 1), (new Complex(-0.2, 0.1), -1), (new Complex(0.2, 0), 2) })
        {
            var w = Special.LambertW(z, k);
            Assert.True(Complex.Abs(w * Complex.Exp(w) - z) < 1e-14 * Complex.Abs(z), $"W_{k}({z})");
        }
        // scipy.special.lambertw(3-4j, 1), SciPy 1.17.1
        var w1 = Special.LambertW(new Complex(3, -4), 1);
        Assert.True(Math.Abs(w1.Imaginary) > 2, "branch 1 lies above the principal strip");
    }

    [Fact]
    public void Elliptic_Identities()
    {
        SpecialAssert.Near(Math.PI / 2, Special.EllipK(0), 1e-15);
        SpecialAssert.Near(Math.PI / 2, Special.EllipE(0), 1e-15);
        Assert.Equal(1.0, Special.EllipE(1));
        Assert.Equal(double.PositiveInfinity, Special.EllipK(1));
        SpecialAssert.Close(1 / Math.Sqrt(2.5), Special.EllipRF(2.5, 2.5, 2.5), 1e-15, "RF(x,x,x)");
        SpecialAssert.Close(Math.Pow(2.5, -1.5), Special.EllipRD(2.5, 2.5, 2.5), 1e-15, "RD(x,x,x)");
        foreach (var m in new[] { 0.1, 0.5, 0.97 })
        {
            // Legendre's relation E K' + E' K - K K' = π/2
            var k = Special.EllipK(m);
            var e = Special.EllipE(m);
            var kp = Special.EllipK(1 - m);
            var ep = Special.EllipE(1 - m);
            SpecialAssert.Close(Math.PI / 2, e * kp + ep * k - k * kp, 1e-14, $"Legendre relation m={m}");
            SpecialAssert.Close(k, Special.EllipKinc(Math.PI / 2, m), 1e-15, "F(π/2|m) = K(m)");
            SpecialAssert.Close(e, Special.EllipEinc(Math.PI / 2, m), 1e-15, "E(π/2|m) = E(m)");
            foreach (var u in new[] { 0.3, 1.7, -4.0 })
            {
                var j = Special.EllipJ(u, m);
                Assert.True(Math.Abs(j.Sn * j.Sn + j.Cn * j.Cn - 1) < 1e-15);
                Assert.True(Math.Abs(j.Dn * j.Dn + m * j.Sn * j.Sn - 1) < 1e-15);
                SpecialAssert.Close(u, Special.EllipKinc(j.Ph, m), 1e-14, $"F(am(u)) = u at u={u}");
            }
        }
        // m → 1: K diverges like log(4/√(1-m)); ellipkm1 keeps it accurate.
        SpecialAssert.Close(Math.Log(4 / Math.Sqrt(1e-12)), Special.EllipKm1(1e-12), 1e-11, "K near m = 1");
    }

    [Fact]
    public void Elementary_FunctionsAreAccurateWhereNaiveFormsFail()
    {
        Assert.Equal(1e-20, Special.Log1p(1e-20));
        Assert.Equal(1e-20, Special.Expm1(1e-20));
        SpecialAssert.Close(-5e-21, Special.Cosm1(1e-10), 1e-15, "cosm1");
        Assert.Equal(-800.0, Special.LogExpit(-800));
        Assert.Equal(0.5, Special.Expit(0));
        Assert.Equal(1.0, Special.Expit(800));
        SpecialAssert.Close(-0.0004000000053332894, Special.Logit(0.4999), 1e-15, "logit near 1/2"); // scipy.special.logit
        SpecialAssert.Close(1.0, Special.Sinc(0), 0, "sinc 0");
        Assert.Equal(0.0, Special.Sinc(3));
        Assert.Equal(0.0, Special.Xlogy(0, 0));
        Assert.Equal(0.0, Special.Xlog1py(0, -1));
        Assert.Equal(0.0, Special.Entr(0));
        Assert.Equal(double.NegativeInfinity, Special.Entr(-1));
        Assert.Equal(1.0, Special.KlDiv(0, 1));
        Assert.Equal(double.PositiveInfinity, Special.RelEntr(1, 0));
        Assert.Equal(0.125, Special.Huber(1, 0.5));
        Assert.Equal(2.5, Special.Huber(1, 3));
        SpecialAssert.Close(5e-17, Special.PseudoHuber(1, 1e-8), 1e-15, "pseudo-Huber small r");
        foreach (var l in new[] { -0.7, 0.0, 0.5, 2.0 })
        {
            SpecialAssert.Close(3.7, Special.InvBoxcox(Special.Boxcox(3.7, l), l), 1e-15, $"boxcox λ={l}");
            SpecialAssert.Close(1e-9, Special.InvBoxcox1p(Special.Boxcox1p(1e-9, l), l), 1e-14, $"boxcox1p λ={l}");
        }
        Assert.Equal(1024.0, Special.Exp2(10));
        SpecialAssert.Near(1000.0, Special.Exp10(3), 1e-12);
        Assert.Equal(-3.0, Special.Cbrt(-27));
        SpecialAssert.Close(1.0, Special.ExpRel(1e-20), 0, "exprel 0");
    }

    [Fact]
    public void LogSumExp_IsStableAndSupportsWeights()
    {
        double[] a = [1000, 1000, 1000];
        SpecialAssert.Close(1000 + Math.Log(3), Special.LogSumExp(a), 1e-16, "lse large");
        SpecialAssert.Close(Math.Log(2 * Math.E - 0.5 * Math.Exp(2)), Special.LogSumExp([1.0, 2.0], [2.0, -0.5]), 1e-15, "weighted");
        Assert.True(double.IsNaN(Special.LogSumExp([1.0, 2.0], [1.0, -1.0])));
        var (v, s) = Special.LogSumExpWithSign([1.0, 2.0], [1.0, -1.0]);
        Assert.Equal(-1.0, s);
        SpecialAssert.Close(Math.Log(Math.Exp(2) - Math.E), v, 1e-15, "signed lse");
        Assert.Equal(double.NegativeInfinity, Special.LogSumExp([double.NegativeInfinity, double.NegativeInfinity]));
        var sm = Special.Softmax([1000.0, 1001.0, 999.0]);
        SpecialAssert.Close(1.0, sm.Sum(), 1e-15, "softmax sums to 1");
        SpecialAssert.Close(Math.Exp(1) / (1 + Math.E + Math.Exp(-1)) * Math.Exp(-0.0), sm[1], 1e-15, "softmax value");
        var ls = Special.LogSoftmax([1000.0, 1001.0, 999.0]);
        SpecialAssert.Close(Math.Log(sm[0]), ls[0], 1e-15, "log softmax");
    }

    [Fact]
    public void Apply_MatchesScalarCalls_SerialAndParallel()
    {
        var rng = new Random(7);
        var small = new double[100];
        var large = new double[50_000];
        for (var i = 0; i < small.Length; i++) small[i] = rng.NextDouble() * 20 - 5;
        for (var i = 0; i < large.Length; i++) large[i] = rng.NextDouble() * 20 - 5;
        foreach (var data in new[] { small, large })
        {
            var arr = new NdArray((double[])data.Clone(), data.Length);
            var g = Special.GammaLn(arr).AsSpan();
            var e = Special.Erfc(arr).AsSpan();
            for (var i = 0; i < data.Length; i++)
            {
                Assert.Equal(Special.GammaLn(data[i]), g[i]);
                Assert.Equal(Special.Erfc(data[i]), e[i]);
            }
        }
        // Shape is preserved, the input untouched, and strided views are handled.
        var m = new NdArray([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 2, 3);
        var t = m.Transpose();
        var r = Special.Apply(t, Special.Gamma);
        Assert.Equal(new[] { 3, 2 }, r.Shape.ToArray());
        Assert.Equal(Special.Gamma(1.5), r[1, 0]);
        Assert.Equal(Special.Gamma(4.5), r[1, 1]);
        Assert.Equal(0.5, m[0, 0]);
        var nd = Special.Ndtr(new NdArray([0.0], 1)).AsSpan();
        Assert.Equal(0.5, nd[0]);
    }
}
