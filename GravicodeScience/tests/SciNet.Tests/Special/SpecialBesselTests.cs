using Gravicode.Science.SciNet;
using Xunit;
using R = Gravicode.Science.Tests.SciNet.SpecialReference;

namespace Gravicode.Science.Tests.SciNet;

public class SpecialBesselTests
{
    [Fact] public void J0_MatchesReference() => SpecialAssert.Table(R.J0, a => Special.J0(a[0]), 2e-15, "J0");
    [Fact] public void J1_MatchesReference() => SpecialAssert.Table(R.J1, a => Special.J1(a[0]), 2e-15, "J1");
    [Fact] public void Y0_MatchesReference() => SpecialAssert.Table(R.Y0, a => Special.Y0(a[0]), 5e-15, "Y0");
    [Fact] public void Y1_MatchesReference() => SpecialAssert.Table(R.Y1, a => Special.Y1(a[0]), 2e-15, "Y1");
    [Fact] public void I0_MatchesReference() => SpecialAssert.Table(R.I0, a => Special.I0(a[0]), 2e-15, "I0");
    [Fact] public void I1_MatchesReference() => SpecialAssert.Table(R.I1, a => Special.I1(a[0]), 2e-15, "I1");
    [Fact] public void K0_MatchesReference() => SpecialAssert.Table(R.K0, a => Special.K0(a[0]), 2e-15, "K0");
    [Fact] public void K1_MatchesReference() => SpecialAssert.Table(R.K1, a => Special.K1(a[0]), 2e-15, "K1");
    [Fact] public void I0e_MatchesReference() => SpecialAssert.Table(R.I0e, a => Special.I0e(a[0]), 1e-15, "I0e");
    [Fact] public void I1e_MatchesReference() => SpecialAssert.Table(R.I1e, a => Special.I1e(a[0]), 1e-15, "I1e");
    [Fact] public void K0e_MatchesReference() => SpecialAssert.Table(R.K0e, a => Special.K0e(a[0]), 1e-15, "K0e");
    [Fact] public void K1e_MatchesReference() => SpecialAssert.Table(R.K1e, a => Special.K1e(a[0]), 1e-15, "K1e");
    [Fact] public void Jv_MatchesReference() => SpecialAssert.Table(R.Jv, a => Special.Jv(a[0], a[1]), 5e-15, "Jv");
    [Fact] public void Yv_MatchesReference() => SpecialAssert.Table(R.Yv, a => Special.Yv(a[0], a[1]), 1e-14, "Yv");
    [Fact] public void Iv_MatchesReference() => SpecialAssert.Table(R.Iv, a => Special.Iv(a[0], a[1]), 5e-15, "Iv");
    [Fact] public void Kv_MatchesReference() => SpecialAssert.Table(R.Kv, a => Special.Kv(a[0], a[1]), 5e-15, "Kv");
    [Fact] public void Ive_MatchesReference() => SpecialAssert.Table(R.Ive, a => Special.Ive(a[0], a[1]), 5e-15, "Ive");
    [Fact] public void Kve_MatchesReference() => SpecialAssert.Table(R.Kve, a => Special.Kve(a[0], a[1]), 5e-15, "Kve");
    [Fact] public void Jn_MatchesReference() => SpecialAssert.Table(R.Jn, a => Special.Jn((int)a[0], a[1]), 4e-15, "Jn");
    [Fact] public void Yn_MatchesReference() => SpecialAssert.Table(R.Yn, a => Special.Yn((int)a[0], a[1]), 3e-15, "Yn");
    [Fact] public void Kn_MatchesReference() => SpecialAssert.Table(R.Kn, a => Special.Kn((int)a[0], a[1]), 4e-15, "Kn");
    [Fact] public void SphericalJn_MatchesReference() => SpecialAssert.Table(R.SphericalJn, a => Special.SphericalJn((int)a[0], a[1]), 2e-14, "SphericalJn");
    [Fact] public void SphericalYn_MatchesReference() => SpecialAssert.Table(R.SphericalYn, a => Special.SphericalYn((int)a[0], a[1]), 3e-15, "SphericalYn");
    [Fact] public void SphericalIn_MatchesReference() => SpecialAssert.Table(R.SphericalIn, a => Special.SphericalIn((int)a[0], a[1]), 4e-15, "SphericalIn");
    [Fact] public void SphericalKn_MatchesReference() => SpecialAssert.Table(R.SphericalKn, a => Special.SphericalKn((int)a[0], a[1]), 2e-15, "SphericalKn");

    [Fact]
    public void Airy_MatchesReference()
        => SpecialAssert.Table(R.Airy, 4, a => { var r = Special.Airy(a[0]); return [r.Ai, r.Aip, r.Bi, r.Bip]; }, 2e-14, "Airy");

    [Fact]
    public void Zeros_MatchReference()
    {
        void Check(double[][] rows, Func<int, int, double[]> f, string name)
        {
            foreach (var row in rows)
            {
                var n = (int)row[0];
                var nt = (int)row[1];
                var z = f(n, nt);
                Assert.Equal(nt, z.Length);
                for (var i = 0; i < nt; i++) SpecialAssert.Close(row[2 + i], z[i], 3e-15, $"{name}({n}) zero {i + 1}");
            }
        }
        Check(R.JnZeros, Special.JnZeros, "jn_zeros");
        Check(R.JnpZeros, Special.JnpZeros, "jnp_zeros");
        Check(R.YnZeros, Special.YnZeros, "yn_zeros");
        Check(R.YnpZeros, Special.YnpZeros, "ynp_zeros");
    }

    [Fact]
    public void Wronskians_Hold()
    {
        // J_ν Y_(ν+1) - J_(ν+1) Y_ν = -2/(πx) and I_ν K_(ν+1) + I_(ν+1) K_ν = 1/x (DLMF 10.5.2, 10.28.2)
        foreach (var nu in new[] { 0.0, 0.4, 2.5, 17.0 })
            foreach (var x in new[] { 0.3, 3.0, 25.0, 140.0 })
            {
                var w = Special.Jv(nu, x) * Special.Yv(nu + 1, x) - Special.Jv(nu + 1, x) * Special.Yv(nu, x);
                SpecialAssert.Close(-2 / (Math.PI * x), w, 1e-12, $"J/Y Wronskian ν={nu}, x={x}");
                var wi = Special.Ive(nu, x) * Special.Kve(nu + 1, x) + Special.Ive(nu + 1, x) * Special.Kve(nu, x);
                SpecialAssert.Close(1 / x, wi, 1e-13, $"I/K Wronskian ν={nu}, x={x}");
            }
        foreach (var x in new[] { -8.0, -1.0, 0.0, 0.7, 3.0 })
        {
            var a = Special.Airy(x);
            SpecialAssert.Close(1 / Math.PI, a.Ai * a.Bip - a.Aip * a.Bi, 1e-14, $"Airy Wronskian at {x}");
        }
    }

    [Fact]
    public void Recurrences_And_ClosedForms()
    {
        foreach (var x in new[] { 0.5, 4.0, 30.0 })
        {
            for (var n = 1; n < 6; n++)
                SpecialAssert.Close(2 * n / x * Special.Jn(n, x), Special.Jn(n - 1, x) + Special.Jn(n + 1, x), 1e-13, $"J recurrence n={n} x={x}");
            SpecialAssert.Close(Math.Sin(x) / x, Special.SphericalJn(0, x), 1e-15, "j0");
            SpecialAssert.Close(Math.Sqrt(2 / (Math.PI * x)) * Math.Sin(x), Special.Jv(0.5, x), 1e-14, "J_1/2 closed form");
            SpecialAssert.Close(Math.Sqrt(Math.PI / (2 * x)) * Math.Exp(-x), Special.Kv(0.5, x), 1e-14, "K_1/2 closed form");
            // derivative flags: j_n' = j_(n-1) - (n+1)/x j_n
            SpecialAssert.Close(Special.SphericalJn(1, x) - 3 / x * Special.SphericalJn(2, x), Special.SphericalJn(2, x, derivative: true), 1e-14, "j_2'");
        }
        Assert.Equal(1.0, Special.Jv(0, 0));
        Assert.Equal(0.0, Special.Jv(2.5, 0));
        Assert.Equal(double.PositiveInfinity, Special.Jv(-0.5, 0));
        Assert.Equal(double.NegativeInfinity, Special.Jv(-1.5, 0));
        Assert.True(double.IsNaN(Special.Jv(2.5, -1)));
        SpecialAssert.Close(Special.Jv(2, 1), Special.Jv(2, -1), 1e-15, "J_2 even");
        Assert.True(double.IsNaN(Special.Yv(1, -1)));
        Assert.Equal(double.NegativeInfinity, Special.Y0(0));
        var h = Special.Hankel1(1.5, 2.0);
        Assert.Equal(Special.Jv(1.5, 2.0), h.Real);
        Assert.Equal(Special.Yv(1.5, 2.0), h.Imaginary);
        Assert.Equal(-h.Imaginary, Special.Hankel2(1.5, 2.0).Imaginary);
    }

    [Fact]
    public void AiryE_ScalesAiry()
    {
        foreach (var x in new[] { 0.5, 3.0, 20.0 })
        {
            var a = Special.Airy(x);
            var e = Special.AiryE(x);
            var z = 2.0 / 3.0 * x * Math.Sqrt(x);
            SpecialAssert.Close(a.Ai * Math.Exp(z), e.Ai, 1e-14, $"eAi({x})");
            SpecialAssert.Close(a.Bi * Math.Exp(-z), e.Bi, 1e-14, $"eBi({x})");
        }
        Assert.True(double.IsNaN(Special.AiryE(-1).Ai));
    }
}
