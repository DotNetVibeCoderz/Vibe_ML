using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

internal static class IntegrateAssert
{
    public static void Close(double expected, double actual, double tol, string what = "")
        => Assert.True(Math.Abs(expected - actual) <= tol, $"{what}: expected {expected:R}, got {actual:R} (|diff| {Math.Abs(expected - actual):E3} > {tol:E1})");

    public static void Rel(double expected, double actual, double rtol, string what = "")
        => Assert.True(Math.Abs(expected - actual) <= rtol * Math.Abs(expected), $"{what}: expected {expected:R}, got {actual:R} (rel {Math.Abs(expected - actual) / Math.Abs(expected):E3} > {rtol:E1})");
}

public class QuadTests
{
    // Reference (value, abserr, neval) triples below came from scipy 1.17.1 quad(..., full_output=1).

    [Fact]
    public void Qags_Polynomial_MatchesScipy()
    {
        var r = Integrate.Quad(x => x * x, 0, 4);
        IntegrateAssert.Close(21.333333333333336, r.Value, 4e-15, "value");
        IntegrateAssert.Rel(2.368475785867001e-13, r.AbsErr, 1e-6, "abserr");
        Assert.Equal(21, r.Neval);
        Assert.Equal(0, r.Ier);
    }

    [Fact]
    public void Qags_LogOverSqrt_EndpointSingularity()
    {
        // ∫0^1 log(x)/sqrt(x) dx = -4 exactly; scipy: (-4.000000000000085, 1.354472090042691e-13, neval 315).
        var r = Integrate.Quad(x => Math.Log(x) / Math.Sqrt(x), 0, 1);
        IntegrateAssert.Close(-4.0, r.Value, 1e-12, "value");
        IntegrateAssert.Close(-4.000000000000085, r.Value, 1e-13, "scipy value");
        IntegrateAssert.Rel(1.354472090042691e-13, r.AbsErr, 1e-3, "abserr");
        Assert.Equal(315, r.Neval);
        Assert.Equal(8, r.Last);
    }

    [Fact]
    public void Qagi_InfiniteRanges_MatchScipy()
    {
        var r = Integrate.Quad(x => Math.Exp(-x), 0, double.PositiveInfinity);
        IntegrateAssert.Close(1.0, r.Value, 1e-14, "exp");
        IntegrateAssert.Rel(5.842606703608969e-11, r.AbsErr, 1e-6, "exp abserr");
        Assert.Equal(135, r.Neval);
        var g = Integrate.Quad(x => Math.Exp(-x * x), double.NegativeInfinity, double.PositiveInfinity);
        IntegrateAssert.Close(Math.Sqrt(Math.PI), g.Value, 1e-14, "gauss");
        IntegrateAssert.Rel(1.4202636756659625e-08, g.AbsErr, 1e-6, "gauss abserr");
        Assert.Equal(270, g.Neval);
        var n = Integrate.Quad(Math.Exp, double.NegativeInfinity, 0);
        IntegrateAssert.Close(1.0, n.Value, 1e-14, "neg");
        Assert.Equal(135, n.Neval);
    }

    [Fact]
    public void Reversed_Limits_Negate()
    {
        var r = Integrate.Quad(x => x * x, 4, 0);
        IntegrateAssert.Close(-21.333333333333336, r.Value, 4e-15, "rev");
    }

    [Fact]
    public void Qagp_BreakPoint()
    {
        var r = Integrate.Quad(x => Math.Abs(x - 0.3), 0, 1, points: [0.3]);
        IntegrateAssert.Close(0.29, r.Value, 1e-15, "value");
        Assert.Equal(42, r.Neval);
    }

    [Fact]
    public void Oscillatory_And_Limit()
    {
        var r = Integrate.Quad(x => Math.Pow(Math.Sin(50 * x), 2), 0, Math.PI);
        IntegrateAssert.Close(Math.PI / 2, r.Value, 1e-13, "sin^2");
        Assert.Equal(147, r.Neval);
        // sin(1/x) hits the 50-subinterval limit in scipy too: (0.5044661486282698, 3.058623065107513e-4, 2079).
        var s = Integrate.Quad(x => Math.Sin(1 / x), 0.001, 1);
        Assert.Equal(1, s.Ier);
        Assert.Equal(2079, s.Neval);
        IntegrateAssert.Close(0.5044661486282698, s.Value, 1e-10, "sin(1/x)");
        IntegrateAssert.Rel(3.058623065107513e-4, s.AbsErr, 1e-4, "sin(1/x) err");
    }

    [Fact]
    public void Qawo_CosineWeight()
    {
        // ∫0^10 e^{-x} cos(20x) dx = (1 - e^{-10}(cos200 - 20 sin200)) / 401.
        var exact = (1 - Math.Exp(-10) * (Math.Cos(200) - 20 * Math.Sin(200))) / 401.0;
        var r = Integrate.Quad(x => Math.Exp(-x), 0, 10, weight: QuadWeight.Cos, wvar: [20]);
        IntegrateAssert.Close(exact, r.Value, 1e-12, "exact");
        IntegrateAssert.Close(0.0024917329900116525, r.Value, 1e-14, "scipy");
        var s = Integrate.Quad(_ => 1.0, 0, 100, weight: QuadWeight.Sin, wvar: [50]);
        IntegrateAssert.Close((1 - Math.Cos(5000)) / 50, s.Value, 1e-13, "sin");
        Assert.Equal(25, s.Neval);
    }

    [Fact]
    public void Qawf_FourierIntegrals()
    {
        var r = Integrate.Quad(x => Math.Exp(-x), 0, double.PositiveInfinity, weight: QuadWeight.Cos, wvar: [2]);
        IntegrateAssert.Close(0.2, r.Value, 1e-12, "cos");
        Assert.Equal(175, r.Neval);
        // ∫1^∞ sin(x)/x dx = π/2 - Si(1) = 0.6247132564795975...
        var s = Integrate.Quad(x => 1 / x, 1, double.PositiveInfinity, weight: QuadWeight.Sin, wvar: [1]);
        IntegrateAssert.Close(0.62471325642771360, s.Value, 1e-8, "sinc");
        IntegrateAssert.Close(0.6247132564795975, s.Value, 1e-12, "scipy");
        Assert.Equal(405, s.Neval);
        // Mirrored: ∫_{-∞}^{-1} sin(x)/x = same.
        var m = Integrate.Quad(x => 1 / x, double.NegativeInfinity, -1, weight: QuadWeight.Sin, wvar: [1]);
        IntegrateAssert.Close(s.Value, m.Value, 1e-12, "mirror");
    }

    [Fact]
    public void Qaws_AlgebraicLogWeights_MatchScipy()
    {
        var r = Integrate.Quad(_ => 1.0, 0, 1, weight: QuadWeight.Alg, wvar: [-0.5, -0.5]);
        IntegrateAssert.Close(Math.PI, r.Value, 1e-14, "beta(1/2,1/2)");
        Assert.Equal(50, r.Neval);
        IntegrateAssert.Close(-4.296154401698846, Integrate.Quad(Math.Exp, 0, 1, weight: QuadWeight.AlgLogA, wvar: [-0.5, 0.3]).Value, 1e-13, "loga");
        IntegrateAssert.Close(-9.64457717159517, Integrate.Quad(Math.Exp, 0, 1, weight: QuadWeight.AlgLogB, wvar: [0.2, -0.5]).Value, 1e-12, "logb");
        // ∫0^1 (1+x) log x log(1-x) dx: scipy 0.5325988997276604 (closed form 2 - π²/6 + ... checked by scipy).
        IntegrateAssert.Close(0.5325988997276604, Integrate.Quad(x => 1 + x, 0, 1, weight: QuadWeight.AlgLog, wvar: [0, 0]).Value, 1e-13, "loglog");
    }

    [Fact]
    public void Qawc_CauchyPrincipalValue()
    {
        // PV ∫_{-1}^{2} dx/(x - 0.5) = log(1.5/1.5) = 0.
        var r = Integrate.Quad(_ => 1.0, -1, 2, weight: QuadWeight.Cauchy, wvar: [0.5]);
        IntegrateAssert.Close(0.0, r.Value, 1e-13, "pv const");
        // PV ∫_{-1}^{1} e^x / x dx = 2 Shi(1) = 2.1145017507514415...
        var e = Integrate.Quad(Math.Exp, -1, 1, weight: QuadWeight.Cauchy, wvar: [0.0]);
        IntegrateAssert.Close(2.114501750751457, e.Value, 1e-13, "shi");
        Assert.Equal(25, e.Neval);
    }

    [Fact]
    public void OscillatoryMoments_AgreeWithPlainQuadrature()
    {
        // f = T_k on [-1, 1] is resolved exactly by the 25-point Chebyshev expansion, so a weighted
        // quad there is the moment itself: this pins both the boundary-value (|p| <= 24) and the
        // forward-recursion branches of the moment computation.
        foreach (var p in new[] { 2.5, 7.0, 23.9, 24.1, 60.0, 400.0 })
            for (var k = 0; k <= 24; k++)
            {
                var kk = k;
                double T(double x) => Math.Cos(kk * Math.Acos(Math.Clamp(x, -1, 1)));
                var weight = k % 2 == 0 ? QuadWeight.Cos : QuadWeight.Sin;
                var got = Integrate.Quad(T, -1, 1, weight: weight, wvar: [p]).Value;
                Func<double, double> g = kk % 2 == 0 ? x => T(x) * Math.Cos(p * x) : x => T(x) * Math.Sin(p * x);
                var exact = Integrate.Quad(g, -1, 1, epsabs: 1e-14, epsrel: 1e-13, limit: 500).Value;
                IntegrateAssert.Close(exact, got, 2e-12, $"p={p} k={k}");
            }
    }

    [Fact]
    public void Dblquad_Tplquad_Nquad()
    {
        IntegrateAssert.Close(0.010416666666666668, Integrate.Dblquad((y, x) => x * y, 0, 0.5, _ => 0, x => 1 - 2 * x).Value, 1e-15, "dbl");
        IntegrateAssert.Close(1.875, Integrate.Tplquad((z, y, x) => x * y * z, 1, 2, _ => 2, _ => 3, (_, _) => 0, (_, _) => 1).Value, 1e-14, "tpl");
        // ∫∫ over the triangle with ranges depending on the outer variable.
        var r = Integrate.Nquad(v => v[0] * v[1], new Func<double[], (double, double)>[] { o => (0, 1 - o[0]), _ => (0, 1) });
        IntegrateAssert.Close(1.0 / 24, r.Value, 1e-15, "nquad");
        var box = Integrate.Nquad(v => Math.Exp(-(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])), [(0, 1), (0, 1), (0, 1)]);
        IntegrateAssert.Close(Math.Pow(0.7468241328124270, 3), box.Value, 1e-13, "gauss box");
    }

    [Fact]
    public void QuadVec_VectorIntegrand()
    {
        // ∫0^2 x^k dx for k = 0..5 and an infinite one.
        var r = Integrate.QuadVec(x => [1, x, x * x, Math.Pow(x, 3), Math.Pow(x, 4), Math.Pow(x, 5)], 0, 2);
        Assert.True(r.Success);
        for (var k = 0; k < 6; k++) IntegrateAssert.Rel(Math.Pow(2, k + 1) / (k + 1), r.Value[k], 1e-12, $"k={k}");
        var inf = Integrate.QuadVec(x => [Math.Exp(-x), x * Math.Exp(-x)], 0, double.PositiveInfinity, epsrel: 1e-10);
        IntegrateAssert.Close(1, inf.Value[0], 1e-9, "e0");
        IntegrateAssert.Close(1, inf.Value[1], 1e-9, "e1");
        var both = Integrate.QuadVec(x => [Math.Exp(-x * x)], double.NegativeInfinity, double.PositiveInfinity, epsrel: 1e-10, norm: QuadVecNorm.Max);
        IntegrateAssert.Close(Math.Sqrt(Math.PI), both.Value[0], 1e-9, "gauss");
        var kink = Integrate.QuadVec(x => [Math.Abs(x - 0.3)], 0, 1, points: [0.3]);
        IntegrateAssert.Close(0.29, kink.Value[0], 1e-12, "kink");
    }
}
