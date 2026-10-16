using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class DifferentiateTests
{
    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-2.5)]
    public void Derivative_Exp_IsAccurateAndErrorEstimateIsHonest(double x)
    {
        var r = Differentiate.Derivative(Math.Exp, x);
        Assert.True(r.Success, $"status {r.Status}");
        var trueErr = Math.Abs(r.Df - Math.Exp(x));
        Assert.True(trueErr <= 1e-12 * Math.Exp(x), $"df {r.Df:R} error {trueErr}");
        // The estimate is the last change, which for a converging sequence bounds the true error.
        Assert.True(trueErr <= Math.Max(r.Error, 1e-15 * Math.Exp(x)) * 10, $"reported {r.Error}, true {trueErr}");
        Assert.Equal(x, r.X);
    }

    [Fact]
    public void Derivative_SinAndPolynomial()
    {
        var s = Differentiate.Derivative(Math.Sin, 0.7);
        Assert.True(Math.Abs(s.Df - Math.Cos(0.7)) < 1e-13, $"{s.Df}");
        // Degree 5 polynomial: an order-8 stencil is exact up to rounding, converging immediately.
        var p = Differentiate.Derivative(t => ((((t - 3) * t + 2) * t - 1) * t + 4) * t - 7, 1.5);
        var exact = 5 * Math.Pow(1.5, 4) - 12 * Math.Pow(1.5, 3) + 6 * 1.5 * 1.5 - 2 * 1.5 + 4;
        Assert.True(Math.Abs(p.Df - exact) < 1e-12, $"{p.Df} vs {exact}");
        Assert.True(p.Nit <= 3, $"nit {p.Nit}");
    }

    [Fact]
    public void Derivative_NfevReusesStencilPoints()
    {
        var calls = 0;
        var r = Differentiate.Derivative(t => { calls++; return Math.Exp(t); }, 0.3);
        Assert.Equal(calls, r.Nfev);
        // Central order 8: four pairs initially, then one new pair per iteration.
        Assert.Equal(8 + 2 * (r.Nit - 1), r.Nfev);
    }

    [Fact]
    public void Derivative_OneSidedAtBoundary_NeverStepsOutside()
    {
        // sqrt'(x) at x=1 using only x >= 1; log at 0.5 using only x <= 0.5.
        var r = Differentiate.Derivative(t =>
        {
            Assert.True(t >= 1.0, $"evaluated at {t}");
            return Math.Sqrt(t);
        }, 1.0, stepDirection: 1, initialStep: 0.5);
        Assert.True(Math.Abs(r.Df - 0.5) < 1e-9, $"{r.Df} status {r.Status}");
        var b = Differentiate.Derivative(t => Math.Log(t), 0.5, stepDirection: -1, initialStep: 0.25);
        Assert.True(Math.Abs(b.Df - 2.0) < 1e-8, $"{b.Df}");
    }

    [Fact]
    public void Derivative_ReportsNonFiniteAndMaxIter()
    {
        var nan = Differentiate.Derivative(t => t < 0 ? double.NaN : t, 0.0);
        Assert.Equal(-3, nan.Status);
        Assert.False(nan.Success);
        var capped = Differentiate.Derivative(Math.Exp, 1.0, rtol: 0, atol: 0, maxIter: 2);
        Assert.Equal(-2, capped.Status);
        Assert.Equal(2, capped.Nit);
        // Demanding the impossible ends when round-off makes the estimate worse.
        var noisy = Differentiate.Derivative(Math.Exp, 1.0, rtol: 0, atol: 0, maxIter: 40);
        Assert.Equal(-1, noisy.Status);
        Assert.True(Math.Abs(noisy.Df - Math.E) < 1e-10, $"kept the better estimate {noisy.Df}");
    }

    [Fact]
    public void Jacobian_Gradient_Hessian_MatchAnalytic()
    {
        double[] x = [0.5, -1.2, 2.0];
        var j = Differentiate.Jacobian(v => [v[0] * v[1], Math.Sin(v[2]) + v[0] * v[0], Math.Exp(v[1] * v[2])], x);
        double[] expected =
        [
            x[1], x[0], 0,
            2 * x[0], 0, Math.Cos(x[2]),
            0, x[2] * Math.Exp(x[1] * x[2]), x[1] * Math.Exp(x[1] * x[2]),
        ];
        Assert.True(j.Success);
        Assert.Equal([3, 3], j.Df.Shape.ToArray());
        var got = j.Df.ToArray();
        for (var i = 0; i < 9; i++) Assert.True(Math.Abs(got[i] - expected[i]) < 1e-11, $"J[{i}] {got[i]} vs {expected[i]}");

        // Rosenbrock.
        static double Rosen(double[] v) => 100 * Math.Pow(v[1] - v[0] * v[0], 2) + Math.Pow(1 - v[0], 2);
        double[] p = [0.7, 0.4];
        var g = Differentiate.Gradient(Rosen, p);
        var gx = -400 * p[0] * (p[1] - p[0] * p[0]) - 2 * (1 - p[0]);
        var gy = 200 * (p[1] - p[0] * p[0]);
        Assert.True(Math.Abs(g.Df[0] - gx) < 1e-9 && Math.Abs(g.Df[1] - gy) < 1e-9, $"grad {g.Df}");

        var h = Differentiate.Hessian(Rosen, p);
        double[] he = [1200 * p[0] * p[0] - 400 * p[1] + 2, -400 * p[0], -400 * p[0], 200];
        var hg = h.Df.ToArray();
        for (var i = 0; i < 4; i++) Assert.True(Math.Abs(hg[i] - he[i]) < 1e-6, $"H[{i}] {hg[i]} vs {he[i]}");
        Assert.True(h.Nfev > 0);
    }

    [Fact]
    public void Fornberg_ReproducesClassicCentralWeights()
    {
        // 5-point central stencil on (-2,-1,1,2): (1, -8, 8, -1)/12.
        var w = Differentiate.FirstDerivativeWeights([-2, -1, 1, 2]);
        double[] e = [1.0 / 12, -8.0 / 12, 8.0 / 12, -1.0 / 12];
        for (var i = 0; i < 4; i++) Assert.True(Math.Abs(w[i] - e[i]) < 1e-15, $"w{i} = {w[i]}");
    }
}
