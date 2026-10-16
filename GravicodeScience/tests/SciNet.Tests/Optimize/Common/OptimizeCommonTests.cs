using System.Numerics;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Assertion helpers shared by the Optimize test files.</summary>
internal static class OptimizeAssert
{
    public static void Close(double expected, double actual, double tol, string what = "")
        => Assert.True(Math.Abs(expected - actual) <= tol, $"{what}: expected {expected:R}, got {actual:R} (|diff| {Math.Abs(expected - actual):E2} > {tol:E1})");

    public static void Close(double[] expected, double[] actual, double tol, string what = "")
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++) Close(expected[i], actual[i], tol, $"{what}[{i}]");
    }

    public static void Relative(double expected, double actual, double rtol, string what = "")
        => Assert.True(Math.Abs(expected - actual) <= rtol * Math.Max(Math.Abs(expected), 1e-300),
            $"{what}: expected {expected:R}, got {actual:R} (rel {Math.Abs(expected - actual) / Math.Abs(expected):E2} > {rtol:E1})");
}

public class OptimizeCommonTests
{
    private static double Cubic(double x) { var t = x + 2; return (x - 2) * x * (t * t); }

    [Fact]
    public void Rosen_DerivativesMatchFiniteDifferences()
    {
        double[] x = [-1.2, 1.0, 0.7, 2.1];
        Assert.Equal(0.0, Optimize.Rosen([1.0, 1.0, 1.0]));
        Assert.True(Optimize.CheckGrad(Optimize.Rosen, Optimize.RosenDer, x) < 1e-4);
        var h = Optimize.RosenHess(x).ToArray();
        double[] p = [0.3, -1.0, 2.0, 0.5];
        var hp = Optimize.RosenHessProd(x, p);
        for (var i = 0; i < 4; i++)
        {
            double s = 0;
            for (var j = 0; j < 4; j++) s += h[i * 4 + j] * p[j];
            OptimizeAssert.Close(s, hp[i], 1e-10, "Hp");
        }
    }

    [Fact]
    public void MinimizeScalar_MatchesScipy()
    {
        // scipy 1.17.1: minimize_scalar((x-2)x(x+2)^2) brent / golden / bounded(-3,-1)
        var b = Optimize.MinimizeScalar(Cubic);
        OptimizeAssert.Close(1.2807764040333458, b.X, 1e-12, "brent x");
        Assert.Equal(11, b.Nit);
        Assert.Equal(14, b.Nfev);
        var g = Optimize.MinimizeScalar(Cubic, method: MinimizeScalarMethod.Golden);
        OptimizeAssert.Close(1.280776401465682, g.X, 1e-12, "golden x");
        Assert.Equal(43, g.Nfev);
        var bd = Optimize.MinimizeScalar(Cubic, bounds: (-3, -1));
        OptimizeAssert.Close(-2.000000202597239, bd.X, 1e-12, "bounded x");
        Assert.Equal(12, bd.Nfev);
        Assert.True(bd.Success);
    }

    [Fact]
    public void Bracket_MatchesScipy()
    {
        var br = Optimize.Bracket(Cubic);
        OptimizeAssert.Close(0.0, br.Xa, 0); OptimizeAssert.Close(1.0, br.Xb, 0);
        OptimizeAssert.Close(2.6180339999999998, br.Xc, 1e-15);
        Assert.Equal(3, br.Nfev);
        Assert.Throws<InvalidOperationException>(() => Optimize.Bracket(x => x, 0, 1, maxiter: 50));
    }

    [Fact]
    public void LineSearch_MatchesScipyOnRosenbrock()
    {
        double[] x = [-1.2, 1.0];
        var p = Optimize.RosenDer(x).Select(v => -v).ToArray();
        var r = Optimize.LineSearch(Optimize.Rosen, Optimize.RosenDer, x, p);
        // scipy 1.17.1 line_search: alpha 0.00093831027587526, fc 11, gc 1, new_fval 4.750587318071407
        Assert.NotNull(r.Alpha);
        OptimizeAssert.Close(0.00093831027587526, r.Alpha!.Value, 1e-14, "alpha");
        Assert.Equal(11, r.Fc);
        Assert.Equal(1, r.Gc);
        OptimizeAssert.Close(4.750587318071407, r.NewFval, 1e-12, "fval");
    }

    [Fact]
    public void ApproxDerivative_HonoursBoundsAndSchemes()
    {
        Func<double[], double> f = x => Math.Exp(x[0]) + x[1] * x[1] * x[1];
        double[] x0 = [0.5, 1.0];
        var exact = new[] { Math.Exp(0.5), 3.0 };
        OptimizeAssert.Close(exact, Optimize.ApproxDerivative(f, x0, FiniteDiffScheme.ThreePoint), 1e-9);
        OptimizeAssert.Close(exact, Optimize.ApproxDerivative(f, x0, FiniteDiffScheme.TwoPoint), 1e-6);
        // At the upper bound the forward step would leave the box: a backward step is used.
        var touched = new List<double>();
        Func<double[], double> guarded = x => { touched.Add(x[1]); return f(x); };
        var g = Optimize.ApproxDerivative(guarded, x0, FiniteDiffScheme.ThreePoint, bounds: new Bounds([0, 0], [2, 1]));
        Assert.All(touched, v => Assert.True(v <= 1.0));
        OptimizeAssert.Close(exact, g, 1e-8);
        var cs = Optimize.ApproxDerivativeComplexStep(z => Complex.Exp(z[0]) + z[1] * z[1] * z[1], x0, 1e-30);
        OptimizeAssert.Close(exact, cs, 1e-15);
    }

    [Fact]
    public void Minimize_RejectsUnsupportedCombinations()
    {
        Assert.Throws<ArgumentException>(() => Optimize.Minimize(Optimize.Rosen, [1.0, 2.0], MinimizeMethod.CG, bounds: Bounds.Uniform(2, 0, 1)));
        Assert.Throws<ArgumentException>(() => Optimize.Minimize(Optimize.Rosen, [1.0, 2.0], MinimizeMethod.Bfgs,
            constraints: [OptimizeConstraint.Ineq(x => x[0])]));
        Assert.Throws<ArgumentException>(() => Optimize.Minimize(Optimize.Rosen, [1.0, 2.0], MinimizeMethod.TrustNcg));
    }
}
