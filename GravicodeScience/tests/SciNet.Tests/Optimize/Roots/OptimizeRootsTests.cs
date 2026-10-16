using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeRootsTests
{
    private static double Cubic(double x) => x * x * x - 2 * x - 5;
    private const double CubicRoot = 2.0945514815423265;

    // (method, function, a, b, scipy 1.17.1 root, iterations, function calls)
    public static TheoryData<string, string, double, int, int> BracketCases => new()
    {
        { "bisect", "cubic", 2.0945514815412025, 39, 41 },
        { "brentq", "cubic", 2.094551481542327, 7, 8 },
        { "brenth", "cubic", 2.0945514815423265, 7, 8 },
        { "ridder", "cubic", 2.094551481543327, 6, 14 },
        { "bisect", "mult", 1.000000000000364, 40, 42 },
        { "brentq", "mult", 1.0000000000001106, 9, 10 },
        { "brenth", "mult", 1.000000000000009, 9, 10 },
        { "ridder", "mult", 1.0000000000010003, 6, 14 },
        { "bisect", "flat", 2.2737367544323206e-13, 42, 44 },
        { "ridder", "flat", 9.768673574825568e-13, 35, 72 },
        { "bisect", "exp", 0.6931471805601177, 41, 43 },
        { "brentq", "exp", 0.6931471805600378, 10, 11 },
        { "brenth", "exp", 0.6931471805599453, 9, 10 },
        { "ridder", "exp", 0.6931471805609318, 5, 12 },
    };

    private static (Func<double, double> F, double A, double B) Problem(string name) => name switch
    {
        "cubic" => (Cubic, 2, 3),
        "mult" => (x => (x - 1) * (x - 2) * (x - 3), 0.5, 1.7),
        "flat" => (x => Math.Pow(x, 9), -1, 4),
        _ => (x => Math.Exp(x) - 2, -1, 3),
    };

    [Theory]
    [MemberData(nameof(BracketCases))]
    public void BracketMethods_MatchScipyExactly(string method, string fn, double root, int iterations, int calls)
    {
        var (f, a, b) = Problem(fn);
        var r = method switch
        {
            "bisect" => Optimize.Bisect(f, a, b),
            "brentq" => Optimize.Brentq(f, a, b),
            "brenth" => Optimize.Brenth(f, a, b),
            _ => Optimize.Ridder(f, a, b),
        };
        Assert.True(r.Converged);
        OptimizeAssert.Close(root, r.Root, 1e-15 * Math.Max(1, Math.Abs(root)), $"{method} {fn}");
        Assert.Equal((iterations, calls), (r.Iterations, r.FunctionCalls));
    }

    [Fact]
    public void Brent_FailsOnX9LikeScipy()
    {
        // scipy 1.17.1: brentq and brenth do not converge on x^9 over [-1, 4] in 100 iterations.
        var r = Optimize.Brentq(x => Math.Pow(x, 9), -1, 4, throwOnFailure: false);
        Assert.False(r.Converged);
        Assert.Equal((100, 102), (r.Iterations, r.FunctionCalls));
        Assert.Throws<InvalidOperationException>(() => Optimize.Brenth(x => Math.Pow(x, 9), -1, 4));
        Assert.Throws<ArgumentException>(() => Optimize.Brentq(Cubic, 3, 4));
    }

    [Fact]
    public void Toms748_MatchesScipy()
    {
        var c1 = Optimize.Toms748(Cubic, 2, 3);
        Assert.Equal((5, 11), (c1.Iterations, c1.FunctionCalls));
        // The interpolation agrees with scipy to the last bit or two, which moves the final midpoint by ~1e-13.
        OptimizeAssert.Close(2.0945514815425765, c1.Root, 1e-12, "toms k1");
        var c2 = Optimize.Toms748(Cubic, 2, 3, k: 2);
        Assert.Equal((3, 9), (c2.Iterations, c2.FunctionCalls));
        var e1 = Optimize.Toms748(x => Math.Exp(x) - 2, -1, 3);
        Assert.Equal((5, 12), (e1.Iterations, e1.FunctionCalls));
        OptimizeAssert.Close(Math.Log(2), e1.Root, 1e-15, "toms exp");
        var fl = Optimize.Toms748(x => Math.Pow(x, 9), -1, 4);
        Assert.Equal((34, 92), (fl.Iterations, fl.FunctionCalls));
        var m = Optimize.Toms748(x => (x - 1) * (x - 2) * (x - 3), 0.5, 1.7);
        Assert.Equal(1.0, m.Root);
        Assert.Equal((5, 10), (m.Iterations, m.FunctionCalls));
    }

    [Fact]
    public void Newton_Secant_Halley_MatchScipy()
    {
        var n = Optimize.Newton(Cubic, 2.0, x => 3 * x * x - 2);
        Assert.Equal((4, 8), (n.Iterations, n.FunctionCalls));
        OptimizeAssert.Close(CubicRoot, n.Root, 1e-15, "newton");
        var s = Optimize.Newton(Cubic, 2.0);
        Assert.Equal((5, 6), (s.Iterations, s.FunctionCalls));
        var h = Optimize.Newton(Cubic, 2.0, x => 3 * x * x - 2, fprime2: x => 6 * x);
        Assert.Equal((3, 9), (h.Iterations, h.FunctionCalls));
        Assert.Equal(RootScalarMethod.Halley, h.Method);
        var s2 = Optimize.Newton(Cubic, 2.0, x1: 2.5);
        Assert.Equal((6, 7), (s2.Iterations, s2.FunctionCalls));
        Assert.Throws<InvalidOperationException>(() => Optimize.Newton(x => x * x + 1, 0.5, x => 2 * x, maxiter: 20));
    }

    [Fact]
    public void Newton_Vectorised()
    {
        double[] c = [1, 4, 9];
        var (r1, ok1) = Optimize.Newton(x => x.Select((v, i) => v * v - c[i]).ToArray(), [1.5, 1.5, 1.5], x => x.Select(v => 2 * v).ToArray());
        OptimizeAssert.Close([1, 2, 3], r1, 1e-12, "newton array");
        Assert.All(ok1, Assert.True);
        var (r2, _) = Optimize.Newton(x => x.Select((v, i) => v * v - c[i]).ToArray(), [1.5, 1.5, 1.5]);
        OptimizeAssert.Close([1, 2, 3], r2, 1e-10, "secant array");
    }

    [Fact]
    public void FixedPoint_MatchesScipyDocExample()
    {
        double[] c1 = [10, 12], c2 = [3, 5];
        var p = Optimize.FixedPoint(x => x.Select((v, i) => Math.Sqrt(c1[i] / (v + c2[i]))).ToArray(), [1.2, 1.3]);
        OptimizeAssert.Close([1.4920333, 1.37228132], p, 1e-7, "fixed point");
        var q = Optimize.FixedPoint(Math.Cos, 1.0);
        OptimizeAssert.Close(0.7390851332151607, q, 1e-9, "dottie");
        var it = Optimize.FixedPoint(Math.Cos, 1.0, useDel2: false, maxiter: 2000);
        OptimizeAssert.Close(0.7390851332151607, it, 1e-7, "plain");
    }

    [Fact]
    public void RootScalar_SelectsMethodsLikeScipy()
    {
        Assert.Equal(RootScalarMethod.Brentq, Optimize.RootScalar(Cubic, bracket: (2, 3)).Method);
        Assert.Equal(RootScalarMethod.Newton, Optimize.RootScalar(Cubic, x0: 2, fprime: x => 3 * x * x - 2).Method);
        Assert.Equal(RootScalarMethod.Halley, Optimize.RootScalar(Cubic, x0: 2, fprime: x => 3 * x * x - 2, fprime2: x => 6 * x).Method);
        Assert.Equal(RootScalarMethod.Secant, Optimize.RootScalar(Cubic, x0: 2, x1: 3).Method);
        var t = Optimize.RootScalar(Cubic, RootScalarMethod.Toms748, (2, 3));
        OptimizeAssert.Close(CubicRoot, t.Root, 1e-12, "toms");
        Assert.False(Optimize.RootScalar(x => Math.Pow(x, 9), RootScalarMethod.Brentq, (-1, 4)).Converged);
        Assert.Throws<ArgumentException>(() => Optimize.RootScalar(Cubic));
    }

    // ----- systems: scipy's root tutorial, solution (0.8411639019140096, 0.1588360980859903) -----

    private static double[] F(double[] x) => [x[0] + 0.5 * Math.Pow(x[0] - x[1], 3) - 1.0, 0.5 * Math.Pow(x[1] - x[0], 3) + x[1]];

    private static NdArray J(double[] x)
    {
        var d = x[0] - x[1];
        return new NdArray([1 + 1.5 * d * d, -1.5 * d * d, -1.5 * d * d, 1 + 1.5 * d * d], 2, 2);
    }

    private static readonly double[] Sol = [0.8411639019140096, 0.1588360980859903];

    [Fact]
    public void Root_HybrMatchesScipy()
    {
        var r = Optimize.Root(F, [0, 0], RootMethod.Hybr, J);
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close(Sol, r.X, 1e-14, "hybrj");
        // scipy 1.17.1 reports 12 (and 14 below): its two input-validation calls of F are counted too.
        Assert.Equal(10, r.Nfev);
        var fd = Optimize.Root(F, [0, 0]);
        OptimizeAssert.Close(Sol, fd.X, 1e-12, "hybrd");
        Assert.Equal(12, fd.Nfev);
        var (x, info, _) = Optimize.Fsolve(F, [0, 0]);
        Assert.Equal(1, info);
        OptimizeAssert.Close(Sol, x, 1e-12, "fsolve");
    }

    [Theory]
    [InlineData(RootMethod.Broyden1, 24)]
    [InlineData(RootMethod.Broyden2, 26)]
    [InlineData(RootMethod.Anderson, 27)]
    [InlineData(RootMethod.DiagBroyden, 31)]
    [InlineData(RootMethod.DfSane, 9)]
    public void Root_QuasiNewtonMatchScipy(RootMethod method, int scipyNfev)
    {
        var r = Optimize.Root(F, [0, 0], method);
        Assert.True(r.Success, r.ToString());
        // The default stopping rule is max-norm residual ≤ 6e-6, so x is only good to ~1e-5.
        OptimizeAssert.Close(Sol, r.X, 2e-5, method.ToString());
        Assert.Equal(scipyNfev, r.Nfev);
    }

    [Fact]
    public void Root_MixingMethodsFailWhereScipyFails()
    {
        // scipy 1.17.1 reports failure for linearmixing / excitingmixing from (0, 0) on this system.
        Assert.False(Optimize.Root(F, [0, 0], RootMethod.LinearMixing).Success);
        Assert.False(Optimize.Root(F, [0, 0], RootMethod.ExcitingMixing).Success);
        // ...while linear mixing converges on a contraction.
        // (SciPy's mixing methods model the Jacobian as -I/alpha, so F = g(x) - x is the natural sign.)
        var c = Optimize.Root(x => [0.5 * Math.Cos(x[1]) - x[0], 0.5 * Math.Sin(x[0]) - x[1]], [0, 0], RootMethod.LinearMixing, options: new RootOptions { Alpha = 1.0 });
        Assert.True(c.Success, c.ToString());
    }

    [Fact]
    public void Root_KrylovSolvesDiscretisedBratu()
    {
        // -u'' = λ eᵘ on (0,1), u(0) = u(1) = 0, λ = 1: the lower branch has max u ≈ 0.140539.
        const int n = 60;
        var h = 1.0 / (n + 1);
        double[] Res(double[] u)
        {
            var r = new double[n];
            for (var i = 0; i < n; i++)
            {
                var ul = i > 0 ? u[i - 1] : 0; var ur = i < n - 1 ? u[i + 1] : 0;
                r[i] = (2 * u[i] - ul - ur) / (h * h) - Math.Exp(u[i]);
            }
            return r;
        }
        var k = Optimize.Root(Res, new double[n], RootMethod.Krylov, options: new RootOptions { Fatol = 1e-9 });
        Assert.True(k.Success, k.ToString());
        Assert.True(Math.Abs(k.X.Max() - 0.140539) < 2e-4, $"max u = {k.X.Max()}");
        var hy = Optimize.Root(Res, new double[n]);
        OptimizeAssert.Close(hy.X, k.X, 1e-8, "krylov vs hybr");
    }

    [Fact]
    public void Root_LmUsesTheMinpackCore()
    {
        var r = Optimize.Root(F, [0, 0], RootMethod.Lm, J);
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close(Sol, r.X, 1e-14, "lm");
        Assert.Equal(7, r.Nfev); // scipy reports 9, including its two validation calls
    }
}
