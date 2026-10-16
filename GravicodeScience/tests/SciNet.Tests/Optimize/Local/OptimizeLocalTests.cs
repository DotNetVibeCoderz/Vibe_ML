using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeLocalTests
{
    private static readonly double[] X5 = [1.3, 0.7, 0.8, 1.9, 1.2];

    private static double[] Start(int n)
    {
        if (n == 2) return [-1.2, 1.0];
        var x = new double[n];
        for (var i = 0; i < n; i++) x[i] = 2.0 * i / (n - 1);
        return x;
    }

    private static void AtOnes(OptimizeResult r, double tol, string what)
    {
        Assert.True(r.Success, $"{what}: {r}");
        foreach (var v in r.X) Assert.True(Math.Abs(v - 1) <= tol, $"{what}: x = [{string.Join(", ", r.X)}]");
    }

    // ----- parity with scipy 1.17.1 on minimize(rosen, [1.3, 0.7, 0.8, 1.9, 1.2]) -----

    [Fact]
    public void NelderMead_MatchesScipyTrajectory()
    {
        var r = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.NelderMead);
        Assert.Equal(141, r.Nit);
        Assert.Equal(243, r.Nfev);
        OptimizeAssert.Relative(6.617481708884532e-05, r.Fun, 1e-8, "fun");
        var a = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.NelderMead, options: new NelderMeadOptions { Adaptive = true });
        Assert.Equal(341, a.Nit);
        Assert.Equal(579, a.Nfev);
        OptimizeAssert.Relative(1.5946193790068098e-09, a.Fun, 1e-6, "adaptive fun");
    }

    [Fact]
    public void Powell_MatchesScipy()
    {
        var r = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.Powell);
        Assert.Equal(18, r.Nit);
        Assert.Equal(988, r.Nfev);
        AtOnes(r, 1e-10, "powell");
    }

    [Fact]
    public void Cg_And_Bfgs_MatchScipyCounts()
    {
        var cg = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.CG, jac: Optimize.RosenDer);
        Assert.Equal((47, 97, 97), (cg.Nit, cg.Nfev, cg.Njev));
        var cgfd = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.CG);
        Assert.Equal((65, 804, 134), (cgfd.Nit, cgfd.Nfev, cgfd.Njev));
        var b = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.Bfgs, jac: Optimize.RosenDer);
        Assert.Equal((25, 30, 30), (b.Nit, b.Nfev, b.Njev));
        OptimizeAssert.Relative(4.0130879821599437e-13, b.Fun, 1e-3, "bfgs fun");
        var bfd = Optimize.Minimize(Optimize.Rosen, X5);
        Assert.Equal((25, 180, 30), (bfd.Nit, bfd.Nfev, bfd.Njev));
        Assert.NotNull(b.HessInv);
    }

    [Fact]
    public void FunAndGrad_CountsEachPointOnce()
    {
        var calls = 0;
        var r = Optimize.Minimize((x, g) => { calls++; return Optimize.RosenDer(x, g); }, X5, MinimizeMethod.Bfgs);
        Assert.Equal(30, calls);
        Assert.Equal(30, r.Nfev);
        AtOnes(r, 1e-5, "fun-and-grad");
    }

    [Fact]
    public void NewtonCg_MatchesScipyCounts()
    {
        var a = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.NewtonCG, jac: Optimize.RosenDer, hess: Optimize.RosenHess);
        Assert.Equal((21, 30, 30, 21), (a.Nit, a.Nfev, a.Njev, a.Nhev));
        var b = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.NewtonCG, jac: Optimize.RosenDer, hessp: Optimize.RosenHessProd);
        Assert.Equal((21, 30, 30, 51), (b.Nit, b.Nfev, b.Njev, b.Nhev));
        var c = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.NewtonCG, jac: Optimize.RosenDer);
        Assert.Equal((21, 30, 111), (c.Nit, c.Nfev, c.Njev));
        AtOnes(c, 1e-3, "newton-cg fd");
    }

    [Fact]
    public void TrustRegion_MatchesScipyCounts()
    {
        var ncg = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.TrustNcg, jac: Optimize.RosenDer, hess: Optimize.RosenHess);
        Assert.Equal((18, 19, 18, 17), (ncg.Nit, ncg.Nfev, ncg.Njev, ncg.Nhev));
        var ncgp = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.TrustNcg, jac: Optimize.RosenDer, hessp: Optimize.RosenHessProd);
        // scipy reports nhev 62: it also counts one call of the placeholder Hessian its ScalarFunction makes when only hessp is given.
        Assert.Equal((18, 19, 18, 61), (ncgp.Nit, ncgp.Nfev, ncgp.Njev, ncgp.Nhev));
        var ex = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.TrustExact, jac: Optimize.RosenDer, hess: Optimize.RosenHess);
        Assert.Equal((12, 13, 12, 13), (ex.Nit, ex.Nfev, ex.Njev, ex.Nhev));
        AtOnes(ex, 1e-5, "trust-exact");
        var kr = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.TrustKrylov, jac: Optimize.RosenDer, hess: Optimize.RosenHess);
        AtOnes(kr, 1e-4, "trust-krylov");
        // Dogleg needs a positive-definite Hessian; at X5 it is not, which SciPy reports as status 3.
        var dl = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.Dogleg, jac: Optimize.RosenDer, hess: Optimize.RosenHess);
        Assert.Equal(3, dl.Status);
        var dl2 = Optimize.Minimize(Optimize.Rosen, [0.8, 0.6], MinimizeMethod.Dogleg, jac: Optimize.RosenDer, hess: Optimize.RosenHess);
        Assert.Equal((10, 11), (dl2.Nit, dl2.Nfev));
        AtOnes(dl2, 1e-7, "dogleg");
    }

    // ----- convergence on Rosenbrock in 2, 5 and 10 dimensions -----

    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(10)]
    public void EveryMethod_ConvergesOnRosenbrock(int n)
    {
        var x0 = Start(n);
        var nm = Optimize.Minimize(Optimize.Rosen, x0, MinimizeMethod.NelderMead,
            options: new NelderMeadOptions { Adaptive = true, MaxFev = 40000, Xatol = 1e-9, Fatol = 1e-12 });
        AtOnes(nm, 1e-6, $"nelder-mead {n}");
        AtOnes(Optimize.Minimize(Optimize.Rosen, x0, MinimizeMethod.Powell, tol: 1e-10), 1e-6, $"powell {n}");
        AtOnes(Optimize.Minimize(Optimize.Rosen, x0, MinimizeMethod.CG, jac: Optimize.RosenDer), 1e-4, $"cg {n}");
        AtOnes(Optimize.Minimize(Optimize.Rosen, x0, MinimizeMethod.Bfgs, jac: Optimize.RosenDer), 1e-5, $"bfgs {n}");
        AtOnes(Optimize.Minimize(Optimize.Rosen, x0, MinimizeMethod.NewtonCG, jac: Optimize.RosenDer, hess: Optimize.RosenHess, tol: 1e-8), 1e-5, $"newton-cg {n}");
        AtOnes(Optimize.Minimize(Optimize.Rosen, x0, MinimizeMethod.TrustNcg, jac: Optimize.RosenDer, hessp: Optimize.RosenHessProd, tol: 1e-8), 1e-6, $"trust-ncg {n}");
        AtOnes(Optimize.Minimize(Optimize.Rosen, x0, MinimizeMethod.TrustExact, jac: Optimize.RosenDer, hess: Optimize.RosenHess, tol: 1e-8), 1e-6, $"trust-exact {n}");
        AtOnes(Optimize.Minimize(Optimize.Rosen, x0, MinimizeMethod.TrustKrylov, jac: Optimize.RosenDer, hess: Optimize.RosenHess, tol: 1e-8), 1e-6, $"trust-krylov {n}");
    }

    [Fact]
    public void BoundedDirectSearch_MatchesScipy()
    {
        // scipy 1.17.1: bounds x0 in [1.5, 3], others free; both stop on their evaluation limits there.
        var b = Bounds.FromPairs((1.5, 3), (null, null), (null, null), (null, null), (null, null));
        var nm = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.NelderMead, bounds: b);
        Assert.Equal(1000, nm.Nfev);
        OptimizeAssert.Relative(69.7923718546748, nm.Fun, 1e-9, "nm fun");
        Assert.True(nm.X[0] >= 1.5);
        var pw = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.Powell, bounds: b);
        Assert.True(pw.X[0] >= 1.5);
        OptimizeAssert.Relative(69.25507059671129, pw.Fun, 1e-3, "powell fun");
    }

    [Fact]
    public void Callback_StopsWithStatus99()
    {
        var seen = 0;
        var r = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.Bfgs, jac: Optimize.RosenDer, callback: (_, _) => ++seen >= 3);
        Assert.Equal(99, r.Status);
        Assert.False(r.Success);
        Assert.Equal(3, r.Nit);
        var nm = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.NelderMead, callback: (_, _) => true);
        Assert.Equal(99, nm.Status);
    }

    [Fact]
    public void MaxIter_IsReported()
    {
        var r = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.Bfgs, jac: Optimize.RosenDer, options: new BfgsOptions { MaxIter = 3 });
        Assert.Equal(1, r.Status);
        Assert.Equal(3, r.Nit);
        Assert.Throws<ArgumentException>(() => Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.Bfgs, options: new CgOptions()));
    }
}
