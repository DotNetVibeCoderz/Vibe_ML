using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeBoundedTests
{
    private static readonly double[] X5 = [1.3, 0.7, 0.8, 1.9, 1.2];

    private static double[] Start(int n)
    {
        if (n == 2) return [-1.2, -1.2];
        var x = new double[n];
        for (var i = 0; i < n; i++) x[i] = 2.0 * i / (n - 1);
        return x;
    }

    [Fact]
    public void LBfgsB_MatchesScipyOnRosenbrock()
    {
        // scipy 1.17.1: nit 24, nfev 26, "RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
        var r = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.LBfgsB, jac: Optimize.RosenDer);
        Assert.Equal((24, 26, 26), (r.Nit, r.Nfev, r.Njev));
        Assert.Equal("CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH", r.Message);
        Assert.True(r.Fun < 1e-10, r.ToString());
        var fd = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.LBfgsB);
        Assert.Equal((24, 156, 26), (fd.Nit, fd.Nfev, fd.Njev));
    }

    [Fact]
    public void LBfgsB_BoundActiveMatchesScipy()
    {
        // scipy 1.17.1: rosen on [-2, 0.5]^5 from X5 (clipped): nit 12, nfev 14, fun 2.645665887626918
        var b = Bounds.Uniform(5, -2, 0.5);
        var r = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.LBfgsB, jac: Optimize.RosenDer, bounds: b);
        Assert.Equal((12, 14), (r.Nit, r.Nfev));
        Assert.Equal("CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL", r.Message);
        OptimizeAssert.Close([0.5, 0.26303738318291764, 0.07996231058449414, 0.016231644140291213, 0.00026345598367933775], r.X, 1e-7, "x");
        OptimizeAssert.Close(2.645665887626918, r.Fun, 1e-12, "fun");
        Assert.Equal(1, r.ActiveMask![0]);
        Assert.Equal(0, r.ActiveMask[1]);
    }

    [Fact]
    public void LBfgsB_QuadraticWithKnownKkt()
    {
        // min ½xᵀAx − bᵀx on [0,1]³: KKT solution (0.25, 0, 1) — x1 at its lower bound with
        // positive gradient, x2 at its upper bound with negative gradient (scipy 1.17.1 agrees).
        double[] a = [4, 1, 0, 1, 3, 1, 0, 1, 2];
        double[] bv = [1, -2, 3];
        Func<double[], double[], double> fg = (x, g) =>
        {
            double f = 0;
            for (var i = 0; i < 3; i++)
            {
                double ax = 0;
                for (var j = 0; j < 3; j++) ax += a[i * 3 + j] * x[j];
                g[i] = ax - bv[i];
                f += 0.5 * x[i] * ax - bv[i] * x[i];
            }
            return f;
        };
        var r = Optimize.Minimize(fg, [0.0, 0.0, 0.0], MinimizeMethod.LBfgsB, bounds: Bounds.Uniform(3, 0, 1),
            options: new LBfgsBOptions { Gtol = 1e-12, Ftol = 1e-15 });
        OptimizeAssert.Close([0.25, 0.0, 1.0], r.X, 1e-12, "x");
        Assert.Equal([0, -1, 1], r.ActiveMask!);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(10)]
    public void LBfgsB_ConvergesOnRosenbrock(int n)
    {
        var r = Optimize.Minimize(Optimize.Rosen, Start(n), MinimizeMethod.LBfgsB, jac: Optimize.RosenDer, options: new LBfgsBOptions { Ftol = 1e-14, Gtol = 1e-9 });
        Assert.True(r.Success, r.ToString());
        foreach (var v in r.X) Assert.True(Math.Abs(v - 1) < 1e-6, r.ToString());
    }

    [Fact]
    public void LBfgsB_RemovesFixedVariablesAndHonoursLimits()
    {
        // scipy 1.17.1, x0 fixed at 1.3: fun 16.386786250467495, x[4] 12.00632660070436
        var b = Bounds.FromPairs((1.3, 1.3), (null, null), (null, null), (null, null), (null, null));
        var r = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.LBfgsB, bounds: b);
        Assert.Equal(1.3, r.X[0]);
        Assert.True(double.IsNaN(r.Jac![0]));
        OptimizeAssert.Close(16.386786250467495, r.Fun, 1e-6, "fun");
        OptimizeAssert.Close(12.00632660070436, r.X[4], 1e-3, "x4");

        var lim = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.LBfgsB, jac: Optimize.RosenDer, options: new LBfgsBOptions { MaxFun = 10 });
        Assert.Equal(1, lim.Status);
        Assert.Equal(11, lim.Nfev);
        Assert.Equal("STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT", lim.Message);

        var stop = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.LBfgsB, jac: Optimize.RosenDer, callback: (_, _) => true);
        Assert.Equal(99, stop.Status);
        Assert.Equal(1, stop.Nit);
    }

    [Fact]
    public void LBfgsB_InverseHessianApproximatesTheTruth()
    {
        // On a quadratic the stored pairs reproduce the true inverse along the directions explored.
        var a = new NdArray([3.0, 1.0, 1.0, 2.0], 2, 2);
        var r = Optimize.Minimize(x => 1.5 * x[0] * x[0] + x[0] * x[1] + x[1] * x[1] - x[0], [2.0, 2.0], MinimizeMethod.LBfgsB,
            jac: x => [3 * x[0] + x[1] - 1, x[0] + 2 * x[1]], options: new LBfgsBOptions { Gtol = 1e-9 });
        OptimizeAssert.Close([0.4, -0.2], r.X, 1e-6, "x");
        Assert.Equal((5, 6), (r.Nit, r.Nfev)); // scipy 1.17.1
        var hinv = LinAlg.Inverse(a).ToArray();
        var h = r.HessInv!.ToArray();
        for (var i = 0; i < 4; i++) OptimizeAssert.Close(hinv[i], h[i], 0.05, "Hinv");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(10)]
    public void Tnc_ConvergesOnRosenbrock(int n)
    {
        var r = Optimize.Minimize(Optimize.Rosen, Start(n), MinimizeMethod.Tnc, jac: Optimize.RosenDer,
            options: new TncOptions { MaxFun = 20000, Gtol = 1e-10 });
        Assert.True(r.Success, r.ToString());
        // TNC is a port of tnc.c now, and stops where tnc.c does: on the f-change test, which scipy
        // 1.17.1 meets 1.01e-4 from the optimum at n = 10 with these options. 1e-4 was the old
        // simplified method's accuracy, not scipy's.
        foreach (var v in r.X) Assert.True(Math.Abs(v - 1) < 1e-3, r.ToString());
    }

    [Fact]
    public void Tnc_FindsTheSameBoundActiveSolutions()
    {
        // Same optimum as scipy's L-BFGS-B and TNC (scipy 1.17.1) on rosen over [-2, 0.5]^5.
        var r = Optimize.Minimize(Optimize.Rosen, X5, MinimizeMethod.Tnc, jac: Optimize.RosenDer, bounds: Bounds.Uniform(5, -2, 0.5));
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close([0.5, 0.26303738318291764, 0.07996231058449414, 0.016231644140291213, 0.00026345598367933775], r.X, 1e-4, "x");
        OptimizeAssert.Close(2.645665887626918, r.Fun, 1e-7, "fun");
        Assert.Equal(1, r.ActiveMask![0]);

        double[] a = [4, 1, 0, 1, 3, 1, 0, 1, 2];
        double[] bv = [1, -2, 3];
        Func<double[], double> f = x =>
        {
            double s = 0;
            for (var i = 0; i < 3; i++) { double ax = 0; for (var j = 0; j < 3; j++) ax += a[i * 3 + j] * x[j]; s += 0.5 * x[i] * ax - bv[i] * x[i]; }
            return s;
        };
        var q = Optimize.Minimize(f, [0.5, 0.5, 0.5], MinimizeMethod.Tnc, bounds: Bounds.Uniform(3, 0, 1));
        Assert.True(q.Success, q.ToString());
        OptimizeAssert.Close([0.25, 0.0, 1.0], q.X, 1e-5, "x");
        Assert.Equal(0.0, q.X[1]);
        Assert.Equal(1.0, q.X[2]);

        // Fixed variables are removed before TNC runs, exactly as SciPy does.
        var fx = Optimize.Minimize(f, [0.5, 0.5, 0.5], MinimizeMethod.Tnc, bounds: new Bounds([0, 0.3, 0], [1, 0.3, 1]));
        Assert.Equal(0.3, fx.X[1]);
        Assert.True(fx.Success, fx.ToString());
    }
}
