using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeLegacyTests
{
    private static readonly (double[] S, double[] Y)[] Pairs =
    [
        ([1, 0.5, 0], [2, 1.2, 0.1]),
        ([0.1, -0.3, 0.2], [0.3, -0.5, 0.6]),
        ([0.2, 0.1, -0.1], [0.5, 0.25, -0.1]),
    ];

    private static void Close(double[,] expected, NdArray actual, double tol, string what)
    {
        var a = actual.AsContiguous().ToArray();
        var n = expected.GetLength(0);
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++) OptimizeAssert.Close(expected[i, j], a[i * n + j], tol, $"{what}[{i},{j}]");
    }

    [Fact]
    public void BfgsHessianUpdate_MatchesScipyAndSatisfiesSecant()
    {
        var b = new BfgsHessianUpdate();
        b.Initialize(3, HessianApproxType.Hess);
        foreach (var (s, y) in Pairs) b.Update(s, y);
        // scipy 1.17.1: BFGS() with the same three updates
        Close(new[,]
        {
            { 2.6778539220618023, 0.3508546277025415, 0.7065624718261458 },
            { 0.3508546277025415, 1.8520252685392997, 0.05373452394438305 },
            { 0.7065624718261458, 0.05373452394438305, 2.466859467596675 },
        }, b.GetMatrix(), 1e-13, "B");
        // The last update satisfies the secant equation B s = y exactly.
        OptimizeAssert.Close(Pairs[2].Y, b.Dot(Pairs[2].S), 1e-13, "secant");
    }

    [Fact]
    public void Sr1AndInverseBfgs_MatchScipy()
    {
        var s1 = new Sr1HessianUpdate();
        s1.Initialize(3, HessianApproxType.Hess);
        var h = new BfgsHessianUpdate();
        h.Initialize(3, HessianApproxType.InvHess);
        for (var k = 0; k < 2; k++) { s1.Update(Pairs[k].S, Pairs[k].Y); h.Update(Pairs[k].S, Pairs[k].Y); }
        Close(new[,]
        {
            { 2.6187327068894266, 0.20679913734232047, 0.5008323525687672 },
            { 0.20679913734232047, 1.2253558255339123, -0.7653658303702917 },
            { 0.5008323525687672, -0.7653658303702917, 1.6015350781601785 },
        }, s1.GetMatrix(), 1e-13, "SR1");
        Close(new[,]
        {
            { 0.48139771034266454, 0.00018466243236886504, -0.0738783031443582 },
            { 0.00018466243236886504, 0.6123253352152436, 0.010178781463185183 },
            { -0.0738783031443582, 0.010178781463185183, 0.3787548027915001 },
        }, h.GetMatrix(), 1e-13, "H");
        OptimizeAssert.Close(Pairs[1].S, h.Dot(Pairs[1].Y), 1e-13, "inverse secant H y = s");
    }

    [Fact]
    public void LbfgsInvHessProduct_MatchesScipy()
    {
        var l = new LbfgsInvHessProduct([Pairs[0].S, Pairs[1].S], [Pairs[0].Y, Pairs[1].Y]);
        Close(new[,]
        {
            { 0.564883711374096, -0.11801158777120316, -0.21411817882971726 },
            { -0.11801158777120316, 0.8271486686390532, 0.24829635108481257 },
            { -0.2141181788297173, 0.24829635108481257, 0.6473060486522025 },
        }, l.ToDense(), 1e-13, "LbfgsInvHessProduct");
        OptimizeAssert.Close(Pairs[1].S, l.Matvec(Pairs[1].Y), 1e-13, "H y = s");
    }

    [Fact]
    public void ScalarWrappers()
    {
        // scipy 1.17.1: brent/golden → 1.9999999999999998, fminbound(sin, 0, 6) → 4.71239046949766
        OptimizeAssert.Close(2, Optimize.Brent(x => (x - 2) * (x - 2) + 1).X, 1e-8, "brent");
        OptimizeAssert.Close(2, Optimize.Golden(x => (x - 2) * (x - 2) + 1).X, 1e-7, "golden");
        OptimizeAssert.Close(4.71239046949766, Optimize.Fminbound(Math.Sin, 0, 6).X, 1e-9, "fminbound");
        OptimizeAssert.Close(3 * Math.PI / 2, Optimize.Fminbound(Math.Sin, 0, 6, xatol: 1e-10).X, 1e-8, "fminbound tight");
    }

    [Fact]
    public void LegacyFminWrappers()
    {
        OptimizeAssert.Close([1, 1], Optimize.Fmin(Optimize.Rosen, [-1.2, 1], xtol: 1e-8, ftol: 1e-10).X, 1e-6, "fmin");
        OptimizeAssert.Close([1, 1], Optimize.FminPowell(Optimize.Rosen, [-1.2, 1], xtol: 1e-8, ftol: 1e-10).X, 1e-5, "powell");
        OptimizeAssert.Close([1, 1], Optimize.FminBfgs(Optimize.Rosen, [-1.2, 1], Optimize.RosenDer).X, 1e-5, "bfgs");
        OptimizeAssert.Close([1, 1], Optimize.FminCg(Optimize.Rosen, [-1.2, 1], Optimize.RosenDer).X, 1e-5, "cg");
        OptimizeAssert.Close([1, 1], Optimize.FminNcg(Optimize.Rosen, [-1.2, 1], Optimize.RosenDer, fhess: Optimize.RosenHess, avextol: 1e-8).X, 1e-6, "ncg");
        var lb = Optimize.FminLBfgsB(Optimize.Rosen, [-1.2, 1], Optimize.RosenDer, Bounds.FromPairs((null, 0.5), (null, null)));
        OptimizeAssert.Close(0.5, lb.X[0], 1e-6, "lbfgsb active bound");
        OptimizeAssert.Close(0.25, lb.X[1], 1e-5, "lbfgsb x1");
        // TNC is a port of tnc.c: scipy 1.17.1's fmin_tnc takes 39 evaluations here, and so does SciNet.
        var tnc = Optimize.FminTnc(Optimize.Rosen, [-1.2, 1], Optimize.RosenDer, Bounds.FromPairs((null, 0.5), (null, null)));
        OptimizeAssert.Close(0.5, tnc.X[0], 1e-6, "tnc");
        Assert.Equal(39, tnc.Nfev);

        // The SLSQP/COBYLA tutorial problem: optimum (1.4, 1.7).
        Func<double[], double> f = x => (x[0] - 1) * (x[0] - 1) + (x[1] - 2.5) * (x[1] - 2.5);
        Func<double[], double>[] cons = [x => x[0] - 2 * x[1] + 2, x => -x[0] - 2 * x[1] + 6, x => -x[0] + 2 * x[1] + 2, x => x[0], x => x[1]];
        OptimizeAssert.Close([1.4, 1.7], Optimize.FminSlsqp(f, [2, 0], ieqcons: cons).X, 1e-6, "slsqp");
        OptimizeAssert.Close([1.4, 1.7], Optimize.FminCobyla(f, [2, 0], cons, rhoend: 1e-7).X, 1e-5, "cobyla");
        var eq = Optimize.FminSlsqp(x => x[0] * x[0] + x[1] * x[1], [3, -1], eqcons: [x => x[0] + x[1] - 1]);
        OptimizeAssert.Close([0.5, 0.5], eq.X, 1e-7, "slsqp eq");
    }

    [Fact]
    public void Leastsq_MatchesScipyIncludingCovariance()
    {
        var xd = Enumerable.Range(0, 12).Select(i => 4.0 * i / 11).ToArray();
        var yd = xd.Select(x => 2.5 * Math.Exp(-1.3 * x) + 0.5 + 0.01 * Math.Sin(7 * x)).ToArray();
        var r = Optimize.Leastsq(p => xd.Select((x, i) => p[0] * Math.Exp(-p[1] * x) + p[2] - yd[i]).ToArray(), [1, 1, 0]);
        Assert.InRange(r.Ier, 1, 4);
        // scipy 1.17.1
        OptimizeAssert.Close([2.500593364570159, 1.3014354963882246, 0.5006970415698191], r.X, 1e-7, "x");
        Close(new[,]
        {
            { 0.9611057064821331, 0.12399632452649866, -0.1716902143124061 },
            { 0.12399632452649866, 1.4613176041366371, 0.4442776534644465 },
            { -0.1716902143124061, 0.4442776534644465, 0.2645276392466337 },
        }, r.CovX!, 1e-5, "cov_x");
    }

    private static double[] Fun(double[] x) => [x[0] + 0.5 * Math.Pow(x[0] - x[1], 3) - 1.0, 0.5 * Math.Pow(x[1] - x[0], 3) + x[1]];

    [Fact]
    public void NonlinFunctionalInterfaces()
    {
        // scipy docs example; scipy 1.17.1 broyden1 → [0.84116396, 0.15883641]
        double[] root = [0.8411639, 0.1588361];
        OptimizeAssert.Close(root, Optimize.Broyden1(Fun, [0, 0]), 2e-6, "broyden1");
        OptimizeAssert.Close(root, Optimize.Broyden2(Fun, [0, 0]), 2e-6, "broyden2");
        OptimizeAssert.Close(root, Optimize.Anderson(Fun, [0, 0]), 5e-6, "anderson");
        OptimizeAssert.Close(root, Optimize.NewtonKrylov(Fun, [0.5, 0.5]), 2e-6, "newton_krylov");
        // A contraction for the mixing schemes, written T(x) − x as their J ≈ −1/α convention expects:
        // x = ½cos y, y = ½sin x.
        Func<double[], double[]> g = x => [0.5 * Math.Cos(x[1]) - x[0], 0.5 * Math.Sin(x[0]) - x[1]];
        foreach (var sol in new[] { Optimize.LinearMixing(g, [0, 0], alpha: 0.5), Optimize.DiagBroyden(g, [0, 0]), Optimize.ExcitingMixing(g, [0, 0], alpha: 0.5) })
            Assert.True(g(sol).Max(Math.Abs) < 1e-5, $"residual {g(sol).Max(Math.Abs)}");
    }

    [Fact]
    public void NonlinThrowsNoConvergence()
    {
        var ex = Assert.Throws<OptimizeNoConvergenceException>(() => Optimize.Broyden1(Fun, [0, 0], maxiter: 2));
        Assert.Equal(2, ex.X.Length);
        Assert.False(ex.Result.Success);
    }
}
