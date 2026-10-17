using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// TNC is a port of tnc.c, so beyond reaching the optimum it must take SciPy's path: the
/// iteration and evaluation counts below come from scipy 1.17.1 (minimize(..., method='TNC',
/// options={'maxfun': 200})) on the Schittkowski problems of scipy's own test_tnc.py.
/// </summary>
public class OptimizeTncTests
{
    private static double F1(double[] x) => 100 * Math.Pow(x[1] - x[0] * x[0], 2) + Math.Pow(1 - x[0], 2);
    private static double[] G1(double[] x) { var d1 = 200 * (x[1] - x[0] * x[0]); return [-2.0 * (x[0] * (d1 - 1.0) + 1.0), d1]; }
    private static double F3(double[] x) => x[1] + Math.Pow(x[1] - x[0], 2) * 1.0e-5;
    private static double[] G3(double[] x) { var d0 = -2.0 * (x[1] - x[0]) * 1.0e-5; return [d0, 1 - d0]; }
    private static double F4(double[] x) => Math.Pow(x[0] + 1.0, 3) / 3.0 + x[1];
    private static double[] G4(double[] x) => [Math.Pow(x[0] + 1.0, 2), 1.0];
    private static double F5(double[] x) => Math.Sin(x[0] + x[1]) + Math.Pow(x[0] - x[1], 2) - 1.5 * x[0] + 2.5 * x[1] + 1.0;
    private static double[] G5(double[] x) { var v1 = Math.Cos(x[0] + x[1]); var v2 = 2.0 * (x[0] - x[1]); return [v1 + v2 - 1.5, v1 - v2 + 2.5]; }
    private static double F38(double[] x) => (100.0 * Math.Pow(x[1] - x[0] * x[0], 2) + Math.Pow(1.0 - x[0], 2) + 90.0 * Math.Pow(x[3] - x[2] * x[2], 2)
        + Math.Pow(1.0 - x[2], 2) + 10.1 * (Math.Pow(x[1] - 1.0, 2) + Math.Pow(x[3] - 1.0, 2)) + 19.8 * (x[1] - 1.0) * (x[3] - 1.0)) * 1.0e-5;
    private static double[] G38(double[] x) =>
    [
        (-400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0])) * 1.0e-5,
        (200.0 * (x[1] - x[0] * x[0]) + 20.2 * (x[1] - 1.0) + 19.8 * (x[3] - 1.0)) * 1.0e-5,
        (-360.0 * x[2] * (x[3] - x[2] * x[2]) - 2.0 * (1.0 - x[2])) * 1.0e-5,
        (180.0 * (x[3] - x[2] * x[2]) + 20.2 * (x[3] - 1.0) + 19.8 * (x[1] - 1.0)) * 1.0e-5,
    ];
    private static double F45(double[] x) => 2.0 - x[0] * x[1] * x[2] * x[3] * x[4] / 120.0;
    private static double[] G45(double[] x)
    {
        var g = new double[5];
        for (var i = 0; i < 5; i++) { var p = 1.0; for (var j = 0; j < 5; j++) if (j != i) p *= x[j]; g[i] = -p / 120.0; }
        return g;
    }

    private const double Inf = double.PositiveInfinity;
    private static readonly TncOptions Opts = new() { MaxFun = 200 };

    private static Bounds B(params (double lo, double hi)[] b) => new(b.Select(t => t.lo).ToArray(), b.Select(t => t.hi).ToArray());

    public static TheoryData<string> Cases => ["t1", "t1fd", "t2", "t3", "t4", "t5", "t38", "t45", "rosb", "rosbfd", "ros5", "rosun"];

    [Theory]
    [MemberData(nameof(Cases))]
    public void Tnc_MatchesScipyCountsAndSolution(string name)
    {
        // name, f, g, x0, bounds, scipy (status, nit, nfev, x, fun). Where x0 lies outside the bounds
        // (t2, t45, ros5) scipy's nfev is one higher: its ScalarFunction evaluates the unclipped x0
        // before TNC clips it, and SciNet clips first, so "- 1" marks that evaluation.
        (Func<double[], double> f, Func<double[], double[]>? g, double[] x0, Bounds? b, int status, int nit, int nfev, double[] x, double fun) c = name switch
        {
            "t1" => (F1, G1, [-2, 1], B((-Inf, Inf), (-1.5, Inf)), 1, 15, 36, [0.9999997276833614, 0.9999994541585596], 7.430233540443515e-14),
            "t1fd" => (F1, null, [-2, 1], B((-Inf, Inf), (-1.5, Inf)), 1, 14, 129, [0.9999943151519535, 0.9999886132807015], 3.2346586398190585e-11),
            "t2" => (F1, G1, [-2, 1], B((-Inf, Inf), (1.5, Inf)), 1, 7, 18 - 1, [-1.2210262421278295, 1.5], 4.941229317989185),
            "t3" => (F3, G3, [10, 1], B((-Inf, Inf), (0, Inf)), 0, 2, 5, [-1.1102230246251565e-12, 0.0], 1.2325951644078311e-29),
            "t4" => (F4, G4, [1.125, 0.125], B((1, Inf), (0, Inf)), 0, 2, 5, [1.0, 0.0], 2.6666666666666665),
            "t5" => (F5, G5, [0, 0], B((-1.5, 4), (-3, 3)), 1, 7, 18, [-0.5471975479691253, -1.5471975495649515], -1.9132229549810362),
            "t38" => (F38, G38, [-3, -1, -3, -1], B((-10, 10), (-10, 10), (-10, 10), (-10, 10)), 1, 11, 43,
                [0.9960651915991068, 0.9922879776175296, 1.0031862327020913, 1.0063902463958123], 6.501792365931289e-10),
            "t45" => (F45, G45, [2, 2, 2, 2, 2], B((0, 1), (0, 2), (0, 3), (0, 4), (0, 5)), 0, 4, 10 - 1, [1, 2, 3, 4, 5], 1.0),
            "rosb" => (Optimize.Rosen, Optimize.RosenDer, [-1.2, 1], B((-Inf, 0.5), (-Inf, Inf)), 0, 12, 39, [0.5, 0.25000000017785295], 0.25),
            "rosbfd" => (Optimize.Rosen, null, [-1.2, 1], B((-Inf, 0.5), (-Inf, Inf)), 1, 11, 126, [0.5, 0.24999999500107806], 0.2500000000000025),
            "ros5" => (Optimize.Rosen, Optimize.RosenDer, [1.3, 0.7, 0.8, 1.9, 1.2], Bounds.Uniform(5, -2, 0.5), 1, 12, 42 - 1,
                [0.5, 0.26303738687073297, 0.07996076054720203, 0.016230978408048546, 0.0002632911044975206], 2.6456658878925126),
            "rosun" => (Optimize.Rosen, Optimize.RosenDer, [-1.2, 1], null, 1, 25, 83, [1.0000803946252816, 1.0001611308365799], 6.47452649798267e-09),
            _ => throw new ArgumentException(name),
        };
        var r = Optimize.Minimize(c.f, c.x0, MinimizeMethod.Tnc, jac: c.g, bounds: c.b, options: Opts);
        Assert.True(r.Status == c.status && r.Nit == c.nit && r.Nfev == c.nfev,
            $"{name}: status/nit/nfev = {r.Status}/{r.Nit}/{r.Nfev}, scipy {c.status}/{c.nit}/{c.nfev}");
        OptimizeAssert.Close(c.x, r.X, 1e-7, name + " x");
        OptimizeAssert.Close(c.fun, r.Fun, 1e-9 * Math.Max(1, Math.Abs(c.fun)), name + " fun");
        Assert.True(r.Success);
    }

    [Fact]
    public void Tnc_FollowsScipysPathWhenTheGradientIsBitIdentical()
    {
        // TNC's Hessian-vector products are gradient differences over a step of ~1e-8, so a one-ulp
        // difference in the gradient becomes a 1e-8 relative difference in the CG direction. With
        // scipy's own summation order for rosen_der the path is scipy's exactly
        // (scipy 1.17.1, maxfun=20000, gtol=1e-10: n=5 -> nit 28, nfev 114; n=10 -> nit 22, nfev 107).
        static double[] Der(double[] x)
        {
            var n = x.Length; var d = new double[n];
            for (var i = 1; i < n - 1; i++) d[i] = 200 * (x[i] - x[i - 1] * x[i - 1]) - 400 * (x[i + 1] - x[i] * x[i]) * x[i] - 2 * (1 - x[i]);
            d[0] = -400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]);
            d[n - 1] = 200 * (x[n - 1] - x[n - 2] * x[n - 2]);
            return d;
        }
        foreach (var (n, nit, nfev, fun) in new[] { (5, 28, 114, 2.876972955e-09), (10, 22, 107, 3.883587311e-09) })
        {
            var x0 = Enumerable.Range(0, n).Select(i => 2.0 * i / (n - 1)).ToArray();
            var r = Optimize.Minimize(Optimize.Rosen, x0, MinimizeMethod.Tnc, jac: Der, options: new TncOptions { MaxFun = 20000, Gtol = 1e-10 });
            Assert.True(r.Nit == nit && r.Nfev == nfev, $"n={n}: nit/nfev {r.Nit}/{r.Nfev}, scipy {nit}/{nfev}");
            OptimizeAssert.Relative(fun, r.Fun, 1e-6, $"n={n} fun");
        }
    }

    [Fact]
    public void Tnc_CallbackSeesEveryIteration()
    {
        var count = 0;
        var r = Optimize.Minimize(F1, [-2, 1], MinimizeMethod.Tnc, jac: G1, bounds: B((-Inf, Inf), (-1.5, Inf)), options: Opts,
            callback: (x, f) => { count++; return false; });
        Assert.Equal(r.Nit, count);
    }

    [Fact]
    public void Tnc_StopsAtMaxFunWithScipysStatus()
    {
        // scipy 1.17.1: rosen from [-1.2, 1] unbounded with maxfun=20 -> status 3, nit 6, nfev 21.
        var r = Optimize.Minimize(Optimize.Rosen, [-1.2, 1], MinimizeMethod.Tnc, jac: Optimize.RosenDer, options: new TncOptions { MaxFun = 20 });
        Assert.Equal(3, r.Status);
        Assert.False(r.Success);
        Assert.True(r.Nfev <= 21, r.ToString());
    }

    [Fact]
    public void Tnc_ScaleAndOffsetAreHonoured()
    {
        // A badly scaled quadratic: with scale = (1, 1e3) TNC sees a well-conditioned problem.
        Func<double[], double> f = x => Math.Pow(x[0] - 1, 2) + Math.Pow((x[1] - 2000) / 1000, 2);
        Func<double[], double[]> g = x => [2 * (x[0] - 1), 2 * (x[1] - 2000) / 1e6];
        var r = Optimize.Minimize(f, [0, 0], MinimizeMethod.Tnc, jac: g, options: new TncOptions { Scale = [1, 1000], Offset = [0, 0] });
        Assert.True(r.Success, r.ToString());
        OptimizeAssert.Close([1, 2000], r.X, 1e-4, "x");

        // A zero scale fixes the variable where it starts.
        var z = Optimize.Minimize(f, [0, 5], MinimizeMethod.Tnc, jac: g, options: new TncOptions { Scale = [1, 0] });
        Assert.Equal(5.0, z.X[1]);
        OptimizeAssert.Close(1, z.X[0], 1e-5, "x0");
    }
}
