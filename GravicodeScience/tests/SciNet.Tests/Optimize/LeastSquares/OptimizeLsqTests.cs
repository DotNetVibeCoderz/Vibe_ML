using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class OptimizeLsqTests
{
    // numpy builds linspace as i*step and its sin/exp may differ in the last ulp, so the data differ
    // from scipy's by ~1e-16 and parameters agree to ~1e-8 rather than to rounding.
    // Deterministic "noisy" exponential decay: y = 2.5 e^{-1.3x} + 0.5 + 0.2 sin(7.3 i), x = linspace(0, 4, 50).
    private static readonly double[] Xd = Enumerable.Range(0, 50).Select(i => 4.0 * i / 49).ToArray();
    private static readonly double[] Yd = Enumerable.Range(0, 50).Select(i => 2.5 * Math.Exp(-1.3 * Xd[i]) + 0.5 + 0.2 * Math.Sin(7.3 * i)).ToArray();

    private static double Model(double x, double[] p) => p[0] * Math.Exp(-p[1] * x) + p[2];

    private static void Cov(double[] expected, NdArray actual, double rtol, string what)
    {
        var a = actual.ToArray();
        for (var i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - a[i]) <= rtol * Math.Abs(expected[i]) + 1e-12, $"{what}[{i}]: {expected[i]} vs {a[i]}");
    }

    [Fact]
    public void CurveFit_LmMatchesScipy()
    {
        // scipy 1.17.1 curve_fit(f, x, y): popt, pcov, nfev 21
        var r = Optimize.CurveFit(Model, Xd, Yd, nParams: 3);
        OptimizeAssert.Close([2.5633230073334787, 1.3238687324585485, 0.4954014248569238], r.Popt, 2e-8, "popt");
        Cov([0.007405827710726433, 0.0030565009894062993, -0.00037827010145888995, 0.0030565009894062993, 0.009158287462022338,
            0.0025857227783759704, -0.00037827010145888995, 0.0025857227783759704, 0.0013871072324158431], r.Pcov, 1e-6, "pcov");
        Assert.Equal(21, r.Nfev);
    }

    [Fact]
    public void CurveFit_TrfDogboxAndBoundsMatchScipy()
    {
        var t = Optimize.CurveFit(Model, Xd, Yd, nParams: 3, method: LeastSquaresMethod.Trf);
        OptimizeAssert.Close([2.5633239639507197, 1.3238715941582704, 0.49540223148253987], t.Popt, 2e-8, "trf popt");
        Cov([0.007406161822461596, 0.0030569244033906394, -0.00037821588048948747, 0.0030569244033906394, 0.00915903136940029,
            0.002585605986419718, -0.0003782158804894874, 0.0025856059864197175, 0.0013869599055556952], t.Pcov, 1e-6, "trf pcov");
        Assert.Equal(7, t.Nfev);
        var d = Optimize.CurveFit(Model, Xd, Yd, nParams: 3, method: LeastSquaresMethod.Dogbox);
        OptimizeAssert.Close([2.5633239549933395, 1.3238715692291945, 0.4954022245695539], d.Popt, 2e-8, "dogbox popt");
        Assert.Equal(7, d.Nfev);
        var b = Optimize.CurveFit(Model, Xd, Yd, bounds: new Bounds([0, 0, 0], [3, 1, 0.5]));
        OptimizeAssert.Close([2.4597788624149226, 0.9999999999999999, 0.3766370027605589], b.Popt, 1e-8, "bounded popt");
        Assert.Equal(8, b.Nfev);
        Cov([0.007857535203153949, 0.0014886464581500094, -0.001153774000489783, 0.0014886464581500094, 0.008503342203431394,
            0.004294137306302198, -0.001153774000489783, 0.004294137306302198, 0.0031768851058504056], b.Pcov, 1e-5, "bounded pcov");
    }

    [Fact]
    public void CurveFit_SigmaAndAbsoluteSigma()
    {
        var sigma = Enumerable.Range(0, 50).Select(i => 0.1 + 0.01 * i).ToArray();
        var r = Optimize.CurveFit(Model, Xd, Yd, nParams: 3, sigma: sigma);
        OptimizeAssert.Close([2.5643756838034677, 1.3654374723232614, 0.5131630452096629], r.Popt, 5e-8, "popt");
        Cov([0.003796670902455695, -0.001899350194717795, -0.0026335417089289235, -0.001899350194717795, 0.007718698619686185,
            0.004035959909795274, -0.0026335417089289235, 0.004035959909795274, 0.0032894025202464443], r.Pcov, 1e-6, "pcov");
        var a = Optimize.CurveFit(Model, Xd, Yd, nParams: 3, sigma: sigma, absoluteSigma: true);
        Cov([0.011587479703122421, -0.005796836859412493, -0.008037597116937986, -0.005796836859412493, 0.023557549729233195,
            0.012317792281422072, -0.008037597116937986, 0.012317792281422072, 0.010039291241730215], a.Pcov, 1e-6, "abs pcov");
    }

    [Fact]
    public void CurveFit_VectorisedAndUnderdetermined()
    {
        var v = Optimize.CurveFit((xs, p) => xs.Select(x => Model(x, p)).ToArray(), Xd, Yd, p0: [1.0, 1.0, 1.0],
            jac: (xs, p) =>
            {
                var j = new double[xs.Length * 3];
                for (var i = 0; i < xs.Length; i++) { var e = Math.Exp(-p[1] * xs[i]); j[i * 3] = e; j[i * 3 + 1] = -p[0] * xs[i] * e; j[i * 3 + 2] = 1; }
                return new NdArray(j, xs.Length, 3);
            });
        OptimizeAssert.Close([2.5633230073334787, 1.3238687324585485, 0.4954014248569238], v.Popt, 1e-7, "analytic jac popt");
        // Three points, three parameters: the fit is exact and the covariance cannot be estimated.
        var u = Optimize.CurveFit(Model, [0.0, 1.0, 2.0], [3.0, 1.5, 1.0], p0: [2.0, 1.0, 1.0], method: LeastSquaresMethod.Trf);
        Assert.False(u.CovarianceEstimated);
        Assert.All(u.Pcov.ToArray(), c => Assert.True(double.IsPositiveInfinity(c)));
    }

    private static double[] Rosen2(double[] p) => [10 * (p[1] - p[0] * p[0]), 1 - p[0]];
    private static NdArray Rosen2Jac(double[] p) => new([-20 * p[0], 10, -1, 0], 2, 2);

    [Fact]
    public void LeastSquares_RosenbrockMatchesScipy()
    {
        var a = Optimize.LeastSquares(Rosen2, [2, 2], Rosen2Jac);
        OptimizeAssert.Close([1, 1], a.X, 1e-15, "trf");
        Assert.Equal((3, 3, 1), (a.Nfev, a.Njev ?? 0, a.Status));
        var fd = Optimize.LeastSquares(Rosen2, [2, 2]);
        OptimizeAssert.Close([1, 1], fd.X, 1e-12, "trf fd");
        Assert.Equal(3, fd.Nfev);
        var b = Optimize.LeastSquares(Rosen2, [2, 2], Rosen2Jac, Bounds.FromPairs((null, null), (1.5, null)));
        OptimizeAssert.Close([1.2243707490009201, 1.500000000000023], b.X, 1e-8, "trf bounded");
        OptimizeAssert.Close(0.025213093946805685, b.Cost, 1e-12, "cost");
        Assert.Equal(9, b.Nfev);
        Assert.Equal([0, -1], b.ActiveMask);
        var d = Optimize.LeastSquares(Rosen2, [2, 2], Rosen2Jac, Bounds.FromPairs((null, null), (1.5, null)), LeastSquaresMethod.Dogbox);
        OptimizeAssert.Close([1.2243707487446318, 1.5], d.X, 1e-9, "dogbox");
        Assert.Equal(6, d.Nfev);
        var lm = Optimize.LeastSquares(Rosen2, [2, 2], Rosen2Jac, method: LeastSquaresMethod.Lm);
        OptimizeAssert.Close([1, 1], lm.X, 1e-15, "lm");
        Assert.Equal((4, 3), (lm.Nfev, lm.Njev ?? 0));
        Assert.Equal(3, lm.Status);
        Assert.Throws<ArgumentException>(() => Optimize.LeastSquares(Rosen2, [2, 2], bounds: Bounds.Uniform(2, 0, 1)));
    }

    [Theory]
    [InlineData(LeastSquaresLoss.Linear, 3.152513206110651, 0.38895536704414135, 7.773697839060088)]
    [InlineData(LeastSquaresLoss.SoftL1, 2.997849603707298, 0.3987519969559226, 0.6531190783189467)]
    [InlineData(LeastSquaresLoss.Huber, 2.996876095147228, 0.3987303945802763, 0.668348378382024)]
    [InlineData(LeastSquaresLoss.Cauchy, 2.985276264814549, 0.3982044944921871, 0.1123132123198707)]
    [InlineData(LeastSquaresLoss.Arctan, 2.984749260135582, 0.39804952979693004, 0.04573941817695189)]
    public void LeastSquares_RobustLossesMatchScipy(LeastSquaresLoss loss, double a, double b, double cost)
    {
        var t = Enumerable.Range(0, 40).Select(i => 10.0 * i / 39).ToArray();
        var y = Enumerable.Range(0, 40).Select(i => 3 * Math.Exp(-0.4 * t[i]) + 0.05 * Math.Sin(5.1 * i)).ToArray();
        y[5] += 2; y[17] -= 1.5; y[30] += 3;
        var r = Optimize.LeastSquares(p => t.Select((ti, i) => p[0] * Math.Exp(-p[1] * ti) - y[i]).ToArray(), [1, 0.1], loss: loss, fScale: 0.1);
        OptimizeAssert.Close([a, b], r.X, 1e-6, loss.ToString());
        OptimizeAssert.Relative(cost, r.Cost, 1e-6, "cost");
    }

    [Fact]
    public void Nnls_MatchesScipy()
    {
        var (x, rn) = Optimize.Nnls(new NdArray([1.0, 0, 1, 0, 0, 1], 3, 2), [2, 1, 1]);
        OptimizeAssert.Close([1.5, 1], x, 1e-14, "nnls");
        OptimizeAssert.Close(0.7071067811865477, rn, 1e-14, "rnorm");
        var a2 = new NdArray([0.6, 0.2, -0.3, 0.4, -0.9, 0.1, 0.3, 0.5, 0.8, -0.2, 0.3, 0.4, 0.9, -0.1, 0.2], 5, 3);
        double[] b2 = [1.0, -2.0, 0.5, 0.3, -0.7];
        var (x2, r2) = Optimize.Nnls(a2, b2);
        OptimizeAssert.Close([0, 2.008333333333333, 0], x2, 1e-13, "nnls 2");
        OptimizeAssert.Close(0.9949455596497063, r2, 1e-13, "rnorm 2");
    }

    [Fact]
    public void Nnls_AgreesWithBruteForceOverActiveSets()
    {
        var rng = new GraviRandom(7);
        for (var trial = 0; trial < 20; trial++)
        {
            int m = 6, n = 4;
            var a = new double[m * n]; var b = new double[m];
            for (var i = 0; i < a.Length; i++) a[i] = rng.Normal();
            for (var i = 0; i < m; i++) b[i] = rng.Normal();
            var (x, rn) = Optimize.Nnls(new NdArray(a, m, n), b);
            // Brute force: best unconstrained LS over every support whose solution is non-negative.
            var best = double.PositiveInfinity;
            for (var mask = 0; mask < (1 << n); mask++)
            {
                var cols = Enumerable.Range(0, n).Where(j => (mask >> j & 1) == 1).ToArray();
                var z = new double[n];
                if (cols.Length > 0)
                {
                    var sub = new double[m * cols.Length];
                    for (var i = 0; i < m; i++) for (var k = 0; k < cols.Length; k++) sub[i * cols.Length + k] = a[i * n + cols[k]];
                    var lsq = LinAlg.LeastSquares(new NdArray(sub, m, cols.Length), new NdArray(b, m)).ToArray();
                    if (lsq.Any(v => v < 0)) continue;
                    for (var k = 0; k < cols.Length; k++) z[cols[k]] = lsq[k];
                }
                double s = 0;
                for (var i = 0; i < m; i++) { double ax = 0; for (var j = 0; j < n; j++) ax += a[i * n + j] * z[j]; s += (ax - b[i]) * (ax - b[i]); }
                best = Math.Min(best, Math.Sqrt(s));
            }
            Assert.All(x, v => Assert.True(v >= 0));
            OptimizeAssert.Close(best, rn, 1e-10, $"trial {trial}");
        }
    }

    [Fact]
    public void LsqLinear_MatchesScipy()
    {
        var a2 = new NdArray([0.6, 0.2, -0.3, 0.4, -0.9, 0.1, 0.3, 0.5, 0.8, -0.2, 0.3, 0.4, 0.9, -0.1, 0.2], 5, 3);
        double[] b2 = [1.0, -2.0, 0.5, 0.3, -0.7];
        var bounds = new Bounds([-0.5, 0, -1], [0.5, 1, 1]);
        var t = Optimize.LsqLinear(a2, b2, bounds);
        OptimizeAssert.Close([-0.2821975382767938, 1.0, -0.4399579705794055], t.X, 1e-8, "trf");
        OptimizeAssert.Close(0.9310604923446414, t.Cost, 1e-12, "trf cost");
        Assert.Equal(1, t.ActiveMask[1]);
        var v = Optimize.LsqLinear(a2, b2, bounds, LsqLinearMethod.Bvls);
        OptimizeAssert.Close([-0.28219753827679395, 1.0, -0.43995797057940506], v.X, 1e-13, "bvls");
        Assert.Equal((1, 2), (v.Status, v.Nit));
        var free = Optimize.LsqLinear(a2, b2);
        Assert.Equal(3, free.Status);
    }
}
