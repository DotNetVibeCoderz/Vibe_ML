using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class BvpTests
{
    private static NdArray Bratu(double[] x, NdArray y)
    {
        var m = x.Length;
        var r = new double[2 * m];
        for (var j = 0; j < m; j++) { r[j] = y[1, j]; r[m + j] = -Math.Exp(y[0, j]); }
        return new NdArray(r, 2, m);
    }

    private static double[] Linspace(double a, double b, int n) => Enumerable.Range(0, n).Select(i => a + (b - a) * i / (n - 1)).ToArray();

    // scipy 1.17.1 docs example: solve_bvp(fun, bc, linspace(0,1,5), y_a / y_b)
    [Theory]
    [InlineData(0.0, 1, 5, 0.140534772543497, 0.28432097718415983, 0.5493492748250857)]
    [InlineData(3.0, 4, 25, 4.09147891514972, 9.605137112098673, 10.846976018307315)]
    public void Bratu_BothSolutions_MatchScipy(double guess, int niter, int nodes, double y05, double yp025, double yp0)
    {
        var x = Linspace(0, 1, 5);
        var y = NdArray.Zeros(2, 5);
        for (var j = 0; j < 5; j++) y[0, j] = guess;
        var r = Integrate.SolveBvp(Bratu, (ya, yb) => [ya[0], yb[0]], x, y);
        Assert.True(r.Success, r.Message);
        Assert.Equal(niter, r.Niter);
        Assert.Equal(nodes, r.X.Length);
        IntegrateAssert.Close(y05, r.Sol.Evaluate(0.5)[0], 1e-8, "y(0.5)");
        IntegrateAssert.Close(yp025, r.Sol.Evaluate(0.25)[1], 1e-8, "y'(0.25)");
        IntegrateAssert.Close(yp0, r.Y[1, 0], 1e-8, "y'(0)");
        Assert.True(r.RmsResiduals.Max() < 1e-3);
    }

    [Fact]
    public void SturmLiouville_EigenvalueParameter()
    {
        // y'' + k^2 y = 0, y(0) = y(1) = 0, y'(0) = k: scipy finds k = 6.283294600464651 (≈ 2π).
        var x = Linspace(0, 1, 5);
        var y = NdArray.Zeros(2, 5);
        y[0, 1] = 1; y[0, 3] = -1;
        var r = Integrate.SolveBvp((xx, yy, p) =>
        {
            var m = xx.Length;
            var o = new double[2 * m];
            for (var j = 0; j < m; j++) { o[j] = yy[1, j]; o[m + j] = -p[0] * p[0] * yy[0, j]; }
            return new NdArray(o, 2, m);
        }, (ya, yb, p) => [ya[0], yb[0], ya[1] - p[0]], x, y, p: [6.0]);
        Assert.True(r.Success, r.Message);
        Assert.Equal(4, r.Niter);
        Assert.Equal(25, r.X.Length);
        IntegrateAssert.Close(6.283294600464651, r.P[0], 1e-8, "k");
        IntegrateAssert.Close(2 * Math.PI, r.P[0], 2e-4, "2π (collocation error at tol 1e-3)");
        IntegrateAssert.Close(0.9510516346631294, r.Sol.Evaluate(0.3)[0], 1e-8, "y(0.3)");
    }

    [Fact]
    public void PeriodicConditions_CoupledEnds()
    {
        // y'' = y - cos(2πx) with y(0) = y(1), y'(0) = y'(1): y = cos(2πx)/(1 + 4π²).
        var x = Linspace(0, 1, 11);
        var r = Integrate.SolveBvp((xx, yy) =>
        {
            var m = xx.Length;
            var o = new double[2 * m];
            for (var j = 0; j < m; j++) { o[j] = yy[1, j]; o[m + j] = yy[0, j] - Math.Cos(2 * Math.PI * xx[j]); }
            return new NdArray(o, 2, m);
        }, (ya, yb) => [ya[0] - yb[0], ya[1] - yb[1]], x, NdArray.Zeros(2, 11), tol: 1e-6);
        Assert.True(r.Success, r.Message);
        Assert.Equal(121, r.X.Length);
        IntegrateAssert.Close(0.007634117398930208, r.Sol.Evaluate(0.2)[0], 1e-10, "scipy");
        IntegrateAssert.Close(Math.Cos(2 * Math.PI * 0.2) / (1 + 4 * Math.PI * Math.PI), r.Sol.Evaluate(0.2)[0], 1e-8, "exact");
    }

    [Fact]
    public void LargeMesh_IsSolvedByTheBandedSystem()
    {
        // 10k-node mesh on a smooth linear problem: y'' = -y, y(0) = 0, y(π/2) = 1 → y = sin x.
        var x = Linspace(0, Math.PI / 2, 10000);
        var r = Integrate.SolveBvp((xx, yy) =>
        {
            var m = xx.Length;
            var o = new double[2 * m];
            for (var j = 0; j < m; j++) { o[j] = yy[1, j]; o[m + j] = -yy[0, j]; }
            return new NdArray(o, 2, m);
        }, (ya, yb) => [ya[0], yb[0] - 1], x, NdArray.Zeros(2, 10000), maxNodes: 20000);
        Assert.True(r.Success, r.Message);
        IntegrateAssert.Close(Math.Sin(0.7), r.Sol.Evaluate(0.7)[0], 1e-10, "sin");
    }
}
