using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Tolerance assertions and small reference builders shared by the Sparse tests.</summary>
internal static class SparseAssert
{
    public static void Close(double expected, double actual, double tol, string what = "")
        => Assert.True(expected.Equals(actual) || Math.Abs(expected - actual) <= tol, $"{what}: expected {expected:R}, got {actual:R} (tol {tol:G3})");

    public static void Close(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual, double tol, string what = "")
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
            Assert.True(expected[i].Equals(actual[i]) || Math.Abs(expected[i] - actual[i]) <= tol,
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R} (tol {tol:G3})");
    }

    public static void Close(NdArray expected, NdArray actual, double tol, string what = "")
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        Close(expected.AsContiguous().ToArray(), actual.AsContiguous().ToArray(), tol, what);
    }

    public static double MaxAbs(NdArray a) => a.AsContiguous().ToArray().Select(Math.Abs).DefaultIfEmpty(0).Max();

    /// <summary>A dense random matrix with roughly the given fraction of non-zeros (fixed seed).</summary>
    public static NdArray RandomDense(int m, int n, double density, int seed, double low = -1, double high = 1)
    {
        var rng = new GraviRandom(seed);
        var d = new double[m * n];
        for (var i = 0; i < d.Length; i++) if (rng.NextDouble() < density) d[i] = rng.Uniform(low, high);
        return new NdArray(d, m, n);
    }

    /// <summary>Plain triple-loop product, the reference for every sparse product.</summary>
    public static NdArray NaiveMatMul(NdArray a, NdArray b)
    {
        int m = a.Shape[0], k = a.Shape[1], n = b.Shape[1];
        var x = a.AsContiguous().ToArray();
        var y = b.AsContiguous().ToArray();
        var c = new double[m * n];
        for (var i = 0; i < m; i++)
            for (var p = 0; p < k; p++)
                for (var j = 0; j < n; j++) c[i * n + j] += x[i * k + p] * y[p * n + j];
        return new NdArray(c, m, n);
    }

    /// <summary>The 1-D second-difference matrix tridiag(−1, 2, −1).</summary>
    public static CsrMatrix Laplacian1D(int n) => SparseMatrices.Diags([[-1.0], [2.0], [-1.0]], [-1, 0, 1], (n, n));

    /// <summary>The 5-point 2-D Poisson matrix on an nx×ny grid (SPD).</summary>
    public static CsrMatrix Poisson2D(int nx, int ny) => SparseMatrices.Kronsum(Laplacian1D(nx), Laplacian1D(ny));

    /// <summary>Upwinded convection–diffusion on an n×n grid: non-symmetric, diagonally dominant.</summary>
    public static CsrMatrix ConvectionDiffusion(int n, double peclet)
    {
        var h = 1.0 / (n + 1);
        var c = peclet * h;
        var oneD = SparseMatrices.Diags([[-1.0 - c], [2.0 + c], [-1.0]], [-1, 0, 1], (n, n));
        return SparseMatrices.Kronsum(oneD, Laplacian1D(n));
    }

    public static double[] Residual(ILinearOperator a, double[] x, double[] b)
    {
        var ax = a.Dot(x);
        for (var i = 0; i < ax.Length; i++) ax[i] = b[i] - ax[i];
        return ax;
    }

    public static double Norm(ReadOnlySpan<double> v)
    {
        var s = 0.0;
        foreach (var x in v) s += x * x;
        return Math.Sqrt(s);
    }
}
