using System.Numerics;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

internal static class LinalgAssert
{
    public static NdArray M(int rows, int cols, params double[] values) => new(values, rows, cols);

    public static NdArray V(params double[] values) => new(values, values.Length);

    public static NdArray Mul(NdArray a, NdArray b) => LinAlg.Dot(a, b);

    public static void Close(NdArray expected, NdArray actual, double tol, string what = "")
    {
        Assert.True(expected.Shape.SequenceEqual(actual.Shape),
            $"{what}: shape [{string.Join(",", expected.Shape.ToArray())}] vs [{string.Join(",", actual.Shape.ToArray())}]");
        Close(expected.ToArray(), actual.ToArray(), tol, what);
    }

    public static void Close(double[] expected, double[] actual, double tol, string what = "")
    {
        Assert.True(expected.Length == actual.Length, $"{what}: length {expected.Length} vs {actual.Length}");
        var scale = 1.0;
        foreach (var v in expected) scale = Math.Max(scale, Math.Abs(v));
        for (var i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol * scale,
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R} (tol {tol} x {scale})");
    }

    public static void Close(double expected, double actual, double tol, string what = "")
        => Assert.True(Math.Abs(expected - actual) <= tol * Math.Max(1.0, Math.Abs(expected)),
            $"{what}: expected {expected:R}, got {actual:R}");

    public static void Identity(NdArray a, double tol, string what = "")
    {
        var n = a.Shape[0];
        Close(NdArray.Eye(n), a, tol, what);
    }

    public static void Orthogonal(NdArray q, double tol, string what = "")
        => Identity(Mul(q.T, q), tol, what + " (QᵀQ)");

    public static double MaxAbs(NdArray a)
    {
        var m = 0.0;
        foreach (var v in a.ToArray()) m = Math.Max(m, Math.Abs(v));
        return m;
    }

    public static NdArray Sub(NdArray a, NdArray b)
    {
        var x = a.ToArray();
        var y = b.ToArray();
        for (var i = 0; i < x.Length; i++) x[i] -= y[i];
        return new NdArray(x, a.Shape.ToArray());
    }

    public static NdArray Add(NdArray a, NdArray b)
    {
        var x = a.ToArray();
        var y = b.ToArray();
        for (var i = 0; i < x.Length; i++) x[i] += y[i];
        return new NdArray(x, a.Shape.ToArray());
    }

    public static Complex[] CMul(Complex[] a, Complex[] b, int n)
    {
        var c = new Complex[n * n];
        for (var i = 0; i < n; i++)
            for (var k = 0; k < n; k++)
                for (var j = 0; j < n; j++) c[i * n + j] += a[i * n + k] * b[k * n + j];
        return c;
    }

    /// <summary>A deterministic pseudo-random matrix (LCG) so tests need no RNG dependency.</summary>
    public static NdArray Random(int rows, int cols, int seed)
    {
        var rng = new GraviRandom(seed);
        var d = new double[rows * cols];
        for (var i = 0; i < d.Length; i++) d[i] = rng.NextDouble() * 2.0 - 1.0;
        return new NdArray(d, rows, cols);
    }

    public static readonly NdArray A4 = M(4, 4, 4, -2, 1, 3, 3, 6, -4, 2, 2, 1, 8, -5, 1, -3, 2, 7);
}
