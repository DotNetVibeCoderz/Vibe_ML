using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Helpers shared by the Spatial and Cluster tests.</summary>
internal static class SpatialAssert
{
    public static void Close(double expected, double actual, double tol, string what = "")
    {
        var ok = (double.IsNaN(expected) && double.IsNaN(actual)) || expected == actual || Math.Abs(expected - actual) <= tol;
        Assert.True(ok, $"{what}: expected {expected:R}, got {actual:R} (tol {tol})");
    }

    public static void Close(double[] expected, double[] actual, double tol, string what = "")
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++) Close(expected[i], actual[i], tol, $"{what}[{i}]");
    }

    /// <summary>
    /// Deterministic points from an integer LCG, reproducible bit-for-bit in Python
    /// (<c>s = (s*1103515245 + 12345) % 2**31; x = (s % 10007) / 10007</c>), so scipy reference
    /// values can be computed on exactly the same input.
    /// </summary>
    public static double[] Lcg(int n, int d, long seed)
    {
        var s = seed;
        var r = new double[n * d];
        for (var i = 0; i < r.Length; i++)
        {
            s = (s * 1103515245 + 12345) % 2147483648;
            r[i] = (s % 10007) / 10007.0;
        }
        return r;
    }

    public static NdArray LcgNd(int n, int d, long seed) => new(Lcg(n, d, seed), n, d);

    public static double[] Random(int n, int seed)
    {
        var rng = new GraviRandom(seed);
        var r = new double[n];
        for (var i = 0; i < n; i++) r[i] = rng.NextDouble();
        return r;
    }
}
