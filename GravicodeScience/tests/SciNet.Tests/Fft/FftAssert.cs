using System.Numerics;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Tolerance assertions and independent references for the FFT tests.</summary>
internal static class FftAssert
{
    /// <summary>
    /// The DFT straight from its definition, sharing no code with the library. The kernel index
    /// is reduced mod n in integer arithmetic so the reference itself does not lose phase accuracy
    /// at large n.
    /// </summary>
    public static Complex[] NaiveDft(Complex[] x, bool inverse = false)
    {
        var n = x.Length;
        var sign = inverse ? 1.0 : -1.0;
        var table = new Complex[n];
        for (var m = 0; m < n; m++)
        {
            var angle = sign * 2.0 * Math.PI * m / n;
            table[m] = new Complex(Math.Cos(angle), Math.Sin(angle));
        }
        var result = new Complex[n];
        for (var k = 0; k < n; k++)
        {
            var sum = Complex.Zero;
            var idx = 0L;
            for (var t = 0; t < n; t++)
            {
                sum += x[t] * table[idx];
                idx += k;
                if (idx >= n) idx -= n;
            }
            result[k] = sum;
        }
        return result;
    }

    /// <summary>One bin of the naive DFT (for lengths where the full O(n²) is too slow).</summary>
    public static Complex NaiveBin(Complex[] x, int k)
    {
        var n = x.Length;
        var sum = Complex.Zero;
        for (var t = 0; t < n; t++)
        {
            var m = (long)t * k % n;
            var angle = -2.0 * Math.PI * m / n;
            sum += x[t] * new Complex(Math.Cos(angle), Math.Sin(angle));
        }
        return sum;
    }

    public static Complex[] RandomComplex(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new Complex[n];
        for (var i = 0; i < n; i++) x[i] = new Complex(rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1);
        return x;
    }

    public static double[] RandomReal(int n, int seed)
    {
        var rng = new Random(seed);
        var x = new double[n];
        for (var i = 0; i < n; i++) x[i] = rng.NextDouble() * 2 - 1;
        return x;
    }

    public static double MaxAbs(ReadOnlySpan<Complex> x)
    {
        double m = 0;
        foreach (var v in x) m = Math.Max(m, v.Magnitude);
        return m;
    }

    /// <summary>Max element-wise error, relative to the largest magnitude of <paramref name="expected"/>.</summary>
    public static void Close(ReadOnlySpan<Complex> expected, ReadOnlySpan<Complex> actual, double tol, string what)
    {
        Assert.True(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        var scale = Math.Max(MaxAbs(expected), 1e-300);
        double worst = 0;
        var at = -1;
        for (var i = 0; i < expected.Length; i++)
        {
            var e = (expected[i] - actual[i]).Magnitude / scale;
            if (e > worst) { worst = e; at = i; }
        }
        Assert.True(worst <= tol, $"{what}: relative error {worst:E3} at index {at} exceeds {tol:E1}");
    }

    public static void Close(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual, double tol, string what)
    {
        Assert.True(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        double scale = 1e-300;
        foreach (var v in expected) scale = Math.Max(scale, Math.Abs(v));
        double worst = 0;
        var at = -1;
        for (var i = 0; i < expected.Length; i++)
        {
            var e = Math.Abs(expected[i] - actual[i]) / scale;
            if (e > worst) { worst = e; at = i; }
        }
        Assert.True(worst <= tol, $"{what}: relative error {worst:E3} at index {at} exceeds {tol:E1}");
    }

    /// <summary>Absolute element-wise comparison (for hard-coded reference values).</summary>
    public static void Abs(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual, double tol, string what)
    {
        Assert.True(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        for (var i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol, $"{what}[{i}]: {actual[i]:R} vs {expected[i]:R}");
    }
}
