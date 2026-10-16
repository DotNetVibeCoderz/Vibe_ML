using System.Numerics;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Tolerance-explicit comparisons for the Signal tests.</summary>
internal static class SignalAssert
{
    /// <summary>|a − e| ≤ atol + rtol·|e| elementwise.</summary>
    public static void Close(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual, double rtol = 1e-10, double atol = 1e-12, string what = "")
    {
        Assert.True(expected.Length == actual.Length, $"{what}: length {actual.Length} != expected {expected.Length}");
        for (var i = 0; i < expected.Length; i++)
        {
            var e = expected[i];
            var a = actual[i];
            if (double.IsNaN(e)) { Assert.True(double.IsNaN(a), $"{what}[{i}]: expected NaN, got {a}"); continue; }
            Assert.True(Math.Abs(a - e) <= atol + rtol * Math.Abs(e), $"{what}[{i}]: {a:R} vs expected {e:R} (diff {Math.Abs(a - e):E2})");
        }
    }

    public static void Close(double expected, double actual, double rtol = 1e-10, double atol = 1e-12, string what = "")
        => Assert.True(Math.Abs(actual - expected) <= atol + rtol * Math.Abs(expected), $"{what}: {actual:R} vs expected {expected:R}");

    /// <summary>Interleaved re/im reference against complex values.</summary>
    public static void Close(double[] expectedInterleaved, Complex[] actual, double rtol = 1e-10, double atol = 1e-12, string what = "")
    {
        Assert.True(expectedInterleaved.Length == 2 * actual.Length, $"{what}: length {actual.Length} != expected {expectedInterleaved.Length / 2}");
        for (var i = 0; i < actual.Length; i++)
        {
            var e = new Complex(expectedInterleaved[2 * i], expectedInterleaved[2 * i + 1]);
            Assert.True(Complex.Abs(actual[i] - e) <= atol + rtol * Complex.Abs(e), $"{what}[{i}]: {actual[i]} vs expected {e}");
        }
    }

    public static Complex[] ToComplex(double[] interleaved)
    {
        var c = new Complex[interleaved.Length / 2];
        for (var i = 0; i < c.Length; i++) c[i] = new Complex(interleaved[2 * i], interleaved[2 * i + 1]);
        return c;
    }

    /// <summary>Roots compared as multisets: each expected root is matched to the nearest unused actual root.</summary>
    public static void SameRoots(double[] expectedInterleaved, Complex[] actual, double tol, string what = "")
    {
        var exp = ToComplex(expectedInterleaved);
        Assert.True(exp.Length == actual.Length, $"{what}: {actual.Length} roots vs expected {exp.Length}");
        var used = new bool[actual.Length];
        foreach (var e in exp)
        {
            var best = -1;
            var bd = double.PositiveInfinity;
            for (var j = 0; j < actual.Length; j++)
            {
                if (used[j]) continue;
                var d = Complex.Abs(actual[j] - e);
                if (d < bd) { bd = d; best = j; }
            }
            used[best] = true;
            Assert.True(bd <= tol * Math.Max(1, Complex.Abs(e)), $"{what}: root {e} nearest {actual[best]} (diff {bd:E2})");
        }
    }
}
