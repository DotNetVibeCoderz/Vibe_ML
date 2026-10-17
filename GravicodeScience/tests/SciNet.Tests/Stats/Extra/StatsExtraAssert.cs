using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Tolerance checks for the scipy.stats additions (survival, MGC, Sobol indices, matrix distributions, ...).</summary>
internal static class StatsExtraAssert
{
    /// <summary>|actual − expected| ≤ atol + rtol·|expected|; NaN matches NaN and infinities must be equal.</summary>
    public static void Near(double expected, double actual, double rtol, double atol, string what)
    {
        if (double.IsNaN(expected)) { Assert.True(double.IsNaN(actual), $"{what}: expected NaN, got {actual}"); return; }
        if (double.IsInfinity(expected)) { Assert.True(expected == actual, $"{what}: expected {expected}, got {actual}"); return; }
        var err = Math.Abs(actual - expected);
        Assert.True(err <= atol + rtol * Math.Abs(expected), $"{what}: expected {expected:R}, got {actual:R} (err {err:E2})");
    }

    /// <summary>Element-wise <see cref="Near"/>.</summary>
    public static void All(double[] expected, double[] actual, double rtol, double atol, string what)
    {
        Assert.True(expected.Length == actual.Length, $"{what}: expected {expected.Length} values, got {actual.Length}");
        for (var i = 0; i < expected.Length; i++) Near(expected[i], actual[i], rtol, atol, $"{what}[{i}]");
    }
}
