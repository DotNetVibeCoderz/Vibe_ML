using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Tolerance checks against the scipy reference table.</summary>
internal static class StatsTestsAssert
{
    public static readonly (string Name, Alternative Alt)[] Alternatives =
        [("two-sided", Alternative.TwoSided), ("less", Alternative.Less), ("greater", Alternative.Greater)];

    /// <summary>Asserts |actual − expected| ≤ atol + rtol·|expected| element-wise (NaN matches NaN).</summary>
    public static void Close(string key, double rtol, double atol, params double[] actual)
    {
        Assert.True(StatsTestsRef.Expected.TryGetValue(key, out var expected), $"missing reference '{key}'");
        Assert.True(expected!.Length == actual.Length, $"{key}: expected {expected.Length} values, got {actual.Length}");
        for (var i = 0; i < actual.Length; i++) Near(expected[i], actual[i], rtol, atol, $"{key}[{i}]");
    }

    /// <summary>Single-value tolerance check.</summary>
    public static void Near(double expected, double actual, double rtol, double atol, string what)
    {
        if (double.IsNaN(expected)) { Assert.True(double.IsNaN(actual), $"{what}: expected NaN, got {actual}"); return; }
        if (double.IsInfinity(expected)) { Assert.True(expected == actual, $"{what}: expected {expected}, got {actual}"); return; }
        var err = Math.Abs(actual - expected);
        Assert.True(err <= atol + rtol * Math.Abs(expected), $"{what}: expected {expected:R}, got {actual:R} (err {err:E2})");
    }

    public static double[] Flat(Gravicode.Science.GraviNum.NdArray a) => a.AsContiguous().ToArray();
}
