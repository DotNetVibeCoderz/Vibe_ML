using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Tolerance assertions for the Interpolate tests (NaN equals NaN).</summary>
internal static class InterpolateAssert
{
    public static void Close(double expected, double actual, double tol, string what)
    {
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual), $"{what}: expected NaN, got {actual:R}");
            return;
        }
        var err = Math.Abs(expected - actual);
        var scale = Math.Max(1.0, Math.Abs(expected));
        Assert.True(err <= tol * scale, $"{what}: expected {expected:R}, got {actual:R} (|diff| {err:E3}, tol {tol:E1})");
    }

    public static void AllClose(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual, double tol, string what)
    {
        Assert.True(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        for (var i = 0; i < expected.Length; i++) Close(expected[i], actual[i], tol, $"{what}[{i}]");
    }
}
