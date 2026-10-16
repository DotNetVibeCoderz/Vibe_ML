using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Tolerance helpers for the special-function tests.</summary>
internal static class SpecialAssert
{
    /// <summary>Relative error, falling back to absolute error when the expected value is zero.</summary>
    public static double RelErr(double expected, double actual)
    {
        if (expected.Equals(actual)) return 0.0; // also NaN == NaN and matching infinities
        if (double.IsNaN(expected) || double.IsNaN(actual) || double.IsInfinity(expected) || double.IsInfinity(actual))
            return double.PositiveInfinity;
        return expected == 0 ? Math.Abs(actual) : Math.Abs(actual - expected) / Math.Abs(expected);
    }

    /// <summary>Asserts |actual - expected| ≤ tol·|expected| (absolute when expected is 0).</summary>
    public static void Close(double expected, double actual, double tol, string what = "")
    {
        var e = RelErr(expected, actual);
        Assert.True(e <= tol, $"{what}: expected {expected:R}, got {actual:R}, relative error {e:E2} > {tol:E1}");
    }

    /// <summary>Asserts |actual - expected| ≤ absTol.</summary>
    public static void Near(double expected, double actual, double absTol)
        => Assert.True(Math.Abs(actual - expected) <= absTol, $"expected {expected:R}, got {actual:R} (|diff| {Math.Abs(actual - expected):E2} > {absTol:E1})");

    /// <summary>
    /// Checks every row of a reference table (argument columns followed by
    /// <paramref name="outputs"/> expected values) against <paramref name="f"/>, reporting the worst row.
    /// </summary>
    public static void Table(double[][] rows, int outputs, Func<double[], double[]> f, double tol, string name)
    {
        var worst = 0.0;
        var worstRow = "";
        foreach (var row in rows)
        {
            var nargs = row.Length - outputs;
            var args = row[..nargs];
            var got = f(args);
            Assert.Equal(outputs, got.Length);
            for (var i = 0; i < outputs; i++)
            {
                var e = RelErr(row[nargs + i], got[i]);
                if (e > worst || double.IsNaN(e))
                {
                    worst = double.IsNaN(e) ? double.PositiveInfinity : e;
                    worstRow = $"({string.Join(", ", args)}) output {i}: expected {row[nargs + i]:R}, got {got[i]:R}";
                }
            }
        }
        Assert.True(worst <= tol, $"{name}: worst relative error {worst:E2} > {tol:E1} at {worstRow}");
    }

    /// <summary>Single-output form of <see cref="Table(double[][], int, Func{double[], double[]}, double, string)"/>.</summary>
    public static void Table(double[][] rows, Func<double[], double> f, double tol, string name)
        => Table(rows, 1, a => [f(a)], tol, name);
}
