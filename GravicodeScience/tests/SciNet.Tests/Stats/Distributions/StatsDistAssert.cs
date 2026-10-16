using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>One continuous distribution checked against scipy at fixed points.</summary>
internal sealed record ContinuousReferenceCase(
    string Label, Func<ContinuousDistribution> Make, double Tol, double MomentTol, double EntropyTol,
    double[] X, double[] Pdf, double[] LogPdf, double[] Cdf, double[] Sf, double[] LogCdf, double[] LogSf,
    double[] Q, double[] Ppf, double[] Isf,
    double Mean, double Var, double Skew, double Kurt, double Entropy, double Median)
{
    public override string ToString() => Label;
}

/// <summary>One discrete distribution checked against scipy at fixed points.</summary>
internal sealed record DiscreteReferenceCase(
    string Label, Func<DiscreteDistribution> Make, double Tol, double MomentTol, double EntropyTol,
    double[] K, double[] Pmf, double[] LogPmf, double[] Cdf, double[] Sf, double[] LogCdf, double[] LogSf,
    double[] Q, double[] Ppf, double[] Isf,
    double Mean, double Var, double Skew, double Kurt, double Entropy, double Median)
{
    public override string ToString() => Label;
}

/// <summary>Tolerance helpers for the distribution tests.</summary>
internal static class StatsDistAssert
{
    /// <summary>Relative error, absolute when the expected value is zero; NaN/∞ must match exactly.</summary>
    public static double RelErr(double expected, double actual)
    {
        if (expected.Equals(actual)) return 0.0;
        if (double.IsNaN(expected) || double.IsNaN(actual) || double.IsInfinity(expected) || double.IsInfinity(actual))
            return double.PositiveInfinity;
        // Values below the smallest normal carry no relative precision in either library.
        if (Math.Abs(expected) < 1e-300 && Math.Abs(actual) < 1e-300) return 0.0;
        return expected == 0 ? Math.Abs(actual) : Math.Abs(actual - expected) / Math.Abs(expected);
    }

    /// <summary>Asserts a relative error bound.</summary>
    public static void Close(double expected, double actual, double tol, string what = "")
    {
        var e = RelErr(expected, actual);
        Assert.True(e <= tol, $"{what}: expected {expected:R}, got {actual:R}, relative error {e:E2} > {tol:E1}");
    }

    /// <summary>Asserts an absolute error bound.</summary>
    public static void Near(double expected, double actual, double absTol, string what = "")
        => Assert.True(Math.Abs(actual - expected) <= absTol, $"{what}: expected {expected:R}, got {actual:R} (|diff| {Math.Abs(actual - expected):E2} > {absTol:E1})");

    /// <summary>Collects mismatches of one vectorised check into <paramref name="errors"/>.</summary>
    public static void Collect(List<string> errors, string what, double[] args, double[] expected, Func<double, double> f, double tol, double absFloor = 0.0)
    {
        for (var i = 0; i < args.Length; i++)
        {
            double got;
            try { got = f(args[i]); }
            catch (Exception ex) { errors.Add($"{what}({args[i]:R}) threw {ex.GetType().Name}: {ex.Message}"); continue; }
            var e = RelErr(expected[i], got);
            if (!(e <= tol) && !(Math.Abs(got - expected[i]) <= absFloor)) errors.Add($"{what}({args[i]:R}): expected {expected[i]:R}, got {got:R}, rel {e:E2}");
        }
    }

    /// <summary>Collects one scalar mismatch.</summary>
    public static void Collect(List<string> errors, string what, double expected, Func<double> f, double tol)
    {
        if (tol < 0) return; // a negative tolerance marks a quantity not checked for this case (too slow to evaluate)
        double got;
        try { got = f(); }
        catch (Exception ex) { errors.Add($"{what} threw {ex.GetType().Name}: {ex.Message}"); return; }
        var e = RelErr(expected, got);
        // Moments that are zero by symmetry come out of scipy's quadrature as ±1e-15.
        if (!(e <= tol) && !(Math.Abs(got - expected) <= 1e-13)) errors.Add($"{what}: expected {expected:R}, got {got:R}, rel {e:E2}");
    }
}
