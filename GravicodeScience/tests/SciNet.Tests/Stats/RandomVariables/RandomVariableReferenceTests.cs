using System.Numerics;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsDistAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>One function of a random variable checked at several arguments (Args2 for two-argument CDFs).</summary>
/// <param name="Function">scipy's method name (<c>"cdf"</c>, <c>"moment_central"</c>, …).</param>
/// <param name="Args">Arguments (moment orders for the moment checks, a dummy 0 for properties).</param>
/// <param name="Expected">scipy's values.</param>
/// <param name="Numeric">Whether scipy computed the value numerically (quadrature, inversion, optimisation).</param>
/// <param name="Args2">Second arguments of a two-argument CDF.</param>
internal sealed record RvCheck(string Function, double[] Args, double[] Expected, bool Numeric, double[]? Args2 = null);

/// <summary>A random variable from scipy's new framework and its reference values.</summary>
internal sealed record RvReferenceCase(string Label, Func<object> Make, double Tol, RvCheck[] Checks)
{
    public override string ToString() => Label;
}

/// <summary>The new random-variable framework against scipy 1.17.1 (RandomVariableReference.Generated.cs).</summary>
public class RandomVariableReferenceTests
{
    public static TheoryData<int> Indices()
    {
        var d = new TheoryData<int>();
        for (var i = 0; i < RandomVariableReference.Cases.Length; i++) d.Add(i);
        return d;
    }

    [Theory]
    [MemberData(nameof(Indices))]
    public void MatchesScipy(int index)
    {
        var c = RandomVariableReference.Cases[index];
        var d = c.Make();
        var errors = new List<string>();
        foreach (var check in c.Checks)
        {
            // Numerically computed values agree as well as two independent quadratures/root finders
            // do; optimisation (the mode) only to about √eps, since the density is flat at its peak.
            var tol = check.Function == "mode" ? 1e-6 : check.Numeric || check.Function.StartsWith("moment") ? 1e-9 : c.Tol;
            var absFloor = check.Function == "mode" ? 1e-7 : check.Numeric || check.Function.StartsWith("moment") ? 1e-12 : 0.0;
            for (var i = 0; i < check.Args.Length; i++)
            {
                var x = check.Args[i];
                double got;
                try { got = Evaluate(d, check.Function, x, check.Args2 is null ? 0 : check.Args2[i]); }
                catch (Exception ex) { errors.Add($"{check.Function}({x:R}) threw {ex.GetType().Name}: {ex.Message}"); continue; }
                var expected = check.Expected[i];
                var e = RelErr(expected, got);
                if (!(e <= tol) && !(Math.Abs(got - expected) <= absFloor))
                    errors.Add($"{check.Function}({x:R}{(check.Args2 is null ? "" : ", " + check.Args2[i].ToString("R"))}): expected {expected:R}, got {got:R}, rel {e:E2}");
            }
        }
        Assert.True(errors.Count == 0, c.Label + ":\n  " + string.Join("\n  ", errors));
    }

    private static double Evaluate(object d, string fn, double x, double y) => d switch
    {
        UnivariateRandomVariable rv => fn switch
        {
            "pdf" => rv.Pdf(x),
            "logpdf" => rv.LogPdf(x),
            "pmf" => rv.Pmf(x),
            "logpmf" => rv.LogPmf(x),
            "cdf" => rv.Cdf(x),
            "ccdf" => rv.Ccdf(x),
            "logcdf" => rv.LogCdf(x),
            "logccdf" => rv.LogCcdf(x),
            "icdf" => rv.Icdf(x),
            "iccdf" => rv.Iccdf(x),
            "ilogcdf" => rv.ILogCdf(x),
            "ilogccdf" => rv.ILogCcdf(x),
            "cdf2" => rv.Cdf(x, y),
            "ccdf2" => rv.Ccdf(x, y),
            "logcdf2" => rv.LogCdf(x, y).Real,
            "logcdf2imag" => rv.LogCdf(x, y).Imaginary,
            "logccdf2" => rv.LogCcdf(x, y),
            "entropy" => rv.Entropy(),
            "logentropy" => rv.LogEntropy().Real,
            "median" => rv.Median(),
            "mode" => rv.Mode(),
            "mean" => rv.Mean(),
            "variance" => rv.Variance(),
            "standard_deviation" => rv.StandardDeviation(),
            "skewness" => rv.Skewness(),
            "kurtosis" => rv.Kurtosis(),
            "moment_raw" => rv.Moment((int)x, MomentKind.Raw),
            "moment_central" => rv.Moment((int)x, MomentKind.Central),
            "moment_standardized" => rv.Moment((int)x, MomentKind.Standardized),
            _ => throw new ArgumentException("unknown function " + fn),
        },
        Mixture m => fn switch
        {
            "pdf" => m.Pdf(x),
            "logpdf" => m.LogPdf(x),
            "cdf" => m.Cdf(x),
            "ccdf" => m.Ccdf(x),
            "logcdf" => m.LogCdf(x),
            "logccdf" => m.LogCcdf(x),
            "icdf" => m.Icdf(x),
            "iccdf" => m.Iccdf(x),
            "ilogcdf" => m.ILogCdf(x),
            "ilogccdf" => m.ILogCcdf(x),
            "cdf2" => m.Cdf(x, y),
            "ccdf2" => m.Ccdf(x, y),
            "entropy" => m.Entropy(),
            "logentropy" => m.LogEntropy().Real,
            "median" => m.Median(),
            "mode" => m.Mode(),
            "mean" => m.Mean(),
            "variance" => m.Variance(),
            "standard_deviation" => m.StandardDeviation(),
            "skewness" => m.Skewness(),
            "kurtosis" => m.Kurtosis(),
            "moment_raw" => m.Moment((int)x, MomentKind.Raw),
            "moment_central" => m.Moment((int)x, MomentKind.Central),
            "moment_standardized" => m.Moment((int)x, MomentKind.Standardized),
            "logccdf2" => m.LogCcdf(x, y),
            _ => throw new ArgumentException("unknown function " + fn),
        },
        _ => throw new ArgumentException("unknown distribution type"),
    };
}
