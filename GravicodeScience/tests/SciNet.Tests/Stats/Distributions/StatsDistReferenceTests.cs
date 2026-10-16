using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsDistAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Every distribution against the scipy 1.17.1 reference table in StatsDistReference.Generated.cs.</summary>
public class StatsDistReferenceTests
{
    public static TheoryData<int> ContinuousIndices()
    {
        var d = new TheoryData<int>();
        for (var i = 0; i < StatsDistReference.Continuous.Length; i++) d.Add(i);
        return d;
    }

    public static TheoryData<int> DiscreteIndices()
    {
        var d = new TheoryData<int>();
        for (var i = 0; i < StatsDistReference.Discrete.Length; i++) d.Add(i);
        return d;
    }

    [Theory]
    [MemberData(nameof(ContinuousIndices))]
    public void ContinuousMatchesScipy(int index)
    {
        var c = StatsDistReference.Continuous[index];
        var d = c.Make();
        var errors = new List<string>();
        Collect(errors, "pdf", c.X, c.Pdf, d.Pdf, c.Tol);
        Collect(errors, "logpdf", c.X, c.LogPdf, d.LogPdf, c.Tol);
        Collect(errors, "cdf", c.X, c.Cdf, d.Cdf, c.Tol);
        Collect(errors, "sf", c.X, c.Sf, d.Sf, c.Tol);
        Collect(errors, "logcdf", c.X, c.LogCdf, d.LogCdf, c.Tol);
        Collect(errors, "logsf", c.X, c.LogSf, d.LogSf, c.Tol);
        Collect(errors, "ppf", c.Q, c.Ppf, d.Ppf, c.Tol, 1e-13);
        Collect(errors, "isf", c.Q, c.Isf, d.Isf, c.Tol, 1e-13);
        Collect(errors, "mean", c.Mean, d.Mean, c.MomentTol);
        Collect(errors, "var", c.Var, d.Var, c.MomentTol);
        Collect(errors, "skew", c.Skew, d.Skew, c.MomentTol);
        Collect(errors, "kurtosis", c.Kurt, d.Kurtosis, c.MomentTol);
        Collect(errors, "entropy", c.Entropy, d.Entropy, c.EntropyTol);
        Collect(errors, "median", [0.5], [c.Median], _ => d.Median(), c.Tol, 1e-13);
        Assert.True(errors.Count == 0, c.Label + ":\n  " + string.Join("\n  ", errors));
    }

    [Theory]
    [MemberData(nameof(DiscreteIndices))]
    public void DiscreteMatchesScipy(int index)
    {
        var c = StatsDistReference.Discrete[index];
        var d = c.Make();
        var errors = new List<string>();
        Collect(errors, "pmf", c.K, c.Pmf, d.Pmf, c.Tol);
        Collect(errors, "logpmf", c.K, c.LogPmf, d.LogPmf, c.Tol);
        Collect(errors, "cdf", c.K, c.Cdf, d.Cdf, c.Tol);
        Collect(errors, "sf", c.K, c.Sf, d.Sf, c.Tol);
        Collect(errors, "logcdf", c.K, c.LogCdf, d.LogCdf, c.Tol);
        Collect(errors, "logsf", c.K, c.LogSf, d.LogSf, c.Tol);
        Collect(errors, "ppf", c.Q, c.Ppf, d.Ppf, 0.0);
        Collect(errors, "isf", c.Q, c.Isf, d.Isf, 0.0);
        Collect(errors, "mean", c.Mean, d.Mean, c.MomentTol);
        Collect(errors, "var", c.Var, d.Var, c.MomentTol);
        Collect(errors, "skew", c.Skew, d.Skew, c.MomentTol);
        Collect(errors, "kurtosis", c.Kurt, d.Kurtosis, c.MomentTol);
        Collect(errors, "entropy", c.Entropy, d.Entropy, c.EntropyTol);
        Collect(errors, "median", c.Median, d.Median, 0.0);
        Assert.True(errors.Count == 0, c.Label + ":\n  " + string.Join("\n  ", errors));
    }
}
