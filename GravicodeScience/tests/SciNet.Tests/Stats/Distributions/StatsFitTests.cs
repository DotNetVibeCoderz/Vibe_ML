using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Maximum-likelihood fits against the true MLEs of scipy 1.17.1 (StatsFitReference.Generated.cs).</summary>
public class StatsFitTests
{
    public static TheoryData<int> FitIndices()
    {
        var d = new TheoryData<int>();
        for (var i = 0; i < StatsFitReference.Cases.Length; i++) d.Add(i);
        return d;
    }

    private static readonly HashSet<string> ClosedForm =
        ["norm", "expon", "uniform", "laplace", "halfnorm", "gumbel_r", "gumbel_l", "gamma_floc", "lognorm_floc", "weibull_floc", "pareto_floc"];

    [Theory]
    [MemberData(nameof(FitIndices))]
    public void FitReachesTheMaximumLikelihood(int index)
    {
        var (name, fit, expected, nll, scipyNll) = StatsFitReference.Cases[index];
        var d = fit();
        var data = name is "norm" or "laplace" or "gumbel_r" or "gumbel_l" or "t" or "logistic" or "cauchy" or "skewnorm"
            ? StatsFitReference.Real
            : name is "uniform" or "beta_fixed" ? StatsFitReference.Unit : StatsFitReference.Pos;
        var got = d.Shapes.Concat([d.Loc, d.Scale]).ToArray();
        Assert.Equal(expected.Length, got.Length);
        var ours = d.NegativeLogLikelihood(data);
        var closed = ClosedForm.Contains(name);
        // Closed forms must reproduce the optimum to rounding; numerical fits must reach it (never a
        // worse likelihood than scipy's own fit) and agree in the parameters to the flatness of the surface.
        Assert.True(ours <= nll + (closed ? 1e-10 : 1e-7) * Math.Abs(nll), $"{name}: nll {ours:R} vs optimum {nll:R}");
        Assert.True(ours <= scipyNll + 1e-9 * Math.Abs(scipyNll), $"{name}: nll {ours:R} worse than scipy's fit {scipyNll:R}");
        // The Laplace likelihood is flat between the two middle order statistics (n is even), so any
        // location there is an MLE; scipy's refined optimum and the median differ only in that sense.
        var tol = name == "laplace" ? 0.05 : closed ? 1e-8 : name == "t" ? 2e-2 : 2e-3;
        for (var i = 0; i < got.Length; i++)
        {
            var err = Math.Abs(got[i] - expected[i]) / Math.Max(Math.Abs(expected[i]), 0.1);
            Assert.True(err <= tol, $"{name}: parameter {i} = {got[i]:R}, expected {expected[i]:R} (rel {err:E2})");
        }
    }

    [Fact]
    public void GenericFitHonoursFixedParameters()
    {
        var data = StatsFitReference.Pos;
        var g = Stats.Gamma(1.0).FitTo(data, fixedShapes: [2.0], floc: 0.0);
        Assert.Equal(2.0, g.Shapes[0]);
        Assert.Equal(0.0, g.Loc);
        // With a and loc fixed the scale MLE is mean/a.
        Assert.True(Math.Abs(g.Scale - data.Average() / 2.0) < 1e-12, $"scale {g.Scale}");

        var n = Stats.Fit(Stats.Norm(), data, fscale: 1.5);
        Assert.Equal(1.5, n.Scale);
        Assert.True(Math.Abs(n.Loc - data.Average()) < 1e-12);

        // The numerical path with a fixed scale still finds the location MLE of a logistic.
        var l = Stats.Logistic().FitTo(StatsFitReference.Real, fscale: 1.0);
        Assert.Equal(1.0, l.Scale);
        var free = Stats.Logistic(l.Loc, 1.0).NegativeLogLikelihood(StatsFitReference.Real);
        foreach (var dx in new[] { -1e-3, 1e-3 })
            Assert.True(Stats.Logistic(l.Loc + dx, 1.0).NegativeLogLikelihood(StatsFitReference.Real) > free);
    }

    [Fact]
    public void FitRejectsBadData()
    {
        Assert.Throws<ArgumentException>(() => NormalDistribution.Fit([]));
        Assert.Throws<ArgumentException>(() => NormalDistribution.Fit([1, double.NaN]));
        Assert.Throws<ArgumentException>(() => GammaDistribution.Fit([1, 2, -3], floc: 0));
    }
}
