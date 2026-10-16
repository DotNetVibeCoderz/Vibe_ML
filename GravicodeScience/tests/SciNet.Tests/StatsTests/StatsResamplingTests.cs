using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsTestsAssert;
using R = Gravicode.Science.Tests.SciNet.StatsTestsRef;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Permutation tests, Monte Carlo tests, bootstrap and jackknife.</summary>
public class StatsResamplingTests
{
    private static double Mean(double[] a) => a.Average();

    [Fact]
    public void ExactPermutationTestsMatchScipy()
    {
        foreach (var (name, alt) in Alternatives)
        {
            var r = Stats.PermutationTest(R.RA, R.RB, (a, b) => Mean(a) - Mean(b), alternative: alt);
            Close($"perm/ind/{name}", 1e-12, 1e-15, r.Statistic, r.Pvalue, r.NullDistribution.Length);
            r = Stats.PermutationTest([R.PY], d => Stats.Pearsonr(R.PX, d[0]).Statistic, PermutationType.Pairings, alternative: alt);
            Close($"perm/pair/{name}", 1e-12, 1e-15, r.Statistic, r.Pvalue, r.NullDistribution.Length);
            r = Stats.PermutationTest([R.SDIFF], d => Mean(d[0]), PermutationType.Samples, alternative: alt);
            Close($"perm/sign/{name}", 1e-12, 1e-15, r.Statistic, r.Pvalue, r.NullDistribution.Length);
        }
        var k3 = Stats.PermutationTest([R.RC, R.RD, R.RE], d => Math.Pow(Mean(d[0]), 2) - Mean(d[1]) + d[2].Max());
        Close("perm/ind3", 1e-12, 1e-15, k3.Statistic, k3.Pvalue, k3.NullDistribution.Length);
        var s2 = Stats.PermutationTest(R.SA, R.SB, (a, b) => a.Zip(b, (u, v) => u - v).Average(), PermutationType.Samples);
        Close("perm/samples2", 1e-12, 1e-15, s2.Statistic, s2.Pvalue, s2.NullDistribution.Length);
    }

    [Fact]
    public void RandomPermutationTestApproachesExact()
    {
        // 9! / (5! 4!) = 126 · ... too few for Monte Carlo; use 8 + 8 (12870 partitions) with B = 4000.
        var a = R.X[..8];
        var b = R.Y[..8];
        Func<double[], double[], double> stat = (u, v) => Mean(u) - Mean(v);
        var exact = Stats.PermutationTest(a, b, stat, nResamples: 20000);
        Assert.Equal(12870, exact.NullDistribution.Length);
        var mc = Stats.PermutationTest(a, b, stat, nResamples: 4000, seed: 3);
        Assert.Equal(4000, mc.NullDistribution.Length);
        // Binomial standard error of the estimate at B = 4000 is under 0.008.
        Assert.True(Math.Abs(mc.Pvalue - exact.Pvalue) < 0.04, $"MC {mc.Pvalue} vs exact {exact.Pvalue}");
        var par = Stats.PermutationTest(a, b, stat, nResamples: 4000, seed: 3, parallel: true);
        Assert.Equal(mc.NullDistribution, par.NullDistribution);
    }

    [Fact]
    public void MonteCarloTestHasUniformNullPvalues()
    {
        // Under H0 (data truly normal) p-values of a skewness test should be roughly uniform.
        var rng = new GraviRandom(11);
        var below = 0;
        const int trials = 60;
        for (var t = 0; t < trials; t++)
        {
            var sample = Enumerable.Range(0, 30).Select(_ => rng.Normal()).ToArray();
            var r = Stats.MonteCarloTest(sample, Stats.Norm(), s => Stats.Skew(s), nResamples: 199, seed: t);
            Assert.InRange(r.Pvalue, 1.0 / 200, 1.0);
            if (r.Pvalue < 0.5) below++;
        }
        Assert.InRange(below, 15, 45);
        // A strongly skewed sample is rejected.
        var skewed = Enumerable.Range(0, 30).Select(_ => rng.Exponential()).ToArray();
        var rej = Stats.MonteCarloTest(skewed, Stats.Norm(), s => Stats.Skew(s), nResamples: 999, alternative: Alternative.Greater, seed: 1);
        Assert.True(rej.Pvalue < 0.01, $"p = {rej.Pvalue}");
    }

    [Fact]
    public void BootstrapIntervalsMatchScipyForAGivenDistribution()
    {
        foreach (var m in new[] { "percentile", "basic", "BCa" })
            foreach (var (name, alt) in Alternatives)
            {
                var r = Stats.Bootstrap(R.BOOTS, Mean, nResamples: 0, confidenceLevel: 0.9, method: m, alternative: alt, bootstrapDistribution: R.BOOTD);
                Close($"boot/{m}/{name}", 1e-12, 1e-14, r.ConfidenceInterval.Low, r.ConfidenceInterval.High, r.StandardError);
            }
        var two = Stats.Bootstrap([R.BOOTS[..12], R.BOOTS[12..]], d => Mean(d[0]) - Stats.Quantile(d[1], 0.5), nResamples: 0, bootstrapDistribution: R.BOOTD);
        Close("boot/bca2", 1e-12, 1e-14, two.ConfidenceInterval.Low, two.ConfidenceInterval.High);
        var paired = Stats.Bootstrap([R.BOOTS[..12], R.BOOTS[12..24]], d => d[0].Zip(d[1], (u, v) => u * v).Average(), nResamples: 0, paired: true, bootstrapDistribution: R.BOOTD);
        Close("boot/bca_paired", 1e-12, 1e-14, paired.ConfidenceInterval.Low, paired.ConfidenceInterval.High);
    }

    [Fact]
    public void BootstrapStandardErrorAndCoverage()
    {
        // The bootstrap SE of the mean approaches s/√n.
        var r = Stats.Bootstrap(R.L1, Mean, nResamples: 4000, seed: 5, method: "percentile");
        var se = Math.Sqrt(R.L1.Select(v => (v - R.L1.Average()) * (v - R.L1.Average())).Sum() / R.L1.Length) / Math.Sqrt(R.L1.Length);
        Assert.True(Math.Abs(r.StandardError - se) / se < 0.06, $"SE {r.StandardError} vs {se}");
        // Coverage of the BCa interval for an exponential mean over repeated experiments.
        var rng = new GraviRandom(21);
        var covered = 0;
        const int reps = 40;
        for (var t = 0; t < reps; t++)
        {
            var s = Enumerable.Range(0, 40).Select(_ => rng.Exponential()).ToArray();
            var ci = Stats.Bootstrap(s, Mean, nResamples: 499, seed: t, parallel: true).ConfidenceInterval;
            if (ci.Low <= 1.0 && 1.0 <= ci.High) covered++;
        }
        Assert.InRange(covered, 31, 40);
        var seq = Stats.Bootstrap(R.X, Mean, nResamples: 300, seed: 9);
        var par = Stats.Bootstrap(R.X, Mean, nResamples: 300, seed: 9, parallel: true);
        Assert.Equal(seq.BootstrapDistribution, par.BootstrapDistribution);
    }

    [Fact]
    public void JackknifeOfTheMeanIsExact()
    {
        var j = Stats.Jackknife(R.X, Mean);
        var n = R.X.Length;
        var s = Math.Sqrt(R.X.Select(v => (v - R.X.Average()) * (v - R.X.Average())).Sum() / (n - 1));
        Near(R.X.Average(), j.Estimate, 1e-13, 1e-15, "estimate");
        Near(0, j.Bias, 0, 1e-13, "bias");
        Near(s / Math.Sqrt(n), j.StandardError, 1e-12, 0, "se = s/√n");
        // The plug-in variance has bias −σ²/n; the jackknife removes it exactly.
        var jv = Stats.Jackknife(R.X, a => Stats.Moment(a, 2));
        Near(s * s, jv.Estimate, 1e-12, 0, "unbiased variance");
    }
}
