using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.StatsTestsAssert;
using R = Gravicode.Science.Tests.SciNet.StatsTestsRef;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Gaussian KDE against scipy 1.17.1 and against a naive kernel sum.</summary>
public class StatsKdeTests
{
    [Fact]
    public void OneDimensionalMatchesScipy()
    {
        foreach (var (key, kde) in new[]
                 {
                     ("scott", new GaussianKde(R.KD1)), ("silverman", new GaussianKde(R.KD1, "silverman")), ("0.3", new GaussianKde(R.KD1, 0.3)),
                 })
        {
            Close($"kde1/{key}", 1e-11, 1e-15, [.. kde.Evaluate(R.KPTS), .. kde.LogPdf(R.KPTS), kde.Factor, .. Flat(kde.Covariance), kde.Neff,
                kde.IntegrateBox1d(-1.0, 1.5), kde.IntegrateGaussian([0.5], new NdArray([0.7], 1, 1))]);
        }
        var w = new GaussianKde(R.KD1, weights: R.KW);
        Close("kde1w", 1e-11, 1e-15, [.. w.Evaluate(R.KPTS), w.Factor, .. Flat(w.Covariance), w.Neff, w.IntegrateBox1d(-1.0, 1.5)]);
        // Far in the tails the density underflows but the log density must not.
        Close("kde1_far", 1e-11, 0, w.LogPdf([-60.0, 80.0]));
    }

    [Fact]
    public void MultiDimensionalMatchesScipy()
    {
        var kd2 = new NdArray(R.KD2, 2, 80);
        var pts = new NdArray(R.KP2, 2, 15);
        var k = new GaussianKde(kd2);
        Close("kde2", 1e-11, 1e-15, [.. k.Evaluate(pts), .. k.LogPdf(pts), k.Factor, .. Flat(k.Covariance),
            k.IntegrateGaussian([0.2, 0.9], new NdArray([0.5, 0.1, 0.1, 0.4], 2, 2))]);
        // Genz's algorithm is randomised; scipy's own answer carries ~1e-5 absolute error.
        Close("kde2_box", 0, 2e-5, k.IntegrateBox([-1.0, 0], [1.0, 2], seed: 1));
        var kw = new GaussianKde(kd2, "silverman", R.KW[..80]);
        Close("kde2w", 1e-11, 1e-15, [.. kw.Evaluate(pts), kw.Factor, .. Flat(kw.Covariance), kw.Neff]);
        Close("kde_kde", 1e-11, 1e-15, k.IntegrateKde(kw), new GaussianKde(R.KD1).IntegrateKde(new GaussianKde(R.KD1[..50], 0.5)));
        var mg = k.Marginal(1);
        Close("kde_marginal", 1e-11, 1e-15, [.. mg.Evaluate(R.KPTS[..5]), .. Flat(mg.Covariance)]);
        var k3 = new GaussianKde(new NdArray(R.KD3, 3, 60));
        Close("kde3", 1e-11, 1e-15, [.. k3.Evaluate(new NdArray(R.KP3, 3, 6)), .. Flat(k3.Covariance)]);
    }

    [Fact]
    public void TwoDimensionalBoxIntegralAgreesWithQuadrature()
    {
        // Σ wᵢ P(box | N(xᵢ, Σ)), each term by Simpson's rule on φ(u)·[Φ((h₂ − ρu)/√(1−ρ²)) − Φ((l₂ − ρu)/√(1−ρ²))].
        foreach (var scale in new[] { 1.0, 3.0 })
        {
            var raw = R.KD2.ToArray();
            for (var i = 80; i < 160; i++) raw[i] = raw[i - 80] * scale + raw[i] * 0.05; // strongly correlated when scale = 3
            var kde = new GaussianKde(new NdArray(raw, 2, 80));
            var c = Flat(kde.Covariance);
            double s1 = Math.Sqrt(c[0]), s2 = Math.Sqrt(c[3]), rho = c[1] / (s1 * s2);
            var sr = Math.Sqrt(1 - rho * rho);
            double[] lo = [-1.0, -0.5], hi = [1.0, 2.0];
            var w = kde.Weights;
            var expected = 0.0;
            for (var i = 0; i < 80; i++)
            {
                double l1 = (lo[0] - raw[i]) / s1, h1 = (hi[0] - raw[i]) / s1, l2 = (lo[1] - raw[80 + i]) / s2, h2 = (hi[1] - raw[80 + i]) / s2;
                const int steps = 4000;
                var s = 0.0;
                for (var j = 0; j <= steps; j++)
                {
                    var u = l1 + (h1 - l1) * j / steps;
                    var f = Math.Exp(-0.5 * u * u) / Math.Sqrt(2 * Math.PI) * (Special.Ndtr((h2 - rho * u) / sr) - Special.Ndtr((l2 - rho * u) / sr));
                    s += (j == 0 || j == steps ? 1 : j % 2 == 1 ? 4 : 2) * f;
                }
                expected += w[i] * s * (h1 - l1) / steps / 3;
            }
            Assert.True(Math.Abs(rho) > (scale > 1 ? 0.95 : 0.3), $"rho = {rho}");
            Near(expected, kde.IntegrateBox(lo, hi), 1e-9, 1e-12, $"box, rho {rho}");
        }
    }

    [Fact]
    public void VectorisedEvaluationAgreesWithNaiveSum()
    {
        // Large enough to take the parallel and SIMD paths; the reference is the definition.
        var rng = new GraviRandom(4);
        var data = Enumerable.Range(0, 3001).Select(_ => rng.Normal()).ToArray();
        var kde = new GaussianKde(data);
        var h = Math.Sqrt(Flat(kde.Covariance)[0]);
        var pts = Enumerable.Range(0, 257).Select(i => -5 + i * 10.0 / 256).ToArray();
        var fast = kde.Evaluate(pts);
        for (var j = 0; j < pts.Length; j++)
        {
            var s = 0.0;
            foreach (var x in data) s += Math.Exp(-0.5 * Math.Pow((pts[j] - x) / h, 2));
            s /= data.Length * h * Math.Sqrt(2 * Math.PI);
            Near(s, fast[j], 1e-12, 1e-300, $"pdf({pts[j]})");
        }
    }

    [Fact]
    public void BinnedApproximationIsCloseWhereTheDensityIsNotTiny()
    {
        var kde = new GaussianKde(R.KD1);
        var exact = kde.Evaluate(R.KPTS);
        var binned = kde.EvaluateBinned(R.KPTS);
        for (var i = 0; i < exact.Length; i++)
            if (exact[i] > 1e-3) Near(exact[i], binned[i], 1e-3, 1e-6, $"binned at {R.KPTS[i]}");
    }

    [Fact]
    public void ResampleReproducesTheMixtureMoments()
    {
        var kde = new GaussianKde(R.KD1);
        var s = Flat(kde.Resample(200_000, seed: 1));
        var mean = s.Average();
        var var = s.Select(v => (v - mean) * (v - mean)).Sum() / s.Length;
        // Mixture mean is the data mean; mixture variance is the (biased) data variance plus h².
        var dm = R.KD1.Average();
        var dv = R.KD1.Select(v => (v - dm) * (v - dm)).Sum() / R.KD1.Length + Flat(kde.Covariance)[0];
        Assert.True(Math.Abs(mean - dm) < 0.02, $"mean {mean} vs {dm}");
        Assert.True(Math.Abs(var - dv) / dv < 0.02, $"var {var} vs {dv}");
        var k2 = new GaussianKde(new NdArray(R.KD2, 2, 80));
        var r2 = k2.Resample(5, seed: 3);
        Assert.Equal([2, 5], r2.Shape.ToArray());
    }

    [Fact]
    public void BandwidthCanBeChanged()
    {
        var kde = new GaussianKde(R.KD1);
        var scott = kde.Factor;
        kde.SetBandwidth("silverman");
        Near(Math.Pow(100 * 3 / 4.0, -1 / 5.0), kde.Factor, 1e-15, 0, "silverman factor");
        kde.SetBandwidth(k => k.ScottsFactor() / 2);
        Near(scott / 2, kde.Factor, 1e-15, 0, "callable factor");
        Assert.Throws<ArgumentException>(() => kde.SetBandwidth("nope"));
        Assert.Throws<ArgumentException>(() => new GaussianKde([1.0]));
    }
}
