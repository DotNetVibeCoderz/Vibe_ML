using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.NdImageAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Filters against scipy 1.17.1, every boundary mode (values in <see cref="NdImageExpectedFilters"/>).</summary>
public class NdImageFilterTests
{
    private static double[] E(string key) => NdImageExpectedFilters.Values[key];

    private static readonly double[] W4 = [0.5, -1, 2, 0.25];

    private static NdArray W2() => Arr(new double[,] { { 1, -2, 0.5 }, { 0.25, 3, -1 }, { 2, 0, -0.75 } });

    private static NdArray W3()
    {
        var d = new double[18];
        for (var i = 0; i < 18; i++) d[i] = i % 5 - 1.5;
        return new NdArray(d, 3, 2, 3);
    }

    private static NdArray Cross() => Arr(new double[,] { { 0, 1, 0 }, { 1, 1, 1 }, { 0, 1, 1 } });

    public static TheoryData<NdImageMode> Modes() => NdImageAssert.Modes();

    [Theory]
    [MemberData(nameof(Modes))]
    public void Correlate1d_MatchesScipy(NdImageMode m)
    {
        var a = A2();
        var n = Name(m);
        Close(E($"corr1d_a0_o1_{n}"), NdImage.Correlate1d(a, W4, axis: 0, mode: m, cval: 1.5, origin: 1), what: "axis 0");
        Close(E($"corr1d_a1_om2_{n}"), NdImage.Correlate1d(a, W4, axis: 1, mode: m, cval: 1.5, origin: -2), what: "axis 1");
        Close(E($"conv1d_a1_o1_{n}"), NdImage.Convolve1d(a, W4, axis: 1, mode: m, cval: 1.5, origin: 1), what: "convolve");
        var longKernel = Enumerable.Range(1, 17).Select(i => (double)i).ToArray();
        Close(E($"corr1d_long_{n}"), NdImage.Correlate1d(a, longKernel, axis: 0, mode: m, cval: 1.5), what: "kernel longer than line");
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void CorrelateNd_MatchesScipy(NdImageMode m)
    {
        var n = Name(m);
        Close(E($"corr2d_{n}"), NdImage.Correlate(A2(), W2(), m, 1.5, [0, 1]), what: "corr2d");
        Close(E($"conv2d_{n}"), NdImage.Convolve(A2(), W2(), m, 1.5, [-1, 0]), what: "conv2d");
        Close(E($"corr3d_{n}"), NdImage.Correlate(A3(), W3(), m, 1.5), what: "corr3d");
        Close(E($"conv3d_{n}"), NdImage.Convolve(A3(), W3(), m, 1.5), what: "conv3d");
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void Gaussian_AllOrders_MatchScipy(NdImageMode m)
    {
        var n = Name(m);
        foreach (var (o0, o1) in new[] { (0, 0), (1, 0), (0, 2), (3, 1) })
            Close(E($"gauss_{o0}{o1}_{n}"), NdImage.GaussianFilter(A2(), [1.2, 0.8], [o0, o1], m, 1.5), what: $"order {o0},{o1}");
        Close(E($"gauss3d_{n}"), NdImage.GaussianFilter(A3(), [1.0], [1, 0, 2], m, 1.5), what: "3-D");
        Close(E($"gauss_r_{n}"), NdImage.GaussianFilter1d(A2(), 2.0, axis: 1, mode: m, cval: 1.5, radius: 2), what: "radius");
        Close(E($"glaplace_{n}"), NdImage.GaussianLaplace(A2(), 1.0, m, 1.5), what: "LoG");
        Close(E($"ggm_{n}"), NdImage.GaussianGradientMagnitude(A2(), [1.0, 1.5], m, 1.5), what: "gradient magnitude");
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void UniformAndMinMax_MatchScipy(NdImageMode m)
    {
        var n = Name(m);
        Close(E($"unif_{n}"), NdImage.UniformFilter(A2(), [3, 4], m, 1.5, [0, -1]), what: "uniform");
        Close(E($"unif3d_{n}"), NdImage.UniformFilter(A3(), 3, m, 1.5), what: "uniform 3-D");
        Close(E($"min_{n}"), NdImage.MinimumFilter(A2(), size: [3, 2], mode: m, cval: 1.5), 0, "min");
        Close(E($"max_{n}"), NdImage.MaximumFilter(A2(), size: [2, 5], mode: m, cval: 1.5, origin: [0, 1]), 0, "max");
        Close(E($"minfp_{n}"), NdImage.MinimumFilter(A2(), footprint: Cross(), mode: m, cval: 1.5), 0, "min footprint");
        Close(E($"max3d_{n}"), NdImage.MaximumFilter(A3(), 3, m, 1.5), 0, "max 3-D");
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void RankFilters_MatchScipy(NdImageMode m)
    {
        var n = Name(m);
        Close(E($"median_{n}"), NdImage.MedianFilter(A2(), 3, m, 1.5), 0, "median");
        Close(E($"rankfp_{n}"), NdImage.RankFilter(A2(), -2, footprint: Cross(), mode: m, cval: 1.5), 0, "rank");
        Close(E($"pct_{n}"), NdImage.PercentileFilter(A2(), 30, size: [2, 3], mode: m, cval: 1.5), 0, "percentile");
        Close(E($"median3d_{n}"), NdImage.MedianFilter(A3(), size: [2, 3, 2], mode: m, cval: 1.5), 0, "median 3-D");
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void EdgeFiltersAndGeneric_MatchScipy(NdImageMode m)
    {
        var n = Name(m);
        Close(E($"sobel0_{n}"), NdImage.Sobel(A2(), 0, m, 1.5), what: "sobel");
        Close(E($"prewitt1_{n}"), NdImage.Prewitt(A2(), 1, m, 1.5), what: "prewitt");
        Close(E($"laplace_{n}"), NdImage.Laplace(A2(), m, 1.5), what: "laplace");
        Close(E($"generic_{n}"), NdImage.GenericFilter(A2(), v =>
        {
            double lo = double.MaxValue, hi = double.MinValue;
            foreach (var x in v) { lo = Math.Min(lo, x); hi = Math.Max(hi, x); }
            return hi - lo;
        }, footprint: Cross(), mode: m, cval: 1.5), what: "generic ptp");
    }

    [Theory]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(3.0)]
    public void GaussianKernel_MatchesScipyIncludingDerivatives(double sigma)
    {
        for (var order = 0; order < 4; order++)
            Close(E($"kernel_{sigma:0.0}_{order}"), NdImage.GaussianKernel1d(sigma, order, (int)(4 * sigma + 0.5)), 1e-14, $"order {order}");
    }

    [Fact]
    public void SeparableGaussian_EqualsDirectNdCorrelation()
    {
        // The outer product of the two 1-D kernels, applied as one n-D correlation, must agree with the separable passes.
        // cval is 0 deliberately: with any other constant the second pass pads the *filtered* rows with cval, whereas the
        // direct correlation sees a cval-filled plane - separable and direct genuinely differ there (in scipy too).
        var gy = NdImage.GaussianKernel1d(1.2, 1, 5).Reverse().ToArray();
        var gx = NdImage.GaussianKernel1d(0.8, 0, 3).Reverse().ToArray();
        var w = new double[gy.Length * gx.Length];
        for (var i = 0; i < gy.Length; i++)
            for (var j = 0; j < gx.Length; j++)
                w[i * gx.Length + j] = gy[i] * gx[j];
        foreach (var m in AllModes)
        {
            var direct = NdImage.Correlate(A2(), new NdArray(w, gy.Length, gx.Length), m, 0.0);
            var separable = NdImage.GaussianFilter(A2(), [1.2, 0.8], [1, 0], m, 0.0);
            Close(direct.ToArray(), separable, 1e-12, m.ToString());
        }
    }

    [Fact]
    public void SeparableMinimum_EqualsFootprintPath()
    {
        var big = RandomImage(37, 41, 5);
        var fp = new double[5 * 3];
        Array.Fill(fp, 1.0);
        fp[0] = 0; // forces the footprint path
        var fpPath = NdImage.MinimumFilter(big, footprint: new NdArray(fp, 5, 3));
        // Reference: brute force minimum over the same footprint with reflect boundaries.
        var reference = NdImage.GenericFilter(big, v => { var m = double.MaxValue; foreach (var x in v) m = Math.Min(m, x); return m; },
            footprint: new NdArray(fp, 5, 3));
        Close(reference.ToArray(), fpPath, 0, "footprint");
        var sep = NdImage.MinimumFilter(big, size: [5, 3]);
        var refFull = NdImage.GenericFilter(big, v => { var m = double.MaxValue; foreach (var x in v) m = Math.Min(m, x); return m; }, size: [5, 3]);
        Close(refFull.ToArray(), sep, 0, "separable");
    }

    [Fact]
    public void LargeImage_ParallelPathsAgreeWithNaiveReference()
    {
        // Large enough to take the parallel / blocked line engine on every axis.
        var img = RandomImage(130, 270, 3);
        var g = NdImage.UniformFilter(img, [5, 7], NdImageMode.Wrap).ToArray();
        var src = img.ToArray();
        for (var t = 0; t < 200; t++)
        {
            var i = t * 37 % 130;
            var j = t * 101 % 270;
            var s = 0.0;
            for (var di = -2; di <= 2; di++)
                for (var dj = -3; dj <= 3; dj++)
                    s += src[((i + di + 130) % 130) * 270 + (j + dj + 270) % 270];
            Assert.True(Math.Abs(s / 35 - g[i * 270 + j]) < 1e-12, $"({i},{j})");
        }
    }

    [Fact]
    public void GenericFilter1d_ReproducesCorrelate1d()
    {
        var w = new[] { 1.0, -3.0, 2.5, 0.5 };
        var viaGeneric = NdImage.GenericFilter1d(A2(), (input, output) =>
        {
            for (var k = 0; k < output.Length; k++)
            {
                var s = 0.0;
                for (var j = 0; j < w.Length; j++) s += input[k + j] * w[j];
                output[k] = s;
            }
        }, 4, axis: 0, mode: NdImageMode.Mirror, origin: 1);
        Close(NdImage.Correlate1d(A2(), w, 0, NdImageMode.Mirror, 0, 1).ToArray(), viaGeneric, 1e-12);
    }

    [Fact]
    public void BadArguments_Throw()
    {
        Assert.Throws<ArgumentException>(() => NdImage.Correlate1d(A2(), [1, 2, 3], origin: 2));
        Assert.Throws<ArgumentException>(() => NdImage.Correlate(A2(), new NdArray([1.0, 2.0], 2)));
        Assert.Throws<ArgumentException>(() => NdImage.RankFilter(A2(), 9, size: [3, 3]));
        Assert.Throws<ArgumentException>(() => NdImage.PercentileFilter(A2(), 120, size: [3]));
    }

    internal static NdArray RandomImage(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var d = new double[rows * cols];
        for (var i = 0; i < d.Length; i++) d[i] = Math.Round(rng.NextDouble() * 100) / 4;
        return new NdArray(d, rows, cols);
    }
}
