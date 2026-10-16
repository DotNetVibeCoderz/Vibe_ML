using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpectralAssert;

namespace Gravicode.Science.Tests.SciNet;

public class SignalConvolutionTests
{
    private static readonly ConvolveMode[] Modes = [ConvolveMode.Full, ConvolveMode.Same, ConvolveMode.Valid];

    private static string Name(ConvolveMode m) => m.ToString().ToLowerInvariant();

    [Fact]
    public void Convolve1D_AllModesAndMethods_MatchScipy()
    {
        var x = R("x50");
        foreach (var hn in new[] { "h7", "h8" })
        {
            var h = R(hn);
            foreach (var m in Modes)
                foreach (var method in new[] { ConvolveMethod.Direct, ConvolveMethod.Fft, ConvolveMethod.Auto })
                {
                    Close($"conv_{Name(m)}_{hn}", Signal.Convolve(x, h, m, method), 1e-12);
                    Close($"corr_{Name(m)}_{hn}", Signal.Correlate(x, h, m, method), 1e-12);
                }
        }
    }

    [Fact]
    public void Convolve1D_ShortFirstInput_SwapsForValidAndKeepsSameLength()
    {
        var a = R("h8");
        var b = R("long60");
        foreach (var m in Modes)
            foreach (var method in new[] { ConvolveMethod.Direct, ConvolveMethod.Fft })
            {
                Close($"conv_{Name(m)}_short_long", Signal.Convolve(a, b, m, method), 1e-12);
                Close($"corr_{Name(m)}_short_long", Signal.Correlate(a, b, m, method), 1e-12);
            }
    }

    [Fact]
    public void CorrelationLags_MatchScipy()
    {
        foreach (var m in Modes)
            foreach (var (a, b) in new[] { (10, 4), (4, 10), (9, 4), (5, 5) })
                Equal($"lags_{Name(m)}_{a}_{b}", Signal.CorrelationLags(a, b, m));
    }

    [Fact]
    public void CorrelationLags_LineUpWithCorrelateOutput()
    {
        // A delayed copy must peak exactly at its lag.
        var x = R("x50");
        var y = new double[50];
        for (var i = 7; i < 50; i++) y[i] = x[i - 7];
        var c = Signal.Correlate(y, x);
        var lags = Signal.CorrelationLags(50, 50);
        var best = Array.IndexOf(c, c.Max());
        Assert.Equal(7, lags[best]);
    }

    [Fact]
    public void ConvolveNd_MatchScipy()
    {
        var a = Nd("a2");
        foreach (var m in Modes)
            foreach (var method in new[] { ConvolveMethod.Direct, ConvolveMethod.Fft })
            {
                Close($"nconv_{Name(m)}", Signal.Convolve(a, Nd("b2"), m, method), 1e-12);
                Close($"ncorr_{Name(m)}", Signal.Correlate(a, Nd("b3"), m, method), 1e-12);
                Close($"conv3_{Name(m)}", Signal.Convolve(Nd("a3"), Nd("b3d"), m, method), 1e-12);
            }
        Close("nconv_full", Signal.Fftconvolve(a, Nd("b2")), 1e-12);
        Close("nconv_same", Signal.Oaconvolve(a, Nd("b2"), ConvolveMode.Same), 1e-12);
    }

    [Fact]
    public void Convolve2d_AllBoundariesAndModes_MatchScipy()
    {
        var a = Nd("a2");
        var b = Nd("b3");
        foreach (var m in Modes)
            foreach (var bd in new[] { Convolve2dBoundary.Fill, Convolve2dBoundary.Wrap, Convolve2dBoundary.Symm })
            {
                var bn = bd.ToString().ToLowerInvariant();
                Close($"c2d_{Name(m)}_{bn}", Signal.Convolve2d(a, b, m, bd, 1.5), 1e-12);
                Close($"r2d_{Name(m)}_{bn}", Signal.Correlate2d(a, b, m, bd, 1.5), 1e-12);
            }
        Close("r2d_valid_swapped", Signal.Correlate2d(b, a, ConvolveMode.Valid), 1e-12);
    }

    [Fact]
    public void Oaconvolve_MatchesScipy()
    {
        var x = R("x1000");
        var h = R("h30");
        foreach (var m in Modes)
        {
            Close($"oa_{Name(m)}", Signal.Oaconvolve(x, h, m), 1e-12);
        }
        Close("oa_full", Signal.Oaconvolve(h, x), 1e-12);
        Close("oa_valid", Signal.Oaconvolve(h, x, ConvolveMode.Valid), 1e-12);
    }

    [Fact]
    public void Oaconvolve_LongSignal_EqualsDirect()
    {
        // Enough blocks to take the parallel even/odd path.
        var rng = new Random(5);
        var x = Enumerable.Range(0, 200_000).Select(_ => rng.NextDouble() - 0.5).ToArray();
        var h = Enumerable.Range(0, 101).Select(_ => rng.NextDouble() - 0.5).ToArray();
        var oa = Signal.Oaconvolve(x, h);
        var direct = Signal.Convolve(x, h, ConvolveMode.Full, ConvolveMethod.Direct);
        Close(direct, oa, 1e-12, "oa vs direct");
    }

    [Fact]
    public void Deconvolve_MatchesScipyAndReconstructs()
    {
        var (q, r) = Signal.Deconvolve([1, 3, -2, 5, 7, 1, 2, 9, 4], [1, 0.5, -0.25]);
        Close("deconv_q", q, 1e-12);
        Close("deconv_r", r, 1e-12);
        var back = Signal.Convolve([1, 0.5, -0.25], q);
        for (var i = 0; i < back.Length; i++) back[i] += r[i];
        Close(new double[] { 1, 3, -2, 5, 7, 1, 2, 9, 4 }, back, 1e-12, "reconstruct");
    }

    [Fact]
    public void DirectKernel_SimdAndScalarAgree()
    {
        var rng = new Random(11);
        var x = Enumerable.Range(0, 5000).Select(_ => rng.NextDouble()).ToArray();
        var h = Enumerable.Range(0, 37).Select(_ => rng.NextDouble()).ToArray();
        var simd = Signal.Convolve(x, h, ConvolveMode.Full, ConvolveMethod.Direct);
        var old = Signal.DirectKernelSimd;
        try
        {
            Signal.DirectKernelSimd = false;
            var scalar = Signal.Convolve(x, h, ConvolveMode.Full, ConvolveMethod.Direct);
            Close(scalar, simd, 1e-13, "simd vs scalar");
        }
        finally { Signal.DirectKernelSimd = old; }
    }

    [Fact]
    public void Direct_LargeParallelPath_MatchesNaiveSum()
    {
        var rng = new Random(3);
        var x = Enumerable.Range(0, 30_000).Select(_ => rng.NextDouble() - 0.5).ToArray();
        var h = Enumerable.Range(0, 40).Select(_ => rng.NextDouble() - 0.5).ToArray();
        var y = Signal.Convolve(x, h, ConvolveMode.Same, ConvolveMethod.Direct);
        var start = (h.Length - 1) / 2;
        foreach (var i in new[] { 0, 1, 19, 20, 15_000, 29_980, 29_999 })
        {
            var k = i + start;
            double s = 0;
            for (var j = 0; j < h.Length; j++) if (k - j >= 0 && k - j < x.Length) s += x[k - j] * h[j];
            Assert.True(Math.Abs(s - y[i]) < 1e-12, $"y[{i}]");
        }
    }

    [Fact]
    public void ChooseConvMethod_PrefersDirectForSmallAndNonFinite_FftForLarge()
    {
        Assert.Equal(ConvolveMethod.Direct, Signal.ChooseConvMethod(new double[100], new double[5]));
        Assert.Equal(ConvolveMethod.Fft, Signal.ChooseConvMethod(new double[100_000], new double[5_000]));
        var big = new double[100_000];
        big[3] = double.NaN;
        Assert.Equal(ConvolveMethod.Direct, Signal.ChooseConvMethod(big, new double[5_000]));
        // Auto on NaN input keeps the NaN local instead of smearing it.
        var x = new double[50];
        x[10] = double.NaN;
        var y = Signal.Convolve(x, [1.0, 1.0]);
        Assert.True(double.IsNaN(y[10]) && double.IsNaN(y[11]) && y[0] == 0 && y[30] == 0);
    }

    [Fact]
    public void Convolve_ValidRequiresOneLargerInEveryDim()
    {
        Assert.Throws<ArgumentException>(() => Signal.Convolve(new NdArray(new double[6], 2, 3), new NdArray(new double[6], 3, 2), ConvolveMode.Valid));
    }
}
