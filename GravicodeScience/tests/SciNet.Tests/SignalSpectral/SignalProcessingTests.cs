using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpectralAssert;

namespace Gravicode.Science.Tests.SciNet;

public class SignalProcessingTests
{
    [Fact]
    public void SavgolCoeffs_MatchScipy()
    {
        Close("sgc_7_2", Signal.SavgolCoeffs(7, 2), 1e-12);
        Close("sgc_8_3", Signal.SavgolCoeffs(8, 3), 1e-12);
        Close("sgc_5_2_d1", Signal.SavgolCoeffs(5, 2, deriv: 1, delta: 0.5), 1e-12);
        Close("sgc_7_3_pos1", Signal.SavgolCoeffs(7, 3, pos: 1), 1e-12);
        Close("sgc_9_4_dot", Signal.SavgolCoeffs(9, 4, use: SavgolUse.Dot), 1e-12);
        Close("sgc_11_2_d2", Signal.SavgolCoeffs(11, 2, deriv: 2), 1e-12);
        // scipy's lstsq on the raw Vandermonde matrix loses ~1e-11 here; the exact rational
        // solution (sympy) is matched to rounding, so pin against that and only loosely against scipy.
        Close("sgc_31_6", Signal.SavgolCoeffs(31, 6), 1e-9);
        Close("sgc_31_6_exact", Signal.SavgolCoeffs(31, 6), 1e-13);
        Close("sgc_5_2_d3", Signal.SavgolCoeffs(5, 2, deriv: 3), 1e-12);
    }

    [Fact]
    public void SavgolCoeffs_ReproducePolynomialsExactly()
    {
        // A window-length-9, order-3 filter must return a cubic's value and derivative unchanged.
        var c0 = Signal.SavgolCoeffs(9, 3, use: SavgolUse.Dot);
        var c1 = Signal.SavgolCoeffs(9, 3, deriv: 1, use: SavgolUse.Dot);
        double P(double t) => 2 - t + 0.5 * t * t - 0.1 * t * t * t;
        double dP(double t) => -1 + t - 0.3 * t * t;
        double v = 0, d = 0;
        for (var i = 0; i < 9; i++) { v += c0[i] * P(i - 4); d += c1[i] * P(i - 4); }
        Assert.True(Math.Abs(v - P(0)) < 1e-12 && Math.Abs(d - dP(0)) < 1e-12);
    }

    [Fact]
    public void SavgolFilter_AllModesMatchScipy()
    {
        var x = R("x50");
        foreach (var mode in new[] { SavgolMode.Interp, SavgolMode.Mirror, SavgolMode.Nearest, SavgolMode.Constant, SavgolMode.Wrap })
        {
            var name = mode.ToString().ToLowerInvariant();
            Close($"sg_{name}", Signal.SavgolFilter(x, 7, 2, mode: mode, cval: 0.3), 1e-12);
            Close($"sg_{name}_d1", Signal.SavgolFilter(x, 9, 3, deriv: 1, delta: 0.1, mode: mode), 1e-12);
            Close($"sg_{name}_even", Signal.SavgolFilter(x, 8, 3, mode: mode), 1e-12);
        }
        Close("sg_interp_d2", Signal.SavgolFilter(x, 11, 4, deriv: 2), 1e-11);
        Close("sg_mirror_long", Signal.SavgolFilter(x[..5], 9, 2, mode: SavgolMode.Mirror), 1e-12);
        Assert.Throws<ArgumentException>(() => Signal.SavgolFilter(x[..5], 9, 2));
    }

    [Fact]
    public void Medfilt_MatchesScipy()
    {
        var x = R("xm");
        foreach (var k in new[] { 3, 5, 11, 21 }) Close($"medfilt_{k}", Signal.Medfilt(x, k), 0);
        var im = Nd("im");
        Close("medfilt2d_3", Signal.Medfilt2d(im), 0);
        Close("medfilt2d_3_5", Signal.Medfilt2d(im, 3, 5), 0);
        Close("medfilt2d_7", Signal.Medfilt2d(im, 7), 0);
        Close("medfilt_nd_3", Signal.Medfilt(Nd("a3")), 0);
        Close("medfilt2d_3_5", Signal.Medfilt(im, [3, 5]), 0);
    }

    [Fact]
    public void Medfilt_SlidingWindowAgreesWithSortingEveryWindow()
    {
        var rng = new Random(9);
        var x = Enumerable.Range(0, 20_000).Select(_ => Math.Round(rng.NextDouble() * 20) - 10).ToArray(); // many ties; long enough for the parallel chunks
        var y = Signal.Medfilt(x, 51);
        for (var i = 0; i < x.Length; i += 7)
        {
            var w = Enumerable.Range(i - 25, 51).Select(j => j < 0 || j >= x.Length ? 0.0 : x[j]).OrderBy(v => v).ToArray();
            Assert.Equal(w[25], y[i]);
        }
    }

    [Fact]
    public void OrderFilterAndWiener_MatchScipy()
    {
        var im = Nd("im");
        var dom = new NdArray(new double[] { 1, 0, 1, 0, 1, 0, 1, 1, 1 }, 3, 3);
        Close("order_r0", Signal.OrderFilter(im, dom, 0), 0);
        Close("order_r3", Signal.OrderFilter(im, dom, 3), 0);
        Close("order_r5", Signal.OrderFilter(im, dom, 5), 0);
        Close("wiener_im", Signal.Wiener(im), 1e-12);
        Close("wiener_im_5_3", Signal.Wiener(im, [5, 3], 0.5), 1e-12);
        Close("wiener_im_4", Signal.Wiener(im, [4]), 1e-12);
        Close("wiener_1d", Signal.Wiener(R("xm"), 5), 1e-12);
    }

    [Fact]
    public void Hilbert_MatchesScipyAndHasRealPartX()
    {
        var x = R("x64");
        Close("hilbert", Signal.Hilbert(x), 1e-12);
        Close("hilbert_50", Signal.Hilbert(x, 50), 1e-12);
        Close("hilbert_81", Signal.Hilbert(x, 81), 1e-12);
        var h = Signal.Hilbert(x);
        for (var i = 0; i < x.Length; i++) Assert.True(Math.Abs(h[i].Real - x[i]) < 1e-12);
        Close("hilbert2", Signal.Hilbert2(Nd("im")), 1e-12);
        Close("hilbert2_8_5", Signal.Hilbert2(Nd("im"), (8, 5)), 1e-12);
        Close("hilbert_axis0", Signal.Hilbert(Nd("im"), axis: 0).Data.ToArray(), 1e-12);
    }

    [Fact]
    public void Hilbert_OfCosineIsSine()
    {
        var n = 256;
        var x = Enumerable.Range(0, n).Select(i => Math.Cos(2 * Math.PI * 8 * i / n)).ToArray();
        var h = Signal.Hilbert(x);
        for (var i = 0; i < n; i++) Assert.True(Math.Abs(h[i].Imaginary - Math.Sin(2 * Math.PI * 8 * i / n)) < 1e-12);
    }

    [Fact]
    public void Envelope_MatchesScipy()
    {
        var x = R("x64");
        var cases = new (int? S, int? E, int? N, bool Sq, EnvelopeResidual Res)[]
        {
            (1, null, null, false, EnvelopeResidual.Lowpass), (2, 10, null, true, EnvelopeResidual.All), (null, null, 40, false, EnvelopeResidual.Lowpass),
            (-5, 7, 90, false, EnvelopeResidual.All), (3, 12, 31, false, EnvelopeResidual.None), (1, null, 64, false, EnvelopeResidual.Lowpass),
        };
        for (var i = 0; i < cases.Length; i++)
        {
            var c = cases[i];
            var (env, res) = Signal.Envelope(x, c.S, c.E, c.N, c.Sq, c.Res);
            Close($"env{i}_e", env, 1e-12);
            if (c.Res == EnvelopeResidual.None) Assert.Null(res);
            else Close($"env{i}_r", res!, 1e-12);
        }
    }

    [Fact]
    public void Detrend_MatchesScipy()
    {
        var x = R("xt");
        Close("detrend_lin", Signal.Detrend(x), 1e-12);
        Close("detrend_const", Signal.Detrend(x, DetrendType.Constant), 1e-12);
        Close("detrend_bp", Signal.Detrend(x, bp: [30, 70]), 1e-12);
        Close("detrend_bp1", Signal.Detrend(x, bp: [99]), 1e-12);
        Close("detrend_axis0", Signal.Detrend(Nd("im"), axis: 0), 1e-12);
    }

    [Fact]
    public void ChirpZ_MatchesScipyAndFft()
    {
        var x = R("xz");
        Close("czt", Signal.Czt(x), 1e-12);
        Close("czt_m20", Signal.Czt(x, 20), 1e-12);
        Close("czt_wa", Signal.Czt(x, 9, 0.98 * Complex.Exp(new Complex(0, -0.3)), 1.1 * Complex.Exp(new Complex(0, 0.2))), 1e-11);
        Close("czt_pts", Signal.CztPoints(5, 0.9 * Complex.Exp(new Complex(0, -0.4)), new Complex(0.8, 0.1)), 1e-13);
        Close("zoom", Signal.ZoomFft(x, 0.2, 0.6, 17), 1e-12);
        Close("zoom_ep", Signal.ZoomFft(x, 0.1, 0.9, 11, fs: 3, endpoint: true), 1e-12);
        Close("zoom_fn", Signal.ZoomFft(x, 0.5), 1e-12);
        var plain = Fft.Fft1D(x);
        var cz = new CZT(x.Length).Transform(x);
        for (var i = 0; i < x.Length; i++) Assert.True((plain[i] - cz[i]).Magnitude < 1e-12);
    }

    [Fact]
    public void Vectorstrength_MatchesScipy()
    {
        var (s, p) = Signal.Vectorstrength(R("events"), [0.7, 1.3, 2.0]);
        Close("vs_strength", s, 1e-12);
        Close("vs_phase", p, 1e-12);
        var (s1, _) = Signal.Vectorstrength([0.0, 1.0, 2.0, 3.0], 1.0);
        Assert.True(Math.Abs(s1 - 1) < 1e-12);
    }

    [Fact]
    public void UnwrapPhase_RemovesJumps()
    {
        var truth = Enumerable.Range(0, 100).Select(i => 0.3 * i).ToArray();
        var wrapped = truth.Select(v => Math.Atan2(Math.Sin(v), Math.Cos(v))).ToArray();
        var u = Signal.UnwrapPhase(wrapped);
        for (var i = 0; i < 100; i++) Assert.True(Math.Abs(u[i] - truth[i]) < 1e-12);
    }
}
