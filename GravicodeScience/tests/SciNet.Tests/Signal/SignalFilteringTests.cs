using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class SignalFilteringTests
{
    private static readonly Dictionary<string, double[]> R = SignalFilterRef.D;
    private static double[] X => R["x"];
    private static NdArray Sos => new(R["sos"].ToArray(), R["sos"].Length / 6, 6);

    [Fact]
    public void LfilterMatchesScipy()
    {
        var (b, a) = (R["b4"], R["a4"]);
        SignalAssert.Close(R["lfilter"], Signal.Lfilter(b, a, X), 1e-11, 1e-13, "lfilter");
        var zi = Signal.LfilterZi(b, a);
        SignalAssert.Close(R["lfilter_zi"], zi, 1e-12, 1e-14, "zi");
        var (y, zf) = Signal.Lfilter(b, a, X, zi.Select(v => v * X[0]).ToArray());
        SignalAssert.Close(R["lfilter_y_zi"], y, 1e-11, 1e-13, "y zi");
        SignalAssert.Close(R["lfilter_zf"], zf, 1e-11, 1e-13, "zf");
        SignalAssert.Close(R["lfilter_unnorm"], Signal.Lfilter([1, 2, 3], [2, 0.5, 0.25, 0.1], X), 1e-11, 1e-13, "unnormalised a");
        SignalAssert.Close(R["lfilter_fir"], Signal.Lfilter([0.5, 0.3, 0.2, -0.1], [1.0], X), 1e-13, 1e-14, "fir");
        SignalAssert.Close(R["lfilter_first"], Signal.Lfilter([0.3], [1, -0.7], X), 1e-12, 1e-14, "first order");
        SignalAssert.Close(R["lfiltic"], Signal.Lfiltic(b, a, [1.0, 0.5, -0.2], [0.3, 0.1]), 1e-12, 1e-14, "lfiltic");
        SignalAssert.Close(R["lfiltic2"], Signal.Lfiltic([1, 2], [1, -0.5, 0.25, 0.1], [2.0]), 1e-12, 1e-14, "lfiltic2");
    }

    [Fact]
    public void StreamingInBlocksEqualsOneShot()
    {
        // The span kernel carries state, so filtering in chunks must equal filtering once.
        var (b, a) = (R["b4"], R["a4"]);
        var y = new double[X.Length];
        var z = new double[4];
        for (var start = 0; start < X.Length; start += 37)
        {
            var len = Math.Min(37, X.Length - start);
            Signal.Lfilter(b, a, X.AsSpan(start, len), y.AsSpan(start, len), z);
        }
        SignalAssert.Close(R["lfilter"], y, 1e-11, 1e-13, "chunked lfilter");
        var s = Sos.ToArray();
        var ys = new double[X.Length];
        var zs = new double[s.Length / 3];
        for (var start = 0; start < X.Length; start += 50)
            Signal.Sosfilt(s, X.AsSpan(start, 50), ys.AsSpan(start, 50), zs);
        SignalAssert.Close(R["sosfilt"], ys, 1e-10, 1e-13, "chunked sosfilt");
    }

    [Fact]
    public void LfilterAgreesWithNaiveDifferenceEquation()
    {
        // Independent reference: a[0] y[n] = Σ b[k] x[n-k] − Σ_{k≥1} a[k] y[n-k].
        double[] b = [0.2, -0.1, 0.4, 0.05, 0.3];
        double[] a = [1.5, -0.4, 0.2, 0.1];
        var y = Signal.Lfilter(b, a, X);
        var naive = new double[X.Length];
        for (var n = 0; n < X.Length; n++)
        {
            var s = 0.0;
            for (var k = 0; k < b.Length && k <= n; k++) s += b[k] * X[n - k];
            for (var k = 1; k < a.Length && k <= n; k++) s -= a[k] * naive[n - k];
            naive[n] = s / a[0];
        }
        SignalAssert.Close(naive, y, 1e-11, 1e-13, "naive");
    }

    [Fact]
    public void SosfiltMatchesScipy()
    {
        SignalAssert.Close(R["sosfilt"], Signal.Sosfilt(Sos, X), 1e-10, 1e-13, "sosfilt");
        var zi = Signal.SosfiltZi(Sos);
        SignalAssert.Close(R["sosfilt_zi"], zi.ToArray(), 1e-10, 1e-13, "sosfilt_zi");
        var scaled = new NdArray(zi.ToArray().Select(v => v * X[0]).ToArray(), zi.Shape[0], 2);
        var (y, zf) = Signal.Sosfilt(Sos, X, scaled);
        SignalAssert.Close(R["sosfilt_y_zi"], y, 1e-10, 1e-13, "y zi");
        SignalAssert.Close(R["sosfilt_zf"], zf.ToArray(), 1e-10, 1e-13, "zf");
        // Cascade equals the product transfer function (lfilter of sos2tf) for this modest order.
        var (bb, aa) = Signal.Sos2Tf(Sos);
        SignalAssert.Close(Signal.Lfilter(bb, aa, X), Signal.Sosfilt(Sos, X), 1e-7, 1e-9, "sos vs tf");
    }

    [Fact]
    public void FiltfiltMatchesScipy()
    {
        var (b, a) = (R["b4"], R["a4"]);
        SignalAssert.Close(R["filtfilt_odd"], Signal.Filtfilt(b, a, X), 1e-10, 1e-13, "odd");
        SignalAssert.Close(R["filtfilt_even"], Signal.Filtfilt(b, a, X, FiltfiltPad.Even, 30), 1e-10, 1e-13, "even");
        SignalAssert.Close(R["filtfilt_const"], Signal.Filtfilt(b, a, X, FiltfiltPad.Constant), 1e-10, 1e-13, "const");
        SignalAssert.Close(R["filtfilt_none"], Signal.Filtfilt(b, a, X, FiltfiltPad.None), 1e-10, 1e-13, "none");
        SignalAssert.Close(R["filtfilt_gust"], Signal.Filtfilt(b, a, X, method: FiltfiltMethod.Gust), 1e-8, 1e-11, "gust");
        SignalAssert.Close(R["filtfilt_gust_ir"], Signal.Filtfilt(b, a, X, method: FiltfiltMethod.Gust, irlen: 40), 1e-8, 1e-11, "gust irlen");
        SignalAssert.Close(R["sosfiltfilt"], Signal.Sosfiltfilt(Sos, X), 1e-10, 1e-13, "sosfiltfilt");
        SignalAssert.Close(R["sosfiltfilt_even"], Signal.Sosfiltfilt(Sos, X, FiltfiltPad.Even, 50), 1e-10, 1e-13, "sosfiltfilt even");
        Assert.Throws<ArgumentException>(() => Signal.Filtfilt(b, a, X[..10]));
    }

    [Fact]
    public void AxisOverloadsMatchScipy()
    {
        var (b, a) = (R["b4"], R["a4"]);
        var x = new NdArray(R["X"].ToArray(), 3, 50);
        SignalAssert.Close(R["lfilter_axis0"], Signal.Lfilter(b, a, x, axis: 0).ToArray(), 1e-11, 1e-13, "axis0");
        SignalAssert.Close(R["lfilter_axis1"], Signal.Lfilter(b, a, x, axis: 1).ToArray(), 1e-11, 1e-13, "axis1");
        var xt = new NdArray(R["X"].ToArray(), 3, 50).Transpose().Copy();
        SignalAssert.Close(R["sosfilt_axis0"], Signal.Sosfilt(Sos, xt, axis: 0).ToArray(), 1e-10, 1e-13, "sos axis0");
        SignalAssert.Close(R["filtfilt_axis1"], Signal.Filtfilt(b, a, x, axis: 1, padlen: 10).ToArray(), 1e-10, 1e-13, "filtfilt axis1");
    }

    [Fact]
    public void ParallelLinesMatchSerial()
    {
        // 64 lines × 2048 crosses the parallel threshold; each line must equal its own 1-D filter.
        var rng = new Random(7);
        var data = new double[64 * 2048];
        for (var i = 0; i < data.Length; i++) data[i] = rng.NextDouble() - 0.5;
        var y = Signal.Sosfilt(Sos, new NdArray(data, 64, 2048)).ToArray();
        for (var line = 0; line < 64; line += 21)
        {
            var one = Signal.Sosfilt(Sos, data.AsSpan(line * 2048, 2048).ToArray());
            SignalAssert.Close(one, y.AsSpan(line * 2048, 2048), 0, 0, $"line {line}");
        }
    }

    [Fact]
    public void ConversionsMatchScipy()
    {
        var zpk = Signal.Tf2Zpk([1, -1.5, 0.7], [1, 0.3, -0.2, 0.1]);
        SignalAssert.SameRoots(R["tf2zpk_z"], zpk.Z, 1e-13, "tf2zpk z");
        SignalAssert.SameRoots(R["tf2zpk_p"], zpk.P, 1e-13, "tf2zpk p");
        SignalAssert.Close(R["tf2zpk_k"][0], zpk.K, 1e-15, 0, "k");
        var (b10, a10) = Signal.Butter(10, 0.3);
        SignalAssert.SameRoots(R["tf2zpk_b10_p"], Signal.Tf2Zpk(b10, a10).P, 1e-9, "butter10 poles via roots");
        var (_, ac) = Signal.Cheby1(12, 0.5, 0.2);
        SignalAssert.SameRoots(R["roots_c12_p"], Signal.Roots(ac), 1e-8, "cheby1(12) roots");

        Complex[] zz = [-1, -1, new(0.5, 0.5), new(0.5, -0.5), -0.3];
        Complex[] pp = [0.9, Complex.FromPolarCoordinates(0.8, 0.3), Complex.FromPolarCoordinates(0.8, -0.3), new(0, 0.5), new(0, -0.5)];
        var f = new FilterZpk(zz, pp, 2.5);
        SignalAssert.Close(R["zpk2sos_nearest"], Signal.Zpk2Sos(f).ToArray(), 1e-12, 1e-14, "nearest");
        SignalAssert.Close(R["zpk2sos_keep_odd"], Signal.Zpk2Sos(f, SosPairing.KeepOdd).ToArray(), 1e-12, 1e-14, "keep_odd");
        SignalAssert.Close(R["zpk2sos_minimal"], Signal.Zpk2Sos(f, SosPairing.Minimal).ToArray(), 1e-12, 1e-14, "minimal");
        SignalAssert.Close(R["zpk2sos_odd_p"], Signal.Zpk2Sos(new FilterZpk([0.2], [0.5, new(0.3, 0.4), new(0.3, -0.4)], 1.5)).ToArray(), 1e-12, 1e-14, "odd");
        SignalAssert.Close(R["zpk2sos_analog"], Signal.Zpk2Sos(new FilterZpk([Complex.ImaginaryOne, -Complex.ImaginaryOne], [-1, new(-0.5, 2), new(-0.5, -2)], 3.0), analog: true).ToArray(), 1e-12, 1e-14, "analog");

        var sz = Signal.Sos2Zpk(Sos);
        SignalAssert.SameRoots(R["sos2zpk_z"], sz.Z, 1e-9, "sos2zpk z");
        SignalAssert.SameRoots(R["sos2zpk_p"], sz.P, 1e-11, "sos2zpk p");
        SignalAssert.Close(R["sos2zpk_k"][0], sz.K, 1e-12, 0, "sos2zpk k");
        var (sb, sa) = Signal.Sos2Tf(Sos);
        SignalAssert.Close(R["sos2tf_b"], sb, 1e-11, 1e-16, "sos2tf b");
        SignalAssert.Close(R["sos2tf_a"], sa, 1e-11, 1e-14, "sos2tf a");
        SignalAssert.Close(R["tf2sos"], Signal.Tf2Sos([1, -1.5, 0.7, 0.2], [1, 0.3, -0.2, 0.1]).ToArray(), 1e-11, 1e-13, "tf2sos");

        var bl = Signal.Bilinear([1, 2], [1, 3, 2], 10);
        SignalAssert.Close(R["bilinear_b"], bl.B, 1e-13, 1e-15, "bilinear b");
        SignalAssert.Close(R["bilinear_a"], bl.A, 1e-13, 1e-15, "bilinear a");
        bl = Signal.Bilinear([2], [1, 0.5, 4, 1], 2);
        SignalAssert.Close(R["bilinear2_b"], bl.B, 1e-13, 1e-15, "bilinear2 b");
        SignalAssert.Close(R["bilinear2_a"], bl.A, 1e-13, 1e-15, "bilinear2 a");

        var (ba, aa) = Signal.Butter(3, 1, analog: true);
        foreach (var (name, r) in new[]
        {
            ("lp2lp", Signal.Lp2Lp(ba, aa, 2.5)), ("lp2hp", Signal.Lp2Hp(ba, aa, 2.5)),
            ("lp2bp", Signal.Lp2Bp(ba, aa, 2.0, 0.5)), ("lp2bs", Signal.Lp2Bs(ba, aa, 2.0, 0.5)),
            ("lp2hp2", Signal.Lp2Hp([1, 0.5], [1, 2, 3], 3.0)), ("lp2bp2", Signal.Lp2Bp([1, 0.5], [1, 2, 3], 1.5, 0.7)),
        })
        {
            SignalAssert.Close(R[name + "_b"], r.B, 1e-12, 1e-14, name + " b");
            SignalAssert.Close(R[name + "_a"], r.A, 1e-12, 1e-14, name + " a");
        }
        var nrm = Signal.Normalize([0, 0, 2, 4], [0, 2, 1]);
        SignalAssert.Close(R["normalize_b"], nrm.B, 0, 0, "normalize b");
        SignalAssert.Close(R["normalize_a"], nrm.A, 0, 0, "normalize a");
        Assert.Throws<ArgumentException>(() => Signal.Poly([new Complex(1, 1)]));
    }

    [Fact]
    public void FrequencyResponsesMatchScipy()
    {
        var (b, a) = Signal.Butter(4, 0.2);
        var (w, h) = Signal.Freqz(b, a, 16);
        SignalAssert.Close(R["freqz_w"], w, 1e-15, 1e-15, "w");
        SignalAssert.Close(R["freqz_h"], h, 1e-11, 1e-13, "h");
        (w, h) = Signal.Freqz(b, a, 16, whole: true, fs: 100);
        SignalAssert.Close(R["freqz_whole_w"], w, 1e-14, 1e-14, "whole w");
        SignalAssert.Close(R["freqz_whole_h"], h, 1e-11, 1e-13, "whole h");
        var fir = Signal.Firwin(21, 0.3);
        (w, h) = Signal.Freqz(fir, null, 17, includeNyquist: true);
        SignalAssert.Close(R["freqz_fir_w"], w, 1e-15, 1e-15, "fir w");
        SignalAssert.Close(R["freqz_fir_h"], h, 1e-12, 1e-14, "fir h");
        (_, h) = Signal.Freqz(fir, null, 10, whole: true);
        SignalAssert.Close(R["freqz_fir_whole_h"], h, 1e-12, 1e-14, "fir whole h");
        (_, h) = Signal.Freqz(b, a, [0, 10, 25, 40], fs: 100);
        SignalAssert.Close(R["freqz_pts_h"], h, 1e-11, 1e-13, "pts");
        (w, h) = Signal.Sosfreqz(Sos, 32);
        SignalAssert.Close(R["sosfreqz_w"], w, 1e-15, 1e-15, "sos w");
        SignalAssert.Close(R["sosfreqz_h"], h, 1e-9, 1e-12, "sos h");
        var ez = Signal.EllipZpk(4, 1, 40, 0.3);
        (_, h) = Signal.FreqzZpk(ez, 16);
        SignalAssert.Close(R["freqz_zpk_h"], h, 1e-10, 1e-13, "zpk");
        (_, h) = Signal.FreqzZpk(ez, [0.1, 0.2, 0.4], fs: 2);
        SignalAssert.Close(R["freqz_zpk_pts_h"], h, 1e-10, 1e-13, "zpk pts");
        var (cb, ca) = Signal.Cheby1(4, 1, 2.0, analog: true);
        (w, h) = Signal.Freqs(cb, ca, 20);
        SignalAssert.Close(R["freqs_w"], w, 1e-13, 0, "freqs w");
        SignalAssert.Close(R["freqs_h"], h, 1e-10, 1e-13, "freqs h");
        (w, h) = Signal.FreqsZpk(Signal.Cheby1Zpk(4, 1, 2.0, analog: true), 20);
        SignalAssert.Close(R["freqs_zpk_w"], w, 1e-13, 0, "freqs_zpk w");
        SignalAssert.Close(R["freqs_zpk_h"], h, 1e-10, 1e-13, "freqs_zpk h");
        var (gw, gd) = Signal.GroupDelay(b, a, 16);
        SignalAssert.Close(R["gd_w"], gw, 1e-15, 1e-15, "gd w");
        SignalAssert.Close(R["gd"], gd, 1e-10, 1e-12, "gd");
        (_, gd) = Signal.GroupDelay(fir, [1.0], [0.05, 0.2, 0.4], fs: 1);
        SignalAssert.Close(R["gd_fir"], gd, 1e-10, 1e-10, "gd fir");
    }

    [Fact]
    public void FastKernelsMatchScalarReferences()
    {
        var rng = new Random(3);
        var x = Enumerable.Range(0, 10_000).Select(_ => rng.NextDouble() - 0.5).ToArray();
        // Cascades of 1..7 sections exercise the 4/2/1 grouping; the arithmetic per section is
        // unchanged, so the blocked kernel must be bit-identical to scipy's sample-major order.
        foreach (var order in new[] { 2, 3, 5, 8, 11, 14 })
        {
            var sos = Signal.EllipSos(order, 0.5, 60, 0.3).ToArray();
            var zi = Enumerable.Range(0, sos.Length / 3).Select(i => 0.1 * i).ToArray();
            var z1 = zi.ToArray();
            var z2 = zi.ToArray();
            var y1 = new double[x.Length];
            var y2 = new double[x.Length];
            Signal.Sosfilt(sos, x, y1, z1);
            Signal.SosfiltReference(sos, x, y2, z2);
            SignalAssert.Close(y2, y1, 0, 0, $"sos {order}");
            SignalAssert.Close(z2, z1, 0, 0, $"sos state {order}");
            // In place.
            var yi = x.ToArray();
            var z3 = zi.ToArray();
            Signal.Sosfilt(sos, yi, yi, z3);
            SignalAssert.Close(y2, yi, 0, 0, $"sos in place {order}");
        }
        // FIR: SIMD convolution vs the DF2T recurrence, with initial state, short and long inputs.
        foreach (var taps in new[] { 8, 13, 64 })
        {
            var b = Signal.Firwin(taps, 0.3, window: "hann");
            foreach (var len in new[] { 5, 200, 10_000 })
            {
                var xs = x[..len];
                var zi = Enumerable.Range(0, taps - 1).Select(i => 0.01 * i).ToArray();
                var zf = zi.ToArray();
                var zr = zi.ToArray();
                var yf = new double[len];
                var yr = new double[len];
                Signal.Lfilter(b, [1.0], xs, yf, zf);
                Signal.Df2tLoop(b, [1.0, .. new double[taps - 1]], xs, yr, zr);
                SignalAssert.Close(yr, yf, 1e-12, 1e-14, $"fir {taps}/{len}");
                SignalAssert.Close(zr, zf, 1e-12, 1e-14, $"fir state {taps}/{len}");
            }
        }
    }
}
