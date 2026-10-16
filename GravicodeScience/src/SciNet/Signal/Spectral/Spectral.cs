using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    // ================================================================== periodogram / welch / csd

    /// <summary>
    /// Periodogram — a single-segment Welch estimate over the whole signal
    /// (<c>scipy.signal.periodogram</c>). An <paramref name="nfft"/> shorter than the signal
    /// truncates it, as scipy does.
    /// </summary>
    /// <param name="x">Signal.</param>
    /// <param name="fs">Sampling frequency.</param>
    /// <param name="window">Window; default boxcar.</param>
    /// <param name="nfft">FFT length; default the signal length.</param>
    /// <param name="detrend">Per-segment detrend; null for none.</param>
    /// <param name="returnOnesided">One-sided spectrum (power folded onto positive frequencies).</param>
    /// <param name="scaling">Density (V²/Hz) or spectrum (V²).</param>
    public static (double[] F, double[] Pxx) Periodogram(double[] x, double fs = 1.0, SpectralWindow? window = null, int? nfft = null,
        DetrendType? detrend = DetrendType.Constant, bool returnOnesided = true, SpectralScaling scaling = SpectralScaling.Density)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Length == 0) return ([], []);
        window ??= "boxcar";
        int nperseg;
        if (nfft is null || nfft == x.Length) nperseg = x.Length;
        else if (nfft > x.Length) nperseg = x.Length;
        else
        {
            x = x.AsSpan(0, nfft.Value).ToArray();
            nperseg = nfft.Value;
            nfft = null;
        }
        if (window.Values is { } wv && wv.Length != nperseg)
            throw new ArgumentException("the size of the window must be the same size of the input on the specified axis.", nameof(window));
        return Welch(x, fs, window, nperseg, 0, nfft, detrend, returnOnesided, scaling);
    }

    /// <summary>
    /// Welch's power spectral density estimate (<c>scipy.signal.welch</c>): overlapping segments,
    /// each detrended, windowed and transformed, then averaged.
    /// </summary>
    /// <remarks>
    /// Segments go through a cached real-FFT plan with per-worker buffers and are processed in
    /// parallel once there is enough work; the mean is accumulated on the fly, so memory stays
    /// O(nfft) however long the signal. A named window longer than the signal is silently
    /// shortened to it (scipy warns). <see cref="SpectralAverage.Median"/> divides by
    /// <c>1 + Σ_{i=1}^{(K−1)/2} (1/(2i+1) − 1/(2i))</c>, the bias of the median of K χ²₂ variates.
    /// </remarks>
    /// <param name="x">Signal.</param>
    /// <param name="fs">Sampling frequency.</param>
    /// <param name="window">Window; default periodic Hann.</param>
    /// <param name="nperseg">Segment length; default 256 (or the window's length).</param>
    /// <param name="noverlap">Overlap; default half a segment.</param>
    /// <param name="nfft">FFT length (zero padding); default nperseg.</param>
    /// <param name="detrend">Per-segment detrend; null for none.</param>
    /// <param name="returnOnesided">One-sided spectrum.</param>
    /// <param name="scaling">Density or spectrum.</param>
    /// <param name="average">Mean or bias-corrected median.</param>
    public static (double[] F, double[] Pxx) Welch(double[] x, double fs = 1.0, SpectralWindow? window = null, int? nperseg = null,
        int? noverlap = null, int? nfft = null, DetrendType? detrend = DetrendType.Constant, bool returnOnesided = true,
        SpectralScaling scaling = SpectralScaling.Density, SpectralAverage average = SpectralAverage.Mean)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Length == 0) return ([], []);
        var (f, pxy, preal) = CsdCore(x, null, fs, window, nperseg, noverlap, nfft, detrend, returnOnesided, scaling, average);
        return (f, preal ?? pxy.Select(v => v.Real).ToArray());
    }

    /// <summary><see cref="Welch(double[], double, SpectralWindow, int?, int?, int?, DetrendType?, bool, SpectralScaling, SpectralAverage)"/>
    /// along one axis; the axis is replaced by frequency.</summary>
    public static (double[] F, NdArray Pxx) Welch(NdArray x, double fs = 1.0, SpectralWindow? window = null, int? nperseg = null,
        int? noverlap = null, int? nfft = null, DetrendType? detrend = DetrendType.Constant, bool returnOnesided = true,
        SpectralScaling scaling = SpectralScaling.Density, SpectralAverage average = SpectralAverage.Mean, int axis = -1)
    {
        var n = SignalAxisLines.Length(x, axis);
        var np = (window?.Values?.Length) ?? Math.Min(nperseg ?? 256, n);
        var nf = nfft ?? np;
        var f = returnOnesided ? Fft.RfftFreq(nf, 1 / fs) : Fft.FftFreq(nf, 1 / fs);
        var p = SignalAxisLines.Map(x, axis, f.Length,
            (s, d) => Welch(s.ToArray(), fs, window, nperseg, noverlap, nfft, detrend, returnOnesided, scaling, average).Pxx.CopyTo(d));
        return (f, p);
    }

    /// <summary>
    /// Cross power spectral density by Welch's method (<c>scipy.signal.csd</c>):
    /// <c>Pxy = mean(conj(X)·Y)</c> with the same scaling as <see cref="Welch(double[], double, SpectralWindow, int?, int?, int?, DetrendType?, bool, SpectralScaling, SpectralAverage)"/>.
    /// The shorter input is zero-padded to the longer.
    /// </summary>
    public static (double[] F, Complex[] Pxy) Csd(double[] x, double[] y, double fs = 1.0, SpectralWindow? window = null, int? nperseg = null,
        int? noverlap = null, int? nfft = null, DetrendType? detrend = DetrendType.Constant, bool returnOnesided = true,
        SpectralScaling scaling = SpectralScaling.Density, SpectralAverage average = SpectralAverage.Mean)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Length == 0 || y.Length == 0) return ([], []);
        if (x.Length != y.Length)
        {
            var n = Math.Max(x.Length, y.Length);
            if (x.Length < n) { var t = new double[n]; x.CopyTo(t, 0); x = t; }
            if (y.Length < n) { var t = new double[n]; y.CopyTo(t, 0); y = t; }
        }
        var (f, pxy, _) = CsdCore(x, y, fs, window, nperseg, noverlap, nfft, detrend, returnOnesided, scaling, average);
        return (f, pxy);
    }

    /// <summary>
    /// Magnitude-squared coherence <c>|Pxy|² / (Pxx·Pyy)</c> (<c>scipy.signal.coherence</c>), in [0, 1].
    /// </summary>
    public static (double[] F, double[] Cxy) Coherence(double[] x, double[] y, double fs = 1.0, SpectralWindow? window = null, int? nperseg = null,
        int? noverlap = null, int? nfft = null, DetrendType? detrend = DetrendType.Constant)
    {
        var (f, pxx) = Welch(x, fs, window, nperseg, noverlap, nfft, detrend);
        var (_, pyy) = Welch(y, fs, window, nperseg, noverlap, nfft, detrend);
        var (_, pxy) = Csd(x, y, fs, window, nperseg, noverlap, nfft, detrend);
        var c = new double[f.Length];
        for (var i = 0; i < c.Length; i++)
        {
            var m = pxy[i].Magnitude;
            c[i] = m * m / pxx[i] / pyy[i];
        }
        return (f, c);
    }

    private static (double[] F, Complex[] Pxy, double[]? Preal) CsdCore(double[] x, double[]? y, double fs, SpectralWindow? window,
        int? nperseg, int? noverlap, int? nfft, DetrendType? detrend, bool onesided, SpectralScaling scaling, SpectralAverage average)
    {
        var n = x.Length;
        window ??= "hann_periodic";
        double[] win;
        int np;
        if (window.Values is null)
        {
            np = nperseg ?? 256;
            if (np < 1) throw new ArgumentException($"Parameter nperseg={np} is not a positive integer!", nameof(nperseg));
            if (n < np) np = n;
            win = GetWindow(window.Spec!, np);
        }
        else
        {
            win = (double[])window.Values.Clone();
            np = nperseg ?? win.Length;
            if (np != win.Length) throw new ArgumentException($"nperseg={np} does not equal len(win)={win.Length}.", nameof(nperseg));
            if (np > n) throw new ArgumentException("window is longer than input signal.", nameof(window));
        }
        var nf = nfft ?? np;
        if (nf < np) throw new ArgumentException($"nfft={nf} must be greater than or equal to nperseg={np}!", nameof(nfft));
        var nov = noverlap ?? np / 2;
        if (nov >= np) throw new ArgumentException($"noverlap={nov} must be less than nperseg={np}!", nameof(noverlap));
        if (nov < 0) throw new ArgumentException("noverlap must be non-negative.", nameof(noverlap));
        var scale = Scale(win, fs, scaling);
        var engine = new SignalSpectralEngine(win, np, nov, nf, detrend, onesided);
        var nseg = engine.Segments(n);
        var bins = engine.Bins;
        var freqs = onesided ? Fft.RfftFreq(nf, 1 / fs) : Fft.FftFreq(nf, 1 / fs);
        var doubleTo = nf % 2 == 0 ? bins - 1 : bins;
        if (average == SpectralAverage.Mean || nseg <= 1)
        {
            var sum = engine.SumCross(x, y, nseg, out var real);
            var factor = scale / nseg;
            if (real is not null)
            {
                for (var k = 0; k < bins; k++) real[k] *= factor * (onesided && k >= 1 && k < doubleTo ? 2 : 1);
                return (freqs, [], real);
            }
            for (var k = 0; k < bins; k++) sum[k] *= factor * (onesided && k >= 1 && k < doubleTo ? 2 : 1);
            return (freqs, sum, null);
        }
        var all = engine.AllSegments(x, y, nseg, power: y is null);
        var bias = MedianBias(nseg);
        var re = new double[nseg];
        var im = new double[nseg];
        var outC = new Complex[bins];
        var outR = y is null ? new double[bins] : null;
        for (var k = 0; k < bins; k++)
        {
            for (var s = 0; s < nseg; s++)
            {
                var v = all[(long)k * nseg + s];
                re[s] = v.Real;
                im[s] = v.Imaginary;
            }
            var f = scale * (onesided && k >= 1 && k < doubleTo ? 2 : 1) / bias;
            var mr = MedianOf(re) * f;
            if (outR is not null) outR[k] = mr;
            else outC[k] = new Complex(mr, MedianOf(im) * f);
        }
        return (freqs, outC, outR);
    }

    private static double Scale(double[] win, double fs, SpectralScaling scaling)
    {
        double s = 0, s2 = 0;
        foreach (var w in win) { s += w; s2 += w * w; }
        return scaling == SpectralScaling.Density ? 1.0 / (fs * s2) : 1.0 / (s * s);
    }

    private static double MedianBias(int n)
    {
        double b = 1;
        for (var i = 1; i <= (n - 1) / 2; i++)
        {
            var ii2 = 2.0 * i;
            b += 1.0 / (ii2 + 1) - 1.0 / ii2;
        }
        return b;
    }

    // ================================================================== spectrogram / stft / istft

    /// <summary>
    /// Spectrogram — consecutive Fourier transforms of overlapping segments
    /// (<c>scipy.signal.spectrogram</c>). Returns frequencies, segment-centre times, and a
    /// bins × segments array.
    /// </summary>
    /// <param name="x">Signal.</param>
    /// <param name="fs">Sampling frequency.</param>
    /// <param name="window">Window; default <c>("tukey_periodic", 0.25)</c>.</param>
    /// <param name="nperseg">Segment length; default 256 (or the window's length).</param>
    /// <param name="noverlap">Overlap; default nperseg/8.</param>
    /// <param name="nfft">FFT length; default nperseg.</param>
    /// <param name="detrend">Per-segment detrend; null for none.</param>
    /// <param name="returnOnesided">One-sided spectrum.</param>
    /// <param name="scaling">Density or spectrum.</param>
    /// <param name="mode">PSD, or the complex/magnitude/angle/phase of the scaled STFT.</param>
    /// <returns>For <see cref="SpectrogramMode.Complex"/> the values are in <c>SxxComplex</c> and
    /// <c>Sxx</c> is null; otherwise the reverse.</returns>
    public static (double[] F, double[] T, NdArray? Sxx, ComplexNdArray? SxxComplex) Spectrogram(double[] x, double fs = 1.0,
        SpectralWindow? window = null, int? nperseg = null, int? noverlap = null, int? nfft = null, DetrendType? detrend = DetrendType.Constant,
        bool returnOnesided = true, SpectralScaling scaling = SpectralScaling.Density, SpectrogramMode mode = SpectrogramMode.Psd)
    {
        ArgumentNullException.ThrowIfNull(x);
        window ??= ("tukey_periodic", 0.25);
        var (win, np) = window.Resolve(nperseg, x.Length);
        var nov = noverlap ?? np / 8;
        var (f, t, z) = SpectralHelper(x, fs, win, np, nov, nfft, detrend, returnOnesided, scaling, mode == SpectrogramMode.Psd, StftBoundary.None, false);
        var bins = f.Length;
        var nseg = t.Length;
        if (mode == SpectrogramMode.Complex) return (f, t, null, new ComplexNdArray(z, bins, nseg));
        var r = new double[z.Length];
        switch (mode)
        {
            case SpectrogramMode.Psd:
                for (var i = 0; i < r.Length; i++) r[i] = z[i].Real;
                break;
            case SpectrogramMode.Magnitude:
                for (var i = 0; i < r.Length; i++) r[i] = z[i].Magnitude;
                break;
            default:
                for (var i = 0; i < r.Length; i++) r[i] = Math.Atan2(z[i].Imaginary, z[i].Real);
                if (mode == SpectrogramMode.Phase)
                {
                    // numpy.unwrap along the frequency axis, column by column.
                    var col = new double[bins];
                    for (var s = 0; s < nseg; s++)
                    {
                        for (var k = 0; k < bins; k++) col[k] = r[k * nseg + s];
                        UnwrapInPlace(col);
                        for (var k = 0; k < bins; k++) r[k * nseg + s] = col[k];
                    }
                }
                break;
        }
        return (f, t, new NdArray(r, bins, nseg), null);
    }

    /// <summary>
    /// Short-time Fourier transform (<c>scipy.signal.stft</c>, legacy interface): the signal is
    /// extended by half a segment at each end (<paramref name="boundary"/>) and zero-padded to a
    /// whole number of steps (<paramref name="padded"/>), so <see cref="Istft"/> can invert it.
    /// </summary>
    /// <param name="x">Signal.</param>
    /// <param name="fs">Sampling frequency.</param>
    /// <param name="window">Window; default periodic Hann.</param>
    /// <param name="nperseg">Segment length (default 256, shortened to the signal length).</param>
    /// <param name="noverlap">Overlap; default half a segment.</param>
    /// <param name="nfft">FFT length; default nperseg.</param>
    /// <param name="detrend">Per-segment detrend; default none.</param>
    /// <param name="returnOnesided">One-sided spectrum.</param>
    /// <param name="boundary">End extension.</param>
    /// <param name="padded">Zero-pad so the last segment is complete.</param>
    /// <param name="scaling">Spectrum (magnitude) or density (scipy's "psd").</param>
    /// <returns>Frequencies, segment times and the bins × segments STFT.</returns>
    public static (double[] F, double[] T, ComplexNdArray Zxx) Stft(double[] x, double fs = 1.0, SpectralWindow? window = null, int nperseg = 256,
        int? noverlap = null, int? nfft = null, DetrendType? detrend = null, bool returnOnesided = true, StftBoundary boundary = StftBoundary.Zeros,
        bool padded = true, SpectralScaling scaling = SpectralScaling.Spectrum)
    {
        ArgumentNullException.ThrowIfNull(x);
        window ??= "hann_periodic";
        var (win, np) = window.Resolve(nperseg, x.Length);
        var (f, t, z) = SpectralHelper(x, fs, win, np, noverlap, nfft, detrend, returnOnesided, scaling, false, boundary, padded);
        return (f, t, new ComplexNdArray(z, f.Length, t.Length));
    }

    /// <summary>
    /// Inverse STFT by weighted overlap-add (<c>scipy.signal.istft</c>) for one-sided input,
    /// returning a real signal.
    /// </summary>
    /// <remarks>Each segment is multiplied by the window and the sum divided by the overlapped
    /// squared window, so any window satisfying NOLA (<see cref="CheckNola"/>) inverts exactly;
    /// samples where the squared-window sum is below 1e-10 are left undivided (scipy warns there).</remarks>
    /// <param name="zxx">STFT, bins × segments.</param>
    /// <param name="fs">Sampling frequency.</param>
    /// <param name="window">Window used for the forward transform.</param>
    /// <param name="nperseg">Segment length; default inferred from the bin count.</param>
    /// <param name="noverlap">Overlap; default half a segment.</param>
    /// <param name="nfft">FFT length; default inferred.</param>
    /// <param name="boundary">Strip the half-segment extension the forward transform added.</param>
    /// <param name="scaling">Scaling the forward transform used.</param>
    public static (double[] T, double[] X) Istft(ComplexNdArray zxx, double fs = 1.0, SpectralWindow? window = null, int? nperseg = null,
        int? noverlap = null, int? nfft = null, bool boundary = true, SpectralScaling scaling = SpectralScaling.Spectrum)
    {
        var (t, c) = IstftCore(zxx, fs, window, nperseg, noverlap, nfft, true, boundary, scaling);
        return (t, c.Select(v => v.Real).ToArray());
    }

    /// <summary>Inverse STFT of two-sided input (complex signal).</summary>
    public static (double[] T, Complex[] X) IstftComplex(ComplexNdArray zxx, double fs = 1.0, SpectralWindow? window = null, int? nperseg = null,
        int? noverlap = null, int? nfft = null, bool boundary = true, SpectralScaling scaling = SpectralScaling.Spectrum)
        => IstftCore(zxx, fs, window, nperseg, noverlap, nfft, false, boundary, scaling);

    private static (double[] T, Complex[] X) IstftCore(ComplexNdArray zxx, double fs, SpectralWindow? window, int? nperseg, int? noverlap,
        int? nfft, bool onesided, bool boundary, SpectralScaling scaling)
    {
        ArgumentNullException.ThrowIfNull(zxx);
        if (zxx.Rank != 2) throw new ArgumentException("Input stft must be 2-D (bins × segments).", nameof(zxx));
        window ??= "hann_periodic";
        var bins = zxx.Shape[0];
        var nseg = zxx.Shape[1];
        var nDefault = onesided ? 2 * (bins - 1) : bins;
        var np = nperseg ?? nDefault;
        if (np < 1) throw new ArgumentException("nperseg must be a positive integer.", nameof(nperseg));
        int nf;
        if (nfft is null) nf = onesided && np == nDefault + 1 ? np : nDefault;
        else if (nfft < np) throw new ArgumentException("nfft must be greater than or equal to nperseg.", nameof(nfft));
        else nf = nfft.Value;
        var nov = noverlap ?? np / 2;
        if (nov >= np) throw new ArgumentException("noverlap must be less than nperseg.", nameof(noverlap));
        var step = np - nov;
        var win = window.ValuesFor(np);
        double gain = 0;
        if (scaling == SpectralScaling.Spectrum) foreach (var w in win) gain += w;
        else { foreach (var w in win) gain += w * w; gain = Math.Sqrt(fs * gain); }
        var outLen = np + (nseg - 1) * step;
        var x = new Complex[outLen];
        var norm = new double[outLen];
        var data = zxx.Data;
        var col = new Complex[bins];
        for (var s = 0; s < nseg; s++)
        {
            for (var k = 0; k < bins; k++) col[k] = data[k * nseg + s];
            Complex[] seg;
            if (onesided)
            {
                var r = Fft.Irfft(col, nf);
                seg = new Complex[np];
                for (var i = 0; i < np && i < r.Length; i++) seg[i] = r[i];
            }
            else seg = Fft.Ifft(col, nf);
            var off = s * step;
            for (var i = 0; i < np; i++)
            {
                x[off + i] += (i < seg.Length ? seg[i] : 0) * gain * win[i];
                norm[off + i] += win[i] * win[i];
            }
        }
        var lo = 0;
        var hi = outLen;
        if (boundary) { lo = np / 2; hi = outLen - np / 2; }
        if (hi < lo) hi = lo;
        var res = new Complex[hi - lo];
        for (var i = lo; i < hi; i++) res[i - lo] = x[i] / (norm[i] > 1e-10 ? norm[i] : 1.0);
        var t = new double[res.Length];
        for (var i = 0; i < t.Length; i++) t[i] = i / fs;
        return (t, res);
    }

    /// <summary>
    /// scipy's legacy <c>_spectral_helper</c> for one real signal: returns frequencies, times and a
    /// bins × segments buffer (power when <paramref name="psd"/>, else the scaled STFT).
    /// </summary>
    private static (double[] F, double[] T, Complex[] Z) SpectralHelper(double[] x, double fs, double[] win, int np, int? noverlap, int? nfft,
        DetrendType? detrend, bool onesided, SpectralScaling scaling, bool psd, StftBoundary boundary, bool padded)
    {
        var nf = nfft ?? np;
        if (nf < np) throw new ArgumentException("nfft must be greater than or equal to nperseg.", nameof(nfft));
        var nov = noverlap ?? np / 2;
        if (nov >= np) throw new ArgumentException("noverlap must be less than nperseg.", nameof(noverlap));
        if (nov < 0) throw new ArgumentException("noverlap must be non-negative.", nameof(noverlap));
        var step = np - nov;
        if (boundary != StftBoundary.None)
        {
            var ext = np / 2;
            if (boundary is StftBoundary.Even or StftBoundary.Odd && ext >= x.Length && ext > 0)
                throw new ArgumentException($"The extension length n ({ext}) is too big. It must not exceed x.shape[axis]-1, which is {x.Length - 1}.");
            var mode = boundary switch
            {
                StftBoundary.Even => UpfirdnPad.Reflect,
                StftBoundary.Odd => UpfirdnPad.Antireflect,
                StftBoundary.Constant => UpfirdnPad.Edge,
                _ => UpfirdnPad.Constant,
            };
            var xe = new double[x.Length + 2 * ext];
            for (var i = 0; i < xe.Length; i++) xe[i] = SignalPadding.Extend(x, i - ext, mode, 0);
            x = xe;
        }
        if (padded)
        {
            var nadd = Mod(Mod(-(x.Length - np), step), np);
            if (nadd > 0)
            {
                var xp = new double[x.Length + nadd];
                x.CopyTo(xp, 0);
                x = xp;
            }
        }
        var scale = Scale(win, fs, scaling);
        if (!psd) scale = Math.Sqrt(scale);
        var engine = new SignalSpectralEngine(win, np, nov, nf, detrend, onesided);
        var nseg = engine.Segments(x.Length);
        var bins = engine.Bins;
        var z = engine.AllSegments(x, null, nseg, power: psd);
        var doubleTo = nf % 2 == 0 ? bins - 1 : bins;
        for (var k = 0; k < bins; k++)
        {
            var f = scale * (psd && onesided && k >= 1 && k < doubleTo ? 2 : 1);
            for (var s = 0; s < nseg; s++) z[(long)k * nseg + s] *= f;
        }
        var freqs = onesided ? Fft.RfftFreq(nf, 1 / fs) : Fft.FftFreq(nf, 1 / fs);
        var t = new double[nseg];
        for (var s = 0; s < nseg; s++)
        {
            t[s] = (np / 2.0 + s * (double)step) / fs;
            if (boundary != StftBoundary.None) t[s] -= np / 2.0 / fs;
        }
        return (freqs, t, z);

        static int Mod(int a, int m) => ((a % m) + m) % m;
    }

    // ================================================================== COLA / NOLA

    /// <summary>
    /// Whether overlapped copies of the window sum to a constant (<c>scipy.signal.check_COLA</c>) —
    /// the condition under which plain overlap-add of unmodified segments reconstructs the signal.
    /// </summary>
    public static bool CheckCola(SpectralWindow window, int nperseg, int noverlap, double tol = 1e-10)
    {
        var binsums = OverlapSums(window, nperseg, noverlap, squared: false);
        var med = MedianOf(binsums);
        double dev = 0;
        foreach (var b in binsums) dev = Math.Max(dev, Math.Abs(b - med));
        return dev < tol;
    }

    /// <summary>
    /// Whether the overlapped squared window is nowhere zero (<c>scipy.signal.check_NOLA</c>) — the
    /// weaker condition <see cref="Istft"/> needs to invert an STFT.
    /// </summary>
    public static bool CheckNola(SpectralWindow window, int nperseg, int noverlap, double tol = 1e-10)
    {
        if (noverlap < 0) throw new ArgumentException("noverlap must be a nonnegative integer.", nameof(noverlap));
        var binsums = OverlapSums(window, nperseg, noverlap, squared: true);
        return binsums.Min() > tol;
    }

    private static double[] OverlapSums(SpectralWindow window, int nperseg, int noverlap, bool squared)
    {
        ArgumentNullException.ThrowIfNull(window);
        if (nperseg < 1) throw new ArgumentException("nperseg must be a positive integer.", nameof(nperseg));
        if (noverlap >= nperseg) throw new ArgumentException("noverlap must be less than nperseg.", nameof(noverlap));
        var win = window.ValuesFor(nperseg);
        var step = nperseg - noverlap;
        var sums = new double[step];
        for (var ii = 0; ii < nperseg / step; ii++)
            for (var j = 0; j < step; j++)
            {
                var w = win[ii * step + j];
                sums[j] += squared ? w * w : w;
            }
        var rem = nperseg % step;
        for (var j = 0; j < rem; j++)
        {
            var w = win[nperseg - rem + j];
            sums[j] += squared ? w * w : w;
        }
        return sums;
    }

    // ================================================================== Lomb-Scargle

    /// <summary>
    /// Generalised Lomb–Scargle periodogram of unevenly sampled data (<c>scipy.signal.lombscargle</c>,
    /// scipy 1.15+ semantics) at the <em>angular</em> frequencies <paramref name="freqs"/>.
    /// </summary>
    /// <remarks>
    /// One <c>SinCos</c> per (sample, frequency): the τ-shifted sums scipy evaluates with a second
    /// pass of cos/sin are rotations of the first-pass sums, <c>Σ w y cos(ωx − τ) = cos τ·Σ w y cos ωx +
    /// sin τ·Σ w y sin ωx</c> and so on, which halves the transcendental work. Frequencies run in
    /// parallel. scipy's deprecated <c>precenter</c> is not offered: subtract the mean from
    /// <paramref name="y"/>, or use <paramref name="floatingMean"/>.
    /// </remarks>
    /// <param name="x">Sample times.</param>
    /// <param name="y">Observations.</param>
    /// <param name="freqs">Angular frequencies.</param>
    /// <param name="floatingMean">Fit a constant offset per frequency (the "floating mean" model).</param>
    /// <param name="normalize">Unnormalised power or normalised to [0, 1].</param>
    /// <param name="weights">Non-negative weights (default all 1); normalised to sum 1.</param>
    public static double[] Lombscargle(double[] x, double[] y, double[] freqs, bool floatingMean = false,
        LombscargleNormalization normalize = LombscargleNormalization.Power, double[]? weights = null)
    {
        var (p, _) = LombscargleCore(x, y, freqs, floatingMean, normalize, weights, amplitude: false);
        return p;
    }

    /// <summary>
    /// The complex amplitude <c>(a + ib)·e^{iτ}</c> of the best-fit sinusoid at each frequency —
    /// scipy's <c>normalize="amplitude"</c>.
    /// </summary>
    public static Complex[] LombscargleAmplitude(double[] x, double[] y, double[] freqs, bool floatingMean = false, double[]? weights = null)
        => LombscargleCore(x, y, freqs, floatingMean, LombscargleNormalization.Power, weights, amplitude: true).Amp!;

    private static (double[] P, Complex[]? Amp) LombscargleCore(double[] x, double[] y, double[] freqs, bool floatingMean,
        LombscargleNormalization normalize, double[]? weights, bool amplitude)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        ArgumentNullException.ThrowIfNull(freqs);
        var n = x.Length;
        if (n == 0 || y.Length != n || (weights is not null && weights.Length != n))
            throw new ArgumentException("Parameters x, y, weights must be 1-D arrays of equal non-zero length!");
        if (freqs.Length == 0) throw new ArgumentException("Parameter freqs must be a 1-D array of non-zero length!", nameof(freqs));
        var w = new double[n];
        double wsum = 0;
        for (var i = 0; i < n; i++)
        {
            w[i] = weights?[i] ?? 1.0;
            if (!(w[i] >= 0)) throw new ArgumentException("Parameter weights must have only non-negative entries which sum to a positive value!", nameof(weights));
            wsum += w[i];
        }
        if (!(wsum > 0)) throw new ArgumentException("Parameter weights must have only non-negative entries which sum to a positive value!", nameof(weights));
        var inv = 1.0 / wsum;
        double Y = 0, YY = 0;
        var wy = new double[n];
        for (var i = 0; i < n; i++)
        {
            w[i] *= inv;
            wy[i] = w[i] * y[i];
            Y += wy[i];
            YY += wy[i] * y[i];
        }
        if (floatingMean) YY -= Y * Y;
        var p = new double[freqs.Length];
        var amp = amplitude ? new Complex[freqs.Length] : null;
        const double epsneg = 1.1102230246251565e-16;
        void Freq(int j)
        {
            var om = freqs[j];
            double cc = 0, cs = 0, c1 = 0, s1 = 0, yc = 0, ys = 0;
            for (var i = 0; i < n; i++)
            {
                var (sn, cn) = Math.SinCos(om * x[i]);
                var wi = w[i];
                cc += wi * cn * cn;
                cs += wi * cn * sn;
                c1 += wi * cn;
                s1 += wi * sn;
                yc += wy[i] * cn;
                ys += wy[i] * sn;
            }
            var ss = 1.0 - cc;
            double CC = cc, SS = ss, CS = cs;
            if (floatingMean)
            {
                CC -= c1 * c1;
                SS -= s1 * s1;
                CS -= c1 * s1;
            }
            var tau = 0.5 * Math.Atan2(2.0 * CS, CC - SS);
            var (st, ct) = Math.SinCos(tau);
            // Rotate the first-pass sums by τ.
            var ycT = ct * yc + st * ys;
            var ysT = ct * ys - st * yc;
            var ccT = ct * ct * cc + 2 * ct * st * cs + st * st * ss;
            var ssT = 1.0 - ccT;
            if (floatingMean)
            {
                var cT = ct * c1 + st * s1;
                var sT = ct * s1 - st * c1;
                ycT -= Y * cT;
                ysT -= Y * sT;
                ccT -= cT * cT;
                ssT -= sT * sT;
            }
            if (ccT < epsneg) ccT = epsneg;
            if (ssT < epsneg) ssT = epsneg;
            var a = ycT / ccT;
            var b = ysT / ssT;
            if (amp is not null)
            {
                amp[j] = new Complex(a, b) * Complex.Exp(new Complex(0, tau));
                return;
            }
            var pg = 2.0 * (a * ycT + b * ysT);
            p[j] = normalize == LombscargleNormalization.Power ? pg * n / 4.0 : pg * 0.5 / YY;
        }
        if ((long)n * freqs.Length > 1 << 15) Parallel.For(0, freqs.Length, Freq);
        else for (var j = 0; j < freqs.Length; j++) Freq(j);
        return (p, amp);
    }
}
