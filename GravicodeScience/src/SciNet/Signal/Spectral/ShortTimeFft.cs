using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Which FFT <see cref="ShortTimeFft"/> uses and how its frequency axis is laid out (scipy's <c>fft_mode</c>).</summary>
public enum ShortTimeFftMode
{
    /// <summary>Real FFT, non-negative frequencies only (scipy's default).</summary>
    Onesided,
    /// <summary>Full complex FFT in FFT order (0, positive, negative frequencies).</summary>
    Twosided,
    /// <summary>Full complex FFT with the zero frequency in the middle (<c>fftshift</c>ed).</summary>
    Centered,
    /// <summary>One-sided with the non-DC/non-Nyquist bins multiplied by 2 (magnitude) or √2 (psd), so
    /// one-sided values carry the energy of both halves. Requires a scaling.</summary>
    Onesided2X,
}

/// <summary>Scaling of a <see cref="ShortTimeFft"/> window (scipy's <c>scaling</c>).</summary>
public enum ShortTimeFftScaling
{
    /// <summary>The window sums to 1: |S| is the amplitude of a sinusoid.</summary>
    Magnitude,
    /// <summary>Σ w² = T: |S|² is a power spectral density.</summary>
    Psd,
    /// <summary>Unitary (only from <see cref="ShortTimeFft.FromWinEqualsDual"/>): window equals its dual up to √mfft.</summary>
    Unitary,
}

/// <summary>How <see cref="ShortTimeFft"/> extends the signal beyond its ends (scipy's <c>padding</c>).</summary>
public enum ShortTimeFftPadding
{
    /// <summary>Zeros.</summary>
    Zeros,
    /// <summary>Repeat the edge sample.</summary>
    Edge,
    /// <summary>Even reflection about the edge sample (numpy <c>reflect</c>).</summary>
    Even,
    /// <summary>Odd (point) reflection about the edge sample.</summary>
    Odd,
}

/// <summary>
/// The short-time Fourier transform as scipy's <c>ShortTimeFFT</c> class: a window slid by
/// <see cref="Hop"/> samples, slice <c>p</c> centred on sample <c>p·hop</c>, and an inverse by the
/// dual window. Slices are indexed from <see cref="PMin"/> (negative: slices that overhang the
/// signal start) to <see cref="PMax"/>(n), so the transform covers every sample with full weight
/// and <see cref="Istft"/> is exact.
/// </summary>
/// <remarks>
/// Only real windows are supported (scipy also accepts complex ones). Input may be real, or
/// complex for the two-sided modes. Slices are transformed in parallel once the work is large
/// enough, except when a custom detrend callback is given (it is called serially, once per slice,
/// with a fresh copy, as scipy does for a 1-D signal).
/// </remarks>
public sealed class ShortTimeFft
{
    private double[] _win;
    private double[]? _dualWin;
    private double _fs;
    private ShortTimeFftMode _fftMode = ShortTimeFftMode.Onesided;
    private int _mfft;
    private int? _phaseShift;
    private (int K, int P)? _preCache;

    /// <summary>Creates the transform (scipy's <c>ShortTimeFFT(win, hop, fs, ...)</c>).</summary>
    /// <param name="win">Window, real, finite, non-empty.</param>
    /// <param name="hop">Step between slices in samples, ≥ 1.</param>
    /// <param name="fs">Sampling frequency.</param>
    /// <param name="fftMode">FFT layout.</param>
    /// <param name="mfft">FFT length (zero padding), at least the window length; default the window length.</param>
    /// <param name="dualWin">Dual window; default computed (canonical dual) on first use.</param>
    /// <param name="scaleTo">Rescale the window to magnitude or psd units.</param>
    /// <param name="phaseShift">Shift of the slice's phase reference in samples (|v| &lt; mfft); 0 makes
    /// the slice centre the time origin of its FFT, null uses the slice start (no roll).</param>
    public ShortTimeFft(double[] win, int hop, double fs, ShortTimeFftMode fftMode = ShortTimeFftMode.Onesided, int? mfft = null,
        double[]? dualWin = null, ShortTimeFftScaling? scaleTo = null, int? phaseShift = 0)
    {
        ArgumentNullException.ThrowIfNull(win);
        if (win.Length == 0) throw new ArgumentException("Parameter win must be 1d and non-empty.", nameof(win));
        foreach (var v in win) if (!double.IsFinite(v)) throw new ArgumentException("Parameter win must have finite entries!", nameof(win));
        if (hop < 1) throw new ArgumentException($"Parameter hop={hop} is not an integer >= 1!", nameof(hop));
        _win = (double[])win.Clone();
        Hop = hop;
        Fs = fs;
        Mfft = mfft ?? win.Length;
        if (dualWin is not null)
        {
            if (dualWin.Length != win.Length) throw new ArgumentException($"dual_win length {dualWin.Length} must equal win length {win.Length}!", nameof(dualWin));
            foreach (var v in dualWin) if (!double.IsFinite(v)) throw new ArgumentException("Parameter dual_win must be a finite array!", nameof(dualWin));
            _dualWin = (double[])dualWin.Clone();
        }
        if (scaleTo is { } s) ScaleTo(s);
        FftMode = fftMode;
        PhaseShift = phaseShift;
    }

    /// <summary>scipy's <c>ShortTimeFFT.from_window</c>: a named window of length <paramref name="nperseg"/>
    /// (periodic unless <paramref name="symmetricWin"/>) with hop = nperseg − noverlap.</summary>
    public static ShortTimeFft FromWindow(WindowSpec window, double fs, int nperseg, int noverlap, bool symmetricWin = false,
        ShortTimeFftMode fftMode = ShortTimeFftMode.Onesided, int? mfft = null, ShortTimeFftScaling? scaleTo = null, int? phaseShift = 0)
    {
        var win = Signal.GetWindow(window, nperseg, fftbins: !symmetricWin);
        return new ShortTimeFft(win, nperseg - noverlap, fs, fftMode, mfft, null, scaleTo, phaseShift);
    }

    /// <summary>scipy's <c>ShortTimeFFT.from_dual</c>: the window is the canonical dual of the given dual window.</summary>
    public static ShortTimeFft FromDual(double[] dualWin, int hop, double fs, ShortTimeFftMode fftMode = ShortTimeFftMode.Onesided,
        int? mfft = null, ShortTimeFftScaling? scaleTo = null, int? phaseShift = 0)
    {
        ArgumentNullException.ThrowIfNull(dualWin);
        var win = CanonicalDual(dualWin, hop);
        return new ShortTimeFft(win, hop, fs, fftMode, mfft, dualWin, scaleTo, phaseShift);
    }

    /// <summary>
    /// scipy's <c>ShortTimeFFT.from_win_equals_dual</c>: the window closest to
    /// <paramref name="desiredWin"/> that is its own dual (each hop-strided comb normalised to unit norm).
    /// </summary>
    public static ShortTimeFft FromWinEqualsDual(double[] desiredWin, int hop, double fs, ShortTimeFftMode fftMode = ShortTimeFftMode.Onesided,
        int? mfft = null, ShortTimeFftScaling? scaleTo = null, int? phaseShift = 0)
    {
        ArgumentNullException.ThrowIfNull(desiredWin);
        if (desiredWin.Length == 0) throw new ArgumentException("Parameter desired_win is not 1d.", nameof(desiredWin));
        foreach (var v in desiredWin) if (!double.IsFinite(v)) throw new ArgumentException("Parameter desired_win must have finite entries!", nameof(desiredWin));
        if (hop < 1 || hop > desiredWin.Length) throw new ArgumentException($"Parameter hop={hop} is not an integer between 1 and len(desired_win)={desiredWin.Length}!", nameof(hop));
        var m = mfft ?? desiredWin.Length;
        var sFac = scaleTo == ShortTimeFftScaling.Unitary ? Math.Sqrt(m) : 1.0;
        var win = (double[])desiredWin.Clone();
        var resolution = 1e-15 * desiredWin.Max();
        for (var k = 0; k < hop; k++)
        {
            double a = 0;
            for (var i = k; i < win.Length; i += hop) a += desiredWin[i] * desiredWin[i];
            a = Math.Sqrt(a);
            if (!(a > resolution)) throw new ArgumentException($"Parameter desired_win does not have valid STFT dual window for hop={hop}!", nameof(desiredWin));
            for (var i = k; i < win.Length; i += hop) win[i] /= a;
        }
        var w = new double[win.Length]; var d = new double[win.Length];
        for (var i = 0; i < win.Length; i++) { w[i] = win[i] / sFac; d[i] = win[i] * sFac; }
        var sft = new ShortTimeFft(w, hop, fs, fftMode, m, d, scaleTo == ShortTimeFftScaling.Unitary ? null : scaleTo, phaseShift);
        if (scaleTo == ShortTimeFftScaling.Unitary) sft.Scaling = ShortTimeFftScaling.Unitary;
        return sft;
    }

    // ------------------------------------------------------------------ properties

    /// <summary>The window (a copy).</summary>
    public double[] Win => (double[])_win.Clone();

    /// <summary>Step between slices in samples.</summary>
    public int Hop { get; }

    /// <summary>Sampling frequency; must be positive.</summary>
    public double Fs
    {
        get => _fs;
        set
        {
            if (!(value > 0)) throw new ArgumentException($"Sampling frequency fs={value} must be positive!");
            _fs = value;
        }
    }

    /// <summary>Sampling interval 1/fs; must be positive.</summary>
    public double T
    {
        get => 1 / _fs;
        set
        {
            if (!(value > 0)) throw new ArgumentException($"Sampling interval T={value} must be positive!");
            _fs = 1 / value;
        }
    }

    /// <summary>FFT layout. <see cref="ShortTimeFftMode.Onesided2X"/> requires a scaling.</summary>
    public ShortTimeFftMode FftMode
    {
        get => _fftMode;
        set
        {
            if (value == ShortTimeFftMode.Onesided2X && Scaling is null)
                throw new ArgumentException("For scaling is None, fft_mode='onesided2X' is invalid! Call ScaleTo(Psd) or ScaleTo(Magnitude)!");
            _fftMode = value;
        }
    }

    /// <summary>FFT length, at least the window length.</summary>
    public int Mfft
    {
        get => _mfft;
        set
        {
            if (value < MNum) throw new ArgumentException($"Attribute mfft={value} needs to be at least the window length m_num={MNum}!");
            _mfft = value;
        }
    }

    /// <summary>Current window scaling, or null for none.</summary>
    public ShortTimeFftScaling? Scaling { get; private set; }

    /// <summary>Phase shift in samples (−mfft &lt; v &lt; mfft), or null.</summary>
    public int? PhaseShift
    {
        get => _phaseShift;
        set
        {
            if (value is { } v && !(-Mfft < v && v < Mfft))
                throw new ArgumentException($"-mfft < phase_shift < mfft does not hold for mfft={Mfft}, phase_shift={v}!");
            _phaseShift = value;
        }
    }

    /// <summary>Rescales the window (and dual window inversely) to magnitude or psd units.</summary>
    public void ScaleTo(ShortTimeFftScaling scaling)
    {
        if (scaling == ShortTimeFftScaling.Unitary) throw new ArgumentException("scaling must be Magnitude or Psd.", nameof(scaling));
        if (Scaling == scaling) return;
        var s = scaling == ShortTimeFftScaling.Psd ? FacPsd : FacMagnitude;
        for (var i = 0; i < _win.Length; i++) _win[i] *= s;
        if (_dualWin is not null) for (var i = 0; i < _dualWin.Length; i++) _dualWin[i] /= s;
        Scaling = scaling;
        _preCache = null;
    }

    /// <summary>Factor that would scale the window to magnitude units (1 when already scaled so).</summary>
    public double FacMagnitude => Scaling == ShortTimeFftScaling.Magnitude ? 1 : 1 / Math.Abs(_win.Sum());

    /// <summary>Factor that would scale the window to psd units (1 when already scaled so).</summary>
    public double FacPsd
    {
        get
        {
            if (Scaling == ShortTimeFftScaling.Psd) return 1;
            double s = 0;
            foreach (var w in _win) s += w * w;
            return 1 / Math.Sqrt(s / T);
        }
    }

    /// <summary>The dual window used by <see cref="Istft"/> (canonical dual unless one was given); a copy.</summary>
    public double[] DualWin => (double[])DualWinRef.Clone();

    private double[] DualWinRef => _dualWin ??= CanonicalDual(_win, Hop);

    /// <summary>Whether a dual window exists, i.e. the STFT can be inverted.</summary>
    public bool Invertible
    {
        get
        {
            try { return DualWinRef.Length > 0; }
            catch (ArgumentException) { return false; }
        }
    }

    /// <summary>Window length.</summary>
    public int MNum => _win.Length;

    /// <summary>Centre index of the window, <c>m_num / 2</c>.</summary>
    public int MNumMid => _win.Length / 2;

    /// <summary>Time increment between slices, <c>T·hop</c>.</summary>
    public double DeltaT => T * Hop;

    /// <summary>Frequency increment, <c>1/(mfft·T)</c>.</summary>
    public double DeltaF => 1 / (Mfft * T);

    /// <summary>Number of frequency points per slice.</summary>
    public int FPts => OnesidedFft ? Mfft / 2 + 1 : Mfft;

    /// <summary>True for the one-sided modes.</summary>
    public bool OnesidedFft => FftMode is ShortTimeFftMode.Onesided or ShortTimeFftMode.Onesided2X;

    /// <summary>The frequencies of the rows of the STFT, in the layout of <see cref="FftMode"/>.</summary>
    public double[] F => FftMode switch
    {
        ShortTimeFftMode.Twosided => Fft.FftFreq(Mfft, T),
        ShortTimeFftMode.Centered => Fft.FftShift(Fft.FftFreq(Mfft, T)),
        _ => Fft.RfftFreq(Mfft, T),
    };

    private (int K, int P) PrePadding
    {
        get
        {
            if (_preCache is { } c) return c;
            var w2 = W2();
            var n0 = -MNumMid;
            var p = 0;
            for (var k = n0; k > n0 - MNum - 1; k -= Hop, p++)
            {
                var next = k - Hop;
                if (next + MNum <= 0 || AllZero(w2, PyIndex(next, w2.Length), w2.Length))
                    return (_preCache = (k, -p)).Value;
            }
            throw new InvalidOperationException("ShortTimeFft pre-padding search failed.");
        }
    }

    /// <summary>Smallest sample index (possibly negative) touched by a slice that still overlaps the signal.</summary>
    public int KMin => PrePadding.K;

    /// <summary>Index of the first slice (≤ 0): slices before 0 overhang the signal start.</summary>
    public int PMin => PrePadding.P;

    private (int K, int P) PostPadding(int n)
    {
        var m2p = MNum - MNumMid;
        if (n < m2p) throw new ArgumentException($"Parameter n must be >= ceil(m_num/2) = {m2p}!", nameof(n));
        var w2 = W2();
        var q1 = FloorDiv(n, Hop);
        var k1 = q1 * Hop - MNumMid;
        var q = q1;
        for (var k = k1; k < n + MNum; k += Hop, q++)
        {
            var next = k + Hop;
            if (next >= n || AllZero(w2, 0, Math.Min(n - next, w2.Length))) return (k + MNum, q + 1);
        }
        throw new InvalidOperationException("ShortTimeFft post-padding search failed.");
    }

    /// <summary>First sample index past the end of the last slice for a signal of length n.</summary>
    public int KMax(int n) => PostPadding(n).K;

    /// <summary>One past the index of the last slice for a signal of length n.</summary>
    public int PMax(int n) => PostPadding(n).P;

    /// <summary>Number of slices for a signal of length n.</summary>
    public int PNum(int n) => PMax(n) - PMin;

    /// <summary>(k, p): first sample and first slice not influenced by the start of the signal.</summary>
    public (int K, int P) LowerBorderEnd
    {
        get
        {
            var w2 = W2();
            var m0 = Array.FindIndex(w2, v => v != 0);
            var k0 = -MNumMid + m0;
            var q = 0;
            for (var k = k0; k < Hop + 1; k += Hop, q++)
                if (k + Hop >= 0) return (k + MNum, q + 1);
            return (0, Math.Max(PMin, 0));
        }
    }

    /// <summary>(k, p): first sample and first slice influenced by the end of a signal of length n.</summary>
    public (int K, int P) UpperBorderBegin(int n)
    {
        var m2p = MNum - MNumMid;
        if (n < m2p) throw new ArgumentException($"Parameter n must be >= ceil(m_num/2) = {m2p}!", nameof(n));
        var w2 = W2();
        var q2 = FloorDiv(n, Hop) + 1;
        var q1 = Math.Max(FloorDiv(n - MNum, Hop) - 1, -1);
        for (var q = q2; q > q1; q--)
        {
            var k = q * Hop + (MNum - MNumMid);
            if (k <= n || AllZero(w2, PyIndex(n - k, w2.Length), w2.Length)) return ((q + 1) * Hop - MNumMid, q + 1);
        }
        throw new InvalidOperationException("ShortTimeFft upper-border search failed.");
    }

    /// <summary>The slice range [p0, p1) for a signal of length n, validated (defaults PMin and PMax(n)).</summary>
    public (int P0, int P1) PRange(int n, int? p0 = null, int? p1 = null)
    {
        var pMax = PMax(n);
        var a = p0 ?? PMin;
        var b = p1 ?? pMax;
        if (!(PMin <= a && a < b && b <= pMax))
            throw new ArgumentException($"Invalid Parameter p0={p0}, p1={p1}, i.e., p_min={PMin} <= p0 < p1 <= p_max={pMax} does not hold for signal length n={n}!");
        return (a, b);
    }

    /// <summary>Times of the slices (scipy's method <c>t(n, p0, p1, k_offset)</c>).</summary>
    public double[] Times(int n, int? p0 = null, int? p1 = null, int kOffset = 0)
    {
        if (n <= 0) throw new ArgumentException($"Parameter n={n} is not a positive integer!", nameof(n));
        var (a, b) = PRange(n, p0, p1);
        var t = new double[b - a];
        for (var i = 0; i < t.Length; i++) t[i] = (a + i) * DeltaT + kOffset * T;
        return t;
    }

    /// <summary>Nearest sample index at a slice position: k rounded down (or up) to a multiple of hop.</summary>
    public int NearestKP(int k, bool left = true)
    {
        var q = FloorDiv(k, Hop);
        if (k - q * Hop == 0) return k;
        return left ? q * Hop : (q + 1) * Hop;
    }

    /// <summary>Plot extent (t0, t1, f0, f1), or (f0, f1, t0, t1) when <paramref name="axesSeqTf"/> is false.
    /// Not defined for <see cref="ShortTimeFftMode.Twosided"/>.</summary>
    public (double, double, double, double) Extent(int n, bool axesSeqTf = true, bool centerBins = false)
    {
        int q0, q1;
        if (OnesidedFft) { q0 = 0; q1 = FPts; }
        else if (FftMode == ShortTimeFftMode.Centered) { q0 = -(Mfft / 2); q1 = Mfft % 2 == 0 ? Mfft / 2 : Mfft / 2 + 1; }
        else throw new InvalidOperationException("Extent requires fft_mode Centered, Onesided or Onesided2X.");
        int p0 = PMin, p1 = PMax(n);
        double t0, t1, f0, f1;
        if (centerBins)
        {
            t0 = DeltaT * (p0 - 0.5); t1 = DeltaT * (p1 - 0.5);
            f0 = DeltaF * (q0 - 0.5); f1 = DeltaF * (q1 - 0.5);
        }
        else
        {
            t0 = DeltaT * p0; t1 = DeltaT * p1;
            f0 = DeltaF * q0; f1 = DeltaF * q1;
        }
        return axesSeqTf ? (t0, t1, f0, f1) : (f0, f1, t0, t1);
    }

    // ------------------------------------------------------------------ transforms

    /// <summary>The STFT of a real signal: an <see cref="FPts"/> × (p1 − p0) complex array.</summary>
    /// <param name="x">Signal; at least ceil(m_num/2) samples.</param>
    /// <param name="p0">First slice; default <see cref="PMin"/>.</param>
    /// <param name="p1">One past the last slice; default <see cref="PMax"/>(n).</param>
    /// <param name="kOffset">Shift of the signal against the slice grid, in samples.</param>
    /// <param name="padding">Extension beyond the signal ends.</param>
    public ComplexNdArray Stft(double[] x, int? p0 = null, int? p1 = null, int kOffset = 0, ShortTimeFftPadding padding = ShortTimeFftPadding.Zeros)
        => StftCore(x, null, null, null, p0, p1, kOffset, padding);

    /// <summary>The STFT of a complex signal (two-sided or centered modes only).</summary>
    public ComplexNdArray Stft(Complex[] x, int? p0 = null, int? p1 = null, int kOffset = 0, ShortTimeFftPadding padding = ShortTimeFftPadding.Zeros)
        => StftCore(null, x, null, null, p0, p1, kOffset, padding);

    /// <summary>The STFT with each slice detrended before windowing (scipy's <c>stft_detrend</c> with a str).</summary>
    public ComplexNdArray StftDetrend(double[] x, DetrendType? detr, int? p0 = null, int? p1 = null, int kOffset = 0, ShortTimeFftPadding padding = ShortTimeFftPadding.Zeros)
        => StftCore(x, null, detr, null, p0, p1, kOffset, padding);

    /// <summary>The STFT with each slice passed through a detrend callback before windowing; the
    /// callback receives a copy of one slice (length m_num) and must return as many values.</summary>
    public ComplexNdArray StftDetrend(double[] x, Func<double[], double[]> detr, int? p0 = null, int? p1 = null, int kOffset = 0, ShortTimeFftPadding padding = ShortTimeFftPadding.Zeros)
        => StftCore(x, null, null, detr ?? throw new ArgumentNullException(nameof(detr)), p0, p1, kOffset, padding);

    /// <summary>The STFT of a complex signal with each slice detrended (real and imaginary parts separately).</summary>
    public ComplexNdArray StftDetrend(Complex[] x, DetrendType? detr, int? p0 = null, int? p1 = null, int kOffset = 0, ShortTimeFftPadding padding = ShortTimeFftPadding.Zeros)
        => StftCore(null, x, detr, null, p0, p1, kOffset, padding);

    /// <summary>|S|², the spectrogram of a real signal.</summary>
    public NdArray Spectrogram(double[] x, DetrendType? detr = null, int? p0 = null, int? p1 = null, int kOffset = 0, ShortTimeFftPadding padding = ShortTimeFftPadding.Zeros)
        => Power(StftCore(x, null, detr, null, p0, p1, kOffset, padding));

    /// <summary>|S|² with a custom per-slice detrend callback.</summary>
    public NdArray Spectrogram(double[] x, Func<double[], double[]> detr, int? p0 = null, int? p1 = null, int kOffset = 0, ShortTimeFftPadding padding = ShortTimeFftPadding.Zeros)
        => Power(StftCore(x, null, null, detr ?? throw new ArgumentNullException(nameof(detr)), p0, p1, kOffset, padding));

    /// <summary>The cross-spectrogram <c>Sx · conj(Sy)</c> of two real signals of equal length.</summary>
    public ComplexNdArray Spectrogram(double[] x, double[] y, DetrendType? detr = null, int? p0 = null, int? p1 = null, int kOffset = 0,
        ShortTimeFftPadding padding = ShortTimeFftPadding.Zeros)
    {
        var sx = StftCore(x, null, detr, null, p0, p1, kOffset, padding);
        if (ReferenceEquals(x, y)) return ComplexNdArray.FromParts(Power(sx));
        var sy = StftCore(y, null, detr, null, p0, p1, kOffset, padding);
        var a = sx.Data; var b = sy.Data;
        for (var i = 0; i < a.Length; i++) a[i] *= Complex.Conjugate(b[i]);
        return sx;
    }

    /// <summary>|S|² of a complex signal (two-sided or centered modes).</summary>
    public NdArray Spectrogram(Complex[] x, DetrendType? detr = null, int? p0 = null, int? p1 = null, int kOffset = 0, ShortTimeFftPadding padding = ShortTimeFftPadding.Zeros)
        => Power(StftCore(null, x, detr, null, p0, p1, kOffset, padding));

    private static NdArray Power(ComplexNdArray s)
    {
        var d = s.Data;
        var r = new double[d.Length];
        for (var i = 0; i < r.Length; i++) r[i] = d[i].Real * d[i].Real + d[i].Imaginary * d[i].Imaginary;
        return new NdArray(r, s.Shape[0], s.Shape[1]);
    }

    private ComplexNdArray StftCore(double[]? xr, Complex[]? xc, DetrendType? detr, Func<double[], double[]>? custom, int? p0, int? p1,
        int kOffset, ShortTimeFftPadding padding)
    {
        if (xr is null && xc is null) throw new ArgumentNullException("x");
        if (xc is not null && OnesidedFft)
            throw new ArgumentException($"Complex-valued `x` not allowed for fft_mode={FftMode}! Set FftMode to Twosided or Centered.");
        var n = xr?.Length ?? xc!.Length;
        var m2p = MNum - MNumMid;
        if (n < m2p) throw new ArgumentException($"len(x)={n} must be >= ceil(m_num/2) = {m2p}!");
        var (a, b) = PRange(n, p0, p1);
        var pn = b - a;
        var fpts = FPts;
        var result = new Complex[(long)fpts * pn];

        // scipy pads x[i0:i1] (not all of x) and walks it in hop steps.
        var k0 = a * Hop - MNumMid + kOffset;
        var i0 = Math.Max(k0, 0);
        var i1 = Math.Min(k0 + pn * Hop + MNum, n);
        if (i1 < i0) i1 = i0;
        var mode = padding switch
        {
            ShortTimeFftPadding.Edge => UpfirdnPad.Edge,
            ShortTimeFftPadding.Even => UpfirdnPad.Reflect,
            ShortTimeFftPadding.Odd => UpfirdnPad.Antireflect,
            _ => UpfirdnPad.Constant,
        };
        if (i1 == i0 && mode != UpfirdnPad.Constant) mode = UpfirdnPad.Constant;
        var re = xr is not null ? xr.AsSpan(i0, i1 - i0).ToArray() : xc!.AsSpan(i0, i1 - i0).ToArray().Select(v => v.Real).ToArray();
        var im = xc is not null ? xc.AsSpan(i0, i1 - i0).ToArray().Select(v => v.Imaginary).ToArray() : null;
        var leftPad = -Math.Min(k0, 0);

        void Slice(int p, double[] bufR, double[]? bufI)
        {
            var start = p * Hop - leftPad; // index into the sliced x[i0:i1]
            for (var j = 0; j < MNum; j++)
            {
                var idx = (long)start + j;
                bufR[j] = SignalPadding.Extend(re, idx, mode, 0);
                if (bufI is not null) bufI[j] = SignalPadding.Extend(im!, idx, mode, 0);
            }
            if (detr is { } d)
            {
                Signal.DetrendInPlace(bufR, d);
                if (bufI is not null) Signal.DetrendInPlace(bufI, d);
            }
            else if (custom is not null)
            {
                var r = custom((double[])bufR.Clone());
                if (r is null || r.Length != MNum) throw new ArgumentException($"The detrend callback must return {MNum} values.");
                r.CopyTo(bufR, 0);
            }
            var spec = Transform(bufR, bufI);
            for (var k = 0; k < fpts; k++) result[(long)k * pn + p] = spec[k];
        }

        var parallel = custom is null && (long)pn * Mfft >= 1 << 16 && pn >= 4;
        if (parallel)
            Parallel.For(0, pn, () => (new double[MNum], im is not null ? new double[MNum] : null),
                (p, _, bufs) => { Slice(p, bufs.Item1, bufs.Item2); return bufs; }, _ => { });
        else
        {
            var bufR = new double[MNum];
            var bufI = im is not null ? new double[MNum] : null;
            for (var p = 0; p < pn; p++) Slice(p, bufR, bufI);
        }
        return new ComplexNdArray(result, fpts, pn);
    }

    /// <summary>scipy's <c>_fft_func</c> on one windowed slice.</summary>
    private Complex[] Transform(double[] sliceRe, double[]? sliceIm)
    {
        var m = Mfft;
        var ps = PhaseShift is { } v ? Mod(v + MNumMid, MNum) : -1;
        // Window, zero-pad to mfft, then roll left by p_s (over the padded length, as numpy does).
        if (sliceIm is null && OnesidedFft)
        {
            var buf = new double[m];
            for (var j = 0; j < MNum; j++)
            {
                var dst = ps < 0 ? j : Mod(j - ps, m);
                buf[dst] = sliceRe[j] * _win[j];
            }
            var X = Fft.Rfft(buf);
            if (FftMode == ShortTimeFftMode.Onesided2X)
            {
                var fac = Scaling == ShortTimeFftScaling.Psd ? Math.Sqrt(2) : 2;
                var end = m % 2 == 0 ? X.Length - 1 : X.Length;
                for (var k = 1; k < end; k++) X[k] *= fac;
            }
            return X;
        }
        var cb = new Complex[m];
        for (var j = 0; j < MNum; j++)
        {
            var dst = ps < 0 ? j : Mod(j - ps, m);
            cb[dst] = new Complex(sliceRe[j] * _win[j], sliceIm is null ? 0 : sliceIm[j] * _win[j]);
        }
        var Y = Fft.Fft1D(cb);
        return FftMode == ShortTimeFftMode.Centered ? Fft.FftShift(Y) : Y;
    }

    /// <summary>scipy's <c>_ifft_func</c>: one slice back to m_num samples (before the dual window).</summary>
    private Complex[] InverseTransform(Complex[] X, out double[]? real)
    {
        var m = Mfft;
        double[]? xr = null;
        Complex[]? xc = null;
        switch (FftMode)
        {
            case ShortTimeFftMode.Twosided: xc = Fft.Ifft(X, m); break;
            case ShortTimeFftMode.Centered: xc = Fft.Ifft(Fft.IfftShift(X), m); break;
            case ShortTimeFftMode.Onesided: xr = Fft.Irfft(X, m); break;
            default:
            {
                var c = (Complex[])X.Clone();
                var fac = Scaling == ShortTimeFftScaling.Psd ? Math.Sqrt(2) : 2;
                var end = m % 2 == 0 ? c.Length - 1 : c.Length;
                for (var k = 1; k < end; k++) c[k] /= fac;
                xr = Fft.Irfft(c, m);
                break;
            }
        }
        var ps = PhaseShift is { } v ? Mod(v + MNumMid, MNum) : 0;
        if (xr is not null)
        {
            real = new double[MNum];
            for (var j = 0; j < MNum; j++) real[j] = xr[Mod(j - ps, m)];
            return [];
        }
        real = null;
        var outC = new Complex[MNum];
        for (var j = 0; j < MNum; j++) outC[j] = xc![Mod(j - ps, m)];
        return outC;
    }

    /// <summary>
    /// Inverse STFT of a one-sided transform (real signal) by overlap-add with the dual window
    /// (scipy's <c>istft</c>), returning samples k0 ≤ k &lt; k1 (default 0 to k_max).
    /// </summary>
    public double[] Istft(ComplexNdArray s, int k0 = 0, int? k1 = null)
    {
        if (!OnesidedFft) throw new InvalidOperationException("Istft returns a real signal; use IstftComplex for the two-sided modes.");
        var (r, _) = IstftCore(s, k0, k1);
        return r!;
    }

    /// <summary>Inverse STFT for the two-sided/centered modes (complex signal); for one-sided modes the imaginary parts are zero.</summary>
    public Complex[] IstftComplex(ComplexNdArray s, int k0 = 0, int? k1 = null)
    {
        var (r, c) = IstftCore(s, k0, k1);
        return c ?? r!.Select(v => new Complex(v, 0)).ToArray();
    }

    private (double[]? Real, Complex[]? Cplx) IstftCore(ComplexNdArray s, int k0, int? k1In)
    {
        ArgumentNullException.ThrowIfNull(s);
        if (s.Rank != 2) throw new ArgumentException("S must be 2-D (frequencies × slices).", nameof(s));
        if (s.Shape[0] != FPts) throw new ArgumentException($"S.shape[f_axis]={s.Shape[0]} must be equal to f_pts={FPts}!", nameof(s));
        var nMin = MNum - MNumMid;
        var qNum = PNum(nMin);
        var ns = s.Shape[1];
        if (ns < qNum) throw new ArgumentException($"S.shape[t_axis]={ns} needs to have at least {qNum} slices!", nameof(s));
        var qMax = ns + PMin;
        var kMax = (qMax - 1) * Hop + MNum - MNumMid;
        var k1 = k1In ?? kMax;
        if (!(KMin <= k0 && k0 < k1 && k1 <= kMax)) throw new ArgumentException($"(k_min={KMin}) <= (k0={k0}) < (k1={k1}) <= (k_max={kMax}) is false!");
        if (k1 - k0 < nMin) throw new ArgumentException($"(k1={k1}) - (k0={k0}) = {k1 - k0} has to be at least the half the window length {nMin}!");
        var q0 = k0 >= 0 ? FloorDiv(k0, Hop) + PMin : FloorDiv(k0, Hop);
        var q1 = Math.Min(PMax(k1), qMax);
        var kq0 = NearestKP(k0);
        var kq1 = NearestKP(k1, left: false);
        var nPts = kq1 - kq0 + MNum - MNumMid;
        var onesided = OnesidedFft;
        var xr = onesided ? new double[nPts] : null;
        var xc = onesided ? null : new Complex[nPts];
        var dual = DualWinRef;
        var data = s.Data;
        var col = new Complex[FPts];
        for (var q = q0; q < q1; q++)
        {
            var c = q - PMin;
            for (var k = 0; k < col.Length; k++) col[k] = data[k * ns + c];
            var seg = InverseTransform(col, out var segR);
            var ii0 = q * Hop - MNumMid;
            var ii1 = Math.Min(ii0 + MNum, nPts + k0);
            int j0 = 0, j1 = ii1 - ii0;
            if (ii0 < k0) { j0 += k0 - ii0; ii0 = k0; }
            for (var j = j0; j < j1; j++)
            {
                var dst = ii0 - k0 + (j - j0);
                if (dst < 0 || dst >= nPts) continue;
                if (xr is not null) xr[dst] += segR![j] * dual[j];
                else xc![dst] += seg[j] * dual[j];
            }
        }
        var len = k1 - k0;
        if (xr is not null) return (xr.AsSpan(0, Math.Min(len, nPts)).ToArray(), null);
        return (null, xc!.AsSpan(0, Math.Min(len, nPts)).ToArray());
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>scipy's <c>_calc_dual_canonical_window</c>: <c>w / Σ_k |w(· − k·hop)|²</c>.</summary>
    internal static double[] CanonicalDual(double[] win, int hop)
    {
        if (hop > win.Length) throw new ArgumentException($"hop={hop} is larger than window length of {win.Length} => STFT not invertible!");
        var dd = OverlapSquares(win, hop);
        var resolution = 1e-15 * dd.Max();
        foreach (var v in dd) if (!(v >= resolution)) throw new ArgumentException("Short-time Fourier Transform not invertible!");
        var r = new double[win.Length];
        for (var i = 0; i < r.Length; i++) r[i] = win[i] / dd[i];
        return r;
    }

    private static double[] OverlapSquares(double[] win, int hop)
    {
        var n = win.Length;
        var w2 = new double[n];
        for (var i = 0; i < n; i++) w2[i] = win[i] * win[i];
        return OverlapAdd(w2, hop);
    }

    /// <summary><c>DD = v + Σ_{k=hop,2hop,..} (v shifted by ±k)</c>, scipy's in-place accumulation.</summary>
    internal static double[] OverlapAdd(double[] v, int hop)
    {
        var n = v.Length;
        var dd = (double[])v.Clone();
        for (var k = hop; k < n; k += hop)
        {
            for (var i = k; i < n; i++) dd[i] += v[i - k];
            for (var i = 0; i < n - k; i++) dd[i] += v[i + k];
        }
        return dd;
    }

    private double[] W2()
    {
        var w2 = new double[_win.Length];
        for (var i = 0; i < w2.Length; i++) w2[i] = _win[i] * _win[i];
        return w2;
    }

    /// <summary>Python slice-start semantics: negative counts from the end, clamped to [0, len].</summary>
    private static int PyIndex(int i, int len) => i < 0 ? Math.Max(0, len + i) : Math.Min(i, len);

    private static bool AllZero(double[] a, int from, int to)
    {
        for (var i = from; i < to; i++) if (a[i] != 0) return false;
        return true;
    }

    private static int FloorDiv(int a, int b) => (int)Math.Floor((double)a / b);

    private static int Mod(int a, int m) => ((a % m) + m) % m;
}

public static partial class Signal
{
    /// <summary>
    /// The dual window closest (least squares) to <paramref name="desiredDual"/> for
    /// <paramref name="win"/> and <paramref name="hop"/> — scipy's <c>closest_STFT_dual_window</c>.
    /// With <paramref name="scaled"/> the result is <c>α·</c>(closest dual) with the best scale α
    /// (returned), otherwise α = 1.
    /// </summary>
    /// <param name="win">Analysis window (real).</param>
    /// <param name="hop">Hop in samples, 1 ≤ hop ≤ len(win).</param>
    /// <param name="desiredDual">Desired dual window; default all ones.</param>
    /// <param name="scaled">Also fit a scale factor.</param>
    public static (double[] Dual, double Alpha) ClosestStftDualWindow(double[] win, int hop, double[]? desiredDual = null, bool scaled = true)
    {
        ArgumentNullException.ThrowIfNull(win);
        desiredDual ??= Enumerable.Repeat(1.0, win.Length).ToArray();
        if (desiredDual.Length != win.Length) throw new ArgumentException("Parameters `win` and `desired_dual` are not 1d arrays of equal length!");
        foreach (var v in win) if (!double.IsFinite(v)) throw new ArgumentException("Parameter win must have finite entries!", nameof(win));
        foreach (var v in desiredDual) if (!double.IsFinite(v)) throw new ArgumentException("Parameter desired_dual must have finite entries!", nameof(desiredDual));
        if (hop < 1 || hop > win.Length) throw new ArgumentException($"Parameter hop={hop} is not an integer between 1 and len(win)={win.Length}!", nameof(hop));
        var wd = ShortTimeFft.CanonicalDual(win, hop);
        var wdd = new double[win.Length];
        for (var i = 0; i < wdd.Length; i++) wdd[i] = win[i] * desiredDual[i];
        var qd = ShortTimeFft.OverlapAdd(wdd, hop);
        for (var i = 0; i < qd.Length; i++) qd[i] *= wd[i];
        var r = new double[win.Length];
        if (!scaled)
        {
            for (var i = 0; i < r.Length; i++) r[i] = wd[i] + desiredDual[i] - qd[i];
            return (r, 1.0);
        }
        double num = 0, den = 0;
        for (var i = 0; i < r.Length; i++) { num += qd[i] * wd[i]; den += qd[i] * qd[i]; }
        if (!(Math.Abs(num) > 0 && den > 1e-15))
            throw new ArgumentException("Unable to calculate scaled closest dual window due to numerically unstable scaling factor! Try setting parameter `scaled` to False.");
        var alpha = num / den;
        for (var i = 0; i < r.Length; i++) r[i] = wd[i] + alpha * (desiredDual[i] - qd[i]);
        return (r, alpha);
    }
}
