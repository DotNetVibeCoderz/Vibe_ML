using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Anti-aliasing filter used by <see cref="Signal.Decimate(double[], int, int?, DecimateFilter, bool)"/>.</summary>
public enum DecimateFilter
{
    /// <summary>Order-8 (default) Chebyshev type I, 0.05 dB ripple, cutoff 0.8/q.</summary>
    Iir,
    /// <summary>Hamming-window FIR of 20·q + 1 taps (default), cutoff 1/q.</summary>
    Fir,
}

/// <summary>Whether <see cref="Signal.Resample(Complex[], int, double[], ResampleDomain)"/> receives samples or a spectrum.</summary>
public enum ResampleDomain
{
    /// <summary>The input is a time series.</summary>
    Time,
    /// <summary>The input is already an FFT (in numpy order).</summary>
    Freq,
}

public static partial class Signal
{
    // ================================================================== resample (Fourier)

    /// <summary>
    /// Fourier resampling to <paramref name="num"/> samples (<c>scipy.signal.resample</c>): the
    /// spectrum is truncated or zero-padded, which assumes the signal is periodic — expect ringing
    /// at the ends of a signal that is not.
    /// </summary>
    /// <remarks>
    /// When the shorter length <c>m = min(num, n)</c> is even, its Nyquist bin is split (upsampling,
    /// ×½) or folded (downsampling, ×2) exactly as scipy 1.17 does, so the result is real and
    /// reproduces scipy's to rounding.
    /// </remarks>
    /// <param name="x">Real signal.</param>
    /// <param name="num">Number of output samples.</param>
    /// <param name="window">Optional window applied in the frequency domain, generated with
    /// <c>get_window(window, n)</c> and <c>fftshift</c>-ed so its peak sits on DC.</param>
    public static double[] Resample(double[] x, int num, WindowSpec? window = null)
        => ResampleCore(x, num, window is null ? null : FftShifted(GetWindow(window, x.Length)));

    /// <summary>
    /// <see cref="Resample(double[], int, WindowSpec)"/> with a frequency-domain weighting given
    /// directly, one value per FFT bin in numpy order (length n).
    /// </summary>
    public static double[] Resample(double[] x, int num, double[] window)
    {
        ArgumentNullException.ThrowIfNull(window);
        if (window.Length != x.Length) throw new ArgumentException($"window length {window.Length} != {x.Length}, the number of frequency bins.", nameof(window));
        return ResampleCore(x, num, (double[])window.Clone());
    }

    /// <summary>
    /// <see cref="Resample(double[], int, WindowSpec)"/> that also returns the resampled sample
    /// positions <c>t[0] + (t[1] − t[0])·(n/num)·k</c> — only the first two entries of
    /// <paramref name="t"/> are read, so the input spacing is assumed uniform.
    /// </summary>
    public static (double[] Y, double[] T) Resample(double[] x, int num, double[] t, WindowSpec? window)
    {
        ArgumentNullException.ThrowIfNull(t);
        if (t.Length < 2) throw new ArgumentException("t needs at least two samples.", nameof(t));
        var y = Resample(x, num, window);
        var s = (double)x.Length / num;
        var nt = new double[num];
        for (var k = 0; k < num; k++) nt[k] = t[0] + (t[1] - t[0]) * s * k;
        return (y, nt);
    }

    /// <summary><see cref="Resample(double[], int, WindowSpec)"/> along one axis of an n-D array.</summary>
    public static NdArray Resample(NdArray x, int num, int axis = 0, WindowSpec? window = null)
    {
        var n = SignalAxisLines.Length(x, axis);
        var w = window is null ? null : FftShifted(GetWindow(window, n));
        return SignalAxisLines.Map(x, axis, num, (s, d) => ResampleCore(s.ToArray(), num, w is null ? null : (double[])w.Clone()).CopyTo(d));
    }

    /// <summary>
    /// Fourier resampling of complex data, or of a spectrum when <paramref name="domain"/> is
    /// <see cref="ResampleDomain.Freq"/> (the full-FFT branch of <c>scipy.signal.resample</c>).
    /// </summary>
    /// <param name="x">Samples, or an FFT in numpy order.</param>
    /// <param name="num">Output length.</param>
    /// <param name="window">Optional per-bin weights in numpy order (length n).</param>
    /// <param name="domain">Meaning of <paramref name="x"/>.</param>
    public static Complex[] Resample(Complex[] x, int num, double[]? window = null, ResampleDomain domain = ResampleDomain.Time)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (num < 1) throw new ArgumentException("num must be positive.", nameof(num));
        var nx = x.Length;
        if (nx == 0) throw new ArgumentException("x must not be empty.", nameof(x));
        var sFac = (double)nx / num;
        var m = Math.Min(num, nx);
        var m2 = m / 2 + 1;
        var X = domain == ResampleDomain.Time ? Fft.Fft1D(x) : (Complex[])x.Clone();
        if (window is not null)
        {
            if (window.Length != nx) throw new ArgumentException("window length must equal the number of bins.", nameof(window));
            for (var i = 0; i < nx; i++) X[i] *= window[i];
        }
        var Y = new Complex[num];
        for (var i = 0; i < Math.Min(m2, nx); i++) Y[i] = X[i];
        if (m2 < m)
            for (var i = 1; i <= m - m2; i++) Y[num - i] = X[nx - i];
        if (m % 2 == 0)
        {
            if (num < nx) Y[num - m / 2] += X[nx - m / 2];
            else if (nx < num)
            {
                Y[m / 2] /= 2;
                Y[num - m / 2] = Y[m / 2];
            }
        }
        for (var i = 0; i < num; i++) Y[i] /= sFac;
        return Fft.Ifft(Y);
    }

    private static double[] FftShifted(double[] w) => Fft.FftShift(w);

    private static double[] ResampleCore(double[] x, int num, double[]? w)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (num < 1) throw new ArgumentException("num must be positive.", nameof(num));
        var nx = x.Length;
        if (nx == 0) throw new ArgumentException("x must not be empty.", nameof(x));
        var sFac = (double)nx / num;
        var m = Math.Min(num, nx);
        var m2 = m / 2 + 1;
        var X = Fft.Rfft(x);
        if (w is not null)
        {
            var nX = X.Length;
            // Fold the negative-frequency half of the window onto the positive bins.
            for (var i = 1; i < nX; i++) w[i] = (w[i] + w[nx - i]) / 2;
            for (var i = 0; i < nX; i++) X[i] *= w[i];
        }
        var bins = num / 2 + 1;
        var Y = new Complex[bins];
        var copy = Math.Min(m2, Math.Min(X.Length, bins));
        for (var i = 0; i < copy; i++) Y[i] = X[i] / sFac;
        if (m % 2 == 0 && num != nx && m / 2 < bins) Y[m / 2] *= num < nx ? 2.0 : 0.5;
        return Fft.Irfft(Y, num);
    }

    // ================================================================== resample_poly

    /// <summary>
    /// Polyphase resampling by <paramref name="up"/>/<paramref name="down"/>
    /// (<c>scipy.signal.resample_poly</c>): a Kaiser-windowed (β = 5 by default) FIR low-pass of
    /// <c>20·max(up, down) + 1</c> taps applied through <see cref="Upfirdn(double[], double[], int, int, UpfirdnPad, double)"/>,
    /// with the group delay removed so the output is aligned with the input.
    /// </summary>
    /// <param name="x">Signal.</param>
    /// <param name="up">Upsampling factor.</param>
    /// <param name="down">Downsampling factor.</param>
    /// <param name="window">Window for the FIR design; default <c>("kaiser", 5.0)</c>.</param>
    /// <param name="padtype">Edge extension; the statistic modes subtract that statistic first.</param>
    /// <param name="cval">Pad value for <see cref="UpfirdnPad.Constant"/>.</param>
    public static double[] ResamplePoly(double[] x, int up, int down, WindowSpec? window = null, UpfirdnPad padtype = UpfirdnPad.Constant, double? cval = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        var (u, d) = ReduceRates(up, down, padtype, cval);
        if (u == 1 && d == 1) return (double[])x.Clone();
        var maxRate = Math.Max(u, d);
        var halfLen = 10 * maxRate;
        var h = Firwin(2 * halfLen + 1, 1.0 / maxRate, window: window ?? new WindowSpec("kaiser", 5.0));
        return ResamplePolyCore(x, u, d, h, halfLen, padtype, cval ?? 0);
    }

    /// <summary>
    /// <see cref="ResamplePoly(double[], int, int, WindowSpec, UpfirdnPad, double?)"/> with the FIR
    /// filter given directly (scipy's array <c>window</c>); it is used as-is apart from the ×up gain.
    /// </summary>
    public static double[] ResamplePoly(double[] x, int up, int down, double[] window, UpfirdnPad padtype = UpfirdnPad.Constant, double? cval = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(window);
        var (u, d) = ReduceRates(up, down, padtype, cval);
        if (u == 1 && d == 1) return (double[])x.Clone();
        return ResamplePolyCore(x, u, d, (double[])window.Clone(), (window.Length - 1) / 2, padtype, cval ?? 0);
    }

    /// <summary><see cref="ResamplePoly(double[], int, int, WindowSpec, UpfirdnPad, double?)"/> along one axis.</summary>
    public static NdArray ResamplePoly(NdArray x, int up, int down, int axis = 0, WindowSpec? window = null, UpfirdnPad padtype = UpfirdnPad.Constant, double? cval = null)
    {
        var n = SignalAxisLines.Length(x, axis);
        var (u, d) = ReduceRates(up, down, padtype, cval);
        var nOut = (int)(((long)n * u + d - 1) / d);
        if (u == 1 && d == 1) return x.Copy();
        var maxRate = Math.Max(u, d);
        var halfLen = 10 * maxRate;
        var h = Firwin(2 * halfLen + 1, 1.0 / maxRate, window: window ?? new WindowSpec("kaiser", 5.0));
        return SignalAxisLines.Map(x, axis, nOut, (s, dst) => ResamplePolyCore(s.ToArray(), u, d, (double[])h.Clone(), halfLen, padtype, cval ?? 0).CopyTo(dst));
    }

    private static (int Up, int Down) ReduceRates(int up, int down, UpfirdnPad padtype, double? cval)
    {
        if (up < 1 || down < 1) throw new ArgumentException("up and down must be >= 1.");
        if (cval is not null && padtype != UpfirdnPad.Constant) throw new ArgumentException($"cval has no effect when padtype is {padtype}.", nameof(cval));
        var g = (int)BigInteger.GreatestCommonDivisor(up, down);
        return (up / g, down / g);
    }

    private static double[] ResamplePolyCore(double[] x, int up, int down, double[] h, int halfLen, UpfirdnPad padtype, double cval)
    {
        var nIn = x.Length;
        var nOut = (int)(((long)nIn * up + down - 1) / down);
        for (var i = 0; i < h.Length; i++) h[i] *= up;
        var nPrePad = down - halfLen % down;
        var nPostPad = 0;
        var nPreRemove = (halfLen + nPrePad) / down;
        while (new UpfirdnPlan(new double[h.Length + nPrePad + nPostPad], up, down).OutputLength(nIn) < nOut + nPreRemove) nPostPad++;
        var hp = new double[nPrePad + h.Length + nPostPad];
        h.CopyTo(hp, nPrePad);
        double background = 0;
        var mode = padtype;
        if (padtype is UpfirdnPad.Mean or UpfirdnPad.Median or UpfirdnPad.Minimum or UpfirdnPad.Maximum)
        {
            background = padtype switch
            {
                UpfirdnPad.Mean => Mean(x),
                UpfirdnPad.Median => MedianOf(x),
                UpfirdnPad.Minimum => x.Length == 0 ? 0 : x.Min(),
                _ => x.Length == 0 ? 0 : x.Max(),
            };
            var shifted = new double[nIn];
            for (var i = 0; i < nIn; i++) shifted[i] = x[i] - background;
            x = shifted;
            mode = UpfirdnPad.Constant;
            cval = 0;
        }
        var plan = new UpfirdnPlan(hp, up, down);
        var full = new double[plan.OutputLength(nIn)];
        plan.Apply(x, full, mode, cval);
        var y = new double[nOut];
        for (var i = 0; i < nOut; i++) y[i] = full[nPreRemove + i] + background;
        return y;
    }

    private static double Mean(ReadOnlySpan<double> x)
    {
        double s = 0;
        foreach (var v in x) s += v;
        return x.Length == 0 ? double.NaN : s / x.Length;
    }

    private static double MedianOf(ReadOnlySpan<double> x)
    {
        if (x.Length == 0) return double.NaN;
        var c = x.ToArray();
        Array.Sort(c);
        var h = c.Length / 2;
        return c.Length % 2 == 1 ? c[h] : 0.5 * (c[h - 1] + c[h]);
    }

    // ================================================================== decimate

    /// <summary>
    /// Anti-alias filter, then keep every <paramref name="q"/>-th sample (<c>scipy.signal.decimate</c>).
    /// </summary>
    /// <remarks>
    /// The IIR default (Chebyshev I, order 8, as second-order sections) is applied forwards and
    /// backwards when <paramref name="zeroPhase"/>; the FIR path then goes through
    /// <see cref="ResamplePoly(double[], int, int, double[], UpfirdnPad, double?)"/> exactly like
    /// scipy. For large q prefer several smaller stages: an order-8 IIR with cutoff 0.8/q becomes
    /// numerically fragile once q exceeds ~13.
    /// </remarks>
    /// <param name="x">Signal.</param>
    /// <param name="q">Downsampling factor.</param>
    /// <param name="n">Filter order; default 8 (IIR) or 20·q (FIR).</param>
    /// <param name="ftype">IIR or FIR.</param>
    /// <param name="zeroPhase">Filter forwards and backwards (IIR) or remove the FIR delay.</param>
    public static double[] Decimate(double[] x, int q, int? n = null, DecimateFilter ftype = DecimateFilter.Iir, bool zeroPhase = true)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (q < 1) throw new ArgumentException("q must be a positive integer.", nameof(q));
        if (ftype == DecimateFilter.Fir)
        {
            var order = n ?? 20 * q;
            var b = Firwin(order + 1, 1.0 / q, window: "hamming");
            if (zeroPhase) return ResamplePoly(x, 1, q, b);
            var nOut = x.Length / q + (x.Length % q != 0 ? 1 : 0);
            var y = Upfirdn(b, x, 1, q);
            return y.AsSpan(0, Math.Min(nOut, y.Length)).ToArray();
        }
        var sos = Cheby1Sos(n ?? 8, 0.05, 0.8 / q);
        var filtered = zeroPhase ? Sosfiltfilt(sos, x) : Sosfilt(sos, x);
        var res = new double[(filtered.Length + q - 1) / q];
        for (var i = 0; i < res.Length; i++) res[i] = filtered[i * q];
        return res;
    }

    /// <summary><see cref="Decimate(double[], int, int?, DecimateFilter, bool)"/> along one axis.</summary>
    public static NdArray Decimate(NdArray x, int q, int? n = null, DecimateFilter ftype = DecimateFilter.Iir, bool zeroPhase = true, int axis = -1)
    {
        var len = SignalAxisLines.Length(x, axis);
        if (q < 1) throw new ArgumentException("q must be a positive integer.", nameof(q));
        var outLen = (len + q - 1) / q;
        return SignalAxisLines.Map(x, axis, outLen, (s, d) => Decimate(s.ToArray(), q, n, ftype, zeroPhase).CopyTo(d));
    }
}
