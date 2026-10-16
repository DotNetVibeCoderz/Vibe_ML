using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A precomputed chirp z-transform (<c>scipy.signal.CZT</c>): evaluates the z-transform of an
/// n-sample signal at the m points <c>z_k = a · w^(−k)</c> on a spiral, by Bluestein's
/// algorithm — one FFT convolution of length <c>next_fast_len(n + m − 1)</c>, so O((n+m) log(n+m))
/// for any n, m, a, w. Instances are immutable and safe to call from several threads.
/// </summary>
public class CZT
{
    private readonly Complex[] _awk2;
    private readonly Complex[] _fwk2;
    private readonly Complex[] _wk2;
    private readonly FftPlan _plan;

    /// <summary>Input length.</summary>
    public int N { get; }

    /// <summary>Number of output points.</summary>
    public int M { get; }

    /// <summary>Ratio between points (<c>exp(−2πi/m)</c> by default: equally spaced on the unit circle).</summary>
    public Complex W { get; }

    /// <summary>Starting point.</summary>
    public Complex A { get; }

    /// <summary>Builds the transform.</summary>
    /// <param name="n">Input length.</param>
    /// <param name="m">Output length; default n.</param>
    /// <param name="w">Point ratio; default <c>exp(−2πi/m)</c>, for which the result equals the FFT.</param>
    /// <param name="a">Starting point; default 1.</param>
    public CZT(int n, int? m = null, Complex? w = null, Complex? a = null)
    {
        if (n < 1) throw new ArgumentException($"Invalid number of CZT data points ({n}) specified. n must be positive.", nameof(n));
        var mm = m ?? n;
        if (mm < 1) throw new ArgumentException($"Invalid number of CZT output points ({mm}) specified. m must be positive.", nameof(m));
        N = n;
        M = mm;
        var aa = a ?? Complex.One;
        var len = Math.Max(mm, n);
        var wk2 = new Complex[len];
        Complex ww;
        if (w is null)
        {
            ww = Complex.Exp(new Complex(0, -2 * Math.PI / mm));
            // k² mod 2m keeps the phase argument small, so it stays exact for large k.
            for (long k = 0; k < len; k++)
                wk2[k] = Complex.Exp(new Complex(0, -Math.PI * (k * k % (2L * mm)) / mm));
        }
        else
        {
            ww = w.Value;
            for (long k = 0; k < len; k++) wk2[k] = Complex.Pow(ww, k * (double)k / 2.0);
        }
        W = ww;
        A = aa;
        _awk2 = new Complex[n];
        for (var k = 0; k < n; k++) _awk2[k] = Complex.Pow(aa, -k) * wk2[k];
        (_plan, _fwk2, _wk2) = Prepare(n, mm, wk2);
    }

    /// <summary>Constructor for subclasses that build their own chirp (see <see cref="ZoomFFT"/>).</summary>
    private protected CZT(int n, int m, Complex w, Complex a, Complex[] awk2, Complex[] wk2Full)
    {
        N = n;
        M = m;
        W = w;
        A = a;
        _awk2 = awk2;
        (_plan, _fwk2, _wk2) = Prepare(n, m, wk2Full);
    }

    private static (FftPlan, Complex[], Complex[]) Prepare(int n, int m, Complex[] wk2)
    {
        var nfft = Fft.NextFastLen(n + m - 1);
        var plan = FftPlan.Get(nfft);
        var h = new Complex[nfft];
        // hstack(wk2[n−1:0:−1], wk2[:m]), reciprocal.
        var idx = 0;
        for (var k = n - 1; k >= 1; k--) h[idx++] = 1.0 / wk2[k];
        for (var k = 0; k < m; k++) h[idx++] = 1.0 / wk2[k];
        plan.ExecuteInPlace(h);
        return (plan, h, wk2.AsSpan(0, m).ToArray());
    }

    /// <summary>Transforms one signal of length <see cref="N"/>.</summary>
    public Complex[] Transform(ReadOnlySpan<Complex> x)
    {
        if (x.Length != N) throw new ArgumentException($"CZT defined for length {N}, not {x.Length}.", nameof(x));
        var nfft = _plan.Length;
        var buf = new Complex[nfft];
        for (var k = 0; k < N; k++) buf[k] = x[k] * _awk2[k];
        _plan.ExecuteInPlace(buf);
        for (var k = 0; k < nfft; k++) buf[k] *= _fwk2[k];
        _plan.ExecuteInPlace(buf, inverse: true);
        var y = new Complex[M];
        var s = 1.0 / nfft;
        for (var k = 0; k < M; k++) y[k] = buf[N - 1 + k] * s * _wk2[k];
        return y;
    }

    /// <summary>Transforms one real signal of length <see cref="N"/>.</summary>
    public Complex[] Transform(ReadOnlySpan<double> x)
    {
        var c = new Complex[x.Length];
        for (var i = 0; i < c.Length; i++) c[i] = x[i];
        return Transform(c);
    }

    /// <summary>The points at which the z-transform is evaluated.</summary>
    public Complex[] Points() => Signal.CztPoints(M, W, A);
}

/// <summary>
/// A CZT restricted to a frequency band (<c>scipy.signal.ZoomFFT</c>): m points equally spaced
/// over <c>[f1, f2)</c> (or <c>[f1, f2]</c> with <c>endpoint</c>) on the unit circle — a
/// high-resolution look at part of the spectrum without a huge zero-padded FFT.
/// </summary>
public sealed class ZoomFFT : CZT
{
    /// <summary>Lower edge of the band.</summary>
    public double F1 { get; }

    /// <summary>Upper edge of the band.</summary>
    public double F2 { get; }

    /// <summary>Sampling frequency.</summary>
    public double Fs { get; }

    /// <summary>Builds the zoom transform.</summary>
    /// <param name="n">Input length.</param>
    /// <param name="f1">Lower band edge.</param>
    /// <param name="f2">Upper band edge.</param>
    /// <param name="m">Number of output points; default n.</param>
    /// <param name="fs">Sampling frequency (default 2, so frequencies are relative to Nyquist = 1).</param>
    /// <param name="endpoint">Include <paramref name="f2"/> as the last point.</param>
    public ZoomFFT(int n, double f1, double f2, int? m = null, double fs = 2, bool endpoint = false)
        : this(n, f1, f2, m ?? n, fs, endpoint, Build(n, f1, f2, m ?? n, fs, endpoint))
    {
    }

    /// <summary>Zoom over <c>[0, fn)</c>.</summary>
    public ZoomFFT(int n, double fn, int? m = null, double fs = 2, bool endpoint = false) : this(n, 0.0, fn, m, fs, endpoint) { }

    private ZoomFFT(int n, double f1, double f2, int m, double fs, bool endpoint, (Complex W, Complex A, Complex[] Awk2, Complex[] Wk2) b)
        : base(n, m, b.W, b.A, b.Awk2, b.Wk2)
    {
        F1 = f1;
        F2 = f2;
        Fs = fs;
    }

    private static (Complex, Complex, Complex[], Complex[]) Build(int n, double f1, double f2, int m, double fs, bool endpoint)
    {
        if (n < 1) throw new ArgumentException("n must be positive.", nameof(n));
        if (m < 1) throw new ArgumentException("m must be positive.", nameof(m));
        var scale = endpoint ? (f2 - f1) * m / (fs * (m - 1)) : (f2 - f1) / fs;
        var a = Complex.Exp(new Complex(0, 2 * Math.PI * f1 / fs));
        var len = Math.Max(m, n);
        var wk2 = new Complex[len];
        for (long k = 0; k < len; k++) wk2[k] = Complex.Exp(new Complex(0, -Math.PI * scale * k * k / m));
        var w = Complex.Exp(new Complex(0, -2 * Math.PI / m * scale));
        var awk2 = new Complex[n];
        for (var k = 0; k < n; k++) awk2[k] = Complex.Exp(new Complex(0, -2 * Math.PI * f1 / fs * k)) * wk2[k];
        return (w, a, awk2, wk2);
    }
}

public static partial class Signal
{
    /// <summary>
    /// The points <c>a · w^(−k)</c>, k = 0…m−1, of a chirp z-transform (<c>scipy.signal.czt_points</c>);
    /// with <paramref name="w"/> null they are the m-th roots of unity scaled by a.
    /// </summary>
    public static Complex[] CztPoints(int m, Complex? w = null, Complex? a = null)
    {
        if (m < 1) throw new ArgumentException("m must be positive.", nameof(m));
        var aa = a ?? Complex.One;
        var p = new Complex[m];
        for (var k = 0; k < m; k++)
            p[k] = w is null ? aa * Complex.Exp(new Complex(0, 2 * Math.PI * k / m)) : aa * Complex.Pow(w.Value, -k);
        return p;
    }

    /// <summary>Chirp z-transform of a complex signal (<c>scipy.signal.czt</c>).</summary>
    public static Complex[] Czt(Complex[] x, int? m = null, Complex? w = null, Complex? a = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        return new CZT(x.Length, m, w, a).Transform(x);
    }

    /// <summary>Chirp z-transform of a real signal (<c>scipy.signal.czt</c>).</summary>
    public static Complex[] Czt(double[] x, int? m = null, Complex? w = null, Complex? a = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        return new CZT(x.Length, m, w, a).Transform(x);
    }

    /// <summary>Spectrum over <c>[f1, f2)</c> by the zoom FFT (<c>scipy.signal.zoom_fft</c>).</summary>
    public static Complex[] ZoomFft(double[] x, double f1, double f2, int? m = null, double fs = 2, bool endpoint = false)
    {
        ArgumentNullException.ThrowIfNull(x);
        return new ZoomFFT(x.Length, f1, f2, m, fs, endpoint).Transform(x);
    }

    /// <summary>Spectrum over <c>[0, fn)</c> by the zoom FFT.</summary>
    public static Complex[] ZoomFft(double[] x, double fn, int? m = null, double fs = 2, bool endpoint = false)
        => ZoomFft(x, 0.0, fn, m, fs, endpoint);

    /// <summary>Complex-input zoom FFT over <c>[f1, f2)</c>.</summary>
    public static Complex[] ZoomFft(Complex[] x, double f1, double f2, int? m = null, double fs = 2, bool endpoint = false)
    {
        ArgumentNullException.ThrowIfNull(x);
        return new ZoomFFT(x.Length, f1, f2, m, fs, endpoint).Transform(x);
    }
}
