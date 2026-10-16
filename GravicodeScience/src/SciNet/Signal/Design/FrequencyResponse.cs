using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    private static double[] LinspaceEndpoint(double start, double stop, int num, bool endpoint)
    {
        if (endpoint) return Linspace(start, stop, num);
        var y = new double[num];
        var step = num > 0 ? (stop - start) / num : 0;
        for (var i = 0; i < num; i++) y[i] = i * step + start;
        return y;
    }

    /// <summary>Σ c[k]·x^k for coefficients <paramref name="c"/> given highest power first, evaluated in z⁻¹ (numpy.polynomial order).</summary>
    private static Complex PolyvalLowFirst(ReadOnlySpan<double> c, Complex x)
    {
        var y = Complex.Zero;
        for (var i = c.Length - 1; i >= 0; i--) y = y * x + c[i];
        return y;
    }

    /// <summary>
    /// Frequency response of a digital filter at <paramref name="worN"/> equally spaced
    /// frequencies (<c>scipy.signal.freqz</c>). An FIR (scalar <paramref name="a"/>) goes through
    /// one FFT, as scipy does; anything else evaluates both polynomials at <c>e^{−jω}</c>.
    /// </summary>
    /// <param name="b">Numerator.</param>
    /// <param name="a">Denominator (default <c>[1]</c>).</param>
    /// <param name="worN">Number of frequencies.</param>
    /// <param name="whole">Span the whole circle [0, fs) rather than [0, fs/2).</param>
    /// <param name="fs">Sampling frequency; default 2π, so w is in rad/sample.</param>
    /// <param name="includeNyquist">Include fs/2 as the last point (only when not <paramref name="whole"/>).</param>
    public static (double[] W, Complex[] H) Freqz(double[] b, double[]? a = null, int worN = 512, bool whole = false,
        double fs = 2 * Math.PI, bool includeNyquist = false)
    {
        ArgumentNullException.ThrowIfNull(b);
        a ??= [1.0];
        if (worN < 0) throw new ArgumentException($"worN must be nonnegative, got {worN}.", nameof(worN));
        var n = worN;
        var last = whole ? 2 * Math.PI : Math.PI;
        var w = LinspaceEndpoint(0, last, n, includeNyquist && !whole);
        var nFft = whole ? n : includeNyquist ? 2 * (n - 1) : 2 * n;
        Complex[] h;
        if (a.Length == 1 && nFft >= b.Length && nFft > 0)
        {
            var buf = new Complex[nFft];
            for (var i = 0; i < b.Length; i++) buf[i] = b[i];
            var f = SignalDft(buf);
            h = new Complex[n];
            for (var i = 0; i < n; i++) h[i] = f[i] / a[0];
        }
        else h = EvalBa(b, a, w);
        for (var i = 0; i < n; i++) w[i] *= fs / (2 * Math.PI);
        return (w, h);
    }

    /// <summary>Frequency response of a digital filter at the given frequencies (in the units of <paramref name="fs"/>).</summary>
    public static (double[] W, Complex[] H) Freqz(double[] b, double[]? a, double[] frequencies, double fs = 2 * Math.PI)
    {
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(frequencies);
        a ??= [1.0];
        var w = frequencies.Select(v => 2 * Math.PI * v / fs).ToArray();
        var h = EvalBa(b, a, w);
        return (w.Select(v => v * (fs / (2 * Math.PI))).ToArray(), h);
    }

    private static Complex[] EvalBa(double[] b, double[] a, double[] w)
    {
        var h = new Complex[w.Length];
        for (var i = 0; i < w.Length; i++)
        {
            var zm1 = Complex.Exp(new Complex(0, -w[i]));
            h[i] = PolyvalLowFirst(b, zm1) / PolyvalLowFirst(a, zm1);
        }
        return h;
    }

    /// <summary>Frequency response of a digital filter in zpk form (<c>freqz_zpk</c>).</summary>
    public static (double[] W, Complex[] H) FreqzZpk(FilterZpk zpk, int worN = 512, bool whole = false, double fs = 2 * Math.PI)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        var w = LinspaceEndpoint(0, whole ? 2 * Math.PI : Math.PI, worN, false);
        return ZpkResponse(zpk, w, fs, digital: true);
    }

    /// <summary>Frequency response of a digital zpk filter at the given frequencies.</summary>
    public static (double[] W, Complex[] H) FreqzZpk(FilterZpk zpk, double[] frequencies, double fs = 2 * Math.PI)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        return ZpkResponse(zpk, frequencies.Select(v => 2 * Math.PI * v / fs).ToArray(), fs, digital: true);
    }

    private static (double[] W, Complex[] H) ZpkResponse(FilterZpk zpk, double[] w, double fs, bool digital)
    {
        var h = new Complex[w.Length];
        for (var i = 0; i < w.Length; i++)
        {
            var s = digital ? Complex.Exp(new Complex(0, w[i])) : new Complex(0, w[i]);
            Complex num = 1, den = 1;
            foreach (var z in zpk.Z) num *= s - z;
            foreach (var p in zpk.P) den *= s - p;
            h[i] = zpk.K * num / den;
        }
        var wo = digital ? w.Select(v => v * (fs / (2 * Math.PI))).ToArray() : w;
        return (wo, h);
    }

    /// <summary>Frequency response of second-order sections (<c>sosfreqz</c> / <c>freqz_sos</c>): the product of each section's response.</summary>
    public static (double[] W, Complex[] H) Sosfreqz(NdArray sos, int worN = 512, bool whole = false, double fs = 2 * Math.PI)
    {
        var s = ValidateSos(sos, out var n);
        if (n == 0) throw new ArgumentException("Cannot compute frequencies with no sections.");
        double[] w = [];
        Complex[]? h = null;
        for (var j = 0; j < n; j++)
        {
            var (wj, hj) = Freqz(s.Slice(j * 6, 3).ToArray(), s.Slice(j * 6 + 3, 3).ToArray(), worN, whole, fs);
            w = wj;
            if (h is null) h = hj;
            else for (var i = 0; i < h.Length; i++) h[i] *= hj[i];
        }
        return (w, h!);
    }

    /// <summary>Frequency response of second-order sections at the given frequencies.</summary>
    public static (double[] W, Complex[] H) Sosfreqz(NdArray sos, double[] frequencies, double fs = 2 * Math.PI)
    {
        var s = ValidateSos(sos, out var n);
        if (n == 0) throw new ArgumentException("Cannot compute frequencies with no sections.");
        double[] w = [];
        Complex[]? h = null;
        for (var j = 0; j < n; j++)
        {
            var (wj, hj) = Freqz(s.Slice(j * 6, 3).ToArray(), s.Slice(j * 6 + 3, 3).ToArray(), frequencies, fs);
            w = wj;
            if (h is null) h = hj;
            else for (var i = 0; i < h.Length; i++) h[i] *= hj[i];
        }
        return (w, h!);
    }

    /// <summary>
    /// scipy's <c>findfreqs</c>: a log-spaced frequency grid spanning the interesting region of an
    /// analog filter, chosen from its pole and zero magnitudes.
    /// </summary>
    public static double[] Findfreqs(Complex[] zeros, Complex[] poles, int n)
    {
        var ep = poles.Length == 0 ? [new Complex(-1000, 0)] : poles;
        var ez = ep.Where(v => v.Imaginary >= 0).Concat(zeros.Where(v => Complex.Abs(v) < 1e5 && v.Imaginary >= 0)).ToArray();
        var hiArg = double.NegativeInfinity;
        var loArg = double.PositiveInfinity;
        foreach (var v in ez)
        {
            var integ = Complex.Abs(v) < 1e-10 ? 1.0 : 0.0;
            hiArg = Math.Max(hiArg, 3 * Math.Abs(v.Real + integ) + 1.5 * v.Imaginary);
            loArg = Math.Min(loArg, Math.Abs(v.Real + integ) + 2 * v.Imaginary);
        }
        var hfreq = Math.Round(Math.Log10(hiArg) + 0.5, MidpointRounding.ToEven);
        var lfreq = Math.Round(Math.Log10(0.1 * loArg) - 0.5, MidpointRounding.ToEven);
        return Linspace(lfreq, hfreq, n).Select(v => Math.Pow(10.0, v)).ToArray();
    }

    /// <summary>Frequency response of an analog filter <c>H(jω)</c> (<c>scipy.signal.freqs</c>) on a <see cref="Findfreqs"/> grid.</summary>
    public static (double[] W, Complex[] H) Freqs(double[] b, double[] a, int worN = 200)
        => Freqs(b, a, Findfreqs(Roots(b), Roots(a), worN));

    /// <summary>Frequency response of an analog filter at the given angular frequencies.</summary>
    public static (double[] W, Complex[] H) Freqs(double[] b, double[] a, double[] w)
    {
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(w);
        var h = new Complex[w.Length];
        for (var i = 0; i < w.Length; i++)
        {
            var s = new Complex(0, w[i]);
            h[i] = Polyval(b, s) / Polyval(a, s);
        }
        return (w.ToArray(), h);
    }

    /// <summary>Frequency response of an analog zpk filter (<c>freqs_zpk</c>) on a <see cref="Findfreqs"/> grid.</summary>
    public static (double[] W, Complex[] H) FreqsZpk(FilterZpk zpk, int worN = 200)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        return FreqsZpk(zpk, Findfreqs(zpk.Z, zpk.P, worN));
    }

    /// <summary>Frequency response of an analog zpk filter at the given angular frequencies.</summary>
    public static (double[] W, Complex[] H) FreqsZpk(FilterZpk zpk, double[] w)
    {
        ArgumentNullException.ThrowIfNull(zpk);
        return ZpkResponse(zpk, w, 1, digital: false);
    }

    /// <summary>
    /// Group delay <c>−dφ/dω</c> of a digital filter in samples (<c>scipy.signal.group_delay</c>),
    /// computed exactly from the polynomial <c>b(z)·a*(1/z)</c> rather than by differentiating the
    /// unwrapped phase. Where the response is singular the delay is set to zero, as scipy does.
    /// </summary>
    public static (double[] W, double[] Gd) GroupDelay(double[] b, double[] a, int w = 512, bool whole = false, double fs = 2 * Math.PI)
        => GroupDelayCore(b, a, LinspaceEndpoint(0, whole ? 2 * Math.PI : Math.PI, w, false), fs);

    /// <summary>Group delay at the given frequencies (units of <paramref name="fs"/>).</summary>
    public static (double[] W, double[] Gd) GroupDelay(double[] b, double[] a, double[] frequencies, double fs = 2 * Math.PI)
        => GroupDelayCore(b, a, frequencies.Select(v => 2 * Math.PI * v / fs).ToArray(), fs);

    private static (double[] W, double[] Gd) GroupDelayCore(double[] b, double[] a, double[] w, double fs)
    {
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(a);
        var ar = a.Reverse().ToArray();
        var c = new double[b.Length + ar.Length - 1];
        for (var i = 0; i < b.Length; i++) for (var j = 0; j < ar.Length; j++) c[i + j] += b[i] * ar[j];
        var cr = new double[c.Length];
        for (var i = 0; i < c.Length; i++) cr[i] = c[i] * i;
        var gd = new double[w.Length];
        for (var i = 0; i < w.Length; i++)
        {
            var z = Complex.Exp(new Complex(0, -w[i]));
            var num = PolyvalLowFirst(cr, z);
            var den = PolyvalLowFirst(c, z);
            var g = (num / den).Real - a.Length + 1;
            gd[i] = double.IsFinite(g) ? g : 0;
        }
        return (w.Select(v => v * (fs / (2 * Math.PI))).ToArray(), gd);
    }
}
