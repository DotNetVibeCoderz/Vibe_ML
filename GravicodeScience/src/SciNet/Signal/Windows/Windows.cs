using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// <c>scipy.signal</c> for .NET: windows, FIR and IIR design, filtering, LTI systems and waveforms.
/// </summary>
/// <remarks>
/// Window functions follow scipy's <c>sym</c> convention exactly: <c>sym: true</c> gives the
/// symmetric window used for filter design; <c>sym: false</c> gives the periodic window used for
/// spectral analysis, computed as a symmetric window of length <c>M + 1</c> with the last sample
/// dropped. That second form is what makes a Hann window sum to a constant under 50% overlap.
/// </remarks>
public static partial class Signal
{
    // ------------------------------------------------------------------ helpers

    private static bool LenGuard(int m)
    {
        if (m < 0) throw new ArgumentException("Window length M must be a non-negative integer.", nameof(m));
        return m <= 1;
    }

    private static double[] Ones(int m)
    {
        var w = new double[m];
        Array.Fill(w, 1.0);
        return w;
    }

    private static double[] Truncate(double[] w, bool needed) => needed ? w[..^1] : w;

    /// <summary>numpy's <c>linspace</c>, including its exact end point.</summary>
    private static double[] Linspace(double start, double stop, int num)
    {
        var y = new double[num];
        if (num == 0) return y;
        if (num == 1) { y[0] = start; return y; }
        var step = (stop - start) / (num - 1);
        for (var i = 0; i < num; i++) y[i] = i * step + start;
        y[num - 1] = stop;
        return y;
    }

    private static double Sinc(double x)
    {
        if (x == 0) return 1.0;
        var px = Math.PI * x;
        return Math.Sin(px) / px;
    }

    /// <summary>
    /// Modified Bessel function of the first kind, order zero. Every term of the power series is
    /// positive, so it sums without cancellation to a few ulp; above 500 the series would overflow
    /// before the answer does, so the asymptotic expansion takes over there.
    /// </summary>
    internal static double BesselI0(double x)
    {
        x = Math.Abs(x);
        if (x > 500)
        {
            // I0(x) ~ e^x / sqrt(2 pi x) * (1 + 1/(8x) + 9/(128x^2) + ...)
            double sum = 1, term = 1;
            for (var k = 1; k < 30; k++)
            {
                term *= (2 * k - 1) * (2 * k - 1) / (8.0 * k * x);
                sum += term;
                if (term < 1e-17 * sum) break;
            }
            return Math.Exp(x) / Math.Sqrt(2 * Math.PI * x) * sum;
        }
        var q = 0.25 * x * x;
        double s = 1, t = 1;
        for (var k = 1; k < 1000; k++)
        {
            t *= q / ((double)k * k);
            s += t;
            if (t < 1e-17 * s) break;
        }
        return s;
    }

    // ------------------------------------------------------------------ cosine family

    /// <summary>A sum of cosines, <c>w(n) = Σ a[k] cos(k·x)</c> with x spanning [−π, π].</summary>
    /// <param name="m">Number of points.</param>
    /// <param name="a">Weights of the cosine terms.</param>
    /// <param name="sym">Symmetric (filter design) or periodic (spectral analysis).</param>
    public static double[] GeneralCosine(int m, double[] a, bool sym = true)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (LenGuard(m)) return Ones(m);
        var needsTrunc = !sym;
        var mm = sym ? m : m + 1;
        var fac = Linspace(-Math.PI, Math.PI, mm);
        var w = new double[mm];
        for (var k = 0; k < a.Length; k++)
            for (var i = 0; i < mm; i++)
                w[i] += a[k] * Math.Cos(k * fac[i]);
        return Truncate(w, needsTrunc);
    }

    /// <summary>A generalised Hamming window, <c>α − (1−α)cos(2πn/(M−1))</c>.</summary>
    public static double[] GeneralHamming(int m, double alpha, bool sym = true) => GeneralCosine(m, [alpha, 1.0 - alpha], sym);

    /// <summary>The Hann window.</summary>
    public static double[] Hann(int m, bool sym = true) => GeneralHamming(m, 0.5, sym);

    /// <summary>The Hamming window.</summary>
    public static double[] Hamming(int m, bool sym = true) => GeneralHamming(m, 0.54, sym);

    /// <summary>The Blackman window (coefficients 0.42, 0.5, 0.08).</summary>
    public static double[] Blackman(int m, bool sym = true) => GeneralCosine(m, [0.42, 0.5, 0.08], sym);

    /// <summary>The minimum four-term Blackman-Harris window (Nuttall's coefficients).</summary>
    public static double[] Nuttall(int m, bool sym = true) => GeneralCosine(m, [0.3635819, 0.4891775, 0.1365995, 0.0106411], sym);

    /// <summary>The four-term Blackman-Harris window.</summary>
    public static double[] BlackmanHarris(int m, bool sym = true) => GeneralCosine(m, [0.35875, 0.48829, 0.14128, 0.01168], sym);

    /// <summary>The flat-top window, for amplitude-accurate spectral measurements.</summary>
    public static double[] Flattop(int m, bool sym = true) => GeneralCosine(m, [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368], sym);

    // ------------------------------------------------------------------ simple shapes

    /// <summary>The rectangular window.</summary>
    public static double[] Boxcar(int m, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        return Ones(m);
    }

    /// <summary>The triangular window, which unlike <see cref="Bartlett"/> has no zero end points.</summary>
    public static double[] Triang(int m, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var half = (mm + 1) / 2;
        var w = new double[mm];
        for (var i = 0; i < half; i++)
        {
            double n = i + 1;
            var v = mm % 2 == 0 ? (2 * n - 1.0) / mm : 2 * n / (mm + 1.0);
            w[i] = v;
            w[mm - 1 - i] = v;
        }
        return Truncate(w, !sym);
    }

    /// <summary>The Bartlett window: triangular with zero end points.</summary>
    public static double[] Bartlett(int m, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var w = new double[mm];
        for (var i = 0; i < mm; i++)
        {
            double n = i;
            w[i] = n <= (mm - 1) / 2.0 ? 2.0 * n / (mm - 1) : 2.0 - 2.0 * n / (mm - 1);
        }
        return Truncate(w, !sym);
    }

    /// <summary>The modified Bartlett-Hann window.</summary>
    public static double[] BartHann(int m, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var w = new double[mm];
        for (var i = 0; i < mm; i++)
        {
            var fac = Math.Abs(i / (mm - 1.0) - 0.5);
            w[i] = 0.62 - 0.48 * fac + 0.38 * Math.Cos(2 * Math.PI * fac);
        }
        return Truncate(w, !sym);
    }

    /// <summary>The Parzen (de la Vallée Poussin) window, a piecewise cubic.</summary>
    public static double[] Parzen(int m, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var w = new double[mm];
        for (var i = 0; i < mm; i++)
        {
            var n = Math.Abs(-(mm - 1) / 2.0 + i);
            var r = n / (mm / 2.0);
            w[i] = n <= (mm - 1) / 4.0 ? 1 - 6 * Math.Pow(r, 2.0) + 6 * Math.Pow(r, 3.0) : 2 * Math.Pow(1 - r, 3.0);
        }
        return Truncate(w, !sym);
    }

    /// <summary>The Bohman window: the convolution of two half-cosines.</summary>
    public static double[] Bohman(int m, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var lin = Linspace(-1, 1, mm);
        var w = new double[mm];
        for (var i = 1; i < mm - 1; i++)
        {
            var fac = Math.Abs(lin[i]);
            w[i] = (1 - fac) * Math.Cos(Math.PI * fac) + 1.0 / Math.PI * Math.Sin(Math.PI * fac);
        }
        return Truncate(w, !sym);
    }

    /// <summary>The cosine (sine) window.</summary>
    public static double[] Cosine(int m, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var w = new double[mm];
        for (var i = 0; i < mm; i++) w[i] = Math.Sin(Math.PI / mm * (i + 0.5));
        return Truncate(w, !sym);
    }

    /// <summary>The exponential (Poisson) window, <c>exp(−|n − center|/τ)</c>.</summary>
    /// <param name="m">Number of points.</param>
    /// <param name="center">Centre; must be null for a symmetric window (scipy raises otherwise).</param>
    /// <param name="tau">Decay constant.</param>
    /// <param name="sym">Symmetric or periodic.</param>
    public static double[] Exponential(int m, double? center = null, double tau = 1.0, bool sym = true)
    {
        if (sym && center is not null) throw new ArgumentException("If sym is true, center must be null.", nameof(center));
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var c = center ?? (mm - 1) / 2.0;
        var w = new double[mm];
        for (var i = 0; i < mm; i++) w[i] = Math.Exp(-Math.Abs(i - c) / tau);
        return Truncate(w, !sym);
    }

    /// <summary>The Tukey (tapered cosine) window; α = 0 is a boxcar and α = 1 a Hann window.</summary>
    public static double[] Tukey(int m, double alpha = 0.5, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        if (alpha <= 0) return Ones(m);
        if (alpha >= 1.0) return Hann(m, sym);
        var mm = sym ? m : m + 1;
        var width = (int)Math.Floor(alpha * (mm - 1) / 2.0);
        var w = new double[mm];
        for (var i = 0; i < mm; i++)
        {
            double n = i;
            if (i <= width) w[i] = 0.5 * (1 + Math.Cos(Math.PI * (-1 + 2.0 * n / alpha / (mm - 1))));
            else if (i < mm - width - 1) w[i] = 1.0;
            else w[i] = 0.5 * (1 + Math.Cos(Math.PI * (-2.0 / alpha + 1 + 2.0 * n / alpha / (mm - 1))));
        }
        return Truncate(w, !sym);
    }

    /// <summary>The Kaiser window, <c>I0(β√(1 − (2n/(M−1) − 1)²)) / I0(β)</c>.</summary>
    public static double[] Kaiser(int m, double beta, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var alpha = (mm - 1) / 2.0;
        var den = BesselI0(beta);
        var w = new double[mm];
        for (var i = 0; i < mm; i++)
        {
            var r = (i - alpha) / alpha;
            w[i] = BesselI0(beta * Math.Sqrt(Math.Max(0.0, 1 - Math.Pow(r, 2.0)))) / den;
        }
        return Truncate(w, !sym);
    }

    /// <summary>
    /// The Kaiser-Bessel-derived window used by MDCT codecs. Only symmetric, even lengths exist.
    /// </summary>
    public static double[] KaiserBesselDerived(int m, double beta, bool sym = true)
    {
        if (!sym) throw new ArgumentException("Kaiser-Bessel Derived windows are only defined for symmetric shapes.", nameof(sym));
        if (m < 1) return [];
        if (m % 2 != 0) throw new ArgumentException("Kaiser-Bessel Derived windows are only defined for even number of points.", nameof(m));
        var k = Kaiser(m / 2 + 1, beta);
        var csum = new double[k.Length];
        var acc = 0.0;
        for (var i = 0; i < k.Length; i++) { acc += k[i]; csum[i] = acc; }
        var half = m / 2;
        var w = new double[m];
        for (var i = 0; i < half; i++)
        {
            var v = Math.Sqrt(csum[i] / csum[^1]);
            w[i] = v;
            w[m - 1 - i] = v;
        }
        return w;
    }

    /// <summary>The Gaussian window with standard deviation <paramref name="std"/> samples.</summary>
    public static double[] Gaussian(int m, double std, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var sig2 = 2 * std * std;
        var w = new double[mm];
        for (var i = 0; i < mm; i++)
        {
            var n = i - (mm - 1.0) / 2.0;
            w[i] = Math.Exp(-(n * n) / sig2);
        }
        return Truncate(w, !sym);
    }

    /// <summary>The generalised Gaussian, <c>exp(−½|n/σ|^(2p))</c>; p = 1 is <see cref="Gaussian"/>.</summary>
    public static double[] GeneralGaussian(int m, double p, double sig, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var w = new double[mm];
        for (var i = 0; i < mm; i++)
        {
            var n = i - (mm - 1.0) / 2.0;
            w[i] = Math.Exp(-0.5 * Math.Pow(Math.Abs(n / sig), 2 * p));
        }
        return Truncate(w, !sym);
    }

    /// <summary>The Lanczos (sinc) window.</summary>
    public static double[] Lanczos(int m, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var w = new double[mm];
        // Right half from arange(start, mm); the left half is its mirror, the centre (odd) is one.
        var start = mm % 2 == 0 ? mm / 2 : (mm + 1) / 2;
        var count = mm - start;
        for (var j = 0; j < count; j++)
        {
            var v = Sinc(2.0 * (start + j) / (mm - 1) - 1.0);
            w[mm - count + j] = v;
            w[count - 1 - j] = v;
        }
        if (mm % 2 == 1) w[count] = 1.0;
        return Truncate(w, !sym);
    }

    /// <summary>
    /// The Dolph-Chebyshev window: the narrowest main lobe for a given sidelobe level
    /// <paramref name="at"/> (dB). Built from the Chebyshev polynomial's DFT, as scipy does.
    /// </summary>
    /// <remarks>Below about 45 dB the equivalent noise bandwidth stops shrinking with attenuation, so
    /// scipy warns; this method does not.</remarks>
    public static double[] Chebwin(int m, double at, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var order = mm - 1.0;
        var beta = Math.Cosh(1.0 / order * Math.Acosh(Math.Pow(10, Math.Abs(at) / 20.0)));
        var p = new Complex[mm];
        for (var k = 0; k < mm; k++)
        {
            var x = beta * Math.Cos(Math.PI * k / mm);
            double v;
            if (x > 1) v = Math.Cosh(order * Math.Acosh(x));
            else if (x < -1) v = (2 * (mm % 2) - 1) * Math.Cosh(order * Math.Acosh(-x));
            else v = Math.Cos(order * Math.Acos(x));
            p[k] = mm % 2 == 1 ? new Complex(v, 0) : v * Complex.FromPolarCoordinates(1.0, Math.PI / mm * k);
        }
        var spec = SignalDft(p);
        double[] w;
        if (mm % 2 == 1)
        {
            var n = (mm + 1) / 2;
            w = new double[2 * n - 1];
            for (var i = 0; i < n; i++) w[n - 1 + i] = spec[i].Real;
            for (var i = 1; i < n; i++) w[n - 1 - i] = spec[i].Real;
        }
        else
        {
            var n = mm / 2 + 1;
            w = new double[2 * (n - 1)];
            for (var i = 1; i < n; i++)
            {
                w[n - 1 - i] = spec[i].Real;
                w[n - 2 + i] = spec[i].Real;
            }
        }
        var max = w.Max();
        for (var i = 0; i < w.Length; i++) w[i] /= max;
        return Truncate(w, !sym);
    }

    /// <summary>
    /// The Taylor window, used in radar to hold <paramref name="nbar"/> − 1 sidelobes near
    /// <paramref name="sll"/> dB below the main lobe.
    /// </summary>
    /// <param name="m">Number of points.</param>
    /// <param name="nbar">Number of nearly constant-level sidelobes adjacent to the main lobe.</param>
    /// <param name="sll">Desired suppression of sidelobe level in dB (positive).</param>
    /// <param name="norm">Scale so the centre is one (for an even length, the value it would have).</param>
    /// <param name="sym">Symmetric or periodic.</param>
    public static double[] Taylor(int m, int nbar = 4, double sll = 30, bool norm = true, bool sym = true)
    {
        if (LenGuard(m)) return Ones(m);
        var mm = sym ? m : m + 1;
        var b = Math.Pow(10, sll / 20);
        var a = Math.Acosh(b) / Math.PI;
        var s2 = nbar * (double)nbar / (a * a + (nbar - 0.5) * (nbar - 0.5));
        var cnt = Math.Max(nbar - 1, 0);
        var fm = new double[cnt];
        for (var mi = 0; mi < cnt; mi++)
        {
            double mv = mi + 1;
            var m2 = mv * mv;
            var numer = mi % 2 == 0 ? 1.0 : -1.0;
            for (var j = 0; j < cnt; j++)
            {
                var ma = j + 1.0;
                numer *= 1 - m2 / s2 / (a * a + (ma - 0.5) * (ma - 0.5));
            }
            var denom = 2.0;
            for (var j = 0; j < cnt; j++)
            {
                if (j == mi) continue;
                var mj = j + 1.0;
                denom *= 1 - m2 / (mj * mj);
            }
            fm[mi] = numer / denom;
        }
        double W(double n)
        {
            var s = 0.0;
            for (var j = 0; j < cnt; j++) s += fm[j] * Math.Cos(2 * Math.PI * (j + 1) * (n - mm / 2.0 + 0.5) / mm);
            return 1 + 2 * s;
        }
        var w = new double[mm];
        for (var i = 0; i < mm; i++) w[i] = W(i);
        if (norm)
        {
            var scale = 1.0 / W((mm - 1) / 2.0);
            for (var i = 0; i < mm; i++) w[i] *= scale;
        }
        return Truncate(w, !sym);
    }

    // ------------------------------------------------------------------ get_window

    /// <summary>
    /// A window by name, the analogue of <c>scipy.signal.get_window</c>. Defaults to the
    /// <b>periodic</b> form (<paramref name="fftbins"/> true), which is what spectral analysis wants;
    /// a <c>_symmetric</c> / <c>_periodic</c> suffix on the name overrides it.
    /// </summary>
    /// <param name="window">Name and parameters, e.g. <c>"hann"</c> or <c>("kaiser", 8.6)</c>.</param>
    /// <param name="nx">Number of samples.</param>
    /// <param name="fftbins">Periodic (true) or symmetric (false).</param>
    public static double[] GetWindow(WindowSpec window, int nx, bool fftbins = true)
    {
        ArgumentNullException.ThrowIfNull(window);
        if (nx <= 0) throw new ArgumentException($"Parameter Nx={nx} is not a positive integer.", nameof(nx));
        var sym = !fftbins;
        var name = window.Name.ToLowerInvariant();
        if (name.EndsWith("_symmetric", StringComparison.Ordinal)) { sym = true; name = name[..^10]; }
        else if (name.EndsWith("_periodic", StringComparison.Ordinal)) { sym = false; name = name[..^9]; }
        var a = window.Parameters ?? [];

        void NoArgs() { if (a.Length > 0) throw new ArgumentException($"'{name}' does not allow parameters."); }
        double Need(int i)
        {
            if (a.Length <= i) throw new ArgumentException($"'{name}' must have parameters.");
            return a[i];
        }

        switch (name)
        {
            case "barthann" or "brthan" or "bth": NoArgs(); return BartHann(nx, sym);
            case "bartlett" or "bart" or "brt": NoArgs(); return Bartlett(nx, sym);
            case "blackman" or "black" or "blk": NoArgs(); return Blackman(nx, sym);
            case "blackmanharris" or "blackharr" or "bkh": NoArgs(); return BlackmanHarris(nx, sym);
            case "bohman" or "bman" or "bmn": NoArgs(); return Bohman(nx, sym);
            case "boxcar" or "box" or "ones" or "rect" or "rectangular": NoArgs(); return Boxcar(nx, sym);
            case "chebwin" or "cheb": return Chebwin(nx, Need(0), sym);
            case "cosine" or "halfcosine": NoArgs(); return Cosine(nx, sym);
            case "dpss": return Dpss(nx, Need(0), sym);
            case "exponential" or "poisson":
                {
                    double? c = a.Length > 0 && !double.IsNaN(a[0]) ? a[0] : null;
                    return Exponential(nx, c, a.Length > 1 ? a[1] : 1.0, sym);
                }
            case "flattop" or "flat" or "flt": NoArgs(); return Flattop(nx, sym);
            case "gaussian" or "gauss" or "gss": return Gaussian(nx, Need(0), sym);
            case "general cosine" or "general_cosine": Need(0); return GeneralCosine(nx, a, sym);
            case "general gaussian" or "general_gaussian" or "general gauss" or "general_gauss" or "ggs":
                return GeneralGaussian(nx, Need(0), Need(1), sym);
            case "general hamming" or "general_hamming": return GeneralHamming(nx, Need(0), sym);
            case "hamming" or "hamm" or "ham": NoArgs(); return Hamming(nx, sym);
            case "hann" or "han": NoArgs(); return Hann(nx, sym);
            case "kaiser" or "ksr": return Kaiser(nx, Need(0), sym);
            case "kaiser bessel derived" or "kaiser_bessel_derived" or "kbd": return KaiserBesselDerived(nx, Need(0), sym);
            case "lanczos" or "sinc": NoArgs(); return Lanczos(nx, sym);
            case "nuttall" or "nutl" or "nut": NoArgs(); return Nuttall(nx, sym);
            case "parzen" or "parz" or "par": NoArgs(); return Parzen(nx, sym);
            case "taylor" or "taylorwin":
                return Taylor(nx, a.Length > 0 ? (int)a[0] : 4, a.Length > 1 ? a[1] : 30, a.Length <= 2 || a[2] != 0, sym);
            case "triangle" or "triang" or "tri": NoArgs(); return Triang(nx, sym);
            case "tukey" or "tuk": return Tukey(nx, a.Length > 0 ? a[0] : 0.5, sym);
            default: throw new ArgumentException($"Invalid window name '{window.Name}'.", nameof(window));
        }
    }

    // ------------------------------------------------------------------ DFT helper

    /// <summary>
    /// Forward DFT through GraviNum's FFT (any length). The SciNet <c>Fft</c> module is a separate
    /// deliverable, so Signal deliberately does not depend on it.
    /// </summary>
    internal static Complex[] SignalDft(Complex[] x) => Gravicode.Science.GraviNum.Signal.Fft.Forward(x);

    /// <summary>Inverse DFT (scaled by 1/n) through GraviNum's FFT.</summary>
    internal static Complex[] SignalIdft(Complex[] x) => Gravicode.Science.GraviNum.Signal.Fft.Inverse(x);
}
