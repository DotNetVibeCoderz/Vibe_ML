using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    // ------------------------------------------------------------------ analog prototypes

    /// <summary>Analog Butterworth low-pass prototype of order <paramref name="n"/>, cutoff 1 rad/s.</summary>
    public static FilterZpk Buttap(int n)
    {
        if (n < 0) throw new ArgumentException("Filter order must be a nonnegative integer.", nameof(n));
        var p = new Complex[n];
        for (var i = 0; i < n; i++)
        {
            double m = -n + 1 + 2 * i;
            p[i] = -Complex.Exp(new Complex(0, Math.PI * m / (2 * n)));
        }
        return new FilterZpk([], p, 1.0);
    }

    /// <summary>Analog Chebyshev type I prototype with <paramref name="rp"/> dB passband ripple.</summary>
    public static FilterZpk Cheb1ap(int n, double rp)
    {
        if (n < 0) throw new ArgumentException("Filter order must be a nonnegative integer.", nameof(n));
        if (n == 0) return new FilterZpk([], [], Math.Pow(10, -rp / 20));
        var eps = Math.Sqrt(Math.Pow(10, 0.1 * rp) - 1.0);
        var mu = 1.0 / n * Math.Asinh(1 / eps);
        var p = new Complex[n];
        var prod = Complex.One;
        for (var i = 0; i < n; i++)
        {
            double m = -n + 1 + 2 * i;
            var theta = Math.PI * m / (2 * n);
            p[i] = -Complex.Sinh(new Complex(mu, theta));
            prod *= -p[i];
        }
        var k = prod.Real;
        if (n % 2 == 0) k /= Math.Sqrt(1 + eps * eps);
        return new FilterZpk([], p, k);
    }

    /// <summary>Analog Chebyshev type II prototype with <paramref name="rs"/> dB stopband attenuation.</summary>
    public static FilterZpk Cheb2ap(int n, double rs)
    {
        if (n < 0) throw new ArgumentException("Filter order must be a nonnegative integer.", nameof(n));
        if (n == 0) return new FilterZpk([], [], 1.0);
        var de = 1.0 / Math.Sqrt(Math.Pow(10, 0.1 * rs) - 1);
        var mu = Math.Asinh(1.0 / de) / n;
        var ms = new List<double>();
        if (n % 2 == 1)
        {
            for (var m = -n + 1; m < 0; m += 2) ms.Add(m);
            for (var m = 2; m < n; m += 2) ms.Add(m);
        }
        else for (var m = -n + 1; m < n; m += 2) ms.Add(m);
        var z = ms.Select(m => Complex.ImaginaryOne / Math.Sin(m * Math.PI / (2 * n))).ToArray();
        var p = new Complex[n];
        for (var i = 0; i < n; i++)
        {
            double m = -n + 1 + 2 * i;
            p[i] = -1 / Complex.Sinh(new Complex(mu, Math.PI * m / (2 * n)));
        }
        var k = (Prod(p, v => -v) / Prod(z, v => -v)).Real;
        return new FilterZpk(z, p, k);
    }

    /// <summary><c>exp(x) − 1</c> without cancellation for small x.</summary>
    private static double ExpM1(double x)
    {
        if (Math.Abs(x) < 0.5)
        {
            // expm1(x) = 2 tanh(x/2) / (1 - tanh(x/2)), accurate across the range.
            var t = Math.Tanh(x / 2);
            return 2 * t / (1 - t);
        }
        return Math.Exp(x) - 1;
    }

    /// <summary>
    /// Analog elliptic (Cauer) prototype with <paramref name="rp"/> dB passband ripple and
    /// <paramref name="rs"/> dB stopband attenuation, following scipy's Orfanidis-based design.
    /// </summary>
    public static FilterZpk Ellipap(int n, double rp, double rs)
    {
        const double epsilon = 2e-16;
        if (n < 0) throw new ArgumentException("Filter order must be a nonnegative integer.", nameof(n));
        if (n == 0) return new FilterZpk([], [], Math.Pow(10, -rp / 20));
        if (n == 1)
        {
            var p1 = -Math.Sqrt(1.0 / ExpM1(Math.Log(10) * (0.1 * rp)));
            return new FilterZpk([], [new Complex(p1, 0)], -p1);
        }
        var epsSq = ExpM1(Math.Log(10) * (0.1 * rp));
        var eps = Math.Sqrt(epsSq);
        var ck1Sq = epsSq / ExpM1(Math.Log(10) * (0.1 * rs));
        if (ck1Sq == 0) throw new ArgumentException("Cannot design a filter with given rp and rs specifications.");
        var val0 = SignalElliptic.K(ck1Sq);
        var m = SignalElliptic.EllipDeg(n, ck1Sq);
        var capk = SignalElliptic.K(m);
        var js = new List<int>();
        for (var j = 1 - n % 2; j < n; j += 2) js.Add(j);
        var s = new double[js.Count];
        var c = new double[js.Count];
        var d = new double[js.Count];
        for (var i = 0; i < js.Count; i++) (s[i], c[i], d[i], _) = SignalElliptic.Ellipj(js[i] * capk / n, m);
        var zl = new List<Complex>();
        foreach (var sv0 in s)
            if (Math.Abs(sv0) > epsilon) zl.Add(Complex.ImaginaryOne * (1.0 / (Math.Sqrt(m) * sv0)));
        var z = zl.Concat(zl.Select(Complex.Conjugate)).ToArray();

        var r = SignalElliptic.ArcJacSc1(1.0 / eps, ck1Sq);
        var v0 = capk * r / (n * val0);
        var (sv, cv, dv, _) = SignalElliptic.Ellipj(v0, 1 - m);
        var p = new Complex[js.Count];
        for (var i = 0; i < js.Count; i++)
        {
            var den = 1 - Math.Pow(d[i] * sv, 2.0);
            p[i] = new Complex(-(c[i] * d[i] * sv * cv) / den, -(s[i] * dv) / den);
        }
        Complex[] pAll;
        if (n % 2 == 1)
        {
            var nrm = 0.0;
            foreach (var v in p) nrm += (v * Complex.Conjugate(v)).Real;
            nrm = Math.Sqrt(nrm);
            var newp = p.Where(v => Math.Abs(v.Imaginary) > epsilon * nrm).Select(Complex.Conjugate);
            pAll = p.Concat(newp).ToArray();
        }
        else pAll = p.Concat(p.Select(Complex.Conjugate)).ToArray();
        var k = (Prod(pAll, v => -v) / Prod(z, v => -v)).Real;
        if (n % 2 == 0) k /= Math.Sqrt(1 + epsSq);
        return new FilterZpk(z, pAll, k);
    }

    // ------------------------------------------------------------------ Bessel

    /// <summary>
    /// Analog Bessel/Thomson prototype. Poles are the reciprocals of the zeros of the Bessel
    /// polynomial <c>y_N</c>, found as scipy does — Aberth iteration from Campos &amp; Calderón's
    /// asymptotic starting values, then Newton — except that <c>y_N</c> is evaluated by its
    /// three-term recurrence rather than through <c>K_{N+1/2}</c>, which is the same function up
    /// to a factor that has no zeros.
    /// </summary>
    /// <param name="n">Order.</param>
    /// <param name="norm">Frequency normalisation.</param>
    public static FilterZpk Besselap(int n, BesselNorm norm = BesselNorm.Phase)
    {
        if (n < 0) throw new ArgumentException("Filter order must be a nonnegative integer.", nameof(n));
        if (n == 0) return new FilterZpk([], [], 1.0);
        var x = BesselZeros(n);
        var p = x.Select(v => 1 / v).ToArray();
        // a_last = (2N)! / (2^N N!), an exact integer; scipy takes math.log10 of the Python int.
        var logALast = Math.Log10(FallingFactorialOverPow2(n));
        double k;
        switch (norm)
        {
            case BesselNorm.Delay:
                k = FallingFactorialOverPow2(n);
                break;
            case BesselNorm.Mag:
                {
                    var aLast = FallingFactorialOverPow2(n);
                    var nf = BesselNormFactor(p, aLast);
                    for (var i = 0; i < n; i++) p[i] /= nf;
                    k = Math.Pow(nf, -n) * aLast;
                    break;
                }
            default:
                {
                    var f = Math.Pow(10, -logALast / n);
                    for (var i = 0; i < n; i++) p[i] *= f;
                    k = 1;
                    break;
                }
        }
        return new FilterZpk([], p, k);
    }

    private static double FallingFactorialOverPow2(int n)
    {
        // Exact while it fits: the integer (2N)!/(N! 2^N) via BigInteger, then converted.
        var v = BigInteger.One;
        for (var i = n + 1; i <= 2 * n; i++) v *= i;
        v /= BigInteger.Pow(2, n);
        return (double)v;
    }

    private static Complex[] BesselZeros(int n)
    {
        if (n == 1) return [new Complex(-1, 0)];
        // Campos & Calderón (2011) starting values.
        double Pv(double xx, params double[] c) { var r = 0.0; for (var i = c.Length - 1; i >= 0; i--) r = r * xx + c[i]; return r; }
        double nn = n;
        var s = Pv(nn, 0, 0, 2, 0, -3, 1);
        var b3 = Pv(nn, 16, -8) / s;
        var b2 = Pv(nn, -24, -12, 12) / s;
        var b1 = Pv(nn, 8, 24, -12, -2) / s;
        var b0 = Pv(nn, 0, -6, 0, 5, -1) / s;
        var r = Pv(nn, 0, 0, 2, 1);
        var a1 = Pv(nn, -6, -6) / r;
        var a2 = 6 / r;
        var x = new Complex[n];
        for (var k = 1; k <= n; k++) x[k - 1] = new Complex(Pv(k, 0, a1, a2), Pv(k, b0, b1, b2, b3));

        // Aberth-Ehrlich iteration on y_n.
        var alpha = new Complex[n];
        for (var it = 0; it < 50; it++)
        {
            var done = true;
            for (var k = 0; k < n; k++)
            {
                var (f, fp) = BesselPoly(n, x[k]);
                alpha[k] = -f / fp;
                if (Complex.Abs(alpha[k]) > 1e-15) done = false;
            }
            for (var k = 0; k < n; k++)
            {
                var beta = Complex.Zero;
                for (var j = 0; j < n; j++) if (j != k) beta += 1 / (x[k] - x[j]);
                x[k] += alpha[k] / (1 + alpha[k] * beta);
            }
            if (x.Any(v => !double.IsFinite(v.Real) || !double.IsFinite(v.Imaginary)))
                throw new InvalidOperationException("Bessel root-finding failed.");
            if (done) break;
        }
        // Newton polish with y_n evaluated in double-double: the recurrence's intermediate terms
        // grow to ~1e7 and cancel at a root, so a double evaluation leaves y_n as 1e-9 of noise
        // and the root uncertain at ~1e-5 by N = 25. Twice the precision restores ~1e-16.
        for (var k = 0; k < n; k++)
        {
            for (var it = 0; it < 20; it++)
            {
                var (f, fp) = BesselPolyDd(n, x[k]);
                if (fp == Complex.Zero) break;
                var step = f / fp;
                x[k] -= step;
                if (Complex.Abs(step) <= 2e-16 * Math.Max(1, Complex.Abs(x[k]))) break;
            }
        }
        var outv = new Complex[n];
        for (var k = 0; k < n; k++) outv[k] = (x[k] + Complex.Conjugate(x[n - 1 - k])) / 2;
        return outv;
    }

    /// <summary>The Bessel polynomial y_n(x) and its derivative by the forward recurrence
    /// y_k = (2k−1)·x·y_{k−1} + y_{k−2}.</summary>
    private static (Complex Y, Complex Dy) BesselPoly(int n, Complex x)
    {
        Complex y0 = 1, d0 = 0;
        Complex y1 = 1 + x, d1 = 1;
        for (var k = 2; k <= n; k++)
        {
            var y2 = (2 * k - 1) * x * y1 + y0;
            var d2 = (2 * k - 1) * (y1 + x * d1) + d0;
            y0 = y1; d0 = d1; y1 = y2; d1 = d2;
        }
        return (y1, d1);
    }

    /// <summary>y_n(x) and y_n'(x) by the same recurrence as <see cref="BesselPoly"/>, carried in double-double.</summary>
    private static (Complex Y, Complex Dy) BesselPolyDd(int n, Complex x)
    {
        var y0 = new SignalDdComplex(1, 0);
        var d0 = new SignalDdComplex(0, 0);
        var y1 = new SignalDdComplex(1 + x.Real, x.Imaginary).Fix(1, x);
        var d1 = new SignalDdComplex(1, 0);
        for (var k = 2; k <= n; k++)
        {
            var c = 2 * k - 1;
            var xy1 = y1.Mul(x).Scale(c);
            var y2 = xy1.Add(y0);
            var d2 = y1.Add(d1.Mul(x)).Scale(c).Add(d0);
            y0 = y1; d0 = d1; y1 = y2; d1 = d2;
        }
        return (y1.ToComplex(), d1.ToComplex());
    }

    /// <summary>Frequency where the Bessel magnitude is −3 dB, by scipy's own secant iteration from 1.5.</summary>
    private static double BesselNormFactor(Complex[] p, double k)
    {
        double Cutoff(double w)
        {
            var prod = Complex.One;
            foreach (var pi in p) prod *= new Complex(0, w) - pi;
            return Complex.Abs(k / prod) - 1 / Math.Sqrt(2);
        }
        return SecantNewton(Cutoff, 1.5);
    }

    /// <summary>scipy's <c>newton</c> without a derivative: the secant method, tol 1.48e-8, 50 iterations.</summary>
    private static double SecantNewton(Func<double, double> f, double x0, double tol = 1.48e-8, int maxiter = 50)
    {
        var p0 = x0;
        const double eps = 1e-4;
        var p1 = x0 * (1 + eps);
        p1 += p1 >= 0 ? eps : -eps;
        var q0 = f(p0);
        var q1 = f(p1);
        if (Math.Abs(q1) < Math.Abs(q0)) { (p0, p1) = (p1, p0); (q0, q1) = (q1, q0); }
        var p = p1;
        for (var it = 0; it < maxiter; it++)
        {
            if (q1 == q0) return (p1 + p0) / 2.0;
            p = Math.Abs(q1) > Math.Abs(q0) ? (-q0 / q1 * p1 + p0) / (1 - q0 / q1) : (-q1 / q0 * p0 + p1) / (1 - q1 / q0);
            if (Math.Abs(p - p1) <= tol) return p;
            p0 = p1; q0 = q1;
            p1 = p; q1 = f(p1);
        }
        throw new InvalidOperationException($"Secant iteration failed to converge; last value {p}.");
    }

    // ------------------------------------------------------------------ iirfilter

    /// <summary>
    /// IIR digital or analog filter design from order and critical frequencies
    /// (<c>scipy.signal.iirfilter</c>): prototype → frequency transform → bilinear (digital), all
    /// in zeros/poles/gain, converted to the requested output only at the end.
    /// </summary>
    /// <param name="n">Order.</param>
    /// <param name="wn">Critical frequency/frequencies. Digital: normalised to Nyquist (0..1) unless <paramref name="fs"/> is given. Analog: rad/s.</param>
    /// <param name="rp">Passband ripple in dB (Chebyshev I, elliptic).</param>
    /// <param name="rs">Stopband attenuation in dB (Chebyshev II, elliptic).</param>
    /// <param name="btype">Band type.</param>
    /// <param name="analog">Analog filter (s-domain) rather than digital.</param>
    /// <param name="ftype">Prototype family.</param>
    /// <param name="output">Representation of the result.</param>
    /// <param name="fs">Sampling frequency of a digital filter (same units as <paramref name="wn"/>).</param>
    /// <param name="besselNorm">Normalisation when <paramref name="ftype"/> is Bessel.</param>
    public static IirFilterResult Iirfilter(int n, CriticalFrequencies wn, double? rp = null, double? rs = null,
        FilterType btype = FilterType.Bandpass, bool analog = false, IirDesign ftype = IirDesign.Butter,
        FilterOutput output = FilterOutput.Ba, double? fs = null, BesselNorm besselNorm = BesselNorm.Phase)
    {
        var zpk = IirZpk(n, wn, rp, rs, btype, analog, ftype, fs, besselNorm);
        return output switch
        {
            FilterOutput.Zpk => new IirFilterResult(null, zpk, null),
            FilterOutput.Sos => new IirFilterResult(null, null, Zpk2Sos(zpk, analog: analog)),
            _ => new IirFilterResult(Zpk2Tf(zpk), null, null),
        };
    }

    private static FilterZpk IirZpk(int n, CriticalFrequencies wnSpec, double? rp, double? rs, FilterType btype, bool analog,
        IirDesign ftype, double? fs, BesselNorm besselNorm)
    {
        var wn = (wnSpec.Values ?? throw new ArgumentException("Wn is required.")).ToArray();
        if (wn.Length is < 1 or > 2) throw new ArgumentException("Wn must have one or two elements.");
        if (fs is double f)
        {
            if (analog) throw new ArgumentException("fs cannot be specified for an analog filter.", nameof(fs));
            if (!(f > 0) || double.IsInfinity(f)) throw new ArgumentException("fs must be a positive finite number.", nameof(fs));
            for (var i = 0; i < wn.Length; i++) wn[i] /= f / 2;
        }
        if (wn.Any(v => v <= 0)) throw new ArgumentException("Filter critical frequencies must be greater than 0.");
        if (wn.Length > 1 && !(wn[0] < wn[1])) throw new ArgumentException("Wn[0] must be less than Wn[1].");
        if (rp is < 0) throw new ArgumentException("Passband ripple (rp) must be positive.");
        if (rs is < 0) throw new ArgumentException("Stopband attenuation (rs) must be positive.");

        var proto = ftype switch
        {
            IirDesign.Butter => Buttap(n),
            IirDesign.Bessel => Besselap(n, besselNorm),
            IirDesign.Cheby1 => Cheb1ap(n, rp ?? throw new ArgumentException("Passband ripple (rp) must be provided to design a Chebyshev I filter.")),
            IirDesign.Cheby2 => Cheb2ap(n, rs ?? throw new ArgumentException("Stopband attenuation (rs) must be provided to design a Chebyshev II filter.")),
            IirDesign.Ellip => Ellipap(n, rp ?? throw new ArgumentException("Both rp and rs must be provided to design an elliptic filter."),
                rs ?? throw new ArgumentException("Both rp and rs must be provided to design an elliptic filter.")),
            _ => throw new ArgumentException($"'{ftype}' is not a valid basic IIR filter."),
        };

        double[] warped;
        const double fsd = 2.0;
        if (!analog)
        {
            if (wn.Any(v => v <= 0 || v >= 1))
                throw new ArgumentException(fs is null ? "Digital filter critical frequencies must be 0 < Wn < 1." : $"Digital filter critical frequencies must be 0 < Wn < fs/2 (fs={fs} -> fs/2={fs / 2}).");
            warped = wn.Select(v => 2 * fsd * Math.Tan(Math.PI * v / fsd)).ToArray();
        }
        else warped = wn;

        FilterZpk zpk;
        switch (btype)
        {
            case FilterType.Lowpass or FilterType.Highpass:
                if (wn.Length != 1) throw new ArgumentException("Must specify a single critical frequency Wn for lowpass or highpass filter.");
                zpk = btype == FilterType.Lowpass ? Lp2LpZpk(proto, warped[0]) : Lp2HpZpk(proto, warped[0]);
                break;
            default:
                if (wn.Length != 2) throw new ArgumentException("Wn must specify start and stop frequencies for bandpass or bandstop filter.");
                var bw = warped[1] - warped[0];
                var wo = Math.Sqrt(warped[0] * warped[1]);
                zpk = btype == FilterType.Bandpass ? Lp2BpZpk(proto, wo, bw) : Lp2BsZpk(proto, wo, bw);
                break;
        }
        if (!analog) zpk = BilinearZpk(zpk, fsd);
        return zpk;
    }

    // ------------------------------------------------------------------ named designs

    /// <summary>Butterworth filter as numerator/denominator (<c>scipy.signal.butter</c>, output='ba').</summary>
    /// <remarks>Above order ~8, and for narrow bands, prefer <see cref="ButterSos"/>: the ba form loses precision fast.</remarks>
    public static FilterBA Butter(int n, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => Zpk2Tf(IirZpk(n, wn, null, null, btype, analog, IirDesign.Butter, fs, default));

    /// <summary>Butterworth filter as zeros/poles/gain.</summary>
    public static FilterZpk ButterZpk(int n, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => IirZpk(n, wn, null, null, btype, analog, IirDesign.Butter, fs, default);

    /// <summary>Butterworth filter as second-order sections.</summary>
    public static NdArray ButterSos(int n, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => Zpk2Sos(ButterZpk(n, wn, btype, analog, fs), analog: analog);

    /// <summary>Chebyshev type I filter with <paramref name="rp"/> dB passband ripple (<c>cheby1</c>).</summary>
    public static FilterBA Cheby1(int n, double rp, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => Zpk2Tf(Cheby1Zpk(n, rp, wn, btype, analog, fs));

    /// <summary>Chebyshev type I filter as zeros/poles/gain.</summary>
    public static FilterZpk Cheby1Zpk(int n, double rp, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => IirZpk(n, wn, rp, null, btype, analog, IirDesign.Cheby1, fs, default);

    /// <summary>Chebyshev type I filter as second-order sections.</summary>
    public static NdArray Cheby1Sos(int n, double rp, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => Zpk2Sos(Cheby1Zpk(n, rp, wn, btype, analog, fs), analog: analog);

    /// <summary>Chebyshev type II filter with <paramref name="rs"/> dB stopband attenuation (<c>cheby2</c>).</summary>
    public static FilterBA Cheby2(int n, double rs, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => Zpk2Tf(Cheby2Zpk(n, rs, wn, btype, analog, fs));

    /// <summary>Chebyshev type II filter as zeros/poles/gain.</summary>
    public static FilterZpk Cheby2Zpk(int n, double rs, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => IirZpk(n, wn, null, rs, btype, analog, IirDesign.Cheby2, fs, default);

    /// <summary>Chebyshev type II filter as second-order sections.</summary>
    public static NdArray Cheby2Sos(int n, double rs, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => Zpk2Sos(Cheby2Zpk(n, rs, wn, btype, analog, fs), analog: analog);

    /// <summary>Elliptic (Cauer) filter (<c>ellip</c>).</summary>
    public static FilterBA Ellip(int n, double rp, double rs, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => Zpk2Tf(EllipZpk(n, rp, rs, wn, btype, analog, fs));

    /// <summary>Elliptic filter as zeros/poles/gain.</summary>
    public static FilterZpk EllipZpk(int n, double rp, double rs, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => IirZpk(n, wn, rp, rs, btype, analog, IirDesign.Ellip, fs, default);

    /// <summary>Elliptic filter as second-order sections.</summary>
    public static NdArray EllipSos(int n, double rp, double rs, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, double? fs = null)
        => Zpk2Sos(EllipZpk(n, rp, rs, wn, btype, analog, fs), analog: analog);

    /// <summary>Bessel/Thomson filter (<c>bessel</c>); <paramref name="norm"/> defaults to scipy's 'phase'.</summary>
    public static FilterBA Bessel(int n, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, BesselNorm norm = BesselNorm.Phase, double? fs = null)
        => Zpk2Tf(BesselZpk(n, wn, btype, analog, norm, fs));

    /// <summary>Bessel filter as zeros/poles/gain.</summary>
    public static FilterZpk BesselZpk(int n, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, BesselNorm norm = BesselNorm.Phase, double? fs = null)
        => IirZpk(n, wn, null, null, btype, analog, IirDesign.Bessel, fs, norm);

    /// <summary>Bessel filter as second-order sections.</summary>
    public static NdArray BesselSos(int n, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass, bool analog = false, BesselNorm norm = BesselNorm.Phase, double? fs = null)
        => Zpk2Sos(BesselZpk(n, wn, btype, analog, norm, fs), analog: analog);

    /// <summary>
    /// Complete IIR design from pass/stop band specifications (<c>scipy.signal.iirdesign</c>):
    /// the order comes from the matching <c>*ord</c> function, the band type from the edges.
    /// </summary>
    /// <param name="wp">Passband edge(s).</param>
    /// <param name="ws">Stopband edge(s).</param>
    /// <param name="gpass">Maximum passband loss (dB).</param>
    /// <param name="gstop">Minimum stopband attenuation (dB).</param>
    /// <param name="analog">Analog design.</param>
    /// <param name="ftype">Family; Bessel has no order selection and is rejected.</param>
    /// <param name="output">Representation.</param>
    /// <param name="fs">Sampling frequency.</param>
    public static IirFilterResult Iirdesign(CriticalFrequencies wp, CriticalFrequencies ws, double gpass, double gstop,
        bool analog = false, IirDesign ftype = IirDesign.Ellip, FilterOutput output = FilterOutput.Ba, double? fs = null)
    {
        if (ftype == IirDesign.Bessel) throw new ArgumentException("Bessel does not have order selection. Use Iirfilter.", nameof(ftype));
        ValidateGpassGstop(gpass, gstop);
        var p = wp.Values; var s = ws.Values;
        if (p.Length != s.Length || p.Length is < 1 or > 2) throw new ArgumentException("wp and ws must have one or two elements each, and the same shape.");
        if (p.Any(v => v <= 0) || s.Any(v => v <= 0)) throw new ArgumentException("Values for wp, ws must be greater than 0.");
        if (!analog)
        {
            var lim = fs is double f ? f / 2 : 1.0;
            if (p.Any(v => v >= lim) || s.Any(v => v >= lim)) throw new ArgumentException($"Values for wp, ws must be less than {lim}.");
        }
        if (p.Length == 2 && !((s[0] < p[0] && p[1] < s[1]) || (p[0] < s[0] && s[1] < p[1])))
            throw new ArgumentException("Passband must lie strictly inside stopband or vice versa.");
        var bandType = 2 * (p.Length - 1) + 1;
        if (p[0] >= s[0]) bandType++;
        var btype = bandType switch { 1 => FilterType.Lowpass, 2 => FilterType.Highpass, 3 => FilterType.Bandstop, _ => FilterType.Bandpass };
        var (n, wn) = ftype switch
        {
            IirDesign.Butter => Buttord(wp, ws, gpass, gstop, analog, fs),
            IirDesign.Cheby1 => Cheb1ord(wp, ws, gpass, gstop, analog, fs),
            IirDesign.Cheby2 => Cheb2ord(wp, ws, gpass, gstop, analog, fs),
            _ => Ellipord(wp, ws, gpass, gstop, analog, fs),
        };
        return Iirfilter(n, wn, gpass, gstop, btype, analog, ftype, output, fs);
    }

    // ------------------------------------------------------------------ notch / peak / comb / gammatone

    /// <summary>Second-order IIR notch at <paramref name="w0"/> with quality factor <paramref name="q"/> (<c>iirnotch</c>).</summary>
    public static FilterBA Iirnotch(double w0, double q, double fs = 2.0) => NotchPeak(w0, q, true, fs);

    /// <summary>Second-order IIR peak (resonator) at <paramref name="w0"/> (<c>iirpeak</c>).</summary>
    public static FilterBA Iirpeak(double w0, double q, double fs = 2.0) => NotchPeak(w0, q, false, fs);

    private static FilterBA NotchPeak(double w0, double q, bool notch, double fs)
    {
        if (!(fs > 0) || double.IsInfinity(fs)) throw new ArgumentException("fs must be a positive finite number.", nameof(fs));
        w0 = 2 * w0 / fs;
        if (w0 > 1.0 || w0 < 0.0) throw new ArgumentException("w0 should be such that 0 < w0 < 1.", nameof(w0));
        var bw = w0 / q * Math.PI;
        w0 *= Math.PI;
        var beta = Math.Tan(bw / 2.0);
        var gain = 1.0 / (1.0 + beta);
        double[] b = notch ? [gain * 1.0, gain * (-2.0 * Math.Cos(w0)), gain * 1.0] : [(1.0 - gain) * 1.0, 0.0, (1.0 - gain) * -1.0];
        double[] a = [1.0, -2.0 * gain * Math.Cos(w0), 2.0 * gain - 1.0];
        return new FilterBA(b, a);
    }

    /// <summary>
    /// IIR comb notch or peak filter (<c>iircomb</c>): notches/peaks at every multiple of
    /// <paramref name="w0"/>, which must divide <paramref name="fs"/>.
    /// </summary>
    /// <param name="w0">Fundamental frequency.</param>
    /// <param name="q">Quality factor.</param>
    /// <param name="peak">Peak filter rather than notch.</param>
    /// <param name="fs">Sampling frequency.</param>
    /// <param name="passZero">Whether the notch/peak is centred on DC as well.</param>
    public static FilterBA Iircomb(double w0, double q, bool peak = false, double fs = 2.0, bool passZero = false)
    {
        if (!(fs > 0)) throw new ArgumentException("fs must be positive.", nameof(fs));
        if (!(0 < w0 && w0 < fs / 2)) throw new ArgumentException($"w0 must be between 0 and {fs / 2} (Nyquist), but given {w0}.", nameof(w0));
        var n = (int)Math.Round(fs / w0, MidpointRounding.ToEven);
        if (Math.Abs(w0 - fs / n) / fs > 1e-14) throw new ArgumentException("fs must be divisible by w0.", nameof(fs));
        var w = 2 * Math.PI * w0 / fs;
        var wDelta = w / q;
        double g0 = peak ? 0 : 1, g = peak ? 1 : 0;
        var beta = Math.Tan(n * wDelta / 4);
        var ax = (1 - beta) / (1 + beta);
        var bx = (g0 + g * beta) / (1 + beta);
        var cx = (g0 - g * beta) / (1 + beta);
        var negative = (peak && passZero) || (!peak && !passZero);
        var sgn = negative ? -1.0 : 1.0;
        var b = new double[n + 1];
        var a = new double[n + 1];
        b[0] = bx; b[^1] = sgn * cx;
        a[0] = 1.0; a[^1] = sgn * ax;
        return new FilterBA(b, a);
    }

    /// <summary>
    /// Gammatone auditory filter (<c>scipy.signal.gammatone</c>): FIR (Slaney's sampled impulse
    /// response) or fourth-order IIR (Slaney's 1993 approximation).
    /// </summary>
    /// <param name="freq">Centre frequency.</param>
    /// <param name="fir">FIR (true) or IIR (false).</param>
    /// <param name="order">FIR order (default 4, at most 24).</param>
    /// <param name="numtaps">FIR length (default max(fs·0.015, 15)).</param>
    /// <param name="fs">Sampling frequency (default 2).</param>
    public static FilterBA Gammatone(double freq, bool fir, int? order = null, int? numtaps = null, double? fs = null)
    {
        var f = fs ?? 2.0;
        if (!(0 < freq && freq < f / 2)) throw new ArgumentException($"The frequency must be between 0 and {f / 2} (Nyquist), but given {freq}.", nameof(freq));
        static double HzToErb(double hz) => hz / 9.26449 + 24.7;
        if (fir)
        {
            var ord = order ?? 4;
            var nt = numtaps ?? Math.Max((int)(f * 0.015), 15);
            if (!(0 < ord && ord <= 24)) throw new ArgumentException("Invalid order: order must be > 0 and <= 24.", nameof(order));
            var bw = 1.019 * HzToErb(freq);
            var scale = 2 * Math.Pow(2 * Math.PI * bw, ord);
            double fact = 1;
            for (var i = 2; i <= ord - 1; i++) fact *= i;
            scale /= fact;
            scale /= f;
            var b = new double[nt];
            for (var i = 0; i < nt; i++)
            {
                var t = i / f;
                b[i] = Math.Pow(t, ord - 1) * Math.Exp(-2 * Math.PI * bw * t) * Math.Cos(2 * Math.PI * freq * t) * scale;
            }
            return new FilterBA(b, [1.0]);
        }
        var T = 1.0 / f;
        var bwr = 2 * Math.PI * 1.019 * HzToErb(freq);
        var fr = 2 * freq * Math.PI * T;
        var bwT = bwr * T;
        var g1 = -2 * Complex.Exp(new Complex(0, 2 * fr)) * T;
        var g2 = 2 * Complex.Exp(new Complex(-bwT, fr)) * T;
        var g3 = Math.Sqrt(3 + Math.Pow(2, 1.5)) * Math.Sin(fr);
        var g4 = Math.Sqrt(3 - Math.Pow(2, 1.5)) * Math.Sin(fr);
        var g5 = Complex.Exp(new Complex(0, 2 * fr));
        var gg = g1 + g2 * (Math.Cos(fr) - g4);
        gg *= g1 + g2 * (Math.Cos(fr) + g4);
        gg *= g1 + g2 * (Math.Cos(fr) - g3);
        gg *= g1 + g2 * (Math.Cos(fr) + g3);
        gg /= Complex.Pow(-2 / Math.Exp(2 * bwT) - 2 * g5 + 2 * (1 + g5) / Math.Exp(bwT), 4);
        var gm = Complex.Abs(gg);
        var T4 = Math.Pow(T, 4);
        double[] bb =
        [
            T4 / gm,
            -4 * T4 * Math.Cos(fr) / Math.Exp(bwr * T) / gm,
            6 * T4 * Math.Cos(2 * fr) / Math.Exp(2 * bwr * T) / gm,
            -4 * T4 * Math.Cos(3 * fr) / Math.Exp(3 * bwr * T) / gm,
            T4 * Math.Cos(4 * fr) / Math.Exp(4 * bwr * T) / gm,
        ];
        double[] aa =
        [
            1,
            -8 * Math.Cos(fr) / Math.Exp(bwr * T),
            4 * (4 + 3 * Math.Cos(2 * fr)) / Math.Exp(2 * bwr * T),
            -8 * (6 * Math.Cos(fr) + Math.Cos(3 * fr)) / Math.Exp(3 * bwr * T),
            2 * (18 + 16 * Math.Cos(2 * fr) + Math.Cos(4 * fr)) / Math.Exp(4 * bwr * T),
            -8 * (6 * Math.Cos(fr) + Math.Cos(3 * fr)) / Math.Exp(5 * bwr * T),
            4 * (4 + 3 * Math.Cos(2 * fr)) / Math.Exp(6 * bwr * T),
            -8 * Math.Cos(fr) / Math.Exp(7 * bwr * T),
            Math.Exp(-8 * bwr * T),
        ];
        return new FilterBA(bb, aa);
    }
}

/// <summary>A complex number with double-double real and imaginary parts (Dekker/Knuth error-free transforms).</summary>
internal readonly struct SignalDdComplex
{
    private readonly double _reHi, _reLo, _imHi, _imLo;

    public SignalDdComplex(double re, double im) { _reHi = re; _reLo = 0; _imHi = im; _imLo = 0; }

    private SignalDdComplex(double rh, double rl, double ih, double il) { _reHi = rh; _reLo = rl; _imHi = ih; _imLo = il; }

    /// <summary>Exact 1 + x for a double x (the constructor's rounding is replaced by an exact sum).</summary>
    public SignalDdComplex Fix(double one, Complex x)
    {
        var (h, l) = TwoSum(one, x.Real);
        return new SignalDdComplex(h, l, x.Imaginary, 0);
    }

    private static (double Hi, double Lo) TwoSum(double a, double b)
    {
        var s = a + b;
        var bb = s - a;
        return (s, (a - (s - bb)) + (b - bb));
    }

    private static (double Hi, double Lo) TwoProd(double a, double b)
    {
        var p = a * b;
        return (p, Math.FusedMultiplyAdd(a, b, -p));
    }

    private static (double Hi, double Lo) AddDd(double ah, double al, double bh, double bl)
    {
        var (s, e) = TwoSum(ah, bh);
        e += al + bl;
        return TwoSum(s, e);
    }

    private static (double Hi, double Lo) MulDd(double ah, double al, double b)
    {
        var (p, e) = TwoProd(ah, b);
        e += al * b;
        return TwoSum(p, e);
    }

    public SignalDdComplex Add(SignalDdComplex o)
    {
        var (rh, rl) = AddDd(_reHi, _reLo, o._reHi, o._reLo);
        var (ih, il) = AddDd(_imHi, _imLo, o._imHi, o._imLo);
        return new SignalDdComplex(rh, rl, ih, il);
    }

    public SignalDdComplex Scale(double c)
    {
        var (rh, rl) = MulDd(_reHi, _reLo, c);
        var (ih, il) = MulDd(_imHi, _imLo, c);
        return new SignalDdComplex(rh, rl, ih, il);
    }

    /// <summary>Product with an ordinary complex number.</summary>
    public SignalDdComplex Mul(Complex x)
    {
        var (a1h, a1l) = MulDd(_reHi, _reLo, x.Real);
        var (a2h, a2l) = MulDd(_imHi, _imLo, -x.Imaginary);
        var (b1h, b1l) = MulDd(_reHi, _reLo, x.Imaginary);
        var (b2h, b2l) = MulDd(_imHi, _imLo, x.Real);
        var (rh, rl) = AddDd(a1h, a1l, a2h, a2l);
        var (ih, il) = AddDd(b1h, b1l, b2h, b2l);
        return new SignalDdComplex(rh, rl, ih, il);
    }

    public Complex ToComplex() => new(_reHi + _reLo, _imHi + _imLo);
}
