using System.Numerics;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    // ------------------------------------------------------------------ Kaiser formulas

    /// <summary>Kaiser β for a stopband attenuation of <paramref name="a"/> dB (Kaiser's empirical formula).</summary>
    public static double KaiserBeta(double a)
    {
        if (a > 50) return 0.1102 * (a - 8.7);
        if (a > 21) return 0.5842 * Math.Pow(a - 21, 0.4) + 0.07886 * (a - 21);
        return 0.0;
    }

    /// <summary>Attenuation (dB) of a Kaiser FIR with <paramref name="numtaps"/> taps and transition <paramref name="width"/> (fraction of Nyquist).</summary>
    public static double KaiserAtten(int numtaps, double width) => 2.285 * (numtaps - 1) * Math.PI * width + 7.95;

    /// <summary>
    /// Kaiser window design parameters: taps and β for <paramref name="ripple"/> dB of ripple and
    /// a transition of <paramref name="width"/> (fraction of Nyquist) (<c>kaiserord</c>).
    /// </summary>
    public static (int Numtaps, double Beta) Kaiserord(double ripple, double width)
    {
        var a = Math.Abs(ripple);
        if (a < 8) throw new ArgumentException($"Requested maximum ripple attenuation {a} is too small for the Kaiser formula.", nameof(ripple));
        var beta = KaiserBeta(a);
        var numtaps = (a - 7.95) / 2.285 / (Math.PI * width) + 1;
        return ((int)Math.Ceiling(numtaps), beta);
    }

    // ------------------------------------------------------------------ firwin

    /// <summary>
    /// FIR filter by the window method (<c>scipy.signal.firwin</c>), with <paramref name="passZero"/>
    /// deciding whether DC is in the passband (scipy's boolean form).
    /// </summary>
    /// <param name="numtaps">Filter length; must be odd if the filter passes Nyquist.</param>
    /// <param name="cutoff">Band edges, strictly increasing, in (0, fs/2).</param>
    /// <param name="width">If set, design a Kaiser window for this transition width (overrides <paramref name="window"/>).</param>
    /// <param name="window">Window; default Hamming.</param>
    /// <param name="passZero">Whether the gain at DC is one.</param>
    /// <param name="scale">Scale so the response at the centre of the first passband is exactly one.</param>
    /// <param name="fs">Sampling frequency; default 2 (cutoffs relative to Nyquist = 1).</param>
    public static double[] Firwin(int numtaps, CriticalFrequencies cutoff, double? width = null, WindowSpec? window = null,
        bool passZero = true, bool scale = true, double? fs = null)
        => FirwinCore(numtaps, cutoff.Values, width, window, passZero, scale, fs);

    /// <summary>
    /// <see cref="Firwin(int, CriticalFrequencies, double?, WindowSpec?, bool, bool, double?)"/> with the
    /// band type spelled out (scipy's string <c>pass_zero</c>), which also validates the cutoff count.
    /// </summary>
    public static double[] Firwin(int numtaps, CriticalFrequencies cutoff, FilterType btype, double? width = null, WindowSpec? window = null,
        bool scale = true, double? fs = null)
    {
        var n = cutoff.Length;
        switch (btype)
        {
            case FilterType.Lowpass or FilterType.Highpass when n != 1:
                throw new ArgumentException($"cutoff must have one element for {btype}.", nameof(cutoff));
            case FilterType.Bandpass or FilterType.Bandstop when n <= 1:
                throw new ArgumentException($"cutoff must have at least two elements for {btype}.", nameof(cutoff));
        }
        var pz = btype is FilterType.Lowpass or FilterType.Bandstop;
        return FirwinCore(numtaps, cutoff.Values, width, window, pz, scale, fs);
    }

    private static double[] FirwinCore(int numtaps, double[] cutoffIn, double? width, WindowSpec? window, bool passZero, bool scale, double? fs)
    {
        var f = fs ?? 2.0;
        var nyq = 0.5 * f;
        if (cutoffIn is null || cutoffIn.Length == 0) throw new ArgumentException("At least one cutoff frequency must be given.");
        var cutoff = cutoffIn.Select(c => c / nyq).ToArray();
        if (cutoff.Min() <= 0 || cutoff.Max() >= 1)
            throw new ArgumentException("Invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2.");
        for (var i = 1; i < cutoff.Length; i++)
            if (cutoff[i] - cutoff[i - 1] <= 0) throw new ArgumentException("Invalid cutoff frequencies: the frequencies must be strictly increasing.");
        var win = window ?? "hamming";
        if (width is double w) win = WindowSpec.Kaiser(KaiserBeta(KaiserAtten(numtaps, w / nyq)));

        var passNyquist = (cutoff.Length % 2 == 0) == passZero;
        if (passNyquist && numtaps % 2 == 0)
            throw new ArgumentException("A filter with an even number of coefficients must have zero response at the Nyquist frequency.");
        var edges = new List<double>();
        if (passZero) edges.Add(0.0);
        edges.AddRange(cutoff);
        if (passNyquist) edges.Add(1.0);

        var alpha = 0.5 * (numtaps - 1);
        var h = new double[numtaps];
        for (var j = 0; j < edges.Count; j += 2)
        {
            double left = edges[j], right = edges[j + 1];
            for (var i = 0; i < numtaps; i++)
            {
                var m = i - alpha;
                h[i] += right * Sinc(right * m);
                h[i] -= left * Sinc(left * m);
            }
        }
        var wv = GetWindow(win, numtaps, fftbins: false);
        for (var i = 0; i < numtaps; i++) h[i] *= wv[i];
        if (scale)
        {
            double l0 = edges[0], r0 = edges[1];
            var sf = l0 == 0 ? 0.0 : r0 == 1 ? 1.0 : 0.5 * (l0 + r0);
            var s = 0.0;
            for (var i = 0; i < numtaps; i++) s += h[i] * Math.Cos(Math.PI * (i - alpha) * sf);
            for (var i = 0; i < numtaps; i++) h[i] /= s;
        }
        return h;
    }

    // ------------------------------------------------------------------ firwin2

    /// <summary>
    /// FIR filter by frequency sampling with a window (<c>scipy.signal.firwin2</c>): the gain
    /// curve is interpolated onto a dense grid, inverse-FFT'd, truncated and windowed.
    /// </summary>
    /// <param name="numtaps">Filter length.</param>
    /// <param name="freq">Frequency points, from 0 to fs/2, nondecreasing; a value may repeat once to make a step.</param>
    /// <param name="gain">Gain at each point.</param>
    /// <param name="nfreqs">Interpolation grid size; default <c>1 + 2^ceil(log2 numtaps)</c>.</param>
    /// <param name="window">Window; default Hamming; pass <c>null</c> with <paramref name="noWindow"/> for none.</param>
    /// <param name="antisymmetric">Type III/IV (odd symmetry) rather than I/II.</param>
    /// <param name="fs">Sampling frequency (default 2).</param>
    /// <param name="noWindow">Skip windowing (scipy's <c>window=None</c>).</param>
    public static double[] Firwin2(int numtaps, double[] freq, double[] gain, int? nfreqs = null, WindowSpec? window = null,
        bool antisymmetric = false, double? fs = null, bool noWindow = false)
    {
        ArgumentNullException.ThrowIfNull(freq);
        ArgumentNullException.ThrowIfNull(gain);
        var nyq = 0.5 * (fs ?? 2.0);
        if (freq.Length != gain.Length) throw new ArgumentException("freq and gain must be of same length.");
        if (nfreqs is int nf0 && numtaps >= nf0) throw new ArgumentException($"ntaps must be less than nfreqs, but firwin2 was called with ntaps={numtaps} and nfreqs={nf0}.");
        if (freq[0] != 0 || freq[^1] != nyq) throw new ArgumentException("freq must start with 0 and end with fs/2.");
        var f = freq.ToArray();
        var anyEqual = false;
        for (var i = 0; i < f.Length - 1; i++)
        {
            if (f[i + 1] - f[i] < 0) throw new ArgumentException("The values in freq must be nondecreasing.");
            if (f[i + 1] == f[i]) anyEqual = true;
            if (i < f.Length - 2 && f[i + 2] - f[i] == 0) throw new ArgumentException("A value in freq must not occur more than twice.");
        }
        if (f[1] == 0) throw new ArgumentException("Value 0 must not be repeated in freq.");
        if (f[^2] == nyq) throw new ArgumentException("Value fs/2 must not be repeated in freq.");
        int ftype = antisymmetric ? (numtaps % 2 == 0 ? 4 : 3) : (numtaps % 2 == 0 ? 2 : 1);
        if (ftype == 2 && gain[^1] != 0.0) throw new ArgumentException("A Type II filter must have zero gain at the Nyquist frequency.");
        if (ftype == 3 && (gain[0] != 0.0 || gain[^1] != 0.0)) throw new ArgumentException("A Type III filter must have zero gain at zero and Nyquist frequencies.");
        if (ftype == 4 && gain[0] != 0.0) throw new ArgumentException("A Type IV filter must have zero gain at zero frequency.");
        var nfreq = nfreqs ?? 1 + (1 << (int)Math.Ceiling(Math.Log2(numtaps)));
        if (anyEqual)
        {
            var eps = 2.220446049250313e-16 * nyq;
            for (var k = 0; k < f.Length - 1; k++)
                if (f[k] == f[k + 1]) { f[k] -= eps; f[k + 1] += eps; }
            for (var k = 0; k < f.Length - 1; k++)
                if (f[k + 1] - f[k] <= 0) throw new ArgumentException($"freq cannot contain numbers that are too close (within eps * (fs/2): {eps}) to a repeated value.");
        }
        var x = Linspace(0.0, nyq, nfreq);
        var spec = new Complex[nfreq];
        for (var i = 0; i < nfreq; i++)
        {
            var fx = Interp(x[i], f, gain);
            var shift = Complex.Exp(-(numtaps - 1) / 2.0 * Complex.ImaginaryOne * Math.PI * x[i] / nyq);
            if (ftype > 2) shift *= Complex.ImaginaryOne;
            spec[i] = fx * shift;
        }
        var full = Gravicode.Science.GraviNum.Signal.Fft.InverseReal(spec, 2 * (nfreq - 1));
        var outv = full[..numtaps];
        if (!noWindow)
        {
            var wv = GetWindow(window ?? "hamming", numtaps, fftbins: false);
            for (var i = 0; i < numtaps; i++) outv[i] *= wv[i];
        }
        if (ftype == 3) outv[numtaps / 2] = 0.0;
        return outv;
    }

    /// <summary>numpy's <c>interp</c> for increasing <paramref name="xp"/>, clamped at the ends.</summary>
    internal static double Interp(double x, double[] xp, double[] fp)
    {
        if (x <= xp[0]) return fp[0];
        if (x >= xp[^1]) return fp[^1];
        var lo = 0;
        var hi = xp.Length - 1;
        while (hi - lo > 1)
        {
            var mid = (lo + hi) >> 1;
            if (xp[mid] <= x) lo = mid; else hi = mid;
        }
        var slope = (fp[hi] - fp[lo]) / (xp[hi] - xp[lo]);
        return slope * (x - xp[lo]) + fp[lo];
    }

    // ------------------------------------------------------------------ firls

    /// <summary>
    /// Linear-phase FIR minimising the weighted integral squared error to a piecewise-linear
    /// desired response (<c>scipy.signal.firls</c>). <paramref name="numtaps"/> must be odd.
    /// </summary>
    /// <param name="numtaps">Odd filter length.</param>
    /// <param name="bands">Band edge pairs, flattened, monotonic, in [0, fs/2].</param>
    /// <param name="desired">Desired gain at each band edge (same length as <paramref name="bands"/>).</param>
    /// <param name="weight">One weight per band (default ones).</param>
    /// <param name="fs">Sampling frequency (default 2).</param>
    public static double[] Firls(int numtaps, double[] bands, double[] desired, double[]? weight = null, double? fs = null)
    {
        ArgumentNullException.ThrowIfNull(bands);
        ArgumentNullException.ThrowIfNull(desired);
        var nyq = 0.5 * (fs ?? 2.0);
        if (numtaps % 2 == 0 || numtaps < 1) throw new ArgumentException("numtaps must be odd and >= 1.", nameof(numtaps));
        if (nyq <= 0) throw new ArgumentException("nyq must be positive.");
        var m = (numtaps - 1) / 2;
        if (bands.Length % 2 != 0) throw new ArgumentException("bands must contain frequency pairs.", nameof(bands));
        var bd = bands.Select(v => v / nyq).ToArray();
        if (bd.Any(v => v < 0 || v > 1)) throw new ArgumentException("bands must be between 0 and 1 relative to Nyquist.");
        if (desired.Length != bd.Length) throw new ArgumentException($"desired must have one entry per frequency, got {desired.Length} gains for {bd.Length} frequencies.");
        var nb = bd.Length / 2;
        for (var i = 0; i < nb; i++)
        {
            if (bd[2 * i + 1] - bd[2 * i] <= 0) throw new ArgumentException("bands must be monotonically nondecreasing and have width > 0.");
            if (i > 0 && bd[2 * i] < bd[2 * i - 2]) throw new ArgumentException("bands must be monotonically nondecreasing and have width > 0.");
            if (i > 0 && bd[2 * i - 1] > bd[2 * i]) throw new ArgumentException("bands must not overlap.");
        }
        if (desired.Any(v => v < 0)) throw new ArgumentException("desired must be non-negative.");
        var wt = weight ?? Enumerable.Repeat(1.0, nb).ToArray();
        if (wt.Length != nb) throw new ArgumentException($"weight must be the same size as the number of band pairs ({nb}).");
        if (wt.Any(v => v < 0)) throw new ArgumentException("weight must be non-negative.");

        // q[n] = Σ_bands w (b1 sinc(b1 n) − b0 sinc(b0 n))
        var q = new double[numtaps];
        for (var n = 0; n < numtaps; n++)
        {
            var s = 0.0;
            for (var k = 0; k < nb; k++)
            {
                double b0 = bd[2 * k], b1 = bd[2 * k + 1];
                s += (Sinc(b1 * n) * b1 - Sinc(b0 * n) * b0) * wt[k];
            }
            q[n] = s;
        }
        var size = m + 1;
        var qm = new double[size * size];
        for (var i = 0; i < size; i++)
            for (var j = 0; j < size; j++)
                qm[i * size + j] = q[Math.Abs(i - j)] + q[i + j];

        var rhs = new double[size];
        for (var n = 0; n < size; n++)
        {
            var s = 0.0;
            for (var k = 0; k < nb; k++)
            {
                double b0 = bd[2 * k], b1 = bd[2 * k + 1];
                double d0 = desired[2 * k], d1 = desired[2 * k + 1];
                var slope = (d1 - d0) / (b1 - b0);
                var c = d0 - b0 * slope;
                double Term(double b)
                {
                    var t = b * (slope * b + c) * Sinc(b * n);
                    if (n == 0) t -= slope * b * b / 2.0;
                    else t += slope * Math.Cos(n * Math.PI * b) / Math.Pow(Math.PI * n, 2);
                    return t;
                }
                s += (Term(b1) - Term(b0)) * wt[k];
            }
            rhs[n] = s;
        }
        var a = SolveSpd(qm, rhs, size) ?? SolveLeastSquares(qm, rhs, size);
        var coeffs = new double[numtaps];
        for (var i = 1; i <= m; i++) { coeffs[m - i] = a[i]; coeffs[m + i] = a[i]; }
        coeffs[m] = 2 * a[0];
        return coeffs;
    }

    /// <summary>Cholesky solve; null when the matrix is not (numerically) positive definite or is ill-conditioned.</summary>
    private static double[]? SolveSpd(double[] a, double[] b, int n)
    {
        var l = new double[n * n];
        double dmax = 0, dmin = double.PositiveInfinity;
        for (var j = 0; j < n; j++)
        {
            var s = a[j * n + j];
            for (var k = 0; k < j; k++) s -= l[j * n + k] * l[j * n + k];
            if (!(s > 0)) return null;
            var d = Math.Sqrt(s);
            l[j * n + j] = d;
            dmax = Math.Max(dmax, d); dmin = Math.Min(dmin, d);
            for (var i = j + 1; i < n; i++)
            {
                var t = a[i * n + j];
                for (var k = 0; k < j; k++) t -= l[i * n + k] * l[j * n + k];
                l[i * n + j] = t / d;
            }
        }
        // scipy falls back to least squares on an "ill-conditioned" warning (rcond < eps).
        if (dmin / dmax < Math.Sqrt(2.220446049250313e-16)) return null;
        var y = new double[n];
        for (var i = 0; i < n; i++)
        {
            var s = b[i];
            for (var k = 0; k < i; k++) s -= l[i * n + k] * y[k];
            y[i] = s / l[i * n + i];
        }
        var x = new double[n];
        for (var i = n - 1; i >= 0; i--)
        {
            var s = y[i];
            for (var k = i + 1; k < n; k++) s -= l[k * n + i] * x[k];
            x[i] = s / l[i * n + i];
        }
        return x;
    }

    /// <summary>Minimum-norm least squares through GraviNum's SVD (the fallback scipy reaches via <c>gelsy</c>).</summary>
    private static double[] SolveLeastSquares(double[] a, double[] b, int n)
    {
        var svd = Gravicode.Science.GraviNum.Decomposition.Svd(new Gravicode.Science.GraviNum.NdArray(a.ToArray(), n, n));
        var u = svd.U.To2DArray();
        var s = svd.SingularValues.ToArray();
        var v = svd.V.To2DArray();
        var tol = s.Max() * n * 2.220446049250313e-16;
        var x = new double[n];
        for (var k = 0; k < s.Length; k++)
        {
            if (s[k] <= tol) continue;
            var c = 0.0;
            for (var i = 0; i < n; i++) c += u[i, k] * b[i];
            c /= s[k];
            for (var i = 0; i < n; i++) x[i] += c * v[i, k];
        }
        return x;
    }

    // ------------------------------------------------------------------ minimum phase

    /// <summary>
    /// Converts a linear-phase FIR to minimum phase (<c>scipy.signal.minimum_phase</c>).
    /// </summary>
    /// <param name="h">Symmetric linear-phase filter, at least 3 taps.</param>
    /// <param name="method">Homomorphic (default) or Hilbert.</param>
    /// <param name="nFft">FFT length; default <c>2^ceil(log2(2(len−1)/0.01))</c>.</param>
    /// <param name="half">Homomorphic only: return the half-length filter whose magnitude is the square root of the original (scipy's default); false keeps the full magnitude.</param>
    public static double[] MinimumPhase(double[] h, MinimumPhaseMethod method = MinimumPhaseMethod.Homomorphic, int? nFft = null, bool half = true)
    {
        ArgumentNullException.ThrowIfNull(h);
        if (h.Length <= 2) throw new ArgumentException("h must be 1-D and at least 2 samples long.", nameof(h));
        if (method == MinimumPhaseMethod.Hilbert && !half) throw new ArgumentException("half=false is only supported for the homomorphic method.");
        var nHalf = h.Length / 2;
        var n = nFft ?? (1 << (int)Math.Ceiling(Math.Log2(2 * (h.Length - 1) / 0.01)));
        if (n < h.Length) throw new ArgumentException($"n_fft must be at least len(h)=={h.Length}.");
        var buf = new Complex[n];
        for (var i = 0; i < h.Length; i++) buf[i] = h[i];
        var hf = SignalDft(buf);
        double[] hMin;
        if (method == MinimumPhaseMethod.Hilbert)
        {
            var hr = new double[n];
            for (var i = 0; i < n; i++)
            {
                var w = i * (2 * Math.PI / n * nHalf);
                hr[i] = (hf[i] * Complex.Exp(new Complex(0, w))).Real;
            }
            var dp = hr.Max() - 1;
            var ds = 0 - hr.Min();
            var sc = 4.0 / Math.Pow(Math.Sqrt(1 + dp + ds) + Math.Sqrt(1 - dp + ds), 2);
            var mag = new Complex[n];
            for (var i = 0; i < n; i++) mag[i] = Math.Sqrt((hr[i] + ds) * sc) + 1e-10;
            // discrete Hilbert transform method
            var sig = new double[n];
            var mid = n / 2;
            for (var i = 1; i < mid; i++) sig[i] = 1;
            for (var i = mid + 1; i < n; i++) sig[i] = -1;
            var logm = SignalIdft(mag.Select(v => Complex.Log(v)).ToArray());
            for (var i = 0; i < n; i++) logm[i] *= sig[i];
            var ph = SignalDft(logm);
            var rec = new Complex[n];
            for (var i = 0; i < n; i++) rec[i] = mag[i] * Complex.Exp(ph[i]);
            hMin = SignalIdft(rec).Select(c => c.Real).ToArray();
        }
        else
        {
            var mag = hf.Select(Complex.Abs).ToArray();
            var minPos = mag.Where(v => v > 0).DefaultIfEmpty(0).Min();
            var lg = new Complex[n];
            for (var i = 0; i < n; i++)
            {
                var v = Math.Log(mag[i] + 1e-7 * minPos);
                lg[i] = half ? v * 0.5 : v;
            }
            var cep = SignalIdft(lg).Select(c => c.Real).ToArray();
            var stop = n / 2;
            var fold = new Complex[n];
            fold[0] = cep[0];
            for (var i = 1; i < stop; i++) fold[i] = 2 * cep[i];
            if (n % 2 == 1) fold[stop] = cep[stop];
            var e = SignalDft(fold);
            for (var i = 0; i < n; i++) e[i] = Complex.Exp(e[i]);
            hMin = SignalIdft(e).Select(c => c.Real).ToArray();
        }
        var nOut = half ? nHalf + h.Length % 2 : h.Length;
        return hMin[..nOut];
    }
}
