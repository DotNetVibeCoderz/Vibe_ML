using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Signal
{
    /// <summary>numpy's <c>mod</c>: the result takes the sign of the divisor.</summary>
    private static double PyMod(double a, double b)
    {
        var m = a % b;
        if (m != 0 && (b < 0) != (m < 0)) m += b;
        return m;
    }

    /// <summary>
    /// Periodic sawtooth/triangle wave with period 2π (<c>scipy.signal.sawtooth</c>): rises from
    /// −1 to 1 over <c>width·2π</c>, then falls back. <paramref name="width"/> 0.5 gives a triangle;
    /// outside [0, 1] the output is NaN, as in scipy.
    /// </summary>
    public static double[] Sawtooth(ReadOnlySpan<double> t, double width = 1)
    {
        var y = new double[t.Length];
        for (var i = 0; i < t.Length; i++)
        {
            if (width > 1 || width < 0) { y[i] = double.NaN; continue; }
            var tm = PyMod(t[i], 2 * Math.PI);
            y[i] = tm < width * 2 * Math.PI ? tm / (Math.PI * width) - 1 : (Math.PI * (width + 1) - tm) / (Math.PI * (1 - width));
        }
        return y;
    }

    /// <summary>Periodic square wave with period 2π and the given duty cycle (<c>scipy.signal.square</c>); NaN outside [0, 1].</summary>
    public static double[] Square(ReadOnlySpan<double> t, double duty = 0.5)
    {
        var y = new double[t.Length];
        for (var i = 0; i < t.Length; i++)
        {
            if (duty > 1 || duty < 0) { y[i] = double.NaN; continue; }
            y[i] = PyMod(t[i], 2 * Math.PI) < duty * 2 * Math.PI ? 1 : -1;
        }
        return y;
    }

    private static double GaussPulseA(double fc, double bw, double bwr)
    {
        if (fc < 0) throw new ArgumentException($"Center frequency (fc={fc:F2}) must be >=0.", nameof(fc));
        if (bw <= 0) throw new ArgumentException($"Fractional bandwidth (bw={bw:F2}) must be > 0.", nameof(bw));
        if (bwr >= 0) throw new ArgumentException($"Reference level for bandwidth (bwr={bwr:F2}) must be < 0 dB.", nameof(bwr));
        var r = Math.Pow(10.0, bwr / 20.0);
        return -Math.Pow(Math.PI * fc * bw, 2) / (4.0 * Math.Log(r));
    }

    /// <summary>
    /// Gaussian-modulated sinusoid <c>exp(−a t²) cos(2π fc t)</c> (<c>scipy.signal.gausspulse</c>),
    /// with <c>a</c> set so the spectrum is <paramref name="bwr"/> dB down at the fractional
    /// bandwidth <paramref name="bw"/>.
    /// </summary>
    public static double[] GaussPulse(ReadOnlySpan<double> t, double fc = 1000, double bw = 0.5, double bwr = -6)
        => GaussPulseFull(t, fc, bw, bwr).I;

    /// <summary>The in-phase pulse, its quadrature (sine) partner and the envelope (<c>retquad=True, retenv=True</c>).</summary>
    public static (double[] I, double[] Q, double[] Envelope) GaussPulseFull(ReadOnlySpan<double> t, double fc = 1000, double bw = 0.5, double bwr = -6)
    {
        var a = GaussPulseA(fc, bw, bwr);
        var yi = new double[t.Length];
        var yq = new double[t.Length];
        var env = new double[t.Length];
        for (var i = 0; i < t.Length; i++)
        {
            var e = Math.Exp(-a * t[i] * t[i]);
            env[i] = e;
            yi[i] = e * Math.Cos(2 * Math.PI * fc * t[i]);
            yq[i] = e * Math.Sin(2 * Math.PI * fc * t[i]);
        }
        return (yi, yq, env);
    }

    /// <summary>Time at which the pulse envelope falls to <paramref name="tpr"/> dB (scipy's <c>gausspulse('cutoff', ...)</c>).</summary>
    public static double GaussPulseCutoff(double fc = 1000, double bw = 0.5, double bwr = -6, double tpr = -60)
    {
        var a = GaussPulseA(fc, bw, bwr);
        if (tpr >= 0) throw new ArgumentException("Reference level for time cutoff must be < 0 dB.", nameof(tpr));
        var tref = Math.Pow(10.0, tpr / 20.0);
        return Math.Sqrt(-Math.Log(tref) / a);
    }

    /// <summary>Instantaneous phase of a chirp (radians), without the phase offset.</summary>
    private static double ChirpPhase(double t, double f0, double t1, double f1, ChirpMethod method, bool vertexZero)
    {
        switch (method)
        {
            case ChirpMethod.Linear:
                {
                    var beta = (f1 - f0) / t1;
                    return 2 * Math.PI * (f0 * t + 0.5 * beta * t * t);
                }
            case ChirpMethod.Quadratic:
                {
                    var beta = (f1 - f0) / (t1 * t1);
                    return vertexZero
                        ? 2 * Math.PI * (f0 * t + beta * (t * t * t) / 3)
                        : 2 * Math.PI * (f1 * t + beta * (Math.Pow(t1 - t, 3) - Math.Pow(t1, 3)) / 3);
                }
            case ChirpMethod.Logarithmic:
                {
                    if (f0 * f1 <= 0.0) throw new ArgumentException("For a logarithmic chirp, f0 and f1 must be nonzero and have the same sign.");
                    if (f0 == f1) return 2 * Math.PI * f0 * t;
                    var beta = t1 / Math.Log(f1 / f0);
                    return 2 * Math.PI * beta * f0 * (Math.Pow(f1 / f0, t / t1) - 1.0);
                }
            default:
                {
                    if (f0 == 0 || f1 == 0) throw new ArgumentException("For a hyperbolic chirp, f0 and f1 must be nonzero.");
                    if (f0 == f1) return 2 * Math.PI * f0 * t;
                    var sing = -f1 * t1 / (f0 - f1);
                    return 2 * Math.PI * (-sing * f0) * Math.Log(Math.Abs(1 - t / sing));
                }
        }
    }

    /// <summary>
    /// Frequency-swept cosine (<c>scipy.signal.chirp</c>): frequency <paramref name="f0"/> at t = 0
    /// and <paramref name="f1"/> at <paramref name="t1"/>, following <paramref name="method"/>.
    /// </summary>
    /// <param name="t">Times.</param>
    /// <param name="f0">Frequency at time 0.</param>
    /// <param name="t1">Time at which <paramref name="f1"/> is reached.</param>
    /// <param name="f1">Frequency at <paramref name="t1"/>.</param>
    /// <param name="method">Sweep law.</param>
    /// <param name="phi">Phase offset in degrees.</param>
    /// <param name="vertexZero">Quadratic only: parabola vertex at t = 0 (true) or at t1.</param>
    public static double[] Chirp(ReadOnlySpan<double> t, double f0, double t1, double f1, ChirpMethod method = ChirpMethod.Linear, double phi = 0, bool vertexZero = true)
    {
        var y = new double[t.Length];
        var off = phi * Math.PI / 180;
        for (var i = 0; i < t.Length; i++) y[i] = Math.Cos(ChirpPhase(t[i], f0, t1, f1, method, vertexZero) + off);
        return y;
    }

    /// <summary>Complex (analytic) chirp <c>exp(iφ(t))</c> (scipy's <c>complex=True</c>).</summary>
    public static Complex[] ChirpComplex(ReadOnlySpan<double> t, double f0, double t1, double f1, ChirpMethod method = ChirpMethod.Linear, double phi = 0, bool vertexZero = true)
    {
        var y = new Complex[t.Length];
        var off = phi * Math.PI / 180;
        for (var i = 0; i < t.Length; i++) y[i] = Complex.FromPolarCoordinates(1.0, ChirpPhase(t[i], f0, t1, f1, method, vertexZero) + off);
        return y;
    }

    /// <summary>
    /// Cosine whose instantaneous frequency is the polynomial <paramref name="poly"/>(t)
    /// (highest power first) (<c>scipy.signal.sweep_poly</c>).
    /// </summary>
    public static double[] SweepPoly(ReadOnlySpan<double> t, double[] poly, double phi = 0)
    {
        ArgumentNullException.ThrowIfNull(poly);
        var n = poly.Length;
        var integ = new double[n + 1];
        for (var i = 0; i < n; i++) integ[i] = poly[i] / (n - i);
        var y = new double[t.Length];
        var off = phi * Math.PI / 180;
        for (var i = 0; i < t.Length; i++) y[i] = Math.Cos(2 * Math.PI * Polyval(integ, t[i]) + off);
        return y;
    }

    /// <summary>Unit impulse (Kronecker delta) of length <paramref name="n"/> at <paramref name="idx"/> (default 0).</summary>
    public static double[] UnitImpulse(int n, int idx = 0)
    {
        if (idx < 0) idx += n;
        if (idx < 0 || idx >= n) throw new ArgumentException("idx out of range.", nameof(idx));
        var y = new double[n];
        y[idx] = 1;
        return y;
    }

    /// <summary>n-D unit impulse; <paramref name="idx"/> null puts it at the origin, <paramref name="mid"/> at the centre (scipy's 'mid').</summary>
    public static NdArray UnitImpulse(int[] shape, int[]? idx = null, bool mid = false)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var y = NdArray.Zeros(shape);
        var pos = mid ? shape.Select(s => s / 2).ToArray() : idx ?? new int[shape.Length];
        if (pos.Length != shape.Length) throw new ArgumentException("idx must have one entry per dimension.", nameof(idx));
        var flat = 0;
        for (var i = 0; i < shape.Length; i++)
        {
            if (pos[i] < 0 || pos[i] >= shape[i]) throw new ArgumentException("idx out of range.", nameof(idx));
            flat = flat * shape[i] + pos[i];
        }
        y.AsSpan()[flat] = 1;
        return y;
    }

    private static readonly Dictionary<int, int[]> MlsTaps = new()
    {
        [2] = [1], [3] = [2], [4] = [3], [5] = [3], [6] = [5], [7] = [6], [8] = [7, 6, 1], [9] = [5], [10] = [7], [11] = [9],
        [12] = [11, 10, 4], [13] = [12, 11, 8], [14] = [13, 12, 2], [15] = [14], [16] = [15, 13, 4], [17] = [14], [18] = [11],
        [19] = [18, 17, 14], [20] = [17], [21] = [19], [22] = [21], [23] = [18], [24] = [23, 22, 17], [25] = [22],
        [26] = [25, 24, 20], [27] = [26, 25, 22], [28] = [25], [29] = [27], [30] = [29, 28, 7], [31] = [28], [32] = [31, 30, 10],
    };

    /// <summary>
    /// Maximum length sequence from a Fibonacci linear-feedback shift register
    /// (<c>scipy.signal.max_len_seq</c>): period <c>2^nbits − 1</c>, flat spectrum, ±1-correlation
    /// of −1/N off peak once mapped to ±1.
    /// </summary>
    /// <param name="nbits">Register length (2..32 with the built-in taps).</param>
    /// <param name="state">Initial state of 0/1 values (default all ones); not all zero.</param>
    /// <param name="length">Samples to produce (default the full period).</param>
    /// <param name="taps">Feedback taps (default scipy's primitive-polynomial table).</param>
    /// <returns>The 0/1 sequence and the final register state (ready to continue the sequence).</returns>
    public static (byte[] Seq, byte[] State) MaxLenSeq(int nbits, byte[]? state = null, long? length = null, int[]? taps = null)
    {
        int[] tp;
        if (taps is null)
        {
            if (!MlsTaps.TryGetValue(nbits, out var t0)) throw new ArgumentException("nbits must be between 2 and 32 if taps is null.", nameof(nbits));
            tp = t0;
        }
        else
        {
            tp = taps.Distinct().OrderByDescending(v => v).ToArray();
            if (tp.Length < 1 || tp.Any(v => v < 0 || v > nbits)) throw new ArgumentException("taps must be non-empty with values between zero and nbits (inclusive).", nameof(taps));
        }
        var nMax = (1L << nbits) - 1;
        var len = length ?? nMax;
        if (len < 0) throw new ArgumentException("length must be greater than or equal to 0.", nameof(length));
        if (len > int.MaxValue) throw new ArgumentException("length too large for a single array.", nameof(length));
        var st = state is null ? Enumerable.Repeat((byte)1, nbits).ToArray() : state.Select(v => v != 0 ? (byte)1 : (byte)0).ToArray();
        if (st.Length != nbits) throw new ArgumentException("state must be a 1-D array of size nbits.", nameof(state));
        if (st.All(v => v == 0)) throw new ArgumentException("state must not be all zeros.", nameof(state));
        var seq = new byte[len];
        var idx = 0;
        for (var i = 0; i < len; i++)
        {
            var feedback = st[idx];
            seq[i] = feedback;
            foreach (var tap in tp) feedback ^= st[(tap + idx) % nbits];
            st[idx] = feedback;
            idx = (idx + 1) % nbits;
        }
        var rolled = new byte[nbits];
        for (var i = 0; i < nbits; i++) rolled[i] = st[(i + idx) % nbits];
        return (seq, rolled);
    }
}
