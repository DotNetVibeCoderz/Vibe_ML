using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Time response of an LTI simulation (<c>lsim</c>/<c>dlsim</c>).</summary>
/// <param name="T">Time points.</param>
/// <param name="Yout">Outputs, one row per time point and one column per output.</param>
/// <param name="Xout">States, one row per time point and one column per state.</param>
public sealed record LtiResponse(double[] T, NdArray Yout, NdArray Xout)
{
    /// <summary>The first (for a SISO system, the only) output as a vector.</summary>
    public double[] Y
    {
        get
        {
            var rows = Yout.Shape[0];
            var cols = Yout.Shape[1];
            var src = Yout.AsContiguous().AsSpan();
            var y = new double[rows];
            for (var i = 0; i < rows; i++) y[i] = src[i * cols];
            return y;
        }
    }
}

public static partial class Signal
{
    // ------------------------------------------------------------------ small dense helpers

    private static double[] MatMul(ReadOnlySpan<double> a, int ar, int ac, ReadOnlySpan<double> b, int bc)
    {
        var r = new double[ar * bc];
        for (var i = 0; i < ar; i++)
            for (var k = 0; k < ac; k++)
            {
                var v = a[i * ac + k];
                if (v == 0) continue;
                for (var j = 0; j < bc; j++) r[i * bc + j] += v * b[k * bc + j];
            }
        return r;
    }

    private static double[] Flat(NdArray m) => m.AsContiguous().ToArray();

    private static double[] Eye(int n)
    {
        var e = new double[n * n];
        for (var i = 0; i < n; i++) e[i * n + i] = 1;
        return e;
    }

    // ------------------------------------------------------------------ representation conversions

    /// <summary>
    /// Controller canonical state-space realisation of a transfer function
    /// (<c>scipy.signal.tf2ss</c>): <c>A</c>'s first row is <c>−den[1:]</c> with ones below the
    /// diagonal, <c>B = e₁</c>, and <c>C = num[1:] − num[0]·den[1:]</c>.
    /// </summary>
    public static (NdArray A, NdArray B, NdArray C, NdArray D) Tf2Ss(double[] num, double[] den)
    {
        var (b, a) = Normalize(num, den);
        var m = b.Length;
        var k = a.Length;
        if (m > k) throw new ArgumentException("Improper transfer function. `num` is longer than `den`.");
        var nb = new double[k];
        Array.Copy(b, 0, nb, k - m, m);
        var d = new NdArray([nb[0]], 1, 1);
        if (k == 1) return (NdArray.Zeros(1, 1), NdArray.Zeros(1, 1), NdArray.Zeros(1, 1), d);
        var n = k - 1;
        var amat = new double[n * n];
        for (var j = 0; j < n; j++) amat[j] = -a[j + 1];
        for (var i = 1; i < n; i++) amat[i * n + i - 1] = 1.0;
        var bmat = new double[n];
        bmat[0] = 1.0;
        var cmat = new double[n];
        for (var j = 0; j < n; j++) cmat[j] = nb[j + 1] - nb[0] * a[j + 1];
        return (new NdArray(amat, n, n), new NdArray(bmat, n, 1), new NdArray(cmat, 1, n), d);
    }

    /// <summary>
    /// Transfer function of one input of a state-space system (<c>scipy.signal.ss2tf</c>):
    /// <c>den = det(sI − A)</c> and, per output k, <c>num_k = det(sI − A + B C_k) + (D_k − 1)·den</c>.
    /// </summary>
    /// <returns>Numerators (one row per output) and the common denominator.</returns>
    public static (NdArray Num, double[] Den) Ss2Tf(NdArray a, NdArray b, NdArray c, NdArray d, int input = 0)
    {
        var sys = new StateSpace(a, b, c, d);
        var n = sys.States;
        var nout = sys.Outputs;
        if (input >= sys.Inputs) throw new ArgumentException("System does not have the input specified.", nameof(input));
        var af = Flat(sys.A);
        var bf = Flat(sys.B);
        var cf = Flat(sys.C);
        var df = Flat(sys.D);
        var nin = sys.Inputs;
        var den = n == 0 ? [1.0] : CharPoly(af, n);
        var num = new double[nout * (n + 1)];
        for (var k = 0; k < nout; k++)
        {
            var m = new double[n * n];
            for (var i = 0; i < n; i++)
                for (var j = 0; j < n; j++)
                    m[i * n + j] = af[i * n + j] - bf[i * nin + input] * cf[k * n + j];
            var pk = n == 0 ? [1.0] : CharPoly(m, n);
            var dk = df[k * nin + input];
            for (var i = 0; i <= n; i++) num[k * (n + 1) + i] = pk[i] + (dk - 1) * den[i];
        }
        return (new NdArray(num, nout, n + 1), den);
    }

    /// <summary>Characteristic polynomial as numpy's <c>poly(A)</c>: from the eigenvalues.</summary>
    private static double[] CharPoly(double[] a, int n)
    {
        var ev = Linalg.Eigvals(new NdArray(a.ToArray(), n, n));
        var c = PolyComplex(ev);
        return c.Select(v => v.Real).ToArray();
    }

    /// <summary>State-space realisation of a zpk system (<c>zpk2ss</c>), via the transfer function.</summary>
    public static (NdArray A, NdArray B, NdArray C, NdArray D) Zpk2Ss(Complex[] z, Complex[] p, double k)
    {
        var (b, a) = Zpk2Tf(z, p, k);
        return Tf2Ss(b, a);
    }

    /// <summary>Zeros, poles and gain of one input of a state-space system (<c>ss2zpk</c>).</summary>
    public static FilterZpk Ss2Zpk(NdArray a, NdArray b, NdArray c, NdArray d, int input = 0)
    {
        var (num, den) = Ss2Tf(a, b, c, d, input);
        if (num.Shape[0] != 1) throw new ArgumentException("ss2zpk requires a single-output system.");
        return Tf2Zpk(num.ToArray(), den);
    }

    // ------------------------------------------------------------------ cont2discrete

    /// <summary>
    /// Discretises a continuous state-space system (<c>scipy.signal.cont2discrete</c>). ZOH and FOH
    /// are exact for piecewise-constant / piecewise-linear inputs, computed from one matrix
    /// exponential of an augmented matrix (Van Loan 1978); the gbt family solves with
    /// <c>I − α·dt·A</c>.
    /// </summary>
    /// <param name="system">Continuous system.</param>
    /// <param name="dt">Sample time.</param>
    /// <param name="method">Discretisation rule.</param>
    /// <param name="alpha">α for <see cref="DiscretizationMethod.Gbt"/>, in [0, 1].</param>
    public static StateSpace Cont2Discrete(StateSpace system, double dt, DiscretizationMethod method = DiscretizationMethod.Zoh, double alpha = double.NaN)
    {
        ArgumentNullException.ThrowIfNull(system);
        if (!(dt > 0)) throw new ArgumentException("dt must be positive.", nameof(dt));
        switch (method)
        {
            case DiscretizationMethod.Bilinear: return Gbt(system, dt, 0.5);
            case DiscretizationMethod.Euler: return Gbt(system, dt, 0.0);
            case DiscretizationMethod.BackwardDiff: return Gbt(system, dt, 1.0);
            case DiscretizationMethod.Gbt:
                if (double.IsNaN(alpha)) throw new ArgumentException("Alpha parameter must be specified for the generalized bilinear transform (gbt) method.", nameof(alpha));
                if (alpha < 0 || alpha > 1) throw new ArgumentException("Alpha parameter must be within the interval [0,1] for the gbt method.", nameof(alpha));
                return Gbt(system, dt, alpha);
        }
        var n = system.States;
        var m = system.Inputs;
        var q = system.Outputs;
        var a = Flat(system.A);
        var b = Flat(system.B);
        var c = Flat(system.C);
        var d = Flat(system.D);
        switch (method)
        {
            case DiscretizationMethod.Zoh:
                {
                    var s = n + m;
                    var em = new double[s * s];
                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++) em[i * s + j] = dt * a[i * n + j];
                        for (var j = 0; j < m; j++) em[i * s + n + j] = dt * b[i * m + j];
                    }
                    var ms = Flat(Linalg.Expm(new NdArray(em, s, s)));
                    var ad = new double[n * n];
                    var bd = new double[n * m];
                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++) ad[i * n + j] = ms[i * s + j];
                        for (var j = 0; j < m; j++) bd[i * m + j] = ms[i * s + n + j];
                    }
                    return new StateSpace(new NdArray(ad, n, n), new NdArray(bd, n, m), system.C, system.D, dt);
                }
            case DiscretizationMethod.Foh:
                {
                    var s = n + 2 * m;
                    var em = new double[s * s];
                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++) em[i * s + j] = a[i * n + j] * dt;
                        for (var j = 0; j < m; j++) em[i * s + n + j] = b[i * m + j] * dt;
                    }
                    for (var i = 0; i < m; i++) em[(n + i) * s + n + m + i] = 1.0;
                    var ms = Flat(Linalg.Expm(new NdArray(em, s, s)));
                    var ms11 = new double[n * n];
                    var ms12 = new double[n * m];
                    var ms13 = new double[n * m];
                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++) ms11[i * n + j] = ms[i * s + j];
                        for (var j = 0; j < m; j++) { ms12[i * m + j] = ms[i * s + n + j]; ms13[i * m + j] = ms[i * s + n + m + j]; }
                    }
                    var p = MatMul(ms11, n, n, ms13, m);
                    var bd = new double[n * m];
                    for (var i = 0; i < n * m; i++) bd[i] = ms12[i] - ms13[i] + p[i];
                    var cm = MatMul(c, q, n, ms13, m);
                    var dd = new double[q * m];
                    for (var i = 0; i < q * m; i++) dd[i] = d[i] + cm[i];
                    return new StateSpace(new NdArray(ms11, n, n), new NdArray(bd, n, m), system.C, new NdArray(dd, q, m), dt);
                }
            case DiscretizationMethod.Impulse:
                {
                    foreach (var v in d)
                        if (Math.Abs(v) > 1e-8) throw new ArgumentException("Impulse method is only applicable to strictly proper systems.");
                    var at = new double[n * n];
                    for (var i = 0; i < n * n; i++) at[i] = a[i] * dt;
                    var ad = Flat(Linalg.Expm(new NdArray(at, n, n)));
                    var bd = MatMul(ad, n, n, b, m);
                    for (var i = 0; i < bd.Length; i++) bd[i] *= dt;
                    var dd = MatMul(c, q, n, b, m);
                    for (var i = 0; i < dd.Length; i++) dd[i] *= dt;
                    return new StateSpace(new NdArray(ad, n, n), new NdArray(bd, n, m), system.C, new NdArray(dd, q, m), dt);
                }
            default:
                throw new ArgumentException($"Unknown transformation method '{method}'.", nameof(method));
        }
    }

    /// <summary>Discretises any continuous system, keeping its representation (<c>lti.to_discrete</c>).</summary>
    public static LtiSystem Cont2Discrete(LtiSystem system, double dt, DiscretizationMethod method = DiscretizationMethod.Zoh, double alpha = double.NaN)
    {
        ArgumentNullException.ThrowIfNull(system);
        if (system.IsDiscrete) throw new ArgumentException("System is already discrete.", nameof(system));
        return system.ToDiscrete(dt, method, alpha);
    }

    private static StateSpace Gbt(StateSpace sys, double dt, double alpha)
    {
        var n = sys.States;
        var m = sys.Inputs;
        var q = sys.Outputs;
        var a = Flat(sys.A);
        var b = Flat(sys.B);
        var c = Flat(sys.C);
        var d = Flat(sys.D);
        var ima = new double[n * n];
        var rhsA = new double[n * n];
        for (var i = 0; i < n * n; i++)
        {
            ima[i] = -alpha * dt * a[i];
            rhsA[i] = (1.0 - alpha) * dt * a[i];
        }
        for (var i = 0; i < n; i++) { ima[i * n + i] += 1; rhsA[i * n + i] += 1; }
        var ad = SolveColumns(ima, rhsA, n, n);
        var bdt = b.Select(v => dt * v).ToArray();
        var bd = SolveColumns(ima, bdt, n, m);
        // cd = solve(ima^T, c^T)^T
        var imaT = new double[n * n];
        for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) imaT[i * n + j] = ima[j * n + i];
        var cT = new double[n * q];
        for (var i = 0; i < q; i++) for (var j = 0; j < n; j++) cT[j * q + i] = c[i * n + j];
        var cdT = SolveColumns(imaT, cT, n, q);
        var cd = new double[q * n];
        for (var i = 0; i < q; i++) for (var j = 0; j < n; j++) cd[i * n + j] = cdT[j * q + i];
        var cb = MatMul(c, q, n, bd, m);
        var dd = new double[q * m];
        for (var i = 0; i < q * m; i++) dd[i] = d[i] + alpha * cb[i];
        return new StateSpace(new NdArray(ad, n, n), new NdArray(bd, n, m), new NdArray(cd, q, n), new NdArray(dd, q, m), dt);
    }

    private static double[] SolveColumns(double[] a, double[] rhs, int n, int cols)
    {
        var x = new double[n * cols];
        for (var j = 0; j < cols; j++)
        {
            var col = new double[n];
            for (var i = 0; i < n; i++) col[i] = rhs[i * cols + j];
            var s = SolveDense(a, col, n);
            for (var i = 0; i < n; i++) x[i * cols + j] = s[i];
        }
        return x;
    }

    // ------------------------------------------------------------------ continuous responses

    /// <summary>
    /// Simulates a continuous system's response to input <paramref name="u"/> sampled at
    /// <paramref name="t"/> (<c>scipy.signal.lsim</c>). With <paramref name="interp"/> the input is
    /// linear between samples (first-order hold), otherwise held constant; either way the
    /// discretisation is exact, from one matrix exponential — not an ODE solver.
    /// </summary>
    /// <param name="system">Continuous system.</param>
    /// <param name="u">Input, one row per time point and one column per input; null for zero input.</param>
    /// <param name="t">Equally spaced, non-negative time points.</param>
    /// <param name="x0">Initial state (default zero).</param>
    /// <param name="interp">Linear (true) or zero-order-hold (false) input between samples.</param>
    public static LtiResponse Lsim(LtiSystem system, NdArray? u, double[] t, double[]? x0 = null, bool interp = true)
    {
        ArgumentNullException.ThrowIfNull(system);
        ArgumentNullException.ThrowIfNull(t);
        if (system.IsDiscrete) throw new ArgumentException("lsim can only be used with continuous-time systems; use Dlsim.", nameof(system));
        var sys = system.ToSs();
        var n = sys.States;
        var m = sys.Inputs;
        var q = sys.Outputs;
        var a = Flat(sys.A);
        var c = Flat(sys.C);
        var d = Flat(sys.D);
        var steps = t.Length;
        if (steps == 0) throw new ArgumentException("T must not be empty.", nameof(t));
        var x = new double[steps * n];
        var init = x0 ?? new double[n];
        if (init.Length != n) throw new ArgumentException($"X0 must have {n} elements.", nameof(x0));
        if (t[0] == 0) Array.Copy(init, x, n);
        else if (t[0] > 0)
        {
            var at = new double[n * n];
            for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) at[i * n + j] = a[j * n + i] * t[0];
            var e = Flat(Linalg.Expm(new NdArray(at, n, n)));
            var r = MatMul(init, 1, n, e, n);
            Array.Copy(r, x, n);
        }
        else throw new ArgumentException("Initial time must be nonnegative.", nameof(t));

        double[]? uf = null;
        if (u is not null)
        {
            var uu = u.Rank == 1 ? new NdArray(u.ToArray(), u.Size, 1) : u;
            if (uu.Shape[0] != steps) throw new ArgumentException("U must have the same number of rows as elements in T.", nameof(u));
            if (uu.Shape[1] != m) throw new ArgumentException("System does not define that many inputs.", nameof(u));
            uf = Flat(uu);
            if (uf.All(v => v == 0)) uf = null;
        }

        if (steps > 1)
        {
            var dt = t[1] - t[0];
            for (var i = 1; i < steps; i++)
                if (Math.Abs(t[i] - t[i - 1] - dt) > 1e-8 + 1e-5 * Math.Abs(dt)) throw new ArgumentException("Time steps are not equally spaced.", nameof(t));
            if (uf is null)
            {
                var at = new double[n * n];
                for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) at[i * n + j] = a[j * n + i] * dt;
                var e = Flat(Linalg.Expm(new NdArray(at, n, n)));
                for (var i = 1; i < steps; i++)
                {
                    var r = MatMul(x.AsSpan((i - 1) * n, n), 1, n, e, n);
                    Array.Copy(r, 0, x, i * n, n);
                }
            }
            else
            {
                var b = Flat(sys.B);
                double[] ad, bd0, bd1;
                if (!interp)
                {
                    var s = n + m;
                    var mm = new double[s * s];
                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++) mm[i * s + j] = a[i * n + j] * dt;
                        for (var j = 0; j < m; j++) mm[i * s + n + j] = b[i * m + j] * dt;
                    }
                    var ex = ExpmTransposed(mm, s);
                    ad = Block(ex, s, 0, n, 0, n);
                    bd0 = Block(ex, s, n, m, 0, n);
                    bd1 = new double[m * n];
                }
                else
                {
                    var s = n + 2 * m;
                    var mm = new double[s * s];
                    for (var i = 0; i < n; i++)
                    {
                        for (var j = 0; j < n; j++) mm[i * s + j] = a[i * n + j] * dt;
                        for (var j = 0; j < m; j++) mm[i * s + n + j] = b[i * m + j] * dt;
                    }
                    for (var i = 0; i < m; i++) mm[(n + i) * s + n + m + i] = 1.0;
                    var ex = ExpmTransposed(mm, s);
                    ad = Block(ex, s, 0, n, 0, n);
                    bd1 = Block(ex, s, n + m, m, 0, n);
                    var b0 = Block(ex, s, n, m, 0, n);
                    bd0 = new double[m * n];
                    for (var i = 0; i < bd0.Length; i++) bd0[i] = b0[i] - bd1[i];
                }
                for (var i = 1; i < steps; i++)
                {
                    var r = MatMul(x.AsSpan((i - 1) * n, n), 1, n, ad, n);
                    var r0 = MatMul(uf.AsSpan((i - 1) * m, m), 1, m, bd0, n);
                    var r1 = MatMul(uf.AsSpan(i * m, m), 1, m, bd1, n);
                    for (var j = 0; j < n; j++) x[i * n + j] = r[j] + r0[j] + r1[j];
                }
            }
        }
        // y = x C^T + u D^T
        var y = new double[steps * q];
        for (var i = 0; i < steps; i++)
            for (var k = 0; k < q; k++)
            {
                var s = 0.0;
                for (var j = 0; j < n; j++) s += x[i * n + j] * c[k * n + j];
                if (uf is not null) for (var j = 0; j < m; j++) s += uf[i * m + j] * d[k * m + j];
                y[i * q + k] = s;
            }
        return new LtiResponse(t.ToArray(), new NdArray(y, steps, q), new NdArray(x, steps, n));
    }

    /// <summary>Single-input convenience overload of <see cref="Lsim(LtiSystem, NdArray, double[], double[], bool)"/>.</summary>
    public static LtiResponse Lsim(LtiSystem system, double[]? u, double[] t, double[]? x0 = null, bool interp = true)
        => Lsim(system, u is null ? null : new NdArray(u.ToArray(), u.Length, 1), t, x0, interp);

    private static double[] ExpmTransposed(double[] mm, int s)
    {
        var mt = new double[s * s];
        for (var i = 0; i < s; i++) for (var j = 0; j < s; j++) mt[i * s + j] = mm[j * s + i];
        return Flat(Linalg.Expm(new NdArray(mt, s, s)));
    }

    private static double[] Block(double[] m, int s, int r0, int rows, int c0, int cols)
    {
        var o = new double[rows * cols];
        for (var i = 0; i < rows; i++) for (var j = 0; j < cols; j++) o[i * cols + j] = m[(r0 + i) * s + c0 + j];
        return o;
    }

    /// <summary>scipy's default time grid for impulse/step: 0 to 7 time constants of the slowest pole.</summary>
    private static double[] DefaultResponseTimes(StateSpace sys, int n)
    {
        var r = double.PositiveInfinity;
        foreach (var v in sys.States == 0 ? [] : Linalg.Eigvals(sys.A)) r = Math.Min(r, Math.Abs(v.Real));
        if (r == 0.0 || double.IsInfinity(r)) r = 1.0;
        return Linspace(0.0, 7 * (1.0 / r), n);
    }

    /// <summary>
    /// Impulse response of a single-input continuous system (<c>scipy.signal.impulse</c>),
    /// simulated as the free response from initial state <c>B (+ X0)</c>.
    /// </summary>
    /// <returns>Times and the first output.</returns>
    public static (double[] T, double[] Y) Impulse(LtiSystem system, double[]? x0 = null, double[]? t = null, int? n = null)
    {
        ArgumentNullException.ThrowIfNull(system);
        var sys = system.ToSs();
        if (sys.Inputs != 1) throw new ArgumentException("Impulse requires a single-input system.");
        var x = Flat(sys.B);
        if (x0 is not null) for (var i = 0; i < x.Length; i++) x[i] += x0[i];
        var tt = t ?? DefaultResponseTimes(sys, n ?? 100);
        var r = Lsim(sys, (NdArray?)null, tt, x, interp: false);
        return (tt, r.Y);
    }

    /// <summary>Step response of a continuous system (<c>scipy.signal.step</c>); returns the first output.</summary>
    public static (double[] T, double[] Y) Step(LtiSystem system, double[]? x0 = null, double[]? t = null, int? n = null)
    {
        ArgumentNullException.ThrowIfNull(system);
        var sys = system.ToSs();
        var tt = t ?? DefaultResponseTimes(sys, n ?? 100);
        var u = new double[tt.Length * sys.Inputs];
        Array.Fill(u, 1.0);
        var r = Lsim(sys, new NdArray(u, tt.Length, sys.Inputs), tt, x0, interp: false);
        return (tt, r.Y);
    }

    /// <summary>Frequency response <c>H(jω)</c> of a continuous SISO system (<c>scipy.signal.freqresp</c>).</summary>
    /// <param name="system">Continuous system.</param>
    /// <param name="w">Angular frequencies; null chooses <paramref name="n"/> log-spaced points (<see cref="Findfreqs"/>).</param>
    /// <param name="n">Number of points when <paramref name="w"/> is null.</param>
    public static (double[] W, Complex[] H) Freqresp(LtiSystem system, double[]? w = null, int n = 10000)
    {
        ArgumentNullException.ThrowIfNull(system);
        if (system.IsDiscrete) throw new ArgumentException("freqresp can only be used with continuous-time systems; use Dfreqresp.", nameof(system));
        if (system is TransferFunction tf)
            return w is null ? Freqs(tf.Num, tf.Den, n) : Freqs(tf.Num, tf.Den, w);
        var zpk = system.ToZpk().ToFilterZpk();
        return w is null ? FreqsZpk(zpk, n) : FreqsZpk(zpk, w);
    }

    /// <summary>Bode magnitude (dB) and unwrapped phase (degrees) of a continuous system (<c>scipy.signal.bode</c>).</summary>
    public static (double[] W, double[] MagDb, double[] PhaseDeg) Bode(LtiSystem system, double[]? w = null, int n = 100)
    {
        var (ww, h) = Freqresp(system, w, n);
        return (ww, h.Select(v => 20.0 * Math.Log10(Complex.Abs(v))).ToArray(), Unwrap(h.Select(v => Math.Atan2(v.Imaginary, v.Real)).ToArray()).Select(v => v * 180.0 / Math.PI).ToArray());
    }

    /// <summary>numpy's <c>unwrap</c> with the default 2π period.</summary>
    internal static double[] Unwrap(double[] p)
    {
        var o = p.ToArray();
        var correction = 0.0;
        for (var i = 1; i < p.Length; i++)
        {
            var dd = p[i] - p[i - 1];
            var ddmod = ((dd + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
            if (ddmod == -Math.PI && dd > 0) ddmod = Math.PI;
            var corr = ddmod - dd;
            if (Math.Abs(dd) < Math.PI) corr = 0;
            correction += corr;
            o[i] = p[i] + correction;
        }
        return o;
    }

    // ------------------------------------------------------------------ discrete responses

    /// <summary>
    /// Simulates a discrete system (<c>scipy.signal.dlsim</c>). Without <paramref name="t"/> one
    /// output is produced per input row; with it the input is linearly interpolated onto the
    /// system's sample grid up to <c>t[^1]</c>.
    /// </summary>
    public static LtiResponse Dlsim(LtiSystem system, NdArray u, double[]? t = null, double[]? x0 = null)
    {
        ArgumentNullException.ThrowIfNull(system);
        ArgumentNullException.ThrowIfNull(u);
        if (!system.IsDiscrete) throw new ArgumentException("dlsim can only be used with discrete-time systems.", nameof(system));
        var sys = system.ToSs();
        var dt = system.Dt!.Value;
        var n = sys.States;
        var m = sys.Inputs;
        var q = sys.Outputs;
        var uu = u.Rank == 1 ? new NdArray(u.ToArray(), u.Size, 1) : u;
        if (uu.Shape[1] != m) throw new ArgumentException("System does not define that many inputs.", nameof(u));
        var uf = Flat(uu);
        int outSamples;
        double stop;
        if (t is null) { outSamples = uu.Shape[0]; stop = (outSamples - 1) * dt; }
        else { stop = t[^1]; outSamples = (int)Math.Floor(stop / dt) + 1; }
        var tout = Linspace(0.0, stop, outSamples);
        double[] ud;
        if (t is null) ud = uf;
        else
        {
            ud = new double[outSamples * m];
            var col = new double[t.Length];
            for (var j = 0; j < m; j++)
            {
                for (var i = 0; i < t.Length; i++) col[i] = uf[i * m + j];
                for (var i = 0; i < outSamples; i++) ud[i * m + j] = InterpLinearExtrap(tout[i], t, col);
            }
        }
        var a = Flat(sys.A);
        var b = Flat(sys.B);
        var c = Flat(sys.C);
        var d = Flat(sys.D);
        var x = new double[outSamples * n];
        if (x0 is not null) Array.Copy(x0, x, n);
        var y = new double[outSamples * q];
        for (var i = 0; i < outSamples; i++)
        {
            for (var k = 0; k < q; k++)
            {
                var s = 0.0;
                for (var j = 0; j < n; j++) s += c[k * n + j] * x[i * n + j];
                for (var j = 0; j < m; j++) s += d[k * m + j] * ud[i * m + j];
                y[i * q + k] = s;
            }
            if (i == outSamples - 1) break;
            for (var r = 0; r < n; r++)
            {
                var s = 0.0;
                for (var j = 0; j < n; j++) s += a[r * n + j] * x[i * n + j];
                for (var j = 0; j < m; j++) s += b[r * m + j] * ud[i * m + j];
                x[(i + 1) * n + r] = s;
            }
        }
        return new LtiResponse(tout, new NdArray(y, outSamples, q), new NdArray(x, outSamples, n));
    }

    /// <summary>Single-input convenience overload of <see cref="Dlsim(LtiSystem, NdArray, double[], double[])"/>.</summary>
    public static LtiResponse Dlsim(LtiSystem system, double[] u, double[]? t = null, double[]? x0 = null)
    {
        ArgumentNullException.ThrowIfNull(u);
        return Dlsim(system, new NdArray(u.ToArray(), u.Length, 1), t, x0);
    }

    /// <summary>Linear interpolation that extrapolates beyond the ends (a k=1 B-spline, as scipy's dlsim uses).</summary>
    private static double InterpLinearExtrap(double x, double[] xp, double[] fp)
    {
        if (xp.Length == 1) return fp[0];
        var i = Array.BinarySearch(xp, x);
        if (i < 0) i = ~i - 1;
        i = Math.Clamp(i, 0, xp.Length - 2);
        return fp[i] + (fp[i + 1] - fp[i]) * (x - xp[i]) / (xp[i + 1] - xp[i]);
    }

    /// <summary>
    /// Impulse response of a discrete system (<c>dimpulse</c>): one response per input, each
    /// <c>steps × outputs</c>. Default <c>n = 100</c> samples.
    /// </summary>
    public static (double[] T, NdArray[] Y) Dimpulse(LtiSystem system, double[]? x0 = null, double[]? t = null, int? n = null)
        => DiscreteTest(system, x0, t, n, impulse: true);

    /// <summary>Step response of a discrete system (<c>dstep</c>), one response per input.</summary>
    public static (double[] T, NdArray[] Y) Dstep(LtiSystem system, double[]? x0 = null, double[]? t = null, int? n = null)
        => DiscreteTest(system, x0, t, n, impulse: false);

    private static (double[] T, NdArray[] Y) DiscreteTest(LtiSystem system, double[]? x0, double[]? t, int? n, bool impulse)
    {
        ArgumentNullException.ThrowIfNull(system);
        if (!system.IsDiscrete) throw new ArgumentException("Requires a discrete-time system.", nameof(system));
        var sys = system.ToSs();
        var dt = system.Dt!.Value;
        var count = n ?? 100;
        var tt = t ?? LinspaceEndpoint(0, count * dt, count, false);
        var outs = new NdArray[sys.Inputs];
        double[] tout = tt;
        for (var i = 0; i < sys.Inputs; i++)
        {
            var u = new double[tt.Length * sys.Inputs];
            if (impulse) u[i] = 1.0;
            else for (var k = 0; k < tt.Length; k++) u[k * sys.Inputs + i] = 1.0;
            var r = Dlsim(sys, new NdArray(u, tt.Length, sys.Inputs), tt, x0);
            outs[i] = r.Yout;
            tout = r.T;
        }
        return (tout, outs);
    }

    /// <summary>Frequency response of a discrete SISO system (<c>dfreqresp</c>); w in rad/sample.</summary>
    public static (double[] W, Complex[] H) Dfreqresp(LtiSystem system, double[]? w = null, int n = 10000, bool whole = false)
    {
        ArgumentNullException.ThrowIfNull(system);
        if (!system.IsDiscrete) throw new ArgumentException("dfreqresp can only be used with discrete-time systems.", nameof(system));
        if (system is ZerosPolesGain zp)
            return w is null ? FreqzZpk(zp.ToFilterZpk(), n, whole) : FreqzZpk(zp.ToFilterZpk(), w);
        var tf = system.ToTf();
        // z -> z^-1: left-pad the shorter polynomial with zeros.
        var num = tf.Num;
        var den = tf.Den;
        var diff = num.Length - den.Length;
        if (diff > 0) den = new double[diff].Concat(den).ToArray();
        else if (diff < 0) num = new double[-diff].Concat(num).ToArray();
        return w is null ? Freqz(num, den, n, whole) : Freqz(num, den, w);
    }

    /// <summary>Bode data of a discrete system (<c>dbode</c>): frequency in rad/time unit (w/dt), magnitude dB, phase degrees.</summary>
    public static (double[] W, double[] MagDb, double[] PhaseDeg) Dbode(LtiSystem system, double[]? w = null, int n = 100)
    {
        var (ww, h) = Dfreqresp(system, w, n);
        var dt = system.Dt!.Value;
        return (ww.Select(v => v / dt).ToArray(), h.Select(v => 20.0 * Math.Log10(Complex.Abs(v))).ToArray(),
            Unwrap(h.Select(v => Math.Atan2(v.Imaginary, v.Real)).ToArray()).Select(v => v * 180.0 / Math.PI).ToArray());
    }
}
