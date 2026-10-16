using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Univariate spline in the B-spline basis (scipy.interpolate.BSpline):
/// <c>S(x) = Σ_j c[j] B_{j,k;t}(x)</c> on the base interval <c>[t[k], t[n]]</c>,
/// n = len(t) - k - 1.
/// </summary>
/// <remarks>
/// <para>Evaluation is de Boor's recurrence on the k+1 non-zero B-splines (a port of scipy's
/// <c>_deBoor_D</c>), so derivatives cost the same as values. Coefficients may carry trailing
/// dimensions (vector-valued splines); as in scipy, c may also have more rows than n — the
/// extra rows are ignored, which is how FITPACK pads its arrays.</para>
/// <para>Derivative and antiderivative are FITPACK's <c>splder</c>/<c>splantider</c>
/// coefficient recurrences, exact in exact arithmetic. A spline with an interior knot of
/// multiplicity k+1 is discontinuous there and cannot be differentiated: that throws rather
/// than returning infinities.</para>
/// </remarks>
public sealed class BSpline : PiecewiseInterpolantBase
{
    private readonly double[] _t;
    private readonly double[] _c; // (nc, nt) row-major, nc >= n
    private readonly int _nc;

    /// <summary>Degree k.</summary>
    public int K { get; }

    /// <summary>Creates a scalar spline from knots, coefficients and degree.</summary>
    public BSpline(double[] t, double[] c, int k, InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate)
        : this(t, new NdArray((double[])(c ?? throw new ArgumentNullException(nameof(c))).Clone(), c.Length), k, extrapolate, 0) { }

    /// <summary>Creates a spline whose coefficients run along <paramref name="axis"/> of c.</summary>
    public BSpline(double[] t, NdArray c, int k, InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate, int axis = 0)
        : base([], extrapolate, NormalizeAxis(axis, c.Rank))
    {
        ArgumentNullException.ThrowIfNull(t);
        if (k < 0) throw new ArgumentException("Spline order cannot be negative.", nameof(k));
        K = k;
        _t = (double[])t.Clone();
        var n = _t.Length - k - 1;
        if (n < k + 1) throw new ArgumentException($"Need at least {2 * k + 2} knots for degree {k}");
        for (var i = 1; i < _t.Length; i++)
            if (_t[i] < _t[i - 1]) throw new ArgumentException("Knots must be in a non-decreasing order.");
        foreach (var v in _t) if (!double.IsFinite(v)) throw new ArgumentException("Knots should not have nans or infs.");
        if (!(_t[n] > _t[k])) throw new ArgumentException("Need at least two internal knots.");
        _c = MoveAxisToFront(c, Axis, out _nc, out var rest);
        _rest = rest;
        _nt = Product(rest);
        if (_nc < n) throw new ArgumentException("Knots, coefficients and degree are inconsistent.");
    }

    private BSpline(double[] t, double[] c, int nc, int[] rest, int k, InterpolateExtrapolate extrapolate, int axis)
        : base(rest, extrapolate, axis)
    {
        _t = t; _c = c; _nc = nc; K = k;
    }

    /// <summary>scipy's <c>construct_fast</c>: no validation or copying; c is (nc, nt) row-major
    /// with trailing shape <paramref name="valueShape"/>.</summary>
    public static BSpline ConstructFast(double[] t, double[] c, int k, InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate,
        int axis = 0, int[]? valueShape = null)
    {
        var rest = valueShape ?? [];
        var nt = Product(rest);
        return new BSpline(t, c, c.Length / Math.Max(nt, 1), rest, k, extrapolate, axis);
    }

    internal static BSpline Raw(double[] t, double[] c, int nc, int[] rest, int k, InterpolateExtrapolate e, int axis) => new(t, c, nc, rest, k, e, axis);

    /// <summary>Knot vector (a copy).</summary>
    public double[] T => (double[])_t.Clone();

    /// <summary>Coefficients, shape (nc, …ValueShape) (a copy).</summary>
    public NdArray C
    {
        get
        {
            var shape = new int[1 + _rest.Length];
            shape[0] = _nc;
            _rest.CopyTo(shape, 1);
            return new NdArray((double[])_c.Clone(), shape);
        }
    }

    /// <summary>Number of basis functions n = len(t) - k - 1.</summary>
    public int BasisCount => _t.Length - K - 1;

    /// <summary>(t, c, k) of a scalar spline, as scipy's <c>tck</c> (copies).</summary>
    public (double[] T, double[] C, int K) Tck
    {
        get
        {
            RequireScalar();
            return ((double[])_t.Clone(), (double[])_c.Clone(), K);
        }
    }

    internal double[] KnotsRaw => _t;
    internal double[] CoefficientsRaw => _c;
    internal int CoefficientRows => _nc;
    internal int Width => _nt;
    internal int[] Trailing => _rest;

    private protected override long CostPerPoint => (long)(K + 1) * (K + 2) + 8;

    private protected override void EvalChunk(ReadOnlySpan<double> xs, Span<double> output, int nu, InterpolateExtrapolate mode)
    {
        var t = _t;
        var k = K;
        var n = t.Length - k - 1;
        var periodic = mode == InterpolateExtrapolate.Periodic;
        var ext = mode == InterpolateExtrapolate.Extrapolate;
        double tk = t[k], per = t[n] - t[k];
        Span<double> h = stackalloc double[k + 1];
        Span<double> w = stackalloc double[k + 1];
        var hint = k;
        var nt = _nt;
        var c = _c;
        if (nu > k)
        {
            // Every derivative above the degree vanishes (inside the base interval).
            for (var p = 0; p < xs.Length; p++)
            {
                var v = xs[p];
                if (periodic) v = tk + PyMod(v - tk, per);
                var l = FindSplineInterval(t, k, v, hint, ext || periodic);
                output.Slice(p * nt, nt).Fill(l < 0 ? double.NaN : 0.0);
            }
            return;
        }
        for (var p = 0; p < xs.Length; p++)
        {
            var v = xs[p];
            if (periodic) v = tk + PyMod(v - tk, per);
            EvaluateSplinePoint(t, c, nt, k, v, nu, ext, ref hint, output.Slice(p * nt, nt), h, w);
        }
    }

    private double[] PaddedCoefficients()
    {
        // Pad c to len(t) rows, as scipy does before splder/splantider.
        var rows = Math.Max(_nc, _t.Length);
        if (rows == _nc) return (double[])_c.Clone();
        var c = new double[rows * _nt];
        Array.Copy(_c, c, _c.Length);
        return c;
    }

    /// <summary>The <paramref name="nu"/>-th derivative as a spline of degree k - nu (FITPACK
    /// <c>splder</c>); negative nu gives the antiderivative.</summary>
    public BSpline Derivative(int nu = 1)
    {
        if (nu < 0) return Antiderivative(-nu);
        var (t, c, k) = SplderCore(_t, PaddedCoefficients(), _nt, K, nu);
        return new BSpline(t, c, c.Length / _nt, _rest, k, Extrapolate, Axis);
    }

    /// <summary>The <paramref name="nu"/>-th antiderivative (FITPACK <c>splantider</c>); periodic
    /// extrapolation becomes None, as in scipy.</summary>
    public BSpline Antiderivative(int nu = 1)
    {
        if (nu < 0) return Derivative(-nu);
        var (t, c, k) = SplantiderCore(_t, PaddedCoefficients(), _nt, K, nu);
        return new BSpline(t, c, c.Length / _nt, _rest, k,
            Extrapolate == InterpolateExtrapolate.Periodic ? InterpolateExtrapolate.None : Extrapolate, Axis);
    }

    internal static (double[] T, double[] C, int K) SplderCore(double[] t, double[] c, int nt, int k, int nu)
    {
        if (nu < 0) return SplantiderCore(t, c, nt, k, -nu);
        if (nu > k) throw new ArgumentException($"Order of derivative (n = {nu}) must be <= order of spline (k = {k})");
        for (var j = 0; j < nu; j++)
        {
            var len = t.Length;
            var rows = c.Length / nt;
            // c = (c[1:-1-k] - c[:-2-k]) * k / dt, dt = t[k+1:-1] - t[1:-k-1]
            var nr = rows - k - 2;
            var nc = new double[(nr + k) * nt];
            for (var i = 0; i < nr; i++)
            {
                var dt = t[k + 1 + i] - t[1 + i];
                if (dt == 0) throw new ArgumentException($"The spline has internal repeated knots and is not differentiable {nu} times");
                for (var q = 0; q < nt; q++) nc[i * nt + q] = (c[(i + 1) * nt + q] - c[i * nt + q]) * k / dt;
            }
            c = nc;
            t = t[1..(len - 1)];
            k--;
        }
        return (t, c, k);
    }

    internal static (double[] T, double[] C, int K) SplantiderCore(double[] t, double[] c, int nt, int k, int nu)
    {
        if (nu < 0) return SplderCore(t, c, nt, k, -nu);
        for (var j = 0; j < nu; j++)
        {
            var len = t.Length;
            var m = len - k - 1; // rows used: c[:-k-1] relative to len(t) rows
            var cum = new double[m * nt];
            for (var q = 0; q < nt; q++)
            {
                var run = 0.0;
                for (var i = 0; i < m; i++)
                {
                    run += c[i * nt + q] * (t[k + 1 + i] - t[i]);
                    cum[i * nt + q] = run / (k + 1);
                }
            }
            // c = [0, cum, cum[-1] * (k+2)]
            var rows = 1 + m + (k + 2);
            var nc = new double[rows * nt];
            Array.Copy(cum, 0, nc, nt, m * nt);
            for (var r = 0; r < k + 2; r++)
                Array.Copy(cum, (m - 1) * nt, nc, (1 + m + r) * nt, nt);
            var nt2 = new double[len + 2];
            nt2[0] = t[0];
            Array.Copy(t, 0, nt2, 1, len);
            nt2[^1] = t[^1];
            t = nt2;
            c = nc;
            k++;
        }
        return (t, c, k);
    }

    /// <summary>Definite integral of a scalar spline.</summary>
    public double Integrate(double a, double b, InterpolateExtrapolate? extrapolate = null)
    {
        RequireScalar();
        return IntegrateValues(a, b, extrapolate)[0];
    }

    /// <summary>
    /// Definite integral of every component. With extrapolation off the range is clipped to
    /// the base interval (FITPACK <c>splint</c> semantics); periodic splits it into whole
    /// periods plus a remainder (scipy's <c>BSpline.integrate</c>).
    /// </summary>
    public double[] IntegrateValues(double a, double b, InterpolateExtrapolate? extrapolate = null)
    {
        var e = extrapolate ?? Extrapolate;
        var sign = 1.0;
        if (b < a) { (a, b) = (b, a); sign = -1; }
        var k = K;
        var n = _t.Length - k - 1;
        var (ta, ca, ka) = SplantiderCore(_t, PaddedCoefficients(), _nt, k, 1);
        var anti = new BSpline(ta, ca, ca.Length / _nt, _rest, ka, InterpolateExtrapolate.Extrapolate, Axis);
        var nt = _nt;
        var res = new double[nt];
        void Seg(double p, double q, double scale)
        {
            var fp = anti.EvaluateValues(p, 0, InterpolateExtrapolate.Extrapolate);
            var fq = anti.EvaluateValues(q, 0, InterpolateExtrapolate.Extrapolate);
            for (var j = 0; j < nt; j++) res[j] += scale * (fq[j] - fp[j]);
        }
        if (e == InterpolateExtrapolate.Periodic)
        {
            double ts = _t[k], te = _t[n];
            var period = te - ts;
            var interval = b - a;
            var nPeriods = Math.Floor(interval / period);
            var left = PyMod(interval, period);
            if (nPeriods > 0) Seg(ts, te, nPeriods);
            a = ts + PyMod(a - ts, period);
            b = a + left;
            if (b <= te) Seg(a, b, 1);
            else
            {
                Seg(a, te, 1);
                Seg(ts, ts + b - te, 1);
            }
        }
        else if (e == InterpolateExtrapolate.None)
        {
            a = Math.Clamp(a, _t[k], _t[n]);
            b = Math.Clamp(b, _t[k], _t[n]);
            Seg(a, b, 1);
        }
        else Seg(a, b, 1);
        for (var j = 0; j < nt; j++) res[j] *= sign;
        return res;
    }

    /// <summary>
    /// The B-spline basis element on knots t (degree len(t) - 2), extended by k knots on each
    /// side so that it can be evaluated anywhere (scipy's <c>BSpline.basis_element</c>).
    /// </summary>
    public static BSpline BasisElement(double[] t, InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate)
    {
        ArgumentNullException.ThrowIfNull(t);
        var k = t.Length - 2;
        if (k < 0) throw new ArgumentException("A basis element needs at least 2 knots.");
        var tt = new double[t.Length + 2 * k];
        for (var i = 0; i < k; i++) { tt[i] = t[0] - 1; tt[^(i + 1)] = t[^1] + 1; }
        Array.Copy(t, 0, tt, k, t.Length);
        var c = new double[tt.Length];
        c[k] = 1.0;
        return new BSpline(tt, c, c.Length, [], k, extrapolate, 0);
    }

    /// <summary>
    /// The sparse collocation ("design") matrix B[i, j] = B_j(x_i), shape (len(x), len(t)-k-1),
    /// in CSR form with k+1 entries per row (scipy's <c>BSpline.design_matrix</c>). Without
    /// extrapolation, x outside [t[k], t[n]] throws.
    /// </summary>
    public static CsrMatrix DesignMatrix(double[] x, double[] t, int k, InterpolateExtrapolate extrapolate = InterpolateExtrapolate.None)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(t);
        if (k < 0) throw new ArgumentException("Spline order cannot be negative.");
        RequireNonDecreasing(t, nameof(t));
        if (t.Length < 2 * k + 2) throw new ArgumentException($"Length t is not enough for k={k}.");
        foreach (var v in x) if (!double.IsFinite(v)) throw new ArgumentException("Array must not contain infs or nans.");
        var n = t.Length - k - 1;
        var xs = (double[])x.Clone();
        var ext = extrapolate == InterpolateExtrapolate.Extrapolate;
        if (extrapolate == InterpolateExtrapolate.Periodic)
        {
            for (var i = 0; i < xs.Length; i++) xs[i] = t[k] + PyMod(xs[i] - t[k], t[n] - t[k]);
        }
        else if (!ext)
        {
            foreach (var v in xs) if (v < t[k] || v > t[n]) throw new ArgumentException($"Out of bounds w/ x = {v}.");
        }
        var nx = xs.Length;
        var data = new double[nx * (k + 1)];
        var indices = new int[nx * (k + 1)];
        var indptr = new int[nx + 1];
        Span<double> h = stackalloc double[k + 1];
        Span<double> w = stackalloc double[k + 1];
        var hint = k;
        for (var i = 0; i < nx; i++)
        {
            var l = FindSplineInterval(t, k, xs[i], hint, true);
            hint = l;
            DeBoorD(t, xs[i], k, l, 0, h, w);
            for (var a = 0; a <= k; a++)
            {
                data[i * (k + 1) + a] = h[a];
                indices[i * (k + 1) + a] = l - k + a;
            }
            indptr[i + 1] = (i + 1) * (k + 1);
        }
        return new CsrMatrix(data, indices, indptr, nx, n, copy: false);
    }

    /// <summary>
    /// Inserts knot x with multiplicity m (Boehm's algorithm; a port of FITPACK's
    /// <c>insert</c> as in scipy's <c>BSpline.insert_knot</c>, including the periodic
    /// boundary bookkeeping). The spline's values do not change.
    /// </summary>
    public BSpline InsertKnot(double x, int m = 1)
    {
        var k = K;
        if (x < _t[k] || x > _t[_t.Length - k - 1]) throw new ArgumentException($"Cannot insert a knot at {x}.");
        if (m <= 0) throw new ArgumentException($"`m` must be positive, got m={m}.");
        var tt = _t;
        var cc = _c;
        var rows = _nc;
        for (var r = 0; r < m; r++)
            (tt, cc, rows) = InsertOnce(x, tt, cc, rows, _nt, k, Extrapolate == InterpolateExtrapolate.Periodic);
        return new BSpline(tt, cc, rows, _rest, k, Extrapolate, Axis);
    }

    internal static (double[] T, double[] C, int Rows) InsertOnce(double xval, double[] t, double[] c, int rows, int nt, int k, bool periodic)
    {
        var interval = FindSplineInterval(t, k, xval, k, false);
        if (interval < 0) throw new ArgumentException($"Cannot insert the knot at {xval}.");
        if (t[interval] == t[interval + k + 1]) interval--;
        if (periodic && interval + 1 <= 2 * k && interval + 1 >= t.Length - 2 * k)
            throw new ArgumentException("Not enough internal knots.");
        var tt = new double[t.Length + 1];
        Array.Copy(t, 0, tt, 0, interval + 1);
        tt[interval + 1] = xval;
        Array.Copy(t, interval + 1, tt, interval + 2, t.Length - interval - 1);
        var cc = new double[(rows + 1) * nt];
        // cc[interval+1:] = c[interval:]
        Array.Copy(c, interval * nt, cc, (interval + 1) * nt, (rows - interval) * nt);
        for (var i = interval; i > interval - k; i--)
        {
            var fac = (xval - tt[i]) / (tt[i + k + 1] - tt[i]);
            for (var q = 0; q < nt; q++) cc[i * nt + q] = fac * c[i * nt + q] + (1.0 - fac) * c[(i - 1) * nt + q];
        }
        // cc[:interval-k+1] = c[:interval-k+1]
        var head = interval - k + 1;
        if (head > 0) Array.Copy(c, 0, cc, 0, head * nt);
        if (periodic)
        {
            var n = tt.Length;
            var nk = n - k - 1;
            var n2k = n - 2 * k - 1;
            var T = tt[nk] - tt[k];
            if (interval >= nk - k)
            {
                for (var i = 0; i < k; i++) tt[i] = tt[nk - k + i] - T;
                Array.Copy(cc, n2k * nt, cc, 0, k * nt);
            }
            if (interval <= 2 * k - 1)
            {
                for (var i = 0; i < k; i++) tt[n - k + i] = tt[k + 1 + i] + T;
                Array.Copy(cc, 0, cc, n2k * nt, k * nt);
            }
        }
        return (tt, cc, rows + 1);
    }

    /// <summary>The spline as a <see cref="PPoly"/> in the power basis (scipy's
    /// <c>PPoly.from_spline</c>); breakpoints are the full knot vector.</summary>
    public PPoly ToPPoly(InterpolateExtrapolate? extrapolate = null)
    {
        var t = _t;
        var k = K;
        var m = t.Length - 1;
        var nt = _nt;
        var cv = new double[(k + 1) * m * nt];
        Span<double> h = stackalloc double[k + 1];
        Span<double> w = stackalloc double[k + 1];
        var o = new double[nt];
        for (var mm = k; mm >= 0; mm--)
        {
            var g = Factorial(mm);
            var hint = k;
            for (var i = 0; i < m; i++)
            {
                EvaluateSplinePoint(t, _c, nt, k, t[i], mm, true, ref hint, o, h, w);
                for (var q = 0; q < nt; q++) cv[((k - mm) * m + i) * nt + q] = o[q] / g;
            }
        }
        return PPoly.ConstructFast(cv, (double[])t.Clone(), _rest, extrapolate ?? Extrapolate, Axis);
    }
}
