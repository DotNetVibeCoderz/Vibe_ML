using System.Numerics;
using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Shared machinery of <see cref="PPoly"/> and <see cref="BPoly"/> (scipy's <c>_PPolyBase</c>):
/// coefficients of shape (k+1, m, ...) over m intervals between m+1 breakpoints that are
/// strictly increasing or strictly decreasing, plus scipy's <c>axis</c> bookkeeping.
/// </summary>
/// <remarks>Internally the coefficients are one contiguous buffer indexed
/// <c>[(q * m + i) * nt + j]</c>: q is the coefficient (highest power first), i the interval and
/// j the flattened trailing ("value") index.</remarks>
public abstract class PiecewisePolynomialBase : PiecewiseInterpolantBase
{
    private protected double[] _c;
    private protected double[] _x;
    private protected int _k1;   // number of coefficients per interval (order + 1)
    private protected int _m;    // number of intervals

    private protected PiecewisePolynomialBase(double[] c, int k1, int m, int[] rest, double[] x, InterpolateExtrapolate extrapolate, int axis)
        : base(rest, extrapolate, axis)
    {
        _c = c; _k1 = k1; _m = m; _x = x;
    }

    /// <summary>Builds the internal buffer from scipy-shaped coefficients (…, k+1, m, …) where
    /// <paramref name="axis"/> is the position of the k+1 axis.</summary>
    private protected static double[] ImportCoefficients(NdArray c, int axis, out int k1, out int m, out int[] rest)
    {
        if (c.Rank < 2) throw new ArgumentException("Coefficients array must be at least 2-dimensional.", nameof(c));
        if (axis < 0 || axis >= c.Rank - 1) throw new ArgumentException($"axis={axis} must be between 0 and {c.Rank - 1}.", nameof(axis));
        var shape = c.Shape.ToArray();
        k1 = shape[axis]; m = shape[axis + 1];
        rest = new int[c.Rank - 2];
        for (int i = 0, j = 0; i < c.Rank; i++) if (i != axis && i != axis + 1) rest[j++] = shape[i];
        if (axis == 0) return c.ToArray();
        var perm = new int[c.Rank];
        perm[0] = axis; perm[1] = axis + 1;
        for (int i = 0, j = 2; i < c.Rank; i++) if (i != axis && i != axis + 1) perm[j++] = i;
        return c.Transpose(perm).ToArray();
    }

    private protected static double[] CheckBreakpoints(ReadOnlySpan<double> x, int m, int k1)
    {
        if (x.Length < 2) throw new ArgumentException("at least 2 breakpoints are needed", nameof(x));
        if (k1 == 0) throw new ArgumentException("polynomial must be at least of order 0");
        if (m != x.Length - 1) throw new ArgumentException("number of coefficients != len(x)-1");
        bool up = true, down = true;
        for (var i = 1; i < x.Length; i++)
        {
            var d = x[i] - x[i - 1];
            if (!(d >= 0)) up = false;
            if (!(d <= 0)) down = false;
        }
        if (!up && !down) throw new ArgumentException("`x` must be strictly increasing or decreasing.", nameof(x));
        return x.ToArray();
    }

    /// <summary>Breakpoints, length m+1 (a copy).</summary>
    public double[] X => (double[])_x.Clone();

    /// <summary>Coefficients in scipy's internal layout, shape (k+1, m, …) (a copy).</summary>
    public NdArray C
    {
        get
        {
            var shape = new int[2 + _rest.Length];
            shape[0] = _k1; shape[1] = _m;
            _rest.CopyTo(shape, 2);
            return new NdArray((double[])_c.Clone(), shape);
        }
    }

    /// <summary>Polynomial degree k (number of coefficients per interval minus one).</summary>
    public int Degree => _k1 - 1;

    /// <summary>Number of intervals m.</summary>
    public int IntervalCount => _m;

    /// <summary>Whether the breakpoints increase (they may also strictly decrease).</summary>
    public bool Ascending => _x[^1] >= _x[0];

    private protected override long CostPerPoint => 2L * _k1 + 8;

    private protected override void EvalChunk(ReadOnlySpan<double> xs, Span<double> output, int nu, InterpolateExtrapolate mode)
    {
        var x = _x;
        var asc = Ascending;
        double x0 = x[0], xe = x[^1], per = xe - x0;
        var periodic = mode == InterpolateExtrapolate.Periodic;
        var ext = mode == InterpolateExtrapolate.Extrapolate;
        var hint = 0;
        var nt = _nt;
        for (var p = 0; p < xs.Length; p++)
        {
            var v = xs[p];
            if (periodic) v = x0 + PyMod(v - x0, per);
            var i = asc ? FindIntervalAscending(x, v, hint, ext) : FindIntervalDescending(x, v, hint, ext);
            if (i < 0)
            {
                output.Slice(p * nt, nt).Fill(double.NaN);
                continue;
            }
            hint = i;
            EvalInterval(i, v, nu, output.Slice(p * nt, nt));
        }
    }

    /// <summary>Evaluates all nt components on interval i at x.</summary>
    private protected abstract void EvalInterval(int i, double x, int nu, Span<double> output);

    /// <summary>Copy of the coefficient buffer, for subclasses building derived polynomials.</summary>
    private protected double[] CoefficientBuffer => _c;

    private protected void ExtendCore(double[] c2, int k2, int m2, double[] x2, Func<double[], int, int, double[]> raise)
    {
        if (x2.Length != m2) throw new ArgumentException($"Shapes of x ({x2.Length}) and c (…, {m2}) are incompatible.");
        if (m2 == 0) return;
        bool up = true, down = true;
        for (var i = 1; i < x2.Length; i++)
        {
            if (!(x2[i] - x2[i - 1] >= 0)) up = false;
            if (!(x2[i] - x2[i - 1] <= 0)) down = false;
        }
        if (!up && !down) throw new ArgumentException("`x` is not sorted.");
        bool append;
        if (_x[^1] >= _x[0])
        {
            if (!(x2[^1] >= x2[0])) throw new ArgumentException("`x` is in the different order than `self.x`.");
            if (x2[0] >= _x[^1]) append = true;
            else if (x2[^1] <= _x[0]) append = false;
            else throw new ArgumentException("`x` is neither on the left or on the right from `self.x`.");
        }
        else
        {
            if (!(x2[^1] <= x2[0])) throw new ArgumentException("`x` is in the different order than `self.x`.");
            if (x2[0] <= _x[^1]) append = true;
            else if (x2[^1] >= _x[0]) append = false;
            else throw new ArgumentException("`x` is neither on the left or on the right from `self.x`.");
        }
        var kk = Math.Max(k2, _k1);
        var own = raise(_c, _k1, kk);
        var other = raise(c2, k2, kk);
        int ownK = own.Length / (_m * _nt), otherK = other.Length / (m2 * _nt);
        var nm = _m + m2;
        var nt = _nt;
        var res = new double[kk * nm * nt];
        void Put(double[] src, int srcK, int srcM, int colOffset)
        {
            var shift = kk - srcK;
            for (var q = 0; q < srcK; q++)
                for (var i = 0; i < srcM; i++)
                    Array.Copy(src, (q * srcM + i) * nt, res, ((q + shift) * nm + i + colOffset) * nt, nt);
        }
        double[] nx;
        if (append)
        {
            Put(own, ownK, _m, 0);
            Put(other, otherK, m2, _m);
            nx = [.. _x, .. x2];
        }
        else
        {
            Put(other, otherK, m2, 0);
            Put(own, ownK, _m, m2);
            nx = [.. x2, .. _x];
        }
        _c = res; _k1 = kk; _m = nm; _x = nx;
        OnChanged();
    }

    /// <summary>Hook for cached derived objects.</summary>
    private protected virtual void OnChanged() { }

    private protected static (double[] C, int K1, int M) ImportExtension(NdArray c, int[] rest)
    {
        if (c.Rank < 2) throw new ArgumentException("invalid dimensions for c");
        var shape = c.Shape.ToArray();
        if (shape.Length != 2 + rest.Length) throw new ArgumentException("Shapes of c and self.c are incompatible.");
        for (var i = 0; i < rest.Length; i++)
            if (shape[2 + i] != rest[i]) throw new ArgumentException("Shapes of c and self.c are incompatible.");
        return (c.ToArray(), shape[0], shape[1]);
    }
}

/// <summary>
/// Piecewise polynomial in the power basis (scipy.interpolate.PPoly): on
/// <c>x[i] &lt;= xp &lt; x[i+1]</c>, <c>S = Σ_q c[q, i] (xp - x[i])^(k-q)</c>.
/// </summary>
/// <remarks>
/// <para>Roots are found per interval: degrees 1–2 in closed form (with the cancellation-free
/// quadratic formula), higher degrees as companion-matrix eigenvalues refined by one Newton
/// step, which is what scipy does. A polynomial that is identically equal to y on an interval
/// reports that interval's left end followed by NaN, as scipy does.</para>
/// <para>Integration goes through the antiderivative, which is built once and cached.</para>
/// </remarks>
public partial class PPoly : PiecewisePolynomialBase
{
    private PPoly? _antiderivative;

    /// <summary>Creates a piecewise polynomial from coefficients of shape (…, k+1, m, …) with the
    /// k+1 axis at <paramref name="axis"/>, and m+1 breakpoints.</summary>
    public PPoly(NdArray c, double[] x, InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate, int axis = 0)
        : this(ImportCoefficients(c, axis, out var k1, out var m, out var rest), k1, m, rest, CheckBreakpoints(x, m, k1), extrapolate, axis)
    {
    }

    /// <summary>Creates a scalar piecewise polynomial from a (k+1, m) coefficient matrix.</summary>
    public PPoly(double[,] c, double[] x, InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate)
        : this(NdArray.FromArray(c), x, extrapolate) { }

    private protected PPoly(double[] c, int k1, int m, int[] rest, double[] x, InterpolateExtrapolate extrapolate, int axis)
        : base(c, k1, m, rest, x, extrapolate, axis) { }

    /// <summary>scipy's <c>construct_fast</c>: no validation; c is (k+1, m, nt) contiguous.</summary>
    public static PPoly ConstructFast(double[] c, double[] x, int[]? valueShape = null,
        InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate, int axis = 0)
    {
        var rest = valueShape ?? [];
        var m = x.Length - 1;
        var nt = Product(rest);
        if (m < 1 || c.Length % (m * nt) != 0) throw new ArgumentException("c and x are inconsistent.");
        return new PPoly(c, c.Length / (m * nt), m, rest, x, extrapolate, axis);
    }

    private protected PPoly Derived(double[] c, int k1, InterpolateExtrapolate e) => new(c, k1, _m, _rest, _x, e, Axis);

    private protected override void OnChanged() => _antiderivative = null;

    private protected override void EvalInterval(int i, double x, int nu, Span<double> output)
    {
        var s = x - _x[i];
        int k1 = _k1, m = _m, nt = _nt;
        var c = _c;
        var k = k1 - 1;
        if (nu == 0)
        {
            if (nt == 1)
            {
                var r = c[i];
                for (var q = 1; q < k1; q++) r = r * s + c[q * m + i];
                output[0] = r;
                return;
            }
            for (var j = 0; j < nt; j++)
            {
                var r = c[i * nt + j];
                for (var q = 1; q < k1; q++) r = r * s + c[(q * m + i) * nt + j];
                output[j] = r;
            }
            return;
        }
        for (var j = 0; j < nt; j++)
        {
            var r = 0.0;
            for (var q = 0; q <= k - nu; q++)
            {
                var p = k - q;
                var f = 1.0;
                for (var t = p; t > p - nu; t--) f *= t;
                r = r * s + c[(q * m + i) * nt + j] * f;
            }
            output[j] = r;
        }
    }

    /// <summary>Value of the nu-th derivative of the polynomial on interval i at local offset
    /// s, column j (scipy's <c>evaluate_poly1</c>).</summary>
    private double PolyAt(int i, int j, double s, int nu)
    {
        int k1 = _k1, m = _m, nt = _nt;
        var k = k1 - 1;
        var r = 0.0;
        for (var q = 0; q <= k - nu; q++)
        {
            var p = k - q;
            var f = 1.0;
            for (var t = p; t > p - nu; t--) f *= t;
            r = r * s + _c[(q * m + i) * nt + j] * f;
        }
        return r;
    }

    /// <summary>The <paramref name="nu"/>-th derivative as a new piecewise polynomial (negative
    /// nu gives the antiderivative).</summary>
    public PPoly Derivative(int nu = 1)
    {
        if (nu < 0) return Antiderivative(-nu);
        var k1 = _k1 - nu;
        int m = _m, nt = _nt;
        if (k1 <= 0) return Derived(new double[m * nt], 1, Extrapolate);
        var c2 = new double[k1 * m * nt];
        for (var q = 0; q < k1; q++)
        {
            // power of this row after differentiation is k1-1-q; it came from k1-1-q+nu
            var p = _k1 - 1 - q;
            var f = 1.0;
            for (var t = p; t > p - nu; t--) f *= t;
            for (var idx = 0; idx < m * nt; idx++) c2[q * m * nt + idx] = _c[q * m * nt + idx] * f;
        }
        return Derived(c2, k1, Extrapolate);
    }

    /// <summary>
    /// The <paramref name="nu"/>-th antiderivative. Integration constants make it continuous
    /// with value 0 at x[0], and up to its (nu-1)-th derivative continuous across breakpoints
    /// (scipy's <c>fix_continuity</c>). Periodic extrapolation becomes "none", as in scipy: an
    /// antiderivative of a periodic function is not periodic.
    /// </summary>
    public PPoly Antiderivative(int nu = 1)
    {
        if (nu <= 0) return Derivative(-nu);
        int k1 = _k1, m = _m, nt = _nt;
        var nk1 = k1 + nu;
        var c2 = new double[nk1 * m * nt];
        for (var q = 0; q < k1; q++)
        {
            var p = k1 - 1 - q; // original power
            var f = 1.0;
            for (var t = p + 1; t <= p + nu; t++) f *= t;
            for (var idx = 0; idx < m * nt; idx++) c2[q * m * nt + idx] = _c[q * m * nt + idx] / f;
        }
        var res = new PPoly(c2, nk1, m, _rest, _x, Extrapolate == InterpolateExtrapolate.Periodic ? InterpolateExtrapolate.None : Extrapolate, Axis);
        res.FixContinuity(nu - 1);
        return res;
    }

    /// <summary>Sets the low-order coefficients of intervals 1.. so that derivatives 0..order are
    /// continuous at each breakpoint (scipy's <c>_ppoly.fix_continuity</c>).</summary>
    private void FixContinuity(int order)
    {
        int k1 = _k1, m = _m, nt = _nt;
        for (var ip = 1; ip < m; ip++)
        {
            var dxl = _x[ip] - _x[ip - 1];
            for (var dx = order; dx >= 0; dx--)
            {
                var fact = Factorial(dx);
                for (var j = 0; j < nt; j++)
                {
                    var v = PolyAt(ip - 1, j, dxl, dx);
                    _c[((k1 - 1 - dx) * m + ip) * nt + j] = v / fact;
                }
            }
        }
    }

    private PPoly AntiderivativeCached()
    {
        var a = Volatile.Read(ref _antiderivative);
        if (a is not null) return a;
        a = Antiderivative(1);
        Interlocked.CompareExchange(ref _antiderivative, a, null);
        return _antiderivative!;
    }

    /// <summary>Definite integral from a to b (scalar-valued interpolants).</summary>
    public double Integrate(double a, double b, InterpolateExtrapolate? extrapolate = null)
    {
        RequireScalar();
        return IntegrateValues(a, b, extrapolate)[0];
    }

    /// <summary>
    /// Definite integral of every value component. Without extrapolation a bound outside the
    /// breakpoints gives NaN (scipy's behaviour); periodic extrapolation splits the range into
    /// whole periods plus a remainder.
    /// </summary>
    public double[] IntegrateValues(double a, double b, InterpolateExtrapolate? extrapolate = null)
    {
        var e = extrapolate ?? Extrapolate;
        var sign = 1.0;
        if (b < a) { (a, b) = (b, a); sign = -1; }
        var nt = _nt;
        var res = new double[nt];
        var anti = AntiderivativeCached();
        double lo = Math.Min(_x[0], _x[^1]), hi = Math.Max(_x[0], _x[^1]);
        void Seg(double p, double q, double[] acc, double scale)
        {
            var fp = anti.EvaluateValues(p, 0, InterpolateExtrapolate.Extrapolate);
            var fq = anti.EvaluateValues(q, 0, InterpolateExtrapolate.Extrapolate);
            for (var j = 0; j < nt; j++) acc[j] += scale * (fq[j] - fp[j]);
        }
        if (e == InterpolateExtrapolate.Periodic)
        {
            double xs = _x[0], xe = _x[^1];
            var period = xe - xs;
            var interval = b - a;
            var nPeriods = Math.Floor(interval / period);
            var left = PyMod(interval, period);
            if (nPeriods > 0) Seg(xs, xe, res, nPeriods);
            a = xs + PyMod(a - xs, period);
            b = a + left;
            if (b <= xe) Seg(a, b, res, 1);
            else
            {
                Seg(a, xe, res, 1);
                Seg(xs, xs + left + a - xe, res, 1);
            }
        }
        else if (e == InterpolateExtrapolate.None && (a < lo || b > hi))
        {
            Array.Fill(res, double.NaN);
            return res;
        }
        else Seg(a, b, res, 1);
        for (var j = 0; j < nt; j++) res[j] *= sign;
        return res;
    }

    /// <summary>Real solutions of S(x) = y (scalar-valued interpolants), ascending for
    /// ascending breakpoints. See <see cref="SolveValues"/>.</summary>
    public double[] Solve(double y = 0.0, bool discontinuity = true, bool? extrapolate = null)
    {
        RequireScalar();
        return SolveColumn(0, y, discontinuity, extrapolate ?? Extrapolate != InterpolateExtrapolate.None);
    }

    /// <summary>Real roots, S(x) = 0 (scipy's <c>roots</c>).</summary>
    public double[] Roots(bool discontinuity = true, bool? extrapolate = null) => Solve(0.0, discontinuity, extrapolate);

    /// <summary>
    /// Real solutions of S(x) = y for each value component. With
    /// <paramref name="discontinuity"/>, a sign change across a jump at a breakpoint is reported
    /// as a root there. With extrapolation (any mode other than None — periodic counts as
    /// extrapolation, as in scipy) the first and last pieces are searched to ±∞.
    /// </summary>
    public double[][] SolveValues(double y = 0.0, bool discontinuity = true, bool? extrapolate = null)
    {
        var ext = extrapolate ?? Extrapolate != InterpolateExtrapolate.None;
        var res = new double[_nt][];
        for (var j = 0; j < _nt; j++) res[j] = SolveColumn(j, y, discontinuity, ext);
        return res;
    }

    private double[] SolveColumn(int j, double y, bool discontinuity, bool extrapolate)
    {
        int m = _m;
        var x = _x;
        var asc = Ascending;
        var roots = new List<double>();
        var last = double.NaN;
        Span<double> coef = stackalloc double[_k1];
        var cand = new List<double>();
        for (var ii = 0; ii < m; ii++)
        {
            var width = Math.Abs(x[ii + 1] - x[ii]);
            if (ii > 0 && discontinuity)
            {
                var va = PolyAt(ii - 1, j, x[ii] - x[ii - 1], 0) - y;
                var vb = PolyAt(ii, j, 0, 0) - y;
                if ((va < 0 && vb > 0) || (va > 0 && vb < 0))
                {
                    if (!IsDuplicate(x[ii], last, width))
                    {
                        last = x[ii];
                        roots.Add(last);
                    }
                }
            }
            for (var q = 0; q < _k1; q++) coef[q] = _c[(q * m + ii) * _nt + j];
            coef[_k1 - 1] -= y;
            cand.Clear();
            var status = PolynomialRealRoots(coef, cand);
            if (status < 0)
            {
                roots.Add(x[ii]);
                roots.Add(double.NaN);
                last = double.NaN;
                continue;
            }
            if (cand.Count == 0) continue;
            for (var r = 0; r < cand.Count; r++)
            {
                var s = cand[r];
                // One Newton step, accepted only if it is a small correction.
                var f = PolyAt(ii, j, s, 0) - y;
                var df = PolyAt(ii, j, s, 1);
                if (df != 0)
                {
                    var d = f / df;
                    if (Math.Abs(d) < Math.Abs(s)) s -= d;
                }
                cand[r] = s + x[ii];
            }
            // scipy sorts by value even for descending breakpoints.
            cand.Sort();
            foreach (var w in cand)
            {
                bool ok;
                if (ii == 0 && extrapolate)
                    ok = m == 1 || (asc ? w <= x[1] : w >= x[1]);
                else if (ii == m - 1 && extrapolate)
                    ok = asc ? w >= x[ii] : w <= x[ii];
                else
                    ok = Math.Min(x[ii], x[ii + 1]) <= w && w <= Math.Max(x[ii], x[ii + 1]);
                if (!ok) continue;
                if (IsDuplicate(w, last, width)) continue;
                last = w;
                roots.Add(w);
            }
        }
        return [.. roots];
    }

    private static bool IsDuplicate(double r, double last, double width) =>
        !double.IsNaN(last) && (r == last || Math.Abs(r - last) <= 1e-12 * Math.Max(width, 1e-300));

    /// <summary>
    /// Real roots (in the local variable) of Σ coef[q] s^(k-q). Returns -1 when the polynomial is
    /// identically zero; otherwise the number of candidates appended. Mirrors scipy's
    /// <c>croots_poly1</c>: leading zeros are stripped, degrees 1–2 are closed form, higher
    /// degrees use companion-matrix eigenvalues.
    /// </summary>
    internal static int PolynomialRealRoots(ReadOnlySpan<double> coef, List<double> output)
    {
        var n = coef.Length;
        var first = -1;
        for (var q = 0; q < n; q++) if (coef[q] != 0) { first = q; break; }
        if (first < 0) return -1;
        var order = n - 1 - first;
        if (order == 0) return 0;
        if (order == 1)
        {
            output.Add(-coef[n - 1] / coef[n - 2]);
            return 1;
        }
        if (order == 2)
        {
            double a0 = coef[n - 3], a1 = coef[n - 2], a2 = coef[n - 1];
            var d = a1 * a1 - 4 * a0 * a2;
            if (d < 0) return 0;
            d = Math.Sqrt(d);
            if (d == 0) { output.Add(-a1 / (2 * a0)); output.Add(-a1 / (2 * a0)); }
            else if (a1 < 0) { output.Add(2 * a2 / (-a1 + d)); output.Add((-a1 + d) / (2 * a0)); }
            else { output.Add((-a1 - d) / (2 * a0)); output.Add(2 * a2 / (-a1 - d)); }
            return 2;
        }
        // Companion matrix: first row = -coef[1..]/coef[0], ones on the subdiagonal.
        var comp = new double[order * order];
        var lead = coef[first];
        for (var j = 0; j < order; j++)
        {
            comp[j] = -coef[first + 1 + j] / lead;
            if (j + 1 < order) comp[(j + 1) * order + j] = 1;
        }
        var ev = Linalg.Eigvals(new NdArray(comp, order, order));
        var added = 0;
        foreach (var z in ev)
        {
            if (Math.Abs(z.Imaginary) > 1e-10 * Math.Max(1.0, Math.Abs(z.Real))) continue;
            output.Add(z.Real);
            added++;
        }
        return added;
    }

    /// <summary>
    /// Appends (or prepends) intervals: <paramref name="x"/> holds one breakpoint per new
    /// interval (the shared end point excluded) and must lie entirely on one side, in the same
    /// order as the existing breakpoints. Orders are equalised by zero-padding the high powers.
    /// </summary>
    public virtual void Extend(NdArray c, double[] x)
    {
        var (c2, k2, m2) = ImportExtension(c, _rest);
        ExtendCore(c2, k2, m2, x, RaisePower);
    }

    /// <summary>Scalar overload of <see cref="Extend(NdArray, double[])"/>.</summary>
    public void Extend(double[,] c, double[] x) => Extend(NdArray.FromArray(c), x);

    private double[] RaisePower(double[] c, int k1, int kk) => c; // power basis: padding happens at copy time

    /// <summary>
    /// Converts a B-spline (t, c, k) to the power basis (scipy's <c>PPoly.from_spline</c>). The
    /// breakpoints are the full knot vector, so repeated knots give zero-length intervals,
    /// exactly as scipy produces.
    /// </summary>
    public static PPoly FromSpline(double[] t, double[] c, int k, InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate)
    {
        var m = t.Length - 1;
        var nc = t.Length - k - 1;
        if (c.Length < nc) throw new ArgumentException("Knots, coefficients and degree are inconsistent.");
        var cv = new double[(k + 1) * m];
        Span<double> h = stackalloc double[k + 1];
        Span<double> w = stackalloc double[k + 1];
        Span<double> o = stackalloc double[1];
        for (var mm = k; mm >= 0; mm--)
        {
            var g = Factorial(mm);
            var hint = k;
            for (var i = 0; i < m; i++)
            {
                EvaluateSplinePoint(t, c, 1, k, t[i], mm, true, ref hint, o, h, w);
                cv[(k - mm) * m + i] = o[0] / g;
            }
        }
        return new PPoly(cv, k + 1, m, [], (double[])t.Clone(), extrapolate, 0);
    }

    /// <summary>Converts a Bernstein-basis polynomial to the power basis (scipy's
    /// <c>from_bernstein_basis</c>).</summary>
    public static PPoly FromBernsteinBasis(BPoly bp, InterpolateExtrapolate? extrapolate = null)
    {
        ArgumentNullException.ThrowIfNull(bp);
        var (bc, k1, m, nt, x, rest) = bp.Raw;
        var k = k1 - 1;
        var c = new double[bc.Length];
        for (var a = 0; a <= k; a++)
        {
            var sa = (a % 2 == 0 ? 1.0 : -1.0) * Binom(k, a);
            for (var s = a; s <= k; s++)
            {
                var val = Binom(k - a, s - a) * (s % 2 == 0 ? 1.0 : -1.0);
                for (var i = 0; i < m; i++)
                {
                    var dx = x[i + 1] - x[i];
                    var scale = sa * val / Math.Pow(dx, s);
                    for (var j = 0; j < nt; j++) c[((k - s) * m + i) * nt + j] += scale * bc[(a * m + i) * nt + j];
                }
            }
        }
        return new PPoly(c, k1, m, rest, (double[])x.Clone(), extrapolate ?? bp.Extrapolate, bp.Axis);
    }

    internal (double[] C, int K1, int M, int Nt, double[] X, int[] Trailing) Raw => (_c, _k1, _m, _nt, _x, _rest);
}

/// <summary>
/// Piecewise polynomial in the Bernstein basis (scipy.interpolate.BPoly): on
/// <c>x[i] &lt;= xp &lt; x[i+1]</c>, <c>S = Σ_a c[a, i] b(a, k; (xp - x[i]) / (x[i+1] - x[i]))</c>
/// with <c>b(a, k; s) = C(k, a) s^a (1 - s)^(k - a)</c>.
/// </summary>
/// <remarks>Evaluation uses de Casteljau's algorithm, which is numerically stable for any
/// degree; derivatives are formed from forward differences of the control coefficients.</remarks>
public class BPoly : PiecewisePolynomialBase
{
    /// <summary>Creates a Bernstein polynomial from coefficients (…, k+1, m, …) with the k+1 axis
    /// at <paramref name="axis"/>, and m+1 breakpoints.</summary>
    public BPoly(NdArray c, double[] x, InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate, int axis = 0)
        : this(ImportCoefficients(c, axis, out var k1, out var m, out var rest), k1, m, rest, CheckBreakpoints(x, m, k1), extrapolate, axis) { }

    /// <summary>Creates a scalar Bernstein polynomial from a (k+1, m) coefficient matrix.</summary>
    public BPoly(double[,] c, double[] x, InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate)
        : this(NdArray.FromArray(c), x, extrapolate) { }

    private BPoly(double[] c, int k1, int m, int[] rest, double[] x, InterpolateExtrapolate extrapolate, int axis)
        : base(c, k1, m, rest, x, extrapolate, axis) { }

    internal (double[] C, int K1, int M, int Nt, double[] X, int[] Trailing) Raw => (_c, _k1, _m, _nt, _x, _rest);

    private protected override void EvalInterval(int i, double x, int nu, Span<double> output)
    {
        int k1 = _k1, m = _m, nt = _nt;
        var k = k1 - 1;
        var h = _x[i + 1] - _x[i];
        var s = (x - _x[i]) / h;
        if (nu > k) { output[..nt].Clear(); return; }
        var kd = k - nu;
        Span<double> b = k1 <= 128 ? stackalloc double[k1] : new double[k1];
        // k!/(k-nu)! / h^nu
        var scale = 1.0;
        for (var t = k; t > kd; t--) scale *= t;
        scale /= Math.Pow(h, nu);
        for (var j = 0; j < nt; j++)
        {
            for (var a = 0; a <= k; a++) b[a] = _c[(a * m + i) * nt + j];
            // nu forward differences
            for (var d = 0; d < nu; d++)
                for (var a = 0; a < k - d; a++) b[a] = b[a + 1] - b[a];
            // de Casteljau on b[0..kd]
            var u = 1 - s;
            for (var r = 1; r <= kd; r++)
                for (var a = 0; a <= kd - r; a++) b[a] = u * b[a] + s * b[a + 1];
            output[j] = scale * b[0];
        }
    }

    private BPoly Derived(double[] c, int k1, InterpolateExtrapolate e) => new(c, k1, _m, _rest, _x, e, Axis);

    /// <summary>The <paramref name="nu"/>-th derivative (negative nu: antiderivative).</summary>
    public BPoly Derivative(int nu = 1)
    {
        if (nu < 0) return Antiderivative(-nu);
        var bp = this;
        for (var r = 0; r < nu; r++) bp = bp.Derivative1();
        return nu == 0 ? Derived((double[])_c.Clone(), _k1, Extrapolate) : bp;
    }

    private BPoly Derivative1()
    {
        int k1 = _k1, m = _m, nt = _nt;
        var k = k1 - 1;
        if (k == 0) return Derived(new double[m * nt], 1, Extrapolate);
        var c2 = new double[k * m * nt];
        for (var a = 0; a < k; a++)
            for (var i = 0; i < m; i++)
            {
                var dx = _x[i + 1] - _x[i];
                for (var j = 0; j < nt; j++)
                    c2[(a * m + i) * nt + j] = k * (_c[((a + 1) * m + i) * nt + j] - _c[(a * m + i) * nt + j]) / dx;
            }
        return Derived(c2, k, Extrapolate);
    }

    /// <summary>The <paramref name="nu"/>-th antiderivative, zero at x[0] and continuous
    /// (periodic extrapolation becomes None, as in scipy).</summary>
    public BPoly Antiderivative(int nu = 1)
    {
        if (nu <= 0) return Derivative(-nu);
        var bp = this;
        for (var r = 0; r < nu; r++) bp = bp.Antiderivative1();
        return bp;
    }

    private BPoly Antiderivative1()
    {
        int k1 = _k1, m = _m, nt = _nt;
        var k = k1; // scipy: k = c.shape[0]
        var c2 = new double[(k + 1) * m * nt];
        for (var i = 0; i < m; i++)
        {
            var delta = _x[i + 1] - _x[i];
            for (var j = 0; j < nt; j++)
            {
                var cum = 0.0;
                for (var a = 0; a < k; a++)
                {
                    cum += _c[(a * m + i) * nt + j];
                    c2[((a + 1) * m + i) * nt + j] = cum / k * delta;
                }
            }
        }
        // Continuity (scipy): c2[:, 1:] += cumsum(c2[k, :])[:-1] — the integration constant on
        // interval i is the sum of the right-end values of all earlier local antiderivatives.
        // The right ends are read before any shifting.
        for (var j = 0; j < nt; j++)
        {
            var run = 0.0;
            var prevEnd = c2[(k * m + 0) * nt + j];
            for (var i = 1; i < m; i++)
            {
                run += prevEnd;
                prevEnd = c2[(k * m + i) * nt + j];
                for (var a = 0; a <= k; a++) c2[(a * m + i) * nt + j] += run;
            }
        }
        return new BPoly(c2, k + 1, m, _rest, _x,
            Extrapolate == InterpolateExtrapolate.Periodic ? InterpolateExtrapolate.None : Extrapolate, Axis);
    }

    /// <summary>Definite integral (scalar-valued).</summary>
    public double Integrate(double a, double b, InterpolateExtrapolate? extrapolate = null)
    {
        RequireScalar();
        return IntegrateValues(a, b, extrapolate)[0];
    }

    /// <summary>Definite integral of every component, via the antiderivative (scipy's
    /// <c>BPoly.integrate</c>, including its periodic splitting).</summary>
    public double[] IntegrateValues(double a, double b, InterpolateExtrapolate? extrapolate = null)
    {
        var ib = Antiderivative1();
        var e = extrapolate ?? Extrapolate;
        var nt = _nt;
        if (e != InterpolateExtrapolate.Periodic)
        {
            var fa = ib.EvaluateValues(a, 0, e);
            var fb = ib.EvaluateValues(b, 0, e);
            for (var j = 0; j < nt; j++) fb[j] -= fa[j];
            return fb;
        }
        var sign = 1.0;
        if (b < a) { (a, b) = (b, a); sign = -1; }
        double xs = _x[0], xe = _x[^1];
        var period = xe - xs;
        var interval = b - a;
        var nPeriods = Math.Floor(interval / period);
        var left = PyMod(interval, period);
        var res = new double[nt];
        double[] F(double v) => ib.EvaluateValues(v, 0, InterpolateExtrapolate.None);
        var fxe = F(xe); var fxs = F(xs);
        for (var j = 0; j < nt; j++) res[j] = nPeriods * (fxe[j] - fxs[j]);
        a = xs + PyMod(a - xs, period);
        b = a + left;
        var fa2 = F(a);
        if (b <= xe)
        {
            var fb2 = F(b);
            for (var j = 0; j < nt; j++) res[j] += fb2[j] - fa2[j];
        }
        else
        {
            var fr = F(xs + left + a - xe);
            for (var j = 0; j < nt; j++) res[j] += fxe[j] - fa2[j] + fr[j] - fxs[j];
        }
        for (var j = 0; j < nt; j++) res[j] *= sign;
        return res;
    }

    /// <summary>Appends or prepends intervals (see <see cref="PPoly.Extend(NdArray, double[])"/>);
    /// degrees are equalised by degree elevation.</summary>
    public void Extend(NdArray c, double[] x)
    {
        var (c2, k2, m2) = ImportExtension(c, _rest);
        ExtendCore(c2, k2, m2, x, (src, k1, kk) => RaiseDegree(src, k1, src.Length / (k1 * _nt) , _nt, kk - k1));
    }

    /// <summary>Scalar overload of <see cref="Extend(NdArray, double[])"/>.</summary>
    public void Extend(double[,] c, double[] x) => Extend(NdArray.FromArray(c), x);

    /// <summary>Degree elevation by d (scipy's <c>_raise_degree</c>); c is (k1, m, nt).</summary>
    internal static double[] RaiseDegree(double[] c, int k1, int m, int nt, int d)
    {
        if (d == 0) return c;
        var k = k1 - 1;
        var o = new double[(k1 + d) * m * nt];
        for (var a = 0; a < k1; a++)
        {
            var ca = Binom(k, a);
            for (var j = 0; j <= d; j++)
            {
                var f = ca * Binom(d, j) / Binom(k + d, a + j);
                for (var idx = 0; idx < m * nt; idx++) o[(a + j) * m * nt + idx] += f * c[a * m * nt + idx];
            }
        }
        return o;
    }

    /// <summary>Converts a power-basis polynomial to the Bernstein basis (scipy's
    /// <c>from_power_basis</c>).</summary>
    public static BPoly FromPowerBasis(PPoly pp, InterpolateExtrapolate? extrapolate = null)
    {
        ArgumentNullException.ThrowIfNull(pp);
        var (pc, k1, m, nt, x, rest) = pp.Raw;
        var k = k1 - 1;
        var c = new double[pc.Length];
        for (var a = 0; a <= k; a++)
            for (var i = 0; i < m; i++)
            {
                var dx = x[i + 1] - x[i];
                var f0 = Math.Pow(dx, k - a) / Binom(k, k - a);
                for (var jj = k - a; jj <= k; jj++)
                {
                    var f = f0 * Binom(jj, k - a);
                    for (var j = 0; j < nt; j++) c[(jj * m + i) * nt + j] += f * pc[(a * m + i) * nt + j];
                }
            }
        return new BPoly(c, k1, m, rest, (double[])x.Clone(), extrapolate ?? pp.Extrapolate, pp.Axis);
    }

    /// <summary>
    /// Piecewise polynomial matching given derivatives at the breakpoints (scipy's
    /// <c>BPoly.from_derivatives</c>, scalar values). <c>yi[i]</c> lists f, f', f'', … at
    /// <c>xi[i]</c>. Without <paramref name="orders"/> each interval uses every supplied
    /// derivative at both ends; with orders, interval i has degree orders[i] and takes
    /// derivatives from the two ends as evenly as possible (scipy's split, including its
    /// quirks).
    /// </summary>
    public static BPoly FromDerivatives(double[] xi, double[][] yi, int[]? orders = null,
        InterpolateExtrapolate extrapolate = InterpolateExtrapolate.Extrapolate)
    {
        ArgumentNullException.ThrowIfNull(xi);
        ArgumentNullException.ThrowIfNull(yi);
        if (xi.Length != yi.Length) throw new ArgumentException("xi and yi need to have the same length");
        for (var i = 1; i < xi.Length; i++)
            if (xi[i] - xi[0] <= 0) throw new ArgumentException("x coordinates are not in increasing order");
        var m = xi.Length - 1;
        if (m < 1) throw new ArgumentException("at least 2 breakpoints are needed");
        var k = 0;
        for (var i = 0; i < m; i++) k = Math.Max(k, yi[i].Length + yi[i + 1].Length);
        if (orders is not null)
        {
            if (orders.Length == 1 && m > 1) orders = Enumerable.Repeat(orders[0], m).ToArray();
            if (orders.Length != m) throw new ArgumentException("orders must have one entry per interval.");
            foreach (var o in orders) if (o <= 0) throw new ArgumentException("Orders must be positive.");
            k = Math.Max(k, orders.Max());
        }
        var cols = new double[m][];
        for (var i = 0; i < m; i++)
        {
            var y1 = yi[i]; var y2 = yi[i + 1];
            int n1, n2;
            if (orders is null) { n1 = y1.Length; n2 = y2.Length; }
            else
            {
                var n = orders[i] + 1;
                n1 = Math.Min(n / 2, y1.Length);
                n2 = Math.Min(n - n1, y2.Length);
                n1 = Math.Min(n - n2, y2.Length);
                if (n1 + n2 != n)
                    throw new ArgumentException($"Point {xi[i]} has {y1.Length} derivatives, point {xi[i + 1]} has {y2.Length} derivatives, but order {orders[i]} requested");
                if (!(n1 <= y1.Length && n2 <= y2.Length)) throw new ArgumentException("`order` input incompatible with length y1 or y2.");
            }
            var b = ConstructFromDerivatives(xi[i], xi[i + 1], y1.AsSpan(0, n1), y2.AsSpan(0, n2));
            if (b.Length < k) b = RaiseDegree(b, b.Length, 1, 1, k - b.Length);
            cols[i] = b;
        }
        var c = new double[k * m];
        for (var i = 0; i < m; i++)
            for (var a = 0; a < k; a++) c[a * m + i] = cols[i][a];
        return new BPoly(c, k, m, [], (double[])xi.Clone(), extrapolate, 0);
    }

    private static double[] ConstructFromDerivatives(double xa, double xb, ReadOnlySpan<double> ya, ReadOnlySpan<double> yb)
    {
        int na = ya.Length, nb = yb.Length, n = na + nb;
        var c = new double[n];
        static double Poch(double z, int m)
        {
            var r = 1.0;
            for (var i = 0; i < m; i++) r *= z + i;
            return r;
        }
        for (var q = 0; q < na; q++)
        {
            c[q] = ya[q] / Poch(n - q, q) * Math.Pow(xb - xa, q);
            for (var j = 0; j < q; j++) c[q] -= ((j + q) % 2 == 0 ? 1 : -1) * Binom(q, j) * c[j];
        }
        for (var q = 0; q < nb; q++)
        {
            c[n - q - 1] = yb[q] / Poch(n - q, q) * (q % 2 == 0 ? 1 : -1) * Math.Pow(xb - xa, q);
            for (var j = 0; j < q; j++)
            {
                // c[-q+j]: for j = 0 that is c[-q] → index n - q (q > 0 here since j < q)
                c[n - q - 1] -= ((j + 1) % 2 == 0 ? 1 : -1) * Binom(q, j + 1) * c[n - q + j];
            }
        }
        return c;
    }
}
