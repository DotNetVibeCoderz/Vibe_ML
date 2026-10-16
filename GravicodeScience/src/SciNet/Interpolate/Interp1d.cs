using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

public static partial class Interpolate
{
    /// <summary>
    /// numpy.interp: piecewise-linear interpolation of (xp, fp) at x, with <paramref name="left"/>
    /// / <paramref name="right"/> (default fp[0] / fp[^1]) outside, or periodic wrapping when
    /// <paramref name="period"/> is given (then left/right are ignored). xp must be increasing
    /// (not checked, as in numpy).
    /// </summary>
    /// <remarks>
    /// Sorted queries are detected and walked with a monotone hint (amortised O(1) per point);
    /// unsorted ones use a branch-free binary search whose loop the JIT compiles to
    /// conditional moves. Large batches run in parallel. Results equal numpy's bit for bit
    /// away from NaN edge cases, including its fallback formula when the forward one
    /// overflows to NaN.
    /// </remarks>
    public static double[] Interp(double[] x, double[] xp, double[] fp, double? left = null, double? right = null, double? period = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(xp);
        ArgumentNullException.ThrowIfNull(fp);
        if (xp.Length != fp.Length) throw new ArgumentException("fp and xp are not of the same length.");
        if (xp.Length == 0) throw new ArgumentException("array of sample points is empty");
        var res = new double[x.Length];
        if (period is double per)
        {
            if (per == 0) throw new ArgumentException("period must be a non-zero value");
            per = Math.Abs(per);
            var n = xp.Length;
            var idx = Enumerable.Range(0, n).ToArray();
            var xpm = new double[n];
            for (var i = 0; i < n; i++) xpm[i] = PyMod(xp[i], per);
            Array.Sort(xpm, idx);
            var xx = new double[n + 2];
            var ff = new double[n + 2];
            for (var i = 0; i < n; i++) { xx[i + 1] = xpm[i]; ff[i + 1] = fp[idx[i]]; }
            xx[0] = xpm[n - 1] - per; ff[0] = fp[idx[n - 1]];
            xx[n + 1] = xpm[0] + per; ff[n + 1] = fp[idx[0]];
            var xm = new double[x.Length];
            for (var i = 0; i < x.Length; i++) xm[i] = PyMod(x[i], per);
            InterpCore(xm, xx, ff, ff[0], ff[^1], res);
            return res;
        }
        InterpCore(x, xp, fp, left ?? fp[0], right ?? fp[^1], res);
        return res;
    }

    /// <summary>Scalar overload of <see cref="Interp(double[], double[], double[], double?, double?, double?)"/>.</summary>
    public static double Interp(double x, double[] xp, double[] fp, double? left = null, double? right = null, double? period = null) =>
        Interp([x], xp, fp, left, right, period)[0];

    private static void InterpCore(double[] x, double[] xp, double[] fp, double lval, double rval, double[] res)
    {
        ForChunks(x.Length, 12, (s, e) => InterpChunk(x.AsSpan(s, e - s), xp, fp, lval, rval, res.AsSpan(s, e - s)));
    }

    private static double Lerp(double[] xp, double[] fp, int j, double v)
    {
        if (xp[j] == v) return fp[j];
        var slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
        var r = slope * (v - xp[j]) + fp[j];
        if (double.IsNaN(r))
        {
            r = slope * (v - xp[j + 1]) + fp[j + 1];
            if (double.IsNaN(r) && fp[j] == fp[j + 1]) r = fp[j];
        }
        return r;
    }

    private static void InterpChunk(ReadOnlySpan<double> x, double[] xp, double[] fp, double lval, double rval, Span<double> res)
    {
        var n = xp.Length;
        if (n == 1)
        {
            for (var i = 0; i < x.Length; i++)
            {
                var v = x[i];
                res[i] = double.IsNaN(v) ? v : v < xp[0] ? lval : v > xp[0] ? rval : fp[0];
            }
            return;
        }
        double x0 = xp[0], xl = xp[n - 1];
        var j = 0;
        var prev = double.NegativeInfinity;
        for (var i = 0; i < x.Length; i++)
        {
            var v = x[i];
            if (double.IsNaN(v)) { res[i] = v; continue; }
            if (v > xl) { res[i] = rval; continue; }
            if (v < x0) { res[i] = lval; continue; }
            if (v == xl) { res[i] = fp[n - 1]; continue; }
            if (v >= prev && j < n - 1)
            {
                // Monotone walk from the last interval; fall back to bisection after a few steps.
                var steps = 0;
                while (xp[j + 1] <= v && steps < 8) { j++; steps++; }
                if (xp[j + 1] <= v) j = BranchlessFloor(xp, v);
            }
            else j = BranchlessFloor(xp, v);
            prev = v;
            res[i] = Lerp(xp, fp, j, v);
        }
    }

    /// <summary>Largest j in [0, n-2] with xp[j] &lt;= v, for xp[0] &lt;= v &lt; xp[n-1], without
    /// data-dependent branches.</summary>
    private static int BranchlessFloor(double[] xp, double v)
    {
        var b = 0;
        var len = xp.Length - 1; // search among xp[0..n-2]
        while (len > 1)
        {
            var half = len >> 1;
            b = xp[b + half] <= v ? b + half : b;
            len -= half;
        }
        return b;
    }

    /// <summary>Monotone piecewise cubic interpolation of (xi, yi) at x, or its
    /// <paramref name="der"/>-th derivative (scipy.interpolate.pchip_interpolate).</summary>
    public static double[] PchipInterpolate(double[] xi, double[] yi, double[] x, int der = 0) =>
        new PchipInterpolator(xi, yi).Evaluate(x, der);
}

/// <summary>Interpolation kind of <see cref="Interp1d"/> (scipy's <c>kind</c> strings).</summary>
public enum Interp1dKind
{
    /// <summary>Piecewise linear.</summary>
    Linear,
    /// <summary>Nearest point; halfway rounds down.</summary>
    Nearest,
    /// <summary>Nearest point; halfway rounds up.</summary>
    NearestUp,
    /// <summary>Zeroth-order spline (piecewise constant from the left point).</summary>
    Zero,
    /// <summary>First-order spline.</summary>
    Slinear,
    /// <summary>Second-order (not-a-knot) spline.</summary>
    Quadratic,
    /// <summary>Third-order (not-a-knot) spline.</summary>
    Cubic,
    /// <summary>Previous data point.</summary>
    Previous,
    /// <summary>Next data point.</summary>
    Next,
}

/// <summary>scipy's <c>fill_value</c> for <see cref="Interp1d"/>: one value, a (below, above)
/// pair (scalars or one value per component), or "extrapolate".</summary>
public sealed class Interp1dFill
{
    private Interp1dFill(double[]? below, double[]? above, bool extrapolate)
    {
        Below = below; Above = above; Extrapolate = extrapolate;
    }

    internal double[]? Below { get; }
    internal double[]? Above { get; }

    /// <summary>Whether this is the "extrapolate" fill.</summary>
    public bool Extrapolate { get; }

    /// <summary>The same constant below and above (NaN by default).</summary>
    public static Interp1dFill Constant(double value) => new([value], [value], false);

    /// <summary>Separate constants below and above the data range.</summary>
    public static Interp1dFill BelowAbove(double below, double above) => new([below], [above], false);

    /// <summary>Per-component constants below and above the data range.</summary>
    public static Interp1dFill BelowAbove(double[] below, double[] above) => new((double[])below.Clone(), (double[])above.Clone(), false);

    /// <summary>Extrapolate (scipy's <c>fill_value="extrapolate"</c>).</summary>
    public static Interp1dFill ExtrapolateFill { get; } = new(null, null, true);

    /// <summary>A scalar converts to <see cref="Constant"/>.</summary>
    public static implicit operator Interp1dFill(double value) => Constant(value);
}

/// <summary>
/// scipy.interpolate.interp1d (legacy, kept for compatibility): a callable interpolant along
/// one axis of y. Splines kinds use <see cref="Interpolate.MakeInterpSpline(double[], NdArray, int, double[], SplineBoundary?, int)"/>.
/// </summary>
/// <remarks>As in scipy, out-of-range points raise unless <c>boundsError: false</c> is given or
/// the fill is "extrapolate"; <c>axis</c> defaults to the last axis.</remarks>
public sealed class Interp1d
{
    private readonly double[] _x;
    private readonly double[] _y;   // (n, nt)
    private readonly int _n, _nt;
    private readonly int[] _rest;
    private readonly int _order;    // -1 for the non-spline kinds
    private readonly Interp1dKind _kind;
    private readonly BSpline? _spline;
    private readonly bool _nanSpline;
    private readonly double[] _below, _above;
    private readonly bool _extrapolate;
    private readonly bool _boundsError;
    private readonly double[]? _xBounds;
    private readonly double[]? _xShift;

    /// <summary>Interpolation axis of y.</summary>
    public int Axis { get; }

    /// <summary>Sorted x.</summary>
    public double[] X => (double[])_x.Clone();

    /// <summary>Scalar data.</summary>
    public Interp1d(double[] x, double[] y, Interp1dKind kind = Interp1dKind.Linear, bool? boundsError = null,
        Interp1dFill? fillValue = null, bool assumeSorted = false)
        : this(x, new NdArray((double[])(y ?? throw new ArgumentNullException(nameof(y))).Clone(), y.Length), kind, -1, boundsError, fillValue, assumeSorted, null) { }

    /// <summary>Scalar data with an integer spline order (scipy's <c>kind=int</c>).</summary>
    public Interp1d(double[] x, double[] y, int order, bool? boundsError = null, Interp1dFill? fillValue = null, bool assumeSorted = false)
        : this(x, new NdArray((double[])(y ?? throw new ArgumentNullException(nameof(y))).Clone(), y.Length), Interp1dKind.Slinear, -1, boundsError, fillValue, assumeSorted, order) { }

    /// <summary>N-D data along <paramref name="axis"/> (default: last axis, as scipy).</summary>
    public Interp1d(double[] x, NdArray y, Interp1dKind kind = Interp1dKind.Linear, int axis = -1, bool? boundsError = null,
        Interp1dFill? fillValue = null, bool assumeSorted = false)
        : this(x, y, kind, axis, boundsError, fillValue, assumeSorted, null) { }

    private Interp1d(double[] x, NdArray y, Interp1dKind kind, int axis, bool? boundsError, Interp1dFill? fillValue, bool assumeSorted, int? order)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        Axis = NormalizeAxis(axis, y.Rank);
        var yy = MoveAxisToFront(y, Axis, out _n, out _rest);
        _nt = Product(_rest);
        if (x.Length != _n) throw new ArgumentException("x and y arrays must be equal in length along interpolation axis.");
        var xs = (double[])x.Clone();
        if (!assumeSorted)
        {
            var idx = Enumerable.Range(0, xs.Length).ToArray();
            Array.Sort(idx, (p, q) => { var c = xs[p].CompareTo(xs[q]); return c != 0 ? c : p.CompareTo(q); });
            var sx = new double[_n];
            var sy = new double[yy.Length];
            for (var i = 0; i < _n; i++)
            {
                sx[i] = xs[idx[i]];
                Array.Copy(yy, idx[i] * _nt, sy, i * _nt, _nt);
            }
            xs = sx; yy = sy;
        }
        _x = xs; _y = yy;
        if (order is int o)
        {
            if (o < 0) throw new ArgumentException("Spline order must be non-negative.");
            _order = o;
        }
        else
        {
            _order = kind switch
            {
                Interp1dKind.Zero => 0,
                Interp1dKind.Slinear => 1,
                Interp1dKind.Quadratic => 2,
                Interp1dKind.Cubic => 3,
                _ => -1,
            };
        }
        _kind = kind;
        var fill = fillValue ?? Interp1dFill.Constant(double.NaN);
        var minval = 1;
        if (_order < 0)
        {
            if (kind is Interp1dKind.Nearest or Interp1dKind.NearestUp)
            {
                _xBounds = new double[_n - 1];
                for (var i = 0; i < _n - 1; i++) _xBounds[i] = _x[i + 1] / 2.0 + _x[i] / 2.0;
            }
            else if (kind == Interp1dKind.Previous)
            {
                _xShift = _x.Select(v => Math.BitDecrement(v)).ToArray();
                if (fill.Extrapolate)
                {
                    if (boundsError == true) throw new ArgumentException("Cannot extrapolate and raise at the same time.");
                    boundsError = false;
                    fill = Interp1dFill.BelowAbove(Enumerable.Repeat(double.NaN, _nt).ToArray(), _y[((_n - 1) * _nt)..(_n * _nt)]);
                }
            }
            else if (kind == Interp1dKind.Next)
            {
                _xShift = _x.Select(v => Math.BitIncrement(v)).ToArray();
                if (fill.Extrapolate)
                {
                    if (boundsError == true) throw new ArgumentException("Cannot extrapolate and raise at the same time.");
                    boundsError = false;
                    fill = Interp1dFill.BelowAbove(_y[..(_nt)], Enumerable.Repeat(double.NaN, _nt).ToArray());
                }
            }
        }
        else
        {
            minval = _order + 1;
            if (_n >= minval)
            {
                var hasNan = _x.Any(double.IsNaN) || _y.Any(double.IsNaN);
                if (hasNan && _order > 1)
                {
                    // scipy: any NaN makes a quadratic/cubic interp1d return NaN everywhere.
                    _nanSpline = true;
                    var lin = new double[_n];
                    for (var i = 0; i < _n; i++) lin[i] = i;
                    _spline = Interpolate.MakeInterpSpline(lin, new NdArray(Enumerable.Repeat(1.0, _y.Length).ToArray(), [_n, .. _rest]), _order);
                }
                else _spline = MakeSplineUnchecked(_x, _y, _order);
            }
        }
        if (_n < minval) throw new ArgumentException($"x and y arrays must have at least {minval} entries");
        if (fill.Extrapolate)
        {
            if (boundsError == true) throw new ArgumentException("Cannot extrapolate and raise at the same time.");
            _boundsError = false;
            _extrapolate = true;
            _below = _above = [];
        }
        else
        {
            _below = Broadcast(fill.Below!, "fill_value (below)");
            _above = Broadcast(fill.Above!, "fill_value (above)");
            _boundsError = boundsError ?? true;
        }
    }

    private BSpline MakeSplineUnchecked(double[] x, double[] y, int k)
    {
        var yy = new NdArray(y, [_n, .. _rest]);
        if (k <= 1)
        {
            // make_interp_spline's k=0/1 shortcuts accept any finite or NaN data.
            double[] t = k == 0 ? [.. x, x[^1]] : [x[0], .. x, x[^1]];
            return BSpline.Raw(t, y, _n, _rest, k, InterpolateExtrapolate.Extrapolate, 0);
        }
        return Interpolate.MakeInterpSpline(x, yy, k);
    }

    private double[] Broadcast(double[] v, string name)
    {
        if (v.Length == _nt) return v;
        if (v.Length == 1) return Enumerable.Repeat(v[0], _nt).ToArray();
        throw new ArgumentException($"{name} argument must be able to broadcast up to the value shape.");
    }

    /// <summary>Interpolated value at x (scalar data).</summary>
    public double Evaluate(double x)
    {
        if (_nt != 1) throw new InvalidOperationException("Vector-valued data: use the NdArray overload.");
        var r = new double[1];
        Core([x], r);
        return r[0];
    }

    /// <summary>Interpolated values at x (scalar data).</summary>
    public double[] Evaluate(double[] x)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (_nt != 1) throw new InvalidOperationException("Vector-valued data: use the NdArray overload.");
        var r = new double[x.Length];
        Core(x, r);
        return r;
    }

    /// <summary>Interpolated values at x of any shape; the x dimensions replace the
    /// interpolation axis of y.</summary>
    public NdArray Evaluate(NdArray x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var xs = x.ToArray();
        var r = new double[xs.Length * _nt];
        Core(xs, r);
        return PlaceResult(r, x.Shape, _rest, Axis);
    }

    private void Core(double[] xnew, double[] output)
    {
        var nt = _nt;
        if (!_extrapolate && _boundsError)
        {
            foreach (var v in xnew)
            {
                if (v < _x[0]) throw new ArgumentException($"A value ({v}) in x_new is below the interpolation range's minimum value ({_x[0]}).");
                if (v > _x[^1]) throw new ArgumentException($"A value ({v}) in x_new is above the interpolation range's maximum value ({_x[^1]}).");
            }
        }
        if (_order >= 0)
        {
            var spl = _spline!.Evaluate(new NdArray(xnew, xnew.Length)).ToArray();
            if (_nanSpline) Array.Fill(spl, double.NaN);
            Array.Copy(spl, output, spl.Length);
        }
        else
        {
            switch (_kind)
            {
                case Interp1dKind.Linear:
                    if (nt == 1 && !_extrapolate)
                    {
                        var r = Interpolate.Interp(xnew, _x, _y);
                        Array.Copy(r, output, r.Length);
                    }
                    else
                    {
                        for (var i = 0; i < xnew.Length; i++)
                        {
                            var idx = Math.Clamp(SearchLeft(_x, xnew[i]), 1, _n - 1);
                            int lo = idx - 1, hi = idx;
                            var dxv = _x[hi] - _x[lo];
                            for (var q = 0; q < nt; q++)
                            {
                                var slope = (_y[hi * nt + q] - _y[lo * nt + q]) / dxv;
                                output[i * nt + q] = slope * (xnew[i] - _x[lo]) + _y[lo * nt + q];
                            }
                        }
                    }
                    break;
                case Interp1dKind.Nearest:
                case Interp1dKind.NearestUp:
                    for (var i = 0; i < xnew.Length; i++)
                    {
                        var idx = _kind == Interp1dKind.Nearest ? SearchLeft(_xBounds!, xnew[i]) : SearchRight(_xBounds!, xnew[i]);
                        idx = Math.Clamp(idx, 0, _n - 1);
                        Array.Copy(_y, idx * nt, output, i * nt, nt);
                    }
                    break;
                case Interp1dKind.Previous:
                    for (var i = 0; i < xnew.Length; i++)
                    {
                        var idx = Math.Clamp(SearchLeft(_xShift!, xnew[i]), 1, _n);
                        Array.Copy(_y, (idx - 1) * nt, output, i * nt, nt);
                    }
                    break;
                case Interp1dKind.Next:
                    for (var i = 0; i < xnew.Length; i++)
                    {
                        var idx = Math.Clamp(SearchRight(_xShift!, xnew[i]), 0, _n - 1);
                        Array.Copy(_y, idx * nt, output, i * nt, nt);
                    }
                    break;
            }
        }
        if (!_extrapolate)
        {
            for (var i = 0; i < xnew.Length; i++)
            {
                if (xnew[i] < _x[0]) Array.Copy(_below, 0, output, i * nt, nt);
                else if (xnew[i] > _x[^1]) Array.Copy(_above, 0, output, i * nt, nt);
            }
        }
    }
}
