using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

/// <summary>Method of <see cref="RegularGridInterpolator"/> and <see cref="Interpolate.Interpn"/>.</summary>
public enum RegularGridMethod
{
    /// <summary>Multilinear.</summary>
    Linear,
    /// <summary>Nearest grid point (ties go to the lower index).</summary>
    Nearest,
    /// <summary>Tensor-product linear B-spline (equal to Linear inside the grid).</summary>
    Slinear,
    /// <summary>Tensor-product not-a-knot cubic spline.</summary>
    Cubic,
    /// <summary>Tensor-product not-a-knot quintic spline.</summary>
    Quintic,
    /// <summary>PCHIP applied one dimension at a time (last dimension first, as scipy).</summary>
    Pchip,
    /// <summary>interpn only: FITPACK bicubic spline (2-D, <see cref="RectBivariateSpline"/>).</summary>
    Splinef2d,
}

/// <summary>
/// Tensor-product B-spline in n dimensions (scipy.interpolate.NdBSpline): knots and degree per
/// dimension, coefficients of shape (n_1, …, n_d, …trailing).
/// </summary>
public sealed class NdBSpline
{
    private readonly double[][] _t;
    private readonly int[] _k;
    private readonly double[] _c;
    private readonly int[] _nc;     // coefficient count per dimension
    private readonly int[] _rest;
    private readonly int _nt;

    /// <summary>Whether evaluation outside the base box extrapolates (otherwise NaN).</summary>
    public bool Extrapolate { get; set; }

    /// <summary>Creates the spline; c has shape (len(t_d) - k_d - 1 for each d, …trailing).</summary>
    public NdBSpline(double[][] t, NdArray c, int[] k, bool extrapolate = true)
    {
        ArgumentNullException.ThrowIfNull(t);
        ArgumentNullException.ThrowIfNull(c);
        ArgumentNullException.ThrowIfNull(k);
        var d = t.Length;
        if (k.Length == 1 && d > 1) k = Enumerable.Repeat(k[0], d).ToArray();
        if (k.Length != d) throw new ArgumentException("len(t) must equal len(k).");
        if (c.Rank < d) throw new ArgumentException("Coefficients must have at least ndim dimensions.");
        _t = t.Select(v => (double[])v.Clone()).ToArray();
        _k = (int[])k.Clone();
        _nc = new int[d];
        for (var i = 0; i < d; i++)
        {
            _nc[i] = _t[i].Length - _k[i] - 1;
            if (_nc[i] < _k[i] + 1) throw new ArgumentException($"Too few knots in dimension {i}.");
            if (c.Shape[i] != _nc[i]) throw new ArgumentException($"Coefficient dimension {i} must have {_nc[i]} entries.");
            RequireNonDecreasing(_t[i], "t");
        }
        _rest = c.Shape[d..].ToArray();
        _nt = Product(_rest);
        _c = c.ToArray();
        Extrapolate = extrapolate;
    }

    internal NdBSpline(double[][] t, int[] k, double[] c, int[] rest, bool extrapolate)
    {
        _t = t; _k = k; _c = c; _rest = rest; _nt = Product(rest);
        _nc = new int[t.Length];
        for (var i = 0; i < t.Length; i++) _nc[i] = t[i].Length - k[i] - 1;
        Extrapolate = extrapolate;
    }

    /// <summary>Number of dimensions.</summary>
    public int NDim => _t.Length;

    /// <summary>Knots per dimension (copies).</summary>
    public double[][] T => _t.Select(v => (double[])v.Clone()).ToArray();

    /// <summary>Degree per dimension.</summary>
    public int[] K => (int[])_k.Clone();

    /// <summary>Coefficients.</summary>
    public NdArray C => new((double[])_c.Clone(), [.. _nc, .. _rest]);

    /// <summary>Shape of the value at one point.</summary>
    public int[] ValueShape => (int[])_rest.Clone();

    /// <summary>
    /// Interpolating tensor-product spline of degree k through gridded values (scipy's
    /// <c>make_ndbspl</c>) with not-a-knot knots per dimension. The collocation matrix is a
    /// Kronecker product, so the coefficients come from one banded solve per dimension —
    /// exact, where scipy solves the full system iteratively (gcrotmk, atol 1e-6).
    /// </summary>
    public static NdBSpline FromGrid(double[][] points, NdArray values, int k = 3)
    {
        var d = points.Length;
        var shape = values.Shape.ToArray();
        for (var i = 0; i < d; i++)
        {
            if (points[i].Length <= k) throw new ArgumentException($"There are {points[i].Length} points in dimension {i}, but order {k} requires at least {k + 1} points per dimension.");
            if (shape[i] != points[i].Length) throw new ArgumentException($"There are {points[i].Length} points and {shape[i]} values in dimension {i}");
        }
        var rest = shape[d..];
        var c = values.ToArray();
        var t = new double[d][];
        var dims = shape[..d];
        var nt = Product(rest);
        var h = new double[k + 1];
        var w = new double[k + 1];
        for (var dim = 0; dim < d; dim++)
        {
            var x = points[dim];
            t[dim] = Interpolate.NotAKnotKnots(x, k);
            if (k <= 1) continue; // k = 0/1: coefficients are the values
            var n = x.Length;
            var lu = new BandedLu(n, k, k);
            var hint = k;
            for (var j = 0; j < n; j++)
            {
                var l = FindSplineInterval(t[dim], k, x[j], hint, false);
                hint = l;
                DeBoorD(t[dim], x[j], k, l, 0, h, w);
                for (var a = 0; a <= k; a++) lu[j, l - k + a] = h[a];
            }
            lu.Factor();
            var outer = 1;
            for (var i = 0; i < dim; i++) outer *= dims[i];
            var inner = nt;
            for (var i = dim + 1; i < d; i++) inner *= dims[i];
            var block = n * inner;
            for (var o = 0; o < outer; o++) lu.Solve(c.AsSpan(o * block, block), inner);
        }
        return new NdBSpline(t, Enumerable.Repeat(k, d).ToArray(), c, rest, true);
    }

    /// <summary>Evaluates at m points given as an (m, ndim) array; nu holds derivative orders
    /// per dimension. Result (m, nt) flattened.</summary>
    internal double[] EvaluateFlat(double[] xi, int m, int[]? nu, bool? extrapolate = null)
    {
        var d = NDim;
        var ext = extrapolate ?? Extrapolate;
        var nuv = nu ?? new int[d];
        if (nuv.Length != d) throw new ArgumentException("nu must have one entry per dimension.");
        for (var i = 0; i < d; i++) if (nuv[i] < 0) throw new ArgumentException("Derivative orders must be non-negative.");
        var res = new double[m * _nt];
        var cost = 1L;
        for (var i = 0; i < d; i++) cost *= _k[i] + 1;
        ForChunks(m, cost * _nt + 16L * d, (s, e) =>
        {
            var kmax = _k.Max();
            var h = new double[d * (kmax + 1)];
            var work = new double[kmax + 1];
            var l = new int[d];
            var hints = new int[d];
            for (var i = 0; i < d; i++) hints[i] = _k[i];
            var strides = new int[d];
            var st = _nt;
            for (var i = d - 1; i >= 0; i--) { strides[i] = st; st *= _nc[i]; }
            var idx = new int[d];
            for (var p = s; p < e; p++)
            {
                var bad = false;
                for (var i = 0; i < d; i++)
                {
                    var v = xi[p * d + i];
                    var li = FindSplineInterval(_t[i], _k[i], v, hints[i], ext);
                    if (li < 0) { bad = true; break; }
                    hints[i] = li;
                    l[i] = li;
                    if (nuv[i] > _k[i]) { Array.Clear(h, i * (kmax + 1), _k[i] + 1); continue; }
                    DeBoorD(_t[i], v, _k[i], li, nuv[i], h.AsSpan(i * (kmax + 1), _k[i] + 1), work);
                }
                var o = res.AsSpan(p * _nt, _nt);
                if (bad) { o.Fill(double.NaN); continue; }
                Array.Clear(idx);
                // iterate the (k+1)^d corner offsets
                while (true)
                {
                    var wgt = 1.0;
                    var off = 0;
                    for (var i = 0; i < d; i++)
                    {
                        wgt *= h[i * (kmax + 1) + idx[i]];
                        off += (l[i] - _k[i] + idx[i]) * strides[i];
                    }
                    if (wgt != 0)
                        for (var q = 0; q < _nt; q++) o[q] += wgt * _c[off + q];
                    var dd = d - 1;
                    while (dd >= 0)
                    {
                        idx[dd]++;
                        if (idx[dd] <= _k[dd]) break;
                        idx[dd] = 0;
                        dd--;
                    }
                    if (dd < 0) break;
                }
            }
        });
        return res;
    }

    /// <summary>Evaluates at points xi of shape (…, ndim); result shape (…, …ValueShape).</summary>
    public NdArray Evaluate(NdArray xi, int[]? nu = null, bool? extrapolate = null)
    {
        var (flat, m, lead) = RegularGridInterpolator.SplitPoints(xi, NDim);
        return new NdArray(EvaluateFlat(flat, m, nu, extrapolate), [.. lead, .. _rest]);
    }

    /// <summary>Evaluates a scalar spline at the rows of an (m, ndim) array.</summary>
    public double[] Evaluate(double[,] xi, int[]? nu = null, bool? extrapolate = null)
    {
        if (_nt != 1) throw new InvalidOperationException("Vector-valued spline: use the NdArray overload.");
        if (xi.GetLength(1) != NDim) throw new ArgumentException($"Points must have {NDim} coordinates.");
        var flat = new double[xi.Length];
        Buffer.BlockCopy(xi, 0, flat, 0, xi.Length * sizeof(double));
        return EvaluateFlat(flat, xi.GetLength(0), nu, extrapolate);
    }
}

/// <summary>
/// Interpolation on a rectilinear grid in any number of dimensions
/// (scipy.interpolate.RegularGridInterpolator).
/// </summary>
/// <remarks>
/// <para>The linear path locates each coordinate with an O(1)-per-step hinted search, forms the
/// 2^d corner weights on the stack and allocates nothing per point; batches are threaded.
/// Spline methods are exact tensor-product interpolants (see <see cref="NdBSpline.FromGrid"/>);
/// scipy reaches the same function through an iterative solver with atol 1e-6, so the two
/// agree to about that accuracy, not to rounding. scipy's <c>*_legacy</c> spline methods
/// produce the same interpolant mathematically and are not separate options here.</para>
/// <para>Grids may be strictly descending per dimension (they are flipped). Values may carry
/// trailing dimensions beyond the grid's.</para>
/// </remarks>
public sealed class RegularGridInterpolator
{
    private readonly double[][] _grid;
    private readonly double[] _values;
    private readonly int[] _shape;  // grid shape
    private readonly int[] _rest;
    private readonly int _nt;
    private NdBSpline?[] _splines = new NdBSpline?[3];

    /// <summary>Default method.</summary>
    public RegularGridMethod Method { get; }

    /// <summary>Throw for points outside the grid (otherwise fill or extrapolate).</summary>
    public bool BoundsError { get; set; }

    /// <summary>Value for outside points when <see cref="BoundsError"/> is false; null
    /// extrapolates.</summary>
    public double? FillValue { get; set; }

    /// <summary>Creates the interpolator. <paramref name="values"/> has shape
    /// (len(points[0]), …, len(points[d-1]), …trailing).</summary>
    public RegularGridInterpolator(double[][] points, NdArray values, RegularGridMethod method = RegularGridMethod.Linear,
        bool boundsError = true, double? fillValue = double.NaN)
    {
        ArgumentNullException.ThrowIfNull(points);
        ArgumentNullException.ThrowIfNull(values);
        if (method == RegularGridMethod.Splinef2d) throw new ArgumentException("Method 'splinef2d' is only available through interpn.");
        var d = points.Length;
        if (d > values.Rank) throw new ArgumentException($"There are {d} point arrays, but values has {values.Rank} dimensions");
        var vals = values.ToArray();
        var vshape = values.Shape.ToArray();
        _grid = new double[d][];
        var flips = new List<int>();
        for (var i = 0; i < d; i++)
        {
            var p = CopyFinite(points[i], "points");
            if (p.Length != vshape[i]) throw new ArgumentException($"There are {p.Length} points and {vshape[i]} values in dimension {i}");
            var asc = true; var desc = true;
            for (var j = 1; j < p.Length; j++) { if (!(p[j] > p[j - 1])) asc = false; if (!(p[j] < p[j - 1])) desc = false; }
            if (!asc)
            {
                if (!desc || p.Length < 2) throw new ArgumentException($"The points in dimension {i} must be strictly ascending or descending");
                Array.Reverse(p);
                flips.Add(i);
            }
            _grid[i] = p;
        }
        if (flips.Count > 0) vals = Flip(vals, vshape, flips);
        _values = vals;
        _shape = vshape[..d];
        _rest = vshape[d..];
        _nt = Product(_rest);
        Method = method;
        BoundsError = boundsError;
        FillValue = fillValue;
        if (method is RegularGridMethod.Slinear or RegularGridMethod.Cubic or RegularGridMethod.Quintic) _ = Spline(method);
        if (method == RegularGridMethod.Pchip)
            for (var i = 0; i < d; i++)
                if (_grid[i].Length <= 3) throw new ArgumentException($"There are {_grid[i].Length} points in dimension {i}, but method pchip requires at least 4 points per dimension.");
    }

    private static double[] Flip(double[] vals, int[] shape, List<int> flips)
    {
        var res = new double[vals.Length];
        var nd = shape.Length;
        var strides = new int[nd];
        var s = 1;
        for (var i = nd - 1; i >= 0; i--) { strides[i] = s; s *= shape[i]; }
        var idx = new int[nd];
        for (var lin = 0; lin < vals.Length; lin++)
        {
            var rem = lin;
            var target = 0;
            for (var i = 0; i < nd; i++)
            {
                idx[i] = rem / strides[i];
                rem %= strides[i];
                var j = flips.Contains(i) ? shape[i] - 1 - idx[i] : idx[i];
                target += j * strides[i];
            }
            res[target] = vals[lin];
        }
        return res;
    }

    /// <summary>Grid coordinates (ascending; copies).</summary>
    public double[][] Grid => _grid.Select(g => (double[])g.Clone()).ToArray();

    private NdBSpline Spline(RegularGridMethod m)
    {
        var (slot, k) = m switch
        {
            RegularGridMethod.Slinear => (0, 1),
            RegularGridMethod.Cubic => (1, 3),
            _ => (2, 5),
        };
        var s = Volatile.Read(ref _splines[slot]);
        if (s is not null) return s;
        s = NdBSpline.FromGrid(_grid, new NdArray(_values, [.. _shape, .. _rest]), k);
        Interlocked.CompareExchange(ref _splines[slot], s, null);
        return _splines[slot]!;
    }

    internal static (double[] Flat, int M, int[] Lead) SplitPoints(NdArray xi, int d)
    {
        ArgumentNullException.ThrowIfNull(xi);
        var shape = xi.Shape.ToArray();
        if (d == 1 && (shape.Length == 1 || shape[^1] != 1))
            return (xi.ToArray(), xi.Size, shape);
        if (shape[^1] != d) throw new ArgumentException($"The requested sample points xi have dimension {shape[^1]} but this interpolator has dimension {d}");
        return (xi.ToArray(), xi.Size / d, shape[..^1]);
    }

    /// <summary>Interpolates at points of shape (…, ndim); result shape (…, …trailing).</summary>
    public NdArray Evaluate(NdArray xi, RegularGridMethod? method = null, int[]? nu = null)
    {
        var (flat, m, lead) = SplitPoints(xi, _grid.Length);
        return new NdArray(EvaluateFlat(flat, m, method ?? Method, nu), [.. lead, .. _rest]);
    }

    /// <summary>Interpolates scalar values at the rows of an (m, ndim) array.</summary>
    public double[] Evaluate(double[,] xi, RegularGridMethod? method = null, int[]? nu = null)
    {
        if (_nt != 1) throw new InvalidOperationException("Vector-valued data: use the NdArray overload.");
        if (xi.GetLength(1) != _grid.Length) throw new ArgumentException($"Points must have {_grid.Length} coordinates.");
        var flat = new double[xi.Length];
        Buffer.BlockCopy(xi, 0, flat, 0, xi.Length * sizeof(double));
        return EvaluateFlat(flat, xi.GetLength(0), method ?? Method, nu);
    }

    internal double[] EvaluateFlat(double[] xi, int m, RegularGridMethod method, int[]? nu)
    {
        var d = _grid.Length;
        if (nu is not null && method is not (RegularGridMethod.Slinear or RegularGridMethod.Cubic or RegularGridMethod.Quintic))
            throw new ArgumentException("Can only compute derivatives for methods slinear, cubic and quintic.");
        if (BoundsError)
        {
            for (var p = 0; p < m; p++)
                for (var i = 0; i < d; i++)
                {
                    var v = xi[p * d + i];
                    if (!(v >= _grid[i][0] && v <= _grid[i][^1]) && !double.IsNaN(v))
                        throw new ArgumentException($"One of the requested xi is out of bounds in dimension {i}");
                }
        }
        double[] res;
        switch (method)
        {
            case RegularGridMethod.Linear: res = EvaluateLinear(xi, m); break;
            case RegularGridMethod.Nearest: res = EvaluateNearest(xi, m); break;
            case RegularGridMethod.Pchip: res = EvaluatePchip(xi, m); break;
            case RegularGridMethod.Splinef2d: throw new ArgumentException("Method 'splinef2d' is only available through interpn.");
            default: res = Spline(method).EvaluateFlat(xi, m, nu, true); break;
        }
        // Out-of-bounds fill and NaN propagation.
        if (!BoundsError && FillValue is double fv)
        {
            for (var p = 0; p < m; p++)
                for (var i = 0; i < d; i++)
                {
                    var v = xi[p * d + i];
                    if (v < _grid[i][0] || v > _grid[i][^1]) { res.AsSpan(p * _nt, _nt).Fill(fv); break; }
                }
        }
        for (var p = 0; p < m; p++)
            for (var i = 0; i < d; i++)
                if (double.IsNaN(xi[p * d + i])) { res.AsSpan(p * _nt, _nt).Fill(double.NaN); break; }
        return res;
    }

    private double[] EvaluateLinear(double[] xi, int m)
    {
        var d = _grid.Length;
        var nt = _nt;
        var res = new double[m * nt];
        var strides = new int[d];
        var st = nt;
        for (var i = d - 1; i >= 0; i--) { strides[i] = st; st *= _shape[i]; }
        ForChunks(m, (2L << Math.Min(d, 20)) * nt + 8L * d, (s, e) =>
        {
            Span<int> idx = stackalloc int[d];
            Span<double> dist = stackalloc double[d];
            Span<int> hints = stackalloc int[d];
            for (var p = s; p < e; p++)
            {
                for (var i = 0; i < d; i++)
                {
                    var v = xi[p * d + i];
                    var g = _grid[i];
                    var j = g.Length == 1 ? 0 : FindIntervalAscending(g, v, hints[i], true);
                    if (j < 0) j = 0;
                    hints[i] = j;
                    idx[i] = j;
                    dist[i] = g.Length == 1 ? 0 : (v - g[j]) / (g[j + 1] - g[j]);
                }
                var o = res.AsSpan(p * nt, nt);
                if (d == 2 && nt == 1)
                {
                    int i0 = idx[0], i1 = idx[1];
                    double t0 = dist[0], t1 = dist[1];
                    var v00 = _values[i0 * strides[0] + i1];
                    var v01 = _values[i0 * strides[0] + i1 + 1];
                    var v10 = _values[(i0 + 1) * strides[0] + i1];
                    var v11 = _values[(i0 + 1) * strides[0] + i1 + 1];
                    o[0] = v00 * (1 - t0) * (1 - t1) + v01 * (1 - t0) * t1 + v10 * t0 * (1 - t1) + v11 * t0 * t1;
                    continue;
                }
                var corners = 1 << d;
                for (var cMask = 0; cMask < corners; cMask++)
                {
                    var w = 1.0;
                    var off = 0;
                    for (var i = 0; i < d; i++)
                    {
                        var up = (cMask >> (d - 1 - i)) & 1;
                        if (_shape[i] == 1) { if (up == 1) { w = 0; break; } off += idx[i] * strides[i]; continue; }
                        w *= up == 1 ? dist[i] : 1 - dist[i];
                        off += (idx[i] + up) * strides[i];
                    }
                    for (var q = 0; q < nt; q++) o[q] += w * _values[off + q];
                }
            }
        });
        return res;
    }

    private double[] EvaluateNearest(double[] xi, int m)
    {
        var d = _grid.Length;
        var nt = _nt;
        var res = new double[m * nt];
        var strides = new int[d];
        var st = nt;
        for (var i = d - 1; i >= 0; i--) { strides[i] = st; st *= _shape[i]; }
        ForChunks(m, 8L * d + nt, (s, e) =>
        {
            for (var p = s; p < e; p++)
            {
                var off = 0;
                for (var i = 0; i < d; i++)
                {
                    var v = xi[p * d + i];
                    var g = _grid[i];
                    if (g.Length == 1) continue;
                    var j = FindIntervalAscending(g, v, 0, true);
                    if (j < 0) j = 0;
                    var t = (v - g[j]) / (g[j + 1] - g[j]);
                    off += (t <= 0.5 ? j : j + 1) * strides[i];
                }
                Array.Copy(_values, off, res, p * nt, nt);
            }
        });
        return res;
    }

    private double[] EvaluatePchip(double[] xi, int m)
    {
        var d = _grid.Length;
        var nt = _nt;
        var res = new double[m * nt];
        var strides = new int[d];
        var st = nt;
        for (var i = d - 1; i >= 0; i--) { strides[i] = st; st *= _shape[i]; }
        ForChunks(m, 1000L * d, (s, e) =>
        {
            for (var p = s; p < e; p++)
                for (var q = 0; q < nt; q++)
                    res[p * nt + q] = PchipFold(0, 0, xi.AsSpan(p * d, d), strides, q);
        });
        return res;
    }

    /// <summary>PCHIP along dimension dim of the function given recursively by the inner
    /// dimensions; only the nodes the local slopes need are evaluated.</summary>
    private double PchipFold(int dim, int offset, ReadOnlySpan<double> x, int[] strides, int q)
    {
        if (dim == _grid.Length) return _values[offset + q];
        var g = _grid[dim];
        var n = g.Length;
        var xv = x[dim];
        var xs = x.ToArray();
        var cache = new Dictionary<int, double>();
        double Y(int j)
        {
            if (!cache.TryGetValue(j, out var v)) cache[j] = v = PchipFold(dim + 1, offset + j * strides[dim], xs, strides, q);
            return v;
        }
        var i = FindIntervalAscending(g, xv, 0, true);
        if (i < 0) return double.NaN;
        double H(int j) => g[j + 1] - g[j];
        double Mk(int j) => (Y(j + 1) - Y(j)) / H(j);
        double Slope(int j)
        {
            if (j == 0) return PchipEdge(H(0), H(1), Mk(0), Mk(1));
            if (j == n - 1) return PchipEdge(H(n - 2), H(n - 3), Mk(n - 2), Mk(n - 3));
            double m0 = Mk(j - 1), m1 = Mk(j);
            if (Math.Sign(m1) != Math.Sign(m0) || m1 == 0 || m0 == 0) return 0;
            var w1 = 2 * H(j) + H(j - 1);
            var w2 = H(j) + 2 * H(j - 1);
            return 1.0 / ((w1 / m0 + w2 / m1) / (w1 + w2));
        }
        double y0 = Y(i), d0 = Slope(i), d1 = Slope(i + 1);
        var h = H(i);
        var slope = (Y(i + 1) - y0) / h;
        var t = (d0 + d1 - 2 * slope) / h;
        var c0 = t / h;
        var c1 = (slope - d0) / h - t;
        var sx = xv - g[i];
        return ((c0 * sx + c1) * sx + d0) * sx + y0;
    }

    private static double PchipEdge(double h0, double h1, double m0, double m1)
    {
        var dd = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
        if (Math.Sign(dd) != Math.Sign(m0)) return 0.0;
        if (Math.Sign(m0) != Math.Sign(m1) && Math.Abs(dd) > 3.0 * Math.Abs(m0)) return 3.0 * m0;
        return dd;
    }
}

public static partial class Interpolate
{
    /// <summary>
    /// Multidimensional interpolation on a regular grid (scipy.interpolate.interpn): points of
    /// shape (…, ndim). <see cref="RegularGridMethod.Splinef2d"/> uses a FITPACK bicubic spline
    /// on 2-D data and fills (never extrapolates) outside the grid.
    /// </summary>
    public static NdArray Interpn(double[][] points, NdArray values, NdArray xi, RegularGridMethod method = RegularGridMethod.Linear,
        bool boundsError = true, double? fillValue = double.NaN)
    {
        if (method != RegularGridMethod.Splinef2d)
            return new RegularGridInterpolator(points, values, method, boundsError, fillValue).Evaluate(xi);
        if (values.Rank != 2 || points.Length != 2) throw new ArgumentException("The method splinef2d can only be used for 2-dimensional input data");
        if (!boundsError && fillValue is null) throw new ArgumentException("The method splinef2d does not support extrapolation.");
        var (flat, m, lead) = RegularGridInterpolator.SplitPoints(xi, 2);
        var gx = points[0]; var gy = points[1];
        double x0 = Math.Min(gx[0], gx[^1]), x1 = Math.Max(gx[0], gx[^1]), y0 = Math.Min(gy[0], gy[^1]), y1 = Math.Max(gy[0], gy[^1]);
        for (var p = 0; p < m; p++)
            if (boundsError && !(flat[2 * p] >= x0 && flat[2 * p] <= x1 && flat[2 * p + 1] >= y0 && flat[2 * p + 1] <= y1))
                throw new ArgumentException("One of the requested xi is out of bounds");
        var spl = new RectBivariateSpline(gx, gy, values);
        var res = new double[m];
        for (var p = 0; p < m; p++)
        {
            double a = flat[2 * p], b = flat[2 * p + 1];
            res[p] = a >= x0 && a <= x1 && b >= y0 && b <= y1 ? spl.Ev(a, b) : fillValue ?? double.NaN;
        }
        return new NdArray(res, lead.Length == 0 ? [1] : lead);
    }
}
