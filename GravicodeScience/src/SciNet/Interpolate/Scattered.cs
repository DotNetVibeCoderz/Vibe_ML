using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

/// <summary>Method of <see cref="Interpolate.Griddata(NdArray, NdArray, NdArray, GriddataMethod, double, bool)"/>.</summary>
public enum GriddataMethod
{
    /// <summary>Value of the nearest data point.</summary>
    Nearest,
    /// <summary>Piecewise linear on the Delaunay triangulation.</summary>
    Linear,
    /// <summary>Clough–Tocher C¹ cubic (2-D), or a cubic spline in 1-D.</summary>
    Cubic,
}

/// <summary>Common input handling of the scattered-data interpolators (scipy's
/// <c>NDInterpolatorBase</c>): points (n, d), values (n, …), optional rescaling of the points
/// to the unit cube.</summary>
public abstract class ScatteredInterpolatorBase
{
    private protected readonly double[] _points; // (n, d), rescaled if requested
    private protected readonly double[] _values; // (n, nt)
    private protected readonly int _n, _d, _nt;
    private protected readonly int[] _rest;
    private protected readonly double[]? _offset, _scale;

    private protected ScatteredInterpolatorBase(NdArray points, NdArray values, bool rescale)
    {
        ArgumentNullException.ThrowIfNull(points);
        ArgumentNullException.ThrowIfNull(values);
        if (points.Rank != 2) throw new ArgumentException("points must have shape (n, ndim).");
        _n = points.Shape[0];
        _d = points.Shape[1];
        if (values.Shape[0] != _n) throw new ArgumentException("different number of values and points");
        _points = CopyFinite(points.ToArray(), "points");
        _values = values.ToArray();
        _rest = values.Shape[1..].ToArray();
        _nt = Product(_rest);
        if (rescale)
        {
            _offset = new double[_d];
            _scale = new double[_d];
            for (var j = 0; j < _d; j++)
            {
                double s = 0, lo = double.PositiveInfinity, hi = double.NegativeInfinity;
                for (var i = 0; i < _n; i++)
                {
                    var v = _points[i * _d + j];
                    s += v; lo = Math.Min(lo, v); hi = Math.Max(hi, v);
                }
                _offset[j] = s / _n;
                var sc = hi - lo;
                _scale[j] = sc > 0 ? sc : 1.0;
                for (var i = 0; i < _n; i++) _points[i * _d + j] = (_points[i * _d + j] - _offset[j]) / _scale[j];
            }
        }
    }

    /// <summary>Number of dimensions.</summary>
    public int NDim => _d;

    private protected double[] ScaleQueries(NdArray xi, out int m, out int[] lead)
    {
        var shape = xi.Shape.ToArray();
        if (shape[^1] != _d) throw new ArgumentException($"number of dimensions in xi does not match x");
        var q = xi.ToArray();
        m = q.Length / _d;
        lead = shape[..^1];
        if (lead.Length == 0) lead = [1];
        if (_offset is not null)
            for (var i = 0; i < m; i++)
                for (var j = 0; j < _d; j++) q[i * _d + j] = (q[i * _d + j] - _offset[j]) / _scale![j];
        return q;
    }

    /// <summary>Interpolates at points of shape (…, ndim); result shape (…, …value shape).</summary>
    public NdArray Evaluate(NdArray xi)
    {
        ArgumentNullException.ThrowIfNull(xi);
        var q = ScaleQueries(xi, out var m, out var lead);
        return new NdArray(EvaluateFlat(q, m), [.. lead, .. _rest]);
    }

    /// <summary>Interpolates scalar values at the rows of an (m, ndim) array.</summary>
    public double[] Evaluate(double[,] xi)
    {
        if (_nt != 1) throw new InvalidOperationException("Vector-valued data: use the NdArray overload.");
        return Evaluate(NdArray.FromArray(xi)).ToArray();
    }

    private protected abstract double[] EvaluateFlat(double[] q, int m);
}

/// <summary>Nearest-neighbour interpolation in N dimensions through a k-d tree
/// (scipy.interpolate.NearestNDInterpolator). Equidistant ties may resolve to a different
/// point than scipy's cKDTree.</summary>
public sealed class NearestNDInterpolator : ScatteredInterpolatorBase
{
    private readonly KDTree _tree;

    /// <summary>Builds the tree over (rescaled) points.</summary>
    public NearestNDInterpolator(NdArray points, NdArray values, bool rescale = false)
        : base(points, values, rescale)
    {
        _tree = new KDTree(new NdArray(_points, _n, _d));
    }

    /// <summary>Scalar-valued convenience constructor.</summary>
    public NearestNDInterpolator(double[,] points, double[] values, bool rescale = false)
        : this(NdArray.FromArray(points), new NdArray((double[])values.Clone(), values.Length), rescale) { }

    /// <summary>Upper bound on the distance to a usable neighbour (scipy's
    /// <c>distance_upper_bound</c> query option); farther queries give NaN.</summary>
    public double DistanceUpperBound { get; set; } = double.PositiveInfinity;

    private protected override double[] EvaluateFlat(double[] q, int m)
    {
        var (dist, idx) = _tree.Query(new NdArray(q, m, _d), 1, distanceUpperBound: DistanceUpperBound);
        var dd = dist.ToArray();
        var res = new double[m * _nt];
        for (var i = 0; i < m; i++)
        {
            if (!double.IsFinite(dd[i])) { res.AsSpan(i * _nt, _nt).Fill(double.NaN); continue; }
            Array.Copy(_values, idx[i, 0] * _nt, res, i * _nt, _nt);
        }
        return res;
    }
}

/// <summary>
/// Piecewise-linear interpolation on the Delaunay triangulation
/// (scipy.interpolate.LinearNDInterpolator), in any dimension ≥ 2. Points outside the convex
/// hull get <see cref="FillValue"/>. Inside, the value is the barycentric combination of the
/// containing simplex's vertex values, so for points in general position (a unique Delaunay
/// triangulation) it agrees with scipy to rounding; on cospherical data the triangulation, and
/// so the interpolant, can legitimately differ.
/// </summary>
public class LinearNDInterpolator : ScatteredInterpolatorBase
{
    private protected readonly Delaunay _tri;
    private protected readonly int[] _simplices; // (ns, d + 1)
    private protected readonly double[] _transform; // (ns, d + 1, d)

    /// <summary>Value outside the convex hull.</summary>
    public double FillValue { get; set; }

    /// <summary>Triangulates the (rescaled) points.</summary>
    public LinearNDInterpolator(NdArray points, NdArray values, double fillValue = double.NaN, bool rescale = false)
        : base(points, values, rescale)
    {
        if (_d < 2) throw new NotSupportedException("LinearNDInterpolator needs points of dimension 2 or more (use Interp1d in 1-D).");
        _tri = new Delaunay(new NdArray((double[])_points.Clone(), _n, _d));
        FillValue = fillValue;
        _simplices = (int[])_tri.TriangleBuffer.Clone();
        _transform = _tri.TransformBuffer;
    }

    /// <summary>Scalar-valued convenience constructor.</summary>
    public LinearNDInterpolator(double[,] points, double[] values, double fillValue = double.NaN, bool rescale = false)
        : this(NdArray.FromArray(points), new NdArray((double[])values.Clone(), values.Length), fillValue, rescale) { }

    /// <summary>The triangulation used.</summary>
    public Delaunay Triangulation => _tri;

    private protected void Barycentric(int s, double x, double y, Span<double> c)
    {
        var o = 6 * s;
        double dx = x - _transform[o + 4], dy = y - _transform[o + 5];
        c[0] = _transform[o] * dx + _transform[o + 1] * dy;
        c[1] = _transform[o + 2] * dx + _transform[o + 3] * dy;
        c[2] = 1 - c[0] - c[1];
    }

    private protected int[] Locate(double[] q, int m) => _tri.FindSimplex(new NdArray(q, m, _d));

    private protected override double[] EvaluateFlat(double[] q, int m)
    {
        var simplex = Locate(q, m);
        var nt = _nt;
        var res = new double[m * nt];
        if (_d != 2)
        {
            var d = _d;
            ForChunks(m, 8 + (d + 1L) * (d + nt), (s, e) =>
            {
                Span<double> c = stackalloc double[d + 1];
                for (var i = s; i < e; i++)
                {
                    var o = res.AsSpan(i * nt, nt);
                    var t = simplex[i];
                    if (t < 0) { o.Fill(FillValue); continue; }
                    _tri.BarycentricInto(_transform, t, q.AsSpan(i * d, d), c);
                    for (var k = 0; k <= d; k++)
                    {
                        var v = _simplices[t * (d + 1) + k];
                        for (var j = 0; j < nt; j++) o[j] += c[k] * _values[v * nt + j];
                    }
                }
            });
            return res;
        }
        ForChunks(m, 8 + 3L * nt, (s, e) =>
        {
            Span<double> c = stackalloc double[3];
            for (var i = s; i < e; i++)
            {
                var o = res.AsSpan(i * nt, nt);
                var t = simplex[i];
                if (t < 0) { o.Fill(FillValue); continue; }
                Barycentric(t, q[2 * i], q[2 * i + 1], c);
                for (var k = 0; k < 3; k++)
                {
                    var v = _simplices[t * 3 + k];
                    for (var j = 0; j < nt; j++) o[j] += c[k] * _values[v * nt + j];
                }
            }
        });
        return res;
    }
}

/// <summary>
/// Piecewise cubic, C¹, curvature-minimising interpolation in 2-D
/// (scipy.interpolate.CloughTocher2DInterpolator; Alfeld 1984, Nielson 1983).
/// </summary>
/// <remarks>
/// <para>Gradients at the vertices come from scipy's global estimate: Gauss–Seidel sweeps
/// that, vertex by vertex, minimise the squared second derivative of the interpolant
/// integrated along the edges, stopping when the largest relative change is below
/// <c>tol</c>. Each triangle then carries a Clough–Tocher split into three cubic Bézier
/// patches, with the cross-boundary derivative direction taken towards the neighbour's
/// centroid so the interpolant is affinely invariant and C¹ — the same choices as scipy, so
/// values agree with scipy to the gradient tolerance.</para>
/// </remarks>
public sealed class CloughTocher2DInterpolator : LinearNDInterpolator
{
    private readonly double[] _grad; // (n, nt, 2)

    /// <summary>Builds the triangulation and estimates gradients (tol 1e-6, at most 400
    /// sweeps, as scipy).</summary>
    public CloughTocher2DInterpolator(NdArray points, NdArray values, double fillValue = double.NaN, double tol = 1e-6, int maxiter = 400, bool rescale = false)
        : base(points, values, fillValue, rescale)
    {
        if (_d != 2) throw new NotSupportedException("CloughTocher2DInterpolator is for 2-D points only.");
        _grad = new double[_n * _nt * 2];
        var (indptr, indices) = _tri.VertexNeighborVertices();
        var col = new double[_n];
        var y = new double[2 * _n];
        for (var j = 0; j < _nt; j++)
        {
            for (var i = 0; i < _n; i++) col[i] = _values[i * _nt + j];
            Array.Clear(y);
            EstimateGradients(_points, col, indptr, indices, maxiter, tol, y);
            for (var i = 0; i < _n; i++) { _grad[(i * _nt + j) * 2] = y[2 * i]; _grad[(i * _nt + j) * 2 + 1] = y[2 * i + 1]; }
        }
    }

    /// <summary>Scalar-valued convenience constructor.</summary>
    public CloughTocher2DInterpolator(double[,] points, double[] values, double fillValue = double.NaN, double tol = 1e-6, int maxiter = 400, bool rescale = false)
        : this(NdArray.FromArray(points), new NdArray((double[])values.Clone(), values.Length), fillValue, tol, maxiter, rescale) { }

    /// <summary>Number of sweeps the gradient estimate took (0 = not converged).</summary>
    public int GradientIterations { get; private set; }

    /// <summary>Estimated gradient at every data point, shape (n, …value shape, 2).</summary>
    public NdArray Gradients => new((double[])_grad.Clone(), [_n, .. _rest, 2]);

    private void EstimateGradients(double[] p, double[] data, int[] indptr, int[] indices, int maxiter, double tol, double[] y)
    {
        Span<double> Q = stackalloc double[4];
        Span<double> s = stackalloc double[2];
        for (var iter = 0; iter < maxiter; iter++)
        {
            var err = 0.0;
            for (var ip = 0; ip < _n; ip++)
            {
                Q.Clear(); s.Clear();
                for (var jp = indptr[ip]; jp < indptr[ip + 1]; jp++)
                {
                    var ip2 = indices[jp];
                    var ex = p[2 * ip2] - p[2 * ip];
                    var ey = p[2 * ip2 + 1] - p[2 * ip + 1];
                    var L = Math.Sqrt(ex * ex + ey * ey);
                    var L3 = L * L * L;
                    double f1 = data[ip], f2 = data[ip2];
                    var df2 = -ex * y[2 * ip2] - ey * y[2 * ip2 + 1];
                    Q[0] += 4 * ex * ex / L3;
                    Q[1] += 4 * ex * ey / L3;
                    Q[3] += 4 * ey * ey / L3;
                    s[0] += (6 * (f1 - f2) - 2 * df2) * ex / L3;
                    s[1] += (6 * (f1 - f2) - 2 * df2) * ey / L3;
                }
                Q[2] = Q[1];
                var det = Q[0] * Q[3] - Q[1] * Q[2];
                var r0 = (Q[3] * s[0] - Q[1] * s[1]) / det;
                var r1 = (-Q[2] * s[0] + Q[0] * s[1]) / det;
                var change = Math.Max(Math.Abs(y[2 * ip] + r0), Math.Abs(y[2 * ip + 1] + r1));
                y[2 * ip] = -r0;
                y[2 * ip + 1] = -r1;
                change /= Math.Max(1.0, Math.Max(Math.Abs(r0), Math.Abs(r1)));
                err = Math.Max(err, change);
            }
            if (err < tol) { GradientIterations = iter + 1; return; }
        }
        GradientIterations = 0;
    }

    private protected override double[] EvaluateFlat(double[] q, int m)
    {
        var simplex = Locate(q, m);
        var nt = _nt;
        var res = new double[m * nt];
        var nbr = _tri.Neighbors;
        ForChunks(m, 120L * nt, (s, e) =>
        {
            Span<double> b = stackalloc double[3];
            Span<double> f = stackalloc double[3];
            Span<double> df = stackalloc double[6];
            for (var i = s; i < e; i++)
            {
                var o = res.AsSpan(i * nt, nt);
                var t = simplex[i];
                if (t < 0) { o.Fill(FillValue); continue; }
                Barycentric(t, q[2 * i], q[2 * i + 1], b);
                for (var j = 0; j < nt; j++)
                {
                    for (var k = 0; k < 3; k++)
                    {
                        var v = _simplices[t * 3 + k];
                        f[k] = _values[v * nt + j];
                        df[2 * k] = _grad[(v * nt + j) * 2];
                        df[2 * k + 1] = _grad[(v * nt + j) * 2 + 1];
                    }
                    o[j] = CloughTocherSingle(t, b, f, df, nbr);
                }
            }
        });
        return res;
    }

    private double CloughTocherSingle(int t, ReadOnlySpan<double> b, ReadOnlySpan<double> f, ReadOnlySpan<double> df, int[,] nbr)
    {
        var p = _points;
        int v0 = _simplices[3 * t], v1 = _simplices[3 * t + 1], v2 = _simplices[3 * t + 2];
        double e12x = p[2 * v1] - p[2 * v0], e12y = p[2 * v1 + 1] - p[2 * v0 + 1];
        double e23x = p[2 * v2] - p[2 * v1], e23y = p[2 * v2 + 1] - p[2 * v1 + 1];
        double e31x = p[2 * v0] - p[2 * v2], e31y = p[2 * v0 + 1] - p[2 * v2 + 1];
        double f1 = f[0], f2 = f[1], f3 = f[2];
        var df12 = +(df[0] * e12x + df[1] * e12y);
        var df21 = -(df[2] * e12x + df[3] * e12y);
        var df23 = +(df[2] * e23x + df[3] * e23y);
        var df32 = -(df[4] * e23x + df[5] * e23y);
        var df31 = +(df[4] * e31x + df[5] * e31y);
        var df13 = -(df[0] * e31x + df[1] * e31y);
        var c3000 = f1;
        var c2100 = (df12 + 3 * c3000) / 3;
        var c2010 = (df13 + 3 * c3000) / 3;
        var c0300 = f2;
        var c1200 = (df21 + 3 * c0300) / 3;
        var c0210 = (df23 + 3 * c0300) / 3;
        var c0030 = f3;
        var c1020 = (df31 + 3 * c0030) / 3;
        var c0120 = (df32 + 3 * c0030) / 3;
        var c2001 = (c2100 + c2010 + c3000) / 3;
        var c0201 = (c1200 + c0300 + c0210) / 3;
        var c0021 = (c1020 + c0120 + c0030) / 3;
        Span<double> g = stackalloc double[3];
        Span<double> c = stackalloc double[3];
        for (var k = 0; k < 3; k++)
        {
            var it = nbr[t, k];
            if (it == -1) { g[k] = -0.5; continue; }
            int a0 = _simplices[3 * it], a1 = _simplices[3 * it + 1], a2 = _simplices[3 * it + 2];
            var yx = (p[2 * a0] + p[2 * a1] + p[2 * a2]) / 3;
            var yy = (p[2 * a0 + 1] + p[2 * a1 + 1] + p[2 * a2 + 1]) / 3;
            Barycentric(t, yx, yy, c);
            g[k] = k switch
            {
                0 => (2 * c[2] + c[1] - 1) / (2 - 3 * c[2] - 3 * c[1]),
                1 => (2 * c[0] + c[2] - 1) / (2 - 3 * c[0] - 3 * c[2]),
                _ => (2 * c[1] + c[0] - 1) / (2 - 3 * c[1] - 3 * c[0]),
            };
        }
        var c0111 = (g[0] * (-c0300 + 3 * c0210 - 3 * c0120 + c0030) + (-c0300 + 2 * c0210 - c0120 + c0021 + c0201)) / 2;
        var c1011 = (g[1] * (-c0030 + 3 * c1020 - 3 * c2010 + c3000) + (-c0030 + 2 * c1020 - c2010 + c2001 + c0021)) / 2;
        var c1101 = (g[2] * (-c3000 + 3 * c2100 - 3 * c1200 + c0300) + (-c3000 + 2 * c2100 - c1200 + c2001 + c0201)) / 2;
        var c1002 = (c1101 + c1011 + c2001) / 3;
        var c0102 = (c1101 + c0111 + c0201) / 3;
        var c0012 = (c1011 + c0111 + c0021) / 3;
        var c0003 = (c1002 + c0102 + c0012) / 3;
        var minval = Math.Min(b[0], Math.Min(b[1], b[2]));
        double b1 = b[0] - minval, b2 = b[1] - minval, b3 = b[2] - minval, b4 = 3 * minval;
        return b1 * b1 * b1 * c3000 + 3 * b1 * b1 * b2 * c2100 + 3 * b1 * b1 * b3 * c2010 +
               3 * b1 * b1 * b4 * c2001 + 3 * b1 * b2 * b2 * c1200 +
               6 * b1 * b2 * b4 * c1101 + 3 * b1 * b3 * b3 * c1020 + 6 * b1 * b3 * b4 * c1011 +
               3 * b1 * b4 * b4 * c1002 + b2 * b2 * b2 * c0300 + 3 * b2 * b2 * b3 * c0210 +
               3 * b2 * b2 * b4 * c0201 + 3 * b2 * b3 * b3 * c0120 + 6 * b2 * b3 * b4 * c0111 +
               3 * b2 * b4 * b4 * c0102 + b3 * b3 * b3 * c0030 + 3 * b3 * b3 * b4 * c0021 +
               3 * b3 * b4 * b4 * c0012 + b4 * b4 * b4 * c0003;
    }
}

public static partial class Interpolate
{
    /// <summary>
    /// Interpolates unstructured data (scipy.interpolate.griddata). points (n, d), values (n, …),
    /// xi (…, d). 1-D data go through <see cref="Interp1d"/> (nearest extrapolates, as scipy);
    /// linear uses the N-d Delaunay triangulation in any dimension, cubic is 2-D only
    /// (Clough–Tocher), nearest works in any dimension.
    /// </summary>
    public static NdArray Griddata(NdArray points, NdArray values, NdArray xi, GriddataMethod method = GriddataMethod.Linear,
        double fillValue = double.NaN, bool rescale = false)
    {
        ArgumentNullException.ThrowIfNull(points);
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(xi);
        var d = points.Rank < 2 ? 1 : points.Shape[1];
        if (d == 1)
        {
            var pts = points.ToArray();
            var kind = method switch { GriddataMethod.Nearest => Interp1dKind.Nearest, GriddataMethod.Linear => Interp1dKind.Linear, _ => Interp1dKind.Cubic };
            var fill = method == GriddataMethod.Nearest ? Interp1dFill.ExtrapolateFill : Interp1dFill.Constant(fillValue);
            var ip = new Interp1d(pts, values, kind, axis: 0, boundsError: false, fillValue: fill);
            var q = xi.Rank >= 2 && xi.Shape[^1] == 1 ? xi.Reshape(xi.Shape[..^1].ToArray() is { Length: > 0 } s ? s : [1]) : xi;
            return ip.Evaluate(q);
        }
        return method switch
        {
            GriddataMethod.Nearest => new NearestNDInterpolator(points, values, rescale).Evaluate(xi),
            GriddataMethod.Linear => new LinearNDInterpolator(points, values, fillValue, rescale).Evaluate(xi),
            GriddataMethod.Cubic when d == 2 => new CloughTocher2DInterpolator(points, values, fillValue, rescale: rescale).Evaluate(xi),
            _ => throw new ArgumentException($"Unknown interpolation method {method} for {d} dimensional data"),
        };
    }

    /// <summary>Scalar 2-D convenience overload: points (n, 2), values (n), xi (m, 2).</summary>
    public static double[] Griddata(double[,] points, double[] values, double[,] xi, GriddataMethod method = GriddataMethod.Linear,
        double fillValue = double.NaN, bool rescale = false) =>
        Griddata(NdArray.FromArray(points), new NdArray((double[])values.Clone(), values.Length), NdArray.FromArray(xi), method, fillValue, rescale).ToArray();
}
