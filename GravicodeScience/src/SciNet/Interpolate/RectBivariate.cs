using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.InterpolateSupport;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Bivariate spline on a rectangular grid (scipy.interpolate.RectBivariateSpline), s = 0
/// (interpolation) only.
/// </summary>
/// <remarks>
/// <para>With s = 0 FITPACK's <c>regrid</c> puts the interior knots at the data (odd degree)
/// or at midpoints (even degree), exactly the not-a-knot choice, and the tensor-product
/// collocation system separates into one banded solve per axis — which is what is done here,
/// so values agree with scipy to rounding. Smoothing (s &gt; 0, FITPACK <c>fpregr</c>) is not
/// implemented and throws.</para>
/// <para>As in FITPACK, evaluation clamps coordinates into the knot box rather than
/// extrapolating, for values and derivatives alike.</para>
/// </remarks>
public sealed class RectBivariateSpline
{
    private readonly double[] _tx, _ty;
    private readonly double[] _c; // (nx, ny) row-major
    private readonly int _kx, _ky;

    /// <summary>Fits z (shape (len(x), len(y))) with degrees kx, ky; bbox = [xb, xe, yb, ye]
    /// defaults to the data range.</summary>
    public RectBivariateSpline(double[] x, double[] y, NdArray z, double[]? bbox = null, int kx = 3, int ky = 3, double s = 0)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        ArgumentNullException.ThrowIfNull(z);
        if (s != 0) throw new NotSupportedException("RectBivariateSpline is implemented for s = 0 (interpolation) only.");
        if (kx < 1 || kx > 5 || ky < 1 || ky > 5) throw new ArgumentException("1 <= kx, ky <= 5 must hold.");
        var xs = CopyFinite(x, "x");
        var ys = CopyFinite(y, "y");
        RequireStrictlyIncreasing(xs, "x");
        RequireStrictlyIncreasing(ys, "y");
        if (z.Rank != 2 || z.Shape[0] != xs.Length || z.Shape[1] != ys.Length) throw new ArgumentException("z must have shape (len(x), len(y)).");
        if (xs.Length <= kx || ys.Length <= ky) throw new ArgumentException("x and y need more than k points.");
        var b = bbox ?? [xs[0], xs[^1], ys[0], ys[^1]];
        if (b.Length != 4) throw new ArgumentException("bbox must have 4 entries.");
        if (b[0] > xs[0] || b[1] < xs[^1] || b[2] > ys[0] || b[3] < ys[^1]) throw new ArgumentException("bbox must contain the data.");
        _kx = kx; _ky = ky;
        _tx = Knots(xs, kx, b[0], b[1]);
        _ty = Knots(ys, ky, b[2], b[3]);
        var c = z.ToArray();
        SolveAxis(c, xs, _tx, kx, 1, xs.Length, ys.Length);
        SolveAxis(c, ys, _ty, ky, xs.Length, ys.Length, 1);
        _c = c;
    }

    private RectBivariateSpline(double[] tx, double[] ty, double[] c, int kx, int ky)
    {
        _tx = tx; _ty = ty; _c = c; _kx = kx; _ky = ky;
    }

    private static double[] Knots(double[] x, int k, double xb, double xe)
    {
        var t = Interpolate.NotAKnotKnots(x, k);
        for (var i = 0; i <= k; i++) { t[i] = xb; t[^(i + 1)] = xe; }
        return t;
    }

    private static void SolveAxis(double[] c, double[] x, double[] t, int k, int outer, int n, int inner)
    {
        var lu = new BandedLu(n, k, k);
        var h = new double[k + 1];
        var w = new double[k + 1];
        var hint = k;
        for (var j = 0; j < n; j++)
        {
            var l = FindSplineInterval(t, k, x[j], hint, true);
            hint = l;
            DeBoorD(t, x[j], k, l, 0, h, w);
            for (var a = 0; a <= k; a++) lu[j, l - k + a] = h[a];
        }
        lu.Factor();
        for (var o = 0; o < outer; o++) lu.Solve(c.AsSpan(o * n * inner, n * inner), inner);
    }

    /// <summary>Knots (tx, ty).</summary>
    public (double[] Tx, double[] Ty) GetKnots() => ((double[])_tx.Clone(), (double[])_ty.Clone());

    /// <summary>Coefficients, row-major (len(tx)-kx-1, len(ty)-ky-1) flattened.</summary>
    public double[] GetCoeffs() => (double[])_c.Clone();

    /// <summary>Residual sum of squares (0 for interpolation).</summary>
    public double GetResidual() => 0.0;

    /// <summary>Degrees (kx, ky).</summary>
    public (int Kx, int Ky) Degrees => (_kx, _ky);

    private int Nx => _tx.Length - _kx - 1;
    private int Ny => _ty.Length - _ky - 1;

    /// <summary>Value (or partial derivative) at one point; coordinates are clamped into the
    /// knot box.</summary>
    public double Ev(double x, double y, int dx = 0, int dy = 0)
    {
        if (dx < 0 || dy < 0) throw new ArgumentException("Derivative orders must be non-negative.");
        if (dx > _kx || dy > _ky) return 0.0;
        x = Math.Clamp(x, _tx[_kx], _tx[Nx]);
        y = Math.Clamp(y, _ty[_ky], _ty[Ny]);
        Span<double> hx = stackalloc double[_kx + 1];
        Span<double> hy = stackalloc double[_ky + 1];
        Span<double> w = stackalloc double[Math.Max(_kx, _ky) + 1];
        var lx = FindSplineInterval(_tx, _kx, x, _kx, true);
        var ly = FindSplineInterval(_ty, _ky, y, _ky, true);
        DeBoorD(_tx, x, _kx, lx, dx, hx, w);
        DeBoorD(_ty, y, _ky, ly, dy, hy, w);
        var s = 0.0;
        var ny = Ny;
        for (var a = 0; a <= _kx; a++)
        {
            var row = (lx - _kx + a) * ny + ly - _ky;
            var acc = 0.0;
            for (var b = 0; b <= _ky; b++) acc += _c[row + b] * hy[b];
            s += hx[a] * acc;
        }
        return s;
    }

    /// <summary>Values at the points (xi[i], yi[i]) (scipy's <c>ev</c>).</summary>
    public double[] Ev(double[] xi, double[] yi, int dx = 0, int dy = 0)
    {
        if (xi.Length != yi.Length) throw new ArgumentException("xi and yi must have the same length.");
        var r = new double[xi.Length];
        ForChunks(xi.Length, 32, (s, e) => { for (var i = s; i < e; i++) r[i] = Ev(xi[i], yi[i], dx, dy); });
        return r;
    }

    /// <summary>Values on the grid x × y, shape (len(x), len(y)) (scipy's <c>__call__</c>
    /// with grid=True; x and y must be ascending).</summary>
    public NdArray Evaluate(double[] x, double[] y, int dx = 0, int dy = 0)
    {
        RequireNonDecreasing(x, "x");
        RequireNonDecreasing(y, "y");
        var r = new double[x.Length * y.Length];
        ForChunks(x.Length, 32L * y.Length, (s, e) =>
        {
            for (var i = s; i < e; i++)
                for (var j = 0; j < y.Length; j++) r[i * y.Length + j] = Ev(x[i], y[j], dx, dy);
        });
        return new NdArray(r, x.Length, y.Length);
    }

    /// <summary>Double integral over [xa, xb] × [ya, yb], clipped to the knot box (FITPACK
    /// <c>dblint</c>).</summary>
    public double Integral(double xa, double xb, double ya, double yb)
    {
        var ix = BasisIntegrals(_tx, _kx, xa, xb);
        var iy = BasisIntegrals(_ty, _ky, ya, yb);
        var s = 0.0;
        var ny = Ny;
        for (var i = 0; i < Nx; i++)
        {
            var acc = 0.0;
            for (var j = 0; j < ny; j++) acc += _c[i * ny + j] * iy[j];
            s += ix[i] * acc;
        }
        return s;
    }

    private static double[] BasisIntegrals(double[] t, int k, double a, double b)
    {
        var n = t.Length - k - 1;
        var eye = new double[n * n];
        for (var i = 0; i < n; i++) eye[i * n + i] = 1;
        var spl = BSpline.Raw(t, eye, n, [n], k, InterpolateExtrapolate.None, 0);
        return spl.IntegrateValues(a, b);
    }

    /// <summary>The (dx, dy) partial derivative as a new spline of degrees (kx-dx, ky-dy).</summary>
    public RectBivariateSpline PartialDerivative(int dx, int dy)
    {
        if (dx < 0 || dy < 0) throw new ArgumentException("Derivative orders must be non-negative.");
        if (dx >= _kx || dy >= _ky) throw new ArgumentException($"order of derivative must be less than the degree ({_kx}, {_ky}).");
        int nx = Nx, ny = Ny;
        // along x: treat c as (nx, ny) with ny columns
        var cx = new double[_tx.Length * ny];
        Array.Copy(_c, cx, _c.Length);
        var (tx, c1, kx) = BSpline.SplderCore(_tx, cx, ny, _kx, dx);
        var nx2 = tx.Length - kx - 1;
        // transpose to (ny, nx2) to differentiate along y
        var ct = new double[_ty.Length * nx2];
        for (var i = 0; i < nx2; i++)
            for (var j = 0; j < ny; j++) ct[j * nx2 + i] = c1[i * ny + j];
        var (ty, c2, ky) = BSpline.SplderCore(_ty, ct, nx2, _ky, dy);
        var ny2 = ty.Length - ky - 1;
        var c = new double[nx2 * ny2];
        for (var i = 0; i < nx2; i++)
            for (var j = 0; j < ny2; j++) c[i * ny2 + j] = c2[j * nx2 + i];
        return new RectBivariateSpline(tx, ty, c, kx, ky);
    }

    internal static RectBivariateSpline FromTck(double[] tx, double[] ty, double[] c, int kx, int ky)
    {
        var nx = tx.Length - kx - 1;
        var ny = ty.Length - ky - 1;
        if (c.Length < nx * ny) throw new ArgumentException("c is too short for the knots.");
        return new RectBivariateSpline((double[])tx.Clone(), (double[])ty.Clone(), c[..(nx * ny)], kx, ky);
    }
}

public static partial class Interpolate
{
    /// <summary>
    /// Evaluates a bivariate B-spline (tx, ty, c, kx, ky) or its partial derivative on the grid
    /// x × y (scipy.interpolate.bisplev); result shape (len(x), len(y)). Coordinates are clamped
    /// into the knot box, as FITPACK does.
    /// </summary>
    public static NdArray Bisplev(double[] x, double[] y, (double[] Tx, double[] Ty, double[] C, int Kx, int Ky) tck, int dx = 0, int dy = 0)
    {
        var (tx, ty, c, kx, ky) = tck;
        if (dx < 0 || (dx > 0 && dx >= kx)) throw new ArgumentException($"0 <= dx = {dx} < kx = {kx} must hold");
        if (dy < 0 || (dy > 0 && dy >= ky)) throw new ArgumentException($"0 <= dy = {dy} < ky = {ky} must hold");
        var spl = RectBivariateSpline.FromTck(tx, ty, c, kx, ky);
        var r = new double[x.Length * y.Length];
        for (var i = 0; i < x.Length; i++)
            for (var j = 0; j < y.Length; j++) r[i * y.Length + j] = spl.Ev(x[i], y[j], dx, dy);
        return new NdArray(r, x.Length, y.Length);
    }

    /// <summary>
    /// Weighted least-squares bivariate spline on scattered data with given full knot vectors
    /// (scipy.interpolate.bisplrep with task=-1). Rank-deficient systems get the minimum-norm
    /// solution, as FITPACK's <c>fprank</c> does. Automatic knot placement (task=0, s ≥ 0, FITPACK
    /// <c>surfit</c>) is not implemented.
    /// </summary>
    public static (double[] Tx, double[] Ty, double[] C, int Kx, int Ky) Bisplrep(double[] x, double[] y, double[] z, double[] tx, double[] ty,
        double[]? w = null, int kx = 3, int ky = 3)
    {
        ArgumentNullException.ThrowIfNull(tx);
        ArgumentNullException.ThrowIfNull(ty);
        var m = x.Length;
        if (y.Length != m || z.Length != m) throw new ArgumentException("len(x)==len(y)==len(z) must hold.");
        if (m < (kx + 1) * (ky + 1)) throw new ArgumentException("m >= (kx+1)(ky+1) must hold");
        var ww = w ?? Enumerable.Repeat(1.0, m).ToArray();
        int nx = tx.Length - kx - 1, ny = ty.Length - ky - 1;
        var a = new double[m * nx * ny];
        var b = new double[m];
        Span<double> hx = stackalloc double[kx + 1];
        Span<double> hy = stackalloc double[ky + 1];
        Span<double> wk = stackalloc double[Math.Max(kx, ky) + 1];
        for (var p = 0; p < m; p++)
        {
            var xv = Math.Clamp(x[p], tx[kx], tx[nx]);
            var yv = Math.Clamp(y[p], ty[ky], ty[ny]);
            var lx = FindSplineInterval(tx, kx, xv, kx, true);
            var ly = FindSplineInterval(ty, ky, yv, ky, true);
            DeBoorD(tx, xv, kx, lx, 0, hx, wk);
            DeBoorD(ty, yv, ky, ly, 0, hy, wk);
            for (var i = 0; i <= kx; i++)
                for (var j = 0; j <= ky; j++)
                    a[p * nx * ny + (lx - kx + i) * ny + (ly - ky + j)] = ww[p] * hx[i] * hy[j];
            b[p] = ww[p] * z[p];
        }
        var sol = Linalg.Lstsq(new NdArray(a, m, nx * ny), new NdArray(b, m)).X.ToArray();
        return ((double[])tx.Clone(), (double[])ty.Clone(), sol, kx, ky);
    }
}
