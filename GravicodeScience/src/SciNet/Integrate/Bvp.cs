using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>The C1 cubic spline solution of a boundary value problem (scipy's PPoly sol).</summary>
public sealed class BvpSolution
{
    private readonly double[] _x;
    private readonly double[] _c; // per interval j, per component i: 4 coefficients (c0 dx^3 + c1 dx^2 + c2 dx + c3)
    private readonly int _n;

    internal BvpSolution(double[] x, double[] c, int n)
    {
        _x = x; _c = c; _n = n;
    }

    /// <summary>System dimension.</summary>
    public int N => _n;

    /// <summary>Mesh nodes (breakpoints).</summary>
    public IReadOnlyList<double> X => _x;

    private int Interval(double xv)
    {
        var m = _x.Length;
        if (xv <= _x[0]) return 0;
        if (xv >= _x[m - 1]) return m - 2;
        int lo = 0, hi = m - 1;
        while (hi - lo > 1)
        {
            var mid = (lo + hi) >> 1;
            if (_x[mid] <= xv) lo = mid; else hi = mid;
        }
        return lo;
    }

    /// <summary>Evaluates the nu-th derivative (0-3) at x into a buffer; extrapolates outside the mesh.</summary>
    public void Evaluate(double x, Span<double> y, int nu = 0)
    {
        var j = Interval(x);
        var dx = x - _x[j];
        for (var i = 0; i < _n; i++)
        {
            var o = (j * _n + i) * 4;
            double c0 = _c[o], c1 = _c[o + 1], c2 = _c[o + 2], c3 = _c[o + 3];
            y[i] = nu switch
            {
                0 => ((c0 * dx + c1) * dx + c2) * dx + c3,
                1 => (3 * c0 * dx + 2 * c1) * dx + c2,
                2 => 6 * c0 * dx + 2 * c1,
                3 => 6 * c0,
                _ => 0.0,
            };
        }
    }

    /// <summary>Solution (or its nu-th derivative) at x.</summary>
    public double[] Evaluate(double x, int nu = 0)
    {
        var y = new double[_n];
        Evaluate(x, y, nu);
        return y;
    }

    /// <summary>Solution at many points, shape (n, len(xs)).</summary>
    public NdArray Evaluate(IReadOnlyList<double> xs, int nu = 0)
    {
        var m = xs.Count;
        var d = new double[_n * m];
        Span<double> tmp = stackalloc double[_n];
        for (var j = 0; j < m; j++)
        {
            Evaluate(xs[j], tmp, nu);
            for (var i = 0; i < _n; i++) d[i * m + j] = tmp[i];
        }
        return new NdArray(d, _n, m);
    }
}

/// <summary>Result of <see cref="Integrate.SolveBvp(Func{double[], NdArray, double[], NdArray}, Func{double[], double[], double[], double[]}, double[], NdArray, double[], double, int, double?)"/>.</summary>
public sealed class BvpResult
{
    /// <summary>Continuous solution.</summary>
    public required BvpSolution Sol { get; init; }
    /// <summary>Found unknown parameters (empty when none).</summary>
    public required double[] P { get; init; }
    /// <summary>Final mesh.</summary>
    public required double[] X { get; init; }
    /// <summary>Solution at the mesh, shape (n, m).</summary>
    public required NdArray Y { get; init; }
    /// <summary>Derivative at the mesh, shape (n, m).</summary>
    public required NdArray Yp { get; init; }
    /// <summary>Relative RMS residual per mesh interval.</summary>
    public required double[] RmsResiduals { get; init; }
    /// <summary>Iterations (Newton solve + mesh refinement rounds).</summary>
    public int Niter { get; init; }
    /// <summary>0 converged; 1 max nodes exceeded; 2 singular Jacobian; 3 boundary-condition tolerance not met.</summary>
    public int Status { get; init; }
    /// <summary>Termination reason.</summary>
    public string Message { get; init; } = "";
    /// <summary>Status 0.</summary>
    public bool Success => Status == 0;
}

public static partial class Integrate
{
    /// <summary>
    /// scipy.integrate.solve_bvp: 4th-order collocation (Lobatto IIIA, Kierzenka and Shampine's
    /// BVP4C scheme) with a damped Newton method and mesh refinement on the relative RMS residual,
    /// following scipy's _bvp.py step for step (finite-difference Jacobians, affine-invariant
    /// Armijo line search, fixed-Jacobian iterations after full steps, 1 or 2 nodes inserted per
    /// interval). The Newton system is not handed to a general sparse LU: the boundary rows coupling
    /// y(a) and y(b), and the unknown parameters p, are made banded by carrying copies z ≡ y(a) and
    /// q ≡ p along the mesh (z' = 0, p' = 0, the standard trick for non-separated conditions). The
    /// augmented system has the same solution and bandwidth O(n + k), so a banded LU with partial
    /// pivoting solves it in O(m (n + k)^3) — 10^4 nodes are routine. The singular term S of scipy
    /// is not supported.
    /// </summary>
    /// <param name="fun">f(x, y, p): x of length m, y of shape (n, m), returns (n, m).</param>
    /// <param name="bc">Boundary residuals bc(ya, yb, p), length n + k.</param>
    /// <param name="x">Initial mesh, strictly increasing.</param>
    /// <param name="y">Initial guess at the mesh, shape (n, m).</param>
    /// <param name="p">Initial guess for the k unknown parameters, or null.</param>
    /// <param name="tol">Tolerance on the relative RMS collocation residual.</param>
    /// <param name="maxNodes">Maximum number of mesh nodes.</param>
    /// <param name="bcTol">Tolerance on the boundary residuals (defaults to tol).</param>
    public static BvpResult SolveBvp(Func<double[], NdArray, double[], NdArray> fun, Func<double[], double[], double[], double[]> bc,
        double[] x, NdArray y, double[]? p = null, double tol = 1e-3, int maxNodes = 1000, double? bcTol = null)
    {
        ArgumentNullException.ThrowIfNull(fun);
        ArgumentNullException.ThrowIfNull(bc);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (y.Rank != 2) throw new ArgumentException("`y` must be 2 dimensional.");
        var m0 = x.Length;
        if (y.Columns != m0) throw new ArgumentException("`y` is expected to have as many columns as `x` has elements.");
        for (var i = 1; i < m0; i++) if (!(x[i] > x[i - 1])) throw new ArgumentException("`x` must be strictly increasing.");
        var solver = new IntegrateBvpSolver(fun, bc, y.Rows, p?.Length ?? 0);
        return solver.Solve((double[])x.Clone(), y.AsContiguous().AsSpan().ToArray(), p is null ? [] : (double[])p.Clone(), tol, maxNodes, bcTol ?? tol);
    }

    /// <summary>solve_bvp without unknown parameters: fun(x, y), bc(ya, yb).</summary>
    public static BvpResult SolveBvp(Func<double[], NdArray, NdArray> fun, Func<double[], double[], double[]> bc,
        double[] x, NdArray y, double tol = 1e-3, int maxNodes = 1000, double? bcTol = null)
    {
        ArgumentNullException.ThrowIfNull(fun);
        ArgumentNullException.ThrowIfNull(bc);
        return SolveBvp((xx, yy, _) => fun(xx, yy), (a, b, _) => bc(a, b), x, y, null, tol, maxNodes, bcTol);
    }
}

/// <summary>The solve_bvp machinery (scipy _bvp.py), on flat row-major (n x m) buffers.</summary>
internal sealed class IntegrateBvpSolver
{
    private const double Eps = 2.220446049250313e-16;
    private readonly Func<double[], NdArray, double[], NdArray> _fun;
    private readonly Func<double[], double[], double[], double[]> _bc;
    private readonly int _n, _k;

    internal IntegrateBvpSolver(Func<double[], NdArray, double[], NdArray> fun, Func<double[], double[], double[], double[]> bc, int n, int k)
    {
        _fun = fun; _bc = bc; _n = n; _k = k;
    }

    private double[] F(double[] x, double[] y, double[] p)
    {
        var r = _fun(x, new NdArray((double[])y.Clone(), _n, x.Length), p);
        if (r.Rank != 2 || r.Rows != _n || r.Columns != x.Length) throw new ArgumentException("`fun` return is expected to have shape (n, m).");
        return r.AsContiguous().AsSpan().ToArray();
    }

    private double[] Bc(double[] ya, double[] yb, double[] p)
    {
        var r = _bc((double[])ya.Clone(), (double[])yb.Clone(), p);
        if (r.Length != _n + _k) throw new ArgumentException("`bc` return is expected to have shape (n + k,).");
        return r;
    }

    private double[] Column(double[] y, int m, int j)
    {
        var c = new double[_n];
        for (var i = 0; i < _n; i++) c[i] = y[i * m + j];
        return c;
    }

    private (double[] ColRes, double[] YMid, double[] Fv, double[] FMid) Collocation(double[] x, double[] h, double[] y, double[] p)
    {
        int n = _n, m = x.Length;
        var f = F(x, y, p);
        var xm = new double[m - 1];
        for (var j = 0; j < m - 1; j++) xm[j] = x[j] + 0.5 * h[j];
        var ymid = new double[n * (m - 1)];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < m - 1; j++)
                ymid[i * (m - 1) + j] = 0.5 * (y[i * m + j + 1] + y[i * m + j]) - 0.125 * h[j] * (f[i * m + j + 1] - f[i * m + j]);
        var fmid = F(xm, ymid, p);
        var col = new double[n * (m - 1)];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < m - 1; j++)
                col[i * (m - 1) + j] = y[i * m + j + 1] - y[i * m + j] - h[j] / 6 * (f[i * m + j] + f[i * m + j + 1] + 4 * fmid[i * (m - 1) + j]);
        return (col, ymid, f, fmid);
    }

    /// <summary>df/dy as [node][row n][col n] and df/dp as [node][row n][col k], by forward differences.</summary>
    private (double[] Dy, double[] Dp) FunJac(double[] x, double[] y, double[] p, double[] f0)
    {
        int n = _n, k = _k, m = x.Length;
        var dy = new double[m * n * n];
        var yn = (double[])y.Clone();
        var hi = new double[m];
        for (var c = 0; c < n; c++)
        {
            for (var j = 0; j < m; j++)
            {
                var v = y[c * m + j];
                var h = Math.Sqrt(Eps) * (1 + Math.Abs(v));
                yn[c * m + j] = v + h;
                hi[j] = yn[c * m + j] - v;
            }
            var fn = F(x, yn, p);
            for (var j = 0; j < m; j++) yn[c * m + j] = y[c * m + j];
            for (var r = 0; r < n; r++)
                for (var j = 0; j < m; j++)
                    dy[(j * n + r) * n + c] = (fn[r * m + j] - f0[r * m + j]) / hi[j];
        }
        var dp = new double[m * n * k];
        if (k > 0)
        {
            var pn = (double[])p.Clone();
            for (var c = 0; c < k; c++)
            {
                var h = Math.Sqrt(Eps) * (1 + Math.Abs(p[c]));
                pn[c] = p[c] + h;
                var hc = pn[c] - p[c];
                var fn = F(x, y, pn);
                pn[c] = p[c];
                for (var r = 0; r < n; r++)
                    for (var j = 0; j < m; j++)
                        dp[(j * n + r) * k + c] = (fn[r * m + j] - f0[r * m + j]) / hc;
            }
        }
        return (dy, dp);
    }

    /// <summary>Boundary Jacobians: (n+k) x n for ya and yb, (n+k) x k for p.</summary>
    private (double[] Da, double[] Db, double[] Dp) BcJac(double[] ya, double[] yb, double[] p, double[] bc0)
    {
        int n = _n, k = _k, r = n + k;
        var da = new double[r * n]; var db = new double[r * n]; var dp = new double[r * k];
        for (var c = 0; c < n; c++)
        {
            var yan = (double[])ya.Clone();
            yan[c] += Math.Sqrt(Eps) * (1 + Math.Abs(ya[c]));
            var h = yan[c] - ya[c];
            var b = Bc(yan, yb, p);
            for (var i = 0; i < r; i++) da[i * n + c] = (b[i] - bc0[i]) / h;
            var ybn = (double[])yb.Clone();
            ybn[c] += Math.Sqrt(Eps) * (1 + Math.Abs(yb[c]));
            h = ybn[c] - yb[c];
            b = Bc(ya, ybn, p);
            for (var i = 0; i < r; i++) db[i * n + c] = (b[i] - bc0[i]) / h;
        }
        for (var c = 0; c < k; c++)
        {
            var pn = (double[])p.Clone();
            pn[c] += Math.Sqrt(Eps) * (1 + Math.Abs(p[c]));
            var h = pn[c] - p[c];
            var b = Bc(ya, yb, pn);
            for (var i = 0; i < r; i++) dp[i * k + c] = (b[i] - bc0[i]) / h;
        }
        return (da, db, dp);
    }

    /// <summary>Assembles and factors the augmented banded Newton matrix. Returns null when singular.</summary>
    private IntegrateBandedLu? Jacobian(double[] x, double[] h, double[] y, double[] p, double[] ymid, double[] f, double[] fmid, double[] bc0)
    {
        int n = _n, k = _k, m = x.Length;
        var xm = new double[m - 1];
        for (var j = 0; j < m - 1; j++) xm[j] = x[j] + 0.5 * h[j];
        var (dfy, dfp) = FunJac(x, y, p, f);
        var (dfyM, dfpM) = FunJac(xm, ymid, p, fmid);
        var (da, db, dbp) = BcJac(Column(y, m, 0), Column(y, m, m - 1), p, bc0);
        var s = 2 * n + k;
        var lu = new IntegrateBandedLu(m * s, n + s - 1, 2 * s - 1 - n);
        for (var i = 0; i < n; i++) { lu.Add(i, n + i, 1.0); lu.Add(i, i, -1.0); }
        var tmp = new double[n * n];
        var tmpP = new double[n * k];
        for (var j = 0; j < m - 1; j++)
        {
            var row = n + j * s;
            var c0 = j * s; var c1 = (j + 1) * s;
            var hj = h[j];
            // dPhi/dy_j and dPhi/dy_{j+1}
            for (var side = 0; side < 2; side++)
            {
                var node = j + side;
                // T = dfyM[j] @ dfy[node]
                for (var r = 0; r < n; r++)
                    for (var c = 0; c < n; c++)
                    {
                        var acc = 0.0;
                        for (var q = 0; q < n; q++) acc += dfyM[(j * n + r) * n + q] * dfy[(node * n + q) * n + c];
                        tmp[r * n + c] = acc;
                    }
                var colBase = side == 0 ? c0 : c1;
                for (var r = 0; r < n; r++)
                    for (var c = 0; c < n; c++)
                    {
                        var v = (r == c ? (side == 0 ? -1.0 : 1.0) : 0.0)
                            - hj / 6 * (dfy[(node * n + r) * n + c] + 2 * dfyM[(j * n + r) * n + c])
                            + (side == 0 ? -1.0 : 1.0) * hj * hj / 12 * tmp[r * n + c];
                        if (v != 0) lu.Add(row + r, colBase + c, v);
                    }
            }
            if (k > 0)
            {
                for (var r = 0; r < n; r++)
                    for (var c = 0; c < k; c++)
                    {
                        var acc = 0.0;
                        for (var q = 0; q < n; q++) acc += dfyM[(j * n + r) * n + q] * (dfp[(j * n + q) * k + c] - dfp[((j + 1) * n + q) * k + c]);
                        tmpP[r * k + c] = dfpM[(j * n + r) * k + c] + 0.125 * hj * acc;
                    }
                for (var r = 0; r < n; r++)
                    for (var c = 0; c < k; c++)
                    {
                        var v = -hj / 6 * (dfp[(j * n + r) * k + c] + dfp[((j + 1) * n + r) * k + c] + 4 * tmpP[r * k + c]);
                        if (v != 0) lu.Add(row + r, c0 + 2 * n + c, v);
                    }
            }
            for (var i = 0; i < n; i++) { lu.Add(row + n + i, c1 + n + i, 1.0); lu.Add(row + n + i, c0 + n + i, -1.0); }
            for (var i = 0; i < k; i++) { lu.Add(row + 2 * n + i, c1 + 2 * n + i, 1.0); lu.Add(row + 2 * n + i, c0 + 2 * n + i, -1.0); }
        }
        var bRow = n + (m - 1) * s;
        var cl = (m - 1) * s;
        for (var r = 0; r < n + k; r++)
        {
            for (var c = 0; c < n; c++)
            {
                if (da[r * n + c] != 0) lu.Add(bRow + r, cl + n + c, da[r * n + c]);
                if (db[r * n + c] != 0) lu.Add(bRow + r, cl + c, db[r * n + c]);
            }
            for (var c = 0; c < k; c++) if (dbp[r * k + c] != 0) lu.Add(bRow + r, cl + 2 * n + c, dbp[r * k + c]);
        }
        return lu.Factor() ? lu : null;
    }

    /// <summary>Solves with the augmented system, mapping the residual in and the (y, p) step out.</summary>
    private (double[] Ystep, double[] Pstep) LinearSolve(IntegrateBandedLu lu, int m, double[] colRes, double[] bcRes)
    {
        int n = _n, k = _k, s = 2 * n + k;
        var b = new double[m * s];
        for (var j = 0; j < m - 1; j++)
            for (var i = 0; i < n; i++) b[n + j * s + i] = colRes[i * (m - 1) + j];
        var bRow = n + (m - 1) * s;
        for (var r = 0; r < n + k; r++) b[bRow + r] = bcRes[r];
        lu.Solve(b);
        var ys = new double[n * m];
        for (var j = 0; j < m; j++)
            for (var i = 0; i < n; i++) ys[i * m + j] = b[j * s + i];
        var ps = new double[k];
        for (var c = 0; c < k; c++) ps[c] = b[2 * n + c];
        return (ys, ps);
    }

    private static double Dot(double[] a, double[] b)
    {
        double s = 0;
        foreach (var v in a) s += v * v;
        foreach (var v in b) s += v * v;
        return s;
    }

    private (double[] Y, double[] P, bool Singular) Newton(double[] x, double[] h, double[] y, double[] p, double bvpTol, double bcTol)
    {
        int n = _n, m = x.Length;
        var tolR = new double[m - 1];
        for (var j = 0; j < m - 1; j++) tolR[j] = 2.0 / 3 * h[j] * 5e-2 * bvpTol;
        const int maxNjev = 4, maxIter = 8, nTrial = 4;
        const double sigma = 0.2, tau = 0.5;
        var (col, ymid, f, fmid) = Collocation(x, h, y, p);
        var bcRes = Bc(Column(y, m, 0), Column(y, m, m - 1), p);
        var njev = 0;
        var recompute = true;
        IntegrateBandedLu? lu = null;
        double[] yStep = [], pStep = [];
        var cost = 0.0;
        for (var iter = 0; iter < maxIter; iter++)
        {
            if (recompute)
            {
                lu = Jacobian(x, h, y, p, ymid, f, fmid, bcRes);
                njev++;
                if (lu is null) return (y, p, true);
                (yStep, pStep) = LinearSolve(lu, m, col, bcRes);
                cost = Dot(yStep, pStep);
            }
            var alpha = 1.0;
            double[] yNew = y, pNew = p;
            double[] stepNewY = [], stepNewP = [];
            var costNew = 0.0;
            for (var trial = 0; trial <= nTrial; trial++)
            {
                yNew = new double[y.Length];
                for (var i = 0; i < y.Length; i++) yNew[i] = y[i] - alpha * yStep[i];
                pNew = new double[p.Length];
                for (var i = 0; i < p.Length; i++) pNew[i] = p[i] - alpha * pStep[i];
                (col, ymid, f, fmid) = Collocation(x, h, yNew, pNew);
                bcRes = Bc(Column(yNew, m, 0), Column(yNew, m, m - 1), pNew);
                (stepNewY, stepNewP) = LinearSolve(lu!, m, col, bcRes);
                costNew = Dot(stepNewY, stepNewP);
                if (costNew < (1 - 2 * alpha * sigma) * cost) break;
                if (trial < nTrial) alpha *= tau;
            }
            y = yNew;
            p = pNew;
            if (njev == maxNjev) break;
            var ok = true;
            for (var i = 0; i < n && ok; i++)
                for (var j = 0; j < m - 1; j++)
                    if (!(Math.Abs(col[i * (m - 1) + j]) < tolR[j] * (1 + Math.Abs(fmid[i * (m - 1) + j])))) { ok = false; break; }
            if (ok) foreach (var v in bcRes) if (!(Math.Abs(v) < bcTol)) { ok = false; break; }
            if (ok) break;
            if (alpha == 1)
            {
                yStep = stepNewY; pStep = stepNewP;
                cost = costNew;
                recompute = false;
            }
            else recompute = true;
        }
        return (y, p, false);
    }

    private BvpSolution Spline(double[] x, double[] h, double[] y, double[] yp)
    {
        int n = _n, m = x.Length;
        var c = new double[(m - 1) * n * 4];
        for (var j = 0; j < m - 1; j++)
            for (var i = 0; i < n; i++)
            {
                var slope = (y[i * m + j + 1] - y[i * m + j]) / h[j];
                var t = (yp[i * m + j] + yp[i * m + j + 1] - 2 * slope) / h[j];
                var o = (j * n + i) * 4;
                c[o] = t / h[j];
                c[o + 1] = (slope - yp[i * m + j]) / h[j] - t;
                c[o + 2] = yp[i * m + j];
                c[o + 3] = y[i * m + j];
            }
        return new BvpSolution((double[])x.Clone(), c, n);
    }

    private double[] RmsResiduals(BvpSolution sol, double[] x, double[] h, double[] p, double[] rMid, double[] fMid)
    {
        int n = _n, m = x.Length;
        var x1 = new double[m - 1]; var x2 = new double[m - 1];
        for (var j = 0; j < m - 1; j++)
        {
            var xmid = x[j] + 0.5 * h[j];
            var s = 0.5 * h[j] * Math.Sqrt(3.0 / 7);
            x1[j] = xmid + s; x2[j] = xmid - s;
        }
        var y1 = sol.Evaluate(x1).AsContiguous().AsSpan().ToArray();
        var y2 = sol.Evaluate(x2).AsContiguous().AsSpan().ToArray();
        var y1p = sol.Evaluate(x1, 1).AsContiguous().AsSpan().ToArray();
        var y2p = sol.Evaluate(x2, 1).AsContiguous().AsSpan().ToArray();
        var f1 = F(x1, y1, p);
        var f2 = F(x2, y2, p);
        var res = new double[m - 1];
        for (var j = 0; j < m - 1; j++)
        {
            double s1 = 0, s2 = 0, sm = 0;
            for (var i = 0; i < n; i++)
            {
                var idx = i * (m - 1) + j;
                var r1 = (y1p[idx] - f1[idx]) / (1 + Math.Abs(f1[idx]));
                var r2 = (y2p[idx] - f2[idx]) / (1 + Math.Abs(f2[idx]));
                var rm = rMid[idx] / (1 + Math.Abs(fMid[idx]));
                s1 += r1 * r1; s2 += r2 * r2; sm += rm * rm;
            }
            res[j] = Math.Sqrt(0.5 * (32.0 / 45 * sm + 49.0 / 90 * (s1 + s2)));
        }
        return res;
    }

    internal BvpResult Solve(double[] x, double[] y, double[] p, double tol, int maxNodes, double bcTol)
    {
        int n = _n;
        if (tol < 100 * Eps) tol = 100 * Eps;
        const int maxIteration = 10;
        var status = 0;
        var iteration = 0;
        double[] h, f, rms;
        BvpSolution sol;
        while (true)
        {
            var m = x.Length;
            h = new double[m - 1];
            for (var j = 0; j < m - 1; j++) h[j] = x[j + 1] - x[j];
            bool singular;
            (y, p, singular) = Newton(x, h, y, p, tol, bcTol);
            iteration++;
            var (col, _, fv, fmid) = Collocation(x, h, y, p);
            f = fv;
            var bcRes = Bc(Column(y, m, 0), Column(y, m, m - 1), p);
            var maxBc = bcRes.Length == 0 ? 0 : bcRes.Max(Math.Abs);
            var rMid = new double[col.Length];
            for (var i = 0; i < n; i++)
                for (var j = 0; j < m - 1; j++) rMid[i * (m - 1) + j] = 1.5 * col[i * (m - 1) + j] / h[j];
            sol = Spline(x, h, y, f);
            rms = RmsResiduals(sol, x, h, p, rMid, fmid);
            if (singular) { status = 2; break; }
            var ins1 = new List<int>();
            var ins2 = new List<int>();
            for (var j = 0; j < m - 1; j++)
            {
                if (rms[j] > tol && rms[j] < 100 * tol) ins1.Add(j);
                else if (rms[j] >= 100 * tol) ins2.Add(j);
            }
            var added = ins1.Count + 2 * ins2.Count;
            if (m + added > maxNodes) { status = 1; break; }
            if (added > 0)
            {
                var nx = new List<double>(x);
                foreach (var j in ins1) nx.Add(0.5 * (x[j] + x[j + 1]));
                foreach (var j in ins2)
                {
                    nx.Add((2 * x[j] + x[j + 1]) / 3);
                    nx.Add((x[j] + 2 * x[j + 1]) / 3);
                }
                nx.Sort();
                x = nx.ToArray();
                y = sol.Evaluate(x).AsContiguous().AsSpan().ToArray();
            }
            else if (maxBc <= bcTol) { status = 0; break; }
            else if (iteration >= maxIteration) { status = 3; break; }
        }
        var mm = x.Length;
        var message = status switch
        {
            0 => "The algorithm converged to the desired accuracy.",
            1 => "The maximum number of mesh nodes is exceeded.",
            2 => "A singular Jacobian encountered when solving the collocation system.",
            _ => "The solver was unable to satisfy boundary conditions tolerance on iteration 10.",
        };
        return new BvpResult
        {
            Sol = sol, P = p, X = x, Y = new NdArray(y, n, mm), Yp = new NdArray(f, n, mm), RmsResiduals = rms,
            Niter = iteration, Status = status, Message = message,
        };
    }
}
