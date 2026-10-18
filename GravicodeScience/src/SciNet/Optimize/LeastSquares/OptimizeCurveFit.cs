using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Result of <c>Optimize.CurveFit</c>.</summary>
public sealed class CurveFitResult
{
    /// <summary>Optimal parameters.</summary>
    public double[] Popt { get; init; } = [];
    /// <summary>Estimated covariance of <see cref="Popt"/>; all +∞ when it cannot be estimated.</summary>
    public NdArray Pcov { get; init; } = NdArray.Zeros(0);
    /// <summary>Residual evaluations.</summary>
    public int Nfev { get; init; }
    /// <summary>Weighted residuals (f(x) − y)/σ at the solution.</summary>
    public double[] Fvec { get; init; } = [];
    /// <summary>Solver status (MINPACK <c>info</c> for Lm, least_squares status otherwise).</summary>
    public int Status { get; init; }
    /// <summary>Solver message.</summary>
    public string Message { get; init; } = "";
    /// <summary>True when the covariance is finite.</summary>
    public bool CovarianceEstimated { get; init; }

    /// <summary>One-standard-deviation parameter errors √diag(Pcov).</summary>
    public double[] Perr
    {
        get
        {
            var n = Popt.Length;
            var p = Pcov.AsContiguous().ToArray();
            var r = new double[n];
            for (var i = 0; i < n; i++) r[i] = Math.Sqrt(p[i * n + i]);
            return r;
        }
    }
}

/// <summary>Result of <see cref="Optimize.LsqLinear"/>.</summary>
public sealed class LsqLinearResult
{
    /// <summary>Solution.</summary>
    public double[] X { get; init; } = [];
    /// <summary>½‖Ax − b‖².</summary>
    public double Cost { get; init; }
    /// <summary>Residuals Ax − b.</summary>
    public double[] Fun { get; init; } = [];
    /// <summary>First-order optimality.</summary>
    public double Optimality { get; init; }
    /// <summary>−1 lower bound active, +1 upper, 0 free.</summary>
    public int[] ActiveMask { get; init; } = [];
    /// <summary>Iterations.</summary>
    public int Nit { get; init; }
    /// <summary>−1 no progress, 0 iteration limit, 1 optimality, 2 cost change, 3 unconstrained solution optimal.</summary>
    public int Status { get; init; }
    /// <summary>Termination message.</summary>
    public string Message { get; init; } = "";
    /// <summary>Status &gt; 0.</summary>
    public bool Success { get; init; }
}

/// <summary>Methods of <see cref="Optimize.LsqLinear"/>.</summary>
public enum LsqLinearMethod
{
    /// <summary>Trust Region Reflective adapted to the linear problem.</summary>
    Trf,
    /// <summary>Bounded-variable least squares (Stark &amp; Parker 1995): an active-set method, exact in finitely many steps.</summary>
    Bvls,
}

public static partial class Optimize
{
    /// <summary>
    /// Fits <c>y ≈ model(x, p)</c> by (weighted) least squares, <c>scipy.optimize.curve_fit</c>.
    /// </summary>
    /// <param name="model">Model evaluated at one x.</param>
    /// <param name="xdata">Independent variable.</param>
    /// <param name="ydata">Observations.</param>
    /// <param name="p0">Initial parameters; defaults to ones (or interior points of the bounds), which needs <paramref name="nParams"/>.</param>
    /// <param name="nParams">Number of parameters when <paramref name="p0"/> is null.</param>
    /// <param name="sigma">Per-point standard deviations; residuals are divided by them.</param>
    /// <param name="absoluteSigma">Treat sigma as absolute (pcov not rescaled by the residual variance).</param>
    /// <param name="bounds">Parameter bounds (switches the default method to Trf).</param>
    /// <param name="method">Lm without bounds, Trf with bounds, unless given.</param>
    /// <param name="jac">Gradient of the model with respect to p at one x.</param>
    /// <param name="maxNfev">Evaluation limit.</param>
    /// <exception cref="InvalidOperationException">The optimiser did not converge (SciPy's RuntimeError).</exception>
    public static CurveFitResult CurveFit(Func<double, double[], double> model, double[] xdata, double[] ydata, double[]? p0 = null, int? nParams = null,
        double[]? sigma = null, bool absoluteSigma = false, Bounds? bounds = null, LeastSquaresMethod? method = null,
        Func<double, double[], double[]>? jac = null, int? maxNfev = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        Func<double[], double[], double[]> vec = (xs, p) => { var r = new double[xs.Length]; for (var i = 0; i < xs.Length; i++) r[i] = model(xs[i], p); return r; };
        Func<double[], double[], NdArray>? vjac = null;
        if (jac is not null)
            vjac = (xs, p) =>
            {
                var rows = new double[xs.Length * p.Length];
                for (var i = 0; i < xs.Length; i++)
                {
                    var g = jac(xs[i], p);
                    if (g.Length != p.Length) throw new ArgumentException("jac must return one derivative per parameter.");
                    Array.Copy(g, 0, rows, i * p.Length, p.Length);
                }
                return new NdArray(rows, xs.Length, p.Length);
            };
        return CurveFit(vec, xdata, ydata, p0, nParams, sigma, absoluteSigma, bounds, method, vjac, maxNfev);
    }

    /// <summary>
    /// the scalar-model <c>CurveFit</c>
    /// with a vectorised model <c>model(xdata, p)</c> returning all predictions at once (and a
    /// Jacobian of shape (m, n) if given) — much cheaper when the model shares work across points.
    /// </summary>
    public static CurveFitResult CurveFit(Func<double[], double[], double[]> model, double[] xdata, double[] ydata, double[]? p0 = null, int? nParams = null,
        double[]? sigma = null, bool absoluteSigma = false, Bounds? bounds = null, LeastSquaresMethod? method = null,
        Func<double[], double[], NdArray>? jac = null, int? maxNfev = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(xdata);
        ArgumentNullException.ThrowIfNull(ydata);
        var m = ydata.Length;
        if (m == 0) throw new ArgumentException("`ydata` must not be empty!");
        foreach (var v in ydata) if (!double.IsFinite(v)) throw new ArgumentException("ydata must be finite.");
        foreach (var v in xdata) if (!double.IsFinite(v)) throw new ArgumentException("xdata must be finite.");
        int n;
        double[] lb, ub;
        if (p0 is not null) n = p0.Length;
        else if (nParams is { } np && np > 0) n = np;
        else if (bounds is not null) n = bounds.Length;
        else throw new ArgumentException("Unable to determine number of fit parameters: pass p0 or nParams.");
        lb = bounds?.Lb ?? Enumerable.Repeat(double.NegativeInfinity, n).ToArray();
        ub = bounds?.Ub ?? Enumerable.Repeat(double.PositiveInfinity, n).ToArray();
        if (lb.Length != n) throw new ArgumentException("Bounds must have one entry per parameter.");
        if (p0 is null)
        {
            // SciPy's _initialize_feasible.
            p0 = new double[n];
            for (var i = 0; i < n; i++)
            {
                var lf = double.IsFinite(lb[i]); var uf = double.IsFinite(ub[i]);
                p0[i] = lf && uf ? 0.5 * (lb[i] + ub[i]) : lf ? lb[i] + 1 : uf ? ub[i] - 1 : 1;
            }
        }
        var bounded = lb.Any(double.IsFinite) || ub.Any(double.IsFinite);
        var meth = method ?? (bounded ? LeastSquaresMethod.Trf : LeastSquaresMethod.Lm);
        if (meth == LeastSquaresMethod.Lm && bounded) throw new ArgumentException("Method 'lm' only works for unconstrained problems. Use 'trf' or 'dogbox' instead.");
        double[]? transform = null;
        if (sigma is not null)
        {
            if (sigma.Length == 1) transform = Enumerable.Repeat(1.0 / sigma[0], m).ToArray();
            else if (sigma.Length == m) transform = sigma.Select(s => 1.0 / s).ToArray();
            else throw new ArgumentException("`sigma` has incorrect shape.");
        }
        var xcopy = (double[])xdata.Clone();
        double[] Residuals(double[] p)
        {
            var y = model(xcopy, p);
            if (y.Length != m) throw new ArgumentException($"The model returned {y.Length} values for {m} observations.");
            // A fresh array, never the model's own: a model may legitimately return a cached buffer.
            var r = new double[m];
            ReadOnlySpan<double> ys = y, yd = ydata;
            if (transform is null) for (var i = 0; i < r.Length; i++) r[i] = ys[i] - yd[i];
            else { ReadOnlySpan<double> t = transform; for (var i = 0; i < r.Length; i++) r[i] = t[i] * (ys[i] - yd[i]); }
            return r;
        }
        Func<double[], NdArray>? rjac = null;
        if (jac is not null)
            rjac = p =>
            {
                var j = jac(xcopy, p).AsContiguous().ToArray();
                if (j.Length != m * n) throw new ArgumentException($"jac must have shape ({m}, {n}).");
                if (transform is not null) for (var i = 0; i < m; i++) for (var k = 0; k < n; k++) j[i * n + k] *= transform[i];
                return new NdArray(j, m, n);
            };

        double[] popt; double[] fvec; NdArray? pcov; int nfev, status; string msg;
        if (meth == LeastSquaresMethod.Lm)
        {
            if (m != 1 && n > m) throw new ArgumentException($"The number of func parameters={n} must not exceed the number of data points={m}");
            // MINPACK copies each residual vector into its own storage before the next call, so the
            // Lm path can reuse one buffer instead of allocating per evaluation.
            var buffer = new double[m];
            double[] ResidualsInto(double[] p)
            {
                var y = model(xcopy, p);
                if (y.Length != m) throw new ArgumentException($"The model returned {y.Length} values for {m} observations.");
                ReadOnlySpan<double> ys = y, yd = ydata;
                if (transform is null) for (var i = 0; i < buffer.Length; i++) buffer[i] = ys[i] - yd[i];
                else { ReadOnlySpan<double> t = transform; for (var i = 0; i < buffer.Length; i++) buffer[i] = t[i] * (ys[i] - yd[i]); }
                return buffer;
            }
            var r = MinpackLm(new MinpackLmProblem { Fun = ResidualsInto, Jac = rjac, X0 = (double[])p0.Clone(), MaxFev = maxNfev ?? 0 });
            if (r.Status is < 1 or > 4) throw new InvalidOperationException("Optimal parameters not found: " + r.Message);
            popt = r.X; fvec = r.FunVector!; pcov = r.HessInv; nfev = r.Nfev; status = r.Status; msg = r.Message;
        }
        else
        {
            var r = LeastSquares(Residuals, p0, rjac, bounded ? new Bounds(lb, ub) : null, meth, maxNfev: maxNfev);
            if (!r.Success) throw new InvalidOperationException("Optimal parameters not found: " + r.Message);
            popt = r.X; fvec = r.Fun; nfev = r.Nfev; status = r.Status; msg = r.Message;
            pcov = PinvCovariance(r.Jac.AsContiguous().ToArray(), m, n);
        }

        var cost = OptimizeDense.Dot(fvec, fvec);
        var finite = pcov is not null && !pcov.ToArray().Any(double.IsNaN);
        double[] pc;
        if (!finite) { pc = Enumerable.Repeat(double.PositiveInfinity, n * n).ToArray(); }
        else
        {
            pc = pcov!.AsContiguous().ToArray();
            if (!absoluteSigma)
            {
                if (m > n) { var s2 = cost / (m - n); for (var i = 0; i < pc.Length; i++) pc[i] *= s2; }
                else { Array.Fill(pc, double.PositiveInfinity); finite = false; }
            }
        }
        return new CurveFitResult
        {
            Popt = popt, Pcov = new NdArray(pc, n, n), Nfev = nfev, Fvec = fvec, Status = status, Message = msg, CovarianceEstimated = finite,
        };
    }

    /// <summary>Moore–Penrose (JᵀJ)⁻¹ through the SVD, discarding singular values below ε·max(m,n)·s₀ — SciPy's curve_fit recipe.</summary>
    private static NdArray PinvCovariance(double[] jm, int m, int n)
    {
        var (_, s, v, k) = OptimizeLsq.Svd(jm, m, n);
        var vc = v.Length / n;
        var threshold = 2.220446049250313e-16 * Math.Max(m, n) * s[0];
        var cov = new double[n * n];
        for (var c = 0; c < k; c++)
        {
            if (!(s[c] > threshold)) continue;
            var w = 1 / (s[c] * s[c]);
            for (var i = 0; i < n; i++)
                for (var j = 0; j < n; j++) cov[i * n + j] += v[i * vc + c] * w * v[j * vc + c];
        }
        return new NdArray(cov, n, n);
    }

    /// <summary>
    /// Non-negative least squares min ‖Ax − b‖ subject to x ≥ 0, by the Lawson–Hanson active-set
    /// algorithm (the one behind <c>scipy.optimize.nnls</c>). Throws when the iteration limit
    /// (default 3n) is reached, as SciPy does.
    /// </summary>
    /// <returns>The solution and the residual norm ‖Ax − b‖.</returns>
    public static (double[] X, double Rnorm) Nnls(NdArray a, double[] b, int? maxiter = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Rank != 2) throw new ArgumentException("Expected a 2D array A.");
        int m = a.Shape[0], n = a.Shape[1];
        if (b.Length != m) throw new ArgumentException($"Incompatible dimensions: A has {m} rows but b has {b.Length} entries.");
        var am = a.AsContiguous().ToArray();
        foreach (var v in am) if (!double.IsFinite(v)) throw new ArgumentException("A must be finite.");
        foreach (var v in b) if (!double.IsFinite(v)) throw new ArgumentException("b must be finite.");
        return OptimizeNnls.Solve(am, m, n, b, maxiter is > 0 ? maxiter.Value : 3 * n);
    }

    /// <summary>
    /// Bounded linear least squares min ½‖Ax − b‖² subject to lb ≤ x ≤ ub (<c>scipy.optimize.lsq_linear</c>,
    /// dense A, exact sub-solves).
    /// </summary>
    public static LsqLinearResult LsqLinear(NdArray a, double[] b, Bounds? bounds = null, LsqLinearMethod method = LsqLinearMethod.Trf, double tol = 1e-10, int? maxIter = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Rank == 1) a = a.Reshape(1, a.Size);
        if (a.Rank != 2) throw new ArgumentException("`A` must have at most 2 dimensions.");
        int m = a.Shape[0], n = a.Shape[1];
        if (b.Length != m) throw new ArgumentException("Inconsistent shapes between `A` and `b`.");
        if (maxIter is <= 0) throw new ArgumentException("`max_iter` must be None or positive integer.");
        var lb = bounds?.Lb ?? Enumerable.Repeat(double.NegativeInfinity, n).ToArray();
        var ub = bounds?.Ub ?? Enumerable.Repeat(double.PositiveInfinity, n).ToArray();
        if (lb.Length != n) throw new ArgumentException("Bounds have wrong shape.");
        for (var i = 0; i < n; i++) if (lb[i] >= ub[i]) throw new ArgumentException("Each lower bound must be strictly less than each upper bound.");
        return OptimizeLsqLinear.Run(a.AsContiguous().ToArray(), m, n, b, lb, ub, method, tol, maxIter);
    }
}

/// <summary>Lawson &amp; Hanson (1974), Algorithm NNLS, with the passive-set sub-problems solved by SVD.</summary>
internal static class OptimizeNnls
{
    public static (double[] X, double Rnorm) Solve(double[] a, int m, int n, double[] b, int maxiter)
    {
        var x = new double[n];
        var passive = new bool[n];
        var w = new double[n];
        var r = new double[m];
        double anorm = 0;
        foreach (var v in a) anorm = Math.Max(anorm, Math.Abs(v));
        var tol = 10 * 2.220446049250313e-16 * Math.Max(m, n) * Math.Max(anorm, 1e-300) * Math.Max(1, OptimizeDense.NormInf(b));
        var iter = 0;
        void Gradient()
        {
            OptimizeDense.MatVec(a, m, n, x, r);
            for (var i = 0; i < m; i++) r[i] = b[i] - r[i];
            OptimizeDense.MatTVec(a, m, n, r, w);
        }
        double[] SolvePassive()
        {
            var idx = Enumerable.Range(0, n).Where(j => passive[j]).ToArray();
            var sub = new double[m * idx.Length];
            for (var i = 0; i < m; i++) for (var k = 0; k < idx.Length; k++) sub[i * idx.Length + k] = a[i * n + idx[k]];
            var zs = OptimizeDense.LstsqSvd(sub, m, idx.Length, b);
            var z = new double[n];
            for (var k = 0; k < idx.Length; k++) z[idx[k]] = zs[k];
            return z;
        }
        Gradient();
        while (true)
        {
            var t = -1; var wmax = tol;
            for (var j = 0; j < n; j++) if (!passive[j] && w[j] > wmax) { wmax = w[j]; t = j; }
            if (t < 0) break;
            passive[t] = true;
            while (true)
            {
                if (++iter > maxiter) throw new InvalidOperationException("Maximum number of iterations reached.");
                var z = SolvePassive();
                var allPositive = true;
                for (var j = 0; j < n; j++) if (passive[j] && z[j] <= 0) { allPositive = false; break; }
                if (allPositive) { x = z; break; }
                var alpha = double.PositiveInfinity;
                for (var j = 0; j < n; j++)
                    if (passive[j] && z[j] <= 0) alpha = Math.Min(alpha, x[j] / (x[j] - z[j]));
                for (var j = 0; j < n; j++) x[j] += alpha * (z[j] - x[j]);
                for (var j = 0; j < n; j++) if (passive[j] && x[j] <= tol) { passive[j] = false; x[j] = 0; }
                if (!passive.Any(p => p)) break;
            }
            Gradient();
        }
        OptimizeDense.MatVec(a, m, n, x, r);
        for (var i = 0; i < m; i++) r[i] -= b[i];
        return (x, OptimizeDense.Norm2(r));
    }
}

/// <summary>Ports of SciPy's <c>lsq_linear</c>, <c>trf_linear</c> and <c>bvls</c>.</summary>
internal static class OptimizeLsqLinear
{
    private static readonly Dictionary<int, string> Messages = new()
    {
        [-1] = "The algorithm was not able to make progress on the last iteration.",
        [0] = "The maximum number of iterations is exceeded.",
        [1] = "The first-order optimality measure is less than `tol`.",
        [2] = "The relative change of the cost function is less than `tol`.",
        [3] = "The unconstrained solution is optimal.",
    };

    private static double[] Residual(double[] a, int m, int n, double[] x, double[] b)
    {
        var r = new double[m];
        OptimizeDense.MatVec(a, m, n, x, r);
        for (var i = 0; i < m; i++) r[i] -= b[i];
        return r;
    }

    private static double[] Grad(double[] a, int m, int n, double[] r) { var g = new double[n]; OptimizeDense.MatTVec(a, m, n, r, g); return g; }

    public static LsqLinearResult Run(double[] a, int m, int n, double[] b, double[] lb, double[] ub, LsqLinearMethod method, double tol, int? maxIter)
    {
        var xLsq = OptimizeDense.LstsqSvd(a, m, n, b);
        var inside = true;
        for (var i = 0; i < n; i++) if (!(xLsq[i] >= lb[i] && xLsq[i] <= ub[i])) inside = false;
        if (inside)
        {
            var r = Residual(a, m, n, xLsq, b);
            var g = Grad(a, m, n, r);
            return new LsqLinearResult { X = xLsq, Fun = r, Cost = 0.5 * OptimizeDense.Dot(r, r), Optimality = OptimizeDense.NormInf(g), ActiveMask = new int[n], Nit = 0, Status = 3, Message = Messages[3], Success = true };
        }
        return method == LsqLinearMethod.Trf ? TrfLinear(a, m, n, b, xLsq, lb, ub, tol, maxIter ?? 100) : Bvls(a, m, n, b, xLsq, lb, ub, tol, maxIter);
    }

    private static double[] Reflective(double[] y, double[] lb, double[] ub)
    {
        var x = (double[])y.Clone();
        for (var i = 0; i < y.Length; i++)
        {
            if (y[i] >= lb[i] && y[i] <= ub[i]) continue;
            var lf = double.IsFinite(lb[i]); var uf = double.IsFinite(ub[i]);
            if (lf && !uf) x[i] = Math.Max(y[i], 2 * lb[i] - y[i]);
            else if (!lf && uf) x[i] = Math.Min(y[i], 2 * ub[i] - y[i]);
            else if (lf && uf)
            {
                var d = ub[i] - lb[i];
                var t = (y[i] - lb[i]) % (2 * d);
                if (t < 0) t += 2 * d;
                x[i] = lb[i] + Math.Min(t, 2 * d - t);
            }
        }
        return x;
    }

    private static double EvalQuad(double[] a, int m, int n, double[] g, double[] s, double[]? diag = null)
    {
        var js = new double[m];
        OptimizeDense.MatVec(a, m, n, s, js);
        var q = OptimizeDense.Dot(js, js);
        if (diag is not null) for (var i = 0; i < n; i++) q += s[i] * diag[i] * s[i];
        return 0.5 * q + OptimizeDense.Dot(s, g);
    }

    private static (double Step, int[] Hits) StepToBound(double[] x, double[] s, double[] lb, double[] ub)
    {
        var n = x.Length;
        var min = double.PositiveInfinity;
        var steps = new double[n];
        for (var i = 0; i < n; i++)
        {
            steps[i] = s[i] != 0 ? Math.Max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i]) : double.PositiveInfinity;
            min = Math.Min(min, steps[i]);
        }
        return (min, steps.Select((v, i) => v == min ? Math.Sign(s[i]) : 0).ToArray());
    }

    private static (double, double) MinQuad1d(double qa, double qb, double lo, double hi, double c = 0)
    {
        var ts = new List<double> { lo, hi };
        if (qa != 0) { var e = -0.5 * qb / qa; if (lo < e && e < hi) ts.Add(e); }
        double bt = lo, by = double.PositiveInfinity;
        foreach (var t in ts) { var y = t * (qa * t + qb) + c; if (y < by) { by = y; bt = t; } }
        return (bt, by);
    }

    private static LsqLinearResult TrfLinear(double[] a, int m, int n, double[] b, double[] xLsq, double[] lb, double[] ub, double tol, int maxIter)
    {
        var x = OptimizeLsq.MakeStrictlyFeasible(Reflective(xLsq, lb, ub), lb, ub, 0.1);
        var r = Residual(a, m, n, x, b);
        var g = Grad(a, m, n, r);
        var cost = 0.5 * OptimizeDense.Dot(r, r);
        int? status = null;
        double gNorm = 0;
        var iteration = 0;
        for (iteration = 0; iteration < maxIter; iteration++)
        {
            var (v, dv) = OptimizeLsq.ClScalingVector(x, g, lb, ub);
            gNorm = 0;
            for (var i = 0; i < n; i++) gNorm = Math.Max(gNorm, Math.Abs(g[i] * v[i]));
            if (gNorm < tol) status = 1;
            if (status is not null) break;
            var diagH = g.Select((gi, i) => gi * dv[i]).ToArray();
            var d = v.Select(Math.Sqrt).ToArray();
            var gh = g.Select((gi, i) => d[i] * gi).ToArray();
            var ah = new double[m * n];
            for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) ah[i * n + j] = a[i * n + j] * d[j];
            // Regularised Gauss–Newton step: min ‖A_h p + r‖² + ‖√diag_h p‖², solved by SVD of the stacked matrix.
            var stacked = new double[(m + n) * n];
            Array.Copy(ah, stacked, m * n);
            for (var j = 0; j < n; j++) stacked[(m + j) * n + j] = Math.Sqrt(diagH[j]);
            var rhs = new double[m + n];
            for (var i = 0; i < m; i++) rhs[i] = -r[i];
            var ph = OptimizeDense.LstsqSvd(stacked, m + n, n, rhs);
            var p = ph.Select((t, i) => d[i] * t).ToArray();
            var pDotG = OptimizeDense.Dot(p, g);
            if (pDotG > 0) status = -1;
            var theta = 1 - Math.Min(0.005, gNorm);
            var step = SelectStep(x, ah, m, n, gh, diagH, p, ph, d, lb, ub, theta);
            var costChange = -EvalQuad(a, m, n, g, step);
            if (costChange < 0)
            {
                // Backtracking along the reflected path.
                double alpha = 1;
                double[] xNew;
                while (true)
                {
                    xNew = Reflective(x.Select((xv, i) => xv + alpha * p[i]).ToArray(), lb, ub);
                    step = xNew.Select((xv, i) => xv - x[i]).ToArray();
                    costChange = -EvalQuad(a, m, n, g, step);
                    if (costChange > -0.1 * alpha * pDotG || alpha < 1e-300) break;
                    alpha *= 0.5;
                }
                var active = OptimizeLsq.FindActiveConstraints(xNew, lb, ub);
                if (active.Any(t => t != 0))
                {
                    xNew = OptimizeLsq.MakeStrictlyFeasible(Reflective(x.Select((xv, i) => xv + theta * alpha * p[i]).ToArray(), lb, ub), lb, ub, 0);
                    step = xNew.Select((xv, i) => xv - x[i]).ToArray();
                    costChange = -EvalQuad(a, m, n, g, step);
                }
                // SciPy returns the unchanged x from backtracking here (its `x` is not advanced).
            }
            else x = OptimizeLsq.MakeStrictlyFeasible(x.Select((xv, i) => xv + step[i]).ToArray(), lb, ub, 0);
            r = Residual(a, m, n, x, b);
            g = Grad(a, m, n, r);
            if (costChange < tol * cost) status = 2;
            cost = 0.5 * OptimizeDense.Dot(r, r);
        }
        var st = status ?? 0;
        return new LsqLinearResult
        {
            X = x, Fun = r, Cost = cost, Optimality = gNorm, ActiveMask = OptimizeLsq.FindActiveConstraints(x, lb, ub, tol), Nit = iteration + 1,
            Status = st, Message = Messages[st], Success = st > 0,
        };
    }

    private static double[] SelectStep(double[] x, double[] ah, int m, int n, double[] gh, double[] ch, double[] p, double[] ph, double[] d,
        double[] lb, double[] ub, double theta)
    {
        if (x.Select((v, i) => v + p[i]).Select((v, i) => v >= lb[i] && v <= ub[i]).All(t => t)) return p;
        var (pStride, hits) = StepToBound(x, p, lb, ub);
        var rh = (double[])ph.Clone();
        for (var i = 0; i < n; i++) if (hits[i] != 0) rh[i] = -rh[i];
        var r = rh.Select((v, i) => d[i] * v).ToArray();
        p = p.Select(v => v * pStride).ToArray();
        ph = ph.Select(v => v * pStride).ToArray();
        var xOn = x.Select((v, i) => v + p[i]).ToArray();
        var (rStrideU, _) = StepToBound(xOn, r, lb, ub);
        var rStrideL = (1 - theta) * rStrideU;
        rStrideU *= theta;
        double rValue;
        if (rStrideU > 0)
        {
            var (qa, qb, qc) = Quad1d(ah, m, n, gh, rh, ch, ph);
            var (rs, rv) = MinQuad1d(qa, qb, rStrideL, rStrideU, qc);
            rValue = rv;
            rh = ph.Select((v, i) => v + rh[i] * rs).ToArray();
            r = rh.Select((v, i) => d[i] * v).ToArray();
        }
        else rValue = double.PositiveInfinity;
        ph = ph.Select(v => v * theta).ToArray();
        p = p.Select(v => v * theta).ToArray();
        var pValue = EvalQuad(ah, m, n, gh, ph, ch);
        var agh = gh.Select(v => -v).ToArray();
        var ag = agh.Select((v, i) => d[i] * v).ToArray();
        var (agU, _) = StepToBound(x, ag, lb, ub);
        agU *= theta;
        var (qa2, qb2, _) = Quad1d(ah, m, n, gh, agh, ch, null);
        var (ags, agValue) = MinQuad1d(qa2, qb2, 0, agU);
        ag = ag.Select(v => v * ags).ToArray();
        if (pValue < rValue && pValue < agValue) return p;
        if (rValue < pValue && rValue < agValue) return r;
        return ag;
    }

    private static (double, double, double) Quad1d(double[] jm, int m, int n, double[] g, double[] s, double[] diag, double[]? s0)
    {
        var v = new double[m];
        OptimizeDense.MatVec(jm, m, n, s, v);
        var qa = OptimizeDense.Dot(v, v);
        for (var i = 0; i < n; i++) qa += s[i] * diag[i] * s[i];
        qa *= 0.5;
        var qb = OptimizeDense.Dot(g, s);
        double qc = 0;
        if (s0 is not null)
        {
            var u = new double[m];
            OptimizeDense.MatVec(jm, m, n, s0, u);
            qb += OptimizeDense.Dot(u, v);
            qc = 0.5 * OptimizeDense.Dot(u, u) + OptimizeDense.Dot(g, s0);
            for (var i = 0; i < n; i++) { qb += s0[i] * diag[i] * s[i]; qc += 0.5 * s0[i] * diag[i] * s0[i]; }
        }
        return (qa, qb, qc);
    }

    private static double Kkt(double[] g, double[] onBound)
    {
        double mx = double.NegativeInfinity;
        for (var i = 0; i < g.Length; i++) mx = Math.Max(mx, onBound[i] == 0 ? Math.Abs(g[i]) : g[i] * onBound[i]);
        return mx;
    }

    private static double[] SubLstsq(double[] a, int m, int n, int[] cols, double[] rhs)
    {
        var sub = new double[m * cols.Length];
        for (var i = 0; i < m; i++) for (var k = 0; k < cols.Length; k++) sub[i * cols.Length + k] = a[i * n + cols[k]];
        return OptimizeDense.LstsqSvd(sub, m, cols.Length, rhs);
    }

    private static double[] BFree(double[] a, int m, int n, double[] b, double[] x, double[] onBound)
    {
        var xa = x.Select((v, i) => onBound[i] != 0 ? v : 0).ToArray();
        var ax = new double[m];
        OptimizeDense.MatVec(a, m, n, xa, ax);
        return b.Select((v, i) => v - ax[i]).ToArray();
    }

    private static LsqLinearResult Bvls(double[] a, int m, int n, double[] b, double[] xLsq, double[] lb, double[] ub, double tol, int? maxIter)
    {
        var x = (double[])xLsq.Clone();
        var onBound = new double[n];
        for (var i = 0; i < n; i++)
        {
            if (x[i] <= lb[i]) { x[i] = lb[i]; onBound[i] = -1; }
            else if (x[i] >= ub[i]) { x[i] = ub[i]; onBound[i] = 1; }
        }
        var freeSet = Enumerable.Range(0, n).Where(i => onBound[i] == 0).ToArray();
        var r = Residual(a, m, n, x, b);
        var cost = 0.5 * OptimizeDense.Dot(r, r);
        var g = Grad(a, m, n, r);
        var iteration = 0;
        while (freeSet.Length > 0)
        {
            iteration++;
            var z = SubLstsq(a, m, n, freeSet, BFree(a, m, n, b, x, onBound));
            var violated = new bool[freeSet.Length];
            for (var k = 0; k < freeSet.Length; k++)
            {
                var i = freeSet[k];
                if (z[k] < lb[i]) { x[i] = lb[i]; onBound[i] = -1; violated[k] = true; }
                else if (z[k] > ub[i]) { x[i] = ub[i]; onBound[i] = 1; violated[k] = true; }
                else x[i] = z[k];
            }
            r = Residual(a, m, n, x, b);
            cost = 0.5 * OptimizeDense.Dot(r, r);
            g = Grad(a, m, n, r);
            if (violated.Any(t => t)) freeSet = freeSet.Where((_, k) => !violated[k]).ToArray();
            else break;
        }
        var maxIt = (maxIter ?? n) + iteration;
        int? status = null;
        var optimality = Kkt(g, onBound);
        for (; iteration < maxIt; iteration++)
        {
            if (optimality < tol) status = 1;
            if (status is not null) break;
            var gb = g.Select((v, i) => v * onBound[i]).ToArray();
            var moveToFree = Array.IndexOf(gb, gb.Max());
            onBound[moveToFree] = 0;
            while (true)
            {
                var fs = Enumerable.Range(0, n).Where(i => onBound[i] == 0).ToArray();
                var xFree = fs.Select(i => x[i]).ToArray();
                var z = SubLstsq(a, m, n, fs, BFree(a, m, n, b, x, onBound));
                var lbv = Enumerable.Range(0, fs.Length).Where(k => z[k] < lb[fs[k]]).ToArray();
                var ubv = Enumerable.Range(0, fs.Length).Where(k => z[k] > ub[fs[k]]).ToArray();
                var v = lbv.Concat(ubv).ToArray();
                if (v.Length > 0)
                {
                    var alphas = lbv.Select(k => (lb[fs[k]] - xFree[k]) / (z[k] - xFree[k]))
                        .Concat(ubv.Select(k => (ub[fs[k]] - xFree[k]) / (z[k] - xFree[k]))).ToArray();
                    var ii = Array.IndexOf(alphas, alphas.Min());
                    var iFree = v[ii];
                    var alpha = alphas[ii];
                    for (var k = 0; k < fs.Length; k++) x[fs[k]] = xFree[k] * (1 - alpha) + alpha * z[k];
                    onBound[fs[iFree]] = ii < lbv.Length ? -1 : 1;
                }
                else
                {
                    for (var k = 0; k < fs.Length; k++) x[fs[k]] = z[k];
                    break;
                }
            }
            r = Residual(a, m, n, x, b);
            var costNew = 0.5 * OptimizeDense.Dot(r, r);
            if (cost - costNew < tol * cost) status = 2;
            cost = costNew;
            g = Grad(a, m, n, r);
            optimality = Kkt(g, onBound);
        }
        var st = status ?? 0;
        return new LsqLinearResult
        {
            X = x, Fun = r, Cost = cost, Optimality = optimality, ActiveMask = onBound.Select(v => (int)v).ToArray(), Nit = iteration + 1,
            Status = st, Message = Messages[st], Success = st > 0,
        };
    }
}
