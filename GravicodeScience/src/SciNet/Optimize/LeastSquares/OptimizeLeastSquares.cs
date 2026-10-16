using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Algorithms of <see cref="Optimize.LeastSquares"/>.</summary>
public enum LeastSquaresMethod
{
    /// <summary>Trust Region Reflective (Branch, Coleman &amp; Li 1999): the default, handles bounds.</summary>
    Trf,
    /// <summary>Dogleg in a rectangular trust region (Voglis &amp; Lagaris 2004), handles bounds.</summary>
    Dogbox,
    /// <summary>MINPACK Levenberg–Marquardt; no bounds, needs m ≥ n.</summary>
    Lm,
}

/// <summary>Robust loss ρ(z) applied to z = (f/fScale)² in <see cref="Optimize.LeastSquares"/>.</summary>
public enum LeastSquaresLoss
{
    /// <summary>ρ(z) = z: ordinary least squares.</summary>
    Linear,
    /// <summary>ρ(z) = 2(√(1+z) − 1): smooth L1.</summary>
    SoftL1,
    /// <summary>ρ(z) = z for z ≤ 1, else 2√z − 1.</summary>
    Huber,
    /// <summary>ρ(z) = ln(1 + z).</summary>
    Cauchy,
    /// <summary>ρ(z) = arctan(z): caps the influence of a single residual.</summary>
    Arctan,
}

/// <summary>Result of <see cref="Optimize.LeastSquares"/> (SciPy's <c>least_squares</c> fields).</summary>
public sealed class LeastSquaresResult
{
    /// <summary>Solution.</summary>
    public double[] X { get; init; } = [];
    /// <summary>½ Σ ρ(f²) at the solution.</summary>
    public double Cost { get; init; }
    /// <summary>Residuals at the solution.</summary>
    public double[] Fun { get; init; } = [];
    /// <summary>Jacobian (m, n) at the solution (scaled by the loss, as SciPy's).</summary>
    public NdArray Jac { get; init; } = NdArray.Zeros(0);
    /// <summary>Gradient of the cost.</summary>
    public double[] Grad { get; init; } = [];
    /// <summary>First-order optimality (scaled projected gradient ∞-norm).</summary>
    public double Optimality { get; init; }
    /// <summary>−1 at a lower bound, +1 at an upper bound, 0 free.</summary>
    public int[] ActiveMask { get; init; } = [];
    /// <summary>Residual evaluations (finite-difference ones excluded, as SciPy).</summary>
    public int Nfev { get; init; }
    /// <summary>Jacobian evaluations.</summary>
    public int? Njev { get; init; }
    /// <summary>−2 callback, −1 improper input, 0 nfev limit, 1 gtol, 2 ftol, 3 xtol, 4 ftol and xtol.</summary>
    public int Status { get; init; }
    /// <summary>Termination message.</summary>
    public string Message { get; init; } = "";
    /// <summary>Status &gt; 0.</summary>
    public bool Success { get; init; }
    /// <summary>For method Lm: SciPy's <c>cov_x</c>, (JᵀJ)⁻¹ from the final QR factor; null when singular.</summary>
    internal NdArray? CovX { get; init; }
}

public static partial class Optimize
{
    /// <summary>
    /// Bounded nonlinear least squares, <c>scipy.optimize.least_squares</c>: minimises
    /// ½ Σ ρ((fᵢ(x)/fScale)²)·fScale² subject to lb ≤ x ≤ ub.
    /// </summary>
    /// <param name="fun">Residuals f(x) (length m).</param>
    /// <param name="x0">Initial guess; must lie within the bounds.</param>
    /// <param name="jac">Jacobian (m, n); forward differences with SciPy's relative step otherwise.</param>
    /// <param name="bounds">Bounds; lb &lt; ub strictly, as SciPy requires.</param>
    /// <param name="method">Trf (default), Dogbox or Lm.</param>
    /// <param name="ftol">Relative cost-change tolerance.</param>
    /// <param name="xtol">Relative step tolerance.</param>
    /// <param name="gtol">Gradient tolerance.</param>
    /// <param name="xScale">Characteristic variable scales; null means 1 (Trf/Dogbox) or Jacobian-based (Lm).</param>
    /// <param name="xScaleJac">Use SciPy's <c>x_scale='jac'</c>: inverse column norms of the Jacobian, updated as it grows.</param>
    /// <param name="loss">Robust loss.</param>
    /// <param name="customLoss">A user loss returning (ρ(z), ρ'(z), ρ''(z)); overrides <paramref name="loss"/>.</param>
    /// <param name="fScale">Soft margin between inlier and outlier residuals.</param>
    /// <param name="diffStep">Relative finite-difference step.</param>
    /// <param name="maxNfev">Residual evaluation limit (default 100 n).</param>
    /// <param name="jacScheme">Finite-difference scheme when <paramref name="jac"/> is null.</param>
    /// <param name="callback">Trf/Dogbox: called per iteration with (x, cost); return true to stop.</param>
    public static LeastSquaresResult LeastSquares(Func<double[], double[]> fun, double[] x0, Func<double[], NdArray>? jac = null, Bounds? bounds = null,
        LeastSquaresMethod method = LeastSquaresMethod.Trf, double ftol = 1e-8, double xtol = 1e-8, double gtol = 1e-8, double[]? xScale = null,
        bool xScaleJac = false, LeastSquaresLoss loss = LeastSquaresLoss.Linear, Func<double, (double Rho, double Drho, double D2rho)>? customLoss = null,
        double fScale = 1.0, double? diffStep = null, int? maxNfev = null, FiniteDiffScheme jacScheme = FiniteDiffScheme.TwoPoint,
        Func<double[], double, bool>? callback = null)
        => OptimizeLsq.Run(fun, x0, jac, bounds, method, ftol, xtol, gtol, xScale, xScaleJac, loss, customLoss, fScale, diffStep, maxNfev, jacScheme, callback);
}

/// <summary>Ports of SciPy's <c>_lsq/least_squares.py</c>, <c>trf.py</c>, <c>dogbox.py</c> and <c>common.py</c>.</summary>
internal static class OptimizeLsq
{
    private const double Eps = 2.220446049250313e-16;

    private static readonly Dictionary<int, string> Messages = new()
    {
        [-2] = "Stopped because `callback` function raised `StopIteration` or returned `True`",
        [-1] = "Improper input parameters status returned from `leastsq`",
        [0] = "The maximum number of function evaluations is exceeded.",
        [1] = "`gtol` termination condition is satisfied.",
        [2] = "`ftol` termination condition is satisfied.",
        [3] = "`xtol` termination condition is satisfied.",
        [4] = "Both `ftol` and `xtol` termination conditions are satisfied.",
    };

    /// <summary>Residuals with evaluation counting and the Jacobian (analytic or finite-difference).</summary>
    private sealed class Problem(Func<double[], double[]> fun, Func<double[], NdArray>? jac, int n, FiniteDiffScheme scheme, double? relStep, double[] lb, double[] ub)
    {
        public int M;
        public int Nfev, Njev;
        public double[] F(double[] x)
        {
            Nfev++;
            var v = fun((double[])x.Clone());
            if (M != 0 && v.Length != M) throw new ArgumentException($"The residual function returned {v.Length} values, expected {M}.");
            return (double[])v.Clone();
        }
        /// <summary>Jacobian at x, row-major (m, n). f is f(x) (already known).</summary>
        public double[] J(double[] x, double[] f)
        {
            Njev++;
            if (jac is not null)
            {
                var j = jac((double[])x.Clone());
                if (j.Rank != 2 || j.Shape[0] != M || j.Shape[1] != n)
                    throw new ArgumentException($"The return value of `jac` has wrong shape: expected ({M}, {n}).");
                return j.AsContiguous().ToArray();
            }
            var jj = new double[M * n];
            OptimizeFiniteDiff.Jacobian(fun, x, f, jj, scheme, null, relStep, lb, ub);
            return jj;
        }
    }

    // ---------- loss functions ----------

    private delegate void LossRho(double[] f, double[] rho0, double[]? rho1, double[]? rho2);

    private static LossRho? BuildLoss(LeastSquaresLoss loss, Func<double, (double, double, double)>? custom, double fScale)
    {
        if (custom is null && loss == LeastSquaresLoss.Linear) return null;
        var s2 = fScale * fScale;
        return (f, r0, r1, r2) =>
        {
            for (var i = 0; i < f.Length; i++)
            {
                var z = (f[i] / fScale) * (f[i] / fScale);
                double a, b, c;
                if (custom is not null) (a, b, c) = custom(z);
                else
                {
                    switch (loss)
                    {
                        case LeastSquaresLoss.SoftL1:
                        {
                            var t = 1 + z;
                            a = 2 * (Math.Sqrt(t) - 1); b = 1 / Math.Sqrt(t); c = -0.5 * Math.Pow(t, -1.5);
                            break;
                        }
                        case LeastSquaresLoss.Huber:
                            if (z <= 1) { a = z; b = 1; c = 0; }
                            else { a = 2 * Math.Sqrt(z) - 1; b = 1 / Math.Sqrt(z); c = -0.5 * Math.Pow(z, -1.5); }
                            break;
                        case LeastSquaresLoss.Cauchy:
                        {
                            var t = 1 + z;
                            a = Math.Log(t); b = 1 / t; c = -1 / (t * t);
                            break;
                        }
                        default:
                        {
                            var t = 1 + z * z;
                            a = Math.Atan(z); b = 1 / t; c = -2 * z / (t * t);
                            break;
                        }
                    }
                }
                r0[i] = a * s2;
                if (r1 is not null) r1[i] = b;
                if (r2 is not null) r2[i] = c / s2;
            }
        };
    }

    private static double LossCost(LossRho? loss, double[] f)
    {
        if (loss is null) return 0.5 * OptimizeDense.Dot(f, f);
        var r0 = new double[f.Length];
        loss(f, r0, null, null);
        double s = 0;
        foreach (var v in r0) s += v;
        return 0.5 * s;
    }

    /// <summary>scale_for_robust_loss_function: rescales J rows and f in place, returns the cost.</summary>
    private static double ApplyLoss(LossRho loss, double[] jm, double[] f, int m, int n)
    {
        var r0 = new double[m]; var r1 = new double[m]; var r2 = new double[m];
        loss(f, r0, r1, r2);
        double cost = 0;
        for (var i = 0; i < m; i++)
        {
            cost += r0[i];
            var js = r1[i] + 2 * r2[i] * f[i] * f[i];
            if (js < Eps) js = Eps;
            js = Math.Sqrt(js);
            f[i] *= r1[i] / js;
            for (var j = 0; j < n; j++) jm[i * n + j] *= js;
        }
        return 0.5 * cost;
    }

    // ---------- common.py ----------

    private static (double T1, double T2) IntersectTrustRegion(double[] x, double[] s, double delta)
    {
        var a = OptimizeDense.Dot(s, s);
        if (a == 0) throw new ArgumentException("`s` is zero.");
        var b = OptimizeDense.Dot(x, s);
        var c = OptimizeDense.Dot(x, x) - delta * delta;
        if (c > 0) throw new ArgumentException("`x` is not within the trust region.");
        var d = Math.Sqrt(b * b - a * c);
        var q = -(b + Math.CopySign(d, b));
        var t1 = q / a; var t2 = c / q;
        return t1 < t2 ? (t1, t2) : (t2, t1);
    }

    private static (double[] P, double Alpha) SolveLsqTrustRegion(int n, int m, double[] uf, double[] s, double[] v, int k, double delta, double initialAlpha)
    {
        // v is row-major (n, k): column j is the j-th right singular vector.
        var suf = new double[k];
        for (var i = 0; i < k; i++) suf[i] = s[i] * uf[i];
        bool fullRank;
        if (m >= n) fullRank = k == n && s[k - 1] > Eps * m * s[0];
        else fullRank = false;
        double[] VDot(double[] w) { var p = new double[n]; for (var r = 0; r < n; r++) { double acc = 0; for (var c = 0; c < k; c++) acc += v[r * k + c] * w[c]; p[r] = acc; } return p; }
        if (fullRank)
        {
            var w = new double[k];
            for (var i = 0; i < k; i++) w[i] = uf[i] / s[i];
            var p = VDot(w);
            for (var i = 0; i < n; i++) p[i] = -p[i];
            if (OptimizeDense.Norm2(p) <= delta) return (p, 0.0);
        }
        (double Phi, double Dphi) PhiD(double alpha)
        {
            double pn2 = 0, sum3 = 0;
            for (var i = 0; i < k; i++)
            {
                var den = s[i] * s[i] + alpha;
                var t = suf[i] / den;
                pn2 += t * t;
                sum3 += suf[i] * suf[i] / (den * den * den);
            }
            var pn = Math.Sqrt(pn2);
            return (pn - delta, -sum3 / pn);
        }
        var alphaUpper = OptimizeDense.Norm2(suf) / delta;
        double alphaLower = 0;
        if (fullRank) { var (ph, pd) = PhiD(0); alphaLower = -ph / pd; }
        double alpha;
        if (!fullRank && initialAlpha == 0) alpha = Math.Max(0.001 * alphaUpper, Math.Sqrt(alphaLower * alphaUpper));
        else alpha = initialAlpha;
        for (var it = 0; it < 10; it++)
        {
            if (alpha < alphaLower || alpha > alphaUpper) alpha = Math.Max(0.001 * alphaUpper, Math.Sqrt(alphaLower * alphaUpper));
            var (phi, dphi) = PhiD(alpha);
            if (phi < 0) alphaUpper = alpha;
            var ratio = phi / dphi;
            alphaLower = Math.Max(alphaLower, alpha - ratio);
            alpha -= (phi + delta) * ratio / delta;
            if (Math.Abs(phi) < 0.01 * delta) break;
        }
        var ww = new double[k];
        for (var i = 0; i < k; i++) ww[i] = suf[i] / (s[i] * s[i] + alpha);
        var pp = VDot(ww);
        var nrm = OptimizeDense.Norm2(pp);
        for (var i = 0; i < n; i++) pp[i] = -pp[i] * delta / nrm;
        return (pp, alpha);
    }

    private static (double Delta, double Ratio) UpdateTrRadius(double delta, double actual, double predicted, double stepNorm, bool boundHit)
    {
        double ratio;
        if (predicted > 0) ratio = actual / predicted;
        else if (predicted == actual && actual == 0) ratio = 1;
        else ratio = 0;
        if (ratio < 0.25) delta = 0.25 * stepNorm;
        else if (ratio > 0.75 && boundHit) delta *= 2.0;
        return (delta, ratio);
    }

    private static double[] JDot(double[] jm, int m, int n, double[] s) { var r = new double[m]; OptimizeDense.MatVec(jm, m, n, s, r); return r; }

    private static (double A, double B, double C) BuildQuadratic1d(double[] jm, int m, int n, double[] g, double[] s, double[]? diag = null, double[]? s0 = null)
    {
        var v = JDot(jm, m, n, s);
        var a = OptimizeDense.Dot(v, v);
        if (diag is not null) for (var i = 0; i < n; i++) a += s[i] * diag[i] * s[i];
        a *= 0.5;
        var b = OptimizeDense.Dot(g, s);
        double c = 0;
        if (s0 is not null)
        {
            var u = JDot(jm, m, n, s0);
            b += OptimizeDense.Dot(u, v);
            c = 0.5 * OptimizeDense.Dot(u, u) + OptimizeDense.Dot(g, s0);
            if (diag is not null)
                for (var i = 0; i < n; i++) { b += s0[i] * diag[i] * s[i]; c += 0.5 * s0[i] * diag[i] * s0[i]; }
        }
        return (a, b, c);
    }

    private static (double T, double Y) MinimizeQuadratic1d(double a, double b, double lb, double ub, double c = 0)
    {
        var ts = new List<double> { lb, ub };
        if (a != 0)
        {
            var ext = -0.5 * b / a;
            if (lb < ext && ext < ub) ts.Add(ext);
        }
        double bt = 0, by = double.PositiveInfinity;
        var first = true;
        foreach (var t in ts)
        {
            var y = t * (a * t + b) + c;
            if (first || y < by) { bt = t; by = y; first = false; }
        }
        return (bt, by);
    }

    private static double EvaluateQuadratic(double[] jm, int m, int n, double[] g, double[] s, double[]? diag = null)
    {
        var js = JDot(jm, m, n, s);
        var q = OptimizeDense.Dot(js, js);
        if (diag is not null) for (var i = 0; i < n; i++) q += s[i] * diag[i] * s[i];
        return 0.5 * q + OptimizeDense.Dot(s, g);
    }

    private static bool InBounds(double[] x, double[] lb, double[] ub)
    {
        for (var i = 0; i < x.Length; i++) if (!(x[i] >= lb[i] && x[i] <= ub[i])) return false;
        return true;
    }

    private static (double Step, int[] Hits) StepSizeToBound(double[] x, double[] s, double[] lb, double[] ub)
    {
        var n = x.Length;
        var steps = new double[n];
        var min = double.PositiveInfinity;
        for (var i = 0; i < n; i++)
        {
            steps[i] = double.PositiveInfinity;
            if (s[i] != 0) steps[i] = Math.Max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i]);
            if (steps[i] < min) min = steps[i];
        }
        var hits = new int[n];
        for (var i = 0; i < n; i++) hits[i] = steps[i] == min ? Math.Sign(s[i]) : 0;
        return (min, hits);
    }

    internal static int[] FindActiveConstraints(double[] x, double[] lb, double[] ub, double rtol = 1e-10)
    {
        var n = x.Length;
        var active = new int[n];
        if (rtol == 0)
        {
            for (var i = 0; i < n; i++) { if (x[i] <= lb[i]) active[i] = -1; if (x[i] >= ub[i]) active[i] = 1; }
            return active;
        }
        for (var i = 0; i < n; i++)
        {
            var ld = x[i] - lb[i]; var ud = ub[i] - x[i];
            var lt = rtol * Math.Max(1, Math.Abs(lb[i])); var ut = rtol * Math.Max(1, Math.Abs(ub[i]));
            if (double.IsFinite(lb[i]) && ld <= Math.Min(ud, lt)) active[i] = -1;
            if (double.IsFinite(ub[i]) && ud <= Math.Min(ld, ut)) active[i] = 1;
        }
        return active;
    }

    internal static double[] MakeStrictlyFeasible(double[] x, double[] lb, double[] ub, double rstep = 1e-10)
    {
        var xn = (double[])x.Clone();
        var active = FindActiveConstraints(x, lb, ub, rstep);
        for (var i = 0; i < x.Length; i++)
        {
            if (active[i] == -1) xn[i] = rstep == 0 ? Math.BitIncrement(lb[i]) : lb[i] + rstep * Math.Max(1, Math.Abs(lb[i]));
            else if (active[i] == 1) xn[i] = rstep == 0 ? Math.BitDecrement(ub[i]) : ub[i] - rstep * Math.Max(1, Math.Abs(ub[i]));
            if (xn[i] < lb[i] || xn[i] > ub[i]) xn[i] = 0.5 * (lb[i] + ub[i]);
        }
        return xn;
    }

    internal static (double[] V, double[] Dv) ClScalingVector(double[] x, double[] g, double[] lb, double[] ub)
    {
        var n = x.Length;
        var v = new double[n]; var dv = new double[n];
        for (var i = 0; i < n; i++)
        {
            v[i] = 1;
            if (g[i] < 0 && double.IsFinite(ub[i])) { v[i] = ub[i] - x[i]; dv[i] = -1; }
            if (g[i] > 0 && double.IsFinite(lb[i])) { v[i] = x[i] - lb[i]; dv[i] = 1; }
        }
        return (v, dv);
    }

    private static int? CheckTermination(double dF, double F, double dxNorm, double xNorm, double ratio, double ftol, double xtol)
    {
        var fs = dF < ftol * F && ratio > 0.25;
        var xs = dxNorm < xtol * (xtol + xNorm);
        if (fs && xs) return 4;
        if (fs) return 2;
        if (xs) return 3;
        return null;
    }

    private static (double[] Scale, double[] ScaleInv) JacScale(double[] jm, int m, int n, double[]? scaleInvOld)
    {
        var si = new double[n];
        for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) si[j] += jm[i * n + j] * jm[i * n + j];
        for (var j = 0; j < n; j++)
        {
            si[j] = Math.Sqrt(si[j]);
            if (scaleInvOld is null) { if (si[j] == 0) si[j] = 1; }
            else si[j] = Math.Max(si[j], scaleInvOld[j]);
        }
        return (si.Select(v => 1 / v).ToArray(), si);
    }

    /// <summary>Thin SVD: U (rows, k), s (k), V (n, k) row-major.</summary>
    internal static (double[] U, double[] S, double[] V, int K) Svd(double[] a, int rows, int n)
    {
        var r = Decomposition.Svd(new NdArray((double[])a.Clone(), rows, n));
        var s = r.SingularValues.ToArray();
        return (r.U.AsContiguous().ToArray(), s, r.V.AsContiguous().ToArray(), s.Length);
    }

    // ---------- driver ----------

    public static LeastSquaresResult Run(Func<double[], double[]> fun, double[] x0In, Func<double[], NdArray>? jac, Bounds? bounds,
        LeastSquaresMethod method, double ftol, double xtol, double gtol, double[]? xScaleIn, bool xScaleJac, LeastSquaresLoss loss,
        Func<double, (double, double, double)>? customLoss, double fScale, double? diffStep, int? maxNfev, FiniteDiffScheme scheme,
        Func<double[], double, bool>? callback)
    {
        ArgumentNullException.ThrowIfNull(fun);
        ArgumentNullException.ThrowIfNull(x0In);
        var n = x0In.Length;
        if (n == 0) throw new ArgumentException("x0 must not be empty.");
        if (method == LeastSquaresMethod.Lm && (loss != LeastSquaresLoss.Linear || customLoss is not null))
            throw new ArgumentException("method Lm supports only the linear loss function.");
        if (maxNfev is <= 0) throw new ArgumentException("maxNfev must be null or positive.");
        var lb = bounds?.Lb ?? Enumerable.Repeat(double.NegativeInfinity, n).ToArray();
        var ub = bounds?.Ub ?? Enumerable.Repeat(double.PositiveInfinity, n).ToArray();
        if (lb.Length != n) throw new ArgumentException("Inconsistent shapes between bounds and x0.");
        var unbounded = lb.All(double.IsNegativeInfinity) && ub.All(double.IsPositiveInfinity);
        if (method == LeastSquaresMethod.Lm && !unbounded) throw new ArgumentException("Method Lm doesn't support bounds.");
        for (var i = 0; i < n; i++) if (lb[i] >= ub[i]) throw new ArgumentException("Each lower bound must be strictly less than each upper bound.");
        if (!InBounds(x0In, lb, ub)) throw new ArgumentException("Initial guess is outside of provided bounds");
        if (method == LeastSquaresMethod.Lm && (ftol < Eps || xtol < Eps || gtol < Eps))
            throw new ArgumentException($"All tolerances must be higher than machine epsilon ({Eps:E2}) for method 'lm'.");
        if (ftol < Eps && xtol < Eps && gtol < Eps) throw new ArgumentException($"At least one of the tolerances must be higher than machine epsilon ({Eps:E2}).");
        var jacScale = xScaleJac || (xScaleIn is null && method == LeastSquaresMethod.Lm);
        double[] xScale;
        if (xScaleIn is not null)
        {
            if (xScaleIn.Length != n) throw new ArgumentException("Inconsistent shapes between `x_scale` and `x0`.");
            if (xScaleIn.Any(v => !(v > 0) || !double.IsFinite(v))) throw new ArgumentException("`x_scale` must be 'jac' or array_like with positive numbers.");
            xScale = (double[])xScaleIn.Clone();
        }
        else xScale = Enumerable.Repeat(1.0, n).ToArray();

        var x0 = method == LeastSquaresMethod.Trf ? MakeStrictlyFeasible(x0In, lb, ub) : (double[])x0In.Clone();
        var prob = new Problem(fun, jac, n, scheme, diffStep, lb, ub);
        var f0 = prob.F(x0);
        prob.M = f0.Length;
        var m = f0.Length;
        if (m == 0) throw new ArgumentException("The residual function returned no values.");
        if (!f0.All(double.IsFinite)) throw new ArgumentException("Residuals are not finite in the initial point.");
        if (method == LeastSquaresMethod.Lm && m < n) throw new ArgumentException("Method 'lm' doesn't work when the number of residuals is less than the number of variables.");
        var lossFn = BuildLoss(loss, customLoss, fScale);

        if (method == LeastSquaresMethod.Lm)
        {
            // MINPACK counts its own evaluations. The Jacobian reuses the residuals of the point
            // MINPACK just evaluated, as SciPy's VectorFunction cache does.
            double[]? lastX = null, lastF = null;
            double[] CachedFun(double[] x)
            {
                var v = (double[])fun((double[])x.Clone()).Clone();
                lastX = (double[])x.Clone(); lastF = v;
                return v;
            }
            NdArray LmJac(double[] x)
            {
                var fx = lastX is not null && x.AsSpan().SequenceEqual(lastX) ? lastF! : fun((double[])x.Clone());
                return new NdArray(prob.J(x, fx), m, n);
            }
            var r = Optimize.MinpackLm(new MinpackLmProblem
            {
                Fun = CachedFun, Jac = LmJac,
                X0 = x0, Ftol = ftol, Xtol = xtol, Gtol = gtol, MaxFev = maxNfev ?? 100 * n,
                Diag = jacScale ? null : xScale.Select(v => 1 / v).ToArray(),
            });
            var fv = r.FunVector!;
            var jf = prob.J(r.X, fv);
            var g = new double[n];
            OptimizeDense.MatTVec(jf, m, n, fv, g);
            var status = r.Status switch { 0 => -1, 1 => 2, 2 => 3, 3 => 4, 4 => 1, 5 => 0, 6 => 2, 7 => 3, _ => 1 };
            return new LeastSquaresResult
            {
                X = r.X, Cost = 0.5 * OptimizeDense.Dot(fv, fv), Fun = fv, Jac = new NdArray(jf, m, n), Grad = g, Optimality = OptimizeDense.NormInf(g),
                ActiveMask = new int[n], Nfev = r.Nfev, Njev = jac is null ? null : r.Njev, Status = status, Message = Messages[status], Success = status > 0,
                CovX = r.HessInv,
            };
        }

        var j0 = prob.J(x0, f0);
        var res = method == LeastSquaresMethod.Trf
            ? (unbounded ? TrfNoBounds(prob, x0, f0, j0, m, n, ftol, xtol, gtol, maxNfev ?? 100 * n, xScale, jacScale, lossFn, callback)
                         : TrfBounds(prob, x0, f0, j0, m, n, lb, ub, ftol, xtol, gtol, maxNfev ?? 100 * n, xScale, jacScale, lossFn, callback))
            : Dogbox(prob, x0, f0, j0, m, n, lb, ub, ftol, xtol, gtol, maxNfev ?? 100 * n, xScale, jacScale, lossFn, callback);
        return res;
    }

    private static LeastSquaresResult Make(double[] x, double cost, double[] fTrue, double[] jm, int m, int n, double[] g, double gNorm, int[] active,
        Problem p, int status) => new()
    {
        X = x, Cost = cost, Fun = fTrue, Jac = new NdArray(jm, m, n), Grad = g, Optimality = gNorm, ActiveMask = active,
        Nfev = p.Nfev, Njev = p.Njev, Status = status, Message = Messages[status], Success = status > 0,
    };

    private static LeastSquaresResult TrfNoBounds(Problem p, double[] x0, double[] f0, double[] j0, int m, int n, double ftol, double xtol, double gtol,
        int maxNfev, double[] xScale, bool jacScale, LossRho? loss, Func<double[], double, bool>? callback)
    {
        var x = (double[])x0.Clone();
        var f = (double[])f0.Clone();
        var fTrue = (double[])f.Clone();
        var jm = (double[])j0.Clone();
        var cost = loss is null ? 0.5 * OptimizeDense.Dot(f, f) : ApplyLoss(loss, jm, f, m, n);
        var g = new double[n];
        OptimizeDense.MatTVec(jm, m, n, f, g);
        double[] scale, scaleInv;
        if (jacScale) (scale, scaleInv) = JacScale(jm, m, n, null);
        else { scale = xScale; scaleInv = xScale.Select(v => 1 / v).ToArray(); }
        var delta = OptimizeDense.Norm2(x0.Select((v, i) => v * scaleInv[i]).ToArray());
        if (delta == 0) delta = 1.0;
        double alpha = 0;
        int? status = null;
        double gNorm;
        while (true)
        {
            gNorm = OptimizeDense.NormInf(g);
            if (gNorm < gtol) status = 1;
            if (status is not null || p.Nfev == maxNfev) break;
            var d = scale;
            var gh = g.Select((v, i) => d[i] * v).ToArray();
            var jh = new double[m * n];
            for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) jh[i * n + j] = jm[i * n + j] * d[j];
            var (u, s, vv, k) = Svd(jh, m, n);
            var uc = u.Length / m;
            var uf = new double[k];
            for (var c = 0; c < k; c++) { double acc = 0; for (var i = 0; i < m; i++) acc += u[i * uc + c] * f[i]; uf[c] = acc; }
            double actual = -1;
            double[] xNew = x, fNew = f;
            double costNew = cost;
            while (actual <= 0 && p.Nfev < maxNfev)
            {
                double[] stepH;
                (stepH, alpha) = SolveLsqTrustRegion(n, m, uf, s, vv, k, delta, alpha);
                var predicted = -EvaluateQuadratic(jh, m, n, gh, stepH);
                var step = stepH.Select((v, i) => d[i] * v).ToArray();
                xNew = x.Select((v, i) => v + step[i]).ToArray();
                fNew = p.F(xNew);
                var stepHNorm = OptimizeDense.Norm2(stepH);
                if (!fNew.All(double.IsFinite)) { delta = 0.25 * stepHNorm; continue; }
                costNew = LossCost(loss, fNew);
                actual = cost - costNew;
                var (deltaNew, ratio) = UpdateTrRadius(delta, actual, predicted, stepHNorm, stepHNorm > 0.95 * delta);
                status = CheckTermination(actual, cost, OptimizeDense.Norm2(step), OptimizeDense.Norm2(x), ratio, ftol, xtol);
                if (status is not null) break;
                alpha *= delta / deltaNew;
                delta = deltaNew;
            }
            if (actual > 0)
            {
                x = xNew; f = fNew; fTrue = (double[])f.Clone(); cost = costNew;
                jm = p.J(x, f);
                if (loss is not null) ApplyLoss(loss, jm, f, m, n);
                OptimizeDense.MatTVec(jm, m, n, f, g);
                if (jacScale) (scale, scaleInv) = JacScale(jm, m, n, scaleInv);
            }
            if (callback is not null && callback((double[])x.Clone(), cost)) { status = -2; break; }
        }
        return Make(x, cost, fTrue, jm, m, n, (double[])g.Clone(), gNorm, new int[n], p, status ?? 0);
    }

    private static LeastSquaresResult TrfBounds(Problem p, double[] x0, double[] f0, double[] j0, int m, int n, double[] lb, double[] ub,
        double ftol, double xtol, double gtol, int maxNfev, double[] xScale, bool jacScale, LossRho? loss, Func<double[], double, bool>? callback)
    {
        var x = (double[])x0.Clone();
        var f = (double[])f0.Clone();
        var fTrue = (double[])f.Clone();
        var jm = (double[])j0.Clone();
        var cost = loss is null ? 0.5 * OptimizeDense.Dot(f, f) : ApplyLoss(loss, jm, f, m, n);
        var g = new double[n];
        OptimizeDense.MatTVec(jm, m, n, f, g);
        double[] scale, scaleInv;
        if (jacScale) (scale, scaleInv) = JacScale(jm, m, n, null);
        else { scale = xScale; scaleInv = xScale.Select(v => 1 / v).ToArray(); }
        var (v0, dv0) = ClScalingVector(x, g, lb, ub);
        for (var i = 0; i < n; i++) if (dv0[i] != 0) v0[i] *= scaleInv[i];
        var delta = OptimizeDense.Norm2(x0.Select((xv, i) => xv * scaleInv[i] / Math.Sqrt(v0[i])).ToArray());
        if (delta == 0) delta = 1.0;
        double alpha = 0;
        int? status = null;
        double gNorm;
        var jaug = new double[(m + n) * n];
        var faug = new double[m + n];
        while (true)
        {
            var (v, dv) = ClScalingVector(x, g, lb, ub);
            gNorm = 0;
            for (var i = 0; i < n; i++) gNorm = Math.Max(gNorm, Math.Abs(g[i] * v[i]));
            if (gNorm < gtol) status = 1;
            if (status is not null || p.Nfev == maxNfev) break;
            for (var i = 0; i < n; i++) if (dv[i] != 0) v[i] *= scaleInv[i];
            var d = v.Select((vi, i) => Math.Sqrt(vi) * scale[i]).ToArray();
            var diagH = g.Select((gi, i) => gi * dv[i] * scale[i]).ToArray();
            var gh = g.Select((gi, i) => d[i] * gi).ToArray();
            Array.Clear(faug);
            Array.Copy(f, faug, m);
            Array.Clear(jaug);
            for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) jaug[i * n + j] = jm[i * n + j] * d[j];
            for (var j = 0; j < n; j++) jaug[(m + j) * n + j] = Math.Sqrt(diagH[j]);
            var jh = jaug.AsSpan(0, m * n).ToArray();
            var (u, s, vv, k) = Svd(jaug, m + n, n);
            var uc = u.Length / (m + n);
            var uf = new double[k];
            for (var c = 0; c < k; c++) { double acc = 0; for (var i = 0; i < m + n; i++) acc += u[i * uc + c] * faug[i]; uf[c] = acc; }
            var theta = Math.Max(0.995, 1 - gNorm);
            double actual = -1;
            double[] xNew = x, fNew = f;
            double costNew = cost;
            while (actual <= 0 && p.Nfev < maxNfev)
            {
                var (ph, al) = SolveLsqTrustRegion(n, m, uf, s, vv, k, delta, alpha);
                alpha = al;
                var pv = ph.Select((t, i) => d[i] * t).ToArray();
                var (step, stepH, predicted) = SelectStep(x, jh, m, n, diagH, gh, pv, ph, d, delta, lb, ub, theta);
                xNew = MakeStrictlyFeasible(x.Select((xv, i) => xv + step[i]).ToArray(), lb, ub, 0);
                fNew = p.F(xNew);
                var stepHNorm = OptimizeDense.Norm2(stepH);
                if (!fNew.All(double.IsFinite)) { delta = 0.25 * stepHNorm; continue; }
                costNew = LossCost(loss, fNew);
                actual = cost - costNew;
                var (deltaNew, ratio) = UpdateTrRadius(delta, actual, predicted, stepHNorm, stepHNorm > 0.95 * delta);
                status = CheckTermination(actual, cost, OptimizeDense.Norm2(step), OptimizeDense.Norm2(x), ratio, ftol, xtol);
                if (status is not null) break;
                alpha *= delta / deltaNew;
                delta = deltaNew;
            }
            if (actual > 0)
            {
                x = xNew; f = fNew; fTrue = (double[])f.Clone(); cost = costNew;
                jm = p.J(x, f);
                if (loss is not null) ApplyLoss(loss, jm, f, m, n);
                OptimizeDense.MatTVec(jm, m, n, f, g);
                if (jacScale) (scale, scaleInv) = JacScale(jm, m, n, scaleInv);
            }
            if (callback is not null && callback((double[])x.Clone(), cost)) { status = -2; break; }
        }
        return Make(x, cost, fTrue, jm, m, n, (double[])g.Clone(), gNorm, FindActiveConstraints(x, lb, ub, xtol), p, status ?? 0);
    }

    private static (double[] Step, double[] StepH, double Predicted) SelectStep(double[] x, double[] jh, int m, int n, double[] diagH, double[] gh,
        double[] p, double[] ph, double[] d, double delta, double[] lb, double[] ub, double theta)
    {
        if (InBounds(x.Select((v, i) => v + p[i]).ToArray(), lb, ub))
            return (p, ph, -EvaluateQuadratic(jh, m, n, gh, ph, diagH));
        var (pStride, hits) = StepSizeToBound(x, p, lb, ub);
        var rh = (double[])ph.Clone();
        for (var i = 0; i < n; i++) if (hits[i] != 0) rh[i] = -rh[i];
        var r = rh.Select((v, i) => d[i] * v).ToArray();
        p = p.Select(v => v * pStride).ToArray();
        ph = ph.Select(v => v * pStride).ToArray();
        var xOnBound = x.Select((v, i) => v + p[i]).ToArray();
        var (_, toTr) = IntersectTrustRegion(ph, rh, delta);
        var (toBound, _) = StepSizeToBound(xOnBound, r, lb, ub);
        var rStride = Math.Min(toBound, toTr);
        double rStrideL, rStrideU;
        if (rStride > 0)
        {
            rStrideL = (1 - theta) * pStride / rStride;
            rStrideU = rStride == toBound ? theta * toBound : toTr;
        }
        else { rStrideL = 0; rStrideU = -1; }
        double rValue;
        if (rStrideL <= rStrideU)
        {
            var (a, b, c) = BuildQuadratic1d(jh, m, n, gh, rh, diagH, ph);
            var (rs, rv) = MinimizeQuadratic1d(a, b, rStrideL, rStrideU, c);
            rValue = rv;
            for (var i = 0; i < n; i++) rh[i] = rh[i] * rs + ph[i];
            r = rh.Select((v, i) => v * d[i]).ToArray();
        }
        else rValue = double.PositiveInfinity;
        p = p.Select(v => v * theta).ToArray();
        ph = ph.Select(v => v * theta).ToArray();
        var pValue = EvaluateQuadratic(jh, m, n, gh, ph, diagH);
        var agh = gh.Select(v => -v).ToArray();
        var ag = agh.Select((v, i) => d[i] * v).ToArray();
        toTr = delta / OptimizeDense.Norm2(agh);
        (toBound, _) = StepSizeToBound(x, ag, lb, ub);
        var agStride = toBound < toTr ? theta * toBound : toTr;
        var (a2, b2, _) = BuildQuadratic1d(jh, m, n, gh, agh, diagH);
        var (ags, agValue) = MinimizeQuadratic1d(a2, b2, 0, agStride);
        agh = agh.Select(v => v * ags).ToArray();
        ag = ag.Select(v => v * ags).ToArray();
        if (pValue < rValue && pValue < agValue) return (p, ph, -pValue);
        if (rValue < pValue && rValue < agValue) return (r, rh, -rValue);
        return (ag, agh, -agValue);
    }

    private static LeastSquaresResult Dogbox(Problem p, double[] x0, double[] f0, double[] j0, int m, int n, double[] lb, double[] ub,
        double ftol, double xtol, double gtol, int maxNfev, double[] xScale, bool jacScale, LossRho? loss, Func<double[], double, bool>? callback)
    {
        var f = (double[])f0.Clone();
        var fTrue = (double[])f.Clone();
        var jm = (double[])j0.Clone();
        var cost = loss is null ? 0.5 * OptimizeDense.Dot(f, f) : ApplyLoss(loss, jm, f, m, n);
        var g = new double[n];
        OptimizeDense.MatTVec(jm, m, n, f, g);
        double[] scale, scaleInv;
        if (jacScale) (scale, scaleInv) = JacScale(jm, m, n, null);
        else { scale = xScale; scaleInv = xScale.Select(v => 1 / v).ToArray(); }
        var delta = OptimizeDense.NormInf(x0.Select((v, i) => v * scaleInv[i]).ToArray());
        if (delta == 0) delta = 1.0;
        var onBound = new int[n];
        for (var i = 0; i < n; i++) { if (x0[i] == lb[i]) onBound[i] = -1; if (x0[i] == ub[i]) onBound[i] = 1; }
        var x = (double[])x0.Clone();
        int? status = null;
        double gNorm;
        double[] gFull;
        while (true)
        {
            var free = new List<int>();
            for (var i = 0; i < n; i++) if (!(onBound[i] * g[i] < 0)) free.Add(i);
            gFull = (double[])g.Clone();
            for (var i = 0; i < n; i++) if (onBound[i] * g[i] < 0) g[i] = 0;
            gNorm = OptimizeDense.NormInf(g);
            if (gNorm < gtol) status = 1;
            if (status is not null || p.Nfev == maxNfev) break;
            var nf = free.Count;
            var xFree = free.Select(i => x[i]).ToArray();
            var lbFree = free.Select(i => lb[i]).ToArray();
            var ubFree = free.Select(i => ub[i]).ToArray();
            var scaleFree = free.Select(i => scale[i]).ToArray();
            var gFree = free.Select(i => g[i]).ToArray();
            var jFree = new double[m * nf];
            for (var r = 0; r < m; r++) for (var c = 0; c < nf; c++) jFree[r * nf + c] = jm[r * n + free[c]];
            var negF = f.Select(v => -v).ToArray();
            var newton = OptimizeDense.LstsqSvd(jFree, m, nf, negF);
            var (qa, qb, _) = BuildQuadratic1d(jFree, m, nf, gFree, gFree.Select(v => -v).ToArray());
            double actual = -1;
            double[] xNew = x, fNew = f;
            double costNew = cost;
            int[] onBoundFree = new int[nf];
            while (actual <= 0 && p.Nfev < maxNfev)
            {
                var trBounds = scaleFree.Select(v => delta * v).ToArray();
                (var stepFree, onBoundFree, var trHit) = DoglegStep(xFree, newton, gFree, qa, qb, trBounds, lbFree, ubFree);
                var step = new double[n];
                for (var c = 0; c < nf; c++) step[free[c]] = stepFree[c];
                var predicted = -EvaluateQuadratic(jFree, m, nf, gFree, stepFree);
                xNew = x.Select((v, i) => Math.Clamp(v + step[i], lb[i], ub[i])).ToArray();
                fNew = p.F(xNew);
                double stepHNorm = 0;
                for (var i = 0; i < n; i++) stepHNorm = Math.Max(stepHNorm, Math.Abs(step[i] * scaleInv[i]));
                if (!fNew.All(double.IsFinite)) { delta = 0.25 * stepHNorm; continue; }
                costNew = LossCost(loss, fNew);
                actual = cost - costNew;
                (delta, var ratio) = UpdateTrRadius(delta, actual, predicted, stepHNorm, trHit);
                status = CheckTermination(actual, cost, OptimizeDense.Norm2(step), OptimizeDense.Norm2(x), ratio, ftol, xtol);
                if (status is not null) break;
            }
            if (actual > 0)
            {
                for (var c = 0; c < nf; c++) onBound[free[c]] = onBoundFree[c];
                x = xNew;
                for (var i = 0; i < n; i++) { if (onBound[i] == -1) x[i] = lb[i]; if (onBound[i] == 1) x[i] = ub[i]; }
                f = fNew; fTrue = (double[])f.Clone(); cost = costNew;
                jm = p.J(x, f);
                if (loss is not null) ApplyLoss(loss, jm, f, m, n);
                OptimizeDense.MatTVec(jm, m, n, f, g);
                if (jacScale) (scale, scaleInv) = JacScale(jm, m, n, scaleInv);
            }
            if (callback is not null && callback((double[])x.Clone(), cost)) { status = -2; break; }
        }
        return Make(x, cost, fTrue, jm, m, n, gFull, gNorm, onBound, p, status ?? 0);
    }

    private static (double[] Step, int[] BoundHits, bool TrHit) DoglegStep(double[] x, double[] newton, double[] g, double a, double b,
        double[] trBounds, double[] lb, double[] ub)
    {
        var n = x.Length;
        var lbT = new double[n]; var ubT = new double[n];
        var origL = new bool[n]; var origU = new bool[n]; var trL = new bool[n]; var trU = new bool[n];
        for (var i = 0; i < n; i++)
        {
            var lc = lb[i] - x[i]; var uc = ub[i] - x[i];
            lbT[i] = Math.Max(lc, -trBounds[i]); ubT[i] = Math.Min(uc, trBounds[i]);
            origL[i] = lbT[i] == lc; origU[i] = ubT[i] == uc; trL[i] = lbT[i] == -trBounds[i]; trU[i] = ubT[i] == trBounds[i];
        }
        var hitsOut = new int[n];
        if (InBounds(newton, lbT, ubT)) return (newton, hitsOut, false);
        var (toBounds, _) = StepSizeToBound(new double[n], g.Select(v => -v).ToArray(), lbT, ubT);
        var t = MinimizeQuadratic1d(a, b, 0, toBounds).T;
        var cauchy = g.Select(v => -t * v).ToArray();
        var diff = newton.Select((v, i) => v - cauchy[i]).ToArray();
        var (stepSize, hits) = StepSizeToBound(cauchy, diff, lbT, ubT);
        var trHit = false;
        for (var i = 0; i < n; i++)
        {
            if (hits[i] < 0 && origL[i]) hitsOut[i] = -1;
            if (hits[i] > 0 && origU[i]) hitsOut[i] = 1;
            if ((hits[i] < 0 && trL[i]) || (hits[i] > 0 && trU[i])) trHit = true;
        }
        return (cauchy.Select((v, i) => v + stepSize * diff[i]).ToArray(), hitsOut, trHit);
    }
}
