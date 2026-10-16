using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>Outcome of a Krylov solver.</summary>
/// <param name="X">The approximate solution.</param>
/// <param name="Info">0 converged; &gt; 0 the iteration limit was hit without converging; &lt; 0 breakdown (SciPy's codes, e.g. −10 for a ρ breakdown).</param>
/// <param name="Iterations">Iterations performed (inner iterations for GMRES/LGMRES).</param>
/// <param name="ResidualNorm">True residual <c>‖b − A x‖₂</c> recomputed at exit — not the solver's internal estimate.</param>
public sealed record IterativeSolveResult(double[] X, int Info, int Iterations, double ResidualNorm)
{
    /// <summary>True when <see cref="Info"/> is zero.</summary>
    public bool Converged => Info == 0;
}

/// <summary>Outcome of <see cref="SparseLinalg.Lsqr"/>, mirroring SciPy's tuple.</summary>
/// <param name="X">Least-squares solution.</param>
/// <param name="IStop">Reason for stopping (SciPy's codes: 0 x=0 solves it, 1 Ax=b within tolerance, 2 least-squares within tolerance, 3 cond(A) &gt; conlim, 4–6 the same at machine precision, 7 iteration limit).</param>
/// <param name="Itn">Iterations performed.</param>
/// <param name="R1Norm"><c>‖b − A x‖</c>.</param>
/// <param name="R2Norm"><c>sqrt(‖b − A x‖² + damp²‖x‖²)</c>.</param>
/// <param name="ANorm">Estimate of the Frobenius norm of <c>[A; damp·I]</c>.</param>
/// <param name="ACond">Estimate of its condition number.</param>
/// <param name="ArNorm">Estimate of <c>‖Aᵀr − damp²x‖</c>.</param>
/// <param name="XNorm"><c>‖x‖</c>.</param>
/// <param name="Var">Diagonal of <c>(AᵀA + damp²I)⁻¹</c> estimate when requested, else zeros.</param>
public sealed record LsqrResult(double[] X, int IStop, int Itn, double R1Norm, double R2Norm, double ANorm, double ACond, double ArNorm, double XNorm, double[] Var);

/// <summary>Outcome of <see cref="SparseLinalg.Lsmr"/>, mirroring SciPy's tuple.</summary>
/// <param name="X">Least-squares solution.</param>
/// <param name="IStop">Reason for stopping (same codes as <see cref="LsqrResult.IStop"/>).</param>
/// <param name="Itn">Iterations performed.</param>
/// <param name="NormR"><c>‖b − A x‖</c> estimate.</param>
/// <param name="NormAr"><c>‖Aᵀ(b − A x)‖</c> estimate.</param>
/// <param name="NormA">Estimate of the Frobenius norm of <c>[A; damp·I]</c>.</param>
/// <param name="CondA">Estimate of its condition number.</param>
/// <param name="NormX"><c>‖x‖</c>.</param>
public sealed record LsmrResult(double[] X, int IStop, int Itn, double NormR, double NormAr, double NormA, double CondA, double NormX);

/// <summary>
/// <c>scipy.sparse.linalg</c>: Krylov solvers, sparse direct factorisations, sparse eigen/SVD
/// solvers and the action of the matrix exponential.
/// </summary>
/// <remarks>
/// Every Krylov solver follows SciPy's contract: it stops once <c>‖b − A x‖ ≤ max(atol, rtol·‖b‖)</c>,
/// the preconditioner <c>M</c> approximates <c>A⁻¹</c> (its <c>MatVec</c> applies <c>M r</c>),
/// <c>maxiter</c> defaults to <c>10 n</c>, and the callback receives the current iterate. The
/// iterations are ports of the Templates algorithms SciPy uses, so iteration counts agree with
/// SciPy's to within rounding on well-conditioned problems.
/// </remarks>
public static partial class SparseLinalg
{
    private const double Eps = 2.220446049250313e-16;

    // ------------------------------------------------------------------ small vector helpers

    private static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b) => LinAlg.InnerSpan(a, b);

    private static double Nrm2(ReadOnlySpan<double> a)
    {
        double scale = 0, ssq = 1;
        foreach (var x in a)
        {
            if (x == 0) continue;
            var ax = Math.Abs(x);
            if (scale < ax) { ssq = 1 + ssq * (scale / ax) * (scale / ax); scale = ax; }
            else ssq += (ax / scale) * (ax / scale);
        }
        return scale * Math.Sqrt(ssq);
    }

    private static void Axpy(double alpha, ReadOnlySpan<double> x, Span<double> y) => LinAlg.AxpySpan(alpha, x, y);

    private static void ScaleInPlace(Span<double> x, double alpha)
    {
        for (var i = 0; i < x.Length; i++) x[i] *= alpha;
    }

    private sealed class SolverSetup
    {
        public required ILinearOperator A;
        public required double[] B;
        public required double[] X;
        public required double[] R;
        public required double BNorm;
        public required double Tol;
        public required int MaxIter;
        public required ILinearOperator? M;

        public void Precondition(ReadOnlySpan<double> r, Span<double> z)
        {
            if (M is null) r.CopyTo(z);
            else M.MatVec(r, z);
        }

        public void PreconditionT(ReadOnlySpan<double> r, Span<double> z)
        {
            if (M is null) r.CopyTo(z);
            else M.RMatVec(r, z);
        }

        public IterativeSolveResult Finish(int info, int iterations)
        {
            var ax = new double[X.Length];
            A.MatVec(X, ax);
            for (var i = 0; i < ax.Length; i++) ax[i] = B[i] - ax[i];
            return new IterativeSolveResult(X, info, iterations, Nrm2(ax));
        }
    }

    private static SolverSetup Setup(ILinearOperator a, double[] b, double[]? x0, double rtol, double atol, int? maxiter, ILinearOperator? m, string name)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Rows != a.Columns) throw new ArgumentException($"{name} needs a square operator, got {a.Rows}x{a.Columns}.");
        var n = a.Rows;
        if (b.Length != n) throw new ArgumentException($"b has {b.Length} entries, expected {n}.");
        if (x0 is not null && x0.Length != n) throw new ArgumentException($"x0 has {x0.Length} entries, expected {n}.");
        if (m is not null && (m.Rows != n || m.Columns != n)) throw new ArgumentException("Preconditioner shape does not match A.");
        if (rtol < 0 || atol < 0) throw new ArgumentException("Tolerances must be non-negative.");
        var x = x0 is null ? new double[n] : (double[])x0.Clone();
        var r = new double[n];
        if (x0 is null) b.CopyTo(r, 0);
        else
        {
            a.MatVec(x, r);
            for (var i = 0; i < n; i++) r[i] = b[i] - r[i];
        }
        var bnorm = Nrm2(b);
        return new SolverSetup
        {
            A = a, B = b, X = x, R = r, BNorm = bnorm, Tol = Math.Max(atol, rtol * bnorm),
            MaxIter = maxiter ?? n * 10, M = m,
        };
    }

    private static IterativeSolveResult ZeroRhs(int n) => new(new double[n], 0, 0, 0.0);

    // ------------------------------------------------------------------ CG

    /// <summary>
    /// Conjugate gradients for symmetric positive definite <c>A</c> (and SPD preconditioner).
    /// </summary>
    /// <param name="a">The SPD operator.</param>
    /// <param name="b">Right-hand side.</param>
    /// <param name="x0">Starting guess (zeros by default).</param>
    /// <param name="rtol">Relative tolerance on the residual.</param>
    /// <param name="atol">Absolute tolerance on the residual.</param>
    /// <param name="maxiter">Iteration limit (default <c>10 n</c>).</param>
    /// <param name="m">Preconditioner approximating <c>A⁻¹</c>.</param>
    /// <param name="callback">Called with the iterate after each step.</param>
    public static IterativeSolveResult Cg(ILinearOperator a, double[] b, double[]? x0 = null, double rtol = 1e-5, double atol = 0.0,
        int? maxiter = null, ILinearOperator? m = null, Action<double[]>? callback = null)
    {
        var s = Setup(a, b, x0, rtol, atol, maxiter, m, nameof(Cg));
        var n = b.Length;
        if (s.BNorm == 0) return ZeroRhs(n);
        var r = s.R; var x = s.X;
        var z = new double[n]; var p = new double[n]; var q = new double[n];
        var rhoPrev = 0.0;
        for (var it = 0; it < s.MaxIter; it++)
        {
            if (Nrm2(r) <= s.Tol) return s.Finish(0, it);
            s.Precondition(r, z);
            var rho = Dot(r, z);
            if (it > 0)
            {
                var beta = rho / rhoPrev;
                for (var i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
            }
            else z.CopyTo(p, 0);
            a.MatVec(p, q);
            var pq = Dot(p, q);
            if (pq == 0) return s.Finish(-11, it + 1);
            var alpha = rho / pq;
            Axpy(alpha, p, x);
            Axpy(-alpha, q, r);
            rhoPrev = rho;
            callback?.Invoke(x);
        }
        return Nrm2(r) <= s.Tol ? s.Finish(0, s.MaxIter) : s.Finish(s.MaxIter, s.MaxIter);
    }

    // ------------------------------------------------------------------ BiCG

    /// <summary>Biconjugate gradients for general square <c>A</c>; needs <c>Aᵀ</c> products (and <c>Mᵀ</c> if preconditioned).</summary>
    /// <inheritdoc cref="Cg" path="/param"/>
    public static IterativeSolveResult Bicg(ILinearOperator a, double[] b, double[]? x0 = null, double rtol = 1e-5, double atol = 0.0,
        int? maxiter = null, ILinearOperator? m = null, Action<double[]>? callback = null)
    {
        var s = Setup(a, b, x0, rtol, atol, maxiter, m, nameof(Bicg));
        var n = b.Length;
        if (s.BNorm == 0) return ZeroRhs(n);
        var r = s.R; var x = s.X;
        var rt = (double[])r.Clone();
        var z = new double[n]; var zt = new double[n]; var p = new double[n]; var pt = new double[n];
        var q = new double[n]; var qt = new double[n];
        double rhoPrev = 0, rhotol = Eps * Eps;
        for (var it = 0; it < s.MaxIter; it++)
        {
            if (Nrm2(r) <= s.Tol) return s.Finish(0, it);
            s.Precondition(r, z);
            s.PreconditionT(rt, zt);
            var rho = Dot(rt, z);
            if (Math.Abs(rho) < rhotol) return s.Finish(-10, it);
            if (it > 0)
            {
                var beta = rho / rhoPrev;
                for (var i = 0; i < n; i++) { p[i] = z[i] + beta * p[i]; pt[i] = zt[i] + beta * pt[i]; }
            }
            else { z.CopyTo(p, 0); zt.CopyTo(pt, 0); }
            a.MatVec(p, q);
            a.RMatVec(pt, qt);
            var rv = Dot(pt, q);
            if (rv == 0) return s.Finish(-11, it);
            var alpha = rho / rv;
            Axpy(alpha, p, x);
            Axpy(-alpha, q, r);
            Axpy(-alpha, qt, rt);
            rhoPrev = rho;
            callback?.Invoke(x);
        }
        return Nrm2(r) <= s.Tol ? s.Finish(0, s.MaxIter) : s.Finish(s.MaxIter, s.MaxIter);
    }

    // ------------------------------------------------------------------ CGS

    /// <summary>Conjugate gradients squared (Sonneveld) for general square <c>A</c>; no transpose products needed.</summary>
    /// <inheritdoc cref="Cg" path="/param"/>
    public static IterativeSolveResult Cgs(ILinearOperator a, double[] b, double[]? x0 = null, double rtol = 1e-5, double atol = 0.0,
        int? maxiter = null, ILinearOperator? m = null, Action<double[]>? callback = null)
    {
        var s = Setup(a, b, x0, rtol, atol, maxiter, m, nameof(Cgs));
        var n = b.Length;
        if (s.BNorm == 0) return ZeroRhs(n);
        var r = s.R; var x = s.X;
        var rt = (double[])r.Clone();
        var p = new double[n]; var u = new double[n]; var q = new double[n];
        var phat = new double[n]; var vhat = new double[n]; var uhat = new double[n]; var qhat = new double[n]; var tmp = new double[n];
        double rhoPrev = 0, rhotol = Eps * Eps;
        for (var it = 0; it < s.MaxIter; it++)
        {
            if (Nrm2(r) <= s.Tol) return s.Finish(0, it);
            var rho = Dot(rt, r);
            if (Math.Abs(rho) < rhotol) return s.Finish(-10, it);
            if (it > 0)
            {
                var beta = rho / rhoPrev;
                for (var i = 0; i < n; i++)
                {
                    u[i] = r[i] + beta * q[i];
                    p[i] = u[i] + beta * (q[i] + beta * p[i]);
                }
            }
            else { r.CopyTo(u, 0); r.CopyTo(p, 0); }
            s.Precondition(p, phat);
            a.MatVec(phat, vhat);
            var rv = Dot(rt, vhat);
            if (rv == 0) return s.Finish(-11, it);
            var alpha = rho / rv;
            for (var i = 0; i < n; i++) { q[i] = u[i] - alpha * vhat[i]; tmp[i] = u[i] + q[i]; }
            s.Precondition(tmp, uhat);
            Axpy(alpha, uhat, x);
            a.MatVec(uhat, qhat);
            Axpy(-alpha, qhat, r);
            rhoPrev = rho;
            callback?.Invoke(x);
        }
        return Nrm2(r) <= s.Tol ? s.Finish(0, s.MaxIter) : s.Finish(s.MaxIter, s.MaxIter);
    }

    // ------------------------------------------------------------------ BiCGSTAB

    /// <summary>Stabilised biconjugate gradients (van der Vorst) for general square <c>A</c>.</summary>
    /// <inheritdoc cref="Cg" path="/param"/>
    public static IterativeSolveResult Bicgstab(ILinearOperator a, double[] b, double[]? x0 = null, double rtol = 1e-5, double atol = 0.0,
        int? maxiter = null, ILinearOperator? m = null, Action<double[]>? callback = null)
    {
        var s = Setup(a, b, x0, rtol, atol, maxiter, m, nameof(Bicgstab));
        var n = b.Length;
        if (s.BNorm == 0) return ZeroRhs(n);
        var r = s.R; var x = s.X;
        var rt = (double[])r.Clone();
        var p = new double[n]; var v = new double[n]; var phat = new double[n]; var shat = new double[n]; var t = new double[n];
        double rhoPrev = 0, omega = 0, alpha = 0, rhotol = Eps * Eps, omegatol = rhotol;
        for (var it = 0; it < s.MaxIter; it++)
        {
            if (Nrm2(r) <= s.Tol) return s.Finish(0, it);
            var rho = Dot(rt, r);
            if (Math.Abs(rho) < rhotol) return s.Finish(-10, it);
            if (it > 0)
            {
                if (Math.Abs(omega) < omegatol) return s.Finish(-11, it);
                var beta = rho / rhoPrev * (alpha / omega);
                for (var i = 0; i < n; i++) p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
            else r.CopyTo(p, 0);
            s.Precondition(p, phat);
            a.MatVec(phat, v);
            var rv = Dot(rt, v);
            if (rv == 0) return s.Finish(-11, it);
            alpha = rho / rv;
            Axpy(-alpha, v, r); // r now holds s
            if (Nrm2(r) <= s.Tol)
            {
                Axpy(alpha, phat, x);
                callback?.Invoke(x);
                return s.Finish(0, it + 1);
            }
            s.Precondition(r, shat);
            a.MatVec(shat, t);
            var tt = Dot(t, t);
            omega = tt == 0 ? 0 : Dot(t, r) / tt;
            Axpy(alpha, phat, x);
            Axpy(omega, shat, x);
            Axpy(-omega, t, r);
            rhoPrev = rho;
            callback?.Invoke(x);
        }
        return Nrm2(r) <= s.Tol ? s.Finish(0, s.MaxIter) : s.Finish(s.MaxIter, s.MaxIter);
    }

    // ------------------------------------------------------------------ GMRES

    /// <summary>
    /// Restarted GMRES (Saad &amp; Schultz) with right preconditioning, so the residual it minimises
    /// is the true one. Arnoldi uses modified Gram–Schmidt with one selective re-orthogonalisation.
    /// </summary>
    /// <param name="a">The square operator.</param>
    /// <param name="b">Right-hand side.</param>
    /// <param name="x0">Starting guess.</param>
    /// <param name="rtol">Relative tolerance.</param>
    /// <param name="atol">Absolute tolerance.</param>
    /// <param name="restart">Krylov dimension per cycle (default 20, capped at n).</param>
    /// <param name="maxiter">Maximum number of restart cycles (default <c>10 n</c>, as SciPy).</param>
    /// <param name="m">Preconditioner approximating <c>A⁻¹</c>.</param>
    /// <param name="callback">Called with the iterate after each restart cycle.</param>
    public static IterativeSolveResult Gmres(ILinearOperator a, double[] b, double[]? x0 = null, double rtol = 1e-5, double atol = 0.0,
        int restart = 20, int? maxiter = null, ILinearOperator? m = null, Action<double[]>? callback = null)
    {
        var s = Setup(a, b, x0, rtol, atol, maxiter, m, nameof(Gmres));
        var n = b.Length;
        if (s.BNorm == 0) return ZeroRhs(n);
        if (restart < 1) throw new ArgumentException("restart must be positive.");
        var mDim = Math.Min(restart, n);
        var r = s.R; var x = s.X;
        var v = new double[mDim + 1][];
        for (var j = 0; j <= mDim; j++) v[j] = new double[n];
        var h = new double[mDim + 1, mDim];
        var cs = new double[mDim]; var sn = new double[mDim]; var g = new double[mDim + 1];
        var z = new double[n]; var w = new double[n]; var dx = new double[n];
        var total = 0;
        for (var cycle = 0; cycle < s.MaxIter; cycle++)
        {
            var beta = Nrm2(r);
            if (beta <= s.Tol) return s.Finish(0, total);
            for (var i = 0; i < n; i++) v[0][i] = r[i] / beta;
            Array.Clear(g);
            g[0] = beta;
            var k = 0;
            for (; k < mDim; k++)
            {
                total++;
                s.Precondition(v[k], z);
                a.MatVec(z, w);
                var wNorm0 = Nrm2(w);
                for (var i = 0; i <= k; i++)
                {
                    var hik = Dot(w, v[i]);
                    h[i, k] = hik;
                    Axpy(-hik, v[i], w);
                }
                var hn = Nrm2(w);
                if (hn < 0.7071 * wNorm0)
                {
                    // Lost orthogonality to cancellation: one more pass (Kahan's "twice is enough").
                    for (var i = 0; i <= k; i++)
                    {
                        var c = Dot(w, v[i]);
                        h[i, k] += c;
                        Axpy(-c, v[i], w);
                    }
                    hn = Nrm2(w);
                }
                h[k + 1, k] = hn;
                if (hn != 0) for (var i = 0; i < n; i++) v[k + 1][i] = w[i] / hn;
                for (var i = 0; i < k; i++)
                {
                    var t = cs[i] * h[i, k] + sn[i] * h[i + 1, k];
                    h[i + 1, k] = -sn[i] * h[i, k] + cs[i] * h[i + 1, k];
                    h[i, k] = t;
                }
                var (c1, s1, rr) = SymOrtho(h[k, k], h[k + 1, k]);
                cs[k] = c1; sn[k] = s1;
                h[k, k] = rr;
                h[k + 1, k] = 0;
                g[k + 1] = -s1 * g[k];
                g[k] = c1 * g[k];
                if (Math.Abs(g[k + 1]) <= s.Tol || hn == 0) { k++; break; }
            }
            // Back-substitute for y and apply x += M (V y).
            var y = new double[k];
            for (var i = k - 1; i >= 0; i--)
            {
                var acc = g[i];
                for (var j = i + 1; j < k; j++) acc -= h[i, j] * y[j];
                y[i] = h[i, i] == 0 ? 0 : acc / h[i, i];
            }
            Array.Clear(dx);
            for (var j = 0; j < k; j++) Axpy(y[j], v[j], dx);
            s.Precondition(dx, z);
            Axpy(1.0, z, x);
            a.MatVec(x, r);
            for (var i = 0; i < n; i++) r[i] = b[i] - r[i];
            callback?.Invoke(x);
        }
        return Nrm2(r) <= s.Tol ? s.Finish(0, total) : s.Finish(Math.Max(1, total), total);
    }

    /// <summary>Stable Givens rotation <c>(c, s, r)</c> with <c>[c s; −s c]ᵀ[a; b] = [r; 0]</c> (SciPy's <c>_sym_ortho</c>).</summary>
    internal static (double C, double S, double R) SymOrtho(double a, double b)
    {
        if (b == 0) return (a == 0 ? 1 : Math.Sign(a), 0, Math.Abs(a));
        if (a == 0) return (0, Math.Sign(b), Math.Abs(b));
        if (Math.Abs(b) > Math.Abs(a))
        {
            var tau = a / b;
            var s = Math.Sign(b) / Math.Sqrt(1 + tau * tau);
            var c = s * tau;
            return (c, s, b / s);
        }
        else
        {
            var tau = b / a;
            var c = Math.Sign(a) / Math.Sqrt(1 + tau * tau);
            var s = c * tau;
            return (c, s, a / c);
        }
    }

    // ------------------------------------------------------------------ LGMRES

    /// <summary>
    /// LGMRES (Baker, Jessup &amp; Manteuffel 2005): restarted GMRES whose Krylov space is augmented
    /// with the last <paramref name="outerK"/> error approximations, which damps the stagnation of
    /// plain restarted GMRES. Left-preconditioned, as SciPy's.
    /// </summary>
    /// <param name="a">The square operator.</param>
    /// <param name="b">Right-hand side.</param>
    /// <param name="x0">Starting guess.</param>
    /// <param name="rtol">Relative tolerance.</param>
    /// <param name="atol">Absolute tolerance.</param>
    /// <param name="maxiter">Maximum number of outer cycles (default 1000, as SciPy).</param>
    /// <param name="m">Preconditioner approximating <c>A⁻¹</c>.</param>
    /// <param name="callback">Called with the iterate after each cycle.</param>
    /// <param name="innerM">Inner Krylov dimension per cycle.</param>
    /// <param name="outerK">Number of augmentation vectors carried between cycles.</param>
    public static IterativeSolveResult Lgmres(ILinearOperator a, double[] b, double[]? x0 = null, double rtol = 1e-5, double atol = 0.0,
        int? maxiter = null, ILinearOperator? m = null, Action<double[]>? callback = null, int innerM = 30, int outerK = 3)
    {
        var s = Setup(a, b, x0, rtol, atol, maxiter ?? 1000, m, nameof(Lgmres));
        var n = b.Length;
        if (s.BNorm == 0) return ZeroRhs(n);
        var r = s.R; var x = s.X;
        var outer = new List<double[]>();
        var total = 0;
        var w = new double[n]; var tmp = new double[n];
        for (var cycle = 0; cycle < s.MaxIter; cycle++)
        {
            if (Nrm2(r) <= s.Tol) return s.Finish(0, total);
            // v0 = M r normalised; the inner problem minimises ‖M(r − A dx)‖.
            var v0 = new double[n];
            s.Precondition(r, v0);
            var inner0 = Nrm2(v0);
            if (inner0 == 0) break;
            ScaleInPlace(v0, 1 / inner0);
            var dim = Math.Min(innerM, n) + outer.Count;
            var vs = new List<double[]> { v0 };
            var zs = new List<double[]>();
            var h = new double[dim + 1, dim];
            var cs = new double[dim]; var sn = new double[dim]; var g = new double[dim + 1];
            g[0] = 1.0;
            var k = 0;
            for (; k < dim; k++)
            {
                total++;
                var innerSteps = dim - outer.Count;
                double[] z = k < innerSteps ? (double[])vs[^1].Clone() : (double[])outer[k - innerSteps].Clone();
                a.MatVec(z, tmp);
                s.Precondition(tmp, w);
                var wn = (double[])w.Clone();
                var w0 = Nrm2(wn);
                for (var i = 0; i <= k; i++)
                {
                    var c = Dot(wn, vs[i]);
                    h[i, k] = c;
                    Axpy(-c, vs[i], wn);
                }
                var hn = Nrm2(wn);
                if (hn < 0.7071 * w0)
                    for (var i = 0; i <= k; i++)
                    {
                        var c = Dot(wn, vs[i]);
                        h[i, k] += c;
                        Axpy(-c, vs[i], wn);
                    }
                hn = Nrm2(wn);
                h[k + 1, k] = hn;
                zs.Add(z);
                if (hn != 0) ScaleInPlace(wn, 1 / hn);
                vs.Add(wn);
                for (var i = 0; i < k; i++)
                {
                    var t = cs[i] * h[i, k] + sn[i] * h[i + 1, k];
                    h[i + 1, k] = -sn[i] * h[i, k] + cs[i] * h[i + 1, k];
                    h[i, k] = t;
                }
                var (c1, s1, rr) = SymOrtho(h[k, k], h[k + 1, k]);
                cs[k] = c1; sn[k] = s1; h[k, k] = rr; h[k + 1, k] = 0;
                g[k + 1] = -s1 * g[k];
                g[k] = c1 * g[k];
                if (Math.Abs(g[k + 1]) * inner0 <= s.Tol * 0.5 || hn == 0) { k++; break; }
            }
            var y = new double[k];
            for (var i = k - 1; i >= 0; i--)
            {
                var acc = g[i];
                for (var j = i + 1; j < k; j++) acc -= h[i, j] * y[j];
                y[i] = h[i, i] == 0 ? 0 : acc / h[i, i];
            }
            var dx = new double[n];
            for (var j = 0; j < k; j++) Axpy(y[j] * inner0, zs[j], dx);
            var nx = Nrm2(dx);
            if (nx > 0)
            {
                var aug = (double[])dx.Clone();
                ScaleInPlace(aug, 1 / nx);
                outer.Insert(0, aug);
                if (outer.Count > outerK) outer.RemoveAt(outer.Count - 1);
            }
            Axpy(1.0, dx, x);
            a.MatVec(x, r);
            for (var i = 0; i < n; i++) r[i] = b[i] - r[i];
            callback?.Invoke(x);
        }
        return Nrm2(r) <= s.Tol ? s.Finish(0, total) : s.Finish(Math.Max(1, total), total);
    }

    // ------------------------------------------------------------------ MINRES

    /// <summary>
    /// MINRES (Paige &amp; Saunders 1975) for symmetric, possibly indefinite <c>A − shift·I</c>; a
    /// port of SciPy's, including its stopping rule <c>‖r‖ ≤ rtol·‖A‖·‖x‖</c> (which is what
    /// <paramref name="rtol"/> means here) plus the usual <c>‖r‖ ≤ max(atol, rtol‖b‖)</c>.
    /// The preconditioner must be symmetric positive definite.
    /// </summary>
    /// <param name="a">Symmetric operator.</param>
    /// <param name="b">Right-hand side.</param>
    /// <param name="x0">Starting guess.</param>
    /// <param name="rtol">Relative tolerance.</param>
    /// <param name="atol">Absolute tolerance on the residual estimate.</param>
    /// <param name="shift">Solves <c>(A − shift·I) x = b</c>.</param>
    /// <param name="maxiter">Iteration limit (default <c>5 n</c>, as SciPy).</param>
    /// <param name="m">SPD preconditioner approximating <c>A⁻¹</c>.</param>
    /// <param name="callback">Called with the iterate after each step.</param>
    public static IterativeSolveResult Minres(ILinearOperator a, double[] b, double[]? x0 = null, double rtol = 1e-5, double atol = 0.0,
        double shift = 0.0, int? maxiter = null, ILinearOperator? m = null, Action<double[]>? callback = null)
    {
        var st = Setup(a, b, x0, rtol, atol, maxiter ?? 5 * b.Length, m, nameof(Minres));
        var n = b.Length;
        if (st.BNorm == 0) return ZeroRhs(n);
        var x = st.X;
        var r1 = (double[])st.R.Clone();
        var y = new double[n];
        st.Precondition(r1, y);
        var beta1 = Dot(r1, y);
        if (beta1 < 0) throw new ArgumentException("MINRES: the preconditioner is not positive definite.");
        if (beta1 == 0) return st.Finish(0, 0);
        beta1 = Math.Sqrt(beta1);

        double oldb = 0, beta = beta1, dbar = 0, epsln = 0, phibar = beta1, rhs1 = beta1, rhs2 = 0, tnorm2 = 0;
        double gmax = 0, gmin = double.MaxValue, cs = -1, sn = 0;
        var w = new double[n]; var w1 = new double[n]; var w2 = new double[n];
        var r2 = (double[])r1.Clone();
        var v = new double[n];
        var istop = 0;
        var itn = 0;
        while (itn < st.MaxIter)
        {
            itn++;
            var sc = 1.0 / beta;
            for (var i = 0; i < n; i++) v[i] = sc * y[i];
            a.MatVec(v, y);
            if (shift != 0) Axpy(-shift, v, y);
            if (itn >= 2) Axpy(-(beta / oldb), r1, y);
            var alfa = Dot(v, y);
            Axpy(-(alfa / beta), r2, y);
            (r1, r2) = (r2, r1);
            y.CopyTo(r2, 0);
            st.Precondition(r2, y);
            oldb = beta;
            beta = Dot(r2, y);
            if (beta < 0) throw new ArgumentException("MINRES: the matrix is not symmetric (or the preconditioner is indefinite).");
            beta = Math.Sqrt(beta);
            tnorm2 += alfa * alfa + oldb * oldb + beta * beta;
            if (itn == 1 && beta / beta1 <= 10 * Eps) istop = -1;

            var oldeps = epsln;
            var delta = cs * dbar + sn * alfa;
            var gbar = sn * dbar - cs * alfa;
            epsln = sn * beta;
            dbar = -cs * beta;
            var root = Hypot(gbar, dbar);
            var gamma = Math.Max(Hypot(gbar, beta), Eps);
            cs = gbar / gamma;
            sn = beta / gamma;
            var phi = cs * phibar;
            phibar = sn * phibar;

            var denom = 1.0 / gamma;
            (w1, w2, w) = (w2, w, w1);
            for (var i = 0; i < n; i++) w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) * denom;
            Axpy(phi, w, x);

            gmax = Math.Max(gmax, gamma);
            gmin = Math.Min(gmin, gamma);
            var zz = rhs1 / gamma;
            rhs1 = rhs2 - delta * zz;
            rhs2 = -epsln * zz;

            var anorm = Math.Sqrt(tnorm2);
            var ynorm = Nrm2(x);
            var epsx = anorm * ynorm * Eps;
            var rnorm = phibar;
            var test1 = ynorm == 0 || anorm == 0 ? double.PositiveInfinity : rnorm / (anorm * ynorm);
            var test2 = anorm == 0 ? double.PositiveInfinity : root / anorm;
            var acond = gmax / gmin;
            if (istop == 0)
            {
                if (1 + test2 <= 1) istop = 2;
                if (1 + test1 <= 1) istop = 1;
                if (itn >= st.MaxIter) istop = 6;
                if (acond >= 0.1 / Eps) istop = 4;
                if (epsx >= beta1) istop = 3;
                if (test2 <= rtol) istop = 2;
                if (test1 <= rtol) istop = 1;
                if (rnorm <= st.Tol) istop = 1;
            }
            callback?.Invoke(x);
            if (istop != 0) break;
        }
        return st.Finish(istop == 6 ? st.MaxIter : 0, itn);
    }

    private static double Hypot(double a, double b)
    {
        a = Math.Abs(a); b = Math.Abs(b);
        if (a < b) (a, b) = (b, a);
        if (a == 0) return 0;
        var t = b / a;
        return a * Math.Sqrt(1 + t * t);
    }

    // ------------------------------------------------------------------ QMR

    /// <summary>
    /// Quasi-minimal residual (Freund &amp; Nachtigal) with optional left (<paramref name="m1"/>) and
    /// right (<paramref name="m2"/>) preconditioners; needs <c>Aᵀ</c> products. A port of SciPy's.
    /// </summary>
    /// <param name="a">The square operator.</param>
    /// <param name="b">Right-hand side.</param>
    /// <param name="x0">Starting guess.</param>
    /// <param name="rtol">Relative tolerance.</param>
    /// <param name="atol">Absolute tolerance.</param>
    /// <param name="maxiter">Iteration limit (default <c>10 n</c>).</param>
    /// <param name="m1">Left preconditioner.</param>
    /// <param name="m2">Right preconditioner.</param>
    /// <param name="callback">Called with the iterate after each step.</param>
    public static IterativeSolveResult Qmr(ILinearOperator a, double[] b, double[]? x0 = null, double rtol = 1e-5, double atol = 0.0,
        int? maxiter = null, ILinearOperator? m1 = null, ILinearOperator? m2 = null, Action<double[]>? callback = null)
    {
        var s = Setup(a, b, x0, rtol, atol, maxiter, null, nameof(Qmr));
        var n = b.Length;
        if (s.BNorm == 0) return ZeroRhs(n);
        void Apply(ILinearOperator? op, bool transpose, ReadOnlySpan<double> src, Span<double> dst)
        {
            if (op is null) src.CopyTo(dst);
            else if (transpose) op.RMatVec(src, dst);
            else op.MatVec(src, dst);
        }
        const double tol = Eps;
        var r = s.R; var x = s.X;
        var vt = (double[])r.Clone();
        var y = new double[n]; Apply(m1, false, vt, y);
        var rho = Nrm2(y);
        var wt = (double[])r.Clone();
        var z = new double[n]; Apply(m2, true, wt, z);
        var xi = Nrm2(z);
        double gamma = 1, eta = -1, theta = 0, epsilon = 0;
        var v = new double[n]; var w = new double[n];
        var p = new double[n]; var q = new double[n]; var d = new double[n]; var sv = new double[n];
        var yt = new double[n]; var zt = new double[n]; var pt = new double[n]; var tmp = new double[n];
        for (var it = 0; it < s.MaxIter; it++)
        {
            if (Nrm2(r) <= s.Tol) return s.Finish(0, it);
            if (Math.Abs(rho) < tol) return s.Finish(-10, it);
            if (Math.Abs(xi) < tol) return s.Finish(-15, it);
            for (var i = 0; i < n; i++) { v[i] = vt[i] / rho; y[i] /= rho; w[i] = wt[i] / xi; z[i] /= xi; }
            var delta = Dot(z, y);
            if (Math.Abs(delta) < tol) return s.Finish(-13, it);
            Apply(m2, false, y, yt);
            Apply(m1, true, z, zt);
            if (it > 0)
            {
                var f1 = xi * delta / epsilon;
                var f2 = rho * delta / epsilon;
                for (var i = 0; i < n; i++) { p[i] = yt[i] - f1 * p[i]; q[i] = zt[i] - f2 * q[i]; }
            }
            else { yt.CopyTo(p, 0); zt.CopyTo(q, 0); }
            a.MatVec(p, pt);
            epsilon = Dot(q, pt);
            if (Math.Abs(epsilon) < tol) return s.Finish(-14, it);
            var beta = epsilon / delta;
            if (Math.Abs(beta) < tol) return s.Finish(-11, it);
            for (var i = 0; i < n; i++) vt[i] = pt[i] - beta * v[i];
            Apply(m1, false, vt, y);
            var rhoPrev = rho;
            rho = Nrm2(y);
            a.RMatVec(q, tmp);
            for (var i = 0; i < n; i++) wt[i] = tmp[i] - beta * w[i];
            Apply(m2, true, wt, z);
            xi = Nrm2(z);
            var gammaPrev = gamma;
            var thetaPrev = theta;
            theta = rho / (gammaPrev * Math.Abs(beta));
            gamma = 1 / Math.Sqrt(1 + theta * theta);
            if (Math.Abs(gamma) < tol) return s.Finish(-12, it);
            eta *= -(rhoPrev / beta) * (gamma / gammaPrev) * (gamma / gammaPrev);
            if (it > 0)
            {
                var f = (thetaPrev * gamma) * (thetaPrev * gamma);
                for (var i = 0; i < n; i++) { d[i] = f * d[i] + eta * p[i]; sv[i] = f * sv[i] + eta * pt[i]; }
            }
            else
                for (var i = 0; i < n; i++) { d[i] = eta * p[i]; sv[i] = eta * pt[i]; }
            Axpy(1.0, d, x);
            Axpy(-1.0, sv, r);
            callback?.Invoke(x);
        }
        return Nrm2(r) <= s.Tol ? s.Finish(0, s.MaxIter) : s.Finish(s.MaxIter, s.MaxIter);
    }

    // ------------------------------------------------------------------ TFQMR

    /// <summary>
    /// Transpose-free QMR (Freund 1993) for general square <c>A</c>, right-preconditioned. Stops on
    /// SciPy's residual bound <c>τ·√(k+1) ≤ tol</c>; <see cref="IterativeSolveResult.ResidualNorm"/> is the true residual.
    /// </summary>
    /// <inheritdoc cref="Cg" path="/param"/>
    public static IterativeSolveResult Tfqmr(ILinearOperator a, double[] b, double[]? x0 = null, double rtol = 1e-5, double atol = 0.0,
        int? maxiter = null, ILinearOperator? m = null, Action<double[]>? callback = null)
    {
        var s = Setup(a, b, x0, rtol, atol, maxiter, m, nameof(Tfqmr));
        var n = b.Length;
        if (s.BNorm == 0) return ZeroRhs(n);
        var x = s.X;
        var r = (double[])s.R.Clone();
        var tmp = new double[n];
        void Op(ReadOnlySpan<double> src, Span<double> dst)
        {
            s.Precondition(src, tmp);
            a.MatVec(tmp, dst);
        }
        var u = (double[])r.Clone();
        var w = (double[])r.Clone();
        var rstar = r;
        var v = new double[n]; Op(r, v);
        var uhat = (double[])v.Clone();
        var d = new double[n]; var z = new double[n]; var uNext = new double[n];
        double theta = 0, eta = 0, alpha = 0;
        var rho = Dot(rstar, r);
        var rhoLast = rho;
        var tau = Math.Sqrt(rho);
        if (tau == 0) return s.Finish(0, 0);
        for (var it = 0; it < s.MaxIter; it++)
        {
            var even = it % 2 == 0;
            if (even)
            {
                var vtr = Dot(rstar, v);
                if (vtr == 0) return s.Finish(-1, it);
                alpha = rho / vtr;
                for (var i = 0; i < n; i++) uNext[i] = u[i] - alpha * v[i];
            }
            Axpy(-alpha, uhat, w);
            var coef = theta * theta / alpha * eta;
            for (var i = 0; i < n; i++) d[i] = u[i] + coef * d[i];
            theta = Nrm2(w) / tau;
            var c = Math.Sqrt(1.0 / (1 + theta * theta));
            tau *= theta * c;
            eta = c * c * alpha;
            s.Precondition(d, z);
            Axpy(eta, z, x);
            callback?.Invoke(x);
            if (tau * Math.Sqrt(it + 1) < s.Tol) return s.Finish(0, it + 1);
            if (!even)
            {
                rho = Dot(rstar, w);
                var beta = rho / rhoLast;
                for (var i = 0; i < n; i++) u[i] = w[i] + beta * u[i];
                for (var i = 0; i < n; i++) v[i] = beta * uhat[i] + beta * beta * v[i];
                Op(u, uhat);
                Axpy(1.0, uhat, v);
            }
            else
            {
                Op(uNext, uhat);
                uNext.CopyTo(u, 0);
                rhoLast = rho;
            }
        }
        var res = s.Finish(s.MaxIter, s.MaxIter);
        return res.ResidualNorm <= s.Tol ? res with { Info = 0 } : res;
    }

    // ------------------------------------------------------------------ LSQR / LSMR

    /// <summary>
    /// LSQR (Paige &amp; Saunders 1982): least squares <c>min ‖A x − b‖² + damp²‖x‖²</c> for any
    /// m×n operator, using only <c>A v</c> and <c>Aᵀ u</c>. A port of SciPy's, including its
    /// stopping rules and return values.
    /// </summary>
    /// <param name="a">The m×n operator.</param>
    /// <param name="b">Right-hand side of length m.</param>
    /// <param name="damp">Tikhonov damping.</param>
    /// <param name="atol">Relative tolerance on <c>‖Aᵀr‖</c>.</param>
    /// <param name="btol">Relative tolerance on <c>‖r‖</c>.</param>
    /// <param name="conlim">Stop when the condition estimate exceeds this.</param>
    /// <param name="iterLim">Iteration limit (default <c>2 n</c>).</param>
    /// <param name="calcVar">Accumulate the variance estimate.</param>
    /// <param name="x0">Starting guess.</param>
    public static LsqrResult Lsqr(ILinearOperator a, double[] b, double damp = 0.0, double atol = 1e-6, double btol = 1e-6,
        double conlim = 1e8, int? iterLim = null, bool calcVar = false, double[]? x0 = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        int m = a.Rows, n = a.Columns;
        if (b.Length != m) throw new ArgumentException($"b has {b.Length} entries, expected {m}.");
        var lim = iterLim ?? 2 * n;
        var var = new double[n];
        int itn = 0, istop = 0;
        var ctol = conlim > 0 ? 1 / conlim : 0;
        double anorm = 0, acond = 0, dampsq = damp * damp, ddnorm = 0, res2 = 0, xnorm = 0, xxnorm = 0, z = 0, cs2 = -1, sn2 = 0;
        var u = (double[])b.Clone();
        var bnorm = Nrm2(b);
        double[] x;
        double beta;
        if (x0 is null) { x = new double[n]; beta = bnorm; }
        else
        {
            x = (double[])x0.Clone();
            var ax = a.Dot(x);
            for (var i = 0; i < m; i++) u[i] -= ax[i];
            beta = Nrm2(u);
        }
        var v = new double[n];
        double alfa = 0;
        if (beta > 0)
        {
            ScaleInPlace(u, 1 / beta);
            a.RMatVec(u, v);
            alfa = Nrm2(v);
        }
        else x.CopyTo(v, 0);
        if (alfa > 0) ScaleInPlace(v, 1 / alfa);
        var w = (double[])v.Clone();
        double rhobar = alfa, phibar = beta, rnorm = beta, r1norm = rnorm, r2norm = rnorm;
        var arnorm = alfa * beta;
        if (arnorm == 0) return new LsqrResult(x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var);
        var tu = new double[m]; var tv = new double[n];
        while (itn < lim)
        {
            itn++;
            a.MatVec(v, tu);
            for (var i = 0; i < m; i++) u[i] = tu[i] - alfa * u[i];
            beta = Nrm2(u);
            if (beta > 0)
            {
                ScaleInPlace(u, 1 / beta);
                anorm = Math.Sqrt(anorm * anorm + alfa * alfa + beta * beta + dampsq);
                a.RMatVec(u, tv);
                for (var i = 0; i < n; i++) v[i] = tv[i] - beta * v[i];
                alfa = Nrm2(v);
                if (alfa > 0) ScaleInPlace(v, 1 / alfa);
            }
            double rhobar1, psi;
            if (damp > 0)
            {
                rhobar1 = Math.Sqrt(rhobar * rhobar + dampsq);
                var cs1 = rhobar / rhobar1;
                var sn1 = damp / rhobar1;
                psi = sn1 * phibar;
                phibar = cs1 * phibar;
            }
            else { rhobar1 = rhobar; psi = 0; }
            var (cs, sn, rho) = SymOrtho(rhobar1, beta);
            var theta = sn * alfa;
            rhobar = -cs * alfa;
            var phi = cs * phibar;
            phibar = sn * phibar;
            var tau = sn * phi;
            var t1 = phi / rho;
            var t2 = -theta / rho;
            var dkn2 = 0.0;
            for (var i = 0; i < n; i++)
            {
                var dk = w[i] / rho;
                dkn2 += dk * dk;
                if (calcVar) var[i] += dk * dk;
                x[i] += t1 * w[i];
                w[i] = v[i] + t2 * w[i];
            }
            ddnorm += dkn2;
            var delta = sn2 * rho;
            var gambar = -cs2 * rho;
            var rhs = phi - delta * z;
            var zbar = rhs / gambar;
            xnorm = Math.Sqrt(xxnorm + zbar * zbar);
            var gamma = Math.Sqrt(gambar * gambar + theta * theta);
            cs2 = gambar / gamma;
            sn2 = theta / gamma;
            z = rhs / gamma;
            xxnorm += z * z;
            acond = anorm * Math.Sqrt(ddnorm);
            var res1 = phibar * phibar;
            res2 += psi * psi;
            rnorm = Math.Sqrt(res1 + res2);
            arnorm = alfa * Math.Abs(tau);
            if (damp > 0)
            {
                var r1sq = rnorm * rnorm - dampsq * xxnorm;
                r1norm = Math.Sqrt(Math.Abs(r1sq));
                if (r1sq < 0) r1norm = -r1norm;
            }
            else r1norm = rnorm;
            r2norm = rnorm;
            var test1 = rnorm / bnorm;
            var test2 = arnorm / (anorm * rnorm + Eps);
            var test3 = 1 / (acond + Eps);
            var tt1 = test1 / (1 + anorm * xnorm / bnorm);
            var rtol = btol + atol * anorm * xnorm / bnorm;
            if (itn >= lim) istop = 7;
            if (1 + test3 <= 1) istop = 6;
            if (1 + test2 <= 1) istop = 5;
            if (1 + tt1 <= 1) istop = 4;
            if (test3 <= ctol) istop = 3;
            if (test2 <= atol) istop = 2;
            if (test1 <= rtol) istop = 1;
            if (istop != 0) break;
        }
        return new LsqrResult(x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var);
    }

    /// <summary>
    /// LSMR (Fong &amp; Saunders 2011): least squares like <see cref="Lsqr"/>, but minimising
    /// <c>‖Aᵀr‖</c> monotonically, so it is safer to stop early. A port of SciPy's.
    /// </summary>
    /// <param name="a">The m×n operator.</param>
    /// <param name="b">Right-hand side of length m.</param>
    /// <param name="damp">Tikhonov damping.</param>
    /// <param name="atol">Relative tolerance on <c>‖Aᵀr‖</c>.</param>
    /// <param name="btol">Relative tolerance on <c>‖r‖</c>.</param>
    /// <param name="conlim">Stop when the condition estimate exceeds this.</param>
    /// <param name="maxiter">Iteration limit (default <c>min(m, n)</c>).</param>
    /// <param name="x0">Starting guess.</param>
    public static LsmrResult Lsmr(ILinearOperator a, double[] b, double damp = 0.0, double atol = 1e-6, double btol = 1e-6,
        double conlim = 1e8, int? maxiter = null, double[]? x0 = null)
    {
        ArgumentNullException.ThrowIfNull(a);
        int m = a.Rows, n = a.Columns;
        if (b.Length != m) throw new ArgumentException($"b has {b.Length} entries, expected {m}.");
        var lim = maxiter ?? Math.Min(m, n);
        double[] x;
        var u = (double[])b.Clone();
        if (x0 is null) x = new double[n];
        else
        {
            x = (double[])x0.Clone();
            var ax = a.Dot(x);
            for (var i = 0; i < m; i++) u[i] -= ax[i];
        }
        var normb = Nrm2(b);
        var beta = Nrm2(u);
        var v = new double[n];
        double alpha = 0;
        if (beta > 0)
        {
            ScaleInPlace(u, 1 / beta);
            a.RMatVec(u, v);
            alpha = Nrm2(v);
        }
        if (alpha > 0) ScaleInPlace(v, 1 / alpha);
        var itn = 0;
        double zetabar = alpha * beta, alphabar = alpha, rho = 1, rhobar = 1, cbar = 1, sbar = 0;
        var h = (double[])v.Clone();
        var hbar = new double[n];
        double betadd = beta, betad = 0, rhodold = 1, tautildeold = 0, thetatilde = 0, zeta = 0, d = 0;
        double normA2 = alpha * alpha, maxrbar = 0, minrbar = 1e100, normA = Math.Sqrt(normA2), condA = 1, normx = 0;
        var istop = 0;
        var ctol = conlim > 0 ? 1 / conlim : 0;
        var normr = beta;
        var normar = alpha * beta;
        if (normar == 0) return new LsmrResult(x, istop, itn, normr, normar, normA, condA, normx);
        if (normb == 0) { Array.Clear(x); return new LsmrResult(x, istop, itn, normr, normar, normA, condA, normx); }
        var tu = new double[m]; var tv = new double[n];
        while (itn < lim)
        {
            itn++;
            a.MatVec(v, tu);
            for (var i = 0; i < m; i++) u[i] = tu[i] - alpha * u[i];
            beta = Nrm2(u);
            if (beta > 0)
            {
                ScaleInPlace(u, 1 / beta);
                a.RMatVec(u, tv);
                for (var i = 0; i < n; i++) v[i] = tv[i] - beta * v[i];
                alpha = Nrm2(v);
                if (alpha > 0) ScaleInPlace(v, 1 / alpha);
            }
            var (chat, shat, alphahat) = SymOrtho(alphabar, damp);
            var rhoold = rho;
            var (c, s, rhoNew) = SymOrtho(alphahat, beta);
            rho = rhoNew;
            var thetanew = s * alpha;
            alphabar = c * alpha;
            var rhobarold = rhobar;
            var zetaold = zeta;
            var thetabar = sbar * rho;
            var rhotemp = cbar * rho;
            (cbar, sbar, rhobar) = SymOrtho(cbar * rho, thetanew);
            zeta = cbar * zetabar;
            zetabar = -sbar * zetabar;
            var fh = -(thetabar * rho / (rhoold * rhobarold));
            var fx = zeta / (rho * rhobar);
            var fhh = -(thetanew / rho);
            for (var i = 0; i < n; i++)
            {
                hbar[i] = fh * hbar[i] + h[i];
                x[i] += fx * hbar[i];
                h[i] = fhh * h[i] + v[i];
            }
            var betaacute = chat * betadd;
            var betacheck = -shat * betadd;
            var betahat = c * betaacute;
            betadd = -s * betaacute;
            var thetatildeold = thetatilde;
            var (ctildeold, stildeold, rhotildeold) = SymOrtho(rhodold, thetabar);
            thetatilde = stildeold * rhobar;
            rhodold = ctildeold * rhobar;
            betad = -stildeold * betad + ctildeold * betahat;
            tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
            var taud = (zeta - thetatilde * tautildeold) / rhodold;
            d += betacheck * betacheck;
            normr = Math.Sqrt(d + (betad - taud) * (betad - taud) + betadd * betadd);
            normA2 += beta * beta;
            normA = Math.Sqrt(normA2);
            normA2 += alpha * alpha;
            maxrbar = Math.Max(maxrbar, rhobarold);
            if (itn > 1) minrbar = Math.Min(minrbar, rhobarold);
            condA = Math.Max(maxrbar, rhotemp) / Math.Min(minrbar, rhotemp);
            normar = Math.Abs(zetabar);
            normx = Nrm2(x);
            var test1 = normr / normb;
            var test2 = normA * normr != 0 ? normar / (normA * normr) : double.PositiveInfinity;
            var test3 = 1 / condA;
            var t1 = test1 / (1 + normA * normx / normb);
            var rtol = btol + atol * normA * normx / normb;
            if (itn >= lim) istop = 7;
            if (1 + test3 <= 1) istop = 6;
            if (1 + test2 <= 1) istop = 5;
            if (1 + t1 <= 1) istop = 4;
            if (test3 <= ctol) istop = 3;
            if (test2 <= atol) istop = 2;
            if (test1 <= rtol) istop = 1;
            if (istop > 0) break;
        }
        return new LsmrResult(x, istop, itn, normr, normar, normA, condA, normx);
    }
}
