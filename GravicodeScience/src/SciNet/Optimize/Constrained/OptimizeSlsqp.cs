namespace Gravicode.Science.SciNet;

/// <summary>Options for <see cref="MinimizeMethod.Slsqp"/>.</summary>
public sealed record SlsqpOptions : MinimizeOptions
{
    /// <summary>Precision goal (<c>acc</c>) for the objective and the constraint violation. Default 1e-6.</summary>
    public double? Ftol { get; init; }
    /// <summary>Absolute forward-difference step for gradients and constraint Jacobians. Default √ε.</summary>
    public double Eps { get; init; } = 1.4901161193847656e-08;
}

/// <summary>Options for <see cref="MinimizeMethod.Cobyla"/>.</summary>
public sealed record CobylaOptions : MinimizeOptions
{
    /// <summary>Initial trust-region radius. Default 1.</summary>
    public double Rhobeg { get; init; } = 1.0;
    /// <summary>Final trust-region radius (SciPy's <c>tol</c>). Default 1e-4.</summary>
    public double? Tol { get; init; }
    /// <summary>Constraint-violation tolerance for declaring success. Default √ε.</summary>
    public double? Catol { get; init; }
    /// <summary>Stop as soon as a feasible point reaches this value.</summary>
    public double FTarget { get; init; } = double.NegativeInfinity;
}

public static partial class Optimize
{
    static partial void RunSlsqp(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeSlsqp.Run(problem);
    static partial void RunCobyla(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeCobyla.Run(problem);
}

/// <summary>
/// Sequential least squares programming after Kraft (1988): a BFGS approximation of the Lagrangian
/// Hessian with Powell's damping, a quadratic subproblem with the linearised equalities,
/// inequalities and bounds, Kraft's relaxation of an inconsistent linearisation (an extra variable
/// δ scaling the violated constraints), and an L1 exact-penalty line search with Kraft's penalty
/// update. The quadratic subproblem is solved by Goldfarb–Idnani rather than Kraft's LSQ chain, so
/// iteration counts can differ from SciPy's while the KKT points agree.
/// </summary>
internal static class OptimizeSlsqp
{
    private static readonly Dictionary<int, string> Messages = new()
    {
        [0] = "Optimization terminated successfully",
        [2] = "More equality constraints than independent variables",
        [4] = "Inequality constraints incompatible",
        [5] = "Singular matrix E in LSQ subproblem",
        [8] = "Positive directional derivative for linesearch",
        [9] = "Iteration limit reached",
    };

    public static OptimizeResult Run(MinimizeProblem p)
    {
        var o = p.OptionsAs<SlsqpOptions>("SLSQP");
        var acc = o.Ftol ?? p.Tol ?? 1e-6;
        var maxiter = o.MaxIter ?? 100;
        var n = p.N;
        var sf = p.CreateObjective(o.Eps);
        var lb = p.Bounds?.Lb; var ub = p.Bounds?.Ub;
        var x = (double[])p.X0.Clone();
        var cs = new OptimizeConstraintSet(p.Constraints, x, o.Eps);
        int meq = cs.MEq, m = cs.M;
        if (meq > n) return Done(2, x, double.NaN, new double[n], new double[m], 0);

        var g = new double[n];
        var f = sf.FunAndGrad(x, g);
        var (ce, ci) = cs.Eval(x);
        var c = (double[])[.. ce, .. ci];
        var a = cs.Jacobian(x, lb, ub);
        var bmat = OptimizeDense.Identity(n);
        var mu = new double[m];
        var lam = new double[m];
        var nit = 0;
        var status = -1;
        while (true)
        {
            if (nit >= maxiter) { status = 9; break; }
            nit++;
            // ---- QP subproblem: min ½dᵀBd + gᵀd, A_eq d + c_eq = 0, A_in d + c_in ≥ 0, bounds on d ----
            var (d, qpLam, delta, qpStatus) = SolveQp(bmat, g, a, c, n, meq, m, x, lb, ub);
            if (qpStatus != 0) { status = qpStatus; break; }
            Array.Copy(qpLam, lam, m);
            var h4 = 1.0 - delta;

            // Kraft's convergence check and penalty update.
            var h1 = Math.Abs(OptimizeDense.Dot(g, d));
            double h2 = 0;
            for (var j = 0; j < m; j++)
            {
                var h3v = j < meq ? c[j] : 0;
                h2 += Math.Max(-c[j], h3v);
                var ar = Math.Abs(lam[j]);
                mu[j] = Math.Max(ar, (mu[j] + ar) / 2);
                h1 += ar * Math.Abs(c[j]);
            }
            if (h1 < acc && h2 < acc) { status = 0; break; }
            double pen = 0;
            for (var j = 0; j < m; j++) pen += mu[j] * Math.Max(-c[j], j < meq ? c[j] : 0);
            var t0 = f + pen;
            var h3 = OptimizeDense.Dot(g, d) - pen * h4;
            if (h3 >= 0) { status = 8; break; }

            // ---- L1 merit line search ----
            var x0 = (double[])x.Clone();
            var g0 = (double[])g.Clone();
            var a0 = (double[])a.Clone();
            var f0 = f;
            var alpha = 1.0;
            var s = (double[])d.Clone();
            for (var line = 1; ; line++)
            {
                for (var i = 0; i < n; i++) x[i] = x0[i] + alpha * d[i];
                if (lb is not null) for (var i = 0; i < n; i++) x[i] = Math.Clamp(x[i], lb[i], ub![i]);
                f = sf.Fun(x);
                (ce, ci) = cs.Eval(x);
                c = [.. ce, .. ci];
                double hp = 0;
                for (var j = 0; j < m; j++) hp += mu[j] * Math.Max(-c[j], j < meq ? c[j] : 0);
                var merit = f + hp;
                if (!double.IsFinite(merit)) { alpha *= 0.1; if (line > 10) break; continue; }
                if (merit - t0 <= 0.1 * alpha * h3 || line > 10) break;
                var ha = alpha * h3;
                alpha *= Math.Min(Math.Max(ha / (2 * (ha - (merit - t0))), 0.1), 1.0);
                alpha = Math.Max(alpha, 1e-10);
            }
            for (var i = 0; i < n; i++) s[i] = x[i] - x0[i];
            if (p.NotifyCallback(x, f)) { status = 99; sf.Grad(x, g); break; }

            // Convergence on the accepted step.
            double viol = 0;
            for (var j = 0; j < m; j++) viol += Math.Max(-c[j], j < meq ? Math.Abs(c[j]) : 0);
            sf.Grad(x, g);
            a = cs.Jacobian(x, lb, ub);
            if ((Math.Abs(f - f0) < acc || OptimizeDense.Norm2(s) < acc) && viol < acc) { status = 0; break; }

            // ---- damped BFGS update of the Lagrangian Hessian ----
            var y = new double[n];
            for (var i = 0; i < n; i++)
            {
                double gl1 = g[i], gl0 = g0[i];
                for (var j = 0; j < m; j++) { gl1 -= lam[j] * a[j * n + i]; gl0 -= lam[j] * a0[j * n + i]; }
                y[i] = gl1 - gl0;
            }
            var bs = new double[n];
            OptimizeDense.MatVec(bmat, n, n, s, bs);
            var sbs = OptimizeDense.Dot(s, bs);
            var sy = OptimizeDense.Dot(s, y);
            if (sbs > 0)
            {
                if (sy < 0.2 * sbs)
                {
                    var theta = 0.8 * sbs / (sbs - sy);
                    for (var i = 0; i < n; i++) y[i] = theta * y[i] + (1 - theta) * bs[i];
                    sy = OptimizeDense.Dot(s, y);
                }
                if (sy > 0)
                    for (var i = 0; i < n; i++)
                        for (var j = 0; j < n; j++) bmat[i * n + j] += y[i] * y[j] / sy - bs[i] * bs[j] / sbs;
            }
        }
        var (fe, fi) = cs.Eval(x);
        var viol2 = OptimizeConstraintSet.Violation(fe, fi);
        return Done(status, x, f, (double[])g.Clone(), lam, nit, viol2, sf.Nfev, sf.Ngev);

        OptimizeResult Done(int st, double[] xr, double fr, double[] gr, double[] l, int it, double cv = 0, int nfev = 0, int njev = 0) => new()
        {
            X = xr, Fun = fr, Jac = gr, Nit = it, Nfev = nfev, Njev = njev, Status = st, Success = st == 0,
            Message = Messages.TryGetValue(st, out var msg) ? msg : "Callback requested stop", Multipliers = l, ConstrViolation = cv,
        };
    }

    /// <summary>The QP, with Kraft's relaxation when the linearisation is inconsistent.</summary>
    private static (double[] D, double[] Lambda, double Delta, int Status) SolveQp(double[] bmat, double[] g, double[] a, double[] c, int n, int meq, int m,
        double[] x, double[]? lb, double[]? ub)
    {
        var cons = new List<double[]>(); var rhs = new List<double>();
        for (var j = 0; j < m; j++) { cons.Add(a.AsSpan(j * n, n).ToArray()); rhs.Add(-c[j]); }
        AddBounds(cons, rhs, n, x, lb, ub, 0);
        var st = OptimizeQuadprog.Solve(bmat, n, g, cons.ToArray(), rhs.ToArray(), meq, out var d, out var lam);
        if (st == 0) return (d, lam.Take(m).ToArray(), 0, 0);

        // Relaxed problem in (d, δ): equalities and violated inequalities are scaled by (1 − δ).
        var n1 = n + 1;
        var g1 = new double[n1 * n1];
        for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) g1[i * n1 + j] = bmat[i * n + j];
        var rho = 100.0;
        for (var tries = 0; tries < 3; tries++, rho *= 100)
        {
            g1[n * n1 + n] = rho;
            var a1 = new double[n1];
            Array.Copy(g, a1, n);
            var cons1 = new List<double[]>(); var rhs1 = new List<double>();
            for (var j = 0; j < m; j++)
            {
                var row = new double[n1];
                Array.Copy(a, j * n, row, 0, n);
                if (j < meq || c[j] < 0) { row[n] = -c[j]; }
                cons1.Add(row); rhs1.Add(-c[j]);
            }
            AddBounds(cons1, rhs1, n1, x, lb, ub, 1);
            var lo = new double[n1]; lo[n] = 1; cons1.Add(lo); rhs1.Add(0);
            var hi = new double[n1]; hi[n] = -1; cons1.Add(hi); rhs1.Add(-1);
            var st1 = OptimizeQuadprog.Solve(g1, n1, a1, cons1.ToArray(), rhs1.ToArray(), meq, out var d1, out var lam1);
            if (st1 != 0) continue;
            var delta = d1[n];
            if (delta >= 1 - 1e-12) return (new double[n], new double[m], 1, 4);
            return (d1.Take(n).ToArray(), lam1.Take(m).ToArray(), delta, 0);
        }
        return (new double[n], new double[m], 1, st == 2 ? 5 : 4);
    }

    private static void AddBounds(List<double[]> cons, List<double> rhs, int dim, double[] x, double[]? lb, double[]? ub, int extra)
    {
        if (lb is null) return;
        var n = x.Length;
        for (var i = 0; i < n; i++)
        {
            if (double.IsFinite(lb[i])) { var r = new double[dim]; r[i] = 1; cons.Add(r); rhs.Add(lb[i] - x[i]); }
            if (double.IsFinite(ub![i])) { var r = new double[dim]; r[i] = -1; cons.Add(r); rhs.Add(x[i] - ub[i]); }
        }
        _ = extra;
    }
}

/// <summary>
/// Constrained optimisation by linear approximations in the manner of Powell's COBYLA (1994):
/// linear interpolation models of f and every constraint on a simplex of n + 1 points, a trust
/// region whose radius ρ shrinks from rhobeg to rhoend, an ℓ∞ merit function with an adaptive
/// penalty, and geometry steps that keep the simplex well conditioned. The trust-region subproblem
/// is an LP over the box ‖d‖∞ ≤ ρ (Powell uses the 2-norm), and bounds are honoured exactly, so
/// iterates and counts differ from SciPy's PRIMA translation; the constrained optima agree.
/// </summary>
internal static class OptimizeCobyla
{
    public static OptimizeResult Run(MinimizeProblem p)
    {
        var o = p.OptionsAs<CobylaOptions>("COBYLA");
        var n = p.N;
        var rhoend = o.Tol ?? p.Tol ?? 1e-4;
        var rho = Math.Max(o.Rhobeg, rhoend);
        var maxfun = o.MaxIter ?? 1000;
        var catol = o.Catol ?? Math.Sqrt(2.220446049250313e-16);
        var lb = p.Bounds?.Lb ?? Enumerable.Repeat(double.NegativeInfinity, n).ToArray();
        var ub = p.Bounds?.Ub ?? Enumerable.Repeat(double.PositiveInfinity, n).ToArray();
        var cs = new OptimizeConstraintSet(p.Constraints, p.X0, 1e-8);
        int meq = cs.MEq, m = cs.M;
        var nfev = 0;
        var stop = false;

        // Equalities become two inequalities for the merit and the models.
        double[] Cons(double[] x)
        {
            var (e, i) = cs.Eval(x);
            var r = new double[2 * meq + i.Length];
            for (var k = 0; k < meq; k++) { r[2 * k] = e[k]; r[2 * k + 1] = -e[k]; }
            Array.Copy(i, 0, r, 2 * meq, i.Length);
            return r;
        }
        var mc = 2 * meq + (m - meq);
        var pts = new double[n + 1][]; var fv = new double[n + 1]; var cv = new double[n + 1][];
        (double F, double[] C) Eval(double[] x)
        {
            nfev++;
            var f = p.Fun is not null ? p.Fun((double[])x.Clone()) : p.FunAndGrad!((double[])x.Clone(), new double[n]);
            if (double.IsNaN(f)) f = double.PositiveInfinity;
            return (f, Cons(x));
        }
        static double Viol(double[] c) { double v = 0; foreach (var t in c) v = Math.Max(v, -t); return v; }
        var sigma = 0.0;
        double Merit(int k) => fv[k] + sigma * Viol(cv[k]);

        var x0 = (double[])p.X0.Clone();
        pts[0] = x0;
        (fv[0], cv[0]) = Eval(x0);
        for (var k = 0; k < n; k++)
        {
            var xk = (double[])x0.Clone();
            xk[k] += rho;
            if (xk[k] > ub[k]) xk[k] = x0[k] - rho;
            if (xk[k] < lb[k]) xk[k] = Math.Abs(ub[k] - x0[k]) >= Math.Abs(x0[k] - lb[k]) ? Math.Min(ub[k], x0[k] + rho) : Math.Max(lb[k], x0[k] - rho);
            pts[k + 1] = xk;
            (fv[k + 1], cv[k + 1]) = Eval(xk);
        }
        int Best()
        {
            var b = 0;
            for (var k = 1; k <= n; k++)
            {
                double vk = Viol(cv[k]), vb = Viol(cv[b]);
                // Violations below catol count as feasible, so rounding-level infeasibility cannot lock in a worse vertex.
                double ek = vk <= catol ? 0 : vk, eb = vb <= catol ? 0 : vb;
                var better = sigma > 0 ? fv[k] + sigma * ek < fv[b] + sigma * eb || (fv[k] + sigma * ek == fv[b] + sigma * eb && vk < vb)
                                       : ek < eb || (ek == eb && fv[k] < fv[b]);
                if (better) b = k;
            }
            return b;
        }
        var info = 0;
        var nit = 0;
        var forced = 0;
        var itersAtRho = 0;
        while (true)
        {
            if (nfev >= maxfun) { info = 3; break; }
            var b = Best();
            if (p.NotifyCallback(pts[b], fv[b])) { stop = true; break; }
            if (fv[b] <= o.FTarget && Viol(cv[b]) <= catol) { info = 1; break; }
            // Linear models from the simplex: rows of D are x_k − x_b.
            var others = Enumerable.Range(0, n + 1).Where(k => k != b).ToArray();
            var dm = new double[n * n];
            for (var r = 0; r < n; r++) for (var j = 0; j < n; j++) dm[r * n + j] = pts[others[r]][j] - pts[b][j];
            var lu = (double[])dm.Clone();
            var piv = new int[n];
            var okLu = OptimizeDense.LuFactor(lu, n, piv);
            double[] Grad(Func<int, double> val)
            {
                var rhs = new double[n];
                for (var r = 0; r < n; r++) rhs[r] = val(others[r]) - val(b);
                // Solve D g = rhs.
                OptimizeDense.LuSolve(lu, n, piv, rhs);
                return rhs;
            }
            if (!okLu)
            {
                if (++forced > 2 * n + 2 || !Regeometry(b, others, true)) { forced = 0; if (!ShrinkRho()) break; }
                continue;
            }
            forced = 0;
            var gf = Grad(k => fv[k]);
            var gc = new double[mc][];
            for (var i = 0; i < mc; i++) { var ii = i; gc[i] = Grad(k => cv[k][ii]); }
            if (gf.Any(t => !double.IsFinite(t))) { if (!Regeometry(b, others, true)) { if (!ShrinkRho()) break; } continue; }

            // Trust-region LP: min gfᵀd s.t. c_b + gcᵀd ≥ 0 (relaxed if needed), box ∩ bounds.
            var lo = new double[n]; var hi = new double[n];
            for (var j = 0; j < n; j++) { lo[j] = Math.Max(-rho, lb[j] - pts[b][j]); hi[j] = Math.Min(rho, ub[j] - pts[b][j]); }
            var d = SolveTrLp(gf, gc, cv[b], lo, hi, n, mc);
            nit++;
            // The box LP always steps to a corner, so near a curved active set the iterates can zigzag
            // with "acceptable" ratios indefinitely; a bounded number of steps per radius guarantees
            // termination (at most 5(n+1)·log2(rhobeg/rhoend) trust-region steps overall).
            if (++itersAtRho > 5 * (n + 1)) { if (!ShrinkRho()) break; continue; }
            var dn = OptimizeDense.NormInf(d);
            if (dn < 0.5 * rho)
            {
                // Small step: improve the simplex if it is poor, otherwise reduce rho.
                if (Regeometry(b, others, false)) continue;
                if (!ShrinkRho()) break;
                continue;
            }
            // Penalty: make the predicted merit reduction positive (Powell's parmu rule).
            var predF = -OptimizeDense.Dot(gf, d);
            var vB = Viol(cv[b]);
            double vPred = 0;
            for (var i = 0; i < mc; i++) { var ci = cv[b][i] + OptimizeDense.Dot(gc[i], d); vPred = Math.Max(vPred, -ci); }
            if (vPred < vB)
            {
                var need = 2 * -predF / (vB - vPred);
                if (sigma < need) sigma = Math.Max(need, 2 * sigma);
            }
            var pred = predF + sigma * (vB - vPred);
            var xn = pts[b].Select((v, j) => Math.Clamp(v + d[j], lb[j], ub[j])).ToArray();
            var (fn, cn) = Eval(xn);
            var actual = Merit(b) - (fn + sigma * Viol(cn));
            var ratio = pred > 0 ? actual / pred : -1;
            // Replace the vertex whose removal keeps the simplex best conditioned: the one with the largest
            // barycentric weight of the new point (weights from D⁻ᵀ d), preferring far vertices.
            // Solve Dᵀ w = d via the LU of D: (LU)ᵀ w = d.
            var dt = OptimizeDense.Transpose(dm, n, n);
            var wsol = OptimizeDense.Solve(dt, n, d) ?? new double[n];
            var replace = -1; double bestScore = -1;
            for (var r = 0; r < n; r++)
            {
                var dist = 0.0;
                for (var j = 0; j < n; j++) dist = Math.Max(dist, Math.Abs(pts[others[r]][j] - (actual > 0 ? xn[j] : pts[b][j])));
                var score = Math.Abs(wsol[r]) * Math.Max(1, dist / rho);
                if (score > bestScore) { bestScore = score; replace = others[r]; }
            }
            if (actual > 0 || bestScore > 1e-2)
            {
                if (replace >= 0) { pts[replace] = xn; fv[replace] = fn; cv[replace] = cn; }
            }
            if (ratio < 0.1)
            {
                if (Regeometry(b, others, false)) continue;
                if (!ShrinkRho()) break;
            }
        }

        bool ShrinkRho()
        {
            if (rho <= rhoend) { info = 0; return false; }
            rho = 0.5 * rho;
            if (rho <= 1.5 * rhoend) rho = rhoend;
            itersAtRho = 0;
            return true;
        }

        // Replaces a vertex that is too far from the best point, or (force) the one that degrades the
        // simplex most, by a point at distance rho along the direction normal to the opposite face.
        bool Regeometry(int b, int[] others, bool force)
        {
            var far = -1; double farD = 0;
            foreach (var k in others)
            {
                double dk = 0;
                for (var j = 0; j < n; j++) dk = Math.Max(dk, Math.Abs(pts[k][j] - pts[b][j]));
                if (dk > farD) { farD = dk; far = k; }
            }
            if (!force && farD <= 2.1 * rho) return false;
            if (nfev >= maxfun) return false;
            var r = Array.IndexOf(others, far);
            // Direction: row r of D⁻ᵀ (normal to the face spanned by the other vertices).
            var dm = new double[n * n];
            for (var rr = 0; rr < n; rr++) for (var j = 0; j < n; j++) dm[rr * n + j] = pts[others[rr]][j] - pts[b][j];
            var e = new double[n]; e[r] = 1;
            var dir = OptimizeDense.Solve(OptimizeDense.Transpose(dm, n, n), n, e);
            if (dir is null || dir.All(v => v == 0)) { dir = new double[n]; dir[r % n] = 1; }
            var nrm = OptimizeDense.NormInf(dir);
            for (var j = 0; j < n; j++) dir[j] /= nrm;
            // Step along ±dir as far as rho allows without leaving the box, on the roomier side.
            double Room(double sgn)
            {
                var t = double.PositiveInfinity;
                for (var j = 0; j < n; j++)
                {
                    var dj = sgn * dir[j];
                    if (dj > 0) t = Math.Min(t, (ub[j] - pts[b][j]) / dj);
                    else if (dj < 0) t = Math.Min(t, (lb[j] - pts[b][j]) / dj);
                }
                return t;
            }
            var rp = Room(1); var rm = Room(-1);
            var sg = rp >= rm ? 1.0 : -1.0;
            var len = Math.Min(rho, Math.Max(rp, rm));
            if (!(len > 0)) len = rho;
            var xn = new double[n];
            for (var j = 0; j < n; j++) xn[j] = Math.Clamp(pts[b][j] + sg * len * dir[j], lb[j], ub[j]);
            pts[far] = xn;
            (fv[far], cv[far]) = Eval(xn);
            return true;
        }

        var bi = Enumerable.Range(0, n + 1).OrderBy(k => Viol(cv[k]) > catol ? 1 : 0).ThenBy(k => Viol(cv[k]) > catol ? Viol(cv[k]) : fv[k]).First();
        var xr = pts[bi];
        var (ee, ii2) = cs.Eval(xr);
        var maxcv = OptimizeConstraintSet.Violation(ee, ii2);
        string[] msgs =
        [
            "Return from COBYLA because the trust region radius reaches its lower bound.",
            "Return from COBYLA because the target function value is achieved.",
            "Return from COBYLA because a trust region step has failed to reduce the quadratic model.",
            "Return from COBYLA because the objective function has been evaluated MAXFUN times.",
        ];
        var success = !stop && (info is 0 or 1) && maxcv <= catol;
        return new OptimizeResult
        {
            X = (double[])xr.Clone(), Fun = fv[bi], Nfev = nfev, Nit = nit, Status = info, Success = success, ConstrViolation = maxcv,
            Message = maxcv > catol ? "Did not converge to a solution satisfying the constraints. See `maxcv` for the magnitude of the violation." : msgs[info],
        };
    }

    /// <summary>LP trust-region step; when the linearised constraints cannot be met, first minimise their worst violation.</summary>
    private static double[] SolveTrLp(double[] gf, double[][] gc, double[] cb, double[] lo, double[] hi, int n, int mc)
    {
        // Variables (d, t): minimise t subject to c + G d + t ≥ 0, t ≥ 0.
        var nv = n + 1;
        var aub = new double[mc * nv];
        var bub = new double[mc];
        for (var i = 0; i < mc; i++)
        {
            for (var j = 0; j < n; j++) aub[i * nv + j] = -gc[i][j];
            aub[i * nv + n] = -1;
            bub[i] = cb[i];
        }
        var l1 = new double[nv]; var u1 = new double[nv];
        Array.Copy(lo, l1, n); Array.Copy(hi, u1, n); l1[n] = 0; u1[n] = double.PositiveInfinity;
        var c1 = new double[nv]; c1[n] = 1;
        var s1 = OptimizeLpSimplex.Solve(new OptimizeLp.Problem(c1, aub, bub, mc, [], [], 0, l1, u1), 1e-11, 10000, false);
        var tstar = s1.Status == 0 ? s1.X[n] : 0;
        // Then minimise the objective model with the violation capped at its minimum (plus slack).
        var c2 = new double[nv];
        Array.Copy(gf, c2, n);
        u1[n] = tstar * (1 + 1e-9) + 1e-15;
        var s2 = OptimizeLpSimplex.Solve(new OptimizeLp.Problem(c2, aub, bub, mc, [], [], 0, l1, u1), 1e-11, 10000, false);
        if (s2.Status == 0) return s2.X.Take(n).ToArray();
        return s1.Status == 0 ? s1.X.Take(n).ToArray() : new double[n];
    }
}
