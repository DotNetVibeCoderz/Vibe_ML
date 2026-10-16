namespace Gravicode.Science.SciNet;

/// <summary>
/// Flattens a list of <see cref="IMinimizeConstraint"/> into equality rows (c(x) = 0) and
/// inequality rows (c(x) ≥ 0), the form SLSQP and COBYLA work in. Two-sided constraints become two
/// inequalities; lb = ub becomes an equality; infinite sides are dropped (SciPy's
/// <c>new_constraint_to_old</c>).
/// </summary>
internal sealed class OptimizeConstraintSet
{
    private sealed record Row(int Source, int Index, double Sign, double Offset, bool IsEq);

    private readonly IReadOnlyList<IMinimizeConstraint> _cons;
    private readonly List<Row> _eq = [], _in = [];
    private readonly int _n;
    private readonly double _eps;

    public int MEq => _eq.Count;
    public int MIn => _in.Count;
    public int M => MEq + MIn;

    public OptimizeConstraintSet(IReadOnlyList<IMinimizeConstraint> cons, double[] x0, double eps)
    {
        _cons = cons; _n = x0.Length; _eps = eps;
        for (var s = 0; s < cons.Count; s++)
        {
            switch (cons[s])
            {
                case LinearConstraint lc:
                    if (lc.A.Shape[1] != _n) throw new ArgumentException($"LinearConstraint has {lc.A.Shape[1]} columns but x has {_n}.");
                    AddTwoSided(s, lc.Lb, lc.Ub);
                    break;
                case NonlinearConstraint nc:
                {
                    var v = nc.Fun((double[])x0.Clone());
                    if (v.Length != nc.Lb.Length) throw new ArgumentException($"NonlinearConstraint returned {v.Length} values but has {nc.Lb.Length} bounds.");
                    AddTwoSided(s, nc.Lb, nc.Ub);
                    break;
                }
                case OptimizeConstraint oc:
                {
                    var v = oc.Fun((double[])x0.Clone());
                    for (var i = 0; i < v.Length; i++)
                        (oc.Type == OptimizeConstraintType.Eq ? _eq : _in).Add(new Row(s, i, 1, 0, oc.Type == OptimizeConstraintType.Eq));
                    break;
                }
                default:
                    throw new ArgumentException($"Unsupported constraint type {cons[s].GetType().Name}.");
            }
        }
    }

    private void AddTwoSided(int s, double[] lb, double[] ub)
    {
        for (var i = 0; i < lb.Length; i++)
        {
            if (lb[i] == ub[i]) { _eq.Add(new Row(s, i, 1, -lb[i], true)); continue; }
            if (double.IsFinite(lb[i])) _in.Add(new Row(s, i, 1, -lb[i], false));
            if (double.IsFinite(ub[i])) _in.Add(new Row(s, i, -1, ub[i], false));
        }
    }

    private double[][] Raw(double[] x)
    {
        var raw = new double[_cons.Count][];
        for (var s = 0; s < _cons.Count; s++)
        {
            raw[s] = _cons[s] switch
            {
                LinearConstraint lc => Mat(lc, x),
                NonlinearConstraint nc => nc.Fun((double[])x.Clone()),
                OptimizeConstraint oc => oc.Fun((double[])x.Clone()),
                _ => [],
            };
        }
        return raw;
    }

    private static double[] Mat(LinearConstraint lc, double[] x)
    {
        int m = lc.A.Shape[0], n = lc.A.Shape[1];
        var r = new double[m];
        OptimizeDense.MatVec(lc.A.AsContiguous().AsSpan(), m, n, x, r);
        return r;
    }

    /// <summary>Equality values (want 0) and inequality values (want ≥ 0).</summary>
    public (double[] Eq, double[] In) Eval(double[] x)
    {
        var raw = Raw(x);
        var eq = _eq.Select(r => r.Sign * raw[r.Source][r.Index] + r.Offset).ToArray();
        var ineq = _in.Select(r => r.Sign * raw[r.Source][r.Index] + r.Offset).ToArray();
        return (eq, ineq);
    }

    /// <summary>All rows stacked (equalities first) — handy for finite differences.</summary>
    public double[] EvalAll(double[] x)
    {
        var (e, i) = Eval(x);
        return [.. e, .. i];
    }

    /// <summary>Jacobian (M, n) of all rows, equalities first; analytic where given, forward differences (absolute step) otherwise.</summary>
    public double[] Jacobian(double[] x, double[]? lb, double[]? ub)
    {
        var jac = new double[M * _n];
        var srcJac = new double[_cons.Count][];
        var needFd = false;
        for (var s = 0; s < _cons.Count; s++)
        {
            switch (_cons[s])
            {
                case LinearConstraint lc: srcJac[s] = lc.A.AsContiguous().ToArray(); break;
                case NonlinearConstraint { Jac: { } j }: srcJac[s] = Check(j((double[])x.Clone()), s); break;
                case OptimizeConstraint { Jac: { } j }: srcJac[s] = Check(j((double[])x.Clone()), s); break;
                default: needFd = true; break;
            }
        }
        double[][]? fd = null;
        if (needFd)
        {
            fd = new double[_cons.Count][];
            var raw0 = Raw(x);
            for (var s = 0; s < _cons.Count; s++)
            {
                if (srcJac[s] is not null) continue;
                var ms = raw0[s].Length;
                var j = new double[ms * _n];
                var src = s;
                OptimizeFiniteDiff.Jacobian(z => Raw(z)[src], x, raw0[s], j, FiniteDiffScheme.TwoPoint, _eps, null, lb, ub);
                fd[s] = j;
            }
        }
        var rows = _eq.Concat(_in).ToList();
        for (var k = 0; k < rows.Count; k++)
        {
            var r = rows[k];
            var src = srcJac[r.Source] ?? fd![r.Source];
            for (var j = 0; j < _n; j++) jac[k * _n + j] = r.Sign * src[r.Index * _n + j];
        }
        return jac;
    }

    private double[] Check(Gravicode.Science.GraviNum.NdArray j, int s)
    {
        var a = j.AsContiguous().ToArray();
        if (a.Length % _n != 0) throw new ArgumentException($"Constraint {s} Jacobian must have {_n} columns.");
        return a;
    }

    /// <summary>Maximum violation at the given values.</summary>
    public static double Violation(double[] eq, double[] ineq)
    {
        double v = 0;
        foreach (var e in eq) v = Math.Max(v, Math.Abs(e));
        foreach (var c in ineq) v = Math.Max(v, -c);
        return v;
    }
}

/// <summary>
/// Goldfarb &amp; Idnani's (1983) dual active-set method for strictly convex QPs
/// min ½xᵀGx + aᵀx subject to Cᵢᵀx = bᵢ (first meq) and Cᵢᵀx ≥ bᵢ (rest), following the structure
/// of Turlach's <c>quadprog</c>: it starts from the unconstrained minimiser, so no feasible point is
/// needed, and reports infeasibility when no step can restore a violated constraint.
/// </summary>
internal static class OptimizeQuadprog
{
    /// <summary>Returns 0 solved, 1 infeasible, 2 G not positive definite / equalities dependent.</summary>
    public static int Solve(double[] g, int n, double[] a, double[][] cons, double[] b, int meq, out double[] x, out double[] lambda)
    {
        var q = cons.Length;
        x = new double[n];
        lambda = new double[q];
        var l = (double[])g.Clone();
        if (!OptimizeDense.Cholesky(l, n)) return 2;
        // J = L⁻ᵀ (columns), stored row-major n×n.
        var jm = new double[n * n];
        for (var c = 0; c < n; c++)
        {
            // Solve Lᵀ col = e_c.
            var col = new double[n];
            col[c] = 1;
            for (var i = n - 1; i >= 0; i--)
            {
                var s = col[i];
                for (var k = i + 1; k < n; k++) s -= l[k * n + i] * col[k];
                col[i] = s / l[i * n + i];
            }
            for (var i = 0; i < n; i++) jm[i * n + c] = col[i];
        }
        // x = -G⁻¹ a
        var xa = a.Select(v => -v).ToArray();
        OptimizeDense.CholeskySolve(l, n, xa);
        Array.Copy(xa, x, n);

        var r = new double[n * n]; // upper triangular, row-major, first nact columns used
        var active = new List<int>();
        var u = new List<double>();
        var inActive = new bool[q];
        var d = new double[n];
        var z = new double[n];
        var rv = new double[n];

        double Dot(double[] v, double[] w) => OptimizeDense.Dot(v, w);

        // Jᵀ v
        void JtV(double[] v, double[] outv)
        {
            for (var c = 0; c < n; c++) { double s = 0; for (var i = 0; i < n; i++) s += jm[i * n + c] * v[i]; outv[c] = s; }
        }

        bool AddConstraint(double[] dvec)
        {
            var nact = active.Count;
            // Zero d[nact+1..n-1] by Givens rotations from the bottom, rotating the columns of J.
            for (var j = n - 1; j > nact; j--)
            {
                var c1 = dvec[j - 1]; var c2 = dvec[j];
                if (c2 == 0) continue;
                var h = Math.Sqrt(c1 * c1 + c2 * c2);
                var cs = c1 / h; var sn = c2 / h;
                dvec[j - 1] = h; dvec[j] = 0;
                for (var i = 0; i < n; i++)
                {
                    var t1 = jm[i * n + j - 1]; var t2 = jm[i * n + j];
                    jm[i * n + j - 1] = cs * t1 + sn * t2;
                    jm[i * n + j] = -sn * t1 + cs * t2;
                }
            }
            if (Math.Abs(dvec[nact]) <= 1e-12 * Math.Max(1, OptimizeDense.NormInf(dvec))) return false;
            for (var i = 0; i <= nact; i++) r[i * n + nact] = dvec[i];
            return true;
        }

        void DropConstraint(int k)
        {
            var nact = active.Count;
            // Remove column k of R and restore triangularity with Givens rotations on rows k.., applied to J too.
            for (var c = k; c < nact - 1; c++)
                for (var i = 0; i < n; i++) r[i * n + c] = r[i * n + c + 1];
            for (var i = 0; i < n; i++) r[i * n + nact - 1] = 0;
            for (var c = k; c < nact - 1; c++)
            {
                var a1 = r[c * n + c]; var a2 = r[(c + 1) * n + c];
                if (a2 == 0) continue;
                var h = Math.Sqrt(a1 * a1 + a2 * a2);
                var cs = a1 / h; var sn = a2 / h;
                for (var cc = c; cc < nact - 1; cc++)
                {
                    var t1 = r[c * n + cc]; var t2 = r[(c + 1) * n + cc];
                    r[c * n + cc] = cs * t1 + sn * t2;
                    r[(c + 1) * n + cc] = -sn * t1 + cs * t2;
                }
                for (var i = 0; i < n; i++)
                {
                    var t1 = jm[i * n + c]; var t2 = jm[i * n + c + 1];
                    jm[i * n + c] = cs * t1 + sn * t2;
                    jm[i * n + c + 1] = -sn * t1 + cs * t2;
                }
            }
            inActive[active[k]] = false;
            active.RemoveAt(k);
            u.RemoveAt(k);
        }

        // Step direction for constraint np: z = J2 d2, rv = R⁻¹ d1.
        void Directions(double[] np)
        {
            var nact = active.Count;
            JtV(np, d);
            Array.Clear(z);
            for (var c = nact; c < n; c++) for (var i = 0; i < n; i++) z[i] += jm[i * n + c] * d[c];
            for (var i = nact - 1; i >= 0; i--)
            {
                var s = d[i];
                for (var k = i + 1; k < nact; k++) s -= r[i * n + k] * rv[k];
                rv[i] = s / r[i * n + i];
            }
        }

        var iter = 0;
        // Constraints are processed: equalities first (forced), then the most violated inequality.
        var eqNext = 0;
        var flipped = new bool[q];
        while (true)
        {
            if (++iter > 50 * (q + n) + 100) return 1;
            int p;
            if (eqNext < meq) p = eqNext++;
            else
            {
                p = -1; var worst = 0.0;
                for (var i = meq; i < q; i++)
                {
                    if (inActive[i]) continue;
                    var s = Dot(cons[i], x) - b[i];
                    var scale = 1 + Math.Abs(b[i]) + OptimizeDense.NormInf(cons[i]) * OptimizeDense.NormInf(x);
                    if (s < -1e-11 * scale && s < worst) { worst = s; p = i; }
                }
                if (p < 0) break;
            }
            var np = cons[p];
            var bp = b[p];
            if (p < meq && Dot(np, x) - bp > 0)
            {
                // Approach an equality from whichever side x is on.
                np = np.Select(v => -v).ToArray(); bp = -bp; flipped[p] = true;
            }
            var uPlus = 0.0;
            while (true)
            {
                Directions(np);
                var nact = active.Count;
                // Partial (dual) step bound.
                var t1 = double.PositiveInfinity; var kDrop = -1;
                for (var k = 0; k < nact; k++)
                {
                    if (active[k] < meq) continue;
                    if (rv[k] > 0 && u[k] / rv[k] < t1) { t1 = u[k] / rv[k]; kDrop = k; }
                }
                var zn = Dot(z, np);
                var sp = Dot(np, x) - bp;
                var t2 = Math.Abs(zn) > 1e-14 * Math.Max(1, Dot(np, np)) ? -sp / zn : double.PositiveInfinity;
                var t = Math.Min(t1, t2);
                if (double.IsPositiveInfinity(t)) return p < meq ? 2 : 1;
                if (double.IsPositiveInfinity(t2))
                {
                    for (var k = 0; k < nact; k++) u[k] -= t * rv[k];
                    uPlus += t;
                    DropConstraint(kDrop);
                    continue;
                }
                for (var i = 0; i < n; i++) x[i] += t * z[i];
                for (var k = 0; k < nact; k++) u[k] -= t * rv[k];
                uPlus += t;
                if (t == t2)
                {
                    var dv = new double[n];
                    JtV(np, dv);
                    if (!AddConstraint(dv)) return p < meq ? 2 : 1;
                    active.Add(p); u.Add(uPlus); inActive[p] = true;
                    break;
                }
                DropConstraint(kDrop);
                // Equalities must become active with an exact step; keep iterating on p.
            }
        }
        for (var k = 0; k < active.Count; k++) lambda[active[k]] = flipped[active[k]] ? -u[k] : u[k];
        return 0;
    }
}
