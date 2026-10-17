using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>y = H·p for a (Lagrangian) Hessian held as an operator.</summary>
internal delegate void TrustConstrMatVec(ReadOnlySpan<double> p, Span<double> y);

/// <summary>
/// Stop test called by the SQP loop once per iteration (and once before the first), as SciPy's
/// <c>stop_criteria(state, x, last_iteration_failed, optimality, constr_violation, tr_radius,
/// penalty, cg_info)</c>.
/// </summary>
internal delegate bool TrustConstrStop(double[] x, bool lastIterationFailed, double optimality, double constrViolation,
    double trRadius, double penalty, int cgNiter, int cgStopCond);

/// <summary>The functions an equality-constrained SQP solve needs (SciPy passes them as callables).</summary>
internal interface ITrustConstrSqpProblem
{
    /// <summary>f(x) and the equality residuals c(x).</summary>
    (double F, double[] C) FunAndConstr(double[] x);

    /// <summary>∇f(x) and the Jacobian of c, row-major (m, n).</summary>
    (double[] G, double[] A) GradAndJac(double[] x);

    /// <summary>Hessian of the Lagrangian at (x, v) as an operator.</summary>
    TrustConstrMatVec LagrangianHessian(double[] x, double[] v);

    /// <summary>Diagonal scaling of the step (null = identity): x⁺ = x + S·d.</summary>
    double[]? Scaling(double[] x);
}

/// <summary>
/// Orthogonal projections onto the null space and row space of a full-row-rank A (m, n), and the
/// least-squares multiplier map, as SciPy's <c>projections</c> with the dense
/// <c>QRFactorization</c> method: Aᵀ P = Q R by Householder QR with column pivoting, falling back to
/// a thresholded SVD when R's last diagonal is below 1e-15 (rank-deficient A) or when m &gt; n.
/// Each null-space projection is iteratively refined until ‖A z‖/(‖A‖_F ‖z‖) ≤ 1e-12, at most three
/// times — the projected CG loses feasibility of its iterates without it (Gould, Hribar &amp; Nocedal 2001).
/// </summary>
internal sealed class TrustConstrProjections
{
    private const double OrthTol = 1e-12;
    private const int MaxRefin = 3;
    private const double RankTol = 1e-15;

    private readonly int _m, _n;
    private readonly double[] _a;
    private readonly double _normA;
    // QR path
    private readonly double[]? _qr, _tau;
    private readonly int[]? _perm;
    // SVD path: A = U S Vᵀ restricted to s > tol.
    private readonly double[]? _u, _v, _s;
    private readonly int _uc, _vc;

    public TrustConstrProjections(double[] a, int m, int n)
    {
        _a = a; _m = m; _n = n;
        double fro = 0;
        foreach (var t in a) fro += t * t;
        _normA = Math.Sqrt(fro);
        if (m == 0) return;
        if (m <= n)
        {
            var qr = new double[n * m];
            for (var i = 0; i < m; i++)
                for (var j = 0; j < n; j++) qr[j * m + i] = a[i * n + j];
            var perm = new int[m];
            var tau = new double[m];
            OptimizeDense.QrPivot(qr, n, m, perm, tau);
            if (Math.Abs(qr[(m - 1) * m + (m - 1)]) >= RankTol)
            {
                _qr = qr; _perm = perm; _tau = tau;
                return;
            }
        }
        var svd = Decomposition.Svd(new NdArray((double[])a.Clone(), m, n));
        var u = svd.U.AsContiguous().ToArray();
        var s = svd.SingularValues.ToArray();
        var v = svd.V.AsContiguous().ToArray();
        _uc = u.Length / m; _vc = v.Length / n;
        var keep = new List<int>();
        for (var k = 0; k < s.Length; k++) if (s[k] > RankTol) keep.Add(k);
        _s = new double[keep.Count];
        _u = new double[m * keep.Count];
        _v = new double[n * keep.Count];
        for (var c = 0; c < keep.Count; c++)
        {
            _s[c] = s[keep[c]];
            for (var i = 0; i < m; i++) _u[i * keep.Count + c] = u[i * _uc + keep[c]];
            for (var j = 0; j < n; j++) _v[j * keep.Count + c] = v[j * _vc + keep[c]];
        }
        _uc = _vc = keep.Count;
    }

    /// <summary>(A Aᵀ)⁻¹ A x — the least-squares multipliers for gradient x.</summary>
    public double[] LeastSquares(ReadOnlySpan<double> x)
    {
        var r = new double[_m];
        if (_m == 0) return r;
        if (_qr is not null)
        {
            var buf = x.ToArray();
            OptimizeDense.ApplyQt(_qr, _n, _m, _tau, buf);
            BackSolve(buf);
            for (var j = 0; j < _m; j++) r[_perm![j]] = buf[j];
            return r;
        }
        var k = _s!.Length;
        var t = new double[k];
        for (var c = 0; c < k; c++)
        {
            double sum = 0;
            for (var j = 0; j < _n; j++) sum += _v![j * k + c] * x[j];
            t[c] = sum / _s[c];
        }
        for (var i = 0; i < _m; i++)
        {
            double sum = 0;
            for (var c = 0; c < k; c++) sum += _u![i * k + c] * t[c];
            r[i] = sum;
        }
        return r;
    }

    /// <summary>Aᵀ (A Aᵀ)⁻¹ y — the minimum-norm x with A x = y.</summary>
    public double[] RowSpace(ReadOnlySpan<double> y)
    {
        var z = new double[_n];
        if (_m == 0) return z;
        if (_qr is not null)
        {
            var aux = new double[_m];
            for (var j = 0; j < _m; j++) aux[j] = y[_perm![j]];
            // Solve Rᵀ w = aux (forward substitution).
            for (var i = 0; i < _m; i++)
            {
                var s = aux[i];
                for (var k = 0; k < i; k++) s -= _qr[k * _m + i] * aux[k];
                aux[i] = s / _qr[i * _m + i];
            }
            Array.Copy(aux, z, _m);
            OptimizeDense.ApplyQ(_qr, _n, _m, _tau, z);
            return z;
        }
        var kk = _s!.Length;
        var t = new double[kk];
        for (var c = 0; c < kk; c++)
        {
            double sum = 0;
            for (var i = 0; i < _m; i++) sum += _u![i * kk + c] * y[i];
            t[c] = sum / _s[c];
        }
        for (var j = 0; j < _n; j++)
        {
            double sum = 0;
            for (var c = 0; c < kk; c++) sum += _v![j * kk + c] * t[c];
            z[j] = sum;
        }
        return z;
    }

    /// <summary>Projection of x onto the null space of A, with iterative refinement.</summary>
    public double[] NullSpace(ReadOnlySpan<double> x)
    {
        var z = x.ToArray();
        if (_m == 0) return z;
        SubtractAt(z, LeastSquares(z));
        for (var k = 0; k < MaxRefin && Orthogonality(z) > OrthTol; k++)
            SubtractAt(z, LeastSquares(z));
        return z;
    }

    private void SubtractAt(double[] z, double[] v)
    {
        for (var i = 0; i < _m; i++)
        {
            var vi = v[i];
            if (vi == 0) continue;
            for (var j = 0; j < _n; j++) z[j] -= _a[i * _n + j] * vi;
        }
    }

    private double Orthogonality(double[] z)
    {
        var nz = OptimizeDense.Norm2(z);
        if (nz == 0 || _normA == 0) return 0;
        var az = new double[_m];
        OptimizeDense.MatVec(_a, _m, _n, z, az);
        return OptimizeDense.Norm2(az) / (_normA * nz);
    }

    private void BackSolve(double[] b)
    {
        for (var i = _m - 1; i >= 0; i--)
        {
            var s = b[i];
            for (var k = i + 1; k < _m; k++) s -= _qr![i * _m + k] * b[k];
            b[i] = s / _qr![i * _m + i];
        }
    }
}

/// <summary>
/// Port of SciPy's <c>_trustregion_constr</c> subproblem solvers and its Byrd–Omojokun SQP and
/// barrier (interior-point) drivers. Constants and their references follow SciPy's source, which
/// follows Byrd, Hribar &amp; Nocedal (1999) and Lalee, Nocedal &amp; Plantenga (1998).
/// </summary>
internal static class OptimizeTrustConstrSqp
{
    // ---------- intersections (qp_subproblem.py) ----------

    public static (double Ta, double Tb, bool Intersect) SphereIntersections(double[] z, double[] d, double trustRadius, bool entireLine = false)
    {
        if (OptimizeDense.Norm2(d) == 0) return (0, 0, false);
        if (double.IsPositiveInfinity(trustRadius))
            return entireLine ? (double.NegativeInfinity, double.PositiveInfinity, true) : (0, 1, true);
        var a = OptimizeDense.Dot(d, d);
        var b = 2 * OptimizeDense.Dot(z, d);
        var c = OptimizeDense.Dot(z, z) - trustRadius * trustRadius;
        var disc = b * b - 4 * a * c;
        if (disc < 0) return (0, 0, false);
        var sq = Math.Sqrt(disc);
        // The numerically stable pair of roots (no cancellation in either).
        var aux = b + Math.CopySign(sq, b);
        var ta = -aux / (2 * a);
        var tb = -2 * c / aux;
        if (ta > tb) (ta, tb) = (tb, ta);
        if (entireLine) return (ta, tb, true);
        if (tb < 0 || ta > 1) return (0, 0, false);
        return (Math.Max(0, ta), Math.Min(1, tb), true);
    }

    public static (double Ta, double Tb, bool Intersect) BoxIntersections(double[] z, double[] d, double[] lb, double[] ub, bool entireLine = false)
    {
        if (OptimizeDense.Norm2(d) == 0) return (0, 0, false);
        var ta = double.NegativeInfinity;
        var tb = double.PositiveInfinity;
        for (var i = 0; i < z.Length; i++)
        {
            if (d[i] == 0)
            {
                if (z[i] < lb[i] || z[i] > ub[i]) return (0, 0, false);
                continue;
            }
            var tl = (lb[i] - z[i]) / d[i];
            var tu = (ub[i] - z[i]) / d[i];
            ta = Math.Max(ta, Math.Min(tl, tu));
            tb = Math.Min(tb, Math.Max(tl, tu));
        }
        var intersect = ta <= tb;
        if (!entireLine)
        {
            if (tb < 0 || ta > 1) return (0, 0, false);
            ta = Math.Max(0, ta);
            tb = Math.Min(1, tb);
        }
        return (ta, tb, intersect);
    }

    public static (double Ta, double Tb, bool Intersect) BoxSphereIntersections(double[] z, double[] d, double[] lb, double[] ub, double trustRadius, bool entireLine = false)
    {
        var (tab, tbb, ib) = BoxIntersections(z, d, lb, ub, entireLine);
        var (tas, tbs, isph) = SphereIntersections(z, d, trustRadius, entireLine);
        var ta = Math.Max(tab, tas);
        var tb = Math.Min(tbb, tbs);
        return (ta, tb, ib && isph && ta <= tb);
    }

    private static bool InsideBox(double[] x, double[] lb, double[] ub)
    {
        for (var i = 0; i < x.Length; i++) if (!(lb[i] <= x[i]) || !(x[i] <= ub[i])) return false;
        return true;
    }

    private static double[] ClipToBox(double[] x, double[] lb, double[] ub)
    {
        var r = new double[x.Length];
        for (var i = 0; i < x.Length; i++) r[i] = Math.Min(Math.Max(x[i], lb[i]), ub[i]);
        return r;
    }

    private static double[] Combine(double[] a, double alpha, double[] p)
    {
        var r = new double[a.Length];
        for (var i = 0; i < a.Length; i++) r[i] = a[i] + alpha * p[i];
        return r;
    }

    /// <summary>
    /// Approximately minimises ½‖A x + b‖² within ‖x‖ ≤ Δ and lb ≤ x ≤ ub: the better of the dogleg
    /// path from the Cauchy point to the Newton point (−Aᵀ(AAᵀ)⁻¹b) and the scaled Newton step.
    /// </summary>
    public static double[] ModifiedDogleg(double[] a, int m, int n, TrustConstrProjections proj, double[] b, double trustRadius, double[] lb, double[] ub)
    {
        var newton = proj.RowSpace(b);
        for (var i = 0; i < n; i++) newton[i] = -newton[i];
        if (InsideBox(newton, lb, ub) && OptimizeDense.Norm2(newton) <= trustRadius) return newton;

        var g = new double[n];
        OptimizeDense.MatTVec(a, m, n, b, g);
        var ag = new double[m];
        OptimizeDense.MatVec(a, m, n, g, ag);
        var den = OptimizeDense.Dot(ag, ag);
        var cauchy = new double[n];
        if (den > 0)
        {
            var scale = -OptimizeDense.Dot(g, g) / den;
            for (var i = 0; i < n; i++) cauchy[i] = scale * g[i];
        }
        var origin = new double[n];

        double[] x1;
        var p = new double[n];
        for (var i = 0; i < n; i++) p[i] = newton[i] - cauchy[i];
        var (_, alpha, intersect) = BoxSphereIntersections(cauchy, p, lb, ub, trustRadius);
        if (intersect) x1 = Combine(cauchy, alpha, p);
        else
        {
            (_, alpha, _) = BoxSphereIntersections(origin, cauchy, lb, ub, trustRadius);
            x1 = Combine(origin, alpha, cauchy);
        }
        (_, alpha, _) = BoxSphereIntersections(origin, newton, lb, ub, trustRadius);
        var x2 = Combine(origin, alpha, newton);

        return ResidualNorm(a, m, n, x1, b) < ResidualNorm(a, m, n, x2, b) ? x1 : x2;
    }

    private static double ResidualNorm(double[] a, int m, int n, double[] x, double[] b)
    {
        var r = new double[m];
        OptimizeDense.MatVec(a, m, n, x, r);
        for (var i = 0; i < m; i++) r[i] += b[i];
        return OptimizeDense.Norm2(r);
    }

    /// <summary>
    /// Projected conjugate gradient for min ½xᵀHx + cᵀx subject to A x = −b, ‖x‖ ≤ Δ, lb ≤ x ≤ ub
    /// (Gould, Hribar &amp; Nocedal 2001, with SciPy's box handling: the iterate stops on the trust
    /// sphere, and a run of box-infeasible iterates falls back to the last feasible point on the path).
    /// Returns the step, the CG iteration count and the stop condition (1 max iter, 2 boundary,
    /// 3 negative curvature, 4 tolerance).
    /// </summary>
    public static (double[] X, int Niter, int StopCond, bool HitsBoundary) ProjectedCg(TrustConstrMatVec h, double[] c, TrustConstrProjections z,
        double[] b, int m, double trustRadius, double[] lb, double[] ub)
    {
        const double CloseToZero = 1e-25;
        var n = c.Length;
        var negB = new double[m];
        for (var i = 0; i < m; i++) negB[i] = -b[i];
        var x = z.RowSpace(negB);
        var hx = new double[n];
        h(x, hx);
        for (var i = 0; i < n; i++) hx[i] += c[i];
        var r = z.NullSpace(hx);
        var g = z.NullSpace(r);
        var p = new double[n];
        for (var i = 0; i < n; i++) p[i] = -g[i];
        var hp = new double[n];
        h(p, hp);
        var gn = OptimizeDense.Norm2(g);
        var rtg = gn * gn;
        var trDistance = trustRadius - OptimizeDense.Norm2(x);
        if (trDistance < 0) throw new InvalidOperationException("Trust region problem does not have a solution.");
        if (trDistance < CloseToZero) return (x, 0, 2, true);
        var tol = Math.Max(Math.Min(0.01 * Math.Sqrt(rtg), 0.1 * rtg), CloseToZero);
        var maxIter = n - m;
        var maxInfeasible = n - m;

        var hitsBoundary = false;
        var stopCond = 1;
        var counter = 0;
        var lastFeasible = new double[n];
        var k = 0;
        for (var it = 0; it < maxIter; it++)
        {
            if (rtg < tol) { stopCond = 4; break; }
            k++;
            var pthp = OptimizeDense.Dot(hp, p);
            if (pthp <= 0)
            {
                if (double.IsPositiveInfinity(trustRadius)) throw new InvalidOperationException("Negative curvature not allowed for unrestricted problems.");
                var (_, alphaB, inter) = BoxSphereIntersections(x, p, lb, ub, trustRadius, entireLine: true);
                if (inter) x = Combine(x, alphaB, p);
                x = ClipToBox(x, lb, ub);
                stopCond = 3; hitsBoundary = true;
                break;
            }
            var alpha = rtg / pthp;
            var xNext = Combine(x, alpha, p);
            if (OptimizeDense.Norm2(xNext) >= trustRadius)
            {
                var ap = new double[n];
                for (var i = 0; i < n; i++) ap[i] = alpha * p[i];
                var (_, theta, inter) = BoxSphereIntersections(x, ap, lb, ub, trustRadius);
                if (inter) x = Combine(x, theta, ap);
                x = ClipToBox(x, lb, ub);
                stopCond = 2; hitsBoundary = true;
                break;
            }
            counter = InsideBox(xNext, lb, ub) ? 0 : counter + 1;
            if (counter > 0)
            {
                var ap = new double[n];
                for (var i = 0; i < n; i++) ap[i] = alpha * p[i];
                var (_, theta, inter) = BoxSphereIntersections(x, ap, lb, ub, trustRadius);
                if (inter)
                {
                    lastFeasible = ClipToBox(Combine(x, theta, ap), lb, ub);
                    counter = 0;
                }
            }
            if (counter > maxInfeasible) break;
            var rNext = Combine(r, alpha, hp);
            var gNext = z.NullSpace(rNext);
            var gnn = OptimizeDense.Norm2(gNext);
            var beta = gnn * gnn / rtg;
            for (var i = 0; i < n; i++) p[i] = -gNext[i] + beta * p[i];
            x = xNext;
            g = gNext;
            r = gNext;
            rtg = gnn * gnn;
            h(p, hp);
        }
        if (!InsideBox(x, lb, ub)) { x = lastFeasible; hitsBoundary = true; }
        return (x, k, stopCond, hitsBoundary);
    }

    // ---------- equality_constrained_sqp.py ----------

    /// <summary>
    /// Byrd–Omojokun trust-region SQP for min f(x) s.t. c(x) = 0. Each step is a normal step
    /// (towards feasibility, within 0.8Δ) plus a tangential step (projected CG in the null space of
    /// the linearised constraints), judged by the ℓ₂ merit function f + μ‖c‖ whose penalty μ is
    /// raised just enough for the step to be a descent direction (Byrd, Hribar &amp; Nocedal eq. 3.52).
    /// A second-order correction rescues steps rejected only because of constraint curvature (Maratos).
    /// </summary>
    public static double[] EqualityConstrainedSqp(ITrustConstrSqpProblem prob, double[] x0, double fun0, double[] grad0,
        double[] constr0, double[] jac0, TrustConstrStop stop, double initialPenalty, double initialTrustRadius,
        double[]? trustLb, double[]? trustUb)
    {
        const double PenaltyFactor = 0.3;
        const double LargeReductionRatio = 0.9;
        const double IntermediaryReductionRatio = 0.3;
        const double SufficientReductionRatio = 1e-8;
        const double TrustEnlargementFactorL = 7.0;
        const double TrustEnlargementFactorS = 2.0;
        const double MaxTrustReduction = 0.5;
        const double MinTrustReduction = 0.1;
        const double SocThreshold = 0.1;
        const double TrFactor = 0.8;
        const double BoxFactor = 0.5;

        var n = x0.Length;
        var m = constr0.Length;
        if (trustLb is null) { trustLb = new double[n]; Array.Fill(trustLb, double.NegativeInfinity); }
        if (trustUb is null) { trustUb = new double[n]; Array.Fill(trustUb, double.PositiveInfinity); }
        var boxLb = new double[n]; var boxUb = new double[n];
        for (var i = 0; i < n; i++) { boxLb[i] = BoxFactor * trustLb[i]; boxUb[i] = BoxFactor * trustUb[i]; }

        var x = (double[])x0.Clone();
        var trustRadius = initialTrustRadius;
        var penalty = initialPenalty;
        var f = fun0;
        var c = grad0;
        var b = constr0;
        var a = jac0;
        var s = prob.Scaling(x);
        var proj = new TrustConstrProjections(a, m, n);
        var v = proj.LeastSquares(c);
        for (var i = 0; i < m; i++) v[i] = -v[i];
        var h = prob.LagrangianHessian(x, v);
        var optimality = LagrangianOptimality(c, a, v, m, n);
        var constrViolation = m > 0 ? OptimizeDense.NormInf(b) : 0;
        var cgNiter = 0; var cgStop = 0;
        var lastFailed = false;

        while (!stop(x, lastFailed, optimality, constrViolation, trustRadius, penalty, cgNiter, cgStop))
        {
            var dn = ModifiedDogleg(a, m, n, proj, b, TrFactor * trustRadius, boxLb, boxUb);

            var hdn = new double[n];
            h(dn, hdn);
            var ct = new double[n];
            for (var i = 0; i < n; i++) ct[i] = hdn[i] + c[i];
            var bt = new double[m];
            var dnNorm = OptimizeDense.Norm2(dn);
            var trT = Math.Sqrt(trustRadius * trustRadius - dnNorm * dnNorm);
            var lbT = new double[n]; var ubT = new double[n];
            for (var i = 0; i < n; i++) { lbT[i] = trustLb[i] - dn[i]; ubT[i] = trustUb[i] - dn[i]; }
            var (dt, cgIt, cgSc, _) = ProjectedCg(h, ct, proj, bt, m, trT, lbT, ubT);
            cgNiter = cgIt; cgStop = cgSc;

            var d = new double[n];
            for (var i = 0; i < n; i++) d[i] = dn[i] + dt[i];
            var hd = new double[n];
            h(d, hd);
            var quadraticModel = 0.5 * OptimizeDense.Dot(hd, d) + OptimizeDense.Dot(c, d);
            var lin = new double[m];
            OptimizeDense.MatVec(a, m, n, d, lin);
            for (var i = 0; i < m; i++) lin[i] += b[i];
            var vpred = Math.Max(1e-16, OptimizeDense.Norm2(b) - OptimizeDense.Norm2(lin));
            var previousPenalty = penalty;
            if (quadraticModel > 0)
                penalty = Math.Max(penalty, quadraticModel / ((1 - PenaltyFactor) * vpred));
            var predictedReduction = -quadraticModel + penalty * vpred;

            var merit = f + penalty * OptimizeDense.Norm2(b);
            var xNext = Step(x, s, d);
            var (fNext, bNext) = prob.FunAndConstr(xNext);
            var meritNext = fNext + penalty * OptimizeDense.Norm2(bNext);
            var reductionRatio = (merit - meritNext) / predictedReduction;

            if (reductionRatio < SufficientReductionRatio && dnNorm <= SocThreshold * OptimizeDense.Norm2(dt))
            {
                var y = proj.RowSpace(bNext);
                for (var i = 0; i < n; i++) y[i] = -y[i];
                var (_, t, intersect) = BoxIntersections(d, y, trustLb, trustUb);
                var dSoc = Combine(d, t, y);
                var xSoc = Step(x, s, dSoc);
                var (fSoc, bSoc) = prob.FunAndConstr(xSoc);
                var ratioSoc = (merit - (fSoc + penalty * OptimizeDense.Norm2(bSoc))) / predictedReduction;
                if (intersect && ratioSoc >= SufficientReductionRatio)
                {
                    xNext = xSoc; fNext = fSoc; bNext = bSoc; reductionRatio = ratioSoc;
                }
            }

            var dNorm = OptimizeDense.Norm2(d);
            if (reductionRatio >= LargeReductionRatio)
                trustRadius = Math.Max(TrustEnlargementFactorL * dNorm, trustRadius);
            else if (reductionRatio >= IntermediaryReductionRatio)
                trustRadius = Math.Max(TrustEnlargementFactorS * dNorm, trustRadius);
            else if (reductionRatio < SufficientReductionRatio)
            {
                // NaN ratios (a NaN objective) land here too and shrink the region, as in SciPy.
                var trustReduction = (1 - SufficientReductionRatio) / (1 - reductionRatio);
                var newTr = trustReduction * dNorm;
                if (newTr >= MaxTrustReduction * trustRadius) trustRadius *= MaxTrustReduction;
                else if (newTr >= MinTrustReduction * trustRadius) trustRadius = newTr;
                else trustRadius *= MinTrustReduction;
            }

            if (reductionRatio >= SufficientReductionRatio)
            {
                x = xNext;
                f = fNext; b = bNext;
                (c, a) = prob.GradAndJac(x);
                s = prob.Scaling(x);
                proj = new TrustConstrProjections(a, m, n);
                v = proj.LeastSquares(c);
                for (var i = 0; i < m; i++) v[i] = -v[i];
                h = prob.LagrangianHessian(x, v);
                lastFailed = false;
                optimality = LagrangianOptimality(c, a, v, m, n);
                constrViolation = m > 0 ? OptimizeDense.NormInf(b) : 0;
            }
            else
            {
                penalty = previousPenalty;
                lastFailed = true;
            }
        }
        return x;
    }

    private static double[] Step(double[] x, double[]? s, double[] d)
    {
        var r = new double[x.Length];
        for (var i = 0; i < x.Length; i++) r[i] = x[i] + (s is null ? d[i] : s[i] * d[i]);
        return r;
    }

    private static double LagrangianOptimality(double[] c, double[] a, double[] v, int m, int n)
    {
        var lg = (double[])c.Clone();
        if (m > 0)
        {
            var atv = new double[n];
            OptimizeDense.MatTVec(a, m, n, v, atv);
            for (var i = 0; i < n; i++) lg[i] += atv[i];
        }
        return OptimizeDense.NormInf(lg);
    }
}
