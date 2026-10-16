using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Algorithms of <see cref="Optimize.Linprog"/>.</summary>
public enum LinprogMethod
{
    /// <summary>SciPy's default name; here it selects the bounded revised simplex (the most robust method in this library).</summary>
    Highs,
    /// <summary>Bounded revised simplex, two phases, Dantzig pricing with Bland's rule against cycling.</summary>
    Simplex,
    /// <summary>Primal–dual interior point with Mehrotra predictor–corrector; statuses other than "optimal" are confirmed by the simplex.</summary>
    InteriorPoint,
}

/// <summary>Options of <see cref="Optimize.Linprog"/>.</summary>
public sealed record LinprogOptions
{
    /// <summary>Feasibility/optimality tolerance. Default 1e-9 (simplex), 1e-8 (interior point).</summary>
    public double? Tol { get; init; }
    /// <summary>Iteration limit. Default 50 000 (simplex), 200 (interior point).</summary>
    public int? MaxIter { get; init; }
    /// <summary>Use Bland's rule from the start (slow but cannot cycle).</summary>
    public bool Bland { get; init; }
}

/// <summary>Result of <see cref="Optimize.Linprog"/>, mirroring SciPy's (HiGHS) fields.</summary>
public sealed class LinprogResult
{
    /// <summary>Solution.</summary>
    public double[] X { get; init; } = [];
    /// <summary>cᵀx.</summary>
    public double Fun { get; init; }
    /// <summary>b_ub − A_ub x (non-negative when feasible).</summary>
    public double[] Slack { get; init; } = [];
    /// <summary>b_eq − A_eq x.</summary>
    public double[] Con { get; init; } = [];
    /// <summary>Status = 0.</summary>
    public bool Success { get; init; }
    /// <summary>0 optimal, 1 iteration limit, 2 infeasible, 3 unbounded, 4 numerical difficulties.</summary>
    public int Status { get; init; }
    /// <summary>Termination message.</summary>
    public string Message { get; init; } = "";
    /// <summary>Iterations.</summary>
    public int Nit { get; init; }
    /// <summary>∂fun/∂b_ub (≤ 0).</summary>
    public double[] IneqlinMarginals { get; init; } = [];
    /// <summary>∂fun/∂b_eq.</summary>
    public double[] EqlinMarginals { get; init; } = [];
    /// <summary>∂fun/∂lb (≥ 0, non-zero only at an active lower bound).</summary>
    public double[] LowerMarginals { get; init; } = [];
    /// <summary>∂fun/∂ub (≤ 0, non-zero only at an active upper bound).</summary>
    public double[] UpperMarginals { get; init; } = [];
}

public static partial class Optimize
{
    /// <summary>
    /// Linear programming, <c>scipy.optimize.linprog</c>: minimise cᵀx subject to A_ub x ≤ b_ub,
    /// A_eq x = b_eq and bounds (default 0 ≤ x &lt; ∞, as SciPy).
    /// </summary>
    public static LinprogResult Linprog(double[] c, NdArray? aUb = null, double[]? bUb = null, NdArray? aEq = null, double[]? bEq = null,
        Bounds? bounds = null, LinprogMethod method = LinprogMethod.Highs, LinprogOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(c);
        var n = c.Length;
        if (n == 0) throw new ArgumentException("c must not be empty.");
        foreach (var v in c) if (!double.IsFinite(v)) throw new ArgumentException("c must be finite.");
        var (au, mu) = OptimizeLp.Matrix(aUb, bUb, n, "A_ub");
        var (ae, me) = OptimizeLp.Matrix(aEq, bEq, n, "A_eq");
        var lb = bounds?.Lb ?? new double[n];
        var ub = bounds?.Ub ?? Enumerable.Repeat(double.PositiveInfinity, n).ToArray();
        if (lb.Length != n) throw new ArgumentException("Bounds must have one entry per variable.");
        var o = options ?? new LinprogOptions();
        var lp = new OptimizeLp.Problem(c, au, bUb ?? [], mu, ae, bEq ?? [], me, lb, ub);
        OptimizeLp.Solution s;
        if (method == LinprogMethod.InteriorPoint)
        {
            s = OptimizeLpIpm.Solve(lp, o.Tol ?? 1e-8, o.MaxIter ?? 200);
            if (s.Status != 0) s = OptimizeLpSimplex.Solve(lp, o.Tol ?? 1e-9, o.MaxIter ?? 50000, o.Bland) with { Nit = s.Nit };
        }
        else s = OptimizeLpSimplex.Solve(lp, o.Tol ?? 1e-9, o.MaxIter ?? 50000, o.Bland);
        return OptimizeLp.ToResult(lp, s);
    }
}

/// <summary>Problem representation and helpers shared by the LP solvers.</summary>
internal static class OptimizeLp
{
    public sealed record Problem(double[] C, double[] AUb, double[] BUb, int MUb, double[] AEq, double[] BEq, int MEq, double[] Lb, double[] Ub)
    {
        public int N => C.Length;
    }

    public sealed record Solution(int Status, double[] X, double[] Y, double[] ReducedCosts, int Nit, string? Detail = null);

    public static (double[] A, int M) Matrix(NdArray? a, double[]? b, int n, string name)
    {
        if (a is null)
        {
            if (b is { Length: > 0 }) throw new ArgumentException($"{name} is missing but its right-hand side is given.");
            return ([], 0);
        }
        if (a.Rank == 1) a = a.Reshape(1, a.Size);
        if (a.Rank != 2 || a.Shape[1] != n) throw new ArgumentException($"{name} must have {n} columns.");
        var m = a.Shape[0];
        if (b is null || b.Length != m) throw new ArgumentException($"The right-hand side of {name} must have {m} entries.");
        var d = a.AsContiguous().ToArray();
        foreach (var v in d) if (!double.IsFinite(v)) throw new ArgumentException($"{name} must be finite.");
        foreach (var v in b) if (!double.IsFinite(v)) throw new ArgumentException($"The right-hand side of {name} must be finite.");
        return (d, m);
    }

    private static readonly string[] Messages =
    [
        "Optimization terminated successfully.",
        "Iteration limit reached.",
        "The problem is infeasible.",
        "The problem is unbounded.",
        "Numerical difficulties encountered.",
    ];

    public static LinprogResult ToResult(Problem p, Solution s)
    {
        var n = p.N;
        var x = s.X;
        double fun = 0;
        for (var j = 0; j < n; j++) fun += p.C[j] * x[j];
        var slack = new double[p.MUb];
        for (var i = 0; i < p.MUb; i++) { double ax = 0; for (var j = 0; j < n; j++) ax += p.AUb[i * n + j] * x[j]; slack[i] = p.BUb[i] - ax; }
        var con = new double[p.MEq];
        for (var i = 0; i < p.MEq; i++) { double ax = 0; for (var j = 0; j < n; j++) ax += p.AEq[i * n + j] * x[j]; con[i] = p.BEq[i] - ax; }
        var lowerM = new double[n]; var upperM = new double[n];
        if (s.Status == 0)
            for (var j = 0; j < n; j++)
            {
                var d = s.ReducedCosts[j];
                if (d > 0) lowerM[j] = d; else if (d < 0) upperM[j] = d;
            }
        return new LinprogResult
        {
            X = x, Fun = fun, Slack = slack, Con = con, Status = s.Status, Success = s.Status == 0, Nit = s.Nit,
            Message = s.Detail ?? Messages[s.Status],
            IneqlinMarginals = s.Status == 0 ? s.Y.Take(p.MUb).ToArray() : new double[p.MUb],
            EqlinMarginals = s.Status == 0 ? s.Y.Skip(p.MUb).Take(p.MEq).ToArray() : new double[p.MEq],
            LowerMarginals = lowerM, UpperMarginals = upperM,
        };
    }
}

/// <summary>
/// Dense bounded revised simplex. Columns: the n structural variables, one slack per inequality
/// row, one artificial per row. Phase I minimises the artificials; Phase II the true cost with the
/// artificials fixed at zero. B⁻¹ is kept explicitly with product-form row updates and rebuilt from
/// scratch every 64 pivots (and the basic values recomputed) so rounding cannot accumulate.
/// Dantzig pricing switches to Bland's rule after a run of degenerate pivots, which rules out cycling.
/// </summary>
internal static class OptimizeLpSimplex
{
    private enum VarState : byte { Basic, AtLower, AtUpper, FreeZero }

    public static OptimizeLp.Solution Solve(OptimizeLp.Problem p, double tol, int maxIter, bool blandAlways)
    {
        int n = p.N, mu = p.MUb, me = p.MEq, m = mu + me;
        var nTot = n + mu + m;
        for (var j = 0; j < n; j++)
        {
            if (p.Lb[j] > p.Ub[j]) return new OptimizeLp.Solution(2, new double[n], new double[m], new double[n], 0);
            if (double.IsPositiveInfinity(p.Lb[j]) || double.IsNegativeInfinity(p.Ub[j])) return new OptimizeLp.Solution(2, new double[n], new double[m], new double[n], 0);
        }
        // Row-major constraint matrix over all columns; artificial columns get their sign below.
        var a = new double[m * nTot];
        var b = new double[m];
        for (var i = 0; i < mu; i++) { for (var j = 0; j < n; j++) a[i * nTot + j] = p.AUb[i * n + j]; a[i * nTot + n + i] = 1; b[i] = p.BUb[i]; }
        for (var i = 0; i < me; i++) { for (var j = 0; j < n; j++) a[(mu + i) * nTot + j] = p.AEq[i * n + j]; b[mu + i] = p.BEq[i]; }
        var lo = new double[nTot]; var hi = new double[nTot];
        for (var j = 0; j < n; j++) { lo[j] = p.Lb[j]; hi[j] = p.Ub[j]; }
        for (var j = n; j < n + mu; j++) { lo[j] = 0; hi[j] = double.PositiveInfinity; }
        for (var j = n + mu; j < nTot; j++) { lo[j] = 0; hi[j] = double.PositiveInfinity; }

        var x = new double[nTot];
        var state = new VarState[nTot];
        for (var j = 0; j < n + mu; j++)
        {
            if (double.IsFinite(lo[j])) { x[j] = lo[j]; state[j] = VarState.AtLower; }
            else if (double.IsFinite(hi[j])) { x[j] = hi[j]; state[j] = VarState.AtUpper; }
            else { x[j] = 0; state[j] = VarState.FreeZero; }
        }
        // Residual of the rows with nonbasic values; choose slack or artificial as the initial basis.
        var basis = new int[m];
        var resid = new double[m];
        for (var i = 0; i < m; i++)
        {
            double s = b[i];
            for (var j = 0; j < n; j++) s -= a[i * nTot + j] * x[j];
            resid[i] = s;
        }
        for (var i = 0; i < m; i++)
        {
            var art = n + mu + i;
            if (i < mu && resid[i] >= 0)
            {
                basis[i] = n + i; state[n + i] = VarState.Basic; x[n + i] = resid[i];
                // The artificial of this row is not needed: fix it at zero.
                hi[art] = 0; state[art] = VarState.AtLower; x[art] = 0;
                a[i * nTot + art] = 1;
            }
            else
            {
                if (i < mu) { x[n + i] = 0; state[n + i] = VarState.AtLower; resid[i] -= 0; }
                var sign = resid[i] >= 0 ? 1.0 : -1.0;
                a[i * nTot + art] = sign;
                basis[i] = art; state[art] = VarState.Basic; x[art] = Math.Abs(resid[i]);
            }
        }

        var binv = new double[m * m];
        var ctx = new Ctx(m, nTot, a, b, lo, hi, x, state, basis, binv, tol);
        if (!ctx.Refactor()) return new OptimizeLp.Solution(4, x.Take(n).ToArray(), new double[m], new double[n], 0);

        // Phase I.
        var c1 = new double[nTot];
        for (var j = n + mu; j < nTot; j++) c1[j] = 1;
        var nit = 0;
        var anyArt = basis.Any(j => j >= n + mu && x[j] > 0);
        if (anyArt)
        {
            var st = ctx.Run(c1, maxIter, ref nit, blandAlways);
            if (st == 1) return new OptimizeLp.Solution(1, x.Take(n).ToArray(), new double[m], new double[n], nit);
            if (st == 4) return new OptimizeLp.Solution(4, x.Take(n).ToArray(), new double[m], new double[n], nit);
            double infeas = 0;
            for (var j = n + mu; j < nTot; j++) infeas += x[j];
            var bnorm = 1 + b.Select(Math.Abs).DefaultIfEmpty(0).Max();
            if (infeas > Math.Max(tol, 1e-9) * bnorm * 10) return new OptimizeLp.Solution(2, x.Take(n).ToArray(), new double[m], new double[n], nit);
        }
        // Fix the artificials at zero and pivot basic ones out where possible.
        for (var j = n + mu; j < nTot; j++) { hi[j] = 0; lo[j] = 0; if (state[j] != VarState.Basic) { state[j] = VarState.AtLower; x[j] = 0; } }
        ctx.DriveOutArtificials(n + mu);

        // Phase II.
        var c2 = new double[nTot];
        Array.Copy(p.C, c2, n);
        var st2 = ctx.Run(c2, maxIter, ref nit, blandAlways);
        var xs = x.Take(n).ToArray();
        // Snap tiny bound violations from rounding.
        for (var j = 0; j < n; j++) xs[j] = Math.Clamp(xs[j], p.Lb[j], p.Ub[j]);
        var y = ctx.Duals(c2);
        var d = new double[n];
        for (var j = 0; j < n; j++)
        {
            if (state[j] == VarState.Basic) continue;
            double s = c2[j];
            for (var i = 0; i < m; i++) s -= y[i] * a[i * nTot + j];
            d[j] = s;
        }
        return new OptimizeLp.Solution(st2, xs, y, d, nit);
    }

    private sealed class Ctx(int m, int nTot, double[] a, double[] b, double[] lo, double[] hi, double[] x, VarState[] state, int[] basis, double[] binv, double tol)
    {
        private int _sinceRefactor;

        /// <summary>Rebuilds B⁻¹ by Gauss–Jordan with partial pivoting and recomputes the basic values.</summary>
        public bool Refactor()
        {
            _sinceRefactor = 0;
            if (m == 0) return true;
            var bm = new double[m * m];
            for (var i = 0; i < m; i++) for (var k = 0; k < m; k++) bm[i * m + k] = a[i * nTot + basis[k]];
            Array.Clear(binv);
            for (var i = 0; i < m; i++) binv[i * m + i] = 1;
            for (var col = 0; col < m; col++)
            {
                var piv = col; var best = Math.Abs(bm[col * m + col]);
                for (var r = col + 1; r < m; r++) { var v = Math.Abs(bm[r * m + col]); if (v > best) { best = v; piv = r; } }
                if (best < 1e-13) return false;
                if (piv != col)
                {
                    for (var k = 0; k < m; k++) { (bm[col * m + k], bm[piv * m + k]) = (bm[piv * m + k], bm[col * m + k]); (binv[col * m + k], binv[piv * m + k]) = (binv[piv * m + k], binv[col * m + k]); }
                }
                var d = bm[col * m + col];
                for (var k = 0; k < m; k++) { bm[col * m + k] /= d; binv[col * m + k] /= d; }
                for (var r = 0; r < m; r++)
                {
                    if (r == col) continue;
                    var f = bm[r * m + col];
                    if (f == 0) continue;
                    for (var k = 0; k < m; k++) { bm[r * m + k] -= f * bm[col * m + k]; binv[r * m + k] -= f * binv[col * m + k]; }
                }
            }
            // x_B = B⁻¹ (b − N x_N)
            var rhs = (double[])b.Clone();
            for (var j = 0; j < nTot; j++)
            {
                if (state[j] == VarState.Basic || x[j] == 0) continue;
                for (var i = 0; i < m; i++) rhs[i] -= a[i * nTot + j] * x[j];
            }
            for (var i = 0; i < m; i++)
            {
                double s = 0;
                for (var k = 0; k < m; k++) s += binv[i * m + k] * rhs[k];
                x[basis[i]] = s;
            }
            return true;
        }

        public double[] Duals(double[] c)
        {
            var y = new double[m];
            for (var i = 0; i < m; i++)
            {
                var cb = c[basis[i]];
                if (cb == 0) continue;
                for (var k = 0; k < m; k++) y[k] += cb * binv[i * m + k];
            }
            return y;
        }

        private double[] Column(int q)
        {
            var alpha = new double[m];
            for (var i = 0; i < m; i++)
            {
                double s = 0;
                var row = i * m;
                for (var k = 0; k < m; k++) { var akq = a[k * nTot + q]; if (akq != 0) s += binv[row + k] * akq; }
                alpha[i] = s;
            }
            return alpha;
        }

        private void Pivot(int r, double[] alpha)
        {
            var ar = alpha[r];
            for (var k = 0; k < m; k++) binv[r * m + k] /= ar;
            for (var i = 0; i < m; i++)
            {
                if (i == r || alpha[i] == 0) continue;
                var f = alpha[i];
                for (var k = 0; k < m; k++) binv[i * m + k] -= f * binv[r * m + k];
            }
        }

        /// <summary>Simplex iterations for cost c. Returns 0 optimal, 1 iteration limit, 3 unbounded, 4 numerical trouble.</summary>
        public int Run(double[] c, int maxIter, ref int nit, bool blandAlways)
        {
            var degenerateRun = 0;
            var d = new double[nTot];
            while (true)
            {
                if (nit >= maxIter) return 1;
                var y = Duals(c);
                var bland = blandAlways || degenerateRun > 50;
                var q = -1; var dir = 0; double bestScore = 0;
                for (var j = 0; j < nTot; j++)
                {
                    if (state[j] == VarState.Basic || lo[j] == hi[j]) continue;
                    double s = c[j];
                    for (var i = 0; i < m; i++) { var aij = a[i * nTot + j]; if (aij != 0) s -= y[i] * aij; }
                    d[j] = s;
                    int dj = 0;
                    if (s < -tol && (state[j] == VarState.AtLower || state[j] == VarState.FreeZero)) dj = 1;
                    else if (s > tol && (state[j] == VarState.AtUpper || state[j] == VarState.FreeZero)) dj = -1;
                    if (dj == 0) continue;
                    if (bland) { q = j; dir = dj; break; }
                    if (Math.Abs(s) > bestScore) { bestScore = Math.Abs(s); q = j; dir = dj; }
                }
                if (q < 0) return 0;
                var alpha = Column(q);
                // Ratio test: basic i changes by −dir·t·alpha_i.
                var tMax = hi[q] - lo[q];
                if (double.IsNaN(tMax)) tMax = double.PositiveInfinity;
                var r = -1; var leaveToUpper = false;
                double bestPiv = 0;
                const double pivTol = 1e-9;
                for (var i = 0; i < m; i++)
                {
                    var g = dir * alpha[i];
                    if (Math.Abs(g) <= pivTol) continue;
                    var bi = basis[i];
                    double t; bool toUpper;
                    if (g > 0)
                    {
                        if (double.IsNegativeInfinity(lo[bi])) continue;
                        t = (x[bi] - lo[bi]) / g; toUpper = false;
                    }
                    else
                    {
                        if (double.IsPositiveInfinity(hi[bi])) continue;
                        t = (hi[bi] - x[bi]) / -g; toUpper = true;
                    }
                    if (t < 0) t = 0;
                    var tie = 1e-12 * Math.Max(1, double.IsInfinity(tMax) ? 1 : Math.Abs(tMax));
                    var better = t < tMax - tie
                        || (r >= 0 && t <= tMax + tie && (bland ? basis[i] < basis[r] : Math.Abs(g) > bestPiv));
                    if (better) { tMax = t; r = i; leaveToUpper = toUpper; bestPiv = Math.Abs(g); }
                }
                if (double.IsPositiveInfinity(tMax)) return 3;
                nit++;
                degenerateRun = tMax <= 1e-12 ? degenerateRun + 1 : 0;
                // Move.
                x[q] += dir * tMax;
                for (var i = 0; i < m; i++) x[basis[i]] -= dir * tMax * alpha[i];
                if (r < 0)
                {
                    // Bound flip of the entering variable.
                    state[q] = dir > 0 ? VarState.AtUpper : VarState.AtLower;
                    x[q] = dir > 0 ? hi[q] : lo[q];
                    continue;
                }
                var leaving = basis[r];
                state[leaving] = leaveToUpper ? VarState.AtUpper : VarState.AtLower;
                x[leaving] = leaveToUpper ? hi[leaving] : lo[leaving];
                basis[r] = q;
                state[q] = VarState.Basic;
                Pivot(r, alpha);
                if (++_sinceRefactor >= 64 && !Refactor()) return 4;
            }
        }

        /// <summary>Degenerate pivots replacing basic artificials by structural/slack columns; redundant rows keep theirs (fixed at 0).</summary>
        public void DriveOutArtificials(int firstArt)
        {
            for (var r = 0; r < m; r++)
            {
                if (basis[r] < firstArt) continue;
                for (var j = 0; j < firstArt; j++)
                {
                    if (state[j] == VarState.Basic) continue;
                    double v = 0;
                    for (var k = 0; k < m; k++) v += binv[r * m + k] * a[k * nTot + j];
                    if (Math.Abs(v) < 1e-7) continue;
                    var alpha = Column(j);
                    var leaving = basis[r];
                    state[leaving] = VarState.AtLower; x[leaving] = 0;
                    basis[r] = j; state[j] = VarState.Basic;
                    Pivot(r, alpha);
                    break;
                }
            }
            Refactor();
        }
    }
}

/// <summary>
/// Primal–dual interior point (Mehrotra 1992) on the standard form min cᵀz, Az = b, z ≥ 0, built
/// by shifting finite lower bounds, reflecting upper-only bounds, splitting free variables and
/// turning finite upper bounds into rows. Normal equations A D Aᵀ are factored by Cholesky with a
/// tiny diagonal regularisation. Only "optimal" is trusted from this solver; anything else is
/// handed to the simplex for the authoritative status.
/// </summary>
internal static class OptimizeLpIpm
{
    public static OptimizeLp.Solution Solve(OptimizeLp.Problem p, double tol, int maxIter)
    {
        int n = p.N, mu = p.MUb, me = p.MEq;
        // Column map from original variable j to standard columns.
        var cols = new List<(int Var, double Sign, double Shift)>();
        var ubRows = new List<(int Col, double U)>();
        var shift = new double[n];
        var plusCol = new int[n]; var minusCol = new int[n];
        for (var j = 0; j < n; j++)
        {
            double l = p.Lb[j], u = p.Ub[j];
            minusCol[j] = -1;
            if (double.IsFinite(l))
            {
                plusCol[j] = cols.Count; cols.Add((j, 1, l)); shift[j] = l;
                if (double.IsFinite(u)) ubRows.Add((plusCol[j], u - l));
            }
            else if (double.IsFinite(u)) { plusCol[j] = cols.Count; cols.Add((j, -1, u)); shift[j] = u; }
            else { plusCol[j] = cols.Count; cols.Add((j, 1, 0)); minusCol[j] = cols.Count; cols.Add((j, -1, 0)); }
        }
        var nStruct = cols.Count;
        var ns = nStruct + mu + ubRows.Count;
        var ms = mu + me + ubRows.Count;
        var A = new double[ms * ns];
        var bvec = new double[ms];
        var c = new double[ns];
        for (var k = 0; k < nStruct; k++) c[k] = cols[k].Sign * p.C[cols[k].Var];
        void FillRow(int row, double[] src, int srcRow, double rhs)
        {
            var r = rhs;
            for (var j = 0; j < n; j++)
            {
                var v = src[srcRow * n + j];
                if (v == 0) continue;
                var pc = plusCol[j];
                A[row * ns + pc] = v * cols[pc].Sign;
                r -= v * cols[pc].Shift;
                if (minusCol[j] >= 0) A[row * ns + minusCol[j]] = -v;
            }
            bvec[row] = r;
        }
        for (var i = 0; i < mu; i++) { FillRow(i, p.AUb, i, p.BUb[i]); A[i * ns + nStruct + i] = 1; }
        for (var i = 0; i < me; i++) FillRow(mu + i, p.AEq, i, p.BEq[i]);
        for (var k = 0; k < ubRows.Count; k++)
        {
            var row = mu + me + k;
            A[row * ns + ubRows[k].Col] = 1;
            A[row * ns + nStruct + mu + k] = 1;
            bvec[row] = ubRows[k].U;
        }

        var z = new double[ns]; var s = new double[ns]; var lam = new double[ms];
        if (!InitialPoint(A, bvec, c, ms, ns, z, s, lam)) return Fail(n, mu + me, 0);
        var bn = 1 + OptimizeDense.Norm2(bvec); var cn = 1 + OptimizeDense.Norm2(c);
        var rb = new double[ms]; var rc = new double[ns];
        var it = 0;
        for (; it < maxIter; it++)
        {
            OptimizeDense.MatVec(A, ms, ns, z, rb);
            for (var i = 0; i < ms; i++) rb[i] -= bvec[i];
            OptimizeDense.MatTVec(A, ms, ns, lam, rc);
            for (var j = 0; j < ns; j++) rc[j] += s[j] - c[j];
            var mu0 = OptimizeDense.Dot(z, s) / ns;
            var pobj = OptimizeDense.Dot(c, z); var dobj = OptimizeDense.Dot(bvec, lam);
            if (OptimizeDense.Norm2(rb) / bn < tol && OptimizeDense.Norm2(rc) / cn < tol && Math.Abs(pobj - dobj) / (1 + Math.Abs(pobj)) < tol) break;
            if (OptimizeDense.NormInf(z) > 1e14 || OptimizeDense.NormInf(lam) > 1e14 || !double.IsFinite(mu0)) return Fail(n, mu + me, it);
            var dd = new double[ns];
            for (var j = 0; j < ns; j++) dd[j] = z[j] / s[j];
            var mtx = new double[ms * ms];
            for (var i = 0; i < ms; i++)
                for (var k = i; k < ms; k++)
                {
                    double acc = 0;
                    for (var j = 0; j < ns; j++) { var aij = A[i * ns + j]; if (aij != 0) acc += aij * dd[j] * A[k * ns + j]; }
                    mtx[i * ms + k] = acc; mtx[k * ms + i] = acc;
                }
            double maxDiag = 0;
            for (var i = 0; i < ms; i++) maxDiag = Math.Max(maxDiag, mtx[i * ms + i]);
            for (var i = 0; i < ms; i++) mtx[i * ms + i] += 1e-14 * Math.Max(1, maxDiag);
            if (!OptimizeDense.Cholesky(mtx, ms)) return Fail(n, mu + me, it);

            (double[] Dz, double[] Dl, double[] Ds) Direction(double[] rxs)
            {
                // M Δλ = −rb − A(S⁻¹ rxs + D rc)
                var tmp = new double[ns];
                for (var j = 0; j < ns; j++) tmp[j] = rxs[j] / s[j] + dd[j] * rc[j];
                var rhs = new double[ms];
                OptimizeDense.MatVec(A, ms, ns, tmp, rhs);
                for (var i = 0; i < ms; i++) rhs[i] = -rb[i] - rhs[i];
                OptimizeDense.CholeskySolve(mtx, ms, rhs);
                var at = new double[ns];
                OptimizeDense.MatTVec(A, ms, ns, rhs, at);
                var dsv = new double[ns]; var dzv = new double[ns];
                for (var j = 0; j < ns; j++) { dsv[j] = -rc[j] - at[j]; dzv[j] = (rxs[j] - z[j] * dsv[j]) / s[j]; }
                return (dzv, rhs, dsv);
            }
            static double MaxStep(double[] v, double[] dv)
            {
                var a = 1.0;
                for (var j = 0; j < v.Length; j++) if (dv[j] < 0) a = Math.Min(a, -v[j] / dv[j]);
                return a;
            }
            var rxsAff = new double[ns];
            for (var j = 0; j < ns; j++) rxsAff[j] = -z[j] * s[j];
            var (dzA, _, dsA) = Direction(rxsAff);
            var ap = MaxStep(z, dzA); var ad = MaxStep(s, dsA);
            double muAff = 0;
            for (var j = 0; j < ns; j++) muAff += (z[j] + ap * dzA[j]) * (s[j] + ad * dsA[j]);
            muAff /= ns;
            var sigma = Math.Pow(muAff / mu0, 3);
            var rxs = new double[ns];
            for (var j = 0; j < ns; j++) rxs[j] = -z[j] * s[j] - dzA[j] * dsA[j] + sigma * mu0;
            var (dz, dl, ds) = Direction(rxs);
            ap = Math.Min(1, 0.995 * MaxStep(z, dz)); ad = Math.Min(1, 0.995 * MaxStep(s, ds));
            for (var j = 0; j < ns; j++) { z[j] += ap * dz[j]; s[j] += ad * ds[j]; }
            for (var i = 0; i < ms; i++) lam[i] += ad * dl[i];
        }
        if (it >= maxIter) return Fail(n, mu + me, it);
        var x = new double[n];
        for (var j = 0; j < n; j++)
        {
            x[j] = cols[plusCol[j]].Shift + cols[plusCol[j]].Sign * z[plusCol[j]];
            if (minusCol[j] >= 0) x[j] -= z[minusCol[j]];
            x[j] = Math.Clamp(x[j], p.Lb[j], p.Ub[j]);
        }
        var y = lam.Take(mu + me).ToArray();
        // Reduced costs of the original variables: c − Aᵀy, cleaned of interior noise.
        var red = new double[n];
        for (var j = 0; j < n; j++)
        {
            double sv = p.C[j];
            for (var i = 0; i < mu; i++) sv -= y[i] * p.AUb[i * n + j];
            for (var i = 0; i < me; i++) sv -= y[mu + i] * p.AEq[i * n + j];
            var atL = Math.Abs(x[j] - p.Lb[j]) <= 1e-7 * (1 + Math.Abs(x[j]));
            var atU = Math.Abs(x[j] - p.Ub[j]) <= 1e-7 * (1 + Math.Abs(x[j]));
            red[j] = (atL && sv > 0) || (atU && sv < 0) ? sv : 0;
        }
        return new OptimizeLp.Solution(0, x, y, red, it);
    }

    private static OptimizeLp.Solution Fail(int n, int m, int it) => new(4, new double[n], new double[m], new double[n], it);

    private static bool InitialPoint(double[] A, double[] b, double[] c, int m, int n, double[] x, double[] s, double[] lam)
    {
        // Mehrotra's starting point from the least-squares solutions of the primal and dual equations.
        var aat = new double[m * m];
        for (var i = 0; i < m; i++)
            for (var k = i; k < m; k++)
            {
                double acc = 0;
                for (var j = 0; j < n; j++) acc += A[i * n + j] * A[k * n + j];
                aat[i * m + k] = acc; aat[k * m + i] = acc;
            }
        for (var i = 0; i < m; i++) aat[i * m + i] += 1e-12 * (1 + aat[i * m + i]);
        if (!OptimizeDense.Cholesky(aat, m)) return false;
        var w = (double[])b.Clone();
        OptimizeDense.CholeskySolve(aat, m, w);
        OptimizeDense.MatTVec(A, m, n, w, x);
        var ac = new double[m];
        OptimizeDense.MatVec(A, m, n, c, ac);
        OptimizeDense.CholeskySolve(aat, m, ac);
        Array.Copy(ac, lam, m);
        var atl = new double[n];
        OptimizeDense.MatTVec(A, m, n, lam, atl);
        for (var j = 0; j < n; j++) s[j] = c[j] - atl[j];
        var dx = Math.Max(-1.5 * x.Min(), 0); var ds = Math.Max(-1.5 * s.Min(), 0);
        for (var j = 0; j < n; j++) { x[j] += dx; s[j] += ds; }
        var xs = OptimizeDense.Dot(x, s);
        var dx2 = 0.5 * xs / Math.Max(s.Sum(), 1e-300); var ds2 = 0.5 * xs / Math.Max(x.Sum(), 1e-300);
        for (var j = 0; j < n; j++) { x[j] += dx2; s[j] += ds2; if (x[j] <= 0) x[j] = 1; if (s[j] <= 0) s[j] = 1; }
        return true;
    }
}
