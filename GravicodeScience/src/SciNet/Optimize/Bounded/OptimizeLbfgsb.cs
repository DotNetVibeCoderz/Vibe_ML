using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Options for <see cref="MinimizeMethod.LBfgsB"/> (SciPy's <c>_minimize_lbfgsb</c> keywords).</summary>
public sealed record LBfgsBOptions : MinimizeOptions
{
    /// <summary>Number of correction pairs kept (SciPy's <c>maxcor</c>). Default 10.</summary>
    public int M { get; init; } = 10;
    /// <summary>Stop when (f_k − f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) ≤ ftol. Default 2.22e-9 (factr 1e7).</summary>
    public double? Ftol { get; init; }
    /// <summary>Stop when the ∞-norm of the projected gradient ≤ gtol. Default 1e-5.</summary>
    public double? Gtol { get; init; }
    /// <summary>Absolute forward-difference step when no gradient is given. Default 1e-8.</summary>
    public double Eps { get; init; } = 1e-8;
    /// <summary>Maximum function evaluations (checked once per iteration, as SciPy). Default 15000.</summary>
    public int MaxFun { get; init; } = 15000;
    /// <summary>Maximum line-search steps per iteration. Default 20.</summary>
    public int MaxLs { get; init; } = 20;
}

public static partial class Optimize
{
    static partial void RunLBfgsB(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeLbfgsb.Run(problem);
}

/// <summary>
/// L-BFGS-B 3.0 (Byrd, Lu, Nocedal &amp; Zhu 1995; Morales &amp; Nocedal 2011), translated routine by
/// routine from the Fortran: <c>cauchy</c> (generalised Cauchy point by breakpoint search with a
/// heap), <c>freev</c>/<c>formk</c>/<c>cmprlb</c>/<c>subsm</c> (direct primal subspace minimisation
/// with the 3.0 projection-and-backtrack fix), <c>lnsrlb</c> + <c>dcsrch</c> (Moré–Thuente),
/// <c>matupd</c>/<c>formt</c>/<c>bmv</c> (compact limited-memory matrices).
/// </summary>
/// <remarks>
/// The code keeps the Fortran's 1-based, column-major indexing internally: every index expression
/// can be checked against the original line by line, which matters more here than idiom.
/// </remarks>
internal static class OptimizeLbfgsb
{
    private const double Epsmch = 2.220446049250313e-16;

    /// <summary>1-based column-major dense matrix (LINPACK conventions).</summary>
    private sealed class FMat(int rows, int cols)
    {
        public readonly double[] A = new double[rows * cols];
        public readonly int Ld = rows;
        public ref double this[int i, int j] => ref A[(i - 1) + (j - 1) * Ld];
    }

    private sealed class State
    {
        public int N, M;
        public double[] L = null!, U = null!; // 1-based
        public int[] Nbd = null!;
        public FMat Ws = null!, Wy = null!, Sy = null!, Ss = null!, Wt = null!, Wn = null!, Snd = null!;
        public int Col, Head = 1, Itail, Iupdat;
        public double Theta = 1;
        public bool Updatd;
        public int Nenter, Ileave;
        public void Refresh() { Col = 0; Head = 1; Theta = 1; Iupdat = 0; Updatd = false; }
    }

    public static OptimizeResult Run(MinimizeProblem p)
    {
        var o = p.OptionsAs<LBfgsBOptions>("L-BFGS-B");
        if (o.M <= 0) throw new ArgumentException("M must be positive.");
        if (o.MaxLs <= 0) throw new ArgumentException("maxls must be positive.");
        var ftol = o.Ftol ?? p.Tol ?? 2.2204460492503131e-09;
        var pgtol = o.Gtol ?? p.Tol ?? 1e-5;
        var maxiter = o.MaxIter ?? 15000;
        var factr = ftol / Epsmch;
        var tol = factr * Epsmch;
        int n = p.N, m = o.M;
        var sf = p.CreateObjective(o.Eps);

        var st = new State { N = n, M = m, Ileave = n + 1, L = new double[n + 1], U = new double[n + 1], Nbd = new int[n + 1] };
        for (var i = 1; i <= n; i++)
        {
            var lo = p.Bounds?.Lb[i - 1] ?? double.NegativeInfinity;
            var hi = p.Bounds?.Ub[i - 1] ?? double.PositiveInfinity;
            var hasL = !double.IsInfinity(lo); var hasU = !double.IsInfinity(hi);
            if (hasL) st.L[i] = lo;
            if (hasU) st.U[i] = hi;
            st.Nbd[i] = hasL && hasU ? 2 : hasL ? 1 : hasU ? 3 : 0;
        }
        st.Ws = new FMat(n, m); st.Wy = new FMat(n, m);
        st.Sy = new FMat(m, m); st.Ss = new FMat(m, m); st.Wt = new FMat(m, m);
        st.Wn = new FMat(2 * m, 2 * m); st.Snd = new FMat(2 * m, 2 * m);

        var x = new double[n + 1];
        for (var i = 1; i <= n; i++) x[i] = p.X0[i - 1];
        var g = new double[n + 1];
        var z = new double[n + 1]; var r = new double[n + 1]; var d = new double[n + 1]; var t = new double[n + 1];
        var xp = new double[n + 1];
        var wa = new double[8 * m + 1];
        var index = new int[n + 1]; var iwhere = new int[n + 1]; var indx2 = new int[n + 1];

        // active: project x and classify variables.
        var cnstnd = false; var boxed = true;
        for (var i = 1; i <= n; i++)
        {
            if (st.Nbd[i] > 0)
            {
                if (st.Nbd[i] <= 2 && x[i] <= st.L[i]) x[i] = st.L[i];
                else if (st.Nbd[i] >= 2 && x[i] >= st.U[i]) x[i] = st.U[i];
            }
            if (st.Nbd[i] != 2) boxed = false;
            if (st.Nbd[i] == 0) iwhere[i] = -1;
            else { cnstnd = true; iwhere[i] = st.Nbd[i] == 2 && st.U[i] - st.L[i] <= 0 ? 3 : 0; }
        }

        double f = Eval(sf, x, g, n);
        var sbgnrm = Projgr(st, x, g);
        var nIter = 0; var mainIter = 0;
        var nfree = n;
        var task = 0; var taskMsg = "";
        double fold = 0, gd = 0, gdold = 0, stp = 0, dtd = 0;

        if (sbgnrm <= pgtol) { task = 4; taskMsg = "NORM OF PROJECTED GRADIENT <= PGTOL"; }
        else
        {
            for (var i = 1; i <= n; i++) index[i] = i;
            while (true)
            {
                // ---- 222: search direction ----
                var wrk = false;
                var info = 0;
                var skipToLineSearch = false;
                if (!cnstnd && st.Col > 0)
                {
                    Array.Copy(x, z, n + 1);
                    wrk = st.Updatd;
                }
                else
                {
                    info = Cauchy(st, x, g, indx2, iwhere, t, d, z, wa, sbgnrm);
                    if (info != 0) { st.Refresh(); continue; }
                    // freev
                    int nenter = 0, ileave = n + 1;
                    if (mainIter > 0 && cnstnd)
                    {
                        for (var i = 1; i <= nfree; i++) { var k = index[i]; if (iwhere[k] > 0) { ileave--; indx2[ileave] = k; } }
                        for (var i = 1 + nfree; i <= n; i++) { var k = index[i]; if (iwhere[k] <= 0) { nenter++; indx2[nenter] = k; } }
                    }
                    wrk = ileave < n + 1 || nenter > 0 || st.Updatd;
                    nfree = 0; var iact = n + 1;
                    for (var i = 1; i <= n; i++)
                    {
                        if (iwhere[i] <= 0) { nfree++; index[nfree] = i; }
                        else { iact--; index[iact] = i; }
                    }
                    st.Nenter = nenter; st.Ileave = ileave;
                }
                if (nfree == 0 || st.Col == 0) skipToLineSearch = true;
                if (!skipToLineSearch)
                {
                    if (wrk) info = Formk(st, nfree, index, indx2);
                    if (info != 0) { st.Refresh(); continue; }
                    info = Cmprlb(st, x, g, z, r, wa, index, nfree, cnstnd);
                    if (info == 0) info = Subsm(st, nfree, index, z, r, xp, x, g, wa);
                    if (info != 0) { st.Refresh(); continue; }
                }

                // ---- 555: line search ----
                for (var i = 1; i <= n; i++) d[i] = z[i] - x[i];
                dtd = 0;
                for (var i = 1; i <= n; i++) dtd += d[i] * d[i];
                var dnorm = Math.Sqrt(dtd);
                var stpmx = 1e10;
                if (cnstnd)
                {
                    if (mainIter == 0) stpmx = 1;
                    else
                        for (var i = 1; i <= n; i++)
                        {
                            var a1 = d[i];
                            if (st.Nbd[i] == 0) continue;
                            if (a1 < 0 && st.Nbd[i] <= 2)
                            {
                                var a2 = st.L[i] - x[i];
                                if (a2 >= 0) stpmx = 0; else if (a1 * stpmx < a2) stpmx = a2 / a1;
                            }
                            else if (a1 > 0 && st.Nbd[i] >= 2)
                            {
                                var a2 = st.U[i] - x[i];
                                if (a2 <= 0) stpmx = 0; else if (a1 * stpmx > a2) stpmx = a2 / a1;
                            }
                        }
                }
                stp = mainIter == 0 && !boxed ? Math.Min(1.0 / dnorm, stpmx) : 1.0;
                Array.Copy(x, t, n + 1);
                Array.Copy(g, r, n + 1);
                fold = f;
                gd = 0;
                for (var i = 1; i <= n; i++) gd += g[i] * d[i];
                gdold = gd;
                string lsTask;
                if (gd >= 0) lsTask = "ERROR: ascent direction";
                else
                {
                    var fl = f; var gl = gd;
                    var zz = z; var dd = d; var tt = t;
                    double Phi(double s, out double der)
                    {
                        if (s == 1) Array.Copy(zz, x, n + 1);
                        else for (var i = 1; i <= n; i++) x[i] = s * dd[i] + tt[i];
                        var fv = Eval(sf, x, g, n);
                        double gdv = 0;
                        for (var i = 1; i <= n; i++) gdv += g[i] * dd[i];
                        der = gdv;
                        return fv;
                    }
                    lsTask = OptimizeLineSearch.Dcsrch(Phi, ref stp, ref fl, ref gl, 1e-3, 0.9, 0.1, 0.0, stpmx, o.MaxLs, out _);
                    f = fl; gd = gl;
                }
                if (!(lsTask.StartsWith("CONV", StringComparison.Ordinal) || (lsTask.StartsWith("WARN", StringComparison.Ordinal) && lsTask != "WARNING: MAXITER")))
                {
                    // Restore the previous iterate.
                    Array.Copy(t, x, n + 1);
                    Array.Copy(r, g, n + 1);
                    f = fold;
                    if (st.Col == 0)
                    {
                        task = 8; taskMsg = "";
                        mainIter++;
                        break;
                    }
                    st.Refresh();
                    continue;
                }

                // ---- NEW_X ----
                mainIter++;
                sbgnrm = Projgr(st, x, g);
                nIter++;
                var xs = new double[n];
                Array.Copy(x, 1, xs, 0, n);
                if (p.NotifyCallback(xs, f)) { task = 5; taskMsg = "CALLBACK REQUESTED HALT"; }
                if (nIter >= maxiter) { task = 5; taskMsg = "TOTAL NO. OF ITERATIONS REACHED LIMIT"; }
                else if (sf.Nfev > o.MaxFun) { task = 5; taskMsg = "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"; }
                if (task == 5) break;

                // ---- 777: termination tests ----
                if (sbgnrm <= pgtol) { task = 4; taskMsg = "NORM OF PROJECTED GRADIENT <= PGTOL"; break; }
                var ddum = Math.Max(Math.Abs(fold), Math.Max(Math.Abs(f), 1.0));
                if (fold - f <= tol * ddum) { task = 4; taskMsg = "RELATIVE REDUCTION OF F <= FACTR*EPSMCH"; break; }

                for (var i = 1; i <= n; i++) r[i] = g[i] - r[i];
                double rr = 0;
                for (var i = 1; i <= n; i++) rr += r[i] * r[i];
                double dr;
                if (stp == 1) { dr = gd - gdold; ddum = -gdold; }
                else
                {
                    dr = (gd - gdold) * stp;
                    for (var i = 1; i <= n; i++) d[i] *= stp;
                    ddum = -gdold * stp;
                }
                if (dr <= Epsmch * ddum) { st.Updatd = false; continue; }
                st.Updatd = true;
                st.Iupdat++;
                Matupd(st, d, r, rr, dr, stp, dtd);
                if (Formt(st) != 0) st.Refresh();
            }
        }

        int warn;
        if (task == 4) warn = 0;
        else if (sf.Nfev > o.MaxFun || nIter >= maxiter) warn = 1;
        else warn = 2;
        string[] statusNames = ["START", "NEW_X", "RESTART", "FG", "CONVERGENCE", "STOP", "WARNING", "ERROR", "ABNORMAL"];
        var xr = new double[n]; var gr = new double[n];
        Array.Copy(x, 1, xr, 0, n); Array.Copy(g, 1, gr, 0, n);
        var active = new int[n];
        for (var i = 1; i <= n; i++)
        {
            if (st.Nbd[i] is 1 or 2 && x[i] <= st.L[i]) active[i - 1] = -1;
            else if (st.Nbd[i] is 2 or 3 && x[i] >= st.U[i]) active[i - 1] = 1;
        }
        return new OptimizeResult
        {
            X = xr, Fun = f, Jac = gr, Nfev = sf.Nfev, Njev = sf.Ngev, Nit = nIter, Status = warn, Success = warn == 0,
            Message = statusNames[task] + ": " + taskMsg, HessInv = n <= 1000 ? InverseHessian(st) : null, ActiveMask = active,
        };
    }

    private static double Eval(OptimizeScalarFunction sf, double[] x1, double[] g1, int n)
    {
        var f = sf.FunAndGrad(x1.AsSpan(1, n), g1.AsSpan(1, n));
        return f;
    }

    private static double Projgr(State st, double[] x, double[] g)
    {
        double s = 0;
        for (var i = 1; i <= st.N; i++)
        {
            var gi = g[i];
            if (st.Nbd[i] != 0)
            {
                if (gi < 0) { if (st.Nbd[i] >= 2) gi = Math.Max(x[i] - st.U[i], gi); }
                else if (st.Nbd[i] <= 2) gi = Math.Min(x[i] - st.L[i], gi);
            }
            s = Math.Max(s, Math.Abs(gi));
        }
        return s;
    }

    // ---- LINPACK helpers on 1-based column-major storage ----

    /// <summary>dpofa on the k×k block at (off+1, off+1): upper Cholesky R with RᵀR = A. Returns 0 or the failing column.</summary>
    private static int Dpofa(FMat a, int off, int k)
    {
        for (var j = 1; j <= k; j++)
        {
            double s = 0;
            for (var kk = 1; kk <= j - 1; kk++)
            {
                var tv = a[off + kk, off + j];
                for (var i = 1; i <= kk - 1; i++) tv -= a[off + i, off + kk] * a[off + i, off + j];
                tv /= a[off + kk, off + kk];
                a[off + kk, off + j] = tv;
                s += tv * tv;
            }
            s = a[off + j, off + j] - s;
            if (s <= 0) return j;
            a[off + j, off + j] = Math.Sqrt(s);
        }
        return 0;
    }

    /// <summary>dtrsl with an upper-triangular T (k×k at the origin): job 01 solves T x = b, job 11 solves Tᵀ x = b.</summary>
    private static int Dtrsl(FMat tm, int k, double[] b, int bOff, bool transpose)
    {
        for (var i = 1; i <= k; i++) if (tm[i, i] == 0) return i;
        if (!transpose)
        {
            for (var i = k; i >= 1; i--)
            {
                var s = b[bOff + i];
                for (var j = i + 1; j <= k; j++) s -= tm[i, j] * b[bOff + j];
                b[bOff + i] = s / tm[i, i];
            }
        }
        else
        {
            for (var i = 1; i <= k; i++)
            {
                var s = b[bOff + i];
                for (var j = 1; j <= i - 1; j++) s -= tm[j, i] * b[bOff + j];
                b[bOff + i] = s / tm[i, i];
            }
        }
        return 0;
    }

    /// <summary>bmv: p = M v for the 2col×2col middle matrix; v at vOff, p at pOff (both 1-based offsets into wa-like arrays).</summary>
    private static int Bmv(State st, double[] v, int vOff, double[] p, int pOff)
    {
        var col = st.Col;
        if (col == 0) return 0;
        var sy = st.Sy;
        p[pOff + col + 1] = v[vOff + col + 1];
        for (var i = 2; i <= col; i++)
        {
            var i2 = col + i;
            double sum = 0;
            for (var k = 1; k <= i - 1; k++) sum += sy[i, k] * v[vOff + k] / sy[k, k];
            p[pOff + i2] = v[vOff + i2] + sum;
        }
        var info = Dtrsl(st.Wt, col, p, pOff + col, true);
        if (info != 0) return info;
        for (var i = 1; i <= col; i++) p[pOff + i] = v[vOff + i] / Math.Sqrt(sy[i, i]);
        info = Dtrsl(st.Wt, col, p, pOff + col, false);
        if (info != 0) return info;
        for (var i = 1; i <= col; i++) p[pOff + i] = -p[pOff + i] / Math.Sqrt(sy[i, i]);
        for (var i = 1; i <= col; i++)
        {
            double sum = 0;
            for (var k = i + 1; k <= col; k++) sum += sy[k, i] * p[pOff + col + k] / sy[i, i];
            p[pOff + i] += sum;
        }
        return 0;
    }

    /// <summary>Generalised Cauchy point. wa layout as Fortran: p = wa(1..2m), c = wa(2m+1..), wbp = wa(4m+1..), v = wa(6m+1..).</summary>
    private static int Cauchy(State st, double[] x, double[] g, int[] iorder, int[] iwhere, double[] t, double[] d, double[] xcp, double[] wa, double sbgnrm)
    {
        int n = st.N, m = st.M, col = st.Col;
        int pO = 0, cO = 2 * m, wbpO = 4 * m, vO = 6 * m;
        var l = st.L; var u = st.U; var nbd = st.Nbd;
        if (sbgnrm <= 0) { Array.Copy(x, xcp, n + 1); return 0; }
        var bnded = true;
        var nfree = n + 1;
        var nbreak = 0; var ibkmin = 0;
        double bkmin = 0;
        var col2 = 2 * col;
        double f1 = 0;
        for (var i = 1; i <= col2; i++) wa[pO + i] = 0;
        double tl = 0, tu = 0;
        for (var i = 1; i <= n; i++)
        {
            var neggi = -g[i];
            if (iwhere[i] != 3 && iwhere[i] != -1)
            {
                if (nbd[i] <= 2) tl = x[i] - l[i];
                if (nbd[i] >= 2) tu = u[i] - x[i];
                var xlower = nbd[i] <= 2 && tl <= 0;
                var xupper = nbd[i] >= 2 && tu <= 0;
                iwhere[i] = 0;
                if (xlower) { if (neggi <= 0) iwhere[i] = 1; }
                else if (xupper) { if (neggi >= 0) iwhere[i] = 2; }
                else if (Math.Abs(neggi) <= 0) iwhere[i] = -3;
            }
            var pointr = st.Head;
            if (iwhere[i] != 0 && iwhere[i] != -1) d[i] = 0;
            else
            {
                d[i] = neggi;
                f1 -= neggi * neggi;
                for (var j = 1; j <= col; j++)
                {
                    wa[pO + j] += st.Wy[i, pointr] * neggi;
                    wa[pO + col + j] += st.Ws[i, pointr] * neggi;
                    pointr = pointr % m + 1;
                }
                if (nbd[i] <= 2 && nbd[i] != 0 && neggi < 0)
                {
                    nbreak++; iorder[nbreak] = i; t[nbreak] = tl / (-neggi);
                    if (nbreak == 1 || t[nbreak] < bkmin) { bkmin = t[nbreak]; ibkmin = nbreak; }
                }
                else if (nbd[i] >= 2 && neggi > 0)
                {
                    nbreak++; iorder[nbreak] = i; t[nbreak] = tu / neggi;
                    if (nbreak == 1 || t[nbreak] < bkmin) { bkmin = t[nbreak]; ibkmin = nbreak; }
                }
                else
                {
                    nfree--; iorder[nfree] = i;
                    if (Math.Abs(neggi) > 0) bnded = false;
                }
            }
        }
        if (st.Theta != 1) for (var j = 1; j <= col; j++) wa[pO + col + j] *= st.Theta;
        Array.Copy(x, xcp, n + 1);
        if (nbreak == 0 && nfree == n + 1) return 0;
        for (var j = 1; j <= col2; j++) wa[cO + j] = 0;
        var f2 = -st.Theta * f1;
        var f2Org = f2;
        if (col > 0)
        {
            var info = Bmv(st, wa, pO, wa, vO);
            if (info != 0) return info;
            double s = 0;
            for (var j = 1; j <= col2; j++) s += wa[vO + j] * wa[pO + j];
            f2 -= s;
        }
        var dtm = -f1 / f2;
        double tsum = 0;
        var gotoEnd = false;
        if (nbreak > 0)
        {
            var nleft = nbreak;
            var iter = 1;
            double tj = 0;
            while (true)
            {
                var tj0 = tj;
                int ibp;
                if (iter == 1) { tj = bkmin; ibp = iorder[ibkmin]; }
                else
                {
                    if (iter == 2 && ibkmin != nbreak) { t[ibkmin] = t[nbreak]; iorder[ibkmin] = iorder[nbreak]; }
                    Hpsolb(nleft, t, iorder, iter - 2);
                    tj = t[nleft];
                    ibp = iorder[nleft];
                }
                var dt = tj - tj0;
                if (dtm < dt) break;
                tsum += dt;
                nleft--;
                iter++;
                var dibp = d[ibp];
                d[ibp] = 0;
                double zibp;
                if (dibp > 0) { zibp = u[ibp] - x[ibp]; xcp[ibp] = u[ibp]; iwhere[ibp] = 2; }
                else { zibp = l[ibp] - x[ibp]; xcp[ibp] = l[ibp]; iwhere[ibp] = 1; }
                if (nleft == 0 && nbreak == n) { dtm = dt; gotoEnd = true; break; }
                var dibp2 = dibp * dibp;
                f1 = f1 + dt * f2 + dibp2 - st.Theta * dibp * zibp;
                f2 -= st.Theta * dibp2;
                if (col > 0)
                {
                    for (var j = 1; j <= col2; j++) wa[cO + j] += dt * wa[pO + j];
                    var pointr = st.Head;
                    for (var j = 1; j <= col; j++)
                    {
                        wa[wbpO + j] = st.Wy[ibp, pointr];
                        wa[wbpO + col + j] = st.Theta * st.Ws[ibp, pointr];
                        pointr = pointr % m + 1;
                    }
                    var info = Bmv(st, wa, wbpO, wa, vO);
                    if (info != 0) return info;
                    double wmc = 0, wmp = 0, wmw = 0;
                    for (var j = 1; j <= col2; j++)
                    {
                        wmc += wa[cO + j] * wa[vO + j];
                        wmp += wa[pO + j] * wa[vO + j];
                        wmw += wa[wbpO + j] * wa[vO + j];
                    }
                    for (var j = 1; j <= col2; j++) wa[pO + j] -= dibp * wa[wbpO + j];
                    f1 += dibp * wmc;
                    f2 += 2.0 * dibp * wmp - dibp2 * wmw;
                }
                f2 = Math.Max(Epsmch * f2Org, f2);
                if (nleft > 0) { dtm = -f1 / f2; continue; }
                if (bnded) { f1 = 0; f2 = 0; dtm = 0; }
                else dtm = -f1 / f2;
                break;
            }
        }
        if (!gotoEnd)
        {
            if (dtm <= 0) dtm = 0;
            tsum += dtm;
            for (var i = 1; i <= n; i++) xcp[i] += tsum * d[i];
        }
        if (col > 0) for (var j = 1; j <= col2; j++) wa[cO + j] += dtm * wa[pO + j];
        return 0;
    }

    private static void Hpsolb(int n, double[] t, int[] iorder, int iheap)
    {
        if (iheap == 0)
        {
            for (var k = 2; k <= n; k++)
            {
                var ddum = t[k]; var indxin = iorder[k];
                var i = k;
                while (i > 1)
                {
                    var j = i / 2;
                    if (ddum < t[j]) { t[i] = t[j]; iorder[i] = iorder[j]; i = j; }
                    else break;
                }
                t[i] = ddum; iorder[i] = indxin;
            }
        }
        if (n > 1)
        {
            var i = 1;
            var outv = t[1]; var indxou = iorder[1];
            var ddum = t[n]; var indxin = iorder[n];
            while (true)
            {
                var j = i + i;
                if (j <= n - 1)
                {
                    if (t[j + 1] < t[j]) j++;
                    if (t[j] < ddum) { t[i] = t[j]; iorder[i] = iorder[j]; i = j; continue; }
                }
                break;
            }
            t[i] = ddum; iorder[i] = indxin;
            t[n] = outv; iorder[n] = indxou;
        }
    }

    private static int Formk(State st, int nsub, int[] ind, int[] indx2)
    {
        int n = st.N, m = st.M, col = st.Col, head = st.Head;
        var wn1 = st.Snd; var wn = st.Wn; var ws = st.Ws; var wy = st.Wy;
        int nenter = st.Nenter, ileave = st.Ileave;
        int upcl;
        if (st.Updatd)
        {
            if (st.Iupdat > m)
            {
                for (var jy = 1; jy <= m - 1; jy++)
                {
                    var js = m + jy;
                    for (var k = 0; k < m - jy; k++) wn1[jy + k, jy] = wn1[jy + 1 + k, jy + 1];
                    for (var k = 0; k < m - jy; k++) wn1[js + k, js] = wn1[js + 1 + k, js + 1];
                    for (var k = 0; k < m - 1; k++) wn1[m + 1 + k, jy] = wn1[m + 2 + k, jy + 1];
                }
            }
            int pbegin = 1, pend = nsub, dbegin = nsub + 1, dend = n;
            var iy = col; var isr = m + col;
            var ipntr = head + col - 1; if (ipntr > m) ipntr -= m;
            var jpntr = head;
            for (var jy = 1; jy <= col; jy++)
            {
                var js = m + jy;
                double temp1 = 0, temp2 = 0, temp3 = 0;
                for (var k = pbegin; k <= pend; k++) { var k1 = ind[k]; temp1 += wy[k1, ipntr] * wy[k1, jpntr]; }
                for (var k = dbegin; k <= dend; k++) { var k1 = ind[k]; temp2 += ws[k1, ipntr] * ws[k1, jpntr]; temp3 += ws[k1, ipntr] * wy[k1, jpntr]; }
                wn1[iy, jy] = temp1;
                wn1[isr, js] = temp2;
                wn1[isr, jy] = temp3;
                jpntr = jpntr % m + 1;
            }
            {
                var jy = col;
                jpntr = head + col - 1; if (jpntr > m) jpntr -= m;
                ipntr = head;
                for (var i = 1; i <= col; i++)
                {
                    var isi = m + i;
                    double temp3 = 0;
                    for (var k = pbegin; k <= pend; k++) { var k1 = ind[k]; temp3 += ws[k1, ipntr] * wy[k1, jpntr]; }
                    ipntr = ipntr % m + 1;
                    wn1[isi, jy] = temp3;
                }
            }
            upcl = col - 1;
        }
        else upcl = col;

        {
            var ipntr = head;
            for (var iy = 1; iy <= upcl; iy++)
            {
                var isr = m + iy;
                var jpntr = head;
                for (var jy = 1; jy <= iy; jy++)
                {
                    var js = m + jy;
                    double temp1 = 0, temp2 = 0, temp3 = 0, temp4 = 0;
                    for (var k = 1; k <= nenter; k++) { var k1 = indx2[k]; temp1 += wy[k1, ipntr] * wy[k1, jpntr]; temp2 += ws[k1, ipntr] * ws[k1, jpntr]; }
                    for (var k = ileave; k <= n; k++) { var k1 = indx2[k]; temp3 += wy[k1, ipntr] * wy[k1, jpntr]; temp4 += ws[k1, ipntr] * ws[k1, jpntr]; }
                    wn1[iy, jy] += temp1 - temp3;
                    wn1[isr, js] += -temp2 + temp4;
                    jpntr = jpntr % m + 1;
                }
                ipntr = ipntr % m + 1;
            }
            ipntr = head;
            for (var isr = m + 1; isr <= m + upcl; isr++)
            {
                var jpntr = head;
                for (var jy = 1; jy <= upcl; jy++)
                {
                    double temp1 = 0, temp3 = 0;
                    for (var k = 1; k <= nenter; k++) { var k1 = indx2[k]; temp1 += ws[k1, ipntr] * wy[k1, jpntr]; }
                    for (var k = ileave; k <= n; k++) { var k1 = indx2[k]; temp3 += ws[k1, ipntr] * wy[k1, jpntr]; }
                    if (isr <= jy + m) wn1[isr, jy] += temp1 - temp3;
                    else wn1[isr, jy] += -temp1 + temp3;
                    jpntr = jpntr % m + 1;
                }
                ipntr = ipntr % m + 1;
            }
        }

        for (var iy = 1; iy <= col; iy++)
        {
            var isr = col + iy;
            var is1 = m + iy;
            for (var jy = 1; jy <= iy; jy++)
            {
                var js = col + jy;
                var js1 = m + jy;
                wn[jy, iy] = wn1[iy, jy] / st.Theta;
                wn[js, isr] = wn1[is1, js1] * st.Theta;
            }
            for (var jy = 1; jy <= iy - 1; jy++) wn[jy, isr] = -wn1[is1, jy];
            for (var jy = iy; jy <= col; jy++) wn[jy, isr] = wn1[is1, jy];
            wn[iy, iy] += st.Sy[iy, iy];
        }
        if (Dpofa(wn, 0, col) != 0) return -1;
        var col2 = 2 * col;
        var colBuf = new double[col + 1];
        for (var js = col + 1; js <= col2; js++)
        {
            for (var i = 1; i <= col; i++) colBuf[i] = wn[i, js];
            Dtrsl(wn, col, colBuf, 0, true);
            for (var i = 1; i <= col; i++) wn[i, js] = colBuf[i];
        }
        for (var isr = col + 1; isr <= col2; isr++)
            for (var js = isr; js <= col2; js++)
            {
                double s = 0;
                for (var k = 1; k <= col; k++) s += wn[k, isr] * wn[k, js];
                wn[isr, js] += s;
            }
        if (Dpofa(wn, col, col) != 0) return -2;
        return 0;
    }

    private static int Cmprlb(State st, double[] x, double[] g, double[] z, double[] r, double[] wa, int[] index, int nfree, bool cnstnd)
    {
        int n = st.N, m = st.M, col = st.Col;
        if (!cnstnd && col > 0)
        {
            for (var i = 1; i <= n; i++) r[i] = -g[i];
            return 0;
        }
        for (var i = 1; i <= nfree; i++) { var k = index[i]; r[i] = -st.Theta * (z[k] - x[k]) - g[k]; }
        if (Bmv(st, wa, 2 * m, wa, 0) != 0) return -8;
        var pointr = st.Head;
        for (var j = 1; j <= col; j++)
        {
            var a1 = wa[j]; var a2 = st.Theta * wa[col + j];
            for (var i = 1; i <= nfree; i++) { var k = index[i]; r[i] += st.Wy[k, pointr] * a1 + st.Ws[k, pointr] * a2; }
            pointr = pointr % m + 1;
        }
        return 0;
    }

    private static int Subsm(State st, int nsub, int[] ind, double[] x, double[] d, double[] xp, double[] xx, double[] gg, double[] wv)
    {
        int n = st.N, m = st.M, col = st.Col;
        var l = st.L; var u = st.U; var nbd = st.Nbd;
        if (nsub <= 0) return 0;
        var pointr = st.Head;
        for (var i = 1; i <= col; i++)
        {
            double temp1 = 0, temp2 = 0;
            for (var j = 1; j <= nsub; j++) { var k = ind[j]; temp1 += st.Wy[k, pointr] * d[j]; temp2 += st.Ws[k, pointr] * d[j]; }
            wv[i] = temp1;
            wv[col + i] = st.Theta * temp2;
            pointr = pointr % m + 1;
        }
        var col2 = 2 * col;
        if (Dtrsl(st.Wn, col2, wv, 0, true) != 0) return 1;
        for (var i = 1; i <= col; i++) wv[i] = -wv[i];
        if (Dtrsl(st.Wn, col2, wv, 0, false) != 0) return 1;
        pointr = st.Head;
        for (var jy = 1; jy <= col; jy++)
        {
            var js = col + jy;
            for (var i = 1; i <= nsub; i++) { var k = ind[i]; d[i] += st.Wy[k, pointr] * wv[jy] / st.Theta + st.Ws[k, pointr] * wv[js]; }
            pointr = pointr % m + 1;
        }
        for (var i = 1; i <= nsub; i++) d[i] /= st.Theta;

        // Projection of the Newton step (3.0): accept it if it is still a descent direction.
        var iword = 0;
        Array.Copy(x, xp, n + 1);
        for (var i = 1; i <= nsub; i++)
        {
            var k = ind[i]; var dk = d[i]; var xk = x[k];
            if (nbd[k] != 0)
            {
                if (nbd[k] == 1) { x[k] = Math.Max(l[k], xk + dk); if (x[k] == l[k]) iword = 1; }
                else if (nbd[k] == 2) { xk = Math.Max(l[k], xk + dk); x[k] = Math.Min(u[k], xk); if (x[k] == l[k] || x[k] == u[k]) iword = 1; }
                else if (nbd[k] == 3) { x[k] = Math.Min(u[k], xk + dk); if (x[k] == u[k]) iword = 1; }
            }
            else x[k] = xk + dk;
        }
        if (iword == 0) return 0;
        double ddp = 0;
        for (var i = 1; i <= n; i++) ddp += (x[i] - xx[i]) * gg[i];
        if (ddp <= 0) return 0;
        Array.Copy(xp, x, n + 1);

        // Otherwise backtrack along the unprojected direction to the first bound.
        double alpha = 1, temp1v = alpha;
        var ibd = 0;
        for (var i = 1; i <= nsub; i++)
        {
            var k = ind[i]; var dk = d[i];
            if (nbd[k] != 0)
            {
                if (dk < 0 && nbd[k] <= 2)
                {
                    var temp2 = l[k] - x[k];
                    if (temp2 >= 0) temp1v = 0; else if (dk * alpha < temp2) temp1v = temp2 / dk;
                }
                else if (dk > 0 && nbd[k] >= 2)
                {
                    var temp2 = u[k] - x[k];
                    if (temp2 <= 0) temp1v = 0; else if (dk * alpha > temp2) temp1v = temp2 / dk;
                }
                if (temp1v < alpha) { alpha = temp1v; ibd = i; }
            }
        }
        if (alpha < 1)
        {
            var dk = d[ibd]; var k = ind[ibd];
            if (dk > 0) { x[k] = u[k]; d[ibd] = 0; }
            else if (dk < 0) { x[k] = l[k]; d[ibd] = 0; }
        }
        for (var i = 1; i <= nsub; i++) { var k = ind[i]; x[k] += alpha * d[i]; }
        return 0;
    }

    private static void Matupd(State st, double[] d, double[] r, double rr, double dr, double stp, double dtd)
    {
        int n = st.N, m = st.M;
        if (st.Iupdat <= m) { st.Col = st.Iupdat; st.Itail = (st.Head + st.Iupdat - 2) % m + 1; }
        else { st.Itail = st.Itail % m + 1; st.Head = st.Head % m + 1; }
        for (var i = 1; i <= n; i++) { st.Ws[i, st.Itail] = d[i]; st.Wy[i, st.Itail] = r[i]; }
        st.Theta = rr / dr;
        var col = st.Col;
        if (st.Iupdat > m)
        {
            for (var j = 1; j <= col - 1; j++)
            {
                for (var k = 0; k < j; k++) st.Ss[1 + k, j] = st.Ss[2 + k, j + 1];
                for (var k = 0; k < col - j; k++) st.Sy[j + k, j] = st.Sy[j + 1 + k, j + 1];
            }
        }
        var pointr = st.Head;
        for (var j = 1; j <= col - 1; j++)
        {
            double a = 0, b = 0;
            for (var i = 1; i <= n; i++) { a += d[i] * st.Wy[i, pointr]; b += st.Ws[i, pointr] * d[i]; }
            st.Sy[col, j] = a;
            st.Ss[j, col] = b;
            pointr = pointr % m + 1;
        }
        st.Ss[col, col] = stp == 1 ? dtd : stp * stp * dtd;
        st.Sy[col, col] = dr;
    }

    private static int Formt(State st)
    {
        var col = st.Col;
        var wt = st.Wt; var sy = st.Sy; var ss = st.Ss;
        for (var j = 1; j <= col; j++) wt[1, j] = st.Theta * ss[1, j];
        for (var i = 2; i <= col; i++)
            for (var j = i; j <= col; j++)
            {
                var k1 = Math.Min(i, j) - 1;
                double ddum = 0;
                for (var k = 1; k <= k1; k++) ddum += sy[i, k] * sy[j, k] / sy[k, k];
                wt[i, j] = ddum + st.Theta * ss[i, j];
            }
        return Dpofa(wt, 0, col) != 0 ? -3 : 0;
    }

    /// <summary>Dense inverse Hessian approximation from the stored pairs (two-loop recursion on unit vectors, H₀ = I).</summary>
    private static NdArray InverseHessian(State st)
    {
        int n = st.N, m = st.M;
        var col = st.Col;
        var s = new double[col][]; var y = new double[col][]; var rho = new double[col];
        var pointr = st.Head;
        for (var j = 0; j < col; j++)
        {
            s[j] = new double[n]; y[j] = new double[n];
            double sy = 0;
            for (var i = 0; i < n; i++) { s[j][i] = st.Ws[i + 1, pointr]; y[j][i] = st.Wy[i + 1, pointr]; sy += s[j][i] * y[j][i]; }
            rho[j] = 1 / sy;
            pointr = pointr % m + 1;
        }
        var h = new double[n * n];
        var alpha = new double[col];
        var q = new double[n];
        for (var c = 0; c < n; c++)
        {
            Array.Clear(q);
            q[c] = 1;
            for (var j = col - 1; j >= 0; j--) { alpha[j] = rho[j] * OptimizeDense.Dot(s[j], q); OptimizeDense.Axpy(-alpha[j], y[j], q); }
            for (var j = 0; j < col; j++) { var beta = rho[j] * OptimizeDense.Dot(y[j], q); OptimizeDense.Axpy(alpha[j] - beta, s[j], q); }
            for (var i = 0; i < n; i++) h[i * n + c] = q[i];
        }
        return new NdArray(h, n, n);
    }
}
