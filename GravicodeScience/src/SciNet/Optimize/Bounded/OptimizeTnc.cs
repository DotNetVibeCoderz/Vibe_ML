namespace Gravicode.Science.SciNet;

/// <summary>Options for <see cref="MinimizeMethod.Tnc"/>; names and defaults follow SciPy's <c>_minimize_tnc</c>.</summary>
public sealed record TncOptions : MinimizeOptions
{
    /// <summary>Absolute forward-difference step when no gradient is given. Default 1e-8.</summary>
    public double Eps { get; init; } = 1e-8;
    /// <summary>Variable scale factors (absolute values are used; a zero fixes the variable at x0); default ub − lb for doubly bounded variables, 1 + |x0| otherwise.</summary>
    public double[]? Scale { get; init; }
    /// <summary>Variable offsets; default (ub + lb)/2 for doubly bounded variables, x0 otherwise.</summary>
    public double[]? Offset { get; init; }
    /// <summary>Maximum CG iterations per Newton step; default max(1, min(50, n/2)); 0 means steepest descent. Capped at n.</summary>
    public int? MaxCGIt { get; init; }
    /// <summary>Maximum function evaluations (tnc.c's count, which includes the Hessian-vector products but not finite-difference gradient calls); default max(100, 10 n).</summary>
    public int? MaxFun { get; init; }
    /// <summary>Line-search severity: the Gill–Murray search stops when |g·p| ≤ eta·|g₀·p|. Outside [0, 1) means 0.25.</summary>
    public double Eta { get; init; } = 0.25;
    /// <summary>Maximum step of the line search in scaled variables (raised ×100 whenever a step reaches it). Below 10√ε means 10.</summary>
    public double StepMx { get; init; } = 10;
    /// <summary>Relative precision of the Hessian-vector finite differences; ≤ ε means √ε.</summary>
    public double Accuracy { get; init; }
    /// <summary>Estimate of the minimum value; only used to size the first trial step of each line search. Default 0 (SciPy's <c>minfev</c>).</summary>
    public double Fmin { get; init; }
    /// <summary>Stop when |f_k − f_(k−1)| ≤ ftol (in the internally rescaled f); negative means <see cref="Accuracy"/>.</summary>
    public double? Ftol { get; init; }
    /// <summary>Stop when the scaled step length is ≤ xtol; negative means √ε.</summary>
    public double? Xtol { get; init; }
    /// <summary>Scaled projected-gradient norm at which to stop; negative means 1e-2·√accuracy.</summary>
    public double? Gtol { get; init; }
    /// <summary>log10 threshold for rescaling f: when |log10 ‖pg‖| exceeds it, f is renormalised. Negative means 1.3.</summary>
    public double Rescale { get; init; } = -1;
}

public static partial class Optimize
{
    static partial void RunTnc(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeTnc.Run(problem);
}

/// <summary>
/// A line-for-line port of Jean-Sebastien Roy's <c>tnc.c</c> (TNC 1.3 as vendored by SciPy), itself
/// a C translation of Stephen Nash's TNBC: a truncated-Newton method whose Newton equations are
/// solved by preconditioned conjugate gradients (a two-step self-scaled BFGS preconditioner with a
/// diagonal BFGS scaling, refreshed by every CG iteration), Hessian-vector products by gradient
/// differences, an active-set treatment of the bounds driven by Lagrange-multiplier signs, and the
/// Gill–Murray safeguarded cubic line search (<c>getptc</c>).
/// </summary>
/// <remarks>
/// Iteration and evaluation counts match SciPy's on its own test problems: the order of floating
/// operations follows the C source, including its overflow-safe norm (<c>dnrm21</c>), because a
/// truncated-CG method amplifies a last-bit difference into a different step count.
/// </remarks>
internal static class OptimizeTnc
{
    private const double Eps = 2.220446049250313e-16;

    private static readonly string[] Messages =
    [
        "Local minimum reached (|pg| ~= 0)", "Converged (|f_n-f_(n-1)| ~= 0)", "Converged (|x_n-x_(n-1)| ~= 0)",
        "Max. number of function evaluations reached", "Linear search failed", "All lower bounds are equal to the upper bounds",
        "Unable to progress", "User requested end of minimization",
    ];

    // tnc_rc values (TNC_MINRC = -3).
    private const int LocalMinimum = 0, FConverged = 1, XConverged = 2, MaxFun = 3, LsFail = 4, Constant = 5, NoProgress = 6, UserAbort = 7;

    private enum LsRc { Ok, MaxFun, Fail, UserAbort }

    public static OptimizeResult Run(MinimizeProblem p)
    {
        var o = p.OptionsAs<TncOptions>("TNC");
        var n = p.N;
        var sf = p.CreateObjective(o.Eps);
        var low = (double[])(p.Bounds?.Lb ?? Enumerable.Repeat(double.NegativeInfinity, n).ToArray()).Clone();
        var up = (double[])(p.Bounds?.Ub ?? Enumerable.Repeat(double.PositiveInfinity, n).ToArray()).Clone();
        if (o.Scale is not null && o.Scale.Length != n) throw new ArgumentException("tnc: scale must have one entry per variable.");
        if (o.Offset is not null && o.Offset.Length != n) throw new ArgumentException("tnc: offset must have one entry per variable.");
        var maxfun = o.MaxFun ?? Math.Max(100, 10 * n);
        var x = (double[])p.X0.Clone();
        var g = new double[n];
        var state = new Tnc(n, sf, p, low, up);
        var rc = state.Driver(x, g, o.Scale, o.Offset, o.MaxCGIt ?? -1, maxfun, o.Eta, o.StepMx, o.Accuracy, o.Fmin,
            o.Ftol ?? p.Tol ?? -1, o.Xtol ?? p.Tol ?? -1, o.Gtol ?? p.Tol ?? -1, o.Rescale, out var nit);

        // SciPy re-evaluates at the returned point (free when it is the last point evaluated).
        var jac = new double[n];
        var f = sf.FunAndGrad(x, jac);
        var active = new int[n];
        for (var i = 0; i < n; i++) active[i] = x[i] <= low[i] ? -1 : x[i] >= up[i] ? 1 : 0;
        return new OptimizeResult
        {
            X = x, Fun = f, Jac = jac, Nit = nit, Nfev = sf.Nfev, Njev = sf.Ngev, Status = rc,
            Success = rc is >= 0 and < 3, Message = Messages[rc], ActiveMask = active,
        };
    }

    private sealed class Tnc(int n, OptimizeScalarFunction sf, MinimizeProblem problem, double[] low, double[] up)
    {
        private int _nfeval;
        private double[] _xscale = [], _xoffset = [];
        private readonly double[] _xbuf = new double[n];

        /// <summary>The user function at an unscaled point; true means abort was requested.</summary>
        private double Function(double[] xUnscaled, double[] grad)
        {
            Array.Copy(xUnscaled, _xbuf, n);
            return sf.FunAndGrad(_xbuf, grad);
        }

        public int Driver(double[] x, double[] g, double[]? scale, double[]? offset, int maxCGit, int maxnfeval,
            double eta, double stepmx, double accuracy, double fmin, double ftol, double xtol, double pgtol, double rescale, out int niter)
        {
            niter = 0;
            Coercex(x);
            if (maxnfeval < 1) return MaxFun;
            var f = Function(x, g);
            _nfeval++;

            var nc = 0;
            for (var i = 0; i < n; i++) if (low[i] == up[i] || (scale is not null && scale[i] == 0.0)) nc++;
            if (nc == n) return Constant;

            _xscale = new double[n]; _xoffset = new double[n];
            for (var i = 0; i < n; i++)
            {
                if (scale is not null)
                {
                    _xscale[i] = Math.Abs(scale[i]);
                    if (_xscale[i] == 0.0) _xoffset[i] = low[i] = up[i] = x[i];
                }
                else if (low[i] != double.NegativeInfinity && up[i] != double.PositiveInfinity)
                {
                    _xscale[i] = up[i] - low[i];
                    _xoffset[i] = (up[i] + low[i]) * 0.5;
                }
                else
                {
                    _xscale[i] = 1.0 + Math.Abs(x[i]);
                    _xoffset[i] = x[i];
                }
                if (offset is not null) _xoffset[i] = offset[i];
            }

            var rteps = Math.Sqrt(Eps);
            if (stepmx < rteps * 10.0) stepmx = 1.0e1;
            if (eta < 0.0 || eta >= 1.0) eta = 0.25;
            if (rescale < 0) rescale = 1.3;
            if (maxCGit < 0) maxCGit = Math.Clamp(n / 2, 1, 50);
            if (maxCGit > n) maxCGit = n;
            if (accuracy <= Eps) accuracy = rteps;
            if (ftol < 0.0) ftol = accuracy;
            if (pgtol < 0.0) pgtol = 1e-2 * Math.Sqrt(accuracy);
            if (xtol < 0.0) xtol = rteps;

            return Minimize(x, ref f, g, maxCGit, maxnfeval, eta, stepmx, accuracy, fmin, ftol, xtol, pgtol, rescale, out niter);
        }

        private int Minimize(double[] x, ref double f, double[] gfull, int maxCGit, int maxnfeval, double eta, double stepmx,
            double accuracy, double fmin, double ftol, double xtol, double pgtol, double rescale, out int niter)
        {
            var oldg = new double[n]; var g = new double[n]; var temp = new double[n]; var diagb = new double[n];
            var pk = new double[n]; var sk = new double[n]; var yk = new double[n]; var sr = new double[n]; var yr = new double[n];
            var pivot = new int[n];
            double alpha = 0.0, fscale = 1.0;
            niter = 0;
            int rc;

            var difnew = 0.0;
            var epsred = 0.05;
            var upd1 = true;
            var icycle = n - 1;
            var newcon = true;
            var lreset = false;
            double yrsr = 0.0, yksk = 0.0;

            Scalex(x);
            f *= fscale;
            SetConstraints(x, pivot);
            Array.Copy(gfull, g, n);
            Scaleg(g, fscale);
            for (var i = 0; i < n; i++) if (-pivot[i] * g[i] < 0.0) pivot[i] = 0;
            Project(g, pivot);
            var gnorm = Dnrm2(g);
            var fLastConstraint = f;
            var fLastReset = f;
            for (var i = 0; i < n; i++) diagb[i] = 1.0;

            while (true)
            {
                if (Dnrm2(g) <= pgtol * fscale) { rc = LocalMinimum; break; }
                if (_nfeval >= maxnfeval) { rc = MaxFun; break; }

                var newscale = Dnrm2(g);
                if (newscale > Eps && Math.Abs(Math.Log10(newscale)) > rescale)
                {
                    newscale = 1.0 / newscale;
                    f *= newscale; fscale *= newscale; gnorm *= newscale;
                    fLastConstraint *= newscale; fLastReset *= newscale; difnew *= newscale;
                    for (var i = 0; i < n; i++) g[i] *= newscale;
                    for (var i = 0; i < n; i++) diagb[i] = 1.0;
                    upd1 = true; icycle = n - 1; newcon = true;
                }

                Array.Copy(x, temp, n);
                Project(temp, pivot);
                var xnorm = Dnrm2(temp);
                var oldnfeval = _nfeval;

                Direction(pk, diagb, x, g, maxCGit, maxnfeval, upd1, yksk, yrsr, sk, yk, sr, yr, lreset, fscale, pivot, accuracy, gnorm, xnorm);

                if (!newcon)
                {
                    if (!lreset)
                    {
                        for (var i = 0; i < n; i++) { sr[i] += sk[i]; yr[i] += yk[i]; }
                        icycle++;
                    }
                    else
                    {
                        Array.Copy(sk, sr, n); Array.Copy(yk, yr, n);
                        fLastReset = f;
                        icycle = 1;
                    }
                }

                Array.Copy(g, oldg, n);
                var oldf = f;
                var oldgtp = Dot(pk, g);
                var ustpmax = stepmx / (Dnrm2(pk) + Eps);
                var spe = StepMax(ustpmax, x, pk, pivot);

                if (spe > 0.0)
                {
                    alpha = InitialStep(f, fmin / fscale, oldgtp, spe);
                    var lsrc = LinearSearch(fscale, pivot, eta, ftol, spe, pk, x, ref f, ref alpha, gfull, maxnfeval);
                    if (lsrc == LsRc.UserAbort) { rc = UserAbort; break; }
                    if (lsrc == LsRc.Fail) { rc = LsFail; break; }
                    if (alpha >= 0.9 * ustpmax) stepmx *= 1e2;
                    if (alpha - spe >= -Eps * 10.0) newcon = true;
                    else
                    {
                        if (lsrc != LsRc.Ok) { rc = lsrc == LsRc.MaxFun ? MaxFun : LsFail; break; }
                        newcon = false;
                    }
                }
                else newcon = true;

                if (newcon)
                {
                    if (!AddConstraint(x, pk, pivot) && _nfeval == oldnfeval) { rc = NoProgress; break; }
                    fLastConstraint = f;
                }

                niter++;
                if (problem.Callback is not null)
                {
                    Array.Copy(x, temp, n);
                    Unscalex(temp);
                    if (problem.NotifyCallback(temp, f / fscale)) { rc = UserAbort; break; }
                }

                var difold = difnew;
                difnew = oldf - f;
                if (icycle == 1)
                {
                    if (difnew > difold * 2.0) epsred += epsred;
                    if (difnew < difold * 0.5) epsred *= 0.5;
                }

                Array.Copy(gfull, g, n);
                Scaleg(g, fscale);
                Array.Copy(g, temp, n);
                Project(temp, pivot);
                gnorm = Dnrm2(temp);

                var remcon = RemoveConstraint(oldgtp, gnorm, pgtol * fscale, f, fLastConstraint, g, pivot);
                if (remcon)
                {
                    Array.Copy(g, temp, n);
                    Project(temp, pivot);
                    gnorm = Dnrm2(temp);
                    fLastConstraint = f;
                }

                if (!remcon && !newcon)
                {
                    if (Math.Abs(difnew) <= ftol * fscale) { rc = FConverged; break; }
                    if (alpha * Dnrm2(pk) <= xtol) { rc = XConverged; break; }
                }

                Project(g, pivot);

                if (!newcon)
                {
                    for (var i = 0; i < n; i++) { yk[i] = g[i] - oldg[i]; sk[i] = alpha * pk[i]; }
                    yksk = Dot(yk, sk);
                    if (icycle == n - 1 || difnew < epsred * (fLastReset - f)) lreset = true;
                    else
                    {
                        yrsr = Dot(yr, sr);
                        lreset = yrsr <= 0.0;
                    }
                    upd1 = false;
                }
            }

            Unscalex(x);
            Coercex(x);
            f /= fscale;
            return rc;
        }

        // ---------------- scaling and the active set ----------------

        private void Coercex(double[] x)
        {
            for (var i = 0; i < n; i++)
            {
                if (x[i] < low[i]) x[i] = low[i];
                else if (x[i] > up[i]) x[i] = up[i];
            }
        }

        private void Unscalex(double[] x) { for (var i = 0; i < n; i++) x[i] = x[i] * _xscale[i] + _xoffset[i]; }

        private void Scalex(double[] x) { for (var i = 0; i < n; i++) if (_xscale[i] > 0.0) x[i] = (x[i] - _xoffset[i]) / _xscale[i]; }

        private void Scaleg(double[] g, double fscale) { for (var i = 0; i < n; i++) g[i] *= _xscale[i] * fscale; }

        private void SetConstraints(double[] x, int[] pivot)
        {
            for (var i = 0; i < n; i++)
            {
                if (_xscale[i] == 0.0) pivot[i] = 2;
                else if (low[i] != double.NegativeInfinity && x[i] * _xscale[i] + _xoffset[i] - low[i] <= Eps * 10.0 * (Math.Abs(low[i]) + 1.0)) pivot[i] = -1;
                else if (up[i] != double.PositiveInfinity && x[i] * _xscale[i] + _xoffset[i] - up[i] >= Eps * 10.0 * (Math.Abs(up[i]) + 1.0)) pivot[i] = 1;
                else pivot[i] = 0;
            }
        }

        private static void Project(double[] x, int[] pivot) { for (var i = 0; i < x.Length; i++) if (pivot[i] != 0) x[i] = 0.0; }

        private double StepMax(double step, double[] x, double[] dir, int[] pivot)
        {
            for (var i = 0; i < n; i++)
            {
                if (pivot[i] != 0 || dir[i] == 0.0) continue;
                if (dir[i] < 0.0)
                {
                    var t = (low[i] - _xoffset[i]) / _xscale[i] - x[i];
                    if (t > step * dir[i]) step = t / dir[i];
                }
                else
                {
                    var t = (up[i] - _xoffset[i]) / _xscale[i] - x[i];
                    if (t < step * dir[i]) step = t / dir[i];
                }
            }
            return step;
        }

        private bool AddConstraint(double[] x, double[] p, int[] pivot)
        {
            var newcon = false;
            for (var i = 0; i < n; i++)
            {
                if (pivot[i] != 0 || p[i] == 0.0) continue;
                if (p[i] < 0.0 && low[i] != double.NegativeInfinity)
                {
                    var tol = Eps * 10.0 * (Math.Abs(low[i]) + 1.0);
                    if (x[i] * _xscale[i] + _xoffset[i] - low[i] <= tol)
                    {
                        pivot[i] = -1;
                        x[i] = (low[i] - _xoffset[i]) / _xscale[i];
                        newcon = true;
                    }
                }
                else if (up[i] != double.PositiveInfinity)
                {
                    // As in tnc.c, a negative direction with an infinite lower bound falls through to
                    // the upper-bound test; it can only trigger at the upper bound itself.
                    var tol = Eps * 10.0 * (Math.Abs(up[i]) + 1.0);
                    if (up[i] - (x[i] * _xscale[i] + _xoffset[i]) <= tol)
                    {
                        pivot[i] = 1;
                        x[i] = (up[i] - _xoffset[i]) / _xscale[i];
                        newcon = true;
                    }
                }
            }
            return newcon;
        }

        private bool RemoveConstraint(double gtpnew, double gnorm, double pgtolfs, double f, double fLastConstraint, double[] g, int[] pivot)
        {
            // Gill, Murray & Wright (1981, p. 308): release a bound only once the free subspace has
            // been minimised enough (or the projected gradient is already tiny).
            if (fLastConstraint - f <= gtpnew * -0.5 && gnorm > pgtolfs) return false;
            var imax = -1;
            var cmax = 0.0;
            for (var i = 0; i < n; i++)
            {
                if (pivot[i] == 2) continue;
                var t = -pivot[i] * g[i];
                if (t < cmax) { cmax = t; imax = i; }
            }
            if (imax == -1) return false;
            pivot[imax] = 0;
            return true;
        }

        // ---------------- search direction (preconditioned truncated CG) ----------------

        private void Direction(double[] zsol, double[] diagb, double[] x, double[] g, int maxCGit, int maxnfeval,
            bool upd1, double yksk, double yrsr, double[] sk, double[] yk, double[] sr, double[] yr, bool lreset,
            double fscale, int[] pivot, double accuracy, double gnorm, double xnorm)
        {
            if (maxCGit == 0)
            {
                for (var i = 0; i < n; i++) zsol[i] = -g[i];
                Project(zsol, pivot);
                return;
            }
            var rhsnrm = gnorm;
            const double tol = 1e-12;
            var qold = 0.0;
            var rzold = 0.0;
            var r = new double[n]; var v = new double[n]; var zk = new double[n]; var emat = new double[n]; var gv = new double[n];

            InitPreconditioner(diagb, emat, lreset, yksk, yrsr, sk, yk, sr, yr, upd1);
            for (var i = 0; i < n; i++) { r[i] = -g[i]; v[i] = 0.0; zsol[i] = 0.0; }

            for (var k = 0; k < maxCGit; k++)
            {
                Project(r, pivot);
                Msolve(r, zk, sk, yk, diagb, sr, yr, upd1, yksk, yrsr, lreset);
                Project(zk, pivot);
                var rz = Dot(r, zk);
                if (rz / rhsnrm < tol || _nfeval >= maxnfeval - 1)
                {
                    if (k == 0)
                    {
                        for (var i = 0; i < n; i++) zsol[i] = -g[i];
                        Project(zsol, pivot);
                    }
                    break;
                }
                var beta = k == 0 ? 0.0 : rz / rzold;
                for (var i = 0; i < n; i++) v[i] = zk[i] + beta * v[i];
                Project(v, pivot);
                HessianTimesVector(v, gv, x, g, fscale, accuracy, xnorm);
                _nfeval++;
                Project(gv, pivot);

                var vgv = Dot(v, gv);
                if (vgv / rhsnrm < tol)
                {
                    if (k == 0)
                    {
                        Msolve(g, zsol, sk, yk, diagb, sr, yr, upd1, yksk, yrsr, lreset);
                        for (var i = 0; i < n; i++) zsol[i] = -zsol[i];
                        Project(zsol, pivot);
                    }
                    break;
                }
                DiagonalScaling(emat, v, gv, r);

                var alpha = rz / vgv;
                for (var i = 0; i < n; i++) zsol[i] += alpha * v[i];
                for (var i = 0; i < n; i++) r[i] += -alpha * gv[i];

                // Nash's quadratic-model test: stop when the relative reduction of q per iteration
                // has become small.
                var gtp = Dot(zsol, g);
                var pr = Dot(r, zsol);
                var qnew = (gtp + pr) * 0.5;
                var qtest = (k + 1) * (1.0 - qold / qnew);
                if (qtest <= 0.5) break;
                if (gtp > 0.0)
                {
                    for (var i = 0; i < n; i++) zsol[i] += -alpha * v[i];
                    break;
                }
                qold = qnew;
                rzold = rz;
            }
            Array.Copy(emat, diagb, n);
        }

        private static void DiagonalScaling(double[] e, double[] v, double[] gv, double[] r)
        {
            var vr = 1.0 / Dot(v, r);
            var vgv = 1.0 / Dot(v, gv);
            for (var i = 0; i < e.Length; i++)
            {
                e[i] += -r[i] * r[i] * vr + gv[i] * gv[i] * vgv;
                if (e[i] <= 1e-6) e[i] = 1.0;
            }
        }

        private static double InitialStep(double fnew, double fmin, double gtp, double smax)
        {
            var d = Math.Abs(fnew - fmin);
            var alpha = 1.0;
            if (d * 2.0 <= -gtp && d >= Eps) alpha = d * -2.0 / gtp;
            if (alpha >= smax) alpha = smax;
            return alpha;
        }

        private readonly double[] _xv = new double[n];

        private void HessianTimesVector(double[] v, double[] gv, double[] x, double[] g, double fscale, double accuracy, double xnorm)
        {
            var delta = accuracy * (xnorm + 1.0);
            for (var i = 0; i < n; i++) _xv[i] = x[i] + delta * v[i];
            Unscalex(_xv);
            Coercex(_xv);
            Function(_xv, gv);
            Scaleg(gv, fscale);
            var dinv = 1.0 / delta;
            for (var i = 0; i < n; i++) gv[i] = (gv[i] - g[i]) * dinv;
            for (var i = 0; i < n; i++) if (_xscale[i] == 0.0) gv[i] = 0.0;
        }

        private void Msolve(double[] g, double[] y, double[] sk, double[] yk, double[] diagb, double[] sr, double[] yr,
            bool upd1, double yksk, double yrsr, bool lreset)
        {
            if (upd1)
            {
                for (var i = 0; i < n; i++) y[i] = g[i] / diagb[i];
                return;
            }
            var gsk = Dot(g, sk);
            var hg = new double[n]; var hyr = new double[n]; var hyk = new double[n];
            if (lreset)
            {
                for (var i = 0; i < n; i++)
                {
                    var rdiagb = 1.0 / diagb[i];
                    hg[i] = g[i] * rdiagb;
                    hyk[i] = yk[i] * rdiagb;
                }
                var ykhyk = Dot(yk, hyk);
                var ghyk = Dot(g, hyk);
                Ssbfgs(1.0, sk, hg, hyk, yksk, ykhyk, gsk, ghyk, y);
            }
            else
            {
                for (var i = 0; i < n; i++)
                {
                    var rdiagb = 1.0 / diagb[i];
                    hg[i] = g[i] * rdiagb;
                    hyk[i] = yk[i] * rdiagb;
                    hyr[i] = yr[i] * rdiagb;
                }
                var gsr = Dot(g, sr);
                var ghyr = Dot(g, hyr);
                var yrhyr = Dot(yr, hyr);
                Ssbfgs(1.0, sr, hg, hyr, yrsr, yrhyr, gsr, ghyr, hg);
                var yksr = Dot(yk, sr);
                var ykhyr = Dot(yk, hyr);
                Ssbfgs(1.0, sr, hyk, hyr, yrsr, yrhyr, yksr, ykhyr, hyk);
                var ykhyk = Dot(hyk, yk);
                var ghyk = Dot(hyk, g);
                Ssbfgs(1.0, sk, hg, hyk, yksk, ykhyk, gsk, ghyk, y);
            }
        }

        private static void Ssbfgs(double gamma, double[] sj, double[] hjv, double[] hjyj, double yjsj, double yjhyj,
            double vsj, double vhyj, double[] hjp1v)
        {
            double beta, delta;
            if (yjsj == 0.0) { delta = 0.0; beta = 0.0; }
            else
            {
                delta = (gamma * yjhyj / yjsj + 1.0) * vsj / yjsj - gamma * vhyj / yjsj;
                beta = -gamma * vsj / yjsj;
            }
            for (var i = 0; i < sj.Length; i++) hjp1v[i] = gamma * hjv[i] + delta * sj[i] + beta * hjyj[i];
        }

        private void InitPreconditioner(double[] diagb, double[] emat, bool lreset, double yksk, double yrsr,
            double[] sk, double[] yk, double[] sr, double[] yr, bool upd1)
        {
            if (upd1) { Array.Copy(diagb, emat, n); return; }
            var bsk = new double[n];
            if (lreset)
            {
                for (var i = 0; i < n; i++) bsk[i] = diagb[i] * sk[i];
                var sds = Dot(sk, bsk);
                if (yksk == 0.0) yksk = 1.0;
                if (sds == 0.0) sds = 1.0;
                for (var i = 0; i < n; i++)
                {
                    var td = diagb[i];
                    emat[i] = td - td * td * sk[i] * sk[i] / sds + yk[i] * yk[i] / yksk;
                }
            }
            else
            {
                for (var i = 0; i < n; i++) bsk[i] = diagb[i] * sr[i];
                var sds = Dot(sr, bsk);
                var srds = Dot(sk, bsk);
                var yrsk = Dot(yr, sk);
                if (yrsr == 0.0) yrsr = 1.0;
                if (sds == 0.0) sds = 1.0;
                for (var i = 0; i < n; i++)
                {
                    var td = diagb[i];
                    bsk[i] = td * sk[i] - bsk[i] * srds / sds + yr[i] * yrsk / yrsr;
                    emat[i] = td - td * td * sr[i] * sr[i] / sds + yr[i] * yr[i] / yrsr;
                }
                sds = Dot(sk, bsk);
                if (yksk == 0.0) yksk = 1.0;
                if (sds == 0.0) sds = 1.0;
                // tnc.c subtracts the y_k term too (Nash's original adds it); kept as SciPy has it.
                for (var i = 0; i < n; i++) emat[i] -= bsk[i] * bsk[i] / sds + yk[i] * yk[i] / yksk;
            }
        }

        // ---------------- Gill–Murray line search ----------------

        private LsRc LinearSearch(double fscale, int[] pivot, double eta, double ftol, double xbnd, double[] p, double[] x,
            ref double f, ref double alpha, double[] gfull, int maxnfeval)
        {
            const int maxlsit = 64;
            var temp = new double[n]; var tempgfull = new double[n]; var newgfull = new double[n];
            Array.Copy(gfull, temp, n);
            Scaleg(temp, fscale);
            var gu = Dot(temp, p);
            Array.Copy(x, temp, n);
            Project(temp, pivot);
            var xnorm = Dnrm2(temp);

            var rteps = Math.Sqrt(Eps);
            var pe = Dnrm2(p) + Eps;
            var ls = new GetPtc
            {
                RelTol = rteps * (xnorm + 1.0) / pe,
                AbsTol = -Eps * (1.0 + Math.Abs(f)) / (gu - Eps),
                TnyTol = Eps * (xnorm + 1.0) / pe,
                FPresn = ftol, Xbnd = xbnd,
                U = alpha, Fu = f, Gu = gu, Fmin = f,
            };
            var itcnt = 0;
            var itest = ls.Init(eta, 1e-4);
            alpha = ls.Xmin;
            while (itest == GetPtc.Rc.Eval)
            {
                if (++itcnt > maxlsit || _nfeval >= maxnfeval) break;
                var ualpha = ls.Xmin + ls.U;
                for (var i = 0; i < n; i++) temp[i] = x[i] + ualpha * p[i];
                Unscalex(temp);
                Coercex(temp);
                var fu = Function(temp, tempgfull);
                _nfeval++;
                ls.Fu = fu * fscale;
                Array.Copy(tempgfull, temp, n);
                Scaleg(temp, fscale);
                ls.Gu = Dot(temp, p);
                itest = ls.Iter();
                if (ls.Xmin == ualpha) Array.Copy(tempgfull, newgfull, n);
            }
            alpha = ls.Xmin;

            if (itest == GetPtc.Rc.Ok)
            {
                f = ls.Fmin;
                for (var i = 0; i < n; i++) x[i] += alpha * p[i];
                Array.Copy(newgfull, gfull, n);
                return LsRc.Ok;
            }
            if (itcnt > maxlsit) return LsRc.Fail;
            if (itest != GetPtc.Rc.Eval) return LsRc.Fail;
            return LsRc.MaxFun;
        }

        /// <summary>The state of Gill and Murray's <c>getptc</c> safeguarded cubic step-length search.</summary>
        private sealed class GetPtc
        {
            public enum Rc { Ok, Eval, EInval, Fail }

            private const double RtSmll = Eps;
            private const double Big = 1.0 / (Eps * Eps);
            public double RelTol, AbsTol, TnyTol, FPresn, Xbnd;
            public double U, Fu, Gu, Xmin, Fmin, Gmin;
            private double _xw, _fw, _gw, _a, _b, _oldf, _b1, _scxbnd, _e, _step, _factor, _gtest1, _gtest2, _tol;
            private bool _braktd;

            public Rc Init(double eta, double rmu)
            {
                if (U <= 0.0 || Xbnd <= TnyTol || Gu > 0.0) return Rc.EInval;
                if (Xbnd < AbsTol) AbsTol = Xbnd;
                _tol = AbsTol;
                _a = 0.0; _xw = 0.0; Xmin = 0.0;
                _oldf = Fu; Fmin = Fu; _fw = Fu; _gw = Gu; Gmin = Gu;
                _step = U; _factor = 5.0;
                _braktd = false;
                _scxbnd = Xbnd;
                _b = _scxbnd + RelTol * Math.Abs(_scxbnd) + AbsTol;
                _e = _b + _b;
                _b1 = _b;
                _gtest1 = -rmu * Gu;
                _gtest2 = -eta * Gu;
                if (_step >= _scxbnd)
                {
                    _step = _scxbnd;
                    _scxbnd -= (RelTol * Math.Abs(Xbnd) + AbsTol) / (1.0 + RelTol);
                }
                SetU();
                return Rc.Eval;
            }

            private void SetU()
            {
                U = _step;
                if (Math.Abs(_step) < _tol && _step < 0.0) U = -_tol;
                if (Math.Abs(_step) < _tol && _step >= 0.0) U = _tol;
            }

            public Rc Iter()
            {
                double p, q, r, s, a1, twotol, xmidpt;
                var accepted = false;
                if (Fu <= Fmin)
                {
                    var chordu = _oldf - (Xmin + U) * _gtest1;
                    if (Fu > chordu)
                    {
                        // Sufficient decrease failed: move the upper bound here and force a bisection
                        // or a linear step towards the root of f(alpha) = chord(alpha).
                        var chordm = _oldf - Xmin * _gtest1;
                        Gu = -Gmin;
                        var denom = chordm - Fmin;
                        if (Math.Abs(denom) < 1e-15)
                        {
                            denom = 1e-15;
                            if (chordm - Fmin < 0.0) denom = -denom;
                        }
                        if (Xmin != 0.0) Gu = Gmin * (chordu - Fu) / denom;
                        Fu = 0.5 * U * (Gmin + Gu) + Fmin;
                        if (Fu < Fmin) Fu = Fmin;
                    }
                    else
                    {
                        _fw = Fmin; Fmin = Fu; _gw = Gmin; Gmin = Gu;
                        Xmin += U; _a -= U; _b -= U; _xw = -U; _scxbnd -= U;
                        if (Gu <= 0.0) _a = 0.0;
                        else { _b = 0.0; _braktd = true; }
                        _tol = Math.Abs(Xmin) * RelTol + AbsTol;
                        accepted = true;
                    }
                }
                if (!accepted)
                {
                    if (U < 0.0) _a = U;
                    else { _b = U; _braktd = true; }
                    _xw = U; _fw = Fu; _gw = Gu;
                }

                twotol = _tol + _tol;
                xmidpt = 0.5 * (_a + _b);
                var convrg = Math.Abs(xmidpt) <= twotol - 0.5 * (_b - _a)
                             || (Math.Abs(Gmin) <= _gtest2 && Fmin < _oldf && (Math.Abs(Xmin - Xbnd) > _tol || !_braktd));
                if (convrg)
                {
                    if (Xmin != 0.0) return Rc.Ok;
                    // No reduction yet: if f changed by more than its estimated precision, shrink tol.
                    if (Math.Abs(_oldf - _fw) <= FPresn) return Rc.Fail;
                    _tol = 0.1 * _tol;
                    if (_tol < TnyTol) return Rc.Fail;
                    RelTol = 0.1 * RelTol;
                    AbsTol = 0.1 * AbsTol;
                    twotol = 0.1 * twotol;
                }

                r = 0.0; q = 0.0; s = 0.0;
                if (Math.Abs(_e) > _tol)
                {
                    var minimumFound = false;
                    r = 3.0 * (Fmin - _fw) / _xw + Gmin + _gw;
                    var absr = Math.Abs(r);
                    q = absr;
                    if (_gw != 0.0 && Gmin != 0.0)
                    {
                        var abgw = Math.Abs(_gw);
                        var abgmin = Math.Abs(Gmin);
                        s = Math.Sqrt(abgmin) * Math.Sqrt(abgw);
                        if (_gw / abgw * Gmin > 0.0)
                        {
                            if (r >= s || r <= -s) q = Math.Sqrt(Math.Abs(r + s)) * Math.Sqrt(Math.Abs(r - s));
                            else { r = 0.0; q = 0.0; minimumFound = true; }
                        }
                        else
                        {
                            // sqrt(r² + s²) without under/overflow.
                            var sumsq = 1.0;
                            p = 0.0;
                            double scale;
                            if (absr >= s)
                            {
                                if (absr > RtSmll) p = absr * RtSmll;
                                if (s >= p) { var value = s / absr; sumsq = 1.0 + value * value; }
                                scale = absr;
                            }
                            else
                            {
                                if (s > RtSmll) p = s * RtSmll;
                                if (absr >= p) { var value = absr / s; sumsq = 1.0 + value * value; }
                                scale = s;
                            }
                            sumsq = Math.Sqrt(sumsq);
                            q = Big;
                            if (scale < Big / sumsq) q = scale * sumsq;
                        }
                    }
                    if (!minimumFound)
                    {
                        if (_xw < 0.0) q = -q;
                        s = _xw * (Gmin - r - q);
                        q = _gw - Gmin + q + q;
                        if (q > 0.0) s = -s;
                        if (q <= 0.0) q = -q;
                        r = _e;
                        if (_b1 != _step || _braktd) _e = _step;
                    }
                }

                // MinimumFound:
                a1 = _a;
                _b1 = _b;
                _step = xmidpt;
                if (!_braktd || (_a == 0.0 && _xw < 0.0) || (_b == 0.0 && _xw > 0.0))
                {
                    if (_braktd)
                    {
                        var d1 = _xw;
                        var d2 = _a;
                        if (_a == 0.0) d2 = _b;
                        U = -d1 / d2;
                        _step = 5.0 * d2 * (0.1 + 1.0 / U) / 11.0;
                        if (U < 1.0) _step = 0.5 * d2 * Math.Sqrt(U);
                    }
                    else
                    {
                        _step = -_factor * _xw;
                        if (_step > _scxbnd) _step = _scxbnd;
                        if (_step != _scxbnd) _factor = 5.0 * _factor;
                    }
                    if (_step <= 0.0) a1 = _step;
                    if (_step > 0.0) _b1 = _step;
                }

                if (Math.Abs(s) <= Math.Abs(0.5 * q * r) || s <= q * a1 || s >= q * _b1) _e = _b - _a;
                else
                {
                    _step = s / q;
                    if (_step - _a < twotol || _b - _step < twotol) _step = xmidpt <= 0.0 ? -_tol : _tol;
                }

                if (_step >= _scxbnd)
                {
                    _step = _scxbnd;
                    _scxbnd -= (RelTol * Math.Abs(Xbnd) + AbsTol) / (1.0 + RelTol);
                }
                SetU();
                return Rc.Eval;
            }
        }

        // ---------------- BLAS-like helpers, with tnc.c's summation order ----------------

        private static double Dot(double[] dx, double[] dy)
        {
            var t = 0.0;
            for (var i = 0; i < dx.Length; i++) t += dy[i] * dx[i];
            return t;
        }

        /// <summary>tnc.c's <c>dnrm21</c>: the scaled (LAPACK dnrm2-style) Euclidean norm.</summary>
        private static double Dnrm2(double[] dx)
        {
            double dssq = 1.0, dscale = 0.0;
            for (var i = 0; i < dx.Length; i++)
            {
                if (dx[i] == 0.0) continue;
                var dabsxi = Math.Abs(dx[i]);
                if (dscale < dabsxi)
                {
                    var ratio = dscale / dabsxi;
                    dssq = 1.0 + dssq * ratio * ratio;
                    dscale = dabsxi;
                }
                else
                {
                    var ratio = dabsxi / dscale;
                    dssq += ratio * ratio;
                }
            }
            return dscale * Math.Sqrt(dssq);
        }
    }
}
