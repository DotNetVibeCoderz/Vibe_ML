namespace Gravicode.Science.SciNet;

/// <summary>Outcome of a line search along a direction.</summary>
/// <param name="Alpha">Accepted step length, or null when no point satisfying the conditions was found.</param>
/// <param name="Fc">Function evaluations used.</param>
/// <param name="Gc">Gradient evaluations used.</param>
/// <param name="NewFval">f(x + alpha p) (or the last value tried when the search failed).</param>
/// <param name="OldFval">f(x).</param>
/// <param name="NewSlope">Directional derivative at the accepted point, when it was evaluated.</param>
/// <param name="Message">Termination message of the inner search ("CONVERGENCE", or a warning).</param>
public sealed record LineSearchResult(double? Alpha, int Fc, int Gc, double NewFval, double OldFval, double? NewSlope, string Message);

/// <summary>
/// The Moré–Thuente line search (MINPACK-2 <c>dcsrch</c>/<c>dcstep</c>) and the Nocedal–Wright
/// zoom search, ported from SciPy's <c>_linesearch.py</c> / <c>_dcsrch.py</c>. BFGS and CG try the
/// first and fall back to the second, exactly as SciPy's <c>_line_search_wolfe12</c>; L-BFGS-B uses
/// <see cref="Dcsrch"/> directly with its own constants.
/// </summary>
internal static class OptimizeLineSearch
{
    /// <summary>One-dimensional merit function: returns phi(alpha) and writes phi'(alpha).</summary>
    public delegate double PhiFunction(double alpha, out double derphi);

    /// <summary>
    /// Moré–Thuente search for a step satisfying <c>phi(stp) &lt;= phi(0) + ftol stp phi'(0)</c> and
    /// <c>|phi'(stp)| &lt;= gtol |phi'(0)|</c>. Returns the final step and the task string
    /// ("CONVERGENCE", "WARNING: ...", "ERROR: ...").
    /// </summary>
    /// <param name="phi">Evaluates the merit function and its slope.</param>
    /// <param name="stp">Initial trial step; overwritten with the last step tried.</param>
    /// <param name="f">phi(0) on entry; phi(stp) on exit.</param>
    /// <param name="g">phi'(0) on entry; phi'(stp) on exit.</param>
    /// <param name="ftol">Sufficient-decrease constant.</param>
    /// <param name="gtol">Curvature constant.</param>
    /// <param name="xtol">Relative width at which the bracket is accepted.</param>
    /// <param name="stpmin">Lower bound on the step.</param>
    /// <param name="stpmax">Upper bound on the step.</param>
    /// <param name="maxiter">Maximum evaluations of <paramref name="phi"/>.</param>
    /// <param name="nfev">Evaluations made.</param>
    public static string Dcsrch(PhiFunction phi, ref double stp, ref double f, ref double g,
        double ftol, double gtol, double xtol, double stpmin, double stpmax, int maxiter, out int nfev)
    {
        nfev = 0;
        const double xtrapl = 1.1, xtrapu = 4.0;
        if (stp < stpmin) return "ERROR: STP .LT. STPMIN";
        if (stp > stpmax) return "ERROR: STP .GT. STPMAX";
        if (g >= 0) return "ERROR: INITIAL G .GE. ZERO";
        if (ftol < 0) return "ERROR: FTOL .LT. ZERO";
        if (gtol < 0) return "ERROR: GTOL .LT. ZERO";
        if (xtol < 0) return "ERROR: XTOL .LT. ZERO";
        if (stpmin < 0) return "ERROR: STPMIN .LT. ZERO";
        if (stpmax < stpmin) return "ERROR: STPMAX .LT. STPMIN";

        var brackt = false;
        var stage = 1;
        var finit = f;
        var ginit = g;
        var gtest = ftol * ginit;
        var width = stpmax - stpmin;
        var width1 = width / 0.5;
        double stx = 0, fx = finit, gx = ginit;
        double sty = 0, fy = finit, gy = ginit;
        double stmin = 0, stmax = stp + xtrapu * stp;

        for (var it = 0; it < maxiter; it++)
        {
            f = phi(stp, out g);
            nfev++;

            var ftest = finit + stp * gtest;
            if (stage == 1 && f <= ftest && g >= 0) stage = 2;

            string? task = null;
            if (brackt && (stp <= stmin || stp >= stmax)) task = "WARNING: ROUNDING ERRORS PREVENT PROGRESS";
            if (brackt && stmax - stmin <= xtol * stmax) task = "WARNING: XTOL TEST SATISFIED";
            if (stp == stpmax && f <= ftest && g <= gtest) task = "WARNING: STP = STPMAX";
            if (stp == stpmin && (f > ftest || g >= gtest)) task = "WARNING: STP = STPMIN";
            if (f <= ftest && Math.Abs(g) <= gtol * (-ginit)) task = "CONVERGENCE";
            if (task is not null) return task;

            if (stage == 1 && f <= fx && f > ftest)
            {
                // Modified function values so the step stays a sufficient-decrease candidate.
                var fm = f - stp * gtest;
                var fxm = fx - stx * gtest;
                var fym = fy - sty * gtest;
                var gm = g - gtest;
                var gxm = gx - gtest;
                var gym = gy - gtest;
                Dcstep(ref stx, ref fxm, ref gxm, ref sty, ref fym, ref gym, ref stp, fm, gm, ref brackt, stmin, stmax);
                fx = fxm + stx * gtest;
                fy = fym + sty * gtest;
                gx = gxm + gtest;
                gy = gym + gtest;
            }
            else
            {
                Dcstep(ref stx, ref fx, ref gx, ref sty, ref fy, ref gy, ref stp, f, g, ref brackt, stmin, stmax);
            }

            if (brackt)
            {
                if (Math.Abs(sty - stx) >= 0.66 * width1) stp = stx + 0.5 * (sty - stx);
                width1 = width;
                width = Math.Abs(sty - stx);
            }

            if (brackt) { stmin = Math.Min(stx, sty); stmax = Math.Max(stx, sty); }
            else { stmin = stp + xtrapl * (stp - stx); stmax = stp + xtrapu * (stp - stx); }

            stp = Math.Max(stp, stpmin);
            stp = Math.Min(stp, stpmax);

            if ((brackt && (stp <= stmin || stp >= stmax)) || (brackt && stmax - stmin <= xtol * stmax)) stp = stx;
        }
        return "WARNING: MAXITER";
    }

    /// <summary>The safeguarded cubic/quadratic step of Moré and Thuente (1994).</summary>
    public static void Dcstep(ref double stx, ref double fx, ref double dx, ref double sty, ref double fy, ref double dy,
        ref double stp, double fp, double dp, ref bool brackt, double stpmin, double stpmax)
    {
        var sgnd = dp * Math.Sign(dx);
        double stpf;
        if (fp > fx)
        {
            var theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            var s = Max3(Math.Abs(theta), Math.Abs(dx), Math.Abs(dp));
            var gamma = s * Math.Sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp < stx) gamma = -gamma;
            var p = (gamma - dx) + theta;
            var q = ((gamma - dx) + gamma) + dp;
            var r = p / q;
            var stpc = stx + r * (stp - stx);
            var stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);
            stpf = Math.Abs(stpc - stx) < Math.Abs(stpq - stx) ? stpc : stpc + (stpq - stpc) / 2.0;
            brackt = true;
        }
        else if (sgnd < 0)
        {
            var theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            var s = Max3(Math.Abs(theta), Math.Abs(dx), Math.Abs(dp));
            var gamma = s * Math.Sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp > stx) gamma = -gamma;
            var p = (gamma - dp) + theta;
            var q = ((gamma - dp) + gamma) + dx;
            var r = p / q;
            var stpc = stp + r * (stx - stp);
            var stpq = stp + (dp / (dp - dx)) * (stx - stp);
            stpf = Math.Abs(stpc - stp) > Math.Abs(stpq - stp) ? stpc : stpq;
            brackt = true;
        }
        else if (Math.Abs(dp) < Math.Abs(dx))
        {
            var theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            var s = Max3(Math.Abs(theta), Math.Abs(dx), Math.Abs(dp));
            var gamma = s * Math.Sqrt(Math.Max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
            if (stp > stx) gamma = -gamma;
            var p = (gamma - dp) + theta;
            var q = (gamma + (dx - dp)) + gamma;
            var r = p / q;
            double stpc;
            if (r < 0 && gamma != 0) stpc = stp + r * (stx - stp);
            else if (stp > stx) stpc = stpmax;
            else stpc = stpmin;
            var stpq = stp + (dp / (dp - dx)) * (stx - stp);
            if (brackt)
            {
                stpf = Math.Abs(stpc - stp) < Math.Abs(stpq - stp) ? stpc : stpq;
                stpf = stp > stx ? Math.Min(stp + 0.66 * (sty - stp), stpf) : Math.Max(stp + 0.66 * (sty - stp), stpf);
            }
            else
            {
                stpf = Math.Abs(stpc - stp) > Math.Abs(stpq - stp) ? stpc : stpq;
                stpf = Math.Min(stpmax, stpf);
                stpf = Math.Max(stpmin, stpf);
            }
        }
        else
        {
            if (brackt)
            {
                var theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
                var s = Max3(Math.Abs(theta), Math.Abs(dy), Math.Abs(dp));
                var gamma = s * Math.Sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
                if (stp > sty) gamma = -gamma;
                var p = (gamma - dp) + theta;
                var q = ((gamma - dp) + gamma) + dy;
                var r = p / q;
                stpf = stp + r * (sty - stp);
            }
            else stpf = stp > stx ? stpmax : stpmin;
        }

        if (fp > fx) { sty = stp; fy = fp; dy = dp; }
        else
        {
            if (sgnd < 0) { sty = stx; fy = fx; dy = dx; }
            stx = stp; fx = fp; dx = dp;
        }
        stp = stpf;
    }

    private static double Max3(double a, double b, double c) => Math.Max(a, Math.Max(b, c));

    /// <summary>
    /// SciPy's <c>scalar_search_wolfe1</c>: Moré–Thuente with the initial step guessed from the
    /// previous decrease. Returns the step, or null on failure.
    /// </summary>
    public static double? ScalarSearchWolfe1(PhiFunction phi, double phi0, double? oldPhi0, double derphi0,
        double c1, double c2, double amax, double amin, double xtol, out double phi1, out int nfev)
    {
        var alpha1 = 1.0;
        if (oldPhi0 is { } op && derphi0 != 0)
        {
            alpha1 = Math.Min(1.0, 1.01 * 2 * (phi0 - op) / derphi0);
            if (alpha1 < 0) alpha1 = 1.0;
        }
        var stp = alpha1;
        var f = phi0;
        var g = derphi0;
        var task = Dcsrch(phi, ref stp, ref f, ref g, c1, c2, xtol, amin, amax, 100, out nfev);
        phi1 = f;
        if (task.StartsWith("ERROR", StringComparison.Ordinal) || task.StartsWith("WARN", StringComparison.Ordinal)) return null;
        return stp;
    }

    /// <summary>
    /// SciPy's <c>scalar_search_wolfe2</c> (Nocedal &amp; Wright, Algorithms 3.5/3.6): bracket by
    /// doubling, then zoom with cubic and quadratic interpolation. <paramref name="derphiAt"/>
    /// is only called once a point passes sufficient decrease, so evaluations are saved.
    /// </summary>
    public static double? ScalarSearchWolfe2(Func<double, double> phiOnly, Func<double, double> derphiAt,
        double phi0, double? oldPhi0, double derphi0, double c1, double c2, double? amax,
        Func<double, double, bool>? extraCondition, int maxiter, out double phiStar, out double? derphiStar)
    {
        extraCondition ??= static (_, _) => true;
        double alpha0 = 0;
        var alpha1 = 1.0;
        if (oldPhi0 is { } op && derphi0 != 0) alpha1 = Math.Min(1.0, 1.01 * 2 * (phi0 - op) / derphi0);
        if (alpha1 < 0) alpha1 = 1.0;
        if (amax is { } am) alpha1 = Math.Min(alpha1, am);

        var phiA1 = phiOnly(alpha1);
        var phiA0 = phi0;
        var derphiA0 = derphi0;
        derphiStar = null;

        for (var i = 0; i < maxiter; i++)
        {
            if (alpha1 == 0 || (amax is { } amx && alpha0 > amx))
            {
                phiStar = phi0;
                return null;
            }
            var notFirst = i > 0;
            if (phiA1 > phi0 + c1 * alpha1 * derphi0 || (phiA1 >= phiA0 && notFirst))
                return Zoom(alpha0, alpha1, phiA0, phiA1, derphiA0, phiOnly, derphiAt, phi0, derphi0, c1, c2, extraCondition, out phiStar, out derphiStar);

            var derphiA1 = derphiAt(alpha1);
            if (Math.Abs(derphiA1) <= -c2 * derphi0 && extraCondition(alpha1, phiA1))
            {
                phiStar = phiA1; derphiStar = derphiA1;
                return alpha1;
            }
            if (derphiA1 >= 0)
                return Zoom(alpha1, alpha0, phiA1, phiA0, derphiA1, phiOnly, derphiAt, phi0, derphi0, c1, c2, extraCondition, out phiStar, out derphiStar);

            var alpha2 = 2 * alpha1;
            if (amax is { } amx2) alpha2 = Math.Min(alpha2, amx2);
            alpha0 = alpha1; alpha1 = alpha2;
            phiA0 = phiA1; phiA1 = phiOnly(alpha1);
            derphiA0 = derphiA1;
        }
        // Stopping on maxiter: SciPy returns the last step with a warning ("did not converge").
        phiStar = phiA1;
        return alpha1;
    }

    private static double? Zoom(double aLo, double aHi, double phiLo, double phiHi, double derphiLo,
        Func<double, double> phi, Func<double, double> derphi, double phi0, double derphi0, double c1, double c2,
        Func<double, double, bool> extraCondition, out double phiStar, out double? derphiStar)
    {
        const int maxiter = 10;
        const double delta1 = 0.2, delta2 = 0.1;
        var i = 0;
        var phiRec = phi0;
        double aRec = 0;
        while (true)
        {
            var dalpha = aHi - aLo;
            double a, b;
            if (dalpha < 0) { a = aHi; b = aLo; } else { a = aLo; b = aHi; }

            double? aj = null;
            var cchk = delta1 * dalpha;
            if (i > 0) aj = CubicMin(aLo, phiLo, derphiLo, aHi, phiHi, aRec, phiRec);
            if (i == 0 || aj is null || aj > b - cchk || aj < a + cchk)
            {
                var qchk = delta2 * dalpha;
                aj = QuadMin(aLo, phiLo, derphiLo, aHi, phiHi);
                if (aj is null || aj > b - qchk || aj < a + qchk) aj = aLo + 0.5 * dalpha;
            }
            var ajv = aj.Value;
            var phiAj = phi(ajv);
            if (phiAj > phi0 + c1 * ajv * derphi0 || phiAj >= phiLo)
            {
                phiRec = phiHi; aRec = aHi; aHi = ajv; phiHi = phiAj;
            }
            else
            {
                var derphiAj = derphi(ajv);
                if (Math.Abs(derphiAj) <= -c2 * derphi0 && extraCondition(ajv, phiAj))
                {
                    phiStar = phiAj; derphiStar = derphiAj;
                    return ajv;
                }
                if (derphiAj * (aHi - aLo) >= 0)
                {
                    phiRec = phiHi; aRec = aHi; aHi = aLo; phiHi = phiLo;
                }
                else { phiRec = phiLo; aRec = aLo; }
                aLo = ajv; phiLo = phiAj; derphiLo = derphiAj;
            }
            i++;
            if (i > maxiter)
            {
                phiStar = phiLo; derphiStar = null;
                return null;
            }
        }
    }

    /// <summary>Minimiser of the cubic through (a, fa, fpa), (b, fb), (c, fc); null if it does not exist.</summary>
    public static double? CubicMin(double a, double fa, double fpa, double b, double fb, double c, double fc)
    {
        var C = fpa;
        var db = b - a;
        var dc = c - a;
        var denom = (db * dc) * (db * dc) * (db - dc);
        if (denom == 0) return null;
        var r1 = fb - fa - C * db;
        var r2 = fc - fa - C * dc;
        var A = (dc * dc * r1 - db * db * r2) / denom;
        var B = (-dc * dc * dc * r1 + db * db * db * r2) / denom;
        var radical = B * B - 3 * A * C;
        if (radical < 0 || A == 0) return null;
        var xmin = a + (-B + Math.Sqrt(radical)) / (3 * A);
        return double.IsFinite(xmin) ? xmin : null;
    }

    /// <summary>Minimiser of the quadratic through (a, fa, fpa), (b, fb); null if it does not exist.</summary>
    public static double? QuadMin(double a, double fa, double fpa, double b, double fb)
    {
        var D = fa;
        var C = fpa;
        var db = b - a;
        var B = (fb - D - C * db) / (db * db);
        if (B == 0 || !double.IsFinite(B)) return null;
        var xmin = a - C / (2.0 * B);
        return double.IsFinite(xmin) ? xmin : null;
    }

    /// <summary>
    /// SciPy's <c>_line_search_wolfe12</c> on a counted objective: Moré–Thuente first, the zoom
    /// search if that fails. On success <paramref name="xNew"/>, <paramref name="gNew"/> and
    /// <paramref name="fNew"/> describe the accepted point. Returns null when both searches fail
    /// (SciPy's <c>_LineSearchError</c>).
    /// </summary>
    public static double? Wolfe12(OptimizeScalarFunction sf, double[] xk, double[] pk, double[] gk, double fk, double? oldOldF,
        double c1, double c2, double amin, double amax, double[] xNew, double[] gNew, out double fNew,
        Func<double, double[], double[], bool>? extraCondition = null)
    {
        var n = xk.Length;
        var xt = new double[n];
        var gt = new double[n];
        void At(double alpha) { for (var i = 0; i < n; i++) xt[i] = xk[i] + alpha * pk[i]; }
        double lastAlphaG = double.NaN;

        double Phi(double alpha, out double d)
        {
            At(alpha);
            var f = sf.FunAndGrad(xt, gt);
            lastAlphaG = alpha;
            d = OptimizeDense.Dot(gt, pk);
            return f;
        }

        var derphi0 = OptimizeDense.Dot(gk, pk);
        var alpha = ScalarSearchWolfe1(Phi, fk, oldOldF, derphi0, c1, c2, amax, amin, 1e-14, out var phi1, out _);
        if (alpha is { } a1)
        {
            At(a1);
            if (lastAlphaG != a1) sf.Grad(xt, gt);
            if (extraCondition is null || extraCondition(a1, xt, gt))
            {
                xt.CopyTo(xNew, 0); gt.CopyTo(gNew, 0);
                fNew = phi1;
                return a1;
            }
        }

        double PhiOnly(double al) { At(al); return sf.Fun(xt); }
        double Derphi(double al) { At(al); sf.Grad(xt, gt); lastAlphaG = al; return OptimizeDense.Dot(gt, pk); }
        Func<double, double, bool>? extra2 = extraCondition is null ? null : (al, _) =>
        {
            At(al);
            if (lastAlphaG != al) { sf.Grad(xt, gt); lastAlphaG = al; }
            return extraCondition(al, xt, gt);
        };
        var a2 = ScalarSearchWolfe2(PhiOnly, Derphi, fk, oldOldF, derphi0, c1, c2, amax, extra2, 10, out var phiStar, out _);
        if (a2 is { } av)
        {
            At(av);
            sf.Grad(xt, gt);
            xt.CopyTo(xNew, 0); gt.CopyTo(gNew, 0);
            fNew = phiStar;
            return av;
        }
        fNew = fk;
        return null;
    }
}
