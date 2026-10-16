namespace Gravicode.Science.SciNet;

/// <summary>Options for <see cref="MinimizeMethod.Tnc"/>; names and defaults follow SciPy's <c>_minimize_tnc</c>.</summary>
public sealed record TncOptions : MinimizeOptions
{
    /// <summary>Absolute forward-difference step when no gradient is given. Default 1e-8.</summary>
    public double Eps { get; init; } = 1e-8;
    /// <summary>Variable scale factors; default ub − lb for doubly bounded variables, 1 + |x0| otherwise.</summary>
    public double[]? Scale { get; init; }
    /// <summary>Variable offsets; default (ub + lb)/2 for doubly bounded variables, x0 otherwise.</summary>
    public double[]? Offset { get; init; }
    /// <summary>Maximum CG iterations per Newton step; default max(1, min(50, n/2)).</summary>
    public int? MaxCGIt { get; init; }
    /// <summary>Maximum function evaluations; default max(100, 10 n).</summary>
    public int? MaxFun { get; init; }
    /// <summary>Line-search severity (Armijo constant scale, 0 &lt; eta &lt; 1). Default 0.25.</summary>
    public double Eta { get; init; } = 0.25;
    /// <summary>Maximum step of the line search in scaled variables. Default 10.</summary>
    public double StepMx { get; init; } = 10;
    /// <summary>Relative precision of the Hessian-vector finite differences; default √ε.</summary>
    public double Accuracy { get; init; }
    /// <summary>Stop when f falls to this value. Default −∞ (disabled).</summary>
    public double Fmin { get; init; } = double.NegativeInfinity;
    /// <summary>Relative change in f at which to stop; negative means 0.</summary>
    public double? Ftol { get; init; }
    /// <summary>Scaled step length at which to stop; negative means √ε.</summary>
    public double? Xtol { get; init; }
    /// <summary>Scaled projected-gradient norm at which to stop; negative means 1e-2·√accuracy.</summary>
    public double? Gtol { get; init; }
}

public static partial class Optimize
{
    static partial void RunTnc(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeTnc.Run(problem);
}

/// <summary>
/// Truncated Newton with simple bounds in the spirit of Nash's TN as wrapped by SciPy's TNC:
/// variables are scaled to the box, the Newton system on the free variables is solved inexactly by
/// conjugate gradients with Hessian-vector products from gradient differences, and a projected
/// Armijo search keeps iterates feasible. This is not a line-for-line port of <c>tnc.c</c>, so
/// iteration counts differ from SciPy's; the termination codes and messages are SciPy's.
/// </summary>
internal static class OptimizeTnc
{
    private static readonly string[] Messages =
    [
        "Local minimum reached (|pg| ~= 0)", "Converged (|f_n-f_(n-1)| ~= 0)", "Converged (|x_n-x_(n-1)| ~= 0)",
        "Max. number of function evaluations reached", "Linear search failed", "All lower bounds are equal to the upper bounds",
        "Unable to progress", "User requested end of minimization",
    ];

    public static OptimizeResult Run(MinimizeProblem p)
    {
        var o = p.OptionsAs<TncOptions>("TNC");
        var n = p.N;
        var sf = p.CreateObjective(o.Eps);
        var lb = p.Bounds?.Lb ?? Enumerable.Repeat(double.NegativeInfinity, n).ToArray();
        var ub = p.Bounds?.Ub ?? Enumerable.Repeat(double.PositiveInfinity, n).ToArray();
        var x0 = p.X0;
        var maxfun = o.MaxFun ?? Math.Max(100, 10 * n);
        var maxCg = o.MaxCGIt ?? Math.Max(1, Math.Min(50, n / 2));
        var accuracy = o.Accuracy <= 2.220446049250313e-16 ? Math.Sqrt(2.220446049250313e-16) : o.Accuracy;
        var ftol = Math.Max(0, o.Ftol ?? p.Tol ?? -1);
        var xtolRaw = o.Xtol ?? p.Tol ?? -1;
        var xtol = xtolRaw < 0 ? Math.Sqrt(2.220446049250313e-16) : xtolRaw;
        var gtolRaw = o.Gtol ?? p.Tol ?? -1;
        var pgtol = gtolRaw < 0 ? 1e-2 * Math.Sqrt(accuracy) : gtolRaw;

        // Scaled variables y = (x - offset) / scale.
        var scale = new double[n]; var offset = new double[n];
        for (var i = 0; i < n; i++)
        {
            var both = !double.IsInfinity(lb[i]) && !double.IsInfinity(ub[i]);
            scale[i] = o.Scale?[i] ?? (both && ub[i] > lb[i] ? ub[i] - lb[i] : 1 + Math.Abs(x0[i]));
            offset[i] = o.Offset?[i] ?? (both ? 0.5 * (ub[i] + lb[i]) : x0[i]);
            if (!(scale[i] > 0)) scale[i] = 1;
        }
        var ylb = new double[n]; var yub = new double[n];
        for (var i = 0; i < n; i++) { ylb[i] = (lb[i] - offset[i]) / scale[i]; yub[i] = (ub[i] - offset[i]) / scale[i]; }
        double[] ToX(double[] y) { var x = new double[n]; for (var i = 0; i < n; i++) x[i] = Math.Clamp(y[i] * scale[i] + offset[i], lb[i], ub[i]); return x; }
        var gx = new double[n];
        double FG(double[] y, double[] gy)
        {
            var f0 = sf.FunAndGrad(ToX(y), gx);
            for (var i = 0; i < n; i++) gy[i] = gx[i] * scale[i];
            return f0;
        }

        var allFixed = true;
        for (var i = 0; i < n; i++) if (lb[i] != ub[i]) allFixed = false;

        var yv = new double[n];
        for (var i = 0; i < n; i++) yv[i] = Math.Clamp((x0[i] - offset[i]) / scale[i], ylb[i], yub[i]);
        var g = new double[n];
        var f = FG(yv, g);
        var status = -2;
        var nit = 0;
        if (allFixed) status = 5;

        var free = new bool[n];
        var pk = new double[n]; var r = new double[n]; var dvec = new double[n]; var hd = new double[n];
        var gp = new double[n]; var yt = new double[n]; var gt = new double[n];
        while (status == -2)
        {
            // Active set: at a bound with the gradient pointing out of the box.
            double pgn = 0;
            for (var i = 0; i < n; i++)
            {
                var atL = yv[i] <= ylb[i]; var atU = yv[i] >= yub[i];
                free[i] = !(ylb[i] == yub[i] || (atL && g[i] > 0) || (atU && g[i] < 0));
                if (free[i]) pgn = Math.Max(pgn, Math.Abs(g[i]));
            }
            if (pgn <= pgtol) { status = 0; break; }
            if (f <= o.Fmin) { status = 0; break; }
            if (sf.Nfev >= maxfun) { status = 3; break; }

            // Truncated CG on the free subspace: H p = -g.
            Array.Clear(pk);
            for (var i = 0; i < n; i++) { r[i] = free[i] ? -g[i] : 0; dvec[i] = r[i]; }
            var rr = OptimizeDense.Dot(r, r);
            var r0 = Math.Sqrt(rr);
            var tolCg = Math.Min(0.5, Math.Sqrt(r0)) * r0;
            for (var it = 0; it < maxCg; it++)
            {
                var dn = OptimizeDense.Norm2(dvec);
                if (dn == 0) break;
                var yn = OptimizeDense.Norm2(yv);
                var h = Math.Sqrt(accuracy) * (1 + yn) / dn;
                // Step away from a bound rather than across it.
                var sign = 1.0;
                for (var i = 0; i < n; i++)
                    if (dvec[i] != 0 && (yv[i] + h * dvec[i] > yub[i] || yv[i] + h * dvec[i] < ylb[i])) { sign = -1; break; }
                for (var i = 0; i < n; i++) yt[i] = yv[i] + sign * h * dvec[i];
                FG(yt, gt);
                for (var i = 0; i < n; i++) hd[i] = free[i] ? (gt[i] - g[i]) / (sign * h) : 0;
                var curv = OptimizeDense.Dot(dvec, hd);
                if (curv <= 1e-16 * dn * dn)
                {
                    if (it == 0) for (var i = 0; i < n; i++) pk[i] = -(free[i] ? g[i] : 0);
                    break;
                }
                var alpha = rr / curv;
                for (var i = 0; i < n; i++) { pk[i] += alpha * dvec[i]; r[i] -= alpha * hd[i]; }
                var rrNew = OptimizeDense.Dot(r, r);
                if (Math.Sqrt(rrNew) <= tolCg) break;
                var beta = rrNew / rr;
                rr = rrNew;
                for (var i = 0; i < n; i++) dvec[i] = r[i] + beta * dvec[i];
            }
            if (OptimizeDense.Dot(pk, g) >= 0) for (var i = 0; i < n; i++) pk[i] = free[i] ? -g[i] : 0;

            // Projected Armijo backtracking.
            var pnorm = OptimizeDense.Norm2(pk);
            var step = Math.Min(1.0, o.StepMx / pnorm);
            var accepted = false;
            double fNew = f;
            for (var ls = 0; ls < 30 && sf.Nfev < maxfun; ls++)
            {
                for (var i = 0; i < n; i++) yt[i] = Math.Clamp(yv[i] + step * pk[i], ylb[i], yub[i]);
                double dec = 0;
                for (var i = 0; i < n; i++) dec += g[i] * (yt[i] - yv[i]);
                fNew = FG(yt, gt);
                if (fNew <= f + 1e-4 * o.Eta * 4 * dec && dec < 0) { accepted = true; break; }
                step *= 0.5;
            }
            if (!accepted) { status = sf.Nfev >= maxfun ? 3 : 4; break; }
            double sn = 0, yn2 = 0;
            for (var i = 0; i < n; i++) { sn += (yt[i] - yv[i]) * (yt[i] - yv[i]); yn2 += yt[i] * yt[i]; }
            var fOld = f;
            Array.Copy(yt, yv, n); Array.Copy(gt, g, n); f = fNew;
            nit++;
            if (p.NotifyCallback(ToX(yv), f)) { status = 7; break; }
            if (Math.Abs(fOld - f) <= ftol * Math.Max(1, Math.Abs(f)) && ftol > 0) { status = 1; break; }
            if (Math.Sqrt(sn) <= xtol * (1 + Math.Sqrt(yn2))) { status = 2; break; }
            if (fOld - f == 0) { status = 1; break; }
        }

        var xr = ToX(yv);
        var jac = new double[n];
        for (var i = 0; i < n; i++) jac[i] = g[i] / scale[i];
        var active = new int[n];
        for (var i = 0; i < n; i++) active[i] = xr[i] <= lb[i] ? -1 : xr[i] >= ub[i] ? 1 : 0;
        return new OptimizeResult
        {
            X = xr, Fun = f, Jac = jac, Nit = nit, Nfev = sf.Nfev, Njev = sf.Ngev, Status = status,
            Success = status is 0 or 1 or 2, Message = Messages[status], ActiveMask = active,
        };
    }
}
