using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Options for <see cref="MinimizeMethod.CG"/>.</summary>
public sealed record CgOptions : MinimizeOptions
{
    /// <summary>Gradient norm at which to stop. Default 1e-5.</summary>
    public double? Gtol { get; init; }
    /// <summary>Order of the gradient norm (∞ by default, as SciPy).</summary>
    public double Norm { get; init; } = double.PositiveInfinity;
    /// <summary>Absolute finite-difference step when no gradient is given.</summary>
    public double Eps { get; init; } = 1.4901161193847656e-08;
    /// <summary>Armijo constant of the line search.</summary>
    public double C1 { get; init; } = 1e-4;
    /// <summary>Curvature constant of the line search (0.4 for CG).</summary>
    public double C2 { get; init; } = 0.4;
}

/// <summary>Options for <see cref="MinimizeMethod.Bfgs"/>.</summary>
public sealed record BfgsOptions : MinimizeOptions
{
    /// <summary>Gradient norm at which to stop. Default 1e-5.</summary>
    public double? Gtol { get; init; }
    /// <summary>Order of the gradient norm (∞ by default, as SciPy).</summary>
    public double Norm { get; init; } = double.PositiveInfinity;
    /// <summary>Absolute finite-difference step when no gradient is given.</summary>
    public double Eps { get; init; } = 1.4901161193847656e-08;
    /// <summary>Relative step size at which to stop (0 disables).</summary>
    public double Xrtol { get; init; }
    /// <summary>Armijo constant.</summary>
    public double C1 { get; init; } = 1e-4;
    /// <summary>Curvature constant.</summary>
    public double C2 { get; init; } = 0.9;
    /// <summary>Initial inverse Hessian (n, n); identity by default.</summary>
    public NdArray? HessInv0 { get; init; }
}

/// <summary>Options for <see cref="MinimizeMethod.NewtonCG"/>.</summary>
public sealed record NewtonCgOptions : MinimizeOptions
{
    /// <summary>Average relative step at which to stop (scaled by n inside). Default 1e-5.</summary>
    public double? Xtol { get; init; }
    /// <summary>Step for the gradient-difference Hessian-vector product.</summary>
    public double Eps { get; init; } = 1.4901161193847656e-08;
    /// <summary>Armijo constant.</summary>
    public double C1 { get; init; } = 1e-4;
    /// <summary>Curvature constant.</summary>
    public double C2 { get; init; } = 0.9;
}

public static partial class Optimize
{
    static partial void RunCG(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeQuasiNewton.Cg(problem);
    static partial void RunBfgs(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeQuasiNewton.Bfgs(problem);
    static partial void RunNewtonCG(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeQuasiNewton.NewtonCg(problem);
}

/// <summary>Ports of SciPy's <c>_minimize_cg</c>, <c>_minimize_bfgs</c> and <c>_minimize_newtoncg</c>.</summary>
internal static class OptimizeQuasiNewton
{
    public static double VecNorm(ReadOnlySpan<double> v, double ord)
    {
        if (double.IsPositiveInfinity(ord)) return OptimizeDense.NormInf(v);
        if (double.IsNegativeInfinity(ord)) { var m = double.PositiveInfinity; foreach (var a in v) m = Math.Min(m, Math.Abs(a)); return m; }
        if (ord == 2) return OptimizeDense.Norm2(v);
        double s = 0;
        foreach (var a in v) s += Math.Pow(Math.Abs(a), ord);
        return Math.Pow(s, 1 / ord);
    }

    public static OptimizeResult Cg(MinimizeProblem p)
    {
        var o = p.OptionsAs<CgOptions>("CG");
        var gtol = o.Gtol ?? p.Tol ?? 1e-5;
        var n = p.N;
        var maxiter = o.MaxIter ?? n * 200;
        var sf = p.CreateObjective(o.Eps, useBounds: false);
        var xk = (double[])p.X0.Clone();
        var oldF = sf.Fun(xk);
        var gfk = sf.Grad(xk);
        var oldOldF = oldF + OptimizeDense.Norm2(gfk) / 2;
        var pk = gfk.Select(v => -v).ToArray();
        var gnorm = VecNorm(gfk, o.Norm);
        const double sigma3 = 0.01;
        var k = 0; var warn = 0;
        var xNew = new double[n]; var gNew = new double[n];
        while (gnorm > gtol && k < maxiter)
        {
            var deltak = OptimizeDense.Dot(gfk, gfk);
            double cachedAlpha = double.NaN;
            double[]? cPk = null, cG = null, cX = null; double cGnorm = 0;
            (double[] X, double[] P, double[] G, double Gn) Step(double alpha, double[] gp)
            {
                var x1 = new double[n];
                for (var i = 0; i < n; i++) x1[i] = xk[i] + alpha * pk[i];
                double yg = 0;
                for (var i = 0; i < n; i++) yg += (gp[i] - gfk[i]) * gp[i];
                var beta = Math.Max(0, yg / deltak);
                var p1 = new double[n];
                for (var i = 0; i < n; i++) p1[i] = -gp[i] + beta * pk[i];
                return (x1, p1, (double[])gp.Clone(), VecNorm(gp, o.Norm));
            }
            bool Descent(double alpha, double[] x1, double[] g1)
            {
                var s = Step(alpha, g1);
                cachedAlpha = alpha; cX = s.X; cPk = s.P; cG = s.G; cGnorm = s.Gn;
                if (cGnorm <= gtol) return true;
                return OptimizeDense.Dot(cPk, cG) <= -sigma3 * OptimizeDense.Dot(cG, cG);
            }
            var prevF = oldF;
            var alphaK = OptimizeLineSearch.Wolfe12(sf, xk, pk, gfk, oldF, oldOldF, o.C1, o.C2, 1e-100, 1e100, xNew, gNew, out var fNew, Descent);
            if (alphaK is null) { warn = 2; break; }
            oldOldF = prevF; oldF = fNew;
            if (alphaK.Value == cachedAlpha) { xk = cX!; pk = cPk!; gfk = cG!; gnorm = cGnorm; }
            else { var s = Step(alphaK.Value, gNew); xk = s.X; pk = s.P; gfk = s.G; gnorm = s.Gn; }
            k++;
            if (p.NotifyCallback(xk, oldF)) break;
        }
        string msg;
        if (warn == 2) msg = OptimizeMessages.PrLoss;
        else if (k >= maxiter) { warn = 1; msg = OptimizeMessages.MaxIter; }
        else if (double.IsNaN(gnorm) || double.IsNaN(oldF) || xk.Any(double.IsNaN)) { warn = 3; msg = OptimizeMessages.Nan; }
        else msg = OptimizeMessages.Success;
        return new OptimizeResult { X = xk, Fun = oldF, Jac = gfk, Status = warn, Success = warn == 0, Message = msg, Nit = k, Nfev = sf.Nfev, Njev = sf.Ngev };
    }

    public static OptimizeResult Bfgs(MinimizeProblem p)
    {
        var o = p.OptionsAs<BfgsOptions>("BFGS");
        var gtol = o.Gtol ?? p.Tol ?? 1e-5;
        var n = p.N;
        var maxiter = o.MaxIter ?? n * 200;
        var sf = p.CreateObjective(o.Eps, useBounds: false);
        var xk = (double[])p.X0.Clone();
        var oldF = sf.Fun(xk);
        var gfk = sf.Grad(xk);
        double[] hk;
        if (o.HessInv0 is null) hk = OptimizeDense.Identity(n);
        else hk = OptimizeDense.ToMatrix(o.HessInv0, n, n, "HessInv0");
        var oldOldF = oldF + OptimizeDense.Norm2(gfk) / 2;
        var k = 0; var warn = 0;
        var gnorm = VecNorm(gfk, o.Norm);
        var pk = new double[n]; var xNew = new double[n]; var gNew = new double[n];
        var sk = new double[n]; var yk = new double[n]; var hy = new double[n];
        while (gnorm > gtol && k < maxiter)
        {
            OptimizeDense.MatVec(hk, n, n, gfk, pk);
            for (var i = 0; i < n; i++) pk[i] = -pk[i];
            var prevF = oldF;
            var alpha = OptimizeLineSearch.Wolfe12(sf, xk, pk, gfk, oldF, oldOldF, o.C1, o.C2, 1e-100, 1e100, xNew, gNew, out var fNew);
            if (alpha is null) { warn = 2; break; }
            oldOldF = prevF; oldF = fNew;
            for (var i = 0; i < n; i++) { sk[i] = alpha.Value * pk[i]; yk[i] = gNew[i] - gfk[i]; }
            xk = (double[])xNew.Clone();
            gfk = (double[])gNew.Clone();
            k++;
            if (p.NotifyCallback(xk, oldF)) break;
            gnorm = VecNorm(gfk, o.Norm);
            if (gnorm <= gtol) break;
            if (alpha.Value * OptimizeDense.Norm2(pk) <= o.Xrtol * (o.Xrtol + OptimizeDense.Norm2(xk))) break;
            if (!double.IsFinite(oldF)) { warn = 2; break; }
            var rhoInv = OptimizeDense.Dot(yk, sk);
            var rho = rhoInv == 0 ? 1000.0 : 1.0 / rhoInv;
            // H+ = (I - ρ s yᵀ) H (I - ρ y sᵀ) + ρ s sᵀ, expanded into an O(n²) rank-two update.
            OptimizeDense.MatVec(hk, n, n, yk, hy);
            var yhy = OptimizeDense.Dot(yk, hy);
            var c = rho * rho * yhy + rho;
            for (var i = 0; i < n; i++)
            {
                var row = hk.AsSpan(i * n, n);
                var si = sk[i]; var hyi = hy[i];
                for (var j = 0; j < n; j++) row[j] += -rho * (hyi * sk[j] + si * hy[j]) + c * si * sk[j];
            }
        }
        string msg;
        if (warn == 2) msg = OptimizeMessages.PrLoss;
        else if (k >= maxiter) { warn = 1; msg = OptimizeMessages.MaxIter; }
        else if (double.IsNaN(gnorm) || double.IsNaN(oldF) || xk.Any(double.IsNaN)) { warn = 3; msg = OptimizeMessages.Nan; }
        else msg = OptimizeMessages.Success;
        return new OptimizeResult
        {
            X = xk, Fun = oldF, Jac = gfk, HessInv = new NdArray(hk, n, n), Nfev = sf.Nfev, Njev = sf.Ngev,
            Status = warn, Success = warn == 0, Message = msg, Nit = k,
        };
    }

    public static OptimizeResult NewtonCg(MinimizeProblem p)
    {
        var o = p.OptionsAs<NewtonCgOptions>("Newton-CG");
        var n = p.N;
        var xtol = n * (o.Xtol ?? p.Tol ?? 1e-5);
        var maxiter = o.MaxIter ?? n * 200;
        var cgMaxiter = 20 * n;
        var sf = p.CreateObjective(o.Eps, useBounds: false);
        var xk = (double[])p.X0.Clone();
        var oldF = sf.Fun(xk);
        double? oldOldF = null;
        var k = 0;
        var updateL1 = double.MaxValue;
        double[] gfk = new double[n];
        var xNew = new double[n]; var gNew = new double[n];
        const double eps64 = 2.220446049250313e-16;

        OptimizeResult Terminate(int warn, string msg) => new()
        {
            X = xk, Fun = oldF, Jac = gfk, Nfev = sf.Nfev, Njev = sf.Ngev, Nhev = p.Nhev, Status = warn, Success = warn == 0, Message = msg, Nit = k,
        };

        while (updateL1 > xtol)
        {
            if (k >= maxiter) return Terminate(1, "Warning: " + OptimizeMessages.MaxIter);
            var g = sf.Grad(xk);
            var b = g.Select(v => -v).ToArray();
            double maggrad = 0;
            foreach (var v in b) maggrad += Math.Abs(v);
            var eta = Math.Min(0.5, Math.Sqrt(maggrad));
            var termcond = eta * maggrad;
            var xsupi = new double[n];
            var ri = g.ToArray();
            var psupi = b.ToArray();
            var i = 0;
            var dri0 = OptimizeDense.Dot(ri, ri);
            double[]? A = null;
            if (p.Hess is not null) A = p.EvalHess(xk).AsContiguous().ToArray();
            var ap = new double[n];
            var converged = false;
            for (var k2 = 0; k2 < cgMaxiter; k2++)
            {
                double l1 = 0;
                foreach (var v in ri) l1 += Math.Abs(v);
                if (l1 <= termcond) { converged = true; break; }
                if (A is not null) OptimizeDense.MatVec(A, n, n, psupi, ap);
                else if (p.Hessp is not null) ap = p.EvalHessp(xk, psupi);
                else
                {
                    // SciPy's approx_fhess_p: base point first (often cached), then the probe.
                    var xp = new double[n];
                    for (var j = 0; j < n; j++) xp[j] = xk[j] + o.Eps * psupi[j];
                    var f1 = sf.Grad(xk);
                    var f2 = sf.Grad(xp);
                    for (var j = 0; j < n; j++) ap[j] = (f2[j] - f1[j]) / o.Eps;
                }
                var curv = OptimizeDense.Dot(psupi, ap);
                if (0 <= curv && curv <= 3 * eps64) { converged = true; break; }
                if (curv < 0)
                {
                    if (i == 0) for (var j = 0; j < n; j++) xsupi[j] = dri0 / (-curv) * b[j];
                    converged = true;
                    break;
                }
                var alphai = dri0 / curv;
                for (var j = 0; j < n; j++) { xsupi[j] += alphai * psupi[j]; ri[j] += alphai * ap[j]; }
                var dri1 = OptimizeDense.Dot(ri, ri);
                var betai = dri1 / dri0;
                for (var j = 0; j < n; j++) psupi[j] = -ri[j] + betai * psupi[j];
                i++;
                dri0 = dri1;
            }
            if (!converged) return Terminate(3, "Warning: CG iterations didn't converge. The Hessian is not positive definite.");
            gfk = g;
            var prevF = oldF;
            var alpha = OptimizeLineSearch.Wolfe12(sf, xk, xsupi, gfk, oldF, oldOldF, o.C1, o.C2, 1e-100, 1e100, xNew, gNew, out var fNew);
            if (alpha is null) return Terminate(2, "Warning: " + OptimizeMessages.PrLoss);
            oldOldF = prevF; oldF = fNew;
            updateL1 = 0;
            for (var j = 0; j < n; j++) { var u = alpha.Value * xsupi[j]; xk[j] += u; updateL1 += Math.Abs(u); }
            k++;
            if (p.NotifyCallback(xk, oldF)) return Terminate(5, "");
        }
        if (double.IsNaN(oldF) || double.IsNaN(updateL1)) return Terminate(3, OptimizeMessages.Nan);
        return Terminate(0, OptimizeMessages.Success);
    }
}
