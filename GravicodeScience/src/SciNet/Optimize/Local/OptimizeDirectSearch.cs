using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Options for <see cref="MinimizeMethod.NelderMead"/> (SciPy's <c>_minimize_neldermead</c>).</summary>
public sealed record NelderMeadOptions : MinimizeOptions
{
    /// <summary>Maximum function evaluations; with neither limit set both default to 200·n.</summary>
    public int? MaxFev { get; init; }
    /// <summary>Absolute spread of the simplex vertices at which to stop. Default 1e-4.</summary>
    public double? Xatol { get; init; }
    /// <summary>Absolute spread of the vertex values at which to stop. Default 1e-4.</summary>
    public double? Fatol { get; init; }
    /// <summary>Gao–Han dimension-dependent coefficients; much better above ~10 variables.</summary>
    public bool Adaptive { get; init; }
    /// <summary>Starting simplex, shape (n+1, n); by default x0 plus 5% steps along each axis.</summary>
    public NdArray? InitialSimplex { get; init; }
}

/// <summary>Options for <see cref="MinimizeMethod.Powell"/>.</summary>
public sealed record PowellOptions : MinimizeOptions
{
    /// <summary>Maximum function evaluations; with neither limit set both default to 1000·n.</summary>
    public int? MaxFev { get; init; }
    /// <summary>Line-search tolerance (scaled by 100 inside, as SciPy). Default 1e-4.</summary>
    public double? Xtol { get; init; }
    /// <summary>Relative decrease per sweep at which to stop. Default 1e-4.</summary>
    public double? Ftol { get; init; }
    /// <summary>Initial direction set, shape (n, n), rows are directions. Default identity.</summary>
    public NdArray? Direc { get; init; }
}

/// <summary>Thrown inside a derivative-free method when the evaluation budget runs out.</summary>
internal sealed class OptimizeMaxFevException : Exception { }

/// <summary>Counted objective with a hard evaluation budget (SciPy's <c>_wrap_scalar_function_maxfun_validation</c>).</summary>
internal sealed class OptimizeBudgetFunction(Func<double[], double>? fun, Func<double[], double[], double>? funGrad, double maxfun)
{
    public int Calls;

    public double Eval(double[] x)
    {
        if (Calls >= maxfun) throw new OptimizeMaxFevException();
        Calls++;
        var xc = (double[])x.Clone();
        return fun is not null ? fun(xc) : funGrad!(xc, new double[xc.Length]);
    }
}

public static partial class Optimize
{
    static partial void RunNelderMead(MinimizeProblem problem, ref OptimizeResult? result)
        => result = OptimizeNelderMead.Run(problem);

    static partial void RunPowell(MinimizeProblem problem, ref OptimizeResult? result)
        => result = OptimizePowell.Run(problem);
}

/// <summary>A line-for-line port of SciPy's Nelder–Mead, including bound clipping.</summary>
internal static class OptimizeNelderMead
{
    public static OptimizeResult Run(MinimizeProblem p)
    {
        var o = p.OptionsAs<NelderMeadOptions>("NelderMead");
        var xatol = o.Xatol ?? p.Tol ?? 1e-4;
        var fatol = o.Fatol ?? p.Tol ?? 1e-4;
        var x0 = p.X0;
        var n = x0.Length;
        double rho = 1, chi = 2, psi = 0.5, sigma = 0.5;
        if (o.Adaptive)
        {
            double dim = n;
            chi = 1 + 2 / dim; psi = 0.75 - 1 / (2 * dim); sigma = 1 - 1 / dim;
        }
        var lb = p.Bounds?.Lb; var ub = p.Bounds?.Ub;
        var sim = new double[n + 1][];
        if (o.InitialSimplex is null)
        {
            sim[0] = (double[])x0.Clone();
            for (var k = 0; k < n; k++)
            {
                var y = (double[])x0.Clone();
                y[k] = y[k] != 0 ? (1 + 0.05) * y[k] : 0.00025;
                sim[k + 1] = y;
            }
        }
        else
        {
            var s = o.InitialSimplex;
            if (s.Rank != 2 || s.Shape[0] != n + 1 || s.Shape[1] != n) throw new ArgumentException($"InitialSimplex must have shape ({n + 1}, {n}).");
            var d = s.AsContiguous().ToArray();
            for (var k = 0; k <= n; k++) sim[k] = d.AsSpan(k * n, n).ToArray();
        }

        double maxiter, maxfun;
        if (o.MaxIter is null && o.MaxFev is null) { maxiter = n * 200; maxfun = n * 200; }
        else if (o.MaxIter is null) { maxfun = o.MaxFev!.Value; maxiter = double.PositiveInfinity; }
        else if (o.MaxFev is null) { maxiter = o.MaxIter.Value; maxfun = double.PositiveInfinity; }
        else { maxiter = o.MaxIter.Value; maxfun = o.MaxFev.Value; }

        if (lb is not null)
        {
            // Reflect vertices that the default construction pushed past an upper bound, then clip.
            foreach (var v in sim)
                for (var j = 0; j < n; j++)
                {
                    if (v[j] > ub![j]) v[j] = 2 * ub[j] - v[j];
                    v[j] = Math.Clamp(v[j], lb[j], ub[j]);
                }
        }

        var bf = new OptimizeBudgetFunction(p.Fun, p.FunAndGrad, maxfun);
        var fsim = new double[n + 1];
        Array.Fill(fsim, double.PositiveInfinity);
        try { for (var k = 0; k <= n; k++) fsim[k] = bf.Eval(sim[k]); }
        catch (OptimizeMaxFevException) { }
        Sort(sim, fsim);

        void Clip(double[] v) { if (lb is not null) for (var j = 0; j < n; j++) v[j] = Math.Clamp(v[j], lb[j], ub![j]); }
        var xbar = new double[n];
        var iterations = 1;
        while (bf.Calls < maxfun && iterations < maxiter)
        {
            var stop = false;
            try
            {
                double maxDx = 0, maxDf = 0;
                for (var k = 1; k <= n; k++)
                {
                    for (var j = 0; j < n; j++) maxDx = Math.Max(maxDx, Math.Abs(sim[k][j] - sim[0][j]));
                    maxDf = Math.Max(maxDf, Math.Abs(fsim[0] - fsim[k]));
                }
                if (maxDx <= xatol && maxDf <= fatol) break;

                Array.Clear(xbar);
                for (var k = 0; k < n; k++) for (var j = 0; j < n; j++) xbar[j] += sim[k][j];
                for (var j = 0; j < n; j++) xbar[j] /= n;
                var worst = sim[n];
                var xr = new double[n];
                for (var j = 0; j < n; j++) xr[j] = (1 + rho) * xbar[j] - rho * worst[j];
                Clip(xr);
                var fxr = bf.Eval(xr);
                var doshrink = false;
                if (fxr < fsim[0])
                {
                    var xe = new double[n];
                    for (var j = 0; j < n; j++) xe[j] = (1 + rho * chi) * xbar[j] - rho * chi * worst[j];
                    Clip(xe);
                    var fxe = bf.Eval(xe);
                    if (fxe < fxr) { sim[n] = xe; fsim[n] = fxe; } else { sim[n] = xr; fsim[n] = fxr; }
                }
                else if (fxr < fsim[n - 1]) { sim[n] = xr; fsim[n] = fxr; }
                else
                {
                    if (fxr < fsim[n])
                    {
                        var xc = new double[n];
                        for (var j = 0; j < n; j++) xc[j] = (1 + psi * rho) * xbar[j] - psi * rho * worst[j];
                        Clip(xc);
                        var fxc = bf.Eval(xc);
                        if (fxc <= fxr) { sim[n] = xc; fsim[n] = fxc; } else doshrink = true;
                    }
                    else
                    {
                        var xcc = new double[n];
                        for (var j = 0; j < n; j++) xcc[j] = (1 - psi) * xbar[j] + psi * worst[j];
                        Clip(xcc);
                        var fxcc = bf.Eval(xcc);
                        if (fxcc < fsim[n]) { sim[n] = xcc; fsim[n] = fxcc; } else doshrink = true;
                    }
                    if (doshrink)
                        for (var k = 1; k <= n; k++)
                        {
                            for (var j = 0; j < n; j++) sim[k][j] = sim[0][j] + sigma * (sim[k][j] - sim[0][j]);
                            Clip(sim[k]);
                            fsim[k] = bf.Eval(sim[k]);
                        }
                }
                iterations++;
            }
            catch (OptimizeMaxFevException) { }
            finally
            {
                Sort(sim, fsim);
                if (p.NotifyCallback(sim[0], fsim[0])) stop = true;
            }
            if (stop) break;
        }

        var fval = fsim.Min();
        int warn; string msg;
        if (bf.Calls >= maxfun) { warn = 1; msg = OptimizeMessages.MaxFev; }
        else if (iterations >= maxiter) { warn = 2; msg = OptimizeMessages.MaxIter; }
        else { warn = 0; msg = OptimizeMessages.Success; }
        var simArr = new double[(n + 1) * n];
        for (var k = 0; k <= n; k++) sim[k].CopyTo(simArr, k * n);
        return new OptimizeResult
        {
            X = (double[])sim[0].Clone(), Fun = fval, Nit = iterations, Nfev = bf.Calls, Status = warn, Success = warn == 0,
            Message = msg, FinalSimplex = new NdArray(simArr, n + 1, n),
        };
    }

    private static void Sort(double[][] sim, double[] fsim)
    {
        // Stable, so equal values keep their order (NumPy's default sort is not, which only matters on exact ties).
        var idx = Enumerable.Range(0, fsim.Length).OrderBy(i => fsim[i], Comparer<double>.Create((a, b) => a.CompareTo(b))).ToArray();
        var s2 = idx.Select(i => sim[i]).ToArray();
        var f2 = idx.Select(i => fsim[i]).ToArray();
        s2.CopyTo(sim, 0); f2.CopyTo(fsim, 0);
    }
}

/// <summary>SciPy's status messages.</summary>
internal static class OptimizeMessages
{
    public const string Success = "Optimization terminated successfully.";
    public const string MaxFev = "Maximum number of function evaluations has been exceeded.";
    public const string MaxIter = "Maximum number of iterations has been exceeded.";
    public const string PrLoss = "Desired error not necessarily achieved due to precision loss.";
    public const string Nan = "NaN result encountered.";
    public const string OutOfBounds = "The result is outside of the provided bounds.";
}

/// <summary>A port of SciPy's <c>_minimize_powell</c> with its Brent / bounded line searches.</summary>
internal static class OptimizePowell
{
    public static OptimizeResult Run(MinimizeProblem p)
    {
        var o = p.OptionsAs<PowellOptions>("Powell");
        var xtol = o.Xtol ?? p.Tol ?? 1e-4;
        var ftol = o.Ftol ?? p.Tol ?? 1e-4;
        var x = (double[])p.X0.Clone();
        var n = x.Length;
        double maxiter, maxfun;
        if (o.MaxIter is null && o.MaxFev is null) { maxiter = n * 1000; maxfun = n * 1000; }
        else if (o.MaxIter is null) { maxfun = o.MaxFev!.Value; maxiter = double.PositiveInfinity; }
        else if (o.MaxFev is null) { maxiter = o.MaxIter.Value; maxfun = double.PositiveInfinity; }
        else { maxiter = o.MaxIter.Value; maxfun = o.MaxFev.Value; }

        var bf = new OptimizeBudgetFunction(p.Fun, p.FunAndGrad, maxfun);
        var direc = new double[n][];
        if (o.Direc is null) for (var i = 0; i < n; i++) { direc[i] = new double[n]; direc[i][i] = 1; }
        else
        {
            if (o.Direc.Rank != 2 || o.Direc.Shape[0] != n || o.Direc.Shape[1] != n) throw new ArgumentException($"Direc must have shape ({n}, {n}).");
            var d = o.Direc.AsContiguous().ToArray();
            for (var i = 0; i < n; i++) direc[i] = d.AsSpan(i * n, n).ToArray();
        }
        var lb = p.Bounds?.Lb; var ub = p.Bounds?.Ub;

        double fval;
        try { fval = bf.Eval(x); }
        catch (OptimizeMaxFevException) { fval = double.NaN; }
        var x1 = (double[])x.Clone();
        var iter = 0;
        while (true)
        {
            try
            {
                var fx = fval;
                var bigind = 0;
                double delta = 0;
                for (var i = 0; i < n; i++)
                {
                    var fx2a = fval;
                    (fval, x, _) = LineSearch(bf, x, direc[i], xtol * 100, lb, ub, fval);
                    if (fx2a - fval > delta) { delta = fx2a - fval; bigind = i; }
                }
                iter++;
                if (p.NotifyCallback(x, fval)) break;
                var bnd = ftol * (Math.Abs(fx) + Math.Abs(fval)) + 1e-20;
                if (2.0 * (fx - fval) <= bnd) break;
                if (bf.Calls >= maxfun) break;
                if (iter >= maxiter) break;
                if (double.IsNaN(fx) && double.IsNaN(fval)) break;

                var direc1 = new double[n];
                for (var j = 0; j < n; j++) direc1[j] = x[j] - x1[j];
                x1 = (double[])x.Clone();
                var lmax = 1.0;
                if (lb is not null) lmax = LineForSearch(x, direc1, lb, ub!).Max;
                var x2 = new double[n];
                var step = Math.Min(lmax, 1);
                for (var j = 0; j < n; j++) x2[j] = x[j] + step * direc1[j];
                var fx2 = bf.Eval(x2);
                if (fx > fx2)
                {
                    var t = 2.0 * (fx + fx2 - 2.0 * fval);
                    var temp = fx - fval - delta;
                    t *= temp * temp;
                    temp = fx - fx2;
                    t -= delta * temp * temp;
                    if (t < 0.0)
                    {
                        (fval, x, direc1) = LineSearch(bf, x, direc1, xtol * 100, lb, ub, fval);
                        if (direc1.Any(v => v != 0))
                        {
                            direc[bigind] = direc[n - 1];
                            direc[n - 1] = direc1;
                        }
                    }
                }
            }
            catch (OptimizeMaxFevException) { break; }
        }

        var warn = 0; var msg = OptimizeMessages.Success;
        if (bf.Calls >= maxfun) { warn = 1; msg = OptimizeMessages.MaxFev; }
        else if (iter >= maxiter) { warn = 2; msg = OptimizeMessages.MaxIter; }
        else if (lb is not null && !p.Bounds!.Contains(x)) { warn = 4; msg = OptimizeMessages.OutOfBounds; }
        if (double.IsNaN(fval) || x.Any(double.IsNaN)) { warn = 3; msg = OptimizeMessages.Nan; }
        var dArr = new double[n * n];
        for (var i = 0; i < n; i++) direc[i].CopyTo(dArr, i * n);
        return new OptimizeResult
        {
            X = x, Fun = fval, Nit = iter, Nfev = bf.Calls, Status = warn, Success = warn == 0, Message = msg,
            Direc = new NdArray(dArr, n, n),
        };
    }

    /// <summary>SciPy's <c>_line_for_search</c>: the interval of t keeping x + t·d inside the box.</summary>
    public static (double Min, double Max) LineForSearch(double[] x, double[] d, double[] lb, double[] ub)
    {
        double lmin = double.NegativeInfinity, lmax = double.PositiveInfinity;
        for (var i = 0; i < x.Length; i++)
        {
            if (d[i] == 0) continue;
            var low = (lb[i] - x[i]) / d[i];
            var high = (ub[i] - x[i]) / d[i];
            if (d[i] > 0) { lmin = Math.Max(lmin, low); lmax = Math.Min(lmax, high); }
            else { lmin = Math.Max(lmin, high); lmax = Math.Min(lmax, low); }
        }
        return lmax >= lmin ? (lmin, lmax) : (0, 0);
    }

    private static (double F, double[] X, double[] Xi) LineSearch(OptimizeBudgetFunction bf, double[] p, double[] xi, double tol,
        double[]? lb, double[]? ub, double fval)
    {
        var n = p.Length;
        if (xi.All(v => v == 0)) return (fval, p, xi);
        double F(double alpha)
        {
            var x = new double[n];
            for (var i = 0; i < n; i++) x[i] = p[i] + alpha * xi[i];
            return bf.Eval(x);
        }
        (double F, double[] X, double[] Xi) Finish(double alpha, double f)
        {
            var step = new double[n];
            var x = new double[n];
            for (var i = 0; i < n; i++) { step[i] = alpha * xi[i]; x[i] = p[i] + step[i]; }
            return (f, x, step);
        }
        if (lb is null) goto unbounded;
        var (lo, hi) = LineForSearch(p, xi, lb, ub!);
        if (double.IsNegativeInfinity(lo) && double.IsPositiveInfinity(hi)) goto unbounded;
        if (!double.IsInfinity(lo) && !double.IsInfinity(hi))
        {
            var r = Optimize.Fminbound(F, lo, hi, tol / 100, 500);
            return Finish(r.X, r.Fun);
        }
        else
        {
            // One-sided: map through arctan so the interval is finite.
            var r = Optimize.Fminbound(t => F(Math.Tan(t)), Math.Atan(lo), Math.Atan(hi), tol / 100, 500);
            return Finish(Math.Tan(r.X), r.Fun);
        }
    unbounded:
        var res = Optimize.BrentScalar(F, null, tol, 500, recover: true);
        return Finish(res.X, res.Fun);
    }
}
