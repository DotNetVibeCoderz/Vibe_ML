using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Everything a local minimiser receives from <see cref="Optimize.Minimize(Func{double[], double}, double[], MinimizeMethod?, Func{double[], double[]}?, Func{double[], NdArray}?, Func{double[], double[], double[]}?, Bounds?, IReadOnlyList{IMinimizeConstraint}?, double?, MinimizeOptions?, OptimizeCallback?, FiniteDiffScheme?)"/>
/// after validation: x0 copied (and clipped into the bounds), fixed variables possibly removed.
/// </summary>
internal sealed class MinimizeProblem
{
    public required Func<double[], double>? Fun { get; init; }
    public required Func<double[], double[]>? Jac { get; init; }
    public required Func<double[], double[], double>? FunAndGrad { get; init; }
    public required Func<double[], NdArray>? Hess { get; init; }
    public required Func<double[], double[], double[]>? Hessp { get; init; }
    public required double[] X0 { get; init; }
    public required Bounds? Bounds { get; init; }
    public required IReadOnlyList<IMinimizeConstraint> Constraints { get; init; }
    public required double? Tol { get; init; }
    public required MinimizeOptions? Options { get; init; }
    public required OptimizeCallback? Callback { get; init; }

    /// <summary>
    /// The scheme the caller asked for explicitly; null means SciPy's <c>jac=None</c> behaviour, a
    /// forward difference with the method's absolute step (its <c>eps</c> option).
    /// </summary>
    public required FiniteDiffScheme? FiniteDiff { get; init; }

    public int N => X0.Length;

    /// <summary>Hessian evaluations made through <see cref="EvalHess"/>/<see cref="EvalHessp"/>.</summary>
    public int Nhev { get; set; }

    /// <summary>
    /// Builds the counted objective. <paramref name="absStep"/> is the method's default absolute
    /// finite-difference step (ignored when an explicit scheme was requested, which uses SciPy's
    /// relative steps instead). Finite differences respect the bounds when <paramref name="useBounds"/>.
    /// </summary>
    public OptimizeScalarFunction CreateObjective(double? absStep = 1.4901161193847656e-08, bool useBounds = true, double? relStep = null)
    {
        var scheme = FiniteDiff ?? FiniteDiffScheme.TwoPoint;
        var abs = FiniteDiff is null ? absStep : null;
        return new OptimizeScalarFunction(N, Fun, Jac, FunAndGrad, scheme, abs, relStep,
            useBounds ? Bounds?.Lb : null, useBounds ? Bounds?.Ub : null);
    }

    /// <summary>The method's options cast to the expected record, or a default instance.</summary>
    public T OptionsAs<T>(string methodName) where T : MinimizeOptions, new()
    {
        if (Options is null) return new T();
        if (Options is T t) return t;
        if (Options.GetType() == typeof(MinimizeOptions)) return new T { MaxIter = Options.MaxIter, Disp = Options.Disp };
        throw new ArgumentException($"{methodName} expects options of type {typeof(T).Name}, not {Options.GetType().Name}.");
    }

    public NdArray EvalHess(double[] x)
    {
        Nhev++;
        var h = Hess!((double[])x.Clone());
        if (h.Rank != 2 || h.Shape[0] != N || h.Shape[1] != N) throw new ArgumentException($"The Hessian must have shape ({N}, {N}).");
        return h;
    }

    public double[] EvalHessp(double[] x, double[] p)
    {
        Nhev++;
        var r = Hessp!((double[])x.Clone(), (double[])p.Clone());
        if (r.Length != N) throw new ArgumentException($"hessp returned {r.Length} values but x has {N}.");
        return r;
    }

    /// <summary>Invokes the callback; true means the user asked to stop.</summary>
    public bool NotifyCallback(double[] x, double f)
    {
        if (Callback is null || !Callback((double[])x.Clone(), f)) return false;
        CallbackStopped = true;
        return true;
    }

    /// <summary>Set when the callback asked to stop; Minimize then reports status 99.</summary>
    public bool CallbackStopped { get; private set; }
}

public static partial class Optimize
{
    // Each method lives in its own file and implements one of these hooks. A missing
    // implementation leaves the result null, which Minimize reports as unsupported.
    static partial void RunNelderMead(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunPowell(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunCG(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunBfgs(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunNewtonCG(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunLBfgsB(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunTnc(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunCobyla(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunSlsqp(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunTrustConstr(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunDogleg(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunTrustNcg(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunTrustExact(MinimizeProblem problem, ref OptimizeResult? result);
    static partial void RunTrustKrylov(MinimizeProblem problem, ref OptimizeResult? result);

    /// <summary>
    /// Minimises a scalar function of one or more variables (<c>scipy.optimize.minimize</c>).
    /// </summary>
    /// <param name="fun">The objective. It receives a private copy of x.</param>
    /// <param name="x0">Initial guess (not modified).</param>
    /// <param name="method">Defaults as SciPy: SLSQP with constraints, L-BFGS-B with bounds, otherwise BFGS.</param>
    /// <param name="jac">Gradient. When null, a forward difference is used (see <paramref name="finiteDiff"/>).</param>
    /// <param name="hess">Hessian for Newton-type methods (NewtonCG, Dogleg, Trust*).</param>
    /// <param name="hessp">Hessian-vector product, an alternative to <paramref name="hess"/>.</param>
    /// <param name="bounds">Box bounds, for methods that support them (NelderMead, Powell, LBfgsB, Tnc, Slsqp, Cobyla, TrustConstr); other methods throw.</param>
    /// <param name="constraints">Constraints for Slsqp, Cobyla and TrustConstr; other methods throw.</param>
    /// <param name="tol">Termination tolerance, mapped onto each method's options exactly as SciPy maps <c>tol</c>.</param>
    /// <param name="options">Method options; pass the method's derived record (e.g. <c>BfgsOptions</c>).</param>
    /// <param name="callback">Called after each iteration; return true to stop.</param>
    /// <param name="finiteDiff">
    /// Explicit finite-difference scheme with SciPy's relative steps (<c>jac='2-point'</c> /
    /// <c>'3-point'</c>). Null keeps SciPy's <c>jac=None</c> behaviour: a forward difference with the
    /// method's absolute <c>eps</c>.
    /// </param>
    public static OptimizeResult Minimize(
        Func<double[], double> fun,
        double[] x0,
        MinimizeMethod? method = null,
        Func<double[], double[]>? jac = null,
        Func<double[], NdArray>? hess = null,
        Func<double[], double[], double[]>? hessp = null,
        Bounds? bounds = null,
        IReadOnlyList<IMinimizeConstraint>? constraints = null,
        double? tol = null,
        MinimizeOptions? options = null,
        OptimizeCallback? callback = null,
        FiniteDiffScheme? finiteDiff = null)
    {
        ArgumentNullException.ThrowIfNull(fun);
        return MinimizeCore(fun, null, jac, x0, method, hess, hessp, bounds, constraints, tol, options, callback, finiteDiff);
    }

    /// <summary>
    /// <see cref="Minimize(Func{double[], double}, double[], MinimizeMethod?, Func{double[], double[]}?, Func{double[], NdArray}?, Func{double[], double[], double[]}?, Bounds?, IReadOnlyList{IMinimizeConstraint}?, double?, MinimizeOptions?, OptimizeCallback?, FiniteDiffScheme?)"/>
    /// with a combined objective that returns f(x) and writes ∇f(x) into its second argument
    /// (SciPy's <c>jac=True</c>). Shared work between f and ∇f is done once, and the gradient
    /// buffer is supplied by the solver.
    /// </summary>
    public static OptimizeResult Minimize(
        Func<double[], double[], double> funAndGrad,
        double[] x0,
        MinimizeMethod? method = null,
        Func<double[], NdArray>? hess = null,
        Func<double[], double[], double[]>? hessp = null,
        Bounds? bounds = null,
        IReadOnlyList<IMinimizeConstraint>? constraints = null,
        double? tol = null,
        MinimizeOptions? options = null,
        OptimizeCallback? callback = null)
    {
        ArgumentNullException.ThrowIfNull(funAndGrad);
        return MinimizeCore(null, funAndGrad, null, x0, method, hess, hessp, bounds, constraints, tol, options, callback, null);
    }

    private static OptimizeResult MinimizeCore(
        Func<double[], double>? fun, Func<double[], double[], double>? funGrad, Func<double[], double[]>? jac,
        double[] x0, MinimizeMethod? method, Func<double[], NdArray>? hess, Func<double[], double[], double[]>? hessp,
        Bounds? bounds, IReadOnlyList<IMinimizeConstraint>? constraints, double? tol, MinimizeOptions? options,
        OptimizeCallback? callback, FiniteDiffScheme? finiteDiff)
    {
        ArgumentNullException.ThrowIfNull(x0);
        if (x0.Length == 0) throw new ArgumentException("x0 must have at least one element.");
        foreach (var v in x0) if (!double.IsFinite(v)) throw new ArgumentException("x0 must be finite.");
        constraints ??= [];
        var m = method ?? (constraints.Count > 0 ? MinimizeMethod.Slsqp : bounds is not null ? MinimizeMethod.LBfgsB : MinimizeMethod.Bfgs);

        if (constraints.Count > 0 && m is not (MinimizeMethod.Cobyla or MinimizeMethod.Slsqp or MinimizeMethod.TrustConstr))
            throw new ArgumentException($"Method {m} cannot handle constraints; use Slsqp, Cobyla or TrustConstr.");
        if (bounds is not null && m is not (MinimizeMethod.NelderMead or MinimizeMethod.Powell or MinimizeMethod.LBfgsB
                or MinimizeMethod.Cobyla or MinimizeMethod.Slsqp or MinimizeMethod.Tnc or MinimizeMethod.TrustConstr))
            throw new ArgumentException($"Method {m} cannot handle bounds.");
        if (m is MinimizeMethod.NewtonCG or MinimizeMethod.Dogleg or MinimizeMethod.TrustNcg or MinimizeMethod.TrustExact or MinimizeMethod.TrustKrylov
            && jac is null && funGrad is null)
            throw new ArgumentException($"Jacobian (gradient) is required for method {m}.");
        if (m is MinimizeMethod.Dogleg or MinimizeMethod.TrustExact && hess is null)
            throw new ArgumentException($"Hessian is required for method {m}.");
        if (m is MinimizeMethod.TrustNcg or MinimizeMethod.TrustKrylov && hess is null && hessp is null)
            throw new ArgumentException($"Either the Hessian or the Hessian-vector product is required for method {m}.");
        foreach (var c in constraints) if (c is null) throw new ArgumentException("constraints contains a null entry.");

        var n = x0.Length;
        var x = (double[])x0.Clone();
        if (bounds is not null)
        {
            if (bounds.Length != n) throw new ArgumentException($"Bounds have {bounds.Length} entries but x0 has {n}.");
            bounds.Clip(x);
        }

        // SciPy removes variables fixed by lb == ub for the methods whose finite differences would
        // otherwise step outside a zero-width box (and always for TNC).
        if (bounds is not null && m is MinimizeMethod.LBfgsB or MinimizeMethod.Tnc or MinimizeMethod.Slsqp)
        {
            var fixedMask = new bool[n];
            var nFixed = 0;
            for (var i = 0; i < n; i++) if (bounds!.Lb[i] == bounds.Ub[i]) { fixedMask[i] = true; nFixed++; }
            var fdNeeded = jac is null && funGrad is null;
            foreach (var c in constraints)
                if (c is NonlinearConstraint { Jac: null } || c is OptimizeConstraint { Jac: null }) fdNeeded = true;
            if (nFixed == n) return AllFixed(fun, funGrad, bounds!, constraints);
            if (nFixed > 0 && (fdNeeded || m == MinimizeMethod.Tnc))
                return MinimizeReduced(fun, funGrad, jac, x, fixedMask, m, hess, hessp, bounds!, constraints, tol, options, callback, finiteDiff);
        }

        var problem = new MinimizeProblem
        {
            Fun = fun, Jac = jac, FunAndGrad = funGrad, Hess = hess, Hessp = hessp, X0 = x, Bounds = bounds,
            Constraints = constraints, Tol = tol, Options = options, Callback = callback, FiniteDiff = finiteDiff,
        };
        OptimizeResult? result = null;
        switch (m)
        {
            case MinimizeMethod.NelderMead: RunNelderMead(problem, ref result); break;
            case MinimizeMethod.Powell: RunPowell(problem, ref result); break;
            case MinimizeMethod.CG: RunCG(problem, ref result); break;
            case MinimizeMethod.Bfgs: RunBfgs(problem, ref result); break;
            case MinimizeMethod.NewtonCG: RunNewtonCG(problem, ref result); break;
            case MinimizeMethod.LBfgsB: RunLBfgsB(problem, ref result); break;
            case MinimizeMethod.Tnc: RunTnc(problem, ref result); break;
            case MinimizeMethod.Cobyla: RunCobyla(problem, ref result); break;
            case MinimizeMethod.Slsqp: RunSlsqp(problem, ref result); break;
            case MinimizeMethod.TrustConstr: RunTrustConstr(problem, ref result); break;
            case MinimizeMethod.Dogleg: RunDogleg(problem, ref result); break;
            case MinimizeMethod.TrustNcg: RunTrustNcg(problem, ref result); break;
            case MinimizeMethod.TrustExact: RunTrustExact(problem, ref result); break;
            case MinimizeMethod.TrustKrylov: RunTrustKrylov(problem, ref result); break;
            default: throw new ArgumentOutOfRangeException(nameof(method));
        }
        if (result is null) throw new NotSupportedException($"Minimize method {m} is not available in this build.");
        if (result.Nhev == 0) result.Nhev = problem.Nhev;
        if (problem.CallbackStopped)
        {
            result.Success = false;
            result.Status = 99;
            result.Message = "`callback` raised `StopIteration`.";
        }
        return result;
    }

    private static OptimizeResult AllFixed(Func<double[], double>? fun, Func<double[], double[], double>? funGrad, Bounds bounds, IReadOnlyList<IMinimizeConstraint> constraints)
    {
        var x = (double[])bounds.Lb.Clone();
        var f = fun is not null ? fun((double[])x.Clone()) : funGrad!((double[])x.Clone(), new double[x.Length]);
        var violation = OptimizeConstraintViolation(constraints, x);
        var feasible = violation <= 1e-8 * Math.Max(1, OptimizeDense.NormInf(x));
        return new OptimizeResult
        {
            X = x, Fun = f, Nfev = 1, Success = feasible, Status = feasible ? 0 : 4,
            Message = feasible ? "All independent variables were fixed by bounds." : "All independent variables were fixed by bounds, but the result violates the constraints.",
            ConstrViolation = constraints.Count > 0 ? violation : null,
        };
    }

    /// <summary>Maximum violation of a constraint list at x (0 when feasible), as SciPy's <c>constr_violation</c>.</summary>
    internal static double OptimizeConstraintViolation(IReadOnlyList<IMinimizeConstraint> constraints, double[] x)
    {
        double worst = 0;
        foreach (var c in constraints)
        {
            switch (c)
            {
                case LinearConstraint lc:
                {
                    var rows = lc.A.Shape[0]; var cols = lc.A.Shape[1];
                    var a = lc.A.AsContiguous().AsSpan();
                    var ax = new double[rows];
                    OptimizeDense.MatVec(a, rows, cols, x, ax);
                    for (var i = 0; i < rows; i++) worst = Math.Max(worst, Math.Max(lc.Lb[i] - ax[i], ax[i] - lc.Ub[i]));
                    break;
                }
                case NonlinearConstraint nc:
                {
                    var v = nc.Fun((double[])x.Clone());
                    for (var i = 0; i < v.Length; i++) worst = Math.Max(worst, Math.Max(nc.Lb[i] - v[i], v[i] - nc.Ub[i]));
                    break;
                }
                case OptimizeConstraint oc:
                {
                    var v = oc.Fun((double[])x.Clone());
                    for (var i = 0; i < v.Length; i++)
                        worst = Math.Max(worst, oc.Type == OptimizeConstraintType.Eq ? Math.Abs(v[i]) : -v[i]);
                    break;
                }
            }
        }
        return Math.Max(0, worst);
    }

    private static OptimizeResult MinimizeReduced(Func<double[], double>? fun, Func<double[], double[], double>? funGrad, Func<double[], double[]>? jac,
        double[] x, bool[] fixedMask, MinimizeMethod m, Func<double[], NdArray>? hess, Func<double[], double[], double[]>? hessp,
        Bounds bounds, IReadOnlyList<IMinimizeConstraint> constraints, double? tol, MinimizeOptions? options, OptimizeCallback? callback,
        FiniteDiffScheme? finiteDiff)
    {
        var n = x.Length;
        var free = new List<int>();
        for (var i = 0; i < n; i++) if (!fixedMask[i]) free.Add(i);
        var nf = free.Count;
        var template = (double[])x.Clone();
        for (var i = 0; i < n; i++) if (fixedMask[i]) template[i] = bounds.Lb[i];

        double[] Expand(double[] r)
        {
            var full = (double[])template.Clone();
            for (var k = 0; k < nf; k++) full[free[k]] = r[k];
            return full;
        }
        double[] Reduce(double[] full)
        {
            var r = new double[nf];
            for (var k = 0; k < nf; k++) r[k] = full[free[k]];
            return r;
        }
        NdArray ReduceColumns(NdArray a)
        {
            var mat = a.Rank == 1 ? a.Reshape(1, a.Size) : a;
            var rows = mat.Shape[0];
            var src = mat.AsContiguous().ToArray();
            var dst = new double[rows * nf];
            for (var r = 0; r < rows; r++) for (var k = 0; k < nf; k++) dst[r * nf + k] = src[r * n + free[k]];
            return new NdArray(dst, rows, nf);
        }

        Func<double[], double>? rf = fun is null ? null : r => fun(Expand(r));
        Func<double[], double[], double>? rfg = funGrad is null ? null : (r, g) =>
        {
            var gf = new double[n];
            var f = funGrad(Expand(r), gf);
            for (var k = 0; k < nf; k++) g[k] = gf[free[k]];
            return f;
        };
        Func<double[], double[]>? rj = jac is null ? null : r => Reduce(jac(Expand(r)));
        Func<double[], NdArray>? rh = null;
        if (hess is not null)
            rh = r =>
            {
                var h = hess(Expand(r)).AsContiguous().ToArray();
                var d = new double[nf * nf];
                for (var a = 0; a < nf; a++) for (var b = 0; b < nf; b++) d[a * nf + b] = h[free[a] * n + free[b]];
                return new NdArray(d, nf, nf);
            };
        Func<double[], double[], double[]>? rhp = hessp is null ? null : (r, p) =>
        {
            var pf = new double[n];
            for (var k = 0; k < nf; k++) pf[free[k]] = p[k];
            return Reduce(hessp(Expand(r), pf));
        };

        var rc = new List<IMinimizeConstraint>(constraints.Count);
        foreach (var c in constraints)
        {
            switch (c)
            {
                case LinearConstraint lc:
                {
                    var rows = lc.A.Shape[0];
                    var a = lc.A.AsContiguous().ToArray();
                    var shift = new double[rows];
                    for (var r = 0; r < rows; r++)
                        for (var i = 0; i < n; i++) if (fixedMask[i]) shift[r] += a[r * n + i] * template[i];
                    var lb = new double[rows]; var ub = new double[rows];
                    for (var r = 0; r < rows; r++) { lb[r] = lc.Lb[r] - shift[r]; ub[r] = lc.Ub[r] - shift[r]; }
                    rc.Add(new LinearConstraint(ReduceColumns(lc.A), lb, ub, lc.KeepFeasible));
                    break;
                }
                case NonlinearConstraint nc:
                    rc.Add(new NonlinearConstraint(r => nc.Fun(Expand(r)), nc.Lb, nc.Ub,
                        nc.Jac is null ? null : r => ReduceColumns(nc.Jac(Expand(r))), nc.KeepFeasible));
                    break;
                case OptimizeConstraint oc:
                    rc.Add(new OptimizeConstraint(oc.Type, r => oc.Fun(Expand(r)),
                        oc.Jac is null ? null : r => ReduceColumns(oc.Jac(Expand(r)))));
                    break;
                default:
                    throw new ArgumentException($"Unsupported constraint type {c.GetType().Name}.");
            }
        }

        OptimizeCallback? rcb = callback is null ? null : (r, f) => callback(Expand(r), f);
        var rb = new Bounds(Reduce(bounds.Lb), Reduce(bounds.Ub), bounds.KeepFeasible);
        var res = MinimizeCore(rf, rfg, rj, Reduce(x), m, rh, rhp, rb, rc, tol, options, rcb, finiteDiff);

        res.X = Expand(res.X);
        if (res.Jac is { } rjac)
        {
            var full = new double[n];
            Array.Fill(full, double.NaN);
            for (var k = 0; k < nf; k++) full[free[k]] = rjac[k];
            res.Jac = full;
        }
        if (res.ActiveMask is { } am)
        {
            var full = new int[n];
            for (var k = 0; k < nf; k++) full[free[k]] = am[k];
            for (var i = 0; i < n; i++) if (fixedMask[i]) full[i] = -1;
            res.ActiveMask = full;
        }
        res.HessInv = null;
        res.Hess = null;
        return res;
    }
}

/// <summary>
/// Input to the MINPACK <c>lmder</c>/<c>lmdif</c> Levenberg–Marquardt core, which is shared by
/// <c>Optimize.Root(method: Lm)</c> and <c>Optimize.LeastSquares(method: Lm)</c>.
/// </summary>
internal sealed class MinpackLmProblem
{
    /// <summary>Residual function f: Rⁿ → Rᵐ (m ≥ n).</summary>
    public required Func<double[], double[]> Fun { get; init; }
    /// <summary>Jacobian (m, n), or null for forward differences with step √epsfcn·|x|.</summary>
    public Func<double[], NdArray>? Jac { get; init; }
    /// <summary>Starting point.</summary>
    public required double[] X0 { get; init; }
    /// <summary>Relative reduction in the sum of squares at which to stop.</summary>
    public double Ftol { get; init; } = 1.49012e-08;
    /// <summary>Relative change in x at which to stop.</summary>
    public double Xtol { get; init; } = 1.49012e-08;
    /// <summary>Orthogonality between f and the Jacobian columns at which to stop.</summary>
    public double Gtol { get; init; }
    /// <summary>Maximum function evaluations (0 = MINPACK's default 100(n+1), or 200(n+1) without a Jacobian).</summary>
    public int MaxFev { get; init; }
    /// <summary>Step-length hint for the forward difference (null = machine epsilon).</summary>
    public double? Epsfcn { get; init; }
    /// <summary>Initial step bound factor.</summary>
    public double Factor { get; init; } = 100.0;
    /// <summary>Variable scaling; null = automatic scaling from the Jacobian column norms.</summary>
    public double[]? Diag { get; init; }
}

public static partial class Optimize
{
    /// <summary>
    /// The MINPACK LM core. Implemented in the least-squares files; the result carries
    /// <c>X</c>, <c>FunVector</c> (residuals), <c>Fun</c> = ½‖f‖², <c>JacMatrix</c> (final Jacobian),
    /// <c>Nfev</c>, <c>Njev</c>, <c>Status</c> = MINPACK <c>info</c>, <c>Message</c>, <c>Success</c>.
    /// </summary>
    static partial void RunMinpackLm(MinpackLmProblem problem, ref OptimizeResult? result);

    /// <summary>Runs the MINPACK core or throws when it is not compiled in.</summary>
    internal static OptimizeResult MinpackLm(MinpackLmProblem problem)
    {
        OptimizeResult? r = null;
        RunMinpackLm(problem, ref r);
        return r ?? throw new NotSupportedException("The MINPACK Levenberg–Marquardt core is not available in this build.");
    }
}
