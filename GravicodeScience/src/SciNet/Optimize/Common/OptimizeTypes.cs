using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Called once per iteration by the minimisers with the current iterate and its objective value.
/// Return <c>true</c> to stop the solver early (SciPy's <c>StopIteration</c>); the result then has
/// <see cref="OptimizeResult.Status"/> 99 and <see cref="OptimizeResult.Success"/> false.
/// </summary>
/// <param name="x">The current iterate. Do not keep a reference: the solver may reuse the buffer.</param>
/// <param name="fun">Objective value at <paramref name="x"/>.</param>
public delegate bool OptimizeCallback(double[] x, double fun);

/// <summary>The local minimisers available through <c>Optimize.Minimize</c>.</summary>
public enum MinimizeMethod
{
    /// <summary>Adaptive Nelder–Mead simplex; derivative-free, bounds by clipping.</summary>
    NelderMead,
    /// <summary>Powell's conjugate direction method with Brent line searches; derivative-free.</summary>
    Powell,
    /// <summary>Nonlinear conjugate gradient (Polak–Ribière+).</summary>
    CG,
    /// <summary>Quasi-Newton BFGS with a strong-Wolfe line search.</summary>
    Bfgs,
    /// <summary>Newton conjugate gradient (truncated Newton, line search).</summary>
    NewtonCG,
    /// <summary>Limited-memory BFGS with bounds (Byrd–Lu–Nocedal–Zhu).</summary>
    LBfgsB,
    /// <summary>Truncated Newton with bounds.</summary>
    Tnc,
    /// <summary>Powell's COBYLA: linear approximations in a trust region, inequality constraints.</summary>
    Cobyla,
    /// <summary>Sequential least squares programming (Kraft).</summary>
    Slsqp,
    /// <summary>Trust-region constrained algorithm.</summary>
    TrustConstr,
    /// <summary>Dog-leg trust region; needs a positive-definite Hessian.</summary>
    Dogleg,
    /// <summary>Trust-region Newton conjugate gradient (Steihaug–Toint).</summary>
    TrustNcg,
    /// <summary>Nearly exact trust-region subproblem solution (Moré–Sorensen).</summary>
    TrustExact,
    /// <summary>Trust region with a Lanczos/Krylov subproblem solver (GLTR).</summary>
    TrustKrylov,
}

/// <summary>Finite-difference scheme used when no analytic derivative is supplied.</summary>
public enum FiniteDiffScheme
{
    /// <summary>Forward (or backward, near an upper bound) difference; one extra evaluation per variable.</summary>
    TwoPoint,
    /// <summary>Central difference; two evaluations per variable and roughly twice the accurate digits.</summary>
    ThreePoint,
}

/// <summary>
/// Marker for constraints accepted by <c>Optimize.Minimize</c>:
/// <see cref="LinearConstraint"/>, <see cref="NonlinearConstraint"/> and the legacy dictionary form
/// <see cref="OptimizeConstraint"/>.
/// </summary>
public interface IMinimizeConstraint { }

/// <summary>
/// Box bounds <c>lb &lt;= x &lt;= ub</c>. Infinite entries mean "unbounded on that side".
/// </summary>
public sealed class Bounds
{
    /// <summary>Lower bounds, one per variable (may contain <see cref="double.NegativeInfinity"/>).</summary>
    public double[] Lb { get; }

    /// <summary>Upper bounds, one per variable (may contain <see cref="double.PositiveInfinity"/>).</summary>
    public double[] Ub { get; }

    /// <summary>
    /// When true, solvers that support it (TrustConstr, finite differences) keep every evaluation
    /// strictly feasible. Mirrors SciPy's <c>keep_feasible</c>.
    /// </summary>
    public bool KeepFeasible { get; }

    /// <summary>Creates bounds from lower and upper arrays of equal length.</summary>
    public Bounds(double[] lb, double[] ub, bool keepFeasible = false)
    {
        ArgumentNullException.ThrowIfNull(lb);
        ArgumentNullException.ThrowIfNull(ub);
        if (lb.Length != ub.Length) throw new ArgumentException("Lower and upper bounds must have the same length.");
        for (var i = 0; i < lb.Length; i++)
        {
            if (double.IsNaN(lb[i]) || double.IsNaN(ub[i])) throw new ArgumentException("Bounds must not be NaN.");
            if (lb[i] > ub[i]) throw new ArgumentException($"Lower bound {lb[i]} exceeds upper bound {ub[i]} for variable {i}.");
        }
        Lb = (double[])lb.Clone();
        Ub = (double[])ub.Clone();
        KeepFeasible = keepFeasible;
    }

    /// <summary>The same scalar interval for each of <paramref name="n"/> variables.</summary>
    public static Bounds Uniform(int n, double lb, double ub)
    {
        var l = new double[n]; var u = new double[n];
        Array.Fill(l, lb); Array.Fill(u, ub);
        return new Bounds(l, u);
    }

    /// <summary>
    /// SciPy's legacy sequence of <c>(min, max)</c> pairs; <c>null</c> on either side means unbounded.
    /// </summary>
    public static Bounds FromPairs(params (double? Lower, double? Upper)[] pairs)
    {
        var l = new double[pairs.Length]; var u = new double[pairs.Length];
        for (var i = 0; i < pairs.Length; i++)
        {
            l[i] = pairs[i].Lower ?? double.NegativeInfinity;
            u[i] = pairs[i].Upper ?? double.PositiveInfinity;
        }
        return new Bounds(l, u);
    }

    /// <summary>Number of variables.</summary>
    public int Length => Lb.Length;

    /// <summary>True when every bound is infinite, i.e. the problem is effectively unbounded.</summary>
    public bool IsUnbounded
    {
        get
        {
            for (var i = 0; i < Lb.Length; i++)
                if (!double.IsNegativeInfinity(Lb[i]) || !double.IsPositiveInfinity(Ub[i])) return false;
            return true;
        }
    }

    /// <summary>Clips <paramref name="x"/> into the box in place and returns it.</summary>
    public double[] Clip(double[] x)
    {
        for (var i = 0; i < x.Length; i++) x[i] = Math.Clamp(x[i], Lb[i], Ub[i]);
        return x;
    }

    /// <summary>True when <paramref name="x"/> lies in the box.</summary>
    public bool Contains(ReadOnlySpan<double> x)
    {
        for (var i = 0; i < x.Length; i++) if (x[i] < Lb[i] || x[i] > Ub[i]) return false;
        return true;
    }
}

/// <summary>
/// Linear constraint <c>lb &lt;= A x &lt;= ub</c>. Equal entries in <c>lb</c> and <c>ub</c> give an
/// equality; infinite entries drop that side.
/// </summary>
public sealed class LinearConstraint : IMinimizeConstraint
{
    /// <summary>The constraint matrix, shape <c>(m, n)</c>.</summary>
    public NdArray A { get; }

    /// <summary>Lower bounds on <c>A x</c>, length <c>m</c>.</summary>
    public double[] Lb { get; }

    /// <summary>Upper bounds on <c>A x</c>, length <c>m</c>.</summary>
    public double[] Ub { get; }

    /// <summary>Whether iterates must satisfy the constraint (honoured by TrustConstr only).</summary>
    public bool KeepFeasible { get; }

    /// <summary>Creates <c>lb &lt;= A x &lt;= ub</c>. A 1-D <paramref name="a"/> is one row.</summary>
    public LinearConstraint(NdArray a, double[] lb, double[] ub, bool keepFeasible = false)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Rank == 1) a = a.Reshape(1, a.Size);
        if (a.Rank != 2) throw new ArgumentException("LinearConstraint expects a 1-D or 2-D matrix.");
        var m = a.Shape[0];
        if (lb.Length != m || ub.Length != m) throw new ArgumentException($"lb and ub must have {m} entries, one per row of A.");
        A = a.AsContiguous();
        Lb = (double[])lb.Clone();
        Ub = (double[])ub.Clone();
        KeepFeasible = keepFeasible;
    }

    /// <summary>Scalar bounds broadcast over every row.</summary>
    public LinearConstraint(NdArray a, double lb, double ub, bool keepFeasible = false)
        : this(a, Filled(a.Rank == 1 ? 1 : a.Shape[0], lb), Filled(a.Rank == 1 ? 1 : a.Shape[0], ub), keepFeasible) { }

    private static double[] Filled(int n, double v) { var r = new double[n]; Array.Fill(r, v); return r; }
}

/// <summary>
/// Nonlinear constraint <c>lb &lt;= fun(x) &lt;= ub</c>. Without <see cref="Jac"/> the Jacobian is
/// estimated by finite differences.
/// </summary>
public sealed class NonlinearConstraint : IMinimizeConstraint
{
    /// <summary>The constraint function, returning <c>m</c> values.</summary>
    public Func<double[], double[]> Fun { get; }

    /// <summary>Lower bounds on <c>fun(x)</c>.</summary>
    public double[] Lb { get; }

    /// <summary>Upper bounds on <c>fun(x)</c>.</summary>
    public double[] Ub { get; }

    /// <summary>Jacobian of <see cref="Fun"/>, shape <c>(m, n)</c>; null for finite differences.</summary>
    public Func<double[], NdArray>? Jac { get; }

    /// <summary>Whether iterates must satisfy the constraint (honoured by TrustConstr only).</summary>
    public bool KeepFeasible { get; }

    /// <summary>Creates <c>lb &lt;= fun(x) &lt;= ub</c>.</summary>
    public NonlinearConstraint(Func<double[], double[]> fun, double[] lb, double[] ub, Func<double[], NdArray>? jac = null, bool keepFeasible = false)
    {
        ArgumentNullException.ThrowIfNull(fun);
        if (lb.Length != ub.Length) throw new ArgumentException("lb and ub must have the same length.");
        Fun = fun; Lb = (double[])lb.Clone(); Ub = (double[])ub.Clone(); Jac = jac; KeepFeasible = keepFeasible;
    }

    /// <summary>
    /// Hessian of <c>Σ vᵢ funᵢ(x)</c> as a function of <c>(x, v)</c>, shape <c>(n, n)</c>, used by
    /// TrustConstr. When null TrustConstr approximates it with <see cref="HessianUpdate"/> (BFGS by
    /// default, as SciPy).
    /// </summary>
    public Func<double[], double[], NdArray>? Hess { get; init; }

    /// <summary>
    /// Quasi-Newton strategy for the constraint Hessian when <see cref="Hess"/> is null (TrustConstr).
    /// Null means a fresh <see cref="BfgsHessianUpdate"/>. The instance is re-initialised by each solve.
    /// </summary>
    public HessianUpdateStrategy? HessianUpdate { get; init; }

    /// <summary>Relative finite-difference step for the Jacobian (TrustConstr); null is √ε.</summary>
    public double? FiniteDiffRelStep { get; init; }

    /// <summary>A scalar constraint <c>lb &lt;= fun(x) &lt;= ub</c>.</summary>
    public NonlinearConstraint(Func<double[], double> fun, double lb, double ub, Func<double[], double[]>? jac = null)
        : this(x => [fun(x)], [lb], [ub], jac is null ? null : x => { var g = jac(x); return new NdArray(g, 1, g.Length); }) { }
}

/// <summary>Kind of a legacy dictionary-style constraint.</summary>
public enum OptimizeConstraintType
{
    /// <summary><c>fun(x) == 0</c>.</summary>
    Eq,
    /// <summary><c>fun(x) &gt;= 0</c> (SciPy's convention: non-negative means satisfied).</summary>
    Ineq,
}

/// <summary>
/// SciPy's legacy dictionary constraint <c>{'type': 'eq'|'ineq', 'fun': ..., 'jac': ...}</c>, as used
/// by SLSQP and COBYLA. Inequalities are <c>fun(x) &gt;= 0</c>.
/// </summary>
public sealed class OptimizeConstraint : IMinimizeConstraint
{
    /// <summary>Equality or inequality.</summary>
    public OptimizeConstraintType Type { get; }

    /// <summary>Constraint values; an equality requires all zero, an inequality all non-negative.</summary>
    public Func<double[], double[]> Fun { get; }

    /// <summary>Jacobian of <see cref="Fun"/>, shape <c>(m, n)</c>; null for finite differences.</summary>
    public Func<double[], NdArray>? Jac { get; }

    /// <summary>Creates a vector-valued constraint.</summary>
    public OptimizeConstraint(OptimizeConstraintType type, Func<double[], double[]> fun, Func<double[], NdArray>? jac = null)
    {
        ArgumentNullException.ThrowIfNull(fun);
        Type = type; Fun = fun; Jac = jac;
    }

    /// <summary>Creates a scalar constraint (the common case in the SciPy tutorials).</summary>
    public OptimizeConstraint(OptimizeConstraintType type, Func<double[], double> fun, Func<double[], double[]>? jac = null)
        : this(type, x => [fun(x)], jac is null ? null : x => { var g = jac(x); return new NdArray(g, 1, g.Length); }) { }

    /// <summary>Shorthand for an equality constraint.</summary>
    public static OptimizeConstraint Eq(Func<double[], double> fun, Func<double[], double[]>? jac = null) => new(OptimizeConstraintType.Eq, fun, jac);

    /// <summary>Shorthand for an inequality constraint <c>fun(x) &gt;= 0</c>.</summary>
    public static OptimizeConstraint Ineq(Func<double[], double> fun, Func<double[], double[]>? jac = null) => new(OptimizeConstraintType.Ineq, fun, jac);
}

/// <summary>
/// Options shared by every minimiser. Each method has a derived record (for example
/// <c>LBfgsBOptions</c>) adding its own settings; passing the wrong derived type throws.
/// </summary>
public record MinimizeOptions
{
    /// <summary>Maximum number of iterations; null selects the method's default.</summary>
    public int? MaxIter { get; init; }

    /// <summary>Print convergence messages to the console (rarely useful outside debugging).</summary>
    public bool Disp { get; init; }
}

/// <summary>
/// The result of an optimisation, mirroring <c>scipy.optimize.OptimizeResult</c>. Properties a method
/// does not produce stay null (or zero for counters).
/// </summary>
public sealed class OptimizeResult
{
    /// <summary>The solution.</summary>
    public double[] X { get; set; } = [];

    /// <summary>Objective value at <see cref="X"/>.</summary>
    public double Fun { get; set; }

    /// <summary>
    /// Vector-valued objective at <see cref="X"/> for methods whose function is vector-valued
    /// (<c>Root</c>: the residuals).
    /// </summary>
    public double[]? FunVector { get; set; }

    /// <summary>Gradient (or Jacobian for <c>Root</c>, flattened in <see cref="JacMatrix"/>) at <see cref="X"/>.</summary>
    public double[]? Jac { get; set; }

    /// <summary>Jacobian matrix for vector-valued problems.</summary>
    public NdArray? JacMatrix { get; set; }

    /// <summary>Hessian at <see cref="X"/>, when the method has it.</summary>
    public NdArray? Hess { get; set; }

    /// <summary>Approximate inverse Hessian (BFGS).</summary>
    public NdArray? HessInv { get; set; }

    /// <summary>Number of iterations.</summary>
    public int Nit { get; set; }

    /// <summary>Objective evaluations, including those spent on finite differences.</summary>
    public int Nfev { get; set; }

    /// <summary>Gradient/Jacobian evaluations (analytic or finite-difference).</summary>
    public int Njev { get; set; }

    /// <summary>Hessian (or Hessian-vector product) evaluations.</summary>
    public int Nhev { get; set; }

    /// <summary>True when the method reports convergence.</summary>
    public bool Success { get; set; }

    /// <summary>Method-specific termination code; 0 conventionally means success.</summary>
    public int Status { get; set; }

    /// <summary>Human-readable termination reason.</summary>
    public string Message { get; set; } = "";

    /// <summary>
    /// For bounded methods: −1 where a variable sits on its lower bound, +1 on its upper bound,
    /// 0 when free.
    /// </summary>
    public int[]? ActiveMask { get; set; }

    /// <summary>Maximum constraint violation at <see cref="X"/> (constrained methods).</summary>
    public double? ConstrViolation { get; set; }

    /// <summary>Lagrange multipliers of the constraints, in the order they were given (SLSQP, TrustConstr).</summary>
    public double[]? Multipliers { get; set; }

    /// <summary>Final simplex for Nelder–Mead: <c>(n+1, n)</c> vertices.</summary>
    public NdArray? FinalSimplex { get; set; }

    /// <summary>Final direction set for Powell.</summary>
    public NdArray? Direc { get; set; }

    /// <summary>Lowest local minimum found by global methods, when different from <see cref="X"/> bookkeeping.</summary>
    public OptimizeResult? LowestOptimizationResult { get; set; }

    /// <summary>Population of a population-based global method at termination.</summary>
    public NdArray? Population { get; set; }

    /// <summary>Objective values of <see cref="Population"/>.</summary>
    public double[]? PopulationEnergies { get; set; }

    /// <summary>Local minimisation results attached by a global method's polishing step.</summary>
    public int Nlfev { get; set; }

    /// <summary>Objective gradient at <see cref="X"/> (TrustConstr's <c>grad</c>).</summary>
    public double[]? Grad { get; set; }

    /// <summary>
    /// Gradient of the Lagrangian <c>∇f + Σ Jᵢᵀ vᵢ</c> at <see cref="X"/> (TrustConstr); its infinity
    /// norm is <see cref="Optimality"/>.
    /// </summary>
    public double[]? LagrangianGrad { get; set; }

    /// <summary>
    /// Lagrange multipliers per constraint, in the order given, with the bounds (when present) as a
    /// final entry — SciPy's <c>v</c>. Sign convention: the Lagrangian gradient is <c>∇f + Jᵀv</c>, so
    /// an active lower bound has a negative multiplier.
    /// </summary>
    public double[][]? V { get; set; }

    /// <summary>Constraint values per constraint at <see cref="X"/> (TrustConstr's <c>constr</c>), bounds last.</summary>
    public double[][]? Constr { get; set; }

    /// <summary>Infinity norm of <see cref="LagrangianGrad"/> (TrustConstr).</summary>
    public double? Optimality { get; set; }

    /// <summary>Final trust-region radius (TrustConstr).</summary>
    public double? TrRadius { get; set; }

    /// <summary>Final penalty parameter of the merit function (TrustConstr).</summary>
    public double? ConstrPenalty { get; set; }

    /// <summary>Final barrier parameter (TrustConstr, interior-point branch only).</summary>
    public double? BarrierParameter { get; set; }

    /// <summary>Final barrier subproblem tolerance (TrustConstr, interior-point branch only).</summary>
    public double? BarrierTolerance { get; set; }

    /// <summary>Total projected-CG iterations (TrustConstr).</summary>
    public int CgNiter { get; set; }

    /// <summary>Why the last projected-CG solve stopped: 1 max iterations, 2 trust boundary, 3 negative curvature, 4 tolerance (TrustConstr).</summary>
    public int CgStopCond { get; set; }

    /// <summary>Wall-clock time in seconds (TrustConstr).</summary>
    public double? ExecutionTime { get; set; }

    /// <summary>Which algorithm ran, e.g. <c>"equality_constrained_sqp"</c> or <c>"tr_interior_point"</c>.</summary>
    public string? Method { get; set; }

    /// <summary>All local minima found by <c>Optimize.Shgo</c>, one per row, sorted by objective value.</summary>
    public NdArray? Xl { get; set; }

    /// <summary>Objective values of <see cref="Xl"/>, ascending.</summary>
    public double[]? Funl { get; set; }

    /// <summary>Gradient evaluations made by local minimisations (Shgo).</summary>
    public int Nljev { get; set; }

    /// <summary>Sampling points evaluated (Shgo's <c>tnev</c>), including infeasible ones that were not evaluated.</summary>
    public int Tnev { get; set; }

    /// <inheritdoc />
    public override string ToString()
        => $"{(Success ? "success" : "failure")} (status {Status}): {Message}; fun={Fun:G10}, nit={Nit}, nfev={Nfev}, njev={Njev}, x=[{string.Join(", ", X.Select(v => v.ToString("G8")))}]";
}
