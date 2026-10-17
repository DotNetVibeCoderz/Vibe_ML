using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Thrown by the <c>scipy.optimize.nonlin</c>-style solvers (<see cref="Optimize.Broyden1"/>,
/// <see cref="Optimize.NewtonKrylov"/>, ...) when they stop without converging, as SciPy raises
/// <c>NoConvergence</c>. The best iterate is attached.
/// </summary>
public sealed class OptimizeNoConvergenceException : Exception
{
    /// <summary>The last iterate.</summary>
    public double[] X { get; }

    /// <summary>The solver's full result.</summary>
    public OptimizeResult Result { get; }

    /// <summary>Creates the exception from a failed result.</summary>
    public OptimizeNoConvergenceException(OptimizeResult result)
        : base($"The nonlinear solver did not converge: {result.Message}")
    {
        Result = result;
        X = (double[])result.X.Clone();
    }
}

/// <summary>Result of <see cref="Optimize.Leastsq"/> (SciPy returns the tuple <c>(x, cov_x, infodict, mesg, ier)</c>).</summary>
/// <param name="X">Solution.</param>
/// <param name="CovX">(JᵀJ)⁻¹ at the solution, or null when JᵀJ is singular. Multiply by the residual variance to get the parameter covariance.</param>
/// <param name="Fvec">Residuals at the solution.</param>
/// <param name="Nfev">Function evaluations.</param>
/// <param name="Ier">MINPACK info: 1–4 mean a solution was found.</param>
/// <param name="Mesg">Termination message.</param>
public sealed record LeastsqResult(double[] X, NdArray? CovX, double[] Fvec, int Nfev, int Ier, string Mesg);

/// <summary>
/// The limited-memory BFGS inverse Hessian as an operator (<c>scipy.optimize.LbfgsInvHessProduct</c>):
/// <c>H = Vₖᵀ…V₁ᵀ H₀ V₁…Vₖ + …</c> with H₀ = I, applied by the two-loop recursion (Nocedal &amp; Wright
/// Algorithm 7.4) without forming the matrix.
/// </summary>
public sealed class LbfgsInvHessProduct
{
    private readonly double[][] _s, _y;
    private readonly double[] _rho;

    /// <summary>Problem dimension.</summary>
    public int N { get; }

    /// <summary>Builds the operator from correction pairs sₖ = xₖ₊₁ − xₖ, yₖ = ∇fₖ₊₁ − ∇fₖ, oldest first.</summary>
    public LbfgsInvHessProduct(double[][] sk, double[][] yk)
    {
        ArgumentNullException.ThrowIfNull(sk);
        ArgumentNullException.ThrowIfNull(yk);
        if (sk.Length != yk.Length || sk.Length == 0) throw new ArgumentException("sk and yk must hold the same, non-zero number of correction pairs.");
        N = sk[0].Length;
        _s = sk.Select(v => (double[])v.Clone()).ToArray();
        _y = yk.Select(v => (double[])v.Clone()).ToArray();
        _rho = new double[sk.Length];
        for (var i = 0; i < sk.Length; i++)
        {
            if (_s[i].Length != N || _y[i].Length != N) throw new ArgumentException("All correction vectors must have the same length.");
            _rho[i] = 1.0 / OptimizeDense.Dot(_s[i], _y[i]);
        }
    }

    /// <summary>H·x.</summary>
    public double[] Matvec(double[] x)
    {
        if (x.Length != N) throw new ArgumentException($"x must have length {N}.");
        var k = _s.Length;
        var q = (double[])x.Clone();
        var alpha = new double[k];
        for (var i = k - 1; i >= 0; i--)
        {
            alpha[i] = _rho[i] * OptimizeDense.Dot(_s[i], q);
            OptimizeDense.Axpy(-alpha[i], _y[i], q);
        }
        for (var i = 0; i < k; i++)
        {
            var beta = _rho[i] * OptimizeDense.Dot(_y[i], q);
            OptimizeDense.Axpy(alpha[i] - beta, _s[i], q);
        }
        return q;
    }

    /// <summary>Hᵀ·x (H is symmetric, so the same as <see cref="Matvec"/>).</summary>
    public double[] Rmatvec(double[] x) => Matvec(x);

    /// <summary>The dense (n, n) matrix.</summary>
    public NdArray ToDense()
    {
        var h = new double[N * N];
        var e = new double[N];
        for (var j = 0; j < N; j++)
        {
            Array.Clear(e);
            e[j] = 1;
            var col = Matvec(e);
            for (var i = 0; i < N; i++) h[i * N + j] = col[i];
        }
        return new NdArray(h, N, N);
    }
}

public static partial class Optimize
{
    // ---------- scalar wrappers ----------

    /// <summary>Brent's method (<c>scipy.optimize.brent</c>) from a bracket (null: SciPy's default search from (0, 1)).</summary>
    public static MinimizeScalarResult Brent(Func<double, double> f, double[]? brack = null, double tol = 1.48e-8, int maxiter = 500)
    {
        ArgumentNullException.ThrowIfNull(f);
        return BrentScalar(f, brack, tol, maxiter);
    }

    /// <summary>Golden-section search (<c>scipy.optimize.golden</c>).</summary>
    public static MinimizeScalarResult Golden(Func<double, double> f, double[]? brack = null, double tol = 1.4901161193847656e-08, int maxiter = 5000)
    {
        ArgumentNullException.ThrowIfNull(f);
        return GoldenScalar(f, brack, tol, maxiter);
    }

    // ---------- legacy fmin_* wrappers ----------
    // SciPy's functional interfaces predate minimize(); each is a thin call into Minimize with the
    // legacy keyword names mapped onto the method's options. They return the OptimizeResult rather
    // than SciPy's positional tuple.

    /// <summary>Nelder–Mead (<c>scipy.optimize.fmin</c>).</summary>
    public static OptimizeResult Fmin(Func<double[], double> f, double[] x0, double xtol = 1e-4, double ftol = 1e-4, int? maxiter = null,
        int? maxfun = null, OptimizeCallback? callback = null)
        => Minimize(f, x0, MinimizeMethod.NelderMead, options: new NelderMeadOptions { Xatol = xtol, Fatol = ftol, MaxIter = maxiter, MaxFev = maxfun }, callback: callback);

    /// <summary>Powell's method (<c>scipy.optimize.fmin_powell</c>).</summary>
    public static OptimizeResult FminPowell(Func<double[], double> f, double[] x0, double xtol = 1e-4, double ftol = 1e-4, int? maxiter = null,
        int? maxfun = null, NdArray? direc = null, OptimizeCallback? callback = null)
        => Minimize(f, x0, MinimizeMethod.Powell, options: new PowellOptions { Xtol = xtol, Ftol = ftol, MaxIter = maxiter, MaxFev = maxfun, Direc = direc }, callback: callback);

    /// <summary>Nonlinear conjugate gradient (<c>scipy.optimize.fmin_cg</c>).</summary>
    public static OptimizeResult FminCg(Func<double[], double> f, double[] x0, Func<double[], double[]>? fprime = null, double gtol = 1e-5,
        double norm = double.PositiveInfinity, double epsilon = 1.4901161193847656e-08, int? maxiter = null, OptimizeCallback? callback = null)
        => Minimize(f, x0, MinimizeMethod.CG, fprime, options: new CgOptions { Gtol = gtol, Norm = norm, Eps = epsilon, MaxIter = maxiter }, callback: callback);

    /// <summary>BFGS (<c>scipy.optimize.fmin_bfgs</c>).</summary>
    public static OptimizeResult FminBfgs(Func<double[], double> f, double[] x0, Func<double[], double[]>? fprime = null, double gtol = 1e-5,
        double norm = double.PositiveInfinity, double epsilon = 1.4901161193847656e-08, int? maxiter = null, OptimizeCallback? callback = null)
        => Minimize(f, x0, MinimizeMethod.Bfgs, fprime, options: new BfgsOptions { Gtol = gtol, Norm = norm, Eps = epsilon, MaxIter = maxiter }, callback: callback);

    /// <summary>Newton-CG (<c>scipy.optimize.fmin_ncg</c>); <paramref name="avextol"/> is the average relative change in x.</summary>
    public static OptimizeResult FminNcg(Func<double[], double> f, double[] x0, Func<double[], double[]> fprime, Func<double[], double[], double[]>? fhessP = null,
        Func<double[], NdArray>? fhess = null, double avextol = 1e-5, double epsilon = 1.4901161193847656e-08, int? maxiter = null, OptimizeCallback? callback = null)
        => Minimize(f, x0, MinimizeMethod.NewtonCG, fprime, fhess, fhessP, options: new NewtonCgOptions { Xtol = avextol, Eps = epsilon, MaxIter = maxiter }, callback: callback);

    /// <summary>
    /// L-BFGS-B (<c>scipy.optimize.fmin_l_bfgs_b</c>). <paramref name="factr"/> is in units of machine
    /// epsilon (ftol = factr·ε: 1e12 low accuracy, 1e7 moderate, 10 extreme).
    /// </summary>
    public static OptimizeResult FminLBfgsB(Func<double[], double> f, double[] x0, Func<double[], double[]>? fprime = null, Bounds? bounds = null,
        int m = 10, double factr = 1e7, double pgtol = 1e-5, double epsilon = 1e-8, int maxfun = 15000, int maxiter = 15000, int maxls = 20,
        OptimizeCallback? callback = null)
        => Minimize(f, x0, MinimizeMethod.LBfgsB, fprime, bounds: bounds ?? new Bounds(Fill(x0.Length, double.NegativeInfinity), Fill(x0.Length, double.PositiveInfinity)),
            options: new LBfgsBOptions { M = m, Ftol = factr * 2.220446049250313e-16, Gtol = pgtol, Eps = epsilon, MaxFun = maxfun, MaxIter = maxiter, MaxLs = maxls },
            callback: callback);

    /// <summary>Truncated Newton with bounds (<c>scipy.optimize.fmin_tnc</c>).</summary>
    public static OptimizeResult FminTnc(Func<double[], double> f, double[] x0, Func<double[], double[]>? fprime = null, Bounds? bounds = null,
        TncOptions? options = null, OptimizeCallback? callback = null)
        => Minimize(f, x0, MinimizeMethod.Tnc, fprime, bounds: bounds ?? new Bounds(Fill(x0.Length, double.NegativeInfinity), Fill(x0.Length, double.PositiveInfinity)),
            options: options, callback: callback);

    /// <summary>
    /// SLSQP (<c>scipy.optimize.fmin_slsqp</c>) with scalar equality (<c>== 0</c>) and inequality
    /// (<c>&gt;= 0</c>) constraint functions.
    /// </summary>
    public static OptimizeResult FminSlsqp(Func<double[], double> f, double[] x0, IReadOnlyList<Func<double[], double>>? eqcons = null,
        IReadOnlyList<Func<double[], double>>? ieqcons = null, Bounds? bounds = null, Func<double[], double[]>? fprime = null,
        int iter = 100, double acc = 1e-6, double epsilon = 1.4901161193847656e-08, OptimizeCallback? callback = null)
    {
        var cons = new List<IMinimizeConstraint>();
        foreach (var c in eqcons ?? []) cons.Add(OptimizeConstraint.Eq(c));
        foreach (var c in ieqcons ?? []) cons.Add(OptimizeConstraint.Ineq(c));
        return Minimize(f, x0, MinimizeMethod.Slsqp, fprime, bounds: bounds, constraints: cons,
            options: new SlsqpOptions { MaxIter = iter, Ftol = acc, Eps = epsilon }, callback: callback);
    }

    /// <summary>COBYLA (<c>scipy.optimize.fmin_cobyla</c>); each constraint must be <c>&gt;= 0</c>.</summary>
    public static OptimizeResult FminCobyla(Func<double[], double> f, double[] x0, IReadOnlyList<Func<double[], double>> cons, double rhobeg = 1.0,
        double rhoend = 1e-4, int maxfun = 1000, double? catol = null, OptimizeCallback? callback = null)
    {
        ArgumentNullException.ThrowIfNull(cons);
        return Minimize(f, x0, MinimizeMethod.Cobyla, constraints: cons.Select(c => (IMinimizeConstraint)OptimizeConstraint.Ineq(c)).ToList(),
            options: new CobylaOptions { Rhobeg = rhobeg, Tol = rhoend, MaxIter = maxfun, Catol = catol }, callback: callback);
    }

    private static double[] Fill(int n, double v) { var r = new double[n]; Array.Fill(r, v); return r; }

    // ---------- leastsq ----------

    /// <summary>
    /// MINPACK Levenberg–Marquardt least squares (<c>scipy.optimize.leastsq</c>): minimises ‖func(x)‖²
    /// with <c>lmder</c> (when <paramref name="dfun"/> is given) or <c>lmdif</c>.
    /// </summary>
    /// <param name="func">Residuals, m ≥ n of them.</param>
    /// <param name="x0">Starting point.</param>
    /// <param name="dfun">Jacobian (m, n); forward differences when null.</param>
    /// <param name="ftol">Relative reduction in the sum of squares.</param>
    /// <param name="xtol">Relative change in x.</param>
    /// <param name="gtol">Orthogonality between residuals and Jacobian columns.</param>
    /// <param name="maxfev">Evaluation limit (0: 200(n+1) without, 100(n+1) with a Jacobian).</param>
    /// <param name="epsfcn">Forward-difference step hint.</param>
    /// <param name="factor">Initial step bound factor.</param>
    /// <param name="diag">Variable scale factors.</param>
    public static LeastsqResult Leastsq(Func<double[], double[]> func, double[] x0, Func<double[], NdArray>? dfun = null, double ftol = 1.49012e-08,
        double xtol = 1.49012e-08, double gtol = 0.0, int maxfev = 0, double? epsfcn = null, double factor = 100, double[]? diag = null)
    {
        ArgumentNullException.ThrowIfNull(func);
        ArgumentNullException.ThrowIfNull(x0);
        var n = x0.Length;
        var m = func((double[])x0.Clone()).Length;
        if (n > m) throw new ArgumentException($"Improper input: func (m={m}) must return at least as many values as there are parameters (n={n}).");
        var r = MinpackLm(new MinpackLmProblem
        {
            Fun = func, Jac = dfun, X0 = (double[])x0.Clone(), Ftol = ftol, Xtol = xtol, Gtol = gtol, MaxFev = maxfev,
            Epsfcn = epsfcn, Factor = factor, Diag = diag,
        });
        // MINPACK's core already forms cov_x = (RᵀR)⁻¹ from the final QR, as SciPy's leastsq does
        // (null when R is singular or the fit failed).
        var cov = r.HessInv;
        return new LeastsqResult(r.X, cov, r.FunVector ?? [], r.Nfev, r.Status, r.Message);
    }

    // ---------- scipy.optimize.nonlin functional interfaces ----------

    private static double[] NonlinSolve(Func<double[], double[]> f, double[] xin, RootMethod method, int? iter, double? alpha, int? maxiter,
        double? fTol, double? fRtol, double? xTol, double? xRtol, RootLineSearch lineSearch, Func<RootOptions, RootOptions>? extra = null)
    {
        ArgumentNullException.ThrowIfNull(f);
        var o = new RootOptions
        {
            Alpha = alpha, MaxIter = iter ?? maxiter, Fatol = fTol, Ftol = fRtol, Xatol = xTol, Xtol = xRtol, LineSearch = lineSearch,
        };
        if (extra is not null) o = extra(o);
        var r = Root(f, xin, method, options: o);
        // A fixed iteration count (SciPy's iter=) is not a failure.
        if (!r.Success && iter is null) throw new OptimizeNoConvergenceException(r);
        return r.X;
    }

    /// <summary>Broyden's first ("good") method (<c>scipy.optimize.broyden1</c>). Throws <see cref="OptimizeNoConvergenceException"/> on failure.</summary>
    public static double[] Broyden1(Func<double[], double[]> f, double[] xin, int? iter = null, double? alpha = null, int? maxiter = null,
        double? fTol = null, double? fRtol = null, double? xTol = null, double? xRtol = null, RootLineSearch lineSearch = RootLineSearch.Armijo)
        => NonlinSolve(f, xin, RootMethod.Broyden1, iter, alpha, maxiter, fTol, fRtol, xTol, xRtol, lineSearch);

    /// <summary>Broyden's second ("bad") method (<c>scipy.optimize.broyden2</c>).</summary>
    public static double[] Broyden2(Func<double[], double[]> f, double[] xin, int? iter = null, double? alpha = null, int? maxiter = null,
        double? fTol = null, double? fRtol = null, double? xTol = null, double? xRtol = null, RootLineSearch lineSearch = RootLineSearch.Armijo)
        => NonlinSolve(f, xin, RootMethod.Broyden2, iter, alpha, maxiter, fTol, fRtol, xTol, xRtol, lineSearch);

    /// <summary>Anderson mixing (<c>scipy.optimize.anderson</c>) with <paramref name="m"/> previous vectors and regularisation <paramref name="w0"/>.</summary>
    public static double[] Anderson(Func<double[], double[]> f, double[] xin, int? iter = null, double? alpha = null, double w0 = 0.01, int m = 5,
        int? maxiter = null, double? fTol = null, double? fRtol = null, double? xTol = null, double? xRtol = null, RootLineSearch lineSearch = RootLineSearch.Armijo)
        => NonlinSolve(f, xin, RootMethod.Anderson, iter, alpha, maxiter, fTol, fRtol, xTol, xRtol, lineSearch, o => o with { W0 = w0, AndersonM = m });

    /// <summary>Diagonal Broyden (<c>scipy.optimize.diagbroyden</c>).</summary>
    public static double[] DiagBroyden(Func<double[], double[]> f, double[] xin, int? iter = null, double? alpha = null, int? maxiter = null,
        double? fTol = null, double? fRtol = null, double? xTol = null, double? xRtol = null, RootLineSearch lineSearch = RootLineSearch.Armijo)
        => NonlinSolve(f, xin, RootMethod.DiagBroyden, iter, alpha, maxiter, fTol, fRtol, xTol, xRtol, lineSearch);

    /// <summary>Linear mixing, J ≈ −1/α (<c>scipy.optimize.linearmixing</c>).</summary>
    public static double[] LinearMixing(Func<double[], double[]> f, double[] xin, int? iter = null, double? alpha = null, int? maxiter = null,
        double? fTol = null, double? fRtol = null, double? xTol = null, double? xRtol = null, RootLineSearch lineSearch = RootLineSearch.Armijo)
        => NonlinSolve(f, xin, RootMethod.LinearMixing, iter, alpha, maxiter, fTol, fRtol, xTol, xRtol, lineSearch);

    /// <summary>Exciting mixing with per-component tuned steps up to <paramref name="alphamax"/> (<c>scipy.optimize.excitingmixing</c>).</summary>
    public static double[] ExcitingMixing(Func<double[], double[]> f, double[] xin, int? iter = null, double? alpha = null, double alphamax = 1.0,
        int? maxiter = null, double? fTol = null, double? fRtol = null, double? xTol = null, double? xRtol = null, RootLineSearch lineSearch = RootLineSearch.Armijo)
        => NonlinSolve(f, xin, RootMethod.ExcitingMixing, iter, alpha, maxiter, fTol, fRtol, xTol, xRtol, lineSearch, o => o with { AlphaMax = alphamax });

    /// <summary>Jacobian-free Newton–Krylov with GMRES inner solves (<c>scipy.optimize.newton_krylov</c>).</summary>
    public static double[] NewtonKrylov(Func<double[], double[]> f, double[] xin, int? iter = null, double? rdiff = null, int innerMaxiter = 20,
        int? maxiter = null, double? fTol = null, double? fRtol = null, double? xTol = null, double? xRtol = null, RootLineSearch lineSearch = RootLineSearch.Armijo)
        => NonlinSolve(f, xin, RootMethod.Krylov, iter, null, maxiter, fTol, fRtol, xTol, xRtol, lineSearch, o => o with { Rdiff = rdiff, InnerMaxIter = innerMaxiter });
}
