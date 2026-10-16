namespace Gravicode.Science.SciNet;

/// <summary>
/// A counted, cached scalar objective with an analytic or finite-difference gradient, the
/// equivalent of SciPy's <c>ScalarFunction</c>. Re-evaluating at the point last seen returns the
/// cached value without calling the user (and without counting), which is what makes a combined
/// fun-and-grad callback cheap: the line search asks for f and g at the same point separately.
/// </summary>
/// <remarks>
/// Every call hands the user a fresh copy of <c>x</c>, so a callback that memoises by reference
/// is safe. The finite-difference evaluations count towards <see cref="Nfev"/> and each gradient
/// (analytic or estimated) towards <see cref="Ngev"/>, exactly as SciPy reports them.
/// </remarks>
internal sealed class OptimizeScalarFunction
{
    private readonly Func<double[], double>? _fun;
    private readonly Func<double[], double[]>? _grad;
    private readonly Func<double[], double[], double>? _funGrad;
    private readonly FiniteDiffScheme _scheme;
    private readonly double? _absStep;
    private readonly double? _relStep;
    private readonly double[]? _lb, _ub;

    private double[]? _fx;     // point of the cached value
    private double _fval;
    private double[]? _gx;     // point of the cached gradient
    private double[]? _gval;

    public int N { get; }
    public int Nfev { get; set; }
    public int Ngev { get; set; }

    /// <summary>True when the gradient is analytic (user-supplied).</summary>
    public bool HasAnalyticGradient => _grad is not null || _funGrad is not null;

    public OptimizeScalarFunction(int n, Func<double[], double>? fun, Func<double[], double[]>? grad,
        Func<double[], double[], double>? funGrad, FiniteDiffScheme scheme, double? absStep, double? relStep,
        double[]? lb, double[]? ub)
    {
        if (fun is null && funGrad is null) throw new ArgumentNullException(nameof(fun));
        N = n; _fun = fun; _grad = grad; _funGrad = funGrad; _scheme = scheme; _absStep = absStep; _relStep = relStep;
        _lb = lb; _ub = ub;
    }

    private static bool Same(double[]? cached, ReadOnlySpan<double> x)
        => cached is not null && x.SequenceEqual(cached);

    /// <summary>The raw objective, counted, without caching (used by finite differences).</summary>
    private double Raw(double[] x)
    {
        Nfev++;
        if (_fun is not null) return _fun(x);
        var g = new double[x.Length];
        var f = _funGrad!(x, g);
        // A combined callback computes the gradient anyway: keep it.
        _gx = (double[])x.Clone(); _gval = g;
        return f;
    }

    /// <summary>f(x), cached.</summary>
    public double Fun(ReadOnlySpan<double> x)
    {
        if (Same(_fx, x)) return _fval;
        var xc = x.ToArray();
        var f = Raw(xc);
        _fx = (double[])xc.Clone(); _fval = f;
        if (_funGrad is not null && _fun is null) Ngev++;
        return f;
    }

    /// <summary>Writes ∇f(x) into <paramref name="g"/>, cached.</summary>
    public void Grad(ReadOnlySpan<double> x, Span<double> g)
    {
        if (Same(_gx, x)) { _gval.AsSpan().CopyTo(g); return; }
        if (_grad is not null)
        {
            var r = _grad(x.ToArray());
            if (r.Length != x.Length) throw new ArgumentException($"The gradient has {r.Length} entries but x has {x.Length}.");
            Ngev++;
            _gx = x.ToArray(); _gval = (double[])r.Clone();
            r.AsSpan().CopyTo(g);
            return;
        }
        if (_funGrad is not null)
        {
            var xc = x.ToArray();
            var gg = new double[x.Length];
            var f = _funGrad(xc, gg);
            Nfev++; Ngev++;
            _fx = (double[])xc.Clone(); _fval = f;
            _gx = xc; _gval = gg;
            gg.AsSpan().CopyTo(g);
            return;
        }
        var f0 = Fun(x);
        Ngev++;
        Nfev += OptimizeFiniteDiff.Gradient(_fun!, x, f0, g, _scheme, _absStep, _relStep, _lb, _ub);
        _gx = x.ToArray(); _gval = g.ToArray();
    }

    /// <summary>f(x) and ∇f(x) together.</summary>
    public double FunAndGrad(ReadOnlySpan<double> x, Span<double> g)
    {
        if (_funGrad is not null && _fun is null && !Same(_fx, x))
        {
            Grad(x, g);
            return _fval;
        }
        var f = Fun(x);
        Grad(x, g);
        return f;
    }

    /// <summary>Gradient as a new array.</summary>
    public double[] Grad(ReadOnlySpan<double> x)
    {
        var g = new double[x.Length];
        Grad(x, g);
        return g;
    }

    /// <summary>A plain delegate view of the counted objective, for passing to helper routines.</summary>
    public Func<double[], double> AsFunc() => x => Fun(x);

    /// <summary>
    /// Hessian-vector product estimated by differencing gradients, <c>(∇f(x + h p) − ∇f(x)) / h</c>,
    /// as SciPy does for Newton-CG when no Hessian is given.
    /// </summary>
    public void HessVecFiniteDiff(ReadOnlySpan<double> x, ReadOnlySpan<double> gx, ReadOnlySpan<double> p, Span<double> hp)
    {
        var pn = OptimizeDense.Norm2(p);
        if (pn == 0) { hp.Clear(); return; }
        var xn = OptimizeDense.Norm2(x);
        var h = 1.4901161193847656e-08 * (1 + xn) / pn;
        var xp = x.ToArray();
        for (var i = 0; i < xp.Length; i++) xp[i] += h * p[i];
        var g2 = new double[x.Length];
        // Use the uncached path so the cache keeps the iterate, not the probe point.
        var saveFx = _fx; var saveF = _fval; var saveGx = _gx; var saveG = _gval;
        Grad(xp, g2);
        _fx = saveFx; _fval = saveF; _gx = saveGx; _gval = saveG;
        for (var i = 0; i < hp.Length; i++) hp[i] = (g2[i] - gx[i]) / h;
    }
}
