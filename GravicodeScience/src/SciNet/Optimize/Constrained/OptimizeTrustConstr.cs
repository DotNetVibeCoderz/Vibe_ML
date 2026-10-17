using System.Diagnostics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Options for <see cref="MinimizeMethod.TrustConstr"/> (<c>scipy.optimize.minimize(method='trust-constr')</c>).
/// Unset tolerances take <c>Minimize</c>'s <c>tol</c> when given, else 1e-8, exactly as SciPy maps <c>tol</c>.
/// </summary>
public sealed record TrustConstrOptions : MinimizeOptions
{
    /// <summary>Stop when the Lagrangian gradient's infinity norm and the constraint violation are both below this.</summary>
    public double? Gtol { get; init; }

    /// <summary>Stop when the trust radius falls below this (and, with inequalities, the barrier parameter below <see cref="BarrierTol"/>).</summary>
    public double? Xtol { get; init; }

    /// <summary>Barrier parameter threshold for the interior-point branch's xtol test.</summary>
    public double? BarrierTol { get; init; }

    /// <summary>Initial trust radius (in the scaled variables of the barrier subproblem).</summary>
    public double InitialTrRadius { get; init; } = 1.0;

    /// <summary>Initial penalty of the merit function f + μ‖c‖.</summary>
    public double InitialConstrPenalty { get; init; } = 1.0;

    /// <summary>Initial barrier parameter; decreased by a factor of 5 per barrier subproblem.</summary>
    public double InitialBarrierParameter { get; init; } = 0.1;

    /// <summary>Initial tolerance for the barrier subproblems; decreased with the barrier parameter.</summary>
    public double InitialBarrierTolerance { get; init; } = 0.1;

    /// <summary>Relative step for finite-difference gradients; null is √ε (2-point) or ε^⅓ (3-point).</summary>
    public double? FiniteDiffRelStep { get; init; }

    /// <summary>
    /// Quasi-Newton strategy for the objective's Hessian when neither <c>hess</c> nor <c>hessp</c> is
    /// supplied; null is a fresh <see cref="BfgsHessianUpdate"/> (SciPy's default).
    /// </summary>
    public HessianUpdateStrategy? HessianUpdate { get; init; }

    /// <summary>Verbosity level; accepted for SciPy compatibility and ignored (nothing is printed).</summary>
    public int Verbose { get; init; }
}

public static partial class Optimize
{
    static partial void RunTrustConstr(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeTrustConstr.Run(problem);
}

/// <summary>Objective wrapper with SciPy's <c>ScalarFunction</c> caching and quasi-Newton update semantics.</summary>
internal sealed class TrustConstrObjective
{
    private readonly MinimizeProblem _p;
    private readonly FiniteDiffScheme _scheme;
    private readonly double? _relStep;
    private readonly double[]? _fdLb, _fdUb;
    private readonly HessianUpdateStrategy? _hus;
    private readonly int _n;

    private double[] _x;
    private bool _fUpd, _gUpd, _hUpd;
    private double[]? _xPrev, _gPrev;
    private double[]? _hMat;                 // callable Hessian at _x
    private double[]? _memoX, _memoG;        // combined fun+grad cache

    public double F { get; private set; }
    public double[] G { get; private set; } = [];
    public int NfevRaw { get; private set; }
    public int NfevFd { get; private set; }
    public int Ngev { get; private set; }
    public int Nhev { get; private set; }
    public int Nfev => NfevRaw + NfevFd;

    public TrustConstrObjective(MinimizeProblem p, double[] x0, double? relStep, double[]? fdLb, double[]? fdUb, HessianUpdateStrategy? hus)
    {
        _p = p; _n = x0.Length; _x = (double[])x0.Clone();
        _scheme = p.FiniteDiff ?? FiniteDiffScheme.TwoPoint;
        _relStep = relStep; _fdLb = fdLb; _fdUb = fdUb;
        UpdateFun();
        UpdateGrad();
        if (p.Hess is null && p.Hessp is null)
        {
            _hus = hus ?? new BfgsHessianUpdate();
            _hus.Initialize(_n, HessianApproxType.Hess);
            _hUpd = true;
        }
        else
        {
            UpdateHess();
        }
    }

    public HessianUpdateStrategy? Strategy => _hus;

    private double RawFun(double[] x)
    {
        if (_p.Fun is not null) return _p.Fun((double[])x.Clone());
        var g = new double[_n];
        var f = _p.FunAndGrad!((double[])x.Clone(), g);
        _memoX = (double[])x.Clone(); _memoG = g;
        return f;
    }

    private void UpdateFun()
    {
        if (_fUpd) return;
        F = RawFun(_x);
        NfevRaw++;
        _fUpd = true;
    }

    private void UpdateGrad()
    {
        if (_gUpd) return;
        var g = new double[_n];
        if (_p.Jac is not null)
        {
            var r = _p.Jac((double[])_x.Clone());
            if (r.Length != _n) throw new ArgumentException($"The gradient has {r.Length} entries but x has {_n}.");
            r.AsSpan().CopyTo(g);
        }
        else if (_p.FunAndGrad is not null)
        {
            if (_memoX is null || !_memoX.AsSpan().SequenceEqual(_x))
            {
                var gg = new double[_n];
                _p.FunAndGrad((double[])_x.Clone(), gg);
                _memoX = (double[])_x.Clone(); _memoG = gg;
            }
            _memoG!.AsSpan().CopyTo(g);
        }
        else
        {
            UpdateFun();
            NfevFd += OptimizeFiniteDiff.Gradient(_p.Fun!, _x, F, g, _scheme, null, _relStep, _fdLb, _fdUb);
        }
        Ngev++;
        G = g;
        _gUpd = true;
    }

    private void UpdateHess()
    {
        if (_hUpd) return;
        if (_hus is not null)
        {
            UpdateGrad();
            var dx = new double[_n]; var dg = new double[_n];
            for (var i = 0; i < _n; i++) { dx[i] = _x[i] - _xPrev![i]; dg[i] = G[i] - _gPrev![i]; }
            _hus.Update(dx, dg);
        }
        else if (_p.Hess is not null)
        {
            _hMat = OptimizeDense.ToMatrix(_p.Hess((double[])_x.Clone()), _n, _n, "hess");
            Nhev++;
        }
        else Nhev++; // hessp: an operator bound to the new x
        _hUpd = true;
    }

    private void UpdateX(double[] x)
    {
        if (_hus is not null)
        {
            // SciPy updates the quasi-Newton matrix on every move, including to trial points that are
            // later rejected; the gradient is therefore evaluated at every trial point.
            UpdateGrad();
            _xPrev = _x; _gPrev = G;
            _x = (double[])x.Clone();
            _fUpd = _gUpd = _hUpd = false;
            UpdateHess();
        }
        else
        {
            _x = (double[])x.Clone();
            _fUpd = _gUpd = _hUpd = false;
        }
    }

    public double Fun(double[] x)
    {
        if (!x.AsSpan().SequenceEqual(_x)) UpdateX(x);
        UpdateFun();
        return F;
    }

    public double[] Grad(double[] x)
    {
        if (!x.AsSpan().SequenceEqual(_x)) UpdateX(x);
        UpdateGrad();
        return G;
    }

    /// <summary>Adds H(x)·p into acc; the returned delegate is bound to the Hessian at x.</summary>
    public Action<double[], double[]> Hess(double[] x)
    {
        if (!x.AsSpan().SequenceEqual(_x)) UpdateX(x);
        UpdateHess();
        if (_hus is FullHessianUpdateStrategy full) return (p, acc) => full.AddDot(p, acc);
        if (_hus is not null)
        {
            var hus = _hus;
            return (p, acc) => { var r = hus.Dot(p); for (var i = 0; i < acc.Length; i++) acc[i] += r[i]; };
        }
        if (_hMat is not null)
        {
            var hm = _hMat; var n = _n;
            return (p, acc) =>
            {
                for (var i = 0; i < n; i++)
                {
                    double s = 0;
                    for (var j = 0; j < n; j++) s += hm[i * n + j] * p[j];
                    acc[i] += s;
                }
            };
        }
        var xc = (double[])_x.Clone();
        var hessp = _p.Hessp!;
        return (p, acc) =>
        {
            var r = hessp((double[])xc.Clone(), (double[])p.Clone());
            if (r.Length != acc.Length) throw new ArgumentException($"hessp returned {r.Length} values but x has {acc.Length}.");
            for (var i = 0; i < acc.Length; i++) acc[i] += r[i];
        };
    }
}

/// <summary>One constraint as a vector function with lb ≤ f(x) ≤ ub (SciPy's <c>PreparedConstraint</c>).</summary>
internal abstract class TrustConstrVectorFunction
{
    public int M { get; protected init; }
    public int N { get; protected init; }
    public required double[] Lb { get; init; }
    public required double[] Ub { get; init; }
    public required bool KeepFeasible { get; init; }
    public double[] F { get; protected set; } = [];
    public double[] J { get; protected set; } = [];
    public double[] V { get; protected set; } = [];
    public int Nfev { get; protected set; }
    public int Njev { get; protected set; }
    public int Nhev { get; protected set; }

    public abstract double[] Fun(double[] x);
    public abstract double[] Jac(double[] x);

    /// <summary>Sets the multipliers and returns the Hessian of Σ vᵢ fᵢ at x as an accumulating operator (null when zero).</summary>
    public abstract Action<double[], double[]>? Hess(double[] x, double[] v);
}

/// <summary>A x, including the identity used for bounds. Its Hessian is zero.</summary>
internal sealed class TrustConstrLinearFunction : TrustConstrVectorFunction
{
    private readonly double[]? _a; // null = identity
    private double[] _x;

    public TrustConstrLinearFunction(double[]? a, int m, int n, double[] x0)
    {
        _a = a; M = m; N = n; _x = (double[])x0.Clone();
        J = a ?? OptimizeDense.Identity(n);
        F = Eval(_x);
        V = new double[m];
    }

    private double[] Eval(double[] x)
    {
        if (_a is null) return (double[])x.Clone();
        var r = new double[M];
        OptimizeDense.MatVec(_a, M, N, x, r);
        return r;
    }

    public override double[] Fun(double[] x)
    {
        if (!x.AsSpan().SequenceEqual(_x)) { _x = (double[])x.Clone(); F = Eval(_x); }
        return F;
    }

    public override double[] Jac(double[] x) => J;

    public override Action<double[], double[]>? Hess(double[] x, double[] v) { V = (double[])v.Clone(); return null; }
}

/// <summary>SciPy's <c>VectorFunction</c> for a nonlinear constraint: cached f and J, Hessian of vᵀf analytic or quasi-Newton.</summary>
internal sealed class TrustConstrNonlinearFunction : TrustConstrVectorFunction
{
    private readonly Func<double[], double[]> _fun;
    private readonly Func<double[], NdArray>? _jac;
    private readonly Func<double[], double[], NdArray>? _hess;
    private readonly HessianUpdateStrategy? _hus;
    private readonly double? _relStep;
    private readonly double[]? _fdLb, _fdUb;
    private double[] _x;
    private bool _fUpd, _jUpd, _hUpd;
    private double[]? _xPrev, _jPrev;
    private double[]? _hMat;

    public TrustConstrNonlinearFunction(Func<double[], double[]> fun, Func<double[], NdArray>? jac, Func<double[], double[], NdArray>? hess,
        HessianUpdateStrategy? hus, double? relStep, double[] x0, double[]? fdLb, double[]? fdUb)
    {
        _fun = fun; _jac = jac; _hess = hess; _relStep = relStep; _fdLb = fdLb; _fdUb = fdUb;
        N = x0.Length; _x = (double[])x0.Clone();
        UpdateFun();
        M = F.Length;
        V = new double[M];
        UpdateJac();
        if (hess is not null) { _hUpd = false; UpdateHess(); }
        else
        {
            _hus = hus ?? new BfgsHessianUpdate();
            _hus.Initialize(N, HessianApproxType.Hess);
            _hUpd = true;
        }
    }

    public HessianUpdateStrategy? Strategy => _hus;

    private void UpdateFun()
    {
        if (_fUpd) return;
        var f = _fun((double[])_x.Clone());
        if (M != 0 && f.Length != M) throw new ArgumentException($"NonlinearConstraint returned {f.Length} values but has {M} bounds.");
        F = (double[])f.Clone();
        Nfev++;
        _fUpd = true;
    }

    private void UpdateJac()
    {
        if (_jUpd) return;
        if (_jac is not null)
        {
            J = OptimizeDense.ToMatrix(_jac((double[])_x.Clone()), M, N, "Constraint Jacobian");
            Njev++;
        }
        else
        {
            UpdateFun();
            var j = new double[M * N];
            Nfev += OptimizeFiniteDiff.Jacobian(_fun, _x, F, j, FiniteDiffScheme.TwoPoint, null, _relStep, _fdLb, _fdUb);
            J = j;
        }
        _jUpd = true;
    }

    private void UpdateHess()
    {
        if (_hUpd) return;
        if (_hess is not null)
        {
            _hMat = OptimizeDense.ToMatrix(_hess((double[])_x.Clone(), (double[])V.Clone()), N, N, "Constraint Hessian");
            Nhev++;
        }
        else
        {
            UpdateJac();
            if (_xPrev is not null && _jPrev is not null)
            {
                var dx = new double[N];
                for (var i = 0; i < N; i++) dx[i] = _x[i] - _xPrev[i];
                var g1 = new double[N]; var g0 = new double[N];
                OptimizeDense.MatTVec(J, M, N, V, g1);
                OptimizeDense.MatTVec(_jPrev, M, N, V, g0);
                for (var i = 0; i < N; i++) g1[i] -= g0[i];
                _hus!.Update(dx, g1);
            }
        }
        _hUpd = true;
    }

    private void UpdateX(double[] x)
    {
        if (x.AsSpan().SequenceEqual(_x)) return;
        if (_hus is not null)
        {
            UpdateJac();
            _xPrev = _x; _jPrev = J;
            _x = (double[])x.Clone();
            _fUpd = _jUpd = _hUpd = false;
            UpdateHess();
        }
        else
        {
            _x = (double[])x.Clone();
            _fUpd = _jUpd = _hUpd = false;
        }
    }

    public override double[] Fun(double[] x) { UpdateX(x); UpdateFun(); return F; }

    public override double[] Jac(double[] x) { UpdateX(x); UpdateJac(); return J; }

    public override Action<double[], double[]>? Hess(double[] x, double[] v)
    {
        if (!v.AsSpan().SequenceEqual(V)) { V = (double[])v.Clone(); _hUpd = false; }
        UpdateX(x);
        UpdateHess();
        if (_hus is FullHessianUpdateStrategy full) return (p, acc) => full.AddDot(p, acc);
        if (_hus is not null)
        {
            var hus = _hus;
            return (p, acc) => { var r = hus.Dot(p); for (var i = 0; i < acc.Length; i++) acc[i] += r[i]; };
        }
        var hm = _hMat!; var n = N;
        return (p, acc) =>
        {
            for (var i = 0; i < n; i++)
            {
                double s = 0;
                for (var j = 0; j < n; j++) s += hm[i * n + j] * p[j];
                acc[i] += s;
            }
        };
    }
}

/// <summary>
/// All constraints in SciPy's canonical form: equalities c_eq(x) = 0 and inequalities
/// c_ineq(x) ≤ 0. Per constraint the rows are ordered as SciPy's <c>CanonicalConstraint</c>:
/// equal rows, then "less" (f − ub), "greater" (lb − f), interval upper (f − ub), interval lower (lb − f).
/// </summary>
internal sealed class TrustConstrCanonical
{
    private readonly record struct Row(int Index, double Sign, double Offset);

    private readonly TrustConstrVectorFunction[] _funcs;
    private readonly Row[][] _eq, _in;

    public int NEq { get; }
    public int NIneq { get; }
    public bool[] KeepFeasible { get; }

    public TrustConstrCanonical(TrustConstrVectorFunction[] funcs)
    {
        _funcs = funcs;
        _eq = new Row[funcs.Length][];
        _in = new Row[funcs.Length][];
        var kf = new List<bool>();
        for (var k = 0; k < funcs.Length; k++)
        {
            var c = funcs[k];
            var eq = new List<Row>();
            var less = new List<Row>(); var greater = new List<Row>(); var iu = new List<Row>(); var il = new List<Row>();
            for (var i = 0; i < c.M; i++)
            {
                double lb = c.Lb[i], ub = c.Ub[i];
                if (lb == ub) eq.Add(new Row(i, 1, -lb));
                else if (double.IsNegativeInfinity(lb) && !double.IsPositiveInfinity(ub)) less.Add(new Row(i, 1, -ub));
                else if (double.IsPositiveInfinity(ub) && !double.IsNegativeInfinity(lb)) greater.Add(new Row(i, -1, lb));
                else if (!double.IsNegativeInfinity(lb) && !double.IsPositiveInfinity(ub))
                {
                    iu.Add(new Row(i, 1, -ub));
                    il.Add(new Row(i, -1, lb));
                }
            }
            _eq[k] = [.. eq];
            _in[k] = [.. less, .. greater, .. iu, .. il];
            NEq += _eq[k].Length;
            NIneq += _in[k].Length;
            for (var r = 0; r < _in[k].Length; r++) kf.Add(c.KeepFeasible);
        }
        KeepFeasible = [.. kf];
    }

    public (double[] Eq, double[] Ineq) Fun(double[] x)
    {
        var eq = new double[NEq]; var ineq = new double[NIneq];
        int ie = 0, ii = 0;
        for (var k = 0; k < _funcs.Length; k++)
        {
            var f = _funcs[k].Fun(x);
            foreach (var r in _eq[k]) eq[ie++] = r.Sign * f[r.Index] + r.Offset;
            foreach (var r in _in[k]) ineq[ii++] = r.Sign * f[r.Index] + r.Offset;
        }
        return (eq, ineq);
    }

    public (double[] JEq, double[] JIneq) Jac(double[] x, int n)
    {
        var je = new double[NEq * n]; var ji = new double[NIneq * n];
        int ie = 0, ii = 0;
        for (var k = 0; k < _funcs.Length; k++)
        {
            var j = _funcs[k].Jac(x);
            foreach (var r in _eq[k]) { for (var c = 0; c < n; c++) je[ie * n + c] = r.Sign * j[r.Index * n + c]; ie++; }
            foreach (var r in _in[k]) { for (var c = 0; c < n; c++) ji[ii * n + c] = r.Sign * j[r.Index * n + c]; ii++; }
        }
        return (je, ji);
    }

    /// <summary>Maps canonical multipliers back to each constraint and collects their Hessian operators.</summary>
    public List<Action<double[], double[]>> Hess(double[] x, ReadOnlySpan<double> vEq, ReadOnlySpan<double> vIneq)
    {
        var ops = new List<Action<double[], double[]>>();
        int ie = 0, ii = 0;
        for (var k = 0; k < _funcs.Length; k++)
        {
            var v = new double[_funcs[k].M];
            foreach (var r in _eq[k]) v[r.Index] = vEq[ie++];
            foreach (var r in _in[k]) v[r.Index] += r.Sign * vIneq[ii++];
            var op = _funcs[k].Hess(x, v);
            if (op is not null) ops.Add(op);
        }
        return ops;
    }
}

/// <summary>Problem state shared by the stop test (SciPy's <c>state</c> OptimizeResult).</summary>
internal sealed class TrustConstrState
{
    public int Nit;
    public double[] X = [];
    public double Fun;
    public double[] Grad = [];
    public double[][] V = [];
    public double[][] Constr = [];
    public double[] LagrangianGrad = [];
    public double Optimality;
    public double ConstrViolation;
    public double TrRadius;
    public double ConstrPenalty;
    public int CgNiter;
    public int CgStopCond;
    public double? BarrierParameter;
    public double? BarrierTolerance;
    public int Status = -1;
}

internal static class OptimizeTrustConstr
{
    private static readonly string[] Messages =
    [
        "The maximum number of function evaluations is exceeded.",
        "`gtol` termination condition is satisfied.",
        "`xtol` termination condition is satisfied.",
        "`callback` raised `StopIteration`.",
        "Constraint violation exceeds 'gtol'",
    ];

    public static OptimizeResult Run(MinimizeProblem p)
    {
        var o = p.OptionsAs<TrustConstrOptions>("TrustConstr");
        var gtol = o.Gtol ?? p.Tol ?? 1e-8;
        var xtol = o.Xtol ?? p.Tol ?? 1e-8;
        var barrierTol = o.BarrierTol ?? p.Tol ?? 1e-8;
        var maxiter = o.MaxIter ?? 1000;
        if (o.InitialTrRadius <= 0 || o.InitialConstrPenalty <= 0 || o.InitialBarrierParameter <= 0 || o.InitialBarrierTolerance <= 0)
            throw new ArgumentException("TrustConstr initial radius, penalty, barrier parameter and tolerance must be positive.");
        var sw = Stopwatch.StartNew();
        var n = p.N;
        var x0 = (double[])p.X0.Clone();

        // SciPy widens finite bounds by one ulp so that x0 sitting exactly on a bound starts strictly
        // inside the barrier, and lb == ub becomes a (tiny) interval instead of a degenerate one.
        double[]? bLb = null, bUb = null, fdLb = null, fdUb = null;
        if (p.Bounds is { } bounds)
        {
            bLb = new double[n]; bUb = new double[n];
            for (var i = 0; i < n; i++)
            {
                bLb[i] = double.IsFinite(bounds.Lb[i]) ? Math.BitDecrement(bounds.Lb[i]) : bounds.Lb[i];
                bUb[i] = double.IsFinite(bounds.Ub[i]) ? Math.BitIncrement(bounds.Ub[i]) : bounds.Ub[i];
            }
            if (bounds.KeepFeasible) { fdLb = bLb; fdUb = bUb; }
        }

        var seen = new HashSet<HessianUpdateStrategy>(ReferenceEqualityComparer.Instance);
        void Claim(HessianUpdateStrategy? s)
        {
            if (s is not null && !seen.Add(s))
                throw new ArgumentException("The same HessianUpdateStrategy instance is used twice in one TrustConstr problem; give each role its own instance.");
        }
        Claim(p.Hess is null && p.Hessp is null ? o.HessianUpdate : null);

        var objective = new TrustConstrObjective(p, x0, o.FiniteDiffRelStep, fdLb, fdUb, o.HessianUpdate);

        var funcs = new List<TrustConstrVectorFunction>();
        foreach (var c in p.Constraints)
        {
            switch (c)
            {
                case LinearConstraint lc:
                {
                    if (lc.A.Shape[1] != n) throw new ArgumentException($"LinearConstraint has {lc.A.Shape[1]} columns but x has {n}.");
                    funcs.Add(new TrustConstrLinearFunction(lc.A.AsContiguous().ToArray(), lc.A.Shape[0], n, x0)
                    { Lb = lc.Lb, Ub = lc.Ub, KeepFeasible = lc.KeepFeasible });
                    break;
                }
                case NonlinearConstraint nc:
                {
                    Claim(nc.Hess is null ? nc.HessianUpdate : null);
                    var f = new TrustConstrNonlinearFunction(nc.Fun, nc.Jac, nc.Hess, nc.HessianUpdate, nc.FiniteDiffRelStep, x0, fdLb, fdUb)
                    { Lb = nc.Lb, Ub = nc.Ub, KeepFeasible = nc.KeepFeasible };
                    if (f.M != nc.Lb.Length) throw new ArgumentException($"NonlinearConstraint returned {f.M} values but has {nc.Lb.Length} bounds.");
                    funcs.Add(f);
                    break;
                }
                case OptimizeConstraint oc:
                {
                    // SciPy's old_constraint_to_new: eq → 0 ≤ f ≤ 0, ineq → 0 ≤ f ≤ ∞.
                    var m = oc.Fun((double[])x0.Clone()).Length;
                    var lb = new double[m];
                    var ub = new double[m];
                    if (oc.Type == OptimizeConstraintType.Ineq) Array.Fill(ub, double.PositiveInfinity);
                    funcs.Add(new TrustConstrNonlinearFunction(oc.Fun, oc.Jac, null, null, null, x0, fdLb, fdUb)
                    { Lb = lb, Ub = ub, KeepFeasible = false });
                    break;
                }
                default:
                    throw new ArgumentException($"Unsupported constraint type {c.GetType().Name}.");
            }
        }
        if (bLb is not null)
            funcs.Add(new TrustConstrLinearFunction(null, n, n, x0) { Lb = bLb, Ub = bUb!, KeepFeasible = p.Bounds!.KeepFeasible });

        var prepared = funcs.ToArray();
        foreach (var c in prepared)
            if (c.KeepFeasible)
            {
                var f = c.F;
                for (var i = 0; i < c.M; i++)
                    if (f[i] < c.Lb[i] || f[i] > c.Ub[i])
                        throw new ArgumentException("x0 is infeasible with respect to some inequality constraint with KeepFeasible set to true.");
            }
        var canonical = new TrustConstrCanonical(prepared);
        var (cEq0, cIneq0) = canonical.Fun(x0);
        var (jEq0, jIneq0) = canonical.Jac(x0, n);
        var method = canonical.NIneq == 0 ? "equality_constrained_sqp" : "tr_interior_point";

        TrustConstrMatVec LagrangianHess(double[] x, ReadOnlySpan<double> vEq, ReadOnlySpan<double> vIneq)
        {
            var ops = new List<Action<double[], double[]>> { objective.Hess(x) };
            ops.AddRange(canonical.Hess(x, vEq, vIneq));
            var opsArr = ops.ToArray();
            return (pv, y) =>
            {
                var pa = pv.ToArray();
                var acc = new double[pa.Length];
                foreach (var op in opsArr) op(pa, acc);
                acc.AsSpan().CopyTo(y);
            };
        }

        var state = new TrustConstrState { Fun = objective.F, Grad = (double[])objective.G.Clone() };

        void UpdateState(double[] x, bool lastFailed, double trRadius, double penalty, int cgNiter, int cgStop)
        {
            state.Nit++;
            if (!lastFailed)
            {
                state.X = (double[])x.Clone();
                state.Fun = objective.F;
                state.Grad = (double[])objective.G.Clone();
                state.V = prepared.Select(c => (double[])c.V.Clone()).ToArray();
                state.Constr = prepared.Select(c => (double[])c.F.Clone()).ToArray();
                var lg = (double[])state.Grad.Clone();
                var tmp = new double[n];
                foreach (var c in prepared)
                {
                    OptimizeDense.MatTVec(c.J, c.M, n, c.V, tmp);
                    for (var i = 0; i < n; i++) lg[i] += tmp[i];
                }
                state.LagrangianGrad = lg;
                state.Optimality = OptimizeDense.NormInf(lg);
                double viol = 0;
                for (var k = 0; k < prepared.Length; k++)
                    for (var i = 0; i < prepared[k].M; i++)
                        viol = Math.Max(viol, Math.Max(prepared[k].Lb[i] - state.Constr[k][i], state.Constr[k][i] - prepared[k].Ub[i]));
                state.ConstrViolation = viol;
            }
            state.TrRadius = trRadius;
            state.ConstrPenalty = penalty;
            state.CgNiter += cgNiter;
            state.CgStopCond = cgStop;
        }

        bool Callback()
        {
            if (!p.NotifyCallback(state.X, state.Fun)) return false;
            state.Status = 3;
            return true;
        }

        if (method == "equality_constrained_sqp")
        {
            bool Stop(double[] x, bool lastFailed, double optimality, double viol, double tr, double penalty, int cgN, int cgS)
            {
                UpdateState(x, lastFailed, tr, penalty, cgN, cgS);
                state.Status = -1;
                if (Callback()) return true;
                if (state.Optimality < gtol && state.ConstrViolation < gtol) state.Status = 1;
                else if (state.TrRadius < xtol) state.Status = 2;
                else if (state.Nit >= maxiter) state.Status = 0;
                return state.Status is 0 or 1 or 2 or 3;
            }
            var prob = new EqualityProblem(objective, canonical, n, LagrangianHess);
            OptimizeTrustConstrSqp.EqualityConstrainedSqp(prob, x0, objective.F, (double[])objective.G.Clone(), cEq0, jEq0, Stop,
                o.InitialConstrPenalty, o.InitialTrRadius, null, null);
        }
        else
        {
            bool GlobalStop(double[] x, bool lastFailed, double tr, double penalty, int cgN, int cgS, double mu, double barTol)
            {
                UpdateState(x, lastFailed, tr, penalty, cgN, cgS);
                state.BarrierParameter = mu;
                state.BarrierTolerance = barTol;
                state.Status = -1;
                if (Callback()) return true;
                if (state.Optimality < gtol && state.ConstrViolation < gtol) state.Status = 1;
                else if (state.TrRadius < xtol && state.BarrierParameter < barrierTol) state.Status = 2;
                else if (state.Nit >= maxiter) state.Status = 0;
                return state.Status is 0 or 1 or 2 or 3;
            }
            TrInteriorPoint(objective, canonical, n, LagrangianHess, x0, cIneq0, jIneq0, cEq0, jEq0, GlobalStop, xtol, state,
                o.InitialBarrierParameter, o.InitialBarrierTolerance, o.InitialConstrPenalty, o.InitialTrRadius,
                fdLb, fdUb);
        }

        var status = state.Status;
        if (status is 1 or 2 && state.ConstrViolation > gtol) status = 4;
        var result = new OptimizeResult
        {
            X = state.X.Length == n ? state.X : x0,
            Fun = state.Fun,
            Grad = state.Grad,
            Jac = (double[])state.Grad.Clone(),
            LagrangianGrad = state.LagrangianGrad,
            V = state.V,
            Constr = state.Constr,
            Multipliers = state.V.SelectMany(v => v).ToArray(),
            Optimality = state.Optimality,
            ConstrViolation = state.ConstrViolation,
            TrRadius = state.TrRadius,
            ConstrPenalty = state.ConstrPenalty,
            BarrierParameter = state.BarrierParameter,
            BarrierTolerance = state.BarrierTolerance,
            CgNiter = state.CgNiter,
            CgStopCond = state.CgStopCond,
            Nit = state.Nit,
            Nfev = objective.Nfev,
            Njev = objective.Ngev,
            Nhev = objective.Nhev,
            Status = status,
            Success = status is 1 or 2,
            Message = Messages[status],
            Method = method,
            ExecutionTime = sw.Elapsed.TotalSeconds,
        };
        var strat = objective.Strategy;
        if (strat is not null) result.Hess = strat.GetMatrix();
        return result;
    }

    private sealed class EqualityProblem(TrustConstrObjective obj, TrustConstrCanonical canon, int n,
        LagrHessDelegate lagr) : ITrustConstrSqpProblem
    {
        public (double F, double[] C) FunAndConstr(double[] x)
        {
            var f = obj.Fun(x);
            return (f, canon.Fun(x).Eq);
        }

        public (double[] G, double[] A) GradAndJac(double[] x)
        {
            var g = (double[])obj.Grad(x).Clone();
            return (g, canon.Jac(x, n).JEq);
        }

        public TrustConstrMatVec LagrangianHessian(double[] x, double[] v) => lagr(x, v, []);

        public double[]? Scaling(double[] x) => null;
    }

    internal delegate TrustConstrMatVec LagrHessDelegate(double[] x, ReadOnlySpan<double> vEq, ReadOnlySpan<double> vIneq);

    internal delegate bool GlobalStopDelegate(double[] x, bool lastFailed, double tr, double penalty, int cgN, int cgS, double mu, double barTol);

    /// <summary>
    /// SciPy's <c>BarrierSubproblem</c>: variables z = (x, s) with slacks s &gt; 0 turning
    /// c_ineq(x) ≤ 0 into c_ineq(x) + s = 0, objective f(x) − μ Σ log sᵢ, step scaled by diag(1, s)
    /// so the slack step is relative (Byrd, Hribar &amp; Nocedal 1999).
    /// </summary>
    private sealed class BarrierSubproblem : ITrustConstrSqpProblem
    {
        private readonly TrustConstrObjective _obj;
        private readonly TrustConstrCanonical _canon;
        private readonly LagrHessDelegate _lagr;
        private readonly int _n, _nIneq, _nEq;
        private readonly bool[] _kf;
        private readonly double[]? _lb, _ub;
        private readonly GlobalStopDelegate _globalStop;
        private readonly double _xtol;
        public double Mu;
        public double Tolerance;
        public bool Terminate;

        public BarrierSubproblem(TrustConstrObjective obj, TrustConstrCanonical canon, LagrHessDelegate lagr, int n,
            double mu, double tol, GlobalStopDelegate globalStop, double xtol, double[]? lb, double[]? ub)
        {
            _obj = obj; _canon = canon; _lagr = lagr; _n = n; _nIneq = canon.NIneq; _nEq = canon.NEq; _kf = canon.KeepFeasible;
            Mu = mu; Tolerance = tol; _globalStop = globalStop; _xtol = xtol; _lb = lb; _ub = ub;
        }

        private double[] Vars(double[] z) => z.AsSpan(0, _n).ToArray();

        /// <summary>f − μ Σ log s; keep-feasible slacks are reset to −c_ineq in place, as SciPy does through a view.</summary>
        public double ComputeFunction(double f, double[] cIneq, double[] z)
        {
            double sumLog = 0;
            for (var i = 0; i < _nIneq; i++)
            {
                if (_kf[i]) z[_n + i] = -cIneq[i];
                var s = z[_n + i];
                sumLog += s > 0 ? Math.Log(s) : double.NegativeInfinity;
            }
            return f - Mu * sumLog;
        }

        public double[] ComputeConstr(double[] cIneq, double[] cEq, double[] z)
        {
            var r = new double[_nEq + _nIneq];
            Array.Copy(cEq, r, _nEq);
            for (var i = 0; i < _nIneq; i++) r[_nEq + i] = cIneq[i] + z[_n + i];
            return r;
        }

        public double[] ComputeGradient(double[] g)
        {
            var r = new double[_n + _nIneq];
            Array.Copy(g, r, _n);
            for (var i = 0; i < _nIneq; i++) r[_n + i] = -Mu;
            return r;
        }

        public double[] ComputeJacobian(double[] jEq, double[] jIneq, double[] z)
        {
            var nz = _n + _nIneq;
            var a = new double[(_nEq + _nIneq) * nz];
            for (var i = 0; i < _nEq; i++) Array.Copy(jEq, i * _n, a, i * nz, _n);
            for (var i = 0; i < _nIneq; i++)
            {
                Array.Copy(jIneq, i * _n, a, (_nEq + i) * nz, _n);
                a[(_nEq + i) * nz + _n + i] = z[_n + i];
            }
            return a;
        }

        public (double F, double[] C) FunAndConstr(double[] z)
        {
            var x = Vars(z);
            var outside = false;
            if (_lb is not null)
                for (var i = 0; i < _n; i++) if (x[i] < _lb[i] || x[i] > _ub![i]) { outside = true; break; }
            double f;
            double[] cEq, cIneq;
            if (outside)
            {
                f = double.PositiveInfinity;
                cEq = new double[_nEq]; cIneq = new double[_nIneq];
            }
            else
            {
                f = _obj.Fun(x);
                (cEq, cIneq) = _canon.Fun(x);
            }
            return (ComputeFunction(f, cIneq, z), ComputeConstr(cIneq, cEq, z));
        }

        public (double[] G, double[] A) GradAndJac(double[] z)
        {
            var x = Vars(z);
            var g = _obj.Grad(x);
            var (je, ji) = _canon.Jac(x, _n);
            return (ComputeGradient(g), ComputeJacobian(je, ji, z));
        }

        public TrustConstrMatVec LagrangianHessian(double[] z, double[] v)
        {
            var x = Vars(z);
            var hx = _lagr(x, v.AsSpan(0, _nEq), v.AsSpan(_nEq, _nIneq));
            var shs = new double[_nIneq];
            for (var i = 0; i < _nIneq; i++)
            {
                var vi = v[_nEq + i];
                shs[i] = vi > 0 ? vi * z[_n + i] : Mu;
            }
            var n = _n; var ni = _nIneq;
            return (p, y) =>
            {
                hx(p[..n], y[..n]);
                for (var i = 0; i < ni; i++) y[n + i] = shs[i] * p[n + i];
            };
        }

        public double[]? Scaling(double[] z)
        {
            var d = new double[_n + _nIneq];
            for (var i = 0; i < _n; i++) d[i] = 1;
            for (var i = 0; i < _nIneq; i++) d[_n + i] = z[_n + i];
            return d;
        }

        public bool Stop(double[] z, bool lastFailed, double optimality, double viol, double tr, double penalty, int cgN, int cgS)
        {
            var x = Vars(z);
            if (_globalStop(x, lastFailed, tr, penalty, cgN, cgS, Mu, Tolerance))
            {
                Terminate = true;
                return true;
            }
            return (optimality < Tolerance && viol < Tolerance) || tr < _xtol;
        }
    }

    private static void TrInteriorPoint(TrustConstrObjective obj, TrustConstrCanonical canon, int n,
        LagrHessDelegate lagr,
        double[] x0, double[] cIneq0, double[] jIneq0, double[] cEq0, double[] jEq0, GlobalStopDelegate stop, double xtol,
        TrustConstrState state, double initialBarrierParameter, double initialTolerance, double initialPenalty, double initialTrustRadius,
        double[]? fdLb, double[]? fdUb)
    {
        const double BoundaryParameter = 0.995;
        const double BarrierDecayRatio = 0.2;
        const double TrustEnlargement = 5;
        var nIneq = canon.NIneq;
        var mu = initialBarrierParameter;
        var tolerance = initialTolerance;
        var trustRadius = initialTrustRadius;

        var sub = new BarrierSubproblem(obj, canon, lagr, n, mu, tolerance, stop, xtol, fdLb, fdUb);
        var z = new double[n + nIneq];
        Array.Copy(x0, z, n);
        for (var i = 0; i < nIneq; i++) z[n + i] = Math.Max(-1.5 * cIneq0[i], 1.0);
        var fun0 = sub.ComputeFunction(obj.F, cIneq0, z);
        var grad0 = sub.ComputeGradient(obj.G);
        var constr0 = sub.ComputeConstr(cIneq0, cEq0, z);
        var jac0 = sub.ComputeJacobian(jEq0, jIneq0, z);

        var trustLb = new double[n + nIneq];
        var trustUb = new double[n + nIneq];
        for (var i = 0; i < n; i++) trustLb[i] = double.NegativeInfinity;
        for (var i = 0; i < nIneq; i++) trustLb[n + i] = -BoundaryParameter;
        Array.Fill(trustUb, double.PositiveInfinity);

        while (true)
        {
            z = OptimizeTrustConstrSqp.EqualityConstrainedSqp(sub, z, fun0, grad0, constr0, jac0, sub.Stop,
                initialPenalty, trustRadius, trustLb, trustUb);
            if (sub.Terminate) break;
            trustRadius = Math.Max(initialTrustRadius, TrustEnlargement * state.TrRadius);
            mu *= BarrierDecayRatio;
            tolerance *= BarrierDecayRatio;
            sub.Mu = mu; sub.Tolerance = tolerance;
            (fun0, constr0) = sub.FunAndConstr(z);
            (grad0, jac0) = sub.GradAndJac(z);
        }
    }
}
