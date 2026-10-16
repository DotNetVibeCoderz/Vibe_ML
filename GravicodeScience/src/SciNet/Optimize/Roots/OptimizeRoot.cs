using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Methods of <see cref="Optimize.Root"/>.</summary>
public enum RootMethod
{
    /// <summary>MINPACK hybrd/hybrj: Powell's dogleg with Broyden updates of a QR-factored Jacobian.</summary>
    Hybr,
    /// <summary>MINPACK Levenberg–Marquardt on the residuals (least-squares sense).</summary>
    Lm,
    /// <summary>Broyden's good method (inverse-Jacobian low-rank updates).</summary>
    Broyden1,
    /// <summary>Broyden's bad method.</summary>
    Broyden2,
    /// <summary>Anderson mixing.</summary>
    Anderson,
    /// <summary>Scalar Jacobian approximation.</summary>
    LinearMixing,
    /// <summary>Diagonal Broyden.</summary>
    DiagBroyden,
    /// <summary>Tuned diagonal mixing.</summary>
    ExcitingMixing,
    /// <summary>Newton–Krylov: inexact Newton with GMRES on finite-difference Jacobian products.</summary>
    Krylov,
    /// <summary>Derivative-free spectral residual method (La Cruz, Martínez &amp; Raydan 2006).</summary>
    DfSane,
}

/// <summary>Line search used by the quasi-Newton <see cref="Optimize.Root"/> methods.</summary>
public enum RootLineSearch
{
    /// <summary>Full steps.</summary>
    None,
    /// <summary>Backtracking Armijo on ‖F‖² (SciPy's default).</summary>
    Armijo,
    /// <summary>Moré–Thuente on ‖F‖².</summary>
    Wolfe,
}

/// <summary>
/// Options for <see cref="Optimize.Root"/>. One record covers every method; each method reads the
/// fields SciPy gives it and ignores the rest.
/// </summary>
public sealed record RootOptions
{
    /// <summary>Hybr/Lm: relative error in x. Quasi-Newton methods: relative step tolerance (<c>xtol</c>).</summary>
    public double? Xtol { get; init; }
    /// <summary>Lm: relative reduction of the sum of squares. Quasi-Newton: relative residual (<c>ftol</c>). DfSane: relative residual norm.</summary>
    public double? Ftol { get; init; }
    /// <summary>Lm: orthogonality tolerance.</summary>
    public double Gtol { get; init; }
    /// <summary>Quasi-Newton: absolute residual (max-norm). Default ε^(1/3) ≈ 6e-6.</summary>
    public double? Fatol { get; init; }
    /// <summary>Quasi-Newton: absolute step.</summary>
    public double? Xatol { get; init; }
    /// <summary>DfSane: absolute residual. Default 1e-300.</summary>
    public double DfSaneFatol { get; init; } = 1e-300;
    /// <summary>Hybr/Lm/DfSane: maximum function evaluations (0 = method default; DfSane 1000).</summary>
    public int MaxFev { get; init; }
    /// <summary>Quasi-Newton: maximum iterations (default 100(n+1)).</summary>
    public int? MaxIter { get; init; }
    /// <summary>Hybr/Lm: forward-difference step hint (<c>eps</c>).</summary>
    public double? Eps { get; init; }
    /// <summary>Hybr/Lm: initial step bound factor.</summary>
    public double Factor { get; init; } = 100;
    /// <summary>Hybr/Lm: variable scaling.</summary>
    public double[]? Diag { get; init; }
    /// <summary>Quasi-Newton line search.</summary>
    public RootLineSearch LineSearch { get; init; } = RootLineSearch.Armijo;
    /// <summary>Initial Jacobian scale for Broyden/Anderson/mixing methods; null autoscales as SciPy.</summary>
    public double? Alpha { get; init; }
    /// <summary>Anderson: number of previous vectors kept. Default 5.</summary>
    public int AndersonM { get; init; } = 5;
    /// <summary>Anderson: regularisation weight. Default 0.01.</summary>
    public double W0 { get; init; } = 0.01;
    /// <summary>ExcitingMixing: largest diagonal entry. Default 1.</summary>
    public double AlphaMax { get; init; } = 1.0;
    /// <summary>Krylov: GMRES iterations per Newton step. Default 20.</summary>
    public int InnerMaxIter { get; init; } = 20;
    /// <summary>Krylov: relative finite-difference step; default √ε.</summary>
    public double? Rdiff { get; init; }
    /// <summary>DfSane: length of the non-monotone memory. Default 10.</summary>
    public int DfSaneM { get; init; } = 10;
    /// <summary>DfSane: use Cheng's line search instead of La Cruz's.</summary>
    public bool DfSaneCheng { get; init; }
}

public static partial class Optimize
{
    /// <summary>
    /// Finds a root of a vector function F: Rⁿ → Rⁿ (<c>scipy.optimize.root</c>). The result's
    /// <see cref="OptimizeResult.FunVector"/> holds F(x).
    /// </summary>
    /// <param name="fun">The system; receives a private copy of x.</param>
    /// <param name="x0">Initial guess.</param>
    /// <param name="method">Solver; Hybr by default.</param>
    /// <param name="jac">Jacobian (n, n) for Hybr and Lm; finite differences otherwise.</param>
    /// <param name="tol">Mapped onto the method's tolerance exactly as SciPy maps it.</param>
    /// <param name="options">Method options.</param>
    /// <param name="callback">Quasi-Newton/DfSane: called with (x, F(x)) every iteration.</param>
    public static OptimizeResult Root(Func<double[], double[]> fun, double[] x0, RootMethod method = RootMethod.Hybr,
        Func<double[], NdArray>? jac = null, double? tol = null, RootOptions? options = null, Action<double[], double[]>? callback = null)
    {
        ArgumentNullException.ThrowIfNull(fun);
        ArgumentNullException.ThrowIfNull(x0);
        if (x0.Length == 0) throw new ArgumentException("x0 must not be empty.");
        var o = options ?? new RootOptions();
        if (tol is { } t)
        {
            o = method switch
            {
                RootMethod.Hybr or RootMethod.Lm => o with { Xtol = o.Xtol ?? t },
                RootMethod.DfSane => o with { Ftol = o.Ftol ?? t },
                _ => o with { Xtol = o.Xtol ?? t, Xatol = o.Xatol ?? double.PositiveInfinity, Ftol = o.Ftol ?? double.PositiveInfinity, Fatol = o.Fatol ?? double.PositiveInfinity },
            };
        }
        switch (method)
        {
            case RootMethod.Hybr:
            {
                var xtol = o.Xtol ?? 1.49012e-08;
                var r = OptimizeMinpack.Hybrd(fun, jac, x0, xtol, o.MaxFev, o.Eps, o.Factor, o.Diag);
                string[] msgs =
                [
                    "Improper input parameters were entered.", "The solution converged.",
                    $"The number of calls to function has reached maxfev = {(o.MaxFev > 0 ? o.MaxFev : (jac is null ? 200 : 100) * (x0.Length + 1))}.",
                    $"xtol={xtol:F6} is too small, no further improvement in the approximate\n  solution is possible.",
                    "The iteration is not making good progress, as measured by the \n  improvement from the last five Jacobian evaluations.",
                    "The iteration is not making good progress, as measured by the \n  improvement from the last ten iterations.",
                ];
                var n = x0.Length;
                // Report the final Jacobian approximation Q R.
                var qr = OptimizeDense.MatMul(r.Q, n, n, r.R, n);
                return new OptimizeResult
                {
                    X = r.X, FunVector = r.Fvec, Fun = 0.5 * OptimizeDense.Dot(r.Fvec, r.Fvec), Nfev = r.Nfev, Njev = r.Njev,
                    Status = r.Info, Success = r.Info == 1, Message = msgs[r.Info], JacMatrix = new NdArray(qr, n, n),
                };
            }
            case RootMethod.Lm:
            {
                var r = MinpackLm(new MinpackLmProblem
                {
                    Fun = fun, Jac = jac, X0 = (double[])x0.Clone(), Xtol = o.Xtol ?? 1.49012e-08, Ftol = o.Ftol ?? 1.49012e-08,
                    Gtol = o.Gtol, MaxFev = o.MaxFev, Epsfcn = o.Eps ?? 0.0, Factor = o.Factor, Diag = o.Diag,
                });
                return r;
            }
            case RootMethod.DfSane:
                return OptimizeRootNonlin.DfSane(fun, x0, o, callback);
            default:
                return OptimizeRootNonlin.Solve(fun, x0, method, o, callback);
        }
    }

    /// <summary>
    /// <c>scipy.optimize.fsolve</c>: MINPACK's hybrid method. Returns the solution and the MINPACK
    /// <c>info</c> code (1 = converged); see <see cref="Root"/> for the full result.
    /// </summary>
    public static (double[] X, int Info, string Message) Fsolve(Func<double[], double[]> func, double[] x0, Func<double[], NdArray>? fprime = null,
        double xtol = 1.49012e-08, int maxfev = 0, double? epsfcn = null, double factor = 100, double[]? diag = null)
    {
        var r = Root(func, x0, RootMethod.Hybr, fprime, options: new RootOptions { Xtol = xtol, MaxFev = maxfev, Eps = epsfcn, Factor = factor, Diag = diag });
        return (r.X, r.Status, r.Message);
    }
}

/// <summary>SciPy's <c>nonlin_solve</c> with its Jacobian approximations, and <c>_root_df_sane</c>.</summary>
internal static class OptimizeRootNonlin
{
    private static double Norm(double[] v) => OptimizeDense.Norm2(v);
    private static double MaxNorm(double[] v) => OptimizeDense.NormInf(v);

    private abstract class Jacobian
    {
        public double[] LastX = [], LastF = [];
        public double? Alpha;
        public virtual void Setup(double[] x, double[] f, Func<double[], double[]> func)
        {
            LastX = x; LastF = f;
            if (Alpha is null && UsesAlpha)
            {
                var nf = Norm(f);
                Alpha = nf != 0 ? 0.5 * Math.Max(Norm(x), 1) / nf : 1.0;
            }
        }
        protected virtual bool UsesAlpha => true;
        public abstract double[] Solve(double[] f, double tol);
        public virtual void Update(double[] x, double[] f)
        {
            var n = x.Length;
            var df = new double[n]; var dx = new double[n];
            for (var i = 0; i < n; i++) { df[i] = f[i] - LastF[i]; dx[i] = x[i] - LastX[i]; }
            UpdateCore(x, f, dx, df, Norm(dx), Norm(df));
            LastF = f; LastX = x;
        }
        protected abstract void UpdateCore(double[] x, double[] f, double[] dx, double[] df, double dxNorm, double dfNorm);
    }

    /// <summary>alpha·I + Σ c dᵀ, kept as vectors until the rank exceeds n.</summary>
    private sealed class LowRank(double alpha, int n)
    {
        private readonly List<double[]> _cs = [], _ds = [];
        private double[]? _collapsed;

        public double[] MatVec(double[] v)
        {
            if (_collapsed is not null) { var r = new double[n]; OptimizeDense.MatVec(_collapsed, n, n, v, r); return r; }
            var w = v.Select(x => alpha * x).ToArray();
            for (var k = 0; k < _cs.Count; k++) OptimizeDense.Axpy(OptimizeDense.Dot(_ds[k], v), _cs[k], w);
            return w;
        }

        public double[] RMatVec(double[] v)
        {
            if (_collapsed is not null) { var r = new double[n]; OptimizeDense.MatTVec(_collapsed, n, n, v, r); return r; }
            var w = v.Select(x => alpha * x).ToArray();
            for (var k = 0; k < _cs.Count; k++) OptimizeDense.Axpy(OptimizeDense.Dot(_cs[k], v), _ds[k], w);
            return w;
        }

        public void Append(double[] c, double[] d)
        {
            if (_collapsed is not null)
            {
                for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) _collapsed[i * n + j] += c[i] * d[j];
                return;
            }
            _cs.Add(c); _ds.Add(d);
            if (_cs.Count > n)
            {
                var g = new double[n * n];
                for (var i = 0; i < n; i++) g[i * n + i] = alpha;
                for (var k = 0; k < _cs.Count; k++)
                    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) g[i * n + j] += _cs[k][i] * _ds[k][j];
                _collapsed = g;
                _cs.Clear(); _ds.Clear();
            }
        }
    }

    private sealed class Broyden(bool second) : Jacobian
    {
        private LowRank _gm = null!;
        private Func<double[], double[]> _func = null!;
        public override void Setup(double[] x, double[] f, Func<double[], double[]> func)
        {
            base.Setup(x, f, func);
            _func = func;
            _gm = new LowRank(-Alpha!.Value, x.Length);
        }
        public override double[] Solve(double[] f, double tol)
        {
            var r = _gm.MatVec(f);
            if (r.All(double.IsFinite)) return r;
            Setup(LastX, LastF, _func);
            return _gm.MatVec(f);
        }
        protected override void UpdateCore(double[] x, double[] f, double[] dx, double[] df, double dxNorm, double dfNorm)
        {
            var n = x.Length;
            var gdf = _gm.MatVec(df);
            var c = new double[n];
            for (var i = 0; i < n; i++) c[i] = dx[i] - gdf[i];
            double[] d;
            if (!second)
            {
                var v = _gm.RMatVec(dx);
                var den = OptimizeDense.Dot(df, v);
                d = v.Select(t => t / den).ToArray();
            }
            else d = df.Select(t => t / (dfNorm * dfNorm)).ToArray();
            _gm.Append(c, d);
        }
    }

    private sealed class AndersonJac(double? alpha, double w0, int m) : Jacobian
    {
        private readonly List<double[]> _dx = [], _df = [];
        private double[] _a = [];
        public override void Setup(double[] x, double[] f, Func<double[], double[]> func) { Alpha = alpha; base.Setup(x, f, func); }
        public override double[] Solve(double[] f, double tol)
        {
            var a = Alpha!.Value;
            var dx = f.Select(v => -a * v).ToArray();
            var n = _dx.Count;
            if (n == 0) return dx;
            var dff = new double[n];
            for (var k = 0; k < n; k++) dff[k] = OptimizeDense.Dot(_df[k], f);
            var gamma = OptimizeDense.Solve(_a, n, dff);
            if (gamma is null) { _dx.Clear(); _df.Clear(); return dx; }
            for (var k = 0; k < n; k++)
                for (var i = 0; i < f.Length; i++) dx[i] += gamma[k] * (_dx[k][i] + a * _df[k][i]);
            return dx;
        }
        protected override void UpdateCore(double[] x, double[] f, double[] dx, double[] df, double dxNorm, double dfNorm)
        {
            if (m == 0) return;
            _dx.Add(dx); _df.Add(df);
            while (_dx.Count > m) { _dx.RemoveAt(0); _df.RemoveAt(0); }
            var n = _dx.Count;
            _a = new double[n * n];
            for (var i = 0; i < n; i++)
                for (var j = i; j < n; j++)
                {
                    var wd = i == j ? w0 * w0 : 0;
                    _a[i * n + j] = (1 + wd) * OptimizeDense.Dot(_df[i], _df[j]);
                    _a[j * n + i] = _a[i * n + j];
                }
        }
    }

    private sealed class DiagBroydenJac : Jacobian
    {
        private double[] _d = [];
        public override void Setup(double[] x, double[] f, Func<double[], double[]> func)
        {
            base.Setup(x, f, func);
            _d = Enumerable.Repeat(1 / Alpha!.Value, x.Length).ToArray();
        }
        public override double[] Solve(double[] f, double tol) => f.Select((v, i) => -v / _d[i]).ToArray();
        protected override void UpdateCore(double[] x, double[] f, double[] dx, double[] df, double dxNorm, double dfNorm)
        {
            var d2 = dxNorm * dxNorm;
            for (var i = 0; i < _d.Length; i++) _d[i] -= (df[i] + _d[i] * dx[i]) * dx[i] / d2;
        }
    }

    private sealed class LinearMixingJac : Jacobian
    {
        public override double[] Solve(double[] f, double tol) => f.Select(v => -v * Alpha!.Value).ToArray();
        protected override void UpdateCore(double[] x, double[] f, double[] dx, double[] df, double dxNorm, double dfNorm) { }
    }

    private sealed class ExcitingMixingJac(double alphamax) : Jacobian
    {
        private double[] _beta = [];
        public override void Setup(double[] x, double[] f, Func<double[], double[]> func)
        {
            base.Setup(x, f, func);
            _beta = Enumerable.Repeat(Alpha!.Value, x.Length).ToArray();
        }
        public override double[] Solve(double[] f, double tol) => f.Select((v, i) => -v * _beta[i]).ToArray();
        protected override void UpdateCore(double[] x, double[] f, double[] dx, double[] df, double dxNorm, double dfNorm)
        {
            for (var i = 0; i < _beta.Length; i++)
            {
                if (f[i] * LastF[i] > 0) _beta[i] += Alpha!.Value; else _beta[i] = Alpha!.Value;
                _beta[i] = Math.Clamp(_beta[i], 0, alphamax);
            }
        }
    }

    /// <summary>Newton–Krylov: restarted GMRES on J v ≈ (F(x + ω v/‖v‖) − F(x))·‖v‖/ω.</summary>
    private sealed class KrylovJac(double? rdiff, int innerMaxIter) : Jacobian
    {
        private Func<double[], double[]> _func = null!;
        private double[] _x0 = [], _f0 = [];
        private double _omega, _rdiff;
        protected override bool UsesAlpha => false;
        public override void Setup(double[] x, double[] f, Func<double[], double[]> func)
        {
            base.Setup(x, f, func);
            _func = func; _x0 = x; _f0 = f;
            _rdiff = rdiff ?? Math.Sqrt(2.220446049250313e-16);
            Step();
        }
        private void Step() => _omega = _rdiff * Math.Max(1, MaxNorm(_x0)) / Math.Max(1, MaxNorm(_f0));
        private double[] MatVec(double[] v)
        {
            var nv = Norm(v);
            if (nv == 0) return new double[v.Length];
            var sc = _omega / nv;
            var xt = new double[v.Length];
            for (var i = 0; i < v.Length; i++) xt[i] = _x0[i] + sc * v[i];
            var ft = _func(xt);
            var r = new double[v.Length];
            for (var i = 0; i < v.Length; i++) r[i] = (ft[i] - _f0[i]) / sc;
            if (!r.All(double.IsFinite)) throw new ArgumentException("Function returned non-finite results");
            return r;
        }
        public override double[] Solve(double[] rhs, double tol) => OptimizeGmres.Solve(MatVec, rhs, tol, innerMaxIter);
        public override void Update(double[] x, double[] f) { _x0 = x; _f0 = f; LastX = x; LastF = f; Step(); }
        protected override void UpdateCore(double[] x, double[] f, double[] dx, double[] df, double dxNorm, double dfNorm) { }
    }

    public static OptimizeResult Solve(Func<double[], double[]> fun, double[] x0, RootMethod method, RootOptions o, Action<double[], double[]>? callback)
    {
        var n = x0.Length;
        var nfev = 0;
        double[] Func(double[] z)
        {
            nfev++;
            var v = fun((double[])z.Clone());
            if (v.Length != n) throw new ArgumentException($"Root needs as many equations as unknowns: got {v.Length} for {n}.");
            return v;
        }
        Jacobian jac = method switch
        {
            RootMethod.Broyden1 => new Broyden(false) { Alpha = o.Alpha },
            RootMethod.Broyden2 => new Broyden(true) { Alpha = o.Alpha },
            RootMethod.Anderson => new AndersonJac(o.Alpha, o.W0, o.AndersonM),
            RootMethod.LinearMixing => new LinearMixingJac { Alpha = o.Alpha },
            RootMethod.DiagBroyden => new DiagBroydenJac { Alpha = o.Alpha },
            RootMethod.ExcitingMixing => new ExcitingMixingJac(o.AlphaMax) { Alpha = o.Alpha },
            RootMethod.Krylov => new KrylovJac(o.Rdiff, o.InnerMaxIter),
            _ => throw new ArgumentOutOfRangeException(nameof(method)),
        };
        var fTol = o.Fatol ?? Math.Pow(2.220446049250313e-16, 1.0 / 3);
        var fRtol = o.Ftol ?? double.PositiveInfinity;
        var xTol = o.Xatol ?? double.PositiveInfinity;
        var xRtol = o.Xtol ?? double.PositiveInfinity;
        var maxiter = o.MaxIter ?? 100 * (n + 1);

        var x = (double[])x0.Clone();
        var dx = Enumerable.Repeat(double.PositiveInfinity, n).ToArray();
        var fx = Func(x);
        var fxNorm = Norm(fx);
        jac.Setup((double[])x.Clone(), fx, Func);
        const double gamma = 0.9, etaMax = 0.9999, etaThreshold = 0.1;
        var eta = 1e-3;
        double? f0Norm = null;
        var iteration = 0;
        var status = 0;
        for (var it = 0; it < maxiter; it++)
        {
            iteration++;
            var fn = MaxNorm(fx); var xn = MaxNorm(x); var dxn = MaxNorm(dx);
            f0Norm ??= fn;
            if (fn == 0) { status = 1; break; }
            if (fn <= fTol && fn / fRtol <= f0Norm && dxn <= xTol && dxn / xRtol <= xn) { status = 1; break; }

            var tol = Math.Min(eta, eta * fxNorm);
            var step = jac.Solve(fx, tol);
            for (var i = 0; i < n; i++) dx[i] = -step[i];
            if (Norm(dx) == 0) throw new InvalidOperationException("Jacobian inversion yielded zero vector. This indicates a bug in the Jacobian approximation.");

            double fxNormNew;
            if (o.LineSearch != RootLineSearch.None)
                (x, fx, fxNormNew) = LineSearch(Func, x, fx, dx, o.LineSearch);
            else
            {
                x = x.Select((v, i) => v + dx[i]).ToArray();
                fx = Func(x);
                fxNormNew = Norm(fx);
            }
            jac.Update((double[])x.Clone(), fx);
            callback?.Invoke((double[])x.Clone(), (double[])fx.Clone());
            var etaA = gamma * fxNormNew * fxNormNew / (fxNorm * fxNorm);
            eta = gamma * eta * eta < etaThreshold ? Math.Min(etaMax, etaA) : Math.Min(etaMax, Math.Max(etaA, gamma * eta * eta));
            fxNorm = fxNormNew;
        }
        if (status == 0) status = 2;
        return new OptimizeResult
        {
            X = x, FunVector = fx, Fun = 0.5 * OptimizeDense.Dot(fx, fx), Nit = iteration, Nfev = nfev, Status = status, Success = status == 1,
            Message = status == 1 ? "A solution was found at the specified tolerance." : "The maximum number of iterations allowed has been reached.",
        };
    }

    private static (double[] X, double[] Fx, double Norm) LineSearch(Func<double[], double[]> func, double[] x, double[] fx, double[] dx, RootLineSearch type)
    {
        const double rdiff = 1e-8, smin = 1e-2;
        var n = x.Length;
        double tmpS = 0; var tmpFx = fx; var tmpPhi = Math.Pow(Norm(fx), 2);
        var sNorm = Norm(x) / Norm(dx);
        double Phi(double s, bool store = true)
        {
            if (s == tmpS) return tmpPhi;
            var xt = new double[n];
            for (var i = 0; i < n; i++) xt[i] = x[i] + s * dx[i];
            var v = func(xt);
            var nv = Norm(v);
            var p = double.IsFinite(nv) ? nv * nv : double.PositiveInfinity;
            if (store) { tmpS = s; tmpPhi = p; tmpFx = v; }
            return p;
        }
        double Derphi(double s)
        {
            var ds = (Math.Abs(s) + sNorm + 1) * rdiff;
            return (Phi(s + ds, false) - Phi(s)) / ds;
        }
        double? sv;
        if (type == RootLineSearch.Wolfe)
        {
            double PhiD(double s, out double d) { var v = Phi(s); d = Derphi(s); return v; }
            sv = OptimizeLineSearch.ScalarSearchWolfe1(PhiD, tmpPhi, null, Derphi(0), 1e-4, 0.9, 50, smin, 1e-2, out _, out _);
        }
        else sv = ScalarSearchArmijo(s => Phi(s), tmpPhi, -tmpPhi, smin);
        var sStep = sv ?? 1.0;
        var xn = new double[n];
        for (var i = 0; i < n; i++) xn[i] = x[i] + sStep * dx[i];
        var fxn = sStep == tmpS ? tmpFx : func(xn);
        return (xn, fxn, Norm(fxn));
    }

    /// <summary>SciPy's <c>scalar_search_armijo</c>: quadratic then cubic backtracking.</summary>
    internal static double? ScalarSearchArmijo(Func<double, double> phi, double phi0, double derphi0, double amin, double c1 = 1e-4, double alpha0 = 1)
    {
        var phiA0 = phi(alpha0);
        if (phiA0 <= phi0 + c1 * alpha0 * derphi0) return alpha0;
        var alpha1 = -derphi0 * alpha0 * alpha0 / 2.0 / (phiA0 - phi0 - derphi0 * alpha0);
        var phiA1 = phi(alpha1);
        if (phiA1 <= phi0 + c1 * alpha1 * derphi0) return alpha1;
        while (alpha1 > amin)
        {
            var factor = alpha0 * alpha0 * alpha1 * alpha1 * (alpha1 - alpha0);
            var a = alpha0 * alpha0 * (phiA1 - phi0 - derphi0 * alpha1) - alpha1 * alpha1 * (phiA0 - phi0 - derphi0 * alpha0);
            a /= factor;
            var b = -alpha0 * alpha0 * alpha0 * (phiA1 - phi0 - derphi0 * alpha1) + alpha1 * alpha1 * alpha1 * (phiA0 - phi0 - derphi0 * alpha0);
            b /= factor;
            var alpha2 = (-b + Math.Sqrt(Math.Abs(b * b - 3 * a * derphi0))) / (3.0 * a);
            var phiA2 = phi(alpha2);
            if (phiA2 <= phi0 + c1 * alpha2 * derphi0) return alpha2;
            if (alpha1 - alpha2 > alpha1 / 2.0 || 1 - alpha2 / alpha1 < 0.96) alpha2 = alpha1 / 2.0;
            alpha0 = alpha1; alpha1 = alpha2; phiA0 = phiA1; phiA1 = phiA2;
        }
        return null;
    }

    private sealed class NoConvergence : Exception { }

    public static OptimizeResult DfSane(Func<double[], double[]> fun, double[] x0, RootOptions o, Action<double[], double[]>? callback)
    {
        var n = x0.Length;
        var maxfev = o.MaxFev > 0 ? o.MaxFev : 1000;
        var ftol = o.Ftol ?? 1e-8;
        var fatol = o.DfSaneFatol;
        const double sigmaEps = 1e-10;
        var nfev = 0;
        (double f, double[] F) Wrap(double[] z)
        {
            if (nfev >= maxfev) throw new NoConvergence();
            nfev++;
            var fv = fun((double[])z.Clone());
            if (fv.Length != n) throw new ArgumentException($"Root needs as many equations as unknowns: got {fv.Length} for {n}.");
            var nm = Norm(fv);
            return (nm * nm, fv);
        }
        var xk = (double[])x0.Clone();
        var (fk, Fk) = Wrap(xk);
        var k = 0;
        var f0 = fk;
        var sigma = 1.0;
        var F0Norm = Math.Sqrt(fk);
        var prev = new Queue<double>();
        prev.Enqueue(fk);
        double Q = 1.0, C = f0;
        var converged = false;
        var message = "too many function evaluations required";
        while (true)
        {
            var FkNorm = Math.Sqrt(fk);
            callback?.Invoke((double[])xk.Clone(), (double[])Fk.Clone());
            if (FkNorm < ftol * F0Norm + fatol) { message = "successful convergence"; converged = true; break; }
            if (Math.Abs(sigma) > 1 / sigmaEps) sigma = 1 / sigmaEps * Math.Sign(sigma);
            else if (Math.Abs(sigma) < sigmaEps) sigma = sigmaEps;
            var d = Fk.Select(v => -sigma * v).ToArray();
            var eta = f0 * f0 / ((1.0 + k) * (1.0 + k));
            double[] xp, Fp; double fp;
            try
            {
                var fbar = o.DfSaneCheng ? C : prev.Max();
                double ap = 1, am = 1;
                while (true)
                {
                    xp = xk.Select((v, i) => v + ap * d[i]).ToArray();
                    (fp, Fp) = Wrap(xp);
                    if (fp <= fbar + eta - 1e-4 * ap * ap * fk) break;
                    var atp = ap * ap * fk / (fp + (2 * ap - 1) * fk);
                    xp = xk.Select((v, i) => v - am * d[i]).ToArray();
                    (fp, Fp) = Wrap(xp);
                    if (fp <= fbar + eta - 1e-4 * am * am * fk) break;
                    var atm = am * am * fk / (fp + (2 * am - 1) * fk);
                    ap = Math.Clamp(atp, 0.1 * ap, 0.5 * ap);
                    am = Math.Clamp(atm, 0.1 * am, 0.5 * am);
                }
                if (o.DfSaneCheng)
                {
                    var qn = 0.85 * Q + 1;
                    C = (0.85 * Q * (C + eta) + fp) / qn;
                    Q = qn;
                }
            }
            catch (NoConvergence) { break; }
            double ss = 0, sy = 0;
            for (var i = 0; i < n; i++) { var s = xp[i] - xk[i]; ss += s * s; sy += s * (Fp[i] - Fk[i]); }
            sigma = ss / sy;
            xk = xp; Fk = Fp; fk = fp;
            if (!o.DfSaneCheng) { prev.Enqueue(fp); while (prev.Count > o.DfSaneM) prev.Dequeue(); }
            k++;
        }
        return new OptimizeResult
        {
            X = xk, FunVector = Fk, Fun = 0.5 * fk, Nfev = nfev, Nit = k, Success = converged, Status = converged ? 1 : 2, Message = message,
        };
    }
}

/// <summary>Restarted-free GMRES (one cycle of at most <c>maxiter</c> Arnoldi steps), with modified Gram–Schmidt and Givens rotations.</summary>
internal static class OptimizeGmres
{
    public static double[] Solve(Func<double[], double[]> a, double[] b, double rtol, int maxiter)
    {
        var n = b.Length;
        var x = new double[n];
        var beta = OptimizeDense.Norm2(b);
        if (beta == 0) return x;
        var m = Math.Min(maxiter, n);
        var v = new List<double[]> { b.Select(t => t / beta).ToArray() };
        var h = new double[m + 1, m];
        var cs = new double[m]; var sn = new double[m];
        var g = new double[m + 1];
        g[0] = beta;
        var k = 0;
        for (; k < m; k++)
        {
            var w = a(v[k]);
            for (var j = 0; j <= k; j++)
            {
                h[j, k] = OptimizeDense.Dot(w, v[j]);
                OptimizeDense.Axpy(-h[j, k], v[j], w);
            }
            h[k + 1, k] = OptimizeDense.Norm2(w);
            for (var j = 0; j < k; j++)
            {
                var t = cs[j] * h[j, k] + sn[j] * h[j + 1, k];
                h[j + 1, k] = -sn[j] * h[j, k] + cs[j] * h[j + 1, k];
                h[j, k] = t;
            }
            var r = Math.Sqrt(h[k, k] * h[k, k] + h[k + 1, k] * h[k + 1, k]);
            if (r == 0) { k++; break; }
            cs[k] = h[k, k] / r; sn[k] = h[k + 1, k] / r;
            var hk1 = h[k + 1, k];
            h[k, k] = r; h[k + 1, k] = 0;
            g[k + 1] = -sn[k] * g[k];
            g[k] = cs[k] * g[k];
            var done = Math.Abs(g[k + 1]) <= rtol * beta || hk1 == 0;
            if (!done) v.Add(w.Select(t => t / hk1).ToArray());
            if (done) { k++; break; }
        }
        var y = new double[k];
        for (var i = k - 1; i >= 0; i--)
        {
            var s = g[i];
            for (var j = i + 1; j < k; j++) s -= h[i, j] * y[j];
            y[i] = h[i, i] != 0 ? s / h[i, i] : 0;
        }
        for (var j = 0; j < k; j++) OptimizeDense.Axpy(y[j], v[j], x);
        return x;
    }
}
