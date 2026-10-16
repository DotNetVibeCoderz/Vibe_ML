using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Options shared by the trust-region methods (Dogleg, TrustNcg, TrustExact, TrustKrylov).</summary>
public sealed record TrustRegionOptions : MinimizeOptions
{
    /// <summary>Starting trust radius. Default 1.</summary>
    public double InitialTrustRadius { get; init; } = 1.0;
    /// <summary>Largest trust radius. Default 1000.</summary>
    public double MaxTrustRadius { get; init; } = 1000.0;
    /// <summary>Acceptance threshold on actual/predicted reduction, in [0, 0.25). Default 0.15.</summary>
    public double Eta { get; init; } = 0.15;
    /// <summary>Gradient norm at which to stop. Default 1e-4.</summary>
    public double? Gtol { get; init; }
    /// <summary>TrustKrylov: solve the subproblem only to the Steihaug accuracy (true, default) or nearly exactly.</summary>
    public bool Inexact { get; init; } = true;
}

public static partial class Optimize
{
    static partial void RunDogleg(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeTrust.Run(problem, OptimizeTrust.Kind.Dogleg);
    static partial void RunTrustNcg(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeTrust.Run(problem, OptimizeTrust.Kind.Ncg);
    static partial void RunTrustExact(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeTrust.Run(problem, OptimizeTrust.Kind.Exact);
    static partial void RunTrustKrylov(MinimizeProblem problem, ref OptimizeResult? result) => result = OptimizeTrust.Run(problem, OptimizeTrust.Kind.Krylov);
}

/// <summary>
/// SciPy's <c>_minimize_trust_region</c> driver with its four subproblem solvers. The quadratic
/// model is lazy, as in SciPy: the gradient and Hessian at a proposed point are only evaluated
/// once the step is accepted, which is what keeps the evaluation counts equal to SciPy's.
/// </summary>
internal static class OptimizeTrust
{
    public enum Kind { Dogleg, Ncg, Exact, Krylov }

    private sealed class Model(MinimizeProblem p, OptimizeScalarFunction sf, double[] x)
    {
        public readonly double[] X = x;
        private double? _f;
        private double[]? _g;
        private double[]? _h;
        public double Fun => _f ??= sf.Fun(X);
        public double[] Jac => _g ??= sf.Grad(X);
        public double JacMag => OptimizeDense.Norm2(Jac);
        public double[] Hess => _h ??= p.EvalHess(X).AsContiguous().ToArray();
        public int N => X.Length;

        public double[] Hessp(double[] v)
        {
            if (p.Hessp is not null && p.Hess is null) return p.EvalHessp(X, v);
            var r = new double[N];
            OptimizeDense.MatVec(Hess, N, N, v, r);
            return r;
        }

        public double Value(double[] s) => Fun + OptimizeDense.Dot(Jac, s) + 0.5 * OptimizeDense.Dot(s, Hessp(s));

        // TrustExact keeps its lambda bracket across solves on the same model.
        public double PreviousRadius = -1;
        public double LambdaLb;
    }

    /// <summary>The two t with ‖z + t d‖ = Δ, ascending.</summary>
    public static (double Ta, double Tb) BoundaryIntersections(double[] z, double[] d, double radius)
    {
        var a = OptimizeDense.Dot(d, d);
        var b = 2 * OptimizeDense.Dot(z, d);
        var c = OptimizeDense.Dot(z, z) - radius * radius;
        var sq = Math.Sqrt(b * b - 4 * a * c);
        var aux = b + Math.CopySign(sq, b);
        var ta = -aux / (2 * a);
        var tb = -2 * c / aux;
        return ta <= tb ? (ta, tb) : (tb, ta);
    }

    public static OptimizeResult Run(MinimizeProblem p, Kind kind)
    {
        var o = p.OptionsAs<TrustRegionOptions>(kind.ToString());
        var gtol = o.Gtol ?? p.Tol ?? 1e-4;
        if (!(0 <= o.Eta && o.Eta < 0.25)) throw new ArgumentException("Eta must lie in [0, 0.25).");
        if (o.MaxTrustRadius <= 0) throw new ArgumentException("MaxTrustRadius must be positive.");
        if (o.InitialTrustRadius <= 0) throw new ArgumentException("InitialTrustRadius must be positive.");
        if (o.InitialTrustRadius >= o.MaxTrustRadius) throw new ArgumentException("InitialTrustRadius must be less than MaxTrustRadius.");
        var n = p.N;
        var maxiter = o.MaxIter ?? n * 200;
        var sf = p.CreateObjective(useBounds: false);
        var radius = o.InitialTrustRadius;
        var m = new Model(p, sf, (double[])p.X0.Clone());
        if (kind == Kind.Exact) _ = m.Hess;
        var k = 0; var warn = 0;
        while (m.JacMag >= gtol)
        {
            double[] step; bool hits;
            try
            {
                (step, hits) = kind switch
                {
                    Kind.Dogleg => SolveDogleg(m, radius),
                    Kind.Ncg => SolveNcg(m, radius),
                    Kind.Exact => SolveExact(m, radius),
                    _ => SolveKrylov(m, radius, o.Inexact),
                };
            }
            catch (ArithmeticException) { warn = 3; break; }
            var predicted = m.Value(step);
            var xp = new double[n];
            for (var i = 0; i < n; i++) xp[i] = m.X[i] + step[i];
            var mp = new Model(p, sf, xp);
            // SciPy's exact subproblem reads the Hessian when the model is built, so every proposal costs one.
            if (kind == Kind.Exact) _ = mp.Hess;
            var actualRed = m.Fun - mp.Fun;
            var predRed = m.Fun - predicted;
            if (predRed <= 0) { warn = 2; break; }
            var rho = actualRed / predRed;
            if (rho < 0.25) radius *= 0.25;
            else if (rho > 0.75 && hits) radius = Math.Min(2 * radius, o.MaxTrustRadius);
            if (rho > o.Eta) m = mp;
            k++;
            if (p.NotifyCallback(m.X, m.Fun)) break;
            if (m.JacMag < gtol) { warn = 0; break; }
            if (k >= maxiter) { warn = 1; break; }
        }
        string[] messages = [OptimizeMessages.Success, OptimizeMessages.MaxIter,
            "A bad approximation caused failure to predict improvement.", "A linalg error occurred, such as a non-psd Hessian."];
        // SciPy counts before it attaches the final Hessian, which may be a fresh evaluation.
        var nhev = p.Nhev;
        var fun = m.Fun; var jac = m.Jac;
        var res = new OptimizeResult
        {
            X = m.X, Fun = fun, Jac = jac, Hess = p.Hess is not null ? new NdArray((double[])m.Hess.Clone(), n, n) : null,
            Nfev = sf.Nfev, Njev = sf.Ngev, Nhev = nhev, Nit = k, Status = warn, Success = warn == 0, Message = messages[warn],
        };
        p.Nhev = nhev;
        return res;
    }

    private static (double[], bool) SolveDogleg(Model m, double radius)
    {
        var n = m.N;
        var l = (double[])m.Hess.Clone();
        if (!OptimizeDense.Cholesky(l, n)) throw new ArithmeticException("Hessian is not positive definite.");
        var pb = m.Jac.Select(v => -v).ToArray();
        OptimizeDense.CholeskySolve(l, n, pb);
        if (OptimizeDense.Norm2(pb) < radius) return (pb, false);
        var g = m.Jac;
        var bg = m.Hessp(g);
        var scale = -(OptimizeDense.Dot(g, g) / OptimizeDense.Dot(g, bg));
        var pu = g.Select(v => scale * v).ToArray();
        var puNorm = OptimizeDense.Norm2(pu);
        if (puNorm >= radius) return (pu.Select(v => v * (radius / puNorm)).ToArray(), true);
        var d = new double[n];
        for (var i = 0; i < n; i++) d[i] = pb[i] - pu[i];
        var (_, tb) = BoundaryIntersections(pu, d, radius);
        var r = new double[n];
        for (var i = 0; i < n; i++) r[i] = pu[i] + tb * d[i];
        return (r, true);
    }

    /// <summary>Steihaug–Toint truncated conjugate gradients.</summary>
    private static (double[], bool) SolveNcg(Model m, double radius)
    {
        var n = m.N;
        var jacMag = m.JacMag;
        var tolerance = Math.Min(0.5, Math.Sqrt(jacMag)) * jacMag;
        var z = new double[n];
        if (jacMag < tolerance) return (z, false);
        var r = (double[])m.Jac.Clone();
        var d = r.Select(v => -v).ToArray();
        while (true)
        {
            var bd = m.Hessp(d);
            var dBd = OptimizeDense.Dot(d, bd);
            if (dBd <= 0)
            {
                var (ta, tb) = BoundaryIntersections(z, d, radius);
                var pa = new double[n]; var pbv = new double[n];
                for (var i = 0; i < n; i++) { pa[i] = z[i] + ta * d[i]; pbv[i] = z[i] + tb * d[i]; }
                return (m.Value(pa) < m.Value(pbv) ? pa : pbv, true);
            }
            var rr = OptimizeDense.Dot(r, r);
            var alpha = rr / dBd;
            var zNext = new double[n];
            for (var i = 0; i < n; i++) zNext[i] = z[i] + alpha * d[i];
            if (OptimizeDense.Norm2(zNext) >= radius)
            {
                var (_, tb) = BoundaryIntersections(z, d, radius);
                var pbd = new double[n];
                for (var i = 0; i < n; i++) pbd[i] = z[i] + tb * d[i];
                return (pbd, true);
            }
            var rNext = new double[n];
            for (var i = 0; i < n; i++) rNext[i] = r[i] + alpha * bd[i];
            var rn2 = OptimizeDense.Dot(rNext, rNext);
            if (Math.Sqrt(rn2) < tolerance) return (zNext, false);
            var beta = rn2 / rr;
            for (var i = 0; i < n; i++) d[i] = -rNext[i] + beta * d[i];
            z = zNext; r = rNext;
        }
    }

    // ----- TrustExact: Moré–Sorensen as SciPy's IterativeSubproblem -----

    /// <summary>Upper Cholesky U (Uᵀ U = A); returns 0 or the 1-based column where it failed (LAPACK info).</summary>
    internal static int CholeskyUpper(double[] a, int n, double[] u)
    {
        Array.Clear(u);
        for (var j = 0; j < n; j++)
        {
            var d = a[j * n + j];
            for (var k = 0; k < j; k++) d -= u[k * n + j] * u[k * n + j];
            if (!(d > 0)) return j + 1;
            d = Math.Sqrt(d);
            u[j * n + j] = d;
            for (var c = j + 1; c < n; c++)
            {
                var s = a[j * n + c];
                for (var k = 0; k < j; k++) s -= u[k * n + j] * u[k * n + c];
                u[j * n + c] = s / d;
            }
        }
        return 0;
    }

    /// <summary>Solves U x = b (transpose: Uᵀ x = b) for upper-triangular U, leading k×k block.</summary>
    internal static double[] SolveUpper(double[] u, int n, double[] b, bool transpose, int k = -1)
    {
        if (k < 0) k = n;
        var x = new double[k];
        if (!transpose)
            for (var i = k - 1; i >= 0; i--)
            {
                var s = b[i];
                for (var j = i + 1; j < k; j++) s -= u[i * n + j] * x[j];
                x[i] = s / u[i * n + i];
            }
        else
            for (var i = 0; i < k; i++)
            {
                var s = b[i];
                for (var j = 0; j < i; j++) s -= u[j * n + i] * x[j];
                x[i] = s / u[i * n + i];
            }
        return x;
    }

    /// <summary>Cline–Moler–Stewart–Wilkinson estimate of the smallest singular value of U and its vector.</summary>
    private static (double S, double[] Z) SmallestSingular(double[] u, int n)
    {
        var p = new double[n];
        var w = new double[n];
        for (var k = 0; k < n; k++)
        {
            var wp = (1 - p[k]) / u[k * n + k];
            var wm = (-1 - p[k]) / u[k * n + k];
            double np = Math.Abs(wp), nm = Math.Abs(wm);
            for (var j = k + 1; j < n; j++) { np += Math.Abs(p[j] + u[k * n + j] * wp); nm += Math.Abs(p[j] + u[k * n + j] * wm); }
            var wk = np >= nm ? wp : wm;
            w[k] = wk;
            for (var j = k + 1; j < n; j++) p[j] += u[k * n + j] * wk;
        }
        var v = SolveUpper(u, n, w, false);
        var vn = OptimizeDense.Norm2(v);
        var s = OptimizeDense.Norm2(w) / vn;
        for (var i = 0; i < n; i++) v[i] /= vn;
        return (s, v);
    }

    private static (double[], bool) SolveExact(Model m, double radius) => ExactCore(m.Hess, m.Jac, m.N, radius, m);

    /// <summary>
    /// Nearly exact solution of min gᵀp + ½pᵀHp, ‖p‖ ≤ Δ (Moré &amp; Sorensen 1983; Conn, Gould
    /// &amp; Toint §7.3), including the hard case. Throws when the iteration cannot make progress.
    /// </summary>
    private static (double[], bool) ExactCore(double[] hess, double[] jac, int n, double radius, Model? state, double kEasy = 0.1, double kHard = 0.2)
    {
        const double update = 0.01, eps = 2.220446049250313e-16;
        var jacMag = OptimizeDense.Norm2(jac);
        double gLb = double.PositiveInfinity, gUb = double.NegativeInfinity, hInf = 0, hFro = 0, minDiag = double.PositiveInfinity;
        for (var i = 0; i < n; i++)
        {
            double rowAbs = 0;
            for (var j = 0; j < n; j++) { var v = hess[i * n + j]; rowAbs += Math.Abs(v); hFro += v * v; }
            var d = hess[i * n + i];
            minDiag = Math.Min(minDiag, d);
            gLb = Math.Min(gLb, d + Math.Abs(d) - rowAbs);
            gUb = Math.Max(gUb, d - Math.Abs(d) + rowAbs);
            hInf = Math.Max(hInf, rowAbs);
        }
        hFro = Math.Sqrt(hFro);
        var closeToZero = n * eps * hInf;
        var lambdaLb = Math.Max(0, Math.Max(-minDiag, jacMag / radius - Math.Min(gUb, Math.Min(hFro, hInf))));
        var lambdaUb = Math.Max(0, jacMag / radius + Math.Min(-gLb, Math.Min(hFro, hInf)));
        if (state is not null && radius < state.PreviousRadius) lambdaLb = Math.Max(state.LambdaLb, lambdaLb);
        var lambda = lambdaLb == 0 ? 0 : Math.Max(Math.Sqrt(lambdaLb * lambdaUb), lambdaLb + update * (lambdaUb - lambdaLb));

        var h = new double[n * n];
        var u = new double[n * n];
        var hits = true;
        var already = false;
        int info = 0;
        double[] p = new double[n];
        for (var iter = 0; ; iter++)
        {
            if (iter > 1000) throw new ArithmeticException("Trust-region subproblem did not converge.");
            if (already) already = false;
            else
            {
                Array.Copy(hess, h, h.Length);
                for (var i = 0; i < n; i++) h[i * n + i] += lambda;
                info = CholeskyUpper(h, n, u);
            }
            if (info == 0 && jacMag > closeToZero)
            {
                // p = -(UᵀU)⁻¹ g
                var y = SolveUpper(u, n, jac.Select(v => -v).ToArray(), true);
                p = SolveUpper(u, n, y, false);
                var pNorm = OptimizeDense.Norm2(p);
                if (pNorm <= radius && lambda == 0) { hits = false; break; }
                var w = SolveUpper(u, n, p, true);
                var wNorm = OptimizeDense.Norm2(w);
                var deltaLambda = (pNorm / wNorm) * (pNorm / wNorm) * (pNorm - radius) / radius;
                var lambdaNew = lambda + deltaLambda;
                if (pNorm < radius)
                {
                    var (sMin, zMin) = SmallestSingular(u, n);
                    var (ta, tb) = BoundaryIntersections(p, zMin, radius);
                    var stepLen = Math.Abs(ta) <= Math.Abs(tb) ? ta : tb;
                    var hp = new double[n];
                    OptimizeDense.MatVec(h, n, n, p, hp);
                    var quad = OptimizeDense.Dot(p, hp);
                    var relErr = stepLen * stepLen * sMin * sMin / (quad + lambda * radius * radius);
                    if (relErr <= kHard)
                    {
                        for (var i = 0; i < n; i++) p[i] += stepLen * zMin[i];
                        break;
                    }
                    lambdaUb = lambda;
                    lambdaLb = Math.Max(lambdaLb, lambda - sMin * sMin);
                    Array.Copy(hess, h, h.Length);
                    for (var i = 0; i < n; i++) h[i * n + i] += lambdaNew;
                    info = CholeskyUpper(h, n, u);
                    if (info == 0) { lambda = lambdaNew; already = true; }
                    else
                    {
                        lambdaLb = Math.Max(lambdaLb, lambdaNew);
                        lambda = Math.Max(Math.Sqrt(lambdaLb * lambdaUb), lambdaLb + update * (lambdaUb - lambdaLb));
                    }
                }
                else
                {
                    var relErr = Math.Abs(pNorm - radius) / radius;
                    if (relErr <= kEasy) break;
                    lambdaLb = lambda;
                    lambda = lambdaNew;
                }
            }
            else if (info == 0 && jacMag <= closeToZero)
            {
                if (lambda == 0) { p = new double[n]; hits = false; break; }
                var (sMin, zMin) = SmallestSingular(u, n);
                if (radius * radius * sMin * sMin <= kHard * lambda * radius * radius)
                {
                    p = zMin.Select(v => radius * v).ToArray();
                    break;
                }
                lambdaUb = lambda;
                lambdaLb = Math.Max(lambdaLb, lambda - sMin * sMin);
                lambda = Math.Max(Math.Sqrt(lambdaLb * lambdaUb), lambdaLb + update * (lambdaUb - lambdaLb));
            }
            else
            {
                // Factorisation failed at column k: a direction of negative curvature of the leading block bounds lambda.
                var k = info;
                double delta = -h[(k - 1) * n + (k - 1)];
                for (var i = 0; i < k - 1; i++) delta += u[i * n + (k - 1)] * u[i * n + (k - 1)];
                var v = new double[n];
                v[k - 1] = 1;
                if (k != 1)
                {
                    var rhs = new double[k - 1];
                    for (var i = 0; i < k - 1; i++) rhs[i] = -u[i * n + (k - 1)];
                    var sol = SolveUpper(u, n, rhs, false, k - 1);
                    Array.Copy(sol, v, k - 1);
                }
                var vn = OptimizeDense.Norm2(v);
                lambdaLb = Math.Max(lambdaLb, lambda + delta / (vn * vn));
                lambda = Math.Max(Math.Sqrt(lambdaLb * lambdaUb), lambdaLb + update * (lambdaUb - lambdaLb));
            }
        }
        if (state is not null) { state.LambdaLb = lambdaLb; state.PreviousRadius = radius; }
        return (p, hits);
    }

    /// <summary>
    /// Trust-region subproblem over a growing Lanczos basis (the GLTR idea of Gould, Lucidi, Roma
    /// &amp; Toint 1999): the problem is projected onto the Krylov space, solved exactly there with
    /// the Moré–Sorensen iteration, and the basis grows until the projected residual is small.
    /// Full reorthogonalisation keeps the basis honest; it costs O(n k) per step.
    /// </summary>
    private static (double[], bool) SolveKrylov(Model m, double radius, bool inexact)
    {
        var n = m.N;
        var g = m.Jac;
        var gNorm = OptimizeDense.Norm2(g);
        if (gNorm == 0) return (new double[n], false);
        var tol = inexact ? Math.Min(0.5, Math.Sqrt(gNorm)) * gNorm : Math.Max(1e-12 * gNorm, 1e-300);
        var q = new List<double[]> { g.Select(v => v / gNorm).ToArray() };
        var alpha = new List<double>();
        var beta = new List<double>();
        double[] y = [];
        var hits = false;
        for (var k = 0; k < n; k++)
        {
            var w = m.Hessp(q[k]);
            var a = OptimizeDense.Dot(w, q[k]);
            alpha.Add(a);
            for (var i = 0; i < n; i++) w[i] -= a * q[k][i] + (k > 0 ? beta[k - 1] * q[k - 1][i] : 0);
            foreach (var qq in q) { var c = OptimizeDense.Dot(w, qq); for (var i = 0; i < n; i++) w[i] -= c * qq[i]; }
            var b = OptimizeDense.Norm2(w);

            var dim = k + 1;
            var t = new double[dim * dim];
            for (var i = 0; i < dim; i++)
            {
                t[i * dim + i] = alpha[i];
                if (i + 1 < dim) { t[i * dim + i + 1] = beta[i]; t[(i + 1) * dim + i] = beta[i]; }
            }
            var gk = new double[dim];
            gk[0] = gNorm;
            (y, hits) = ExactCore(t, gk, dim, radius, null, 1e-6, 1e-6);
            var residual = b * Math.Abs(y[dim - 1]);
            if (residual <= tol || b <= 1e-14 * Math.Max(1, Math.Abs(a)) || k == n - 1) break;
            beta.Add(b);
            q.Add(w.Select(v => v / b).ToArray());
        }
        var p = new double[n];
        for (var k = 0; k < y.Length; k++) OptimizeDense.Axpy(y[k], q[k], p);
        return (p, hits);
    }
}
