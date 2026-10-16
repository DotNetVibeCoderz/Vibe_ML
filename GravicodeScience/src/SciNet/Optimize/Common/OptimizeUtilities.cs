using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Optimize
{
    /// <summary>
    /// The Rosenbrock function <c>Σ 100 (x[i+1] − x[i]²)² + (1 − x[i])²</c>; minimum 0 at all ones.
    /// </summary>
    public static double Rosen(ReadOnlySpan<double> x)
    {
        double s = 0;
        for (var i = 0; i + 1 < x.Length; i++)
        {
            var a = x[i + 1] - x[i] * x[i];
            var b = 1 - x[i];
            s += 100.0 * (a * a) + b * b;
        }
        return s;
    }

    /// <summary><see cref="Rosen(ReadOnlySpan{double})"/> as a <c>Func&lt;double[], double&gt;</c>-compatible method.</summary>
    public static double Rosen(double[] x) => Rosen((ReadOnlySpan<double>)x);

    /// <summary>Gradient of the Rosenbrock function.</summary>
    public static double[] RosenDer(double[] x)
    {
        var n = x.Length;
        var d = new double[n];
        RosenDer(x, d);
        return d;
    }

    /// <summary>Gradient of the Rosenbrock function written into <paramref name="grad"/>; returns f.</summary>
    public static double RosenDer(ReadOnlySpan<double> x, Span<double> grad)
    {
        var n = x.Length;
        grad.Clear();
        double f = 0;
        for (var i = 0; i + 1 < n; i++)
        {
            var a = x[i + 1] - x[i] * x[i];
            var b = 1 - x[i];
            f += 100.0 * (a * a) + b * b;
            grad[i] += -400.0 * x[i] * a - 2.0 * b;
            grad[i + 1] += 200.0 * a;
        }
        return f;
    }

    /// <summary>Hessian of the Rosenbrock function, shape (n, n).</summary>
    public static NdArray RosenHess(double[] x)
    {
        var n = x.Length;
        var h = new double[n * n];
        for (var i = 0; i + 1 < n; i++)
        {
            h[i * n + i] += 1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0;
            h[(i + 1) * n + i + 1] += 200.0;
            h[i * n + i + 1] = -400.0 * x[i];
            h[(i + 1) * n + i] = -400.0 * x[i];
        }
        return new NdArray(h, n, n);
    }

    /// <summary>Product of the Rosenbrock Hessian at <paramref name="x"/> with <paramref name="p"/>.</summary>
    public static double[] RosenHessProd(double[] x, double[] p)
    {
        var n = x.Length;
        var hp = new double[n];
        for (var i = 0; i + 1 < n; i++)
        {
            hp[i] += (1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0) * p[i] - 400.0 * x[i] * p[i + 1];
            hp[i + 1] += -400.0 * x[i] * p[i] + 200.0 * p[i + 1];
        }
        return hp;
    }

    /// <summary>
    /// Forward-difference gradient with an absolute step, <c>scipy.optimize.approx_fprime</c>.
    /// </summary>
    public static double[] ApproxFprime(double[] xk, Func<double[], double> f, double epsilon = 1.4901161193847656e-08)
    {
        ArgumentNullException.ThrowIfNull(f);
        var g = new double[xk.Length];
        var f0 = f((double[])xk.Clone());
        for (var i = 0; i < xk.Length; i++)
        {
            var xp = (double[])xk.Clone();
            xp[i] += epsilon;
            g[i] = (f(xp) - f0) / epsilon;
        }
        return g;
    }

    /// <summary>Forward-difference Jacobian of a vector function with an absolute step; shape (m, n).</summary>
    public static NdArray ApproxFprime(double[] xk, Func<double[], double[]> f, double epsilon = 1.4901161193847656e-08)
    {
        ArgumentNullException.ThrowIfNull(f);
        var f0 = f((double[])xk.Clone());
        var m = f0.Length; var n = xk.Length;
        var j = new double[m * n];
        for (var i = 0; i < n; i++)
        {
            var xp = (double[])xk.Clone();
            xp[i] += epsilon;
            var fp = f(xp);
            for (var r = 0; r < m; r++) j[r * n + i] = (fp[r] - f0[r]) / epsilon;
        }
        return new NdArray(j, m, n);
    }

    /// <summary>
    /// Gradient by finite differences with SciPy's <c>approx_derivative</c> conventions: relative
    /// step (√ε two-point, ∛ε three-point), scaled by <c>max(1, |x|)</c>, and kept inside
    /// <paramref name="bounds"/> by switching to one-sided formulas.
    /// </summary>
    public static double[] ApproxDerivative(Func<double[], double> f, double[] x0, FiniteDiffScheme method = FiniteDiffScheme.ThreePoint,
        double? relStep = null, double? absStep = null, Bounds? bounds = null)
    {
        ArgumentNullException.ThrowIfNull(f);
        var g = new double[x0.Length];
        OptimizeFiniteDiff.Gradient(f, x0, f((double[])x0.Clone()), g, method, absStep, relStep, bounds?.Lb, bounds?.Ub);
        return g;
    }

    /// <summary>Jacobian (m, n) of a vector function by finite differences; see the scalar overload.</summary>
    public static NdArray ApproxDerivative(Func<double[], double[]> f, double[] x0, FiniteDiffScheme method = FiniteDiffScheme.ThreePoint,
        double? relStep = null, double? absStep = null, Bounds? bounds = null)
    {
        ArgumentNullException.ThrowIfNull(f);
        var f0 = f((double[])x0.Clone());
        var j = new double[f0.Length * x0.Length];
        OptimizeFiniteDiff.Jacobian(f, x0, f0, j, method, absStep, relStep, bounds?.Lb, bounds?.Ub);
        return new NdArray(j, f0.Length, x0.Length);
    }

    /// <summary>
    /// Complex-step gradient <c>Im f(x + i h e_k) / h</c>: no subtractive cancellation, so it is
    /// accurate to machine precision for any small h. The function must be analytic and written
    /// with <see cref="Complex"/> arithmetic (no <c>Math.Abs</c>, comparisons or conjugation on the
    /// perturbed value).
    /// </summary>
    public static double[] ApproxDerivativeComplexStep(Func<Complex[], Complex> f, double[] x0, double? step = null)
    {
        ArgumentNullException.ThrowIfNull(f);
        return OptimizeFiniteDiff.ComplexStepGradient(f, x0, step);
    }

    /// <summary>
    /// Euclidean distance between an analytic gradient and a forward-difference estimate
    /// (<c>scipy.optimize.check_grad</c>). Small relative to <c>‖grad‖</c> means the gradient is right.
    /// </summary>
    public static double CheckGrad(Func<double[], double> f, Func<double[], double[]> grad, double[] x0, double epsilon = 1.4901161193847656e-08)
    {
        ArgumentNullException.ThrowIfNull(grad);
        var g = grad((double[])x0.Clone());
        var a = ApproxFprime(x0, f, epsilon);
        double s = 0;
        for (var i = 0; i < g.Length; i++) s += (g[i] - a[i]) * (g[i] - a[i]);
        return Math.Sqrt(s);
    }

    /// <summary>
    /// Strong-Wolfe line search along <paramref name="pk"/> (<c>scipy.optimize.line_search</c>,
    /// i.e. the Nocedal–Wright bracketing/zoom algorithm). Returns <c>Alpha = null</c> when no
    /// acceptable step was found.
    /// </summary>
    /// <param name="f">Objective.</param>
    /// <param name="fprime">Gradient.</param>
    /// <param name="xk">Start point.</param>
    /// <param name="pk">Search direction (should be a descent direction).</param>
    /// <param name="gfk">Gradient at <paramref name="xk"/> if already known.</param>
    /// <param name="oldFval">f(xk) if already known.</param>
    /// <param name="oldOldFval">f at the previous iterate, used to guess the first step.</param>
    /// <param name="c1">Armijo constant.</param>
    /// <param name="c2">Curvature constant.</param>
    /// <param name="amax">Maximum step.</param>
    /// <param name="maxiter">Maximum bracketing iterations.</param>
    public static LineSearchResult LineSearch(Func<double[], double> f, Func<double[], double[]> fprime, double[] xk, double[] pk,
        double[]? gfk = null, double? oldFval = null, double? oldOldFval = null, double c1 = 1e-4, double c2 = 0.9, double? amax = null, int maxiter = 10)
    {
        ArgumentNullException.ThrowIfNull(f);
        ArgumentNullException.ThrowIfNull(fprime);
        if (!(0 < c1 && c1 < c2 && c2 < 1)) throw new ArgumentException("Line search requires 0 < c1 < c2 < 1.");
        var n = xk.Length;
        int fc = 0, gc = 0;
        double[]? lastG = null;
        double Phi(double a)
        {
            fc++;
            var x = new double[n];
            for (var i = 0; i < n; i++) x[i] = xk[i] + a * pk[i];
            return f(x);
        }
        double Derphi(double a)
        {
            gc++;
            var x = new double[n];
            for (var i = 0; i < n; i++) x[i] = xk[i] + a * pk[i];
            lastG = fprime(x);
            return OptimizeDense.Dot(lastG, pk);
        }
        var g0 = gfk ?? fprime((double[])xk.Clone());
        var phi0 = oldFval ?? Phi(0.0);
        var derphi0 = OptimizeDense.Dot(g0, pk);
        var alpha = OptimizeLineSearch.ScalarSearchWolfe2(Phi, Derphi, phi0, oldOldFval, derphi0, c1, c2, amax, null, maxiter, out var phiStar, out var derStar);
        return new LineSearchResult(alpha, fc, gc, phiStar, phi0, derStar, alpha is null ? "The line search algorithm did not converge" : "CONVERGENCE");
    }
}
