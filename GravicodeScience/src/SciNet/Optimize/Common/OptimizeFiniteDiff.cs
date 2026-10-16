using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Finite-difference derivatives in the style of <c>scipy.optimize._numdiff.approx_derivative</c>:
/// steps scaled by <c>max(1, |x|)</c>, the step re-measured as <c>(x + h) - x</c> so the divisor is
/// the step the arithmetic actually took, and steps flipped or shortened to stay inside bounds.
/// </summary>
internal static class OptimizeFiniteDiff
{
    private const double Eps = 2.220446049250313e-16;

    /// <summary>SciPy's default relative steps: √ε for one-sided, ∛ε for central differences.</summary>
    public static double DefaultRelStep(FiniteDiffScheme scheme)
        => scheme == FiniteDiffScheme.TwoPoint ? 1.4901161193847656e-08 : 6.055454452393343e-06;

    /// <summary>
    /// Chooses the finite-difference step and whether the difference is one-sided.
    /// Returns h (signed) for one coordinate and sets <paramref name="oneSided"/> when the box
    /// forces a one-sided formula.
    /// </summary>
    public static double Step(double x0, double lb, double ub, FiniteDiffScheme scheme, double? absStep, double? relStep, out bool oneSided)
    {
        oneSided = scheme == FiniteDiffScheme.TwoPoint;
        var signX = x0 >= 0 ? 1.0 : -1.0;
        double h;
        if (absStep is { } a)
        {
            h = a;
            if ((x0 + h) - x0 == 0) h = DefaultRelStep(scheme) * signX * Math.Max(1.0, Math.Abs(x0));
        }
        else
        {
            var r = relStep ?? DefaultRelStep(scheme);
            h = r * signX * Math.Max(1.0, Math.Abs(x0));
            h = (x0 + h) - x0;
            if (h == 0) h = DefaultRelStep(scheme) * signX * Math.Max(1.0, Math.Abs(x0));
        }

        if (double.IsNegativeInfinity(lb) && double.IsPositiveInfinity(ub)) return h;

        var lowerDist = x0 - lb;
        var upperDist = ub - x0;
        if (scheme == FiniteDiffScheme.TwoPoint)
        {
            var x = x0 + h;
            var violated = x < lb || x > ub;
            var fitting = Math.Abs(h) <= Math.Max(lowerDist, upperDist);
            if (violated && fitting) h = -h;
            else if (!fitting)
                h = upperDist >= lowerDist ? upperDist : -lowerDist;
            return h;
        }

        var habs = Math.Abs(h);
        if (lowerDist >= habs && upperDist >= habs) { oneSided = false; return habs; }
        var hadj = upperDist >= lowerDist ? Math.Min(habs, 0.5 * upperDist) : -Math.Min(habs, 0.5 * lowerDist);
        var minDist = Math.Min(lowerDist, upperDist);
        if (Math.Abs(hadj) <= minDist)
        {
            // SciPy's "adjusted central": shrink to the nearer bound and keep the central formula.
            oneSided = false;
            return minDist;
        }
        oneSided = true;
        return hadj;
    }

    /// <summary>
    /// Gradient of a scalar function. <paramref name="f0"/> is f(x) (already known by the caller);
    /// returns the number of function evaluations spent.
    /// </summary>
    public static int Gradient(Func<double[], double> f, ReadOnlySpan<double> x, double f0, Span<double> grad,
        FiniteDiffScheme scheme, double? absStep, double? relStep, double[]? lb, double[]? ub)
    {
        var n = x.Length;
        var nfev = 0;
        for (var i = 0; i < n; i++)
        {
            var l = lb?[i] ?? double.NegativeInfinity;
            var u = ub?[i] ?? double.PositiveInfinity;
            var h = Step(x[i], l, u, scheme, absStep, relStep, out var oneSided);
            if (h == 0) { grad[i] = 0; continue; }
            if (scheme == FiniteDiffScheme.TwoPoint)
            {
                var xp = x.ToArray(); xp[i] = x[i] + h;
                var dx = xp[i] - x[i];
                grad[i] = (f(xp) - f0) / dx; nfev++;
            }
            else if (oneSided)
            {
                var x1 = x.ToArray(); x1[i] = x[i] + h;
                var x2 = x.ToArray(); x2[i] = x[i] + 2 * h;
                var dx = x2[i] - x[i];
                grad[i] = (-3.0 * f0 + 4.0 * f(x1) - f(x2)) / dx; nfev += 2;
            }
            else
            {
                var x1 = x.ToArray(); x1[i] = x[i] - h;
                var x2 = x.ToArray(); x2[i] = x[i] + h;
                var dx = x2[i] - x1[i];
                grad[i] = (f(x2) - f(x1)) / dx; nfev += 2;
            }
        }
        return nfev;
    }

    /// <summary>
    /// Jacobian of a vector function, row-major (m, n). Returns the number of function evaluations.
    /// </summary>
    public static int Jacobian(Func<double[], double[]> f, ReadOnlySpan<double> x, ReadOnlySpan<double> f0, double[] jac,
        FiniteDiffScheme scheme, double? absStep, double? relStep, double[]? lb, double[]? ub)
    {
        var n = x.Length;
        var m = f0.Length;
        var nfev = 0;
        for (var i = 0; i < n; i++)
        {
            var l = lb?[i] ?? double.NegativeInfinity;
            var u = ub?[i] ?? double.PositiveInfinity;
            var h = Step(x[i], l, u, scheme, absStep, relStep, out var oneSided);
            if (h == 0) { for (var r = 0; r < m; r++) jac[r * n + i] = 0; continue; }
            if (scheme == FiniteDiffScheme.TwoPoint)
            {
                var xp = x.ToArray(); xp[i] = x[i] + h;
                var dx = xp[i] - x[i];
                var fp = Checked(f(xp), m); nfev++;
                for (var r = 0; r < m; r++) jac[r * n + i] = (fp[r] - f0[r]) / dx;
            }
            else if (oneSided)
            {
                var x1 = x.ToArray(); x1[i] = x[i] + h;
                var x2 = x.ToArray(); x2[i] = x[i] + 2 * h;
                var dx = x2[i] - x[i];
                var f1 = Checked(f(x1), m); var f2 = Checked(f(x2), m); nfev += 2;
                for (var r = 0; r < m; r++) jac[r * n + i] = (-3.0 * f0[r] + 4.0 * f1[r] - f2[r]) / dx;
            }
            else
            {
                var x1 = x.ToArray(); x1[i] = x[i] - h;
                var x2 = x.ToArray(); x2[i] = x[i] + h;
                var dx = x2[i] - x1[i];
                var f1 = Checked(f(x1), m); var f2 = Checked(f(x2), m); nfev += 2;
                for (var r = 0; r < m; r++) jac[r * n + i] = (f2[r] - f1[r]) / dx;
            }
        }
        return nfev;
    }

    /// <summary>Complex-step gradient: exact to rounding because there is no subtraction.</summary>
    public static double[] ComplexStepGradient(Func<Complex[], Complex> f, ReadOnlySpan<double> x, double? step)
    {
        var n = x.Length;
        var g = new double[n];
        var z = new Complex[n];
        for (var i = 0; i < n; i++)
        {
            for (var k = 0; k < n; k++) z[k] = x[k];
            var h = step ?? 1.4901161193847656e-08 * Math.Max(1.0, Math.Abs(x[i]));
            z[i] = new Complex(x[i], h);
            g[i] = f(z).Imaginary / h;
        }
        return g;
    }

    private static double[] Checked(double[] v, int m)
    {
        if (v.Length != m) throw new ArgumentException($"The function returned {v.Length} values at a perturbed point but {m} at the base point.");
        return v;
    }
}
