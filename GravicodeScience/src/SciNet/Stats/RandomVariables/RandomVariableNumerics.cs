using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// The numerical back end of the random-variable framework: quadrature (plain and log-space),
/// summation over integers, monotone inversion and bounded minimisation. These are the
/// counterparts of scipy's <c>_tanhsinh</c>, <c>nsum</c>, <c>_bracket_root</c> +
/// <c>_chandrupatla</c> and <c>_bracket_minimum</c> + <c>_chandrupatla_minimize</c>.
/// </summary>
internal static class RandomVariableNumerics
{
    internal const double Eps = 2.220446049250313e-16;

    /// <summary>numpy's <c>spacing</c>: the distance from x to the next double away from zero.</summary>
    public static double Spacing(double x)
    {
        if (double.IsNaN(x) || double.IsInfinity(x)) return double.NaN;
        var a = Math.Abs(x);
        var s = Math.BitIncrement(a) - a;
        return x < 0 ? -s : s;
    }

    /// <summary>Tanh-sinh quadrature of f over [a, b], scipy's default relative tolerance eps^0.75.</summary>
    public static double Quad(Func<double, double> f, double a, double b, double? rtol)
    {
        if (double.IsNaN(a) || double.IsNaN(b)) return double.NaN;
        if (a == b) return 0.0;
        double G(double x)
        {
            var v = f(x);
            return double.IsNaN(v) ? 0.0 : v;
        }
        return Integrate.TanhSinh(G, a, b, rtol: rtol).Integral;
    }

    /// <summary>Tanh-sinh quadrature of f over [a, b], split at the given interior points.</summary>
    public static double QuadSplit(Func<double, double> f, double a, double b, IEnumerable<double> breakpoints, double? rtol)
    {
        var cuts = breakpoints.Where(t => t > a && t < b && double.IsFinite(t)).Distinct().Order().ToList();
        var total = 0.0;
        var lo = a;
        foreach (var c in cuts) { total += Quad(f, lo, c, rtol); lo = c; }
        return total + Quad(f, lo, b, rtol);
    }

    /// <summary>
    /// log ∫ exp(logf) over [a, b]. A coarse pass finds the largest log integrand on the nodes, and
    /// the integral of exp(logf − max) is taken at full accuracy — so a tail mass of 1e-400 is still
    /// a finite logarithm, which is all scipy's <c>log=True</c> integration is for.
    /// </summary>
    public static double LogQuad(Func<double, double> logf, double a, double b, double? rtol)
    {
        if (double.IsNaN(a) || double.IsNaN(b)) return double.NaN;
        if (a == b) return double.NegativeInfinity;
        var max = double.NegativeInfinity;
        double Probe(double x)
        {
            var l = logf(x);
            if (l > max) max = l;
            return 0.0;
        }
        Integrate.TanhSinh(Probe, a, b, maxlevel: 5, minlevel: 5);
        if (double.IsNegativeInfinity(max)) return double.NegativeInfinity;
        if (double.IsPositiveInfinity(max)) return double.PositiveInfinity;
        var shift = max;
        double G(double x)
        {
            var l = logf(x);
            return double.IsNaN(l) ? 0.0 : Math.Exp(l - shift);
        }
        var integral = Integrate.TanhSinh(G, a, b, rtol: rtol).Integral;
        return Math.Log(integral) + shift;
    }

    /// <summary>
    /// Σ f(k) over the integers in [a, b] (either end may be infinite). Finite ranges up to two
    /// million terms are summed directly; longer ones are summed outward until the terms have
    /// fallen below eps of the total and are still falling.
    /// </summary>
    public static double Sum(Func<double, double> f, double a, double b)
    {
        if (double.IsNaN(a) || double.IsNaN(b)) return double.NaN;
        var lo = Math.Ceiling(a);
        var hi = Math.Floor(b);
        if (lo > hi) return 0.0;
        if (double.IsFinite(lo) && double.IsFinite(hi) && hi - lo <= 2e6)
        {
            var s = 0.0; var c = 0.0;
            for (var k = lo; k <= hi; k++) KahanAdd(ref s, ref c, Safe(f(k)));
            return s;
        }
        var start = double.IsFinite(lo) ? lo : double.IsFinite(hi) ? hi : 0.0;
        var total = 0.0; var comp = 0.0;
        KahanAdd(ref total, ref comp, Safe(f(start)));
        if (start < hi) SumDirection(f, start + 1, hi, +1, ref total, ref comp);
        if (start > lo) SumDirection(f, start - 1, lo, -1, ref total, ref comp);
        return total;
    }

    private static void SumDirection(Func<double, double> f, double from, double to, int dir, ref double total, ref double comp)
    {
        var prev = double.PositiveInfinity;
        var quiet = 0;
        for (var k = from; dir > 0 ? k <= to : k >= to; k += dir)
        {
            var t = Safe(f(k));
            KahanAdd(ref total, ref comp, t);
            if (Math.Abs(t) <= 1e-18 * Math.Abs(total) && Math.Abs(t) <= prev) { if (++quiet >= 16) break; }
            else quiet = 0;
            if (total == 0 && t == 0 && Math.Abs(k - from) > 1e7) break;
            prev = Math.Abs(t);
        }
    }

    /// <summary>log Σ exp(logf(k)) over the integers in [a, b].</summary>
    public static double LogSum(Func<double, double> logf, double a, double b)
    {
        if (double.IsNaN(a) || double.IsNaN(b)) return double.NaN;
        var lo = Math.Ceiling(a);
        var hi = Math.Floor(b);
        if (lo > hi) return double.NegativeInfinity;
        // Pick the shift from a handful of points, then sum exp(logf - shift).
        var shift = double.NegativeInfinity;
        if (double.IsFinite(lo) && double.IsFinite(hi) && hi - lo <= 2e6)
        {
            for (var k = lo; k <= hi; k++) { var l = logf(k); if (l > shift) shift = l; }
        }
        else
        {
            var start = double.IsFinite(lo) ? lo : double.IsFinite(hi) ? hi : 0.0;
            for (var step = 0.0; step < 1e7; step = step == 0 ? 1 : step * 2)
            {
                foreach (var k in new[] { start + step, start - step })
                    if (k >= lo && k <= hi) { var l = logf(k); if (l > shift) shift = l; }
            }
        }
        if (double.IsNegativeInfinity(shift)) return double.NegativeInfinity;
        var s = Sum(k => Math.Exp(logf(k) - shift), lo, hi);
        return Math.Log(s) + shift;
    }

    private static double Safe(double v) => double.IsNaN(v) ? 0.0 : v;

    private static void KahanAdd(ref double sum, ref double comp, double value)
    {
        var y = value - comp;
        var t = sum + y;
        comp = t - sum - y;
        sum = t;
    }

    /// <summary>
    /// The x in [lo, hi] with g(x) = target for a monotone g (scipy's <c>_solve_bounded</c>): the
    /// bracket starts where scipy's <c>_guess_bracket</c> puts it and is expanded outwards.
    /// </summary>
    public static double Solve(Func<double, double> g, double target, double lo, double hi, bool increasing)
    {
        if (double.IsNaN(target)) return double.NaN;
        double start;
        if (double.IsFinite(lo) && double.IsFinite(hi)) start = 0.5 * (lo + hi);
        else if (double.IsFinite(lo)) start = lo + 1;
        else if (double.IsFinite(hi)) start = hi - 1;
        else start = 0.0;
        return StatsDistMath.InvertMonotone(g, target, lo, hi, increasing, start);
    }

    /// <summary>
    /// Smallest integer k in [lo, hi] for which <paramref name="accept"/>(k) holds, given that the
    /// predicate is monotone (false … false true … true). Returns hi + 1 if it never holds.
    /// </summary>
    public static double SmallestInteger(Func<double, bool> accept, double lo, double hi)
    {
        lo = Math.Ceiling(lo); hi = Math.Floor(hi);
        if (accept(lo)) return lo;
        // Find an accepting upper bound.
        double good;
        if (double.IsFinite(hi)) good = hi;
        else
        {
            var step = 1.0;
            good = lo + step;
            while (!accept(good))
            {
                step *= 2;
                good = lo + step;
                if (good > 1e300) return double.PositiveInfinity;
            }
        }
        if (!accept(good)) return hi + 1;
        var bad = lo;
        while (good - bad > 1)
        {
            var mid = Math.Floor(bad + (good - bad) / 2);
            if (mid <= bad) mid = bad + 1;
            if (accept(mid)) good = mid; else bad = mid;
        }
        return good;
    }

    /// <summary>
    /// A local minimiser of f on [a, b] near x0 (scipy's <c>_bracket_minimum</c> then
    /// <c>_chandrupatla_minimize</c>): the bracket grows from x0 until it straddles a minimum or
    /// hits an end of the support, in which case that end is returned.
    /// </summary>
    public static double Minimize(Func<double, double> f, double x0, double a, double b, double xatol = 0.0)
    {
        if (double.IsNaN(x0)) x0 = double.IsFinite(a) && double.IsFinite(b) ? 0.5 * (a + b) : double.IsFinite(a) ? a + 1 : double.IsFinite(b) ? b - 1 : 0.0;
        double F(double x)
        {
            var v = f(x);
            return double.IsNaN(v) ? double.PositiveInfinity : v;
        }
        // scipy's _bracket_minimum: start at x0 and x0 + 1, walk downhill growing the step by 2,
        // stopping at an end of the support if the function is still falling there.
        var p0 = Math.Clamp(x0, a, b);
        var p1 = p0 + 1.0;
        if (p1 > b) p1 = double.IsFinite(b) ? 0.5 * (p0 + b) : p1;
        if (p1 == p0) p1 = p0 - 1.0 < a ? 0.5 * (p0 + a) : p0 - 1.0;
        double f0 = F(p0), f1 = F(p1);
        if (f1 > f0) { (p0, p1) = (p1, p0); (f0, f1) = (f1, f0); }
        while (true)
        {
            var p2 = p1 + 2.0 * (p1 - p0);
            if (p2 > b) p2 = b;
            if (p2 < a) p2 = a;
            var f2 = F(p2);
            if (f2 > f1)
            {
                var lo = Math.Min(p0, p2);
                var hi = Math.Max(p0, p2);
                return Brent(F, lo, p1, hi, f1, xatol);
            }
            if (p2 == a || p2 == b) return p2;
            if (Math.Abs(p2) > 1e300) return p2;
            (p0, f0) = (p1, f1);
            (p1, f1) = (p2, f2);
        }
    }

    /// <summary>
    /// Golden-section search on a bracket xl &lt; xm &lt; xr with f(xm) lowest. Brent's parabolic
    /// steps were tried and rejected: at a density's jump (a mixture with a uniform component) the
    /// parabola keeps proposing steps of one tolerance whose function values differ only by rounding,
    /// and the bracket collapses onto a point 0.06 from the true maximum. Golden section compares
    /// points a fixed fraction of the bracket apart, so it cannot be trapped that way.
    /// </summary>
    private static double Brent(Func<double, double> f, double ax, double bx, double cx, double fbx, double xatol)
    {
        const double r = 0.6180339887498949;
        double a = Math.Min(ax, cx), b = Math.Max(ax, cx);
        double x1, x2, f1, f2;
        // The interior points must stay in the golden ratio, or a later step puts x1 beyond x2;
        // bx and f(bx) only guarantee that the bracket holds a minimum.
        _ = bx; _ = fbx;
        x1 = b - r * (b - a); f1 = f(x1);
        x2 = a + r * (b - a); f2 = f(x2);
        for (var iter = 0; iter < 400; iter++)
        {
            var tol = Math.Max(4 * Eps * Math.Max(Math.Abs(x1), Math.Abs(x2)), xatol) + 1e-300;
            if (b - a <= tol) break;
            if (f1 <= f2)
            {
                b = x2; x2 = x1; f2 = f1;
                x1 = b - r * (b - a);
                f1 = f(x1);
            }
            else
            {
                a = x1; x1 = x2; f1 = f2;
                x2 = a + r * (b - a);
                f2 = f(x2);
            }
        }
        return f1 <= f2 ? x1 : x2;
    }

    /// <summary>log(exp(x) − exp(y)) as a complex number: real for x &gt; y, with iπ added for x &lt; y.</summary>
    public static Complex LogExpXmExpY(double x, double y)
    {
        if (x == y) return new Complex(double.NegativeInfinity, 0);
        if (x > y) return new Complex(StatsDistMath.LogSubExp(x, y), 0);
        return new Complex(StatsDistMath.LogSubExp(y, x), Math.PI);
    }

    /// <summary>scipy's <c>_log_real_standardize</c>: the log of a positive real has imaginary part
    /// 0, the log of a negative real π.</summary>
    public static Complex LogRealStandardize(Complex z)
    {
        var negative = Math.Cos(z.Imaginary) < 0.5 ? true : false;
        return negative ? new Complex(z.Real, Math.PI) : new Complex(z.Real, 0);
    }

    /// <summary>The complex logarithm of a real number (iπ for negatives).</summary>
    public static Complex LogReal(double x)
        => double.IsNaN(x) ? new Complex(double.NaN, 0) : x < 0 ? new Complex(Math.Log(-x), Math.PI) : new Complex(Math.Log(x), 0);
}
