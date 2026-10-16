using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Numerical helpers shared by the distribution classes.</summary>
internal static class StatsDistMath
{
    public const double Ln2 = 0.69314718055994530942;
    public const double LnSqrt2Pi = 0.91893853320467274178;
    public const double Sqrt2 = 1.41421356237309504880;
    public const double Sqrt2Pi = 2.50662827463100050242;
    public const double EulerGamma = 0.57721566490153286061;

    /// <summary>log(1 − exp(x)) for x ≤ 0 without cancellation (Mächler 2012).</summary>
    public static double Log1mExp(double x)
    {
        if (x > 0) return double.NaN;
        return x > -Ln2 ? Math.Log(-Special.Expm1(x)) : Special.Log1p(-Math.Exp(x));
    }

    /// <summary>log(exp(a) + exp(b)).</summary>
    public static double LogAddExp(double a, double b)
    {
        if (double.IsNegativeInfinity(a)) return b;
        if (double.IsNegativeInfinity(b)) return a;
        var m = Math.Max(a, b);
        return m + Special.Log1p(Math.Exp(-Math.Abs(a - b)));
    }

    /// <summary>log(exp(a) − exp(b)) for a ≥ b.</summary>
    public static double LogSubExp(double a, double b)
    {
        if (double.IsNegativeInfinity(b)) return a;
        if (b > a) return double.NaN;
        if (a == b) return double.NegativeInfinity;
        return a + Log1mExp(b - a);
    }

    /// <summary>log(p) for a probability, but computed as log1p(−q) when the complement is known exactly.</summary>
    public static double LogFromComplement(double q) => Special.Log1p(-q);

    /// <summary>A uniform on the open interval (0, 1): never 0, never 1, so inverse CDFs stay finite.</summary>
    public static double OpenUniform(GraviRandom rng) => ((rng.NextUInt64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);

    /// <summary>
    /// Brent's root finder with a relative tolerance near machine precision. Inverting a CDF needs
    /// relative, not absolute, accuracy: a quantile of 1e-200 is meaningless to an absolute xtol.
    /// Returns the bracket end with the smaller residual if the iteration budget runs out.
    /// </summary>
    public static double Brent(Func<double, double> f, double a, double b, double fa, double fb, int maxIter = 300)
    {
        if (fa == 0) return a;
        if (fb == 0) return b;
        if (double.IsNaN(fa) || double.IsNaN(fb) || Math.Sign(fa) == Math.Sign(fb)) return double.NaN;
        double c = a, fc = fa, d = b - a, e = d;
        for (var iter = 0; iter < maxIter; iter++)
        {
            if (Math.Sign(fb) == Math.Sign(fc)) { c = a; fc = fa; d = b - a; e = d; }
            if (Math.Abs(fc) < Math.Abs(fb))
            {
                a = b; b = c; c = a;
                fa = fb; fb = fc; fc = fa;
            }
            var tol = 2.0 * 2.220446049250313e-16 * Math.Abs(b) + 1e-300;
            var m = 0.5 * (c - b);
            if (Math.Abs(m) <= tol || fb == 0) return b;
            if (Math.Abs(e) >= tol && Math.Abs(fa) > Math.Abs(fb))
            {
                double p, q, r;
                var s = fb / fa;
                if (a == c) { p = 2.0 * m * s; q = 1.0 - s; }
                else
                {
                    q = fa / fc; r = fb / fc;
                    p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                }
                if (p > 0) q = -q; else p = -p;
                if (2.0 * p < Math.Min(3.0 * m * q - Math.Abs(tol * q), Math.Abs(e * q))) { e = d; d = p / q; }
                else { d = m; e = m; }
            }
            else { d = m; e = m; }
            a = b; fa = fb;
            b += Math.Abs(d) > tol ? d : (m > 0 ? tol : -tol);
            fb = f(b);
            if (double.IsNaN(fb)) return double.NaN;
        }
        return b;
    }

    /// <summary>
    /// Solves <c>g(x) = target</c> for a monotone <paramref name="g"/> on [lo, hi] (either may be
    /// infinite), expanding a finite bracket outward first, like scipy's <c>_ppf_single</c>.
    /// <paramref name="increasing"/> says which way g runs.
    /// </summary>
    public static double InvertMonotone(Func<double, double> g, double target, double lo, double hi, bool increasing, double start = 0.0)
    {
        // Residual with the sign arranged so that it increases in x.
        double R(double x) => increasing ? g(x) - target : target - g(x);
        double a, b;
        if (double.IsFinite(lo) && double.IsFinite(hi)) { a = lo; b = hi; }
        else
        {
            if (double.IsFinite(lo)) a = lo;
            else
            {
                a = double.IsFinite(hi) ? Math.Min(hi - 1.0, start - 1.0) : Math.Min(start - 1.0, -1.0);
                var step = Math.Max(1.0, Math.Abs(a));
                while (R(a) > 0 && a > -1e300) { a -= step; step *= 2; }
            }
            if (double.IsFinite(hi)) b = hi;
            else
            {
                b = Math.Max(a + 1.0, Math.Max(start + 1.0, 1.0));
                var step = Math.Max(1.0, Math.Abs(b));
                while (R(b) < 0 && b < 1e300) { b += step; step *= 2; }
            }
        }
        // The hooks are only defined strictly inside the support, so an end point may evaluate to
        // NaN; its sign is known from monotonicity, which is all the bracket needs.
        var fa = R(a);
        var fb = R(b);
        if (double.IsNaN(fa)) fa = -1.0;
        if (double.IsNaN(fb)) fb = 1.0;
        if (fa > 0) return a;
        if (fb < 0) return b;
        return Brent(R, a, b, fa, fb);
    }

    /// <summary>Adaptive quadrature of f over [a, b] (infinite ends allowed); value only.</summary>
    public static double Quad(Func<double, double> f, double a, double b, double epsabs = 1e-13, double epsrel = 1e-12, int limit = 200)
    {
        if (a == b) return 0.0;
        return Integrate.Quad(f, a, b, epsabs: epsabs, epsrel: epsrel, limit: limit).Value;
    }

    /// <summary>
    /// Inverse of <see cref="Special.LogNdtr"/>: the x with log Φ(x) = y (scipy's <c>ndtri_exp</c>).
    /// Below y = −700 the probability itself underflows, so the asymptotic tail relation
    /// y ≈ −x²/2 − log(−x√2π) gives a start that Newton on LogNdtr then polishes.
    /// </summary>
    public static double NdtriExp(double y)
    {
        if (double.IsNaN(y) || y > 0) return double.NaN;
        if (y == 0) return double.PositiveInfinity;
        if (double.IsNegativeInfinity(y)) return double.NegativeInfinity;
        if (y >= -Ln2) return -Special.Ndtri(-Special.Expm1(y));
        if (y > -700) return Special.Ndtri(Math.Exp(y));
        var x = -Math.Sqrt(-2.0 * y);
        for (var i = 0; i < 4; i++) x = -Math.Sqrt(-2.0 * (y + Math.Log(-x) + LnSqrt2Pi));
        for (var i = 0; i < 4; i++)
        {
            var ln = Special.LogNdtr(x);
            var d = Math.Exp(-0.5 * x * x - LnSqrt2Pi - ln);
            var step = (ln - y) / d;
            x -= step;
            if (Math.Abs(step) <= 1e-16 * Math.Abs(x)) break;
        }
        return x;
    }

    /// <summary>
    /// Φ(x). Box probabilities with one open side evaluate it at ±∞ constantly, so the limits are
    /// guarded here as well as in <see cref="Special.Ndtr(double)"/> (which once returned NaN at −∞).
    /// </summary>
    public static double Phi(double x) => double.IsNegativeInfinity(x) ? 0.0 : Special.Ndtr(x);

    /// <summary>log φ(x) for the standard normal density.</summary>
    public static double LogPhi(double x) => -0.5 * x * x - LnSqrt2Pi;

    /// <summary>Horner evaluation of a polynomial with coefficients in ascending order.</summary>
    public static double PolyEval(ReadOnlySpan<double> coeffs, double x)
    {
        var r = 0.0;
        for (var i = coeffs.Length - 1; i >= 0; i--) r = r * x + coeffs[i];
        return r;
    }
}
