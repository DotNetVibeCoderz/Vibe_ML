using System.Runtime.CompilerServices;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Special functions in the spirit of <c>scipy.special</c>: gamma and beta families, error
/// functions, incomplete integrals and the distribution helpers built on them, Bessel and Airy
/// functions, orthogonal polynomials and Gauss quadrature rules, hypergeometric and zeta
/// functions, exponential integrals, elliptic integrals and a set of numerically careful
/// elementary helpers.
/// </summary>
/// <remarks>
/// <para>
/// Every function is scalar and allocation-free; <see cref="Apply(Gravicode.Science.GraviNum.NdArray, Func{double, double})"/>
/// and the <c>NdArray</c> overloads map them over arrays. Argument conventions, edge values
/// (what is returned at a pole, for an integer that is out of domain, for a signed zero) and
/// the order of parameters follow SciPy 1.17 so that a reference value can be copied across
/// without re-deriving it.
/// </para>
/// <para>
/// The algorithms are the proven ones, named in each member's remarks: Cephes (Moshier) for
/// most of the elementary Bessel and probability kernels, Boost's Lanczos approximation and
/// incomplete-gamma helpers (which SciPy also uses), the Faddeeva package for <c>erfcx</c>,
/// Temme's method with Steed's continued fractions for real-order Bessel functions, Carlson's
/// duplication theorem for elliptic integrals and Golub–Welsch plus Newton refinement for
/// Gauss quadrature. Coefficient tables are copied verbatim from those sources; series
/// coefficients that are only known as closed forms were generated with mpmath.
/// </para>
/// <para>Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil.</para>
/// </remarks>
public static partial class Special
{
    /// <summary>2^-53, the unit roundoff Cephes calls MACHEP.</summary>
    internal const double MachEp = 1.11022302462515654042e-16;

    /// <summary>log(double.MaxValue).</summary>
    internal const double MaxLog = 7.09782712893383996843e2;

    /// <summary>log of the smallest subnormal.</summary>
    internal const double MinLog = -7.451332191019412076235e2;

    /// <summary>The Euler–Mascheroni constant.</summary>
    internal const double EulerGamma = 0.57721566490153286060651209008240243;

    /// <summary>sqrt(pi).</summary>
    internal const double SqrtPi = 1.77245385090551602729816748334114518;

    /// <summary>1/sqrt(pi).</summary>
    internal const double InvSqrtPi = 0.56418958354775628694807945156077259;

    /// <summary>log(sqrt(2 pi)).</summary>
    internal const double LnSqrt2Pi = 0.91893853320467274178032973640561764;

    /// <summary>log(pi).</summary>
    internal const double LnPi = 1.14472988584940017414342735135305871;

    /// <summary>sqrt(2).</summary>
    internal const double Sqrt2 = 1.41421356237309504880168872420969808;

    /// <summary>1/sqrt(2).</summary>
    internal const double InvSqrt2 = 0.70710678118654752440084436210484904;

    /// <summary>The g of Boost's lanczos13m53.</summary>
    internal const double LanczosG = 6.024680040776729583740234375;

    /// <summary>Evaluates a polynomial whose coefficients are stored highest degree first (Cephes polevl).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Polevl(double x, ReadOnlySpan<double> c)
    {
        var r = c[0];
        for (var i = 1; i < c.Length; i++) r = r * x + c[i];
        return r;
    }

    /// <summary>As <see cref="Polevl"/> with an implied leading coefficient of one (Cephes p1evl).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double P1evl(double x, ReadOnlySpan<double> c)
    {
        var r = x + c[0];
        for (var i = 1; i < c.Length; i++) r = r * x + c[i];
        return r;
    }

    /// <summary>Clenshaw sum of a Chebyshev series in Cephes' convention (chbevl).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Chbevl(double x, ReadOnlySpan<double> c)
    {
        double b0 = c[0], b1 = 0, b2 = 0;
        for (var i = 1; i < c.Length; i++)
        {
            b2 = b1;
            b1 = b0;
            b0 = x * b1 - b2 + c[i];
        }
        return 0.5 * (b0 - b2);
    }

    /// <summary>
    /// sin(pi x) with the argument reduced exactly first, so sin(pi n) is exactly zero and the
    /// result keeps full relative accuracy near every integer.
    /// </summary>
    internal static double SinPi(double x)
    {
        if (!double.IsFinite(x)) return double.NaN;
        var s = 1.0;
        if (x < 0) { x = -x; s = -1.0; }
        var r = x % 2.0; // exact
        if (r >= 1.0) { r -= 1.0; s = -s; }
        if (r > 0.5) r = 1.0 - r; // exact on [0.5, 1]
        if (r == 0.0) return 0.0 * s;
        return s * (r <= 0.25 ? Math.Sin(Math.PI * r) : Math.Cos(Math.PI * (0.5 - r)));
    }

    /// <summary>cos(pi x) with exact argument reduction; exactly zero at half-integers.</summary>
    internal static double CosPi(double x)
    {
        if (!double.IsFinite(x)) return double.NaN;
        var r = Math.Abs(x) % 2.0;
        if (r > 1.0) r = 2.0 - r;
        var s = 1.0;
        if (r > 0.5) { r = 1.0 - r; s = -1.0; }
        if (r == 0.5) return 0.0;
        return s * (r <= 0.25 ? Math.Cos(Math.PI * r) : Math.Sin(Math.PI * (0.5 - r)));
    }

    /// <summary>True for 0, -1, -2, ... (the poles of the gamma function).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsNonPositiveInteger(double x) => x <= 0 && x == Math.Floor(x);

    /// <summary>
    /// exp(-x^2) without the error amplification of forming x^2 first: the rounding error of
    /// x*x is recovered with a fused multiply-add and applied as a first-order correction.
    /// For x = 26 the naive form is off by ~7e-14; this one by a few ulp.
    /// </summary>
    internal static double ExpMinusSquare(double x)
    {
        var hi = x * x;
        var lo = Math.FusedMultiplyAdd(x, x, -hi);
        return Math.Exp(-hi) * (1.0 - lo);
    }

    /// <summary>exp(x^2), with the same fused correction as <see cref="ExpMinusSquare"/>.</summary>
    internal static double ExpSquare(double x)
    {
        var hi = x * x;
        var lo = Math.FusedMultiplyAdd(x, x, -hi);
        return Math.Exp(hi) * (1.0 + lo);
    }

    /// <summary>log(1 + x) - x, accurate for small x (series) and for large x.</summary>
    internal static double Log1pmx(double x)
    {
        if (Math.Abs(x) < 0.5)
        {
            // -x^2/2 + x^3/3 - ...; alternating, |x| < 1/2 needs at most ~50 terms.
            var xfac = x;
            var res = 0.0;
            for (var n = 2; n < 500; n++)
            {
                xfac *= -x;
                var term = xfac / n;
                res += term;
                if (Math.Abs(term) < MachEp * Math.Abs(res)) break;
            }
            return res;
        }
        return Math.Log(1.0 + x) - x;
    }

    /// <summary>
    /// Boost lanczos13m53 sum scaled by exp(-g), as a rational function evaluated in z or 1/z
    /// (whichever is below one) so it neither overflows nor loses digits.
    /// </summary>
    internal static double LanczosSumExpgScaled(double z)
    {
        var num = SpecialTables.LanczosNum;
        var den = SpecialTables.LanczosDen;
        double n, d;
        if (Math.Abs(z) <= 1.0)
        {
            n = num[0];
            for (var i = 1; i < num.Length; i++) n = n * z + num[i];
            d = den[0];
            for (var i = 1; i < den.Length; i++) d = d * z + den[i];
        }
        else
        {
            var y = 1.0 / z;
            n = num[^1];
            for (var i = num.Length - 2; i >= 0; i--) n = n * y + num[i];
            d = den[^1];
            for (var i = den.Length - 2; i >= 0; i--) d = d * y + den[i];
        }
        return n / d;
    }

    /// <summary>
    /// z + g - 1/2 as an unevaluated sum hi + lo (Knuth's TwoSum), so powers of it carry no
    /// rounding error amplified by the exponent.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static (double Hi, double Lo) LanczosZgh(double z)
    {
        const double c = LanczosG - 0.5;
        var hi = z + c;
        var bb = hi - z;
        var lo = (z - (hi - bb)) + (c - bb);
        return (hi, lo);
    }
}
