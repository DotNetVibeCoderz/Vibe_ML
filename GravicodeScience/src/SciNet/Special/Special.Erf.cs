using System.Numerics;

namespace Gravicode.Science.SciNet;

public static partial class Special
{
    private const double InvSqrt2Lo = -4.833646656726457e-17; // 1/sqrt(2) - (double)(1/sqrt(2))

    /// <summary>The scaled complementary error function erfcx(x) = exp(x²) erfc(x) (scipy.special.erfcx).</summary>
    /// <remarks>
    /// S. G. Johnson's Faddeeva-package algorithm (the one SciPy uses): 100 Chebyshev pieces in
    /// y = 4/(4 + x) for x ≥ 0, a five-term continued fraction above 50, and
    /// 2 exp(x²) - erfcx(-x) for negative x (overflowing to ∞ below -26.7).
    /// </remarks>
    public static double Erfcx(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x >= 0)
        {
            if (x > 50)
            {
                if (x > 5e7) return InvSqrtPi / x;
                var x2 = x * x;
                return InvSqrtPi * (x2 * (x2 + 4.5) + 2) / (x * (x2 * (x2 + 5) + 3.75));
            }
            return ErfcxY100(400.0 / (4.0 + x));
        }
        if (x < -26.7) return double.PositiveInfinity;
        if (x < -6.1) return 2.0 * ExpSquare(x);
        return 2.0 * ExpSquare(x) - ErfcxY100(400.0 / (4.0 - x));
    }

    private static double ErfcxY100(double y100)
    {
        var k = (int)y100;
        if (k >= 100) return 1.0;
        var c = SpecialTables.ErfcxY100.AsSpan(7 * k, 7);
        var t = 2.0 * y100 - (2 * k + 1);
        return c[0] + (c[1] + (c[2] + (c[3] + (c[4] + (c[5] + c[6] * t) * t) * t) * t) * t) * t;
    }

    /// <summary>The error function erf(x) (scipy.special.erf).</summary>
    /// <remarks>Maclaurin series for |x| &lt; 1/2; 1 - erfc(|x|) beyond, with erfc from <see cref="Erfcx"/>.</remarks>
    public static double Erf(double x)
    {
        if (double.IsNaN(x)) return x;
        var ax = Math.Abs(x);
        if (ax < 0.5) return ErfSeries(x);
        if (ax > 6.0) return Math.Sign(x);
        var r = 1.0 - ExpMinusSquare(ax) * Erfcx(ax);
        return x < 0 ? -r : r;
    }

    private static double ErfSeries(double x)
    {
        // erf x = 2/sqrt(pi) Σ (-1)^n x^(2n+1) / (n! (2n+1)); |x| < 1/2 needs ~12 terms.
        var x2 = x * x;
        var term = x;
        var sum = x;
        for (var n = 1; n < 40; n++)
        {
            term *= -x2 / n;
            var t = term / (2 * n + 1);
            sum += t;
            if (Math.Abs(t) < 1e-17 * Math.Abs(sum)) break;
        }
        return 2.0 * InvSqrtPi * sum;
    }

    /// <summary>The complementary error function erfc(x) = 1 - erf(x) (scipy.special.erfc).</summary>
    /// <remarks>
    /// exp(-x²) erfcx(x) for x ≥ 1/2, with exp(-x²) formed from a fused-multiply-add split of x²
    /// so the far tail keeps full relative accuracy down to the subnormal range.
    /// </remarks>
    public static double Erfc(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x < 0.5)
        {
            if (x > -0.5) return 1.0 - ErfSeries(x);
            if (x < -6.0) return 2.0;
            return 2.0 - ExpMinusSquare(-x) * Erfcx(-x);
        }
        if (x > 27.3) return 0.0;
        return ExpMinusSquare(x) * Erfcx(x);
    }

    /// <summary>The imaginary error function erfi(x) = -i erf(ix) = 2/√π e^(x²) D(x) (scipy.special.erfi).</summary>
    public static double Erfi(double x)
    {
        if (double.IsNaN(x)) return x;
        if (Math.Abs(x) > 26.65) return x > 0 ? double.PositiveInfinity : double.NegativeInfinity;
        return 2.0 * InvSqrtPi * ExpSquare(x) * Dawsn(x);
    }

    /// <summary>Dawson's integral D(x) = e^(-x²) ∫₀ˣ e^(t²) dt (scipy.special.dawsn).</summary>
    /// <remarks>
    /// Maclaurin series Σ (-2)^k x^(2k+1)/(2k+1)!! for |x| &lt; 0.5, otherwise the continued fraction
    /// of x ₁F₁(1; 3/2; -x²), D(x) = x/(1 + 2x² - 4x²/(3 + 2x² - 8x²/(5 + 2x² - ...))), evaluated
    /// by modified Lentz; for |x| &gt; 1e8 the leading asymptotic term 1/(2x).
    /// </remarks>
    public static double Dawsn(double x)
    {
        if (double.IsNaN(x)) return x;
        var ax = Math.Abs(x);
        double r;
        if (ax < 0.5)
        {
            var x2 = x * x;
            var term = ax;
            r = ax;
            for (var k = 1; k < 60; k++)
            {
                term *= -2.0 * x2 / (2 * k + 1);
                r += term;
                if (Math.Abs(term) < 1e-17 * r) break;
            }
        }
        else if (ax > 1e8)
        {
            r = 0.5 / ax;
        }
        else
        {
            var x2 = ax * ax;
            const double tiny = 1e-300;
            var f = 1.0 + 2.0 * x2; // b0
            var c = f;
            var d = 0.0;
            for (var k = 1; k < 5000; k++)
            {
                var a = -4.0 * k * x2;
                var b = 2 * k + 1 + 2.0 * x2;
                d = b + a * d;
                if (Math.Abs(d) < tiny) d = tiny;
                c = b + a / c;
                if (Math.Abs(c) < tiny) c = tiny;
                d = 1.0 / d;
                var delta = c * d;
                f *= delta;
                if (Math.Abs(delta - 1.0) < 3e-16) break;
            }
            r = ax / f;
        }
        return x < 0 ? -r : r;
    }

    /// <summary>The inverse error function (scipy.special.erfinv).</summary>
    /// <remarks>
    /// Starts from the Cephes normal quantile and applies Halley steps on <see cref="Erf(double)"/>; for
    /// |y| &gt; 1/2 the problem is handed to <see cref="ErfcInv"/> on the exactly computed 1 - |y|
    /// so values close to ±1 keep their accuracy. ±1 give ±∞; |y| &gt; 1 gives NaN.
    /// </remarks>
    public static double ErfInv(double y)
    {
        if (double.IsNaN(y)) return y;
        if (y <= -1.0) return y == -1.0 ? double.NegativeInfinity : double.NaN;
        if (y >= 1.0) return y == 1.0 ? double.PositiveInfinity : double.NaN;
        if (y == 0.0) return y;
        var ay = Math.Abs(y);
        if (ay > 0.5)
        {
            var r = ErfcInv(1.0 - ay);
            return y < 0 ? -r : r;
        }
        if (ay < 1e-8)
        {
            // erf x = 2/√π (x - x³/3 + ...): invert the series.
            var u = y * SqrtPi / 2;
            return u + u * u * u * Math.PI / 12.0;
        }
        var x = Ndtri(0.5 * (y + 1.0)) * InvSqrt2;
        for (var i = 0; i < 3; i++)
        {
            var f = Erf(x) - y;
            var fp = 2.0 * InvSqrtPi * Math.Exp(-x * x);
            var q = f / fp;
            var dx = q / (1.0 + x * q);
            x -= dx;
            if (Math.Abs(dx) <= 1e-17 * Math.Abs(x)) break;
        }
        return x;
    }

    /// <summary>The inverse complementary error function (scipy.special.erfcinv), for 0 ≤ y ≤ 2.</summary>
    /// <remarks>Cephes normal quantile of y/2 as the start, then Halley steps on <see cref="Erfc(double)"/>.</remarks>
    public static double ErfcInv(double y)
    {
        if (double.IsNaN(y)) return y;
        if (y <= 0.0) return y == 0.0 ? double.PositiveInfinity : double.NaN;
        if (y >= 2.0) return y == 2.0 ? double.NegativeInfinity : double.NaN;
        if (y == 1.0) return 0.0;
        if (y > 1.0) return -ErfcInv(2.0 - y);
        if (y > 0.5) return ErfInv(1.0 - y);
        var x = -Ndtri(0.5 * y) * InvSqrt2;
        for (var i = 0; i < 4; i++)
        {
            var f = Erfc(x) - y;
            var fp = -2.0 * InvSqrtPi * ExpMinusSquare(x);
            if (fp == 0) break;
            var q = f / fp;
            var dx = q / (1.0 + x * q);
            x -= dx;
            if (Math.Abs(dx) <= 1e-17 * Math.Abs(x)) break;
        }
        return x;
    }

    /// <summary>The standard normal CDF Φ(x) (scipy.special.ndtr).</summary>
    /// <remarks>
    /// ½ erfc(-x/√2), with the rounding error of x/√2 carried and applied as a first-order
    /// correction; without it the lower tail is off by ~2t² ulp (1e-13 at x = -30).
    /// </remarks>
    public static double Ndtr(double x)
    {
        if (double.IsNaN(x)) return x;
        // The split below computes x/√2 - t, which is ∞ - ∞ = NaN at -∞.
        if (double.IsNegativeInfinity(x)) return 0.0;
        var t = x * InvSqrt2;
        if (x >= -1.0) return x > 9 ? 1.0 - 0.5 * Erfc(t) : 0.5 * (1.0 + Erf(t));
        var tlo = Math.FusedMultiplyAdd(x, InvSqrt2, -t) + x * InvSqrt2Lo;
        var u = -t; // erfc(u - tlo) with u > 0
        var e = Erfc(u);
        return 0.5 * e * (1.0 + tlo * 2.0 * InvSqrtPi / Erfcx(u));
    }

    /// <summary>log Φ(x), accurate far into both tails (scipy.special.log_ndtr).</summary>
    /// <remarks>
    /// Below -1 it is log(erfcx(-x/√2)/2) - x²/2 with x² split exactly, so it stays accurate to
    /// x = -1e150 where Φ itself has long underflowed; above 0 it is log1p(-Φ(-x)).
    /// </remarks>
    public static double LogNdtr(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x < -1.0)
        {
            if (double.IsNegativeInfinity(x)) return double.NegativeInfinity;
            var hi = x * x;
            var lo = Math.FusedMultiplyAdd(x, x, -hi);
            return Math.Log(0.5 * Erfcx(-x * InvSqrt2)) - 0.5 * hi - 0.5 * lo;
        }
        if (x > 0.0) return Log1p(-Ndtr(-x));
        return Math.Log(Ndtr(x));
    }

    /// <summary>The inverse of the standard normal CDF (scipy.special.ndtri).</summary>
    /// <remarks>Cephes ndtri: three rational approximations, in y - 1/2 centrally and in 1/sqrt(-2 log y) in the tails.</remarks>
    public static double Ndtri(double y0)
    {
        if (double.IsNaN(y0)) return y0;
        if (y0 == 0.0) return double.NegativeInfinity;
        if (y0 == 1.0) return double.PositiveInfinity;
        if (y0 < 0.0 || y0 > 1.0) return double.NaN;
        const double s2pi = 2.50662827463100050242E0;
        const double expm2 = 0.13533528323661269189;
        var code = true;
        var y = y0;
        if (y > 1.0 - expm2)
        {
            y = 1.0 - y;
            code = false;
        }
        if (y > expm2)
        {
            y -= 0.5;
            var y2 = y * y;
            var xc = y + y * (y2 * Polevl(y2, SpecialTables.NdtriP0) / Polevl(y2, SpecialTables.NdtriQ0));
            return xc * s2pi;
        }
        var x = Math.Sqrt(-2.0 * Math.Log(y));
        var x0 = x - Math.Log(x) / x;
        var z = 1.0 / x;
        var x1 = x < 8.0
            ? z * Polevl(z, SpecialTables.NdtriP1) / Polevl(z, SpecialTables.NdtriQ1)
            : z * Polevl(z, SpecialTables.NdtriP2) / Polevl(z, SpecialTables.NdtriQ2);
        x = x0 - x1;
        return code ? -x : x;
    }

    /// <summary>Fresnel integrals (S(x), C(x)) with the π t²/2 convention (scipy.special.fresnel).</summary>
    /// <remarks>
    /// Power series for |x| ≤ 1.5, otherwise the complex continued fraction for erfc evaluated by
    /// modified Lentz (Numerical Recipes frenel). The phase π x²/2 is reduced exactly via a
    /// fused split of x², so large arguments keep their accuracy.
    /// </remarks>
    public static (double S, double C) Fresnel(double x)
    {
        if (double.IsNaN(x)) return (x, x);
        if (double.IsInfinity(x)) return x > 0 ? (0.5, 0.5) : (-0.5, -0.5);
        var ax = Math.Abs(x);
        double s, c;
        if (ax < 1e-150) { s = 0; c = ax; }
        else if (ax <= 1.5)
        {
            double sum = 0, sums = 0, sumc = ax, sign = 1, fact = Math.PI / 2 * ax * ax, term = ax;
            var odd = true;
            var n = 3;
            for (var k = 1; k < 200; k++)
            {
                term *= fact / k;
                sum += sign * term / n;
                var test = Math.Abs(sum) * 1e-17;
                if (odd) { sign = -sign; sums = sum; sum = sumc; }
                else { sumc = sum; sum = sums; }
                if (term < test) break;
                odd = !odd;
                n += 2;
            }
            s = sums;
            c = sumc;
        }
        else
        {
            var hi = ax * ax;
            var lo = Math.FusedMultiplyAdd(ax, ax, -hi);
            Complex h;
            if (ax > 1e7)
            {
                // Asymptotic: erfc continued fraction has converged to its first term.
                h = new Complex(0, 0); // unused
                var piX = Math.PI * ax;
                var (sn, cs) = SinCosHalfPiSquare(hi, lo);
                c = 0.5 + sn / piX;
                s = 0.5 - cs / piX;
                return x < 0 ? (-s, -c) : (s, c);
            }
            var pix2 = Math.PI * hi;
            var b = new Complex(1.0, -pix2);
            var cc = new Complex(1e300, 0);
            var d = 1.0 / b;
            h = d;
            var nn = -1;
            for (var k = 2; k < 1000; k++)
            {
                nn += 2;
                var a = -(double)nn * (nn + 1);
                b += 4.0;
                d = 1.0 / (a * d + b);
                cc = b + a / cc;
                var del = cc * d;
                h *= del;
                if (Math.Abs(del.Real - 1.0) + Math.Abs(del.Imaginary) < 3e-16) break;
            }
            h *= new Complex(ax, -ax);
            var (sn2, cs2) = SinCosHalfPiSquare(hi, lo);
            var res = new Complex(0.5, 0.5) * (1.0 - new Complex(cs2, sn2) * h);
            c = res.Real;
            s = res.Imaginary;
        }
        return x < 0 ? (-s, -c) : (s, c);
    }

    /// <summary>sin and cos of (π/2)(hi + lo) with exact reduction of hi/2.</summary>
    private static (double Sin, double Cos) SinCosHalfPiSquare(double hi, double lo)
    {
        var sn = SinPi(0.5 * hi);
        var cs = CosPi(0.5 * hi);
        var e = Math.PI / 2 * lo;
        return (sn + cs * e, cs - sn * e);
    }

    /// <summary>The Faddeeva function w(z) = exp(-z²) erfc(-iz) (scipy.special.wofz).</summary>
    /// <remarks>
    /// Upper half plane: Weideman's 48-term rational approximation for |z| &lt; 6 and the Laplace
    /// continued fraction (20 levels) beyond; measured against mpmath at 40 digits the worst
    /// relative error in |w| is 1.1e-15. The lower half plane uses w(z) = 2 exp(-z²) - w(-z).
    /// On the real axis w(x) = exp(-x²) + (2i/√π) D(x) exactly. Near the real axis with |x| &gt; 5,
    /// Re w is tiny and only accurate relative to |w|, as with any method not specialised there.
    /// </remarks>
    public static Complex Wofz(Complex z)
    {
        var x = z.Real;
        var y = z.Imaginary;
        if (double.IsNaN(x) || double.IsNaN(y)) return new Complex(double.NaN, double.NaN);
        if (y == 0)
            return new Complex(ExpMinusSquare(x), 2.0 * InvSqrtPi * Dawsn(x));
        if (y < 0)
            return 2.0 * Complex.Exp(-z * z) - Wofz(-z);
        var r2 = x * x + y * y;
        if (r2 > 36.0)
        {
            Complex r = 0;
            for (var k = 20; k >= 1; k--) r = (k / 2.0) / (z - r);
            return new Complex(0, InvSqrtPi) / (z - r);
        }
        var l = SpecialTables.WeidemanL;
        var lmiz = new Complex(l + y, -x);   // L - i z
        var zz = new Complex(l - y, x) / lmiz; // (L + i z)/(L - i z)
        var a = SpecialTables.Weideman;
        Complex p = a[0];
        for (var i = 1; i < a.Length; i++) p = p * zz + a[i];
        return 2.0 * p / (lmiz * lmiz) + InvSqrtPi / lmiz;
    }

    /// <summary>
    /// The Voigt profile, the convolution of a normal density (standard deviation σ) and a
    /// Cauchy density (half-width γ) (scipy.special.voigt_profile).
    /// </summary>
    public static double VoigtProfile(double x, double sigma, double gamma)
    {
        if (double.IsNaN(x) || double.IsNaN(sigma) || double.IsNaN(gamma)) return double.NaN;
        if (sigma < 0 || gamma < 0) return double.NaN;
        if (sigma == 0)
            return gamma == 0 ? (x == 0 ? double.PositiveInfinity : 0.0) : gamma / (Math.PI * (x * x + gamma * gamma));
        if (gamma == 0) return Math.Exp(-0.5 * x * x / (sigma * sigma)) / (sigma * Math.Sqrt(2 * Math.PI));
        var z = new Complex(x, gamma) / (sigma * Sqrt2);
        return Wofz(z).Real / (sigma * Math.Sqrt(2 * Math.PI));
    }

    /// <summary>Owen's T function T(h, a) = (1/2π) ∫₀ᵃ exp(-h²(1 + x²)/2)/(1 + x²) dx (scipy.special.owens_t).</summary>
    /// <remarks>
    /// For |a| ≤ 1 the integral is computed with the Gaussian factor exp(-h²/2) taken outside and
    /// the rest by composite 20-point Gauss–Legendre over the part of [0, a] where exp(-h²x²/2)
    /// is not negligible, so it is accurate relative to T even when T underflows towards zero.
    /// |a| &gt; 1 uses T(h, a) = ½Q(h) + ½Q(ah) - Q(h)Q(ah) - T(ah, 1/a) (Q the upper normal tail,
    /// h ≥ 0), which avoids the 1 - Φ cancellation of the textbook form.
    /// </remarks>
    public static double OwensT(double h, double a)
    {
        if (double.IsNaN(h) || double.IsNaN(a)) return double.NaN;
        if (a < 0) return -OwensT(h, -a);
        if (a == 0) return 0.0;
        h = Math.Abs(h);
        if (double.IsPositiveInfinity(a))
            return h == 0 ? 0.25 : 0.5 * Ndtr(-h) * 1.0; // T(h, ∞) = ½Q(h) for h > 0 (¼ at 0)
        if (h == 0) return Math.Atan(a) / (2 * Math.PI);
        if (a <= 1.0) return OwensTSmallA(h, a);
        var ah = a * h;
        var qh = Ndtr(-h);
        var qah = Ndtr(-ah);
        return 0.5 * qh + 0.5 * qah - qh * qah - OwensTSmallA(ah, 1.0 / a);
    }

    private static double OwensTSmallA(double h, double a)
    {
        var pre = Math.Exp(-0.5 * h * h);
        if (pre == 0) return 0.0;
        // Integrand exp(-h²x²/2)/(1+x²) on [0, b], b where the Gaussian drops below 1e-17.
        var b = Math.Min(a, 9.0 / h);
        var panels = h * b > 3 ? 4 : (h * b > 1 ? 2 : 1);
        var (nodes, weights) = GaussLegendre20Rule.Value;
        var width = b / panels;
        var sum = 0.0;
        for (var p = 0; p < panels; p++)
        {
            var mid = (p + 0.5) * width;
            var half = 0.5 * width;
            for (var i = 0; i < nodes.Length; i++)
            {
                var x = mid + half * nodes[i];
                sum += weights[i] * half * Math.Exp(-0.5 * h * h * x * x) / (1 + x * x);
            }
        }
        return pre * sum / (2 * Math.PI);
    }
}
