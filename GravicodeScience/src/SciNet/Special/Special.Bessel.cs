using System.Numerics;

namespace Gravicode.Science.SciNet;

public static partial class Special
{
    private const double TwoOverPi = 0.636619772367581343075535053490057448;
    private const double SqrtTwoOverPi = 0.797884560802865355879892119868763737;

    /// <summary>Taylor coefficients of 1/Γ(1 + z) about 0 (mpmath, 40 digits).</summary>
    private static readonly double[] RGamma1pTaylor =
    [
        1.0, 5.7721566490153286061E-1, -6.5587807152025388108E-1, -4.2002635034095235529E-2,
        1.665386113822914895E-1, -4.2197734555544336748E-2, -9.6219715278769735621E-3, 7.2189432466630995424E-3,
        -1.1651675918590651121E-3, -2.1524167411495097282E-4, 1.2805028238811618615E-4, -2.0134854780788238656E-5,
        -1.2504934821426706573E-6, 1.1330272319816958824E-6, -2.0563384169776071035E-7, 6.1160951044814158179E-9,
        5.0020076444692229301E-9, -1.1812745704870201446E-9, 1.0434267116911005105E-10, 7.782263439905071254E-12,
        -3.6968056186422057082E-12, 5.100370287454475979E-13, -2.0583260535665067832E-14, -5.3481225394230179824E-15,
        1.2267786282382607902E-15, -1.1812593016974587695E-16, 1.1866922547516003326E-18, 1.4123806553180317816E-18,
        -2.2987456844353702066E-19, 1.7144063219273374334E-20,
    ];

    // ---------------------------------------------------------------- Cephes integer orders

    /// <summary>Bessel function of the first kind of order 0 (scipy.special.j0).</summary>
    /// <remarks>Cephes j0: rational approximation on [0, 5], Hankel-type modulus/phase rationals beyond.</remarks>
    public static double J0(double x)
    {
        if (double.IsNaN(x)) return x;
        x = Math.Abs(x);
        if (double.IsPositiveInfinity(x)) return 0.0;
        if (x <= 5.0)
        {
            var z = x * x;
            if (x < 1e-5) return 1.0 - z / 4.0;
            var p = (z - 5.78318596294678452118e+00) * (z - 3.04712623436620863991e+01);
            return p * Polevl(z, SpecialTables.J0RP) / P1evl(z, SpecialTables.J0RQ);
        }
        var w = 5.0 / x;
        var q = 25.0 / (x * x);
        var pp = Polevl(q, SpecialTables.J0PP) / Polevl(q, SpecialTables.J0PQ);
        var qq = Polevl(q, SpecialTables.J0QP) / P1evl(q, SpecialTables.J0QQ);
        var (s, c) = SinCosShift(x, 0.25);
        return (pp * c - w * qq * s) * SqrtTwoOverPi / Math.Sqrt(x);
    }

    /// <summary>Bessel function of the first kind of order 1 (scipy.special.j1). Cephes j1.</summary>
    public static double J1(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x < 0) return -J1(-x);
        if (double.IsPositiveInfinity(x)) return 0.0;
        if (x <= 5.0)
        {
            var z = x * x;
            var w0 = Polevl(z, SpecialTables.J1RP) / P1evl(z, SpecialTables.J1RQ);
            return w0 * x * (z - 1.46819706421238932572e+01) * (z - 4.92184563216946036703e+01);
        }
        var w = 5.0 / x;
        var q = w * w;
        var pp = Polevl(q, SpecialTables.J1PP) / Polevl(q, SpecialTables.J1PQ);
        var qq = Polevl(q, SpecialTables.J1QP) / P1evl(q, SpecialTables.J1QQ);
        var (s, c) = SinCosShift(x, 0.75);
        return (pp * c - w * qq * s) * SqrtTwoOverPi / Math.Sqrt(x);
    }

    /// <summary>Bessel function of the second kind of order 0 (scipy.special.y0). Cephes y0; NaN for x &lt; 0, -∞ at 0.</summary>
    public static double Y0(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x < 0) return double.NaN;
        if (x == 0) return double.NegativeInfinity;
        if (double.IsPositiveInfinity(x)) return 0.0;
        if (x <= 5.0)
        {
            var z = x * x;
            var w0 = Polevl(z, SpecialTables.Y0YP) / P1evl(z, SpecialTables.Y0YQ);
            return w0 + TwoOverPi * Math.Log(x) * J0(x);
        }
        var w = 5.0 / x;
        var q = 25.0 / (x * x);
        var pp = Polevl(q, SpecialTables.J0PP) / Polevl(q, SpecialTables.J0PQ);
        var qq = Polevl(q, SpecialTables.J0QP) / P1evl(q, SpecialTables.J0QQ);
        var (s, c) = SinCosShift(x, 0.25);
        return (pp * s + w * qq * c) * SqrtTwoOverPi / Math.Sqrt(x);
    }

    /// <summary>Bessel function of the second kind of order 1 (scipy.special.y1). Cephes y1.</summary>
    public static double Y1(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x < 0) return double.NaN;
        if (x == 0) return double.NegativeInfinity;
        if (double.IsPositiveInfinity(x)) return 0.0;
        if (x <= 5.0)
        {
            var z = x * x;
            var w0 = x * (Polevl(z, SpecialTables.Y1YP) / P1evl(z, SpecialTables.Y1YQ));
            return w0 + TwoOverPi * (J1(x) * Math.Log(x) - 1.0 / x);
        }
        var w = 5.0 / x;
        var q = w * w;
        var pp = Polevl(q, SpecialTables.J1PP) / Polevl(q, SpecialTables.J1PQ);
        var qq = Polevl(q, SpecialTables.J1QP) / P1evl(q, SpecialTables.J1QQ);
        var (s, c) = SinCosShift(x, 0.75);
        return (pp * s + w * qq * c) * SqrtTwoOverPi / Math.Sqrt(x);
    }

    /// <summary>sin(x - qπ) and cos(x - qπ), without forming x - φ (whose rounding error grows with x).</summary>
    private static (double Sin, double Cos) SinCosShift(double x, double q)
    {
        var sx = Math.Sin(x);
        var cx = Math.Cos(x);
        var sp = SinPi(q);
        var cp = CosPi(q);
        return (sx * cp - cx * sp, cx * cp + sx * sp);
    }

    /// <summary>Modified Bessel function I0 (scipy.special.i0). Cephes i0 (Chebyshev, two intervals).</summary>
    public static double I0(double x)
    {
        if (double.IsNaN(x)) return x;
        x = Math.Abs(x);
        if (x > 713) return double.PositiveInfinity;
        return Math.Exp(x) * I0e(x);
    }

    /// <summary>Exponentially scaled I0: e^-|x| I0(x) (scipy.special.i0e).</summary>
    public static double I0e(double x)
    {
        if (double.IsNaN(x)) return x;
        x = Math.Abs(x);
        if (x <= 8.0) return Chbevl(x / 2.0 - 2.0, SpecialTables.I0eA);
        return Chbevl(32.0 / x - 2.0, SpecialTables.I0eB) / Math.Sqrt(x);
    }

    /// <summary>Modified Bessel function I1 (scipy.special.i1). Cephes i1.</summary>
    public static double I1(double x)
    {
        if (double.IsNaN(x)) return x;
        var ax = Math.Abs(x);
        if (ax > 713) return x > 0 ? double.PositiveInfinity : double.NegativeInfinity;
        return Math.Exp(ax) * I1e(x);
    }

    /// <summary>Exponentially scaled I1: e^-|x| I1(x) (scipy.special.i1e).</summary>
    public static double I1e(double x)
    {
        if (double.IsNaN(x)) return x;
        var z = Math.Abs(x);
        double r;
        if (z <= 8.0) r = Chbevl(z / 2.0 - 2.0, SpecialTables.I1eA) * z;
        else r = Chbevl(32.0 / z - 2.0, SpecialTables.I1eB) / Math.Sqrt(z);
        return x < 0 ? -r : r;
    }

    /// <summary>Modified Bessel function of the second kind K0 (scipy.special.k0). Cephes k0.</summary>
    public static double K0(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x < 0) return double.NaN;
        if (x == 0) return double.PositiveInfinity;
        if (x <= 2.0) return Chbevl(x * x - 2.0, SpecialTables.K0A) - Math.Log(0.5 * x) * I0(x);
        if (x > 745) return 0.0;
        return Math.Exp(-x) * Chbevl(8.0 / x - 2.0, SpecialTables.K0B) / Math.Sqrt(x);
    }

    /// <summary>Exponentially scaled K0: eˣ K0(x) (scipy.special.k0e).</summary>
    public static double K0e(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x < 0) return double.NaN;
        if (x == 0) return double.PositiveInfinity;
        if (x <= 2.0) return (Chbevl(x * x - 2.0, SpecialTables.K0A) - Math.Log(0.5 * x) * I0(x)) * Math.Exp(x);
        return Chbevl(8.0 / x - 2.0, SpecialTables.K0B) / Math.Sqrt(x);
    }

    /// <summary>Modified Bessel function of the second kind K1 (scipy.special.k1). Cephes k1.</summary>
    public static double K1(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x < 0) return double.NaN;
        if (x == 0) return double.PositiveInfinity;
        if (x <= 2.0) return Math.Log(0.5 * x) * I1(x) + Chbevl(x * x - 2.0, SpecialTables.K1A) / x;
        if (x > 745) return 0.0;
        return Math.Exp(-x) * Chbevl(8.0 / x - 2.0, SpecialTables.K1B) / Math.Sqrt(x);
    }

    /// <summary>Exponentially scaled K1: eˣ K1(x) (scipy.special.k1e).</summary>
    public static double K1e(double x)
    {
        if (double.IsNaN(x)) return x;
        if (x < 0) return double.NaN;
        if (x == 0) return double.PositiveInfinity;
        if (x <= 2.0) return (Math.Log(0.5 * x) * I1(x) + Chbevl(x * x - 2.0, SpecialTables.K1A) / x) * Math.Exp(x);
        return Chbevl(8.0 / x - 2.0, SpecialTables.K1B) / Math.Sqrt(x);
    }

    // ---------------------------------------------------------------- integer order wrappers

    /// <summary>Bessel function J_n(x) of integer order (scipy.special.jn / jv with integer order).</summary>
    public static double Jn(int n, double x)
    {
        if (n < 0) return (n % 2 == 0 ? 1 : -1) * Jn(-n, x);
        if (n == 0) return J0(x);
        if (n == 1) return J1(x);
        if (x < 0) return (n % 2 == 0 ? 1 : -1) * Jn(n, -x);
        return Jv(n, x);
    }

    /// <summary>Bessel function Y_n(x) of integer order (scipy.special.yn), by upward recurrence from Y0 and Y1 (stable).</summary>
    public static double Yn(int n, double x)
    {
        var sign = 1.0;
        if (n < 0) { n = -n; if (n % 2 == 1) sign = -1.0; }
        if (n == 0) return Y0(x);
        if (n == 1) return sign * Y1(x);
        if (double.IsNaN(x)) return x;
        if (x < 0) return double.NaN;
        if (x == 0) return -sign * double.PositiveInfinity;
        if (x > 25 + 0.5 * (double)n * n) return sign * Yv(n, x);
        double a = Y0(x), b = Y1(x);
        for (var k = 1; k < n; k++)
        {
            var c = 2.0 * k / x * b - a;
            a = b;
            b = c;
            if (double.IsNegativeInfinity(b)) break;
        }
        return sign * b;
    }

    /// <summary>Modified Bessel function K_n(x) of integer order (scipy.special.kn), by upward recurrence (stable).</summary>
    public static double Kn(int n, double x)
    {
        n = Math.Abs(n);
        if (n == 0) return K0(x);
        if (n == 1) return K1(x);
        if (double.IsNaN(x)) return x;
        if (x < 0) return double.NaN;
        if (x == 0) return double.PositiveInfinity;
        if (x > 2 && x > 0.1 * n) return Kv(n, x);
        double a = K0(x), b = K1(x);
        for (var k = 1; k < n; k++)
        {
            var c = 2.0 * k / x * b + a;
            a = b;
            b = c;
            if (double.IsPositiveInfinity(b)) break;
        }
        return b;
    }

    // ---------------------------------------------------------------- real order

    /// <summary>Bessel function of the first kind of real order v (scipy.special.jv).</summary>
    /// <remarks>
    /// Temme's series (x &lt; 2) or Steed's CF2 (x ≥ 2) for the order reduced to |μ| ≤ 1/2, with
    /// CF1 and downward recurrence for the rest (Numerical Recipes bessjy, Temme 1975); Hankel's
    /// asymptotic expansion when x &gt; 25 + v²/2, with the phase computed as sin/cos of x combined
    /// with exact sin/cos of (v/2 + 1/4)π. Negative order uses the reflection formula; negative x
    /// is defined only for integer order (NaN otherwise, as in SciPy).
    /// </remarks>
    public static double Jv(double v, double x)
    {
        if (double.IsNaN(v) || double.IsNaN(x)) return double.NaN;
        var vInt = v == Math.Floor(v);
        if (x < 0)
        {
            if (!vInt) return double.NaN;
            var r = Jv(v, -x);
            return (Math.Abs(v % 2.0) == 1.0) ? -r : r;
        }
        if (v < 0)
        {
            if (vInt) return (Math.Abs(v % 2.0) == 1.0 ? -1 : 1) * Jv(-v, x);
            if (x == 0) return GammaSgn(1 + v) * double.PositiveInfinity;
            var (jp, yp) = BesselJY(-v, x);
            return CosPi(v) * jp + SinPi(v) * yp;
        }
        if (x == 0) return v == 0 ? 1.0 : 0.0;
        if (double.IsPositiveInfinity(x)) return 0.0;
        if (v == 0) return J0(x);
        if (v == 1) return J1(x);
        return BesselJY(v, x).J;
    }

    /// <summary>Bessel function of the second kind of real order v (scipy.special.yv); NaN for x &lt; 0, -∞ at 0.</summary>
    public static double Yv(double v, double x)
    {
        if (double.IsNaN(v) || double.IsNaN(x)) return double.NaN;
        if (x < 0) return double.NaN;
        if (x == 0) return double.NegativeInfinity;
        if (double.IsPositiveInfinity(x)) return 0.0;
        var vInt = v == Math.Floor(v);
        if (v < 0)
        {
            if (vInt) return (Math.Abs(v % 2.0) == 1.0 ? -1 : 1) * Yv(-v, x);
            var (jp, yp) = BesselJY(-v, x);
            return -SinPi(v) * jp + CosPi(v) * yp;
        }
        if (v == 0) return Y0(x);
        if (v == 1) return Y1(x);
        return BesselJY(v, x).Y;
    }

    /// <summary>Hankel function of the first kind H⁽¹⁾_v(x) = J_v(x) + i Y_v(x) for real x &gt; 0 (scipy.special.hankel1).</summary>
    public static Complex Hankel1(double v, double x) => new(Jv(v, x), Yv(v, x));

    /// <summary>Hankel function of the second kind H⁽²⁾_v(x) = J_v(x) - i Y_v(x) for real x &gt; 0 (scipy.special.hankel2).</summary>
    public static Complex Hankel2(double v, double x) => new(Jv(v, x), -Yv(v, x));

    /// <summary>J_ν and Y_ν for ν ≥ 0, x &gt; 0.</summary>
    /// <remarks>
    /// Where x &gt; ν + 1 the order is reduced to μ ∈ [0, 1) and J, Y are carried up by forward
    /// recurrence (stable in the oscillatory region for both). Steed's CF1 at the full order
    /// would need ~x terms there, and its rounding accumulates to 1e-12 by x = 150.
    /// </remarks>
    internal static (double J, double Y) BesselJY(double nu, double x)
    {
        if (x > 25 + 0.5 * nu * nu) return HankelAsymptotic(nu, x);
        if (x <= nu + 1 || nu < 2)
        {
            var (j, y, _, _) = BesselJYCore(nu, x);
            return (j, y);
        }
        var n = (int)Math.Floor(nu);
        var mu = nu - n;
        var (j0, y0) = BesselJYSmall(mu, x);
        var (j1, y1) = BesselJYSmall(mu + 1, x);
        for (var k = 1; k < n; k++)
        {
            var f = 2 * (mu + k) / x;
            var j2 = f * j1 - j0;
            var y2 = f * y1 - y0;
            j0 = j1; j1 = j2;
            y0 = y1; y1 = y2;
        }
        return (j1, y1);
    }

    private static (double J, double Y) BesselJYSmall(double nu, double x)
    {
        if (x > 25 + 0.5 * nu * nu) return HankelAsymptotic(nu, x);
        var (j, y, _, _) = BesselJYCore(nu, x);
        return (j, y);
    }

    private static (double J, double Y) HankelAsymptotic(double nu, double x)
    {
        var mu = 4 * nu * nu;
        double p = 0, q = 0, term = 1;
        var inv8x = 1.0 / (8 * x);
        var last = double.PositiveInfinity;
        for (var k = 0; k < 200; k++)
        {
            // term_k = Π_{i=1..k} (mu - (2i-1)^2) / (k! (8x)^k)
            if (k > 0)
            {
                var o = 2 * k - 1;
                term *= (mu - (double)o * o) / k * inv8x;
            }
            var at = Math.Abs(term);
            if (at > last && k > 2) break;
            switch (k % 4)
            {
                case 0: p += term; break;
                case 1: q += term; break;
                case 2: p -= term; break;
                default: q -= term; break;
            }
            last = at;
            if (at < 1e-17 * Math.Max(Math.Abs(p), 1e-300)) break;
            if (term == 0) break;
        }
        // χ = x - (ν/2 + 1/4)π
        var (s, c) = SinCosShift(x, 0.5 * nu + 0.25);
        var f = Math.Sqrt(2.0 / (Math.PI * x));
        return (f * (p * c - q * s), f * (p * s + q * c));
    }

    /// <summary>
    /// Numerical Recipes bessjy (Temme + Steed), returning J_ν, Y_ν, J'_ν, Y'_ν for ν ≥ 0, x &gt; 0.
    /// The gamma-function factors of Temme's series are taken from the Taylor series of 1/Γ(1 ± μ)
    /// rather than NR's single-precision Chebyshev fits.
    /// </summary>
    private static (double J, double Y, double Jp, double Yp) BesselJYCore(double xnu, double x)
    {
        const double eps = 3e-16, fpmin = 1e-300; // a 1e-16 test can stall at |del - 1| = 1 ulp and run to maxit
        const int maxit = 1_000_000;
        var nl = x < 2.0 ? (int)(xnu + 0.5) : Math.Max(0, (int)(xnu - x + 1.5));
        var xmu = xnu - nl;
        var xmu2 = xmu * xmu;
        var xi = 1.0 / x;
        var xi2 = 2.0 * xi;
        var w = xi2 / Math.PI;
        var isign = 1;
        var h = xnu * xi;
        if (h < fpmin) h = fpmin;
        var b = xi2 * xnu;
        var d = 0.0;
        var c = h;
        int i;
        for (i = 1; i <= maxit; i++)
        {
            b += xi2;
            d = b - d;
            if (Math.Abs(d) < fpmin) d = fpmin;
            c = b - 1.0 / c;
            if (Math.Abs(c) < fpmin) c = fpmin;
            d = 1.0 / d;
            var del = c * d;
            h = del * h;
            if (d < 0.0) isign = -isign;
            if (Math.Abs(del - 1.0) < eps) break;
        }
        var rjl = isign * fpmin;
        var rjpl = h * rjl;
        var rjl1 = rjl;
        var rjp1 = rjpl;
        var fact = xnu * xi;
        for (var l = nl; l >= 1; l--)
        {
            // (μ + l)/x formed afresh each step: NR's running "fact -= 1/x" drifts by ~nl ulp,
            // and that drift lands in the result as a relative error of ~1e-12 at order 100.
            fact = (xmu + l) / x;
            var rjtemp = fact * rjl + rjpl;
            fact = (xmu + l - 1) / x;
            rjpl = fact * rjtemp - rjl;
            rjl = rjtemp;
            if (Math.Abs(rjl) > 1e250)
            {
                rjl *= 1e-250; rjpl *= 1e-250; rjl1 *= 1e-250; rjp1 *= 1e-250;
            }
        }
        if (rjl == 0.0) rjl = eps;
        var f = rjpl / rjl;
        double rjmu, rymu, rymup, ry1;
        if (x < 2.0)
        {
            var x2 = 0.5 * x;
            var pimu = Math.PI * xmu;
            fact = Math.Abs(pimu) < eps ? 1.0 : pimu / Math.Sin(pimu);
            d = -Math.Log(x2);
            var e = xmu * d;
            var fact2 = Math.Abs(e) < eps ? 1.0 : Math.Sinh(e) / e;
            var (gam1, gam2, gampl, gammi) = TemmeGammas(xmu);
            var ff = 2.0 / Math.PI * fact * (gam1 * Math.Cosh(e) + gam2 * fact2 * d);
            e = Math.Exp(e);
            var p = e / (gampl * Math.PI);
            var q = 1.0 / (e * Math.PI * gammi);
            var pimu2 = 0.5 * pimu;
            var fact3 = Math.Abs(pimu2) < eps ? 1.0 : Math.Sin(pimu2) / pimu2;
            var r = Math.PI * pimu2 * fact3 * fact3;
            c = 1.0;
            d = -x2 * x2;
            var sum = ff + r * q;
            var sum1 = p;
            for (i = 1; i <= maxit; i++)
            {
                ff = (i * ff + p + q) / (i * (double)i - xmu2);
                c *= d / i;
                p /= i - xmu;
                q /= i + xmu;
                var del = c * (ff + r * q);
                sum += del;
                var del1 = c * p - i * del;
                sum1 += del1;
                if (Math.Abs(del) < (1.0 + Math.Abs(sum)) * eps * 0.1) break;
            }
            rymu = -sum;
            ry1 = -sum1 * xi2;
            rymup = xmu * xi * rymu - ry1;
            rjmu = w / (rymup - f * rymu);
        }
        else
        {
            var a = 0.25 - xmu2;
            var p = -0.5 * xi;
            var q = 1.0;
            var br = 2.0 * x;
            var bi = 2.0;
            fact = a * xi / (p * p + q * q);
            var cr = br + q * fact;
            var ci = bi + p * fact;
            var den = br * br + bi * bi;
            var dr = br / den;
            var di = -bi / den;
            var dlr = cr * dr - ci * di;
            var dli = cr * di + ci * dr;
            var temp = p * dlr - q * dli;
            q = p * dli + q * dlr;
            p = temp;
            for (i = 2; i <= maxit; i++)
            {
                a += 2 * (i - 1);
                bi += 2.0;
                dr = a * dr + br;
                di = a * di + bi;
                if (Math.Abs(dr) + Math.Abs(di) < fpmin) dr = fpmin;
                fact = a / (cr * cr + ci * ci);
                cr = br + cr * fact;
                ci = bi - ci * fact;
                if (Math.Abs(cr) + Math.Abs(ci) < fpmin) cr = fpmin;
                den = dr * dr + di * di;
                dr /= den;
                di /= -den;
                dlr = cr * dr - ci * di;
                dli = cr * di + ci * dr;
                temp = p * dlr - q * dli;
                q = p * dli + q * dlr;
                p = temp;
                if (Math.Abs(dlr - 1.0) + Math.Abs(dli) < eps) break;
            }
            var gam = (p - f) / q;
            rjmu = Math.Sqrt(w / ((p - f) * gam + q));
            rjmu = rjl < 0 ? -rjmu : rjmu;
            rymu = rjmu * gam;
            rymup = rymu * (p + q / gam);
            ry1 = xmu * xi * rymu - rymup;
        }
        fact = rjmu / rjl;
        var rj = rjl1 * fact;
        var rjp = rjp1 * fact;
        for (i = 1; i <= nl; i++)
        {
            var rytemp = (xmu + i) * xi2 * ry1 - rymu;
            rymu = ry1;
            ry1 = rytemp;
            if (double.IsInfinity(ry1)) { rymu = ry1; break; }
        }
        var ry = rymu;
        var ryp = xnu * xi * rymu - ry1;
        return (rj, ry, rjp, ryp);
    }

    /// <summary>
    /// Temme's Γ₁(μ) = (1/Γ(1-μ) - 1/Γ(1+μ))/(2μ) and Γ₂(μ) = (1/Γ(1-μ) + 1/Γ(1+μ))/2 plus
    /// 1/Γ(1+μ), 1/Γ(1-μ), for |μ| ≤ 1/2, from the even/odd parts of the series of 1/Γ(1+z) —
    /// no cancellation as μ → 0.
    /// </summary>
    private static (double Gam1, double Gam2, double GamPl, double GamMi) TemmeGammas(double mu)
    {
        var c = RGamma1pTaylor;
        double even = 0, odd = 0;
        var m2 = mu * mu;
        for (var k = c.Length - 1; k >= 0; k--)
        {
            if (k % 2 == 0) even = even * m2 + c[k];
        }
        for (var k = c.Length - 1; k >= 1; k--)
        {
            if (k % 2 == 1) odd = odd * m2 + c[k];
        }
        // 1/Γ(1+μ) = even + μ·odd, 1/Γ(1-μ) = even - μ·odd
        var gampl = even + mu * odd;
        var gammi = even - mu * odd;
        return (-odd, even, gampl, gammi);
    }

    /// <summary>Modified Bessel function of the first kind of real order (scipy.special.iv).</summary>
    /// <remarks>
    /// Numerical Recipes bessik (Temme for x &lt; 2, Steed's CF2 otherwise, CF1 for the ratio and the
    /// Wronskian for normalisation) evaluated in exponentially scaled form, or the large-argument
    /// expansion DLMF 10.40.1 when x &gt; 25 + v²/2. Negative non-integer order uses
    /// I_(-ν) = I_ν + (2/π) sin(νπ) K_ν; negative x is defined for integer order only.
    /// </remarks>
    public static double Iv(double v, double x)
    {
        if (double.IsNaN(v) || double.IsNaN(x)) return double.NaN;
        if (x < 0)
        {
            if (v != Math.Floor(v)) return double.NaN;
            var r = Iv(v, -x);
            return Math.Abs(v % 2.0) == 1.0 ? -r : r;
        }
        if (x == 0)
        {
            if (v == 0) return 1.0;
            if (v > 0 || v == Math.Floor(v)) return 0.0;
            return GammaSgn(1 + v) * double.PositiveInfinity;
        }
        if (double.IsPositiveInfinity(x)) return double.PositiveInfinity;
        var ive = Ive(v, x);
        if (x > 700)
        {
            var lg = x + Math.Log(Math.Abs(ive));
            return lg > MaxLog ? Math.Sign(ive) * double.PositiveInfinity : Math.Sign(ive) * Math.Exp(lg);
        }
        return ive * Math.Exp(x);
    }

    /// <summary>Exponentially scaled modified Bessel function e^-|x| I_v(x) (scipy.special.ive).</summary>
    public static double Ive(double v, double x)
    {
        if (double.IsNaN(v) || double.IsNaN(x)) return double.NaN;
        if (x < 0)
        {
            if (v != Math.Floor(v)) return double.NaN;
            var r = Ive(v, -x);
            return Math.Abs(v % 2.0) == 1.0 ? -r : r;
        }
        if (x == 0) return Iv(v, 0);
        if (double.IsPositiveInfinity(x)) return 0.0;
        var nu = Math.Abs(v);
        if (v < 0 && v == Math.Floor(v)) v = nu;
        if (nu == 0) return I0e(x);
        if (nu == 1) { if (v > 0) return I1e(x); }
        var (ie, ke) = BesselIKScaled(nu, x);
        if (v >= 0) return ie;
        // I_(-ν) = I_ν + (2/π) sin(νπ) K_ν, scaled: K e^-x = ke e^-2x
        return ie + 2.0 / Math.PI * SinPi(nu) * ke * Math.Exp(-2 * x);
    }

    /// <summary>Modified Bessel function of the second kind of real order (scipy.special.kv). K_(-v) = K_v.</summary>
    public static double Kv(double v, double x)
    {
        if (double.IsNaN(v) || double.IsNaN(x)) return double.NaN;
        if (x < 0) return double.NaN;
        if (x == 0) return double.PositiveInfinity;
        if (double.IsPositiveInfinity(x)) return 0.0;
        var nu = Math.Abs(v);
        if (nu == 0) return K0(x);
        if (nu == 1) return K1(x);
        var ke = BesselIKScaled(nu, x).Ke;
        if (x > 700) return Math.Exp(Math.Log(ke) - x);
        return ke * Math.Exp(-x);
    }

    /// <summary>Exponentially scaled e^x K_v(x) (scipy.special.kve).</summary>
    public static double Kve(double v, double x)
    {
        if (double.IsNaN(v) || double.IsNaN(x)) return double.NaN;
        if (x < 0) return double.NaN;
        if (x == 0) return double.PositiveInfinity;
        if (double.IsPositiveInfinity(x)) return 0.0;
        var nu = Math.Abs(v);
        if (nu == 0) return K0e(x);
        if (nu == 1) return K1e(x);
        return BesselIKScaled(nu, x).Ke;
    }

    /// <summary>(e^-x I_ν(x), e^x K_ν(x)) for ν ≥ 0, x &gt; 0.</summary>
    internal static (double Ie, double Ke) BesselIKScaled(double nu, double x)
    {
        if (x > 25 + 0.5 * nu * nu) return BesselIKAsymptotic(nu, x);
        var r = BesselIKCore(nu, x);
        return (r.Ie, r.Ke);
    }

    private static (double Ie, double Ke) BesselIKAsymptotic(double nu, double x)
    {
        var mu = 4 * nu * nu;
        double si = 1, sk = 1, term = 1, last = double.PositiveInfinity;
        var inv8x = 1.0 / (8 * x);
        for (var k = 1; k < 200; k++)
        {
            var o = 2 * k - 1;
            term *= (mu - (double)o * o) / k * inv8x;
            var at = Math.Abs(term);
            if (at > last) break;
            last = at;
            si += (k % 2 == 0) ? term : -term;
            sk += term;
            if (at < 1e-17) break;
        }
        return (si / Math.Sqrt(2 * Math.PI * x), sk * Math.Sqrt(Math.PI / (2 * x)));
    }

    /// <summary>Numerical Recipes bessik in exponentially scaled form; also returns scaled derivatives.</summary>
    private static (double Ie, double Ke, double Iep, double Kep) BesselIKCore(double xnu, double x)
    {
        const double eps = 3e-16, fpmin = 1e-300; // a 1e-16 test can stall at |del - 1| = 1 ulp and run to maxit
        const int maxit = 1_000_000;
        var nl = (int)(xnu + 0.5);
        var xmu = xnu - nl;
        var xmu2 = xmu * xmu;
        var xi = 1.0 / x;
        var xi2 = 2.0 * xi;
        var h = xnu * xi;
        if (h < fpmin) h = fpmin;
        var b = xi2 * xnu;
        var d = 0.0;
        var c = h;
        int i;
        for (i = 1; i <= maxit; i++)
        {
            b += xi2;
            d = 1.0 / (b + d);
            c = b + 1.0 / c;
            var del = c * d;
            h = del * h;
            if (Math.Abs(del - 1.0) < eps) break;
        }
        var ril = fpmin;
        var ripl = h * ril;
        var ril1 = ril;
        var rip1 = ripl;
        var fact = xnu * xi;
        for (var l = nl; l >= 1; l--)
        {
            fact = (xmu + l) / x;
            var ritemp = fact * ril + ripl;
            fact = (xmu + l - 1) / x;
            ripl = fact * ritemp + ril;
            ril = ritemp;
            if (Math.Abs(ril) > 1e250)
            {
                ril *= 1e-250; ripl *= 1e-250; ril1 *= 1e-250; rip1 *= 1e-250;
            }
        }
        var f = ripl / ril;
        double rkmu, rk1;
        if (x < 2.0)
        {
            var x2 = 0.5 * x;
            var pimu = Math.PI * xmu;
            fact = Math.Abs(pimu) < eps ? 1.0 : pimu / Math.Sin(pimu);
            d = -Math.Log(x2);
            var e = xmu * d;
            var fact2 = Math.Abs(e) < eps ? 1.0 : Math.Sinh(e) / e;
            var (gam1, gam2, gampl, gammi) = TemmeGammas(xmu);
            var ff = fact * (gam1 * Math.Cosh(e) + gam2 * fact2 * d);
            var sum = ff;
            e = Math.Exp(e);
            var p = 0.5 * e / gampl;
            var q = 0.5 / (e * gammi);
            c = 1.0;
            d = x2 * x2;
            var sum1 = p;
            for (i = 1; i <= maxit; i++)
            {
                ff = (i * ff + p + q) / (i * (double)i - xmu2);
                c *= d / i;
                p /= i - xmu;
                q /= i + xmu;
                var del = c * ff;
                sum += del;
                var del1 = c * (p - i * ff);
                sum1 += del1;
                if (Math.Abs(del) < Math.Abs(sum) * eps * 0.1) break;
            }
            var ex = Math.Exp(x);
            rkmu = sum * ex;
            rk1 = sum1 * xi2 * ex;
        }
        else
        {
            b = 2.0 * (1.0 + x);
            d = 1.0 / b;
            var hh = d;
            var delh = d;
            var q1 = 0.0;
            var q2 = 1.0;
            var a1 = 0.25 - xmu2;
            var q = a1;
            c = a1;
            var a = -a1;
            var s = 1.0 + q * delh;
            for (i = 2; i <= maxit; i++)
            {
                a -= 2 * (i - 1);
                c = -a * c / i;
                var qnew = (q1 - b * q2) / a;
                q1 = q2;
                q2 = qnew;
                q += c * qnew;
                b += 2.0;
                d = 1.0 / (b + a * d);
                delh = (b * d - 1.0) * delh;
                hh += delh;
                var dels = q * delh;
                s += dels;
                if (Math.Abs(dels / s) < eps) break;
            }
            hh = a1 * hh;
            rkmu = Math.Sqrt(Math.PI / (2.0 * x)) / s;
            rk1 = rkmu * (xmu + x + 0.5 - hh) * xi;
        }
        var rkmup = xmu * xi * rkmu - rk1;
        var rimu = xi / (f * rkmu - rkmup);
        var ri = rimu * ril1 / ril;
        var rip = rimu * rip1 / ril;
        for (i = 1; i <= nl; i++)
        {
            var rktemp = (xmu + i) * xi2 * rk1 + rkmu;
            rkmu = rk1;
            rk1 = rktemp;
            if (double.IsInfinity(rk1)) { rkmu = rk1; break; }
        }
        var rk = rkmu;
        var rkp = xnu * xi * rkmu - rk1;
        return (ri, rk, rip, rkp);
    }

    // ---------------------------------------------------------------- spherical

    /// <summary>Spherical Bessel function j_n(x) or its derivative (scipy.special.spherical_jn).</summary>
    /// <remarks>Upward recurrence from sin/cos when x &gt; n (where it is stable), √(π/2x) J_(n+½)(x) otherwise.</remarks>
    public static double SphericalJn(int n, double x, bool derivative = false)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Order must be non-negative.");
        if (derivative)
        {
            if (n == 0) return -SphericalJn(1, x);
            if (x == 0) return n == 1 ? 1.0 / 3.0 : 0.0;
            return SphericalJn(n - 1, x) - (n + 1) * SphericalJn(n, x) / x;
        }
        if (double.IsNaN(x)) return x;
        if (x < 0) return (n % 2 == 0 ? 1 : -1) * SphericalJn(n, -x);
        if (x == 0) return n == 0 ? 1.0 : 0.0;
        if (double.IsPositiveInfinity(x)) return 0.0;
        if (x > n)
        {
            var s = Math.Sin(x);
            var c = Math.Cos(x);
            var j0 = s / x;
            if (n == 0) return j0;
            var j1 = (j0 - c) / x;
            for (var k = 1; k < n; k++)
            {
                var j2 = (2 * k + 1) / x * j1 - j0;
                j0 = j1;
                j1 = j2;
            }
            return j1;
        }
        return Math.Sqrt(Math.PI / (2 * x)) * Jv(n + 0.5, x);
    }

    /// <summary>Spherical Bessel function y_n(x) or its derivative (scipy.special.spherical_yn); upward recurrence.</summary>
    public static double SphericalYn(int n, double x, bool derivative = false)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Order must be non-negative.");
        if (derivative)
        {
            if (n == 0) return -SphericalYn(1, x);
            return SphericalYn(n - 1, x) - (n + 1) * SphericalYn(n, x) / x;
        }
        if (double.IsNaN(x)) return x;
        if (x < 0) return (n % 2 == 0 ? -1 : 1) * SphericalYn(n, -x);
        if (x == 0) return double.NegativeInfinity;
        if (double.IsPositiveInfinity(x)) return 0.0;
        var s = Math.Sin(x);
        var c = Math.Cos(x);
        var y0 = -c / x;
        if (n == 0) return y0;
        var y1 = (y0 - s) / x;
        for (var k = 1; k < n; k++)
        {
            var y2 = (2 * k + 1) / x * y1 - y0;
            y0 = y1;
            y1 = y2;
            if (double.IsInfinity(y1)) break;
        }
        return y1;
    }

    /// <summary>Modified spherical Bessel function i_n(x) or its derivative (scipy.special.spherical_in).</summary>
    public static double SphericalIn(int n, double x, bool derivative = false)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Order must be non-negative.");
        if (derivative)
        {
            if (n == 0) return SphericalIn(1, x);
            if (x == 0) return n == 1 ? 1.0 / 3.0 : 0.0;
            return SphericalIn(n - 1, x) - (n + 1) * SphericalIn(n, x) / x;
        }
        if (double.IsNaN(x)) return x;
        if (x < 0) return (n % 2 == 0 ? 1 : -1) * SphericalIn(n, -x);
        if (x == 0) return n == 0 ? 1.0 : 0.0;
        if (double.IsPositiveInfinity(x)) return double.PositiveInfinity;
        if (n == 0) return x < 1e-4 ? 1 + x * x / 6 : Math.Sinh(x) / x;
        return Math.Sqrt(Math.PI / (2 * x)) * Iv(n + 0.5, x);
    }

    /// <summary>Modified spherical Bessel function k_n(x) = √(π/2x) K_(n+½)(x) or its derivative (scipy.special.spherical_kn).</summary>
    public static double SphericalKn(int n, double x, bool derivative = false)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Order must be non-negative.");
        if (derivative)
        {
            if (n == 0) return -SphericalKn(1, x);
            return -SphericalKn(n - 1, x) - (n + 1) * SphericalKn(n, x) / x;
        }
        if (double.IsNaN(x)) return x;
        if (x < 0) return double.NaN;
        if (x == 0) return double.PositiveInfinity;
        if (double.IsPositiveInfinity(x)) return 0.0;
        // Upward recurrence from k_0 = (π/2) e^-x / x is stable for k.
        var k0 = Math.PI / 2 * Math.Exp(-x) / x;
        if (n == 0) return k0;
        var k1 = k0 * (1 + 1 / x);
        for (var k = 1; k < n; k++)
        {
            var k2 = k0 + (2 * k + 1) / x * k1;
            k0 = k1;
            k1 = k2;
            if (double.IsInfinity(k1)) break;
        }
        return k1;
    }

    // ---------------------------------------------------------------- zeros

    /// <summary>The first <paramref name="nt"/> positive zeros of J_n (scipy.special.jn_zeros).</summary>
    /// <remarks>
    /// Zeros of J_n, J'_n, Y_n and Y'_n are never closer than ~2.9, so they are bracketed by a
    /// unit-step sign scan starting below the first zero (n + 0.4 n^(1/3)) and refined by Newton's
    /// method using the Bessel differential equation for the second derivative, with bisection as
    /// the safeguard. Each zero costs a handful of Bessel evaluations.
    /// </remarks>
    public static double[] JnZeros(int n, int nt) => BesselZeros(n, nt, false, false);

    /// <summary>The first nt positive zeros of J'_n (scipy.special.jnp_zeros); the zero at x = 0 is excluded.</summary>
    public static double[] JnpZeros(int n, int nt) => BesselZeros(n, nt, false, true);

    /// <summary>The first nt positive zeros of Y_n (scipy.special.yn_zeros).</summary>
    public static double[] YnZeros(int n, int nt) => BesselZeros(n, nt, true, false);

    /// <summary>The first nt positive zeros of Y'_n (scipy.special.ynp_zeros).</summary>
    public static double[] YnpZeros(int n, int nt) => BesselZeros(n, nt, true, true);

    private static double[] BesselZeros(int n, int nt, bool secondKind, bool derivative)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Order must be non-negative.");
        if (nt < 1) throw new ArgumentOutOfRangeException(nameof(nt), "nt must be positive.");
        // value, first and second derivative of the function whose zeros are wanted
        (double F, double Fp) Eval(double x)
        {
            double z, zp;
            if (secondKind) { z = Yn(n, x); zp = 0.5 * (Yn(n - 1, x) - Yn(n + 1, x)); }
            else { z = Jn(n, x); zp = 0.5 * (Jn(n - 1, x) - Jn(n + 1, x)); }
            if (!derivative) return (z, zp);
            // Bessel ODE: z'' = -z'/x - (1 - n²/x²) z
            var zpp = -zp / x - (1 - (double)n * n / (x * x)) * z;
            return (zp, zpp);
        }
        var result = new double[nt];
        var start = n == 0 ? (secondKind && !derivative ? 0.1 : 0.5) : n + 0.4 * Math.Cbrt(n);
        if (n == 0 && !secondKind && !derivative) start = 1.0;
        var a = start;
        var fa = Eval(a).F;
        for (var k = 0; k < nt; k++)
        {
            var b = a + 1.0;
            var fb = Eval(b).F;
            while (Math.Sign(fa) == Math.Sign(fb) && fb != 0)
            {
                a = b;
                fa = fb;
                b = a + 1.0;
                fb = Eval(b).F;
            }
            // root in (a, b]
            double lo = a, hi = b, flo = fa;
            var x = 0.5 * (lo + hi);
            for (var it = 0; it < 100; it++)
            {
                var (f, fp) = Eval(x);
                if (f == 0) break;
                if (Math.Sign(f) == Math.Sign(flo)) { lo = x; flo = f; } else hi = x;
                var xn = x - f / fp;
                if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
                if (Math.Abs(xn - x) <= 4e-16 * x) { x = xn; break; }
                x = xn;
            }
            result[k] = x;
            a = b;
            fa = fb;
        }
        return result;
    }
}
