namespace Gravicode.Science.SciNet;

public static partial class Special
{
    /// <summary>
    /// Confluent hypergeometric function of the second kind U(a, b, x) (scipy.special.hyperu),
    /// the solution of Kummer's equation that decays like x^(−a) as x → ∞. x ≥ 0.
    /// </summary>
    /// <remarks>
    /// A port of scipy's route through Zhang &amp; Jin's <c>CHGU</c>: the small-x series (DLMF 13.2.42),
    /// the large-x asymptotic series (13.7.3), the logarithmic series for integer b, and 60-point
    /// Gauss–Legendre quadrature of the integral representation (13.4.4), choosing by the estimated
    /// number of correct digits; NaN when fewer than about 6 digits are expected (scipy's
    /// "no result"). At x = 0 it is Poch(1 − b + a, −a) for b ≤ 1 and +∞ otherwise. Gamma values use
    /// SciNet's <see cref="Gamma(double)"/> instead of specfun's 26-term series, so results may differ from
    /// scipy's in the last digits.
    /// </remarks>
    public static double Hyperu(double a, double b, double x)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(x)) return double.NaN;
        if (x < 0.0) return double.NaN;
        if (x == 0.0) return b > 1.0 ? double.PositiveInfinity : Poch(1.0 - b + a, -a);
        if (b == 1.0 && x < 1.0 && -0.25 < a && a < 0.3)
            return (x + 1.0 + 2.0 * a) * Hypu(a + 1.0, 1.0, x) - Math.Pow(a + 1.0, 2) * Hypu(a + 2.0, 1.0, x);
        return Hypu(a, b, x);
    }

    private static double Hypu(double a, double b, double x)
    {
        var hu = Chgu(x, a, b, out var isfer);
        if (hu == 1e300) hu = double.PositiveInfinity;
        if (isfer != 0) hu = double.NaN;
        return hu;
    }

    private static bool IsInt(double v) => v == Math.Truncate(v) && Math.Abs(v) < int.MaxValue;

    /// <summary>specfun's GAMMA2 convention: 1e300 at the poles.</summary>
    private static double Gamma2(double x) => x == Math.Truncate(x) && x <= 0 ? 1e300 : Gamma(x);

    private static double Chgu(double x, double a, double b, out int isfer)
    {
        var aa = a - b + 1.0;
        isfer = 0;
        var il1 = IsInt(a) && a <= 0.0;
        var il2 = IsInt(aa) && aa <= 0.0;
        var il3 = Math.Abs(a * (a - b + 1.0)) / x <= 2.0;
        var bl1 = x <= 5.0 || (x <= 10.0 && a <= 2.0);
        var bl2 = x > 5.0 && x <= 12.5 && a >= 1.0 && b >= a + 4.0;
        var bl3 = x > 12.5 && a >= 5.0 && b >= a + 5.0;
        var bn = IsInt(b) && b != 0.0;
        int id = -100, id1 = 0;
        double hu = 0.0, hu1 = 0.0;
        if (!IsInt(b))
        {
            hu = Chgus(x, a, b, out id1);
            if (id1 >= 9) return hu;
            hu1 = hu;
        }
        if (il1 || il2 || il3)
        {
            hu = Chgul(x, a, b, out id);
            if (id >= 9) return hu;
            if (id1 > id) { id = id1; hu = hu1; }
        }
        if (a >= 1.0)
        {
            hu = bn && (bl1 || bl2 || bl3) ? Chgubi(x, a, b, out id) : Chguit(x, a, b, out id);
        }
        else
        {
            if (b <= a)
            {
                var b00 = b;
                a -= b - 1.0;
                b = 2.0 - b;
                hu = Chguit(x, a, b, out id);
                hu *= Math.Pow(x, 1.0 - b00);
            }
            else if (bn) // specfun tests "bn && ~il1", and ~ of a C logical is always true.
                hu = Chgubi(x, a, b, out id);
        }
        if (id < 6) isfer = 6;
        return hu;
    }

    private static double Chgubi(double x, double a, double b, out int id)
    {
        const double el = 0.5772156649015329;
        var n = (int)Math.Abs(b - 1);
        double rn1 = 1.0, rn = 1.0;
        for (var j = 1; j <= n; j++)
        {
            rn *= j;
            if (j == n - 1) rn1 = rn;
        }
        var ps = Psi(a);
        var ga = Gamma2(a);
        double a0, a1, a2, ga1, ua, ub;
        if (b > 0.0)
        {
            a0 = a; a1 = a - n; a2 = a1;
            ga1 = Gamma2(a1);
            ua = Math.Pow(-1, n - 1) / (rn * ga1);
            ub = rn1 / ga * Math.Pow(x, -n);
        }
        else
        {
            a0 = a + n; a1 = a0; a2 = a;
            ga1 = Gamma2(a1);
            ua = Math.Pow(-1, n - 1) / (rn * ga) * Math.Pow(x, n);
            ub = rn1 / ga1;
        }
        double hm1 = 1.0, r = 1.0, hmax = 0.0, hmin = 1e300, h0 = 0.0;
        for (var k = 1; k <= 150; k++)
        {
            r = r * (a0 + k - 1) * x / ((n + k) * k);
            hm1 += r;
            var hu1 = Math.Abs(hm1);
            if (hu1 > hmax) hmax = hu1;
            if (hu1 < hmin) hmin = hu1;
            if (Math.Abs(hm1 - h0) < Math.Abs(hm1) * 1.0e-15) break;
            h0 = hm1;
        }
        var da1 = Math.Log10(hmax);
        var da2 = hmin != 0 ? Math.Log10(hmin) : 0;
        id = 15 - (int)Math.Abs(da1 - da2);
        hm1 *= Math.Log(x);
        double s0 = 0;
        for (var m = 1; m <= n; m++)
        {
            if (b >= 0) s0 -= 1.0 / m;
            if (b < 0) s0 += (1.0 - a) / (m * (a + m - 1));
        }
        var hm2 = ps + 2 * el + s0;
        r = 1; hmax = 0; hmin = 1.0e300;
        for (var k = 1; k <= 150; k++)
        {
            double s1 = 0, s2 = 0;
            if (b > 0)
            {
                for (var m = 1; m <= k; m++) s1 -= (m + 2 * a - 2) / (m * (m + a - 1));
                for (var m = 1; m <= n; m++) s2 += 1.0 / (k + m);
            }
            else
            {
                for (var m = 1; m <= k + n; m++) s1 += (1.0 - a) / (m * (m + a - 1));
                for (var m = 1; m <= k; m++) s2 += 1.0 / m;
            }
            var hw = 2 * el + ps + s1 - s2;
            r = r * (a0 + k - 1) * x / ((n + k) * k);
            hm2 += r * hw;
            var hu2 = Math.Abs(hm2);
            if (hu2 > hmax) hmax = hu2;
            if (hu2 < hmin) hmin = hu2;
            if (Math.Abs((hm2 - h0) / hm2) < 1.0e-15) break;
            h0 = hm2;
        }
        var db1 = Math.Log10(hmax);
        var db2 = hmin != 0.0 ? Math.Log10(hmin) : 0.0;
        var id1 = 15 - (int)Math.Abs(db1 - db2);
        if (id1 < id) id = id1;
        var hm3 = n == 0 ? 0.0 : 1.0;
        r = 1.0;
        for (var k = 1; k < n; k++)
        {
            r = r * (a2 + k - 1.0) / ((k - n) * k) * x;
            hm3 += r;
        }
        var sa = ua * (hm1 + hm2);
        var sb = ub * hm3;
        var hu = sa + sb;
        var id2 = 0;
        if (sa != 0.0) id1 = (int)Math.Log10(Math.Abs(sa));
        if (hu != 0.0) id2 = (int)Math.Log10(Math.Abs(hu));
        if (sa * sb < 0.0) id -= Math.Abs(id1 - id2);
        return hu;
    }

    private static readonly double[] ChguitT =
    [
        0.259597723012478e-01, 0.778093339495366e-01, 0.129449135396945e+00, 0.180739964873425e+00, 0.231543551376029e+00,
        0.281722937423262e+00, 0.331142848268448e+00, 0.379670056576798e+00, 0.427173741583078e+00, 0.473525841761707e+00,
        0.518601400058570e+00, 0.562278900753945e+00, 0.604440597048510e+00, 0.644972828489477e+00, 0.683766327381356e+00,
        0.720716513355730e+00, 0.755723775306586e+00, 0.788693739932264e+00, 0.819537526162146e+00, 0.848171984785930e+00,
        0.874519922646898e+00, 0.898510310810046e+00, 0.920078476177628e+00, 0.939166276116423e+00, 0.955722255839996e+00,
        0.969701788765053e+00, 0.981067201752598e+00, 0.989787895222222e+00, 0.995840525118838e+00, 0.999210123227436e+00,
    ];

    private static readonly double[] ChguitW =
    [
        0.519078776312206e-01, 0.517679431749102e-01, 0.514884515009810e-01, 0.510701560698557e-01, 0.505141845325094e-01,
        0.498220356905502e-01, 0.489955754557568e-01, 0.480370318199712e-01, 0.469489888489122e-01, 0.457343797161145e-01,
        0.443964787957872e-01, 0.429388928359356e-01, 0.413655512355848e-01, 0.396806954523808e-01, 0.378888675692434e-01,
        0.359948980510845e-01, 0.340038927249464e-01, 0.319212190192963e-01, 0.297524915007890e-01, 0.275035567499248e-01,
        0.251804776215213e-01, 0.227895169439978e-01, 0.203371207294572e-01, 0.178299010142074e-01, 0.152746185967848e-01,
        0.126781664768159e-01, 0.100475571822880e-01, 0.738993116334531e-02, 0.471272992695363e-02, 0.202681196887362e-02,
    ];

    private static double Chguit(double x, double a, double b, out int id)
    {
        id = 9;
        var a1 = a - 1.0;
        var b1 = b - a - 1.0;
        var c = 12.0 / x;
        double hu0 = 0.0, hu1 = 0.0, hu2 = 0.0;
        for (var m = 10; m <= 100; m += 5)
        {
            hu1 = 0.0;
            var g = 0.5 * c / m;
            var d = g;
            for (var j = 1; j < m + 1; j++)
            {
                double s = 0.0;
                for (var k = 0; k < 30; k++)
                {
                    var t1 = d + g * ChguitT[k];
                    var t2 = d - g * ChguitT[k];
                    var f1 = Math.Exp(-x * t1) * Math.Pow(t1, a1) * Math.Pow(1.0 + t1, b1);
                    var f2 = Math.Exp(-x * t2) * Math.Pow(t2, a1) * Math.Pow(1.0 + t2, b1);
                    s += ChguitW[k] * (f1 + f2);
                }
                hu1 += s * g;
                d += 2.0 * g;
            }
            if (Math.Abs(1.0 - hu0 / hu1) < 1.0e-9) break;
            hu0 = hu1;
        }
        var ga = Gamma2(a);
        hu1 /= ga;
        for (var m = 2; m <= 10; m += 2)
        {
            hu2 = 0.0;
            var g = 0.5 / m;
            var d = g;
            for (var j = 1; j <= m; j++)
            {
                double s = 0.0;
                for (var k = 0; k < 30; k++)
                {
                    var t1 = d + g * ChguitT[k];
                    var t2 = d - g * ChguitT[k];
                    var t3 = c / (1.0 - t1);
                    var t4 = c / (1.0 - t2);
                    var f1 = t3 * t3 / c * Math.Exp(-x * t3) * Math.Pow(t3, a1) * Math.Pow(1.0 + t3, b1);
                    var f2 = t4 * t4 / c * Math.Exp(-x * t4) * Math.Pow(t4, a1) * Math.Pow(1.0 + t4, b1);
                    s += ChguitW[k] * (f1 + f2);
                }
                hu2 += s * g;
                d += 2.0 * g;
            }
            if (Math.Abs(1.0 - hu0 / hu2) < 1.0e-9) break;
            hu0 = hu2;
        }
        hu2 /= ga;
        return hu1 + hu2;
    }

    private static double Chgul(double x, double a, double b, out int id)
    {
        id = -100;
        var aa = a - b + 1.0;
        var il1 = IsInt(a) && a <= 0.0;
        var il2 = IsInt(aa) && aa <= 0.0;
        var nm = 0;
        if (il1) nm = (int)Math.Abs(a);
        if (il2) nm = (int)Math.Abs(aa);
        double hu, r;
        if (il1 || il2)
        {
            hu = 1.0; r = 1.0;
            for (var k = 1; k <= nm; k++)
            {
                r = -r * (a + k - 1.0) * (a - b + k) / (k * x);
                hu += r;
            }
            hu *= Math.Pow(x, -a);
            id = 10;
            return hu;
        }
        hu = 1.0; r = 1.0;
        double r0 = 0.0, ra = 0.0;
        for (var k = 1; k <= 25; k++)
        {
            r = -r * (a + k - 1.0) * (a - b + k) / (k * x);
            ra = Math.Abs(r);
            if ((k > 5 && ra >= r0) || ra < 1e-15) break;
            r0 = ra;
            hu += r;
        }
        id = (int)Math.Abs(Math.Log10(ra));
        return hu * Math.Pow(x, -a);
    }

    private static double Chgus(double x, double a, double b, out int id)
    {
        const double pi = 3.141592653589793;
        var ga = Gamma2(a);
        var gb = Gamma2(b);
        var gab = Gamma2(1.0 + a - b);
        var gb2 = Gamma2(2.0 - b);
        var hu0 = pi / Math.Sin(pi * b);
        var r1 = hu0 / (gab * gb);
        var r2 = hu0 * Math.Pow(x, 1.0 - b) / (ga * gb2);
        var hu = r1 - r2;
        double hmax = 0.0, hmin = 1e300, h0 = 0.0;
        for (var j = 1; j < 151; j++)
        {
            r1 = r1 * (a + j - 1.0) / (j * (b + j - 1.0)) * x;
            r2 = r2 * (a - b + j) / (j * (1.0 - b + j)) * x;
            hu += r1 - r2;
            var hua = Math.Abs(hu);
            if (hua > hmax) hmax = hua;
            if (hua < hmin) hmin = hua;
            if (Math.Abs(hu - h0) < Math.Abs(hu) * 1e-15) break;
            h0 = hu;
        }
        var d1 = Math.Log10(hmax);
        var d2 = hmin != 0.0 ? Math.Log10(hmin) : 0.0;
        id = (int)(15 - Math.Abs(d1 - d2));
        return hu;
    }
}
