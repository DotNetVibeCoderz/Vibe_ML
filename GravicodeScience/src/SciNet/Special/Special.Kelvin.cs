using System.Numerics;

namespace Gravicode.Science.SciNet;

public static partial class Special
{
    /// <summary>Kelvin function ber x (scipy.special.ber); even in x.</summary>
    /// <remarks>
    /// All eight Kelvin functions are Zhang &amp; Jin's <c>KLVNA</c> (as in scipy's specfun): power
    /// series below |x| = 10, an 18- (10 above 40) term asymptotic expansion beyond. Like scipy, the
    /// expansion limits accuracy around x ≈ 10–20 to roughly 1e-10 relative, and the growing
    /// functions overflow to ±∞ near x ≈ 1000.
    /// </remarks>
    public static double Ber(double x) => KelvinAll(Math.Abs(x)).Ber;

    /// <summary>Kelvin function bei x (scipy.special.bei); even in x.</summary>
    public static double Bei(double x) => KelvinAll(Math.Abs(x)).Bei;

    /// <summary>Kelvin function ker x (scipy.special.ker); NaN for x &lt; 0, +∞ at 0.</summary>
    public static double Ker(double x) => x < 0 ? double.NaN : KelvinAll(x).Ker;

    /// <summary>Kelvin function kei x (scipy.special.kei); NaN for x &lt; 0.</summary>
    public static double Kei(double x) => x < 0 ? double.NaN : KelvinAll(x).Kei;

    /// <summary>Derivative of ber x (scipy.special.berp); odd in x.</summary>
    public static double Berp(double x) => x < 0 ? -KelvinAll(-x).Berp : KelvinAll(x).Berp;

    /// <summary>Derivative of bei x (scipy.special.beip); odd in x.</summary>
    public static double Beip(double x) => x < 0 ? -KelvinAll(-x).Beip : KelvinAll(x).Beip;

    /// <summary>Derivative of ker x (scipy.special.kerp); NaN for x &lt; 0.</summary>
    public static double Kerp(double x) => x < 0 ? double.NaN : KelvinAll(x).Kerp;

    /// <summary>Derivative of kei x (scipy.special.keip); NaN for x &lt; 0.</summary>
    public static double Keip(double x) => x < 0 ? double.NaN : KelvinAll(x).Keip;

    /// <summary>
    /// All Kelvin functions at once as complex numbers (scipy.special.kelvin):
    /// Be = ber + i bei, Ke = ker + i kei, and their derivatives. For x &lt; 0 scipy's sign rules
    /// apply (Ke and Kep are NaN-free only for x ≥ 0).
    /// </summary>
    public static (Complex Be, Complex Ke, Complex Bep, Complex Kep) Kelvin(double x)
    {
        var flag = x < 0;
        var k = KelvinAll(Math.Abs(x));
        var be = new Complex(k.Ber, k.Bei);
        var ke = new Complex(k.Ker, k.Kei);
        var bep = new Complex(k.Berp, k.Beip);
        var kep = new Complex(k.Kerp, k.Keip);
        if (flag)
        {
            bep = -bep;
            ke = new Complex(double.NaN, double.NaN);
            kep = new Complex(double.NaN, double.NaN);
        }
        return (be, ke, bep, kep);
    }

    private static double ConvInf(double v) => v == 1e300 ? double.PositiveInfinity : v == -1e300 ? double.NegativeInfinity : v;

    private static (double Ber, double Bei, double Ker, double Kei, double Berp, double Beip, double Kerp, double Keip) KelvinAll(double x)
    {
        if (double.IsNaN(x)) return (double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN);
        var r = Klvna(x);
        return (ConvInf(r.Ber), ConvInf(r.Bei), ConvInf(r.Ker), ConvInf(r.Kei), ConvInf(r.Berp), ConvInf(r.Beip), ConvInf(r.Kerp), ConvInf(r.Keip));
    }

    private static (double Ber, double Bei, double Ker, double Kei, double Berp, double Beip, double Kerp, double Keip) Klvna(double x)
    {
        const double pi = 3.141592653589793, el = 0.5772156649015329, eps = 1.0e-15;
        double ber, bei, ger, gei, der, dei, her, hei;
        if (x == 0.0) return (1.0, 0.0, 1.0e300, -0.25 * pi, 0.0, 0.0, -1.0e300, 0.0);
        var x2 = 0.25 * x * x;
        var x4 = x2 * x2;
        if (Math.Abs(x) < 10.0)
        {
            double r, gs;
            ber = 1.0; r = 1.0;
            for (var m = 1; m <= 60; m++)
            {
                r = -0.25 * r / (m * m) / Math.Pow(2.0 * m - 1.0, 2) * x4;
                ber += r;
                if (Math.Abs(r) < Math.Abs(ber) * eps) break;
            }
            bei = x2; r = x2;
            for (var m = 1; m <= 60; m++)
            {
                r = -0.25 * r / (m * m) / Math.Pow(2.0 * m + 1.0, 2) * x4;
                bei += r;
                if (Math.Abs(r) < Math.Abs(bei) * eps) break;
            }
            var lg = Math.Log(x / 2.0) + el;
            ger = -lg * ber + 0.25 * pi * bei;
            r = 1.0; gs = 0.0;
            for (var m = 1; m <= 60; m++)
            {
                r = -0.25 * r / (m * m) / Math.Pow(2.0 * m - 1.0, 2) * x4;
                gs = gs + 1.0 / (2.0 * m - 1.0) + 1.0 / (2.0 * m);
                ger += r * gs;
                if (Math.Abs(r * gs) < Math.Abs(ger) * eps) break;
            }
            gei = x2 - lg * bei - 0.25 * pi * ber;
            r = x2; gs = 1.0;
            for (var m = 1; m <= 60; m++)
            {
                r = -0.25 * r / (m * m) / Math.Pow(2.0 * m + 1.0, 2) * x4;
                gs = gs + 1.0 / (2.0 * m) + 1.0 / (2.0 * m + 1.0);
                gei += r * gs;
                if (Math.Abs(r * gs) < Math.Abs(gei) * eps) break;
            }
            der = -0.25 * x * x2; r = der;
            for (var m = 1; m <= 60; m++)
            {
                r = -0.25 * r / m / (m + 1.0) / Math.Pow(2.0 * m + 1.0, 2) * x4;
                der += r;
                if (Math.Abs(r) < Math.Abs(der) * eps) break;
            }
            dei = 0.5 * x; r = dei;
            for (var m = 1; m <= 60; m++)
            {
                r = -0.25 * r / (m * m) / (2.0 * m - 1.0) / (2.0 * m + 1.0) * x4;
                dei += r;
                if (Math.Abs(r) < Math.Abs(dei) * eps) break;
            }
            r = -0.25 * x * x2; gs = 1.5;
            her = 1.5 * r - ber / x - lg * der + 0.25 * pi * dei;
            for (var m = 1; m <= 60; m++)
            {
                r = -0.25 * r / m / (m + 1.0) / Math.Pow(2.0 * m + 1.0, 2) * x4;
                gs = gs + 1.0 / (2.0 * m + 1.0) + 1.0 / (2 * m + 2.0);
                her += r * gs;
                if (Math.Abs(r * gs) < Math.Abs(her) * eps) break;
            }
            r = 0.5 * x; gs = 1.0;
            hei = 0.5 * x - bei / x - lg * dei - 0.25 * pi * der;
            for (var m = 1; m <= 60; m++)
            {
                r = -0.25 * r / (m * m) / (2 * m - 1.0) / (2 * m + 1.0) * x4;
                gs = gs + 1.0 / (2.0 * m) + 1.0 / (2 * m + 1.0);
                hei += r * gs;
                if (Math.Abs(r * gs) < Math.Abs(hei) * eps) break;
            }
            return (ber, bei, ger, gei, der, dei, her, hei);
        }

        double pp0 = 1.0, pn0 = 1.0, qp0 = 0.0, qn0 = 0.0, r0 = 1.0;
        var km = Math.Abs(x) >= 40.0 ? 10 : 18;
        var fac = 1.0;
        for (var k = 1; k <= km; k++)
        {
            fac = -fac;
            var xt = 0.25 * k * pi - Math.Truncate(0.125 * k) * 2.0 * pi;
            var cs = Math.Cos(xt); var ss = Math.Sin(xt);
            r0 = 0.125 * r0 * Math.Pow(2.0 * k - 1.0, 2) / k / x;
            var rc = r0 * cs; var rs = r0 * ss;
            pp0 += rc; pn0 += fac * rc; qp0 += rs; qn0 += fac * rs;
        }
        var xd = x / Math.Sqrt(2.0);
        var xe1 = Math.Exp(xd); var xe2 = Math.Exp(-xd);
        var xc1 = 1.0 / Math.Sqrt(2.0 * pi * x); var xc2 = Math.Sqrt(0.5 * pi / x);
        var cp0 = Math.Cos(xd + 0.125 * pi); var cn0 = Math.Cos(xd - 0.125 * pi);
        var sp0 = Math.Sin(xd + 0.125 * pi); var sn0 = Math.Sin(xd - 0.125 * pi);
        ger = xc2 * xe2 * (pn0 * cp0 - qn0 * sp0);
        gei = xc2 * xe2 * (-pn0 * sp0 - qn0 * cp0);
        ber = xc1 * xe1 * (pp0 * cn0 + qp0 * sn0) - gei / pi;
        bei = xc1 * xe1 * (pp0 * sn0 - qp0 * cn0) + ger / pi;
        double pp1 = 1.0, pn1 = 1.0, qp1 = 0.0, qn1 = 0.0, r1 = 1.0;
        fac = 1.0;
        for (var k = 1; k <= km; k++)
        {
            fac = -fac;
            var xt = 0.25 * k * pi - (int)(0.125 * k) * 2.0 * pi;
            var cs = Math.Cos(xt); var ss = Math.Sin(xt);
            r1 = 0.125 * r1 * (4.0 - Math.Pow(2.0 * k - 1.0, 2)) / (k * x);
            var rc = r1 * cs; var rs = r1 * ss;
            pp1 += fac * rc; pn1 += rc; qp1 += fac * rs; qn1 += rs;
        }
        her = xc2 * xe2 * (-pn1 * cn0 + qn1 * sn0);
        hei = xc2 * xe2 * (pn1 * sn0 + qn1 * cn0);
        der = xc1 * xe1 * (pp1 * cp0 + qp1 * sp0) - hei / pi;
        dei = xc1 * xe1 * (pp1 * sp0 - qp1 * cp0) + her / pi;
        return (ber, bei, ger, gei, der, dei, her, hei);
    }
}
