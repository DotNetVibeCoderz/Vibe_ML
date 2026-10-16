namespace Gravicode.Science.SciNet;

/// <summary>Airy functions and their derivatives at one point (scipy.special.airy / airye).</summary>
public readonly record struct AiryResult(double Ai, double Aip, double Bi, double Bip);

public static partial class Special
{
    private const double AiryC1 = 0.355028053887817239260; // Ai(0)
    private const double AiryC2 = 0.258819403792806798405; // -Ai'(0)
    private const double Sqrt3 = 1.73205080756887729353;

    /// <summary>Airy functions Ai, Ai', Bi, Bi' of a real argument (scipy.special.airy).</summary>
    /// <remarks>
    /// Maclaurin series for |x| ≤ 1; beyond, the Bessel representations DLMF 9.6.1–9.6.9 in terms of
    /// K and I (x &gt; 0) or J and Y (x &lt; 0) of orders 1/3 and 2/3 at ζ = (2/3)|x|^(3/2), computed by
    /// <see cref="Kve"/>/<see cref="Ive"/>-style scaled kernels so Ai underflows and Bi overflows
    /// only when the true values do.
    /// </remarks>
    public static AiryResult Airy(double x)
    {
        if (double.IsNaN(x)) return new(x, x, x, x);
        if (double.IsPositiveInfinity(x)) return new(0, 0, double.PositiveInfinity, double.PositiveInfinity);
        if (double.IsNegativeInfinity(x)) return new(0, 0, 0, 0);
        if (Math.Abs(x) <= 1.0) return AiryMaclaurin(x);
        var ax = Math.Abs(x);
        var rx = Math.Sqrt(ax);
        var z = 2.0 * ax * rx / 3.0;
        if (x > 0)
        {
            var (ie1, ke1) = BesselIKScaled(1.0 / 3.0, z);
            var (ie2, ke2) = BesselIKScaled(2.0 / 3.0, z);
            var em = Math.Exp(-z);
            var ai = rx / Sqrt3 / Math.PI * ke1 * em;
            var aip = -x / (Math.PI * Sqrt3) * ke2 * em;
            double bi, bip;
            if (z < 700)
            {
                var ep = Math.Exp(z);
                bi = rx * (ke1 * em / Math.PI + 2.0 / Sqrt3 * ie1 * ep);
                bip = x * (ke2 * em / Math.PI + 2.0 / Sqrt3 * ie2 * ep);
            }
            else
            {
                bi = Math.Exp(z + Math.Log(rx * 2.0 / Sqrt3 * ie1));
                bip = Math.Exp(z + Math.Log(x * 2.0 / Sqrt3 * ie2));
            }
            return new(ai, aip, bi, bip);
        }
        var (j1, y1) = BesselJY(1.0 / 3.0, z);
        var (j2, y2) = BesselJY(2.0 / 3.0, z);
        return new(
            0.5 * rx * (j1 - y1 / Sqrt3),
            0.5 * ax * (y2 / Sqrt3 + j2),
            -0.5 * rx * (y1 + j1 / Sqrt3),
            0.5 * ax * (j2 / Sqrt3 - y2));
    }

    /// <summary>
    /// Exponentially scaled Airy functions (scipy.special.airye): Ai·e^ζ, Ai'·e^ζ, Bi·e^-|ζ|,
    /// Bi'·e^-|ζ| with ζ = (2/3) x^(3/2). For x &lt; 0, ζ is imaginary: the Ai pair is NaN (as in
    /// SciPy for real input) and the Bi pair is unscaled.
    /// </summary>
    public static AiryResult AiryE(double x)
    {
        if (double.IsNaN(x)) return new(x, x, x, x);
        if (x < 0)
        {
            var r = Airy(x);
            return new(double.NaN, double.NaN, r.Bi, r.Bip);
        }
        if (x <= 1.0)
        {
            var r = AiryMaclaurin(x);
            var zz = 2.0 / 3.0 * x * Math.Sqrt(x);
            var ep = Math.Exp(zz);
            return new(r.Ai * ep, r.Aip * ep, r.Bi / ep, r.Bip / ep);
        }
        var rx = Math.Sqrt(x);
        var z = 2.0 / 3.0 * x * rx;
        var (ie1, ke1) = BesselIKScaled(1.0 / 3.0, z);
        var (ie2, ke2) = BesselIKScaled(2.0 / 3.0, z);
        var e2 = Math.Exp(-2 * z);
        return new(
            rx / Sqrt3 / Math.PI * ke1,
            -x / (Math.PI * Sqrt3) * ke2,
            rx * (ke1 * e2 / Math.PI + 2.0 / Sqrt3 * ie1),
            x * (ke2 * e2 / Math.PI + 2.0 / Sqrt3 * ie2));
    }

    private static AiryResult AiryMaclaurin(double x)
    {
        var x3 = x * x * x;
        double f = 1, g = x, fp = 0, gp = 1;
        double tf = 1, tg = x, tfp = 0.5 * x * x, tgp = 1;
        fp = tfp;
        for (var k = 0; k < 60; k++)
        {
            tf *= x3 / ((3 * k + 2.0) * (3 * k + 3));
            tg *= x3 / ((3 * k + 3.0) * (3 * k + 4));
            if (k > 0) tfp *= x3 / ((3.0 * k) * (3 * k + 2));
            tgp *= x3 / ((3 * k + 1.0) * (3 * k + 3));
            f += tf;
            g += tg;
            if (k > 0) fp += tfp;
            gp += tgp;
            if (Math.Abs(tf) + Math.Abs(tg) + Math.Abs(tfp) + Math.Abs(tgp) < 1e-18) break;
        }
        return new(
            AiryC1 * f - AiryC2 * g,
            AiryC1 * fp - AiryC2 * gp,
            Sqrt3 * (AiryC1 * f + AiryC2 * g),
            Sqrt3 * (AiryC1 * fp + AiryC2 * gp));
    }
}
