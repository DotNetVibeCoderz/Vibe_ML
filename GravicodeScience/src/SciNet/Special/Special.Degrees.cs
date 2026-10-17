namespace Gravicode.Science.SciNet;

public static partial class Special
{
    // Cephes sindg/cosdg/tandg/cotdg (as vendored by scipy's xsf): the reduction is done in degrees,
    // where multiples of 45° are exact, so sindg(180) is exactly 0 and tandg(45) exactly 1 — the
    // reason these exist next to Math.Sin(x·π/180).

    private const double Pi180 = 1.74532925199432957692e-2;
    private const double DegLossThreshold = 1.0e14;
    private static readonly double[] SinDgCof = [1.58962301572218447952E-10, -2.50507477628503540135E-8, 2.75573136213856773549E-6,
        -1.98412698295895384658E-4, 8.33333333332211858862E-3, -1.66666666666666307295E-1];
    private static readonly double[] CosDgCof = [1.13678171382044553091E-11, -2.08758833757683644217E-9, 2.75573155429816611547E-7,
        -2.48015872936186303776E-5, 1.38888888888806666760E-3, -4.16666666666666348141E-2, 4.99999999999999999798E-1];

    /// <summary>Sine of an angle in degrees (scipy.special.sindg); exact zeros at multiples of 180°.</summary>
    /// <remarks>Returns 0 for |x| &gt; 1e14, where no digit of the reduced angle is left (cephes "total loss").</remarks>
    public static double Sindg(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        var sign = 1;
        if (double.IsNegative(x)) { x = -x; sign = -1; }
        if (x > DegLossThreshold) return 0.0;
        var (j, z) = Octant(x, out var flip);
        if (flip) sign = -sign;
        var zz = z * z;
        var y = j is 1 or 2 ? 1.0 - zz * Polevl(zz, CosDgCof) : z + z * (zz * Polevl(zz, SinDgCof));
        return sign < 0 ? -y : y;
    }

    /// <summary>Cosine of an angle in degrees (scipy.special.cosdg); exact zeros at 90° + k·180°.</summary>
    public static double Cosdg(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (x < 0) x = -x;
        if (x > DegLossThreshold) return 0.0;
        var sign = 1;
        var (j, z) = Octant(x, out var flip);
        if (flip) sign = -sign;
        if (j > 1) sign = -sign;
        var zz = z * z;
        var y = j is 1 or 2 ? z + z * (zz * Polevl(zz, SinDgCof)) : 1.0 - zz * Polevl(zz, CosDgCof);
        return sign < 0 ? -y : y;
    }

    /// <summary>Cephes' octant reduction: octant j in 0..3 (after reflection), remainder in radians.</summary>
    private static (int J, double Z) Octant(double x, out bool flip)
    {
        var y = Math.Floor(x / 45.0);
        var z = Math.Floor(Math.ScaleB(y, -4));
        z = y - Math.ScaleB(z, 4);
        var j = (int)z;
        if ((j & 1) != 0) { j += 1; y += 1.0; }
        j &= 7;
        flip = false;
        if (j > 3) { flip = true; j -= 4; }
        return (j, (x - y * 45.0) * Pi180);
    }

    /// <summary>Tangent of an angle in degrees (scipy.special.tandg): exactly ±1 at 45°, +∞ at 90° + k·180°.</summary>
    public static double Tandg(double x) => TanCot(x, cot: false);

    /// <summary>Cotangent of an angle in degrees (scipy.special.cotdg): exactly ±1 at 45°, +∞ at multiples of 180°.</summary>
    public static double Cotdg(double x) => TanCot(x, cot: true);

    private static double TanCot(double xx, bool cot)
    {
        if (double.IsNaN(xx)) return double.NaN;
        double x;
        int sign;
        if (double.IsNegative(xx)) { x = -xx; sign = -1; } else { x = xx; sign = 1; }
        if (x > DegLossThreshold) return 0.0;
        var k = Math.Floor(x / 180.0);
        x -= 180.0 * k;
        if (cot)
        {
            if (x <= 90.0) x = 90.0 - x;
            else { x -= 90.0; sign = -sign; }
        }
        else if (x > 90.0) { x = 180.0 - x; sign = -sign; }
        // scipy 1.17.1 returns an unsigned +0 at the zeros and +∞ at the poles, whatever the side.
        if (x == 0.0) return 0.0;
        if (x == 45.0) return sign * 1.0;
        if (x == 90.0) return double.PositiveInfinity;
        return sign * Math.Tan(x * Pi180);
    }

    /// <summary>Degrees, minutes and seconds to radians (scipy.special.radian).</summary>
    public static double Radian(double d, double m, double s) => ((d * 60.0 + m) * 60.0 + s) * 4.848136811095359935899141023579479759563533023727e-6;

    /// <summary>
    /// Arithmetic–geometric mean of a and b (scipy.special.agm): NaN for opposite signs, the sign of
    /// the arguments when both are negative.
    /// </summary>
    /// <remarks>
    /// In the normal range it is the closed form π(a + b)/(4 K(m)) with m = 1 − 4ab/(a + b)² taken
    /// as the complementary parameter (no cancellation); extreme magnitudes use the iteration, which
    /// cannot overflow.
    /// </remarks>
    public static double Agm(double a, double b)
    {
        const double sqrtHalfMax = 9.480751908109176e+153, invSqrtHalfMax = 1.0547686614863e-154;
        if (double.IsNaN(a) || double.IsNaN(b)) return double.NaN;
        if ((a < 0 && b > 0) || (a > 0 && b < 0)) return double.NaN;
        if ((double.IsInfinity(a) || double.IsInfinity(b)) && (a == 0 || b == 0)) return double.NaN;
        if (a == 0 || b == 0) return 0.0;
        if (a == b) return a;
        var sgn = 1;
        if (a < 0) { sgn = -1; a = -a; b = -b; }
        if (invSqrtHalfMax < a && a < sqrtHalfMax && invSqrtHalfMax < b && b < sqrtHalfMax)
        {
            var e = 4 * a * b / ((a + b) * (a + b));
            return sgn * (Math.PI / 4) * (a + b) / EllipKm1(e);
        }
        var amean = 0.5 * a + 0.5 * b;
        for (var count = 20; count > 0 && amean != a && amean != b; count--)
        {
            var g = Math.Sqrt(a) * Math.Sqrt(b);
            a = amean; b = g;
            amean = 0.5 * a + 0.5 * b;
        }
        return sgn * amean;
    }
}
