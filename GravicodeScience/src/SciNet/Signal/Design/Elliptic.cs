namespace Gravicode.Science.SciNet;

/// <summary>
/// The elliptic-function machinery the Cauer design needs, embedded so Signal does not depend on
/// SciNet's Special module: complete integrals by the arithmetic-geometric mean, Jacobi functions
/// by Cephes' descending-Landen <c>ellpj</c>, and scipy's own degree equation and inverse <c>sc</c>.
/// </summary>
internal static class SignalElliptic
{
    private const double MachEp = 1.11022302462515654042e-16;

    /// <summary>Complete elliptic integral K(m), parameter convention (m = k²).</summary>
    public static double K(double m)
    {
        if (m < 0 || m > 1) return double.NaN;
        if (m == 1) return double.PositiveInfinity;
        return Math.PI / (2 * Agm(1.0, Math.Sqrt(1.0 - m)));
    }

    /// <summary>K(1 − p), accurate for tiny p where forming 1 − p would lose it (scipy's <c>ellipkm1</c>).</summary>
    public static double Km1(double p)
    {
        if (p < 0 || p > 1) return double.NaN;
        if (p == 0) return double.PositiveInfinity;
        return Math.PI / (2 * Agm(1.0, Math.Sqrt(p)));
    }

    private static double Agm(double a, double b)
    {
        for (var i = 0; i < 64; i++)
        {
            var an = 0.5 * (a + b);
            var bn = Math.Sqrt(a * b);
            if (Math.Abs(an - bn) <= 1e-17 * an) return 0.5 * (an + bn);
            a = an; b = bn;
        }
        return a;
    }

    /// <summary>Jacobi elliptic functions sn, cn, dn and amplitude φ (Cephes <c>ellpj</c>).</summary>
    public static (double Sn, double Cn, double Dn, double Ph) Ellipj(double u, double m)
    {
        if (m < 0 || m > 1 || double.IsNaN(m)) return (double.NaN, double.NaN, double.NaN, double.NaN);
        double t, b, ai, phi, twon;
        if (m < 1.0e-9)
        {
            t = Math.Sin(u);
            b = Math.Cos(u);
            ai = 0.25 * m * (u - t * b);
            return (t - ai * b, b + ai * t, 1.0 - 0.5 * m * t * t, u - ai);
        }
        if (m >= 0.9999999999)
        {
            ai = 0.25 * (1.0 - m);
            b = Math.Cosh(u);
            t = Math.Tanh(u);
            phi = 1.0 / b;
            twon = b * Math.Sinh(u);
            var sn = t + ai * (twon - u) / (b * b);
            var ph = 2.0 * Math.Atan(Math.Exp(u)) - Math.PI / 2 + ai * (twon - u) / b;
            ai *= t * phi;
            return (sn, phi - ai * (twon - u), phi + ai * (twon + u), ph);
        }
        Span<double> a = stackalloc double[9];
        Span<double> c = stackalloc double[9];
        a[0] = 1.0;
        b = Math.Sqrt(1.0 - m);
        c[0] = Math.Sqrt(m);
        twon = 1.0;
        var i = 0;
        while (Math.Abs(c[i] / a[i]) > MachEp)
        {
            if (i > 7) break;
            ai = a[i];
            ++i;
            c[i] = (ai - b) / 2.0;
            t = Math.Sqrt(ai * b);
            a[i] = (ai + b) / 2.0;
            b = t;
            twon *= 2.0;
        }
        phi = twon * a[i] * u;
        do
        {
            t = c[i] * Math.Sin(phi) / a[i];
            b = phi;
            phi = (Math.Asin(t) + phi) / 2.0;
        } while (--i != 0);
        var snv = Math.Sin(phi);
        t = Math.Cos(phi);
        return (snv, t, t / Math.Cos(phi - b), phi);
    }

    /// <summary>Solves the degree equation for the selectivity m given order n and m1 (scipy's <c>_ellipdeg</c>).</summary>
    public static double EllipDeg(int n, double m1)
    {
        const int mmax = 7;
        var k1 = K(m1);
        var k1p = Km1(m1);
        var q1 = Math.Exp(-Math.PI * k1p / k1);
        var q = Math.Pow(q1, 1.0 / n);
        var num = 0.0;
        for (var i = 0; i <= mmax; i++) num += Math.Pow(q, i * (i + 1));
        var den = 0.0;
        for (var i = 1; i <= mmax + 1; i++) den += Math.Pow(q, i * i);
        den = 1 + 2 * den;
        return 16 * q * Math.Pow(num / den, 4);
    }

    /// <summary>
    /// Real inverse of the Jacobi <c>sc</c> function, evaluated as scipy does: <c>arcsn(i·w)</c> by
    /// descending Landen transformations, which keeps the argument purely imaginary throughout, so
    /// it is done here in real arithmetic with <c>arcsin(iy) = i·asinh(y)</c>.
    /// </summary>
    public static double ArcJacSc1(double w, double m)
    {
        static double Complement(double kx) => Math.Sqrt((1 - kx) * (1 + kx));
        var k = Math.Sqrt(m);
        if (k > 1) return double.NaN;
        if (k == 1) return Math.Atan(w); // atanh(i w) = i atan(w)
        var ks = new List<double> { k };
        var niter = 0;
        while (ks[^1] != 0)
        {
            var kk = ks[^1];
            var kp = Complement(kk);
            ks.Add((1 - kp) / (1 + kp));
            if (++niter > 10) throw new InvalidOperationException("Landen transformation not converging.");
        }
        var bigK = Math.PI / 2;
        for (var i = 1; i < ks.Count; i++) bigK *= 1 + ks[i];
        var v = w;
        for (var i = 0; i < ks.Count - 1; i++)
        {
            var kn = ks[i];
            var knext = ks[i + 1];
            // complement(kn * i v) = sqrt(1 + kn^2 v^2), real.
            v = 2 * v / ((1 + knext) * (1 + Math.Sqrt(1 + kn * v * (kn * v))));
        }
        return bigK * (2 / Math.PI) * Math.Asinh(v);
    }
}
