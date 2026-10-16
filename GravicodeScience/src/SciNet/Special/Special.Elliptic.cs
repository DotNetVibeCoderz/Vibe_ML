namespace Gravicode.Science.SciNet;

/// <summary>Jacobian elliptic functions sn, cn, dn and the amplitude φ (scipy.special.ellipj).</summary>
public readonly record struct JacobiEllipticResult(double Sn, double Cn, double Dn, double Ph);

public static partial class Special
{
    // Carlson's duplication algorithms (B. C. Carlson, Numer. Algorithms 10, 1995) with the
    // seventh-order Taylor tails of DLMF 19.36.1/19.36.2, so the iteration stops as soon as the
    // arguments agree to ~1e-3 rather than to full precision.

    /// <summary>Carlson's symmetric integral R_F(x, y, z) (scipy.special.elliprf). At most one argument may be zero.</summary>
    public static double EllipRF(double x, double y, double z)
    {
        if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(z)) return double.NaN;
        if (x < 0 || y < 0 || z < 0) return double.NaN;
        if ((x == 0 ? 1 : 0) + (y == 0 ? 1 : 0) + (z == 0 ? 1 : 0) > 1) return double.PositiveInfinity;
        if (double.IsInfinity(x) || double.IsInfinity(y) || double.IsInfinity(z)) return 0.0;
        var a0 = (x + y + z) / 3;
        var q = Math.Pow(3e-16, -1.0 / 6) * Math.Max(Math.Abs(a0 - x), Math.Max(Math.Abs(a0 - y), Math.Abs(a0 - z)));
        var a = a0;
        double f = 1;
        double xm = x, ym = y, zm = z;
        for (var m = 0; m < 100 && q * f >= Math.Abs(a); m++)
        {
            var sx = Math.Sqrt(xm);
            var sy = Math.Sqrt(ym);
            var sz = Math.Sqrt(zm);
            var lam = sx * sy + sy * sz + sz * sx;
            a = 0.25 * (a + lam);
            xm = 0.25 * (xm + lam);
            ym = 0.25 * (ym + lam);
            zm = 0.25 * (zm + lam);
            f *= 0.25;
        }
        var X = (a0 - x) * f / a;
        var Y = (a0 - y) * f / a;
        var Z = -(X + Y);
        var e2 = X * Y - Z * Z;
        var e3 = X * Y * Z;
        var poly = 1 - e2 / 10 + e3 / 14 + e2 * e2 / 24 - 3 * e2 * e3 / 44
                   - 5 * e2 * e2 * e2 / 208 + 3 * e3 * e3 / 104 + e2 * e2 * e3 / 16;
        return poly / Math.Sqrt(a);
    }

    /// <summary>Carlson's degenerate integral R_C(x, y) (scipy.special.elliprc); Cauchy principal value for y &lt; 0.</summary>
    public static double EllipRC(double x, double y)
    {
        if (double.IsNaN(x) || double.IsNaN(y) || x < 0 || y == 0) return y == 0 ? double.PositiveInfinity : double.NaN;
        if (y < 0) return Math.Sqrt(x / (x - y)) * EllipRC(x - y, -y);
        if (double.IsPositiveInfinity(x) || double.IsPositiveInfinity(y)) return 0.0;
        var a0 = (x + 2 * y) / 3;
        var q = Math.Pow(3e-16, -1.0 / 8) * Math.Abs(a0 - x);
        var a = a0;
        double f = 1, xm = x, ym = y;
        for (var m = 0; m < 100 && q * f >= Math.Abs(a); m++)
        {
            var lam = 2 * Math.Sqrt(xm) * Math.Sqrt(ym) + ym;
            a = 0.25 * (a + lam);
            xm = 0.25 * (xm + lam);
            ym = 0.25 * (ym + lam);
            f *= 0.25;
        }
        var s = (y - a0) * f / a;
        var poly = 1 + s * s * (3.0 / 10 + s * (1.0 / 7 + s * (3.0 / 8 + s * (9.0 / 22 + s * (159.0 / 208 + s * (9.0 / 8))))));
        return poly / Math.Sqrt(a);
    }

    /// <summary>Carlson's symmetric integral of the second kind R_D(x, y, z) (scipy.special.elliprd).</summary>
    public static double EllipRD(double x, double y, double z)
    {
        if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(z) || x < 0 || y < 0 || z < 0) return double.NaN;
        if (z == 0 || (x == 0 && y == 0)) return double.PositiveInfinity;
        if (double.IsInfinity(x) || double.IsInfinity(y) || double.IsInfinity(z)) return 0.0;
        var a0 = (x + y + 3 * z) / 5;
        var q = Math.Pow(0.25e-16, -1.0 / 6) * Math.Max(Math.Abs(a0 - x), Math.Max(Math.Abs(a0 - y), Math.Abs(a0 - z)));
        var a = a0;
        double f = 1, sum = 0, xm = x, ym = y, zm = z;
        for (var m = 0; m < 100 && q * f >= Math.Abs(a); m++)
        {
            var sx = Math.Sqrt(xm);
            var sy = Math.Sqrt(ym);
            var sz = Math.Sqrt(zm);
            var lam = sx * sy + sy * sz + sz * sx;
            sum += f / (sz * (zm + lam));
            a = 0.25 * (a + lam);
            xm = 0.25 * (xm + lam);
            ym = 0.25 * (ym + lam);
            zm = 0.25 * (zm + lam);
            f *= 0.25;
        }
        var X = (a0 - x) * f / a;
        var Y = (a0 - y) * f / a;
        var Z = -(X + Y) / 3;
        var xy = X * Y;
        var z2 = Z * Z;
        var e2 = xy - 6 * z2;
        var e3 = (3 * xy - 8 * z2) * Z;
        var e4 = 3 * (xy - z2) * z2;
        var e5 = xy * z2 * Z;
        var poly = 1 - 3 * e2 / 14 + e3 / 6 + 9 * e2 * e2 / 88 - 3 * e4 / 22 - 9 * e2 * e3 / 52 + 3 * e5 / 26
                   - e2 * e2 * e2 / 16 + 3 * e3 * e3 / 40 + 3 * e2 * e4 / 20 + 45 * e2 * e2 * e3 / 272
                   - 9 * (e3 * e4 + e2 * e5) / 68;
        return f / (a * Math.Sqrt(a)) * poly + 3 * sum;
    }

    /// <summary>Carlson's symmetric integral of the third kind R_J(x, y, z, p) (scipy.special.elliprj); principal value for p &lt; 0.</summary>
    public static double EllipRJ(double x, double y, double z, double p)
    {
        if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(z) || double.IsNaN(p) || x < 0 || y < 0 || z < 0) return double.NaN;
        if (p == 0) return double.PositiveInfinity;
        if ((x == 0 ? 1 : 0) + (y == 0 ? 1 : 0) + (z == 0 ? 1 : 0) > 1) return double.PositiveInfinity;
        if (p < 0)
        {
            // Principal value (Numerical Recipes rj, after Carlson 1995): with x ≤ y ≤ z,
            // R_J(x,y,z,p) = a (b R_J(x,y,z,y+b) + 3 (R_C(xz/y, p(y+b)/y) - R_F(x,y,z))), a = 1/(y-p), b = a(z-y)(y-x).
            Span<double> v = [x, y, z];
            v.Sort();
            double xs = v[0], ys = v[1], zs = v[2];
            var aa = 1.0 / (ys - p);
            var bb = aa * (zs - ys) * (ys - xs);
            var pt = ys + bb;
            var rcx = EllipRC(xs * zs / ys, p * pt / ys);
            return aa * (bb * EllipRJ(xs, ys, zs, pt) + 3.0 * (rcx - EllipRF(xs, ys, zs)));
        }
        if (double.IsInfinity(x) || double.IsInfinity(y) || double.IsInfinity(z) || double.IsInfinity(p)) return 0.0;
        var a0 = (x + y + z + 2 * p) / 5;
        var delta = (p - x) * (p - y) * (p - z);
        var qq = Math.Pow(0.25e-16, -1.0 / 6) * Math.Max(Math.Max(Math.Abs(a0 - x), Math.Abs(a0 - y)), Math.Max(Math.Abs(a0 - z), Math.Abs(a0 - p)));
        var a = a0;
        double f = 1, f3 = 1, sum = 0, xm = x, ym = y, zm = z, pm = p;
        for (var m = 0; m < 100 && qq * f >= Math.Abs(a); m++)
        {
            var sx = Math.Sqrt(xm);
            var sy = Math.Sqrt(ym);
            var sz = Math.Sqrt(zm);
            var sp = Math.Sqrt(pm);
            var lam = sx * sy + sy * sz + sz * sx;
            var d = (sp + sx) * (sp + sy) * (sp + sz);
            var e = f3 * delta / (d * d);
            sum += f / d * EllipRC(1, 1 + e);
            a = 0.25 * (a + lam);
            xm = 0.25 * (xm + lam);
            ym = 0.25 * (ym + lam);
            zm = 0.25 * (zm + lam);
            pm = 0.25 * (pm + lam);
            f *= 0.25;
            f3 *= 1.0 / 64;
        }
        var X = (a0 - x) * f / a;
        var Y = (a0 - y) * f / a;
        var Z = (a0 - z) * f / a;
        var P = -(X + Y + Z) / 2;
        var e2 = X * Y + X * Z + Y * Z - 3 * P * P;
        var e3 = X * Y * Z + 2 * e2 * P + 4 * P * P * P;
        var e4 = (2 * X * Y * Z + e2 * P + 3 * P * P * P) * P;
        var e5 = X * Y * Z * P * P;
        var poly = 1 - 3 * e2 / 14 + e3 / 6 + 9 * e2 * e2 / 88 - 3 * e4 / 22 - 9 * e2 * e3 / 52 + 3 * e5 / 26
                   - e2 * e2 * e2 / 16 + 3 * e3 * e3 / 40 + 3 * e2 * e4 / 20 + 45 * e2 * e2 * e3 / 272
                   - 9 * (e3 * e4 + e2 * e5) / 68;
        return f / (a * Math.Sqrt(a)) * poly + 6 * sum;
    }

    /// <summary>Carlson's completely symmetric integral R_G(x, y, z) (scipy.special.elliprg).</summary>
    public static double EllipRG(double x, double y, double z)
    {
        if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(z) || x < 0 || y < 0 || z < 0) return double.NaN;
        // Put the largest argument last so z ≠ 0 unless all are zero.
        if (x > z) (x, z) = (z, x);
        if (y > z) (y, z) = (z, y);
        if (z == 0) return 0.0;
        if (x == 0 && y == 0) return 0.5 * Math.Sqrt(z);
        var s = x == 0 || y == 0 ? 0.0 : Math.Sqrt(x * y / z);
        return 0.5 * (z * EllipRF(x, y, z) - (x - z) * (y - z) * EllipRD(x, y, z) / 3 + s);
    }

    /// <summary>Complete elliptic integral of the first kind K(m), parameter m = k² (scipy.special.ellipk).</summary>
    /// <remarks>K(m) = R_F(0, 1 - m, 1). Near m = 1 the parameter itself loses digits; use <see cref="EllipKm1"/>.</remarks>
    public static double EllipK(double m)
    {
        if (double.IsNaN(m) || m > 1) return double.NaN;
        if (m == 1) return double.PositiveInfinity;
        if (double.IsNegativeInfinity(m)) return 0.0;
        return EllipRF(0, 1 - m, 1);
    }

    /// <summary>K(1 - p), accurate as p → 0 (scipy.special.ellipkm1).</summary>
    public static double EllipKm1(double p)
    {
        if (double.IsNaN(p) || p < 0) return double.NaN;
        if (p == 0) return double.PositiveInfinity;
        if (double.IsPositiveInfinity(p)) return 0.0;
        return EllipRF(0, p, 1);
    }

    /// <summary>Complete elliptic integral of the second kind E(m) (scipy.special.ellipe).</summary>
    public static double EllipE(double m)
    {
        if (double.IsNaN(m) || m > 1) return double.NaN;
        if (m == 1) return 1.0;
        if (double.IsNegativeInfinity(m)) return double.PositiveInfinity;
        var y = 1 - m;
        return EllipRF(0, y, 1) - m / 3 * EllipRD(0, y, 1);
    }

    /// <summary>Incomplete elliptic integral of the first kind F(φ | m) (scipy.special.ellipkinc).</summary>
    /// <remarks>sin φ R_F(cos²φ, 1 - m sin²φ, 1) after reducing φ by multiples of π (each adds 2K(m)).</remarks>
    public static double EllipKinc(double phi, double m)
    {
        if (double.IsNaN(phi) || double.IsNaN(m)) return double.NaN;
        if (double.IsInfinity(phi)) return phi;
        var n = Math.Round(phi / Math.PI);
        var r = phi - n * Math.PI;
        var s = Math.Sin(r);
        var c = Math.Cos(r);
        var d = 1 - m * s * s;
        if (d < 0) return double.NaN;
        double val;
        if (d == 0) val = Math.Sign(r) * double.PositiveInfinity;
        else val = s * EllipRF(c * c, d, 1);
        if (n == 0) return val;
        if (m >= 1) return n > 0 ? double.PositiveInfinity : double.NegativeInfinity;
        return 2 * n * EllipK(m) + val;
    }

    /// <summary>Incomplete elliptic integral of the second kind E(φ | m) (scipy.special.ellipeinc).</summary>
    public static double EllipEinc(double phi, double m)
    {
        if (double.IsNaN(phi) || double.IsNaN(m)) return double.NaN;
        if (double.IsInfinity(phi)) return phi;
        var n = Math.Round(phi / Math.PI);
        var r = phi - n * Math.PI;
        var s = Math.Sin(r);
        var c = Math.Cos(r);
        var d = 1 - m * s * s;
        if (d < 0) return double.NaN;
        var c2 = c * c;
        var val = s * EllipRF(c2, d, 1) - m / 3 * s * s * s * EllipRD(c2, d, 1);
        if (n == 0) return val;
        return 2 * n * EllipE(m) + val;
    }

    /// <summary>Jacobian elliptic functions sn, cn, dn and amplitude φ for parameter 0 ≤ m ≤ 1 (scipy.special.ellipj).</summary>
    /// <remarks>Cephes ellpj: the arithmetic-geometric mean and descending Landen transformation, with series near m = 0 and m = 1.</remarks>
    public static JacobiEllipticResult EllipJ(double u, double m)
    {
        if (double.IsNaN(u) || double.IsNaN(m) || m < 0 || m > 1) return new(double.NaN, double.NaN, double.NaN, double.NaN);
        if (m < 1e-9)
        {
            var t = Math.Sin(u);
            var b = Math.Cos(u);
            var ai = 0.25 * m * (u - t * b);
            return new(t - ai * b, b + ai * t, 1.0 - 0.5 * m * t * t, u - ai);
        }
        if (m >= 0.9999999999)
        {
            var ai = 0.25 * (1.0 - m);
            var b = Math.Cosh(u);
            var t = Math.Tanh(u);
            var phi = 1.0 / b;
            var twon = b * Math.Sinh(u);
            var sn = t + ai * (twon - u) / (b * b);
            var ph = 2.0 * Math.Atan(Math.Exp(u)) - Math.PI / 2 + ai * (twon - u) / b;
            ai *= t * phi;
            return new(sn, phi - ai * (twon - u), phi + ai * (twon + u), ph);
        }
        Span<double> a = stackalloc double[9];
        Span<double> c = stackalloc double[9];
        a[0] = 1.0;
        var bb = Math.Sqrt(1.0 - m);
        c[0] = Math.Sqrt(m);
        var twoN = 1.0;
        var i = 0;
        while (Math.Abs(c[i] / a[i]) > MachEp)
        {
            if (i > 7) break;
            var ai = a[i];
            i++;
            c[i] = (ai - bb) / 2.0;
            var tt = Math.Sqrt(ai * bb);
            a[i] = (ai + bb) / 2.0;
            bb = tt;
            twoN *= 2.0;
        }
        var phi2 = twoN * a[i] * u;
        double bprev;
        do
        {
            var t = c[i] * Math.Sin(phi2) / a[i];
            bprev = phi2;
            phi2 = (Math.Asin(t) + phi2) / 2.0;
        } while (--i > 0);
        var snr = Math.Sin(phi2);
        var cnr = Math.Cos(phi2);
        return new(snr, cnr, cnr / Math.Cos(phi2 - bprev), phi2);
    }
}
