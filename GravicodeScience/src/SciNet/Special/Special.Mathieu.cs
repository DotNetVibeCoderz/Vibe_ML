namespace Gravicode.Science.SciNet;

public static partial class Special
{
    // Mathieu functions (scipy.special.mathieu_a, mathieu_b, mathieu_cem, mathieu_sem,
    // mathieu_modcem1/2, mathieu_modsem1/2).
    //
    // Not a port of scipy's Zhang & Jin code (cva2/mtu0/mtu12). The characteristic values are
    // eigenvalues of the symmetric tridiagonal recurrence matrices of DLMF 28.4, found by Sturm
    // bisection to the last bit (the matrices are graded, so even a_0(q) ≈ −q²/2 at q = 1e-3 keeps
    // its relative accuracy); the Fourier coefficients are the eigenvector, by inverse iteration;
    // the radial functions are the Bessel-product series of DLMF 28.24.6-9 with the pivot index at
    // the largest coefficient. Conventions are scipy's: x in degrees for ce/se and derivatives with
    // respect to x in radians; ∫₀^{2π} ce² = π; ce_m(π/2) has the sign it has at q = 0
    // (equivalently ce_m(0) > 0 and se_m'(0) > 0); negative q through DLMF 28.2.34-37.

    private enum MathieuKind { CeEven, CeOdd, SeOdd, SeEven }

    private sealed record MathieuCoefficients(double CharValue, double[] C, int Offset, int N);

    /// <summary>Characteristic value a_m(q) of the even Mathieu function ce_m (scipy's <c>mathieu_a</c>); m a non-negative integer.</summary>
    public static double MathieuA(double m, double q)
    {
        if (!IsOrder(m, 0) || double.IsNaN(q) || double.IsInfinity(q)) return double.NaN;
        var mi = (int)m;
        if (q < 0 && mi % 2 == 1) return MathieuCoeffs(false, mi, -q).CharValue;
        return MathieuCoeffs(true, mi, Math.Abs(q)).CharValue;
    }

    /// <summary>Characteristic value b_m(q) of the odd Mathieu function se_m (scipy's <c>mathieu_b</c>); m a positive integer.</summary>
    public static double MathieuB(double m, double q)
    {
        if (!IsOrder(m, 1) || double.IsNaN(q) || double.IsInfinity(q)) return double.NaN;
        var mi = (int)m;
        if (q < 0 && mi % 2 == 1) return MathieuCoeffs(true, mi, -q).CharValue;
        return MathieuCoeffs(false, mi, Math.Abs(q)).CharValue;
    }

    /// <summary>
    /// The even Mathieu function ce_m(x, q) and its derivative with respect to x in radians
    /// (scipy's <c>mathieu_cem</c>; <paramref name="x"/> in degrees).
    /// </summary>
    public static (double Value, double Derivative) MathieuCem(double m, double q, double x)
    {
        if (!IsOrder(m, 0) || double.IsNaN(q) || double.IsNaN(x) || double.IsInfinity(x) || double.IsInfinity(q)) return (double.NaN, double.NaN);
        var mi = (int)m;
        if (q < 0)
        {
            var n = mi / 2;
            var sgn = n % 2 == 0 ? 1.0 : -1.0;
            var (v, d) = mi % 2 == 0 ? Angular(true, mi, -q, 90 - x) : Angular(false, mi, -q, 90 - x);
            return (sgn * v, -sgn * d);
        }
        return Angular(true, mi, q, x);
    }

    /// <summary>
    /// The odd Mathieu function se_m(x, q) and its derivative with respect to x in radians
    /// (scipy's <c>mathieu_sem</c>; <paramref name="x"/> in degrees; m ≥ 1).
    /// </summary>
    public static (double Value, double Derivative) MathieuSem(double m, double q, double x)
    {
        if (!IsOrder(m, 1) || double.IsNaN(q) || double.IsNaN(x) || double.IsInfinity(x) || double.IsInfinity(q)) return (double.NaN, double.NaN);
        var mi = (int)m;
        if (q < 0)
        {
            var n = mi % 2 == 1 ? (mi - 1) / 2 : (mi - 2) / 2;
            var sgn = n % 2 == 0 ? 1.0 : -1.0;
            var (v, d) = mi % 2 == 1 ? Angular(true, mi, -q, 90 - x) : Angular(false, mi, -q, 90 - x);
            return (sgn * v, -sgn * d);
        }
        return Angular(false, mi, q, x);
    }

    /// <summary>The even modified Mathieu function of the first kind Mc⁽¹⁾_m(x, q) and its x-derivative (scipy's <c>mathieu_modcem1</c>; q ≥ 0).</summary>
    public static (double Value, double Derivative) MathieuModCem1(double m, double q, double x) => Radial(true, 1, m, q, x);

    /// <summary>The even modified Mathieu function of the second kind Mc⁽²⁾_m(x, q) and its x-derivative (scipy's <c>mathieu_modcem2</c>; q &gt; 0).</summary>
    public static (double Value, double Derivative) MathieuModCem2(double m, double q, double x) => Radial(true, 2, m, q, x);

    /// <summary>The odd modified Mathieu function of the first kind Ms⁽¹⁾_m(x, q) and its x-derivative (scipy's <c>mathieu_modsem1</c>; m ≥ 1, q ≥ 0).</summary>
    public static (double Value, double Derivative) MathieuModSem1(double m, double q, double x) => Radial(false, 1, m, q, x);

    /// <summary>The odd modified Mathieu function of the second kind Ms⁽²⁾_m(x, q) and its x-derivative (scipy's <c>mathieu_modsem2</c>; m ≥ 1, q &gt; 0).</summary>
    public static (double Value, double Derivative) MathieuModSem2(double m, double q, double x) => Radial(false, 2, m, q, x);

    private static bool IsOrder(double m, int min) => m >= min && m == Math.Floor(m) && m < 100000;

    private static (double Value, double Derivative) Angular(bool even, int m, double q, double xdeg)
    {
        var c = MathieuCoeffs(even, m, q);
        // cos(k x°) and sin(k x°) through the degree functions, so 90° and 180° give exact zeros.
        var xr = Math.IEEERemainder(xdeg, 360.0);
        double v = 0, d = 0;
        for (var i = c.N - 1; i >= 0; i--)
        {
            var k = c.Offset + 2 * i;
            var arg = Math.IEEERemainder(k * xr, 360.0);
            var cs = Cosdg(arg);
            var sn = Sindg(arg);
            if (even) { v += c.C[i] * cs; d -= k * c.C[i] * sn; }
            else { v += c.C[i] * sn; d += k * c.C[i] * cs; }
        }
        return (v, d);
    }

    private static (double Value, double Derivative) Radial(bool even, int kind, double m, double q, double z)
    {
        if (!IsOrder(m, even ? 0 : 1) || !(q >= 0) || double.IsNaN(z) || double.IsInfinity(q)) return (double.NaN, double.NaN);
        var mi = (int)m;
        var c = MathieuCoeffs(even, mi, q);
        var h = Math.Sqrt(q);
        var u1 = h * Math.Exp(-z);
        var u2 = h * Math.Exp(z);
        // Index of the pivot coefficient: the largest, which keeps the series well conditioned.
        var s = 0;
        for (var i = 1; i < c.N; i++) if (Math.Abs(c.C[i]) > Math.Abs(c.C[s])) s = i;
        var offset = c.Offset is 0 or 1 ? c.Offset : 2;   // ce_2n: 0, ce/se_2n+1: 1, se_2n+2: 2
        var subtract = !even;
        var n = mi % 2 == 0 ? (even ? mi / 2 : (mi - 2) / 2) : (mi - 1) / 2;
        var eps = even && mi % 2 == 0 && s == 0 ? 2.0 : 1.0;
        // Orders needed: from -s-1 up to N + s + offset + 1.
        var lo = -s - 1;
        var hi = c.N + s + offset + 1;
        var jv = new double[hi - lo + 1];
        var zv = new double[hi - lo + 1];
        for (var k = lo; k <= hi; k++)
        {
            jv[k - lo] = BesselIntOrder(k, u1, 1);
            zv[k - lo] = BesselIntOrder(k, u2, kind);
        }
        double J(int k) => jv[k - lo];
        double Z(int k) => zv[k - lo];
        double DJ(int k) => 0.5 * (J(k - 1) - J(k + 1));
        double DZ(int k) => 0.5 * (Z(k - 1) - Z(k + 1));
        double val = 0, der = 0;
        for (var l = c.N - 1; l >= 0; l--)
        {
            var sg = (l + n) % 2 == 0 ? 1.0 : -1.0;
            var a = l - s;
            var b = l + s + offset;
            var t1 = J(a) * Z(b);
            var t2 = J(b) * Z(a);
            var d1 = -u1 * DJ(a) * Z(b) + J(a) * u2 * DZ(b);
            var d2 = -u1 * DJ(b) * Z(a) + J(b) * u2 * DZ(a);
            if (subtract) { val += sg * c.C[l] * (t1 - t2); der += sg * c.C[l] * (d1 - d2); }
            else { val += sg * c.C[l] * (t1 + t2); der += sg * c.C[l] * (d1 + d2); }
        }
        var scale = eps * c.C[s];
        return (val / scale, der / scale);
    }

    /// <summary>J_k(u) (kind 1) or Y_k(u) (kind 2) for any integer k, using J₋ₖ = (−1)ᵏ Jₖ.</summary>
    private static double BesselIntOrder(int k, double u, int kind)
    {
        var sign = k < 0 && (-k) % 2 == 1 ? -1.0 : 1.0;
        var n = Math.Abs(k);
        return sign * (kind == 1 ? Jn(n, u) : Yn(n, u));
    }

    /// <summary>Characteristic value and normalised Fourier coefficients of ce_m (even) or se_m, q ≥ 0.</summary>
    private static MathieuCoefficients MathieuCoeffs(bool even, int m, double q)
    {
        MathieuKind kind;
        int j;
        if (even) { kind = m % 2 == 0 ? MathieuKind.CeEven : MathieuKind.CeOdd; j = m / 2; }
        else { kind = m % 2 == 1 ? MathieuKind.SeOdd : MathieuKind.SeEven; j = (m - 1) / 2; }
        var n = j + 30 + (int)(3 * Math.Sqrt(q));
        var d = new double[n];
        var e = new double[n - 1];
        var offset = kind switch { MathieuKind.CeEven => 0, MathieuKind.SeEven => 2, _ => 1 };
        for (var k = 0; k < n; k++) { var idx = offset + 2 * k; d[k] = (double)idx * idx; }
        for (var k = 0; k < n - 1; k++) e[k] = q;
        if (kind == MathieuKind.CeEven) e[0] = Math.Sqrt(2) * q;
        if (kind == MathieuKind.CeOdd) d[0] += q;
        if (kind == MathieuKind.SeOdd) d[0] -= q;

        var lambda = TridiagonalEigenvalue(d, e, j);
        var v = TridiagonalEigenvector(d, e, lambda);
        if (kind == MathieuKind.CeEven) v[0] /= Math.Sqrt(2);

        // Sign: the value (or slope, where the value vanishes) at π/2 keeps its q = 0 sign.
        double sgnSum = 0;
        for (var k = 0; k < n; k++)
        {
            var idx = offset + 2 * k;
            sgnSum += kind switch
            {
                MathieuKind.CeEven => (k % 2 == 0 ? 1 : -1) * v[k],
                MathieuKind.CeOdd => (k % 2 == 0 ? 1 : -1) * idx * v[k],
                MathieuKind.SeOdd => (k % 2 == 0 ? 1 : -1) * v[k],
                _ => ((k + 1) % 2 == 0 ? 1 : -1) * idx * v[k],
            };
        }
        var want = kind == MathieuKind.SeEven ? ((j + 1) % 2 == 0 ? 1 : -1) : (j % 2 == 0 ? 1 : -1);
        if (Math.Sign(sgnSum) != want) for (var k = 0; k < n; k++) v[k] = -v[k];
        // Trim the negligible tail.
        var last = n - 1;
        while (last > j + 1 && Math.Abs(v[last]) < 1e-300) last--;
        return new MathieuCoefficients(lambda, v[..(last + 1)], offset, last + 1);
    }

    /// <summary>The <paramref name="index"/>-th smallest eigenvalue of a symmetric tridiagonal matrix, by Sturm bisection to the last bit.</summary>
    private static double TridiagonalEigenvalue(double[] d, double[] e, int index)
    {
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        for (var i = 0; i < d.Length; i++)
        {
            var r = (i > 0 ? Math.Abs(e[i - 1]) : 0) + (i < e.Length ? Math.Abs(e[i]) : 0);
            lo = Math.Min(lo, d[i] - r);
            hi = Math.Max(hi, d[i] + r);
        }
        lo -= 1; hi += 1;
        while (true)
        {
            var mid = 0.5 * (lo + hi);
            if (mid <= lo || mid >= hi) break;
            if (SturmCount(d, e, mid) > index) hi = mid; else lo = mid;
        }
        return 0.5 * (lo + hi);
    }

    /// <summary>Number of eigenvalues below x (the LDLᵀ pivot signs of T − xI).</summary>
    private static int SturmCount(double[] d, double[] e, double x)
    {
        var count = 0;
        var p = d[0] - x;
        if (p < 0) count++;
        for (var i = 1; i < d.Length; i++)
        {
            if (p == 0) p = 1e-300;
            p = d[i] - x - e[i - 1] * e[i - 1] / p;
            if (p < 0) count++;
        }
        return count;
    }

    /// <summary>Unit eigenvector for the eigenvalue <paramref name="lambda"/> by inverse iteration
    /// (tridiagonal LU with partial pivoting, as LAPACK's dgttrf).</summary>
    private static double[] TridiagonalEigenvector(double[] d, double[] e, double lambda)
    {
        var n = d.Length;
        // LU of T - λI with partial pivoting: rows can swap with the next, giving a second superdiagonal.
        var dl = (double[])e.Clone();
        var dd = new double[n];
        var du = (double[])e.Clone();
        var du2 = new double[Math.Max(n - 2, 0)];
        var piv = new bool[n];
        for (var i = 0; i < n; i++) dd[i] = d[i] - lambda;
        var tiny = 1e-300;
        for (var i = 0; i < n - 1; i++)
        {
            if (Math.Abs(dd[i]) >= Math.Abs(dl[i]))
            {
                if (dd[i] == 0) dd[i] = tiny;
                var f = dl[i] / dd[i];
                dl[i] = f;
                dd[i + 1] -= f * du[i];
                if (i < n - 2) du2[i] = 0;
            }
            else
            {
                piv[i] = true;
                var f = dd[i] / dl[i];
                dd[i] = dl[i];
                dl[i] = f;
                var t = du[i];
                du[i] = dd[i + 1];
                dd[i + 1] = t - f * dd[i + 1];
                if (i < n - 2)
                {
                    du2[i] = du[i + 1];
                    du[i + 1] = -f * du[i + 1];
                }
            }
        }
        if (dd[n - 1] == 0) dd[n - 1] = tiny;
        var x = new double[n];
        Array.Fill(x, 1.0);
        for (var iter = 0; iter < 3; iter++)
        {
            // Forward: apply L⁻¹ with the recorded swaps.
            for (var i = 0; i < n - 1; i++)
            {
                if (piv[i]) { (x[i], x[i + 1]) = (x[i + 1], x[i]); }
                x[i + 1] -= dl[i] * x[i];
            }
            // Backward: U x = y.
            x[n - 1] /= dd[n - 1];
            if (n > 1) x[n - 2] = (x[n - 2] - du[n - 2] * x[n - 1]) / dd[n - 2];
            for (var i = n - 3; i >= 0; i--) x[i] = (x[i] - du[i] * x[i + 1] - du2[i] * x[i + 2]) / dd[i];
            var norm = 0.0;
            var scale = 0.0;
            foreach (var v in x) scale = Math.Max(scale, Math.Abs(v));
            for (var i = 0; i < n; i++) { x[i] /= scale; norm += x[i] * x[i]; }
            norm = Math.Sqrt(norm);
            for (var i = 0; i < n; i++) x[i] /= norm;
        }
        return x;
    }
}
