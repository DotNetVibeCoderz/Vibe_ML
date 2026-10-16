namespace Gravicode.Science.SciNet;

public static partial class Special
{
    /// <summary>The Riemann zeta function ζ(x) for any real x (scipy.special.zeta with q omitted).</summary>
    /// <remarks>
    /// x &gt; 1: Euler–Maclaurin summation (Cephes zeta). 0 ≤ x &lt; 1: Borwein's alternating-series
    /// algorithm for the Dirichlet eta function, ζ = η / (1 - 2^(1-x)) with the denominator from
    /// expm1 so the pole at 1 keeps relative accuracy. x &lt; 0: the functional equation, in
    /// logarithms when Γ(1 - x) would overflow. ζ(1) = +∞.
    /// </remarks>
    public static double Zeta(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (x == 1.0) return double.PositiveInfinity;
        if (double.IsPositiveInfinity(x)) return 1.0;
        if (x > 1.0) return x > 60 ? 1.0 + Zetac(x) : HurwitzZeta(x, 1.0);
        if (x >= 0.0) return DirichletEta(x) / -Expm1((1.0 - x) * Math.Log(2.0));
        if (double.IsNegativeInfinity(x)) return double.NaN;
        // ζ(x) = 2^x π^(x-1) sin(πx/2) Γ(1-x) ζ(1-x)
        var s = SinPi(0.5 * x);
        if (s == 0.0) return 0.0;
        var z1 = Zeta(1.0 - x);
        if (1.0 - x < 170.0)
            return Math.Pow(2.0, x) * Math.Pow(Math.PI, x - 1.0) * s * Gamma(1.0 - x) * z1;
        var lg = x * Math.Log(2.0) + (x - 1.0) * LnPi + GammaLn(1.0 - x) + Math.Log(Math.Abs(s));
        return Math.Sign(s) * Math.Exp(lg) * z1;
    }

    /// <summary>The Hurwitz zeta function ζ(x, q) = Σ (k + q)^-x (scipy.special.zeta(x, q)).</summary>
    /// <remarks>
    /// Cephes zeta: direct summation until the terms are small enough for the Euler–Maclaurin
    /// tail. Defined for x &gt; 1 (x = 1 gives ∞, x &lt; 1 NaN, as in SciPy); for q ≤ 0 it is ∞ at
    /// integer q and NaN unless x is an integer.
    /// </remarks>
    public static double Zeta(double x, double q) => HurwitzZeta(x, q);

    /// <summary>ζ(x) - 1 without cancellation for large x (scipy.special.zetac).</summary>
    public static double Zetac(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (double.IsPositiveInfinity(x)) return 0.0;
        if (x == 1.0) return double.PositiveInfinity;
        if (x > 1.0)
        {
            if (x > 127) return Math.Pow(2.0, -x) + Math.Pow(3.0, -x); // 4^-x is below 2^-x * 2^-127
            return HurwitzZeta(x, 2.0);
        }
        return Zeta(x) - 1.0;
    }

    /// <summary>Dirichlet eta η(s) = Σ (-1)^(k-1) k^-s by Borwein's algorithm 2 (n = 30).</summary>
    private static double DirichletEta(double s)
    {
        const int n = 30;
        // d_k = n Σ_{i=0}^{k} (n + i - 1)! 4^i / ((n - i)! (2i)!)
        Span<double> d = stackalloc double[n + 1];
        double term = 1.0 / n, sum = term;
        d[0] = n * sum;
        for (var i = 1; i <= n; i++)
        {
            term *= (double)(n + i - 1) * (n - i + 1) * 4.0 / ((2.0 * i - 1) * (2.0 * i));
            sum += term;
            d[i] = n * sum;
        }
        var acc = 0.0;
        for (var k = 0; k < n; k++)
        {
            var t = (d[k] - d[n]) * Math.Pow(k + 1, -s);
            acc += (k % 2 == 0) ? t : -t;
        }
        return -acc / d[n];
    }

    /// <summary>Cephes zeta.c (Hurwitz), shared by <see cref="Zeta(double, double)"/> and <see cref="Polygamma"/>.</summary>
    internal static double HurwitzZeta(double x, double q)
    {
        if (double.IsNaN(x) || double.IsNaN(q)) return double.NaN;
        if (x == 1.0) return double.PositiveInfinity;
        if (x < 1.0) return double.NaN;
        if (q <= 0)
        {
            if (q == Math.Floor(q)) return double.PositiveInfinity;
            if (x != Math.Floor(x)) return double.NaN;
        }
        if (double.IsPositiveInfinity(q)) return 0.0;
        var a = SpecialTables.ZetaEulerMaclaurin;
        var s = Math.Pow(q, -x);
        var aa = q;
        var i = 0;
        var b = 0.0;
        while (i < 9 || aa <= 9.0)
        {
            i++;
            aa += 1.0;
            b = Math.Pow(aa, -x);
            s += b;
            if (Math.Abs(b) < MachEp * Math.Abs(s)) return s;
        }
        var w = aa;
        s += b * w / (x - 1.0);
        s -= 0.5 * b;
        var fac = 1.0;
        var k = 0.0;
        for (i = 0; i < 12; i++)
        {
            fac *= x + k;
            b /= w;
            var t = fac * b / a[i];
            s += t;
            if (Math.Abs(t / s) < MachEp) return s;
            k += 1.0;
            fac *= x + k;
            b /= w;
            k += 1.0;
        }
        return s;
    }
}
