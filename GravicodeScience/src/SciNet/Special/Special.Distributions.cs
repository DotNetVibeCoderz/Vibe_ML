namespace Gravicode.Science.SciNet;

public static partial class Special
{
    // Cephes-named distribution helpers. All take the SciPy argument order and return NaN outside
    // the domain rather than throwing, because they are evaluated element-wise over arrays.

    /// <summary>Student t CDF with <paramref name="df"/> degrees of freedom at t (scipy.special.stdtr).</summary>
    /// <remarks>
    /// The tail is ½ I_{df/(df+t²)}(df/2, ½) with df/(df+t²) and t²/(df+t²) formed separately, and
    /// the upper half uses the complement directly so neither side loses digits to 1 - x.
    /// df ≥ 1e10 is treated as normal (the difference is below 1e-10 · t⁴).
    /// </remarks>
    public static double Stdtr(double df, double t)
    {
        if (double.IsNaN(df) || double.IsNaN(t) || df <= 0) return double.NaN;
        if (double.IsPositiveInfinity(t)) return 1.0;
        if (double.IsNegativeInfinity(t)) return 0.0;
        if (t == 0) return 0.5;
        if (df >= 1e10) return Ndtr(t);
        var x2 = t * t;
        var z = df / (df + x2);
        var w = x2 / (df + x2);
        var (I, Ic) = IncompleteBeta(0.5 * df, 0.5, z, w);
        return t < 0 ? 0.5 * I : 0.5 + 0.5 * Ic;
    }

    /// <summary>Inverse of <see cref="Stdtr"/> in t: Stdtr(df, t) = p (scipy.special.stdtrit).</summary>
    /// <remarks>Closed forms for df = 1 and 2; otherwise the inverse incomplete beta on the smaller tail.</remarks>
    public static double Stdtrit(double df, double p)
    {
        if (double.IsNaN(df) || double.IsNaN(p) || df <= 0 || p < 0 || p > 1) return double.NaN;
        if (p == 0) return double.NegativeInfinity;
        if (p == 1) return double.PositiveInfinity;
        if (p == 0.5) return 0.0;
        if (df == 1) return -CosPi(p) / SinPi(p);
        var lower = p < 0.5;
        var pp = lower ? p : 1.0 - p;
        if (df == 2)
        {
            var a = 2 * pp - 1; // negative
            var r = a / Math.Sqrt(2 * pp * (1 - pp));
            return lower ? r : -r;
        }
        if (df >= 1e10) return Ndtri(p);
        var (z, w) = InverseIncompleteBeta(0.5 * df, 0.5, 2 * pp, 1 - 2 * pp);
        var t = Math.Sqrt(df * (w / z));
        return lower ? -t : t;
    }

    /// <summary>Chi-square CDF with v degrees of freedom (scipy.special.chdtr) = P(v/2, x/2).</summary>
    public static double Chdtr(double v, double x)
    {
        if (double.IsNaN(v) || double.IsNaN(x) || v <= 0) return double.NaN;
        if (x < 0) return 0.0;
        return GammaInc(0.5 * v, 0.5 * x);
    }

    /// <summary>Chi-square survival function (scipy.special.chdtrc) = Q(v/2, x/2).</summary>
    public static double Chdtrc(double v, double x)
    {
        if (double.IsNaN(v) || double.IsNaN(x) || v <= 0) return double.NaN;
        if (x < 0) return 1.0;
        return GammaIncC(0.5 * v, 0.5 * x);
    }

    /// <summary>Inverse of <see cref="Chdtrc"/> in x: the x whose upper tail is y (scipy.special.chdtri).</summary>
    public static double Chdtri(double v, double y)
    {
        if (double.IsNaN(v) || double.IsNaN(y) || v <= 0 || y < 0 || y > 1) return double.NaN;
        return 2.0 * GammaIncCInv(0.5 * v, y);
    }

    /// <summary>F distribution CDF (scipy.special.fdtr): I_w(dfn/2, dfd/2), w = dfn x/(dfd + dfn x).</summary>
    public static double Fdtr(double dfn, double dfd, double x)
    {
        if (double.IsNaN(dfn) || double.IsNaN(dfd) || double.IsNaN(x) || dfn <= 0 || dfd <= 0) return double.NaN;
        if (x < 0) return double.NaN;
        if (double.IsPositiveInfinity(x)) return 1.0;
        var den = dfd + dfn * x;
        return IncompleteBeta(0.5 * dfn, 0.5 * dfd, dfn * x / den, dfd / den).I;
    }

    /// <summary>F distribution survival function (scipy.special.fdtrc).</summary>
    public static double Fdtrc(double dfn, double dfd, double x)
    {
        if (double.IsNaN(dfn) || double.IsNaN(dfd) || double.IsNaN(x) || dfn <= 0 || dfd <= 0) return double.NaN;
        if (x < 0) return double.NaN;
        if (double.IsPositiveInfinity(x)) return 0.0;
        var den = dfd + dfn * x;
        return IncompleteBeta(0.5 * dfn, 0.5 * dfd, dfn * x / den, dfd / den).Ic;
    }

    /// <summary>Inverse of <see cref="Fdtr"/> in x: the p-th quantile of F(dfn, dfd) (scipy.special.fdtri).</summary>
    public static double Fdtri(double dfn, double dfd, double p)
    {
        if (double.IsNaN(dfn) || double.IsNaN(dfd) || double.IsNaN(p) || dfn <= 0 || dfd <= 0 || p < 0 || p > 1) return double.NaN;
        if (p == 0) return 0.0;
        if (p == 1) return double.PositiveInfinity;
        var (w, wc) = InverseIncompleteBeta(0.5 * dfn, 0.5 * dfd, p, 1.0 - p);
        return dfd * w / (dfn * wc);
    }

    /// <summary>
    /// Binomial CDF: P(X ≤ k) for X ~ Bin(n, p) (scipy.special.bdtr), = I_{1-p}(n - k, k + 1).
    /// k is floored; k &lt; 0 gives 0 and k ≥ n gives 1.
    /// </summary>
    public static double Bdtr(double k, double n, double p)
    {
        if (double.IsNaN(k) || double.IsNaN(n) || double.IsNaN(p) || p < 0 || p > 1 || n < 0) return double.NaN;
        k = Math.Floor(k);
        if (k < 0) return 0.0;
        if (k >= n) return 1.0;
        if (p == 0) return 1.0;
        if (p == 1) return 0.0;
        return IncompleteBeta(k + 1, n - k, p, 1.0 - p).Ic;
    }

    /// <summary>Binomial survival function P(X &gt; k) (scipy.special.bdtrc) = I_p(k + 1, n - k).</summary>
    public static double Bdtrc(double k, double n, double p)
    {
        if (double.IsNaN(k) || double.IsNaN(n) || double.IsNaN(p) || p < 0 || p > 1 || n < 0) return double.NaN;
        k = Math.Floor(k);
        if (k < 0) return 1.0;
        if (k >= n) return 0.0;
        if (p == 0) return 0.0;
        if (p == 1) return 1.0;
        return IncompleteBeta(k + 1, n - k, p, 1.0 - p).I;
    }

    /// <summary>Inverse of <see cref="Bdtr"/> in p: the success probability with Bdtr(k, n, p) = y (scipy.special.bdtri).</summary>
    public static double Bdtri(double k, double n, double y)
    {
        if (double.IsNaN(k) || double.IsNaN(n) || double.IsNaN(y) || y < 0 || y > 1) return double.NaN;
        k = Math.Floor(k);
        if (k < 0 || n <= k) return double.NaN;
        return InverseIncompleteBeta(k + 1, n - k, 1.0 - y, y).X;
    }

    /// <summary>Poisson CDF P(X ≤ k) with mean m (scipy.special.pdtr) = Q(k + 1, m); k is floored.</summary>
    public static double Pdtr(double k, double m)
    {
        if (double.IsNaN(k) || double.IsNaN(m) || m < 0) return double.NaN;
        if (k < 0) return 0.0;
        if (m == 0) return 1.0;
        return GammaIncC(Math.Floor(k) + 1, m);
    }

    /// <summary>Poisson survival function P(X &gt; k) (scipy.special.pdtrc) = P(k + 1, m).</summary>
    public static double Pdtrc(double k, double m)
    {
        if (double.IsNaN(k) || double.IsNaN(m) || m < 0) return double.NaN;
        if (k < 0) return 1.0;
        if (m == 0) return 0.0;
        return GammaInc(Math.Floor(k) + 1, m);
    }

    /// <summary>Inverse of <see cref="Pdtr"/> in m: the Poisson mean with Pdtr(k, m) = y (scipy.special.pdtri).</summary>
    public static double Pdtri(double k, double y)
    {
        if (double.IsNaN(k) || double.IsNaN(y) || k < 0 || y < 0 || y > 1) return double.NaN;
        return GammaIncCInv(Math.Floor(k) + 1, y);
    }

    /// <summary>
    /// Negative binomial CDF: probability of k or fewer failures before the n-th success
    /// (scipy.special.nbdtr) = I_p(n, k + 1).
    /// </summary>
    public static double Nbdtr(double k, double n, double p)
    {
        if (double.IsNaN(k) || double.IsNaN(n) || double.IsNaN(p) || p < 0 || p > 1 || n <= 0) return double.NaN;
        if (k < 0) return 0.0;
        return IncompleteBeta(n, Math.Floor(k) + 1, p, 1.0 - p).I;
    }

    /// <summary>Negative binomial survival function (scipy.special.nbdtrc) = 1 - I_p(n, k + 1).</summary>
    public static double Nbdtrc(double k, double n, double p)
    {
        if (double.IsNaN(k) || double.IsNaN(n) || double.IsNaN(p) || p < 0 || p > 1 || n <= 0) return double.NaN;
        if (k < 0) return 1.0;
        return IncompleteBeta(n, Math.Floor(k) + 1, p, 1.0 - p).Ic;
    }

    /// <summary>Inverse of <see cref="Nbdtr"/> in p (scipy.special.nbdtri).</summary>
    public static double Nbdtri(double k, double n, double y)
    {
        if (double.IsNaN(k) || double.IsNaN(n) || double.IsNaN(y) || k < 0 || n <= 0 || y < 0 || y > 1) return double.NaN;
        return InverseIncompleteBeta(n, Math.Floor(k) + 1, y, 1.0 - y).X;
    }

    /// <summary>Gamma distribution CDF with rate a and shape b (scipy.special.gdtr) = P(b, a x).</summary>
    public static double Gdtr(double a, double b, double x)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(x) || a <= 0 || b <= 0) return double.NaN;
        if (x < 0) return 0.0;
        return GammaInc(b, a * x);
    }

    /// <summary>Gamma distribution survival function (scipy.special.gdtrc) = Q(b, a x).</summary>
    public static double Gdtrc(double a, double b, double x)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(x) || a <= 0 || b <= 0) return double.NaN;
        if (x < 0) return 1.0;
        return GammaIncC(b, a * x);
    }

    /// <summary>
    /// Survival function of the limiting Kolmogorov distribution, P(√n D_n &gt; y) as n → ∞
    /// (scipy.special.kolmogorov).
    /// </summary>
    /// <remarks>
    /// 2 Σ (-1)^(k-1) exp(-2k²y²) for y ≥ 1; for y &lt; 1 the Jacobi-transformed series
    /// 1 - √(2π)/y Σ exp(-(2k-1)²π²/(8y²)), which converges in a few terms where the first one would not.
    /// </remarks>
    public static double Kolmogorov(double y)
    {
        if (double.IsNaN(y)) return y;
        if (y <= 0) return 1.0;
        if (double.IsPositiveInfinity(y)) return 0.0;
        return KolmogorovSfCdf(y).Sf;
    }

    private static (double Sf, double Cdf, double Pdf) KolmogorovSfCdf(double y)
    {
        if (y < 1.0)
        {
            if (y < 0.04) return (1.0, 0.0, 0.0);
            var w = Math.PI * Math.PI / (8 * y * y);
            double cdf = 0, dcdf = 0;
            for (var k = 1; k < 30; k++)
            {
                var m = 2 * k - 1;
                var e = Math.Exp(-m * m * w);
                cdf += e;
                dcdf += e * (m * m * 2 * w / y);
                if (e < 1e-18 * cdf) break;
            }
            var c = Math.Sqrt(2 * Math.PI) / y;
            var pdf = c * (dcdf - cdf / y);
            cdf *= c;
            return (1.0 - cdf, cdf, pdf);
        }
        double sf = 0, dsf = 0, sign = 1;
        for (var k = 1; k < 100; k++)
        {
            var e = Math.Exp(-2.0 * k * k * y * y);
            sf += sign * e;
            dsf += sign * k * k * e;
            if (e < 1e-18 * Math.Abs(sf)) break;
            sign = -sign;
        }
        return (2 * sf, 1 - 2 * sf, 8 * y * dsf);
    }

    /// <summary>Inverse of <see cref="Kolmogorov"/>: the y with survival probability p (scipy.special.kolmogi).</summary>
    public static double KolmogorovInverse(double p)
    {
        if (double.IsNaN(p) || p < 0 || p > 1) return double.NaN;
        if (p == 0) return double.PositiveInfinity;
        if (p == 1) return 0.0;
        // Start from the one-term tail p ≈ 2 exp(-2y²); Newton with a bracket.
        var y = p < 0.9 ? Math.Sqrt(-0.5 * Math.Log(0.5 * p)) : 0.5;
        double lo = 0.0, hi = 50.0;
        for (var i = 0; i < 200; i++)
        {
            var (sf, cdf, pdf) = KolmogorovSfCdf(y);
            // g increases with y; the complement form keeps accuracy for p near 1.
            var g = p > 0.5 ? cdf - (1 - p) : p - sf;
            if (g < 0) lo = y; else hi = y;
            if (g == 0) return y;
            var yn = pdf > 0 ? y - g / pdf : 0.5 * (lo + hi);
            if (!(yn > lo && yn < hi)) yn = 0.5 * (lo + hi);
            if (Math.Abs(yn - y) <= 1e-16 * y) return yn;
            y = yn;
        }
        return y;
    }

    /// <summary>
    /// Exact one-sided Kolmogorov–Smirnov survival function P(D_n⁺ ≥ d) for sample size n
    /// (scipy.special.smirnov), by the Birnbaum–Tingey sum.
    /// </summary>
    /// <remarks>
    /// d Σ_{j ≤ n(1-d)} C(n, j) (d + j/n)^(j-1) (1 - d - j/n)^(n-j), all terms positive and summed in
    /// log space with a running maximum. The powers amplify the rounding of their bases by their
    /// exponents, so relative accuracy is about n·1e-16 (1e-13 at n = 1000).
    /// </remarks>
    public static double Smirnov(int n, double d)
    {
        if (n <= 0 || double.IsNaN(d)) return double.NaN;
        if (d <= 0) return 1.0;
        if (d >= 1) return 0.0;
        if (n == 1) return 1.0 - d;
        var jmax = (int)Math.Floor(n * (1 - d));
        if (n * (1 - d) - jmax < 1e-12 && jmax > 0 && (double)(n - jmax) / n - d <= 0) jmax--;
        var logTerms = new double[jmax + 1];
        var lgn = GammaLn(n + 1.0);
        var m = double.NegativeInfinity;
        for (var j = 0; j <= jmax; j++)
        {
            var rest = (double)(n - j) / n - d;
            if (rest <= 0) { logTerms[j] = double.NegativeInfinity; continue; }
            var lc = lgn - GammaLn(j + 1.0) - GammaLn(n - j + 1.0);
            var lt = lc + (j - 1) * Math.Log(d + (double)j / n) + (n - j) * Math.Log(rest);
            logTerms[j] = lt;
            if (lt > m) m = lt;
        }
        var s = 0.0;
        for (var j = 0; j <= jmax; j++) s += Math.Exp(logTerms[j] - m);
        return Math.Min(1.0, d * s * Math.Exp(m));
    }

    /// <summary>Inverse of <see cref="Smirnov"/> in d (scipy.special.smirnovi).</summary>
    public static double SmirnovInverse(int n, double p)
    {
        if (n <= 0 || double.IsNaN(p) || p < 0 || p > 1) return double.NaN;
        if (p == 0) return 1.0;
        if (p == 1) return 0.0;
        if (n == 1) return 1.0 - p;
        double lo = 0, hi = 1;
        var d = Math.Min(0.999, Math.Sqrt(-Math.Log(p) / (2.0 * n)));
        for (var i = 0; i < 200; i++)
        {
            var f = Smirnov(n, d) - p; // decreasing in d
            if (f > 0) lo = d; else hi = d;
            if (f == 0) return d;
            // Secant-free: numerical derivative is unreliable at kinks, so bisect once close.
            var h = 1e-7 * Math.Max(d, 1e-3);
            var fp = (Smirnov(n, Math.Min(d + h, 1)) - Smirnov(n, Math.Max(d - h, 0))) / (Math.Min(d + h, 1) - Math.Max(d - h, 0));
            var dn = fp < 0 ? d - f / fp : 0.5 * (lo + hi);
            if (!(dn > lo && dn < hi)) dn = 0.5 * (lo + hi);
            if (Math.Abs(dn - d) <= 1e-15 * d || hi - lo < 1e-15) return dn;
            d = dn;
        }
        return d;
    }
}
