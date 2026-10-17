namespace Gravicode.Science.SciNet;

public static partial class Special
{
    // The cdflib-style inverses scipy exposes: each solves CDF(parameter) = p for one argument of an
    // existing CDF. cdflib searches with a 1e-10 relative tolerance; these bracket the root and then
    // run Brent's method to full precision, so results agree with scipy to ~1e-10 and round-trip
    // through the CDF to ~1e-14. Where p is not attainable for any value of the parameter, cdflib
    // returns its search bound (e.g. ±1e100, 1e10 or 0 with an error flag); these return NaN instead,
    // except where the bound is the natural answer (a count of 0 for the discrete inverses).

    // ------------------------------------------------------------------ noncentral CDFs

    /// <summary>Non-central chi-squared CDF (scipy.special.chndtr): P(X ≤ x), df &gt; 0, nc ≥ 0.</summary>
    /// <remarks>A Poisson(nc/2) mixture of central chi-squared CDFs, summed outward from the Poisson mode.</remarks>
    public static double Chndtr(double x, double df, double nc)
    {
        if (double.IsNaN(x) || double.IsNaN(df) || double.IsNaN(nc) || !(df > 0) || nc < 0 || double.IsInfinity(nc)) return double.NaN;
        if (x <= 0) return 0.0;
        if (double.IsPositiveInfinity(x)) return 1.0;
        var lower = PoissonMixtureSum(nc / 2, j => GammaInc(df / 2 + j, x / 2));
        var upper = PoissonMixtureSum(nc / 2, j => GammaIncC(df / 2 + j, x / 2));
        // Each tail is a positive sum; return the one that is not a complement.
        return lower <= 0.5 ? Math.Min(1, lower) : Math.Max(0, 1 - upper);
    }

    /// <summary>Inverse of <see cref="Chndtr"/> in x (scipy.special.chndtrix).</summary>
    public static double Chndtrix(double p, double df, double nc)
    {
        if (!ValidP(p) || !(df > 0) || !(nc >= 0)) return double.NaN;
        if (p == 0) return 0;
        if (p == 1) return double.PositiveInfinity;
        return SolveIncreasing(x => Chndtr(x, df, nc), p, 0, double.PositiveInfinity, Math.Max(df + nc, 1));
    }

    /// <summary>Inverse of <see cref="Chndtr"/> in df (scipy.special.chndtridf).</summary>
    public static double Chndtridf(double x, double p, double nc)
    {
        if (!ValidP(p) || !(x >= 0) || !(nc >= 0)) return double.NaN;
        return SolveDecreasing(df => Chndtr(x, df, nc), p, 0, double.PositiveInfinity, 1);
    }

    /// <summary>Inverse of <see cref="Chndtr"/> in nc (scipy.special.chndtrinc).</summary>
    public static double Chndtrinc(double x, double df, double p)
    {
        if (!ValidP(p) || !(x >= 0) || !(df > 0)) return double.NaN;
        return SolveDecreasing(nc => Chndtr(x, df, nc), p, 0, 1e9, 1);
    }

    /// <summary>Non-central Student t CDF (scipy.special.nctdtr): P(T ≤ t), df &gt; 0.</summary>
    /// <remarks>Lenth's AS 243 series (with the AS R80 correction); ~1e-12 in the body, the far upper
    /// tail is a complement and loses relative accuracy (scipy uses Boost there).</remarks>
    public static double Nctdtr(double df, double nc, double t)
    {
        if (double.IsNaN(df) || double.IsNaN(nc) || double.IsNaN(t) || !(df > 0) || double.IsInfinity(nc)) return double.NaN;
        if (double.IsPositiveInfinity(t)) return 1.0;
        if (double.IsNegativeInfinity(t)) return 0.0;
        return t >= 0 ? Tnc(t, df, nc) : 1 - Tnc(-t, df, -nc);
    }

    /// <summary>Inverse of <see cref="Nctdtr"/> in t (scipy.special.nctdtrit).</summary>
    public static double Nctdtrit(double df, double nc, double p)
    {
        if (!ValidP(p) || !(df > 0) || double.IsNaN(nc)) return double.NaN;
        if (p == 0) return double.NegativeInfinity;
        if (p == 1) return double.PositiveInfinity;
        return SolveIncreasing(t => Nctdtr(df, nc, t), p, double.NegativeInfinity, double.PositiveInfinity, nc);
    }

    /// <summary>Inverse of <see cref="Nctdtr"/> in df (scipy.special.nctdtridf).</summary>
    public static double Nctdtridf(double p, double nc, double t)
    {
        if (!ValidP(p) || double.IsNaN(nc) || double.IsNaN(t)) return double.NaN;
        var f1 = Nctdtr(1, nc, t);
        var f2 = Nctdtr(1e6, nc, t);
        return f1 <= f2 ? SolveIncreasing(df => Nctdtr(df, nc, t), p, 0, 1e10, 1) : SolveDecreasing(df => Nctdtr(df, nc, t), p, 0, 1e10, 1);
    }

    /// <summary>Inverse of <see cref="Nctdtr"/> in nc (scipy.special.nctdtrinc).</summary>
    public static double Nctdtrinc(double df, double p, double t)
    {
        if (!ValidP(p) || !(df > 0) || double.IsNaN(t)) return double.NaN;
        return SolveDecreasing(nc => Nctdtr(df, nc, t), p, double.NegativeInfinity, double.PositiveInfinity, t);
    }

    /// <summary>Non-central F CDF (scipy.special.ncfdtr): dfn, dfd &gt; 0, nc ≥ 0, f ≥ 0.</summary>
    /// <remarks>A Poisson(nc/2) mixture of regularized incomplete beta functions.</remarks>
    public static double Ncfdtr(double dfn, double dfd, double nc, double f)
    {
        if (double.IsNaN(dfn) || double.IsNaN(dfd) || double.IsNaN(nc) || double.IsNaN(f) || !(dfn > 0) || !(dfd > 0) || nc < 0 || double.IsInfinity(nc)) return double.NaN;
        if (f <= 0) return 0.0;
        if (double.IsPositiveInfinity(f)) return 1.0;
        var den = dfn * f + dfd;
        var y = dfn * f / den;
        var y1 = dfd / den;
        var lower = PoissonMixtureSum(nc / 2, j => y <= 0.5 ? BetaInc(dfn / 2 + j, dfd / 2, y) : BetaIncC(dfd / 2, dfn / 2 + j, y1));
        if (lower <= 0.5) return Math.Min(1, lower);
        var upper = PoissonMixtureSum(nc / 2, j => y <= 0.5 ? BetaIncC(dfn / 2 + j, dfd / 2, y) : BetaInc(dfd / 2, dfn / 2 + j, y1));
        return Math.Max(0, 1 - upper);
    }

    /// <summary>Inverse of <see cref="Ncfdtr"/> in f (scipy.special.ncfdtri).</summary>
    public static double Ncfdtri(double dfn, double dfd, double nc, double p)
    {
        if (!ValidP(p) || !(dfn > 0) || !(dfd > 0) || !(nc >= 0)) return double.NaN;
        if (p == 0) return 0;
        if (p == 1) return double.PositiveInfinity;
        return SolveIncreasing(f => Ncfdtr(dfn, dfd, nc, f), p, 0, double.PositiveInfinity, 1);
    }

    /// <summary>Inverse of <see cref="Ncfdtr"/> in dfd (scipy.special.ncfdtridfd).</summary>
    public static double Ncfdtridfd(double dfn, double p, double nc, double f)
    {
        if (!ValidP(p) || !(dfn > 0) || !(nc >= 0) || !(f >= 0)) return double.NaN;
        return SolveIncreasing(dfd => Ncfdtr(dfn, dfd, nc, f), p, 0, 1e10, 1);
    }

    /// <summary>Inverse of <see cref="Ncfdtr"/> in dfn (scipy.special.ncfdtridfn).</summary>
    public static double Ncfdtridfn(double p, double dfd, double nc, double f)
    {
        if (!ValidP(p) || !(dfd > 0) || !(nc >= 0) || !(f >= 0)) return double.NaN;
        var a = Ncfdtr(1e-3, dfd, nc, f);
        var b = Ncfdtr(1e3, dfd, nc, f);
        return a <= b ? SolveIncreasing(dfn => Ncfdtr(dfn, dfd, nc, f), p, 0, 1e10, 1) : SolveDecreasing(dfn => Ncfdtr(dfn, dfd, nc, f), p, 0, 1e10, 1);
    }

    /// <summary>Inverse of <see cref="Ncfdtr"/> in nc (scipy.special.ncfdtrinc).</summary>
    public static double Ncfdtrinc(double dfn, double dfd, double p, double f)
    {
        if (!ValidP(p) || !(dfn > 0) || !(dfd > 0) || !(f >= 0)) return double.NaN;
        return SolveDecreasing(nc => Ncfdtr(dfn, dfd, nc, f), p, 0, 1e4, 1);
    }

    // ------------------------------------------------------------------ central-family inverses

    /// <summary>Inverse of <see cref="Gdtr"/> in x (scipy.special.gdtrix): Gdtr(a, b, x) = p, with rate a and shape b.</summary>
    public static double Gdtrix(double a, double b, double p)
    {
        if (!ValidP(p) || !(a > 0) || !(b > 0)) return double.NaN;
        return GammaIncInv(b, p) / a;
    }

    /// <summary>Inverse of <see cref="Gdtr"/> in the rate a (scipy.special.gdtria).</summary>
    public static double Gdtria(double p, double b, double x)
    {
        if (!ValidP(p) || !(b > 0) || !(x >= 0)) return double.NaN;
        if (x == 0) return double.NaN;
        return GammaIncInv(b, p) / x;
    }

    /// <summary>Inverse of <see cref="Gdtr"/> in the shape b (scipy.special.gdtrib).</summary>
    public static double Gdtrib(double a, double p, double x)
    {
        if (!ValidP(p) || !(a > 0) || !(x >= 0)) return double.NaN;
        return SolveDecreasing(b => GammaInc(b, a * x), p, 0, double.PositiveInfinity, 1);
    }

    /// <summary>Inverse of <see cref="Stdtr"/> in df (scipy.special.stdtridf); NaN when p is not attainable for any df.</summary>
    public static double Stdtridf(double p, double t)
    {
        if (!ValidP(p) || double.IsNaN(t)) return double.NaN;
        // For t > 0 the CDF increases with df (towards Φ(t)); for t < 0 it decreases.
        return t >= 0 ? SolveIncreasing(df => Stdtr(df, t), p, 0, 1e10, 1) : SolveDecreasing(df => Stdtr(df, t), p, 0, 1e10, 1);
    }

    /// <summary>Inverse of <see cref="Fdtr"/> in dfd (scipy.special.fdtridfd).</summary>
    public static double Fdtridfd(double dfn, double p, double x)
    {
        if (!ValidP(p) || !(dfn > 0) || !(x >= 0)) return double.NaN;
        var a = Fdtr(dfn, 1e-3, x);
        var b = Fdtr(dfn, 1e6, x);
        return a <= b ? SolveIncreasing(d => Fdtr(dfn, d, x), p, 0, 1e10, 1) : SolveDecreasing(d => Fdtr(dfn, d, x), p, 0, 1e10, 1);
    }

    /// <summary>Inverse of the beta CDF I_x(a, b) in a (scipy.special.btdtria).</summary>
    public static double Btdtria(double p, double b, double x)
    {
        if (!ValidP(p) || !(b > 0) || !(x >= 0 && x <= 1)) return double.NaN;
        return SolveDecreasing(a => BetaInc(a, b, x), p, 0, 1e10, 1);
    }

    /// <summary>Inverse of the beta CDF I_x(a, b) in b (scipy.special.btdtrib).</summary>
    public static double Btdtrib(double a, double p, double x)
    {
        if (!ValidP(p) || !(a > 0) || !(x >= 0 && x <= 1)) return double.NaN;
        return SolveIncreasing(b => BetaInc(a, b, x), p, 0, 1e10, 1);
    }

    // ------------------------------------------------------------------ discrete (continuous in the count)

    /// <summary>
    /// Inverse of the binomial CDF in the number of successes k (scipy.special.bdtrik), with cdflib's
    /// continuous extension P(k) = I_{1−pr}(n − k, k + 1); 0 when p is below P(0), n when above P(n).
    /// </summary>
    public static double Bdtrik(double p, double n, double pr)
    {
        if (!ValidP(p) || !(n >= 0) || !(pr >= 0 && pr <= 1)) return double.NaN;
        double F(double k) => k >= n ? 1.0 : BetaIncC(k + 1, n - k, pr);
        if (p <= F(0)) return 0;
        if (p >= 1) return n;
        return SolveIncreasing(F, p, 0, n, n / 2);
    }

    /// <summary>Inverse of the binomial CDF in the number of trials n (scipy.special.bdtrin), continuous extension.</summary>
    public static double Bdtrin(double k, double p, double pr)
    {
        if (!ValidP(p) || !(k >= 0) || !(pr >= 0 && pr <= 1)) return double.NaN;
        double F(double n) => k >= n ? 1.0 : BetaIncC(k + 1, n - k, pr);
        return SolveDecreasing(F, p, k, 1e10, k + 1);
    }

    /// <summary>Inverse of the Poisson CDF in the count k (scipy.special.pdtrik), continuous extension Q(k + 1, m); 0 below P(0).</summary>
    public static double Pdtrik(double p, double m)
    {
        if (!ValidP(p) || !(m >= 0)) return double.NaN;
        double F(double k) => GammaIncC(k + 1, m);
        if (p <= F(0)) return 0;
        if (p == 1) return double.PositiveInfinity;
        return SolveIncreasing(F, p, 0, double.PositiveInfinity, Math.Max(m, 1));
    }

    /// <summary>Inverse of the negative binomial CDF in the failures k (scipy.special.nbdtrik), continuous extension I_pr(n, k + 1); 0 below P(0).</summary>
    public static double Nbdtrik(double p, double n, double pr)
    {
        if (!ValidP(p) || !(n > 0) || !(pr >= 0 && pr <= 1)) return double.NaN;
        double F(double k) => BetaInc(n, k + 1, pr);
        if (p <= F(0)) return 0;
        if (p == 1) return double.PositiveInfinity;
        return SolveIncreasing(F, p, 0, double.PositiveInfinity, n);
    }

    /// <summary>Inverse of the negative binomial CDF in the successes n (scipy.special.nbdtrin), continuous extension.</summary>
    public static double Nbdtrin(double k, double p, double pr)
    {
        if (!ValidP(p) || !(k >= 0) || !(pr >= 0 && pr <= 1)) return double.NaN;
        return SolveDecreasing(n => BetaInc(n, k + 1, pr), p, 0, 1e10, 1);
    }

    // ------------------------------------------------------------------ machinery

    private static bool ValidP(double p) => p >= 0 && p <= 1;

    /// <summary>Σ_j Pois(j; λ)·term(j), outward from the Poisson mode until the weights stop mattering.</summary>
    private static double PoissonMixtureSum(double lambda, Func<int, double> term)
    {
        if (lambda == 0) return term(0);
        var mode = (int)Math.Floor(lambda);
        var logLam = Math.Log(lambda);
        double W(int j) => Math.Exp(-lambda + j * logLam - GammaLn(j + 1));
        double sum = 0, wsum = 0;
        for (var j = mode; ; j++)
        {
            var w = W(j);
            var t = w * term(j);
            sum += t; wsum += w;
            if ((w < 1e-17 * wsum && t <= 1e-17 * sum) || j - mode > 100000) break;
        }
        for (var j = mode - 1; j >= 0; j--)
        {
            var w = W(j);
            var t = w * term(j);
            sum += t;
            if (w < 1e-17 * wsum && t <= 1e-17 * sum) break;
        }
        return sum;
    }

    /// <summary>AS 243 (Lenth 1989): P(T ≤ t) for t ≥ 0 of the noncentral t.</summary>
    private static double Tnc(double t, double df, double delta)
    {
        const double errmax = 1e-14;
        const int itrmax = 5000;
        var x = t * t / (t * t + df);
        double tnc = 0;
        if (x > 0)
        {
            var lambda = delta * delta;
            var p = 0.5 * Math.Exp(-0.5 * lambda);
            var q = Math.Sqrt(2 / Math.PI) * p * delta;
            var s = -0.5 * Expm1(-0.5 * lambda);
            var a = 0.5;
            var b = 0.5 * df;
            var rxb = Math.Pow(1 - x, b);
            var albeta = 0.5 * Math.Log(Math.PI) + GammaLn(b) - GammaLn(0.5 + b);
            var xodd = BetaInc(a, b, x);
            var godd = 2 * rxb * Math.Exp(a * Math.Log(x) - albeta);
            var bx = b * x;
            var xeven = bx < 2.2e-16 ? bx : -Expm1(b * Log1p(-x));
            var geven = bx * rxb;
            tnc = p * xodd + q * xeven;
            for (var en = 1.0; en <= itrmax; en++)
            {
                a += 1;
                xodd -= godd;
                xeven -= geven;
                godd *= x * (a + b - 1) / a;
                geven *= x * (a + b - 0.5) / (a + 0.5);
                p *= lambda / (2 * en);
                q *= lambda / (2 * en + 1);
                s -= p;
                tnc += p * xodd + q * xeven;
                var errbd = 2 * s * (xodd - godd);
                if (Math.Abs(errbd) <= errmax && en > lambda / 2) break;
            }
        }
        tnc += Ndtr(-delta);
        return Math.Clamp(tnc, 0, 1);
    }

    private static double SolveIncreasing(Func<double, double> f, double p, double lo, double hi, double guess)
        => SolveMonotone(v => f(v) - p, lo, hi, guess);

    private static double SolveDecreasing(Func<double, double> f, double p, double lo, double hi, double guess)
        => SolveMonotone(v => p - f(v), lo, hi, guess);

    /// <summary>
    /// Root of an increasing g on (lo, hi): brackets outward from <paramref name="guess"/> (geometrically
    /// on a half-line, so tiny and huge roots are both reached), then Brent's method. NaN when g has
    /// no sign change inside the domain.
    /// </summary>
    private static double SolveMonotone(Func<double, double> g, double lo, double hi, double guess)
    {
        if (double.IsNaN(guess) || guess <= lo || guess >= hi)
            guess = double.IsInfinity(lo) ? (double.IsInfinity(hi) ? 0 : hi - 1) : double.IsInfinity(hi) ? lo + 1 : 0.5 * (lo + hi);
        double a, b, ga, gb;
        var g0 = g(guess);
        if (double.IsNaN(g0)) return double.NaN;
        if (g0 == 0) return guess;
        var step = Math.Max(1.0, Math.Abs(guess));
        if (g0 > 0)
        {
            b = guess; gb = g0; a = guess; ga = g0;
            for (var it = 0; ; it++)
            {
                var na = double.IsNegativeInfinity(lo) ? a - step : lo + (a - lo) / 4;
                step *= 2;
                if (it > 3000 || na <= lo || na == a) return double.NaN;
                a = na; ga = g(a);
                if (double.IsNaN(ga)) return double.NaN;
                if (ga <= 0) break;
                b = a; gb = ga;
            }
        }
        else
        {
            a = guess; ga = g0; b = guess; gb = g0;
            for (var it = 0; ; it++)
            {
                var nb = double.IsPositiveInfinity(hi) ? (b < 1 ? b + step : b * 4) : hi - (hi - b) / 4;
                step *= 2;
                if (it > 3000 || nb >= hi || nb == b || double.IsInfinity(nb)) return double.NaN;
                b = nb; gb = g(b);
                if (double.IsNaN(gb)) return double.NaN;
                if (gb >= 0) break;
                a = b; ga = gb;
            }
        }
        if (ga == 0) return a;
        if (gb == 0) return b;
        return Brent(g, a, b, ga, gb);
    }

    private static double Brent(Func<double, double> f, double a, double b, double fa, double fb)
    {
        double c = a, fc = fa, d = b - a, e = d;
        for (var it = 0; it < 300; it++)
        {
            if (Math.Sign(fb) == Math.Sign(fc)) { c = a; fc = fa; d = b - a; e = d; }
            if (Math.Abs(fc) < Math.Abs(fb)) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }
            var tol = 2 * 2.220446049250313e-16 * Math.Abs(b) + 1e-300;
            var m = 0.5 * (c - b);
            if (Math.Abs(m) <= tol || fb == 0) return b;
            if (Math.Abs(e) >= tol && Math.Abs(fa) > Math.Abs(fb))
            {
                double p, q, r;
                var s = fb / fa;
                if (a == c) { p = 2 * m * s; q = 1 - s; }
                else
                {
                    q = fa / fc; r = fb / fc;
                    p = s * (2 * m * q * (q - r) - (b - a) * (r - 1));
                    q = (q - 1) * (r - 1) * (s - 1);
                }
                if (p > 0) q = -q; else p = -p;
                if (2 * p < Math.Min(3 * m * q - Math.Abs(tol * q), Math.Abs(e * q))) { e = d; d = p / q; }
                else { d = m; e = d; }
            }
            else { d = m; e = d; }
            a = b; fa = fb;
            b += Math.Abs(d) > tol ? d : (m > 0 ? tol : -tol);
            fb = f(b);
        }
        return b;
    }
}
