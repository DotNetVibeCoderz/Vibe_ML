namespace Gravicode.Science.SciNet;

public static partial class Special
{
    // ---------------------------------------------------------------- incomplete gamma

    /// <summary>The regularised lower incomplete gamma function P(a, x) (scipy.special.gammainc).</summary>
    /// <remarks>
    /// SciPy's igam (Cephes + Boost): the uniform asymptotic expansion of Temme (DLMF 8.12.3, 25 × 25
    /// coefficient table) when a is large and x ≈ a, the power series DLMF 8.11.4 otherwise, and 1 - Q
    /// for x &gt; max(1, a). The prefactor x^a e^-x / Γ(a) uses the Lanczos form, which is accurate for
    /// large a where a·log x - x - lgamma(a) cancels. P(0, x &gt; 0) = 1; negative arguments give NaN.
    /// </remarks>
    public static double GammaInc(double a, double x)
    {
        if (double.IsNaN(a) || double.IsNaN(x) || x < 0 || a < 0) return double.NaN;
        if (a == 0) return x > 0 ? 1.0 : double.NaN;
        if (x == 0) return 0.0;
        if (double.IsPositiveInfinity(a)) return double.IsPositiveInfinity(x) ? double.NaN : 0.0;
        if (double.IsPositiveInfinity(x)) return 1.0;
        var absxmaA = Math.Abs(x - a) / a;
        if ((a > 20 && a < 200 && absxmaA < 0.3) || (a > 200 && absxmaA < 4.5 / Math.Sqrt(a)))
            return IgamAsymptotic(a, x, true);
        if (x > 1.0 && x > a) return 1.0 - GammaIncC(a, x);
        return IgamSeries(a, x);
    }

    /// <summary>The regularised upper incomplete gamma function Q(a, x) = 1 - P(a, x) (scipy.special.gammaincc).</summary>
    /// <remarks>
    /// SciPy's igamc: Temme's uniform expansion near the transition, Legendre's continued fraction
    /// (DLMF 8.9.2) for x &gt; 1.1 and x ≥ a, the cancellation-free series DLMF 8.7.3 for small x,
    /// 1 - P elsewhere. Q(0, x &gt; 0) = 0.
    /// </remarks>
    public static double GammaIncC(double a, double x)
    {
        if (double.IsNaN(a) || double.IsNaN(x) || x < 0 || a < 0) return double.NaN;
        if (a == 0) return x > 0 ? 0.0 : double.NaN;
        if (x == 0) return 1.0;
        if (double.IsPositiveInfinity(a)) return double.IsPositiveInfinity(x) ? double.NaN : 1.0;
        if (double.IsPositiveInfinity(x)) return 0.0;
        var absxmaA = Math.Abs(x - a) / a;
        if ((a > 20 && a < 200 && absxmaA < 0.3) || (a > 200 && absxmaA < 4.5 / Math.Sqrt(a)))
            return IgamAsymptotic(a, x, false);
        if (x > 1.1)
            return x < a ? 1.0 - IgamSeries(a, x) : IgamcContinuedFraction(a, x);
        if (x <= 0.5)
            return -0.4 / Math.Log(x) < a ? 1.0 - IgamSeries(a, x) : IgamcSeries(a, x);
        return x * 1.1 < a ? 1.0 - IgamSeries(a, x) : IgamcSeries(a, x);
    }

    /// <summary>x^a e^-x / Γ(a), via Lanczos when x is near a (Boost igamma prefix, SciPy igam_fac).</summary>
    internal static double IgamFac(double a, double x)
    {
        if (Math.Abs(a - x) > 0.4 * Math.Abs(a))
        {
            var ax = a * Math.Log(x) - x - GammaLn(a);
            if (ax < -MaxLog) return 0.0;
            return Math.Exp(ax);
        }
        var fac = a + LanczosG - 0.5;
        var res = Math.Sqrt(fac / Math.E) / LanczosSumExpgScaled(a);
        if (a < 200 && x < 200)
        {
            res *= Math.Exp(a - x) * Math.Pow(x / fac, a);
        }
        else
        {
            var num = x - a - LanczosG + 0.5;
            res *= Math.Exp(a * Log1pmx(num / fac) + x * (0.5 - LanczosG) / fac);
        }
        return res;
    }

    private static double IgamSeries(double a, double x)
    {
        var ax = IgamFac(a, x);
        if (ax == 0.0) return 0.0;
        double r = a, c = 1.0, ans = 1.0;
        for (var i = 0; i < 100000; i++)
        {
            r += 1.0;
            c *= x / r;
            ans += c;
            if (c <= MachEp * ans) break;
        }
        return ans * ax / a;
    }

    private static double IgamcSeries(double a, double x)
    {
        double fac = 1, sum = 0;
        for (var n = 1; n < 2000; n++)
        {
            fac *= -x / n;
            var term = fac / (a + n);
            sum += term;
            if (Math.Abs(term) <= MachEp * Math.Abs(sum)) break;
        }
        var logx = Math.Log(x);
        var t = -Expm1(a * logx - GammaLn(1 + a));
        return t - Math.Exp(a * logx - GammaLn(a)) * sum;
    }

    private static double IgamAsymptotic(double a, double x, bool lower)
    {
        var d = SpecialTables.IgamAsymptoticD;
        var lambda = x / a;
        var sigma = (x - a) / a;
        double eta;
        if (lambda > 1) eta = Math.Sqrt(-2 * Log1pmx(sigma));
        else if (lambda < 1) eta = -Math.Sqrt(-2 * Log1pmx(sigma));
        else eta = 0;
        var sgn = lower ? -1.0 : 1.0;
        var res = 0.5 * Erfc(sgn * eta * Math.Sqrt(a / 2));
        Span<double> etapow = stackalloc double[25];
        etapow[0] = 1;
        var maxpow = 0;
        double sum = 0, afac = 1, absoldterm = double.PositiveInfinity;
        for (var k = 0; k < 25; k++)
        {
            var ck = d[25 * k];
            for (var n = 1; n < 25; n++)
            {
                if (n > maxpow)
                {
                    etapow[n] = eta * etapow[n - 1];
                    maxpow++;
                }
                var ckterm = d[25 * k + n] * etapow[n];
                ck += ckterm;
                if (Math.Abs(ckterm) < MachEp * Math.Abs(ck)) break;
            }
            var term = ck * afac;
            var absterm = Math.Abs(term);
            if (absterm > absoldterm) break;
            sum += term;
            if (absterm < MachEp * Math.Abs(sum)) break;
            absoldterm = absterm;
            afac /= a;
        }
        res += sgn * Math.Exp(-0.5 * a * eta * eta) * sum / Math.Sqrt(2 * Math.PI * a);
        return res;
    }

    private static double IgamcContinuedFraction(double a, double x)
    {
        const double big = 4.503599627370496e15, bigInv = 2.22044604925031308085e-16;
        var ax = IgamFac(a, x);
        if (ax == 0.0) return 0.0;
        var y = 1.0 - a;
        var z = x + y + 1.0;
        var c = 0.0;
        double pkm2 = 1.0, qkm2 = x, pkm1 = x + 1.0, qkm1 = z * x;
        var ans = pkm1 / qkm1;
        for (var i = 0; i < 100000; i++)
        {
            c += 1.0;
            y += 1.0;
            z += 2.0;
            var yc = y * c;
            var pk = pkm1 * z - pkm2 * yc;
            var qk = qkm1 * z - qkm2 * yc;
            double t;
            if (qk != 0)
            {
                var r = pk / qk;
                t = Math.Abs((ans - r) / r);
                ans = r;
            }
            else t = 1.0;
            pkm2 = pkm1; pkm1 = pk; qkm2 = qkm1; qkm1 = qk;
            if (Math.Abs(pk) > big)
            {
                pkm2 *= bigInv; pkm1 *= bigInv; qkm2 *= bigInv; qkm1 *= bigInv;
            }
            if (t <= MachEp) break;
        }
        return ans * ax;
    }

    /// <summary>Inverse of <see cref="GammaInc"/> in x: P(a, x) = p (scipy.special.gammaincinv).</summary>
    /// <remarks>
    /// Wilson–Hilferty / small-a starting values (DiDonato &amp; Morris style, as in Numerical Recipes
    /// invgammp) refined by safeguarded Halley iteration on whichever of P and Q is below one half,
    /// so both tails converge to full relative accuracy.
    /// </remarks>
    public static double GammaIncInv(double a, double p)
    {
        if (double.IsNaN(a) || double.IsNaN(p) || a < 0 || p < 0 || p > 1) return double.NaN;
        if (p == 0) return 0.0;
        if (p == 1) return double.PositiveInfinity;
        if (a == 0) return 0.0;
        return InverseGammaInc(a, p, 1.0 - p);
    }

    /// <summary>Inverse of <see cref="GammaIncC"/> in x: Q(a, x) = q (scipy.special.gammainccinv).</summary>
    public static double GammaIncCInv(double a, double q)
    {
        if (double.IsNaN(a) || double.IsNaN(q) || a < 0 || q < 0 || q > 1) return double.NaN;
        if (q == 0) return double.PositiveInfinity;
        if (q == 1) return 0.0;
        if (a == 0) return 0.0;
        return InverseGammaInc(a, 1.0 - q, q);
    }

    private static double InverseGammaInc(double a, double p, double q)
    {
        // Work with the smaller tail; the residual is taken in logarithms, so a target of 1e-300
        // converges as quickly as one of 0.3 and a start in the wrong decade costs a few steps.
        var usep = p <= q;
        var target = usep ? p : q;
        var logTarget = Math.Log(target);
        double x;
        if (a > 1)
        {
            var t = Math.Sqrt(-2 * logTarget);
            var z = (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t;
            if (usep) z = -z; // z is now the standard-normal quantile of P
            var c = 1 - 1 / (9 * a) + z / (3 * Math.Sqrt(a));
            x = c > 0 ? a * c * c * c : 0;
            // Lower tail: the leading term of the series, P ≈ x^a e^-x / Γ(a+1).
            if (usep && (x <= 0 || x < 0.2 * a))
                x = Math.Exp((logTarget + GammaLn(a + 1)) / a);
            if (!(x > 0)) x = a;
        }
        else
        {
            var t = 1 - a * (0.253 + a * 0.12);
            x = p < t ? Math.Exp((Math.Log(p / t)) / a) : 1 - Math.Log(q / (1 - t));
            if (!(x > 0)) x = double.Epsilon;
        }
        double lo = 0, hi = double.PositiveInfinity;
        for (var j = 0; j < 300; j++)
        {
            var P = GammaInc(a, x);
            var Q = GammaIncC(a, x);
            var cur = usep ? P : Q;
            if (cur == 0)
            {
                // Underflowed: too far into the tail being matched.
                if (usep) { lo = x; x = double.IsPositiveInfinity(hi) ? 2 * x : 0.5 * (x + hi); }
                else { hi = x; x = 0.5 * (lo + x); }
                continue;
            }
            var g = Math.Log(cur) - logTarget; // increasing in x for P, decreasing for Q
            var tooBig = usep ? g > 0 : g < 0;
            if (tooBig) hi = Math.Min(hi, x); else lo = Math.Max(lo, x);
            if (g == 0) break;
            var dens = IgamFac(a, x) / x; // dP/dx
            double xn;
            if (dens > 0 && double.IsFinite(dens))
            {
                // d log(cur)/dx = ±dens/cur. The lower tail is solved in log x (log P ≈ a log x
                // there), the upper in x; both steps are limited so a far start cannot overshoot.
                var slope = (usep ? dens : -dens) / cur;
                if (usep)
                {
                    var dt = Math.Clamp(-g / (x * slope), -5.0, 5.0);
                    xn = x * Math.Exp(dt);
                }
                else
                {
                    xn = x - Math.Clamp(g / slope, -Math.Max(x, 10.0), Math.Max(x, 10.0));
                }
            }
            else xn = double.NaN;
            if (!(xn > lo && xn < hi))
                xn = double.IsPositiveInfinity(hi) ? Math.Max(2 * x, lo + 1) : (lo > 0 ? Math.Sqrt(lo * hi) : 0.5 * hi);
            if (Math.Abs(xn - x) <= 4e-16 * x) { x = xn; break; }
            x = xn;
        }
        return x;
    }

    // ---------------------------------------------------------------- incomplete beta

    /// <summary>The regularised incomplete beta function I_x(a, b) (scipy.special.betainc).</summary>
    /// <remarks>
    /// Lentz evaluation of the classical continued fraction on whichever side of the mean it
    /// converges (x &lt; (a + 1)/(a + b + 2), else via symmetry), with the prefactor
    /// x^a (1-x)^b / B(a, b) computed in Lanczos form as Boost's ibeta_power_terms does, so large
    /// parameters do not lose digits to lgamma cancellation. The fraction needs O(√max(a, b))
    /// terms. a = 0 or b = 0 follow SciPy's limits.
    /// </remarks>
    public static double BetaInc(double a, double b, double x) => IncompleteBeta(a, b, x, 1.0 - x).I;

    /// <summary>The complement 1 - I_x(a, b), computed without cancellation (scipy.special.betaincc).</summary>
    public static double BetaIncC(double a, double b, double x) => IncompleteBeta(a, b, x, 1.0 - x).Ic;

    /// <summary>I_x(a, b) and its complement, with x and y = 1 - x supplied separately so neither is rounded.</summary>
    internal static (double I, double Ic) IncompleteBeta(double a, double b, double x, double y)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(x) || double.IsNaN(y)) return (double.NaN, double.NaN);
        if (a < 0 || b < 0 || x < 0 || x > 1) return (double.NaN, double.NaN);
        if (a == 0 && b == 0) return (double.NaN, double.NaN);
        if (a == 0) return (1.0, 0.0);
        if (b == 0) return (0.0, 1.0);
        if (x == 0) return (0.0, 1.0);
        if (y == 0 || x == 1) return (1.0, 0.0);
        if (double.IsPositiveInfinity(a) || double.IsPositiveInfinity(b))
        {
            if (double.IsPositiveInfinity(a) && double.IsPositiveInfinity(b)) return (double.NaN, double.NaN);
            return double.IsPositiveInfinity(a) ? (0.0, 1.0) : (1.0, 0.0);
        }
        if (x > (a + 1) / (a + b + 2))
        {
            var (i2, ic2) = IncompleteBeta(b, a, y, x);
            return (ic2, i2);
        }
        var pre = BetaPowerTerms(a, b, x, y);
        if (pre == 0) return (0.0, 1.0);
        var I = pre / BetaFraction2(a, b, x, y);
        if (I > 0.5)
        {
            // The complement would come out as 1 - I; try the fraction for it directly. Past the
            // mean it may not converge, in which case the subtraction stands.
            var pre2 = BetaPowerTerms(b, a, y, x);
            var f2 = BetaFraction2(b, a, y, x, out var converged);
            if (converged && pre2 > 0)
            {
                var ic = pre2 / f2;
                if (ic >= 0 && ic < 0.5) return (1.0 - ic, ic);
            }
        }
        return (I, 1.0 - I);
    }

    /// <summary>x^a y^b / B(a, b) in Lanczos form (Boost ibeta_power_terms).</summary>
    internal static double BetaPowerTerms(double a, double b, double x, double y)
    {
        var c = a + b;
        if (c < 100 && a > 1e-3 && b > 1e-3 && a < 100 && b < 100)
        {
            var la = a * Math.Log(x) + b * Math.Log(y);
            if (la > -600 && la < 600)
            {
                var direct = Math.Pow(x, a) * Math.Pow(y, b) / Beta(a, b);
                if (double.IsFinite(direct) && direct > 1e-290) return direct;
            }
        }
        const double gh = LanczosG - 0.5;
        var agh = a + gh;
        var bgh = b + gh;
        var cgh = c + gh;
        var lr = LanczosSumExpgScaled(c) / (LanczosSumExpgScaled(a) * LanczosSumExpgScaled(b));
        // d1 = x cgh/agh - 1 and d2 = y cgh/bgh - 1, written in terms of the smaller of x and y
        // only. x and y arrive separately rounded, and with a ~ 5000 the inconsistency
        // x + y - 1 ~ 1e-16 alone would move a·log1p(d) by 5e-13.
        double d1, d2;
        if (x <= y)
        {
            d1 = (x * cgh - agh) / agh;
            d2 = (a - x * cgh) / bgh;
        }
        else
        {
            d1 = (b - y * cgh) / agh;
            d2 = (y * cgh - bgh) / bgh;
        }
        var scale = lr * Math.Sqrt(agh / cgh * bgh);
        // Near the mean the d-forms are exact enough. In a tail, a·log(x cgh/agh) would carry
        // a·ulp(log x) of error (1e-13 at a = 300); instead the small variable is raised with
        // Math.Pow, which is accurate to an ulp whatever the exponent, the one near 1 with
        // log1p of the small one, and the Lanczos ratio (cgh/agh = 1 + b/agh) with log1p.
        double e = -0.5, f = 1.0;
        var xs = x <= y;
        if (Math.Abs(d1) < 0.5) e += a * Log1p(d1);
        else if (xs && a * Math.Log(x) > -700) { f *= Math.Pow(x, a); e += a * Log1p(b / agh); }
        else e += a * ((xs ? Math.Log(x) : Log1p(-y)) + Log1p(b / agh));
        if (Math.Abs(d2) < 0.5) e += b * Log1p(d2);
        else if (!xs && b * Math.Log(y) > -700) { f *= Math.Pow(y, b); e += b * Log1p(a / bgh); }
        else e += b * ((xs ? Log1p(-x) : Math.Log(y)) + Log1p(a / bgh));
        if (f == 0 || !double.IsFinite(f)) return 0.0;
        var lf = Math.Log(f);
        if (e + lf < MinLog - 50) return 0.0;
        if (Math.Abs(e) > 700) return scale * Math.Exp(e + lf);
        return scale * f * Math.Exp(e);
    }

    /// <summary>
    /// The continued fraction of DiDonato &amp; Morris (TOMS 708, as Boost's ibeta_fraction2):
    /// I_x(a, b) = x^a y^b / (B(a, b) · F). Unlike the classical Numerical Recipes fraction its
    /// partial terms use y = 1 - x as given, and near the mean with a in the thousands it keeps
    /// ~1e-15 where the classical one drifts to 3e-13.
    /// </summary>
    private static double BetaFraction2(double a, double b, double x, double y) => BetaFraction2(a, b, x, y, out _);

    private static double BetaFraction2(double a, double b, double x, double y, out bool converged)
    {
        converged = false;
        const double tiny = 1e-300;
        // m = 0 term
        var f = a * (a * y - b * x + 1) / (a + 1);
        if (f == 0) f = tiny;
        var c = f;
        var d = 0.0;
        var maxIt = 1000 + (int)Math.Min(1e7, 50 * Math.Sqrt(Math.Max(a, b)));
        for (var m = 1; m <= maxIt; m++)
        {
            var denom = a + 2 * m - 1;
            var an = (a + m - 1) * (a + b + m - 1) * m * (b - m) * x * x / (denom * denom);
            var bn = m + m * (b - m) * x / (a + 2 * m - 1) + (a + m) * (a * y - b * x + 1 + m * (2 - x)) / (a + 2 * m + 1);
            d = bn + an * d;
            if (d == 0) d = tiny;
            c = bn + an / c;
            if (c == 0) c = tiny;
            d = 1.0 / d;
            var delta = c * d;
            f *= delta;
            if (Math.Abs(delta - 1.0) <= 3e-16) { converged = true; break; }
        }
        return f;
    }

    /// <summary>Inverse of <see cref="BetaInc"/> in x: I_x(a, b) = y (scipy.special.betaincinv).</summary>
    /// <remarks>
    /// Numerical Recipes / AS 109 starting value, then safeguarded Halley iteration on the
    /// incomplete beta, solving for 1 - x through the symmetry relation when y &gt; 1/2 so the upper
    /// tail keeps its relative accuracy.
    /// </remarks>
    public static double BetaIncInv(double a, double b, double y)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(y) || a <= 0 || b <= 0 || y < 0 || y > 1) return double.NaN;
        return InverseIncompleteBeta(a, b, y, 1.0 - y).X;
    }

    /// <summary>Inverse of <see cref="BetaIncC"/> in x: 1 - I_x(a, b) = y (scipy.special.betainccinv).</summary>
    public static double BetaIncCInv(double a, double b, double y)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(y) || a <= 0 || b <= 0 || y < 0 || y > 1) return double.NaN;
        return InverseIncompleteBeta(a, b, 1.0 - y, y).X;
    }

    /// <summary>Solves I_x(a, b) = p (with q = 1 - p given exactly); returns x and 1 - x.</summary>
    /// <remarks>
    /// Iterates on whichever of x and 1 - x is below one half (decided from the starting value),
    /// matching whichever of p and q is smaller, so neither the variable nor the residual is ever
    /// formed as 1 minus something close to 1.
    /// </remarks>
    internal static (double X, double Y) InverseIncompleteBeta(double a, double b, double p, double q)
    {
        if (p <= 0) return (0.0, 1.0);
        if (q <= 0) return (1.0, 0.0);
        var (x0, y0) = IncompleteBetaInverseGuess(a, b, p, q);
        if (y0 < x0)
        {
            // The answer is nearer 1: iterate on 1 - x, whose guess was formed directly (it can be
            // far below the ulp of 1, and then x itself correctly rounds to 1).
            var (yy, xx) = SolveIncompleteBeta(b, a, q, p, y0);
            return (xx, yy);
        }
        return SolveIncompleteBeta(a, b, p, q, x0);
    }

    /// <summary>Numerical Recipes (invbetai) starting value as (x, 1 - x), each formed directly.</summary>
    private static (double X, double Y) IncompleteBetaInverseGuess(double a, double b, double p, double q)
    {
        double x, y;
        if (a >= 1 && b >= 1)
        {
            var t = Math.Sqrt(-2 * Math.Log(Math.Min(p, q)));
            var z = (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t;
            if (p < q) z = -z; // NR sign convention: large positive z for a small lower tail
            var al = (z * z - 3) / 6;
            var h = 2 / (1 / (2 * a - 1) + 1 / (2 * b - 1));
            var w = z * Math.Sqrt(al + h) / h - (1 / (2 * b - 1) - 1 / (2 * a - 1)) * (al + 5.0 / 6 - 2 / (3 * h));
            var e = b * Math.Exp(2 * w);
            x = a / (a + e);
            y = e / (a + e);
        }
        else
        {
            var lna = Math.Log(a / (a + b));
            var lnb = Math.Log(b / (a + b));
            var t = Math.Exp(a * lna) / a;
            var u = Math.Exp(b * lnb) / b;
            var w = t + u;
            if (p < t / w)
            {
                x = Math.Exp((Math.Log(a * w) + Math.Log(p)) / a);
                y = 1 - x;
            }
            else
            {
                y = Math.Exp((Math.Log(b * w) + Math.Log(q)) / b);
                x = 1 - y;
            }
        }
        if (!(x > 0 && y > 0) || double.IsNaN(x) || double.IsNaN(y)) { x = a / (a + b); y = b / (a + b); }
        return (x, y);
    }

    private static (double X, double Y) SolveIncompleteBeta(double a, double b, double p, double q, double x)
    {
        var usep = p <= q;
        var target = usep ? p : q;
        double lo = 0, hi = 1;
        if (!(x > 0 && x < 1)) x = 0.5;
        for (var j = 0; j < 300; j++)
        {
            var y = 1 - x;
            var (I, Ic) = IncompleteBeta(a, b, x, y);
            var err = usep ? I - p : q - Ic; // increasing in x
            if (err > 0) hi = Math.Min(hi, x); else lo = Math.Max(lo, x);
            if (err == 0) break;
            var dens = BetaPowerTerms(a, b, x, y) / (x * y);
            double xn;
            if (dens > 0 && double.IsFinite(dens))
            {
                var u = err / dens;
                var corr = u * ((a - 1) / x - (b - 1) / y);
                xn = x - u / (1 - 0.5 * Math.Max(-1.0, Math.Min(1.0, corr)));
            }
            else xn = double.NaN;
            if (!(xn > lo && xn < hi))
                xn = lo > 0 ? Math.Sqrt(lo * hi) : 0.5 * (lo + hi) * (target < 1e-10 ? 1e-3 : 1);
            if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
            if (Math.Abs(xn - x) <= 2e-16 * x) { x = xn; break; }
            x = xn;
        }
        return (x, 1 - x);
    }
}
