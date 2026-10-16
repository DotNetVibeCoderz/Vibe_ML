using System.Globalization;
using System.Numerics;

namespace Gravicode.Science.SciNet;

public static partial class Special
{
    private static readonly double[] FactorialTable = BuildFactorialTable();

    private static double[] BuildFactorialTable()
    {
        // Correctly rounded n! for n = 0..170: exact BigInteger products, then a decimal round
        // trip (double.Parse is correctly rounded), so gamma(n + 1) is exact wherever n! is
        // representable and within half an ulp everywhere else.
        var t = new double[171];
        var f = BigInteger.One;
        t[0] = 1.0;
        for (var n = 1; n <= 170; n++)
        {
            f *= n;
            t[n] = double.Parse(f.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
        }
        return t;
    }

    /// <summary>The gamma function Γ(x) (scipy.special.gamma).</summary>
    /// <remarks>
    /// Integers up to 171 come from a correctly rounded factorial table. Otherwise 0 &lt; x &lt; 10
    /// uses Boost's 13-term Lanczos approximation with z + g - 1/2 carried as a double-double so
    /// the power does not amplify its rounding; x ≥ 10 uses Stirling's series with the exact
    /// argument in the power (Cephes' split power avoids overflow up to 171.6). Negative
    /// arguments use the reflection formula with an exactly reduced sin(πx). Relative error is a
    /// few ulp. Γ(±0) = ±∞, Γ(negative integer) = NaN, overflow above 171.62 gives +∞.
    /// </remarks>
    public static double Gamma(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (double.IsPositiveInfinity(x)) return double.PositiveInfinity;
        if (double.IsNegativeInfinity(x)) return double.NaN;
        if (x == 0.0) return double.IsNegative(x) ? double.NegativeInfinity : double.PositiveInfinity;
        if (x < 0)
        {
            if (x == Math.Floor(x)) return double.NaN;
            var s = SinPi(x);
            var oneMinus = 1.0 - x;
            if (oneMinus > 171.0)
            {
                // Γ(1-x) overflows; the result underflows towards zero with the right sign.
                var lg = LnPi - Math.Log(Math.Abs(s)) - GammaLn(oneMinus);
                return Math.Sign(s) * Math.Exp(lg);
            }
            return Math.PI / (s * Gamma(oneMinus));
        }
        if (x > 171.61447887182298) return double.PositiveInfinity;
        if (x == Math.Floor(x) && x <= 171) return FactorialTable[(int)x - 1];
        if (x >= 10.0) return GammaStirling(x);
        return GammaLanczos(x);
    }

    private static double GammaLanczos(double z)
    {
        var (hi, lo) = LanczosZgh(z);
        var e = z - 0.5;
        return LanczosSumExpgScaled(z) * Math.Pow(hi, e) * Math.Exp(e * (lo / hi) - e);
    }

    private static double GammaStirling(double x)
    {
        var s = StirlingTail(x);
        // x^(x - 1/2) e^-x, split so the power cannot overflow before the division.
        var v = Math.Pow(x, 0.5 * x - 0.25);
        var y = v * (v / Math.Exp(x));
        return 2.5066282746310005024 * y * Math.Exp(s);
    }

    /// <summary>The Stirling correction Σ B_2k / (2k(2k-1) x^(2k-1)), for x ≥ ~8.</summary>
    private static double StirlingTail(double x)
    {
        var w = 1.0 / x;
        var w2 = w * w;
        var c = SpecialTables.StirlingSeries;
        var p = w;
        var sum = 0.0;
        for (var k = 0; k < c.Length; k++)
        {
            var t = c[k] * p;
            sum += t;
            if (Math.Abs(t) < 1e-18 * Math.Abs(sum)) break;
            p *= w2;
        }
        return sum;
    }

    /// <summary>log|Γ(x)| (scipy.special.gammaln).</summary>
    /// <remarks>
    /// Relative accuracy is kept near the zeros at 1 and 2 by the Taylor series of log Γ(1 + z)
    /// (coefficients ζ(k)/k), and near 0 by log Γ(1 + x) - log x. Large x uses Stirling's series.
    /// Negative x uses the reflection formula, so near the zeros of log|Γ| on the negative axis
    /// accuracy is absolute rather than relative (as in SciPy). Poles give +∞.
    /// </remarks>
    public static double GammaLn(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (double.IsInfinity(x)) return double.PositiveInfinity;
        if (x <= 0 && x == Math.Floor(x)) return double.PositiveInfinity;
        if (x < 0)
        {
            if (x < -1e15) // sin(pi x) loses meaning; |Γ| underflows anyway.
                return double.NegativeInfinity;
            return LnPi - Math.Log(Math.Abs(SinPi(x))) - GammaLn(1.0 - x);
        }
        if (x < 0.25) return LogGamma1p(x) - Math.Log(x);
        if (Math.Abs(x - 1.0) <= 0.25) return LogGamma1p(x - 1.0);
        if (Math.Abs(x - 2.0) <= 0.25)
        {
            var z = x - 2.0;
            return Math.Log(1.0 + z) + LogGamma1p(z);
        }
        if (x < 10.0) return Math.Log(GammaLanczos(x));
        if (x > 1e300) return x * (Math.Log(x) - 1.0);
        return (x - 0.5) * Math.Log(x) - x + LnSqrt2Pi + StirlingTail(x);
    }

    /// <summary>log Γ(1 + z) for |z| ≤ 1/4 from its Taylor series (DLMF 5.7.3).</summary>
    private static double LogGamma1p(double z)
    {
        var c = SpecialTables.LogGammaTaylor1;
        // Horner from the highest coefficient that matters: |z|^k ≤ 4^-k, so 28 terms reach 1e-17.
        var n = Math.Abs(z) < 0.1 ? 18 : 30;
        var r = c[n - 1];
        for (var k = n - 2; k >= 0; k--) r = r * z + c[k];
        return r * z;
    }

    /// <summary>Sign of Γ(x) (scipy.special.gammasgn): 1, -1, or NaN at a negative integer.</summary>
    /// <remarks>gammasgn(+0) = 1 and gammasgn(-0) = -1, as in SciPy.</remarks>
    public static double GammaSgn(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (x > 0) return 1.0;
        if (x == 0) return double.IsNegative(x) ? -1.0 : 1.0;
        if (double.IsNegativeInfinity(x)) return double.NaN;
        var f = Math.Floor(x);
        if (f == x) return double.NaN;
        return ((-f) % 2.0 == 1.0) ? -1.0 : 1.0;
    }

    /// <summary>1/Γ(x) (scipy.special.rgamma); exactly zero at the poles 0, -1, -2, ...</summary>
    public static double RGamma(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (double.IsPositiveInfinity(x)) return 0.0;
        if (double.IsNegativeInfinity(x)) return double.NaN;
        if (IsNonPositiveInteger(x)) return 0.0;
        if (x > 171.6) return 0.0;
        if (x < 0)
        {
            var oneMinus = 1.0 - x;
            if (oneMinus > 171.0)
                return SinPi(x) / Math.PI * Math.Exp(GammaLn(oneMinus)); // overflows to ±inf, as it must
            return SinPi(x) * Gamma(oneMinus) / Math.PI;
        }
        return 1.0 / Gamma(x);
    }

    /// <summary>The digamma function ψ(x) = Γ'(x)/Γ(x) (scipy.special.psi / digamma).</summary>
    /// <remarks>
    /// Cephes psi: exact harmonic sums for small integers, recurrence up to x ≥ 10 and the
    /// asymptotic series beyond; reflection for x &lt; 0. Near the positive root
    /// x0 = 1.46163... a Taylor series in x - x0 (with x0 held to 34 digits) keeps relative
    /// accuracy where the shifted sum would cancel — the same remedy SciPy applies.
    /// ψ(+0) = -∞, ψ(-0) = +∞, ψ(negative integer) = NaN.
    /// </remarks>
    public static double Digamma(double x)
    {
        if (double.IsNaN(x)) return double.NaN;
        if (double.IsPositiveInfinity(x)) return double.PositiveInfinity;
        if (double.IsNegativeInfinity(x)) return double.NaN;
        if (x == 0) return double.IsNegative(x) ? double.PositiveInfinity : double.NegativeInfinity;
        if (x < 0)
        {
            if (x == Math.Floor(x)) return double.NaN;
            // ψ(x) = ψ(1 - x) - π / tan(πx), with tan from the exactly reduced fraction.
            var r = x - Math.Floor(x);
            return Digamma(1.0 - x) - Math.PI * CosPi(r) / SinPi(r);
        }
        var dx = (x - SpecialTables.DigammaRootHi) - SpecialTables.DigammaRootLo;
        if (Math.Abs(dx) < 0.3)
        {
            var c = SpecialTables.DigammaRootTaylor;
            var r = c[^1];
            for (var k = c.Length - 2; k >= 0; k--) r = r * dx + c[k];
            return r * dx;
        }
        if (x <= 10 && x == Math.Floor(x))
        {
            var n = (int)x;
            var s = 0.0;
            for (var k = 1; k < n; k++) s += 1.0 / k;
            return s - EulerGamma;
        }
        var result = 0.0;
        while (x < 10)
        {
            result -= 1.0 / x;
            x += 1.0;
        }
        var y = 0.0;
        if (x < 1e17)
        {
            var z = 1.0 / (x * x);
            y = z * Polevl(z, SpecialTables.PsiAsymptotic);
        }
        return result + Math.Log(x) - 0.5 / x - y;
    }

    /// <summary>Alias of <see cref="Digamma(double)"/> (scipy.special.psi).</summary>
    public static double Psi(double x) => Digamma(x);

    /// <summary>The polygamma function ψ⁽ⁿ⁾(x) (scipy.special.polygamma).</summary>
    /// <remarks>
    /// n = 0 is <see cref="Digamma(double)"/>; n ≥ 1 is (-1)^(n+1) n! ζ(n + 1, x) with Cephes' Hurwitz zeta,
    /// which also sums correctly for negative non-integer x. Throws for negative n.
    /// </remarks>
    public static double Polygamma(int n, double x)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Polygamma order must be non-negative.");
        if (n == 0) return Digamma(x);
        var sign = (n % 2 == 1) ? 1.0 : -1.0;
        return sign * Gamma(n + 1.0) * HurwitzZeta(n + 1.0, x);
    }

    /// <summary>The beta function B(a, b) = Γ(a)Γ(b)/Γ(a + b) (scipy.special.beta).</summary>
    /// <remarks>
    /// For positive arguments the Lanczos ratio is formed directly (Boost's beta_imp), so B stays
    /// accurate when one argument is huge and the other small — the case where
    /// exp(lgamma + lgamma - lgamma) cancels catastrophically. Other signs go through
    /// <see cref="GammaLn(double)"/> with the gamma signs; poles follow SciPy (B(-1, 2) = ∞, B(-1/2, -1/2) = 0).
    /// </remarks>
    public static double Beta(double a, double b)
    {
        if (double.IsNaN(a) || double.IsNaN(b)) return double.NaN;
        if (a > 0 && b > 0)
        {
            var lb = BetaLnPositive(a, b, out var direct);
            return direct ?? Math.Exp(lb);
        }
        if (IsNonPositiveInteger(a) || IsNonPositiveInteger(b))
        {
            if (b == 0 || a == 0) return double.PositiveInfinity;
            // Γ pole in the numerator; SciPy reports +∞ unless the denominator also has one.
            if (IsNonPositiveInteger(a + b) && !(IsNonPositiveInteger(a) && IsNonPositiveInteger(b))) return double.PositiveInfinity;
            return double.PositiveInfinity;
        }
        if (IsNonPositiveInteger(a + b)) return 0.0;
        var sign = GammaSgn(a) * GammaSgn(b) * GammaSgn(a + b);
        if (Math.Abs(a + b) < 171 && Math.Abs(a) < 171 && Math.Abs(b) < 171)
            return Gamma(a) * (Gamma(b) / Gamma(a + b));
        return sign * Math.Exp(GammaLn(a) + GammaLn(b) - GammaLn(a + b));
    }

    /// <summary>log|B(a, b)| (scipy.special.betaln).</summary>
    public static double BetaLn(double a, double b)
    {
        if (double.IsNaN(a) || double.IsNaN(b)) return double.NaN;
        if (a > 0 && b > 0)
        {
            var lb = BetaLnPositive(a, b, out var direct);
            return direct.HasValue ? Math.Log(direct.Value) : lb;
        }
        if (IsNonPositiveInteger(a) || IsNonPositiveInteger(b)) return double.PositiveInfinity;
        if (IsNonPositiveInteger(a + b)) return double.NegativeInfinity;
        return GammaLn(a) + GammaLn(b) - GammaLn(a + b);
    }

    /// <summary>
    /// log B(a, b) for a, b &gt; 0 via the Lanczos ratio; <paramref name="direct"/> is set when the
    /// value itself is cheaper and more accurate (small arguments, or a = 1 / b = 1).
    /// </summary>
    private static double BetaLnPositive(double a, double b, out double? direct)
    {
        direct = null;
        if (a < b) (a, b) = (b, a); // a ≥ b
        var c = a + b;
        if (b == 1.0) { direct = 1.0 / a; return -Math.Log(a); }
        if (c == a && b < MachEp) { direct = 1.0 / b; return -Math.Log(b); }
        if (c < 171.0 && a < 171.0)
        {
            var v = Gamma(a) * (Gamma(b) / Gamma(c));
            if (v > 1e-300 && double.IsFinite(v)) { direct = v; return Math.Log(v); }
        }
        if (a > 1e6 && b < 1e-6 * a && b < 10)
        {
            // Asymptotic: log B = lgamma(b) - b log a + log(1 + O(b^2/a)) (series of Γ(a)/Γ(a+b)).
            var r = GammaLn(b) - b * Math.Log(a);
            return r + Log1p(-b * (b - 1) / (2 * a));
        }
        var agh = a + LanczosG - 0.5;
        var bgh = b + LanczosG - 0.5;
        var cgh = c + LanczosG - 0.5;
        var res = Math.Log(LanczosSumExpgScaled(a)) + Math.Log(LanczosSumExpgScaled(b) / LanczosSumExpgScaled(c));
        var ambh = a - 0.5 - b;
        // (agh/cgh)^(a - 1/2 - b): agh/cgh = 1 - b/cgh, so use log1p when b is small against cgh.
        res += ambh * Log1p(-b / cgh);
        res += b * (Math.Log(agh / cgh) + Math.Log(bgh / cgh));
        res += 0.5 * (1.0 - Math.Log(bgh));
        return res;
    }

    /// <summary>The Pochhammer symbol (a)_m = Γ(a + m)/Γ(a) (scipy.special.poch).</summary>
    /// <remarks>
    /// SciPy's algorithm: integer parts of m are peeled off with the recurrence, a large a uses the
    /// asymptotic expansion of the gamma ratio, and the remainder uses log-gamma differences with
    /// signs. Poles give ∞ and zeros of 1/Γ give 0 exactly.
    /// </remarks>
    public static double Poch(double a, double m)
    {
        if (double.IsNaN(a) || double.IsNaN(m)) return double.NaN;
        var r = 1.0;
        while (m >= 1.0)
        {
            if (a + m == 1) break;
            m -= 1.0;
            r *= a + m;
            if (!double.IsFinite(r) || r == 0) break;
        }
        while (m <= -1.0)
        {
            if (a + m == 0) break;
            r /= a + m;
            m += 1.0;
            if (!double.IsFinite(r) || r == 0) break;
        }
        if (m == 0) return r;
        if (a > 1e4 && Math.Abs(m) <= 1)
        {
            return r * Math.Pow(a, m) * (1
                + m * (m - 1) / (2 * a)
                + m * (m - 1) * (m - 2) * (3 * m - 1) / (24 * a * a)
                + m * m * (m - 1) * (m - 1) * (m - 2) * (m - 3) / (48 * a * a * a));
        }
        if (IsNonPositiveInteger(a + m) && !IsNonPositiveInteger(a) && a + m != m) return double.PositiveInfinity;
        if (!IsNonPositiveInteger(a + m) && IsNonPositiveInteger(a)) return 0.0;
        return r * Math.Exp(GammaLn(a + m) - GammaLn(a)) * GammaSgn(a + m) * GammaSgn(a);
    }

    /// <summary>The binomial coefficient for real n and k (scipy.special.binom).</summary>
    /// <remarks>
    /// SciPy/Cephes algorithm: an exact multiplicative formula for integer k below 20 (with
    /// symmetry), asymptotic forms for k ≫ n and n ≫ k, otherwise 1/((n + 1) B(n - k + 1, k + 1)).
    /// A negative integer n gives NaN, as in SciPy.
    /// </remarks>
    public static double Binom(double n, double k)
    {
        if (double.IsNaN(n) || double.IsNaN(k)) return double.NaN;
        if (n < 0 && n == Math.Floor(n)) return double.NaN;
        var kx = Math.Floor(k);
        if (k == kx && (Math.Abs(n) > 1e-8 || n == 0))
        {
            var nx = Math.Floor(n);
            if (nx == n && kx > nx / 2 && nx > 0) kx = nx - kx;
            if (kx >= 0 && kx < 20)
            {
                double num = 1.0, den = 1.0;
                for (var i = 1; i < 1 + (int)kx; i++)
                {
                    num *= i + n - kx;
                    den *= i;
                    if (Math.Abs(num) > 1e50) { num /= den; den = 1.0; }
                }
                return num / den;
            }
        }
        if (n >= 1e10 * k && k > 0)
            return Math.Exp(-BetaLn(1 + n - k, 1 + k) - Math.Log(n + 1));
        if (k > 1e8 * Math.Abs(n))
        {
            var num = Gamma(1 + n) / Math.Abs(k) + Gamma(1 + n) * n / (2 * k * k);
            num /= Math.PI * Math.Pow(Math.Abs(k), n);
            if (k > 0)
            {
                kx = Math.Floor(k);
                double dk, sgn;
                if (kx == k && Math.Abs(kx) < 9e15) { dk = k - kx; sgn = ((long)kx % 2 == 0) ? 1 : -1; }
                else { dk = k; sgn = 1; }
                return num * Math.Sin((dk - n) * Math.PI) * sgn;
            }
            kx = Math.Floor(k);
            if (kx == k) return 0;
            return num * Math.Sin(k * Math.PI);
        }
        return 1 / (n + 1) / Beta(1 + n - k, 1 + k);
    }

    /// <summary>
    /// Number of combinations, N choose k, as a double (scipy.special.comb with exact=False).
    /// Zero when k &gt; N, N &lt; 0 or k &lt; 0. With <paramref name="repetition"/> it is
    /// comb(N + k - 1, k).
    /// </summary>
    public static double Comb(double n, double k, bool repetition = false)
    {
        if (repetition) return Comb(n + k - 1, k);
        if (k <= n && n >= 0 && k >= 0) return Binom(n, k);
        return 0.0;
    }

    /// <summary>Exact N choose k as a <see cref="BigInteger"/> (scipy.special.comb with exact=True).</summary>
    public static BigInteger CombExact(long n, long k, bool repetition = false)
    {
        if (repetition) return CombExact(n + k - 1, k);
        if (k > n || n < 0 || k < 0) return BigInteger.Zero;
        if (k > n - k) k = n - k;
        var r = BigInteger.One;
        for (long i = 1; i <= k; i++) r = r * (n - k + i) / i; // exact at every step
        return r;
    }

    /// <summary>Permutations of k items from N, N!/(N - k)! (scipy.special.perm with exact=False).</summary>
    public static double Perm(double n, double k)
    {
        if (k > n || n < 0 || k < 0) return 0.0;
        return Poch(n - k + 1, k);
    }

    /// <summary>Exact N!/(N - k)! (scipy.special.perm with exact=True).</summary>
    public static BigInteger PermExact(long n, long k)
    {
        if (k > n || n < 0 || k < 0) return BigInteger.Zero;
        var r = BigInteger.One;
        for (var i = n - k + 1; i <= n; i++) r *= i;
        return r;
    }

    /// <summary>n! = Γ(n + 1) for real n ≥ 0, and 0 for n &lt; 0 (scipy.special.factorial, exact=False).</summary>
    public static double Factorial(double n)
    {
        if (double.IsNaN(n)) return double.NaN;
        if (n < 0) return 0.0;
        return Gamma(n + 1);
    }

    /// <summary>Exact n! as a <see cref="BigInteger"/> (scipy.special.factorial, exact=True).</summary>
    public static BigInteger FactorialExact(int n)
    {
        if (n < 0) return BigInteger.Zero;
        var r = BigInteger.One;
        for (var i = 2; i <= n; i++) r *= i;
        return r;
    }

    /// <summary>
    /// Double factorial n!! = n (n - 2) (n - 4) ... (scipy.special.factorial2, exact=False):
    /// 1 for n = 0, 0 for negative n.
    /// </summary>
    /// <remarks>Uses the gamma-function form 2^(n/2) Γ(n/2 + 1) (× sqrt(2/π) for odd n), as SciPy does.</remarks>
    public static double Factorial2(int n)
    {
        if (n < 0) return 0.0;
        if (n == 0) return 1.0;
        if (n < 40)
        {
            var r = 1.0;
            for (var i = n; i > 1; i -= 2) r *= i;
            return r;
        }
        var h = n / 2.0;
        var v = Math.Pow(2.0, h) * Gamma(h + 1);
        return (n % 2 == 1) ? v * Math.Sqrt(2.0 / Math.PI) : v;
    }

    /// <summary>Exact double factorial (scipy.special.factorial2, exact=True).</summary>
    public static BigInteger Factorial2Exact(int n) => FactorialKExact(n, 2);

    /// <summary>
    /// Multifactorial n(!k) = n (n - k) (n - 2k) ... down to the first value ≥ 1
    /// (scipy.special.factorialk): 1 for n = 0, 0 for negative n.
    /// </summary>
    public static double FactorialK(int n, int k)
    {
        if (k < 1) throw new ArgumentOutOfRangeException(nameof(k), "k must be a positive integer.");
        if (n < 0) return 0.0;
        var r = 1.0;
        for (var i = n; i > 0; i -= k) r *= i;
        return r;
    }

    /// <summary>Exact multifactorial as a <see cref="BigInteger"/>.</summary>
    public static BigInteger FactorialKExact(int n, int k)
    {
        if (k < 1) throw new ArgumentOutOfRangeException(nameof(k), "k must be a positive integer.");
        if (n < 0) return BigInteger.Zero;
        var r = BigInteger.One;
        for (var i = n; i > 0; i -= k) r *= i;
        return r;
    }
}
