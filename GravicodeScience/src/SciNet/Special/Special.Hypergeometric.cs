namespace Gravicode.Science.SciNet;

public static partial class Special
{
    /// <summary>The confluent hypergeometric limit function ₀F₁(; b; x) (scipy.special.hyp0f1).</summary>
    /// <remarks>
    /// Power series for x ≥ 0 (all terms positive) and for -1 ≤ x &lt; 0; below -1 the alternating
    /// series cancels, so Γ(b) |x|^((1-b)/2) J_(b-1)(2√|x|) is used instead. b a non-positive
    /// integer gives ∞ (NaN at x = 0 would be 1/Γ-style ambiguous; SciPy returns ∞).
    /// </remarks>
    public static double Hyp0f1(double b, double x)
    {
        if (double.IsNaN(b) || double.IsNaN(x)) return double.NaN;
        if (IsNonPositiveInteger(b)) return double.PositiveInfinity;
        if (x == 0) return 1.0;
        if (x >= -1.0 && x < 700)
        {
            double term = 1, sum = 1;
            for (var k = 0; k < 100000; k++)
            {
                term *= x / ((b + k) * (k + 1));
                sum += term;
                if (Math.Abs(term) <= 1e-17 * Math.Abs(sum)) break;
                if (!double.IsFinite(sum)) break;
            }
            if (double.IsFinite(sum) || x < 0) return sum;
        }
        if (x < 0)
        {
            var s = Math.Sqrt(-x);
            return Gamma(b) * Math.Pow(s, 1 - b) * Jv(b - 1, 2 * s);
        }
        {
            var s = Math.Sqrt(x);
            // Γ(b) x^((1-b)/2) I_(b-1)(2√x), in logs because both factors overflow separately.
            var ive = Ive(b - 1, 2 * s);
            var lg = GammaLn(b) + (1 - b) * Math.Log(s) + 2 * s + Math.Log(Math.Abs(ive));
            return GammaSgn(b) * Math.Sign(ive) * Math.Exp(lg);
        }
    }

    /// <summary>Kummer's confluent hypergeometric function M(a, b, x) = ₁F₁(a; b; x) (scipy.special.hyp1f1).</summary>
    /// <remarks>
    /// Negative x is moved to positive x by Kummer's transformation M(a, b, x) = eˣ M(b - a, b, -x),
    /// so the series that is summed has terms of one sign once k &gt; -a; large x (beyond 700, where
    /// eˣ overflows on its own) uses the asymptotic expansion DLMF 13.7.2 in logarithms.
    /// Terminating cases (a a non-positive integer) are summed exactly. Accuracy degrades when
    /// |a| or |b| are large compared with x and the terms cancel; that is the region where SciPy
    /// switches to Boost's much larger machinery.
    /// </remarks>
    public static double Hyp1f1(double a, double b, double x)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(x)) return double.NaN;
        if (IsNonPositiveInteger(b) && !(IsNonPositiveInteger(a) && a > b)) return double.PositiveInfinity;
        if (a == 0 || x == 0) return 1.0;
        if (a == b) return Math.Exp(x);
        if (IsNonPositiveInteger(a)) return Hyp1f1Series(a, b, x);
        if (x < 0)
        {
            if (IsNonPositiveInteger(b - a)) return Math.Exp(x) * Hyp1f1Series(b - a, b, -x);
            if (x > -700)
                return Math.Exp(x) * Hyp1f1(b - a, b, -x);
            // e^x M(b-a, b, -x) with M large: the asymptotic form of M(b-a,b,y) ~ Γ(b)/Γ(b-a) e^y y^(-a) + Γ(b)/Γ(a) (-y)^(a-b)...
            // For x → -∞ only the algebraic part survives: Γ(b)/Γ(b-a) (-x)^(-a).
            var y = -x;
            var sum = AsymptoticSum(a, 1 + a - b, 1.0 / y);
            return Gamma(b) * RGamma(b - a) * Math.Pow(y, -a) * sum;
        }
        if (x < 700) return Hyp1f1Series(a, b, x);
        var s2 = AsymptoticSum(b - a, 1 - a, 1.0 / x);
        var lg = GammaLn(b) - GammaLn(a) + x + (a - b) * Math.Log(x);
        return GammaSgn(b) * GammaSgn(a) * Math.Exp(lg) * s2;
    }

    private static double Hyp1f1Series(double a, double b, double x)
    {
        double term = 1, sum = 1;
        for (var k = 0; k < 100000; k++)
        {
            term *= (a + k) / (b + k) * x / (k + 1);
            sum += term;
            if (term == 0) break;
            if (Math.Abs(term) <= 1e-17 * Math.Abs(sum) && k > Math.Abs(a)) break;
            if (!double.IsFinite(sum)) break;
        }
        return sum;
    }

    /// <summary>Σ (p)_k (q)_k / k! z^k, truncated at its smallest term (asymptotic series).</summary>
    private static double AsymptoticSum(double p, double q, double z)
    {
        double term = 1, sum = 1, last = double.PositiveInfinity;
        for (var k = 0; k < 500; k++)
        {
            var next = term * (p + k) * (q + k) / (k + 1) * z;
            if (Math.Abs(next) >= last) break;
            last = Math.Abs(next);
            term = next;
            sum += term;
            if (Math.Abs(term) <= 1e-17 * Math.Abs(sum)) break;
        }
        return sum;
    }

    /// <summary>Gauss's hypergeometric function ₂F₁(a, b; c; z) for real z ≤ 1 (scipy.special.hyp2f1).</summary>
    /// <remarks>
    /// |z| ≤ 1/2: the defining series. 1/2 &lt; z &lt; 1: the z → 1 - z connection formula (DLMF 15.8.4),
    /// with the logarithmic forms of Abramowitz &amp; Stegun 15.3.10–11 when c - a - b is an integer.
    /// z &lt; -1/2: Pfaff's transformation to z/(z - 1) ∈ (1/3, 1), then the above. z = 1 uses Gauss's
    /// theorem. Polynomial cases sum exactly for any z. z &gt; 1 is complex-valued and returns NaN.
    /// When c - a - b is within ~1e-8 of an integer without being one, the connection formula
    /// cancels and loses roughly as many digits as that distance has.
    /// </remarks>
    public static double Hyp2f1(double a, double b, double c, double z)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(c) || double.IsNaN(z)) return double.NaN;
        if (z == 0) return 1.0;
        var aPoly = IsNonPositiveInteger(a);
        var bPoly = IsNonPositiveInteger(b);
        if (IsNonPositiveInteger(c))
        {
            var ok = (aPoly && a > c) || (bPoly && b > c);
            if (!ok) return double.PositiveInfinity;
        }
        if (a == 0 || b == 0) return 1.0;
        if (aPoly || bPoly)
        {
            var n = aPoly && bPoly ? Math.Max(a, b) : (aPoly ? a : b);
            return Hyp2f1Polynomial(a, b, c, z, (int)(-n));
        }
        if (z > 1) return double.NaN;
        if (z == 1)
        {
            var s = c - a - b;
            if (s <= 0) return double.PositiveInfinity;
            return Math.Exp(GammaLn(c) + GammaLn(s) - GammaLn(c - a) - GammaLn(c - b))
                   * GammaSgn(c) * GammaSgn(s) * GammaSgn(c - a) * GammaSgn(c - b);
        }
        if (Math.Abs(z) <= 0.5) return Hyp2f1Series(a, b, c, z);
        if (z < -0.5)
        {
            var w = z / (z - 1);
            // Pfaff: (1-z)^-a F(a, c-b; c; w) or (1-z)^-b F(c-a, b; c; w); prefer a terminating one.
            if (IsNonPositiveInteger(c - a) && !IsNonPositiveInteger(c - b))
                return Math.Pow(1 - z, -b) * Hyp2f1(c - a, b, c, w);
            return Math.Pow(1 - z, -a) * Hyp2f1(a, c - b, c, w);
        }
        return Hyp2f1OneMinusZ(a, b, c, z);
    }

    private static double Hyp2f1Polynomial(double a, double b, double c, double z, int n)
    {
        double term = 1, sum = 1;
        for (var k = 0; k < n; k++)
        {
            term *= (a + k) * (b + k) / ((c + k) * (k + 1)) * z;
            sum += term;
        }
        return sum;
    }

    private static double Hyp2f1Series(double a, double b, double c, double z)
    {
        double term = 1, sum = 1;
        for (var k = 0; k < 200000; k++)
        {
            term *= (a + k) * (b + k) / ((c + k) * (k + 1)) * z;
            sum += term;
            if (Math.Abs(term) <= 1e-17 * Math.Abs(sum) && k > Math.Max(Math.Abs(a), Math.Abs(b))) break;
            if (term == 0 || !double.IsFinite(sum)) break;
        }
        return sum;
    }

    private static double Hyp2f1OneMinusZ(double a, double b, double c, double z)
    {
        var y = 1 - z; // exact for z in [1/2, 1]
        var m = c - a - b;
        var mr = Math.Round(m);
        // c - a - b that differs from an integer only by the rounding of "nice" decimal inputs
        // (2.3 + 0.7 is not 3) is treated as the integer: the connection formula would divide
        // Γ(m)Γ(-m) ~ 1e16 terms into each other, and moving c by 1e-15 changes F by far less.
        if (Math.Abs(m - mr) < 1e-13 * Math.Max(1.0, Math.Abs(c))) m = mr;
        if (m != mr)
        {
            var t1 = Gamma(c) * Gamma(m) * RGamma(c - a) * RGamma(c - b);
            var t2 = Gamma(c) * Gamma(-m) * RGamma(a) * RGamma(b);
            var f1 = t1 == 0 ? 0 : Hyp2f1Series(a, b, 1 - m, y);
            var f2 = t2 == 0 ? 0 : Hyp2f1Series(c - a, c - b, 1 + m, y);
            if (double.IsFinite(t1) && double.IsFinite(t2))
                return t1 * f1 + Math.Pow(y, m) * t2 * f2;
            // Large parameters: combine in logarithms.
            double Part(double g1, double g2, double ga, double gb, double f, double pw)
            {
                if (f == 0) return 0;
                var lg = GammaLn(g1) + GammaLn(g2) - GammaLn(ga) - GammaLn(gb) + pw + Math.Log(Math.Abs(f));
                return GammaSgn(g1) * GammaSgn(g2) * GammaSgn(ga) * GammaSgn(gb) * Math.Sign(f) * Math.Exp(lg);
            }
            return Part(c, m, c - a, c - b, f1, 0) + Part(c, -m, a, b, f2, m * Math.Log(y));
        }
        var mi = (int)mr;
        if (mi < 0)
        {
            // Euler: F(a,b;c;z) = (1-z)^(c-a-b) F(c-a, c-b; c; z), whose c-a-b is -m > 0.
            return Math.Pow(y, m) * Hyp2f1Log(c - a, c - b, -mi, y);
        }
        return Hyp2f1Log(a, b, mi, y);
    }

    /// <summary>F(a, b; a + b + m; 1 - y) for integer m ≥ 0 (A&amp;S 15.3.10–11).</summary>
    private static double Hyp2f1Log(double a, double b, int mi, double y)
    {
        // A&S 15.3.10 (m = 0) and 15.3.11 (m ≥ 1).
        var lny = Math.Log(y);
        double finite = 0;
        if (mi > 0)
        {
            var pre = Gamma(mi) * Gamma(a + b + mi) * RGamma(a + mi) * RGamma(b + mi);
            double term = 1, sum = 1;
            for (var k = 0; k < mi - 1; k++)
            {
                term *= (a + k) * (b + k) / ((k + 1) * (1 - mi + k)) * y;
                sum += term;
            }
            finite = pre * sum;
        }
        var pre2 = Gamma(a + b + mi) * RGamma(a) * RGamma(b);
        if (pre2 == 0) return finite;
        // Σ_n (a+m)_n (b+m)_n / (n! (n+m)!) y^n [ln y - ψ(n+1) - ψ(n+m+1) + ψ(a+n+m) + ψ(b+n+m)]
        double coef = 1.0 / Gamma(mi + 1.0), s = 0;
        var psi1 = -EulerGamma;               // ψ(n+1) at n = 0
        var psi2 = Digamma(mi + 1.0);         // ψ(n+m+1)
        var psi3 = Digamma(a + mi);           // ψ(a+n+m)
        var psi4 = Digamma(b + mi);           // ψ(b+n+m)
        for (var n = 0; n < 100000; n++)
        {
            var t = coef * (lny - psi1 - psi2 + psi3 + psi4);
            s += t;
            if (Math.Abs(t) <= 1e-17 * Math.Abs(s) && n > 5) break;
            coef *= (a + mi + n) * (b + mi + n) / ((n + 1.0) * (n + mi + 1.0)) * y;
            psi1 += 1.0 / (n + 1);
            psi2 += 1.0 / (n + mi + 1);
            psi3 += 1.0 / (a + mi + n);
            psi4 += 1.0 / (b + mi + n);
            if (coef == 0) break;
        }
        var sign = (mi % 2 == 0) ? 1.0 : -1.0; // (z-1)^m = (-1)^m y^m
        return finite - sign * Math.Pow(y, mi) * pre2 * s;
    }
}
