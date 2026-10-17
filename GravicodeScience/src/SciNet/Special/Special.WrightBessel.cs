namespace Gravicode.Science.SciNet;

public static partial class Special
{
    /// <summary>
    /// Wright's generalised Bessel function Φ(a, b; x) = Σ_k x^k / (k! Γ(a k + b))
    /// (scipy.special.wright_bessel), for a ≥ 0, b ≥ 0, x ≥ 0.
    /// </summary>
    /// <remarks>
    /// On that domain every term is non-negative and log-concave in k, so the series is summed
    /// outward from its largest term — no cancellation, and no overflow until the result itself
    /// overflows. scipy instead switches between the series, asymptotic expansions and an integral
    /// representation; the values agree to ~1e-13 relative for moderate arguments. The cost grows
    /// like √(peak index), and accuracy degrades slowly (as ε·log Γ of the peak index) when the
    /// peak lies beyond ~10⁶ terms.
    /// </remarks>
    public static double WrightBessel(double a, double b, double x)
    {
        var (lm, s) = WrightBesselCore(a, b, x);
        if (double.IsNaN(lm)) return double.NaN;
        if (double.IsNegativeInfinity(lm)) return 0.0;
        return Math.Exp(lm) * s;
    }

    /// <summary>log Φ(a, b; x) (scipy.special.log_wright_bessel), finite where Φ itself would overflow.</summary>
    public static double LogWrightBessel(double a, double b, double x)
    {
        var (lm, s) = WrightBesselCore(a, b, x);
        if (double.IsNaN(lm)) return double.NaN;
        if (double.IsNegativeInfinity(lm)) return double.NegativeInfinity;
        return lm + Math.Log(s);
    }

    /// <summary>Returns (log of the peak term, sum of terms relative to the peak).</summary>
    private static (double LogPeak, double Sum) WrightBesselCore(double a, double b, double x)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(x) || a < 0 || b < 0 || x < 0) return (double.NaN, 0);
        if (double.IsInfinity(a) || double.IsInfinity(b)) return (double.NaN, 0);
        if (double.IsPositiveInfinity(x)) return (double.PositiveInfinity, 1);
        if (x == 0) return b == 0 ? (double.NegativeInfinity, 0) : (-GammaLn(b), 1);
        if (a == 0 && b == 0) return (double.NegativeInfinity, 0);
        var lx = Math.Log(x);
        double LogTerm(double k)
        {
            var arg = a * k + b;
            if (arg == 0) return double.NegativeInfinity; // 1/Γ(0) = 0
            return k * lx - GammaLn(k + 1) - GammaLn(arg);
        }
        // Peak of the concave log-term: the root of ln x − ψ(k+1) − a ψ(ak+b) in k ≥ 0.
        double Slope(double k) => lx - Psi(k + 1) - (a * k + b > 0 ? a * Psi(a * k + b) : double.PositiveInfinity);
        long peak = 0;
        var k0 = b == 0 ? 1.0 : 0.0;
        if (Slope(k0) > 0)
        {
            var lo = k0;
            var hi = Math.Max(1.0, 2 * k0);
            while (Slope(hi) > 0 && hi < 1e15) { lo = hi; hi *= 2; }
            for (var it = 0; it < 200 && hi - lo > 0.5; it++)
            {
                var mid = 0.5 * (lo + hi);
                if (Slope(mid) > 0) lo = mid; else hi = mid;
            }
            peak = (long)Math.Floor(lo);
            if (LogTerm(peak + 1) > LogTerm(peak)) peak++;
        }
        else peak = (long)k0;
        var lp = LogTerm(peak);
        double sum = 1.0;
        // Upward and downward from the peak; each side is a decreasing sequence.
        for (var k = peak + 1; ; k++)
        {
            var t = Math.Exp(LogTerm(k) - lp);
            sum += t;
            if (t < 1e-17 * sum || k - peak > 100_000_000) break;
        }
        for (var k = peak - 1; k >= 0; k--)
        {
            var t = Math.Exp(LogTerm(k) - lp);
            sum += t;
            if (t < 1e-17 * sum) break;
        }
        return (lp, sum);
    }
}
