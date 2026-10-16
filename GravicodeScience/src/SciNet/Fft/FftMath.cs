using System.Numerics;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Number theory and trigonometry shared by the FFT plans: exact twiddle factors, factorisation,
/// and the smooth-number searches behind <see cref="Fft.NextFastLen"/>.
/// </summary>
internal static class FftMath
{
    /// <summary>
    /// <c>exp(-2πi m / n)</c>, with the argument reduced <em>exactly</em> before any rounding.
    /// </summary>
    /// <remarks>
    /// Computing <c>Math.SinCos(-2π m / n)</c> directly loses accuracy as the angle grows: the
    /// product <c>2π m</c> is rounded before the division, and for large <c>m</c> that rounding is
    /// several ulps of the result. Reducing <c>m/n</c> to the first octant in integer arithmetic
    /// (using the symmetries of sine and cosine) keeps every twiddle within about one ulp, which is
    /// what keeps the round-trip error of a 2^20 transform near 1e-16 rather than 1e-13.
    /// </remarks>
    public static Complex Twiddle(long m, long n)
    {
        m %= n;
        if (m < 0) m += n;
        // Work with x = m/n in [0, 1). The forward root is exp(-2πix) = cos(2πx) - i sin(2πx).
        var (c, s) = CosSinTurn(m, n);
        return new Complex(c, -s);
    }

    /// <summary>cos and sin of <c>2π p/q</c> for <c>0 ≤ p &lt; q</c>, octant-reduced.</summary>
    private static (double Cos, double Sin) CosSinTurn(long p, long q)
    {
        // Upper half: angle 2π - θ.
        if (2 * p > q)
        {
            var (c, s) = CosSinTurn(q - p, q);
            return (c, -s);
        }
        // Second quadrant: θ = π - φ with φ = 2π (q - 2p)/(2q).
        if (4 * p > q)
        {
            var (c, s) = CosSinFirstQuadrant(q - 2 * p, 2 * q);
            return (-c, s);
        }
        return CosSinFirstQuadrant(p, q);
    }

    /// <summary>cos and sin of <c>2π p/q</c> for <c>0 ≤ p/q ≤ 1/4</c>.</summary>
    private static (double Cos, double Sin) CosSinFirstQuadrant(long p, long q)
    {
        // Upper octant: θ = π/2 - φ, φ = 2π (q - 4p)/(4q).
        if (8 * p > q)
        {
            var phi = 2.0 * Math.PI * (q - 4 * p) / (4.0 * q);
            var (s, c) = Math.SinCos(phi);
            return (s, c);
        }
        var theta = 2.0 * Math.PI * p / q;
        var (sin, cos) = Math.SinCos(theta);
        return (cos, sin);
    }

    /// <summary>The prime factors of <paramref name="n"/> in ascending order, with multiplicity.</summary>
    public static List<int> PrimeFactors(int n)
    {
        var result = new List<int>();
        while ((n & 1) == 0 && n > 1) { result.Add(2); n >>= 1; }
        for (var p = 3; (long)p * p <= n; p += 2)
            while (n % p == 0) { result.Add(p); n /= p; }
        if (n > 1) result.Add(n);
        return result;
    }

    /// <summary>Largest prime factor (1 for n = 1).</summary>
    public static int LargestPrimeFactor(int n)
    {
        var factors = PrimeFactors(n);
        return factors.Count == 0 ? 1 : factors[^1];
    }

    /// <summary>
    /// A rough operation count for a direct mixed-radix transform, used only to choose between it
    /// and Bluestein. Same model as pocketfft's <c>cost_guess</c>: a radix-p pass costs about p
    /// operations per element, with a penalty above 5 for the generic butterfly.
    /// </summary>
    public static double CostGuess(int n)
    {
        const double genericPenalty = 1.1;
        double result = 0;
        var remaining = n;
        while ((remaining & 3) == 0) { result += 2; remaining >>= 2; }
        while ((remaining & 1) == 0) { result += 2; remaining >>= 1; }
        for (var x = 3; (long)x * x <= remaining; x += 2)
            while (remaining % x == 0) { result += x <= 5 ? x : genericPenalty * x; remaining /= x; }
        if (remaining > 1) result += remaining <= 5 ? remaining : genericPenalty * remaining;
        return result * n;
    }

    /// <summary>
    /// Smallest integer ≥ <paramref name="target"/> whose prime factors are all ≤
    /// <paramref name="largestPrime"/> (5 or 11).
    /// </summary>
    public static long NextSmooth(long target, int largestPrime)
    {
        if (target <= 1) return target;
        if (IsSmooth(target, largestPrime)) return target;

        // Enumerate 2^a 3^b 5^c (7^d 11^e) products above target; keep the best. The search is
        // over at most a few thousand candidates for any int-sized target.
        long best = long.MaxValue;
        long bound = 2 * target; // a power of two is always available below 2*target
        best = 1;
        while (best < target) best <<= 1;

        int[] primes = largestPrime >= 11 ? [11, 7, 5, 3] : [5, 3];
        Search(1, 0);
        return best;

        void Search(long value, int index)
        {
            if (index == primes.Length)
            {
                // Fill with powers of two.
                var v = value;
                while (v < target) v <<= 1;
                if (v < best) best = v;
                return;
            }
            var p = primes[index];
            for (var v = value; v < bound; v *= p)
            {
                Search(v, index + 1);
                if (v >= target) break;
            }
        }
    }

    /// <summary>Largest integer ≤ target whose prime factors are all ≤ largestPrime.</summary>
    public static long PrevSmooth(long target, int largestPrime)
    {
        if (target <= 1) return target;
        long best = 1;
        int[] primes = largestPrime >= 11 ? [11, 7, 5, 3] : [5, 3];
        Search(1, 0);
        return best;

        void Search(long value, int index)
        {
            if (index == primes.Length)
            {
                var v = value;
                while (v * 2 <= target) v <<= 1;
                if (v > best) best = v;
                return;
            }
            for (var v = value; v <= target; v *= primes[index]) Search(v, index + 1);
        }
    }

    public static bool IsSmooth(long n, int largestPrime)
    {
        if (n <= 0) return false;
        while ((n & 1) == 0) n >>= 1;
        foreach (var p in largestPrime >= 11 ? new[] { 3, 5, 7, 11 } : new[] { 3, 5 })
            while (n % p == 0) n /= p;
        return n == 1;
    }
}
