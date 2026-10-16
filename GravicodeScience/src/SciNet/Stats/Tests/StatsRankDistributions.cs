namespace Gravicode.Science.SciNet;

/// <summary>
/// Exact null distributions of the rank statistics (Mann–Whitney U, Wilcoxon signed rank,
/// Ansari–Bradley, Kendall's τ) and the signed-rank statistic itself.
/// </summary>
internal static class StatsRankDistributions
{
    // ------------------------------------------------------------------ Mann–Whitney U

    /// <summary>
    /// P(U ≥ k) under H₀ for sample sizes n1, n2 (no ties), scipy's <c>_MWU.sf</c>.
    /// </summary>
    /// <remarks>
    /// Frequencies come from the recursion of Löffler (1983) / Harding (1984):
    /// u·f(u) = Σ_{i=1..u} σ(i) f(u − i) with σ(i) = Σ_{d | i, d ≤ n1} d − Σ_{d | i, n2 &lt; d ≤ n1 + n2} d,
    /// O(u²) instead of the O(n1·n2·u) table, and only up to the u actually needed.
    /// </remarks>
    public static double MwuSf(int n1, int n2, long k)
    {
        if (n1 > n2) (n1, n2) = (n2, n1);
        var total = (long)n1 * n2;
        if (k <= 0) return 1.0;
        if (k > total) return 0.0;
        var kc = total - k;
        if (k < kc)
        {
            var pmf = MwuPmf(n1, n2, (int)k);
            var cdf = 0.0;
            for (var i = 0; i <= k; i++) cdf += pmf[i];
            return 1.0 - cdf + pmf[k];
        }
        else
        {
            var pmf = MwuPmf(n1, n2, (int)kc);
            var cdf = 0.0;
            for (var i = 0; i <= kc; i++) cdf += pmf[i];
            return cdf;
        }
    }

    /// <summary>The null pmf of U for u = 0…maxu.</summary>
    public static double[] MwuPmf(int n1, int n2, int maxu)
    {
        if (n1 > n2) (n1, n2) = (n2, n1);
        var s = new double[maxu + 1];
        for (var d = 1; d <= n1; d++)
            for (var i = d; i <= maxu; i += d) s[i] += d;
        for (var d = n2 + 1; d <= n2 + n1; d++)
            for (var i = d; i <= maxu; i += d) s[i] -= d;
        var conf = new double[maxu + 1];
        conf[0] = 1;
        for (var u = 1; u <= maxu; u++)
        {
            var acc = 0.0;
            for (var j = 0; j < u; j++) acc += conf[j] * s[u - j];
            var v = acc / u;
            // scipy stores these in uint64 (truncating) until they overflow it.
            conf[u] = v < 1.8e19 ? Math.Floor(v) : v;
        }
        var totalWays = Special.Binom(n1 + n2, n1);
        for (var u = 0; u <= maxu; u++) conf[u] /= totalWays;
        return conf;
    }

    // ------------------------------------------------------------------ Wilcoxon signed rank

    /// <summary>The signed-rank statistic with scipy's zero handling and variance terms.</summary>
    public readonly record struct WilcoxonStat(double RPlus, double RMinus, double Mean, double Se, int Count, bool HasTies, double[] Ranks);

    public static WilcoxonStat WilcoxonStatistic(double[] d, string zeroMethod)
    {
        var n = d.Length;
        var nZero = 0;
        foreach (var v in d) if (v == 0) nZero++;
        double[] absd;
        int[] map;
        if (zeroMethod == "wilcox")
        {
            map = Enumerable.Range(0, n).Where(i => d[i] != 0).ToArray();
            absd = map.Select(i => Math.Abs(d[i])).ToArray();
        }
        else
        {
            map = Enumerable.Range(0, n).ToArray();
            absd = d.Select(Math.Abs).ToArray();
        }
        var count = zeroMethod == "wilcox" ? n - nZero : n;
        var r = StatsTestMath.RankAverage(absd, out _);
        var sorted = StatsTestMath.Sorted(absd);
        var counts = StatsTestMath.TieCounts(sorted);
        var hasTies = counts.Any(c => c > 1);
        double rp = 0, rm = 0, rz = 0;
        var fullRanks = new double[n];
        for (var k = 0; k < map.Length; k++)
        {
            var v = d[map[k]];
            fullRanks[map[k]] = r[k];
            if (v > 0) rp += r[k];
            else if (v < 0) rm += r[k];
            else rz += r[k];
        }
        if (zeroMethod == "zsplit") { rp += rz / 2; rm += rz / 2; }
        double c1 = count;
        var mn = c1 * (c1 + 1) * 0.25;
        var se = c1 * (c1 + 1) * (2 * c1 + 1);
        if (zeroMethod == "pratt")
        {
            double z = nZero;
            mn -= z * (z + 1) * 0.25;
            se -= z * (z + 1) * (2 * z + 1);
            // The zero group (the smallest |d|) leaves the tie correction.
            if (nZero > 0) counts.RemoveAt(0);
        }
        var tie = 0.0;
        foreach (var t in counts) tie += (double)t * t * t - t;
        se = Math.Sqrt((se - tie / 2) / 24);
        return new WilcoxonStat(rp, rm, mn, se, count, hasTies, fullRanks);
    }

    /// <summary>Null pmf of R⁺ for n untied, non-zero differences (scipy's <c>_get_wilcoxon_distr</c>).</summary>
    public static double[] WilcoxonPmf(int n)
    {
        var c = new double[] { 1.0 };
        for (var k = 1; k <= n; k++)
        {
            var next = new double[k * (k + 1) / 2 + 1];
            var m = c.Length;
            for (var i = 0; i < m; i++) next[i] = c[i] * 0.5;
            for (var i = 0; i < m; i++) next[next.Length - m + i] += c[i] * 0.5;
            c = next;
        }
        return c;
    }

    public static double WilcoxonCdf(double[] pmf, int n, int k)
    {
        var mean = n * (n + 1) / 4.0;
        if (k <= mean) return SumRange(pmf, 0, k);
        return 1 - SumRange(pmf, k + 1, pmf.Length - 1);
    }

    public static double WilcoxonSf(double[] pmf, int n, int k)
    {
        var mean = n * (n + 1) / 4.0;
        if (k <= mean) return SumRange(pmf, k, pmf.Length - 1);
        return 1 - SumRange(pmf, 0, k - 1);
    }

    private static double SumRange(double[] a, int lo, int hi)
    {
        lo = Math.Max(lo, 0);
        hi = Math.Min(hi, a.Length - 1);
        var s = 0.0;
        for (var i = lo; i <= hi; i++) s += a[i];
        return s;
    }

    /// <summary>
    /// Exhaustive sign-flip p-value for R⁺, what scipy's <c>permutation_test(permutation_type='samples')</c>
    /// computes when 2ⁿ fits in its resample budget.
    /// </summary>
    public static double WilcoxonSignFlipPvalue(double[] d, string zeroMethod, double observed, Alternative alt)
    {
        var st = WilcoxonStatistic(d, zeroMethod);
        var n = d.Length;
        if (n > 20) throw new ArgumentException("exhaustive sign flipping is limited to 20 observations.");
        var r = st.Ranks;
        double zeroHalf = 0;
        if (zeroMethod == "zsplit") for (var i = 0; i < n; i++) if (d[i] == 0) zeroHalf += r[i] / 2;
        var total = 1 << n;
        var gamma = Math.Abs(Math.Pow(2, -52) * 100 * observed);
        long less = 0, greater = 0;
        for (var mask = 0; mask < total; mask++)
        {
            var rp = zeroHalf;
            for (var i = 0; i < n; i++)
            {
                if (d[i] == 0) continue;
                var positive = d[i] > 0;
                if (((mask >> i) & 1) == 1) positive = !positive;
                if (positive) rp += r[i];
            }
            if (rp <= observed + gamma) less++;
            if (rp >= observed - gamma) greater++;
        }
        double pl = (double)less / total, pg = (double)greater / total;
        return alt switch
        {
            Alternative.Less => pl,
            Alternative.Greater => pg,
            _ => Math.Min(1.0, 2 * Math.Min(pl, pg)),
        };
    }

    // ------------------------------------------------------------------ Ansari–Bradley

    /// <summary>
    /// Exact null frequencies of the Ansari–Bradley statistic for n test and m other observations, by
    /// subset-sum dynamic programming over the symmetric scores min(i, N + 1 − i). Returns the smallest
    /// attainable statistic and the frequency of each integer value from there.
    /// </summary>
    public static (int Start, double[] Freq) AnsariFrequencies(int n, int m)
    {
        var N = n + m;
        var scores = new int[N];
        for (var i = 1; i <= N; i++) scores[i - 1] = Math.Min(i, N - i + 1);
        var maxSum = scores.OrderByDescending(v => v).Take(n).Sum();
        // dp[j][s]: number of size-j subsets with score sum s.
        var dp = new double[n + 1][];
        for (var j = 0; j <= n; j++) dp[j] = new double[maxSum + 1];
        dp[0][0] = 1;
        foreach (var sc in scores)
            for (var j = n; j >= 1; j--)
            {
                var prev = dp[j - 1];
                var cur = dp[j];
                for (var s = maxSum; s >= sc; s--) cur[s] += prev[s - sc];
            }
        var f = dp[n];
        var start = 0;
        while (start < f.Length && f[start] == 0) start++;
        return (start, f[start..]);
    }

    // ------------------------------------------------------------------ Kendall

    /// <summary>scipy's <c>_kendall_p_exact</c>: exact p-value for c concordant pairs among n untied pairs.</summary>
    public static double KendallPExact(int n, long c, Alternative alternative)
    {
        long half = (long)n * (n - 1) / 2;
        var inRightTail = c >= half - c;
        var altGreater = alternative == Alternative.Greater;
        c = Math.Min(c, half - c);
        double prob, pmassAtC = 0;
        if (n <= 0) throw new ArgumentException("n must be positive.");
        if (n == 1) { prob = 1.0; pmassAtC = 1; }
        else if (n == 2) { prob = 1.0; pmassAtC = 0.5; }
        else if (c == 0)
        {
            prob = n < 171 ? 2.0 / Factorial(n) : 0.0;
            pmassAtC = prob / 2;
        }
        else if (c == 1)
        {
            prob = n < 172 ? 2.0 / Factorial(n - 1) : 0.0;
            pmassAtC = (n - 1) / Factorial(n);
        }
        else if (4 * c == (long)n * (n - 1) && alternative == Alternative.TwoSided) prob = 1.0;
        else if (n < 171)
        {
            var nw = new double[c + 1];
            nw[0] = 1; nw[1] = 1;
            for (var j = 3; j <= n; j++)
            {
                for (var i = 1; i <= c; i++) nw[i] += nw[i - 1];
                if (j <= c)
                    for (var i = (int)c; i >= j; i--) nw[i] -= nw[i - j];
            }
            var s = 0.0;
            foreach (var v in nw) s += v;
            var f = Factorial(n);
            prob = 2.0 * s / f;
            pmassAtC = nw[c] / f;
        }
        else
        {
            var nw = new double[c + 1];
            nw[0] = 1; nw[1] = 1;
            for (var j = 3; j <= n; j++)
            {
                for (var i = 1; i <= c; i++) nw[i] += nw[i - 1];
                for (var i = 0; i <= c; i++) nw[i] /= j;
                if (j <= c)
                    for (var i = (int)c; i >= j; i--) nw[i] -= nw[i - j];
            }
            prob = nw.Sum();
            pmassAtC = nw[c] / 2;
        }
        if (alternative != Alternative.TwoSided)
        {
            if (inRightTail == altGreater) prob /= 2;
            else prob = 1 - prob / 2 + pmassAtC;
        }
        return Math.Clamp(prob, 0, 1);
    }

    private static double Factorial(int n)
    {
        var f = 1.0;
        for (var i = 2; i <= n; i++) f *= i;
        return f;
    }
}
