using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>scipy.stats multiple comparisons, p-value combination and trend tests.</summary>
public static partial class Stats
{
    /// <summary>
    /// Tukey's honestly significant difference test for all pairwise mean differences, scipy's
    /// <c>tukey_hsd</c> (Tukey–Kramer for unequal sizes). <paramref name="equalVar"/> false gives the
    /// Games–Howell test with Welch degrees of freedom per pair.
    /// </summary>
    public static TukeyHsdResult TukeyHsd(double[][] samples, bool equalVar = true)
    {
        StatsTestMath.CheckSamples(samples, 2);
        foreach (var s in samples)
            if (s.Length <= 1) throw new ArgumentException("Input sample size must be greater than one.", nameof(samples));
        var k = samples.Length;
        var means = samples.Select(s => StatsTestMath.Mean(s)).ToArray();
        var ns = samples.Select(s => (double)s.Length).ToArray();
        var vars = samples.Select(s => StatsTestMath.Var(s, 1)).ToArray();
        var nobs = ns.Sum();
        var se = new double[k * k];
        var df = new double[k * k];
        if (equalVar)
        {
            var mse = 0.0;
            for (var i = 0; i < k; i++) mse += vars[i] * (ns[i] - 1);
            mse /= nobs - k;
            var equal = ns.All(v => v == ns[0]);
            for (var i = 0; i < k; i++)
                for (var j = 0; j < k; j++)
                {
                    var normalize = equal ? 2 / ns[0] : 1 / ns[j] + 1 / ns[i];
                    se[i * k + j] = Math.Sqrt(normalize * mse / 2);
                    df[i * k + j] = nobs - k;
                }
        }
        else
        {
            for (var i = 0; i < k; i++)
                for (var j = 0; j < k; j++)
                {
                    double si = vars[i] / ns[i], sj = vars[j] / ns[j];
                    se[i * k + j] = Math.Sqrt(si + sj) / Math.Sqrt(2);
                    df[i * k + j] = (si + sj) * (si + sj) / (si * si / (ns[i] - 1) + sj * sj / (ns[j] - 1));
                }
        }
        var diff = new double[k * k];
        var p = new double[k * k];
        var dists = new Dictionary<double, StudentizedRangeDistribution>();
        for (var i = 0; i < k; i++)
            for (var j = 0; j < k; j++)
            {
                var f = i * k + j;
                diff[f] = means[i] - means[j];
                var t = Math.Abs(diff[f]) / se[f];
                if (!dists.TryGetValue(df[f], out var d)) dists[df[f]] = d = new StudentizedRangeDistribution(k, df[f]);
                p[f] = d.Sf(t);
            }
        return new TukeyHsdResult(new NdArray(diff, k, k), new NdArray(p, k, k), k, df, se);
    }

    /// <summary>Tukey's HSD over the given samples (equal variances).</summary>
    public static TukeyHsdResult TukeyHsd(params double[][] samples) => TukeyHsd(samples, true);

    /// <summary>
    /// Adjusted p-values controlling the false discovery rate, scipy's <c>false_discovery_control</c>:
    /// <c>"bh"</c> (Benjamini–Hochberg, independent or positively dependent tests) or <c>"by"</c>
    /// (Benjamini–Yekutieli, any dependence).
    /// </summary>
    public static double[] FalseDiscoveryControl(double[] ps, string method = "bh")
    {
        ArgumentNullException.ThrowIfNull(ps);
        foreach (var v in ps) if (!(v >= 0 && v <= 1)) throw new ArgumentException("`ps` must include only numbers between 0 and 1.", nameof(ps));
        var meth = method.ToLowerInvariant();
        if (meth is not ("bh" or "by")) throw new ArgumentException($"Unrecognized `method` '{method}'. Method must be one of bh, by.", nameof(method));
        var m = ps.Length;
        if (m <= 1) return (double[])ps.Clone();
        var order = Enumerable.Range(0, m).ToArray();
        var keys = (double[])ps.Clone();
        Array.Sort(keys, order);
        var adj = new double[m];
        var c = 1.0;
        if (meth == "by")
        {
            c = 0.0;
            for (var i = 1; i <= m; i++) c += 1.0 / i;
        }
        for (var i = 0; i < m; i++) adj[i] = keys[i] * ((double)m / (i + 1)) * c;
        for (var i = m - 2; i >= 0; i--) adj[i] = Math.Min(adj[i], adj[i + 1]);
        var r = new double[m];
        for (var i = 0; i < m; i++) r[order[i]] = Math.Clamp(adj[i], 0, 1);
        return r;
    }

    /// <summary>
    /// Combines independent p-values of tests of the same hypothesis, scipy's <c>combine_pvalues</c>:
    /// <c>"fisher"</c>, <c>"pearson"</c>, <c>"tippett"</c>, <c>"stouffer"</c> (optionally weighted) or
    /// <c>"mudholkar_george"</c>.
    /// </summary>
    public static SignificanceResult CombinePvalues(double[] pvalues, string method = "fisher", double[]? weights = null)
    {
        ArgumentNullException.ThrowIfNull(pvalues);
        double n = pvalues.Length;
        if (n == 0) return new SignificanceResult(double.NaN, double.NaN);
        switch (method.ToLowerInvariant())
        {
            case "fisher":
            {
                var stat = -2 * StatsTestMath.Sum(pvalues.Select(p => Math.Log(p)).ToArray());
                return new SignificanceResult(stat, StatsTestMath.Chi2Sf(stat, 2 * n));
            }
            case "pearson":
            {
                var stat = 2 * StatsTestMath.Sum(pvalues.Select(p => Special.Log1p(-p)).ToArray());
                return new SignificanceResult(stat, Special.Chdtr(2 * n, -stat));
            }
            case "mudholkar_george":
            {
                var norm = Math.Sqrt(3 / n) / Math.PI;
                var stat = -StatsTestMath.Sum(pvalues.Select(p => Math.Log(p)).ToArray()) + StatsTestMath.Sum(pvalues.Select(p => Special.Log1p(-p)).ToArray());
                var nu = 5 * n + 4;
                var approx = Math.Sqrt(nu / (nu - 2));
                return new SignificanceResult(stat, Special.Stdtr(nu, -(stat * norm * approx)));
            }
            case "tippett":
            {
                var stat = pvalues.Min();
                return new SignificanceResult(stat, -Special.Expm1(n * Special.Log1p(-stat)));
            }
            case "stouffer":
            {
                weights ??= Enumerable.Repeat(1.0, pvalues.Length).ToArray();
                if (weights.Length != pvalues.Length) throw new ArgumentException("pvalues and weights must be of the same length.", nameof(weights));
                double num = 0, w2 = 0;
                for (var i = 0; i < pvalues.Length; i++)
                {
                    num += weights[i] * -Special.Ndtri(pvalues[i]);
                    w2 += weights[i] * weights[i];
                }
                var stat = num / Math.Sqrt(w2);
                return new SignificanceResult(stat, Special.Ndtr(-stat));
            }
            default:
                throw new ArgumentException($"Invalid method '{method}'. Valid methods are 'fisher', 'pearson', 'mudholkar_george', 'tippett', and 'stouffer'", nameof(method));
        }
    }

    /// <summary>
    /// Page's test for an ordered trend across treatments, scipy's <c>page_trend_test</c>. Rows of
    /// <paramref name="data"/> are subjects (m ≥ 2), columns treatments (n ≥ 3) in the predicted order.
    /// <paramref name="method"/> <c>"auto"</c> is exact for small tables (Page's recursion), asymptotic otherwise.
    /// </summary>
    public static PageTrendTestResult PageTrendTest(NdArray data, bool ranked = false, int[]? predictedRanks = null, string method = "auto")
    {
        var (m, n, d) = StatsTestMath.Matrix(data, nameof(data));
        if (m < 2 || n < 3) throw new ArgumentException("Page's L is only appropriate for data with two or more rows and three or more columns.", nameof(data));
        if (d.Any(double.IsNaN)) throw new ArgumentException("`data` contains NaNs, which cannot be ranked meaningfully", nameof(data));
        var colsums = new double[n];
        for (var i = 0; i < m; i++)
        {
            var row = d.AsSpan(i * n, n).ToArray();
            var r = ranked ? row : Rankdata(row);
            for (var j = 0; j < n; j++) colsums[j] += r[j];
        }
        predictedRanks ??= Enumerable.Range(1, n).ToArray();
        if (predictedRanks.Length != n || !predictedRanks.OrderBy(v => v).SequenceEqual(Enumerable.Range(1, n)))
            throw new ArgumentException($"`predicted_ranks` must include each integer from 1 to {n} exactly once.", nameof(predictedRanks));
        var L = 0.0;
        for (var j = 0; j < n; j++) L += predictedRanks[j] * colsums[j];
        var meth = method.ToLowerInvariant();
        if (meth == "auto") meth = n > 8 || (m > 12 && n > 3) || m > 20 ? "asymptotic" : "exact";
        double p;
        if (meth == "asymptotic")
        {
            var e0 = m * (double)n * (n + 1) * (n + 1) / 4;
            var v0 = m * (double)n * n * (n + 1) * ((double)n * n - 1) / 144;
            p = Special.Ndtr(-((L - e0) / Math.Sqrt(v0)));
        }
        else if (meth == "exact") p = PageLSf((int)L, m, n);
        else throw new ArgumentException("method must be one of auto, exact, asymptotic.", nameof(method));
        return new PageTrendTestResult(L, p, meth);
    }

    /// <summary>Exact survival function of Page's L: the single-row distribution convolved m times.</summary>
    private static double PageLSf(int l, int rows, int k)
    {
        int a = k * (k + 1) * (k + 2) / 6, b = k * (k + 1) * (2 * k + 1) / 6;
        // Single-row pmf of Σ i·π(i) over permutations π, by DP over positions and used-rank sets.
        var single = new double[b - a + 1];
        var perm = Enumerable.Range(1, k).ToArray();
        var count = 0L;
        void Enumerate(int pos)
        {
            if (pos == k)
            {
                var s = 0;
                for (var i = 0; i < k; i++) s += (i + 1) * perm[i];
                single[s - a]++;
                count++;
                return;
            }
            for (var i = pos; i < k; i++)
            {
                (perm[pos], perm[i]) = (perm[i], perm[pos]);
                Enumerate(pos + 1);
                (perm[pos], perm[i]) = (perm[i], perm[pos]);
            }
        }
        Enumerate(0);
        for (var i = 0; i < single.Length; i++) single[i] /= count;
        var dist = (double[])single.Clone();
        for (var r = 1; r < rows; r++)
        {
            var next = new double[dist.Length + single.Length - 1];
            for (var i = 0; i < dist.Length; i++)
                for (var j = 0; j < single.Length; j++) next[i + j] += dist[i] * single[j];
            dist = next;
        }
        var lo = rows * a;
        var sum = 0.0;
        for (var v = Math.Max(l, lo); v - lo < dist.Length; v++) sum += dist[v - lo];
        return sum;
    }

    /// <summary>
    /// Dunnett's test comparing each treatment's mean with a control's, scipy's <c>dunnett</c>, with the
    /// pooled variance and the multivariate t null.
    /// </summary>
    /// <remarks>
    /// scipy integrates the multivariate t with Genz's randomised algorithm, so its p-values carry
    /// ~1e-4 noise. Dunnett's correlation matrix has the one-factor form ρᵢⱼ = λᵢλⱼ with
    /// λᵢ = (1 + n₀/nᵢ)^(−½), which reduces the probability to a two-dimensional integral (over the
    /// common factor and the χ scale); it is evaluated here by adaptive quadrature to ~1e-10.
    /// </remarks>
    public static DunnettResult Dunnett(double[][] samples, double[] control, Alternative alternative = Alternative.TwoSided)
    {
        StatsTestMath.CheckSamples(samples, 1);
        ArgumentNullException.ThrowIfNull(control);
        var k = samples.Length;
        var ns = samples.Select(s => (double)s.Length).ToArray();
        double n0 = control.Length;
        var df = ns.Sum() + n0 - k - 1;
        if (df < 1) throw new ArgumentException("Not enough observations for the pooled variance.");
        var mc = StatsTestMath.Mean(control);
        var means = samples.Select(s => StatsTestMath.Mean(s)).ToArray();
        var ss = StatsTestMath.Var(control) * n0;
        for (var i = 0; i < k; i++) ss += StatsTestMath.Var(samples[i]) * ns[i];
        var std = Math.Sqrt(ss / df);
        var lambda = new double[k];
        var stat = new double[k];
        for (var i = 0; i < k; i++)
        {
            lambda[i] = 1 / Math.Sqrt(n0 / ns[i] + 1);
            stat[i] = (means[i] - mc) / Math.Sqrt(1 / ns[i] + 1 / n0) / std;
        }
        var p = stat.Select(t => DunnettPvalue(t, lambda, df, alternative)).ToArray();
        return new DunnettResult(stat, p, lambda, df, std, means, mc, ns, n0, alternative);
    }

    /// <summary>P-value of a Dunnett statistic: 1 − P(all Tⱼ inside the acceptance region of t).</summary>
    internal static double DunnettPvalue(double t, double[] lambda, double df, Alternative alternative)
    {
        double lo, hi;
        switch (alternative)
        {
            case Alternative.Greater: lo = double.NegativeInfinity; hi = t; break;
            case Alternative.Less: lo = t; hi = double.PositiveInfinity; break;
            default: hi = Math.Abs(t); lo = -hi; break;
        }
        return Math.Clamp(1 - DunnettInside(lo, hi, lambda, df), 0, 1);
    }

    /// <summary>P(lo &lt; Tⱼ &lt; hi for all j) for the one-factor multivariate t.</summary>
    private static double DunnettInside(double lo, double hi, double[] lambda, double df)
    {
        double Mvn(double scale)
        {
            // Zⱼ = λⱼU + √(1 − λⱼ²)Eⱼ: condition on the common factor U.
            double Inner(double u)
            {
                var prod = Math.Exp(-0.5 * u * u) / Math.Sqrt(2 * Math.PI);
                foreach (var l in lambda)
                {
                    var r = Math.Sqrt(1 - l * l);
                    var up = double.IsPositiveInfinity(hi) ? 1.0 : Special.Ndtr((hi * scale - l * u) / r);
                    var dn = double.IsNegativeInfinity(lo) ? 0.0 : Special.Ndtr((lo * scale - l * u) / r);
                    prod *= up - dn;
                    if (prod == 0) break;
                }
                return prod;
            }
            return Integrate.Quad(Inner, -9, 9, 1e-13, 1e-11, limit: 200).Value;
        }
        if (df > 1e6) return Mvn(1.0);
        // S = √(W/ν) with W ~ χ²_ν: density ∝ s^(ν−1) exp(−ν s²/2), concentrated near 1.
        var logc = df / 2 * Math.Log(df) - (df / 2 - 1) * Math.Log(2) - Special.GammaLn(df / 2);
        double Outer(double s) => s <= 0 ? 0 : Math.Exp(logc + (df - 1) * Math.Log(s) - df * s * s / 2) * Mvn(s);
        var upper = Math.Max(1 + 40 / Math.Sqrt(df), 3);
        return Integrate.Quad(Outer, 0, 1, 1e-12, 1e-10, limit: 200).Value + Integrate.Quad(Outer, 1, upper, 1e-12, 1e-10, limit: 200).Value;
    }
}
