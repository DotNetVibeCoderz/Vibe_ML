using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>How <c>Stats.PermutationTest</c> generates the null distribution (scipy's <c>permutation_type</c>).</summary>
public enum PermutationType
{
    /// <summary>Pool the samples and re-partition them into groups of the original sizes (independent samples).</summary>
    Independent,
    /// <summary>Permute each sample's order independently, changing which observations are paired (correlation tests).</summary>
    Pairings,
    /// <summary>Permute observations between samples within each index (paired samples); one sample means sign flips.</summary>
    Samples,
}

/// <summary>Result of <see cref="Stats.Jackknife"/>.</summary>
/// <param name="Values">The statistic with each observation left out in turn.</param>
/// <param name="Estimate">The bias-corrected jackknife estimate n·θ̂ − (n − 1)·mean(θ̂₍ᵢ₎).</param>
/// <param name="Bias">The jackknife bias estimate (n − 1)(mean(θ̂₍ᵢ₎) − θ̂).</param>
/// <param name="StandardError">The jackknife standard error.</param>
public sealed record JackknifeResult(double[] Values, double Estimate, double Bias, double StandardError);

/// <summary>scipy.stats resampling methods: permutation tests, Monte Carlo tests, the bootstrap and the jackknife.</summary>
public static partial class Stats
{
    // ------------------------------------------------------------------ permutation test

    /// <summary>
    /// Permutation test of the statistic on <paramref name="data"/> (one array per sample), scipy's
    /// <c>permutation_test</c>. When every distinct permutation fits in <paramref name="nResamples"/> the
    /// null distribution is enumerated and the p-value is exact (no +1 adjustment); otherwise random
    /// permutations are drawn and the p-value is (count + 1)/(B + 1). Comparisons use scipy's relative
    /// tolerance of 100 ε so that statistics equal up to rounding count as ties.
    /// </summary>
    /// <param name="data">The samples.</param>
    /// <param name="statistic">The test statistic of the (resampled) samples. Must not keep the arrays it receives.</param>
    /// <param name="permutationType">How samples are permuted.</param>
    /// <param name="nResamples">Monte Carlo budget; exact enumeration is used when the permutation count fits.</param>
    /// <param name="alternative">Which tail(s) count as extreme.</param>
    /// <param name="seed">Seed for the random permutations.</param>
    /// <param name="parallel">Evaluate the statistic on several threads (the statistic must then be thread-safe).
    /// The result is identical either way: each resample has its own seed derived from <paramref name="seed"/>.</param>
    public static ResamplingTestResult PermutationTest(double[][] data, Func<double[][], double> statistic, PermutationType permutationType = PermutationType.Independent,
        int nResamples = 9999, Alternative alternative = Alternative.TwoSided, int? seed = null, bool parallel = false)
    {
        StatsTestMath.CheckSamples(data, 1, nameof(data));
        ArgumentNullException.ThrowIfNull(statistic);
        if (nResamples < 1) throw new ArgumentException("n_resamples must be a positive integer.", nameof(nResamples));
        var k = data.Length;
        if (permutationType != PermutationType.Independent)
        {
            var n0 = data[0].Length;
            foreach (var s in data) if (s.Length != n0) throw new ArgumentException("All samples must contain the same number of observations for this permutation type.", nameof(data));
        }
        else if (k < 2) throw new ArgumentException("Independent permutations need at least two samples.", nameof(data));
        var observed = statistic(data);
        var nMax = PermutationCount(data, permutationType);
        double[] nulls;
        bool exact;
        if (nMax <= nResamples)
        {
            exact = true;
            nulls = EnumeratePermutations(data, statistic, permutationType);
        }
        else
        {
            exact = false;
            var master = new GraviRandom(seed ?? Random.Shared.Next());
            var seeds = new ulong[nResamples];
            for (var b = 0; b < nResamples; b++) seeds[b] = master.NextUInt64();
            nulls = new double[nResamples];
            void One(int b)
            {
                var rng = new GraviRandom(seeds[b]);
                nulls[b] = statistic(RandomPermutation(data, permutationType, rng));
            }
            if (parallel) Parallel.For(0, nResamples, One);
            else for (var b = 0; b < nResamples; b++) One(b);
        }
        return new ResamplingTestResult(observed, ResamplingPvalue(nulls, observed, alternative, exact ? 0 : 1), nulls);
    }

    /// <summary>Two-sample convenience overload of <see cref="PermutationTest(double[][], Func{double[][], double}, PermutationType, int, Alternative, int?, bool)"/>.</summary>
    public static ResamplingTestResult PermutationTest(double[] x, double[] y, Func<double[], double[], double> statistic, PermutationType permutationType = PermutationType.Independent,
        int nResamples = 9999, Alternative alternative = Alternative.TwoSided, int? seed = null, bool parallel = false)
    {
        ArgumentNullException.ThrowIfNull(statistic);
        return PermutationTest([x, y], d => statistic(d[0], d[1]), permutationType, nResamples, alternative, seed, parallel);
    }

    private static double ResamplingPvalue(double[] nulls, double observed, Alternative alternative, int adjustment)
    {
        var gamma = Math.Abs(Math.Pow(2, -52) * 100 * observed);
        long less = 0, greater = 0;
        foreach (var v in nulls)
        {
            if (v <= observed + gamma) less++;
            if (v >= observed - gamma) greater++;
        }
        double n = nulls.Length;
        double pl = (less + adjustment) / (n + adjustment), pg = (greater + adjustment) / (n + adjustment);
        var p = alternative switch
        {
            Alternative.Less => pl,
            Alternative.Greater => pg,
            _ => 2 * Math.Min(pl, pg),
        };
        return Math.Clamp(p, 0, 1);
    }

    private static double PermutationCount(double[][] data, PermutationType type)
    {
        var k = data.Length;
        switch (type)
        {
            case PermutationType.Independent:
            {
                // Multinomial N! / Π nᵢ!, as a product of binomials.
                double count = 1;
                var cum = 0;
                foreach (var s in data)
                {
                    cum += s.Length;
                    count *= Special.Binom(cum, s.Length);
                }
                return count;
            }
            case PermutationType.Pairings:
                return Math.Pow(Special.Factorial(data[0].Length), k);
            default:
                return Math.Pow(Special.Factorial(k == 1 ? 2 : k), data[0].Length);
        }
    }

    private static double[][] RandomPermutation(double[][] data, PermutationType type, GraviRandom rng)
    {
        var k = data.Length;
        switch (type)
        {
            case PermutationType.Independent:
            {
                var pooled = StatsTestMath.Concat(data);
                for (var i = pooled.Length - 1; i > 0; i--)
                {
                    var j = rng.Next(i + 1);
                    (pooled[i], pooled[j]) = (pooled[j], pooled[i]);
                }
                var r = new double[k][];
                var o = 0;
                for (var s = 0; s < k; s++) { r[s] = pooled[o..(o + data[s].Length)]; o += data[s].Length; }
                return r;
            }
            case PermutationType.Pairings:
            {
                var r = new double[k][];
                for (var s = 0; s < k; s++)
                {
                    var a = (double[])data[s].Clone();
                    for (var i = a.Length - 1; i > 0; i--)
                    {
                        var j = rng.Next(i + 1);
                        (a[i], a[j]) = (a[j], a[i]);
                    }
                    r[s] = a;
                }
                return r;
            }
            default:
            {
                var n = data[0].Length;
                if (k == 1)
                {
                    var a = new double[n];
                    for (var i = 0; i < n; i++) a[i] = rng.Next(2) == 0 ? data[0][i] : -data[0][i];
                    return [a];
                }
                var r = new double[k][];
                for (var s = 0; s < k; s++) r[s] = new double[n];
                var perm = new int[k];
                for (var i = 0; i < n; i++)
                {
                    for (var s = 0; s < k; s++) perm[s] = s;
                    for (var s = k - 1; s > 0; s--)
                    {
                        var j = rng.Next(s + 1);
                        (perm[s], perm[j]) = (perm[j], perm[s]);
                    }
                    for (var s = 0; s < k; s++) r[s][i] = data[perm[s]][i];
                }
                return r;
            }
        }
    }

    private static double[] EnumeratePermutations(double[][] data, Func<double[][], double> statistic, PermutationType type)
    {
        var k = data.Length;
        var result = new List<double>();
        switch (type)
        {
            case PermutationType.Independent:
            {
                var pooled = StatsTestMath.Concat(data);
                var sizes = data.Select(s => s.Length).ToArray();
                var assign = new int[pooled.Length];
                var remaining = (int[])sizes.Clone();
                // Assign each pooled position to a sample, in lexicographic order of the assignment.
                void Rec(int pos)
                {
                    if (pos == pooled.Length)
                    {
                        var r = new double[k][];
                        var fill = new int[k];
                        for (var s = 0; s < k; s++) r[s] = new double[sizes[s]];
                        for (var i = 0; i < pooled.Length; i++) r[assign[i]][fill[assign[i]]++] = pooled[i];
                        result.Add(statistic(r));
                        return;
                    }
                    for (var s = 0; s < k; s++)
                    {
                        if (remaining[s] == 0) continue;
                        remaining[s]--;
                        assign[pos] = s;
                        Rec(pos + 1);
                        remaining[s]++;
                    }
                }
                Rec(0);
                break;
            }
            case PermutationType.Pairings:
            {
                var n = data[0].Length;
                var perms = AllPermutations(n);
                var idx = new int[k];
                while (true)
                {
                    var r = new double[k][];
                    for (var s = 0; s < k; s++)
                    {
                        var p = perms[idx[s]];
                        r[s] = new double[n];
                        for (var i = 0; i < n; i++) r[s][i] = data[s][p[i]];
                    }
                    result.Add(statistic(r));
                    var c = k - 1;
                    while (c >= 0 && ++idx[c] == perms.Count) { idx[c] = 0; c--; }
                    if (c < 0) break;
                }
                break;
            }
            default:
            {
                var n = data[0].Length;
                var src = k == 1 ? [data[0], data[0].Select(v => -v).ToArray()] : data;
                var kk = src.Length;
                var perms = AllPermutations(kk);
                var idx = new int[n];
                while (true)
                {
                    var r = new double[kk][];
                    for (var s = 0; s < kk; s++) r[s] = new double[n];
                    for (var i = 0; i < n; i++)
                    {
                        var p = perms[idx[i]];
                        for (var s = 0; s < kk; s++) r[s][i] = src[p[s]][i];
                    }
                    result.Add(statistic(k == 1 ? [r[0]] : r));
                    var c = n - 1;
                    while (c >= 0 && ++idx[c] == perms.Count) { idx[c] = 0; c--; }
                    if (c < 0) break;
                }
                break;
            }
        }
        return result.ToArray();
    }

    private static List<int[]> AllPermutations(int n)
    {
        var list = new List<int[]>();
        var a = Enumerable.Range(0, n).ToArray();
        void Rec(int pos)
        {
            if (pos == n) { list.Add((int[])a.Clone()); return; }
            for (var i = pos; i < n; i++)
            {
                (a[pos], a[i]) = (a[i], a[pos]);
                Rec(pos + 1);
                (a[pos], a[i]) = (a[i], a[pos]);
            }
        }
        Rec(0);
        return list;
    }

    // ------------------------------------------------------------------ Monte Carlo test

    /// <summary>
    /// Monte Carlo hypothesis test, scipy's <c>monte_carlo_test</c>: the statistic of the observed samples
    /// against its distribution over samples drawn from the null by <paramref name="rvs"/> (one generator
    /// per sample, called with the sample size). p = (count + 1)/(B + 1).
    /// </summary>
    public static ResamplingTestResult MonteCarloTest(double[][] data, Func<int, GraviRandom, double[]>[] rvs, Func<double[][], double> statistic,
        int nResamples = 9999, Alternative alternative = Alternative.TwoSided, int? seed = null, bool parallel = false)
    {
        StatsTestMath.CheckSamples(data, 1, nameof(data));
        ArgumentNullException.ThrowIfNull(rvs);
        ArgumentNullException.ThrowIfNull(statistic);
        if (rvs.Length != data.Length) throw new ArgumentException("If `rvs` is a sequence, `len(rvs)` must equal `len(data)`.", nameof(rvs));
        if (nResamples < 1) throw new ArgumentException("n_resamples must be a positive integer.", nameof(nResamples));
        var observed = statistic(data);
        var master = new GraviRandom(seed ?? Random.Shared.Next());
        var seeds = new ulong[nResamples];
        for (var b = 0; b < nResamples; b++) seeds[b] = master.NextUInt64();
        var nulls = new double[nResamples];
        void One(int b)
        {
            var rng = new GraviRandom(seeds[b]);
            var sample = new double[data.Length][];
            for (var s = 0; s < data.Length; s++) sample[s] = rvs[s](data[s].Length, rng);
            nulls[b] = statistic(sample);
        }
        if (parallel) Parallel.For(0, nResamples, One);
        else for (var b = 0; b < nResamples; b++) One(b);
        return new ResamplingTestResult(observed, ResamplingPvalue(nulls, observed, alternative, 1), nulls);
    }

    /// <summary>One-sample Monte Carlo test with the null given as a frozen distribution.</summary>
    public static ResamplingTestResult MonteCarloTest(double[] sample, IUnivariateDistribution nullDistribution, Func<double[], double> statistic,
        int nResamples = 9999, Alternative alternative = Alternative.TwoSided, int? seed = null, bool parallel = false)
    {
        ArgumentNullException.ThrowIfNull(nullDistribution);
        ArgumentNullException.ThrowIfNull(statistic);
        return MonteCarloTest([sample], [(n, rng) => nullDistribution.Rvs(n, rng)], d => statistic(d[0]), nResamples, alternative, seed, parallel);
    }

    // ------------------------------------------------------------------ bootstrap

    /// <summary>
    /// Bootstrap confidence interval and standard error of a statistic, scipy's <c>bootstrap</c>.
    /// <paramref name="method"/>: <c>"percentile"</c>, <c>"basic"</c> (reverse percentile) or <c>"BCa"</c>
    /// (bias-corrected and accelerated, Efron 1987; acceleration from the jackknife). With
    /// <paramref name="paired"/> every sample is resampled with the same indices. A previous
    /// <paramref name="bootstrapDistribution"/> is extended by <paramref name="nResamples"/> new draws
    /// (scipy's <c>bootstrap_result</c>), so <c>nResamples: 0</c> recomputes the interval only.
    /// </summary>
    public static BootstrapResult Bootstrap(double[][] data, Func<double[][], double> statistic, int nResamples = 9999, double confidenceLevel = 0.95,
        string method = "BCa", bool paired = false, Alternative alternative = Alternative.TwoSided, int? seed = null,
        double[]? bootstrapDistribution = null, bool parallel = false)
    {
        StatsTestMath.CheckSamples(data, 1, nameof(data));
        ArgumentNullException.ThrowIfNull(statistic);
        var meth = method.ToLowerInvariant();
        if (meth is not ("percentile" or "basic" or "bca")) throw new ArgumentException("`method` must be in {'percentile', 'basic', 'bca'}", nameof(method));
        if (!(confidenceLevel > 0 && confidenceLevel < 1)) throw new ArgumentException("confidence_level must lie in (0, 1).", nameof(confidenceLevel));
        if (nResamples < 0) throw new ArgumentException("n_resamples must be a non-negative integer.", nameof(nResamples));
        if (paired)
        {
            var n0 = data[0].Length;
            foreach (var s in data) if (s.Length != n0) throw new ArgumentException("When `paired is True`, all samples must have the same length along `axis`.", nameof(data));
        }
        foreach (var s in data) if (s.Length < 2) throw new ArgumentException("Each sample in `data` must contain two or more observations along `axis`.", nameof(data));
        var master = new GraviRandom(seed ?? Random.Shared.Next());
        var seeds = new ulong[nResamples];
        for (var b = 0; b < nResamples; b++) seeds[b] = master.NextUInt64();
        var fresh = new double[nResamples];
        void One(int b)
        {
            var rng = new GraviRandom(seeds[b]);
            var r = new double[data.Length][];
            if (paired)
            {
                var n = data[0].Length;
                var idx = new int[n];
                for (var i = 0; i < n; i++) idx[i] = rng.Next(n);
                for (var s = 0; s < data.Length; s++)
                {
                    r[s] = new double[n];
                    for (var i = 0; i < n; i++) r[s][i] = data[s][idx[i]];
                }
            }
            else
                for (var s = 0; s < data.Length; s++)
                {
                    var n = data[s].Length;
                    r[s] = new double[n];
                    for (var i = 0; i < n; i++) r[s][i] = data[s][rng.Next(n)];
                }
            fresh[b] = statistic(r);
        }
        if (parallel) Parallel.For(0, nResamples, One);
        else for (var b = 0; b < nResamples; b++) One(b);
        var thetaB = bootstrapDistribution is null ? fresh : [.. bootstrapDistribution, .. fresh];
        if (thetaB.Length == 0) throw new ArgumentException("There must be at least one bootstrap resample.", nameof(nResamples));
        var alpha = alternative == Alternative.TwoSided ? (1 - confidenceLevel) / 2 : 1 - confidenceLevel;
        double a1 = alpha, a2 = 1 - alpha;
        if (meth == "bca") (a1, a2) = BcaLevels(data, statistic, paired, alpha, thetaB);
        double lo = double.IsNaN(a1) ? double.NaN : Quantile(thetaB, a1);
        double hi = double.IsNaN(a2) ? double.NaN : Quantile(thetaB, a2);
        if (meth == "basic")
        {
            var th = statistic(data);
            (lo, hi) = (2 * th - hi, 2 * th - lo);
        }
        if (alternative == Alternative.Less) lo = double.NegativeInfinity;
        else if (alternative == Alternative.Greater) hi = double.PositiveInfinity;
        return new BootstrapResult(new ConfidenceInterval(lo, hi), thetaB, Math.Sqrt(StatsTestMath.Var(thetaB, 1)));
    }

    /// <summary>One-sample convenience overload of <see cref="Bootstrap(double[][], Func{double[][], double}, int, double, string, bool, Alternative, int?, double[], bool)"/>.</summary>
    public static BootstrapResult Bootstrap(double[] sample, Func<double[], double> statistic, int nResamples = 9999, double confidenceLevel = 0.95,
        string method = "BCa", Alternative alternative = Alternative.TwoSided, int? seed = null, double[]? bootstrapDistribution = null, bool parallel = false)
    {
        ArgumentNullException.ThrowIfNull(statistic);
        return Bootstrap([sample], d => statistic(d[0]), nResamples, confidenceLevel, method, false, alternative, seed, bootstrapDistribution, parallel);
    }

    /// <summary>scipy's <c>_bca_interval</c>: the adjusted percentile levels.</summary>
    private static (double A1, double A2) BcaLevels(double[][] data, Func<double[][], double> statistic, bool paired, double alpha, double[] thetaB)
    {
        var thetaHat = statistic(data);
        double below = 0, belowEq = 0;
        foreach (var t in thetaB)
        {
            if (t < thetaHat) below++;
            if (t <= thetaHat) belowEq++;
        }
        var z0 = Special.Ndtri((below + belowEq) / (2.0 * thetaB.Length));
        double num = 0, den = 0;
        // With paired data scipy jackknifes the shared index vector: one sample of n index rows.
        if (paired)
        {
            var n = data[0].Length;
            var ti = new double[n];
            for (var i = 0; i < n; i++)
            {
                var r = data.Select(s => s.Where((_, j) => j != i).ToArray()).ToArray();
                ti[i] = statistic(r);
            }
            Accumulate(ti, ref num, ref den);
        }
        else
            for (var s = 0; s < data.Length; s++)
            {
                var n = data[s].Length;
                var ti = new double[n];
                for (var i = 0; i < n; i++)
                {
                    var r = (double[][])data.Clone();
                    var reduced = new double[n - 1];
                    for (int j = 0, o = 0; j < n; j++) if (j != i) reduced[o++] = data[s][j];
                    r[s] = reduced;
                    ti[i] = statistic(r);
                }
                Accumulate(ti, ref num, ref den);
            }
        var a = 1.0 / 6 * num / Math.Pow(den, 1.5);
        var za = Special.Ndtri(alpha);
        var n1 = z0 + za;
        var a1 = Special.Ndtr(z0 + n1 / (1 - a * n1));
        var n2 = z0 - za;
        var a2 = Special.Ndtr(z0 + n2 / (1 - a * n2));
        return (a1, a2);

        static void Accumulate(double[] ti, ref double num, ref double den)
        {
            double n = ti.Length;
            var mean = StatsTestMath.Mean(ti);
            double s3 = 0, s2 = 0;
            foreach (var t in ti)
            {
                var u = (n - 1) * (mean - t);
                s3 += u * u * u;
                s2 += u * u;
            }
            num += s3 / (n * n * n);
            den += s2 / (n * n);
        }
    }

    // ------------------------------------------------------------------ jackknife

    /// <summary>
    /// Leave-one-out jackknife of a statistic (Quenouille–Tukey): the leave-one-out values, the
    /// bias-corrected estimate, the bias and the standard error √((n−1)/n Σ(θ̂₍ᵢ₎ − θ̄)²).
    /// </summary>
    public static JackknifeResult Jackknife(double[] sample, Func<double[], double> statistic)
    {
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(statistic);
        var n = sample.Length;
        if (n < 2) throw new ArgumentException("The jackknife needs at least two observations.", nameof(sample));
        var values = new double[n];
        var buf = new double[n - 1];
        for (var i = 0; i < n; i++)
        {
            for (int j = 0, o = 0; j < n; j++) if (j != i) buf[o++] = sample[j];
            values[i] = statistic((double[])buf.Clone());
        }
        var full = statistic(sample);
        var mean = StatsTestMath.Mean(values);
        var bias = (n - 1) * (mean - full);
        var ss = 0.0;
        foreach (var v in values) ss += (v - mean) * (v - mean);
        return new JackknifeResult(values, full - bias, bias, Math.Sqrt((n - 1.0) / n * ss));
    }
}
