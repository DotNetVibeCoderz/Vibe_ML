using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>scipy.stats goodness-of-fit for counts, contingency tables and exact tests on proportions.</summary>
public static partial class Stats
{
    // ------------------------------------------------------------------ power divergence family

    /// <summary>
    /// Pearson's chi-square goodness-of-fit test, scipy's <c>chisquare</c>. Expected frequencies default
    /// to uniform; when given, their total must match the observed total to √ε relative tolerance.
    /// </summary>
    public static SignificanceResult Chisquare(double[] fObs, double[]? fExp = null, int ddof = 0, bool sumCheck = true)
        => PowerDivergence(fObs, fExp, ddof, 1.0, sumCheck);

    /// <summary>
    /// Cressie–Read power-divergence statistic and its χ²(k − 1 − ddof) p-value, scipy's
    /// <c>power_divergence</c>, with <paramref name="lambda"/> named: <c>"pearson"</c> (1),
    /// <c>"log-likelihood"</c> (0, the G-test), <c>"freeman-tukey"</c> (−½), <c>"mod-log-likelihood"</c> (−1),
    /// <c>"neyman"</c> (−2) or <c>"cressie-read"</c> (⅔).
    /// </summary>
    public static SignificanceResult PowerDivergence(double[] fObs, double[]? fExp, int ddof, string lambda, bool sumCheck = true)
        => PowerDivergence(fObs, fExp, ddof, PowerDivergenceLambda(lambda), sumCheck);

    /// <summary>Power-divergence test with a numeric λ (1 is Pearson's χ²).</summary>
    public static SignificanceResult PowerDivergence(double[] fObs, double[]? fExp = null, int ddof = 0, double lambda = 1.0, bool sumCheck = true)
    {
        ArgumentNullException.ThrowIfNull(fObs);
        var k = fObs.Length;
        var exp = new double[k];
        if (fExp is null)
        {
            var m = StatsTestMath.Mean(fObs);
            for (var i = 0; i < k; i++) exp[i] = m;
        }
        else
        {
            if (fExp.Length != k) throw new ArgumentException("f_exp must have the same length as f_obs.", nameof(fExp));
            Array.Copy(fExp, exp, k);
            if (sumCheck)
            {
                double so = StatsTestMath.Sum(fObs), se = StatsTestMath.Sum(exp);
                var rel = Math.Abs(so - se) / Math.Min(so, se);
                if (rel > Math.Sqrt(Math.Pow(2, -52)))
                    throw new ArgumentException($"For each axis slice, the sum of the observed frequencies must agree with the sum of the expected frequencies to a relative tolerance of {Math.Sqrt(Math.Pow(2, -52))}, but the percent differences are: {rel}");
            }
        }
        var stat = PowerDivergenceStatistic(fObs, exp, lambda);
        return new SignificanceResult(stat, StatsTestMath.Chi2Sf(stat, k - 1 - ddof));
    }

    private static double PowerDivergenceLambda(string name) => name.ToLowerInvariant() switch
    {
        "pearson" => 1.0,
        "log-likelihood" => 0.0,
        "freeman-tukey" => -0.5,
        "mod-log-likelihood" => -1.0,
        "neyman" => -2.0,
        "cressie-read" => 2.0 / 3.0,
        _ => throw new ArgumentException($"invalid string for lambda_: '{name}'.", nameof(name)),
    };

    private static double PowerDivergenceStatistic(ReadOnlySpan<double> obs, ReadOnlySpan<double> exp, double lambda)
    {
        var terms = new double[obs.Length];
        for (var i = 0; i < obs.Length; i++)
        {
            double o = obs[i], e = exp[i];
            terms[i] = lambda switch
            {
                1.0 => (o - e) * (o - e) / e,
                0.0 => 2.0 * Special.Xlogy(o, o / e),
                -1.0 => 2.0 * Special.Xlogy(e, e / o),
                _ => o * (Math.Pow(o / e, lambda) - 1) / (0.5 * lambda * (lambda + 1)),
            };
        }
        return StatsTestMath.Sum(terms);
    }

    // ------------------------------------------------------------------ contingency tables

    /// <summary>
    /// Marginal sums of an n-dimensional table, scipy's <c>contingency.margins</c>: entry i holds the
    /// sums over every axis but i, as a flat array of length shape[i].
    /// </summary>
    public static double[][] Margins(NdArray table)
    {
        ArgumentNullException.ThrowIfNull(table);
        var shape = table.Shape.ToArray();
        var data = table.AsContiguous().ToArray();
        var d = shape.Length;
        var m = new double[d][];
        for (var i = 0; i < d; i++) m[i] = new double[shape[i]];
        var idx = new int[d];
        for (var f = 0; f < data.Length; f++)
        {
            var rem = f;
            for (var i = d - 1; i >= 0; i--) { idx[i] = rem % shape[i]; rem /= shape[i]; }
            for (var i = 0; i < d; i++) m[i][idx[i]] += data[f];
        }
        return m;
    }

    /// <summary>
    /// Expected frequencies under independence, scipy's <c>contingency.expected_freq</c>: the product
    /// of the margins divided by total^(d−1).
    /// </summary>
    public static NdArray ExpectedFreq(NdArray observed)
    {
        ArgumentNullException.ThrowIfNull(observed);
        var shape = observed.Shape.ToArray();
        var d = shape.Length;
        var margins = Margins(observed);
        var total = margins[0].Sum();
        var size = observed.Size;
        var r = new double[size];
        var idx = new int[d];
        var denom = Math.Pow(total, d - 1);
        for (var f = 0; f < size; f++)
        {
            var rem = f;
            for (var i = d - 1; i >= 0; i--) { idx[i] = rem % shape[i]; rem /= shape[i]; }
            var p = 1.0;
            for (var i = 0; i < d; i++) p *= margins[i][idx[i]];
            r[f] = p / denom;
        }
        return new NdArray(r, shape);
    }

    /// <summary>
    /// Chi-square test of independence for an n-dimensional contingency table, scipy's
    /// <c>chi2_contingency</c>. Yates' continuity correction is applied only when dof = 1 (and
    /// <paramref name="correction"/> is true); <paramref name="lambda"/> selects a power-divergence statistic.
    /// </summary>
    public static Chi2ContingencyResult Chi2Contingency(NdArray observed, bool correction = true, double? lambda = null)
    {
        ArgumentNullException.ThrowIfNull(observed);
        var obs = observed.AsContiguous().ToArray();
        if (obs.Any(v => v < 0)) throw new ArgumentException("All values in `observed` must be nonnegative.", nameof(observed));
        if (obs.Length == 0) throw new ArgumentException("No data; `observed` has size 0.", nameof(observed));
        var expected = ExpectedFreq(observed);
        var exp = expected.AsContiguous().ToArray();
        if (exp.Any(v => v == 0)) throw new ArgumentException("The internally computed table of expected frequencies has a zero element.", nameof(observed));
        var shape = observed.Shape.ToArray();
        var dof = exp.Length - shape.Sum() + shape.Length - 1;
        if (dof == 0) return new Chi2ContingencyResult(0.0, 1.0, 0, expected);
        if (dof == 1 && correction)
        {
            obs = (double[])obs.Clone();
            for (var i = 0; i < obs.Length; i++)
            {
                var diff = exp[i] - obs[i];
                obs[i] += Math.Min(0.5, Math.Abs(diff)) * Math.Sign(diff);
            }
        }
        var stat = PowerDivergenceStatistic(obs, exp, lambda ?? 1.0);
        return new Chi2ContingencyResult(stat, StatsTestMath.Chi2Sf(stat, dof), dof, expected);
    }

    /// <summary>
    /// Strength of association in a 2-D table, scipy's <c>contingency.association</c>:
    /// <c>"cramer"</c> (Cramér's V), <c>"tschuprow"</c> (Tschuprow's T) or <c>"pearson"</c> (contingency coefficient).
    /// </summary>
    public static double Association(NdArray observed, string method = "cramer", bool correction = false, double? lambda = null)
    {
        ArgumentNullException.ThrowIfNull(observed);
        if (observed.Rank != 2) throw new ArgumentException("method only accepts 2d arrays", nameof(observed));
        var chi2 = Chi2Contingency(observed, correction, lambda).Statistic;
        var total = observed.AsContiguous().ToArray().Sum();
        var phi2 = chi2 / total;
        int r = observed.Shape[0], c = observed.Shape[1];
        var v = method.ToLowerInvariant() switch
        {
            "cramer" => phi2 / Math.Min(c - 1, r - 1),
            "tschuprow" => phi2 / Math.Sqrt((r - 1.0) * (c - 1.0)),
            "pearson" => phi2 / (1 + phi2),
            _ => throw new ArgumentException("method must be 'cramer', 'tschuprow', or 'pearson'", nameof(method)),
        };
        return Math.Sqrt(v);
    }

    /// <summary>
    /// Cross-tabulation of paired label sequences, scipy's <c>contingency.crosstab</c>: the sorted unique
    /// values of each sequence and the table of counts (shape = number of uniques per sequence).
    /// </summary>
    public static (double[][] Elements, NdArray Count) Crosstab(params double[][] args)
    {
        ArgumentNullException.ThrowIfNull(args);
        if (args.Length == 0) throw new ArgumentException("At least one input sequence is required.", nameof(args));
        var n = args[0].Length;
        foreach (var a in args) if (a.Length != n) throw new ArgumentException("All input sequences must have the same length.", nameof(args));
        var d = args.Length;
        var elements = new double[d][];
        var shape = new int[d];
        for (var i = 0; i < d; i++)
        {
            elements[i] = args[i].Distinct().OrderBy(v => v).ToArray();
            shape[i] = elements[i].Length;
        }
        var size = shape.Aggregate(1, (a, b) => a * b);
        var count = new double[size];
        for (var j = 0; j < n; j++)
        {
            var f = 0;
            for (var i = 0; i < d; i++) f = f * shape[i] + Array.BinarySearch(elements[i], args[i][j]);
            count[f]++;
        }
        return (elements, new NdArray(count, shape));
    }

    // ------------------------------------------------------------------ exact tests for 2×2 tables

    private static (long A, long B, long C, long D) Table2x2(NdArray table)
    {
        ArgumentNullException.ThrowIfNull(table);
        if (table.Rank != 2 || table.Shape[0] != 2 || table.Shape[1] != 2) throw new ArgumentException("The input `table` must be of shape (2, 2).", nameof(table));
        var t = table.AsContiguous().ToArray();
        foreach (var v in t)
        {
            if (v < 0) throw new ArgumentException("All values in `table` must be nonnegative.", nameof(table));
            if (v != Math.Floor(v)) throw new ArgumentException("The table must hold integer counts.", nameof(table));
        }
        return ((long)t[0], (long)t[1], (long)t[2], (long)t[3]);
    }

    /// <summary>
    /// Fisher's exact test on a 2×2 table, scipy's <c>fisher_exact</c>. The statistic is the sample odds
    /// ratio; the two-sided p-value sums the tables no more probable than the observed one (with scipy's
    /// 1e-14 relative tolerance for "equally probable").
    /// </summary>
    public static SignificanceResult FisherExact(NdArray table, Alternative alternative = Alternative.TwoSided)
    {
        var (a, b, c, d) = Table2x2(table);
        if (a + c == 0 || b + d == 0 || a + b == 0 || c + d == 0) return new SignificanceResult(double.NaN, 1.0);
        var oddsratio = c > 0 && b > 0 ? (double)(a * d) / (c * b) : double.PositiveInfinity;
        var n1 = a + b;
        var n2 = c + d;
        var n = a + c;
        var hg = new HypergeometricDistribution((int)(n1 + n2), (int)n1, (int)n);
        double pvalue;
        switch (alternative)
        {
            case Alternative.Less:
                pvalue = hg.Cdf(a);
                break;
            case Alternative.Greater:
                pvalue = new HypergeometricDistribution((int)(n1 + n2), (int)n1, (int)(b + d)).Cdf(b);
                break;
            default:
            {
                var mode = (long)((n + 1) * (n1 + 1) / (double)(n1 + n2 + 2));
                var pexact = hg.Density(a);
                var pmode = hg.Density(mode);
                const double epsilon = 1e-14;
                var gamma = 1 + epsilon;
                if (Math.Abs(pexact - pmode) / Math.Max(pexact, pmode) <= epsilon) return new SignificanceResult(oddsratio, 1.0);
                if (a < mode)
                {
                    var plower = hg.Cdf(a);
                    if (hg.Density(n) > pexact * gamma) return new SignificanceResult(oddsratio, plower);
                    var guess = BinarySearchPmf(x => -hg.Density(x), -pexact * gamma, mode, n);
                    pvalue = plower + hg.Sf(guess);
                }
                else
                {
                    var pupper = hg.Sf(a - 1);
                    if (hg.Density(0) > pexact * gamma) return new SignificanceResult(oddsratio, pupper);
                    var guess = BinarySearchPmf(x => hg.Density(x), pexact * gamma, 0, mode);
                    pvalue = pupper + hg.Cdf(guess);
                }
                break;
            }
        }
        return new SignificanceResult(oddsratio, Math.Min(pvalue, 1.0));
    }

    /// <summary>scipy's <c>_binary_search_for_binom_tst</c>: the last x in [lo, hi] with a(x) ≤ d for monotone a.</summary>
    internal static long BinarySearchPmf(Func<long, double> a, double d, long lo, long hi)
    {
        while (lo < hi)
        {
            var mid = lo + (hi - lo) / 2;
            var midval = a(mid);
            if (midval < d) lo = mid + 1;
            else if (midval > d) hi = mid - 1;
            else return mid;
        }
        return a(lo) <= d ? lo : lo - 1;
    }

    /// <summary>
    /// Barnard's unconditional exact test on a 2×2 table (columns are the two binomial samples), scipy's
    /// <c>barnard_exact</c>. The statistic is the Wald statistic (pooled or unpooled variance); the p-value
    /// is maximised over the nuisance success probability. scipy uses SHGO with 32 Sobol points; this
    /// scans a dense grid and polishes every local maximum with Brent, so it finds the same supremum or
    /// a slightly larger one when SHGO's sampling misses a narrow peak.
    /// </summary>
    public static SignificanceResult BarnardExact(NdArray table, Alternative alternative = Alternative.TwoSided, bool pooled = true)
    {
        var (a, b, c, d) = Table2x2(table);
        long c1 = a + c, c2 = b + d;
        if (c1 == 0 || c2 == 0) return new SignificanceResult(double.NaN, 1.0);
        var wald = new double[c1 + 1, c2 + 1];
        for (var x1 = 0L; x1 <= c1; x1++)
            for (var x2 = 0L; x2 <= c2; x2++)
            {
                double p1 = (double)x1 / c1, p2 = (double)x2 / c2;
                double v;
                if (pooled)
                {
                    var p = (double)(x1 + x2) / (c1 + c2);
                    v = p * (1 - p) * (1.0 / c1 + 1.0 / c2);
                }
                else v = p1 * (1 - p1) / c1 + p2 * (1 - p2) / c2;
                wald[x1, x2] = p1 == p2 ? 0 : (p1 - p2) / Math.Sqrt(v);
            }
        var obs = wald[a, b];
        var index = new bool[c1 + 1, c2 + 1];
        for (var x1 = 0; x1 <= c1; x1++)
            for (var x2 = 0; x2 <= c2; x2++)
            {
                var w = wald[x1, x2];
                index[x1, x2] = alternative switch
                {
                    Alternative.Less => w <= obs,
                    Alternative.Greater => w >= obs,
                    _ => Math.Abs(w) >= Math.Abs(obs),
                };
            }
        return new SignificanceResult(obs, MaximiseNuisance(c1, c2, index));
    }

    /// <summary>
    /// Boschloo's exact test on a 2×2 table, scipy's <c>boschloo_exact</c>: Fisher's one-sided p-value used
    /// as the ordering statistic of an unconditional test (uniformly more powerful than Fisher's).
    /// Two-sided is twice the smaller one-sided p-value, as scipy.
    /// </summary>
    public static SignificanceResult BoschlooExact(NdArray table, Alternative alternative = Alternative.TwoSided)
    {
        var (a, b, c, d) = Table2x2(table);
        long c1 = a + c, c2 = b + d;
        if (c1 == 0 || c2 == 0) return new SignificanceResult(double.NaN, double.NaN);
        if (alternative == Alternative.TwoSided)
        {
            var less = BoschlooExact(table, Alternative.Less);
            var greater = BoschlooExact(table, Alternative.Greater);
            var r = less.Pvalue < greater.Pvalue ? less : greater;
            return new SignificanceResult(r.Statistic, Math.Clamp(2 * r.Pvalue, 0, 1));
        }
        var total = c1 + c2;
        var pv = new double[c1 + 1, c2 + 1];
        for (var x1 = 0; x1 <= c1; x1++)
            for (var x2 = 0; x2 <= c2; x2++)
            {
                var hg = new HypergeometricDistribution((int)total, x1 + x2, alternative == Alternative.Less ? (int)c1 : (int)c2);
                pv[x1, x2] = hg.Cdf(alternative == Alternative.Less ? x1 : x2);
            }
        var fisherStat = pv[a, b];
        var index = new bool[c1 + 1, c2 + 1];
        for (var x1 = 0; x1 <= c1; x1++)
            for (var x2 = 0; x2 <= c2; x2++) index[x1, x2] = pv[x1, x2] <= fisherStat * (1 + 1e-13);
        return new SignificanceResult(fisherStat, MaximiseNuisance(c1, c2, index));
    }

    /// <summary>
    /// sup over π of Σ_{(x1,x2) ∈ region} C(c1,x1) C(c2,x2) π^{x1+x2} (1−π)^{n−x1−x2}, evaluated in logs.
    /// </summary>
    private static double MaximiseNuisance(long c1, long c2, bool[,] index)
    {
        var lc1 = LogComb(c1);
        var lc2 = LogComb(c2);
        var n = c1 + c2;
        double LogP(double pi)
        {
            var lp = pi > 0 ? Math.Log(pi) : 0;
            var lq = 1 - pi > 0 ? Math.Log(1 - pi) : 0;
            var max = double.NegativeInfinity;
            var vals = new List<double>();
            for (var x1 = 0; x1 <= c1; x1++)
                for (var x2 = 0; x2 <= c2; x2++)
                {
                    if (!index[x1, x2]) continue;
                    var s = x1 + x2;
                    var v = lc1[x1] + lc2[x2] + (s == 0 ? 0 : lp * s) + (s == n ? 0 : lq * (n - s));
                    if (pi == 0 && s > 0 || pi == 1 && s < n) v = double.NegativeInfinity;
                    vals.Add(v);
                    if (v > max) max = v;
                }
            if (double.IsNegativeInfinity(max)) return max;
            var acc = 0.0;
            foreach (var v in vals) acc += Math.Exp(v - max);
            return max + Math.Log(acc);
        }
        const int grid = 400;
        var g = new double[grid + 1];
        for (var i = 0; i <= grid; i++) g[i] = LogP((double)i / grid);
        var best = g.Max();
        for (var i = 0; i <= grid; i++)
        {
            var left = i > 0 ? g[i - 1] : double.NegativeInfinity;
            var right = i < grid ? g[i + 1] : double.NegativeInfinity;
            if (g[i] < left || g[i] < right) continue;
            double lo = Math.Max(0, (i - 1.0) / grid), hi = Math.Min(1, (i + 1.0) / grid);
            // Golden-section refinement of the local maximum.
            const double gr = 0.6180339887498949;
            double x1 = hi - gr * (hi - lo), x2 = lo + gr * (hi - lo);
            double f1 = LogP(x1), f2 = LogP(x2);
            for (var it = 0; it < 80 && hi - lo > 1e-12; it++)
            {
                if (f1 < f2) { lo = x1; x1 = x2; f1 = f2; x2 = lo + gr * (hi - lo); f2 = LogP(x2); }
                else { hi = x2; x2 = x1; f2 = f1; x1 = hi - gr * (hi - lo); f1 = LogP(x1); }
            }
            best = Math.Max(best, Math.Max(f1, f2));
        }
        return Math.Clamp(Math.Exp(best), 0, 1);
    }

    private static double[] LogComb(long n)
    {
        var r = new double[n + 1];
        var gn = Special.GammaLn(n + 1.0);
        for (var k = 0; k <= n; k++) r[k] = gn - Special.GammaLn(k + 1.0) - Special.GammaLn(n - k + 1.0);
        return r;
    }

    // ------------------------------------------------------------------ binomial and Poisson tests

    /// <summary>
    /// Exact binomial test of H₀: P(success) = <paramref name="p"/>, scipy's <c>binomtest</c>. The
    /// two-sided p-value sums every outcome no more probable than k (relative tolerance 1e-7).
    /// </summary>
    public static BinomTestResult Binomtest(int k, int n, double p = 0.5, Alternative alternative = Alternative.TwoSided)
    {
        if (n < 1) throw new ArgumentException("n must be at least 1.", nameof(n));
        if (k < 0 || k > n) throw new ArgumentException($"k ({k}) must lie in [0, n].", nameof(k));
        if (!(p >= 0 && p <= 1)) throw new ArgumentException($"p ({p}) must be in range [0,1]", nameof(p));
        var bin = new BinomialDistribution(n, p);
        double pval;
        switch (alternative)
        {
            case Alternative.Less: pval = bin.Cdf(k); break;
            case Alternative.Greater: pval = bin.Sf(k - 1); break;
            default:
            {
                var d = bin.Density(k);
                const double rerr = 1 + 1e-7;
                if (k == p * n) pval = 1.0;
                else if (k < p * n)
                {
                    var ix = BinarySearchPmf(x => -bin.Density(x), -d * rerr, (long)Math.Ceiling(p * n), n);
                    var y = n - ix + (d * rerr == bin.Density(ix) ? 1 : 0);
                    pval = bin.Cdf(k) + bin.Sf(n - y);
                }
                else
                {
                    var ix = BinarySearchPmf(x => bin.Density(x), d * rerr, 0, (long)Math.Floor(p * n));
                    var y = ix + 1;
                    pval = bin.Cdf(y - 1) + bin.Sf(k - 1);
                }
                pval = Math.Min(1.0, pval);
                break;
            }
        }
        return new BinomTestResult(k, n, alternative, pval);
    }

    /// <summary>
    /// Exact test that two Poisson rates differ by <paramref name="diff"/> (Krishnamoorthy &amp; Thomson's
    /// E-test), scipy's <c>poisson_means_test</c>; k events observed over n units of exposure in each sample.
    /// </summary>
    public static SignificanceResult PoissonMeansTest(int k1, double n1, int k2, double n2, double diff = 0, Alternative alternative = Alternative.TwoSided)
    {
        if (k1 < 0 || k2 < 0) throw new ArgumentException("`k1` and `k2` must be greater than or equal to 0.");
        if (n1 <= 0 || n2 <= 0) throw new ArgumentException("`n1` and `n2` must be greater than 0.");
        if (diff < 0) throw new ArgumentException("diff must be greater than or equal to 0.", nameof(diff));
        var l2 = (k1 + k2) / (n1 + n2) - diff * n1 / (n1 + n2);
        if (l2 <= 0) return new SignificanceResult(0, 1);
        var var = k1 / (n1 * n1) + k2 / (n2 * n2);
        var t = (k1 / n1 - k2 / n2 - diff) / Math.Sqrt(var);
        var nl1 = n1 * (l2 + diff);
        var nl2 = n2 * l2;
        var po1 = new PoissonDistribution(nl1);
        var po2 = new PoissonDistribution(nl2);
        int lb1 = (int)po1.Ppf(1e-10), ub1 = (int)po1.Ppf(1 - 1e-16);
        int lb2 = (int)po2.Ppf(1e-10), ub2 = (int)po2.Ppf(1 - 1e-16);
        var pr1 = new double[ub1 - lb1 + 1];
        for (var i = 0; i < pr1.Length; i++) pr1[i] = po1.Density(lb1 + i);
        var p = 0.0;
        for (var x2 = lb2; x2 <= ub2; x2++)
        {
            var px2 = po2.Density(x2);
            var lx2 = x2 / n2;
            for (var x1 = lb1; x1 <= ub1; x1++)
            {
                var lx1 = x1 / n1;
                var tx = (lx1 - lx2 - diff) / Math.Sqrt(lx1 / n1 + lx2 / n2);
                var hit = alternative switch
                {
                    Alternative.Less => tx <= t,
                    Alternative.Greater => tx >= t,
                    _ => Math.Abs(tx) >= Math.Abs(t),
                };
                if (hit) p += pr1[x1 - lb1] * px2;
            }
        }
        return new SignificanceResult(t, p);
    }
}

/// <summary>Confidence intervals for a binomial proportion (scipy's <c>BinomTestResult.proportion_ci</c>).</summary>
internal static class StatsContingencyKernels
{
    public static ConfidenceInterval BinomCi(int k, int n, Alternative alternative, double confidenceLevel, string method)
    {
        if (!(confidenceLevel >= 0 && confidenceLevel <= 1)) throw new ArgumentException("confidence_level must be in the interval [0, 1].", nameof(confidenceLevel));
        switch (method.ToLowerInvariant())
        {
            case "exact":
            {
                // Clopper–Pearson limits are beta quantiles; scipy root-finds the same equations.
                double lo, hi;
                var alpha = alternative == Alternative.TwoSided ? (1 - confidenceLevel) / 2 : 1 - confidenceLevel;
                lo = alternative == Alternative.Less || k == 0 ? 0.0 : Special.BetaIncInv(k, n - k + 1, alpha);
                hi = alternative == Alternative.Greater || k == n ? 1.0 : Special.BetaIncInv(k + 1, n - k, 1 - alpha);
                return new ConfidenceInterval(lo, hi);
            }
            case "wilson":
            case "wilsoncc":
            {
                var correction = method.Equals("wilsoncc", StringComparison.OrdinalIgnoreCase);
                var p = (double)k / n;
                var z = alternative == Alternative.TwoSided ? Special.Ndtri(0.5 + 0.5 * confidenceLevel) : Special.Ndtri(confidenceLevel);
                var denom = 2 * (n + z * z);
                var center = (2 * n * p + z * z) / denom;
                var q = 1 - p;
                double lo, hi;
                if (correction)
                {
                    lo = alternative == Alternative.Less || k == 0 ? 0.0
                        : center - (1 + z * Math.Sqrt(z * z - 2 - 1.0 / n + 4 * p * (n * q + 1))) / denom;
                    hi = alternative == Alternative.Greater || k == n ? 1.0
                        : center + (1 + z * Math.Sqrt(z * z + 2 - 1.0 / n + 4 * p * (n * q - 1))) / denom;
                }
                else
                {
                    var delta = z / denom * Math.Sqrt(4 * n * p * q + z * z);
                    lo = alternative == Alternative.Less || k == 0 ? 0.0 : center - delta;
                    hi = alternative == Alternative.Greater || k == n ? 1.0 : center + delta;
                }
                return new ConfidenceInterval(lo, hi);
            }
            default:
                throw new ArgumentException("method must be one of 'exact', 'wilson' or 'wilsoncc'.", nameof(method));
        }
    }
}
