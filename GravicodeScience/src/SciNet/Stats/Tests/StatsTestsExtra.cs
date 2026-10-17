using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>Result of <see cref="Stats.QuantileTest"/> (scipy's <c>QuantileTestResult</c>).</summary>
public sealed record QuantileTestResult : SignificanceResult
{
    private readonly double[] _x;
    private readonly double _p;
    private readonly Alternative _alternative;

    internal QuantileTestResult(double statistic, int statisticType, double pvalue, double[] x, double p, Alternative alternative) : base(statistic, pvalue)
    {
        StatisticType = statisticType;
        _x = x;
        _p = p;
        _alternative = alternative;
    }

    /// <summary>1 when the statistic is T1 = #(x ≤ q), 2 when it is T2 = #(x &lt; q).</summary>
    public int StatisticType { get; }

    /// <summary>
    /// Distribution-free confidence interval for the p-th quantile from order statistics (Conover 1999),
    /// scipy's <c>confidence_interval</c>; a bound is NaN when the sample is too small to provide it.
    /// </summary>
    public ConfidenceInterval ConfidenceInterval(double confidenceLevel = 0.95)
    {
        if (!(confidenceLevel > 0 && confidenceLevel < 1)) throw new ArgumentException("confidenceLevel must be a number between 0 and 1.", nameof(confidenceLevel));
        var x = StatsTestMath.Sorted(_x);
        var n = x.Length;
        var bd = new BinomialDistribution(n, _p);
        double low = double.NegativeInfinity, high = double.PositiveInfinity;
        var a = _alternative == Alternative.TwoSided ? (1 - confidenceLevel) / 2 : 1 - confidenceLevel;
        if (_alternative != Alternative.Greater)
        {
            var hi = (int)bd.Isf(a);
            high = hi < n ? x[hi] : double.NaN;
        }
        if (_alternative != Alternative.Less)
        {
            var lo = (int)bd.Ppf(a) - 1;
            low = lo >= 0 ? x[lo] : double.NaN;
        }
        return new ConfidenceInterval(low, high);
    }
}

/// <summary>Result of <see cref="Stats.GoodnessOfFit(ContinuousDistribution, double[], IReadOnlyDictionary{string, double}?, IReadOnlyDictionary{string, double}?, IReadOnlyDictionary{string, double}?, string, int, int?, bool)"/> (scipy's <c>GoodnessOfFitResult</c>).</summary>
/// <param name="FitResult">The distribution fitted under the null hypothesis (known and fit parameters fixed).</param>
/// <param name="Statistic">The statistic comparing the data with <paramref name="FitResult"/>.</param>
/// <param name="Pvalue">Monte Carlo p-value, (count + 1)/(B + 1).</param>
/// <param name="NullDistribution">The statistic on each simulated (and refitted) sample.</param>
public sealed record GoodnessOfFitResult(ContinuousDistribution FitResult, double Statistic, double Pvalue, double[] NullDistribution);

/// <summary>scipy.stats tests added in phase 2: BWS, quantile test, Monte Carlo goodness of fit, PPCC and mvsdist.</summary>
public static partial class Stats
{
    // ------------------------------------------------------------------ Baumgartner–Weiss–Schindler

    /// <summary>
    /// Baumgartner–Weiss–Schindler two-sample test (Neuhäuser 2005), scipy's <c>bws_test</c>: a
    /// permutation test of the BWS statistic on the pooled ranks. Exact when every partition fits in
    /// <paramref name="nResamples"/> (as scipy's <c>PermutationMethod</c>), random permutations otherwise.
    /// One-sided alternatives use the signed statistic of Neuhäuser; <c>Less</c> means x tends to be smaller.
    /// </summary>
    public static ResamplingTestResult BwsTest(double[] x, double[] y, Alternative alternative = Alternative.TwoSided, int nResamples = 9999, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        if (x.Length == 0 || y.Length == 0) throw new ArgumentException("x and y must be of nonzero size.");
        foreach (var v in x.Concat(y)) if (double.IsNaN(v)) throw new ArgumentException("x and y must not contain NaNs.");
        var z = Rankdata([.. x, .. y]);
        var rx = z[..x.Length];
        var ry = z[x.Length..];
        var permAlt = alternative == Alternative.Less ? Alternative.Less : Alternative.Greater;
        return PermutationTest(rx, ry, (a, b) => BwsStatistic(a, b, alternative), nResamples: nResamples, alternative: permAlt, seed: seed);
    }

    private static double BwsStatistic(double[] x, double[] y, Alternative alternative)
    {
        var ri = StatsTestMath.Sorted(x);
        var hj = StatsTestMath.Sorted(y);
        double n = ri.Length, m = hj.Length;
        var twoSided = alternative == Alternative.TwoSided;
        double bx = 0, by = 0;
        for (var i = 1; i <= ri.Length; i++)
        {
            var num = ri[i - 1] - (m + n) / n * i;
            num *= twoSided ? num : Math.Abs(num);
            bx += num / (i / (n + 1) * (1 - i / (n + 1)) * m * (m + n) / n);
        }
        for (var j = 1; j <= hj.Length; j++)
        {
            var num = hj[j - 1] - (m + n) / m * j;
            num *= twoSided ? num : Math.Abs(num);
            by += num / (j / (m + 1) * (1 - j / (m + 1)) * n * (m + n) / m);
        }
        bx /= n;
        by /= m;
        return twoSided ? (bx + by) / 2 : (bx - by) / 2;
    }

    // ------------------------------------------------------------------ quantile test

    /// <summary>
    /// Test that <paramref name="q"/> is the <paramref name="p"/>-th population quantile, scipy's
    /// <c>quantile_test</c> (Conover 1999): binomial tests of T1 = #(x ≤ q) and T2 = #(x &lt; q).
    /// <c>Less</c> is the alternative that the quantile is below q.
    /// </summary>
    public static QuantileTestResult QuantileTest(double[] x, double q = 0, double p = 0.5, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (!(p > 0 && p < 1)) throw new ArgumentException("p must be a float strictly between 0 and 1.", nameof(p));
        if (!double.IsFinite(q)) throw new ArgumentException("q must be a finite scalar.", nameof(q));
        int t1 = 0, t2 = 0;
        foreach (var v in x) { if (v <= q) t1++; if (v < q) t2++; }
        var y = new BinomialDistribution(x.Length, p);
        double stat, pv;
        int type;
        switch (alternative)
        {
            case Alternative.Less: pv = y.Sf(t2 - 1); stat = t2; type = 2; break;
            case Alternative.Greater: pv = y.Cdf(t1); stat = t1; type = 1; break;
            default:
                var pg = y.Cdf(t1);
                var pl = y.Sf(t2 - 1);
                // numpy argsort is stable: ties keep "greater" (T1) first.
                if (pl < pg) { pv = 2 * pl; stat = t2; type = 2; }
                else { pv = 2 * pg; stat = t1; type = 1; }
                pv = Math.Clamp(pv, 0, 1);
                break;
        }
        return new QuantileTestResult(stat, type, pv, (double[])x.Clone(), p, alternative);
    }

    // ------------------------------------------------------------------ mvsdist

    /// <summary>
    /// Frozen posterior distributions of the mean, variance and standard deviation under Jeffreys'
    /// prior, scipy's <c>mvsdist</c>: Student t, inverse gamma and generalised gamma (c = −2), or
    /// normal approximations above 1000 points. <see cref="BayesMvs"/> reports their means and intervals.
    /// </summary>
    public static (ContinuousDistribution Mean, ContinuousDistribution Var, ContinuousDistribution Std) Mvsdist(double[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        var n = data.Length;
        if (n < 2) throw new ArgumentException("Need at least 2 data-points.", nameof(data));
        var xbar = data.Average();
        var c = 0.0;
        foreach (var v in data) c += (v - xbar) * (v - xbar);
        c /= n;
        if (n > 1000)
            return (new NormalDistribution(xbar, Math.Sqrt(c / n)), new NormalDistribution(c, Math.Sqrt(2.0 / n) * c),
                    new NormalDistribution(Math.Sqrt(c), Math.Sqrt(c / (2.0 * n))));
        var nm1 = n - 1;
        var fac = n * c / 2.0;
        var val = nm1 / 2.0;
        return (new StudentTDistribution(nm1, xbar, Math.Sqrt(c / nm1)), new InvGammaDistribution(val, 0, fac),
                new GenGammaDistribution(val, -2, 0, Math.Sqrt(fac)));
    }

    // ------------------------------------------------------------------ PPCC

    /// <summary>
    /// The shape parameter maximising the probability-plot correlation coefficient, scipy's
    /// <c>ppcc_max</c>: Brent's method on 1 − r from the bracket <paramref name="brack"/>.
    /// <paramref name="family"/> maps a shape to a distribution; the default is Tukey's lambda, as scipy.
    /// </summary>
    public static double PpccMax(double[] x, double[]? brack = null, Func<double, ContinuousDistribution>? family = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Length < 2) throw new ArgumentException("x needs at least two values.", nameof(x));
        family ??= s => new TukeyLambdaDistribution(s);
        var u = UniformOrderStatisticMedians(x.Length);
        var osr = StatsTestMath.Sorted(x);
        double F(double shape)
        {
            ContinuousDistribution d;
            try { d = family(shape); }
            catch (ArgumentException) { return double.PositiveInfinity; }
            var osm = new double[u.Length];
            for (var i = 0; i < u.Length; i++) osm[i] = d.Ppf(u[i]);
            return 1 - Corr(osm, osr);
        }
        return Optimize.Brent(F, brack ?? [0.0, 1.0]).X;
    }

    /// <summary>
    /// The probability-plot correlation coefficient at <paramref name="n"/> equally spaced shapes from
    /// <paramref name="a"/> to <paramref name="b"/>, scipy's <c>ppcc_plot</c> (data only; plot it yourself).
    /// </summary>
    public static (double[] Svals, double[] Ppcc) PpccPlot(double[] x, double a, double b, Func<double, ContinuousDistribution>? family = null, int n = 80)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (b <= a) throw new ArgumentException("b has to be larger than a.", nameof(b));
        if (n < 2) throw new ArgumentException("n must be at least 2.", nameof(n));
        family ??= s => new TukeyLambdaDistribution(s);
        var sv = new double[n];
        var r = new double[n];
        for (var i = 0; i < n; i++)
        {
            sv[i] = a + (b - a) * i / (n - 1);
            r[i] = Probplot(x, family(sv[i])).R;
        }
        return (sv, r);
    }

    /// <summary>
    /// Probability-plot correlation of the Box–Cox transformed data against the normal at <paramref name="n"/>
    /// λ values from <paramref name="la"/> to <paramref name="lb"/>, scipy's <c>boxcox_normplot</c> (data only).
    /// </summary>
    public static (double[] Lmbdas, double[] Ppcc) BoxcoxNormplot(double[] x, double la, double lb, int n = 80)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Length == 0) return ([], []);
        if (lb <= la) throw new ArgumentException("lb has to be larger than la.", nameof(lb));
        if (x.Any(v => v <= 0)) throw new ArgumentException("Data must be positive.", nameof(x));
        return NormplotCore(x, la, lb, n, Boxcox);
    }

    /// <summary>
    /// Probability-plot correlation of the Yeo–Johnson transformed data against the normal, scipy's
    /// <c>yeojohnson_normplot</c> (data only).
    /// </summary>
    public static (double[] Lmbdas, double[] Ppcc) YeojohnsonNormplot(double[] x, double la, double lb, int n = 80)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Length == 0) return ([], []);
        if (lb <= la) throw new ArgumentException("lb has to be larger than la.", nameof(lb));
        return NormplotCore(x, la, lb, n, Yeojohnson);
    }

    private static (double[], double[]) NormplotCore(double[] x, double la, double lb, int n, Func<double[], double, double[]> transform)
    {
        if (n < 2) throw new ArgumentException("n must be at least 2.", nameof(n));
        var l = new double[n];
        var r = new double[n];
        for (var i = 0; i < n; i++)
        {
            l[i] = la + (lb - la) * i / (n - 1);
            r[i] = Probplot(transform(x, l[i])).R;
        }
        return (l, r);
    }

    /// <summary>
    /// <paramref name="size"/> directions uniformly distributed on the unit sphere S^(dim−1), scipy's
    /// <c>uniform_direction.rvs</c>: normalised standard Gaussian vectors, as an (size, dim) array.
    /// </summary>
    public static NdArray UniformDirection(int dim, int size = 1, int? seed = null)
    {
        if (dim < 1) throw new ArgumentException("Dimension of vector must be specified, and must be an integer greater than 0.", nameof(dim));
        if (size < 0) throw new ArgumentException("size must be non-negative.", nameof(size));
        var rng = StatsDistributions.MakeRng(seed);
        var r = new double[size * dim];
        for (var s = 0; s < size; s++)
        {
            double norm;
            do
            {
                norm = 0;
                for (var i = 0; i < dim; i++) { var z = StatsRandom.Normal(rng); r[s * dim + i] = z; norm += z * z; }
            } while (norm == 0);
            norm = Math.Sqrt(norm);
            for (var i = 0; i < dim; i++) r[s * dim + i] /= norm;
        }
        return new NdArray(r, size, dim);
    }

    private static double Corr(double[] a, double[] b)
    {
        var n = a.Length;
        double ma = 0, mb = 0;
        for (var i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n;
        mb /= n;
        double sab = 0, saa = 0, sbb = 0;
        for (var i = 0; i < n; i++)
        {
            var da = a[i] - ma;
            var db = b[i] - mb;
            sab += da * db;
            saa += da * da;
            sbb += db * db;
        }
        return Math.Clamp(sab / Math.Sqrt(saa * sbb), -1, 1);
    }

    // ------------------------------------------------------------------ goodness_of_fit

    /// <summary>
    /// Monte Carlo goodness-of-fit test with parameter estimation, scipy's <c>goodness_of_fit</c>
    /// (the parametric bootstrap of Stephens / Lilliefors). The family's free parameters are fitted by
    /// maximum likelihood, samples are drawn from the fit, and every sample is <em>refitted</em> before
    /// its statistic is computed — which is what makes the p-value valid when parameters are estimated.
    /// </summary>
    /// <param name="family">The family; its shape values are the fit's starting point. For the normal the
    /// fit is scipy's <c>_fit_norm</c> (sample mean and the ddof = 1 standard deviation), not the MLE.</param>
    /// <param name="data">The sample.</param>
    /// <param name="knownParams">Parameters (by scipy name: shape names, <c>"loc"</c>, <c>"scale"</c>) that are known; never fitted.</param>
    /// <param name="fitParams">Parameters fixed when fitting the null distribution but refitted for each simulated sample (used as their starting guess).</param>
    /// <param name="guessedParams">Starting values for shape parameters (loc/scale guesses are not used: <see cref="ContinuousDistribution.FitTo(double[], double?[], double?, double?)"/> derives its own).</param>
    /// <param name="statistic"><c>"ad"</c> (Anderson–Darling, default), <c>"ks"</c>, <c>"cvm"</c> or <c>"filliben"</c>.</param>
    /// <param name="nMcSamples">Number of Monte Carlo samples.</param>
    /// <param name="seed">Seed.</param>
    /// <param name="parallel">Simulate on several threads (the result does not depend on it).</param>
    public static GoodnessOfFitResult GoodnessOfFit(ContinuousDistribution family, double[] data, IReadOnlyDictionary<string, double>? knownParams = null,
        IReadOnlyDictionary<string, double>? fitParams = null, IReadOnlyDictionary<string, double>? guessedParams = null, string statistic = "ad",
        int nMcSamples = 9999, int? seed = null, bool parallel = false)
    {
        var name = (statistic ?? "").ToLowerInvariant();
        Func<ContinuousDistribution, double[], double> compare = name switch
        {
            "ad" => GofAndersonDarling,
            "ks" => GofKolmogorovSmirnov,
            "cvm" => GofCramerVonMises,
            "filliben" => GofFilliben,
            _ => throw new ArgumentException("statistic must be one of {'ad', 'ks', 'cvm', 'filliben'}.", nameof(statistic)),
        };
        return GoodnessOfFit(family, data, compare, name == "filliben" ? Alternative.Less : Alternative.Greater,
            knownParams, fitParams, guessedParams, nMcSamples, seed, parallel);
    }

    /// <summary>
    /// <see cref="GoodnessOfFit(ContinuousDistribution, double[], IReadOnlyDictionary{string, double}?, IReadOnlyDictionary{string, double}?, IReadOnlyDictionary{string, double}?, string, int, int?, bool)"/>
    /// with a user statistic of (fitted distribution, sample); <paramref name="alternative"/> says which
    /// tail is evidence against the fit (<c>Greater</c> for distances, as scipy's default).
    /// </summary>
    public static GoodnessOfFitResult GoodnessOfFit(ContinuousDistribution family, double[] data, Func<ContinuousDistribution, double[], double> statistic,
        Alternative alternative = Alternative.Greater, IReadOnlyDictionary<string, double>? knownParams = null, IReadOnlyDictionary<string, double>? fitParams = null,
        IReadOnlyDictionary<string, double>? guessedParams = null, int nMcSamples = 9999, int? seed = null, bool parallel = false)
    {
        ArgumentNullException.ThrowIfNull(family);
        ArgumentNullException.ThrowIfNull(statistic);
        StatsDistributions.CheckData(data);
        if (nMcSamples < 1) throw new ArgumentException("nMcSamples must be a positive integer.", nameof(nMcSamples));
        var names = family.ShapeNames.Concat(["loc", "scale"]).ToArray();
        void CheckNames(IReadOnlyDictionary<string, double>? d, string arg)
        {
            if (d == null) return;
            foreach (var k in d.Keys)
                if (!names.Contains(k)) throw new ArgumentException($"Unknown parameter '{k}' for {family.Name}; expected one of {string.Join(", ", names)}.", arg);
        }
        CheckNames(knownParams, nameof(knownParams));
        CheckNames(fitParams, nameof(fitParams));
        CheckNames(guessedParams, nameof(guessedParams));
        var empty = new Dictionary<string, double>();
        knownParams ??= empty;
        fitParams ??= empty;
        guessedParams ??= empty;
        var nhdFixed = new Dictionary<string, double>(knownParams);
        foreach (var kv in fitParams) nhdFixed[kv.Key] = kv.Value;
        var rfdGuess = new Dictionary<string, double>(fitParams);
        foreach (var kv in guessedParams) rfdGuess[kv.Key] = kv.Value;
        var nhdFit = GofFitFunction(family, names, nhdFixed, guessedParams);
        var rfdFit = GofFitFunction(family, names, knownParams, rfdGuess);
        var nhd = nhdFit(data);
        var mc = MonteCarloTest([data], [(n, rng) => nhd.Rvs(n, rng)], d => statistic(rfdFit(d[0]), StatsTestMath.Sorted(d[0])),
            nMcSamples, alternative, seed, parallel);
        return new GoodnessOfFitResult(nhd, mc.Statistic, mc.Pvalue, mc.NullDistribution);
    }

    private static Func<double[], ContinuousDistribution> GofFitFunction(ContinuousDistribution family, string[] names,
        IReadOnlyDictionary<string, double> fixedParams, IReadOnlyDictionary<string, double> guessed)
    {
        var k = family.ShapeNames.Length;
        var shapes = (double[])family.Shapes.Clone();
        for (var i = 0; i < k; i++) if (guessed.TryGetValue(names[i], out var g)) shapes[i] = g;
        var fixedShapes = new double?[k];
        for (var i = 0; i < k; i++) fixedShapes[i] = fixedParams.TryGetValue(names[i], out var f) ? f : null;
        double? floc = fixedParams.TryGetValue("loc", out var l) ? l : null;
        double? fscale = fixedParams.TryGetValue("scale", out var s) ? s : null;
        if (fixedShapes.All(v => v.HasValue) && floc.HasValue && fscale.HasValue)
        {
            var allFixed = family.WithParameters(fixedShapes.Select(v => v!.Value).ToArray(), floc.Value, fscale.Value);
            return _ => allFixed;
        }
        if (family is NormalDistribution)
        {
            // scipy's _fit_norm: with both free, the scale is the ddof=1 standard deviation, not the MLE.
            return d =>
            {
                var m = floc ?? d.Average();
                double sc;
                if (fscale is double f) sc = f;
                else
                {
                    var ss = 0.0;
                    foreach (var v in d) ss += (v - m) * (v - m);
                    sc = floc.HasValue ? Math.Sqrt(ss / d.Length) : Math.Sqrt(ss / (d.Length - 1));
                }
                return new NormalDistribution(m, sc);
            };
        }
        var start = family.WithParameters(shapes, family.Loc, family.Scale);
        return d => start.FitTo(d, fixedShapes, floc, fscale);
    }

    /// <summary>A² of a sorted sample against a distribution, through log cdf/log sf as scipy's <c>_anderson_darling</c>.</summary>
    private static double GofAndersonDarling(ContinuousDistribution dist, double[] x)
    {
        var n = x.Length;
        var s = 0.0;
        for (var i = 1; i <= n; i++) s += (2.0 * i - 1) / n * (dist.LogCdf(x[i - 1]) + dist.LogSf(x[n - i]));
        return -n - s;
    }

    private static double GofKolmogorovSmirnov(ContinuousDistribution dist, double[] x)
    {
        var n = x.Length;
        double dp = double.NegativeInfinity, dm = double.NegativeInfinity;
        for (var i = 0; i < n; i++)
        {
            var c = dist.Cdf(x[i]);
            dp = Math.Max(dp, (i + 1.0) / n - c);
            dm = Math.Max(dm, c - (double)i / n);
        }
        return Math.Max(dp, dm);
    }

    private static double GofCramerVonMises(ContinuousDistribution dist, double[] x)
    {
        var n = x.Length;
        var w = 1.0 / (12 * n);
        for (var i = 1; i <= n; i++)
        {
            var d = (2.0 * i - 1) / (2 * n) - dist.Cdf(x[i - 1]);
            w += d * d;
        }
        return w;
    }

    /// <summary>Filliben's statistic: correlation of the sorted data with the fitted ppf at the order-statistic medians (Beta(k, n+1−k) medians).</summary>
    private static double GofFilliben(ContinuousDistribution dist, double[] x)
    {
        var n = x.Length;
        var m = new double[n];
        for (var k = 1; k <= n; k++) m[k - 1] = dist.Ppf(Special.BetaIncInv(k, n + 1 - k, 0.5));
        return Corr(x, m);
    }
}
