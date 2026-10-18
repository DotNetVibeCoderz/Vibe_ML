using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// Configuration of a resampling method passed as <c>method</c> to a hypothesis test or confidence
/// interval (scipy's <c>ResamplingMethod</c>).
/// </summary>
/// <param name="NResamples">Number of resamples (scipy's <c>n_resamples</c>, default 9999).</param>
/// <param name="Batch">scipy's <c>batch</c>: how many resamples are evaluated per vectorised call. It only
/// bounds memory in scipy; SciNet evaluates one resample at a time, so it is accepted and ignored.</param>
public abstract record ResamplingMethod(int NResamples = 9999, int? Batch = null)
{
    /// <summary>Seed of the pseudo-random generator (scipy's <c>rng</c>); null draws a fresh one.</summary>
    public int? Seed { get; init; }

    /// <summary>Evaluate the statistic on several threads (results do not depend on it).</summary>
    public bool Parallel { get; init; }

    internal void Validate()
    {
        if (NResamples < 1) throw new ArgumentException("n_resamples must be a positive integer.");
        if (Batch is { } b && b < 1) throw new ArgumentException("batch must be a positive integer or null.");
    }
}

/// <summary>
/// A permutation test (scipy's <c>PermutationMethod</c>): the null distribution is enumerated
/// exactly when every distinct permutation fits in <see cref="ResamplingMethod.NResamples"/>,
/// otherwise sampled.
/// </summary>
public sealed record PermutationMethod(int NResamples = 9999, int? Batch = null) : ResamplingMethod(NResamples, Batch);

/// <summary>
/// A Monte Carlo test (scipy's <c>MonteCarloMethod</c>): samples are drawn from the null by
/// <see cref="Rvs"/> — one generator per sample, called with the sample size — or, when that is
/// null, from the standard normal as scipy's default. <see cref="Rvs"/> and
/// <see cref="ResamplingMethod.Seed"/> may both be given here, since the generator receives the
/// seeded <see cref="GraviRandom"/>.
/// </summary>
public sealed record MonteCarloMethod(int NResamples = 9999, int? Batch = null, Func<int, GraviRandom, double[]>[]? Rvs = null)
    : ResamplingMethod(NResamples, Batch);

/// <summary>
/// A bootstrap confidence interval (scipy's <c>BootstrapMethod</c>), with the interval
/// <see cref="Method"/> <c>"BCa"</c> (default), <c>"percentile"</c> or <c>"basic"</c>.
/// </summary>
public sealed record BootstrapMethod(int NResamples = 9999, int? Batch = null, string Method = "BCa") : ResamplingMethod(NResamples, Batch);

public static partial class Stats
{
    private static Func<int, GraviRandom, double[]> NormalRvs => (n, rng) =>
    {
        var r = new double[n];
        for (var i = 0; i < n; i++) r[i] = StatsRandom.Normal(rng);
        return r;
    };

    private static Func<int, GraviRandom, double[]>[] RvsFor(MonteCarloMethod m, int samples)
    {
        if (m.Rvs is null) return Enumerable.Repeat(NormalRvs, samples).ToArray();
        if (m.Rvs.Length == 1 && samples > 1) return Enumerable.Repeat(m.Rvs[0], samples).ToArray();
        if (m.Rvs.Length != samples) throw new ArgumentException($"`rvs` must hold {samples} callables (one per sample).");
        return m.Rvs;
    }

    private static ResamplingTestResult Resample(ResamplingMethod method, double[][] data, Func<double[][], double> statistic,
        PermutationType permutationType, Alternative alternative, string function)
    {
        ArgumentNullException.ThrowIfNull(method);
        method.Validate();
        return method switch
        {
            PermutationMethod p => PermutationTest(data, statistic, permutationType, p.NResamples, alternative, p.Seed, p.Parallel),
            MonteCarloMethod mc => MonteCarloTest(data, RvsFor(mc, data.Length), statistic, mc.NResamples, alternative, mc.Seed, mc.Parallel),
            _ => throw new ArgumentException($"`method` of `{function}` must be a PermutationMethod or a MonteCarloMethod."),
        };
    }

    /// <summary>
    /// Pearson's r with a resampled p-value (scipy's <c>pearsonr(x, y, method=...)</c>): a
    /// <see cref="PermutationMethod"/> permutes the pairing of y with x; a <see cref="MonteCarloMethod"/>
    /// draws both samples from <see cref="MonteCarloMethod.Rvs"/> (standard normal by default).
    /// </summary>
    public static PearsonRResult Pearsonr(double[] x, double[] y, ResamplingMethod method, Alternative alternative = Alternative.TwoSided)
    {
        var r = Pearsonr(x, y, alternative);
        ResamplingTestResult res = method switch
        {
            PermutationMethod => Resample(method, [y], d => PearsonCoefficient(x, d[0]), PermutationType.Pairings, alternative, "pearsonr"),
            MonteCarloMethod => Resample(method, [x, y], d => PearsonCoefficient(d[0], d[1]), PermutationType.Independent, alternative, "pearsonr"),
            _ => throw new ArgumentException("`method` must be an instance of `PermutationMethod`, `MonteCarloMethod`, or null."),
        };
        return new PearsonRResult(res.Statistic, res.Pvalue, r.N, alternative) { X = x, Y = y };
    }

    /// <summary>
    /// Spearman's ρ with a resampled p-value (scipy's <c>spearmanrho(x, y, method=...)</c>): Pearson's r
    /// of the ranks, resampled as <see cref="Pearsonr(double[], double[], ResamplingMethod, Alternative)"/>.
    /// </summary>
    public static SignificanceResult Spearmanrho(double[] x, double[] y, ResamplingMethod method, Alternative alternative = Alternative.TwoSided)
    {
        var r = Pearsonr(Rankdata(x), Rankdata(y), method, alternative);
        return new SignificanceResult(r.Statistic, r.Pvalue);
    }

    /// <summary>
    /// The two-sample t-test with a resampled p-value (scipy's <c>ttest_ind(a, b, method=...)</c>): the
    /// statistic is the usual t (Student's, Welch's or Yuen's), its null distribution from permutations
    /// of the pooled sample or from Monte Carlo samples (standard normal by default). Degrees of
    /// freedom, estimate and standard error are those of the parametric test.
    /// </summary>
    public static TtestResult TtestInd(double[] a, double[] b, ResamplingMethod method, bool equalVar = true, Alternative alternative = Alternative.TwoSided, double trim = 0.0)
    {
        var t = TtestInd(a, b, equalVar, alternative, trim);
        if (double.IsNaN(t.Statistic)) return t;
        var res = Resample(method, [a, b], d => TtestInd(d[0], d[1], equalVar, alternative, trim).Statistic, PermutationType.Independent, alternative, "ttest_ind");
        return new TtestResult(t.Statistic, res.Pvalue, t.Df, t.Estimate, t.StandardError, alternative);
    }

    /// <summary>
    /// The Mann–Whitney U test with a permutation p-value (scipy's <c>mannwhitneyu(x, y, method=PermutationMethod())</c>).
    /// </summary>
    public static SignificanceResult MannWhitneyU(double[] x, double[] y, PermutationMethod method, bool useContinuity = true, Alternative alternative = Alternative.TwoSided)
    {
        var u = MannWhitneyU(x, y, useContinuity, alternative, "asymptotic");
        var res = Resample(method, [x, y], d => MannWhitneyU(d[0], d[1], useContinuity, alternative, "asymptotic").Statistic,
            PermutationType.Independent, alternative, "mannwhitneyu");
        return new SignificanceResult(u.Statistic, res.Pvalue);
    }

    /// <summary>
    /// The Wilcoxon signed-rank test with a permutation (sign-flip) p-value (scipy's
    /// <c>wilcoxon(x, y, method=PermutationMethod())</c>). The permuted statistic is R⁺.
    /// </summary>
    public static SignificanceResult Wilcoxon(double[] x, double[]? y, PermutationMethod method, string zeroMethod = "wilcox", Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(x);
        var asym = Wilcoxon(x, y, zeroMethod, false, alternative, "asymptotic");
        var d = y is null ? (double[])x.Clone() : x.Select((v, i) => v - y[i]).ToArray();
        var zm = zeroMethod.ToLowerInvariant();
        var res = Resample(method, [d], s => StatsRankDistributions.WilcoxonStatistic(s[0], zm).RPlus, PermutationType.Samples, alternative, "wilcoxon");
        return new SignificanceResult(asym.Statistic, res.Pvalue);
    }

    /// <summary>
    /// Chatterjee's ξ with a permutation p-value (scipy's <c>chatterjeexi(x, y, method=PermutationMethod())</c>):
    /// y is re-paired with x; large ξ is evidence of dependence.
    /// </summary>
    public static SignificanceResult Chatterjeexi(double[] x, double[] y, PermutationMethod method, bool yContinuous = false)
    {
        var xi = Chatterjeexi(x, y, yContinuous);
        var res = Resample(method, [y], d => Chatterjeexi(x, d[0], yContinuous).Statistic, PermutationType.Pairings, Alternative.Greater, "chatterjeexi");
        return new SignificanceResult(xi.Statistic, res.Pvalue);
    }

    /// <summary>The Baumgartner–Weiss–Schindler test configured by a <see cref="PermutationMethod"/> (scipy's <c>bws_test(x, y, method=...)</c>).</summary>
    public static ResamplingTestResult BwsTest(double[] x, double[] y, PermutationMethod method, Alternative alternative = Alternative.TwoSided)
    {
        ArgumentNullException.ThrowIfNull(method);
        method.Validate();
        return BwsTest(x, y, alternative, method.NResamples, method.Seed);
    }

    /// <summary>
    /// The k-sample Anderson–Darling test with a permutation p-value (scipy's
    /// <c>anderson_ksamp(samples, method=PermutationMethod())</c>) in place of the interpolated one.
    /// </summary>
    public static AndersonKsampResult AndersonKsamp(double[][] samples, PermutationMethod method, string variant = "midrank")
    {
        var r = AndersonKsamp(samples, variant);
        var res = Resample(method, samples, s => AndersonKsamp(s, variant).Statistic, PermutationType.Independent, Alternative.Greater, "anderson_ksamp");
        return r with { Pvalue = res.Pvalue };
    }

    /// <summary>
    /// The Anderson–Darling test with a Monte Carlo p-value (scipy's <c>anderson(x, dist, method=MonteCarloMethod())</c>):
    /// <see cref="GoodnessOfFit(ContinuousDistribution, double[], IReadOnlyDictionary{string, double}?, IReadOnlyDictionary{string, double}?, IReadOnlyDictionary{string, double}?, string, int, int?, bool)"/>
    /// with the AD statistic, refitting the family to every simulated sample (the exponential with
    /// its location known to be 0, as scipy). <see cref="MonteCarloMethod.Rvs"/> and the batch are ignored, as in scipy.
    /// </summary>
    public static SignificanceResult Anderson(double[] x, string dist, MonteCarloMethod method)
    {
        ArgumentNullException.ThrowIfNull(method);
        method.Validate();
        var a = Anderson(x, dist);
        var d = a.Distribution;
        ContinuousDistribution family = d switch
        {
            "norm" => Norm(),
            "expon" => Expon(),
            "logistic" => Logistic(),
            "gumbel_l" => GumbelL(),
            "gumbel_r" => GumbelR(),
            "weibull_min" => WeibullMin(1.0),
            _ => throw new ArgumentException($"Unsupported distribution {dist}.", nameof(dist)),
        };
        IReadOnlyDictionary<string, double>? known = d == "expon" ? new Dictionary<string, double> { ["loc"] = 0.0 } : null;
        var g = GoodnessOfFit(family, x, known, statistic: "ad", nMcSamples: method.NResamples, seed: method.Seed, parallel: method.Parallel);
        return new SignificanceResult(a.Statistic, g.Pvalue);
    }

    /// <summary>
    /// The chi-square test of independence with a resampled p-value (scipy's
    /// <c>chi2_contingency(observed, correction=False, method=...)</c>, two-way tables only): a
    /// <see cref="PermutationMethod"/> re-pairs the row and column labels of the untabulated
    /// observations; a <see cref="MonteCarloMethod"/> draws tables with the observed margins
    /// (its <c>Rvs</c> must be null). Degrees of freedom are reported as NaN, as scipy.
    /// </summary>
    public static (double Statistic, double Pvalue, double Dof, NdArray ExpectedFreq) Chi2Contingency(NdArray observed, ResamplingMethod method)
    {
        ArgumentNullException.ThrowIfNull(observed);
        ArgumentNullException.ThrowIfNull(method);
        if (observed.Rank != 2) throw new ArgumentException("Use of `method` is only compatible with two-way tables.", nameof(observed));
        var res0 = Chi2Contingency(observed, correction: false);
        var exp = res0.ExpectedFreq.AsContiguous().ToArray();
        int rows = observed.Shape[0], cols = observed.Shape[1];
        var obs = observed.AsContiguous().ToArray();
        double Stat(ReadOnlySpan<double> table)
        {
            var s = 0.0;
            for (var i = 0; i < table.Length; i++) s += (table[i] - exp[i]) * (table[i] - exp[i]) / exp[i];
            return s;
        }
        ResamplingTestResult res;
        if (method is PermutationMethod)
        {
            // Untabulate: one observation per count, labelled by its row (x) and column (y).
            var xs = new List<double>();
            var ys = new List<int>();
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++)
                    for (var c = 0; c < (int)obs[i * cols + j]; c++) { xs.Add(i); ys.Add(j); }
            var yl = ys.ToArray();
            res = Resample(method, [xs.ToArray()], d =>
            {
                Span<double> table = stackalloc double[rows * cols];
                table.Clear();
                var xv = d[0];
                for (var k = 0; k < xv.Length; k++) table[(int)xv[k] * cols + yl[k]]++;
                return Stat(table);
            }, PermutationType.Pairings, Alternative.Greater, "chi2_contingency");
        }
        else if (method is MonteCarloMethod mc)
        {
            if (mc.Rvs is not null) throw new ArgumentException("If the `method` argument of `chi2_contingency` is a MonteCarloMethod, its `Rvs` must be unspecified; use `Seed`.");
            var rowSums = new int[rows];
            var colSums = new int[cols];
            for (var i = 0; i < rows; i++) for (var j = 0; j < cols; j++) { rowSums[i] += (int)obs[i * cols + j]; colSums[j] += (int)obs[i * cols + j]; }
            var rt = RandomTable(rowSums, colSums);
            Func<int, GraviRandom, double[]> rvs = (_, rng) => rt.Rvs(1, rng)[0].AsContiguous().ToArray();
            res = MonteCarloTest([obs], [rvs], d => Stat(d[0]), mc.NResamples, Alternative.Greater, mc.Seed, mc.Parallel);
        }
        else throw new ArgumentException("`method` must be an instance of `PermutationMethod` or `MonteCarloMethod`.");
        return (res.Statistic, res.Pvalue, double.NaN, res0.ExpectedFreq);
    }
}
