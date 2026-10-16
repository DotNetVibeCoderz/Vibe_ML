using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>The alternative hypothesis of a test, scipy's <c>alternative=</c> keyword.</summary>
public enum Alternative
{
    /// <summary>The effect may lie on either side (scipy's <c>'two-sided'</c>).</summary>
    TwoSided,
    /// <summary>The statistic (or first sample's location) is smaller under the alternative.</summary>
    Less,
    /// <summary>The statistic (or first sample's location) is larger under the alternative.</summary>
    Greater,
}

/// <summary>What to do when the input contains NaN, scipy's <c>nan_policy=</c>.</summary>
public enum NanPolicy
{
    /// <summary>Return NaN results (scipy's default).</summary>
    Propagate,
    /// <summary>Throw an <see cref="ArgumentException"/>.</summary>
    Raise,
    /// <summary>Drop the NaN values and compute on the rest.</summary>
    Omit,
}

/// <summary>A confidence interval, scipy's <c>ConfidenceInterval(low, high)</c>.</summary>
/// <param name="Low">Lower bound (may be −∞ for a one-sided interval).</param>
/// <param name="High">Upper bound (may be +∞ for a one-sided interval).</param>
public readonly record struct ConfidenceInterval(double Low, double High);

/// <summary>
/// A test statistic and its p-value — the shape of most scipy results (<c>SignificanceResult</c>,
/// <c>LeveneResult</c>, <c>KruskalResult</c>, …, which all carry exactly these two attributes).
/// </summary>
/// <param name="Statistic">The test statistic.</param>
/// <param name="Pvalue">The p-value for the requested alternative.</param>
public record SignificanceResult(double Statistic, double Pvalue);

/// <summary>
/// Result of the t-tests (scipy's <c>TtestResult</c>): statistic, p-value, degrees of freedom and a
/// <see cref="ConfidenceInterval(double)"/> for the tested location parameter.
/// </summary>
public sealed record TtestResult : SignificanceResult
{
    internal TtestResult(double statistic, double pvalue, double df, double estimate, double standardError, Alternative alternative)
        : base(statistic, pvalue)
    {
        Df = df;
        Estimate = estimate;
        StandardError = standardError;
        Alternative = alternative;
    }

    /// <summary>Degrees of freedom of the reference t distribution (Welch's are fractional).</summary>
    public double Df { get; }

    /// <summary>Point estimate of the tested quantity: the mean, the difference of means, or the mean difference.</summary>
    public double Estimate { get; }

    /// <summary>Standard error of <see cref="Estimate"/>.</summary>
    public double StandardError { get; }

    /// <summary>The alternative the p-value was computed for; it also fixes the interval's sidedness.</summary>
    public Alternative Alternative { get; }

    /// <summary>
    /// Confidence interval for the population mean (one-sample), difference of means (independent)
    /// or mean difference (related). One-sided alternatives give a half-infinite interval, as scipy.
    /// </summary>
    public ConfidenceInterval ConfidenceInterval(double confidenceLevel = 0.95)
    {
        if (!(confidenceLevel > 0 && confidenceLevel < 1)) throw new ArgumentException("confidenceLevel must lie in (0, 1).", nameof(confidenceLevel));
        switch (Alternative)
        {
            case Alternative.Less:
                return new(double.NegativeInfinity, Estimate + Special.Stdtrit(Df, confidenceLevel) * StandardError);
            case Alternative.Greater:
                return new(Estimate - Special.Stdtrit(Df, confidenceLevel) * StandardError, double.PositiveInfinity);
            default:
                var c = Special.Stdtrit(Df, 0.5 + confidenceLevel / 2);
                return new(Estimate - c * StandardError, Estimate + c * StandardError);
        }
    }
}

/// <summary>Result of the Kolmogorov–Smirnov tests (scipy's <c>KstestResult</c>).</summary>
public sealed record KstestResult : SignificanceResult
{
    internal KstestResult(double statistic, double pvalue, double location, int sign) : base(statistic, pvalue)
    {
        StatisticLocation = location;
        StatisticSign = sign;
    }

    /// <summary>The data value at which the statistic (largest CDF gap) is attained.</summary>
    public double StatisticLocation { get; }

    /// <summary>+1 when the empirical CDF lies above the reference (or second sample) there, −1 below.</summary>
    public int StatisticSign { get; }
}

/// <summary>Result of <see cref="Stats.Anderson"/> (scipy's <c>AndersonResult</c>).</summary>
/// <param name="Statistic">The Anderson–Darling statistic A².</param>
/// <param name="CriticalValues">Critical values at <paramref name="SignificanceLevel"/>.</param>
/// <param name="SignificanceLevel">Significance levels in percent, matching the critical values.</param>
/// <param name="FitLocation">Location of the fitted distribution.</param>
/// <param name="FitScale">Scale of the fitted distribution.</param>
public sealed record AndersonResult(double Statistic, double[] CriticalValues, double[] SignificanceLevel, double FitLocation, double FitScale)
{
    /// <summary>
    /// p-value by linear interpolation of the critical-value table (scipy's <c>method='interpolate'</c>),
    /// clamped to the tabulated range — so it is only informative between the extreme levels.
    /// </summary>
    public double Pvalue
    {
        get
        {
            var c = CriticalValues;
            var n = c.Length;
            if (Statistic <= c[0]) return SignificanceLevel[0] / 100;
            if (Statistic >= c[n - 1]) return SignificanceLevel[n - 1] / 100;
            var i = 0;
            while (Statistic > c[i + 1]) i++;
            var f = (Statistic - c[i]) / (c[i + 1] - c[i]);
            return (SignificanceLevel[i] + f * (SignificanceLevel[i + 1] - SignificanceLevel[i])) / 100;
        }
    }
}

/// <summary>Result of <see cref="Stats.AndersonKsamp"/> (scipy's <c>Anderson_ksampResult</c>).</summary>
/// <param name="Statistic">The normalised k-sample statistic.</param>
/// <param name="CriticalValues">Critical values at 25%, 10%, 5%, 2.5%, 1%, 0.5%, 0.1%.</param>
/// <param name="Pvalue">Approximate p-value, floored at 0.001 and capped at 0.25 as scipy.</param>
public sealed record AndersonKsampResult(double Statistic, double[] CriticalValues, double Pvalue);

/// <summary>Result of <see cref="Stats.Chi2Contingency"/> (scipy's <c>Chi2ContingencyResult</c>).</summary>
/// <param name="Statistic">The test statistic.</param>
/// <param name="Pvalue">The p-value.</param>
/// <param name="Dof">Degrees of freedom.</param>
/// <param name="ExpectedFreq">Expected frequencies under independence, same shape as the table.</param>
public sealed record Chi2ContingencyResult(double Statistic, double Pvalue, int Dof, NdArray ExpectedFreq);

/// <summary>Result of <see cref="Stats.Binomtest"/> (scipy's <c>BinomTestResult</c>).</summary>
public sealed record BinomTestResult : SignificanceResult
{
    internal BinomTestResult(int k, int n, Alternative alternative, double pvalue) : base((double)k / n, pvalue)
    {
        K = k;
        N = n;
        Alternative = alternative;
    }

    /// <summary>Number of successes.</summary>
    public int K { get; }

    /// <summary>Number of trials.</summary>
    public int N { get; }

    /// <summary>The alternative of the test; it fixes the sidedness of the interval.</summary>
    public Alternative Alternative { get; }

    /// <summary>
    /// Confidence interval for the success probability: <c>"exact"</c> (Clopper–Pearson),
    /// <c>"wilson"</c> or <c>"wilsoncc"</c> (Wilson with continuity correction), as scipy.
    /// </summary>
    public ConfidenceInterval ProportionCi(double confidenceLevel = 0.95, string method = "exact")
        => StatsContingencyKernels.BinomCi(K, N, Alternative, confidenceLevel, method);
}

/// <summary>Result of <see cref="Stats.Linregress(double[], double[], Alternative)"/> (scipy's <c>LinregressResult</c>).</summary>
/// <param name="Slope">Slope of the least-squares line.</param>
/// <param name="Intercept">Intercept.</param>
/// <param name="Rvalue">Pearson correlation coefficient.</param>
/// <param name="Pvalue">p-value of the slope (t test with n−2 degrees of freedom).</param>
/// <param name="Stderr">Standard error of the slope.</param>
/// <param name="InterceptStderr">Standard error of the intercept.</param>
public sealed record LinregressResult(double Slope, double Intercept, double Rvalue, double Pvalue, double Stderr, double InterceptStderr);

/// <summary>Result of <see cref="Stats.Theilslopes"/> (scipy's <c>TheilslopesResult</c>).</summary>
/// <param name="Slope">Median of pairwise slopes.</param>
/// <param name="Intercept">Intercept.</param>
/// <param name="LowSlope">Lower bound of the slope confidence interval.</param>
/// <param name="HighSlope">Upper bound of the slope confidence interval.</param>
public sealed record TheilslopesResult(double Slope, double Intercept, double LowSlope, double HighSlope);

/// <summary>Result of <see cref="Stats.Siegelslopes"/> (scipy's <c>SiegelslopesResult</c>).</summary>
/// <param name="Slope">Repeated-median slope.</param>
/// <param name="Intercept">Intercept.</param>
public sealed record SiegelslopesResult(double Slope, double Intercept);

/// <summary>Result of <see cref="Stats.Pearsonr"/> (scipy's <c>PearsonRResult</c>).</summary>
public sealed record PearsonRResult : SignificanceResult
{
    internal PearsonRResult(double r, double p, int n, Alternative alternative) : base(r, p)
    {
        N = n;
        Alternative = alternative;
    }

    /// <summary>Number of observation pairs.</summary>
    public int N { get; }

    /// <summary>The alternative of the test.</summary>
    public Alternative Alternative { get; }

    /// <summary>Confidence interval for ρ by the Fisher transformation, scipy's default method.</summary>
    public ConfidenceInterval ConfidenceInterval(double confidenceLevel = 0.95)
    {
        var r = Statistic;
        var zr = Math.Atanh(r);
        var se = Math.Sqrt(1.0 / (N - 3));
        switch (Alternative)
        {
            case Alternative.Less:
                return new(-1.0, Math.Tanh(zr + se * Special.Ndtri(confidenceLevel)));
            case Alternative.Greater:
                return new(Math.Tanh(zr - se * Special.Ndtri(confidenceLevel)), 1.0);
            default:
                var h = se * Special.Ndtri(0.5 + confidenceLevel / 2);
                return new(Math.Tanh(zr - h), Math.Tanh(zr + h));
        }
    }
}

/// <summary>Result of <see cref="Stats.Describe(double[], int, bool)"/> (scipy's <c>DescribeResult</c>).</summary>
/// <param name="Nobs">Number of observations.</param>
/// <param name="Min">Minimum.</param>
/// <param name="Max">Maximum.</param>
/// <param name="Mean">Arithmetic mean.</param>
/// <param name="Variance">Variance with the requested ddof (scipy default 1).</param>
/// <param name="Skewness">Skewness.</param>
/// <param name="Kurtosis">Kurtosis (Fisher, i.e. excess, by default).</param>
public sealed record DescribeResult(int Nobs, double Min, double Max, double Mean, double Variance, double Skewness, double Kurtosis);

/// <summary>Result of <see cref="Stats.Mode(double[])"/> (scipy's <c>ModeResult</c>).</summary>
/// <param name="Mode">The smallest of the most frequent values.</param>
/// <param name="Count">How often it occurs.</param>
public sealed record ModeResult(double Mode, int Count);

/// <summary>Result of <see cref="Stats.Cumfreq"/> and <see cref="Stats.Relfreq"/>.</summary>
/// <param name="Frequency">Cumulative counts (cumfreq) or relative frequencies (relfreq) per bin.</param>
/// <param name="Lowerlimit">Lower edge of the first bin.</param>
/// <param name="Binsize">Width of each bin.</param>
/// <param name="Extrapoints">Points that fell outside the limits.</param>
public sealed record FrequencyResult(double[] Frequency, double Lowerlimit, double Binsize, int Extrapoints);

/// <summary>Result of <see cref="Stats.Sigmaclip"/> (scipy's <c>SigmaclipResult</c>).</summary>
/// <param name="Clipped">The values that survived clipping.</param>
/// <param name="Lower">The final lower threshold.</param>
/// <param name="Upper">The final upper threshold.</param>
public sealed record SigmaclipResult(double[] Clipped, double Lower, double Upper);

/// <summary>Result of the binned statistics (scipy's <c>BinnedStatisticResult</c> family).</summary>
/// <param name="Statistic">The statistic per bin, shape (nx) / (nx, ny) / (n1, …, nd).</param>
/// <param name="BinEdges">The edges of each dimension.</param>
/// <param name="Binnumber">Linearised bin index of each sample (with the outlier bins, as scipy's <c>expand_binnumbers=False</c>).</param>
public sealed record BinnedStatisticResult(NdArray Statistic, double[][] BinEdges, int[] Binnumber);

/// <summary>Result of <c>Stats.TukeyHsd</c> (scipy's <c>TukeyHSDResult</c>).</summary>
public sealed class TukeyHsdResult
{
    private readonly double[] _df;
    private readonly double[] _standardError;
    private readonly int _k;

    internal TukeyHsdResult(NdArray statistic, NdArray pvalue, int k, double[] df, double[] standardError)
    {
        Statistic = statistic;
        Pvalue = pvalue;
        _k = k;
        _df = df;
        _standardError = standardError;
    }

    /// <summary>k×k matrix of mean differences, entry (i, j) = mean_i − mean_j.</summary>
    public NdArray Statistic { get; }

    /// <summary>k×k matrix of adjusted p-values from the studentized range distribution.</summary>
    public NdArray Pvalue { get; }

    /// <summary>
    /// Simultaneous confidence intervals for the differences: Tukey–Kramer, or Games–Howell when the
    /// test was run with unequal variances.
    /// </summary>
    public (NdArray Low, NdArray High) ConfidenceInterval(double confidenceLevel = 0.95)
    {
        if (!(confidenceLevel > 0 && confidenceLevel < 1)) throw new ArgumentException("confidenceLevel must lie in (0, 1).", nameof(confidenceLevel));
        var stat = Statistic.AsContiguous().ToArray();
        var lo = new double[_k * _k];
        var hi = new double[_k * _k];
        var cache = new Dictionary<double, double>();
        for (var i = 0; i < _k * _k; i++)
        {
            if (!cache.TryGetValue(_df[i], out var q))
                cache[_df[i]] = q = new StudentizedRangeDistribution(_k, _df[i]).Ppf(confidenceLevel);
            lo[i] = stat[i] - q * _standardError[i];
            hi[i] = stat[i] + q * _standardError[i];
        }
        return (new NdArray(lo, _k, _k), new NdArray(hi, _k, _k));
    }
}

/// <summary>Result of <c>Stats.PermutationTest</c> and <c>Stats.MonteCarloTest</c>.</summary>
/// <param name="Statistic">The observed statistic.</param>
/// <param name="Pvalue">The p-value.</param>
/// <param name="NullDistribution">The statistic under each resample of the null.</param>
public sealed record ResamplingTestResult(double Statistic, double Pvalue, double[] NullDistribution);

/// <summary>Result of <c>Stats.Bootstrap</c> (scipy's <c>BootstrapResult</c>).</summary>
/// <param name="ConfidenceInterval">The bootstrap confidence interval.</param>
/// <param name="BootstrapDistribution">The statistic on each resample.</param>
/// <param name="StandardError">Standard deviation (ddof 1) of the bootstrap distribution.</param>
public sealed record BootstrapResult(ConfidenceInterval ConfidenceInterval, double[] BootstrapDistribution, double StandardError);

/// <summary>Result of <see cref="Stats.PageTrendTest"/> (scipy's <c>PageTrendTestResult</c>).</summary>
/// <param name="Statistic">Page's L.</param>
/// <param name="Pvalue">One-sided p-value for an increasing trend in the predicted order.</param>
/// <param name="Method">"exact" or "asymptotic", whichever was used.</param>
public sealed record PageTrendTestResult(double Statistic, double Pvalue, string Method);

/// <summary>Result of <see cref="Stats.Somersd(double[], double[], Alternative)"/> (scipy's <c>SomersDResult</c>).</summary>
/// <param name="Statistic">Somers' D(Y|X).</param>
/// <param name="Pvalue">p-value of the asymptotic test of D = 0.</param>
/// <param name="Table">The contingency table the statistic was computed from.</param>
public sealed record SomersDResult(double Statistic, double Pvalue, NdArray Table);

/// <summary>Result of <see cref="Stats.Dunnett"/> (scipy's <c>DunnettResult</c>).</summary>
public sealed class DunnettResult
{
    private readonly double[] _lambda;
    private readonly double _df;
    private readonly double _std;
    private readonly double[] _means;
    private readonly double _meanControl;
    private readonly double[] _n;
    private readonly double _n0;
    private readonly Alternative _alternative;

    internal DunnettResult(double[] statistic, double[] pvalue, double[] lambda, double df, double std, double[] means, double meanControl, double[] n, double n0, Alternative alternative)
    {
        Statistic = statistic;
        Pvalue = pvalue;
        _lambda = lambda;
        _df = df;
        _std = std;
        _means = means;
        _meanControl = meanControl;
        _n = n;
        _n0 = n0;
        _alternative = alternative;
    }

    /// <summary>The statistic of each treatment-vs-control comparison.</summary>
    public double[] Statistic { get; }

    /// <summary>The adjusted p-value of each comparison.</summary>
    public double[] Pvalue { get; }

    /// <summary>
    /// Simultaneous confidence intervals for the mean differences (treatment − control). The critical
    /// value is solved by bisection to ~1e-12 rather than scipy's minimiser tolerance of 1e-3, so the
    /// bounds agree with scipy to about three digits.
    /// </summary>
    public (double[] Low, double[] High) ConfidenceInterval(double confidenceLevel = 0.95)
    {
        if (!(confidenceLevel > 0 && confidenceLevel < 1)) throw new ArgumentException("confidenceLevel must lie in (0, 1).", nameof(confidenceLevel));
        var alpha = 1 - confidenceLevel;
        double P(double c) => Stats.DunnettPvalue(c, _lambda, _df, _alternative);
        double lo, hi;
        if (_alternative == Alternative.Less) { lo = -50; hi = 0; } else { lo = 0; hi = 50; }
        for (var it = 0; it < 50; it++)
        {
            var mid = (lo + hi) / 2;
            var moveHigh = _alternative == Alternative.Less ? P(mid) > alpha : P(mid) < alpha;
            if (moveHigh) hi = mid; else lo = mid;
        }
        var crit = Math.Abs((lo + hi) / 2);
        var k = _means.Length;
        var low = new double[k];
        var high = new double[k];
        for (var i = 0; i < k; i++)
        {
            var allowance = crit * _std * Math.Sqrt(1 / _n[i] + 1 / _n0);
            var d = _means[i] - _meanControl;
            low[i] = _alternative == Alternative.Less ? double.NegativeInfinity : d - allowance;
            high[i] = _alternative == Alternative.Greater ? double.PositiveInfinity : d + allowance;
        }
        return (low, high);
    }
}
