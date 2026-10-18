using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A user-defined continuous distribution for <see cref="Stats.MakeDistribution(ContinuousDistributionDefinition)"/>:
/// scipy's duck-typed class with <c>support</c>, <c>pdf</c> and optional <c>cdf</c>, <c>icdf</c>, ….
/// Parameters are whatever the delegates close over.
/// </summary>
/// <remarks>Only <see cref="Support"/> and one of <see cref="Pdf"/>/<see cref="LogPdf"/> are required;
/// each delegate given becomes that function's formula, and everything else uses the framework's
/// fall-backs (quadrature, inversion, optimisation) exactly as for a built-in distribution.</remarks>
public sealed class ContinuousDistributionDefinition
{
    /// <summary>The name printed by <c>ToString</c> (scipy uses the class name).</summary>
    public string Name { get; init; } = "CustomDistribution";
    /// <summary>The support [lower, upper].</summary>
    public (double Lower, double Upper) Support { get; init; } = (double.NegativeInfinity, double.PositiveInfinity);
    /// <summary>Whether the ends belong to the support (scipy's <c>inclusive</c>; default neither).</summary>
    public (bool Lower, bool Upper) SupportInclusive { get; init; }
    /// <summary>Probability density.</summary>
    public Func<double, double>? Pdf { get; init; }
    /// <summary>Log density.</summary>
    public Func<double, double>? LogPdf { get; init; }
    /// <summary>CDF.</summary>
    public Func<double, double>? Cdf { get; init; }
    /// <summary>Log CDF.</summary>
    public Func<double, double>? LogCdf { get; init; }
    /// <summary>Complementary CDF.</summary>
    public Func<double, double>? Ccdf { get; init; }
    /// <summary>Log complementary CDF.</summary>
    public Func<double, double>? LogCcdf { get; init; }
    /// <summary>Two-argument CDF P(x ≤ X ≤ y).</summary>
    public Func<double, double, double>? Cdf2 { get; init; }
    /// <summary>Two-argument complementary CDF.</summary>
    public Func<double, double, double>? Ccdf2 { get; init; }
    /// <summary>Inverse CDF.</summary>
    public Func<double, double>? Icdf { get; init; }
    /// <summary>Inverse complementary CDF.</summary>
    public Func<double, double>? Iccdf { get; init; }
    /// <summary>Inverse log CDF.</summary>
    public Func<double, double>? ILogCdf { get; init; }
    /// <summary>Inverse log complementary CDF.</summary>
    public Func<double, double>? ILogCcdf { get; init; }
    /// <summary>Entropy.</summary>
    public Func<double>? Entropy { get; init; }
    /// <summary>Median.</summary>
    public Func<double>? Median { get; init; }
    /// <summary>Mode.</summary>
    public Func<double>? Mode { get; init; }
    /// <summary>Moments: return the moment of the given order and kind, or null where there is no formula
    /// (scipy's <c>moment(order, kind)</c>).</summary>
    public Func<int, MomentKind, double?>? Moment { get; init; }
    /// <summary>A sampler for one variate.</summary>
    public Func<GraviRandom, double>? Sample { get; init; }
}

/// <summary>A distribution built from a <see cref="ContinuousDistributionDefinition"/>.</summary>
internal sealed class CustomContinuousDistribution : ContinuousRandomVariable
{
    private readonly ContinuousDistributionDefinition _d;

    public CustomContinuousDistribution(ContinuousDistributionDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        if (definition.Pdf is null && definition.LogPdf is null) throw new ArgumentException("A distribution definition needs Pdf or LogPdf.", nameof(definition));
        var (a, b) = definition.Support;
        if (double.IsNaN(a) || double.IsNaN(b) || !(a < b)) throw new ArgumentException("The support must satisfy lower < upper.", nameof(definition));
        _d = definition;
    }

    public override string Name => _d.Name;

    internal override bool Overrides(RandomVariableFormula formula) => formula switch
    {
        RandomVariableFormula.Pdf => _d.Pdf is not null,
        RandomVariableFormula.LogPdf => _d.LogPdf is not null,
        RandomVariableFormula.Pmf or RandomVariableFormula.LogPmf => true,
        RandomVariableFormula.Cdf => _d.Cdf is not null,
        RandomVariableFormula.LogCdf => _d.LogCdf is not null,
        RandomVariableFormula.Ccdf => _d.Ccdf is not null,
        RandomVariableFormula.LogCcdf => _d.LogCcdf is not null,
        RandomVariableFormula.Cdf2 => _d.Cdf2 is not null,
        RandomVariableFormula.Ccdf2 => _d.Ccdf2 is not null,
        RandomVariableFormula.Icdf => _d.Icdf is not null,
        RandomVariableFormula.Iccdf => _d.Iccdf is not null,
        RandomVariableFormula.ILogCdf => _d.ILogCdf is not null,
        RandomVariableFormula.ILogCcdf => _d.ILogCcdf is not null,
        RandomVariableFormula.Entropy => _d.Entropy is not null,
        RandomVariableFormula.Median => _d.Median is not null,
        RandomVariableFormula.Mode => _d.Mode is not null,
        RandomVariableFormula.Sample => _d.Sample is not null,
        _ => false,
    };

    protected internal override (double Lower, double Upper) SupportCore() => _d.Support;
    protected internal override (bool Lower, bool Upper) SupportInclusive => _d.SupportInclusive;
    protected override double PdfFormula(double x) => _d.Pdf!(x);
    protected override double LogPdfFormula(double x) => _d.LogPdf!(x);
    protected override double CdfFormula(double x) => _d.Cdf!(x);
    protected override double LogCdfFormula(double x) => _d.LogCdf!(x);
    protected override double CcdfFormula(double x) => _d.Ccdf!(x);
    protected override double LogCcdfFormula(double x) => _d.LogCcdf!(x);
    protected override double Cdf2Formula(double x, double y) => _d.Cdf2!(x, y);
    protected override double Ccdf2Formula(double x, double y) => _d.Ccdf2!(x, y);
    protected override double IcdfFormula(double p) => _d.Icdf!(p);
    protected override double IccdfFormula(double p) => _d.Iccdf!(p);
    protected override double ILogCdfFormula(double logp) => _d.ILogCdf!(logp);
    protected override double ILogCcdfFormula(double logp) => _d.ILogCcdf!(logp);
    protected override double EntropyFormula() => _d.Entropy!();
    protected override double MedianFormula() => _d.Median!();
    protected override double ModeFormula() => _d.Mode!();
    protected override double? MomentRawFormula(int order) => _d.Moment?.Invoke(order, MomentKind.Raw);
    protected override double? MomentCentralFormula(int order) => _d.Moment?.Invoke(order, MomentKind.Central);
    protected override double? MomentStandardizedFormula(int order) => _d.Moment?.Invoke(order, MomentKind.Standardized);

    protected override void SampleFormula(Span<double> destination, GraviRandom rng)
    {
        for (var i = 0; i < destination.Length; i++) destination[i] = _d.Sample!(rng);
    }

    public override string ToString() => $"{_d.Name}()";
}

/// <summary>
/// A frozen <see cref="ContinuousDistribution"/> (scipy's <c>rv_continuous</c>) in the new framework,
/// what <c>make_distribution(stats.gamma)</c> produces. The existing implementations become the
/// formulas — pdf, CDF, survival function, quantiles, entropy, median and the four moments of
/// <c>Stats()</c> — and only what they lack (higher moments, the mode) falls back to numerics.
/// </summary>
internal sealed class RvContinuousAdapter : ContinuousRandomVariable
{
    private readonly ContinuousDistribution _d;
    private DistributionMoments? _stats;

    public RvContinuousAdapter(ContinuousDistribution distribution)
    {
        ArgumentNullException.ThrowIfNull(distribution);
        _d = distribution;
    }

    public ContinuousDistribution Distribution => _d;

    public override string Name => StatsNames.NewName(_d.Name);

    internal override bool Overrides(RandomVariableFormula formula) => formula is RandomVariableFormula.Pdf or RandomVariableFormula.LogPdf
        or RandomVariableFormula.Pmf or RandomVariableFormula.LogPmf
        or RandomVariableFormula.Cdf or RandomVariableFormula.LogCdf or RandomVariableFormula.Ccdf or RandomVariableFormula.LogCcdf
        or RandomVariableFormula.Icdf or RandomVariableFormula.Iccdf or RandomVariableFormula.Entropy or RandomVariableFormula.Median
        or RandomVariableFormula.Sample;

    protected internal override (double Lower, double Upper) SupportCore() => _d.Support();
    protected internal override (bool Lower, bool Upper) SupportInclusive => (true, true);
    protected override double PdfFormula(double x) => _d.Pdf(x);
    protected override double LogPdfFormula(double x) => _d.LogPdf(x);
    protected override double CdfFormula(double x) => _d.Cdf(x);
    protected override double LogCdfFormula(double x) => _d.LogCdf(x);
    protected override double CcdfFormula(double x) => _d.Sf(x);
    protected override double LogCcdfFormula(double x) => _d.LogSf(x);
    protected override double IcdfFormula(double p) => _d.Ppf(p);
    protected override double IccdfFormula(double p) => _d.Isf(p);
    protected override double EntropyFormula() => _d.Entropy();
    protected override double MedianFormula() => _d.Median();

    private DistributionMoments S => _stats ??= _d.Stats();

    protected override double? MomentRawFormula(int order) => order switch
    {
        0 => 1.0,
        1 => S.Mean,
        >= 5 => _d.Moment(order),
        _ => null,
    };

    protected override double? MomentCentralFormula(int order) => order == 2 ? S.Variance : null;

    protected override double? MomentStandardizedFormula(int order) => order switch
    {
        3 => S.Skewness,
        4 => S.Kurtosis + 3,
        _ => null,
    };

    protected override void SampleFormula(Span<double> destination, GraviRandom rng) => _d.Rvs(destination.Length, rng).CopyTo(destination);

    public override string ToString()
    {
        var shapes = _d.ShapeNames.Zip(_d.Shapes, (n, v) => $"{n}={PyFormat.F(v)}");
        var core = $"{Name}({string.Join(", ", shapes)})";
        if (_d.Scale != 1 || _d.Loc != 0) return $"{PyFormat.F(_d.Scale)}*{core}" + (_d.Loc < 0 ? $" - {PyFormat.F(-_d.Loc)}" : _d.Loc != 0 ? $" + {PyFormat.F(_d.Loc)}" : "");
        return core;
    }
}

/// <summary>A frozen <see cref="DiscreteDistribution"/> (scipy's <c>rv_discrete</c>) in the new framework.</summary>
internal sealed class RvDiscreteAdapter : DiscreteRandomVariable
{
    private readonly DiscreteDistribution _d;
    private DistributionMoments? _stats;

    public RvDiscreteAdapter(DiscreteDistribution distribution)
    {
        ArgumentNullException.ThrowIfNull(distribution);
        _d = distribution;
    }

    public override string Name => StatsNames.NewName(_d.Name);

    internal override bool Overrides(RandomVariableFormula formula) => formula is RandomVariableFormula.Pmf or RandomVariableFormula.LogPmf
        or RandomVariableFormula.Pdf or RandomVariableFormula.LogPdf
        or RandomVariableFormula.Cdf or RandomVariableFormula.LogCdf or RandomVariableFormula.Ccdf or RandomVariableFormula.LogCcdf
        or RandomVariableFormula.Icdf or RandomVariableFormula.Iccdf or RandomVariableFormula.Entropy or RandomVariableFormula.Median
        or RandomVariableFormula.Sample;

    protected internal override (double Lower, double Upper) SupportCore() => _d.Support();
    protected override double PmfFormula(double x) => _d.Pmf(x);
    protected override double LogPmfFormula(double x) => _d.LogPmf(x);
    protected override double CdfFormula(double x) => _d.Cdf(x);
    protected override double LogCdfFormula(double x) => _d.LogCdf(x);
    protected override double CcdfFormula(double x) => _d.Sf(x);
    protected override double LogCcdfFormula(double x) => _d.LogSf(x);
    protected override double IcdfFormula(double p) => _d.Ppf(p);
    protected override double IccdfFormula(double p) => _d.Isf(p);
    protected override double EntropyFormula() => _d.Entropy();
    protected override double MedianFormula() => _d.Median();

    private DistributionMoments S => _stats ??= _d.Stats();

    protected override double? MomentRawFormula(int order) => order switch
    {
        0 => 1.0,
        1 => S.Mean,
        >= 5 => _d.Moment(order),
        _ => null,
    };

    protected override double? MomentCentralFormula(int order) => order == 2 ? S.Variance : null;

    protected override double? MomentStandardizedFormula(int order) => order switch
    {
        3 => S.Skewness,
        4 => S.Kurtosis + 3,
        _ => null,
    };

    protected override void SampleFormula(Span<double> destination, GraviRandom rng) => _d.Rvs(destination.Length, rng).CopyTo(destination);

    public override string ToString()
    {
        var shapes = _d.ShapeNames.Zip(_d.Shapes, (n, v) => $"{n}={PyFormat.F(v)}");
        return $"{Name}({string.Join(", ", shapes)})";
    }
}

/// <summary>scipy's table mapping <c>rv_generic</c> names to the new framework's class names.</summary>
internal static class StatsNames
{
    private static readonly Dictionary<string, string> s_names = new()
    {
        ["argus"] = "ARGUS", ["betaprime"] = "BetaPrime", ["chi2"] = "ChiSquared", ["crystalball"] = "CrystalBall",
        ["dgamma"] = "DoubleGamma", ["dweibull"] = "DoubleWeibull", ["expon"] = "Exponential",
        ["exponnorm"] = "ExponentiallyModifiedNormal", ["exponweib"] = "ExponentialWeibull", ["exponpow"] = "ExponentialPower",
        ["fatiguelife"] = "FatigueLife", ["foldcauchy"] = "FoldedCauchy", ["foldnorm"] = "FoldedNormal",
        ["genlogistic"] = "GeneralizedLogistic", ["gennorm"] = "GeneralizedNormal", ["genpareto"] = "GeneralizedPareto",
        ["genexpon"] = "GeneralizedExponential", ["genextreme"] = "GeneralizedExtremeValue", ["gausshyper"] = "GaussHypergeometric",
        ["gengamma"] = "GeneralizedGamma", ["genhalflogistic"] = "GeneralizedHalfLogistic", ["geninvgauss"] = "GeneralizedInverseGaussian",
        ["gumbel_r"] = "Gumbel", ["gumbel_l"] = "ReflectedGumbel", ["halfcauchy"] = "HalfCauchy", ["halflogistic"] = "HalfLogistic",
        ["halfnorm"] = "HalfNormal", ["halfgennorm"] = "HalfGeneralizedNormal", ["hypsecant"] = "HyperbolicSecant",
        ["invgamma"] = "InverseGammma", ["invgauss"] = "InverseGaussian", ["invweibull"] = "InverseWeibull", ["irwinhall"] = "IrwinHall",
        ["jf_skew_t"] = "JonesFaddySkewT", ["johnsonsb"] = "JohnsonSB", ["johnsonsu"] = "JohnsonSU", ["ksone"] = "KSOneSided",
        ["kstwo"] = "KSTwoSided", ["kstwobign"] = "KSTwoSidedAsymptotic", ["laplace_asymmetric"] = "LaplaceAsymmetric",
        ["levy_l"] = "LevyLeft", ["levy_stable"] = "LevyStable", ["loggamma"] = "ExpGamma", ["loglaplace"] = "LogLaplace",
        ["lognorm"] = "LogNormal", ["loguniform"] = "LogUniform", ["ncx2"] = "NoncentralChiSquared", ["nct"] = "NoncentralT",
        ["norm"] = "Normal", ["norminvgauss"] = "NormalInverseGaussian", ["powerlaw"] = "PowerLaw", ["powernorm"] = "PowerNormal",
        ["rdist"] = "R", ["rel_breitwigner"] = "RelativisticBreitWigner", ["recipinvgauss"] = "ReciprocalInverseGaussian",
        ["reciprocal"] = "LogUniform", ["semicircular"] = "SemiCircular", ["skewcauchy"] = "SkewCauchy", ["skewnorm"] = "SkewNormal",
        ["studentized_range"] = "StudentizedRange", ["t"] = "StudentT", ["trapezoid"] = "Trapezoidal", ["triang"] = "Triangular",
        ["truncexpon"] = "TruncatedExponential", ["truncnorm"] = "TruncatedNormal", ["truncpareto"] = "TruncatedPareto",
        ["truncweibull_min"] = "TruncatedWeibull", ["tukeylambda"] = "TukeyLambda", ["vonmises_line"] = "VonMisesLine",
        ["weibull_min"] = "Weibull", ["weibull_max"] = "ReflectedWeibull", ["wrapcauchy"] = "WrappedCauchyLine",
        ["betabinom"] = "BetaBinomial", ["betanbinom"] = "BetaNegativeBinomial", ["dlaplace"] = "LaplaceDiscrete",
        ["geom"] = "Geometric", ["hypergeom"] = "Hypergeometric", ["logser"] = "LogarithmicSeries", ["nbinom"] = "NegativeBinomial",
        ["nchypergeom_fisher"] = "NoncentralHypergeometricFisher", ["nchypergeom_wallenius"] = "NoncentralHypergeometricWallenius",
        ["nhypergeom"] = "NegativeHypergeometric", ["poisson_binom"] = "PoissonBinomial", ["randint"] = "UniformDiscrete",
        ["yulesimon"] = "YuleSimon", ["zipf"] = "Zeta",
    };

    /// <summary>scipy's <c>_distribution_names</c> lookup, capitalising names it does not list.</summary>
    public static string NewName(string oldName)
        => s_names.TryGetValue(oldName, out var n) ? n : oldName.Length == 0 ? oldName : char.ToUpperInvariant(oldName[0]) + oldName[1..];
}

public static partial class Stats
{
    /// <summary>
    /// scipy.stats.make_distribution for a frozen continuous distribution: the same distribution in the
    /// new random-variable framework, its existing functions serving as the formulas. Unlike scipy,
    /// the location and scale of the frozen distribution are kept (scipy's version drops them).
    /// </summary>
    /// <example><code>
    /// var g = Stats.MakeDistribution(Stats.Gamma(2.5));
    /// var y = 3.0 * g + 1.0;                 // shift and scale with arithmetic
    /// double m = y.Moment(3, MomentKind.Central);
    /// </code></example>
    public static ContinuousRandomVariable MakeDistribution(ContinuousDistribution distribution) => new RvContinuousAdapter(distribution);

    /// <summary>scipy.stats.make_distribution for a frozen discrete distribution.</summary>
    public static DiscreteRandomVariable MakeDistribution(DiscreteDistribution distribution) => new RvDiscreteAdapter(distribution);

    /// <summary>
    /// scipy.stats.make_distribution for a user definition: a support, a density, and any other
    /// functions you have formulas for. The rest is computed numerically.
    /// </summary>
    /// <example><code>
    /// var x = Stats.MakeDistribution(new ContinuousDistributionDefinition
    /// {
    ///     Name = "MyTriangle", Support = (0.0, 1.0), SupportInclusive = (true, true),
    ///     Pdf = x => 2 * x, Cdf = x => x * x,
    /// });
    /// double median = x.Median();            // root finding on the CDF: 1/√2
    /// </code></example>
    public static ContinuousRandomVariable MakeDistribution(ContinuousDistributionDefinition definition) => new CustomContinuousDistribution(definition);

    /// <summary>scipy.stats.truncate: X restricted to [lb, ub] and renormalised.</summary>
    public static ContinuousRandomVariable Truncate(ContinuousRandomVariable x, double lb = double.NegativeInfinity, double ub = double.PositiveInfinity)
        => new TruncatedDistribution(x, lb, ub);

    /// <summary>scipy.stats.order_statistic: the r-th smallest of n draws of X.</summary>
    public static ContinuousRandomVariable OrderStatistic(ContinuousRandomVariable x, int r, int n) => new OrderStatisticDistribution(x, r, n);

    /// <summary>scipy.stats.abs: the distribution of |X|.</summary>
    public static ContinuousRandomVariable Abs(ContinuousRandomVariable x) { ArgumentNullException.ThrowIfNull(x); return x.Abs(); }

    /// <summary>scipy.stats.exp: the distribution of exp(X).</summary>
    public static ContinuousRandomVariable Exp(ContinuousRandomVariable x) { ArgumentNullException.ThrowIfNull(x); return x.Exp(); }

    /// <summary>scipy.stats.log: the distribution of log(X) for X with non-negative support.</summary>
    public static ContinuousRandomVariable Log(ContinuousRandomVariable x) { ArgumentNullException.ThrowIfNull(x); return x.Log(); }
}
