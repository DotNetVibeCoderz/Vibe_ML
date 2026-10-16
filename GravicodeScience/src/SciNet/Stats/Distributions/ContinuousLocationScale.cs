using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.StatsDistMath;

namespace Gravicode.Science.SciNet;

/// <summary>The normal distribution, scipy's <c>norm</c>.</summary>
/// <remarks>Both tails go through <see cref="Special.Ndtr(double)"/> / <see cref="Special.LogNdtr(double)"/>
/// of the reflected argument, so <c>Sf(30)</c> is 4.9e-198 rather than 1 - 1 = 0, and <c>LogSf(40)</c> is
/// log(3.7e-350) even though <c>Sf(40)</c> itself underflows. Variates come from a 256-layer ziggurat.</remarks>
public sealed class NormalDistribution : ContinuousDistribution
{
    /// <summary>A normal distribution with mean <paramref name="loc"/> and standard deviation <paramref name="scale"/>.</summary>
    public NormalDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "norm";

    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new NormalDistribution(loc, scale);

    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(-0.5 * z * z) / Sqrt2Pi;
    /// <inheritdoc />
    protected override double LogPdf0(double z) => -0.5 * z * z - LnSqrt2Pi;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Ndtr(z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Ndtr(-z);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Special.LogNdtr(z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => Special.LogNdtr(-z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Special.Ndtri(q);
    /// <inheritdoc />
    protected override double Isf0(double q) => -Special.Ndtri(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(0, 1, 0, 0);
    /// <inheritdoc />
    protected override double Munp0(int n) => n % 2 == 1 ? 0.0 : Special.Factorial2(n - 1);
    /// <inheritdoc />
    protected override double Entropy0() => 0.5 * (Math.Log(2 * Math.PI) + 1);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Normal(rng);
    /// <inheritdoc />
    protected override void Rvs0(GraviRandom rng, Span<double> dest)
    {
        for (var i = 0; i < dest.Length; i++) dest[i] = StatsRandom.Normal(rng);
    }

    /// <summary>Closed-form MLE: the sample mean and the (biased, ddof = 0) standard deviation.</summary>
    public static NormalDistribution Fit(double[] data, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        var loc = floc ?? data.Average();
        if (fscale is double s) return new NormalDistribution(loc, s);
        var ss = 0.0;
        foreach (var x in data) ss += (x - loc) * (x - loc);
        return new NormalDistribution(loc, Math.Sqrt(ss / data.Length));
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null) => Fit(data, floc, fscale);
}

/// <summary>The uniform distribution on [loc, loc + scale], scipy's <c>uniform</c>.</summary>
public sealed class UniformDistribution : ContinuousDistribution
{
    /// <summary>Uniform on [loc, loc + scale].</summary>
    public UniformDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "uniform";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new UniformDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => 1.0;
    /// <inheritdoc />
    protected override double LogPdf0(double z) => 0.0;
    /// <inheritdoc />
    protected override double Cdf0(double z) => z;
    /// <inheritdoc />
    protected override double Sf0(double z) => 1.0 - z;
    /// <inheritdoc />
    protected override double LogSf0(double z) => Special.Log1p(-z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => q;
    /// <inheritdoc />
    protected override double Isf0(double q) => 1.0 - q;
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(0.5, 1.0 / 12.0, 0, -1.2);
    /// <inheritdoc />
    protected override double Munp0(int n) => 1.0 / (n + 1);
    /// <inheritdoc />
    protected override double Entropy0() => 0.0;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => rng.NextDouble();

    /// <summary>Closed-form MLE, as scipy: the sample range (centred on the data when only the scale is fixed).</summary>
    public static UniformDistribution Fit(double[] data, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        var min = data.Min();
        var max = data.Max();
        if (floc is double l && fscale is double s) return new UniformDistribution(l, s);
        if (floc is double l2)
        {
            if (l2 > min) throw new ArgumentException("floc must not exceed the smallest data point.", nameof(floc));
            return new UniformDistribution(l2, max - l2);
        }
        if (fscale is double s2)
        {
            var diff = s2 - (max - min);
            if (diff < 0) throw new ArgumentException("fscale is smaller than the data range.", nameof(fscale));
            return new UniformDistribution(min - 0.5 * diff, s2);
        }
        return new UniformDistribution(min, max - min);
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null) => Fit(data, floc, fscale);
}

/// <summary>The exponential distribution, scipy's <c>expon</c> (scale = 1/rate).</summary>
public sealed class ExponentialDistribution : ContinuousDistribution
{
    /// <summary>Exponential with the given location and scale (mean loc + scale).</summary>
    public ExponentialDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "expon";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new ExponentialDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(-z);
    /// <inheritdoc />
    protected override double LogPdf0(double z) => -z;
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(-z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(-z);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(-z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => -z;
    /// <inheritdoc />
    protected override double Ppf0(double q) => -Special.Log1p(-q);
    /// <inheritdoc />
    protected override double Isf0(double q) => -Math.Log(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(1, 1, 2, 6);
    /// <inheritdoc />
    protected override double Munp0(int n) => Special.Factorial(n);
    /// <inheritdoc />
    protected override double Entropy0() => 1.0;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => StatsRandom.Exponential(rng);

    /// <summary>Closed-form MLE, as scipy: loc = min(data), scale = mean − loc.</summary>
    public static ExponentialDistribution Fit(double[] data, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        var min = data.Min();
        var loc = floc ?? min;
        if (floc.HasValue && min < loc) throw new ArgumentException("floc exceeds the smallest data point.", nameof(floc));
        var scale = fscale ?? data.Average() - loc;
        return new ExponentialDistribution(loc, scale);
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null) => Fit(data, floc, fscale);
}

/// <summary>The Cauchy distribution, scipy's <c>cauchy</c>.</summary>
/// <remarks>The CDF uses atan(−1/z)/π for z &lt; 0 and the quantile −1/tan(πq) for q &lt; ½: the textbook
/// ½ + atan(z)/π and tan(π(q − ½)) both lose every digit in the tails.</remarks>
public sealed class CauchyDistribution : ContinuousDistribution
{
    /// <summary>Cauchy with median <paramref name="loc"/> and half-width <paramref name="scale"/>.</summary>
    public CauchyDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "cauchy";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new CauchyDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => 1.0 / (Math.PI * (1.0 + z * z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => -Math.Log(Math.PI) - Special.Log1p(z * z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => z < 0 ? Math.Atan(-1.0 / z) / Math.PI : 0.5 + Math.Atan(z) / Math.PI;
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(-z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => q < 0.5 ? -1.0 / Math.Tan(Math.PI * q) : q == 0.5 ? 0.0 : 1.0 / Math.Tan(Math.PI * (1.0 - q));
    /// <inheritdoc />
    protected override double Isf0(double q) => -Ppf0(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(double.NaN, double.NaN, double.NaN, double.NaN);
    /// <inheritdoc />
    protected override double Munp0(int n) => double.NaN;
    /// <inheritdoc />
    protected override double Entropy0() => Math.Log(4 * Math.PI);
    /// <inheritdoc />
    protected override double[] FitStart(double[] data)
    {
        // Moments do not exist; the median and half the interquartile range are the natural start.
        var s = (double[])data.Clone();
        Array.Sort(s);
        double Q(double p) { var h = (s.Length - 1) * p; var i = (int)Math.Floor(h); return i + 1 < s.Length ? s[i] + (h - i) * (s[i + 1] - s[i]) : s[i]; }
        var iqr = Q(0.75) - Q(0.25);
        return [Q(0.5), iqr > 0 ? iqr / 2 : 1.0];
    }
}

/// <summary>The Laplace (double exponential) distribution, scipy's <c>laplace</c>.</summary>
public sealed class LaplaceDistribution : ContinuousDistribution
{
    /// <summary>Laplace with the given location and scale.</summary>
    public LaplaceDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "laplace";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LaplaceDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => 0.5 * Math.Exp(-Math.Abs(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => -Ln2 - Math.Abs(z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => z < 0 ? 0.5 * Math.Exp(z) : 1.0 - 0.5 * Math.Exp(-z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(-z);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => z < 0 ? z - Ln2 : Special.Log1p(-0.5 * Math.Exp(-z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => LogCdf0(-z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => q <= 0.5 ? Math.Log(2 * q) : -Math.Log(2 * (1 - q));
    /// <inheritdoc />
    protected override double Isf0(double q) => -Ppf0(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(0, 2, 0, 3);
    /// <inheritdoc />
    protected override double Munp0(int n) => n % 2 == 1 ? 0.0 : Special.Factorial(n);
    /// <inheritdoc />
    protected override double Entropy0() => Ln2 + 1;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => (rng.NextUInt64() & 1) == 0 ? StatsRandom.Exponential(rng) : -StatsRandom.Exponential(rng);

    /// <summary>Closed-form MLE: the sample median and the mean absolute deviation about it.</summary>
    public static LaplaceDistribution Fit(double[] data, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        var loc = floc ?? StatsFitHelpers.Median(data);
        var scale = fscale ?? data.Average(x => Math.Abs(x - loc));
        return new LaplaceDistribution(loc, scale);
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null) => Fit(data, floc, fscale);
}

/// <summary>The logistic distribution, scipy's <c>logistic</c>.</summary>
public sealed class LogisticDistribution : ContinuousDistribution
{
    /// <summary>Logistic with the given location and scale.</summary>
    public LogisticDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "logistic";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LogisticDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z)
    {
        var e = Math.Exp(-Math.Abs(z));
        return e / ((1 + e) * (1 + e));
    }
    /// <inheritdoc />
    protected override double LogPdf0(double z) { var a = Math.Abs(z); return -a - 2 * Special.Log1p(Math.Exp(-a)); }
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Expit(z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Expit(-z);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Special.LogExpit(z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => Special.LogExpit(-z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Log(q) - Special.Log1p(-q);
    /// <inheritdoc />
    protected override double Isf0(double q) => -Ppf0(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(0, Math.PI * Math.PI / 3, 0, 1.2);
    /// <inheritdoc />
    protected override double Entropy0() => 2.0;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) { var u = StatsRandom.U(rng); return Math.Log(u) - Special.Log1p(-u); }
}

/// <summary>The right-skewed Gumbel (extreme value type I for maxima), scipy's <c>gumbel_r</c>.</summary>
public sealed class GumbelRDistribution : ContinuousDistribution
{
    /// <summary>Gumbel (maxima) with the given location and scale.</summary>
    public GumbelRDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "gumbel_r";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GumbelRDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => -z - Math.Exp(-z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => Math.Exp(-Math.Exp(-z));
    /// <inheritdoc />
    protected override double Sf0(double z) => -Special.Expm1(-Math.Exp(-z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => -Math.Exp(-z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => Log1mExp(-Math.Exp(-z));
    /// <inheritdoc />
    protected override double Ppf0(double q) => -Math.Log(-Math.Log(q));
    /// <inheritdoc />
    protected override double Isf0(double q) => -Math.Log(-Special.Log1p(-q));
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(EulerGamma, Math.PI * Math.PI / 6, 12 * Math.Sqrt(6) * 1.2020569031595942854 / (Math.PI * Math.PI * Math.PI), 2.4);
    /// <inheritdoc />
    protected override double Entropy0() => EulerGamma + 1;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => -Math.Log(StatsRandom.Exponential(rng));

    /// <summary>
    /// MLE. With the location free this is the one-dimensional scale equation
    /// s = mean(x) − Σx·e^(−x/s) / Σe^(−x/s) (scipy solves the same one), then loc = −s·log mean e^(−x/s).
    /// </summary>
    public static GumbelRDistribution Fit(double[] data, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        var (loc, scale) = StatsFitHelpers.GumbelR(data, floc, fscale);
        return new GumbelRDistribution(loc, scale);
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null) => Fit(data, floc, fscale);
}

/// <summary>The left-skewed Gumbel (extreme value type I for minima), scipy's <c>gumbel_l</c>.</summary>
public sealed class GumbelLDistribution : ContinuousDistribution
{
    /// <summary>Gumbel (minima) with the given location and scale.</summary>
    public GumbelLDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "gumbel_l";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new GumbelLDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => z - Math.Exp(z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(-Math.Exp(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(-Math.Exp(z));
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(-Math.Exp(z));
    /// <inheritdoc />
    protected override double LogSf0(double z) => -Math.Exp(z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Log(-Special.Log1p(-q));
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Log(-Math.Log(q));
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(-EulerGamma, Math.PI * Math.PI / 6, -12 * Math.Sqrt(6) * 1.2020569031595942854 / (Math.PI * Math.PI * Math.PI), 2.4);
    /// <inheritdoc />
    protected override double Entropy0() => EulerGamma + 1;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Log(StatsRandom.Exponential(rng));

    /// <summary>MLE via the <see cref="GumbelRDistribution"/> fit of the negated data.</summary>
    public static GumbelLDistribution Fit(double[] data, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        var neg = data.Select(x => -x).ToArray();
        var (loc, scale) = StatsFitHelpers.GumbelR(neg, floc.HasValue ? -floc : null, fscale);
        return new GumbelLDistribution(-loc, scale);
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null) => Fit(data, floc, fscale);
}

/// <summary>The half-normal distribution |Z|, scipy's <c>halfnorm</c>.</summary>
public sealed class HalfNormalDistribution : ContinuousDistribution
{
    /// <summary>Half-normal with the given location and scale.</summary>
    public HalfNormalDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "halfnorm";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new HalfNormalDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => 2 * Math.Exp(-0.5 * z * z) / Sqrt2Pi;
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Ln2 - 0.5 * z * z - LnSqrt2Pi;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Erf(z / Sqrt2);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Erfc(z / Sqrt2);
    /// <inheritdoc />
    protected override double LogSf0(double z) => z < 1 ? Special.Log1p(-Special.Erf(z / Sqrt2)) : Ln2 + Special.LogNdtr(-z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Sqrt2 * Special.ErfInv(q);
    /// <inheritdoc />
    protected override double Isf0(double q) => Sqrt2 * Special.ErfcInv(q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
        => new(Math.Sqrt(2 / Math.PI), 1 - 2 / Math.PI,
               Math.Sqrt(2) * (4 - Math.PI) / Math.Pow(Math.PI - 2, 1.5),
               8 * (Math.PI - 3) / ((Math.PI - 2) * (Math.PI - 2)));
    /// <inheritdoc />
    protected override double Entropy0() => 0.5 * Math.Log(Math.PI / 2) + 0.5;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Abs(StatsRandom.Normal(rng));

    /// <summary>Closed-form MLE, as scipy: loc = min(data), scale = root mean square of x − loc.</summary>
    public static HalfNormalDistribution Fit(double[] data, double? floc = null, double? fscale = null)
    {
        StatsDistributions.CheckData(data);
        var loc = floc ?? data.Min();
        if (fscale is double s) return new HalfNormalDistribution(loc, s);
        var ss = 0.0;
        foreach (var x in data) ss += (x - loc) * (x - loc);
        return new HalfNormalDistribution(loc, Math.Sqrt(ss / data.Length));
    }

    /// <inheritdoc />
    public override ContinuousDistribution FitTo(double[] data, double?[]? fixedShapes = null, double? floc = null, double? fscale = null) => Fit(data, floc, fscale);
}

/// <summary>The half-Cauchy distribution, scipy's <c>halfcauchy</c>.</summary>
public sealed class HalfCauchyDistribution : ContinuousDistribution
{
    /// <summary>Half-Cauchy with the given location and scale.</summary>
    public HalfCauchyDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "halfcauchy";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new HalfCauchyDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => 2.0 / (Math.PI * (1 + z * z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(2 / Math.PI) - Special.Log1p(z * z);
    /// <inheritdoc />
    protected override double Cdf0(double z) => 2.0 / Math.PI * Math.Atan(z);
    /// <inheritdoc />
    protected override double Sf0(double z) => 2.0 / Math.PI * Math.Atan(1.0 / z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Tan(Math.PI / 2 * q);
    /// <inheritdoc />
    protected override double Isf0(double q) => 1.0 / Math.Tan(Math.PI / 2 * q);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(double.PositiveInfinity, double.PositiveInfinity, double.NaN, double.NaN);
    /// <inheritdoc />
    protected override double Munp0(int n) => double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Entropy0() => Math.Log(2 * Math.PI);
}

/// <summary>The Rayleigh distribution, scipy's <c>rayleigh</c>.</summary>
public sealed class RayleighDistribution : ContinuousDistribution
{
    /// <summary>Rayleigh with the given location and scale.</summary>
    public RayleighDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "rayleigh";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new RayleighDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z * Math.Exp(-0.5 * z * z);
    /// <inheritdoc />
    protected override double LogPdf0(double z) => Math.Log(z) - 0.5 * z * z;
    /// <inheritdoc />
    protected override double Cdf0(double z) => -Special.Expm1(-0.5 * z * z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Math.Exp(-0.5 * z * z);
    /// <inheritdoc />
    protected override double LogCdf0(double z) => Log1mExp(-0.5 * z * z);
    /// <inheritdoc />
    protected override double LogSf0(double z) => -0.5 * z * z;
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Sqrt(-2 * Special.Log1p(-q));
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Sqrt(-2 * Math.Log(q));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var v = (4 - Math.PI) / 2;
        return new(Math.Sqrt(Math.PI / 2), v, 2 * Math.Sqrt(Math.PI) * (Math.PI - 3) / Math.Pow(4 - Math.PI, 1.5),
            -(6 * Math.PI * Math.PI - 24 * Math.PI + 16) / ((4 - Math.PI) * (4 - Math.PI)));
    }
    /// <inheritdoc />
    protected override double Munp0(int n) => Math.Pow(2, n / 2.0) * Special.Gamma(1 + n / 2.0);
    /// <inheritdoc />
    protected override double Entropy0() => EulerGamma / 2 + 1 - 0.5 * Math.Log(2);
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) => Math.Sqrt(2 * StatsRandom.Exponential(rng));
}

/// <summary>The Maxwell–Boltzmann speed distribution, scipy's <c>maxwell</c> (chi with 3 degrees of freedom).</summary>
public sealed class MaxwellDistribution : ContinuousDistribution
{
    /// <summary>Maxwell with the given location and scale.</summary>
    public MaxwellDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "maxwell";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new MaxwellDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Sqrt(2 / Math.PI) * z * z * Math.Exp(-0.5 * z * z);
    /// <inheritdoc />
    protected override double LogPdf0(double z) => 0.5 * Math.Log(2 / Math.PI) + 2 * Math.Log(z) - 0.5 * z * z;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.GammaInc(1.5, 0.5 * z * z);
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.GammaIncC(1.5, 0.5 * z * z);
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Sqrt(2 * Special.GammaIncInv(1.5, q));
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Sqrt(2 * Special.GammaIncCInv(1.5, q));
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var pi = Math.PI;
        return new(2 * Math.Sqrt(2 / pi), 3 - 8 / pi, 2 * Math.Sqrt(2) * (16 - 5 * pi) / Math.Pow(3 * pi - 8, 1.5),
            4 * (-96 + 40 * pi - 3 * pi * pi) / ((3 * pi - 8) * (3 * pi - 8)));
    }
    /// <inheritdoc />
    protected override double Entropy0() => EulerGamma + 0.5 * Math.Log(2 * Math.PI) - 0.5;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        var a = StatsRandom.Normal(rng);
        var b = StatsRandom.Normal(rng);
        var c = StatsRandom.Normal(rng);
        return Math.Sqrt(a * a + b * b + c * c);
    }
}

/// <summary>The arcsine distribution on [0, 1], scipy's <c>arcsine</c> (Beta(½, ½)).</summary>
public sealed class ArcsineDistribution : ContinuousDistribution
{
    /// <summary>Arcsine with the given location and scale.</summary>
    public ArcsineDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "arcsine";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new ArcsineDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => 1.0 / (Math.PI * Math.Sqrt(z * (1 - z)));
    /// <inheritdoc />
    protected override double Cdf0(double z) => 2.0 / Math.PI * Math.Asin(Math.Sqrt(z));
    /// <inheritdoc />
    protected override double Sf0(double z) => 2.0 / Math.PI * Math.Acos(Math.Sqrt(z));
    /// <inheritdoc />
    protected override double Ppf0(double q)
    {
        // Above ½ use the complement so that q near 1 is not rounded away inside the sine.
        var s = q <= 0.5 ? Math.Sin(Math.PI / 2 * q) : Math.Cos(Math.PI / 2 * (1 - q));
        return s * s;
    }
    /// <inheritdoc />
    protected override double Isf0(double q)
    {
        var c = q <= 0.5 ? Math.Cos(Math.PI / 2 * q) : Math.Sin(Math.PI / 2 * (1 - q));
        return c * c;
    }
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(0.5, 0.125, 0, -1.5);
    /// <inheritdoc />
    protected override double Entropy0() => Math.Log(Math.PI / 4);
}

/// <summary>The Wigner semicircle distribution on [−1, 1], scipy's <c>semicircular</c>.</summary>
public sealed class SemicircularDistribution : ContinuousDistribution
{
    /// <summary>Semicircular with the given location and scale.</summary>
    public SemicircularDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "semicircular";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new SemicircularDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => -1.0;
    /// <inheritdoc />
    protected override double Upper0 => 1.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => 2.0 / Math.PI * Math.Sqrt((1 - z) * (1 + z));
    /// <inheritdoc />
    protected override double Cdf0(double z) => 0.5 + (z * Math.Sqrt((1 - z) * (1 + z)) + Math.Asin(z)) / Math.PI;
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(-z);
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(0, 0.25, 0, -1);
    /// <inheritdoc />
    protected override double Entropy0() => 0.64472988584940017414;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng)
    {
        // Radius sqrt(U) of a uniform point in the unit disc, projected: x = r cos θ.
        var r = Math.Sqrt(rng.NextDouble());
        return r * Math.Cos(Math.PI * StatsRandom.U(rng));
    }
}

/// <summary>The raised-cosine distribution on [−π, π], scipy's <c>cosine</c>.</summary>
public sealed class CosineDistribution : ContinuousDistribution
{
    /// <summary>Cosine with the given location and scale.</summary>
    public CosineDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "cosine";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new CosineDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => -Math.PI;
    /// <inheritdoc />
    protected override double Upper0 => Math.PI;
    /// <inheritdoc />
    protected override double Pdf0(double z) => (1 + Math.Cos(z)) / (2 * Math.PI);
    /// <inheritdoc />
    protected override double Cdf0(double z)
    {
        // Near −π the expression π + z + sin z cancels to third order; use its Taylor series there.
        var t = z + Math.PI;
        if (t < 0.1) { var t2 = t * t; return t * t2 / 6 * (1 - t2 / 20 * (1 - t2 / 42 * (1 - t2 / 72))) / (2 * Math.PI); }
        return (Math.PI + z + Math.Sin(z)) / (2 * Math.PI);
    }
    /// <inheritdoc />
    protected override double Sf0(double z) => Cdf0(-z);
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var pi2 = Math.PI * Math.PI;
        return new(0, pi2 / 3 - 2, 0, -6 * (pi2 * pi2 - 90) / (5 * (pi2 - 6) * (pi2 - 6)));
    }
    /// <inheritdoc />
    protected override double Entropy0() => Math.Log(4 * Math.PI) - 1;
}

/// <summary>The anglit distribution on [−π/4, π/4], scipy's <c>anglit</c>.</summary>
public sealed class AnglitDistribution : ContinuousDistribution
{
    /// <summary>Anglit with the given location and scale.</summary>
    public AnglitDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "anglit";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new AnglitDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => -Math.PI / 4;
    /// <inheritdoc />
    protected override double Upper0 => Math.PI / 4;
    /// <inheritdoc />
    protected override double Pdf0(double z) => Math.Cos(2 * z);
    /// <inheritdoc />
    protected override double Cdf0(double z) { var s = Math.Sin(z + Math.PI / 4); return s * s; }
    /// <inheritdoc />
    protected override double Sf0(double z) { var c = Math.Cos(z + Math.PI / 4); return c * c; }
    /// <inheritdoc />
    protected override double Ppf0(double q) => Math.Asin(Math.Sqrt(q)) - Math.PI / 4;
    /// <inheritdoc />
    protected override double Isf0(double q) => Math.Acos(Math.Sqrt(q)) - Math.PI / 4;
    /// <inheritdoc />
    protected override DistributionMoments Moments0()
    {
        var pi2 = Math.PI * Math.PI;
        return new(0, pi2 / 16 - 0.5, 0, -2 * (pi2 * pi2 - 96) / ((pi2 - 8) * (pi2 - 8)));
    }
    /// <inheritdoc />
    protected override double Entropy0() => 1 - Math.Log(2);
}

/// <summary>The Lévy distribution, scipy's <c>levy</c> (a stable law with α = ½).</summary>
public sealed class LevyDistribution : ContinuousDistribution
{
    /// <summary>Lévy with the given location and scale.</summary>
    public LevyDistribution(double loc = 0.0, double scale = 1.0) : base(loc, scale) { }

    /// <inheritdoc />
    public override string Name => "levy";
    /// <inheritdoc />
    public override ContinuousDistribution WithParameters(double[] shapes, double loc, double scale) => new LevyDistribution(loc, scale);
    /// <inheritdoc />
    protected override double Lower0 => 0.0;
    /// <inheritdoc />
    protected override double Pdf0(double z) => z == 0 ? 0.0 : Math.Exp(LogPdf0(z));
    /// <inheritdoc />
    protected override double LogPdf0(double z) => z == 0 ? double.NegativeInfinity : -LnSqrt2Pi - 1.5 * Math.Log(z) - 0.5 / z;
    /// <inheritdoc />
    protected override double Cdf0(double z) => Special.Erfc(1 / Math.Sqrt(2 * z));
    /// <inheritdoc />
    protected override double Sf0(double z) => Special.Erf(1 / Math.Sqrt(2 * z));
    /// <inheritdoc />
    protected override double Ppf0(double q) { var e = Special.ErfcInv(q); return 0.5 / (e * e); }
    /// <inheritdoc />
    protected override double Isf0(double q) { var e = Special.ErfInv(q); return 0.5 / (e * e); }
    /// <inheritdoc />
    protected override DistributionMoments Moments0() => new(double.PositiveInfinity, double.PositiveInfinity, double.NaN, double.NaN);
    /// <inheritdoc />
    protected override double Munp0(int n) => double.PositiveInfinity;
    /// <inheritdoc />
    protected override double Entropy0() => (1 + 3 * EulerGamma + Math.Log(16 * Math.PI)) / 2;
    /// <inheritdoc />
    protected override double Rvs0(GraviRandom rng) { var z = StatsRandom.Normal(rng); return 1 / (z * z); }
}
