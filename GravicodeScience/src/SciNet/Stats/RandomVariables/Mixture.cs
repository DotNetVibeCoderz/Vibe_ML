using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// A finite mixture of continuous random variables (scipy's <c>stats.Mixture</c>): a component is
/// picked with probability given by its weight, then realised.
/// </summary>
/// <remarks>
/// Densities and CDFs are the weighted sums of the components' (log-space sums for the logarithms);
/// the mean and higher moments are combined from the components' moments, re-centred on the mixture
/// mean; the quantiles invert the mixture CDF by root finding; the entropy is a quadrature of
/// −p log p and the mode a maximisation of the density started at 1, as in scipy. Like scipy's,
/// none of the methods takes a <c>method</c> argument.
/// </remarks>
public sealed class Mixture : IUnivariateDistribution
{
    private readonly ContinuousRandomVariable[] _components;
    private readonly double[] _weights;

    /// <summary>A mixture of <paramref name="components"/>; equal weights unless given (non-negative, summing to 1).</summary>
    public Mixture(IEnumerable<ContinuousRandomVariable> components, double[]? weights = null)
    {
        ArgumentNullException.ThrowIfNull(components);
        _components = components.ToArray();
        if (_components.Length == 0) throw new ArgumentException("`components` must contain at least one random variable.", nameof(components));
        foreach (var c in _components) ArgumentNullException.ThrowIfNull(c, nameof(components));
        if (weights is null)
        {
            _weights = Enumerable.Repeat(1.0 / _components.Length, _components.Length).ToArray();
        }
        else
        {
            if (weights.Length != _components.Length) throw new ArgumentException("`components` and `weights` must have the same length.", nameof(weights));
            var sum = weights.Sum();
            if (!(Math.Abs(sum - 1.0) <= 1e-8 + 1e-5)) throw new ArgumentException("`weights` must sum to 1.0.", nameof(weights));
            if (weights.Any(w => !(w >= 0))) throw new ArgumentException("All `weights` must be non-negative.", nameof(weights));
            _weights = (double[])weights.Clone();
        }
    }

    /// <summary>The components (a copy of the list).</summary>
    public IReadOnlyList<ContinuousRandomVariable> Components => _components.ToArray();

    /// <summary>The weights (a copy).</summary>
    public double[] Weights => (double[])_weights.Clone();

    private double Sum(Func<ContinuousRandomVariable, double> f)
    {
        var s = 0.0;
        for (var i = 0; i < _components.Length; i++) s += f(_components[i]) * _weights[i];
        return s;
    }

    private double LogSum(Func<ContinuousRandomVariable, double> f)
    {
        var s = double.NegativeInfinity;
        for (var i = 0; i < _components.Length; i++) s = StatsDistMath.LogAddExp(s, f(_components[i]) + Math.Log(_weights[i]));
        return s;
    }

    /// <summary>The union of the components' supports.</summary>
    public (double Lower, double Upper) Support()
    {
        double a = double.PositiveInfinity, b = double.NegativeInfinity;
        foreach (var c in _components)
        {
            var (l, u) = c.Support();
            a = Math.Min(a, l);
            b = Math.Max(b, u);
        }
        return (a, b);
    }

    /// <summary>Probability density.</summary>
    public double Pdf(double x) => Sum(c => c.Pdf(x));
    /// <summary>Log density.</summary>
    public double LogPdf(double x) => LogSum(c => c.LogPdf(x));
    /// <summary>Probability mass (zero: the components are continuous).</summary>
    public double Pmf(double x) => Sum(c => c.Pmf(x));
    /// <summary>Log mass (−∞).</summary>
    public double LogPmf(double x) => LogSum(c => c.LogPmf(x));
    /// <summary>CDF.</summary>
    public double Cdf(double x) => Sum(c => c.Cdf(x));
    /// <summary>P(x ≤ X ≤ y), negative when y &lt; x.</summary>
    public double Cdf(double x, double y) => Sum(c => c.Cdf(x, y));
    /// <summary>Log CDF.</summary>
    public double LogCdf(double x) => LogSum(c => c.LogCdf(x));
    /// <summary>Complementary CDF.</summary>
    public double Ccdf(double x) => Sum(c => c.Ccdf(x));
    /// <summary>Mass outside [x, y].</summary>
    public double Ccdf(double x, double y) => Sum(c => c.Ccdf(x, y));
    /// <summary>Log complementary CDF.</summary>
    public double LogCcdf(double x) => LogSum(c => c.LogCcdf(x));
    /// <summary>Log of the mass outside [x, y].</summary>
    public double LogCcdf(double x, double y) => LogSum(c => c.LogCcdf(x, y));

    private double Invert(Func<double, double> f, double p, bool increasing)
    {
        if (double.IsNaN(p)) return double.NaN;
        var (a, b) = Support();
        return RandomVariableNumerics.Solve(f, p, a, b, increasing);
    }

    /// <summary>Inverse CDF by root finding.</summary>
    public double Icdf(double p)
    {
        if (!(p >= 0 && p <= 1)) return double.NaN;
        var (a, b) = Support();
        if (p == 0) return a;
        if (p == 1) return b;
        return Invert(Cdf, p, true);
    }

    /// <summary>Inverse complementary CDF.</summary>
    public double Iccdf(double p)
    {
        if (!(p >= 0 && p <= 1)) return double.NaN;
        var (a, b) = Support();
        if (p == 0) return b;
        if (p == 1) return a;
        return Invert(Ccdf, p, false);
    }

    /// <summary>Inverse log CDF.</summary>
    public double ILogCdf(double logp) => logp > 0 ? double.NaN : Invert(LogCdf, logp, true);
    /// <summary>Inverse log complementary CDF.</summary>
    public double ILogCcdf(double logp) => logp > 0 ? double.NaN : Invert(LogCcdf, logp, false);

    /// <summary>Differential entropy by quadrature of −p log p.</summary>
    public double Entropy()
    {
        var (a, b) = Support();
        // Split at every component's finite support ends (where the density may jump) and at the
        // median; scipy integrates in one piece and is 1e-5 off for a uniform component.
        var cuts = _components.SelectMany(c => { var (l, u) = c.Support(); return new[] { l, u }; }).Append(Median());
        return RandomVariableNumerics.QuadSplit(x =>
        {
            var p = Pdf(x);
            return p == 0 ? 0.0 : -p * LogPdf(x);
        }, a, b, cuts, null);
    }

    /// <summary>Log of the entropy (complex when it is negative).</summary>
    public Complex LogEntropy() => RandomVariableNumerics.LogReal(Entropy());

    /// <summary>The mode, maximising the density from x = 1 as scipy does (so a multimodal mixture returns the peak found from there).</summary>
    public double Mode()
    {
        var (a, b) = Support();
        return RandomVariableNumerics.Minimize(x => -Pdf(x), 1.0, a, b);
    }

    /// <summary>The median, the 0.5 quantile.</summary>
    public double Median() => Icdf(0.5);
    /// <summary>The mean, the weighted mean of the components' means.</summary>
    public double Mean() => Sum(c => c.Mean());
    /// <summary>The variance.</summary>
    public double Variance() => MomentCentral(2);
    /// <summary>The standard deviation.</summary>
    public double StandardDeviation() => Math.Sqrt(Variance());
    /// <summary>The skewness.</summary>
    public double Skewness() => MomentStandardized(3);
    /// <summary>The (non-excess) kurtosis, scipy's <c>Mixture.kurtosis</c>.</summary>
    public double Kurtosis() => MomentStandardized(4);

    /// <summary>A raw, central or standardized moment.</summary>
    public double Moment(int order = 1, MomentKind kind = MomentKind.Raw)
    {
        if (order < 0) throw new ArgumentException("Argument `order` of `moment` must be a finite, positive integer.", nameof(order));
        return kind switch
        {
            MomentKind.Raw => Sum(c => c.Moment(order, MomentKind.Raw)),
            MomentKind.Central => MomentCentral(order),
            MomentKind.Standardized => MomentStandardized(order),
            _ => throw new ArgumentException("Unknown moment kind.", nameof(kind)),
        };
    }

    private double MomentCentral(int order)
    {
        var b = Mean();
        var total = 0.0;
        for (var i = 0; i < _components.Length; i++)
        {
            var c = _components[i];
            var moments = new double[order + 1];
            for (var k = 0; k <= order; k++) moments[k] = c.Moment(k, MomentKind.Central);
            total += UnivariateRandomVariable.TransformCenter(order, moments, c.Mean(), b) * _weights[i];
        }
        return total;
    }

    private double MomentStandardized(int order) => MomentCentral(order) / Math.Pow(StandardDeviation(), order);

    /// <summary>
    /// <paramref name="n"/> variates: the number from each component is multinomial in the weights,
    /// each component is sampled in bulk, and the result is shuffled (scipy's scheme).
    /// </summary>
    public double[] Sample(int n, GraviRandom? rng = null)
    {
        if (n < 0) throw new ArgumentException("n must be non-negative.", nameof(n));
        rng ??= StatsDistributions.MakeRng(null);
        var counts = new int[_components.Length];
        for (var i = 0; i < n; i++) counts[rng.Categorical(_weights)]++;
        var all = new double[n];
        var pos = 0;
        for (var i = 0; i < _components.Length; i++)
        {
            if (counts[i] == 0) continue;
            _components[i].Sample(counts[i], rng).CopyTo(all, pos);
            pos += counts[i];
        }
        for (var i = n - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (all[i], all[j]) = (all[j], all[i]);
        }
        return all;
    }

    /// <summary>One variate.</summary>
    public double Sample(GraviRandom? rng = null) => Sample(1, rng)[0];

    /// <summary>Variates in an array of the given shape.</summary>
    public NdArray Sample(int[] shape, GraviRandom? rng = null)
    {
        ArgumentNullException.ThrowIfNull(shape);
        var n = 1;
        foreach (var d in shape) n *= d;
        return new NdArray(Sample(n, rng), (int[])shape.Clone());
    }

    /// <inheritdoc />
    public override string ToString()
        => "Mixture(\n    [\n" + string.Concat(_components.Select(c => $"        {c},\n")) + "    ],\n    weights=[" + string.Join(", ", _weights.Select(PyFormat.F)) + "],\n)";

    // ------------------------------------------------------------------ IUnivariateDistribution
    string IUnivariateDistribution.Name => "Mixture";
    bool IUnivariateDistribution.IsDiscrete => false;
    double IUnivariateDistribution.Density(double x) => Pdf(x);
    double IUnivariateDistribution.LogDensity(double x) => LogPdf(x);
    double IUnivariateDistribution.Sf(double x) => Ccdf(x);
    double IUnivariateDistribution.LogSf(double x) => LogCcdf(x);
    double IUnivariateDistribution.Ppf(double q) => Icdf(q);
    double IUnivariateDistribution.Isf(double q) => Iccdf(q);
    double IUnivariateDistribution.Var() => Variance();
    double IUnivariateDistribution.Std() => StandardDeviation();
    double IUnivariateDistribution.Skew() => Skewness();
    double IUnivariateDistribution.Kurtosis() => Kurtosis() - 3;
    DistributionMoments IUnivariateDistribution.Stats() => new(Mean(), Variance(), Skewness(), Kurtosis() - 3);
    double IUnivariateDistribution.Moment(int n) => Moment(n);
    (double Lower, double Upper) IUnivariateDistribution.Interval(double confidence)
    {
        var a = 0.5 * (1 - confidence);
        return (Icdf(a), Iccdf(a));
    }
    double IUnivariateDistribution.Rvs(GraviRandom rng) => Sample(rng);
    double[] IUnivariateDistribution.Rvs(int size, GraviRandom rng) => Sample(size, rng);
}
