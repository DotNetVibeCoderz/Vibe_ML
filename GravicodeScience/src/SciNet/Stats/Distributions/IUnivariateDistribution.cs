using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>
/// The first four moments of a distribution, scipy's <c>stats(moments='mvsk')</c>. Kurtosis is the
/// excess (Fisher) kurtosis, zero for the normal, as scipy's <c>'k'</c>.
/// </summary>
/// <param name="Mean">The mean (may be infinite or NaN where it does not exist).</param>
/// <param name="Variance">The variance.</param>
/// <param name="Skewness">The skewness, E[(X−μ)³]/σ³.</param>
/// <param name="Kurtosis">The excess kurtosis, E[(X−μ)⁴]/σ⁴ − 3.</param>
public readonly record struct DistributionMoments(double Mean, double Variance, double Skewness, double Kurtosis);

/// <summary>
/// What every frozen univariate distribution offers, continuous or discrete — the surface that
/// generic code (goodness-of-fit tests, confidence intervals, resampling) programs against.
/// </summary>
/// <remarks>
/// For a discrete distribution <see cref="Density"/> is the probability mass function. <see cref="Sf"/>
/// and <see cref="Isf"/> are computed directly, never as <c>1 − Cdf</c>, so they stay accurate deep in
/// the upper tail where <c>Cdf</c> has already rounded to 1.
/// </remarks>
public interface IUnivariateDistribution
{
    /// <summary>scipy's name for the family, e.g. <c>"norm"</c>, <c>"binom"</c>.</summary>
    string Name { get; }

    /// <summary>True for distributions on the integers.</summary>
    bool IsDiscrete { get; }

    /// <summary>The closed support [lower, upper]; either end may be infinite.</summary>
    (double Lower, double Upper) Support();

    /// <summary>The density (pdf) or, for discrete distributions, the mass function (pmf).</summary>
    double Density(double x);

    /// <summary>The log density or log mass.</summary>
    double LogDensity(double x);

    /// <summary>P(X ≤ x).</summary>
    double Cdf(double x);

    /// <summary>log P(X ≤ x).</summary>
    double LogCdf(double x);

    /// <summary>P(X &gt; x), computed directly.</summary>
    double Sf(double x);

    /// <summary>log P(X &gt; x).</summary>
    double LogSf(double x);

    /// <summary>Quantile function, the inverse of <see cref="Cdf"/>.</summary>
    double Ppf(double q);

    /// <summary>Inverse survival function: the x with <c>Sf(x) = q</c>.</summary>
    double Isf(double q);

    /// <summary>The mean.</summary>
    double Mean();

    /// <summary>The variance.</summary>
    double Var();

    /// <summary>The standard deviation.</summary>
    double Std();

    /// <summary>The median, <c>Ppf(0.5)</c>.</summary>
    double Median();

    /// <summary>The skewness.</summary>
    double Skew();

    /// <summary>The excess kurtosis.</summary>
    double Kurtosis();

    /// <summary>All four moments at once.</summary>
    DistributionMoments Stats();

    /// <summary>The n-th non-central moment E[Xⁿ].</summary>
    double Moment(int n);

    /// <summary>The differential (continuous) or Shannon (discrete) entropy in nats.</summary>
    double Entropy();

    /// <summary>The equal-tailed interval containing the given probability mass.</summary>
    (double Lower, double Upper) Interval(double confidence);

    /// <summary>One random variate.</summary>
    double Rvs(GraviRandom rng);

    /// <summary><paramref name="size"/> random variates.</summary>
    double[] Rvs(int size, GraviRandom rng);
}
