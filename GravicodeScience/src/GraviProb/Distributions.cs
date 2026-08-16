using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;

namespace Gravicode.Science.GraviProb;

/// <summary>
/// A univariate probability distribution.
/// </summary>
/// <remarks>
/// <see cref="LogDensity"/> rather than a plain density is the primitive because inference
/// multiplies many densities together: in linear space a few hundred observations underflow to
/// zero, while in log space they simply add. Every sampler in this library works with log
/// densities for that reason.
/// </remarks>
public abstract class Distribution
{
    /// <summary>Name used in reports.</summary>
    public abstract string Name { get; }

    /// <summary>Natural log of the density (or mass) at <paramref name="x"/>.</summary>
    public abstract double LogDensity(double x);

    /// <summary>
    /// The same log density built as a tape expression, so it can be differentiated with respect
    /// to <paramref name="x"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is what lets Hamiltonian Monte Carlo and variational inference get an exact gradient
    /// of the log posterior instead of a finite-difference estimate. The distribution's own
    /// parameters are fixed at construction, so this differentiates with respect to the value
    /// only — which is exactly what a prior on a latent variable needs. A likelihood whose
    /// parameters are themselves latent goes through <see cref="DistributionSpec"/> instead.
    /// </para>
    /// <para>
    /// Notice that no term here ever needs a differentiable log-gamma: every <c>lgamma</c> in
    /// these densities takes a fixed hyperparameter or an observed count, never a latent, so it
    /// stays a constant and the digamma function is never required.
    /// </para>
    /// <para>
    /// Distributions that do not override this cannot be used with gradient-based inference;
    /// <see cref="IsDifferentiable"/> reports which.
    /// </para>
    /// </remarks>
    public virtual Tensor LogDensity(Tensor x)
        => throw new NotSupportedException(
            $"{Name} has no differentiable log density, so it cannot be used with gradient-based inference.");

    /// <summary>Whether <see cref="LogDensity(Tensor)"/> is implemented.</summary>
    public virtual bool IsDifferentiable => false;

    /// <summary>Draws one value.</summary>
    public abstract double Sample(GraviRandom rng);

    /// <summary>Expected value.</summary>
    public abstract double Mean { get; }

    /// <summary>Variance.</summary>
    public abstract double Variance { get; }

    /// <summary>Standard deviation.</summary>
    public double StandardDeviation => Math.Sqrt(Variance);

    /// <summary>True when <paramref name="x"/> lies in the distribution's support.</summary>
    public virtual bool Supports(double x) => !double.IsNaN(x);

    /// <summary>
    /// The interval the distribution puts mass on, as (low, high) with infinities for unbounded ends.
    /// </summary>
    /// <remarks>
    /// Gradient-based inference needs this. Variational inference optimises an unconstrained
    /// Gaussian, so a parameter living on (0, 1) or (0, infinity) has to be mapped to the whole
    /// real line first - see <c>Inference.MeanFieldVariational</c>. Without the bounds the
    /// optimiser walks straight out of the support and the fit is meaningless.
    /// </remarks>
    public virtual (double Low, double High) SupportBounds
        => (double.NegativeInfinity, double.PositiveInfinity);

    /// <summary>Density (or mass) at <paramref name="x"/>.</summary>
    public double Density(double x) => Math.Exp(LogDensity(x));

    /// <summary>Cumulative distribution function.</summary>
    public virtual double Cdf(double x) => throw new NotSupportedException($"{Name} has no closed-form CDF.");

    /// <summary>Draws <paramref name="count"/> independent values.</summary>
    public NdArray Sample(GraviRandom rng, int count)
    {
        var result = NdArray.Zeros(count);
        for (var i = 0; i < count; i++) result.SetAt(i, Sample(rng));
        return result;
    }

    /// <summary>Total log density of a dataset under this distribution.</summary>
    public double LogLikelihood(IReadOnlyList<double> data)
    {
        var total = 0.0;
        foreach (var x in data) total += LogDensity(x);
        return total;
    }

    /// <inheritdoc />
    public override string ToString() => $"{Name}(mean={Mean:G6}, sd={StandardDeviation:G6})";

    // ---------------------------------------------------------------- factories

    /// <summary>A normal (Gaussian) distribution.</summary>
    public static Normal Normal(double mean = 0.0, double stdDev = 1.0) => new(mean, stdDev);

    /// <summary>A uniform distribution over <c>[low, high]</c>.</summary>
    public static Uniform Uniform(double low = 0.0, double high = 1.0) => new(low, high);

    /// <summary>A Bernoulli distribution.</summary>
    public static Bernoulli Bernoulli(double p) => new(p);

    /// <summary>A binomial distribution.</summary>
    public static Binomial Binomial(int trials, double probability) => new(trials, probability);

    /// <summary>A Poisson distribution.</summary>
    public static Poisson Poisson(double rate) => new(rate);

    /// <summary>A gamma distribution in the shape/rate parameterisation.</summary>
    public static Gamma Gamma(double shape, double rate = 1.0) => new(shape, rate);

    /// <summary>A beta distribution.</summary>
    public static Beta Beta(double alpha, double beta) => new(alpha, beta);

    /// <summary>An exponential distribution.</summary>
    public static Exponential Exponential(double rate = 1.0) => new(rate);

    /// <summary>A Student-t distribution.</summary>
    public static StudentT StudentT(double degreesOfFreedom, double location = 0.0, double scale = 1.0)
        => new(degreesOfFreedom, location, scale);

    /// <summary>A log-normal distribution.</summary>
    public static LogNormal LogNormal(double mu = 0.0, double sigma = 1.0) => new(mu, sigma);

    /// <summary>A half-normal distribution, the usual weakly informative prior for a scale.</summary>
    public static HalfNormal HalfNormal(double sigma = 1.0) => new(sigma);
}

/// <summary>The normal distribution.</summary>
public sealed class Normal(double mean, double stdDev) : Distribution
{
    private readonly double _logNormaliser = -Math.Log(stdDev) - 0.5 * Math.Log(2 * Math.PI);

    /// <inheritdoc />
    public override string Name => $"Normal({mean:G4}, {stdDev:G4})";

    /// <inheritdoc />
    public override double Mean => mean;

    /// <inheritdoc />
    public override double Variance => stdDev * stdDev;

    /// <summary>The scale parameter.</summary>
    public double Sigma => stdDev;

    /// <inheritdoc />
    public override double LogDensity(double x)
    {
        if (stdDev <= 0) return double.NegativeInfinity;
        var z = (x - mean) / stdDev;
        return _logNormaliser - 0.5 * z * z;
    }

    /// <inheritdoc />
    public override bool IsDifferentiable => stdDev > 0;

    /// <inheritdoc />
    public override Tensor LogDensity(Tensor x)
    {
        var z = (x - Tensor.Constant(mean)) / Tensor.Constant(stdDev);
        return Tensor.Constant(_logNormaliser) - Tensor.Constant(0.5) * z * z;
    }

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => rng.Normal(mean, stdDev);

    /// <inheritdoc />
    public override double Cdf(double x) => MathUtil.NormalCdf((x - mean) / stdDev);

    /// <summary>The inverse CDF.</summary>
    public double Quantile(double p) => mean + stdDev * MathUtil.NormalQuantile(p);
}

/// <summary>The continuous uniform distribution.</summary>
public sealed class Uniform(double low, double high) : Distribution
{
    /// <inheritdoc />
    public override string Name => $"Uniform({low:G4}, {high:G4})";

    /// <inheritdoc />
    public override double Mean => (low + high) / 2;

    /// <inheritdoc />
    public override double Variance => (high - low) * (high - low) / 12;

    /// <inheritdoc />
    public override (double Low, double High) SupportBounds => (low, high);

    /// <inheritdoc />
    public override bool Supports(double x) => x >= low && x <= high;

    /// <inheritdoc />
    public override double LogDensity(double x) => Supports(x) ? -Math.Log(high - low) : double.NegativeInfinity;

    /// <inheritdoc />
    public override bool IsDifferentiable => high > low;

    /// <inheritdoc />
    /// <remarks>
    /// Flat inside the support, so the gradient is zero — correct, and the reason a uniform prior
    /// contributes nothing to the Hamiltonian beyond bounding the region.
    /// </remarks>
    public override Tensor LogDensity(Tensor x)
        => Tensor.Constant(-Math.Log(high - low)) + Tensor.Constant(0.0) * x;

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => rng.Uniform(low, high);

    /// <inheritdoc />
    public override double Cdf(double x) => x <= low ? 0 : x >= high ? 1 : (x - low) / (high - low);
}

/// <summary>A single-trial Bernoulli distribution.</summary>
public sealed class Bernoulli(double p) : Distribution
{
    /// <inheritdoc />
    public override string Name => $"Bernoulli({p:G4})";

    /// <inheritdoc />
    public override double Mean => p;

    /// <inheritdoc />
    public override double Variance => p * (1 - p);

    /// <summary>The success probability.</summary>
    public double P => p;

    /// <inheritdoc />
    public override bool Supports(double x) => x is 0.0 or 1.0;

    /// <inheritdoc />
    public override double LogDensity(double x)
    {
        if (p is < 0 or > 1) return double.NegativeInfinity;
        if (x == 1.0) return p <= 0 ? double.NegativeInfinity : Math.Log(p);
        if (x == 0.0) return p >= 1 ? double.NegativeInfinity : Math.Log(1 - p);
        return double.NegativeInfinity;
    }

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => rng.NextDouble() < p ? 1.0 : 0.0;
}

/// <summary>The binomial distribution.</summary>
public sealed class Binomial(int trials, double probability) : Distribution
{
    /// <inheritdoc />
    public override string Name => $"Binomial({trials}, {probability:G4})";

    /// <inheritdoc />
    public override double Mean => trials * probability;

    /// <inheritdoc />
    public override double Variance => trials * probability * (1 - probability);

    /// <summary>Number of trials.</summary>
    public int Trials => trials;

    /// <summary>Success probability of one trial.</summary>
    public double Probability => probability;

    /// <inheritdoc />
    public override bool Supports(double x) => x >= 0 && x <= trials && x == Math.Floor(x);

    /// <inheritdoc />
    public override double LogDensity(double x)
    {
        if (!Supports(x) || probability is < 0 or > 1) return double.NegativeInfinity;
        var k = (int)x;
        if (probability == 0) return k == 0 ? 0.0 : double.NegativeInfinity;
        if (probability == 1) return k == trials ? 0.0 : double.NegativeInfinity;

        return MathUtil.LogBinomialCoefficient(trials, k)
            + k * Math.Log(probability)
            + (trials - k) * Math.Log(1 - probability);
    }

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => rng.Binomial(trials, probability);
}

/// <summary>The Poisson distribution.</summary>
public sealed class Poisson(double rate) : Distribution
{
    /// <inheritdoc />
    public override string Name => $"Poisson({rate:G4})";

    /// <inheritdoc />
    public override double Mean => rate;

    /// <inheritdoc />
    public override double Variance => rate;

    /// <inheritdoc />
    public override bool Supports(double x) => x >= 0 && x == Math.Floor(x);

    /// <inheritdoc />
    public override double LogDensity(double x)
    {
        if (!Supports(x) || rate <= 0) return double.NegativeInfinity;
        return x * Math.Log(rate) - rate - MathUtil.LogFactorial((int)x);
    }

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => rng.Poisson(rate);
}

/// <summary>The gamma distribution in the shape/rate parameterisation.</summary>
public sealed class Gamma(double shape, double rate) : Distribution
{
    /// <inheritdoc />
    public override string Name => $"Gamma({shape:G4}, {rate:G4})";

    /// <inheritdoc />
    public override double Mean => shape / rate;

    /// <inheritdoc />
    public override double Variance => shape / (rate * rate);

    /// <inheritdoc />
    public override (double Low, double High) SupportBounds => (0.0, double.PositiveInfinity);

    /// <inheritdoc />
    public override bool Supports(double x) => x > 0;

    /// <inheritdoc />
    public override double LogDensity(double x)
    {
        if (!Supports(x) || shape <= 0 || rate <= 0) return double.NegativeInfinity;
        return shape * Math.Log(rate) + (shape - 1) * Math.Log(x) - rate * x - MathUtil.LogGamma(shape);
    }

    /// <inheritdoc />
    public override bool IsDifferentiable => shape > 0 && rate > 0;

    /// <inheritdoc />
    public override Tensor LogDensity(Tensor x)
        => Tensor.Constant(shape * Math.Log(rate) - MathUtil.LogGamma(shape))
           + Tensor.Constant(shape - 1) * x.Log()
           - Tensor.Constant(rate) * x;

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => rng.Gamma(shape, 1.0 / rate);

    /// <inheritdoc />
    public override double Cdf(double x) => x <= 0 ? 0 : MathUtil.GammaP(shape, rate * x);
}

/// <summary>The beta distribution.</summary>
public sealed class Beta(double alpha, double beta) : Distribution
{
    private readonly double _logNormaliser = -MathUtil.LogBeta(alpha, beta);

    /// <inheritdoc />
    public override string Name => $"Beta({alpha:G4}, {beta:G4})";

    /// <inheritdoc />
    public override double Mean => alpha / (alpha + beta);

    /// <inheritdoc />
    public override double Variance
        => alpha * beta / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1));

    /// <summary>The first shape parameter.</summary>
    public double Alpha => alpha;

    /// <summary>The second shape parameter.</summary>
    public double BetaParameter => beta;

    /// <inheritdoc />
    public override (double Low, double High) SupportBounds => (0.0, 1.0);

    /// <inheritdoc />
    public override bool Supports(double x) => x is > 0 and < 1;

    /// <inheritdoc />
    public override double LogDensity(double x)
    {
        if (x is <= 0 or >= 1 || alpha <= 0 || beta <= 0) return double.NegativeInfinity;
        return _logNormaliser + (alpha - 1) * Math.Log(x) + (beta - 1) * Math.Log(1 - x);
    }

    /// <inheritdoc />
    public override bool IsDifferentiable => alpha > 0 && beta > 0;

    /// <inheritdoc />
    public override Tensor LogDensity(Tensor x)
        => Tensor.Constant(_logNormaliser)
           + Tensor.Constant(alpha - 1) * x.Log()
           + Tensor.Constant(beta - 1) * (Tensor.Constant(1.0) - x).Log();

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => rng.Beta(alpha, beta);

    /// <inheritdoc />
    public override double Cdf(double x) => MathUtil.BetaInc(alpha, beta, x);

    /// <summary>
    /// The posterior after observing binomial data, using conjugacy.
    /// </summary>
    /// <remarks>
    /// Beta is conjugate to the binomial likelihood, so the posterior is available in closed form:
    /// <c>Beta(alpha + successes, beta + failures)</c>. Tests use this as ground truth for the
    /// MCMC samplers - if the sampler is right, its posterior mean must match this exactly.
    /// </remarks>
    public Beta PosteriorAfter(int successes, int failures) => new(alpha + successes, beta + failures);
}

/// <summary>The exponential distribution.</summary>
public sealed class Exponential(double rate) : Distribution
{
    /// <inheritdoc />
    public override string Name => $"Exponential({rate:G4})";

    /// <inheritdoc />
    public override double Mean => 1.0 / rate;

    /// <inheritdoc />
    public override double Variance => 1.0 / (rate * rate);

    /// <inheritdoc />
    public override (double Low, double High) SupportBounds => (0.0, double.PositiveInfinity);

    /// <inheritdoc />
    public override bool Supports(double x) => x >= 0;

    /// <inheritdoc />
    public override double LogDensity(double x)
        => !Supports(x) || rate <= 0 ? double.NegativeInfinity : Math.Log(rate) - rate * x;

    /// <inheritdoc />
    public override bool IsDifferentiable => rate > 0;

    /// <inheritdoc />
    public override Tensor LogDensity(Tensor x)
        => Tensor.Constant(Math.Log(rate)) - Tensor.Constant(rate) * x;

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => rng.Exponential(rate);

    /// <inheritdoc />
    public override double Cdf(double x) => x <= 0 ? 0 : 1 - Math.Exp(-rate * x);
}

/// <summary>The Student-t distribution, a heavy-tailed alternative to the normal.</summary>
public sealed class StudentT(double degreesOfFreedom, double location, double scale) : Distribution
{
    /// <inheritdoc />
    public override string Name => $"StudentT({degreesOfFreedom:G4}, {location:G4}, {scale:G4})";

    /// <inheritdoc />
    public override double Mean => degreesOfFreedom > 1 ? location : double.NaN;

    /// <inheritdoc />
    public override double Variance => degreesOfFreedom > 2
        ? scale * scale * degreesOfFreedom / (degreesOfFreedom - 2)
        : double.PositiveInfinity;

    /// <inheritdoc />
    public override double LogDensity(double x)
    {
        var z = (x - location) / scale;
        return MathUtil.LogGamma((degreesOfFreedom + 1) / 2)
            - MathUtil.LogGamma(degreesOfFreedom / 2)
            - 0.5 * Math.Log(degreesOfFreedom * Math.PI)
            - Math.Log(scale)
            - (degreesOfFreedom + 1) / 2 * Math.Log(1 + z * z / degreesOfFreedom);
    }

    /// <inheritdoc />
    public override bool IsDifferentiable => scale > 0 && degreesOfFreedom > 0;

    /// <inheritdoc />
    public override Tensor LogDensity(Tensor x)
    {
        var normaliser = MathUtil.LogGamma((degreesOfFreedom + 1) / 2)
            - MathUtil.LogGamma(degreesOfFreedom / 2)
            - 0.5 * Math.Log(degreesOfFreedom * Math.PI)
            - Math.Log(scale);

        var z = (x - Tensor.Constant(location)) / Tensor.Constant(scale);
        return Tensor.Constant(normaliser)
               - Tensor.Constant((degreesOfFreedom + 1) / 2)
                 * (Tensor.Constant(1.0) + z * z / Tensor.Constant(degreesOfFreedom)).Log();
    }

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => location + scale * rng.StudentT(degreesOfFreedom);
}

/// <summary>The log-normal distribution.</summary>
public sealed class LogNormal(double mu, double sigma) : Distribution
{
    /// <inheritdoc />
    public override string Name => $"LogNormal({mu:G4}, {sigma:G4})";

    /// <inheritdoc />
    public override double Mean => Math.Exp(mu + sigma * sigma / 2);

    /// <inheritdoc />
    public override double Variance => (Math.Exp(sigma * sigma) - 1) * Math.Exp(2 * mu + sigma * sigma);

    /// <inheritdoc />
    public override (double Low, double High) SupportBounds => (0.0, double.PositiveInfinity);

    /// <inheritdoc />
    public override bool Supports(double x) => x > 0;

    /// <inheritdoc />
    public override double LogDensity(double x)
    {
        if (!Supports(x)) return double.NegativeInfinity;
        var z = (Math.Log(x) - mu) / sigma;
        return -Math.Log(x * sigma * Math.Sqrt(2 * Math.PI)) - 0.5 * z * z;
    }

    /// <inheritdoc />
    public override bool IsDifferentiable => sigma > 0;

    /// <inheritdoc />
    public override Tensor LogDensity(Tensor x)
    {
        var z = (x.Log() - Tensor.Constant(mu)) / Tensor.Constant(sigma);
        return Tensor.Constant(-Math.Log(sigma * Math.Sqrt(2 * Math.PI)))
               - x.Log()
               - Tensor.Constant(0.5) * z * z;
    }

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => rng.LogNormal(mu, sigma);
}

/// <summary>A normal folded at zero; the standard weakly informative prior for a scale parameter.</summary>
public sealed class HalfNormal(double sigma) : Distribution
{
    /// <inheritdoc />
    public override string Name => $"HalfNormal({sigma:G4})";

    /// <inheritdoc />
    public override double Mean => sigma * Math.Sqrt(2 / Math.PI);

    /// <inheritdoc />
    public override double Variance => sigma * sigma * (1 - 2 / Math.PI);

    /// <inheritdoc />
    public override (double Low, double High) SupportBounds => (0.0, double.PositiveInfinity);

    /// <inheritdoc />
    public override bool Supports(double x) => x >= 0;

    /// <inheritdoc />
    public override double LogDensity(double x)
    {
        if (!Supports(x)) return double.NegativeInfinity;
        return 0.5 * Math.Log(2 / Math.PI) - Math.Log(sigma) - x * x / (2 * sigma * sigma);
    }

    /// <inheritdoc />
    public override bool IsDifferentiable => sigma > 0;

    /// <inheritdoc />
    public override Tensor LogDensity(Tensor x)
        => Tensor.Constant(0.5 * Math.Log(2 / Math.PI) - Math.Log(sigma))
           - x * x / Tensor.Constant(2 * sigma * sigma);

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => Math.Abs(rng.Normal(0, sigma));
}

/// <summary>A distribution over a fixed set of outcomes.</summary>
public sealed class Categorical : Distribution
{
    private readonly double[] _probabilities;

    /// <summary>Normalises <paramref name="weights"/> into a probability vector.</summary>
    public Categorical(IReadOnlyList<double> weights)
    {
        var total = weights.Sum();
        if (total <= 0) throw new ArgumentException("Weights must sum to a positive value.", nameof(weights));
        _probabilities = weights.Select(w => w / total).ToArray();
    }

    /// <summary>The normalised probabilities.</summary>
    public IReadOnlyList<double> Probabilities => _probabilities;

    /// <inheritdoc />
    public override string Name => $"Categorical({_probabilities.Length})";

    /// <inheritdoc />
    public override double Mean => _probabilities.Select((p, i) => p * i).Sum();

    /// <inheritdoc />
    public override double Variance
    {
        get
        {
            var mean = Mean;
            return _probabilities.Select((p, i) => p * (i - mean) * (i - mean)).Sum();
        }
    }

    /// <inheritdoc />
    public override bool Supports(double x) => x >= 0 && x < _probabilities.Length && x == Math.Floor(x);

    /// <inheritdoc />
    public override double LogDensity(double x)
        => !Supports(x) ? double.NegativeInfinity : Math.Log(Math.Max(_probabilities[(int)x], 1e-300));

    /// <inheritdoc />
    public override double Sample(GraviRandom rng) => rng.Categorical(_probabilities);
}
