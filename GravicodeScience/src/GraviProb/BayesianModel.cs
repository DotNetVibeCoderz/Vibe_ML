using System.Text;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviProb;

/// <summary>
/// A distribution that may depend on other model variables, resolved at evaluation time.
/// </summary>
/// <remarks>
/// This is what turns a collection of distributions into a model: writing
/// <c>Distribution.Binomial(10, "theta")</c> records a dependency on the latent variable
/// <c>theta</c> rather than a fixed probability, so the sampler can re-evaluate the likelihood
/// as <c>theta</c> moves.
/// </remarks>
public sealed class DistributionSpec
{
    private readonly Func<IReadOnlyDictionary<string, double>, Distribution> _resolve;

    private DistributionSpec(Func<IReadOnlyDictionary<string, double>, Distribution> resolve,
        IReadOnlyList<string> dependencies)
    {
        _resolve = resolve;
        Dependencies = dependencies;
    }

    /// <summary>Names of the variables this specification reads.</summary>
    public IReadOnlyList<string> Dependencies { get; }

    /// <summary>Builds the concrete distribution for a set of parameter values.</summary>
    public Distribution Resolve(IReadOnlyDictionary<string, double> values) => _resolve(values);

    /// <summary>Wraps a fixed distribution.</summary>
    public static DistributionSpec Constant(Distribution distribution) => new(_ => distribution, []);

    /// <summary>Builds a specification from an arbitrary function of the model's variables.</summary>
    public static DistributionSpec From(Func<IReadOnlyDictionary<string, double>, Distribution> resolve,
        params string[] dependencies) => new(resolve, dependencies);

    /// <summary>A fixed distribution used as a specification.</summary>
    public static implicit operator DistributionSpec(Distribution distribution) => Constant(distribution);

    /// <summary>A binomial likelihood whose success probability is a named model variable.</summary>
    public static DistributionSpec Binomial(int trials, string probabilityVariable)
        => From(values => new Binomial(trials, values[probabilityVariable]), probabilityVariable);

    /// <summary>A Bernoulli likelihood whose success probability is a named model variable.</summary>
    public static DistributionSpec Bernoulli(string probabilityVariable)
        => From(values => new Bernoulli(values[probabilityVariable]), probabilityVariable);

    /// <summary>A normal likelihood whose mean and scale are named model variables.</summary>
    public static DistributionSpec Normal(string meanVariable, string scaleVariable)
        => From(values => new Normal(values[meanVariable], values[scaleVariable]), meanVariable, scaleVariable);

    /// <summary>A normal likelihood with a fitted mean and a fixed scale.</summary>
    public static DistributionSpec Normal(string meanVariable, double scale)
        => From(values => new Normal(values[meanVariable], scale), meanVariable);

    /// <summary>A Poisson likelihood whose rate is a named model variable.</summary>
    public static DistributionSpec Poisson(string rateVariable)
        => From(values => new Poisson(values[rateVariable]), rateVariable);
}

/// <summary>
/// A probabilistic model built from named priors and observed likelihoods.
/// </summary>
/// <remarks>
/// <para>
/// The model is a joint log density over its latent variables: the sum of every prior's log
/// density at the current parameter values plus every observation's log likelihood. Inference
/// never needs anything else - <see cref="LogPosterior"/> is the single function all the samplers
/// in <see cref="Inference"/> explore.
/// </para>
/// <para>
/// The posterior is known only up to a constant, because the marginal likelihood in Bayes' rule
/// is not computed. That is exactly why MCMC is used: Metropolis-Hastings only ever looks at
/// ratios of densities, and the unknown constant cancels.
/// </para>
/// </remarks>
public sealed class BayesianModel
{
    private readonly List<(string Name, Distribution Prior)> _priors = [];
    private readonly List<(string Name, DistributionSpec Likelihood, double[] Data)> _observations = [];

    /// <summary>Names of the latent variables, in declaration order.</summary>
    public IReadOnlyList<string> ParameterNames => _priors.Select(p => p.Name).ToList();

    /// <summary>Number of latent variables.</summary>
    public int ParameterCount => _priors.Count;

    /// <summary>The priors, by variable name.</summary>
    public IReadOnlyDictionary<string, Distribution> Priors =>
        _priors.ToDictionary(p => p.Name, p => p.Prior, StringComparer.Ordinal);

    /// <summary>Number of observed data points across all likelihoods.</summary>
    public int ObservationCount => _observations.Sum(o => o.Data.Length);

    /// <summary>Declares a latent variable and its prior.</summary>
    public BayesianModel AddDistribution(string name, Distribution prior)
    {
        if (_priors.Any(p => p.Name == name))
            throw new ArgumentException($"Variable '{name}' is already declared.");
        _priors.Add((name, prior));
        return this;
    }

    /// <summary>Declares observed data and the likelihood that generated it.</summary>
    public BayesianModel AddObservation(string name, DistributionSpec likelihood, params double[] data)
    {
        foreach (var dependency in likelihood.Dependencies)
            if (!_priors.Any(p => p.Name == dependency))
                throw new ArgumentException(
                    $"Observation '{name}' depends on '{dependency}', which has no prior. Declare it with AddDistribution first.");

        _observations.Add((name, likelihood, data));
        return this;
    }

    /// <summary>Declares observed data from a sequence.</summary>
    public BayesianModel AddObservation(string name, DistributionSpec likelihood, IEnumerable<double> data)
        => AddObservation(name, likelihood, data.ToArray());

    /// <summary>
    /// Log of the unnormalised posterior: prior plus likelihood, evaluated at
    /// <paramref name="values"/>.
    /// </summary>
    public double LogPosterior(IReadOnlyDictionary<string, double> values)
    {
        var total = 0.0;

        foreach (var (name, prior) in _priors)
        {
            if (!values.TryGetValue(name, out var value)) return double.NegativeInfinity;
            var density = prior.LogDensity(value);
            // A proposal outside the support is rejected rather than allowed to poison the sum.
            if (double.IsNegativeInfinity(density) || double.IsNaN(density)) return double.NegativeInfinity;
            total += density;
        }

        foreach (var (_, likelihood, data) in _observations)
        {
            Distribution resolved;
            try { resolved = likelihood.Resolve(values); }
            catch (KeyNotFoundException) { return double.NegativeInfinity; }

            foreach (var x in data)
            {
                var density = resolved.LogDensity(x);
                if (double.IsNaN(density)) return double.NegativeInfinity;
                total += density;
                if (double.IsNegativeInfinity(total)) return double.NegativeInfinity;
            }
        }
        return total;
    }

    /// <summary>Log of the prior alone, useful for diagnosing a badly specified model.</summary>
    public double LogPrior(IReadOnlyDictionary<string, double> values)
    {
        var total = 0.0;
        foreach (var (name, prior) in _priors)
            total += values.TryGetValue(name, out var value) ? prior.LogDensity(value) : double.NegativeInfinity;
        return total;
    }

    /// <summary>Draws one set of parameter values from the priors.</summary>
    public Dictionary<string, double> SampleFromPrior(GraviRandom rng)
        => _priors.ToDictionary(p => p.Name, p => p.Prior.Sample(rng), StringComparer.Ordinal);

    /// <summary>
    /// Draws data from the model by sampling the priors and then the likelihoods - the prior
    /// predictive check that reveals a prior implying impossible data before any fitting happens.
    /// </summary>
    public IReadOnlyDictionary<string, NdArray> PriorPredictive(int draws, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var result = _observations.ToDictionary(o => o.Name, _ => NdArray.Zeros(draws), StringComparer.Ordinal);

        for (var d = 0; d < draws; d++)
        {
            var values = SampleFromPrior(rng);
            foreach (var (name, likelihood, _) in _observations)
                result[name].SetAt(d, likelihood.Resolve(values).Sample(rng));
        }
        return result;
    }

    /// <summary>
    /// Runs Metropolis-Hastings and returns the posterior samples.
    /// </summary>
    public PosteriorTrace SampleMCMC(int iterations = 10_000, int chains = 4, int warmup = -1,
        int thin = 1, int seed = 42)
        => Inference.MetropolisHastings(this, iterations, chains, warmup < 0 ? iterations / 2 : warmup, thin, seed);

    /// <summary>Runs component-wise Metropolis within Gibbs.</summary>
    public PosteriorTrace SampleGibbs(int iterations = 10_000, int chains = 4, int warmup = -1, int seed = 42)
        => Inference.MetropolisWithinGibbs(this, iterations, chains, warmup < 0 ? iterations / 2 : warmup, seed);

    /// <summary>Runs mean-field variational inference.</summary>
    public VariationalResult FitVariational(int iterations = 2000, double learningRate = 0.05,
        int monteCarloSamples = 8, int seed = 42)
        => Inference.MeanFieldVariational(this, iterations, learningRate, monteCarloSamples, seed);

    /// <inheritdoc />
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"BayesianModel: {ParameterCount} parameters, {ObservationCount} observations");
        foreach (var (name, prior) in _priors) sb.AppendLine($"  {name} ~ {prior.Name}");
        foreach (var (name, _, data) in _observations) sb.AppendLine($"  {name}: {data.Length} observations");
        return sb.ToString();
    }
}
