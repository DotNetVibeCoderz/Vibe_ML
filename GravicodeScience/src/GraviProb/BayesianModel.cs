using System.Text;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;

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
    private readonly Func<IReadOnlyDictionary<string, Tensor>, NdArray, Tensor>? _resolveTensor;

    private DistributionSpec(Func<IReadOnlyDictionary<string, double>, Distribution> resolve,
        IReadOnlyList<string> dependencies,
        Func<IReadOnlyDictionary<string, Tensor>, NdArray, Tensor>? resolveTensor = null)
    {
        _resolve = resolve;
        Dependencies = dependencies;
        _resolveTensor = resolveTensor;
    }

    /// <summary>Names of the variables this specification reads.</summary>
    public IReadOnlyList<string> Dependencies { get; }

    /// <summary>Whether this likelihood can be differentiated with respect to its variables.</summary>
    public bool IsDifferentiable => _resolveTensor is not null;

    /// <summary>Builds the concrete distribution for a set of parameter values.</summary>
    public Distribution Resolve(IReadOnlyDictionary<string, double> values) => _resolve(values);

    /// <summary>
    /// Builds the total log density of a whole dataset as one tape expression over the model's
    /// variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The dataset goes in as a single array rather than one observation at a time, and that is a
    /// performance decision, not a stylistic one. Per-observation terms would put a handful of
    /// tape nodes on the graph for every row — thousands of allocations for every gradient, and a
    /// gradient is evaluated at every leapfrog step of every iteration of every chain. Broadcasting
    /// the scalar parameters against the data vector instead keeps the graph a fixed size whatever
    /// the dataset, and each node's forward pass gets the vectorised <c>UFunc</c> kernels.
    /// </para>
    /// <para>
    /// The observations are constants, so any <c>lgamma</c> or binomial coefficient they appear in
    /// is precomputed; only the latent parameters carry a gradient.
    /// </para>
    /// </remarks>
    public Tensor LogDensity(IReadOnlyDictionary<string, Tensor> values, NdArray data)
        => _resolveTensor is null
            ? throw new NotSupportedException(
                "This likelihood has no differentiable form, so it cannot be used with gradient-based inference.")
            : _resolveTensor(values, data);

    /// <summary>Wraps a fixed distribution.</summary>
    public static DistributionSpec Constant(Distribution distribution)
        => new(_ => distribution, [],
            distribution.IsDifferentiable
                ? (_, data) => distribution.LogDensity(Tensor.Constant(data)).Sum()
                : null);

    /// <summary>Builds a specification from an arbitrary function of the model's variables.</summary>
    /// <remarks>
    /// A specification built this way has no differentiable form — the resolver is an opaque
    /// function of doubles — so a model using it falls back to the gradient-free samplers. Use the
    /// named factories below, or <see cref="FromTensor"/>, to keep gradients available.
    /// </remarks>
    public static DistributionSpec From(Func<IReadOnlyDictionary<string, double>, Distribution> resolve,
        params string[] dependencies) => new(resolve, dependencies);

    /// <summary>
    /// Builds a specification that supplies both a concrete distribution and a differentiable
    /// log density.
    /// </summary>
    /// <param name="resolve">Builds the concrete distribution for a set of parameter values.</param>
    /// <param name="logDensity">
    /// Builds the <em>total</em> log density of the whole dataset — the sum over rows — as one
    /// tape expression.
    /// </param>
    /// <param name="dependencies">Names of the variables read.</param>
    public static DistributionSpec FromTensor(
        Func<IReadOnlyDictionary<string, double>, Distribution> resolve,
        Func<IReadOnlyDictionary<string, Tensor>, NdArray, Tensor> logDensity,
        params string[] dependencies) => new(resolve, dependencies, logDensity);

    /// <summary>A fixed distribution used as a specification.</summary>
    public static implicit operator DistributionSpec(Distribution distribution) => Constant(distribution);

    /// <summary>A binomial likelihood whose success probability is a named model variable.</summary>
    public static DistributionSpec Binomial(int trials, string probabilityVariable)
        => FromTensor(
            values => new Binomial(trials, values[probabilityVariable]),
            (tensors, data) =>
            {
                // Sufficient statistics: the likelihood depends on the data only through the
                // total number of successes and failures, so the whole dataset collapses to two
                // numbers before the tape sees it.
                double successes = 0, coefficient = 0;
                for (var i = 0; i < data.Size; i++)
                {
                    var k = (int)data.At(i);
                    successes += k;
                    coefficient += MathUtil.LogBinomialCoefficient(trials, k);
                }

                var p = tensors[probabilityVariable];
                return Tensor.Constant(coefficient)
                       + Tensor.Constant(successes) * p.Log()
                       + Tensor.Constant(trials * (double)data.Size - successes) * (Tensor.Constant(1.0) - p).Log();
            },
            probabilityVariable);

    /// <summary>A Bernoulli likelihood whose success probability is a named model variable.</summary>
    public static DistributionSpec Bernoulli(string probabilityVariable)
        => FromTensor(
            values => new Bernoulli(values[probabilityVariable]),
            (tensors, data) =>
            {
                var successes = Statistics.Sum(data);
                var p = tensors[probabilityVariable];
                return Tensor.Constant(successes) * p.Log()
                       + Tensor.Constant(data.Size - successes) * (Tensor.Constant(1.0) - p).Log();
            },
            probabilityVariable);

    /// <summary>A normal likelihood whose mean and scale are named model variables.</summary>
    public static DistributionSpec Normal(string meanVariable, string scaleVariable)
        => FromTensor(
            values => new Normal(values[meanVariable], values[scaleVariable]),
            (tensors, data) =>
            {
                var sigma = tensors[scaleVariable];
                var z = (Tensor.Constant(data) - tensors[meanVariable]) / sigma;
                return Tensor.Constant(-0.5 * data.Size * Math.Log(2 * Math.PI))
                       - Tensor.Constant((double)data.Size) * sigma.Log()
                       - Tensor.Constant(0.5) * (z * z).Sum();
            },
            meanVariable, scaleVariable);

    /// <summary>A normal likelihood with a fitted mean and a fixed scale.</summary>
    public static DistributionSpec Normal(string meanVariable, double scale)
        => FromTensor(
            values => new Normal(values[meanVariable], scale),
            (tensors, data) =>
            {
                var z = (Tensor.Constant(data) - tensors[meanVariable]) / Tensor.Constant(scale);
                return Tensor.Constant(data.Size * (-Math.Log(scale) - 0.5 * Math.Log(2 * Math.PI)))
                       - Tensor.Constant(0.5) * (z * z).Sum();
            },
            meanVariable);

    /// <summary>A Poisson likelihood whose rate is a named model variable.</summary>
    public static DistributionSpec Poisson(string rateVariable)
        => FromTensor(
            values => new Poisson(values[rateVariable]),
            (tensors, data) =>
            {
                var total = Statistics.Sum(data);
                var logFactorials = 0.0;
                for (var i = 0; i < data.Size; i++) logFactorials += MathUtil.LogFactorial((int)data.At(i));

                var rate = tensors[rateVariable];
                return Tensor.Constant(total) * rate.Log()
                       - Tensor.Constant((double)data.Size) * rate
                       - Tensor.Constant(logFactorials);
            },
            rateVariable);
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

    /// <summary>
    /// Whether every prior and likelihood in this model has a differentiable log density, and so
    /// whether the gradient-based samplers can be used.
    /// </summary>
    public bool IsDifferentiable =>
        _priors.All(p => p.Prior.IsDifferentiable) && _observations.All(o => o.Likelihood.IsDifferentiable);

    /// <summary>
    /// The log posterior and its gradient with respect to every latent variable, in
    /// <see cref="ParameterNames"/> order.
    /// </summary>
    /// <remarks>
    /// <para>
    /// One forward pass builds the whole joint density as a tape expression; one backward pass
    /// then yields every partial derivative at once. That is the property Hamiltonian Monte Carlo
    /// needs: central differences would cost two extra log-posterior evaluations per parameter per
    /// leapfrog step, which for a model with a real dataset behind it is the entire budget.
    /// </para>
    /// <para>
    /// Outside the support the density is <c>-inf</c> and the gradient is not defined; the
    /// gradient comes back as zeros and callers are expected to reject on the density.
    /// </para>
    /// </remarks>
    public (double LogDensity, double[] Gradient) LogPosteriorGradient(IReadOnlyDictionary<string, double> values)
    {
        if (!IsDifferentiable)
            throw new NotSupportedException(
                "This model contains a prior or likelihood with no differentiable log density. "
                + "Use SampleMCMC or SampleGibbs instead.");

        var names = ParameterNames;
        var zeros = new double[names.Count];

        // A point outside any prior's support has no usable gradient, and building the tape there
        // would produce NaNs that propagate silently. Reject on the density instead.
        var plainDensity = LogPosterior(values);
        if (!double.IsFinite(plainDensity)) return (plainDensity, zeros);

        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var name in names) tensors[name] = Tensor.Parameter(values[name]);

        var total = LogPosteriorTensor(tensors);
        total.Backward();

        var gradient = new double[names.Count];
        for (var i = 0; i < names.Count; i++)
            gradient[i] = tensors[names[i]].Gradient?.At(0) ?? 0.0;

        return (total.Item, gradient);
    }

    /// <summary>
    /// The joint log density as a tape expression over tensor-valued parameters.
    /// </summary>
    /// <remarks>
    /// Kept separate from <see cref="LogPosteriorGradient"/> so a caller can put its own
    /// expression in front of the parameters — which is how the gradient-based samplers work in
    /// unconstrained space: they feed in <c>x(z)</c> rather than <c>x</c>, and the tape carries the
    /// chain rule through the transform instead of anyone deriving it by hand.
    /// </remarks>
    public Tensor LogPosteriorTensor(IReadOnlyDictionary<string, Tensor> values)
    {
        Tensor total = Tensor.Constant(0.0);

        foreach (var (name, prior) in _priors) total += prior.LogDensity(values[name]);

        foreach (var (_, likelihood, data) in _observations)
            total += likelihood.LogDensity(values, NdArray.FromValues(data));

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

    /// <summary>
    /// Runs Hamiltonian Monte Carlo, which needs <see cref="IsDifferentiable"/>.
    /// </summary>
    public PosteriorTrace SampleHMC(int iterations = 2000, int chains = 4, int warmup = -1,
        int leapfrogSteps = 20, int seed = 42)
        => Inference.HamiltonianMonteCarlo(this, iterations, chains,
            warmup < 0 ? iterations / 2 : warmup, leapfrogSteps, seed: seed);

    /// <summary>
    /// Runs the No-U-Turn Sampler, which needs <see cref="IsDifferentiable"/>.
    /// </summary>
    /// <remarks>This is the default worth reaching for when the model has gradients.</remarks>
    public PosteriorTrace SampleNUTS(int iterations = 2000, int chains = 4, int warmup = -1,
        int maxTreeDepth = 10, int seed = 42)
        => Inference.NoUTurnSampler(this, iterations, chains,
            warmup < 0 ? iterations / 2 : warmup, maxTreeDepth, seed: seed);

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
