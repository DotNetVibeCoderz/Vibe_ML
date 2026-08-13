using System.Text;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviProb;

/// <summary>Posterior draws for every parameter, organised by chain.</summary>
/// <remarks>
/// Chains are kept separate rather than concatenated because the convergence diagnostics need
/// them that way: <see cref="RHat"/> compares between-chain and within-chain variance, which is
/// meaningless once the draws are pooled.
/// </remarks>
public sealed class PosteriorTrace
{
    private readonly Dictionary<string, double[][]> _draws;

    internal PosteriorTrace(IReadOnlyList<string> parameters, double[][][] chains, double acceptanceRate,
        int warmup, int iterations)
    {
        ParameterNames = parameters;
        ChainCount = chains.Length;
        DrawsPerChain = chains.Length > 0 ? chains[0].Length : 0;
        AcceptanceRate = acceptanceRate;
        Warmup = warmup;
        Iterations = iterations;

        _draws = new Dictionary<string, double[][]>(StringComparer.Ordinal);
        for (var p = 0; p < parameters.Count; p++)
        {
            var perChain = new double[ChainCount][];
            for (var c = 0; c < ChainCount; c++)
            {
                perChain[c] = new double[DrawsPerChain];
                for (var i = 0; i < DrawsPerChain; i++) perChain[c][i] = chains[c][i][p];
            }
            _draws[parameters[p]] = perChain;
        }
    }

    /// <summary>Names of the sampled parameters.</summary>
    public IReadOnlyList<string> ParameterNames { get; }

    /// <summary>Number of independent chains.</summary>
    public int ChainCount { get; }

    /// <summary>Retained draws per chain, after warmup and thinning.</summary>
    public int DrawsPerChain { get; }

    /// <summary>Total retained draws.</summary>
    public int TotalDraws => ChainCount * DrawsPerChain;

    /// <summary>Fraction of proposals accepted.</summary>
    public double AcceptanceRate { get; }

    /// <summary>Warmup iterations discarded per chain.</summary>
    public int Warmup { get; }

    /// <summary>Iterations requested per chain.</summary>
    public int Iterations { get; }

    /// <summary>Every retained draw of one parameter, pooled across chains.</summary>
    public NdArray this[string parameter]
    {
        get
        {
            var chains = _draws[parameter];
            var result = NdArray.Zeros(TotalDraws);
            var k = 0;
            foreach (var chain in chains)
                foreach (var value in chain) result.SetAt(k++, value);
            return result;
        }
    }

    /// <summary>The draws of one parameter from one chain.</summary>
    public NdArray Chain(string parameter, int chain) => new(_draws[parameter][chain], DrawsPerChain);

    /// <summary>Posterior mean.</summary>
    public double Mean(string parameter) => Statistics.Mean(this[parameter]);

    /// <summary>Posterior standard deviation.</summary>
    public double StandardDeviation(string parameter) => Statistics.Std(this[parameter], ddof: 1);

    /// <summary>Posterior median.</summary>
    public double Median(string parameter) => Statistics.Median(this[parameter]);

    /// <summary>An equal-tailed credible interval.</summary>
    public (double Lower, double Upper) CredibleInterval(string parameter, double mass = 0.95)
    {
        var draws = this[parameter];
        var tail = (1 - mass) / 2;
        return (Statistics.Quantile(draws, tail), Statistics.Quantile(draws, 1 - tail));
    }

    /// <summary>
    /// The highest-density interval: the shortest interval containing <paramref name="mass"/> of
    /// the posterior.
    /// </summary>
    /// <remarks>
    /// Preferred over the equal-tailed interval for skewed posteriors, where the equal-tailed
    /// version can exclude the mode - the single most probable value - while the HDI cannot.
    /// </remarks>
    public (double Lower, double Upper) HighestDensityInterval(string parameter, double mass = 0.95)
    {
        var sorted = this[parameter].ToArray();
        Array.Sort(sorted);
        var n = sorted.Length;
        var window = Math.Max(1, (int)Math.Floor(mass * n));

        var bestWidth = double.MaxValue;
        var bestStart = 0;
        for (var i = 0; i + window - 1 < n; i++)
        {
            var width = sorted[i + window - 1] - sorted[i];
            if (width < bestWidth) { bestWidth = width; bestStart = i; }
        }
        return (sorted[bestStart], sorted[bestStart + window - 1]);
    }

    /// <summary>
    /// The Gelman-Rubin statistic. Values near 1 indicate the chains have mixed; above about
    /// 1.01 the run has not converged and its summaries should not be trusted.
    /// </summary>
    public double RHat(string parameter)
    {
        var chains = _draws[parameter];
        if (ChainCount < 2 || DrawsPerChain < 2) return double.NaN;

        var n = DrawsPerChain;
        var means = chains.Select(c => c.Average()).ToArray();
        var grandMean = means.Average();

        // B is the between-chain variance, W the average within-chain variance.
        var b = n * means.Sum(m => (m - grandMean) * (m - grandMean)) / (ChainCount - 1);
        var w = chains.Select(c =>
        {
            var mean = c.Average();
            return c.Sum(v => (v - mean) * (v - mean)) / (n - 1);
        }).Average();

        if (w <= 0) return double.NaN;
        var varianceEstimate = (n - 1.0) / n * w + b / n;
        return Math.Sqrt(varianceEstimate / w);
    }

    /// <summary>
    /// Effective sample size: how many independent draws the correlated chain is worth.
    /// </summary>
    public double EffectiveSampleSize(string parameter)
    {
        var draws = this[parameter].ToArray();
        var n = draws.Length;
        if (n < 4) return n;

        var mean = draws.Average();
        var variance = draws.Sum(v => (v - mean) * (v - mean)) / n;
        if (variance <= 0) return n;

        // Sum autocorrelations until they turn negative, the standard initial-positive-sequence rule.
        var sum = 0.0;
        for (var lag = 1; lag < Math.Min(n / 2, 1000); lag++)
        {
            var acf = 0.0;
            for (var i = 0; i < n - lag; i++) acf += (draws[i] - mean) * (draws[i + lag] - mean);
            acf /= n * variance;
            if (acf <= 0) break;
            sum += acf;
        }
        return n / (1 + 2 * sum);
    }

    /// <summary>A summary table with means, intervals and diagnostics.</summary>
    public string Summary(double credibleMass = 0.95)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"  draws={TotalDraws} ({ChainCount} chains x {DrawsPerChain}), warmup={Warmup}, acceptance={AcceptanceRate:P1}");
        sb.AppendLine($"  {"parameter",-14}{"mean",10}{"sd",10}{"hdi_low",12}{"hdi_high",12}{"ess",10}{"r_hat",8}");
        sb.AppendLine("  " + new string('-', 76));

        foreach (var parameter in ParameterNames)
        {
            var (low, high) = HighestDensityInterval(parameter, credibleMass);
            sb.AppendLine(
                $"  {parameter,-14}{Mean(parameter),10:F4}{StandardDeviation(parameter),10:F4}" +
                $"{low,12:F4}{high,12:F4}{EffectiveSampleSize(parameter),10:F0}{RHat(parameter),8:F3}");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Draws replicated data from the fitted posterior - the posterior predictive check, which
    /// answers "could this model have produced the data I actually saw?".
    /// </summary>
    public NdArray PosteriorPredictive(Func<IReadOnlyDictionary<string, double>, GraviRandom, double> generate,
        int draws = 1000, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var result = NdArray.Zeros(draws);
        for (var d = 0; d < draws; d++)
        {
            var index = rng.Next(DrawsPerChain);
            var chain = rng.Next(ChainCount);
            var values = ParameterNames.ToDictionary(p => p, p => _draws[p][chain][index], StringComparer.Ordinal);
            result.SetAt(d, generate(values, rng));
        }
        return result;
    }
}

/// <summary>The fitted mean-field approximation.</summary>
/// <param name="Means">Posterior mean of each parameter.</param>
/// <param name="StandardDeviations">Posterior standard deviation of each parameter.</param>
/// <param name="EvidenceLowerBound">ELBO after each iteration.</param>
public sealed record VariationalResult(
    IReadOnlyDictionary<string, double> Means,
    IReadOnlyDictionary<string, double> StandardDeviations,
    IReadOnlyList<double> EvidenceLowerBound)
{
    /// <summary>A summary table of the fitted approximation.</summary>
    public string Summary()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"  ELBO: {EvidenceLowerBound.LastOrDefault():F4} after {EvidenceLowerBound.Count} iterations");
        sb.AppendLine($"  {"parameter",-14}{"mean",12}{"sd",12}");
        sb.AppendLine("  " + new string('-', 38));
        foreach (var (name, mean) in Means)
            sb.AppendLine($"  {name,-14}{mean,12:F4}{StandardDeviations[name],12:F4}");
        return sb.ToString();
    }
}

/// <summary>Posterior inference algorithms.</summary>
public static class Inference
{
    /// <summary>
    /// Random-walk Metropolis-Hastings with per-parameter adaptive step sizes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each chain proposes a Gaussian step, then accepts it with probability
    /// <c>min(1, exp(logPosterior(proposal) - logPosterior(current)))</c>. Because only the
    /// difference of log densities appears, the unknown normalising constant of the posterior
    /// cancels - which is what makes the method usable at all.
    /// </para>
    /// <para>
    /// During warmup the proposal scale is tuned toward a 0.234 acceptance rate, the asymptotically
    /// optimal value for random-walk Metropolis. Too large a step is rejected almost always and the
    /// chain stalls; too small a step is accepted almost always but explores nothing. Adaptation
    /// stops when warmup ends, because a proposal that keeps changing breaks the Markov property
    /// and biases the result.
    /// </para>
    /// </remarks>
    public static PosteriorTrace MetropolisHastings(BayesianModel model, int iterations = 10_000,
        int chains = 4, int warmup = 5_000, int thin = 1, int seed = 42)
    {
        if (model.ParameterCount == 0) throw new ArgumentException("The model has no parameters to sample.");
        if (thin < 1) throw new ArgumentOutOfRangeException(nameof(thin));

        var parameters = model.ParameterNames;
        var retained = new double[chains][][];
        var acceptedTotal = 0L;
        var proposedTotal = 0L;
        var padlock = new object();

        // Probe for a feasible starting point before going parallel, so a misspecified model
        // surfaces its own exception instead of an AggregateException from the thread pool.
        _ = InitialPoint(model, new GraviRandom(seed));

        // Chains are independent, which is exactly what makes multi-chain MCMC embarrassingly
        // parallel - and what makes R-hat a meaningful convergence check.
        Parallel.For(0, chains, chain =>
        {
            var rng = new GraviRandom(seed + chain * 7919);
            var current = InitialPoint(model, rng);
            var currentDensity = model.LogPosterior(current);

            var scale = parameters.ToDictionary(p => p, _ => 1.0, StringComparer.Ordinal);
            var accepted = 0;
            var windowAccepted = 0;
            var draws = new List<double[]>();

            for (var iteration = 0; iteration < iterations; iteration++)
            {
                var proposal = new Dictionary<string, double>(current, StringComparer.Ordinal);
                foreach (var name in parameters) proposal[name] = current[name] + rng.Normal(0, scale[name]);

                var proposalDensity = model.LogPosterior(proposal);
                var logRatio = proposalDensity - currentDensity;

                if (logRatio >= 0 || Math.Log(rng.NextDouble()) < logRatio)
                {
                    current = proposal;
                    currentDensity = proposalDensity;
                    accepted++;
                    windowAccepted++;
                }

                // Adapt only during warmup, in windows of 50.
                if (iteration < warmup && (iteration + 1) % 50 == 0)
                {
                    var rate = windowAccepted / 50.0;
                    var factor = Math.Exp((rate - 0.234) * 1.5);
                    foreach (var name in parameters)
                        scale[name] = Math.Clamp(scale[name] * factor, 1e-6, 1e4);
                    windowAccepted = 0;
                }

                if (iteration >= warmup && (iteration - warmup) % thin == 0)
                    draws.Add(parameters.Select(p => current[p]).ToArray());
            }

            retained[chain] = draws.ToArray();
            lock (padlock)
            {
                acceptedTotal += accepted;
                proposedTotal += iterations;
            }
        });

        return new PosteriorTrace(parameters, retained,
            proposedTotal == 0 ? 0 : (double)acceptedTotal / proposedTotal, warmup, iterations);
    }

    /// <summary>
    /// Metropolis within Gibbs: one parameter is updated at a time, holding the rest fixed.
    /// </summary>
    /// <remarks>
    /// Updating coordinates individually gives a much higher acceptance rate than moving all of
    /// them at once, because each proposal only has to be good in one dimension. The trade-off is
    /// slow mixing when parameters are strongly correlated, since single-coordinate moves cannot
    /// travel along a diagonal ridge.
    /// </remarks>
    public static PosteriorTrace MetropolisWithinGibbs(BayesianModel model, int iterations = 10_000,
        int chains = 4, int warmup = 5_000, int seed = 42)
    {
        var parameters = model.ParameterNames;
        if (parameters.Count == 0) throw new ArgumentException("The model has no parameters to sample.");

        var retained = new double[chains][][];
        var acceptedTotal = 0L;
        var proposedTotal = 0L;
        var padlock = new object();

        _ = InitialPoint(model, new GraviRandom(seed));

        Parallel.For(0, chains, chain =>
        {
            var rng = new GraviRandom(seed + chain * 104_729);
            var current = InitialPoint(model, rng);
            var currentDensity = model.LogPosterior(current);

            var scale = parameters.ToDictionary(p => p, _ => 1.0, StringComparer.Ordinal);
            var accepted = new Dictionary<string, int>(parameters.ToDictionary(p => p, _ => 0, StringComparer.Ordinal));
            var totalAccepted = 0;
            var draws = new List<double[]>();

            for (var iteration = 0; iteration < iterations; iteration++)
            {
                foreach (var name in parameters)
                {
                    var proposal = new Dictionary<string, double>(current, StringComparer.Ordinal)
                    {
                        [name] = current[name] + rng.Normal(0, scale[name]),
                    };

                    var proposalDensity = model.LogPosterior(proposal);
                    var logRatio = proposalDensity - currentDensity;
                    if (logRatio >= 0 || Math.Log(rng.NextDouble()) < logRatio)
                    {
                        current = proposal;
                        currentDensity = proposalDensity;
                        accepted[name]++;
                        totalAccepted++;
                    }
                }

                if (iteration < warmup && (iteration + 1) % 50 == 0)
                {
                    foreach (var name in parameters)
                    {
                        // A single-coordinate move targets a higher acceptance rate than a joint one.
                        var rate = accepted[name] / 50.0;
                        scale[name] = Math.Clamp(scale[name] * Math.Exp((rate - 0.44) * 1.5), 1e-6, 1e4);
                        accepted[name] = 0;
                    }
                }

                if (iteration >= warmup) draws.Add(parameters.Select(p => current[p]).ToArray());
            }

            retained[chain] = draws.ToArray();
            lock (padlock)
            {
                acceptedTotal += totalAccepted;
                proposedTotal += (long)iterations * parameters.Count;
            }
        });

        return new PosteriorTrace(parameters, retained,
            proposedTotal == 0 ? 0 : (double)acceptedTotal / proposedTotal, warmup, iterations);
    }

    /// <summary>
    /// Mean-field variational inference: fits an independent Gaussian per parameter by maximising
    /// the evidence lower bound.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Variational inference turns integration into optimisation, so it converges in seconds where
    /// MCMC takes minutes. The price is the mean-field assumption: the approximation treats
    /// parameters as independent, so it systematically underestimates posterior variance and
    /// cannot represent correlations. Use it to explore quickly, then confirm with MCMC.
    /// </para>
    /// <para>
    /// Gradients are estimated by central finite differences rather than by automatic
    /// differentiation, which keeps the implementation dependency-free and is accurate enough for
    /// the handful of parameters this method is aimed at.
    /// </para>
    /// </remarks>
    public static VariationalResult MeanFieldVariational(BayesianModel model, int iterations = 2000,
        double learningRate = 0.05, int monteCarloSamples = 8, int seed = 42)
    {
        var parameters = model.ParameterNames;
        if (parameters.Count == 0) throw new ArgumentException("The model has no parameters to fit.");
        var rng = new GraviRandom(seed);

        // Each parameter is mapped onto the whole real line so the Gaussian family is valid: a
        // Beta parameter on (0,1) becomes a logit, a scale on (0,inf) becomes a log.
        var transforms = model.Priors.ToDictionary(
            kv => kv.Key, kv => SupportTransform.For(kv.Value.SupportBounds), StringComparer.Ordinal);

        var mu = parameters.ToDictionary(p => p, _ => 0.0, StringComparer.Ordinal);
        var logSigma = parameters.ToDictionary(p => p, _ => -1.0, StringComparer.Ordinal);

        var start = InitialPoint(model, rng);
        foreach (var name in parameters) mu[name] = transforms[name].ToUnconstrained(start[name]);

        var elboHistory = new List<double>();
        const double step = 1e-5;

        // The objective in unconstrained space is logPosterior(x(z)) plus the log Jacobian of the
        // transform; omitting the Jacobian would silently target the wrong distribution.
        double Objective(IReadOnlyDictionary<string, double> z)
        {
            var constrained = new Dictionary<string, double>(StringComparer.Ordinal);
            var logJacobian = 0.0;
            foreach (var name in parameters)
            {
                var transform = transforms[name];
                constrained[name] = transform.ToConstrained(z[name]);
                logJacobian += transform.LogJacobian(z[name]);
            }
            var density = model.LogPosterior(constrained);
            return double.IsNegativeInfinity(density) || double.IsNaN(density)
                ? double.NegativeInfinity
                : density + logJacobian;
        }

        for (var iteration = 0; iteration < iterations; iteration++)
        {
            var gradientMu = parameters.ToDictionary(p => p, _ => 0.0, StringComparer.Ordinal);
            var gradientLogSigma = parameters.ToDictionary(p => p, _ => 0.0, StringComparer.Ordinal);
            var objectiveTotal = 0.0;
            var usable = 0;

            for (var s = 0; s < monteCarloSamples; s++)
            {
                // Reparameterisation: draw epsilon once, then z = mu + sigma * epsilon, so the
                // gradient flows through mu and sigma rather than through the sampling step.
                var epsilon = parameters.ToDictionary(p => p, _ => rng.Normal(), StringComparer.Ordinal);
                var z = parameters.ToDictionary(
                    p => p, p => mu[p] + Math.Exp(logSigma[p]) * epsilon[p], StringComparer.Ordinal);

                var value = Objective(z);
                if (double.IsNegativeInfinity(value) || double.IsNaN(value)) continue;
                objectiveTotal += value;
                usable++;

                foreach (var name in parameters)
                {
                    var forward = new Dictionary<string, double>(z, StringComparer.Ordinal);
                    var backward = new Dictionary<string, double>(z, StringComparer.Ordinal);
                    forward[name] += step;
                    backward[name] -= step;

                    var forwardValue = Objective(forward);
                    var backwardValue = Objective(backward);
                    if (!double.IsFinite(forwardValue) || !double.IsFinite(backwardValue)) continue;

                    var derivative = (forwardValue - backwardValue) / (2 * step);
                    gradientMu[name] += derivative;
                    gradientLogSigma[name] += derivative * Math.Exp(logSigma[name]) * epsilon[name];
                }
            }

            if (usable == 0) continue;

            foreach (var name in parameters)
            {
                // The entropy term of the ELBO contributes exactly 1 to each d/dLogSigma.
                var stepMu = learningRate * gradientMu[name] / usable;
                var stepLogSigma = learningRate * (gradientLogSigma[name] / usable + 1.0);

                // Clipping keeps a large finite-difference gradient from throwing the fit away.
                mu[name] += Math.Clamp(stepMu, -1.0, 1.0);
                logSigma[name] = Math.Clamp(logSigma[name] + Math.Clamp(stepLogSigma, -0.5, 0.5), -12, 6);
            }

            var entropy = parameters.Sum(p => logSigma[p] + 0.5 * Math.Log(2 * Math.PI * Math.E));
            elboHistory.Add(objectiveTotal / usable + entropy);
        }

        // Report the posterior on the original scale by pushing draws back through the transform.
        var reportMeans = new Dictionary<string, double>(StringComparer.Ordinal);
        var reportDeviations = new Dictionary<string, double>(StringComparer.Ordinal);
        const int reportDraws = 4000;

        foreach (var name in parameters)
        {
            var transform = transforms[name];
            var sigma = Math.Exp(logSigma[name]);
            var draws = new double[reportDraws];
            for (var i = 0; i < reportDraws; i++)
                draws[i] = transform.ToConstrained(mu[name] + sigma * rng.Normal());

            var array = new NdArray(draws, reportDraws);
            reportMeans[name] = Statistics.Mean(array);
            reportDeviations[name] = Statistics.Std(array, ddof: 1);
        }

        return new VariationalResult(reportMeans, reportDeviations, elboHistory);
    }

    /// <summary>
    /// Maps a constrained parameter onto the real line and back, with the log Jacobian the
    /// change of variables requires.
    /// </summary>
    private sealed class SupportTransform
    {
        private readonly Func<double, double> _toConstrained;
        private readonly Func<double, double> _toUnconstrained;
        private readonly Func<double, double> _logJacobian;

        private SupportTransform(Func<double, double> toConstrained, Func<double, double> toUnconstrained,
            Func<double, double> logJacobian)
        {
            _toConstrained = toConstrained;
            _toUnconstrained = toUnconstrained;
            _logJacobian = logJacobian;
        }

        public double ToConstrained(double z) => _toConstrained(z);

        public double ToUnconstrained(double x) => _toUnconstrained(x);

        public double LogJacobian(double z) => _logJacobian(z);

        public static SupportTransform For((double Low, double High) bounds)
        {
            var (low, high) = bounds;
            var lowFinite = double.IsFinite(low);
            var highFinite = double.IsFinite(high);

            // Already on the real line: nothing to do.
            if (!lowFinite && !highFinite)
                return new SupportTransform(z => z, x => x, _ => 0.0);

            // Half-bounded below: exponential map, log Jacobian is z itself.
            if (lowFinite && !highFinite)
                return new SupportTransform(
                    z => low + Math.Exp(z),
                    x => Math.Log(Math.Max(x - low, 1e-12)),
                    z => z);

            // Half-bounded above: mirror of the case above.
            if (!lowFinite)
                return new SupportTransform(
                    z => high - Math.Exp(z),
                    x => Math.Log(Math.Max(high - x, 1e-12)),
                    z => z);

            // Bounded both sides: scaled logistic, whose derivative is (high-low)*s*(1-s).
            var width = high - low;
            return new SupportTransform(
                z => low + width * MathUtil.Sigmoid(z),
                x => MathUtil.Logit(Math.Clamp((x - low) / width, 1e-12, 1 - 1e-12)),
                z =>
                {
                    var s = MathUtil.Sigmoid(z);
                    return Math.Log(width) + Math.Log(Math.Max(s, 1e-300)) + Math.Log(Math.Max(1 - s, 1e-300));
                });
        }
    }

    /// <summary>
    /// A starting point drawn from the priors, retried until the posterior is finite there.
    /// </summary>
    private static Dictionary<string, double> InitialPoint(BayesianModel model, GraviRandom rng)
    {
        for (var attempt = 0; attempt < 200; attempt++)
        {
            var candidate = model.SampleFromPrior(rng);
            if (!double.IsNegativeInfinity(model.LogPosterior(candidate)) &&
                !double.IsNaN(model.LogPosterior(candidate)))
                return candidate;
        }
        throw new InvalidOperationException(
            "Could not find a starting point with finite posterior density. Check that the priors " +
            "put mass where the likelihood is defined.");
    }
}
