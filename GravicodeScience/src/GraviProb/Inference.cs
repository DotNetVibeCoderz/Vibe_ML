using System.Text;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;

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
    /// Gradients come from the tape in <c>GraviNum.Autodiff</c> once the model has enough
    /// parameters for it to pay off, and from central differences below that. It is worth being
    /// precise about why, because the obvious assumption is wrong: the tape is <em>slower</em> on
    /// a two-parameter model — measured at roughly 11× — since building and walking a graph costs
    /// more than four cheap scalar evaluations. It wins only once <c>2d</c> extra posterior
    /// evaluations outgrow one backward pass, which on this machine happens near 50 parameters.
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

        // The objective in unconstrained space is logPosterior(x(z)) plus the log Jacobian of the
        // transform; omitting the Jacobian would silently target the wrong distribution.
        var target = new UnconstrainedTarget(model, parameters, transforms);

        // Central differences cost 2d extra evaluations of the log posterior; the tape costs one
        // backward pass whatever d is. So the tape wins asymptotically but loses on a small model,
        // where building and walking the graph costs more than a handful of cheap scalar
        // evaluations. Measured on this machine the crossover sits near 50 parameters, drifting
        // lower as the dataset grows; 32 is a conservative place to switch.
        const int autodiffWorthwhileFrom = 32;
        var useAutodiff = model.IsDifferentiable && parameters.Count >= autodiffWorthwhileFrom;

        double Objective(IReadOnlyDictionary<string, double> z)
            => target.LogDensity([.. parameters.Select(p => z[p])]);

        double[] Gradient(IReadOnlyDictionary<string, double> z)
        {
            var point = parameters.Select(p => z[p]).ToArray();

            if (useAutodiff) return target.LogDensityGradient(point).Gradient;

            // Central differences: accurate to O(h²), and two evaluations per parameter.
            const double step = 1e-5;
            var result = new double[parameters.Count];

            for (var i = 0; i < parameters.Count; i++)
            {
                var forward = (double[])point.Clone();
                var backward = (double[])point.Clone();
                forward[i] += step;
                backward[i] -= step;

                var high = target.LogDensity(forward);
                var low = target.LogDensity(backward);
                result[i] = double.IsFinite(high) && double.IsFinite(low) ? (high - low) / (2 * step) : 0.0;
            }

            return result;
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

                var derivatives = Gradient(z);

                for (var i = 0; i < parameters.Count; i++)
                {
                    var derivative = derivatives[i];
                    if (!double.IsFinite(derivative)) continue;

                    var name = parameters[i];
                    gradientMu[name] += derivative;

                    // Chain rule through z = mu + exp(logSigma) * epsilon.
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

    // ================================================================ gradient-based samplers

    /// <summary>
    /// The unconstrained log density and its gradient, with the support transform folded in.
    /// </summary>
    /// <remarks>
    /// Hamiltonian methods need an unbounded space: a trajectory that leaves the support would
    /// otherwise have to be rejected, and near a boundary almost all of them do. Every parameter
    /// is therefore mapped to the whole real line, and the log Jacobian of that map is added so
    /// the target stays the same distribution.
    /// </remarks>
    private sealed class UnconstrainedTarget(
        BayesianModel model, IReadOnlyList<string> parameters,
        IReadOnlyDictionary<string, SupportTransform> transforms)
    {
        /// <summary>Number of parameters.</summary>
        public int Dimension => parameters.Count;

        /// <summary>Maps an unconstrained point back to the model's own scale.</summary>
        public Dictionary<string, double> ToConstrained(double[] z)
        {
            var result = new Dictionary<string, double>(StringComparer.Ordinal);
            for (var i = 0; i < parameters.Count; i++)
                result[parameters[i]] = transforms[parameters[i]].ToConstrained(z[i]);
            return result;
        }

        /// <summary>Maps a point on the model's scale into the unconstrained space.</summary>
        public double[] ToUnconstrained(IReadOnlyDictionary<string, double> x)
        {
            var z = new double[parameters.Count];
            for (var i = 0; i < parameters.Count; i++)
                z[i] = transforms[parameters[i]].ToUnconstrained(x[parameters[i]]);
            return z;
        }

        /// <summary>Log density at an unconstrained point, without building a tape.</summary>
        public double LogDensity(double[] z)
        {
            var constrained = new Dictionary<string, double>(StringComparer.Ordinal);
            var logJacobian = 0.0;

            for (var i = 0; i < parameters.Count; i++)
            {
                var transform = transforms[parameters[i]];
                constrained[parameters[i]] = transform.ToConstrained(z[i]);
                logJacobian += transform.LogJacobian(z[i]);
            }

            var density = model.LogPosterior(constrained);
            return double.IsFinite(density) ? density + logJacobian : double.NegativeInfinity;
        }

        /// <summary>Log density and its gradient at an unconstrained point, in one tape pass.</summary>
        public (double LogDensity, double[] Gradient) LogDensityGradient(double[] z)
        {
            var zeros = new double[parameters.Count];

            // Building the tape at a point of zero density produces NaNs that would then poison
            // the trajectory, so the cheap check comes first.
            if (!double.IsFinite(LogDensity(z))) return (double.NegativeInfinity, zeros);

            var zTensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            var xTensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);

            for (var i = 0; i < parameters.Count; i++)
            {
                var name = parameters[i];
                var zt = Tensor.Parameter(z[i]);
                zTensors[name] = zt;
                xTensors[name] = transforms[name].ToConstrainedTensor(zt);
            }

            var total = model.LogPosteriorTensor(xTensors);
            foreach (var name in parameters) total += transforms[name].LogJacobianTensor(zTensors[name]);

            total.Backward();

            var gradient = new double[parameters.Count];
            for (var i = 0; i < parameters.Count; i++)
                gradient[i] = zTensors[parameters[i]].Gradient?.At(0) ?? 0.0;

            var value = total.Item;
            return double.IsFinite(value) ? (value, gradient) : (double.NegativeInfinity, zeros);
        }
    }

    /// <summary>One leapfrog step of the Hamiltonian dynamics, in place on copies.</summary>
    /// <remarks>
    /// The half-step/full-step/half-step ordering is what makes leapfrog symplectic and reversible,
    /// which is what lets the Metropolis correction at the end be exact. Reordering it into plain
    /// Euler integration produces a sampler that drifts and quietly targets the wrong distribution.
    /// </remarks>
    private static (double[] Position, double[] Momentum, double LogDensity, double[] Gradient) Leapfrog(
        UnconstrainedTarget target, double[] position, double[] momentum, double[] gradient, double stepSize)
    {
        var p = (double[])momentum.Clone();
        var q = (double[])position.Clone();

        for (var i = 0; i < p.Length; i++) p[i] += 0.5 * stepSize * gradient[i];
        for (var i = 0; i < q.Length; i++) q[i] += stepSize * p[i];

        var (density, newGradient) = target.LogDensityGradient(q);
        for (var i = 0; i < p.Length; i++) p[i] += 0.5 * stepSize * newGradient[i];

        return (q, p, density, newGradient);
    }

    private static double KineticEnergy(double[] momentum)
    {
        var total = 0.0;
        foreach (var p in momentum) total += p * p;
        return 0.5 * total;
    }

    /// <summary>
    /// Hamiltonian Monte Carlo with dual-averaging adaptation of the step size.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Random-walk Metropolis explores by diffusion, so on a correlated posterior it needs
    /// <c>O(d²)</c> steps to cross the distribution once. HMC instead simulates a physical
    /// trajectory using the gradient of the log density, which travels across the typical set
    /// rather than stumbling around it, and proposals stay far apart even in high dimension.
    /// </para>
    /// <para>
    /// The gradient comes from the tape in <c>GraviNum.Autodiff</c>. That is the reason this
    /// sampler could not exist before: with central differences each leapfrog step would cost
    /// <c>2d</c> extra evaluations of the whole log posterior.
    /// </para>
    /// <para>
    /// Step size is tuned during warmup by dual averaging toward an acceptance rate of 0.8, the
    /// value Hoffman and Gelman found near-optimal for HMC; it is deliberately much higher than
    /// the 0.234 a random walk targets, because a rejected Hamiltonian trajectory wastes far more
    /// work than a rejected random-walk step.
    /// </para>
    /// </remarks>
    public static PosteriorTrace HamiltonianMonteCarlo(BayesianModel model, int iterations = 2000,
        int chains = 4, int warmup = -1, int leapfrogSteps = 20, double targetAcceptance = 0.8, int seed = 42)
    {
        if (!model.IsDifferentiable)
            throw new NotSupportedException(
                "Hamiltonian Monte Carlo needs gradients, and this model has a prior or likelihood "
                + "with no differentiable log density. Use SampleMCMC instead.");

        var parameters = model.ParameterNames;
        if (parameters.Count == 0) throw new ArgumentException("The model has no parameters to sample.");
        if (warmup < 0) warmup = iterations / 2;

        var transforms = model.Priors.ToDictionary(
            kv => kv.Key, kv => SupportTransform.For(kv.Value.SupportBounds), StringComparer.Ordinal);
        var target = new UnconstrainedTarget(model, parameters, transforms);

        // Fail on the calling thread rather than inside Parallel.For, where the real exception
        // would be buried in an AggregateException.
        _ = InitialPoint(model, new GraviRandom(seed));

        var chainDraws = new double[chains][][];
        var chainAccepted = new double[chains];

        Parallel.For(0, chains, chain =>
        {
            var rng = new GraviRandom(seed + chain * 7919);
            var z = target.ToUnconstrained(InitialPoint(model, rng));
            var (density, gradient) = target.LogDensityGradient(z);

            var stepSize = FindReasonableStepSize(target, z, rng);
            var dualAveraging = new DualAveraging(stepSize, targetAcceptance);

            var draws = new List<double[]>(iterations - warmup);
            var accepted = 0;

            for (var iteration = 0; iteration < iterations; iteration++)
            {
                var momentum = new double[z.Length];
                for (var i = 0; i < momentum.Length; i++) momentum[i] = rng.Normal();

                var startEnergy = density - KineticEnergy(momentum);

                var proposalZ = z;
                var proposalMomentum = momentum;
                var proposalDensity = density;
                var proposalGradient = gradient;
                var diverged = false;

                for (var step = 0; step < leapfrogSteps; step++)
                {
                    (proposalZ, proposalMomentum, proposalDensity, proposalGradient) =
                        Leapfrog(target, proposalZ, proposalMomentum, proposalGradient, stepSize);

                    if (double.IsFinite(proposalDensity)) continue;
                    diverged = true;
                    break;
                }

                var acceptProbability = 0.0;
                if (!diverged)
                {
                    var endEnergy = proposalDensity - KineticEnergy(proposalMomentum);
                    acceptProbability = Math.Min(1.0, Math.Exp(endEnergy - startEnergy));

                    if (rng.NextDouble() < acceptProbability)
                    {
                        z = proposalZ;
                        density = proposalDensity;
                        gradient = proposalGradient;
                        if (iteration >= warmup) accepted++;
                    }
                }

                // Adapt during warmup only; afterwards the chain must be a fixed transition
                // kernel or its stationary distribution is not the posterior.
                stepSize = iteration < warmup
                    ? dualAveraging.Adapt(acceptProbability, iteration + 1)
                    : dualAveraging.Final;

                if (iteration >= warmup)
                {
                    var constrained = target.ToConstrained(z);
                    draws.Add([.. parameters.Select(p => constrained[p])]);
                }
            }

            chainDraws[chain] = [.. draws];
            chainAccepted[chain] = draws.Count > 0 ? (double)accepted / draws.Count : 0.0;
        });

        return new PosteriorTrace(parameters, chainDraws, chainAccepted.Average(), warmup, iterations);
    }

    /// <summary>
    /// The No-U-Turn Sampler: Hamiltonian Monte Carlo that chooses its own trajectory length.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Plain HMC has one awkward tuning knob left: how many leapfrog steps to take. Too few and it
    /// behaves like a random walk; too many and the trajectory curves back on itself and the work
    /// is wasted. NUTS removes the knob by doubling the trajectory in a random direction until the
    /// two ends start moving towards each other — the U-turn — and then sampling from the states
    /// it visited.
    /// </para>
    /// <para>
    /// Doubling in a random direction, and rejecting any sub-trajectory that already contains a
    /// U-turn, is what keeps the transition reversible; a simpler "keep going until you turn
    /// around" rule biases the result.
    /// </para>
    /// </remarks>
    public static PosteriorTrace NoUTurnSampler(BayesianModel model, int iterations = 2000,
        int chains = 4, int warmup = -1, int maxTreeDepth = 10, double targetAcceptance = 0.8, int seed = 42)
    {
        if (!model.IsDifferentiable)
            throw new NotSupportedException(
                "NUTS needs gradients, and this model has a prior or likelihood with no "
                + "differentiable log density. Use SampleMCMC instead.");

        var parameters = model.ParameterNames;
        if (parameters.Count == 0) throw new ArgumentException("The model has no parameters to sample.");
        if (warmup < 0) warmup = iterations / 2;

        var transforms = model.Priors.ToDictionary(
            kv => kv.Key, kv => SupportTransform.For(kv.Value.SupportBounds), StringComparer.Ordinal);
        var target = new UnconstrainedTarget(model, parameters, transforms);

        _ = InitialPoint(model, new GraviRandom(seed));

        var chainDraws = new double[chains][][];
        var chainAccepted = new double[chains];

        Parallel.For(0, chains, chain =>
        {
            var rng = new GraviRandom(seed + chain * 7919);
            var z = target.ToUnconstrained(InitialPoint(model, rng));
            var (density, gradient) = target.LogDensityGradient(z);

            var stepSize = FindReasonableStepSize(target, z, rng);
            var dualAveraging = new DualAveraging(stepSize, targetAcceptance);

            var draws = new List<double[]>(iterations - warmup);
            var acceptanceTotal = 0.0;
            var acceptanceCount = 0;

            for (var iteration = 0; iteration < iterations; iteration++)
            {
                var momentum = new double[z.Length];
                for (var i = 0; i < momentum.Length; i++) momentum[i] = rng.Normal();

                var energy = density - KineticEnergy(momentum);

                // The slice variable is what makes the trajectory a valid Markov transition:
                // states are only eligible if they sit above it, drawn uniformly under the
                // current density.
                var logSlice = energy + Math.Log(rng.NextDouble());

                var minusZ = z;
                var minusMomentum = momentum;
                var minusGradient = gradient;
                var plusZ = z;
                var plusMomentum = momentum;
                var plusGradient = gradient;

                var sampleZ = z;
                var sampleDensity = density;
                var sampleGradient = gradient;

                var validStates = 1;
                var keepGoing = true;
                var stepAcceptance = 0.0;
                var stepAcceptanceCount = 0;

                for (var depth = 0; depth < maxTreeDepth && keepGoing; depth++)
                {
                    var direction = rng.NextDouble() < 0.5 ? -1 : 1;
                    BuildTreeResult subtree;

                    if (direction < 0)
                    {
                        subtree = BuildTree(target, minusZ, minusMomentum, minusGradient,
                            logSlice, direction, depth, stepSize, energy, rng);
                        minusZ = subtree.MinusPosition;
                        minusMomentum = subtree.MinusMomentum;
                        minusGradient = subtree.MinusGradient;
                    }
                    else
                    {
                        subtree = BuildTree(target, plusZ, plusMomentum, plusGradient,
                            logSlice, direction, depth, stepSize, energy, rng);
                        plusZ = subtree.PlusPosition;
                        plusMomentum = subtree.PlusMomentum;
                        plusGradient = subtree.PlusGradient;
                    }

                    // Accept from the new half with probability proportional to how many valid
                    // states it holds, which keeps every eligible state equally likely overall.
                    if (subtree.KeepGoing && subtree.ValidStates > 0 &&
                        rng.NextDouble() < (double)subtree.ValidStates / Math.Max(validStates, 1))
                    {
                        sampleZ = subtree.SamplePosition;
                        sampleDensity = subtree.SampleDensity;
                        sampleGradient = subtree.SampleGradient;
                    }

                    validStates += subtree.ValidStates;
                    stepAcceptance += subtree.AcceptanceSum;
                    stepAcceptanceCount += subtree.AcceptanceCount;

                    keepGoing = subtree.KeepGoing && NoUTurn(minusZ, plusZ, minusMomentum, plusMomentum);
                }

                z = sampleZ;
                density = sampleDensity;
                gradient = sampleGradient;

                var meanAcceptance = stepAcceptanceCount > 0 ? stepAcceptance / stepAcceptanceCount : 0.0;
                stepSize = iteration < warmup
                    ? dualAveraging.Adapt(meanAcceptance, iteration + 1)
                    : dualAveraging.Final;

                if (iteration < warmup) continue;

                acceptanceTotal += meanAcceptance;
                acceptanceCount++;

                var constrained = target.ToConstrained(z);
                draws.Add([.. parameters.Select(p => constrained[p])]);
            }

            chainDraws[chain] = [.. draws];
            chainAccepted[chain] = acceptanceCount > 0 ? acceptanceTotal / acceptanceCount : 0.0;
        });

        return new PosteriorTrace(parameters, chainDraws, chainAccepted.Average(), warmup, iterations);
    }

    /// <summary>One doubling of the NUTS trajectory.</summary>
    private readonly record struct BuildTreeResult(
        double[] MinusPosition, double[] MinusMomentum, double[] MinusGradient,
        double[] PlusPosition, double[] PlusMomentum, double[] PlusGradient,
        double[] SamplePosition, double SampleDensity, double[] SampleGradient,
        int ValidStates, bool KeepGoing, double AcceptanceSum, int AcceptanceCount);

    /// <summary>
    /// Builds one side of the NUTS trajectory, doubling recursively to depth
    /// <paramref name="depth"/>.
    /// </summary>
    /// <remarks>
    /// The U-turn test is applied to every sub-trajectory, not only the whole one. Checking only
    /// at the top would let the sampler keep a state reached after the trajectory had already
    /// folded back, which breaks detailed balance.
    /// </remarks>
    private static BuildTreeResult BuildTree(UnconstrainedTarget target,
        double[] position, double[] momentum, double[] gradient,
        double logSlice, int direction, int depth, double stepSize, double initialEnergy, GraviRandom rng)
    {
        const double divergenceLimit = 1000.0;

        if (depth == 0)
        {
            var (q, p, density, g) = Leapfrog(target, position, momentum, gradient, direction * stepSize);
            var energy = density - KineticEnergy(p);

            var valid = double.IsFinite(energy) && logSlice <= energy ? 1 : 0;
            var alive = double.IsFinite(energy) && logSlice < energy + divergenceLimit;

            return new BuildTreeResult(q, p, g, q, p, g, q, density, g, valid, alive,
                Math.Min(1.0, Math.Exp(energy - initialEnergy)), 1);
        }

        var first = BuildTree(target, position, momentum, gradient,
            logSlice, direction, depth - 1, stepSize, initialEnergy, rng);

        if (!first.KeepGoing) return first;

        var second = direction < 0
            ? BuildTree(target, first.MinusPosition, first.MinusMomentum, first.MinusGradient,
                logSlice, direction, depth - 1, stepSize, initialEnergy, rng)
            : BuildTree(target, first.PlusPosition, first.PlusMomentum, first.PlusGradient,
                logSlice, direction, depth - 1, stepSize, initialEnergy, rng);

        var minusPosition = direction < 0 ? second.MinusPosition : first.MinusPosition;
        var minusMomentum = direction < 0 ? second.MinusMomentum : first.MinusMomentum;
        var minusGradient = direction < 0 ? second.MinusGradient : first.MinusGradient;
        var plusPosition = direction < 0 ? first.PlusPosition : second.PlusPosition;
        var plusMomentum = direction < 0 ? first.PlusMomentum : second.PlusMomentum;
        var plusGradient = direction < 0 ? first.PlusGradient : second.PlusGradient;

        var total = first.ValidStates + second.ValidStates;

        var samplePosition = first.SamplePosition;
        var sampleDensity = first.SampleDensity;
        var sampleGradient = first.SampleGradient;

        if (second.ValidStates > 0 && total > 0 && rng.NextDouble() < (double)second.ValidStates / total)
        {
            samplePosition = second.SamplePosition;
            sampleDensity = second.SampleDensity;
            sampleGradient = second.SampleGradient;
        }

        var keepGoing = first.KeepGoing && second.KeepGoing
            && NoUTurn(minusPosition, plusPosition, minusMomentum, plusMomentum);

        return new BuildTreeResult(
            minusPosition, minusMomentum, minusGradient,
            plusPosition, plusMomentum, plusGradient,
            samplePosition, sampleDensity, sampleGradient,
            total, keepGoing,
            first.AcceptanceSum + second.AcceptanceSum,
            first.AcceptanceCount + second.AcceptanceCount);
    }

    /// <summary>
    /// True while the two ends of the trajectory are still moving apart.
    /// </summary>
    /// <remarks>
    /// The test is whether the momentum at each end still has a positive component along the
    /// vector joining them. Once either dot product turns negative, that end has begun to come
    /// back and extending the trajectory only retraces ground already covered.
    /// </remarks>
    private static bool NoUTurn(double[] minus, double[] plus, double[] minusMomentum, double[] plusMomentum)
    {
        var forward = 0.0;
        var backward = 0.0;

        for (var i = 0; i < minus.Length; i++)
        {
            var span = plus[i] - minus[i];
            forward += span * plusMomentum[i];
            backward += span * minusMomentum[i];
        }

        return forward >= 0 && backward >= 0;
    }

    /// <summary>
    /// Doubles or halves a trial step size until one leapfrog step lands near an acceptance of a
    /// half, giving dual averaging a sane place to start.
    /// </summary>
    private static double FindReasonableStepSize(UnconstrainedTarget target, double[] position, GraviRandom rng)
    {
        var stepSize = 1.0;
        var (density, gradient) = target.LogDensityGradient(position);

        var momentum = new double[position.Length];
        for (var i = 0; i < momentum.Length; i++) momentum[i] = rng.Normal();

        var energy = density - KineticEnergy(momentum);

        var (_, newMomentum, newDensity, _) = Leapfrog(target, position, momentum, gradient, stepSize);
        var newEnergy = newDensity - KineticEnergy(newMomentum);

        // Which way to move depends on whether the first trial was too bold or too timid.
        var direction = newEnergy - energy > Math.Log(0.5) ? 1 : -1;

        for (var attempt = 0; attempt < 100; attempt++)
        {
            stepSize *= direction == 1 ? 2.0 : 0.5;

            (_, newMomentum, newDensity, _) = Leapfrog(target, position, momentum, gradient, stepSize);
            newEnergy = newDensity - KineticEnergy(newMomentum);

            var ratio = newEnergy - energy;
            if (!double.IsFinite(ratio)) { if (direction == 1) { stepSize *= 0.5; break; } continue; }

            if (direction == 1 ? ratio <= Math.Log(0.5) : ratio >= Math.Log(0.5)) break;
        }

        return Math.Clamp(stepSize, 1e-6, 10.0);
    }

    /// <summary>
    /// Nesterov dual averaging, which tunes the step size toward a target acceptance rate.
    /// </summary>
    /// <remarks>
    /// A plain multiplicative rule keeps reacting to the noise in individual acceptances. Dual
    /// averaging instead accumulates the error and reports a running average, so the value it
    /// settles on is stable enough to freeze at the end of warmup.
    /// </remarks>
    private sealed class DualAveraging(double initialStepSize, double target)
    {
        private const double Gamma = 0.05;
        private const double Kappa = 0.75;
        private const double T0 = 10.0;

        private readonly double _mu = Math.Log(10 * initialStepSize);
        private double _logAveraged = Math.Log(initialStepSize);
        private double _errorSum;

        /// <summary>The averaged step size to use once warmup ends.</summary>
        public double Final => Math.Exp(_logAveraged);

        /// <summary>Updates from one iteration's acceptance probability.</summary>
        public double Adapt(double acceptance, int iteration)
        {
            if (!double.IsFinite(acceptance)) acceptance = 0.0;

            _errorSum += (target - acceptance - _errorSum) / (iteration + T0);
            var logStep = _mu - Math.Sqrt(iteration) / Gamma * _errorSum;

            var weight = Math.Pow(iteration, -Kappa);
            _logAveraged = weight * logStep + (1 - weight) * _logAveraged;

            return Math.Clamp(Math.Exp(logStep), 1e-8, 10.0);
        }
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

        /// <summary>The forward map, on the tape.</summary>
        /// <remarks>
        /// The tensor forms exist so a gradient-based sampler can work in unconstrained space
        /// without anyone differentiating the transform by hand: it builds <c>x(z)</c> and the
        /// Jacobi term as tape expressions, and the chain rule back to <c>z</c> comes for free.
        /// </remarks>
        public required Func<Tensor, Tensor> ToConstrainedTensor { get; init; }

        /// <summary>The log Jacobian of <see cref="ToConstrainedTensor"/>, on the tape.</summary>
        public required Func<Tensor, Tensor> LogJacobianTensor { get; init; }

        public static SupportTransform For((double Low, double High) bounds)
        {
            var (low, high) = bounds;
            var lowFinite = double.IsFinite(low);
            var highFinite = double.IsFinite(high);

            // Already on the real line: nothing to do.
            if (!lowFinite && !highFinite)
                return new SupportTransform(z => z, x => x, _ => 0.0)
                {
                    ToConstrainedTensor = z => z,
                    LogJacobianTensor = _ => Tensor.Constant(0.0),
                };

            // Half-bounded below: exponential map, log Jacobian is z itself.
            if (lowFinite && !highFinite)
                return new SupportTransform(
                    z => low + Math.Exp(z),
                    x => Math.Log(Math.Max(x - low, 1e-12)),
                    z => z)
                {
                    ToConstrainedTensor = z => Tensor.Constant(low) + z.Exp(),
                    LogJacobianTensor = z => z,
                };

            // Half-bounded above: mirror of the case above.
            if (!lowFinite)
                return new SupportTransform(
                    z => high - Math.Exp(z),
                    x => Math.Log(Math.Max(high - x, 1e-12)),
                    z => z)
                {
                    ToConstrainedTensor = z => Tensor.Constant(high) - z.Exp(),
                    LogJacobianTensor = z => z,
                };

            // Bounded both sides: scaled logistic, whose derivative is (high-low)*s*(1-s).
            var width = high - low;
            return new SupportTransform(
                z => low + width * MathUtil.Sigmoid(z),
                x => MathUtil.Logit(Math.Clamp((x - low) / width, 1e-12, 1 - 1e-12)),
                z =>
                {
                    var s = MathUtil.Sigmoid(z);
                    return Math.Log(width) + Math.Log(Math.Max(s, 1e-300)) + Math.Log(Math.Max(1 - s, 1e-300));
                })
            {
                ToConstrainedTensor = z => Tensor.Constant(low) + Tensor.Constant(width) * z.Sigmoid(),
                LogJacobianTensor = z =>
                {
                    var s = z.Sigmoid();
                    return Tensor.Constant(Math.Log(width)) + s.Log() + (Tensor.Constant(1.0) - s).Log();
                },
            };
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
