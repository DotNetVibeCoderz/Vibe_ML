using System.Text;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviProb.Models;

/// <summary>
/// A discrete Bayesian network: a directed acyclic graph of variables with conditional
/// probability tables.
/// </summary>
/// <remarks>
/// The factorisation is the whole point. A joint distribution over <c>n</c> binary variables has
/// <c>2^n - 1</c> free parameters; writing it as a product of each variable's conditional given
/// its parents reduces that to the sum of the tables, which is exponentially smaller when the
/// graph is sparse. Inference here is exact enumeration, which is correct for the small networks
/// this type targets and exponential beyond them.
/// </remarks>
public sealed class BayesianNetwork
{
    private readonly List<string> _order = [];
    private readonly Dictionary<string, Node> _nodes = new(StringComparer.Ordinal);

    private sealed class Node
    {
        public required string Name { get; init; }
        public required int StateCount { get; init; }
        public required string[] Parents { get; init; }
        public required double[][] Table { get; init; }   // [parent configuration][state]
    }

    /// <summary>Names of the variables, in insertion order.</summary>
    public IReadOnlyList<string> Variables => _order;

    /// <summary>Number of states a variable can take.</summary>
    public int StateCount(string variable) => _nodes[variable].StateCount;

    /// <summary>
    /// Adds a variable with its conditional probability table.
    /// </summary>
    /// <param name="name">Variable name.</param>
    /// <param name="stateCount">Number of states.</param>
    /// <param name="parents">Parent variables, which must already exist.</param>
    /// <param name="table">
    /// One row per parent configuration in row-major order over the parents' states, each row a
    /// distribution over this variable's states.
    /// </param>
    public BayesianNetwork AddVariable(string name, int stateCount, string[] parents, double[][] table)
    {
        foreach (var parent in parents)
            if (!_nodes.ContainsKey(parent))
                throw new ArgumentException($"Parent '{parent}' of '{name}' has not been added yet.");

        var expectedRows = parents.Aggregate(1, (acc, p) => acc * _nodes[p].StateCount);
        if (table.Length != expectedRows)
            throw new ArgumentException(
                $"'{name}' has {parents.Length} parents giving {expectedRows} configurations, but {table.Length} rows were supplied.");

        foreach (var row in table)
        {
            if (row.Length != stateCount)
                throw new ArgumentException($"Each row of '{name}' must have {stateCount} entries.");
            var total = row.Sum();
            if (Math.Abs(total - 1.0) > 1e-6)
                throw new ArgumentException($"A row of '{name}' sums to {total:F6}, not 1.");
        }

        _nodes[name] = new Node { Name = name, StateCount = stateCount, Parents = parents, Table = table };
        _order.Add(name);
        return this;
    }

    /// <summary>Adds a root variable with no parents.</summary>
    public BayesianNetwork AddVariable(string name, params double[] distribution)
        => AddVariable(name, distribution.Length, [], [distribution]);

    /// <summary>The conditional probability of one state given an assignment to the parents.</summary>
    public double Conditional(string variable, int state, IReadOnlyDictionary<string, int> assignment)
    {
        var node = _nodes[variable];
        var row = 0;
        foreach (var parent in node.Parents) row = row * _nodes[parent].StateCount + assignment[parent];
        return node.Table[row][state];
    }

    /// <summary>The joint probability of a complete assignment.</summary>
    public double JointProbability(IReadOnlyDictionary<string, int> assignment)
    {
        var probability = 1.0;
        foreach (var name in _order) probability *= Conditional(name, assignment[name], assignment);
        return probability;
    }

    /// <summary>
    /// Exact marginal of <paramref name="target"/> given evidence, by enumerating every
    /// consistent assignment.
    /// </summary>
    public double[] Infer(string target, IReadOnlyDictionary<string, int>? evidence = null)
    {
        var known = evidence ?? new Dictionary<string, int>(StringComparer.Ordinal);
        var result = new double[_nodes[target].StateCount];

        foreach (var assignment in Enumerate(0, new Dictionary<string, int>(StringComparer.Ordinal), known))
            result[assignment[target]] += JointProbability(assignment);

        var total = result.Sum();
        if (total <= 0)
            throw new InvalidOperationException("The evidence has zero probability under this network.");
        for (var i = 0; i < result.Length; i++) result[i] /= total;
        return result;
    }

    private IEnumerable<Dictionary<string, int>> Enumerate(int index,
        Dictionary<string, int> partial, IReadOnlyDictionary<string, int> evidence)
    {
        if (index == _order.Count)
        {
            yield return new Dictionary<string, int>(partial, StringComparer.Ordinal);
            yield break;
        }

        var name = _order[index];
        if (evidence.TryGetValue(name, out var fixedState))
        {
            partial[name] = fixedState;
            foreach (var complete in Enumerate(index + 1, partial, evidence)) yield return complete;
            partial.Remove(name);
            yield break;
        }

        for (var state = 0; state < _nodes[name].StateCount; state++)
        {
            partial[name] = state;
            foreach (var complete in Enumerate(index + 1, partial, evidence)) yield return complete;
        }
        partial.Remove(name);
    }

    /// <summary>Draws one assignment by ancestral sampling, parents before children.</summary>
    public Dictionary<string, int> Sample(GraviRandom rng)
    {
        var assignment = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var name in _order)
        {
            var node = _nodes[name];
            var row = 0;
            foreach (var parent in node.Parents) row = row * _nodes[parent].StateCount + assignment[parent];
            assignment[name] = rng.Categorical(node.Table[row]);
        }
        return assignment;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var sb = new StringBuilder($"BayesianNetwork({_order.Count} variables)");
        sb.AppendLine();
        foreach (var name in _order)
        {
            var node = _nodes[name];
            sb.AppendLine(node.Parents.Length == 0
                ? $"  {name} ({node.StateCount} states)"
                : $"  {name} ({node.StateCount} states) | {string.Join(", ", node.Parents)}");
        }
        return sb.ToString();
    }
}

/// <summary>
/// A hidden Markov model over discrete states and discrete observations.
/// </summary>
/// <remarks>
/// The three classic problems each have an exact algorithm here: <see cref="LogLikelihood"/>
/// (evaluation) uses the forward recursion, <see cref="Viterbi"/> (decoding) uses dynamic
/// programming over the most probable path, and <see cref="Fit"/> (learning) uses Baum-Welch,
/// which is expectation-maximisation. All of them work in log space or with per-step
/// normalisation, because the raw probabilities underflow within a few dozen time steps.
/// </remarks>
public sealed class HiddenMarkovModel
{
    /// <summary>Number of hidden states.</summary>
    public int StateCount { get; }

    /// <summary>Number of distinct observation symbols.</summary>
    public int SymbolCount { get; }

    /// <summary>Initial state distribution.</summary>
    public double[] InitialProbabilities { get; private set; }

    /// <summary>State-to-state transition probabilities.</summary>
    public double[,] Transitions { get; private set; }

    /// <summary>State-to-symbol emission probabilities.</summary>
    public double[,] Emissions { get; private set; }

    /// <summary>Creates a model with explicit parameters.</summary>
    public HiddenMarkovModel(double[] initial, double[,] transitions, double[,] emissions)
    {
        StateCount = initial.Length;
        SymbolCount = emissions.GetLength(1);
        InitialProbabilities = initial;
        Transitions = transitions;
        Emissions = emissions;
    }

    /// <summary>Creates a model with random parameters, ready for <see cref="Fit"/>.</summary>
    public static HiddenMarkovModel Random(int states, int symbols, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var initial = Normalize(Enumerable.Range(0, states).Select(_ => rng.NextDouble() + 0.1).ToArray());

        var transitions = new double[states, states];
        var emissions = new double[states, symbols];
        for (var i = 0; i < states; i++)
        {
            var row = Normalize(Enumerable.Range(0, states).Select(_ => rng.NextDouble() + 0.1).ToArray());
            for (var j = 0; j < states; j++) transitions[i, j] = row[j];

            var emissionRow = Normalize(Enumerable.Range(0, symbols).Select(_ => rng.NextDouble() + 0.1).ToArray());
            for (var k = 0; k < symbols; k++) emissions[i, k] = emissionRow[k];
        }
        return new HiddenMarkovModel(initial, transitions, emissions);
    }

    private static double[] Normalize(double[] values)
    {
        var total = values.Sum();
        return values.Select(v => v / total).ToArray();
    }

    /// <summary>
    /// The forward pass with per-step scaling, returning the alphas and the scaling factors.
    /// </summary>
    private (double[,] Alpha, double[] Scale) Forward(IReadOnlyList<int> observations)
    {
        var t = observations.Count;
        var alpha = new double[t, StateCount];
        var scale = new double[t];

        for (var i = 0; i < StateCount; i++)
            alpha[0, i] = InitialProbabilities[i] * Emissions[i, observations[0]];
        scale[0] = Rescale(alpha, 0);

        for (var step = 1; step < t; step++)
        {
            for (var j = 0; j < StateCount; j++)
            {
                var accumulator = 0.0;
                for (var i = 0; i < StateCount; i++) accumulator += alpha[step - 1, i] * Transitions[i, j];
                alpha[step, j] = accumulator * Emissions[j, observations[step]];
            }
            scale[step] = Rescale(alpha, step);
        }
        return (alpha, scale);
    }

    private double Rescale(double[,] matrix, int step)
    {
        var total = 0.0;
        for (var i = 0; i < StateCount; i++) total += matrix[step, i];
        if (total <= 0) return 1.0;
        for (var i = 0; i < StateCount; i++) matrix[step, i] /= total;
        return total;
    }

    private double[,] Backward(IReadOnlyList<int> observations, double[] scale)
    {
        var t = observations.Count;
        var beta = new double[t, StateCount];
        for (var i = 0; i < StateCount; i++) beta[t - 1, i] = 1.0 / scale[t - 1];

        for (var step = t - 2; step >= 0; step--)
            for (var i = 0; i < StateCount; i++)
            {
                var accumulator = 0.0;
                for (var j = 0; j < StateCount; j++)
                    accumulator += Transitions[i, j] * Emissions[j, observations[step + 1]] * beta[step + 1, j];
                beta[step, i] = accumulator / scale[step];
            }
        return beta;
    }

    /// <summary>Log probability of an observation sequence under the model.</summary>
    public double LogLikelihood(IReadOnlyList<int> observations)
    {
        var (_, scale) = Forward(observations);
        // The scaling factors multiply back to the sequence probability, so their logs sum to it.
        return scale.Sum(s => Math.Log(Math.Max(s, 1e-300)));
    }

    /// <summary>
    /// The single most probable hidden state sequence, by the Viterbi algorithm.
    /// </summary>
    /// <remarks>
    /// This is not the same as taking the most probable state at each step independently: Viterbi
    /// maximises the joint probability of the whole path, so it can never return a sequence
    /// containing a transition the model forbids.
    /// </remarks>
    public int[] Viterbi(IReadOnlyList<int> observations)
    {
        var t = observations.Count;
        var delta = new double[t, StateCount];
        var backpointer = new int[t, StateCount];

        for (var i = 0; i < StateCount; i++)
            delta[0, i] = Math.Log(Math.Max(InitialProbabilities[i], 1e-300))
                + Math.Log(Math.Max(Emissions[i, observations[0]], 1e-300));

        for (var step = 1; step < t; step++)
            for (var j = 0; j < StateCount; j++)
            {
                var best = double.NegativeInfinity;
                var bestState = 0;
                for (var i = 0; i < StateCount; i++)
                {
                    var score = delta[step - 1, i] + Math.Log(Math.Max(Transitions[i, j], 1e-300));
                    if (score > best) { best = score; bestState = i; }
                }
                delta[step, j] = best + Math.Log(Math.Max(Emissions[j, observations[step]], 1e-300));
                backpointer[step, j] = bestState;
            }

        var path = new int[t];
        var final = 0;
        for (var i = 1; i < StateCount; i++) if (delta[t - 1, i] > delta[t - 1, final]) final = i;
        path[t - 1] = final;
        for (var step = t - 2; step >= 0; step--) path[step] = backpointer[step + 1, path[step + 1]];
        return path;
    }

    /// <summary>Posterior probability of each state at each time step.</summary>
    public double[,] StatePosteriors(IReadOnlyList<int> observations)
    {
        var (alpha, scale) = Forward(observations);
        var beta = Backward(observations, scale);
        var t = observations.Count;

        var gamma = new double[t, StateCount];
        for (var step = 0; step < t; step++)
        {
            var total = 0.0;
            for (var i = 0; i < StateCount; i++)
            {
                gamma[step, i] = alpha[step, i] * beta[step, i];
                total += gamma[step, i];
            }
            if (total <= 0) continue;
            for (var i = 0; i < StateCount; i++) gamma[step, i] /= total;
        }
        return gamma;
    }

    /// <summary>
    /// Fits the parameters to observation sequences by Baum-Welch (EM).
    /// </summary>
    /// <remarks>
    /// The likelihood is guaranteed not to decrease between iterations, but the algorithm only
    /// finds a local optimum - which is why the initial parameters matter and why a real fit runs
    /// several random restarts.
    /// </remarks>
    public double Fit(IReadOnlyList<IReadOnlyList<int>> sequences, int iterations = 50, double tolerance = 1e-6)
    {
        var previousLikelihood = double.NegativeInfinity;
        var likelihood = double.NegativeInfinity;

        for (var iteration = 0; iteration < iterations; iteration++)
        {
            var initialAccumulator = new double[StateCount];
            var transitionAccumulator = new double[StateCount, StateCount];
            var emissionAccumulator = new double[StateCount, SymbolCount];
            var stateAccumulator = new double[StateCount];
            var transitionDenominator = new double[StateCount];

            likelihood = 0.0;

            foreach (var observations in sequences)
            {
                var t = observations.Count;
                if (t == 0) continue;

                var (alpha, scale) = Forward(observations);
                var beta = Backward(observations, scale);
                likelihood += scale.Sum(s => Math.Log(Math.Max(s, 1e-300)));

                var gamma = new double[t, StateCount];
                for (var step = 0; step < t; step++)
                {
                    var total = 0.0;
                    for (var i = 0; i < StateCount; i++)
                    {
                        gamma[step, i] = alpha[step, i] * beta[step, i];
                        total += gamma[step, i];
                    }
                    if (total <= 0) continue;
                    for (var i = 0; i < StateCount; i++) gamma[step, i] /= total;
                }

                for (var i = 0; i < StateCount; i++) initialAccumulator[i] += gamma[0, i];

                for (var step = 0; step < t - 1; step++)
                {
                    var total = 0.0;
                    var xi = new double[StateCount, StateCount];
                    for (var i = 0; i < StateCount; i++)
                        for (var j = 0; j < StateCount; j++)
                        {
                            xi[i, j] = alpha[step, i] * Transitions[i, j]
                                * Emissions[j, observations[step + 1]] * beta[step + 1, j];
                            total += xi[i, j];
                        }
                    if (total <= 0) continue;

                    for (var i = 0; i < StateCount; i++)
                        for (var j = 0; j < StateCount; j++)
                            transitionAccumulator[i, j] += xi[i, j] / total;
                    for (var i = 0; i < StateCount; i++) transitionDenominator[i] += gamma[step, i];
                }

                for (var step = 0; step < t; step++)
                    for (var i = 0; i < StateCount; i++)
                    {
                        emissionAccumulator[i, observations[step]] += gamma[step, i];
                        stateAccumulator[i] += gamma[step, i];
                    }
            }

            var sequenceCount = Math.Max(1, sequences.Count);
            InitialProbabilities = initialAccumulator.Select(v => v / sequenceCount).ToArray();

            var transitions = new double[StateCount, StateCount];
            var emissions = new double[StateCount, SymbolCount];
            for (var i = 0; i < StateCount; i++)
            {
                var denominator = Math.Max(transitionDenominator[i], 1e-300);
                for (var j = 0; j < StateCount; j++) transitions[i, j] = transitionAccumulator[i, j] / denominator;

                var emissionDenominator = Math.Max(stateAccumulator[i], 1e-300);
                for (var k = 0; k < SymbolCount; k++) emissions[i, k] = emissionAccumulator[i, k] / emissionDenominator;
            }
            Transitions = transitions;
            Emissions = emissions;

            if (Math.Abs(likelihood - previousLikelihood) < tolerance) break;
            previousLikelihood = likelihood;
        }
        return likelihood;
    }

    /// <summary>Generates a state and observation sequence from the model.</summary>
    public (int[] States, int[] Observations) Generate(int length, GraviRandom rng)
    {
        var states = new int[length];
        var observations = new int[length];

        states[0] = rng.Categorical(InitialProbabilities);
        for (var step = 1; step < length; step++)
        {
            var row = new double[StateCount];
            for (var j = 0; j < StateCount; j++) row[j] = Transitions[states[step - 1], j];
            states[step] = rng.Categorical(row);
        }

        for (var step = 0; step < length; step++)
        {
            var row = new double[SymbolCount];
            for (var k = 0; k < SymbolCount; k++) row[k] = Emissions[states[step], k];
            observations[step] = rng.Categorical(row);
        }
        return (states, observations);
    }
}

/// <summary>
/// Bayesian linear regression with a conjugate normal-inverse-gamma prior.
/// </summary>
/// <remarks>
/// Conjugacy means the posterior has a closed form, so no sampling is needed and the answer is
/// exact. The payoff over ordinary least squares is that every coefficient comes with a full
/// posterior distribution rather than a point estimate, and predictions carry an interval that
/// widens where the model has seen little data.
/// </remarks>
public sealed class BayesianLinearRegression(double priorPrecision = 1e-3, double noisePrecision = 1.0)
{
    /// <summary>Posterior mean of the coefficients.</summary>
    public NdArray CoefficientMeans { get; private set; } = NdArray.Zeros(0);

    /// <summary>Posterior covariance of the coefficients.</summary>
    public NdArray CoefficientCovariance { get; private set; } = NdArray.Zeros(0, 0);

    /// <summary>True once the model has been fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>Fits the posterior over coefficients.</summary>
    public BayesianLinearRegression Fit(NdArray x, NdArray y)
    {
        var design = WithIntercept(x);
        var features = design.Shape[1];

        // Posterior precision = prior precision * I + beta * X'X, exactly as in the conjugate update.
        var precision = LinAlg.Dot(design.T, design) * noisePrecision;
        for (var j = 0; j < features; j++) precision[j, j] += priorPrecision;

        CoefficientCovariance = LinAlg.Inverse(precision);
        CoefficientMeans = LinAlg.Dot(CoefficientCovariance, LinAlg.Dot(design.T, y) * noisePrecision);
        IsFitted = true;
        return this;
    }

    /// <summary>Posterior mean prediction for each row.</summary>
    public NdArray Predict(NdArray x)
    {
        RequireFitted();
        return LinAlg.Dot(WithIntercept(x), CoefficientMeans);
    }

    /// <summary>
    /// Predictions with their standard deviation, combining coefficient uncertainty with
    /// observation noise.
    /// </summary>
    public (NdArray Mean, NdArray StandardDeviation) PredictWithUncertainty(NdArray x)
    {
        RequireFitted();
        var design = WithIntercept(x);
        var mean = LinAlg.Dot(design, CoefficientMeans);
        var deviation = NdArray.Zeros(design.Shape[0]);

        for (var i = 0; i < design.Shape[0]; i++)
        {
            var row = design.Row(i).Copy();
            // Predictive variance = 1/beta + x' Sigma x: noise plus parameter uncertainty.
            var variance = 1.0 / noisePrecision + LinAlg.Inner(row, LinAlg.Dot(CoefficientCovariance, row));
            deviation.SetAt(i, Math.Sqrt(Math.Max(variance, 0)));
        }
        return (mean, deviation);
    }

    /// <summary>A credible interval for each prediction.</summary>
    public (NdArray Lower, NdArray Upper) PredictInterval(NdArray x, double mass = 0.95)
    {
        var (mean, deviation) = PredictWithUncertainty(x);
        var z = MathUtil.NormalQuantile(1 - (1 - mass) / 2);

        var lower = NdArray.Zeros(mean.Size);
        var upper = NdArray.Zeros(mean.Size);
        for (var i = 0; i < mean.Size; i++)
        {
            lower.SetAt(i, mean.At(i) - z * deviation.At(i));
            upper.SetAt(i, mean.At(i) + z * deviation.At(i));
        }
        return (lower, upper);
    }

    /// <summary>Draws coefficient vectors from the posterior.</summary>
    public NdArray SampleCoefficients(int draws, int seed = 42)
    {
        RequireFitted();
        return new GraviRandom(seed).MultivariateNormal(CoefficientMeans, CoefficientCovariance, draws);
    }

    private static NdArray WithIntercept(NdArray x)
    {
        var design = NdArray.Zeros(x.Shape[0], x.Shape[1] + 1);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            design[i, 0] = 1.0;
            for (var j = 0; j < x.Shape[1]; j++) design[i, j + 1] = x[i, j];
        }
        return design;
    }

    private void RequireFitted()
    {
        if (!IsFitted) throw new InvalidOperationException("The model must be fitted before use.");
    }
}
