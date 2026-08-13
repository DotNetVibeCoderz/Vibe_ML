using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviText.Sequence;

/// <summary>
/// A linear-chain conditional random field for sequence labelling.
/// </summary>
/// <remarks>
/// <para>
/// The problem a per-token classifier cannot solve. Labelling each token independently produces
/// sequences that are locally plausible and globally impossible — an <c>I-PER</c> with no
/// <c>B-PER</c> before it, a tag that cannot follow the one preceding it. A CRF adds transition
/// scores between adjacent labels and decodes the highest-scoring <em>sequence</em>, so those
/// transitions can be made expensive or forbidden outright.
/// </para>
/// <para>
/// The model scores a labelling as the sum of per-token emission scores and per-adjacent-pair
/// transition scores, then normalises over every possible labelling. That normaliser is what
/// separates a CRF from a structured perceptron or an HMM: it is computed exactly, over
/// exponentially many sequences, by the forward algorithm in <c>O(n·k²)</c>. Training maximises the
/// conditional log likelihood, and decoding runs Viterbi over the same lattice.
/// </para>
/// <para>
/// <b>Emissions come from outside.</b> This takes a matrix of per-token scores — from a linear
/// model, a transformer, anything — and learns only the transitions. That is deliberate: it is what
/// makes the layer composable, and it matches how a CRF is used in practice, as the final layer of
/// a tagger rather than as a feature-engineering exercise.
/// </para>
/// </remarks>
public sealed class LinearChainCrf
{
    private readonly double[,] _transitions;
    private readonly double[] _start;
    private readonly double[] _end;

    /// <summary>Creates a CRF over a fixed label set.</summary>
    /// <param name="labelCount">Number of distinct labels.</param>
    public LinearChainCrf(int labelCount)
    {
        if (labelCount < 1) throw new ArgumentOutOfRangeException(nameof(labelCount));

        LabelCount = labelCount;
        _transitions = new double[labelCount, labelCount];
        _start = new double[labelCount];
        _end = new double[labelCount];
    }

    /// <summary>Number of distinct labels.</summary>
    public int LabelCount { get; }

    /// <summary>Score of moving from one label to another.</summary>
    public double Transition(int from, int to) => _transitions[from, to];

    /// <summary>Score of a label starting a sequence.</summary>
    public double StartScore(int label) => _start[label];

    /// <summary>Score of a label ending a sequence.</summary>
    public double EndScore(int label) => _end[label];

    /// <summary>Sets a transition score directly.</summary>
    /// <remarks>
    /// For encoding prior knowledge before training, or for building a model by hand. Training
    /// starts from whatever is set here rather than resetting it.
    /// </remarks>
    public void SetTransition(int from, int to, double score) => _transitions[from, to] = score;

    /// <summary>Sets the score of a label starting a sequence.</summary>
    public void SetStartScore(int label, double score) => _start[label] = score;

    /// <summary>Sets the score of a label ending a sequence.</summary>
    public void SetEndScore(int label, double score) => _end[label] = score;

    /// <summary>
    /// Forbids a transition outright, by giving it a score no path can recover from.
    /// </summary>
    /// <remarks>
    /// The practical reason to reach for a CRF. In BIO tagging an <c>I-PER</c> may not follow an
    /// <c>O</c>, and stating that as a hard constraint is better than hoping the training data
    /// teaches it — a learned penalty can always be outvoted by a confident emission, and then the
    /// output is a labelling the scheme says cannot exist.
    /// </remarks>
    public void Forbid(int from, int to) => _transitions[from, to] = double.NegativeInfinity;

    /// <summary>Forbids a label from starting a sequence.</summary>
    public void ForbidStart(int label) => _start[label] = double.NegativeInfinity;

    /// <summary>
    /// Applies the BIO constraints: an <c>I-X</c> may only follow a <c>B-X</c> or another <c>I-X</c>.
    /// </summary>
    /// <param name="labels">The label names, indexed as the model indexes them.</param>
    /// <remarks>
    /// Encoding the scheme's own rules, so the decoder cannot emit a sequence that violates them.
    /// The entity type has to match too: <c>I-LOC</c> following <c>B-PER</c> is as invalid as
    /// <c>I-LOC</c> following <c>O</c>, and checking only the prefix misses it.
    /// </remarks>
    public void ApplyBioConstraints(IReadOnlyList<string> labels)
    {
        ArgumentNullException.ThrowIfNull(labels);
        if (labels.Count != LabelCount)
            throw new ArgumentException($"Expected {LabelCount} label names but got {labels.Count}.", nameof(labels));

        for (var to = 0; to < LabelCount; to++)
        {
            if (!labels[to].StartsWith("I-", StringComparison.Ordinal)) continue;

            var type = labels[to][2..];

            // Nothing may open an entity with a continuation tag.
            ForbidStart(to);

            for (var from = 0; from < LabelCount; from++)
            {
                var valid = (labels[from].StartsWith("B-", StringComparison.Ordinal)
                             || labels[from].StartsWith("I-", StringComparison.Ordinal))
                            && labels[from][2..] == type;

                if (!valid) Forbid(from, to);
            }
        }
    }

    /// <summary>
    /// The highest-scoring label sequence, by Viterbi decoding.
    /// </summary>
    /// <param name="emissions">Per-token label scores, one row per token.</param>
    /// <remarks>
    /// Exact, and the reason a CRF beats independent per-token argmax: the best sequence is
    /// generally not the sequence of best tokens. Taking each token's top label independently
    /// ignores every transition score, which is the only thing the model added.
    /// </remarks>
    public int[] Decode(NdArray emissions)
    {
        var steps = ValidateEmissions(emissions);
        if (steps == 0) return [];

        var best = new double[steps, LabelCount];
        var backpointer = new int[steps, LabelCount];

        for (var label = 0; label < LabelCount; label++)
            best[0, label] = _start[label] + emissions[0, label];

        for (var t = 1; t < steps; t++)
            for (var to = 0; to < LabelCount; to++)
            {
                var bestScore = double.NegativeInfinity;
                var bestFrom = 0;

                for (var from = 0; from < LabelCount; from++)
                {
                    var score = best[t - 1, from] + _transitions[from, to];
                    if (score <= bestScore) continue;

                    bestScore = score;
                    bestFrom = from;
                }

                best[t, to] = bestScore + emissions[t, to];
                backpointer[t, to] = bestFrom;
            }

        var final = 0;
        var finalScore = double.NegativeInfinity;

        for (var label = 0; label < LabelCount; label++)
        {
            var score = best[steps - 1, label] + _end[label];
            if (score <= finalScore) continue;

            finalScore = score;
            final = label;
        }

        var path = new int[steps];
        path[^1] = final;
        for (var t = steps - 1; t > 0; t--) path[t - 1] = backpointer[t, path[t]];

        return path;
    }

    /// <summary>The unnormalised score of one labelling.</summary>
    public double Score(NdArray emissions, IReadOnlyList<int> labels)
    {
        var steps = ValidateEmissions(emissions);
        if (labels.Count != steps)
            throw new ArgumentException($"Expected {steps} labels but got {labels.Count}.", nameof(labels));
        if (steps == 0) return 0.0;

        var total = _start[labels[0]] + emissions[0, labels[0]];

        for (var t = 1; t < steps; t++)
            total += _transitions[labels[t - 1], labels[t]] + emissions[t, labels[t]];

        return total + _end[labels[^1]];
    }

    /// <summary>
    /// The log of the sum over every possible labelling — the partition function.
    /// </summary>
    /// <remarks>
    /// Computed exactly by the forward algorithm in <c>O(n·k²)</c>, over what would otherwise be
    /// <c>kⁿ</c> sequences. This is the quantity that makes a CRF a probability model rather than a
    /// scoring function, and having it exactly is why the gradient below is exact too.
    /// </remarks>
    public double LogPartition(NdArray emissions)
    {
        var steps = ValidateEmissions(emissions);
        if (steps == 0) return 0.0;

        var alpha = new double[LabelCount];
        for (var label = 0; label < LabelCount; label++)
            alpha[label] = _start[label] + emissions[0, label];

        var next = new double[LabelCount];

        for (var t = 1; t < steps; t++)
        {
            for (var to = 0; to < LabelCount; to++)
            {
                var accumulator = double.NegativeInfinity;
                for (var from = 0; from < LabelCount; from++)
                    accumulator = LogAdd(accumulator, alpha[from] + _transitions[from, to]);

                next[to] = accumulator + emissions[t, to];
            }

            Array.Copy(next, alpha, LabelCount);
        }

        var total = double.NegativeInfinity;
        for (var label = 0; label < LabelCount; label++) total = LogAdd(total, alpha[label] + _end[label]);
        return total;
    }

    /// <summary>The conditional log likelihood of a labelling: its score minus the partition.</summary>
    public double LogLikelihood(NdArray emissions, IReadOnlyList<int> labels)
        => Score(emissions, labels) - LogPartition(emissions);

    /// <summary>
    /// Per-token marginal probabilities, by the forward-backward algorithm.
    /// </summary>
    /// <remarks>
    /// Different from the Viterbi path and often more useful: these say how confident the model is
    /// about each token given the whole sequence, which is what a downstream threshold or an
    /// abstention rule needs. The most probable label at each position need not form the most
    /// probable sequence.
    /// </remarks>
    public NdArray Marginals(NdArray emissions)
    {
        var steps = ValidateEmissions(emissions);
        var result = NdArray.Zeros(Math.Max(steps, 1), LabelCount);
        if (steps == 0) return result;

        var (alpha, beta) = ForwardBackward(emissions, steps);
        var evidence = LogPartition(emissions);

        for (var t = 0; t < steps; t++)
            for (var label = 0; label < LabelCount; label++)
                result[t, label] = Math.Exp(alpha[t, label] + beta[t, label] - evidence);

        return result;
    }

    /// <summary>
    /// Trains the transition scores by gradient ascent on the conditional log likelihood.
    /// </summary>
    /// <param name="sequences">Emission matrices, one per sequence.</param>
    /// <param name="labels">The true labelling of each sequence.</param>
    /// <param name="epochs">Passes over the data.</param>
    /// <param name="learningRate">Step size.</param>
    /// <param name="l2">Weight decay on the transition scores.</param>
    /// <remarks>
    /// <para>
    /// The gradient of the log likelihood with respect to a transition score is its observed count
    /// minus its expected count under the model — the classic exponential-family form. Both are
    /// available exactly: the observed count from the true labelling, the expected count from
    /// forward-backward. No sampling and no approximation are involved.
    /// </para>
    /// <para>
    /// Forbidden transitions stay forbidden: a negative-infinity score has no gradient and must not
    /// acquire one, or the constraints dissolve over training.
    /// </para>
    /// </remarks>
    public LinearChainCrf Fit(IReadOnlyList<NdArray> sequences, IReadOnlyList<int[]> labels,
        int epochs = 50, double learningRate = 0.1, double l2 = 1e-4)
    {
        ArgumentNullException.ThrowIfNull(sequences);
        ArgumentNullException.ThrowIfNull(labels);
        if (sequences.Count != labels.Count)
            throw new ArgumentException("There must be one labelling per sequence.", nameof(labels));

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            var transitionGradient = new double[LabelCount, LabelCount];
            var startGradient = new double[LabelCount];
            var endGradient = new double[LabelCount];

            for (var s = 0; s < sequences.Count; s++)
            {
                var emissions = sequences[s];
                var truth = labels[s];
                var steps = emissions.Shape[0];
                if (steps == 0) continue;

                // Observed counts from the true labelling.
                startGradient[truth[0]] += 1;
                endGradient[truth[^1]] += 1;
                for (var t = 1; t < steps; t++) transitionGradient[truth[t - 1], truth[t]] += 1;

                // Expected counts under the model.
                var (alpha, beta) = ForwardBackward(emissions, steps);
                var evidence = LogPartition(emissions);

                for (var label = 0; label < LabelCount; label++)
                {
                    startGradient[label] -= Math.Exp(alpha[0, label] + beta[0, label] - evidence);
                    endGradient[label] -= Math.Exp(alpha[steps - 1, label] + beta[steps - 1, label] - evidence);
                }

                for (var t = 1; t < steps; t++)
                    for (var from = 0; from < LabelCount; from++)
                        for (var to = 0; to < LabelCount; to++)
                        {
                            if (double.IsNegativeInfinity(_transitions[from, to])) continue;

                            var edge = alpha[t - 1, from] + _transitions[from, to]
                                       + emissions[t, to] + beta[t, to] - evidence;
                            transitionGradient[from, to] -= Math.Exp(edge);
                        }
            }

            for (var from = 0; from < LabelCount; from++)
            {
                if (!double.IsNegativeInfinity(_start[from]))
                    _start[from] += learningRate * (startGradient[from] / sequences.Count - l2 * _start[from]);

                if (!double.IsNegativeInfinity(_end[from]))
                    _end[from] += learningRate * (endGradient[from] / sequences.Count - l2 * _end[from]);

                for (var to = 0; to < LabelCount; to++)
                {
                    // A forbidden transition has no gradient, and must not acquire one.
                    if (double.IsNegativeInfinity(_transitions[from, to])) continue;

                    _transitions[from, to] += learningRate
                        * (transitionGradient[from, to] / sequences.Count - l2 * _transitions[from, to]);
                }
            }
        }

        return this;
    }

    // ------------------------------------------------------------------ helpers

    private int ValidateEmissions(NdArray emissions)
    {
        ArgumentNullException.ThrowIfNull(emissions);

        if (emissions.Rank != 2)
            throw new ArgumentException("Emissions must be a rank 2 array of shape (tokens, labels).", nameof(emissions));

        if (emissions.Shape[1] != LabelCount)
            throw new ArgumentException(
                $"Emissions have {emissions.Shape[1]} labels but the model has {LabelCount}.", nameof(emissions));

        return emissions.Shape[0];
    }

    /// <summary>The forward and backward log-space tables.</summary>
    private (double[,] Alpha, double[,] Beta) ForwardBackward(NdArray emissions, int steps)
    {
        var alpha = new double[steps, LabelCount];
        var beta = new double[steps, LabelCount];

        for (var label = 0; label < LabelCount; label++)
            alpha[0, label] = _start[label] + emissions[0, label];

        for (var t = 1; t < steps; t++)
            for (var to = 0; to < LabelCount; to++)
            {
                var accumulator = double.NegativeInfinity;
                for (var from = 0; from < LabelCount; from++)
                    accumulator = LogAdd(accumulator, alpha[t - 1, from] + _transitions[from, to]);

                alpha[t, to] = accumulator + emissions[t, to];
            }

        for (var label = 0; label < LabelCount; label++) beta[steps - 1, label] = _end[label];

        for (var t = steps - 2; t >= 0; t--)
            for (var from = 0; from < LabelCount; from++)
            {
                var accumulator = double.NegativeInfinity;
                for (var to = 0; to < LabelCount; to++)
                    accumulator = LogAdd(accumulator,
                        _transitions[from, to] + emissions[t + 1, to] + beta[t + 1, to]);

                beta[t, from] = accumulator;
            }

        return (alpha, beta);
    }

    /// <summary>log(e^a + e^b), computed without leaving log space.</summary>
    private static double LogAdd(double a, double b)
    {
        if (double.IsNegativeInfinity(a)) return b;
        if (double.IsNegativeInfinity(b)) return a;

        var max = Math.Max(a, b);
        return max + Math.Log(Math.Exp(a - max) + Math.Exp(b - max));
    }
}
