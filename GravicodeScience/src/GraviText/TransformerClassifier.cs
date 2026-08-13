using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;

namespace Gravicode.Science.GraviText.Transformers;

/// <summary>Adam over a tape parameter, updating its value in place.</summary>
internal sealed class TapeAdam(Tensor parameter, double weightDecay = 0.0)
{
    private readonly NdArray _m = NdArray.ZerosLike(parameter.Value);
    private readonly NdArray _v = NdArray.ZerosLike(parameter.Value);
    private int _step;

    public void Step(double learningRate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        if (parameter.Gradient is null) return;

        _step++;
        var correction1 = 1 - Math.Pow(beta1, _step);
        var correction2 = 1 - Math.Pow(beta2, _step);

        var value = parameter.Value;
        var gradient = parameter.Gradient;

        for (var i = 0; i < value.Size; i++)
        {
            var g = gradient.At(i) + weightDecay * value.At(i);
            _m.SetAt(i, beta1 * _m.At(i) + (1 - beta1) * g);
            _v.SetAt(i, beta2 * _v.At(i) + (1 - beta2) * g * g);

            var mHat = _m.At(i) / correction1;
            var vHat = _v.At(i) / correction2;
            value.SetAt(i, value.At(i) - learningRate * mHat / (Math.Sqrt(vHat) + epsilon));
        }
    }
}

/// <summary>
/// A transformer encoder with a classification head, trained end to end.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="TransformerModel"/> computes a forward pass and nothing else, so a fresh one stays
/// randomly initialised: its output is structured noise, useful for measuring inference cost but
/// not for prediction. This class is the same architecture built on the autodiff tape, which makes
/// the weights reachable by a gradient and therefore trainable on your own labelled text.
/// </para>
/// <para>
/// <b>This is not a way to obtain BERT.</b> A model trained here learns from the corpus you give
/// it, and a few hundred documents will not produce general language understanding — what they can
/// produce is a task-specific classifier. For a small labelled set,
/// <see cref="Vectorization.TfidfVectorizer"/> feeding a linear model remains the stronger and far
/// cheaper baseline, and is worth beating before reaching for this.
/// </para>
/// </remarks>
public sealed class TransformerClassifier
{
    private readonly TransformerConfig _config;
    private readonly int _seed;

    private Tensor _tokenEmbeddings = Tensor.Constant(0.0);
    private Tensor _positionEmbeddings = Tensor.Constant(0.0);
    private Tensor _embeddingGamma = Tensor.Constant(0.0);
    private Tensor _embeddingBeta = Tensor.Constant(0.0);
    private TransformerTape.EncoderWeights[] _layers = [];
    private Tensor _head = Tensor.Constant(0.0);
    private Tensor _headBias = Tensor.Constant(0.0);

    /// <summary>Creates an untrained classifier.</summary>
    public TransformerClassifier(TransformerConfig config, int seed = 42)
    {
        _config = config;
        _seed = seed;
    }

    /// <summary>Number of classes, known once trained.</summary>
    public int ClassCount { get; private set; }

    /// <summary>True once <see cref="Fit"/> has run.</summary>
    public bool IsTrained { get; private set; }

    /// <summary>Mean training loss after each epoch.</summary>
    public IReadOnlyList<double> LossHistory { get; private set; } = [];

    /// <summary>
    /// Trains on tokenised sequences and their labels.
    /// </summary>
    /// <param name="sequences">Token id arrays, one per document.</param>
    /// <param name="labels">Class index per document.</param>
    /// <param name="epochs">Passes over the data.</param>
    /// <param name="learningRate">Adam step size.</param>
    /// <remarks>
    /// One sequence at a time rather than in batches. Transformers normalise per token, so the
    /// result does not depend on batch size, and a batch of one keeps this readable — the cost is
    /// speed, not correctness.
    /// </remarks>
    public TransformerClassifier Fit(IReadOnlyList<int[]> sequences, IReadOnlyList<int> labels,
        int epochs = 20, double learningRate = 0.005)
    {
        if (sequences.Count != labels.Count)
            throw new ArgumentException($"Got {sequences.Count} sequences for {labels.Count} labels.");
        if (sequences.Count == 0) throw new ArgumentException("There is nothing to train on.");

        ClassCount = labels.Max() + 1;
        var rng = new GraviRandom(_seed);
        Initialise(rng);

        var parameters = AllParameters().ToArray();
        var optimisers = parameters.Select(p => new TapeAdam(p, weightDecay: 0.01)).ToArray();

        var order = Enumerable.Range(0, sequences.Count).ToArray();
        var history = new List<double>();

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            // Shuffling matters here: presented in label order, a single-sequence update rule
            // spends each epoch unlearning the previous class.
            for (var i = order.Length - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            var total = 0.0;
            foreach (var index in order)
            {
                var logits = ForwardTape(sequences[index]);
                var loss = TensorOps.SoftmaxCrossEntropy(logits, [labels[index]], [0]);

                loss.Backward();
                for (var p = 0; p < optimisers.Length; p++) optimisers[p].Step(learningRate);

                total += loss.Item;
            }

            history.Add(total / sequences.Count);
        }

        LossHistory = history;
        IsTrained = true;
        return this;
    }

    /// <summary>Class scores for one sequence.</summary>
    public NdArray PredictLogits(int[] tokenIds)
    {
        RequireTrained();
        return ForwardTape(tokenIds).Value;
    }

    /// <summary>The most likely class for one sequence.</summary>
    public int Predict(int[] tokenIds)
    {
        var logits = PredictLogits(tokenIds);
        var best = 0;
        for (var c = 1; c < ClassCount; c++) if (logits[0, c] > logits[0, best]) best = c;
        return best;
    }

    /// <summary>Fraction of sequences classified correctly.</summary>
    public double Score(IReadOnlyList<int[]> sequences, IReadOnlyList<int> labels)
    {
        var correct = 0;
        for (var i = 0; i < sequences.Count; i++)
            if (Predict(sequences[i]) == labels[i]) correct++;
        return (double)correct / sequences.Count;
    }

    /// <summary>
    /// The forward pass: embed, encode, pool, classify.
    /// </summary>
    /// <remarks>
    /// Used for both training and prediction, so there is one definition of what the model
    /// computes. The result is <c>(1, classes)</c> — a single row — which is what the masked
    /// cross-entropy expects.
    /// </remarks>
    private Tensor ForwardTape(int[] tokenIds)
    {
        var x = TransformerTape.LayerNorm(
            TransformerTape.Embed(_tokenEmbeddings, _positionEmbeddings, tokenIds),
            _embeddingGamma, _embeddingBeta);

        foreach (var layer in _layers)
            x = TransformerTape.EncoderLayer(x, layer, _config.Heads);

        return TransformerTape.Dense(TransformerTape.MeanPool(x), _head, _headBias);
    }

    private void Initialise(GraviRandom rng)
    {
        // Embeddings start small: layer norm follows immediately, and large initial values would
        // saturate it before any gradient arrives.
        _tokenEmbeddings = Tensor.Parameter(rng.Normal(0, 0.02, _config.VocabularySize, _config.HiddenSize));
        _positionEmbeddings = Tensor.Parameter(rng.Normal(0, 0.02, _config.MaxPositions, _config.HiddenSize));
        _embeddingGamma = Tensor.Parameter(NdArray.Ones(_config.HiddenSize));
        _embeddingBeta = Tensor.Parameter(NdArray.Zeros(_config.HiddenSize));

        _layers = new TransformerTape.EncoderWeights[_config.Layers];
        for (var i = 0; i < _config.Layers; i++)
            _layers[i] = new TransformerTape.EncoderWeights(
                Query: Tensor.Parameter(Xavier(_config.HiddenSize, _config.HiddenSize, rng)),
                Key: Tensor.Parameter(Xavier(_config.HiddenSize, _config.HiddenSize, rng)),
                Value: Tensor.Parameter(Xavier(_config.HiddenSize, _config.HiddenSize, rng)),
                AttentionOutput: Tensor.Parameter(Xavier(_config.HiddenSize, _config.HiddenSize, rng)),
                AttentionGamma: Tensor.Parameter(NdArray.Ones(_config.HiddenSize)),
                AttentionBeta: Tensor.Parameter(NdArray.Zeros(_config.HiddenSize)),
                Intermediate: Tensor.Parameter(Xavier(_config.HiddenSize, _config.IntermediateSize, rng)),
                IntermediateBias: Tensor.Parameter(NdArray.Zeros(_config.IntermediateSize)),
                OutputProjection: Tensor.Parameter(Xavier(_config.IntermediateSize, _config.HiddenSize, rng)),
                OutputBias: Tensor.Parameter(NdArray.Zeros(_config.HiddenSize)),
                OutputGamma: Tensor.Parameter(NdArray.Ones(_config.HiddenSize)),
                OutputBeta: Tensor.Parameter(NdArray.Zeros(_config.HiddenSize)));

        _head = Tensor.Parameter(Xavier(_config.HiddenSize, ClassCount, rng));
        _headBias = Tensor.Parameter(NdArray.Zeros(ClassCount));
    }

    private IEnumerable<Tensor> AllParameters()
    {
        yield return _tokenEmbeddings;
        yield return _positionEmbeddings;
        yield return _embeddingGamma;
        yield return _embeddingBeta;

        foreach (var layer in _layers)
            foreach (var parameter in layer.Parameters)
                yield return parameter;

        yield return _head;
        yield return _headBias;
    }

    /// <summary>Xavier initialisation, which keeps activation variance roughly constant with depth.</summary>
    private static NdArray Xavier(int inputs, int outputs, GraviRandom rng)
    {
        var limit = Math.Sqrt(6.0 / (inputs + outputs));
        return rng.Uniform(-limit, limit, inputs, outputs);
    }

    private void RequireTrained()
    {
        if (!IsTrained) throw new InvalidOperationException("The classifier must be trained before use.");
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"TransformerClassifier(layers={_config.Layers}, hidden={_config.HiddenSize}, "
        + $"heads={_config.Heads}, classes={ClassCount}, trained={IsTrained})";
}
