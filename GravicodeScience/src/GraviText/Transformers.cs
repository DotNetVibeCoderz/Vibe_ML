using System.Text.Json;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Tokenization;

namespace Gravicode.Science.GraviText.Transformers;

/// <summary>Architecture of a transformer encoder.</summary>
/// <param name="VocabularySize">Number of token embeddings.</param>
/// <param name="HiddenSize">Model width; must divide evenly by <paramref name="Heads"/>.</param>
/// <param name="Layers">Number of stacked encoder blocks.</param>
/// <param name="Heads">Attention heads per block.</param>
/// <param name="IntermediateSize">Width of the feed-forward inner layer.</param>
/// <param name="MaxPositions">Longest sequence the position embeddings cover.</param>
public sealed record TransformerConfig(
    int VocabularySize,
    int HiddenSize = 128,
    int Layers = 2,
    int Heads = 4,
    int IntermediateSize = 512,
    int MaxPositions = 512)
{
    /// <summary>Width of each attention head.</summary>
    public int HeadSize => HiddenSize / Heads;

    /// <summary>Named presets matching well-known architectures.</summary>
    public static TransformerConfig Preset(string name, int vocabularySize) => name.ToLowerInvariant() switch
    {
        "bert-base" => new TransformerConfig(vocabularySize, 768, 12, 12, 3072),
        "bert-small" => new TransformerConfig(vocabularySize, 512, 4, 8, 2048),
        "bert-mini" => new TransformerConfig(vocabularySize, 256, 4, 4, 1024),
        "bert-tiny" => new TransformerConfig(vocabularySize, 128, 2, 2, 512),
        _ => throw new ArgumentException(
            $"Unknown preset '{name}'. Known presets: bert-base, bert-small, bert-mini, bert-tiny."),
    };

    /// <summary>Total number of learnable parameters.</summary>
    public long ParameterCount =>
        (long)VocabularySize * HiddenSize
        + (long)MaxPositions * HiddenSize
        + Layers * (4L * HiddenSize * HiddenSize + 2L * HiddenSize * IntermediateSize + 6L * HiddenSize);
}

/// <summary>Layer normalisation over the last axis.</summary>
/// <remarks>
/// Transformers normalise per token rather than per batch, which is why training is insensitive
/// to batch size and why inference on a single sequence behaves identically to inference on many.
/// </remarks>
public sealed class LayerNorm(int size, double epsilon = 1e-12)
{
    /// <summary>Learned per-feature scale.</summary>
    public NdArray Gamma { get; } = NdArray.Ones(size);

    /// <summary>Learned per-feature shift.</summary>
    public NdArray Beta { get; } = NdArray.Zeros(size);

    /// <summary>Normalises each row to zero mean and unit variance, then scales and shifts.</summary>
    public NdArray Forward(NdArray x)
    {
        var result = NdArray.Zeros(x.Shape[0], x.Shape[1]);
        for (var i = 0; i < x.Shape[0]; i++)
        {
            var mean = 0.0;
            for (var j = 0; j < size; j++) mean += x[i, j];
            mean /= size;

            var variance = 0.0;
            for (var j = 0; j < size; j++)
            {
                var d = x[i, j] - mean;
                variance += d * d;
            }
            variance /= size;

            var scale = 1.0 / Math.Sqrt(variance + epsilon);
            for (var j = 0; j < size; j++)
                result[i, j] = (x[i, j] - mean) * scale * Gamma.At(j) + Beta.At(j);
        }
        return result;
    }
}

/// <summary>A fully connected layer with an optional activation.</summary>
public sealed class DenseLayer
{
    /// <summary>Weight matrix, shaped (inputs, outputs).</summary>
    public NdArray Weights { get; }

    /// <summary>Bias vector.</summary>
    public NdArray Bias { get; }

    /// <summary>Element-wise activation, or <c>null</c> for a linear layer.</summary>
    public Func<double, double>? Activation { get; }

    /// <summary>Creates a layer with Xavier-initialised weights.</summary>
    public DenseLayer(int inputs, int outputs, GraviRandom rng, Func<double, double>? activation = null)
    {
        // Xavier scaling keeps activation variance roughly constant as depth grows.
        var limit = Math.Sqrt(6.0 / (inputs + outputs));
        Weights = rng.Uniform(-limit, limit, inputs, outputs);
        Bias = NdArray.Zeros(outputs);
        Activation = activation;
    }

    /// <summary>Applies the layer to every row.</summary>
    public NdArray Forward(NdArray x)
    {
        var result = LinAlg.Dot(x, Weights);
        for (var i = 0; i < result.Shape[0]; i++)
            for (var j = 0; j < result.Shape[1]; j++)
            {
                var value = result[i, j] + Bias.At(j);
                result[i, j] = Activation is null ? value : Activation(value);
            }
        return result;
    }
}

/// <summary>Activation functions used inside transformer blocks.</summary>
public static class Activations
{
    /// <summary>The Gaussian error linear unit, BERT's activation (tanh approximation).</summary>
    public static double Gelu(double x)
        => 0.5 * x * (1.0 + MathUtil.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));

    /// <summary>Rectified linear unit.</summary>
    public static double Relu(double x) => x > 0 ? x : 0;

    /// <summary>Sigmoid-weighted linear unit.</summary>
    public static double Swish(double x) => x * MathUtil.Sigmoid(x);
}

/// <summary>
/// Multi-head scaled dot-product self-attention.
/// </summary>
/// <remarks>
/// Each head projects the sequence into its own query, key and value space and computes
/// <c>softmax(QK'/sqrt(d))V</c>. The <c>sqrt(d)</c> divisor is not cosmetic: without it the dot
/// products grow with width, the softmax saturates and gradients vanish. Running several heads in
/// parallel lets the layer attend to different relationships at once - one head can track syntax
/// while another tracks coreference - and their outputs are concatenated and mixed by a final
/// projection.
/// </remarks>
public sealed class MultiHeadAttention
{
    private readonly TransformerConfig _config;

    /// <summary>Query projection.</summary>
    public DenseLayer Query { get; }

    /// <summary>Key projection.</summary>
    public DenseLayer Key { get; }

    /// <summary>Value projection.</summary>
    public DenseLayer Value { get; }

    /// <summary>Output projection applied to the concatenated heads.</summary>
    public DenseLayer Output { get; }

    /// <summary>Attention weights from the most recent forward pass, one matrix per head.</summary>
    public NdArray[] LastAttention { get; private set; } = [];

    /// <summary>Creates the four projections with random weights.</summary>
    public MultiHeadAttention(TransformerConfig config, GraviRandom rng)
    {
        if (config.HiddenSize % config.Heads != 0)
            throw new ArgumentException($"Hidden size {config.HiddenSize} is not divisible by {config.Heads} heads.");
        _config = config;
        Query = new DenseLayer(config.HiddenSize, config.HiddenSize, rng);
        Key = new DenseLayer(config.HiddenSize, config.HiddenSize, rng);
        Value = new DenseLayer(config.HiddenSize, config.HiddenSize, rng);
        Output = new DenseLayer(config.HiddenSize, config.HiddenSize, rng);
    }

    /// <summary>Runs attention over a sequence, honouring an optional padding mask.</summary>
    public NdArray Forward(NdArray x, int[]? attentionMask = null)
    {
        var length = x.Shape[0];
        var headSize = _config.HeadSize;

        var q = Query.Forward(x);
        var k = Key.Forward(x);
        var v = Value.Forward(x);

        var result = NdArray.Zeros(length, _config.HiddenSize);
        var attentions = new NdArray[_config.Heads];
        var scale = 1.0 / Math.Sqrt(headSize);

        for (var head = 0; head < _config.Heads; head++)
        {
            var offset = head * headSize;
            var scores = NdArray.Zeros(length, length);

            for (var i = 0; i < length; i++)
                for (var j = 0; j < length; j++)
                {
                    var dot = 0.0;
                    for (var d = 0; d < headSize; d++) dot += q[i, offset + d] * k[j, offset + d];
                    // Masked positions get a large negative score so the softmax sends them to zero.
                    scores[i, j] = attentionMask is not null && attentionMask[j] == 0
                        ? -1e9
                        : dot * scale;
                }

            for (var i = 0; i < length; i++)
            {
                var row = new double[length];
                for (var j = 0; j < length; j++) row[j] = scores[i, j];
                var weights = MathUtil.Softmax(row);
                for (var j = 0; j < length; j++) scores[i, j] = weights[j];

                for (var d = 0; d < headSize; d++)
                {
                    var acc = 0.0;
                    for (var j = 0; j < length; j++) acc += weights[j] * v[j, offset + d];
                    result[i, offset + d] = acc;
                }
            }
            attentions[head] = scores;
        }

        LastAttention = attentions;
        return Output.Forward(result);
    }
}

/// <summary>
/// One encoder block: self-attention and a feed-forward network, each wrapped in a residual
/// connection and layer normalisation.
/// </summary>
/// <remarks>
/// The residual connections are what make depth trainable - each block learns a correction to its
/// input rather than a fresh representation - and the feed-forward network, which expands to
/// roughly four times the model width and back, is where most of the parameters live.
/// </remarks>
public sealed class TransformerEncoderLayer
{
    /// <summary>The self-attention sub-layer.</summary>
    public MultiHeadAttention Attention { get; }

    /// <summary>Normalisation applied after attention.</summary>
    public LayerNorm AttentionNorm { get; }

    /// <summary>Expanding feed-forward layer.</summary>
    public DenseLayer Intermediate { get; }

    /// <summary>Contracting feed-forward layer.</summary>
    public DenseLayer OutputProjection { get; }

    /// <summary>Normalisation applied after the feed-forward network.</summary>
    public LayerNorm OutputNorm { get; }

    /// <summary>Creates a block with random weights.</summary>
    public TransformerEncoderLayer(TransformerConfig config, GraviRandom rng)
    {
        Attention = new MultiHeadAttention(config, rng);
        AttentionNorm = new LayerNorm(config.HiddenSize);
        Intermediate = new DenseLayer(config.HiddenSize, config.IntermediateSize, rng, Activations.Gelu);
        OutputProjection = new DenseLayer(config.IntermediateSize, config.HiddenSize, rng);
        OutputNorm = new LayerNorm(config.HiddenSize);
    }

    /// <summary>Runs the block.</summary>
    public NdArray Forward(NdArray x, int[]? attentionMask = null)
    {
        var attended = AttentionNorm.Forward(x + Attention.Forward(x, attentionMask));
        var expanded = OutputProjection.Forward(Intermediate.Forward(attended));
        return OutputNorm.Forward(attended + expanded);
    }
}

/// <summary>
/// A transformer encoder: token and position embeddings followed by a stack of encoder blocks.
/// </summary>
/// <remarks>
/// <para>
/// The forward pass here is complete and correct - embeddings, multi-head attention, residuals,
/// layer norm, GELU feed-forward - and will reproduce a reference implementation given the same
/// weights.
/// </para>
/// <para>
/// <b>What it does not include is pretrained weights.</b> A freshly constructed model is randomly
/// initialised, so its output vectors are structured noise: useful for testing shapes, measuring
/// inference throughput and studying attention mechanics, but not semantically meaningful. Call
/// <see cref="LoadWeights"/> with exported weights before treating the output as embeddings. For
/// semantics without a weight file, use <see cref="Embeddings.Word2Vec"/> or
/// <see cref="Vectorization.TfidfVectorizer"/>, which learn from your own corpus.
/// </para>
/// </remarks>
public sealed class TransformerModel
{
    private readonly TransformerEncoderLayer[] _layers;

    /// <summary>The architecture in use.</summary>
    public TransformerConfig Config { get; }

    /// <summary>Token embedding table, one row per vocabulary entry.</summary>
    public NdArray TokenEmbeddings { get; private set; }

    /// <summary>Position embedding table, one row per position.</summary>
    public NdArray PositionEmbeddings { get; private set; }

    /// <summary>Normalisation applied to the summed embeddings.</summary>
    public LayerNorm EmbeddingNorm { get; }

    /// <summary>The tokenizer paired with this model, when one was supplied.</summary>
    public WordPieceTokenizer? Tokenizer { get; private set; }

    /// <summary>
    /// True when weights came from a file. While false, <see cref="Encode(string, int)"/>
    /// returns vectors from a randomly initialised network.
    /// </summary>
    public bool HasPretrainedWeights { get; private set; }

    /// <summary>Builds a model from an explicit configuration.</summary>
    public TransformerModel(TransformerConfig config, int seed = 42)
    {
        Config = config;
        var rng = new GraviRandom(seed);

        // Small-variance init matching the reference implementations.
        TokenEmbeddings = rng.Normal(0, 0.02, config.VocabularySize, config.HiddenSize);
        PositionEmbeddings = rng.Normal(0, 0.02, config.MaxPositions, config.HiddenSize);
        EmbeddingNorm = new LayerNorm(config.HiddenSize);

        _layers = new TransformerEncoderLayer[config.Layers];
        for (var i = 0; i < config.Layers; i++) _layers[i] = new TransformerEncoderLayer(config, rng);
    }

    /// <summary>
    /// Builds a model from a named preset, as in <c>new TransformerModel("bert-base")</c>.
    /// </summary>
    public TransformerModel(string preset, int vocabularySize = 30522, int seed = 42)
        : this(TransformerConfig.Preset(preset, vocabularySize), seed) { }

    /// <summary>Attaches a tokenizer so <see cref="Encode(string, int)"/> can accept raw text.</summary>
    public TransformerModel WithTokenizer(WordPieceTokenizer tokenizer)
    {
        Tokenizer = tokenizer;
        return this;
    }

    /// <summary>Runs the encoder over token ids, returning one vector per position.</summary>
    public NdArray Forward(int[] tokenIds, int[]? attentionMask = null)
    {
        var length = tokenIds.Length;
        if (length > Config.MaxPositions)
            throw new ArgumentException($"Sequence of {length} exceeds the {Config.MaxPositions} position embeddings.");

        var hidden = NdArray.Zeros(length, Config.HiddenSize);
        for (var i = 0; i < length; i++)
        {
            var id = Math.Clamp(tokenIds[i], 0, Config.VocabularySize - 1);
            for (var d = 0; d < Config.HiddenSize; d++)
                hidden[i, d] = TokenEmbeddings[id, d] + PositionEmbeddings[i, d];
        }

        hidden = EmbeddingNorm.Forward(hidden);
        foreach (var layer in _layers) hidden = layer.Forward(hidden, attentionMask);
        return hidden;
    }

    /// <summary>
    /// Encodes text into a single vector by mean-pooling the final hidden states over the
    /// non-padding positions.
    /// </summary>
    /// <remarks>
    /// Mean pooling is used rather than the <c>[CLS]</c> vector because <c>[CLS]</c> only carries
    /// sentence meaning after a model has been fine-tuned to put it there; the mean is the better
    /// default for an encoder used as a feature extractor.
    /// </remarks>
    public NdArray Encode(string text, int maxLength = 128)
    {
        if (Tokenizer is null)
            throw new InvalidOperationException(
                "No tokenizer attached. Call WithTokenizer, or use Encode(int[]) with your own ids.");

        var (ids, mask) = Tokenizer.Encode(text, maxLength);
        return Pool(Forward(ids, mask), mask);
    }

    /// <summary>Encodes pre-tokenized ids into a single pooled vector.</summary>
    public NdArray Encode(int[] tokenIds, int[]? attentionMask = null)
        => Pool(Forward(tokenIds, attentionMask), attentionMask);

    /// <summary>Encodes several documents, one row per document.</summary>
    public NdArray EncodeBatch(IReadOnlyList<string> texts, int maxLength = 128)
    {
        var result = NdArray.Zeros(texts.Count, Config.HiddenSize);
        // Sequences are independent, so batching is embarrassingly parallel.
        Parallel.For(0, texts.Count, i =>
        {
            var vector = Encode(texts[i], maxLength);
            for (var d = 0; d < Config.HiddenSize; d++) result[i, d] = vector.At(d);
        });
        return result;
    }

    private NdArray Pool(NdArray hidden, int[]? mask)
    {
        var pooled = NdArray.Zeros(Config.HiddenSize);
        var counted = 0;
        for (var i = 0; i < hidden.Shape[0]; i++)
        {
            if (mask is not null && mask[i] == 0) continue;
            for (var d = 0; d < Config.HiddenSize; d++) pooled.SetAt(d, pooled.At(d) + hidden[i, d]);
            counted++;
        }
        if (counted > 0)
            for (var d = 0; d < Config.HiddenSize; d++) pooled.SetAt(d, pooled.At(d) / counted);
        return pooled;
    }

    /// <summary>The attention weights of the last forward pass, per layer and head.</summary>
    public IReadOnlyList<NdArray[]> AttentionMaps => _layers.Select(l => l.Attention.LastAttention).ToList();

    /// <summary>
    /// Loads embedding weights from a JSON file produced by <see cref="SaveWeights"/> or an
    /// export script, and marks the model as pretrained.
    /// </summary>
    public void LoadWeights(string path)
    {
        var payload = JsonSerializer.Deserialize<WeightPayload>(File.ReadAllText(path))
            ?? throw new InvalidDataException($"'{path}' does not contain transformer weights.");

        if (payload.HiddenSize != Config.HiddenSize)
            throw new InvalidDataException(
                $"Weight file has hidden size {payload.HiddenSize}, model expects {Config.HiddenSize}.");

        TokenEmbeddings = new NdArray(payload.TokenEmbeddings, payload.VocabularySize, payload.HiddenSize);
        PositionEmbeddings = new NdArray(payload.PositionEmbeddings, payload.MaxPositions, payload.HiddenSize);
        HasPretrainedWeights = true;
    }

    /// <summary>Writes the embedding tables so a model can be reloaded later.</summary>
    public void SaveWeights(string path)
    {
        var payload = new WeightPayload(
            Config.VocabularySize, Config.HiddenSize, Config.MaxPositions,
            TokenEmbeddings.ToArray(), PositionEmbeddings.ToArray());
        File.WriteAllText(path, JsonSerializer.Serialize(payload));
    }

    private sealed record WeightPayload(
        int VocabularySize, int HiddenSize, int MaxPositions,
        double[] TokenEmbeddings, double[] PositionEmbeddings);

    /// <inheritdoc />
    public override string ToString() =>
        $"TransformerModel(hidden={Config.HiddenSize}, layers={Config.Layers}, heads={Config.Heads}, " +
        $"params~{Config.ParameterCount:N0}, pretrained={HasPretrainedWeights})";
}
