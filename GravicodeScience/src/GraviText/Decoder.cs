using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Tokenization;
using Gravicode.Science.GraviText.Transformers;

namespace Gravicode.Science.GraviText.Generation;

/// <summary>
/// Causal self-attention: attention that cannot look forward.
/// </summary>
/// <remarks>
/// <para>
/// The only structural difference between an encoder and a decoder. Position <c>i</c> may attend to
/// positions <c>0..i</c> and no further, enforced by setting the scores above the diagonal to a
/// value the softmax sends to zero.
/// </para>
/// <para>
/// This is what makes generation-time training possible: every position predicts its successor, and
/// because no position can see its own answer, all <c>n</c> predictions can be trained in one
/// forward pass rather than one per token. Leaving the mask off does not produce a worse model — it
/// produces a model that appears to train beautifully and generates nothing, because at inference
/// the future it learned to rely on is not there.
/// </para>
/// </remarks>
public sealed class CausalSelfAttention
{
    private readonly TransformerConfig _config;

    /// <summary>Query projection.</summary>
    public DenseLayer Query { get; }

    /// <summary>Key projection.</summary>
    public DenseLayer Key { get; }

    /// <summary>Value projection.</summary>
    public DenseLayer Value { get; }

    /// <summary>Output projection over the concatenated heads.</summary>
    public DenseLayer Output { get; }

    /// <summary>Attention weights from the most recent forward pass, one matrix per head.</summary>
    public NdArray[] LastAttention { get; private set; } = [];

    /// <summary>Creates the four projections with random weights.</summary>
    public CausalSelfAttention(TransformerConfig config, GraviRandom rng)
    {
        if (config.HiddenSize % config.Heads != 0)
            throw new ArgumentException($"Hidden size {config.HiddenSize} is not divisible by {config.Heads} heads.");

        _config = config;
        Query = new DenseLayer(config.HiddenSize, config.HiddenSize, rng);
        Key = new DenseLayer(config.HiddenSize, config.HiddenSize, rng);
        Value = new DenseLayer(config.HiddenSize, config.HiddenSize, rng);
        Output = new DenseLayer(config.HiddenSize, config.HiddenSize, rng);
    }

    /// <summary>Runs masked attention over a sequence.</summary>
    public NdArray Forward(NdArray x)
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
            {
                var row = new double[length];

                for (var j = 0; j < length; j++)
                {
                    // The causal mask. Everything strictly after i is unreachable.
                    if (j > i) { row[j] = -1e9; continue; }

                    var dot = 0.0;
                    for (var d = 0; d < headSize; d++) dot += q[i, offset + d] * k[j, offset + d];
                    row[j] = dot * scale;
                }

                var weights = MathUtil.Softmax(row);
                for (var j = 0; j < length; j++) scores[i, j] = weights[j];

                for (var d = 0; d < headSize; d++)
                {
                    var accumulator = 0.0;
                    for (var j = 0; j <= i; j++) accumulator += weights[j] * v[j, offset + d];
                    result[i, offset + d] = accumulator;
                }
            }

            attentions[head] = scores;
        }

        LastAttention = attentions;
        return Output.Forward(result);
    }
}

/// <summary>One decoder block: causal self-attention then a feed-forward network, both residual.</summary>
public sealed class TransformerDecoderLayer
{
    /// <summary>The masked self-attention sub-layer.</summary>
    public CausalSelfAttention Attention { get; }

    /// <summary>Normalisation applied after attention.</summary>
    public LayerNorm AttentionNorm { get; }

    /// <summary>Expansion layer of the feed-forward network.</summary>
    public DenseLayer Intermediate { get; }

    /// <summary>Projection back to the model width.</summary>
    public DenseLayer OutputLayer { get; }

    /// <summary>Normalisation applied after the feed-forward network.</summary>
    public LayerNorm OutputNorm { get; }

    /// <summary>Creates a decoder block.</summary>
    public TransformerDecoderLayer(TransformerConfig config, GraviRandom rng)
    {
        Attention = new CausalSelfAttention(config, rng);
        AttentionNorm = new LayerNorm(config.HiddenSize);
        Intermediate = new DenseLayer(config.HiddenSize, config.IntermediateSize, rng);
        OutputLayer = new DenseLayer(config.IntermediateSize, config.HiddenSize, rng);
        OutputNorm = new LayerNorm(config.HiddenSize);
    }

    /// <summary>Runs one block.</summary>
    public NdArray Forward(NdArray x)
    {
        var attended = AttentionNorm.Forward(x + Attention.Forward(x));

        var hidden = Intermediate.Forward(attended);
        for (var i = 0; i < hidden.Size; i++) hidden.SetAt(i, Activations.Gelu(hidden.At(i)));

        return OutputNorm.Forward(attended + OutputLayer.Forward(hidden));
    }
}

/// <summary>How the next token is chosen from the model's distribution.</summary>
/// <param name="Temperature">
/// Divides the logits. Below one sharpens the distribution towards the most likely token; above one
/// flattens it. At zero the choice is deterministic.
/// </param>
/// <param name="TopK">Consider only the <c>k</c> most likely tokens. Zero disables the cut.</param>
/// <param name="TopP">
/// Consider the smallest set of tokens whose probability sums to <c>p</c> — nucleus sampling. Zero
/// disables the cut.
/// </param>
/// <param name="RepetitionPenalty">
/// Divides the logit of any token already generated. Above one discourages repetition.
/// </param>
public readonly record struct SamplingOptions(
    double Temperature = 1.0, int TopK = 0, double TopP = 0.0, double RepetitionPenalty = 1.0)
{
    /// <summary>Always take the most likely token.</summary>
    public static SamplingOptions Greedy => new(Temperature: 0.0);

    /// <summary>A reasonable default for open-ended text: nucleus sampling at 0.9.</summary>
    public static SamplingOptions Nucleus => new(Temperature: 1.0, TopP: 0.9);
}

/// <summary>
/// A decoder-only transformer: the architecture that generates text.
/// </summary>
/// <remarks>
/// <para>
/// Structurally an encoder with two changes — the attention is causally masked, and a language-model
/// head projects each position back to the vocabulary. Those two changes are what turn a model that
/// represents text into one that continues it.
/// </para>
/// <para>
/// This is <b>forward-only</b>, like <see cref="TransformerModel"/>: it runs a model whose weights
/// came from somewhere else, and generates from it. It is deliberately not a training path — a
/// trainable decoder belongs on the autodiff tape alongside <see cref="TransformerTape"/>, where the
/// gradient does not have to be derived by hand.
/// </para>
/// <para>
/// <b>Generation is quadratic here, and knowingly so.</b> Each new token re-runs the whole prefix
/// rather than caching the keys and values of the tokens already processed. A KV cache makes this
/// linear and is the single most valuable optimisation for a real generator; it is left out because
/// it doubles the state a reader has to hold to follow the code, and the shapes this runs at do not
/// need it.
/// </para>
/// </remarks>
public sealed class TransformerDecoder
{
    private readonly TransformerDecoderLayer[] _layers;
    private readonly Vocabulary _vocabulary;

    /// <summary>The model's configuration.</summary>
    public TransformerConfig Config { get; }

    /// <summary>Token embedding table, one row per vocabulary entry.</summary>
    public NdArray TokenEmbeddings { get; }

    /// <summary>Learned position embeddings.</summary>
    public NdArray PositionEmbeddings { get; }

    /// <summary>Normalisation applied to the summed embeddings.</summary>
    public LayerNorm EmbeddingNorm { get; }

    /// <summary>Projection from the model width back to the vocabulary.</summary>
    /// <remarks>
    /// Separate from <see cref="TokenEmbeddings"/> rather than tied to it. Weight tying saves
    /// parameters and usually helps, but making it optional keeps loading a foreign checkpoint —
    /// which may or may not be tied — a matter of assigning weights rather than of matching a
    /// structural assumption.
    /// </remarks>
    public DenseLayer LanguageModelHead { get; }

    /// <summary>Creates a decoder with random weights.</summary>
    public TransformerDecoder(TransformerConfig config, Vocabulary vocabulary, GraviRandom? rng = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(vocabulary);

        rng ??= new GraviRandom(42);

        Config = config;
        _vocabulary = vocabulary;

        TokenEmbeddings = rng.StandardNormal(vocabulary.Count, config.HiddenSize) * 0.02;
        PositionEmbeddings = rng.StandardNormal(config.MaxPositions, config.HiddenSize) * 0.02;
        EmbeddingNorm = new LayerNorm(config.HiddenSize);

        _layers = new TransformerDecoderLayer[config.Layers];
        for (var i = 0; i < config.Layers; i++) _layers[i] = new TransformerDecoderLayer(config, rng);

        LanguageModelHead = new DenseLayer(config.HiddenSize, vocabulary.Count, rng);
    }

    /// <summary>The vocabulary this model generates over.</summary>
    public Vocabulary Vocabulary => _vocabulary;

    /// <summary>Hidden states for a sequence, one row per token.</summary>
    public NdArray Forward(IReadOnlyList<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (tokenIds.Count == 0) throw new ArgumentException("There is nothing to run.", nameof(tokenIds));

        if (tokenIds.Count > Config.MaxPositions)
            throw new ArgumentException(
                $"The sequence is {tokenIds.Count} tokens but the model handles {Config.MaxPositions}.",
                nameof(tokenIds));

        var x = NdArray.Zeros(tokenIds.Count, Config.HiddenSize);

        for (var t = 0; t < tokenIds.Count; t++)
        {
            var id = Math.Clamp(tokenIds[t], 0, TokenEmbeddings.Shape[0] - 1);
            for (var d = 0; d < Config.HiddenSize; d++)
                x[t, d] = TokenEmbeddings[id, d] + PositionEmbeddings[t, d];
        }

        x = EmbeddingNorm.Forward(x);
        foreach (var layer in _layers) x = layer.Forward(x);
        return x;
    }

    /// <summary>Vocabulary logits at every position.</summary>
    public NdArray Logits(IReadOnlyList<int> tokenIds) => LanguageModelHead.Forward(Forward(tokenIds));

    /// <summary>Vocabulary logits for the next token only.</summary>
    public double[] NextTokenLogits(IReadOnlyList<int> tokenIds)
    {
        var logits = Logits(tokenIds);
        var last = logits.Shape[0] - 1;

        var result = new double[logits.Shape[1]];
        for (var v = 0; v < result.Length; v++) result[v] = logits[last, v];
        return result;
    }

    /// <summary>
    /// Generates a continuation of <paramref name="prompt"/>.
    /// </summary>
    /// <param name="prompt">Token ids to continue from.</param>
    /// <param name="maxNewTokens">How many tokens to generate.</param>
    /// <param name="options">How to choose each token.</param>
    /// <param name="rng">Source of randomness. Not used when the sampling is greedy.</param>
    /// <param name="stopTokens">Ids that end generation when produced.</param>
    /// <returns>The prompt followed by the generated tokens.</returns>
    public int[] Generate(IReadOnlyList<int> prompt, int maxNewTokens = 32,
        SamplingOptions options = default, GraviRandom? rng = null, IReadOnlySet<int>? stopTokens = null)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (prompt.Count == 0) throw new ArgumentException("Generation needs at least one token.", nameof(prompt));
        if (maxNewTokens < 0) throw new ArgumentOutOfRangeException(nameof(maxNewTokens));

        if (options == default) options = new SamplingOptions();
        rng ??= new GraviRandom(42);

        var tokens = new List<int>(prompt);

        for (var step = 0; step < maxNewTokens; step++)
        {
            // The context window is finite, so an over-long sequence keeps its tail rather than
            // failing — the recent tokens are what the next one depends on.
            var context = tokens.Count > Config.MaxPositions
                ? tokens.Skip(tokens.Count - Config.MaxPositions).ToArray()
                : tokens.ToArray();

            var logits = NextTokenLogits(context);
            var next = ChooseToken(logits, tokens, options, rng);

            tokens.Add(next);
            if (stopTokens is not null && stopTokens.Contains(next)) break;
        }

        return [.. tokens];
    }

    /// <summary>Generates from a text prompt and decodes the result.</summary>
    public string Generate(string prompt, ITokenizer tokenizer, int maxNewTokens = 32,
        SamplingOptions options = default, GraviRandom? rng = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);

        var ids = _vocabulary.Encode(tokenizer.Tokenize(prompt));
        if (ids.Length == 0) ids = [_vocabulary.ClassId];

        return _vocabulary.Decode(Generate(ids, maxNewTokens, options, rng));
    }

    /// <summary>
    /// The average negative log likelihood the model assigns to a sequence.
    /// </summary>
    /// <remarks>
    /// The quantity a language model is actually judged on. Every position predicts the next token,
    /// and because the attention is causally masked all of those predictions come from one forward
    /// pass — which is the whole reason the mask is worth having. Exponentiating gives perplexity.
    /// </remarks>
    public double CrossEntropy(IReadOnlyList<int> tokenIds)
    {
        if (tokenIds.Count < 2)
            throw new ArgumentException("At least two tokens are needed to score a prediction.", nameof(tokenIds));

        var logits = Logits(tokenIds);
        var total = 0.0;

        for (var t = 0; t < tokenIds.Count - 1; t++)
        {
            var row = new double[logits.Shape[1]];
            for (var v = 0; v < row.Length; v++) row[v] = logits[t, v];

            var probabilities = MathUtil.Softmax(row);
            total -= Math.Log(Math.Max(probabilities[tokenIds[t + 1]], 1e-12));
        }

        return total / (tokenIds.Count - 1);
    }

    /// <summary>Perplexity: the exponential of the cross entropy.</summary>
    public double Perplexity(IReadOnlyList<int> tokenIds) => Math.Exp(CrossEntropy(tokenIds));

    // ------------------------------------------------------------------ sampling

    /// <summary>Applies the sampling options to a logit vector and picks a token.</summary>
    /// <remarks>
    /// Order matters and is the usual one: penalise repeats, then scale by temperature, then cut by
    /// top-k, then by top-p. Applying temperature after the cuts would change which tokens the cuts
    /// should have selected.
    /// </remarks>
    private static int ChooseToken(double[] logits, List<int> generated, SamplingOptions options, GraviRandom rng)
    {
        var scores = (double[])logits.Clone();

        if (options.RepetitionPenalty is > 0 and not 1.0)
            foreach (var token in generated.Distinct())
            {
                if ((uint)token >= (uint)scores.Length) continue;

                // Dividing a negative logit by the penalty would make it larger, which is the
                // opposite of a penalty — so the sign decides which way it applies.
                scores[token] = scores[token] > 0
                    ? scores[token] / options.RepetitionPenalty
                    : scores[token] * options.RepetitionPenalty;
            }

        if (options.Temperature <= 0)
        {
            var best = 0;
            for (var v = 1; v < scores.Length; v++) if (scores[v] > scores[best]) best = v;
            return best;
        }

        for (var v = 0; v < scores.Length; v++) scores[v] /= options.Temperature;

        var order = Enumerable.Range(0, scores.Length).OrderByDescending(v => scores[v]).ToArray();
        var allowed = new HashSet<int>(order);

        if (options.TopK > 0 && options.TopK < order.Length)
            allowed = [.. order.Take(options.TopK)];

        if (options.TopP is > 0 and < 1.0)
        {
            var probabilities = MathUtil.Softmax(scores);
            var nucleus = new HashSet<int>();
            var cumulative = 0.0;

            foreach (var token in order)
            {
                if (!allowed.Contains(token)) continue;

                nucleus.Add(token);
                cumulative += probabilities[token];

                // The token that crosses the threshold is kept, so the set is never empty and the
                // mass covered is at least p rather than just under it.
                if (cumulative >= options.TopP) break;
            }

            allowed = nucleus;
        }

        // Renormalise over the surviving tokens and draw.
        var kept = order.Where(allowed.Contains).ToArray();
        var max = kept.Max(v => scores[v]);

        var weights = kept.Select(v => Math.Exp(scores[v] - max)).ToArray();
        var total = weights.Sum();
        var draw = rng.NextDouble() * total;

        var running = 0.0;
        for (var i = 0; i < kept.Length; i++)
        {
            running += weights[i];
            if (running >= draw) return kept[i];
        }

        return kept[^1];
    }
}
