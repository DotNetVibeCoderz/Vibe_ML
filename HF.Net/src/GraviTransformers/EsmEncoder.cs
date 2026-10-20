using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>
/// The ESM encoder - ESM-1b, ESM-1v, ESM-2 and the Nucleotide Transformer v1 - on the shared kernels.
/// </summary>
/// <remarks>
/// <para>
/// ESM keeps BERT's tensor names and changes nearly everything they mean, which is why it is built
/// here straight from the checkpoint rather than through the foundation's post-norm encoder:
/// </para>
/// <list type="bullet">
/// <item><b>Pre-norm.</b> <c>attention.LayerNorm</c> runs before attention and <c>LayerNorm</c> before
/// the feed-forward; a final <c>emb_layer_norm_after</c> closes the stack. Read as BERT's post-norm,
/// every parameter loads and the output is confident nonsense.</item>
/// <item><b>Rotary positions</b> (ESM-2): no position table; queries and keys are rotated by position,
/// "rotate half" pairing, base 10000.</item>
/// <item><b>Token dropout.</b> <c>&lt;mask&gt;</c> embeddings are zeroed and every embedding scaled by
/// <c>0.88 / (1 - masked / tokens)</c> - the compensation for having trained with 15% x 80% of tokens
/// masked. Without it every hidden state is about 12% too small.</item>
/// <item><b>Absolute positions</b> (ESM-1b) count from <c>pad + 1</c> over non-pad tokens, as RoBERTa's do,
/// and ESM-1b adds a norm before the first layer.</item>
/// </list>
/// <para>
/// The GELU is always the exact one: ESM's modules call <c>erf</c> directly whatever
/// <c>hidden_act</c> says. Weights are float32 and activations double, as for every other encoder here.
/// </para>
/// </remarks>
internal sealed class EsmEncoder : IInferenceEncoder
{
    /// <summary>The ratio ESM's token dropout was trained with: 15% of tokens selected, 80% of those masked.</summary>
    private const double TrainedMaskRatio = 0.15 * 0.8;

    private readonly EsmOptions _options;
    private readonly int _hidden;
    private readonly int _vocabulary;
    private readonly float[] _tokens;        // [vocabulary, hidden]
    private readonly float[]? _positions;    // [positions, hidden], absolute only
    private readonly int _positionRows;
    private readonly Norm? _embeddingNorm;
    private readonly EncoderBlock[] _blocks;
    private readonly Norm _finalNorm;

    private EsmEncoder(
        EsmOptions options, int hidden, int vocabulary, float[] tokens, float[]? positions,
        Norm? embeddingNorm, EncoderBlock[] blocks, Norm finalNorm)
    {
        _options = options;
        _hidden = hidden;
        _vocabulary = vocabulary;
        _tokens = tokens;
        _positions = positions;
        _positionRows = positions is null ? 0 : positions.Length / hidden;
        _embeddingNorm = embeddingNorm;
        _blocks = blocks;
        _finalNorm = finalNorm;
    }

    /// <summary>Reads an ESM encoder from a checkpoint.</summary>
    /// <exception cref="InvalidDataException">A tensor the encoder needs is missing or misshapen.</exception>
    internal static (EsmEncoder Encoder, LoadReport Report) Load(WeightStore weights, PretrainedConfig config)
    {
        var options = config.Esm ?? throw new ArgumentException("The config is not an ESM config.", nameof(config));
        var prefix = PrefixOf(weights);
        var hidden = config.HiddenSize;
        var epsilon = config.LayerNormEpsilon;
        var loaded = new List<string>();

        float[] Floats(string name, long size)
        {
            var values = weights.ReadFloats(name);
            if (values.Length != size)
            {
                throw new InvalidDataException($"'{name}' holds {values.Length} values but the config implies {size}.");
            }

            loaded.Add(name);
            return values;
        }

        Linear Dense(string name, int inputs, int outputs)
        {
            var layer = Linear.Load(weights, name, inputs, outputs);
            loaded.Add(name + ".weight");
            loaded.Add(name + ".bias");
            return layer;
        }

        Norm LayerNorm(string name)
        {
            var norm = LoadNorm(weights, name, hidden, epsilon, out var scale, out var shift);
            loaded.Add(scale);
            loaded.Add(shift);
            return norm;
        }

        var tokens = Floats($"{prefix}embeddings.word_embeddings.weight", (long)config.VocabularySize * hidden);

        float[]? positions = null;
        if (options.PositionEmbeddingType == "absolute")
        {
            var name = $"{prefix}embeddings.position_embeddings.weight";
            positions = Floats(name, (long)weights.ShapeOf(name)[0] * hidden);
        }

        var embeddingNorm = options.EmbeddingNormBefore ? LayerNorm($"{prefix}embeddings.layer_norm") : null;
        var rotary = options.PositionEmbeddingType == "rotary" ? LoadRotary(weights, prefix, config, loaded) : null;

        var blocks = new EncoderBlock[config.Layers];
        for (var i = 0; i < blocks.Length; i++)
        {
            var layer = $"{prefix}encoder.layer.{i}.";
            blocks[i] = new EncoderBlock(
                NormOrder.Pre,
                config.Heads,
                Dense(layer + "attention.self.query", hidden, hidden),
                Dense(layer + "attention.self.key", hidden, hidden),
                Dense(layer + "attention.self.value", hidden, hidden),
                Dense(layer + "attention.output.dense", hidden, hidden),
                LayerNorm(layer + "attention.LayerNorm"),
                Dense(layer + "intermediate.dense", hidden, config.IntermediateSize),
                Dense(layer + "output.dense", config.IntermediateSize, hidden),
                LayerNorm(layer + "LayerNorm"),
                Activation.ExactGelu,
                rotary);
        }

        var finalNorm = LayerNorm($"{prefix}encoder.emb_layer_norm_after");
        var encoder = new EsmEncoder(options, hidden, config.VocabularySize, tokens, positions, embeddingNorm, blocks, finalNorm);

        return (encoder, new LoadReport(loaded, [], prefix));
    }

    /// <summary>
    /// The rotary embedding, from the checkpoint's stored inverse frequencies when it has them.
    /// </summary>
    /// <remarks>
    /// The published ESM-2 checkpoints store <c>encoder.layer.{i}.attention.self.rotary_embeddings.inv_freq</c>
    /// rounded to float16; transformers 5 moves the first of them to the model-level
    /// <c>rotary_embeddings.inv_freq</c>, keeps its values, and saves under that name. Both spellings are
    /// read, layer 0's winning as it does in transformers' load hook. A checkpoint with neither falls back
    /// to <c>1 / theta^(2i/d)</c>.
    /// </remarks>
    private static RotaryEmbedding LoadRotary(WeightStore weights, string prefix, PretrainedConfig config, List<string> loaded)
    {
        var options = config.Esm!;
        var shared = $"{prefix}rotary_embeddings.inv_freq";
        var perLayer = Enumerable.Range(0, config.Layers)
            .Select(i => $"{prefix}encoder.layer.{i}.attention.self.rotary_embeddings.inv_freq")
            .Where(weights.Contains)
            .ToList();

        var source = weights.Contains(shared) ? shared : perLayer.FirstOrDefault();
        if (source is null) return new RotaryEmbedding(config.HeadSize, options.RopeTheta);

        var stored = weights.Read(source);
        if (stored.Size != config.HeadSize / 2)
        {
            throw new InvalidDataException(
                $"'{source}' holds {stored.Size} frequencies but a {config.HeadSize}-wide head rotates {config.HeadSize / 2} pairs.");
        }

        var frequencies = new double[config.HeadSize / 2];
        for (var i = 0; i < frequencies.Length; i++) frequencies[i] = stored.At(i);

        if (weights.Contains(shared)) loaded.Add(shared);
        loaded.AddRange(perLayer);
        return new RotaryEmbedding(frequencies);
    }

    /// <summary><c>esm.</c> for a model with a head (<c>EsmForMaskedLM</c>), nothing for a bare <c>EsmModel</c>.</summary>
    internal static string PrefixOf(WeightStore weights)
    {
        foreach (var prefix in (string[])["esm.", ""])
        {
            if (weights.Contains($"{prefix}embeddings.word_embeddings.weight")) return prefix;
        }

        throw new InvalidDataException(
            "The checkpoint has no esm.embeddings.word_embeddings.weight; it is not an ESM checkpoint. "
            + "Available names begin: " + string.Join(", ", weights.Names.Take(5)) + ".");
    }

    /// <summary>
    /// A layer norm under either spelling: <c>weight</c>/<c>bias</c>, or <c>gamma</c>/<c>beta</c>,
    /// which transformers 5 writes for ESM's block norms when it saves a model.
    /// </summary>
    internal static Norm LoadNorm(WeightStore weights, string name, int size, double epsilon, out string scale, out string shift)
    {
        scale = weights.Contains($"{name}.weight") ? $"{name}.weight" : $"{name}.gamma";
        shift = weights.Contains($"{name}.bias") ? $"{name}.bias" : $"{name}.beta";

        if (!weights.Contains(scale) || !weights.Contains(shift))
        {
            throw new InvalidDataException($"The checkpoint has no layer norm '{name}' (looked for .weight/.bias and .gamma/.beta).");
        }

        var gamma = weights.Read(scale);
        var beta = weights.Read(shift);
        if (gamma.Size != size || beta.Size != size)
        {
            throw new InvalidDataException($"The layer norm '{name}' has {gamma.Size} values but the model is {size} wide.");
        }

        var g = new double[size];
        var b = new double[size];
        for (var i = 0; i < size; i++)
        {
            g[i] = gamma.At(i);
            b[i] = beta.At(i);
        }

        return new Norm(g, b, epsilon);
    }

    /// <inheritdoc />
    public NdArray Forward(int[] ids, int[] mask, int[]? typeIds = null, NdArray? segmentDelta = null)
        => Forward(ids, mask, attentions: null);

    /// <summary>Runs the encoder, optionally keeping every layer's attention weights.</summary>
    /// <param name="ids">Token ids.</param>
    /// <param name="mask">1 for a real token, 0 for padding.</param>
    /// <param name="attentions">When given, receives one <c>[heads, rows, rows]</c> array per layer.</param>
    internal NdArray Forward(int[] ids, int[] mask, List<double[]>? attentions)
    {
        ArgumentNullException.ThrowIfNull(ids);
        ArgumentNullException.ThrowIfNull(mask);
        if (ids.Length == 0) throw new ArgumentException("There are no tokens to encode.", nameof(ids));
        if (mask.Length != ids.Length) throw new ArgumentException("The attention mask must have one entry per token.", nameof(mask));

        var rows = ids.Length;
        var width = _hidden;
        var hidden = new double[rows * width];

        // Token dropout's observed ratio is computed in float32 by the reference, even in a float64
        // model: (ids == mask).sum().float() / lengths. Taking it in double moves states by ~1e-8.
        var unmasked = 1.0;
        if (_options.TokenDropout)
        {
            var masked = ids.Count(id => id == _options.MaskTokenId);
            var observed = (float)masked / mask.Sum();
            unmasked = 1f - observed;
        }

        var position = _options.PadTokenId;
        for (var i = 0; i < rows; i++)
        {
            var id = ids[i];
            if ((uint)id >= (uint)_vocabulary)
            {
                throw new ArgumentOutOfRangeException(nameof(ids), $"Token id {id} at position {i} is outside the {_vocabulary}-entry vocabulary.");
            }

            var row = hidden.AsSpan(i * width, width);
            var embedding = _tokens.AsSpan(id * width, width);
            if (!_options.TokenDropout)
            {
                for (var d = 0; d < width; d++) row[d] = embedding[d];
            }
            else if (id != _options.MaskTokenId)
            {
                // Multiplied, then divided, in the reference's order.
                for (var d = 0; d < width; d++) row[d] = embedding[d] * (1 - TrainedMaskRatio) / unmasked;
            }

            if (_positions is null) continue;

            // Positions count non-pad tokens from pad + 1; a pad reads the pad row.
            var slot = id == _options.PadTokenId ? _options.PadTokenId : ++position;
            if (slot >= _positionRows)
            {
                throw new ArgumentException(
                    $"The sequence needs position {slot} but the model has {_positionRows} position embeddings. Truncate it.", nameof(ids));
            }

            var table = _positions.AsSpan(slot * width, width);
            for (var d = 0; d < width; d++) row[d] += table[d];
        }

        if (_embeddingNorm is not null) hidden = _embeddingNorm.Apply(hidden, rows);

        // The reference zeroes padding rows before the first layer.
        for (var i = 0; i < rows; i++)
        {
            if (mask[i] == 0) hidden.AsSpan(i * width, width).Clear();
        }

        foreach (var block in _blocks)
        {
            double[]? probabilities = null;
            if (attentions is not null)
            {
                probabilities = new double[block.Heads * rows * rows];
                attentions.Add(probabilities);
            }

            hidden = block.Forward(hidden, rows, mask, causal: false, probabilities);
        }

        return new NdArray(_finalNorm.Apply(hidden, rows), [rows, width]);
    }
}

/// <summary>
/// ESM's contact head: a logistic regression over every layer's and head's attention map, made
/// symmetric and corrected for the average product.
/// </summary>
/// <remarks>
/// <para>
/// transformers' <c>EsmContactPredictionHead</c>, step for step: zero the rows and columns of
/// <c>&lt;eos&gt;</c> (id 2, which transformers fixes rather than reading from the vocabulary), drop
/// the last position and then the first (<c>&lt;cls&gt;</c>), symmetrise each map (<c>A + Aᵀ</c>),
/// subtract the average product <c>rowsum · colsum / total</c>, weight the <c>layers × heads</c>
/// maps with <c>contact_head.regression</c>, and take the sigmoid.
/// </para>
/// <para>
/// The result is one probability per residue pair, <c>[L, L]</c> for <c>L</c> residues - the
/// tokens between <c>&lt;cls&gt;</c> and <c>&lt;eos&gt;</c>.
/// </para>
/// </remarks>
internal sealed class EsmContactHead
{
    /// <summary>transformers' <c>eos_idx</c>, a constructor default it never overrides.</summary>
    private const int EosId = 2;

    private readonly double[] _weight;   // [layers * heads]
    private readonly double _bias;

    private EsmContactHead(double[] weight, double bias)
    {
        _weight = weight;
        _bias = bias;
    }

    /// <summary>Loads the head, or returns null when the checkpoint has none.</summary>
    internal static EsmContactHead? TryLoad(WeightStore weights, PretrainedConfig config)
    {
        if (config.Esm is null) return null;

        var prefix = EsmEncoder.PrefixOf(weights);
        if (!weights.TryRead($"{prefix}contact_head.regression.weight", out var weight)) return null;

        var channels = config.Layers * config.Heads;
        if (weight.Size != channels)
        {
            throw new InvalidDataException(
                $"'{prefix}contact_head.regression.weight' has {weight.Size} values but the model has {channels} attention maps.");
        }

        var bias = weights.TryRead($"{prefix}contact_head.regression.bias", out var b) ? b.At(0) : 0.0;
        return new EsmContactHead([.. Enumerable.Range(0, channels).Select(i => weight.At(i))], bias);
    }

    /// <summary>Contact probabilities from the attention maps of one unpadded sequence.</summary>
    /// <param name="ids">The token ids, <c>&lt;cls&gt;</c> first and <c>&lt;eos&gt;</c> last.</param>
    /// <param name="attentions">Per layer, <c>[heads, rows, rows]</c>.</param>
    /// <param name="heads">Attention heads per layer.</param>
    internal NdArray Apply(int[] ids, IReadOnlyList<double[]> attentions, int heads)
    {
        var rows = ids.Length;
        var size = rows - 2;
        if (size < 1) throw new ArgumentException("Contact prediction needs at least one residue between <cls> and <eos>.", nameof(ids));

        var keep = new double[rows];
        for (var i = 0; i < rows; i++) keep[i] = ids[i] == EosId ? 0.0 : 1.0;

        var channels = attentions.Count * heads;

        // Symmetrised value at (i, j) of one map, with <eos> zeroed and <cls> and the last position
        // dropped. A + Aᵀ is exactly symmetric in floating point, so row sums are column sums.
        double Symmetric(int channel, int i, int j)
        {
            var map = attentions[channel / heads];
            var origin = (long)(channel % heads) * rows * rows;
            var a = map[origin + (long)(i + 1) * rows + (j + 1)] * keep[i + 1] * keep[j + 1];
            var b = map[origin + (long)(j + 1) * rows + (i + 1)] * keep[j + 1] * keep[i + 1];
            return a + b;
        }

        // Pass one, per map: the row sums and the total the average product needs.
        var sums = new double[channels][];
        var totals = new double[channels];
        Parallel.For(0, channels, channel =>
        {
            var rowSums = new double[size];
            var total = 0.0;
            for (var i = 0; i < size; i++)
            {
                for (var j = 0; j < size; j++)
                {
                    var value = Symmetric(channel, i, j);
                    rowSums[i] += value;
                    total += value;
                }
            }

            sums[channel] = rowSums;
            totals[channel] = total;
        });

        // Pass two, per row of the result, so every sum over maps runs in the same order every time.
        var result = NdArray.Zeros(size, size);
        var values = new double[size * size];
        Parallel.For(0, size, i =>
        {
            for (var j = 0; j < size; j++)
            {
                var logit = _bias;
                for (var channel = 0; channel < channels; channel++)
                {
                    var corrected = Symmetric(channel, i, j) - sums[channel][i] * sums[channel][j] / totals[channel];
                    logit += _weight[channel] * corrected;
                }

                values[i * size + j] = 1.0 / (1.0 + Math.Exp(-logit));
            }
        });

        for (var i = 0; i < size; i++)
        {
            for (var j = 0; j < size; j++) result[i, j] = values[i * size + j];
        }

        return result;
    }
}
