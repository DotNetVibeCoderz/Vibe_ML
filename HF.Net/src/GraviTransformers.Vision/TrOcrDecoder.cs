using System.Runtime.CompilerServices;

namespace Gravicode.HFNet.GraviTransformers.Vision;

/// <summary>One image's encoder states, projected once into every decoder layer's cross-attention keys and values.</summary>
/// <remarks>
/// The image does not change while its text is generated, so neither do these. Computing them once
/// per image rather than once per token is the cross-attention half of the key/value cache.
/// </remarks>
internal sealed class CrossAttentionStates
{
    internal CrossAttentionStates(double[][][] keys, double[][][] values, int rows)
    {
        Keys = keys;
        Values = values;
        Rows = rows;
    }

    /// <summary><c>[layer][head][row * headSize + d]</c>.</summary>
    internal double[][][] Keys { get; }

    /// <summary><c>[layer][head][row * headSize + d]</c>.</summary>
    internal double[][][] Values { get; }

    /// <summary>How many encoder positions there are.</summary>
    internal int Rows { get; }
}

/// <summary>A text being generated: its tokens so far, its self-attention cache, and the image it reads.</summary>
internal sealed class DecoderSequence
{
    internal DecoderSequence(List<int> tokens, KeyValueCache cache, CrossAttentionStates image)
    {
        Tokens = tokens;
        Cache = cache;
        Image = image;
    }

    internal List<int> Tokens { get; }

    internal KeyValueCache Cache { get; }

    internal CrossAttentionStates Image { get; }

    /// <summary>How many of the tokens fed so far are not padding - what sinusoidal positions count.</summary>
    internal int Unpadded { get; set; }

    /// <summary>An independent copy, for a beam that two continuations share.</summary>
    internal DecoderSequence Fork()
    {
        var source = Cache;
        var copy = new KeyValueCache(source.Keys.Length, source.Keys[0].Length, source.HeadSize, Math.Max(1, source.Capacity));
        var used = source.Length * source.HeadSize;

        for (var layer = 0; layer < source.Keys.Length; layer++)
        {
            for (var head = 0; head < source.Keys[layer].Length; head++)
            {
                Array.Copy(source.Keys[layer][head], copy.Keys[layer][head], used);
                Array.Copy(source.Values[layer][head], copy.Values[layer][head], used);
            }
        }

        copy.Length = source.Length;
        return new DecoderSequence([.. Tokens], copy, Image) { Unpadded = Unpadded };
    }
}

/// <summary>
/// TrOCR's text decoder: token and position embeddings, then blocks of causal self-attention,
/// cross-attention to the image and a feed-forward layer, each post-norm.
/// </summary>
/// <remarks>
/// <para>
/// Post-norm, like BERT and unlike the ViT encoder in front of it: the residual is added first and
/// the sum normalised, three times per block. The parameters are those of any pre-norm decoder, so a
/// block run in the wrong order loads without complaint and produces fluent nonsense.
/// </para>
/// <para>
/// Every step runs all the live sequences - several images, or several beams of one - as rows of one
/// matrix product, so each weight is read once per step however many texts are being decoded. Only
/// the attention, which reads each sequence's own cache and image, runs per sequence.
/// </para>
/// </remarks>
internal sealed class TrOcrDecoder
{
    private readonly TrOcrDecoderConfig _config;
    private readonly float[] _tokenTable;         // [vocab, hidden]
    private readonly double _embedScale;
    private readonly double[] _positions;         // [rows, hidden]: learned (offset 2) or sinusoidal (offset pad + 1)
    private readonly int _positionOffset;
    private readonly Norm? _embeddingNorm;
    private readonly Layer[] _layers;
    private readonly Linear? _bridge;             // enc_to_dec_proj, where the encoder is wider than the decoder expects
    private readonly Linear _output;

    private sealed record Layer(
        Linear SelfQueryKeyValue, Linear SelfOutput, Norm SelfNorm,
        Linear CrossQuery, Linear CrossKeyValue, Linear CrossOutput, Norm CrossNorm,
        Linear Up, Linear Down, Norm FinalNorm, Func<double, double> Activation);

    /// <summary>The epsilon of every norm in the decoder: <c>nn.LayerNorm</c>'s default, which TrOCR never overrides.</summary>
    private const double Epsilon = 1e-5;

    internal TrOcrDecoder(TrOcrDecoderConfig config, WeightStore weights, int encoderWidth)
    {
        _config = config;
        var hidden = config.HiddenSize;
        const string Prefix = "decoder.model.decoder.";

        _tokenTable = weights.ReadFloats($"{Prefix}embed_tokens.weight");
        if (_tokenTable.Length != (long)config.VocabularySize * hidden)
        {
            throw new InvalidDataException(
                $"The token table holds {_tokenTable.Length} values; {config.VocabularySize} tokens of width {hidden} need "
                + $"{(long)config.VocabularySize * hidden}. The config and the weights disagree.");
        }

        _embedScale = config.ScaleEmbedding ? Math.Sqrt(hidden) : 1.0;

        if (config.LearnedPositions)
        {
            // fairseq's numbering: the first real position is 2, after the two it reserves.
            _positions = weights.Read($"{Prefix}embed_positions.weight").AsContiguous().ToArray();
            _positionOffset = 2;
        }
        else
        {
            _positionOffset = config.PadTokenId + 1;
            _positions = Sinusoids(config.MaxPositions + _positionOffset, hidden, config.PadTokenId);
        }

        _embeddingNorm = config.LayerNormEmbedding
            ? Norm.Load(weights, $"{Prefix}layernorm_embedding", hidden, Epsilon)
            : null;

        // VisionEncoderDecoderModel adds this projection only when the encoder's width is not the one
        // the decoder's cross-attention was built for; TrOCR's checkpoints size the cross-attention
        // to the encoder instead and have none.
        var crossWidth = config.CrossAttentionSize;
        if (weights.Contains("enc_to_dec_proj.weight"))
        {
            _bridge = Linear.Load(weights, "enc_to_dec_proj", encoderWidth, hidden);
            crossWidth = hidden;
        }
        else if (encoderWidth != crossWidth)
        {
            throw new InvalidDataException(
                $"The encoder is {encoderWidth} wide but the decoder's cross-attention reads {crossWidth}, and the "
                + "checkpoint has no enc_to_dec_proj to bridge them.");
        }

        var activation = Activation.For(config.Activation);
        _layers = new Layer[config.Layers];
        for (var i = 0; i < config.Layers; i++)
        {
            var p = $"{Prefix}layers.{i}";
            _layers[i] = new Layer(
                Linear.Stack(
                    Dense(weights, $"{p}.self_attn.q_proj", hidden, hidden),
                    Dense(weights, $"{p}.self_attn.k_proj", hidden, hidden),
                    Dense(weights, $"{p}.self_attn.v_proj", hidden, hidden)),
                Dense(weights, $"{p}.self_attn.out_proj", hidden, hidden),
                Norm.Load(weights, $"{p}.self_attn_layer_norm", hidden, Epsilon),
                Dense(weights, $"{p}.encoder_attn.q_proj", hidden, hidden),
                Linear.Stack(
                    Dense(weights, $"{p}.encoder_attn.k_proj", crossWidth, hidden),
                    Dense(weights, $"{p}.encoder_attn.v_proj", crossWidth, hidden)),
                Dense(weights, $"{p}.encoder_attn.out_proj", hidden, hidden),
                Norm.Load(weights, $"{p}.encoder_attn_layer_norm", hidden, Epsilon),
                Dense(weights, $"{p}.fc1", hidden, config.FeedForwardSize),
                Dense(weights, $"{p}.fc2", config.FeedForwardSize, hidden),
                Norm.Load(weights, $"{p}.final_layer_norm", hidden, Epsilon),
                activation);
        }

        // trocr-small keeps its own output layer; trocr-base ties it to the token table and stores none.
        if (!config.TieWordEmbeddings && weights.Contains("decoder.output_projection.weight"))
        {
            var outputs = weights.ShapeOf("decoder.output_projection.weight")[0];
            _output = Linear.FromFloats(weights.ReadFloats("decoder.output_projection.weight"), new double[outputs], hidden, outputs);
        }
        else if (config.TieWordEmbeddings)
        {
            _output = Linear.FromFloats(_tokenTable, new double[config.VocabularySize], hidden, config.VocabularySize);
        }
        else
        {
            throw new InvalidDataException(
                "The checkpoint has no decoder.output_projection.weight and does not tie its output layer to the token table.");
        }
    }

    /// <summary>How many tokens the output layer scores.</summary>
    internal int VocabularySize => _output.Outputs;

    /// <summary>The longest text the positions cover, start token included.</summary>
    internal int MaxPositions => _positions.Length / _config.HiddenSize - _positionOffset;

    private static Linear Dense(WeightStore weights, string name, int inputs, int outputs)
    {
        var shape = weights.ShapeOf($"{name}.weight");
        if (shape.Length != 2 || shape[0] != outputs || shape[1] != inputs)
        {
            throw new InvalidDataException(
                $"'{name}.weight' is [{string.Join(", ", shape)}]; the config makes it [{outputs}, {inputs}].");
        }

        return LlamaBlock.Dense(weights, name, inputs, outputs, bias: true);
    }

    /// <summary>
    /// fairseq's sinusoidal table: sines in the first half of each row, cosines in the second, with
    /// the padding row zeroed.
    /// </summary>
    /// <remarks>
    /// Computed in float32, as transformers does even inside a float64 model - the table is a plain
    /// attribute, not a parameter, so <c>model.double()</c> never converts it.
    /// </remarks>
    internal static double[] Sinusoids(int rows, int width, int padding)
    {
        var half = width / 2;
        var step = (float)(Math.Log(10000) / (half - 1));
        var table = new double[rows * width];

        for (var p = 0; p < rows; p++)
        {
            if (p == padding) continue;

            for (var i = 0; i < half; i++)
            {
                var frequency = MathF.Exp(i * -step);
                var angle = p * frequency;
                table[p * width + i] = MathF.Sin(angle);
                table[p * width + half + i] = MathF.Cos(angle);
            }
        }

        return table;
    }

    /// <summary>Projects one image's encoder states into every layer's cross-attention keys and values.</summary>
    /// <param name="encoderStates">Row-major <c>[rows, encoderWidth]</c>.</param>
    /// <param name="rows">How many encoder positions.</param>
    internal CrossAttentionStates Prepare(double[] encoderStates, int rows)
    {
        var hidden = _config.HiddenSize;
        var heads = _config.Heads;
        var size = _config.HeadSize;
        var states = _bridge is null ? encoderStates : _bridge.Apply(encoderStates, rows);

        var keys = new double[_layers.Length][][];
        var values = new double[_layers.Length][][];

        for (var layer = 0; layer < _layers.Length; layer++)
        {
            var projected = _layers[layer].CrossKeyValue.Apply(states, rows);   // [rows, k | v]
            keys[layer] = new double[heads][];
            values[layer] = new double[heads][];

            for (var head = 0; head < heads; head++)
            {
                var k = new double[rows * size];
                var v = new double[rows * size];
                for (var r = 0; r < rows; r++)
                {
                    Array.Copy(projected, r * 2 * hidden + head * size, k, r * size, size);
                    Array.Copy(projected, r * 2 * hidden + hidden + head * size, v, r * size, size);
                }

                keys[layer][head] = k;
                values[layer][head] = v;
            }
        }

        return new CrossAttentionStates(keys, values, rows);
    }

    /// <summary>A new, empty sequence for an image, to be started with the decoder start token.</summary>
    internal DecoderSequence Start(CrossAttentionStates image)
        => new([], new KeyValueCache(_layers.Length, _config.Heads, _config.HeadSize, capacity: 32), image);

    /// <summary>
    /// Feeds each sequence one token at its next position and returns the next-token logits,
    /// <c>[sequences, vocab]</c> row-major. The tokens are appended and the caches extended.
    /// </summary>
    internal double[] Step(IReadOnlyList<DecoderSequence> sequences, IReadOnlyList<int> tokens)
    {
        var n = sequences.Count;
        var width = _config.HiddenSize;
        var hidden = new double[n * width];

        for (var s = 0; s < n; s++)
        {
            var id = tokens[s];
            if ((uint)id >= (uint)_config.VocabularySize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokens), id, "Token id outside the vocabulary.");
            }

            var position = sequences[s].Cache.Length;
            if (position >= MaxPositions)
            {
                throw new InvalidOperationException($"The decoder's {MaxPositions} positions are used up.");
            }

            // Learned positions count every token from 2. fairseq's sinusoidal ones count only the
            // tokens that are not padding, from pad + 1, and a pad token reads the zeroed padding row -
            // transformers' full forward pass, which is what TrOCR's generate runs (use_cache: false).
            var isPad = id == _config.PadTokenId;
            var row = (_config.LearnedPositions ? position + _positionOffset
                : isPad ? _config.PadTokenId
                : sequences[s].Unpadded + _positionOffset) * width;
            if (!isPad) sequences[s].Unpadded++;
            for (var d = 0; d < width; d++)
            {
                hidden[s * width + d] = _tokenTable[(long)id * width + d] * _embedScale + _positions[row + d];
            }

            sequences[s].Cache.Reserve(position + 1);
        }

        if (_embeddingNorm is not null) hidden = _embeddingNorm.Apply(hidden, n);

        for (var layer = 0; layer < _layers.Length; layer++) hidden = Forward(_layers[layer], layer, hidden, sequences);

        for (var s = 0; s < n; s++)
        {
            sequences[s].Cache.Length++;
            sequences[s].Tokens.Add(tokens[s]);
        }

        return _output.Apply(hidden, n);
    }

    private double[] Forward(Layer block, int layer, double[] hidden, IReadOnlyList<DecoderSequence> sequences)
    {
        var n = sequences.Count;
        var width = _config.HiddenSize;
        var heads = _config.Heads;
        var size = _config.HeadSize;

        // Causal self-attention, each sequence over its own cache.
        var projected = block.SelfQueryKeyValue.Apply(hidden, n);
        var context = new double[n * width];
        for (var s = 0; s < n; s++)
        {
            var row = projected.AsSpan(s * 3 * width, 3 * width).ToArray();
            var cache = sequences[s].Cache;
            var attended = CachedAttention.Attend(row, 1, heads, heads, size, cache, layer, cache.Length, window: null);
            Array.Copy(attended, 0, context, s * width, width);
        }

        hidden = block.SelfNorm.Apply(CachedAttention.Add(hidden, block.SelfOutput.Apply(context, n)), n);

        // Cross-attention, each sequence over its own image.
        var queries = block.CrossQuery.Apply(hidden, n);
        var crossed = new double[n * width];
        var scale = 1.0 / Math.Sqrt(size);
        Parallel.For(0, n * heads, item => Cross(queries, crossed, sequences[item / heads].Image, layer, item / heads, item % heads, width, size, scale));

        hidden = block.CrossNorm.Apply(CachedAttention.Add(hidden, block.CrossOutput.Apply(crossed, n)), n);

        var inner = block.Down.Apply(block.Up.Apply(hidden, n, block.Activation), n);
        return block.FinalNorm.Apply(CachedAttention.Add(hidden, inner), n);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Cross(
        double[] queries, double[] result, CrossAttentionStates image, int layer, int sequence, int head, int width, int size, double scale)
    {
        var rows = image.Rows;
        var weights = new double[rows];
        var query = queries.AsSpan(sequence * width + head * size, size);

        Simd.Scores(query, image.Keys[layer][head], rows, size, scale, weights);

        var largest = double.NegativeInfinity;
        for (var j = 0; j < rows; j++) if (weights[j] > largest) largest = weights[j];

        var total = 0.0;
        for (var j = 0; j < rows; j++)
        {
            weights[j] = Math.Exp(weights[j] - largest);
            total += weights[j];
        }

        for (var j = 0; j < rows; j++) weights[j] /= total;
        Simd.Combine(weights, image.Values[layer][head], rows, size, result.AsSpan(sequence * width + head * size, size));
    }
}
