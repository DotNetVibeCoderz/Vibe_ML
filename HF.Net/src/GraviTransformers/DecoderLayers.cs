using System.Runtime.CompilerServices;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>Every layer's keys and values for the tokens seen so far, grown as the text grows.</summary>
/// <remarks>
/// Grown rather than sized for the model's full context: Llama 3 declares 131,072 positions, and a
/// cache that size would be gigabytes before the first token.
/// </remarks>
internal sealed class KeyValueCache
{
    internal KeyValueCache(int layers, int keyValueHeads, int headSize, int capacity = 64)
    {
        HeadSize = headSize;
        Capacity = capacity;
        Keys = new double[layers][][];
        Values = new double[layers][][];

        for (var layer = 0; layer < layers; layer++)
        {
            Keys[layer] = new double[keyValueHeads][];
            Values[layer] = new double[keyValueHeads][];
            for (var head = 0; head < keyValueHeads; head++)
            {
                Keys[layer][head] = new double[capacity * headSize];
                Values[layer][head] = new double[capacity * headSize];
            }
        }
    }

    internal int HeadSize { get; }

    internal int Capacity { get; private set; }

    /// <summary><c>[layer][key/value head][position * headSize + d]</c>.</summary>
    internal double[][][] Keys { get; }

    /// <summary><c>[layer][key/value head][position * headSize + d]</c>.</summary>
    internal double[][][] Values { get; }

    /// <summary>How many positions are filled.</summary>
    internal int Length { get; set; }

    /// <summary>Makes room for <paramref name="positions"/> positions in all.</summary>
    internal void Reserve(int positions)
    {
        if (positions <= Capacity) return;

        var capacity = Math.Max(positions, 2 * Capacity);
        foreach (var layer in Keys) for (var head = 0; head < layer.Length; head++) Array.Resize(ref layer[head], capacity * HeadSize);
        foreach (var layer in Values) for (var head = 0; head < layer.Length; head++) Array.Resize(ref layer[head], capacity * HeadSize);
        Capacity = capacity;
    }
}

/// <summary>Rotary position embedding: pairs of dimensions rotated by an angle that grows with position.</summary>
/// <remarks>
/// <para>
/// transformers' "rotate half" convention: dimension <c>i</c> pairs with <c>i + d/2</c>, not with
/// <c>i + 1</c>. The two conventions give the same model only if the weights were permuted to match,
/// which is exactly what the Hub's converted checkpoints did.
/// </para>
/// <para>
/// The angles are computed in double. transformers computes them in float32 even inside a float64
/// model, so on a long text the two differ in the seventh digit of an angle - below anything
/// greedy decoding can see.
/// </para>
/// </remarks>
internal sealed class RotaryEmbedding
{
    private readonly double[] _inverseFrequency;   // [dimensions / 2]
    private readonly List<double[]> _cos = [];
    private readonly List<double[]> _sin = [];

    internal RotaryEmbedding(CausalLanguageModelConfig config)
    {
        Dimensions = config.RotaryDimensions;
        var half = Dimensions / 2;
        _inverseFrequency = new double[half];
        for (var i = 0; i < half; i++) _inverseFrequency[i] = 1.0 / Math.Pow(config.RopeTheta, 2.0 * i / Dimensions);

        var p = config.RopeParameters;
        switch (config.RopeType)
        {
            case "linear":
                for (var i = 0; i < half; i++) _inverseFrequency[i] /= p["factor"];
                break;

            case "llama3":
            {
                // transformers' _compute_llama3_parameters: long wavelengths slowed by the factor,
                // short ones kept, the band between blended.
                var factor = p["factor"];
                var low = p["low_freq_factor"];
                var high = p["high_freq_factor"];
                var original = p["original_max_position_embeddings"];
                var lowWavelength = original / low;
                var highWavelength = original / high;

                for (var i = 0; i < half; i++)
                {
                    var frequency = _inverseFrequency[i];
                    var wavelength = 2 * Math.PI / frequency;
                    var scaled = wavelength > lowWavelength ? frequency / factor : frequency;

                    if (!(wavelength < highWavelength) && !(wavelength > lowWavelength))
                    {
                        var smooth = (original / wavelength - low) / (high - low);
                        scaled = (1 - smooth) * scaled / factor + smooth * scaled;
                    }

                    _inverseFrequency[i] = scaled;
                }

                break;
            }
        }
    }

    internal int Dimensions { get; }

    /// <summary>Rotates the first <see cref="Dimensions"/> values of one head's vector for <paramref name="position"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal void Apply(Span<double> head, int position)
    {
        var (cos, sin) = Table(position);
        var half = Dimensions / 2;

        for (var i = 0; i < half; i++)
        {
            var first = head[i];
            var second = head[i + half];
            head[i] = first * cos[i] + -second * sin[i];
            head[i + half] = second * cos[i] + first * sin[i];
        }
    }

    private (double[] Cos, double[] Sin) Table(int position)
    {
        lock (_cos)
        {
            while (_cos.Count <= position)
            {
                var p = _cos.Count;
                var cos = new double[_inverseFrequency.Length];
                var sin = new double[_inverseFrequency.Length];
                for (var i = 0; i < cos.Length; i++)
                {
                    var angle = p * _inverseFrequency[i];
                    cos[i] = Math.Cos(angle);
                    sin[i] = Math.Sin(angle);
                }

                _cos.Add(cos);
                _sin.Add(sin);
            }

            return (_cos[position], _sin[position]);
        }
    }
}

/// <summary>Root-mean-square normalization: <c>x / sqrt(mean(x^2) + eps) * weight</c>, no mean subtracted and no shift.</summary>
internal sealed class RmsNorm(double[] weight, double epsilon)
{
    internal int Width => weight.Length;

    /// <summary>Normalizes each row of <paramref name="input"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal double[] Apply(double[] input, int rows)
    {
        var width = weight.Length;
        var result = new double[input.Length];

        for (var r = 0; r < rows; r++)
        {
            var row = input.AsSpan(r * width, width);
            var squares = 0.0;
            foreach (var value in row) squares += value * value;

            var inverse = 1.0 / Math.Sqrt(squares / width + epsilon);
            var target = result.AsSpan(r * width, width);
            for (var d = 0; d < width; d++) target[d] = weight[d] * (row[d] * inverse);
        }

        return result;
    }

    /// <summary>Normalizes one head-sized slice in place - Qwen3's per-head query and key norm.</summary>
    internal void ApplyInPlace(Span<double> values)
    {
        var squares = 0.0;
        foreach (var value in values) squares += value * value;

        var inverse = 1.0 / Math.Sqrt(squares / values.Length + epsilon);
        for (var d = 0; d < values.Length; d++) values[d] = weight[d] * (values[d] * inverse);
    }
}

/// <summary>A decoder block, run over new rows with the cache holding everything before them.</summary>
internal interface IDecoderLayer
{
    /// <summary>
    /// The block's output for <paramref name="rows"/> new rows at positions <paramref name="start"/>
    /// onwards; appends their keys and values to <paramref name="cache"/>.
    /// </summary>
    double[] Forward(double[] hidden, int rows, KeyValueCache cache, int layer, int start);
}

/// <summary>Causal attention of new rows over the cache, with grouped-query heads and an optional sliding window.</summary>
internal static class CachedAttention
{
    /// <summary>
    /// Appends the new rows' keys and values to the cache and returns each new row's attention output.
    /// </summary>
    /// <param name="projected">Rows of <c>[q: heads * size | k: kvHeads * size | v: kvHeads * size]</c>, positions already applied.</param>
    /// <param name="rows">How many new rows.</param>
    /// <param name="heads">Query heads.</param>
    /// <param name="keyValueHeads">Key/value heads; query head <c>h</c> reads key/value head <c>h / (heads / kvHeads)</c>.</param>
    /// <param name="size">Head width.</param>
    /// <param name="cache">The cache, with room already reserved.</param>
    /// <param name="layer">Which layer's cache.</param>
    /// <param name="start">The first new row's position.</param>
    /// <param name="window">Each row sees at most this many positions, itself included; <c>null</c> for all.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static double[] Attend(
        double[] projected, int rows, int heads, int keyValueHeads, int size, KeyValueCache cache, int layer, int start, int? window)
    {
        var queryWidth = heads * size;
        var keyWidth = keyValueHeads * size;
        var stride = queryWidth + 2 * keyWidth;
        var group = heads / keyValueHeads;
        var scale = 1.0 / Math.Sqrt(size);

        var keys = cache.Keys[layer];
        var values = cache.Values[layer];
        for (var head = 0; head < keyValueHeads; head++)
        {
            for (var r = 0; r < rows; r++)
            {
                Array.Copy(projected, r * stride + queryWidth + head * size, keys[head], (start + r) * size, size);
                Array.Copy(projected, r * stride + queryWidth + keyWidth + head * size, values[head], (start + r) * size, size);
            }
        }

        var result = new double[rows * queryWidth];
        Parallel.For(0, heads * rows, item =>
        {
            var head = item / rows;
            var r = item % rows;
            var position = start + r;
            var first = window is { } w ? Math.Max(0, position - w + 1) : 0;
            var count = position + 1 - first;

            var query = projected.AsSpan(r * stride + head * size, size);
            var k = keys[head / group];
            var v = values[head / group];

            var weights = new double[count];
            Simd.Scores(query, k, count, size, scale, weights, first);

            var largest = double.NegativeInfinity;
            for (var j = 0; j < count; j++) if (weights[j] > largest) largest = weights[j];

            var total = 0.0;
            for (var j = 0; j < count; j++)
            {
                weights[j] = Math.Exp(weights[j] - largest);
                total += weights[j];
            }

            for (var j = 0; j < count; j++) weights[j] /= total;
            Simd.Combine(weights, v, count, size, result.AsSpan(r * queryWidth + head * size, size), first);
        });

        return result;
    }

    internal static double[] Add(double[] a, double[] b)
    {
        var result = new double[a.Length];
        for (var i = 0; i < a.Length; i++) result[i] = a[i] + b[i];
        return result;
    }
}

/// <summary>One GPT-2 block: pre-norm, causal attention over a cache, pre-norm feed-forward.</summary>
internal sealed class Gpt2Block : IDecoderLayer
{
    private readonly Norm _first;
    private readonly Linear _queryKeyValue;   // c_attn, [q | k | v]
    private readonly Linear _projection;      // attn.c_proj
    private readonly Norm _second;
    private readonly Linear _up;              // mlp.c_fc
    private readonly Linear _down;            // mlp.c_proj
    private readonly Func<double, double> _activation;
    private readonly int _heads;

    private Gpt2Block(Norm first, Linear queryKeyValue, Linear projection, Norm second, Linear up, Linear down, Func<double, double> activation, int heads)
    {
        _first = first;
        _queryKeyValue = queryKeyValue;
        _projection = projection;
        _second = second;
        _up = up;
        _down = down;
        _activation = activation;
        _heads = heads;
    }

    internal static Gpt2Block Load(WeightStore weights, string prefix, CausalLanguageModelConfig config)
    {
        var hidden = config.HiddenSize;
        return new Gpt2Block(
            Norm.Load(weights, $"{prefix}.ln_1", hidden, config.LayerNormEpsilon),
            Conv1D(weights, $"{prefix}.attn.c_attn", hidden, 3 * hidden),
            Conv1D(weights, $"{prefix}.attn.c_proj", hidden, hidden),
            Norm.Load(weights, $"{prefix}.ln_2", hidden, config.LayerNormEpsilon),
            Conv1D(weights, $"{prefix}.mlp.c_fc", hidden, config.IntermediateSize),
            Conv1D(weights, $"{prefix}.mlp.c_proj", config.IntermediateSize, hidden),
            Activation.For(config.Activation),
            config.Heads);
    }

    /// <summary>A GPT-2 <c>Conv1D</c>: a linear layer stored <c>[inputs, outputs]</c>, turned to <c>[outputs, inputs]</c>.</summary>
    private static Linear Conv1D(WeightStore weights, string name, int inputs, int outputs)
    {
        var shape = weights.ShapeOf($"{name}.weight");
        if (shape.Length != 2 || shape[0] != inputs || shape[1] != outputs)
        {
            throw new InvalidDataException(
                $"'{name}.weight' is [{string.Join(", ", shape)}]; a GPT-2 Conv1D from {inputs} to {outputs} is [{inputs}, {outputs}].");
        }

        return Linear.FromFloats(weights.ReadFloats($"{name}.weight"), weights.ReadRange($"{name}.bias", 0, outputs), inputs, outputs, inputsFirst: true);
    }

    public double[] Forward(double[] hidden, int rows, KeyValueCache cache, int layer, int start)
    {
        var size = _projection.Outputs / _heads;
        var projected = _queryKeyValue.Apply(_first.Apply(hidden, rows), rows);
        var context = CachedAttention.Attend(projected, rows, _heads, _heads, size, cache, layer, start, window: null);

        var attended = CachedAttention.Add(hidden, _projection.Apply(context, rows));
        return CachedAttention.Add(attended, _down.Apply(_up.Apply(_second.Apply(attended, rows), rows, _activation), rows));
    }
}

/// <summary>
/// One Llama-family block: RMSNorm, attention with rotary positions and grouped-query heads, RMSNorm,
/// and a gated feed-forward <c>down(act(gate(x)) * up(x))</c>.
/// </summary>
internal sealed class LlamaBlock : IDecoderLayer
{
    private readonly RmsNorm _first;
    private readonly Linear _queryKeyValue;   // [q | k | v]
    private readonly RmsNorm? _queryNorm;
    private readonly RmsNorm? _keyNorm;
    private readonly Linear _output;
    private readonly RmsNorm _second;
    private readonly Linear _gateUp;          // [gate | up]
    private readonly Linear _down;
    private readonly Func<double, double> _activation;
    private readonly RotaryEmbedding _rotary;
    private readonly CausalLanguageModelConfig _config;

    private LlamaBlock(
        RmsNorm first, Linear queryKeyValue, RmsNorm? queryNorm, RmsNorm? keyNorm, Linear output, RmsNorm second,
        Linear gateUp, Linear down, RotaryEmbedding rotary, CausalLanguageModelConfig config)
    {
        _first = first;
        _queryKeyValue = queryKeyValue;
        _queryNorm = queryNorm;
        _keyNorm = keyNorm;
        _output = output;
        _second = second;
        _gateUp = gateUp;
        _down = down;
        _rotary = rotary;
        _config = config;
        _activation = Activation.For(config.Activation);
    }

    internal static LlamaBlock Load(WeightStore weights, string prefix, CausalLanguageModelConfig config, RotaryEmbedding rotary)
    {
        var hidden = config.HiddenSize;
        var queryWidth = config.Heads * config.HeadSize;
        var keyWidth = config.KeyValueHeads * config.HeadSize;
        var inner = config.IntermediateSize;

        RmsNorm Norm(string name, int width) => new(weights.ReadRange($"{name}.weight", 0, width), config.LayerNormEpsilon);

        return new LlamaBlock(
            Norm($"{prefix}.input_layernorm", hidden),
            Linear.Stack(
                Dense(weights, $"{prefix}.self_attn.q_proj", hidden, queryWidth, config.AttentionBias),
                Dense(weights, $"{prefix}.self_attn.k_proj", hidden, keyWidth, config.AttentionBias),
                Dense(weights, $"{prefix}.self_attn.v_proj", hidden, keyWidth, config.AttentionBias)),
            config.QueryKeyNorm ? Norm($"{prefix}.self_attn.q_norm", config.HeadSize) : null,
            config.QueryKeyNorm ? Norm($"{prefix}.self_attn.k_norm", config.HeadSize) : null,
            Dense(weights, $"{prefix}.self_attn.o_proj", queryWidth, hidden, config.OutputBias),
            Norm($"{prefix}.post_attention_layernorm", hidden),
            Linear.Stack(
                Dense(weights, $"{prefix}.mlp.gate_proj", hidden, inner, config.MlpBias),
                Dense(weights, $"{prefix}.mlp.up_proj", hidden, inner, config.MlpBias)),
            Dense(weights, $"{prefix}.mlp.down_proj", inner, hidden, config.MlpBias),
            rotary,
            config);
    }

    /// <summary>An <c>nn.Linear</c> read straight to float32, with a bias only where the architecture has one.</summary>
    internal static Linear Dense(WeightStore weights, string name, int inputs, int outputs, bool bias)
        => Linear.FromFloats(
            weights.ReadFloats($"{name}.weight"),
            bias && weights.Contains($"{name}.bias") ? weights.ReadRange($"{name}.bias", 0, outputs) : new double[outputs],
            inputs,
            outputs);

    public double[] Forward(double[] hidden, int rows, KeyValueCache cache, int layer, int start)
    {
        var heads = _config.Heads;
        var kvHeads = _config.KeyValueHeads;
        var size = _config.HeadSize;
        var stride = (heads + 2 * kvHeads) * size;

        var projected = _queryKeyValue.Apply(_first.Apply(hidden, rows), rows);

        // Per-head norms (Qwen3), then rotary positions, on every query and key head.
        for (var r = 0; r < rows; r++)
        {
            for (var h = 0; h < heads + kvHeads; h++)
            {
                var slice = projected.AsSpan(r * stride + h * size, size);
                if (h < heads) _queryNorm?.ApplyInPlace(slice);
                else _keyNorm?.ApplyInPlace(slice);
                _rotary.Apply(slice, start + r);
            }
        }

        var context = CachedAttention.Attend(projected, rows, heads, kvHeads, size, cache, layer, start, _config.SlidingWindow);
        var attended = CachedAttention.Add(hidden, _output.Apply(context, rows));

        var inner = _config.IntermediateSize;
        var gateUp = _gateUp.Apply(_second.Apply(attended, rows), rows);
        var gated = new double[rows * inner];
        for (var r = 0; r < rows; r++)
        {
            for (var i = 0; i < inner; i++) gated[r * inner + i] = _activation(gateUp[r * 2 * inner + i]) * gateUp[r * 2 * inner + inner + i];
        }

        return CachedAttention.Add(attended, _down.Apply(gated, rows));
    }
}

/// <summary>
/// One GPT-NeoX block: LayerNorm, attention with rotary positions on part of each head, and - with a
/// parallel residual - a feed-forward that reads the block's input rather than the attention's output.
/// </summary>
internal sealed class NeoXBlock : IDecoderLayer
{
    private readonly Norm _first;
    private readonly Linear _queryKeyValue;   // reordered to [q | k | v]
    private readonly Linear _output;
    private readonly Norm _second;
    private readonly Linear _up;
    private readonly Linear _down;
    private readonly Func<double, double> _activation;
    private readonly RotaryEmbedding _rotary;
    private readonly CausalLanguageModelConfig _config;

    private NeoXBlock(Norm first, Linear queryKeyValue, Linear output, Norm second, Linear up, Linear down, RotaryEmbedding rotary, CausalLanguageModelConfig config)
    {
        _first = first;
        _queryKeyValue = queryKeyValue;
        _output = output;
        _second = second;
        _up = up;
        _down = down;
        _rotary = rotary;
        _config = config;
        _activation = Activation.For(config.Activation);
    }

    internal static NeoXBlock Load(WeightStore weights, string prefix, CausalLanguageModelConfig config, RotaryEmbedding rotary)
    {
        var hidden = config.HiddenSize;
        return new NeoXBlock(
            Norm.Load(weights, $"{prefix}.input_layernorm", hidden, config.LayerNormEpsilon),
            FusedQueryKeyValue(weights, $"{prefix}.attention.query_key_value", config),
            LlamaBlock.Dense(weights, $"{prefix}.attention.dense", hidden, hidden, config.OutputBias),
            Norm.Load(weights, $"{prefix}.post_attention_layernorm", hidden, config.LayerNormEpsilon),
            LlamaBlock.Dense(weights, $"{prefix}.mlp.dense_h_to_4h", hidden, config.IntermediateSize, true),
            LlamaBlock.Dense(weights, $"{prefix}.mlp.dense_4h_to_h", config.IntermediateSize, hidden, true),
            rotary,
            config);
    }

    /// <summary>
    /// NeoX's fused projection, whose rows run head by head - <c>[q_h | k_h | v_h]</c> for each head in
    /// turn - regrouped into all queries, then all keys, then all values.
    /// </summary>
    private static Linear FusedQueryKeyValue(WeightStore weights, string name, CausalLanguageModelConfig config)
    {
        var hidden = config.HiddenSize;
        var size = config.HeadSize;
        var stored = weights.ReadFloats($"{name}.weight");
        var storedBias = config.AttentionBias && weights.Contains($"{name}.bias") ? weights.ReadRange($"{name}.bias", 0, 3 * hidden) : new double[3 * hidden];

        var weight = new float[stored.Length];
        var bias = new double[3 * hidden];
        for (var head = 0; head < config.Heads; head++)
        {
            for (var part = 0; part < 3; part++)
            {
                for (var d = 0; d < size; d++)
                {
                    var from = head * 3 * size + part * size + d;
                    var to = part * hidden + head * size + d;
                    Array.Copy(stored, from * hidden, weight, to * hidden, hidden);
                    bias[to] = storedBias[from];
                }
            }
        }

        return Linear.FromFloats(weight, bias, hidden, 3 * hidden);
    }

    public double[] Forward(double[] hidden, int rows, KeyValueCache cache, int layer, int start)
    {
        var heads = _config.Heads;
        var size = _config.HeadSize;
        var stride = 3 * heads * size;

        var projected = _queryKeyValue.Apply(_first.Apply(hidden, rows), rows);
        for (var r = 0; r < rows; r++)
        {
            for (var h = 0; h < 2 * heads; h++) _rotary.Apply(projected.AsSpan(r * stride + h * size, size), start + r);
        }

        var context = CachedAttention.Attend(projected, rows, heads, heads, size, cache, layer, start, window: null);
        var attention = _output.Apply(context, rows);

        if (_config.ParallelResidual)
        {
            var feedForward = _down.Apply(_up.Apply(_second.Apply(hidden, rows), rows, _activation), rows);
            var result = new double[hidden.Length];
            for (var i = 0; i < result.Length; i++) result[i] = feedForward[i] + attention[i] + hidden[i];
            return result;
        }

        var attended = CachedAttention.Add(hidden, attention);
        return CachedAttention.Add(attended, _down.Apply(_up.Apply(_second.Apply(attended, rows), rows, _activation), rows));
    }
}
