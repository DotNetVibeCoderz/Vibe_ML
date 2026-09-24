using Gravicode.HFNet.GraviTransformers;
using Gravicode.Science.GraviNum;
using Encoder = Gravicode.Science.GraviText.Transformers.TransformerModel;

namespace Gravicode.HFNet.GraviPEFT;

/// <summary>The six projections in a BERT block that a LoRA adapter can attach to.</summary>
internal enum Projection
{
    Query,
    Key,
    Value,
    AttentionOutput,
    Intermediate,
    Output,
}

/// <summary>
/// A post-norm BERT encoder with LoRA adapters in the loop, and its backward pass.
/// </summary>
/// <remarks>
/// <para>
/// The forward pass is the inference encoder's, term for term: the same float32 weight copies, the
/// same linear kernel, the activation named by <c>hidden_act</c>, each norm's epsilon from the
/// config, and the attention biases every pretrained BERT has. That last point is why this exists.
/// The foundation's autodiff encoder has no biases on its query, key and value projections, so a
/// gradient taken through it would be the gradient of a slightly different model - one that
/// trains, converges, and produces adapters that are quietly wrong for the model they are applied
/// to.
/// </para>
/// <para>
/// The backward pass is written out by hand rather than recorded on a tape. The base weights are
/// frozen, so the only gradients wanted are those of the adapter matrices; everything else is
/// propagated and discarded, and below the lowest adapted layer nothing is propagated at all.
/// </para>
/// </remarks>
internal sealed class LoraEncoder
{
    private readonly Layer[] _layers;
    private readonly NdArray _tokens;
    private readonly NdArray _positions;
    private readonly double[] _embeddingScale;
    private readonly double[] _embeddingShift;
    private readonly double _epsilon;
    private readonly int _hidden;
    private readonly int _heads;
    private readonly Func<double, double> _activation;
    private readonly Func<double, double> _derivative;
    private readonly int _lowest;

    private LoraEncoder(
        Layer[] layers, NdArray tokens, NdArray positions, double[] embeddingScale, double[] embeddingShift,
        double epsilon, int hidden, int heads, Func<double, double> activation, Func<double, double> derivative)
    {
        _layers = layers;
        _tokens = tokens;
        _positions = positions;
        _embeddingScale = embeddingScale;
        _embeddingShift = embeddingShift;
        _epsilon = epsilon;
        _hidden = hidden;
        _heads = heads;
        _activation = activation;
        _derivative = derivative;

        _lowest = Array.FindIndex(layers, l => l.Adapters.Any(a => a is not null));
        if (_lowest < 0) _lowest = layers.Length;
    }

    /// <summary>Model width.</summary>
    internal int Hidden => _hidden;

    /// <summary>The adapters in the loop, in layer order.</summary>
    internal IEnumerable<LoraAdapter> Adapters => _layers.SelectMany(l => l.Adapters).OfType<LoraAdapter>();

    /// <summary>Copies a loaded encoder's frozen weights and attaches adapters to it.</summary>
    /// <param name="source">The foundation encoder, the model of record.</param>
    /// <param name="config">Supplies the activation and the norm epsilon.</param>
    /// <param name="adapterAt">The adapter on a layer's projection, or <c>null</c> for none.</param>
    internal static LoraEncoder Build(
        Encoder source, PretrainedConfig config, Func<int, Projection, LoraAdapter?> adapterAt)
    {
        var layers = new Layer[source.Layers.Count];

        Parallel.For(0, layers.Length, i =>
        {
            var block = source.Layers[i];
            var adapters = new LoraAdapter?[6];
            for (var p = 0; p < adapters.Length; p++) adapters[p] = adapterAt(i, (Projection)p);

            layers[i] = new Layer(
                Linear.Stack(
                    Linear.From(block.Attention.Query),
                    Linear.From(block.Attention.Key),
                    Linear.From(block.Attention.Value)),
                Linear.From(block.Attention.Output),
                Linear.From(block.Intermediate),
                Linear.From(block.OutputProjection),
                block.AttentionNorm.Gamma.ToArray(),
                block.AttentionNorm.Beta.ToArray(),
                block.OutputNorm.Gamma.ToArray(),
                block.OutputNorm.Beta.ToArray(),
                adapters);
        });

        return new LoraEncoder(
            layers,
            source.TokenEmbeddings,
            source.PositionEmbeddings,
            source.EmbeddingNorm.Gamma.ToArray(),
            source.EmbeddingNorm.Beta.ToArray(),
            config.LayerNormEpsilon,
            source.Config.HiddenSize,
            source.Config.Heads,
            Activation.For(config.Activation),
            Activation.DerivativeFor(config.Activation));
    }

    // ------------------------------------------------------------------ forward

    /// <summary>Runs the encoder, returning <c>[rows, hidden]</c> row-major.</summary>
    /// <param name="ids">Token ids. Every position is attended to; there is no padding.</param>
    /// <param name="tape">
    /// Receives what the backward pass needs, or <c>null</c> for inference, which also turns
    /// adapter dropout off.
    /// </param>
    /// <param name="dropout">Draws the adapter dropout masks; unused without a tape.</param>
    internal double[] Forward(int[] ids, Tape? tape = null, Random? dropout = null)
    {
        var rows = ids.Length;
        var width = _hidden;

        if (rows > _positions.Shape[0])
        {
            throw new ArgumentException(
                $"The sequence is {rows} tokens but the model has {_positions.Shape[0]} position "
                + "embeddings. Truncate it with maxLength.", nameof(ids));
        }

        var vocabulary = _tokens.Shape[0];
        var hidden = new double[rows * width];

        for (var i = 0; i < rows; i++)
        {
            var id = Math.Clamp(ids[i], 0, vocabulary - 1);
            for (var d = 0; d < width; d++) hidden[i * width + d] = _tokens[id, d] + _positions[i, d];
        }

        hidden = Normalize(hidden, rows, _embeddingScale, _embeddingShift, out _, out _);

        if (tape is not null) tape.Rows = rows;

        foreach (var layer in _layers)
        {
            var record = tape is null ? null : new LayerTape();
            hidden = LayerForward(layer, hidden, rows, record, tape is null ? null : dropout);
            tape?.Layers.Add(record!);
        }

        return hidden;
    }

    private double[] LayerForward(Layer layer, double[] x, int rows, LayerTape? tape, Random? dropout)
    {
        var width = _hidden;
        var inner = layer.Intermediate.Outputs;

        var qkv = layer.QueryKeyValue.Apply(x, rows);
        for (var p = Projection.Query; p <= Projection.Value; p++)
        {
            AddLora(layer, p, x, rows, qkv, 3 * width, (int)p * width, tape, dropout);
        }

        var context = Attend(qkv, rows, out var probabilities);

        var attended = layer.AttentionOutput.Apply(context, rows);
        AddLora(layer, Projection.AttentionOutput, context, rows, attended, width, 0, tape, dropout);

        for (var i = 0; i < attended.Length; i++) attended[i] += x[i];
        var a = Normalize(attended, rows, layer.FirstScale, layer.FirstShift, out var firstHat, out var firstInverse);

        var z = layer.Intermediate.Apply(a, rows);
        AddLora(layer, Projection.Intermediate, a, rows, z, inner, 0, tape, dropout);

        var g = new double[z.Length];
        for (var i = 0; i < z.Length; i++) g[i] = _activation(z[i]);

        var o = layer.Output.Apply(g, rows);
        AddLora(layer, Projection.Output, g, rows, o, width, 0, tape, dropout);

        for (var i = 0; i < o.Length; i++) o[i] += a[i];
        var y = Normalize(o, rows, layer.SecondScale, layer.SecondShift, out var secondHat, out var secondInverse);

        if (tape is not null)
        {
            tape.QueryKeyValue = qkv;
            tape.Probabilities = probabilities;
            tape.FirstHat = firstHat;
            tape.FirstInverse = firstInverse;
            tape.PreActivation = z;
            tape.SecondHat = secondHat;
            tape.SecondInverse = secondInverse;
        }

        return y;
    }

    /// <summary>
    /// Adds <c>scaling * B A dropout(input)</c> into columns
    /// <c>[offset, offset + outputs)</c> of <paramref name="output"/>, as PEFT does.
    /// </summary>
    private static void AddLora(
        Layer layer, Projection projection, double[] input, int rows,
        double[] output, int stride, int offset, LayerTape? tape, Random? dropout)
    {
        var adapter = layer.Adapters[(int)projection];
        if (adapter is null) return;

        var inputs = adapter.Inputs;
        var outputs = adapter.Outputs;
        var rank = adapter.Config.Rank;
        var scaling = adapter.Config.Scaling;
        var a = adapter.A.AsSpan();
        var b = adapter.B.AsSpan();

        var source = input;
        double[]? factors = null;
        var p = adapter.Config.Dropout;

        if (dropout is not null && p > 0)
        {
            source = new double[input.Length];
            factors = new double[input.Length];
            var keep = 1.0 / (1.0 - p);

            for (var i = 0; i < input.Length; i++)
            {
                factors[i] = dropout.NextDouble() < p ? 0.0 : keep;
                source[i] = input[i] * factors[i];
            }
        }

        var u = new double[rows * rank];
        for (var r = 0; r < rows; r++)
        {
            var row = source.AsSpan(r * inputs, inputs);
            for (var k = 0; k < rank; k++) u[r * rank + k] = Simd.Dot(row, a.Slice(k * inputs, inputs));

            for (var o = 0; o < outputs; o++)
            {
                var sum = 0.0;
                for (var k = 0; k < rank; k++) sum += u[r * rank + k] * b[o * rank + k];
                output[r * stride + offset + o] += scaling * sum;
            }
        }

        if (tape is not null) tape.Lora[(int)projection] = new LoraTape(source, factors, u);
    }

    /// <summary>Softmax attention over every head, keeping the probabilities for the backward pass.</summary>
    private double[] Attend(double[] qkv, int rows, out double[][] probabilities)
    {
        var width = _hidden;
        var heads = _heads;
        var size = width / heads;
        var stride = 3 * width;
        var scale = 1.0 / Math.Sqrt(size);

        var result = new double[rows * width];
        var weights = new double[heads][];

        Parallel.For(0, heads, head =>
        {
            var k = new double[rows * size];
            var v = new double[rows * size];
            for (var j = 0; j < rows; j++)
            {
                Array.Copy(qkv, j * stride + width + head * size, k, j * size, size);
                Array.Copy(qkv, j * stride + 2 * width + head * size, v, j * size, size);
            }

            var p = new double[rows * rows];
            for (var i = 0; i < rows; i++)
            {
                var query = qkv.AsSpan(i * stride + head * size, size);
                var row = p.AsSpan(i * rows, rows);

                var largest = double.NegativeInfinity;
                for (var j = 0; j < rows; j++)
                {
                    row[j] = Simd.Dot(query, k.AsSpan(j * size, size)) * scale;
                    if (row[j] > largest) largest = row[j];
                }

                var total = 0.0;
                for (var j = 0; j < rows; j++)
                {
                    row[j] = Math.Exp(row[j] - largest);
                    total += row[j];
                }

                var target = result.AsSpan(i * width + head * size, size);
                for (var j = 0; j < rows; j++)
                {
                    row[j] /= total;
                    Simd.Axpy(row[j], v.AsSpan(j * size, size), target);
                }
            }

            weights[head] = p;
        });

        probabilities = weights;
        return result;
    }

    private double[] Normalize(
        double[] input, int rows, double[] scale, double[] shift, out double[] hat, out double[] inverse)
    {
        var width = scale.Length;
        var result = new double[input.Length];
        hat = new double[input.Length];
        inverse = new double[rows];

        for (var r = 0; r < rows; r++)
        {
            var row = input.AsSpan(r * width, width);

            var mean = 0.0;
            foreach (var value in row) mean += value;
            mean /= width;

            var variance = 0.0;
            foreach (var value in row) variance += (value - mean) * (value - mean);
            variance /= width;

            var inv = 1.0 / Math.Sqrt(variance + _epsilon);
            inverse[r] = inv;

            for (var d = 0; d < width; d++)
            {
                var h = (row[d] - mean) * inv;
                hat[r * width + d] = h;
                result[r * width + d] = (row[d] - mean) * inv * scale[d] + shift[d];
            }
        }

        return result;
    }

    // ------------------------------------------------------------------ backward

    /// <summary>
    /// Propagates <paramref name="gradient"/> - the loss's gradient with respect to the output
    /// hidden states - back through the recorded pass, adding each adapter's gradients into
    /// <paramref name="gradients"/>.
    /// </summary>
    internal void Backward(double[] gradient, Tape tape, LoraGradients gradients)
    {
        var rows = tape.Rows;
        var width = _hidden;
        var dy = gradient;

        for (var index = _layers.Length - 1; index >= _lowest; index--)
        {
            var layer = _layers[index];
            var record = tape.Layers[index];
            var inner = layer.Intermediate.Outputs;
            var needInput = index > _lowest;

            // y = LN2(a + o)
            var dyPre = NormalizeBackward(dy, rows, layer.SecondScale, record.SecondHat, record.SecondInverse);
            var da = (double[])dyPre.Clone();

            // o = Output(g)
            var dg = layer.Output.ApplyTransposed(dyPre, rows);
            LoraBackward(layer, Projection.Output, record, dyPre, rows, width, 0, dg, gradients);

            // g = act(z)
            var z = record.PreActivation;
            for (var i = 0; i < dg.Length; i++) dg[i] *= _derivative(z[i]);

            // z = Intermediate(a)
            var fromInner = layer.Intermediate.ApplyTransposed(dg, rows);
            for (var i = 0; i < da.Length; i++) da[i] += fromInner[i];
            LoraBackward(layer, Projection.Intermediate, record, dg, rows, inner, 0, da, gradients);

            // a = LN1(x + AttentionOutput(context))
            var daPre = NormalizeBackward(da, rows, layer.FirstScale, record.FirstHat, record.FirstInverse);

            var dContext = layer.AttentionOutput.ApplyTransposed(daPre, rows);
            LoraBackward(layer, Projection.AttentionOutput, record, daPre, rows, width, 0, dContext, gradients);

            var dQkv = AttendBackward(dContext, record.QueryKeyValue, record.Probabilities, rows);

            double[]? dx = null;
            if (needInput)
            {
                dx = layer.QueryKeyValue.ApplyTransposed(dQkv, rows);
                for (var i = 0; i < dx.Length; i++) dx[i] += daPre[i];
            }

            for (var p = Projection.Query; p <= Projection.Value; p++)
            {
                LoraBackward(layer, p, record, dQkv, rows, 3 * width, (int)p * width, dx, gradients);
            }

            dy = dx!;
        }
    }

    /// <summary>
    /// The backward pass of <see cref="AddLora"/>: accumulates <c>dA</c> and <c>dB</c>, and adds the
    /// adapter's share of the input gradient when <paramref name="inputGradient"/> is not null.
    /// </summary>
    private static void LoraBackward(
        Layer layer, Projection projection, LayerTape tape, double[] outputGradient, int rows,
        int stride, int offset, double[]? inputGradient, LoraGradients gradients)
    {
        var adapter = layer.Adapters[(int)projection];
        if (adapter is null) return;

        var saved = tape.Lora[(int)projection]!;
        var inputs = adapter.Inputs;
        var outputs = adapter.Outputs;
        var rank = adapter.Config.Rank;
        var scaling = adapter.Config.Scaling;
        var a = adapter.A.AsSpan();
        var b = adapter.B.AsSpan();
        var (gradA, gradB) = gradients.For(adapter);

        var du = new double[rows * rank];

        for (var r = 0; r < rows; r++)
        {
            for (var o = 0; o < outputs; o++)
            {
                var g = scaling * outputGradient[r * stride + offset + o];
                if (g == 0) continue;

                for (var k = 0; k < rank; k++)
                {
                    gradB[o * rank + k] += g * saved.Down[r * rank + k];
                    du[r * rank + k] += g * b[o * rank + k];
                }
            }

            var input = saved.Input.AsSpan(r * inputs, inputs);
            for (var k = 0; k < rank; k++)
            {
                Simd.Axpy(du[r * rank + k], input, gradA.AsSpan(k * inputs, inputs));
            }

            if (inputGradient is null) continue;

            var target = inputGradient.AsSpan(r * inputs, inputs);
            if (saved.Factors is null)
            {
                for (var k = 0; k < rank; k++) Simd.Axpy(du[r * rank + k], a.Slice(k * inputs, inputs), target);
            }
            else
            {
                var row = new double[inputs];
                for (var k = 0; k < rank; k++) Simd.Axpy(du[r * rank + k], a.Slice(k * inputs, inputs), row);
                for (var i = 0; i < inputs; i++) target[i] += row[i] * saved.Factors[r * inputs + i];
            }
        }
    }

    /// <summary>The gradient of the stacked query/key/value projection from the context's.</summary>
    private double[] AttendBackward(double[] dContext, double[] qkv, double[][] probabilities, int rows)
    {
        var width = _hidden;
        var heads = _heads;
        var size = width / heads;
        var stride = 3 * width;
        var scale = 1.0 / Math.Sqrt(size);

        var result = new double[rows * stride];

        Parallel.For(0, heads, head =>
        {
            var p = probabilities[head];
            var dScores = new double[rows];

            for (var i = 0; i < rows; i++)
            {
                var dc = dContext.AsSpan(i * width + head * size, size);
                var row = p.AsSpan(i * rows, rows);

                // dV_j += P_ij dC_i, and dP_ij = dC_i . V_j
                var weighted = 0.0;
                for (var j = 0; j < rows; j++)
                {
                    Simd.Axpy(row[j], dc, result.AsSpan(j * stride + 2 * width + head * size, size));
                    dScores[j] = Simd.Dot(dc, qkv.AsSpan(j * stride + 2 * width + head * size, size));
                    weighted += row[j] * dScores[j];
                }

                // Through the softmax, then the scale: dS_ij = P_ij (dP_ij - sum_k P_ik dP_ik).
                var query = qkv.AsSpan(i * stride + head * size, size);
                var dQuery = result.AsSpan(i * stride + head * size, size);

                for (var j = 0; j < rows; j++)
                {
                    var ds = row[j] * (dScores[j] - weighted) * scale;
                    if (ds == 0) continue;

                    Simd.Axpy(ds, qkv.AsSpan(j * stride + width + head * size, size), dQuery);
                    Simd.Axpy(ds, query, result.AsSpan(j * stride + width + head * size, size));
                }
            }
        });

        return result;
    }

    /// <summary>
    /// <c>dx = inverse (g - mean(g) - hat mean(g hat))</c> per row, with <c>g = gradient * scale</c>.
    /// </summary>
    private static double[] NormalizeBackward(double[] gradient, int rows, double[] scale, double[] hat, double[] inverse)
    {
        var width = scale.Length;
        var result = new double[gradient.Length];

        for (var r = 0; r < rows; r++)
        {
            var sum = 0.0;
            var dotHat = 0.0;

            for (var d = 0; d < width; d++)
            {
                var g = gradient[r * width + d] * scale[d];
                sum += g;
                dotHat += g * hat[r * width + d];
            }

            var mean = sum / width;
            var meanHat = dotHat / width;

            for (var d = 0; d < width; d++)
            {
                var g = gradient[r * width + d] * scale[d];
                result[r * width + d] = inverse[r] * (g - mean - hat[r * width + d] * meanHat);
            }
        }

        return result;
    }

    // ------------------------------------------------------------------ state

    private sealed record Layer(
        Linear QueryKeyValue,
        Linear AttentionOutput,
        Linear Intermediate,
        Linear Output,
        double[] FirstScale,
        double[] FirstShift,
        double[] SecondScale,
        double[] SecondShift,
        LoraAdapter?[] Adapters);

    /// <summary>What one forward pass leaves behind for its backward pass.</summary>
    internal sealed class Tape
    {
        internal int Rows { get; set; }

        internal List<LayerTape> Layers { get; } = [];
    }

    internal sealed class LayerTape
    {
        internal double[] QueryKeyValue { get; set; } = [];

        internal double[][] Probabilities { get; set; } = [];

        internal double[] FirstHat { get; set; } = [];

        internal double[] FirstInverse { get; set; } = [];

        internal double[] PreActivation { get; set; } = [];

        internal double[] SecondHat { get; set; } = [];

        internal double[] SecondInverse { get; set; } = [];

        internal LoraTape?[] Lora { get; } = new LoraTape?[6];
    }

    /// <summary>An adapter's input after dropout, the dropout factors, and <c>A x</c>.</summary>
    internal sealed record LoraTape(double[] Input, double[]? Factors, double[] Down);
}

/// <summary>Gradient accumulators for a set of adapters, shaped like their A and B.</summary>
internal sealed class LoraGradients
{
    private readonly Dictionary<LoraAdapter, (double[] A, double[] B)> _gradients =
        new(ReferenceEqualityComparer.Instance);

    internal (double[] A, double[] B) For(LoraAdapter adapter)
    {
        if (!_gradients.TryGetValue(adapter, out var pair))
        {
            pair = (new double[adapter.A.Size], new double[adapter.B.Size]);
            _gradients[adapter] = pair;
        }

        return pair;
    }

    internal void Clear()
    {
        foreach (var (a, b) in _gradients.Values)
        {
            Array.Clear(a);
            Array.Clear(b);
        }
    }
}
