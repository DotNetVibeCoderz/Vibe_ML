using System.Numerics;
using System.Runtime.CompilerServices;
using Gravicode.HFNet.GraviHub;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Transformers;

namespace Gravicode.HFNet.GraviTransformers;

// The inner loops every encoder forward pass spends its time in. Shared by the text encoder and the
// vision encoder, which differ in the order of their norms and in nothing that happens in here.
//
// Two decisions run through all of it.
//
// Weights are held as float32 and activations as double. Every checkpoint this loads stores its
// weights in F32, BF16 or F16, all of which float32 holds exactly, so keeping them as double
// bought no precision and doubled the bytes every forward pass has to stream through the cache.
// On a short text input that stream IS the cost. Activations and every sum stay double.
//
// The hot methods are marked AggressiveOptimization. A loop-heavy method called a handful of
// times otherwise runs as unoptimised tier-0 code for its first several calls, and a linear layer
// measured twelve times slower there than once the JIT had caught up. Twelve blocks and one image
// is exactly "a handful of times".

/// <summary>Vectorised primitives.</summary>
internal static class Simd
{
    /// <summary>Dot product of two equal-length spans.</summary>
    /// <remarks>
    /// Both operands have to be contiguous: a strided operand cannot be loaded into a vector
    /// register at all, so the layout decisions elsewhere and this loop are one decision.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var width = Vector<double>.Count;
        var lanes = Vector<double>.Zero;
        var i = 0;

        for (; i <= a.Length - width; i += width)
        {
            lanes += new Vector<double>(a[i..]) * new Vector<double>(b[i..]);
        }

        var sum = Vector.Sum(lanes);
        for (; i < a.Length; i++) sum += a[i] * b[i];

        return sum;
    }

    /// <summary><c>y += alpha * x</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Axpy(double alpha, ReadOnlySpan<double> x, Span<double> y)
    {
        var width = Vector<double>.Count;
        var scale = new Vector<double>(alpha);
        var i = 0;

        for (; i <= x.Length - width; i += width)
        {
            (new Vector<double>(y[i..]) + scale * new Vector<double>(x[i..])).CopyTo(y[i..]);
        }

        for (; i < x.Length; i++) y[i] += alpha * x[i];
    }
}

/// <summary>The feed-forward activations a config's <c>hidden_act</c> can name.</summary>
internal static class Activation
{
    /// <summary>The activation <paramref name="name"/> refers to.</summary>
    /// <remarks>
    /// Transformers reads <c>gelu</c> as the exact GELU, <c>0.5x(1 + erf(x/sqrt 2))</c>. The
    /// foundation's <c>Activations.Gelu</c> is the tanh approximation, and running it in place of
    /// the exact one left <c>google/vit-base-patch16-224</c> agreeing with torch only to the third
    /// decimal of a probability given the very same pixels. An unknown name is refused rather
    /// than guessed, for the same reason.
    /// </remarks>
    internal static Func<double, double> For(string name) => name.ToLowerInvariant() switch
    {
        "gelu" or "gelu_python" => ExactGelu,
        "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => Activations.Gelu,
        "relu" => Activations.Relu,
        _ => throw new NotSupportedException(
            $"The config asks for the activation '{name}'. This encoder runs gelu, gelu_new, "
            + "gelu_pytorch_tanh, gelu_fast and relu; export the model to ONNX and use GraviOptimum "
            + "for anything else."),
    };

    /// <summary>The derivative of the activation <paramref name="name"/> refers to.</summary>
    /// <remarks>
    /// Paired with <see cref="For"/> name for name, so a backward pass differentiates the function
    /// the forward pass actually ran rather than a neighbour of it. The tanh GELU's derivative is
    /// written against the foundation's own formula, which is what <c>gelu_new</c> runs.
    /// </remarks>
    internal static Func<double, double> DerivativeFor(string name) => name.ToLowerInvariant() switch
    {
        "gelu" or "gelu_python" => ExactGeluDerivative,
        "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => TanhGeluDerivative,
        "relu" => x => x > 0 ? 1.0 : 0.0,
        _ => throw new NotSupportedException(
            $"The config asks for the activation '{name}', which has no derivative here; only gelu, "
            + "gelu_new, gelu_pytorch_tanh, gelu_fast and relu can be trained through."),
    };

    internal static double ExactGelu(double x) => 0.5 * x * (1 + Erf(x * InverseSqrtTwo));

    /// <summary><c>Phi(x) + x phi(x)</c>: the normal CDF plus x times the normal density.</summary>
    internal static double ExactGeluDerivative(double x)
        => 0.5 * (1 + Erf(x * InverseSqrtTwo)) + x * InverseSqrtTwoPi * Math.Exp(-0.5 * x * x);

    /// <summary>The derivative of <c>0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))</c>.</summary>
    internal static double TanhGeluDerivative(double x)
    {
        const double c = 0.79788456080286535588;   // sqrt(2/pi)
        const double k = 0.044715;

        var t = Math.Tanh(c * (x + k * x * x * x));
        return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * k * x * x);
    }

    private const double InverseSqrtTwoPi = 0.39894228040143267794;

    private const double InverseSqrtTwo = 0.70710678118654752440;

    // erf is tabulated at every 1/64 on [0, 6] and continued to the exact point by a Taylor series.
    // Beyond 6, erf is 1 to within 2e-17, below the last bit of a double.
    private const int Steps = 64;
    private const double Limit = 6.0;
    private const int Points = (int)(Limit * Steps) + 1;
    private static readonly double[] SlopeAt = BuildSlopes();
    private static readonly double[] ErfAt = BuildTable();

    private static double[] BuildSlopes()
    {
        var slopes = new double[Points];
        for (var k = 0; k < Points; k++)
        {
            var x = (double)k / Steps;
            slopes[k] = 2 / Math.Sqrt(Math.PI) * Math.Exp(-x * x);
        }

        return slopes;
    }

    /// <summary>erf at every grid point, stepped out from erf(0) = 0.</summary>
    /// <remarks>
    /// Stepped rather than evaluated with <c>MathUtil.Erf</c>, whose Maclaurin series loses about
    /// three digits to cancellation near 3 - measured 4.9e-14 off against a correctly rounded erf on
    /// this grid. Stepping 1/64 at a time with sixteen Taylor terms stays within 4.4e-16 across the
    /// whole table.
    /// </remarks>
    private static double[] BuildTable()
    {
        var table = new double[Points];
        for (var k = 1; k < Points; k++) table[k] = table[k - 1] + SlopeAt[k - 1] * Taylor((double)(k - 1) / Steps, 1.0 / Steps, 16);

        return table;
    }

    /// <summary>
    /// <c>sum over n of h^n/n! * (-1)^(n-1) H(n-1, origin)</c> for n = 1..terms: erf's Taylor
    /// series about <paramref name="origin"/>, less the slope factor.
    /// </summary>
    private static double Taylor(double origin, double h, int terms)
    {
        double previous = 0;        // H(n-2)
        double current = 1;         // H(n-1), starting at H(0)
        var power = h;              // (-1)^(n-1) h^n / n!
        var sum = h;

        for (var n = 1; n < terms; n++)
        {
            // H(n) = 2x H(n-1) - 2(n-1) H(n-2)
            var next = 2 * origin * current - 2 * (n - 1) * previous;
            previous = current;
            current = next;

            power *= -h / (n + 1);
            sum += current * power;
        }

        return sum;
    }

    /// <summary>The error function, to within a few units in the last place.</summary>
    /// <remarks>
    /// <para>
    /// The foundation's <c>MathUtil.Erf</c> sums a Maclaurin series of up to fifty terms, and GELU
    /// calls erf once per feed-forward activation - seven million times per ViT image. It was 40%
    /// of the time spent in a BERT feed-forward layer. This is three times faster and more
    /// accurate.
    /// </para>
    /// <para>
    /// It reads erf at the nearest multiple of 1/64 from a table, then steps the remaining distance
    /// h (at most 1/128) with nine Taylor terms. The n-th derivative of erf is
    /// <c>(-1)^(n-1) H(n-1, x) * 2/sqrt(pi) * exp(-x^2)</c>, with H the physicists' Hermite
    /// polynomials, so every term comes from one three-term recurrence. The first term left out is
    /// below h^10/10! - under 1e-25 before the Gaussian factor shrinks it further - so the table is
    /// the only error that remains.
    /// </para>
    /// </remarks>
    internal static double Erf(double x)
    {
        var sign = 1.0;
        if (x < 0)
        {
            x = -x;
            sign = -1.0;
        }

        if (x >= Limit) return sign;
        if (double.IsNaN(x)) return x;

        var k = (int)(x * Steps + 0.5);
        var origin = (double)k / Steps;

        return sign * (ErfAt[k] + SlopeAt[k] * Taylor(origin, x - origin, 9));
    }
}

/// <summary>
/// A fully connected layer: float32 weights packed for <see cref="Gemm"/>, double activations.
/// </summary>
/// <remarks>
/// Built from the <c>(outputs, inputs)</c> order Hugging Face stores and kept in
/// <see cref="Gemm"/>'s panels of twelve outputs, which is what lets the product run as a
/// register-blocked kernel rather than one dot product at a time.
/// </remarks>
internal sealed class Linear
{
    private readonly float[] _packed;    // Gemm panels of W, which is [outputs, inputs]
    private readonly double[] _bias;     // [outputs]

    private Linear(float[] weights, double[] bias, int inputs, int outputs)
    {
        _packed = Gemm.Pack(weights, inputs, outputs);
        _bias = bias;
        Inputs = inputs;
        Outputs = outputs;
    }

    internal int Inputs { get; }

    internal int Outputs { get; }

    /// <summary>Reads <c>{name}.weight</c> and <c>{name}.bias</c> as the checkpoint stores them.</summary>
    internal static Linear Load(WeightStore weights, string name, int inputs, int outputs)
    {
        var matrix = weights.Read($"{name}.weight");
        if (matrix.Size != (long)inputs * outputs)
        {
            throw new InvalidDataException(
                $"'{name}.weight' holds {matrix.Size} values but the layer is {inputs}x{outputs}.");
        }

        var flat = new float[inputs * outputs];
        for (var i = 0; i < flat.Length; i++) flat[i] = (float)matrix.At(i);

        var biasTensor = weights.Read($"{name}.bias");
        var bias = new double[outputs];
        for (var i = 0; i < outputs; i++) bias[i] = biasTensor.At(i);

        return new Linear(flat, bias, inputs, outputs);
    }

    /// <summary>Copies a matrix stored <c>(outputs, inputs)</c>, with an optional bias.</summary>
    internal static Linear From(NdArray weight, NdArray? bias)
    {
        var outputs = weight.Shape[0];
        var inputs = weight.Shape[1];

        var flat = new float[inputs * outputs];
        for (var i = 0; i < flat.Length; i++) flat[i] = (float)weight.At(i);

        var shift = new double[outputs];
        if (bias is not null)
        {
            for (var o = 0; o < outputs; o++) shift[o] = bias.At(o);
        }

        return new Linear(flat, shift, inputs, outputs);
    }

    /// <summary>Copies one of the foundation's layers, which hold <c>(inputs, outputs)</c>.</summary>
    internal static Linear From(DenseLayer layer)
    {
        var inputs = layer.Weights.Shape[0];
        var outputs = layer.Weights.Shape[1];
        var source = layer.Weights.ToArray();

        var flat = new float[inputs * outputs];
        for (var i = 0; i < inputs; i++)
        {
            for (var o = 0; o < outputs; o++) flat[o * inputs + i] = (float)source[i * outputs + o];
        }

        return new Linear(flat, layer.Bias.ToArray(), inputs, outputs);
    }

    /// <summary>Stacks layers that share an input into one, so the input is read once.</summary>
    /// <remarks>Used for query, key and value, whose outputs then sit side by side in each row.</remarks>
    internal static Linear Stack(params Linear[] layers)
    {
        var inputs = layers[0].Inputs;
        if (layers.Any(l => l.Inputs != inputs))
        {
            throw new ArgumentException("Only layers with the same input width can be stacked.", nameof(layers));
        }

        return new Linear(
            [.. layers.SelectMany(l => l.RowMajor())],
            [.. layers.SelectMany(l => l._bias)],
            inputs,
            layers.Sum(l => l.Outputs));
    }

    /// <summary>The weights back in <c>(outputs, inputs)</c> order.</summary>
    private float[] RowMajor()
    {
        var weights = new float[Outputs * Inputs];
        for (var o = 0; o < Outputs; o++)
        {
            for (var i = 0; i < Inputs; i++) weights[o * Inputs + i] = Gemm.At(_packed, Inputs, o, i);
        }

        return weights;
    }

    /// <summary>Applies the layer to <paramref name="rows"/> row-major rows of <paramref name="input"/>.</summary>
    /// <remarks>
    /// The product runs in <see cref="Gemm"/>; the bias and the activation are applied here, per
    /// element, afterwards.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal double[] Apply(double[] input, int rows, Func<double, double>? activation = null)
    {
        var outputs = Outputs;
        var result = Gemm.Multiply(input, rows, _packed, Inputs, outputs);

        for (var r = 0; r < rows; r++)
        {
            var row = result.AsSpan(r * outputs, outputs);
            for (var o = 0; o < outputs; o++)
            {
                var value = row[o] + _bias[o];
                row[o] = activation is null ? value : activation(value);
            }
        }

        return result;
    }

    /// <summary>
    /// The gradient with respect to the input: <c>dx = dy W</c>, for <paramref name="rows"/> rows
    /// of <paramref name="gradient"/> shaped <c>[rows, outputs]</c>.
    /// </summary>
    /// <remarks>
    /// The backward pass of <see cref="Apply"/>, from the same packed weights, with no transposed
    /// copy kept - for bert-base that would be another 340 MB.
    /// </remarks>
    internal double[] ApplyTransposed(double[] gradient, int rows)
        => Gemm.MultiplyTransposed(gradient, rows, _packed, Inputs, Outputs);
}

/// <summary>A layer norm over each row.</summary>
internal sealed class Norm
{
    private readonly double[] _scale;
    private readonly double[] _shift;
    private readonly double _epsilon;

    internal Norm(double[] scale, double[] shift, double epsilon)
    {
        _scale = scale;
        _shift = shift;
        _epsilon = epsilon;
    }

    /// <summary>Reads <c>{prefix}.weight</c> and <c>{prefix}.bias</c>.</summary>
    internal static Norm Load(WeightStore weights, string prefix, int size, double epsilon)
    {
        var scale = weights.Read($"{prefix}.weight");
        var shift = weights.Read($"{prefix}.bias");

        var gamma = new double[size];
        var beta = new double[size];
        for (var i = 0; i < size; i++)
        {
            gamma[i] = scale.At(i);
            beta[i] = shift.At(i);
        }

        return new Norm(gamma, beta, epsilon);
    }

    /// <summary>Copies one of the foundation's norms, with the epsilon the config names.</summary>
    /// <remarks>
    /// The epsilon is taken from the config rather than from the foundation's norm, which does not
    /// expose it and defaults to 1e-12 - right for BERT, wrong for RoBERTa's 1e-5.
    /// </remarks>
    internal static Norm From(LayerNorm norm, double epsilon)
        => new(norm.Gamma.ToArray(), norm.Beta.ToArray(), epsilon);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal double[] Apply(double[] input, int rows)
    {
        var width = _scale.Length;
        var result = new double[input.Length];

        for (var r = 0; r < rows; r++)
        {
            var row = input.AsSpan(r * width, width);

            var mean = 0.0;
            foreach (var value in row) mean += value;
            mean /= width;

            var variance = 0.0;
            foreach (var value in row) variance += (value - mean) * (value - mean);
            variance /= width;

            var inverse = 1.0 / Math.Sqrt(variance + _epsilon);
            var target = result.AsSpan(r * width, width);

            for (var d = 0; d < width; d++) target[d] = (row[d] - mean) * inverse * _scale[d] + _shift[d];
        }

        return result;
    }
}

/// <summary>Scaled dot-product attention over every head at once.</summary>
internal static class Attention
{
    /// <summary>
    /// Attends over a stacked query/key/value projection, rows shaped <c>[q | k | v]</c>.
    /// </summary>
    /// <param name="projected">Row-major <c>[rows, 3 * hidden]</c>.</param>
    /// <param name="rows">Sequence length.</param>
    /// <param name="hidden">Model width.</param>
    /// <param name="heads">Attention heads.</param>
    /// <param name="mask">1 for a real position, 0 for padding; <c>null</c> for none.</param>
    /// <returns>Row-major <c>[rows, hidden]</c>, heads concatenated.</returns>
    /// <remarks>
    /// Each head's keys and values are first copied out into their own contiguous block, so every
    /// score is one vectorised dot product and every output row one vectorised sum. The work is
    /// then split by head and by block of query rows: heads alone would leave cores idle on a
    /// twelve-head model, and rows alone would repeat the copy.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static double[] Apply(double[] projected, int rows, int hidden, int heads, int[]? mask)
    {
        var size = hidden / heads;
        var stride = 3 * hidden;
        var scale = 1.0 / Math.Sqrt(size);

        var keys = new double[heads][];
        var values = new double[heads][];

        Parallel.For(0, heads, head =>
        {
            var k = new double[rows * size];
            var v = new double[rows * size];

            for (var j = 0; j < rows; j++)
            {
                Array.Copy(projected, j * stride + hidden + head * size, k, j * size, size);
                Array.Copy(projected, j * stride + 2 * hidden + head * size, v, j * size, size);
            }

            keys[head] = k;
            values[head] = v;
        });

        const int Block = 16;
        var blocks = (rows + Block - 1) / Block;
        var result = new double[rows * hidden];

        Parallel.For(0, heads * blocks, item =>
        {
            var head = item / blocks;
            var first = (item % blocks) * Block;
            var last = Math.Min(first + Block, rows);

            var k = keys[head];
            var v = values[head];
            var weights = new double[rows];

            for (var i = first; i < last; i++)
            {
                var query = projected.AsSpan(i * stride + head * size, size);

                var largest = double.NegativeInfinity;
                for (var j = 0; j < rows; j++)
                {
                    // The same -1e9 the reference uses, so a fully masked row behaves the same too.
                    weights[j] = mask is not null && mask[j] == 0
                        ? -1e9
                        : Simd.Dot(query, k.AsSpan(j * size, size)) * scale;

                    if (weights[j] > largest) largest = weights[j];
                }

                var total = 0.0;
                for (var j = 0; j < rows; j++)
                {
                    weights[j] = Math.Exp(weights[j] - largest);
                    total += weights[j];
                }

                var target = result.AsSpan(i * hidden + head * size, size);
                for (var j = 0; j < rows; j++) Simd.Axpy(weights[j] / total, v.AsSpan(j * size, size), target);
            }
        });

        return result;
    }
}

/// <summary>One transformer encoder block.</summary>
/// <remarks>
/// <para>
/// BERT and ViT use exactly the same parameters in a different order, and that order is the only
/// thing <see cref="NormOrder"/> changes. Every parameter has the same shape either way, so a
/// checkpoint run with the wrong one loads in silence and returns confident nonsense; the order is
/// therefore a required argument with no default, never inferred.
/// </para>
/// <para>
/// Post-norm (BERT): <c>a = LN1(x + Attn(x))</c>, <c>y = LN2(a + FFN(a))</c>.
/// Pre-norm (ViT): <c>a = x + Attn(LN1(x))</c>, <c>y = a + FFN(LN2(a))</c>.
/// </para>
/// </remarks>
internal sealed class EncoderBlock
{
    private readonly NormOrder _order;
    private readonly int _heads;
    private readonly Linear _queryKeyValue;
    private readonly Linear _attentionOutput;
    private readonly Norm _first;
    private readonly Linear _intermediate;
    private readonly Linear _output;
    private readonly Norm _second;
    private readonly Func<double, double> _activation;

    internal EncoderBlock(
        NormOrder order, int heads,
        Linear query, Linear key, Linear value, Linear attentionOutput, Norm first,
        Linear intermediate, Linear output, Norm second, Func<double, double> activation)
    {
        _order = order;
        _heads = heads;
        _queryKeyValue = Linear.Stack(query, key, value);
        _attentionOutput = attentionOutput;
        _first = first;
        _intermediate = intermediate;
        _output = output;
        _second = second;
        _activation = activation;
    }

    internal int Hidden => _attentionOutput.Outputs;

    internal double[] Forward(double[] hidden, int rows, int[]? mask)
    {
        if (_order == NormOrder.Pre)
        {
            var attended = Add(hidden, Attend(_first.Apply(hidden, rows), rows, mask));
            return Add(attended, FeedForward(_second.Apply(attended, rows), rows));
        }

        var normed = _first.Apply(Add(hidden, Attend(hidden, rows, mask)), rows);
        return _second.Apply(Add(normed, FeedForward(normed, rows)), rows);
    }

    private double[] Attend(double[] input, int rows, int[]? mask)
    {
        var projected = _queryKeyValue.Apply(input, rows);
        return _attentionOutput.Apply(Attention.Apply(projected, rows, Hidden, _heads, mask), rows);
    }

    private double[] FeedForward(double[] input, int rows)
        => _output.Apply(_intermediate.Apply(input, rows, _activation), rows);

    private static double[] Add(double[] a, double[] b)
    {
        var result = new double[a.Length];
        for (var i = 0; i < a.Length; i++) result[i] = a[i] + b[i];

        return result;
    }
}

/// <summary>Where an encoder block normalises.</summary>
internal enum NormOrder
{
    /// <summary>Before each sublayer, residual after - ViT, DeiT.</summary>
    Pre,

    /// <summary>After the residual - BERT and its descendants.</summary>
    Post,
}
