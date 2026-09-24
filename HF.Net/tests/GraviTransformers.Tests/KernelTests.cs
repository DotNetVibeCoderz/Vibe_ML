using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Transformers;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>
/// The shared inference kernels, each pinned to something written independently of it: a naive
/// loop inside the test, the foundation's own layers, or a correctly rounded reference value.
/// </summary>
/// <remarks>
/// Every random value is rounded to float32 first. The kernels hold weights as float32 - exactly,
/// for any real checkpoint - so drawing test values that float32 cannot hold would measure that
/// rounding rather than the kernel.
/// </remarks>
public sealed class KernelTests
{
    private static double[] Values(int count, int seed, double scale = 1.0)
    {
        var random = new GraviRandom(seed);
        return [.. Enumerable.Range(0, count).Select(_ => (double)(float)(random.Normal(0, 1) * scale))];
    }

    private static NdArray Matrix(int rows, int columns, int seed, double scale = 1.0)
        => new(Values(rows * columns, seed, scale), [rows, columns]);

    private static void Close(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            var error = Math.Abs(expected[i] - actual[i]);
            Assert.True(error <= tolerance, $"{what}[{i}]: {actual[i]} against {expected[i]} (off by {error:E2})");
        }
    }

    // ------------------------------------------------------------------ erf

    [Theory]
    // x, math.erf(x) from CPython, which is correctly rounded to within an ulp.
    [InlineData(0.001, 0.0011283787909692365)]
    [InlineData(0.3, 0.3286267594591274)]
    [InlineData(1.0, 0.8427007929497149)]
    [InlineData(1.7, 0.9837904585907745)]
    [InlineData(2.5, 0.999593047982555)]
    [InlineData(2.95, 0.9999697969579359)]
    [InlineData(3.3, 0.9999969422902035)]
    [InlineData(4.5, 0.9999999998033839)]
    [InlineData(5.9, 1.0)]
    [InlineData(6.5, 1.0)]
    public void Erf_agrees_with_a_correctly_rounded_reference(double x, double expected)
    {
        // 2.95 is the point to watch: the foundation's Maclaurin series is off by about 5e-14 near
        // there, and a table built from it inherited that.
        Assert.True(Math.Abs(Activation.Erf(x) - expected) < 5e-16, $"erf({x}) = {Activation.Erf(x):R}");
        Assert.True(Math.Abs(Activation.Erf(-x) + expected) < 5e-16, $"erf(-{x}) = {Activation.Erf(-x):R}");
    }

    [Fact]
    public void Erf_is_smooth_across_the_table_s_grid_points()
    {
        // A table lookup that picked the wrong neighbour, or stepped the wrong way, would show as
        // a jump exactly at a multiple of 1/128. Either side of every boundary should agree with
        // the slope, 2/sqrt(pi) exp(-x^2), to second order.
        for (var k = 1; k < 6 * 128; k += 2)
        {
            var boundary = k / 128.0;
            const double Epsilon = 1e-7;

            var slope = (Activation.Erf(boundary + Epsilon) - Activation.Erf(boundary - Epsilon)) / (2 * Epsilon);
            var expected = 2 / Math.Sqrt(Math.PI) * Math.Exp(-boundary * boundary);

            Assert.True(Math.Abs(slope - expected) < 1e-7, $"slope at {boundary}: {slope} against {expected}");
        }
    }

    // ------------------------------------------------------------------ linear

    [Theory]
    [InlineData(1, 7, 1)]      // one row, an input width no vector covers, one output
    [InlineData(3, 13, 3)]     // a partial block of rows and an odd output count
    [InlineData(5, 16, 65)]    // one row past a block, one output past a tile
    [InlineData(33, 40, 130)]  // past a row tile as well
    public void Linear_agrees_with_a_naive_product_at_awkward_shapes(int rows, int inputs, int outputs)
    {
        // Every one of these shapes lands on a remainder somewhere - the vector tail, the last
        // pair of outputs, the last partial block of four rows. Those are where a kernel like this
        // goes wrong, and the error is small enough to pass for rounding in an end-to-end test.
        var weight = Matrix(outputs, inputs, 1);
        var bias = Values(outputs, 2);
        var input = Values(rows * inputs, 3);

        var layer = Linear.From(weight, new NdArray(bias, [outputs]));
        var actual = layer.Apply(input, rows, Math.Tanh);

        var expected = new double[rows * outputs];
        for (var r = 0; r < rows; r++)
        {
            for (var o = 0; o < outputs; o++)
            {
                var sum = bias[o];
                for (var i = 0; i < inputs; i++) sum += input[r * inputs + i] * weight[o, i];
                expected[r * outputs + o] = Math.Tanh(sum);
            }
        }

        Close(expected, actual, 1e-12, "y");
    }

    [Fact]
    public void A_foundation_layer_is_copied_with_its_transpose_undone()
    {
        // The foundation stores (inputs, outputs); the kernels want (outputs, inputs). Pinned to
        // the foundation's own forward pass, which computes the same product another way.
        var layer = new DenseLayer(12, 20, new GraviRandom(5));
        var weights = Values(12 * 20, 6);
        var bias = Values(20, 7);
        for (var i = 0; i < weights.Length; i++) layer.Weights.SetAt(i, weights[i]);
        for (var i = 0; i < bias.Length; i++) layer.Bias.SetAt(i, bias[i]);

        var input = Matrix(6, 12, 8);

        Close(layer.Forward(input).ToArray(), Linear.From(layer).Apply(input.ToArray(), 6), 1e-12, "y");
    }

    [Fact]
    public void Stacked_layers_put_their_outputs_side_by_side()
    {
        var first = Linear.From(Matrix(5, 9, 10), new NdArray(Values(5, 11), [5]));
        var second = Linear.From(Matrix(3, 9, 12), new NdArray(Values(3, 13), [3]));
        var input = Values(4 * 9, 14);

        var a = first.Apply(input, 4);
        var b = second.Apply(input, 4);
        var stacked = Linear.Stack(first, second).Apply(input, 4);

        var expected = new double[4 * 8];
        for (var r = 0; r < 4; r++)
        {
            Array.Copy(a, r * 5, expected, r * 8, 5);
            Array.Copy(b, r * 3, expected, r * 8 + 5, 3);
        }

        Close(expected, stacked, 0, "stacked");
    }

    // ------------------------------------------------------------------ attention

    [Theory]
    [InlineData(1, 4, 1)]
    [InlineData(5, 6, 3)]
    [InlineData(19, 8, 2)]     // more rows than one block of queries
    public void Attention_agrees_with_the_textbook_formula(int rows, int hidden, int heads)
    {
        var projected = Values(rows * 3 * hidden, 20 + rows);
        var mask = Enumerable.Range(0, rows).Select(i => i == rows - 1 && rows > 1 ? 0 : 1).ToArray();

        var actual = Attention.Apply(projected, rows, hidden, heads, mask);

        // softmax(q k^T / sqrt(d)) v per head, with the masked key sent to -1e9 - written the slow
        // way, on the [q | k | v] row layout the kernel reads.
        var size = hidden / heads;
        var expected = new double[rows * hidden];

        for (var head = 0; head < heads; head++)
        {
            for (var i = 0; i < rows; i++)
            {
                var scores = new double[rows];
                for (var j = 0; j < rows; j++)
                {
                    var dot = 0.0;
                    for (var d = 0; d < size; d++)
                    {
                        dot += projected[i * 3 * hidden + head * size + d]
                            * projected[j * 3 * hidden + hidden + head * size + d];
                    }

                    scores[j] = mask[j] == 0 ? -1e9 : dot / Math.Sqrt(size);
                }

                var weights = MathUtil.Softmax(scores);

                for (var d = 0; d < size; d++)
                {
                    var sum = 0.0;
                    for (var j = 0; j < rows; j++) sum += weights[j] * projected[j * 3 * hidden + 2 * hidden + head * size + d];
                    expected[i * hidden + head * size + d] = sum;
                }
            }
        }

        Close(expected, actual, 1e-12, "attended");
    }

    // ------------------------------------------------------------------ the block

    [Fact]
    public void A_post_norm_block_matches_the_foundation_s_BERT_layer()
    {
        // The foundation's TransformerEncoderLayer is an independent implementation of the same
        // post-norm block, with the tanh GELU - so with gelu_new and its fixed 1e-12 epsilon the
        // two must agree. Getting the norm order wrong, or the q/k/v stacking, fails this by
        // whole units rather than by rounding.
        const int Hidden = 12, Heads = 3, Intermediate = 20, Rows = 7;
        var config = new TransformerConfig(1, Hidden, 1, Heads, Intermediate, Rows);
        var layer = new TransformerEncoderLayer(config, new GraviRandom(30));

        var seed = 40;
        foreach (var dense in (DenseLayer[])[layer.Attention.Query, layer.Attention.Key, layer.Attention.Value,
                     layer.Attention.Output, layer.Intermediate, layer.OutputProjection])
        {
            var w = Values((int)dense.Weights.Size, seed++, 0.3);
            var b = Values((int)dense.Bias.Size, seed++, 0.1);
            for (var i = 0; i < w.Length; i++) dense.Weights.SetAt(i, w[i]);
            for (var i = 0; i < b.Length; i++) dense.Bias.SetAt(i, b[i]);
        }

        foreach (var norm in (LayerNorm[])[layer.AttentionNorm, layer.OutputNorm])
        {
            var g = Values(Hidden, seed++, 0.2);
            var b = Values(Hidden, seed++, 0.2);
            for (var i = 0; i < Hidden; i++)
            {
                norm.Gamma.SetAt(i, 1 + g[i]);
                norm.Beta.SetAt(i, b[i]);
            }
        }

        var block = new EncoderBlock(
            NormOrder.Post, Heads,
            Linear.From(layer.Attention.Query), Linear.From(layer.Attention.Key),
            Linear.From(layer.Attention.Value), Linear.From(layer.Attention.Output),
            Norm.From(layer.AttentionNorm, 1e-12),
            Linear.From(layer.Intermediate), Linear.From(layer.OutputProjection),
            Norm.From(layer.OutputNorm, 1e-12),
            Activation.For("gelu_new"));

        var input = Matrix(Rows, Hidden, 50);
        var mask = new[] { 1, 1, 1, 1, 1, 0, 0 };

        Close(
            layer.Forward(input, mask).ToArray(),
            block.Forward(input.ToArray(), Rows, mask),
            1e-11,
            "block");
    }

    // ------------------------------------------------------------------ backward

    [Theory]
    [InlineData(1, 7, 1)]
    [InlineData(3, 13, 3)]
    [InlineData(5, 130, 65)]   // past an input tile, with a partial vector at its end
    [InlineData(9, 256, 40)]   // two whole input tiles and a partial block of rows
    public void Transposed_linear_agrees_with_a_naive_product(int rows, int inputs, int outputs)
    {
        // dx = dy W: the input gradient of a layer whose weights are frozen. A tile boundary that
        // drops a column or a block that reuses the wrong row is invisible in training - the loss
        // still falls - so it is pinned here against the loop it replaces.
        var weight = Matrix(outputs, inputs, 21);
        var gradient = Values(rows * outputs, 22);

        var actual = Linear.From(weight, null).ApplyTransposed(gradient, rows);

        var expected = new double[rows * inputs];
        for (var r = 0; r < rows; r++)
        {
            for (var i = 0; i < inputs; i++)
            {
                var sum = 0.0;
                for (var o = 0; o < outputs; o++) sum += gradient[r * outputs + o] * weight[o, i];
                expected[r * inputs + i] = sum;
            }
        }

        Close(expected, actual, 1e-12, "dx");
    }

    [Theory]
    [InlineData("gelu")]
    [InlineData("gelu_new")]
    [InlineData("relu")]
    public void Each_activation_s_derivative_is_the_slope_of_the_function_itself(string name)
    {
        // Pinned to the forward function that inference runs, not to a formula for it: a derivative
        // of the exact GELU paired with the tanh GELU would train, and train the wrong model.
        var function = Activation.For(name);
        var derivative = Activation.DerivativeFor(name);
        const double H = 1e-6;

        for (var x = -5.0; x <= 5.0; x += 0.173)
        {
            var slope = (function(x + H) - function(x - H)) / (2 * H);
            Assert.True(Math.Abs(derivative(x) - slope) < 1e-8, $"{name}'({x}) = {derivative(x)} against {slope}");
        }
    }

    [Fact]
    public void An_activation_with_no_derivative_is_refused_by_name()
    {
        var error = Assert.Throws<NotSupportedException>(() => Activation.DerivativeFor("swish"));
        Assert.Contains("swish", error.Message);
    }
}
