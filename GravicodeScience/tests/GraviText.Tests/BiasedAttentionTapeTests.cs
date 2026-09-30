using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;
using Gravicode.Science.GraviText.Transformers;
using Xunit;

namespace Gravicode.Science.Tests.GraviText;

/// <summary>
/// The tape's attention with its projection biases, and the exact GELU - what a pretrained BERT
/// actually computes.
/// </summary>
/// <remarks>
/// Before these, the tape's attention had no biases, so the only way to compare it with the
/// forward-only block was to zero the block's biases first. A gradient taken through the tape was
/// then the gradient of a different model from any checkpoint's. These tests compare against the
/// forward-only block with its biases in place, and check every new gradient by finite differences.
/// </remarks>
public class BiasedAttentionTapeTests
{
    private const int Length = 5;
    private const int Width = 8;

    /// <summary>A block whose biases, norms and weights are all moved off their initial values.</summary>
    private static TransformerEncoderLayer Block(int seed)
    {
        var config = new TransformerConfig(VocabularySize: 20, HiddenSize: Width, Layers: 1, Heads: 2, IntermediateSize: 16);
        var rng = new GraviRandom(seed);
        var layer = new TransformerEncoderLayer(config, rng);

        void Shift(NdArray values, double centre, double scale)
        {
            for (var i = 0; i < values.Size; i++) values.SetAt(i, centre + scale * rng.Normal(0, 1));
        }

        foreach (var dense in new[] { layer.Attention.Query, layer.Attention.Key, layer.Attention.Value, layer.Attention.Output,
                     layer.Intermediate, layer.OutputProjection })
        {
            Shift(dense.Bias, 0, 0.3);
        }

        Shift(layer.AttentionNorm.Gamma, 1, 0.2);
        Shift(layer.AttentionNorm.Beta, 0, 0.2);
        Shift(layer.OutputNorm.Gamma, 1, 0.2);
        Shift(layer.OutputNorm.Beta, 0, 0.2);
        return layer;
    }

    [Fact]
    public void Attention_with_biases_is_the_forward_only_attention_with_biases()
    {
        var layer = Block(3);
        var attention = layer.Attention;
        var x = new GraviRandom(4).StandardNormal(Length, Width);

        var expected = attention.Forward(x);
        var actual = TransformerTape.MultiHeadAttention(
            Tensor.Constant(x),
            Tensor.Constant(attention.Query.Weights), Tensor.Constant(attention.Key.Weights),
            Tensor.Constant(attention.Value.Weights), Tensor.Constant(attention.Output.Weights),
            heads: 2, attentionMask: null,
            Tensor.Constant(attention.Query.Bias), Tensor.Constant(attention.Key.Bias),
            Tensor.Constant(attention.Value.Bias), Tensor.Constant(attention.Output.Bias)).Value;

        Assert.True(UFunc.AllClose(actual, expected, 1e-10));
    }

    [Fact]
    public void A_block_copied_onto_the_tape_computes_what_the_block_computes()
    {
        // EncoderWeights.From carries every weight across, the attention biases included.
        var layer = Block(5);
        var x = new GraviRandom(6).StandardNormal(Length, Width);

        var expected = layer.Forward(x);
        var actual = TransformerTape.EncoderLayer(Tensor.Constant(x), TransformerTape.EncoderWeights.From(layer), heads: 2).Value;

        Assert.True(UFunc.AllClose(actual, expected, 1e-10));
    }

    [Fact]
    public void The_copied_parameters_include_the_attention_biases_and_are_copies()
    {
        var layer = Block(7);
        var weights = TransformerTape.EncoderWeights.From(layer);

        Assert.Equal(16, weights.Parameters.Count());

        weights.QueryBias!.Value.SetAt(0, 99.0);
        Assert.NotEqual(99.0, layer.Attention.Query.Bias.At(0));
    }

    [Fact]
    public void Erf_and_its_gradient_are_right()
    {
        var x = NdArray.FromValues([-2.5, -0.7, 0.0, 0.3, 1.9]).Reshape(1, 5);
        var value = Tensor.Constant(x).Erf().Value;
        for (var j = 0; j < 5; j++) Assert.Equal(MathUtil.Erf(x[0, j]), value[0, j], 15);

        var result = GradientCheck.Check(t => t.Erf().Sum(), x);
        Assert.True(result.Passed(1e-7), result.ToString());
    }

    [Fact]
    public void The_exact_GELU_is_the_erf_formula_and_differs_from_the_tanh_form()
    {
        var x = NdArray.FromValues([-3.0, -0.5, 0.0, 0.7, 2.5]).Reshape(1, 5);
        var exact = TransformerTape.GeluExact(Tensor.Constant(x)).Value;
        var tanh = TransformerTape.Gelu(Tensor.Constant(x)).Value;

        var largestGap = 0.0;
        for (var j = 0; j < 5; j++)
        {
            var v = x[0, j];
            Assert.Equal(0.5 * v * (1 + MathUtil.Erf(v / Math.Sqrt(2))), exact[0, j], 14);
            largestGap = Math.Max(largestGap, Math.Abs(exact[0, j] - tanh[0, j]));
        }

        // The two are not interchangeable: they differ in the fourth decimal place.
        Assert.True(largestGap > 1e-4, $"largest gap {largestGap}");

        var result = GradientCheck.Check(t => TransformerTape.GeluExact(t).Sum(), x);
        Assert.True(result.Passed(1e-7), result.ToString());
    }

    [Theory]
    [InlineData("query")]
    [InlineData("key")]
    [InlineData("value")]
    [InlineData("output")]
    public void Every_attention_bias_gradient_agrees_with_finite_differences(string which)
    {
        var layer = Block(11);
        var copied = TransformerTape.EncoderWeights.From(layer);
        var x = Tensor.Constant(new GraviRandom(12).StandardNormal(Length, Width));
        var readout = Tensor.Constant(new GraviRandom(13).StandardNormal(Length, Width));

        var start = which switch
        {
            "query" => copied.QueryBias!.Value,
            "key" => copied.KeyBias!.Value,
            "value" => copied.ValueBias!.Value,
            _ => copied.AttentionOutputBias!.Value,
        };

        Tensor Loss(Tensor bias)
        {
            var weights = which switch
            {
                "query" => copied with { QueryBias = bias },
                "key" => copied with { KeyBias = bias },
                "value" => copied with { ValueBias = bias },
                _ => copied with { AttentionOutputBias = bias },
            };

            return (TransformerTape.EncoderLayer(x, weights, heads: 2, exactGelu: true) * readout).Sum();
        }

        var result = GradientCheck.Check(Loss, start.Copy());

        // The key bias adds the same amount to every score in a query's row, which the softmax
        // cancels: its gradient is zero up to rounding, and the check has nothing to compare. The
        // others must agree to 1e-6 of the gradient's size.
        if (which == "key")
        {
            Assert.True(Math.Abs(result.Analytic) < 1e-10 && Math.Abs(result.Numeric) < 1e-8, result.ToString());
        }
        else
        {
            Assert.True(result.Passed(1e-6), result.ToString());
        }
    }

    [Fact]
    public void The_whole_block_with_biases_and_exact_GELU_differentiates_correctly_in_its_input()
    {
        var copied = TransformerTape.EncoderWeights.From(Block(17));
        var readout = Tensor.Constant(new GraviRandom(18).StandardNormal(Length, Width));

        var result = GradientCheck.Check(
            x => (TransformerTape.EncoderLayer(x, copied, heads: 2, exactGelu: true) * readout).Sum(),
            new GraviRandom(19).StandardNormal(Length, Width));

        Assert.True(result.Passed(1e-5), result.ToString());
    }
}
