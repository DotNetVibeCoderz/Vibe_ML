using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Autodiff;
using Gravicode.Science.GraviText.Transformers;
using Xunit;

namespace Gravicode.Science.Tests.GraviText;

/// <summary>
/// Tests for the transformer encoder on the autodiff tape.
/// </summary>
/// <remarks>
/// The forward-only classes in <c>Transformers.cs</c> could never be trained — a fresh model
/// stayed randomly initialised for ever. These check that the tape version computes the same
/// architecture, that its gradients agree with central finite differences, and that a model built
/// on it actually learns.
/// </remarks>
public class TransformerTapeTests
{
    private const int Length = 5;
    private const int Width = 8;

    // ---------------------------------------------------------------- agreement with the original

    [Fact]
    public void LayerNorm_MatchesTheForwardOnlyImplementation()
    {
        var rng = new GraviRandom(3);
        var x = rng.StandardNormal(Length, Width);

        var original = new LayerNorm(Width);
        for (var j = 0; j < Width; j++)
        {
            original.Gamma.SetAt(j, 1.0 + 0.1 * j);
            original.Beta.SetAt(j, -0.05 * j);
        }

        var expected = original.Forward(x);
        var actual = TransformerTape.LayerNorm(
            Tensor.Constant(x), Tensor.Constant(original.Gamma), Tensor.Constant(original.Beta)).Value;

        Assert.True(UFunc.AllClose(actual, expected, 1e-10));
    }

    [Fact]
    public void Gelu_MatchesTheScalarActivation()
    {
        var x = NdArray.FromValues([-3.0, -0.5, 0.0, 0.7, 2.5]).Reshape(1, 5);
        var actual = TransformerTape.Gelu(Tensor.Constant(x)).Value;

        for (var j = 0; j < 5; j++)
            Assert.Equal(Activations.Gelu(x[0, j]), actual[0, j], 12);
    }

    [Fact]
    public void MultiHeadAttention_MatchesTheForwardOnlyImplementation()
    {
        var config = new TransformerConfig(VocabularySize: 20, HiddenSize: Width, Layers: 1, Heads: 2,
            IntermediateSize: 16);
        var rng = new GraviRandom(7);
        var attention = new MultiHeadAttention(config, rng);

        // The tape version keeps no bias on the projections, so zero them to compare like for like.
        foreach (var layer in new[] { attention.Query, attention.Key, attention.Value, attention.Output })
            for (var j = 0; j < layer.Bias.Size; j++)
                layer.Bias.SetAt(j, 0.0);

        var x = rng.StandardNormal(Length, Width);
        var expected = attention.Forward(x);

        var actual = TransformerTape.MultiHeadAttention(
            Tensor.Constant(x),
            Tensor.Constant(attention.Query.Weights), Tensor.Constant(attention.Key.Weights),
            Tensor.Constant(attention.Value.Weights), Tensor.Constant(attention.Output.Weights),
            heads: 2).Value;

        Assert.True(UFunc.AllClose(actual, expected, 1e-10));
    }

    [Fact]
    public void MultiHeadAttention_RespectsThePaddingMask()
    {
        var rng = new GraviRandom(11);
        var x = Tensor.Constant(rng.StandardNormal(4, Width));
        var identity = Tensor.Constant(NdArray.Eye(Width));

        // With the last two positions masked, changing their content must not move the output.
        int[] mask = [1, 1, 0, 0];
        var before = TransformerTape.MultiHeadAttention(x, identity, identity, identity, identity, 2, mask).Value;

        var altered = x.Value.Copy();
        for (var j = 0; j < Width; j++) { altered[2, j] += 5.0; altered[3, j] -= 3.0; }

        var after = TransformerTape.MultiHeadAttention(
            Tensor.Constant(altered), identity, identity, identity, identity, 2, mask).Value;

        // Only the unmasked rows are compared: masked rows still produce output, it is just ignored.
        for (var i = 0; i < 2; i++)
            for (var j = 0; j < Width; j++)
                Assert.Equal(before[i, j], after[i, j], 10);
    }

    // ---------------------------------------------------------------- gradients

    [Fact]
    public void LayerNormGradient_AgreesWithFiniteDifferences()
    {
        var rng = new GraviRandom(13);
        var gamma = Tensor.Constant(rng.StandardNormal(Width));
        var beta = Tensor.Constant(rng.StandardNormal(Width));
        var weights = Tensor.Constant(rng.StandardNormal(Length, Width));

        var result = GradientCheck.Check(
            x => (TransformerTape.LayerNorm(x, gamma, beta) * weights).Sum(),
            rng.StandardNormal(Length, Width));

        Assert.True(result.Passed(1e-6), result.ToString());
    }

    [Fact]
    public void SoftmaxRowsGradient_AgreesWithFiniteDifferences()
    {
        var rng = new GraviRandom(17);
        var weights = Tensor.Constant(rng.StandardNormal(Length, Length));

        var result = GradientCheck.Check(
            s => (TransformerTape.SoftmaxRows(s) * weights).Sum(),
            rng.StandardNormal(Length, Length));

        Assert.True(result.Passed(1e-6), result.ToString());
    }

    [Fact]
    public void SoftmaxRows_SumsToOneAndSurvivesLargeScores()
    {
        var scores = NdArray.FromArray(new double[,]
        {
            { 900.0, 901.0, 899.0 },
            { -2.0, 0.0, 2.0 },
        });

        var attention = TransformerTape.SoftmaxRows(Tensor.Constant(scores)).Value;

        for (var i = 0; i < 2; i++)
        {
            var total = 0.0;
            for (var j = 0; j < 3; j++)
            {
                Assert.True(double.IsFinite(attention[i, j]), "softmax overflowed");
                total += attention[i, j];
            }
            Assert.Equal(1.0, total, 12);
        }
    }

    [Fact]
    public void AttentionGradient_FlowsToTheQueryProjection()
    {
        // The query reaches the loss only through the scaled dot product and then the row softmax,
        // which is the path a hand-written backward pass finds hardest to get right.
        var rng = new GraviRandom(19);
        var x = Tensor.Constant(rng.StandardNormal(Length, Width));
        var key = Tensor.Constant(rng.StandardNormal(Width, Width));
        var value = Tensor.Constant(rng.StandardNormal(Width, Width));
        var output = Tensor.Constant(rng.StandardNormal(Width, Width));
        var weights = Tensor.Constant(rng.StandardNormal(Length, Width));

        var result = GradientCheck.Check(
            q => (TransformerTape.MultiHeadAttention(x, q, key, value, output, heads: 2) * weights).Sum(),
            rng.StandardNormal(Width, Width));

        Assert.True(result.Passed(1e-6), result.ToString());
    }

    [Fact]
    public void EncoderLayerGradient_ReachesEveryParameter()
    {
        var rng = new GraviRandom(23);
        var weights = Tensor.Constant(rng.StandardNormal(Length, Width));
        var x = Tensor.Parameter(rng.StandardNormal(Length, Width));

        var layer = new TransformerTape.EncoderWeights(
            Query: Tensor.Parameter(rng.StandardNormal(Width, Width)),
            Key: Tensor.Parameter(rng.StandardNormal(Width, Width)),
            Value: Tensor.Parameter(rng.StandardNormal(Width, Width)),
            AttentionOutput: Tensor.Parameter(rng.StandardNormal(Width, Width)),
            AttentionGamma: Tensor.Parameter(NdArray.Ones(Width)),
            AttentionBeta: Tensor.Parameter(NdArray.Zeros(Width)),
            Intermediate: Tensor.Parameter(rng.StandardNormal(Width, 16)),
            IntermediateBias: Tensor.Parameter(NdArray.Zeros(16)),
            OutputProjection: Tensor.Parameter(rng.StandardNormal(16, Width)),
            OutputBias: Tensor.Parameter(NdArray.Zeros(Width)),
            OutputGamma: Tensor.Parameter(NdArray.Ones(Width)),
            OutputBeta: Tensor.Parameter(NdArray.Zeros(Width)));

        (TransformerTape.EncoderLayer(x, layer, heads: 2) * weights).Sum().Backward();

        // A parameter with no gradient is one the block silently ignores.
        foreach (var parameter in layer.Parameters)
            Assert.NotNull(parameter.Gradient);

        Assert.NotNull(x.Gradient);
    }

    [Fact]
    public void EncoderLayerGradient_AgreesWithFiniteDifferences()
    {
        var rng = new GraviRandom(29);
        var weights = Tensor.Constant(rng.StandardNormal(Length, Width));

        var layer = new TransformerTape.EncoderWeights(
            Query: Tensor.Constant(rng.StandardNormal(Width, Width)),
            Key: Tensor.Constant(rng.StandardNormal(Width, Width)),
            Value: Tensor.Constant(rng.StandardNormal(Width, Width)),
            AttentionOutput: Tensor.Constant(rng.StandardNormal(Width, Width)),
            AttentionGamma: Tensor.Constant(NdArray.Ones(Width)),
            AttentionBeta: Tensor.Constant(NdArray.Zeros(Width)),
            Intermediate: Tensor.Constant(rng.StandardNormal(Width, 16)),
            IntermediateBias: Tensor.Constant(NdArray.Zeros(16)),
            OutputProjection: Tensor.Constant(rng.StandardNormal(16, Width)),
            OutputBias: Tensor.Constant(NdArray.Zeros(Width)),
            OutputGamma: Tensor.Constant(NdArray.Ones(Width)),
            OutputBeta: Tensor.Constant(NdArray.Zeros(Width)));

        var result = GradientCheck.Check(
            x => (TransformerTape.EncoderLayer(x, layer, heads: 2) * weights).Sum(),
            rng.StandardNormal(Length, Width));

        Assert.True(result.Passed(1e-5), result.ToString());
    }

    [Fact]
    public void EmbeddingGradient_AccumulatesForARepeatedToken()
    {
        // Token 2 appears twice, so its embedding row must collect both occurrences' gradient.
        var table = Tensor.Parameter(NdArray.Zeros(4, 3));
        var positions = Tensor.Constant(NdArray.Zeros(4, 3));

        TransformerTape.Embed(table, positions, [2, 0, 2]).Sum().Backward();

        for (var j = 0; j < 3; j++)
        {
            Assert.Equal(2.0, table.Gradient![2, j], 12);
            Assert.Equal(1.0, table.Gradient![0, j], 12);
            Assert.Equal(0.0, table.Gradient![1, j], 12);
        }
    }

    [Fact]
    public void MeanPool_IgnoresPadding()
    {
        var x = Tensor.Constant(NdArray.FromArray(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 99.0, 99.0 },   // padding
        }));

        var pooled = TransformerTape.MeanPool(x, [1, 1, 0]).Value;

        Assert.Equal(2.0, pooled[0, 0], 12);
        Assert.Equal(3.0, pooled[0, 1], 12);
    }

    // ---------------------------------------------------------------- it actually learns

    [Fact]
    public void Classifier_LearnsAPatternThatNeedsOrder()
    {
        // "a then b" is class 0, "b then a" is class 1. Bag-of-words cannot separate these — both
        // classes contain exactly the same tokens — so a model that scores well here has genuinely
        // used position, which is what the position embeddings and attention are for.
        int[][] sequences =
        [
            [1, 2, 3], [1, 2, 4], [1, 2, 5], [3, 1, 2], [4, 1, 2],
            [2, 1, 3], [2, 1, 4], [2, 1, 5], [3, 2, 1], [4, 2, 1],
        ];
        int[] labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        var config = new TransformerConfig(VocabularySize: 8, HiddenSize: 16, Layers: 1, Heads: 2,
            IntermediateSize: 32, MaxPositions: 8);

        var model = new TransformerClassifier(config, seed: 5).Fit(sequences, labels, epochs: 40,
            learningRate: 0.01);

        Assert.True(model.LossHistory[^1] < model.LossHistory[0] / 2,
            $"loss went {model.LossHistory[0]:F4} -> {model.LossHistory[^1]:F4}");
        Assert.True(model.Score(sequences, labels) >= 0.9,
            $"training accuracy {model.Score(sequences, labels):P0}");
    }

    [Fact]
    public void Classifier_RefusesToPredictBeforeTraining()
    {
        var config = new TransformerConfig(VocabularySize: 8, HiddenSize: 8, Layers: 1, Heads: 2,
            IntermediateSize: 16, MaxPositions: 8);

        var error = Assert.Throws<InvalidOperationException>(
            () => new TransformerClassifier(config).Predict([1, 2]));
        Assert.Contains("trained", error.Message, StringComparison.OrdinalIgnoreCase);
    }
}
