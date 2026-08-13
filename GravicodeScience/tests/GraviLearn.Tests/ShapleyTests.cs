using Gravicode.Science.GraviLearn.Explain;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviLearn;

/// <summary>
/// Tests for Shapley attributions.
/// </summary>
/// <remarks>
/// Attributions are exactly the kind of output that looks reasonable while being wrong, so these
/// pin the axioms rather than the numbers wherever possible: efficiency, symmetry, the dummy
/// property, and — for a linear model — the closed form <c>φᵢ = wᵢ(xᵢ − E[xᵢ])</c>, which is known
/// independently of anything in the implementation.
/// </remarks>
public class ShapleyTests
{
    private static NdArray Background(int seed = 5, int rows = 60, int features = 3)
    {
        var rng = new GraviRandom(seed);
        var x = NdArray.Zeros(rows, features);
        for (var i = 0; i < rows; i++)
            for (var f = 0; f < features; f++)
                x[i, f] = rng.NextDouble() * 4 - 2;
        return x;
    }

    /// <summary>A linear model with known weights — the case Shapley values have a closed form for.</summary>
    private static Func<NdArray, NdArray> Linear(double[] weights, double intercept = 0.0)
        => batch =>
        {
            var result = NdArray.Zeros(batch.Shape[0]);
            for (var i = 0; i < batch.Shape[0]; i++)
            {
                var sum = intercept;
                for (var f = 0; f < weights.Length; f++) sum += weights[f] * batch[i, f];
                result.SetAt(i, sum);
            }
            return result;
        };

    [Fact]
    public void ExactValuesMatchTheClosedFormForALinearModel()
    {
        // For an additive model the Shapley value is wᵢ(xᵢ − E[xᵢ]), with no interaction terms to
        // divide up. That is a genuinely independent reference.
        var weights = new[] { 2.0, -3.0, 0.5 };
        var background = Background();
        var instance = NdArray.FromValues([1.0, 1.0, 1.0]);

        var attribution = ShapleyValues.Exact(Linear(weights), instance, background);

        for (var f = 0; f < 3; f++)
        {
            var mean = 0.0;
            for (var i = 0; i < background.Shape[0]; i++) mean += background[i, f];
            mean /= background.Shape[0];

            var expected = weights[f] * (instance.At(f) - mean);
            Assert.Equal(expected, attribution.Contributions[f], 10);
        }
    }

    [Fact]
    public void ContributionsSumToThePrediction()
    {
        // Efficiency: the whole point of the attribution is that the parts add up to the answer.
        // An explanation that leaves a residue is one you cannot reason about.
        var background = Background();
        var instance = NdArray.FromValues([1.5, -0.5, 2.0]);
        var predict = Linear([2.0, -3.0, 0.5], intercept: 1.0);

        var attribution = ShapleyValues.Exact(predict, instance, background);

        var actual = predict(Reshape(instance)).At(0);
        Assert.Equal(actual, attribution.Prediction, 10);
    }

    [Fact]
    public void EfficiencyHoldsForANonLinearModelToo()
    {
        // Nothing above assumed additivity; an interaction term must still be divided up without
        // any of the prediction going missing.
        var background = Background(seed: 9);
        var instance = NdArray.FromValues([1.0, 2.0, -1.0]);

        Func<NdArray, NdArray> predict = batch =>
        {
            var result = NdArray.Zeros(batch.Shape[0]);
            for (var i = 0; i < batch.Shape[0]; i++)
                result.SetAt(i, batch[i, 0] * batch[i, 1] + Math.Tanh(batch[i, 2]));
            return result;
        };

        var attribution = ShapleyValues.Exact(predict, instance, background);
        Assert.Equal(predict(Reshape(instance)).At(0), attribution.Prediction, 10);
    }

    [Fact]
    public void AFeatureTheModelIgnoresGetsZeroCredit()
    {
        // The dummy axiom. Feature 2 is not read at all, so no amount of averaging may hand it a
        // share of the prediction.
        var background = Background();
        var instance = NdArray.FromValues([1.0, 1.0, 5.0]);

        var attribution = ShapleyValues.Exact(Linear([2.0, -3.0, 0.0]), instance, background);
        Assert.Equal(0.0, attribution.Contributions[2], 12);
    }

    [Fact]
    public void FeaturesThatContributeEquallyGetEqualCredit()
    {
        // Symmetry. Two features with identical weights and identical values cannot be told apart
        // by the model, so any difference in their credit would come from the algorithm.
        var background = Background();
        var instance = NdArray.FromValues([1.3, 1.3, 0.0]);

        var attribution = ShapleyValues.Exact(Linear([2.0, 2.0, 1.0]), instance, background);

        // Their background means differ, so the equality to check is on the model's terms: build a
        // background where the two columns are identical.
        var symmetric = NdArray.Zeros(background.Shape[0], 3);
        for (var i = 0; i < background.Shape[0]; i++)
        {
            symmetric[i, 0] = background[i, 0];
            symmetric[i, 1] = background[i, 0];
            symmetric[i, 2] = background[i, 2];
        }

        var fair = ShapleyValues.Exact(Linear([2.0, 2.0, 1.0]), instance, symmetric);
        Assert.Equal(fair.Contributions[0], fair.Contributions[1], 10);
        Assert.NotEqual(0.0, attribution.Contributions[0]);
    }

    [Fact]
    public void TheBaseValueIsTheModelsAverageOverTheBackground()
    {
        var background = Background();
        var predict = Linear([2.0, -3.0, 0.5], intercept: 1.0);

        var predictions = predict(background);
        var mean = 0.0;
        for (var i = 0; i < background.Shape[0]; i++) mean += predictions.At(i);
        mean /= background.Shape[0];

        var attribution = ShapleyValues.Exact(predict, NdArray.FromValues([1.0, 1.0, 1.0]), background);
        Assert.Equal(mean, attribution.BaseValue, 10);
    }

    [Fact]
    public void SamplingConvergesOnTheExactValues()
    {
        // The estimator is only useful if it agrees with the thing it approximates.
        var background = Background(seed: 15);
        var instance = NdArray.FromValues([1.0, 2.0, -1.0]);

        Func<NdArray, NdArray> predict = batch =>
        {
            var result = NdArray.Zeros(batch.Shape[0]);
            for (var i = 0; i < batch.Shape[0]; i++)
                result.SetAt(i, batch[i, 0] * batch[i, 1] + 0.5 * batch[i, 2]);
            return result;
        };

        var exact = ShapleyValues.Exact(predict, instance, background);
        var sampled = ShapleyValues.Sample(predict, instance, background, samples: 4000, seed: 3);

        for (var f = 0; f < 3; f++)
            Assert.True(Math.Abs(exact.Contributions[f] - sampled.Contributions[f]) < 0.15,
                $"feature {f}: exact {exact.Contributions[f]:F4} vs sampled {sampled.Contributions[f]:F4}");
    }

    [Fact]
    public void SamplingSatisfiesEfficiencyAtAnySampleSize()
    {
        // Because each permutation telescopes, this holds exactly rather than approximately — even
        // with a single sample. Only the split between features is noisy.
        var background = Background(seed: 19);
        var instance = NdArray.FromValues([0.7, -1.2, 1.9]);
        var predict = Linear([1.0, 2.0, -0.5], intercept: 0.25);

        foreach (var samples in new[] { 1, 5, 50 })
        {
            var attribution = ShapleyValues.Sample(predict, instance, background, samples, seed: 11);
            Assert.Equal(predict(Reshape(instance)).At(0), attribution.Prediction, 8);
        }
    }

    [Fact]
    public void RankedPutsTheStrongestInfluenceFirstRegardlessOfSign()
    {
        // A large negative contribution is as much an explanation as a large positive one.
        var background = Background();
        var instance = NdArray.FromValues([0.1, 2.0, 0.05]);

        var ranked = ShapleyValues.Exact(Linear([0.1, -5.0, 0.1]), instance, background).Ranked;
        Assert.Equal(1, ranked[0].Feature);
        Assert.True(ranked[0].Contribution < 0);
    }

    [Fact]
    public void ExplainSummarisesAcrossRows()
    {
        var background = Background();
        var instances = NdArray.FromArray(new double[,] { { 1, 1, 1 }, { -1, 2, 0 }, { 0.5, -2, 1 } });

        var (perRow, meanAbsolute) = ShapleyValues.Explain(
            Linear([2.0, -3.0, 0.0]), instances, background, samples: 200);

        Assert.Equal(3, perRow.Count);

        // Feature 2 has zero weight, so it moves nothing on any row.
        Assert.True(meanAbsolute[2] < 1e-9, $"an ignored feature averaged {meanAbsolute[2]:E3}");
        Assert.True(meanAbsolute[1] > meanAbsolute[0], "the heavier weight should dominate on this data");
    }

    [Fact]
    public void TooManyFeaturesForExactEnumerationIsRefused()
    {
        // 2^n subsets stops being computable well before it stops being expressible, so this fails
        // loudly rather than appearing to hang.
        var background = NdArray.Zeros(2, 25);
        var instance = NdArray.Zeros(25);

        var error = Assert.Throws<ArgumentException>(
            () => ShapleyValues.Exact(Linear(new double[25]), instance, background));
        Assert.Contains("Sample", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnInstanceOfTheWrongWidthIsRejected()
    {
        Assert.Throws<ArgumentException>(() => ShapleyValues.Exact(
            Linear([1.0, 1.0, 1.0]), NdArray.FromValues([1.0, 2.0]), Background()));
    }

    private static NdArray Reshape(NdArray instance)
    {
        var row = NdArray.Zeros(1, instance.Size);
        for (var f = 0; f < instance.Size; f++) row[0, f] = instance.At(f);
        return row;
    }
}
