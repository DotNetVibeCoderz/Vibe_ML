using Gravicode.Science.GraviLearn.Explain;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.Trees;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviLearn;

/// <summary>
/// Tests for permutation importance, calibration diagnostics and isotonic regression.
/// </summary>
/// <remarks>
/// Each is pinned against something known independently of the implementation: a dataset where the
/// informative feature is known by construction, a hand-worked pool-adjacent-violators example, and
/// a brute-force search over monotone fits.
/// </remarks>
public class ExplainTests
{
    /// <summary>
    /// A dataset whose label depends on feature 0 alone; features 1 and 2 are noise.
    /// </summary>
    private static (NdArray X, NdArray Y) SignalAndNoise(int seed = 1, int rows = 400)
    {
        var rng = new GraviRandom(seed);
        var x = NdArray.Zeros(rows, 3);
        var y = NdArray.Zeros(rows);

        for (var i = 0; i < rows; i++)
        {
            var signal = rng.NextDouble() * 2 - 1;
            x[i, 0] = signal;
            x[i, 1] = rng.NextDouble() * 2 - 1;
            x[i, 2] = rng.NextDouble() * 2 - 1;
            y.SetAt(i, signal > 0 ? 1 : 0);
        }

        return (x, y);
    }

    // ------------------------------------------------------- permutation importance

    [Fact]
    public void ImportanceFindsTheFeatureTheLabelActuallyDependsOn()
    {
        var (x, y) = SignalAndNoise();
        var model = new DecisionTree(maxDepth: 4);
        model.Fit(x, y);

        var ranked = PermutationImportance.Ranked(model, x, y, repeats: 5);

        // Feature 0 is the label, by construction. Nothing else can compete.
        Assert.Equal(0, ranked[0].Feature);
        Assert.True(ranked[0].Mean > 0.25, $"informative feature scored only {ranked[0].Mean:F4}");
    }

    [Fact]
    public void NoiseFeaturesScoreAboutZero()
    {
        var (x, y) = SignalAndNoise();
        var model = new DecisionTree(maxDepth: 4);
        model.Fit(x, y);

        var importances = PermutationImportance.Compute(model, x, y, repeats: 5);

        // Shuffling a column the model never learned from cannot cost it much. A shallow tree may
        // have split on noise once, so this is a loose bound rather than an exact zero.
        Assert.True(importances[1].Mean < 0.05, $"noise feature 1 scored {importances[1].Mean:F4}");
        Assert.True(importances[2].Mean < 0.05, $"noise feature 2 scored {importances[2].Mean:F4}");
    }

    [Fact]
    public void ImportanceIsReproducibleForAGivenSeed()
    {
        var (x, y) = SignalAndNoise();
        var model = new DecisionTree(maxDepth: 3);
        model.Fit(x, y);

        var first = PermutationImportance.Compute(model, x, y, repeats: 3, seed: 7);
        var second = PermutationImportance.Compute(model, x, y, repeats: 3, seed: 7);

        for (var i = 0; i < first.Count; i++)
            Assert.Equal(first[i].Mean, second[i].Mean, 12);
    }

    [Fact]
    public void ImportanceDoesNotDisturbTheCallersData()
    {
        // The method shuffles a column; doing that in place would corrupt the caller's matrix and
        // every importance measured after the first.
        var (x, y) = SignalAndNoise(rows: 60);
        var before = x.Copy().ToArray();

        var model = new DecisionTree(maxDepth: 3);
        model.Fit(x, y);
        PermutationImportance.Compute(model, x, y, repeats: 2);

        Assert.Equal(before, x.ToArray());
    }

    // ------------------------------------------------------------------ calibration

    [Fact]
    public void APerfectlyCalibratedGeneratorHasNearZeroError()
    {
        // Draw the outcome with exactly the probability claimed. By construction the observed
        // frequency in each bin must match its mean prediction, up to sampling noise.
        var rng = new GraviRandom(21);
        var probabilities = NdArray.Zeros(20000);
        var labels = NdArray.Zeros(20000);

        for (var i = 0; i < 20000; i++)
        {
            var p = rng.NextDouble();
            probabilities.SetAt(i, p);
            labels.SetAt(i, rng.NextDouble() < p ? 1 : 0);
        }

        Assert.True(Calibration.ExpectedError(probabilities, labels) < 0.02,
            $"ECE was {Calibration.ExpectedError(probabilities, labels):F4}");
    }

    [Fact]
    public void AnOverconfidentModelIsCaught()
    {
        // Claims 0.95 but is right 60% of the time — the failure accuracy and AUC cannot see,
        // because the ranking is untouched.
        var rng = new GraviRandom(22);
        var probabilities = NdArray.Full(0.95, 2000);
        var labels = NdArray.Zeros(2000);
        for (var i = 0; i < 2000; i++) labels.SetAt(i, rng.NextDouble() < 0.6 ? 1 : 0);

        var ece = Calibration.ExpectedError(probabilities, labels);
        Assert.True(ece > 0.3, $"ECE was only {ece:F4}");
    }

    [Fact]
    public void TheCurveReportsBinPopulationsAndSkipsEmptyBins()
    {
        var probabilities = NdArray.FromValues([0.05, 0.06, 0.95]);
        var labels = NdArray.FromValues([0, 0, 1]);

        var curve = Calibration.Curve(probabilities, labels, bins: 10);

        // Only two bins hold anything; the eight empty ones must not appear, or the curve would be
        // drawn through regions where there is no evidence at all.
        Assert.Equal(2, curve.Count);
        Assert.Equal(2, curve[0].Count);
        Assert.Equal(0.0, curve[0].ObservedFraction);
        Assert.Equal(1.0, curve[1].ObservedFraction);
    }

    [Fact]
    public void BrierScoreMatchesTheHandComputedValue()
    {
        var probabilities = NdArray.FromValues([0.9, 0.1, 0.8]);
        var labels = NdArray.FromValues([1, 0, 0]);

        // (0.01 + 0.01 + 0.64) / 3
        Assert.Equal(0.22, Calibration.BrierScore(probabilities, labels), 10);
    }

    // ---------------------------------------------------------- isotonic regression

    [Fact]
    public void PoolAdjacentViolatorsMatchesTheHandWorkedExample()
    {
        // [3, 2, 4] violates at the second point: 3 and 2 pool to 2.5, and 4 is already above.
        var model = new IsotonicRegression().Fit(
            NdArray.FromValues([1, 2, 3]), NdArray.FromValues([3, 2, 4]));

        var steps = model.Steps;
        Assert.Equal(2, steps.Count);
        Assert.Equal(2.5, steps[0].Value, 12);
        Assert.Equal(4.0, steps[1].Value, 12);
    }

    [Fact]
    public void PoolingCascadesBackwardsThroughEarlierBlocks()
    {
        // The case a single forward pass gets wrong. Merging 2 into 3 gives 2.5, which is still
        // above the 1 that follows; the merge has to repeat backwards until the whole prefix is
        // monotone, giving one block of (3 + 2 + 1) / 3.
        var model = new IsotonicRegression().Fit(
            NdArray.FromValues([1, 2, 3, 4]), NdArray.FromValues([3, 2, 1, 5]));

        Assert.Equal(2, model.Steps.Count);
        Assert.Equal(2.0, model.Steps[0].Value, 12);
        Assert.Equal(5.0, model.Steps[1].Value, 12);
    }

    [Fact]
    public void AlreadyMonotoneDataIsLeftAlone()
    {
        var x = NdArray.FromValues([1, 2, 3, 4]);
        var y = NdArray.FromValues([1, 2, 5, 9]);

        var fitted = new IsotonicRegression().Fit(x, y).Predict(x);
        Assert.True(UFunc.AllClose(fitted, y, 1e-12));
    }

    [Fact]
    public void TheFitIsTheBestMonotoneOneInSquaredError()
    {
        // The defining property, checked against a brute-force search rather than against the same
        // algorithm: no monotone step function on a grid beats what PAVA found.
        var x = NdArray.FromValues([1, 2, 3, 4, 5]);
        var y = NdArray.FromValues([4, 1, 3, 2, 5]);

        var fitted = new IsotonicRegression().Fit(x, y).Predict(x);
        var best = SquaredError(fitted, y);

        var grid = new[] { 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 };
        foreach (var candidate in MonotoneCandidates(grid, 5))
        {
            var error = 0.0;
            for (var i = 0; i < 5; i++)
            {
                var d = candidate[i] - y.At(i);
                error += d * d;
            }
            Assert.True(best <= error + 1e-9,
                $"a monotone candidate scored {error:F6}, better than the fit's {best:F6}");
        }
    }

    [Fact]
    public void PredictionsNeverDecrease()
    {
        var rng = new GraviRandom(31);
        var x = NdArray.Zeros(200);
        var y = NdArray.Zeros(200);
        for (var i = 0; i < 200; i++)
        {
            x.SetAt(i, i / 200.0);
            y.SetAt(i, rng.NextDouble());       // pure noise, so violations abound
        }

        var model = new IsotonicRegression().Fit(x, y);
        var previous = double.NegativeInfinity;

        for (var i = 0; i < 200; i++)
        {
            var value = model.PredictOne(i / 200.0);
            Assert.True(value >= previous - 1e-12, $"output decreased at {i}");
            previous = value;
        }
    }

    [Fact]
    public void IsotonicRegressionRecalibratesAnOverconfidentClassifier()
    {
        // What the method is actually for. Squash a set of well-ranked but badly scaled scores and
        // check that fitting them back onto the outcomes restores the calibration.
        var rng = new GraviRandom(41);
        var scores = NdArray.Zeros(4000);
        var labels = NdArray.Zeros(4000);

        for (var i = 0; i < 4000; i++)
        {
            var truth = rng.NextDouble();
            // A monotone but badly wrong reported probability: the ranking is perfect, the numbers
            // are not.
            scores.SetAt(i, Math.Sqrt(truth));
            labels.SetAt(i, rng.NextDouble() < truth ? 1 : 0);
        }

        var before = Calibration.ExpectedError(scores, labels);
        var after = Calibration.ExpectedError(
            new IsotonicRegression().Fit(scores, labels).Predict(scores), labels);

        Assert.True(before > 0.1, $"the test setup was not miscalibrated: ECE {before:F4}");
        Assert.True(after < 0.02, $"recalibration left ECE at {after:F4}");
    }

    [Fact]
    public void PredictingBeforeFittingThrows()
    {
        Assert.Throws<InvalidOperationException>(
            () => new IsotonicRegression().PredictOne(0.5));
    }

    // ------------------------------------------------------------------- helpers

    private static double SquaredError(NdArray fitted, NdArray y)
    {
        var total = 0.0;
        for (var i = 0; i < y.Size; i++)
        {
            var d = fitted.At(i) - y.At(i);
            total += d * d;
        }
        return total;
    }

    /// <summary>Every non-decreasing sequence of the given length drawn from <paramref name="grid"/>.</summary>
    private static IEnumerable<double[]> MonotoneCandidates(double[] grid, int length)
    {
        var current = new double[length];

        IEnumerable<double[]> Extend(int position, int from)
        {
            if (position == length)
            {
                yield return (double[])current.Clone();
                yield break;
            }

            for (var g = from; g < grid.Length; g++)
            {
                current[position] = grid[g];
                foreach (var result in Extend(position + 1, g)) yield return result;
            }
        }

        return Extend(0, 0);
    }
}
