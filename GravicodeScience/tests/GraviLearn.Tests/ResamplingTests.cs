using Gravicode.Science.GraviLearn.Resampling;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviLearn;

/// <summary>
/// Tests for the imbalanced-data resamplers.
/// </summary>
/// <remarks>
/// The properties worth pinning are the ones a plausible-looking implementation gets wrong: that
/// the balance really is equal afterwards, that no synthetic point escapes the region its parents
/// occupy, and that the original rows survive oversampling rather than being replaced.
/// </remarks>
public class ResamplingTests
{
    /// <summary>90 rows of class 0 and 10 of class 1, in two well-separated blobs.</summary>
    private static (NdArray X, NdArray Y) Imbalanced(int seed = 3)
    {
        var rng = new GraviRandom(seed);
        var x = NdArray.Zeros(100, 2);
        var y = NdArray.Zeros(100);

        for (var i = 0; i < 100; i++)
        {
            var minority = i >= 90;
            x[i, 0] = (minority ? 10 : 0) + rng.NextDouble();
            x[i, 1] = (minority ? 10 : 0) + rng.NextDouble();
            y.SetAt(i, minority ? 1 : 0);
        }

        return (x, y);
    }

    private static Dictionary<double, int> Counts(NdArray y)
    {
        var counts = new Dictionary<double, int>();
        for (var i = 0; i < y.Size; i++) counts[y.At(i)] = counts.GetValueOrDefault(y.At(i)) + 1;
        return counts;
    }

    [Fact]
    public void OverSamplingLevelsEveryClassUpToTheLargest()
    {
        var (x, y) = Imbalanced();
        var result = Resampler.OverSample(x, y);

        var counts = Counts(result.Y);
        Assert.Equal(90, counts[0]);
        Assert.Equal(90, counts[1]);
        Assert.Equal(180, result.X.Shape[0]);
    }

    [Fact]
    public void OverSamplingKeepsEveryOriginalRow()
    {
        // Sampling with replacement across the whole class, rather than keeping the originals and
        // topping up, would quietly drop some real observations.
        var (x, y) = Imbalanced();
        var result = Resampler.OverSample(x, y);

        var kept = new HashSet<(double, double)>();
        for (var i = 0; i < result.X.Shape[0]; i++) kept.Add((result.X[i, 0], result.X[i, 1]));

        for (var i = 0; i < 100; i++)
            Assert.Contains((x[i, 0], x[i, 1]), kept);
    }

    [Fact]
    public void UnderSamplingLevelsDownToTheSmallest()
    {
        var (x, y) = Imbalanced();
        var result = Resampler.UnderSample(x, y);

        var counts = Counts(result.Y);
        Assert.Equal(10, counts[0]);
        Assert.Equal(10, counts[1]);
    }

    [Fact]
    public void UnderSamplingDrawsWithoutReplacement()
    {
        // Sampling with replacement here would keep fewer distinct majority rows than it claims,
        // which is the opposite of the point.
        var (x, y) = Imbalanced();
        var result = Resampler.UnderSample(x, y);

        var distinct = new HashSet<(double, double)>();
        for (var i = 0; i < result.X.Shape[0]; i++) distinct.Add((result.X[i, 0], result.X[i, 1]));

        Assert.Equal(result.X.Shape[0], distinct.Count);
    }

    [Fact]
    public void SmoteBalancesTheClasses()
    {
        var (x, y) = Imbalanced();
        var result = Resampler.Smote(x, y);

        var counts = Counts(result.Y);
        Assert.Equal(90, counts[0]);
        Assert.Equal(90, counts[1]);
    }

    [Fact]
    public void SyntheticPointsStayInsideTheMinorityRegion()
    {
        // The property that makes SMOTE defensible: interpolation between same-class neighbours
        // cannot leave the convex hull of that class. If a synthetic point turned up in the
        // majority blob, the interpolation would be reaching across classes.
        var (x, y) = Imbalanced();
        var result = Resampler.Smote(x, y);

        for (var i = 0; i < result.X.Shape[0]; i++)
        {
            if (result.Y.At(i) != 1) continue;
            Assert.InRange(result.X[i, 0], 10.0, 11.0);
            Assert.InRange(result.X[i, 1], 10.0, 11.0);
        }
    }

    [Fact]
    public void SmoteCreatesGenuinelyNewPointsRatherThanDuplicates()
    {
        // Distinguishes it from plain oversampling: with a continuous feature the interpolation
        // parameter is almost never 0 or 1, so most synthetic rows should be unseen.
        var (x, y) = Imbalanced();
        var result = Resampler.Smote(x, y);

        var originals = new HashSet<(double, double)>();
        for (var i = 0; i < 100; i++) originals.Add((x[i, 0], x[i, 1]));

        var novel = 0;
        for (var i = 0; i < result.X.Shape[0]; i++)
            if (result.Y.At(i) == 1 && !originals.Contains((result.X[i, 0], result.X[i, 1]))) novel++;

        Assert.True(novel > 60, $"only {novel} of the 80 synthetic rows were new");
    }

    [Fact]
    public void SmoteFallsBackToDuplicationForASingletonClass()
    {
        // One row has no neighbour to interpolate towards. Inventing a spread would be making the
        // data up, so the honest answer is a duplicate.
        var x = NdArray.FromArray(new double[,] { { 0, 0 }, { 1, 1 }, { 2, 2 }, { 9, 9 } });
        var y = NdArray.FromValues([0, 0, 0, 1]);

        var result = Resampler.Smote(x, y);

        Assert.Equal(6, result.X.Shape[0]);
        for (var i = 0; i < result.X.Shape[0]; i++)
            if (result.Y.At(i) == 1)
            {
                Assert.Equal(9.0, result.X[i, 0]);
                Assert.Equal(9.0, result.X[i, 1]);
            }
    }

    [Fact]
    public void AlreadyBalancedDataIsLeftAtTheSameSize()
    {
        var x = NdArray.FromArray(new double[,] { { 0, 0 }, { 1, 1 }, { 8, 8 }, { 9, 9 } });
        var y = NdArray.FromValues([0, 0, 1, 1]);

        Assert.Equal(4, Resampler.OverSample(x, y).X.Shape[0]);
        Assert.Equal(4, Resampler.UnderSample(x, y).X.Shape[0]);
        Assert.Equal(4, Resampler.Smote(x, y, neighbours: 1).X.Shape[0]);
    }

    [Fact]
    public void FeaturesAndTargetsStayAlignedAfterTheShuffle()
    {
        // The bug that would make every resampler useless while every count still looked right:
        // shuffling X and Y independently. Class 1 lives at (10, 10), so the label has to match.
        var (x, y) = Imbalanced();
        var result = Resampler.OverSample(x, y);

        for (var i = 0; i < result.X.Shape[0]; i++)
            Assert.Equal(result.X[i, 0] > 5 ? 1.0 : 0.0, result.Y.At(i));
    }

    [Fact]
    public void ClassWeightsAreInverselyProportionalToFrequency()
    {
        var y = NdArray.FromValues([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        var weights = Resampler.ClassWeights(y);

        // n / (k · nᶜ): 10 / (2 · 9) and 10 / (2 · 1).
        Assert.Equal(10.0 / 18.0, weights[0], 12);
        Assert.Equal(5.0, weights[1], 12);

        // And the total weight each class carries is then equal, which is the point.
        Assert.Equal(9 * weights[0], 1 * weights[1], 12);
    }

    [Fact]
    public void ClassBalanceReportsCountsLargestFirst()
    {
        var (_, y) = Imbalanced();
        var balance = Resampler.ClassBalance(y);

        Assert.Equal((0.0, 90), balance[0]);
        Assert.Equal((1.0, 10), balance[1]);
    }

    [Fact]
    public void MismatchedInputsAreRejected()
    {
        var x = NdArray.Zeros(4, 2);
        Assert.Throws<ArgumentException>(() => Resampler.OverSample(x, NdArray.Zeros(3)));
        Assert.Throws<ArgumentException>(() => Resampler.OverSample(NdArray.Zeros(4), NdArray.Zeros(4)));
    }
}
