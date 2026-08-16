using Gravicode.Science.GraviLearn.Clustering;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviLearn;

/// <summary>
/// Tests for HDBSCAN.
/// </summary>
/// <remarks>
/// The reference that matters most is DBSCAN itself: the varying-density case below is one where
/// no single <c>eps</c> works, so the comparison is run in the test rather than asserted from the
/// literature. Everything else is pinned against a dataset whose true grouping is known by
/// construction.
/// </remarks>
public class HdbscanTests
{
    /// <summary>Three well-separated Gaussian blobs of equal density, 40 points each.</summary>
    private static NdArray ThreeBlobs(out int[] truth, int seed = 5)
    {
        var rng = new GraviRandom(seed);
        var centres = new[] { (0.0, 0.0), (10.0, 0.0), (5.0, 9.0) };

        var x = NdArray.Zeros(120, 2);
        var labels = new int[120];
        var row = 0;

        for (var c = 0; c < 3; c++)
            for (var i = 0; i < 40; i++, row++)
            {
                x[row, 0] = centres[c].Item1 + rng.Normal() * 0.5;
                x[row, 1] = centres[c].Item2 + rng.Normal() * 0.5;
                labels[row] = c;
            }

        truth = labels;
        return x;
    }

    /// <summary>
    /// Two tight clusters close together and one diffuse cluster far away — the configuration a
    /// single density threshold cannot describe.
    /// </summary>
    private static NdArray VaryingDensity(out int[] truth, int seed = 5)
    {
        var rng = new GraviRandom(seed);
        var x = NdArray.Zeros(150, 2);
        var labels = new int[150];

        for (var i = 0; i < 50; i++)
        {
            x[i, 0] = rng.Normal() * 0.3;
            x[i, 1] = rng.Normal() * 0.3;
            labels[i] = 0;
        }

        for (var i = 50; i < 100; i++)
        {
            x[i, 0] = 3 + rng.Normal() * 0.3;
            x[i, 1] = rng.Normal() * 0.3;
            labels[i] = 1;
        }

        for (var i = 100; i < 150; i++)
        {
            x[i, 0] = 25 + rng.Normal() * 3.0;
            x[i, 1] = 25 + rng.Normal() * 3.0;
            labels[i] = 2;
        }

        truth = labels;
        return x;
    }

    /// <summary>Checks that each true group maps onto exactly one predicted label, and vice versa.</summary>
    private static void AssertPartitionsAgree(NdArray predicted, int[] truth, int groups)
    {
        var used = new HashSet<double>();

        for (var group = 0; group < groups; group++)
        {
            var assigned = new HashSet<double>();
            for (var i = 0; i < truth.Length; i++)
                if (truth[i] == group) assigned.Add(predicted.At(i));

            Assert.True(assigned.Count == 1,
                $"true group {group} was split across labels {string.Join(", ", assigned)}");

            var label = assigned.Single();
            Assert.True(label >= 0, $"true group {group} was labelled noise");
            Assert.True(used.Add(label), $"label {label} covers more than one true group");
        }
    }

    [Fact]
    public void ThreeSeparatedBlobsAreRecoveredExactly()
    {
        var x = ThreeBlobs(out var truth);
        var model = new Hdbscan(minClusterSize: 10).Fit(x);

        Assert.Equal(3, model.ClusterCount);
        AssertPartitionsAgree(model.Labels, truth, 3);
    }

    [Theory]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(15)]
    public void TheAnswerIsStableAcrossMinClusterSize(int minClusterSize)
    {
        // The selling point over DBSCAN's eps: the one parameter left is a statement about the
        // problem, and the result should not swing on small changes to it.
        var x = ThreeBlobs(out var truth);
        var model = new Hdbscan(minClusterSize).Fit(x);

        Assert.Equal(3, model.ClusterCount);
        AssertPartitionsAgree(model.Labels, truth, 3);
    }

    [Fact]
    public void VaryingDensityIsHandledWhereNoSingleEpsilonWorks()
    {
        // Run DBSCAN across a sweep of eps first, and assert that none of them gets this right —
        // so the comparison is demonstrated here rather than taken on faith.
        var x = VaryingDensity(out var truth);

        foreach (var epsilon in new[] { 0.5, 1.0, 2.0, 4.0, 6.0 })
        {
            var dbscan = new Dbscan(epsilon, minSamples: 5);
            dbscan.Fit(x);

            var correct = true;
            try { AssertPartitionsAgree(dbscan.Labels, truth, 3); }
            catch (Xunit.Sdk.XunitException) { correct = false; }

            Assert.False(correct, $"DBSCAN at eps={epsilon} recovered all three groups after all");
        }

        // HDBSCAN, with no density threshold to pick, gets all three.
        var model = new Hdbscan(minClusterSize: 10).Fit(x);
        Assert.Equal(3, model.ClusterCount);
        AssertPartitionsAgree(model.Labels, truth, 3);
    }

    [Fact]
    public void ScatteredPointsBetweenClustersBecomeNoise()
    {
        var x = ThreeBlobs(out _);
        var padded = NdArray.Zeros(126, 2);

        for (var i = 0; i < 120; i++) { padded[i, 0] = x[i, 0]; padded[i, 1] = x[i, 1]; }

        // Six points scattered far from every blob and from each other, so they cannot form a
        // cluster of their own either.
        var stray = new[] { (-25.0, -25.0), (30.0, -20.0), (-30.0, 25.0), (40.0, 40.0), (0.0, -35.0), (-40.0, 5.0) };
        for (var i = 0; i < 6; i++) { padded[120 + i, 0] = stray[i].Item1; padded[120 + i, 1] = stray[i].Item2; }

        var model = new Hdbscan(minClusterSize: 10).Fit(padded);

        Assert.Equal(3, model.ClusterCount);
        for (var i = 120; i < 126; i++)
            Assert.Equal(-1.0, model.Labels.At(i));
    }

    [Fact]
    public void NoClusterIsSmallerThanMinClusterSize()
    {
        // The parameter's whole meaning. A cluster below the threshold in the output would mean
        // the condense step let a split through that it should have read as points falling away.
        var x = VaryingDensity(out _);
        var model = new Hdbscan(minClusterSize: 20).Fit(x);

        var counts = new Dictionary<double, int>();
        for (var i = 0; i < model.Labels.Size; i++)
        {
            var label = model.Labels.At(i);
            if (label < 0) continue;
            counts[label] = counts.GetValueOrDefault(label) + 1;
        }

        foreach (var (label, count) in counts)
            Assert.True(count >= 20, $"cluster {label} holds only {count} points");
    }

    [Fact]
    public void CoreDistancesAreLargerInTheSparseCluster()
    {
        // The local density estimate everything else is built on. If this were flat, mutual
        // reachability would collapse back to plain distance and the algorithm to single linkage.
        var x = VaryingDensity(out var truth);
        var model = new Hdbscan(minClusterSize: 10).Fit(x);

        var tight = 0.0;
        var diffuse = 0.0;
        for (var i = 0; i < truth.Length; i++)
            if (truth[i] == 0) tight += model.CoreDistances.At(i);
            else if (truth[i] == 2) diffuse += model.CoreDistances.At(i);

        Assert.True(diffuse / 50 > 5 * (tight / 50),
            $"sparse core distance {diffuse / 50:F3} was not clearly above the tight one {tight / 50:F3}");
    }

    [Fact]
    public void MembershipStrengthIsZeroForNoiseAndPositiveInsideClusters()
    {
        var x = ThreeBlobs(out _);
        var padded = NdArray.Zeros(123, 2);
        for (var i = 0; i < 120; i++) { padded[i, 0] = x[i, 0]; padded[i, 1] = x[i, 1]; }
        padded[120, 0] = -30; padded[120, 1] = -30;
        padded[121, 0] = 35; padded[121, 1] = -25;
        padded[122, 0] = -35; padded[122, 1] = 30;

        var model = new Hdbscan(minClusterSize: 10).Fit(padded);

        for (var i = 0; i < model.Labels.Size; i++)
        {
            if (model.Labels.At(i) < 0) Assert.Equal(0.0, model.Probabilities.At(i));
            else Assert.InRange(model.Probabilities.At(i), 0.0, 1.0);
        }

        for (var i = 120; i < 123; i++) Assert.Equal(-1.0, model.Labels.At(i));
    }

    [Fact]
    public void ASingleUniformCloudIsNotSplitIntoInventedClusters()
    {
        // The failure mode that makes a clusterer untrustworthy: finding structure in a blob that
        // has none. One cluster, or all noise, are both defensible answers; two or more is not.
        var rng = new GraviRandom(77);
        var x = NdArray.Zeros(200, 2);
        for (var i = 0; i < 200; i++) { x[i, 0] = rng.Normal(); x[i, 1] = rng.Normal(); }

        var model = new Hdbscan(minClusterSize: 25).Fit(x);
        Assert.True(model.ClusterCount <= 1,
            $"a single Gaussian cloud was split into {model.ClusterCount} clusters");
    }

    [Fact]
    public void FewerPointsThanMinClusterSizeIsAllNoise()
    {
        // Returning one cluster here would contradict the parameter the caller just set.
        var x = NdArray.FromArray(new double[,] { { 0, 0 }, { 0.1, 0.1 }, { 0.2, 0 } });
        var model = new Hdbscan(minClusterSize: 10).Fit(x);

        Assert.Equal(0, model.ClusterCount);
        for (var i = 0; i < 3; i++) Assert.Equal(-1.0, model.Labels.At(i));
    }

    [Fact]
    public void DuplicatePointsDoNotProduceInfiniteStability()
    {
        // Identical points merge at distance zero, so lambda is infinite there. Letting that into
        // the stability sum would make one cluster infinitely preferable to every other.
        var x = NdArray.Zeros(60, 2);
        for (var i = 0; i < 30; i++) { x[i, 0] = 0; x[i, 1] = 0; }
        for (var i = 30; i < 60; i++) { x[i, 0] = 10; x[i, 1] = 10; }

        var model = new Hdbscan(minClusterSize: 5).Fit(x);

        for (var i = 0; i < 60; i++)
            Assert.False(double.IsNaN(model.Probabilities.At(i)) || double.IsInfinity(model.Probabilities.At(i)),
                $"point {i} got a membership strength of {model.Probabilities.At(i)}");
    }

    [Fact]
    public void FitPredictAgreesWithFitThenLabels()
    {
        var x = ThreeBlobs(out _);
        var model = new Hdbscan(minClusterSize: 10);

        var direct = model.FitPredict(x);
        Assert.Equal(model.Labels.ToArray(), direct.ToArray());
    }

    [Fact]
    public void AMinClusterSizeBelowTwoIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Hdbscan(minClusterSize: 1));
    }
}
