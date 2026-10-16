using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class VqTests
{
    [Fact]
    public void VqAndWhiten_MatchScipy()
    {
        var x = LcgNd(40, 3, 12); var c = LcgNd(4, 3, 13);
        var (codes, dist) = Cluster.Vq(x, c);
        // scipy 1.17.1: vq(lcg_points(40, 3, 12), lcg_points(4, 3, 13))
        Assert.Equal(new[] { 0, 3, 2, 1, 2, 1, 2, 0, 0, 0, 1, 1, 1, 3, 0, 1, 1, 2, 0, 0, 2, 0, 0, 2, 2, 1, 0, 2, 0, 1, 2, 0, 1, 1, 1, 0, 1, 1, 2, 1 }, codes);
        Close([0.5696865551287917, 0.40895187951143863, 0.3627931904654112, 0.21663806713113007, 0.2639065753640823], dist[..5], 1e-15, "distortion");
        var w = Cluster.Whiten(x).ToArray();
        Close([1.4669454416512948, 2.5815972878919804, 0.1484514792570685], w[..3], 1e-14, "whiten first");
        Close([2.9854813542975327, 0.8488526984268289, 0.2420632355885846], w[^3..], 1e-14, "whiten last");
    }

    [Fact]
    public void KMeansFromGuess_MatchesScipyDocstring()
    {
        var f = new NdArray([1.9, 2.3, 1.5, 2.5, 0.8, 0.6, 0.4, 1.8, 1.0, 1.0], 5, 2);
        var guess = new NdArray([1.9, 2.3, 0.8, 0.6], 2, 2);
        var (book, distortion) = Cluster.KMeans(f, guess);
        Close([1.7, 2.4, 0.7333333333333334, 1.1333333333333333], book.ToArray(), 1e-15, "book");
        Close(0.4056391669772859, distortion, 1e-15, "distortion");
    }

    [Fact]
    public void KMeans2FromMatrix_MatchesScipy()
    {
        var x = LcgNd(40, 3, 12); var c = LcgNd(4, 3, 13);
        var (cb, labels) = Cluster.KMeans2(x, c, iter: 7);
        Close([0.43854301988607974, 0.7195762965923853, 0.2629159588288199, 0.48917424469537996, 0.2085623396955464, 0.2393407947769894,
               0.36334565803937247, 0.24282002598181274, 0.7826721295093435, 0.8418482062556211, 0.3476066753272709, 0.6215399220545618],
            cb.ToArray(), 1e-14, "centroids");
        Assert.Equal(new[] { 0, 3, 2, 1, 0, 1, 2, 3, 0, 0, 1, 2, 1, 3, 3, 1, 1, 0, 0, 3, 2, 3, 0, 2, 0, 2, 1, 2, 0, 3, 2, 0, 2, 1, 1, 1, 1, 3, 2, 1 }, labels);
    }

    private static (NdArray X, int[] Truth) Blobs(int perBlob, int seed)
    {
        // Three well separated Gaussian blobs, Iris-like in scale (4 features).
        double[][] centres = [[5.0, 3.4, 1.5, 0.2], [5.9, 2.8, 4.3, 1.3], [6.6, 3.0, 5.6, 2.0]];
        var rng = new GraviRandom(seed);
        var x = new double[3 * perBlob * 4];
        var truth = new int[3 * perBlob];
        for (var b = 0; b < 3; b++)
            for (var i = 0; i < perBlob; i++)
            {
                var r = b * perBlob + i;
                truth[r] = b;
                for (var j = 0; j < 4; j++) x[r * 4 + j] = centres[b][j] + 0.12 * rng.Normal();
            }
        return (new NdArray(x, 3 * perBlob, 4), truth);
    }

    private static void SamePartition(int[] truth, int[] labels)
    {
        // Labels are arbitrary: the partition must match up to renaming.
        var map = new Dictionary<int, int>();
        for (var i = 0; i < truth.Length; i++)
        {
            if (map.TryGetValue(truth[i], out var l)) Assert.Equal(l, labels[i]);
            else map[truth[i]] = labels[i];
        }
        Assert.Equal(3, map.Values.Distinct().Count());
    }

    [Theory]
    [InlineData(KMeansInit.PlusPlus)]
    [InlineData(KMeansInit.Points)]
    [InlineData(KMeansInit.Random)]
    public void KMeans2_RecoversSeparatedBlobs(KMeansInit init)
    {
        var (x, truth) = Blobs(50, 7);
        // Random/points initialisation can leave a blob uncovered; scipy's remedy is restarts.
        int[]? best = null; var bestCost = double.PositiveInfinity;
        for (var seed = 0; seed < 5; seed++)
        {
            var (c, labels) = Cluster.KMeans2(x, 3, iter: 20, minit: init, seed: seed);
            var (_, d) = Cluster.Vq(x, c);
            var cost = d.Sum(v => v * v);
            if (cost < bestCost) { bestCost = cost; best = labels; }
        }
        SamePartition(truth, best!);
    }

    [Fact]
    public void KMeans_RestartsAndKeepsLowestDistortion()
    {
        var (x, truth) = Blobs(40, 3);
        var (book, distortion) = Cluster.KMeans(Cluster.Whiten(x), 3, seed: 1);
        Assert.Equal(3, book.Shape[0]);
        var (codes, d) = Cluster.Vq(Cluster.Whiten(x), book);
        SamePartition(truth, codes);
        Close(d.Average(), distortion, 1e-5, "reported distortion is the mean distance");
    }

    [Fact]
    public void Missing_RaiseThrowsOnEmptyCluster()
    {
        var x = new NdArray([0, 0, 0.1, 0, 0, 0.1, 10, 10.0], 4, 2);
        var init = new NdArray([0, 0, 10, 10, 100, 100.0], 3, 2);
        Assert.Throws<InvalidOperationException>(() => Cluster.KMeans2(x, init, missing: KMeansMissing.Raise));
        var (c, _) = Cluster.KMeans2(x, init);
        Close([100.0, 100.0], c.ToArray()[4..], 0, "empty cluster keeps its centroid");
    }
}
