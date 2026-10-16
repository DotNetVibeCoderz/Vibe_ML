using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class DistanceTests
{
    private static readonly double[] U = [1.5, -2.0, 3.25, 0.5, 4.0];
    private static readonly double[] V = [0.5, 1.0, 2.0, -1.5, 3.0];
    private static readonly double[] W = [0.5, 2.0, 1.0, 1.5, 0.25];

    [Fact]
    public void NumericMetrics_MatchScipy()
    {
        // All expected values: scipy 1.17.1, scipy.spatial.distance.<metric>(u, v[, w]).
        Close(4.0697051490249265, Distance.Euclidean(U, V), 1e-14, "euclidean");
        Close(5.129571132170798, Distance.Euclidean(U, V, W), 1e-14, "euclidean w");
        Close(16.5625, Distance.SqEuclidean(U, V), 1e-13, "sqeuclidean");
        Close(26.3125, Distance.SqEuclidean(U, V, W), 1e-13, "sqeuclidean w");
        Close(3.389852239885402, Distance.Minkowski(U, V, 3), 1e-14, "minkowski 3");
        Close(4.095675094423062, Distance.Minkowski(U, V, 3, W), 1e-14, "minkowski 3 w");
        Close(39.241433926709625, Distance.Minkowski(U, V, 0.5), 1e-12, "minkowski 0.5");
        Close(8.25, Distance.Cityblock(U, V), 1e-14, "cityblock");
        Close(11.0, Distance.Cityblock(U, V, W), 1e-14, "cityblock w");
        Close(3.0, Distance.Chebyshev(U, V), 0, "chebyshev");
        Close(0.29356187785774257, Distance.Cosine(U, V), 1e-14, "cosine");
        Close(0.7175092036652947, Distance.Cosine(U, V, W), 1e-14, "cosine w");
        Close(0.42559324380226604, Distance.Correlation(U, V), 1e-14, "correlation");
        Close(0.7536140474553198, Distance.Correlation(U, V, W), 1e-14, "correlation w");
        Close(2.880952380952381, Distance.Canberra(U, V), 1e-14, "canberra");
        Close(4.023809523809524, Distance.Canberra(U, V, W), 1e-14, "canberra w");
        Close(0.5076923076923077, Distance.BrayCurtis(U, V), 1e-14, "braycurtis");
        Close(0.9565217391304348, Distance.BrayCurtis(U, V, W), 1e-14, "braycurtis w");
        Close(3.8619403585935053, Distance.SEuclidean(U, V, [0.5, 1.5, 2.0, 0.75, 1.25]), 1e-14, "seuclidean");
        var vi = new NdArray([2, 0.3, 0, 0, 0.1, 0.3, 1, 0.2, 0, 0, 0, 0.2, 1.5, 0.1, 0, 0, 0, 0.1, 1, 0.2, 0.1, 0, 0, 0.2, 3.0], 5, 5);
        Close(4.306245464438831, Distance.Mahalanobis(U, V, vi), 1e-14, "mahalanobis");
        double[] p = [0.1, 0.4, 0.2, 0.3], q = [0.3, 0.1, 0.5, 0.1];
        Close(0.36567493874072776, Distance.JensenShannon(p, q), 1e-14, "jensenshannon");
        Close(0.43922036325310043, Distance.JensenShannon(p, q, 2), 1e-14, "jensenshannon base 2");
    }

    [Fact]
    public void BooleanMetrics_MatchScipy()
    {
        double[] a = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1];
        double[] b = [1, 1, 0, 1, 0, 1, 1, 0, 0, 1];
        double[] w = [1, 2, 1, 0.5, 1, 3, 1, 1, 2, 1.0];
        // scipy 1.17.1 on the boolean arrays, unweighted then weighted.
        Close(0.3333333333333333, Distance.Dice(a, b), 1e-15, "dice"); Close(0.5333333333333333, Distance.Dice(a, b, w), 1e-15, "dice w");
        Close(0.5, Distance.Jaccard(a, b), 1e-15, "jaccard"); Close(0.6956521739130435, Distance.Jaccard(a, b, w), 1e-15, "jaccard w");
        Close(0.5714285714285714, Distance.RogersTanimoto(a, b), 1e-15, "rt"); Close(0.7441860465116279, Distance.RogersTanimoto(a, b, w), 1e-15, "rt w");
        Close(0.6, Distance.RussellRao(a, b), 1e-15, "rr"); Close(0.7407407407407407, Distance.RussellRao(a, b, w), 1e-15, "rr w");
        Close(0.6666666666666666, Distance.SokalSneath(a, b), 1e-15, "ss"); Close(0.8205128205128205, Distance.SokalSneath(a, b, w), 1e-15, "ss w");
        Close(0.6666666666666666, Distance.Yule(a, b), 1e-15, "yule"); Close(1.3636363636363635, Distance.Yule(a, b, w), 1e-15, "yule w");
        Close(0.4, Distance.Hamming(a, b), 1e-15, "hamming"); Close(0.5925925925925926, Distance.Hamming(a, b, w), 1e-15, "hamming w");
        // Non-zero counts as true: scaling the inputs changes nothing.
        Close(Distance.Jaccard(a, b), Distance.Jaccard(a.Select(x => 3.5 * x).ToArray(), b.Select(x => -2 * x).ToArray()), 0, "non-zero is true");
        // Removed from scipy 1.17; closed forms from their old definitions: ntt=4, ntf=2, nft=2, nff=2.
        Close(4.0 / 4.0, Distance.Kulczynski1(a, b), 1e-15, "kulczynski1");
        Close(8.0 / 14.0, Distance.SokalMichener(a, b), 1e-15, "sokalmichener");
    }

    [Fact]
    public void ScipyEdgeCases()
    {
        double[] z = [0, 0, 0];
        Assert.True(double.IsNaN(Distance.Dice(z, z)));
        Close(0, Distance.Jaccard(z, z), 0, "jaccard zeros");
        Close(0, Distance.Yule(z, z), 0, "yule zeros");
        Assert.True(double.IsNaN(Distance.Cosine(z, [1, 0, 1])));
        Close(0, Distance.Cosine([1, 2, 3], [2, 4, 6]), 1e-15, "parallel");
        Close(0, Distance.Canberra(z, z), 0, "canberra 0/0 skipped");
    }

    [Fact]
    public void DirectedHausdorff_MatchesScipy()
    {
        var u = new NdArray([0, 0, 1, 0, 0, 1.0], 3, 2);
        var v = new NdArray([3, 0, 0, 4, 1, 1.0], 3, 2);
        var (d, i, j) = Distance.DirectedHausdorff(u, v, seed: 3);
        Close(Math.Sqrt(2), d, 1e-15, "hausdorff");
        Assert.Equal(0, i); Assert.Equal(2, j);
        // Naive max-min on random sets.
        var a = LcgNd(70, 3, 1); var b = LcgNd(50, 3, 2);
        var cd = Distance.Cdist(a, b).ToArray();
        var naive = Enumerable.Range(0, 70).Max(r => Enumerable.Range(0, 50).Min(c => cd[r * 50 + c]));
        Close(naive, Distance.DirectedHausdorff(a, b, 9).Distance, 1e-15, "random hausdorff");
    }

    [Theory]
    [InlineData(DistanceMetric.Euclidean)] [InlineData(DistanceMetric.SqEuclidean)] [InlineData(DistanceMetric.Cityblock)]
    [InlineData(DistanceMetric.Chebyshev)] [InlineData(DistanceMetric.Minkowski)] [InlineData(DistanceMetric.Cosine)]
    [InlineData(DistanceMetric.Correlation)] [InlineData(DistanceMetric.Canberra)] [InlineData(DistanceMetric.BrayCurtis)]
    [InlineData(DistanceMetric.Hamming)] [InlineData(DistanceMetric.Jaccard)] [InlineData(DistanceMetric.JensenShannon)]
    [InlineData(DistanceMetric.Dice)] [InlineData(DistanceMetric.Yule)] [InlineData(DistanceMetric.RussellRao)]
    [InlineData(DistanceMetric.RogersTanimoto)] [InlineData(DistanceMetric.SokalSneath)]
    public void PdistAndCdist_AgreeWithPerPairFunctions(DistanceMetric metric)
    {
        // 67 columns: not a multiple of the SIMD width, so vector body and scalar tail both run.
        var boolean = metric is DistanceMetric.Jaccard or DistanceMetric.Dice or DistanceMetric.Yule or DistanceMetric.RussellRao
            or DistanceMetric.RogersTanimoto or DistanceMetric.SokalSneath or DistanceMetric.Hamming;
        var raw = Lcg(40 * 67, 1, 13);
        if (boolean) raw = raw.Select(x => x > 0.5 ? 1.0 : 0.0).ToArray();
        var x = new NdArray(raw, 40, 67);
        var y = new NdArray(raw[..(9 * 67)].Select(t => boolean ? 1 - t : t * 0.9 + 0.05).ToArray(), 9, 67);
        Func<double[], double[], double> pair = metric switch
        {
            DistanceMetric.Euclidean => (a, b) => Distance.Euclidean(a, b),
            DistanceMetric.SqEuclidean => (a, b) => Distance.SqEuclidean(a, b),
            DistanceMetric.Cityblock => (a, b) => Distance.Cityblock(a, b),
            DistanceMetric.Chebyshev => (a, b) => Distance.Chebyshev(a, b),
            DistanceMetric.Minkowski => (a, b) => Distance.Minkowski(a, b, 3),
            DistanceMetric.Cosine => (a, b) => Distance.Cosine(a, b),
            DistanceMetric.Correlation => (a, b) => Distance.Correlation(a, b),
            DistanceMetric.Canberra => (a, b) => Distance.Canberra(a, b),
            DistanceMetric.BrayCurtis => (a, b) => Distance.BrayCurtis(a, b),
            DistanceMetric.Hamming => (a, b) => Distance.Hamming(a, b),
            DistanceMetric.Jaccard => (a, b) => Distance.Jaccard(a, b),
            DistanceMetric.JensenShannon => (a, b) => Distance.JensenShannon(a, b),
            DistanceMetric.Dice => (a, b) => Distance.Dice(a, b),
            DistanceMetric.Yule => (a, b) => Distance.Yule(a, b),
            DistanceMetric.RussellRao => (a, b) => Distance.RussellRao(a, b),
            DistanceMetric.RogersTanimoto => (a, b) => Distance.RogersTanimoto(a, b),
            DistanceMetric.SokalSneath => (a, b) => Distance.SokalSneath(a, b),
            _ => throw new InvalidOperationException(),
        };
        // A plain scalar reference for the Euclidean family, written here.
        if (metric == DistanceMetric.Euclidean) pair = (a, b) => Math.Sqrt(a.Zip(b, (s, t) => (s - t) * (s - t)).Sum());
        var rows = Enumerable.Range(0, 40).Select(i => raw.AsSpan(i * 67, 67).ToArray()).ToArray();
        var yrows = Enumerable.Range(0, 9).Select(i => y.ToArray().AsSpan(i * 67, 67).ToArray()).ToArray();
        var pd = Distance.Pdist(x, metric, p: 3);
        var k = 0;
        for (var i = 0; i < 40; i++)
            for (var j = i + 1; j < 40; j++)
                Close(pair(rows[i], rows[j]), pd[k++], 1e-12, $"pdist {metric} ({i},{j})");
        var cd = Distance.Cdist(x, y, metric, p: 3).ToArray();
        for (var i = 0; i < 40; i++)
            for (var j = 0; j < 9; j++)
                Close(pair(rows[i], yrows[j]), cd[i * 9 + j], 1e-12, $"cdist {metric} ({i},{j})");
    }

    [Fact]
    public void LargeEuclideanCdist_ParallelBlockedPathMatchesScalar()
    {
        var a = LcgNd(300, 13, 3); var b = LcgNd(517, 13, 4);
        var cd = Distance.Cdist(a, b).ToArray();
        var av = a.ToArray(); var bv = b.ToArray();
        for (var i = 0; i < 300; i += 7)
            for (var j = 0; j < 517; j++)
            {
                var s = 0.0;
                for (var k = 0; k < 13; k++) { var d = av[i * 13 + k] - bv[j * 13 + k]; s += d * d; }
                Close(Math.Sqrt(s), cd[i * 517 + j], 1e-14, "cdist");
            }
        var serial = Distance.Cdist(a, b, workers: 1).ToArray();
        Assert.Equal(serial, cd);
    }

    [Fact]
    public void DefaultVarianceAndInverseCovariance_MatchScipy()
    {
        var x = LcgNd(15, 4, 5); var y = LcgNd(6, 4, 6);
        // scipy 1.17.1: pdist(X, 'seuclidean') / 'mahalanobis' and cdist(X, Y, ...) with defaults.
        var ps = Distance.Pdist(x, DistanceMetric.SEuclidean);
        Close(3.2276407957650126, ps[0], 1e-12, "se0"); Close(2.255259342634647, ps[50], 1e-12, "se50"); Close(3.8569999500695125, ps[^1], 1e-12, "se-1");
        var pm = Distance.Pdist(x, DistanceMetric.Mahalanobis);
        Close(2.7808103499499697, pm[0], 1e-11, "m0"); Close(3.239333407189032, pm[50], 1e-11, "m50"); Close(3.890188188314891, pm[^1], 1e-11, "m-1");
        var cm = Distance.Cdist(x, y, DistanceMetric.Mahalanobis).ToArray();
        Close(2.818072644078376, cm[0], 1e-11, "cm00"); Close(2.9003614618119387, cm[14 * 6 + 5], 1e-11, "cm145");
        var cs = Distance.Cdist(x, y, DistanceMetric.SEuclidean).ToArray();
        Close(2.7178350969906804, cs[0], 1e-12, "cs00"); Close(2.835784774835382, cs[14 * 6 + 5], 1e-12, "cs145");
    }

    [Fact]
    public void Squareform_RoundTripsAndChecks()
    {
        var x = LcgNd(7, 3, 2);
        var y = Distance.Pdist(x);
        var m = Distance.Squareform(y);
        Assert.Equal(new[] { 7, 7 }, m.Shape.ToArray());
        Assert.Equal(y, Distance.Squareform(m));
        Assert.Equal(m.ToArray(), Distance.Cdist(x, x).ToArray().Select((v, i) => i / 7 == i % 7 ? 0 : v).ToArray());
        Assert.Throws<ArgumentException>(() => Distance.Squareform(new double[4]));
        var bad = m.Copy(); bad[0, 1] = 5;
        Assert.Throws<ArgumentException>(() => Distance.Squareform(bad));
        Assert.Equal(1, Distance.Squareform(Array.Empty<double>()).Shape[0]);
        Assert.Equal(Distance.Cdist(x, x, DistanceMetric.Cityblock).ToArray(), Distance.DistanceMatrix(x, x, 1).ToArray());
    }

    [Fact]
    public void CustomMetric_IsUsed()
    {
        var x = LcgNd(5, 2, 1);
        var custom = Distance.Pdist(x, (a, b) => Math.Abs(a[0] - b[0]));
        var reference = Distance.Pdist(new NdArray(x.ToArray().Where((_, i) => i % 2 == 0).ToArray(), 5, 1), DistanceMetric.Cityblock);
        Close(reference, custom, 0, "custom");
    }
}
