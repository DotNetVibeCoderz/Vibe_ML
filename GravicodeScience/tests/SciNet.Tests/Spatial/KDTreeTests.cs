using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class KDTreeTests
{
    private static double Dist(double[] x, int i, double[] q, int j, int d, double p)
        => Distance.Minkowski(x.AsSpan(i * d, d), q.AsSpan(j * d, d), p);

    public static TheoryData<int, double, bool, int> Configs()
    {
        var data = new TheoryData<int, double, bool, int>();
        foreach (var d in new[] { 2, 3, 5, 8 })
            foreach (var p in new[] { 1.0, 2.0, 3.0, double.PositiveInfinity })
                data.Add(d, p, d % 2 == 0, d == 5 ? 1 : 10);
        return data;
    }

    [Theory]
    [MemberData(nameof(Configs))]
    public void Query_MatchesBruteForce(int d, double p, bool balanced, int leafSize)
    {
        const int n = 400, q = 60, k = 7;
        var x = Lcg(n, d, 100 + d);
        var qs = Lcg(q, d, 200 + d);
        var tree = new KDTree(new NdArray(x, n, d), leafSize: leafSize, balancedTree: balanced, compactNodes: !balanced);
        var (dist, idx) = tree.Query(new NdArray(qs, q, d), k, p: p);
        var dv = dist.ToArray();
        for (var j = 0; j < q; j++)
        {
            var brute = Enumerable.Range(0, n).Select(i => (D: Dist(x, i, qs, j, d, p), I: i)).OrderBy(t => t.D).ThenBy(t => t.I).Take(k).ToArray();
            for (var r = 0; r < k; r++)
            {
                Close(brute[r].D, dv[j * k + r], 1e-12, $"dist d={d} p={p}");
                Assert.Equal(brute[r].I, idx[j, r]);
            }
        }
    }

    [Fact]
    public void Ties_AreBrokenBySmallerIndex_AndUpperBoundIsStrict()
    {
        // Integer lattice: many exactly equal distances.
        var pts = new List<double>();
        for (var i = 0; i < 6; i++) for (var j = 0; j < 6; j++) { pts.Add(i); pts.Add(j); }
        var x = pts.ToArray();
        var tree = new KDTree(new NdArray(x, 36, 2), leafSize: 2);
        var (dist, idx) = tree.Query(new double[] { 2, 2 }, 5);
        Assert.Equal(new[] { 14, 8, 13, 15, 20 }, idx);
        Assert.Equal(new[] { 0, 1, 1, 1, 1.0 }, dist);
        // scipy: neighbours must be strictly closer than distance_upper_bound; missing -> (inf, n).
        var (d2, i2) = tree.Query(new double[] { 2, 2 }, 3, distanceUpperBound: 1.0);
        Assert.Equal(new[] { 14, 36, 36 }, i2);
        Assert.True(double.IsPositiveInfinity(d2[1]) && double.IsPositiveInfinity(d2[2]));
        // Ball queries are inclusive.
        Assert.Equal(new[] { 8, 13, 14, 15, 20 }, tree.QueryBallPoint(new double[] { 2, 2 }, 1.0, returnSorted: true));
        Assert.Equal(5, tree.QueryBallPointLength(new double[] { 2, 2 }, 1.0));
    }

    [Fact]
    public void ScipyExamples()
    {
        // scipy 1.17.1: cKDTree([[0],[1],[3]])
        var tree = new KDTree(new double[,] { { 0 }, { 1 }, { 3 } });
        var (d, i) = tree.Query(new double[] { 0 }, 3, distanceUpperBound: 1.0);
        Assert.Equal(new[] { 0, 3, 3 }, i);
        Assert.Equal(0.0, d[0]);
        var (dk, ik) = tree.Query(new NdArray([0.0, 1.0], 2, 1), new[] { 1, 3 });
        Assert.Equal(new[] { 0.0, 3, 0, 2 }, dk.ToArray());
        Assert.Equal(new[,] { { 0, 2 }, { 1, 2 } }, ik);
        Assert.Equal(new long[] { 3, 2, 2 }, tree.CountNeighbors(tree, [0, 1, 2], cumulative: false));
        Assert.Equal(new long[] { 7, 3, 5 }, tree.CountNeighbors(tree, [2, 0, 1]));
        Assert.Equal(new HashSet<(int, int)> { (0, 1), (1, 2) }, tree.QueryPairs(2.0));
        var sdm = tree.SparseDistanceMatrixDictionary(tree, 1.0);
        Assert.Equal(5, sdm.Count);
        Assert.Equal(1.0, sdm[(1, 0)]);
    }

    [Theory]
    [InlineData(2, 2.0)] [InlineData(3, 1.0)] [InlineData(4, double.PositiveInfinity)] [InlineData(6, 2.5)]
    public void BallQueries_PairsAndCounts_MatchBruteForce(int d, double p)
    {
        const int n = 300;
        var x = Lcg(n, d, 7 + d);
        var y = Lcg(150, d, 9 + d);
        var tx = new KDTree(new NdArray(x, n, d), leafSize: 8);
        var ty = new KDTree(new NdArray(y, 150, d), leafSize: 5, balancedTree: false);
        var r = d switch { 2 => 0.1, 3 => 0.3, 4 => 0.2, _ => 0.6 };

        var balls = tx.QueryBallPoint(new NdArray(y, 150, d), r, p: p);
        for (var j = 0; j < 150; j++)
        {
            var brute = Enumerable.Range(0, n).Where(i => Dist(x, i, y, j, d, p) <= r).ToArray();
            Assert.Equal(brute, balls[j]);
        }
        var lengths = tx.QueryBallPointLength(new NdArray(y, 150, d), r, p: p);
        Assert.Equal(balls.Select(b => (long)b.Length).ToArray(), lengths);

        var pairs = tx.QueryPairs(r, p);
        var brutePairs = new HashSet<(int, int)>();
        for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) if (Dist(x, i, x, j, d, p) <= r) brutePairs.Add((i, j));
        Assert.True(brutePairs.SetEquals(pairs));

        var radii = new[] { r / 4, r, r / 2, 2 * r, 0 };
        var counts = ty.CountNeighbors(tx, radii, p);
        for (var k = 0; k < radii.Length; k++)
        {
            long c = 0;
            for (var i = 0; i < 150; i++) for (var j = 0; j < n; j++) if (Dist(y, i, x, j, d, p) <= radii[k]) c++;
            Assert.Equal(c, counts[k]);
        }

        var bt = ty.QueryBallTree(tx, r, p);
        var sdm = ty.SparseDistanceMatrix(tx, r, p);
        Assert.Equal(bt.Sum(b => b.Length), sdm.Count);
        foreach (var (i, j, dist) in sdm) Close(Dist(y, i, x, j, d, p), dist, 0, "sdm");
    }

    [Fact]
    public void Approximate_EpsRespectsItsBound()
    {
        const int n = 2000, d = 4;
        var x = Lcg(n, d, 1); var q = Lcg(100, d, 2);
        var tree = new KDTree(new NdArray(x, n, d));
        var (exact, _) = tree.Query(new NdArray(q, 100, d), 3);
        var (approx, _) = tree.Query(new NdArray(q, 100, d), 3, eps: 0.5);
        var e = exact.ToArray(); var a = approx.ToArray();
        for (var i = 0; i < e.Length; i++) Assert.True(a[i] <= 1.5 * e[i] + 1e-15 && a[i] >= e[i] - 1e-15);
    }

    [Fact]
    public void DuplicatePoints_FormALeaf_AndParallelEqualsSerial()
    {
        var x = new double[5000 * 3]; // all identical but a few
        for (var i = 0; i < 30; i++) x[3 * i] = i;
        var tree = new KDTree(new NdArray(x, 5000, 3), leafSize: 4, balancedTree: false);
        var (d, idx) = tree.Query(new double[] { 0, 0, 0 }, 3);
        Assert.Equal(new[] { 0, 30, 31 }, idx);
        Assert.Equal(new[] { 0.0, 0, 0 }, d);

        var pts = LcgNd(3000, 3, 4);
        var tq = new KDTree(pts);
        var qs = LcgNd(1000, 3, 5);
        var (ds, ix) = tq.Query(qs, 4, workers: 1);
        var (dp, ixp) = tq.Query(qs, 4, workers: -1);
        Assert.Equal(ds.ToArray(), dp.ToArray());
        Assert.Equal(ix, ixp);
    }
}
