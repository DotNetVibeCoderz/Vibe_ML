using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class KDTreePeriodicTests
{
    private static NdArray Queries()
    {
        var q = Lcg(5, 3, 9);
        for (var i = 0; i < q.Length; i++) q[i] = q[i] * 1.2 - 0.1; // some outside [0, 1): wrapped
        return new NdArray(q, 5, 3);
    }

    [Fact]
    public void Query_MatchesScipy()
    {
        // scipy 1.17.1: KDTree(lcg_points(200, 3, 4), boxsize=1.0).query(q, k=3)
        var t = new KDTree(LcgNd(200, 3, 4), [1.0]);
        var (d, i) = t.Query(Queries(), 3);
        Close([0.1085006119596501, 0.11921163934561506, 0.12677533374195352, 0.12955748098065079, 0.1682760555205246, 0.17116351893494855, 0.09803488571088394, 0.14746649494138564, 0.15390020049960146, 0.05043019103666733, 0.060176825897202496, 0.08223708696384462, 0.09238648142986208, 0.09567056504107563, 0.16172196848732223],
            d.ToArray(), 1e-14, "distances");
        Assert.Equal(new[] { 18, 180, 92, 14, 35, 63, 171, 75, 162, 73, 110, 197, 31, 119, 182 }, i.Cast<int>().ToArray());
    }

    [Fact]
    public void BallAndPairs_MatchScipy()
    {
        var t = new KDTree(LcgNd(200, 3, 4), [1.0]);
        // scipy 1.17.1: [len(x) for x in query_ball_point(q, 0.2)], query_ball_point(q[0], 0.15)
        Assert.Equal(new[] { 10, 5, 7, 13, 10 }, t.QueryBallPoint(Queries(), 0.2).Select(x => x.Length).ToArray());
        Assert.Equal(new[] { 10L, 5, 7, 13, 10 }, t.QueryBallPointLength(Queries(), 0.2));
        Assert.Equal(new[] { 18, 92, 112, 180 }, t.QueryBallPoint(Queries().ToArray().AsSpan(0, 3), 0.15, returnSorted: true));
        Assert.Equal(99, t.QueryPairs(0.1).Count);
    }

    [Fact]
    public void BruteForceAgreement_AnisotropicBox()
    {
        // Box (1, 2) in 2-D; scipy 1.17.1: query([[0.01, 1.99]], k=2) distances.
        var p = Lcg(100, 2, 3);
        for (var i = 0; i < 100; i++) p[2 * i + 1] *= 2;
        var t = new KDTree(new NdArray(p, 100, 2), [1.0, 2.0]);
        var (d, _) = t.Query(new NdArray(new[] { 0.01, 1.99 }, 1, 2), 2);
        Close([0.07133313271211847, 0.09464070214043789], d.ToArray(), 1e-14, "corner query");
        // Brute force minimum-image k-NN for many queries.
        var q = Lcg(300, 2, 11);
        for (var i = 0; i < 300; i++) q[2 * i + 1] *= 2;
        var (dd, ii) = t.Query(new NdArray(q, 300, 2), 4);
        var da = dd.ToArray();
        for (var i = 0; i < 300; i++)
        {
            var all = Enumerable.Range(0, 100).Select(j =>
            {
                var dx = Math.Abs(q[2 * i] - p[2 * j]); dx = Math.Min(dx, 1 - dx);
                var dy = Math.Abs(q[2 * i + 1] - p[2 * j + 1]); dy = Math.Min(dy, 2 - dy);
                return Math.Sqrt(dx * dx + dy * dy);
            }).OrderBy(x => x).Take(4).ToArray();
            Close(all, da.AsSpan(4 * i, 4).ToArray(), 1e-14, $"query {i}");
        }
    }

    [Fact]
    public void CountNeighbors_PeriodicAndWeighted_MatchScipy()
    {
        var t = new KDTree(LcgNd(200, 3, 4), [1.0]);
        var t2 = new KDTree(LcgNd(150, 3, 6), [1.0]);
        double[] r = [0.05, 0.1, 0.3];
        // scipy 1.17.1: t.count_neighbors(t2, r) and with weights=(w1, w2)
        Assert.Equal(new[] { 10L, 119, 3437 }, t.CountNeighbors(t2, r));
        var w1 = Lcg(200, 1, 1); var w2 = Lcg(150, 1, 2);
        Close([3.8516310192738175, 34.053162736119674, 964.8861916374738], t.CountNeighbors(t2, r, w1, w2), 1e-10, "periodic weighted");
        var tn = new KDTree(LcgNd(200, 3, 4)); var tn2 = new KDTree(LcgNd(150, 3, 6));
        Close([3.0957125255650717, 30.833265370181707, 695.6669268356367], tn.CountNeighbors(tn2, r, w1, w2), 1e-10, "weighted");
        Close([4.844708703907265, 60.70740481662835, 1301.0109923053894], tn.CountNeighbors(tn2, r, w1, null), 1e-10, "self weights only");
        Close([5.554511841710802, 49.405316278604985, 1260.5506145698077], tn.CountNeighbors(tn2, r, null, w2, cumulative: false), 1e-10, "bins");
        // Unit weights reproduce the integer counts of the fast path.
        var ones = tn.CountNeighbors(tn2, r, null, null);
        var ints = tn.CountNeighbors(tn2, r);
        for (var i = 0; i < 3; i++) Assert.Equal(ints[i], (long)ones[i]);
    }

    [Fact]
    public void Construction_ValidatesTheBox()
    {
        var p = Lcg(20, 3, 4);
        p[0] = 1.5;
        Assert.Throws<ArgumentException>(() => new KDTree(new NdArray(p, 20, 3), [1.0]));
        Assert.Throws<ArgumentException>(() => new KDTree(LcgNd(20, 3, 4), [1.0, 0.0, 1.0]));
        var open = new KDTree(LcgNd(20, 3, 4), [1.0, double.PositiveInfinity, 1.0]);
        Assert.Equal(new[] { 1.0, double.PositiveInfinity, 1.0 }, open.BoxSize);
        Assert.Null(new KDTree(LcgNd(20, 3, 4)).BoxSize);
    }
}
