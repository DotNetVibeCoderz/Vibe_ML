using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class VoronoiNdTests
{
    [Fact]
    public void Random3D_MatchesScipy()
    {
        // scipy 1.17.1: v = Voronoi(lcg_points(16, 3, 5)); ridges as "i-j:<finite count>[u]".
        const string expected = "0-10:4 0-11:2u 0-2:5u 0-3:3u 0-4:2u 0-5:6 0-6:4 0-7:4u 0-8:5 1-10:5 1-13:5 1-14:4 1-15:3u 1-3:4u 1-7:2u 1-8:3 1-9:4u 10-12:5 10-13:4 10-14:6 10-15:5 11-15:2u 12-14:3 12-15:4u 13-14:4 13-15:4 14-15:6 2-10:4 2-11:2u 2-15:5u 2-5:4 2-6:4 2-7:3 3-10:4 3-13:4 3-4:2u 3-5:3 3-6:4 3-7:4u 3-8:7u 4-11:2u 4-12:2u 4-15:2u 4-5:6 4-8:2u 5-10:5 5-11:4 5-12:4 5-15:5 5-8:5 6-10:7 6-13:5 6-14:3 6-15:5 6-7:6 6-8:4 7-13:4 7-15:4u 7-8:3 8-10:7 8-12:3u 8-9:3u 9-10:4 9-12:3u 9-14:4 9-15:2u";
        var pts = LcgNd(16, 3, 5);
        var vor = new Voronoi(pts);
        Assert.Equal(3, vor.NDim);
        var v = vor.Vertices.ToArray();
        Assert.Equal(44, v.Length / 3);
        double sx = 0, sy = 0, sz = 0;
        for (var i = 0; i < 44; i++) { sx += v[3 * i]; sy += v[3 * i + 1]; sz += v[3 * i + 2]; }
        Close(34.020784510324326, sx, 1e-9, "sum x");
        Close(-7.482171899988727, sy, 1e-9, "sum y");
        Close(28.36777634664042, sz, 1e-9, "sum z");
        var items = new List<string>();
        for (var r = 0; r < vor.RidgePoints.GetLength(0); r++)
        {
            var rv = vor.RidgeVertexLists[r];
            int a = vor.RidgePoints[r, 0], b = vor.RidgePoints[r, 1];
            items.Add($"{Math.Min(a, b)}-{Math.Max(a, b)}:{rv.Count(x => x >= 0)}{(rv.Contains(-1) ? "u" : "")}");
        }
        items.Sort(string.CompareOrdinal);
        Assert.Equal(expected, string.Join(' ', items));
        // scipy 1.17.1: region sizes (including -1) and boundedness per point.
        int[] sizes = [13, 11, 10, 13, 7, 14, 14, 11, 15, 8, 20, 5, 9, 10, 10, 17];
        bool[] unbounded = [true, true, true, true, true, false, false, true, true, true, false, true, true, false, false, true];
        for (var i = 0; i < 16; i++)
        {
            var region = vor.Regions[vor.PointRegion[i]];
            Assert.Equal(sizes[i], region.Length);
            Assert.Equal(unbounded[i], region.Contains(-1));
        }
        Assert.Throws<InvalidOperationException>(() => vor.RidgeVertices);
    }

    [Fact]
    public void Random3D_VerticesAreEmptySphereCentres_RidgesLieOnBisectorsInCyclicOrder()
    {
        var n = 40;
        var p = Lcg(n, 3, 13);
        var vor = new Voronoi(new NdArray(p, n, 3));
        var v = vor.Vertices.ToArray();
        // Every region vertex is equidistant from its site and no site is closer.
        for (var i = 0; i < n; i++)
            foreach (var k in vor.Regions[vor.PointRegion[i]])
            {
                if (k < 0) continue;
                var di = Dist2(v, k, p, i);
                for (var j = 0; j < n; j++) Assert.True(Dist2(v, k, p, j) >= di * (1 - 1e-9), $"site {j} closer to vertex {k} than {i}");
            }
        for (var r = 0; r < vor.RidgePoints.GetLength(0); r++)
        {
            int a = vor.RidgePoints[r, 0], b = vor.RidgePoints[r, 1];
            var rv = vor.RidgeVertexLists[r].Where(x => x >= 0).ToArray();
            foreach (var k in rv) Close(Dist2(v, k, p, a), Dist2(v, k, p, b), 1e-9 * (1 + Dist2(v, k, p, a)), "bisector");
            if (vor.RidgeVertexLists[r][0] == -1 || rv.Length < 4) continue;
            // A bounded ridge is a convex polygon: listed cyclically, consecutive edge normals
            // (taken along the Delaunay edge axis) all turn the same way.
            double ax = p[3 * b] - p[3 * a], ay = p[3 * b + 1] - p[3 * a + 1], az = p[3 * b + 2] - p[3 * a + 2];
            var signs = new HashSet<int>();
            for (var t = 0; t < rv.Length; t++)
            {
                int u = rv[t], w = rv[(t + 1) % rv.Length], z = rv[(t + 2) % rv.Length];
                double e1x = v[3 * w] - v[3 * u], e1y = v[3 * w + 1] - v[3 * u + 1], e1z = v[3 * w + 2] - v[3 * u + 2];
                double e2x = v[3 * z] - v[3 * w], e2y = v[3 * z + 1] - v[3 * w + 1], e2z = v[3 * z + 2] - v[3 * w + 2];
                var turn = ax * (e1y * e2z - e1z * e2y) + ay * (e1z * e2x - e1x * e2z) + az * (e1x * e2y - e1y * e2x);
                if (Math.Abs(turn) > 1e-12) signs.Add(Math.Sign(turn));
            }
            Assert.True(signs.Count <= 1, $"ridge {r} is not in cyclic order");
        }
    }

    private static double Dist2(double[] v, int k, double[] p, int i)
    {
        double s = 0;
        for (var j = 0; j < 3; j++) s += (v[3 * k + j] - p[3 * i + j]) * (v[3 * k + j] - p[3 * i + j]);
        return s;
    }

    [Fact]
    public void CubeCorners_OneVertex_NoRidges()
    {
        // scipy 1.17.1: Voronoi(cube corners) -> vertices [[.5,.5,.5]], no ridges, regions [-1, 0].
        var vor = new Voronoi(new NdArray(NdGeometry.Grid(2, 3), 8, 3));
        Close([0.5, 0.5, 0.5], vor.Vertices.ToArray(), 1e-15, "centre");
        Assert.Equal(0, vor.RidgePoints.GetLength(0));
        for (var i = 0; i < 8; i++) Assert.Equal(new[] { -1, 0 }, vor.Regions[vor.PointRegion[i]]);
    }

    [Fact]
    public void CubeWithCentre_MatchesScipy()
    {
        // scipy 1.17.1: 8 corners + centre -> 6 vertices (centre ± 0.75 along each axis), 20 ridges;
        // the centre's region is bounded with all 6 vertices, every corner's is unbounded with 3.
        var p = new List<double>(NdGeometry.Grid(2, 3)) { 0.5, 0.5, 0.5 };
        var vor = new Voronoi(new NdArray(p.ToArray(), 9, 3));
        var v = vor.Vertices.ToArray();
        Assert.Equal(6, v.Length / 3);
        var found = new List<string>();
        for (var i = 0; i < 6; i++) found.Add($"{v[3 * i]:F2},{v[3 * i + 1]:F2},{v[3 * i + 2]:F2}");
        found.Sort(string.CompareOrdinal);
        Assert.Equal(new[] { "-0.25,0.50,0.50", "0.50,-0.25,0.50", "0.50,0.50,-0.25", "0.50,0.50,1.25", "0.50,1.25,0.50", "1.25,0.50,0.50" }, found);
        Assert.Equal(20, vor.RidgePoints.GetLength(0));
        Assert.Equal(6, vor.Regions[vor.PointRegion[8]].Length);
        Assert.DoesNotContain(-1, vor.Regions[vor.PointRegion[8]]);
        for (var i = 0; i < 8; i++)
        {
            var r = vor.Regions[vor.PointRegion[i]];
            Assert.Equal(4, r.Length);
            Assert.Equal(-1, r[0]);
        }
    }

    [Fact]
    public void FourD_RegionsAreConsistent()
    {
        var n = 30;
        var p = Lcg(n, 4, 3);
        var vor = new Voronoi(new NdArray(p, n, 4));
        var v = vor.Vertices.ToArray();
        Assert.Equal(vor.Triangulation.NSimplex, v.Length / 4); // general position: no merging
        for (var i = 0; i < n; i++)
            foreach (var k in vor.Regions[vor.PointRegion[i]])
            {
                if (k < 0) continue;
                double di = 0;
                for (var j = 0; j < 4; j++) di += (v[4 * k + j] - p[4 * i + j]) * (v[4 * k + j] - p[4 * i + j]);
                for (var q = 0; q < n; q++)
                {
                    double dq = 0;
                    for (var j = 0; j < 4; j++) dq += (v[4 * k + j] - p[4 * q + j]) * (v[4 * k + j] - p[4 * q + j]);
                    Assert.True(dq >= di * (1 - 1e-9));
                }
            }
        Assert.All(vor.RidgeVertexLists, rv => Assert.True(rv.Length >= 4));
    }
}
