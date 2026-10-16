using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class VoronoiTests
{
    [Fact]
    public void SmallSet_MatchesScipyRidgesAndVertices()
    {
        // scipy 1.17.1: Voronoi(lcg_points(12, 2, 21)); ridges as sorted (i, j, unbounded).
        var expected = new (int, int, bool)[]
        {
            (0, 5, false), (0, 8, false), (0, 9, false), (0, 10, false), (1, 3, false), (1, 6, false), (1, 7, true),
            (1, 9, true), (1, 10, false), (2, 4, false), (2, 6, false), (2, 7, false), (2, 8, false), (2, 11, false),
            (3, 4, false), (3, 6, false), (3, 8, false), (3, 10, false), (4, 6, false), (4, 8, false), (5, 8, false),
            (5, 9, false), (5, 11, false), (6, 7, false), (7, 11, true), (8, 10, false), (8, 11, false), (9, 10, false), (9, 11, true),
        };
        var xs = new[] { -0.2144740581, 0.0133728452, 0.1232926952, 0.2553780169, 0.2611001399, 0.2989694688, 0.337998673, 0.3705246379, 0.4120101434, 0.4611507995, 0.5112846816, 0.5864745037, 0.5898257727, 0.6229207977, 0.6784396696, 0.6791137095, 1.2404403577, 2.226666837 };
        var ys = new[] { -0.1838236719, 0.2390359737, 0.3372660645, 0.369152733, 0.4161684888, 0.4298897288, 0.5180094943, 0.5193350374, 0.5599573606, 0.587765141, 0.6139256509, 0.648484557, 0.684212304, 0.7190717647, 0.791239054, 0.9122814926, 1.0541291356, 2.0456452126 };

        var vor = new Voronoi(LcgNd(12, 2, 21));
        var ridges = new List<(int, int, bool)>();
        for (var r = 0; r < vor.RidgePoints.GetLength(0); r++)
            ridges.Add((vor.RidgePoints[r, 0], vor.RidgePoints[r, 1], vor.RidgeVertices[r, 0] == -1 || vor.RidgeVertices[r, 1] == -1));
        Assert.Equal(expected, ridges.OrderBy(t => t.Item1).ThenBy(t => t.Item2).ToArray());

        var v = vor.Vertices.ToArray();
        Assert.Equal(18, v.Length / 2);
        Close(xs, Enumerable.Range(0, 18).Select(i => v[2 * i]).OrderBy(x => x).ToArray(), 1e-9, "x");
        Close(ys, Enumerable.Range(0, 18).Select(i => v[2 * i + 1]).OrderBy(x => x).ToArray(), 1e-9, "y");
    }

    [Fact]
    public void Grid_MergesCocircularVertices()
    {
        // scipy 1.17.1 on the 4x4 lattice: 9 vertices at the cell centres, 24 ridges.
        var g = new double[16, 2];
        for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) { g[4 * i + j, 0] = i; g[4 * i + j, 1] = j; }
        var vor = new Voronoi(g);
        Assert.Equal(9, vor.Vertices.Shape[0]);
        Assert.Equal(24, vor.RidgePoints.GetLength(0));
        // Interior point (1,1) -> index 5 has a bounded square region.
        Assert.Equal(4, vor.Regions[vor.PointRegion[5]].Length);
        Assert.DoesNotContain(-1, vor.Regions[vor.PointRegion[5]]);
        Assert.Contains(-1, vor.Regions[vor.PointRegion[0]]);
    }

    [Fact]
    public void FinitePolygons_TileTheBox()
    {
        var pts = LcgNd(40, 2, 17);
        var vor = new Voronoi(pts);
        var polys = vor.FinitePolygons(-0.5, 1.5, -0.5, 1.5);
        var p = pts.ToArray();
        double total = 0;
        for (var i = 0; i < 40; i++)
        {
            var q = polys[i].ToArray();
            var k = q.Length / 2;
            double a = 0;
            for (var j = 0; j < k; j++) a += q[2 * j] * q[2 * ((j + 1) % k) + 1] - q[2 * ((j + 1) % k)] * q[2 * j + 1];
            Assert.True(a > 0, "polygon not counter-clockwise");
            total += a / 2;
            // Every polygon vertex is at least as close to its own site as to any other site.
            for (var j = 0; j < k; j++)
            {
                var own = Math.Pow(q[2 * j] - p[2 * i], 2) + Math.Pow(q[2 * j + 1] - p[2 * i + 1], 2);
                for (var o = 0; o < 40; o++)
                    Assert.True(own <= Math.Pow(q[2 * j] - p[2 * o], 2) + Math.Pow(q[2 * j + 1] - p[2 * o + 1], 2) + 1e-9);
            }
        }
        Close(4.0, total, 1e-9, "cells tile the box");
    }

    [Fact]
    public void Regions_AreCounterClockwiseAroundTheirSite()
    {
        var pts = LcgNd(30, 2, 4);
        var vor = new Voronoi(pts);
        var v = vor.Vertices.ToArray();
        var p = pts.ToArray();
        for (var i = 0; i < 30; i++)
        {
            var reg = vor.Regions[vor.PointRegion[i]].Where(x => x >= 0).ToArray();
            for (var j = 0; j + 1 < reg.Length; j++)
            {
                int a = reg[j], b = reg[j + 1];
                var cross = (v[2 * a] - p[2 * i]) * (v[2 * b + 1] - p[2 * i + 1]) - (v[2 * a + 1] - p[2 * i + 1]) * (v[2 * b] - p[2 * i]);
                Assert.True(cross > 0, $"region {i} not counter-clockwise");
            }
        }
    }
}
