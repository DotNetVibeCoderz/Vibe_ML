using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class HalfspaceTests
{
    [Fact]
    public void SquarePyramid_ApexMergesFourDualFacets()
    {
        // scipy 1.17.1: 4 base corners and the apex; the apex is where 4 halfspaces meet, and
        // z <= 5 is redundant.
        var h = new double[,] { { 0, 0, -1, 0 }, { 1, 0, 1, -1 }, { -1, 0, 1, -1 }, { 0, 1, 1, -1 }, { 0, -1, 1, -1 }, { 0, 0, 1, -5 } };
        var hs = new HalfspaceIntersection(h, [0, 0, 0.3]);
        var v = hs.Intersections.ToArray();
        Assert.Equal(5, v.Length / 3);
        var keys = new List<string>();
        for (var i = 0; i < 5; i++) keys.Add($"{v[3 * i]:F6},{v[3 * i + 1]:F6},{v[3 * i + 2]:F6}");
        keys.Sort(string.CompareOrdinal);
        Assert.Equal(new[] { "-1.000000,-1.000000,0.000000", "-1.000000,1.000000,0.000000", "0.000000,0.000000,1.000000", "1.000000,-1.000000,0.000000", "1.000000,1.000000,0.000000" }, keys);
        var facets = hs.DualFacets.Select(f => string.Join(",", f)).OrderBy(s => s, StringComparer.Ordinal).ToArray();
        Assert.Equal(new[] { "0,1,3", "0,1,4", "0,2,3", "0,2,4", "1,2,3,4" }, facets);
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, hs.DualVertices);
    }

    [Fact]
    public void TangentPlanesOfSphere_MatchScipy()
    {
        // scipy 1.17.1: n = lcg_points(20, 3, 6) - 0.5, normalised; halfspaces n·x - 1 <= 0.
        var p = Lcg(20, 3, 6);
        var h = new double[20 * 4];
        for (var i = 0; i < 20; i++)
        {
            double s = 0;
            for (var j = 0; j < 3; j++) { p[3 * i + j] -= 0.5; s += p[3 * i + j] * p[3 * i + j]; }
            s = Math.Sqrt(s);
            for (var j = 0; j < 3; j++) h[4 * i + j] = p[3 * i + j] / s;
            h[4 * i + 3] = -1;
        }
        var hs = new HalfspaceIntersection(new NdArray(h, 20, 4), [0, 0, 0]);
        var v = hs.Intersections.ToArray();
        Assert.Equal(36, v.Length / 3);
        double sx = 0, sy = 0, sz = 0, s2 = 0;
        for (var i = 0; i < 36; i++) { sx += v[3 * i]; sy += v[3 * i + 1]; sz += v[3 * i + 2]; s2 += v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]; }
        Close(0.6632306409988565, sx, 1e-9, "sum x");
        Close(3.562304771338622, sy, 1e-9, "sum y");
        Close(-4.177581358193916, sz, 1e-9, "sum z");
        Close(54.44481440462794, s2, 1e-8, "sum of squares");
        Close(2.509336538512847, hs.DualVolume, 1e-12, "dual volume");
        Close(9.645123386431822, hs.DualArea, 1e-12, "dual area");
        Assert.Equal(Enumerable.Range(0, 20).ToArray(), hs.DualVertices);
        // Every vertex satisfies every halfspace, with exactly its dual facet active.
        for (var k = 0; k < 36; k++)
            for (var i = 0; i < 20; i++)
            {
                var s = h[4 * i + 3];
                for (var j = 0; j < 3; j++) s += h[4 * i + j] * v[3 * k + j];
                Assert.True(s <= 1e-12);
                Assert.Equal(hs.DualFacets[k].Contains(i), Math.Abs(s) < 1e-9);
            }
    }

    [Fact]
    public void Square2D_AndBadInteriorPoint()
    {
        var h = new double[,] { { -1, 0, 0 }, { 1, 0, -1 }, { 0, -1, 0 }, { 0, 1, -1 }, { 1, 1, -5 } };
        var hs = new HalfspaceIntersection(h, [0.5, 0.5]);
        Assert.Equal(4, hs.Intersections.Shape[0]);
        Assert.Equal(new[] { 0, 1, 2, 3 }, hs.DualVertices);
        Assert.Throws<ArgumentException>(() => new HalfspaceIntersection(h, [1.0, 0.5]));
    }
}
