using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class SphericalVoronoiTests
{
    private static double[] SpherePoints(int n, long seed, double radius, double[] centre)
    {
        var p = Lcg(n, 3, seed);
        for (var i = 0; i < n; i++)
        {
            double s = 0;
            for (var j = 0; j < 3; j++) { p[3 * i + j] -= 0.5; s += p[3 * i + j] * p[3 * i + j]; }
            s = Math.Sqrt(s);
            for (var j = 0; j < 3; j++) p[3 * i + j] = radius * (p[3 * i + j] / s) + centre[j];
        }
        return p;
    }

    [Fact]
    public void Random3D_AreasMatchScipy()
    {
        // scipy 1.17.1: n = lcg_points(30, 3, 8) - 0.5, normalised; p = 2n + [1, 2, 3];
        // SphericalVoronoi(p, radius=2, center=[1, 2, 3]).
        double[] c = [1, 2, 3];
        var p = SpherePoints(30, 8, 2, c);
        var sv = new SphericalVoronoi(new NdArray(p, 30, 3), radius: 2, center: c);
        Assert.Equal(56, sv.Vertices.Shape[0]);
        var a = sv.CalculateAreas();
        Close([1.5757455907304263, 2.299644616717458, 0.8575659271897029, 1.1657135114860369, 1.4194560237702118], a[..5], 1e-10, "areas");
        Close(16 * Math.PI, a.Sum(), 1e-10, "total area");
        Assert.Equal(new[] { 6, 5, 4, 4, 5, 5, 5, 3, 9, 6 }, sv.Regions[..10].Select(r => r.Length).ToArray());
        // Each region's vertices are on the sphere and equidistant from their generator, which
        // is the nearest generator to them.
        var v = sv.Vertices.ToArray();
        for (var i = 0; i < 30; i++)
            foreach (var k in sv.Regions[i])
            {
                double r2 = 0;
                for (var j = 0; j < 3; j++) r2 += (v[3 * k + j] - c[j]) * (v[3 * k + j] - c[j]);
                Close(4.0, r2, 1e-12, "on sphere");
                var di = D2(v, k, p, i);
                for (var q = 0; q < 30; q++) Assert.True(D2(v, k, p, q) >= di - 1e-12);
            }
    }

    private static double D2(double[] v, int k, double[] p, int i)
    {
        double s = 0;
        for (var j = 0; j < 3; j++) s += (v[3 * k + j] - p[3 * i + j]) * (v[3 * k + j] - p[3 * i + j]);
        return s;
    }

    [Fact]
    public void Circle_ArcLengthsAreHalfGapSums()
    {
        // scipy 1.17.1's 2-D calculate_areas returns the arc of every hull *edge* (between
        // consecutive generators), indexed by Qhull facet: [1.3876, 1.6419, 0.4584, 0.3460,
        // 0.9895, 0.8345, 0.6254] here, which are the gaps between the sorted angles. A Voronoi
        // region's length is half the gap on each side; both sum to 2π.
        var t = Lcg(7, 1, 3);
        var p = new double[14];
        for (var i = 0; i < 7; i++) { p[2 * i] = Math.Cos(t[i] * 2 * Math.PI); p[2 * i + 1] = Math.Sin(t[i] * 2 * Math.PI); }
        var sv = new SphericalVoronoi(new NdArray(p, 7, 2));
        var a = sv.CalculateAreas();
        Close(2 * Math.PI, a.Sum(), 1e-12, "circumference");
        var sorted = t.Select((x, i) => (Angle: x * 2 * Math.PI, I: i)).OrderBy(z => z.Angle).ToArray();
        for (var k = 0; k < 7; k++)
        {
            var prev = sorted[(k + 6) % 7].Angle; var next = sorted[(k + 1) % 7].Angle; var cur = sorted[k].Angle;
            var gapPrev = (cur - prev + 2 * Math.PI) % (2 * Math.PI);
            var gapNext = (next - cur + 2 * Math.PI) % (2 * Math.PI);
            Close(0.5 * (gapPrev + gapNext), a[sorted[k].I], 1e-12, $"region {sorted[k].I}");
        }
        Assert.Throws<InvalidOperationException>(() => sv.SortVerticesOfRegions());
    }

    [Fact]
    public void CubeCorners_DuplicateVerticesLikeScipy_AreasEqual()
    {
        var p = NdGeometry.Grid(2, 3);
        for (var i = 0; i < p.Length; i++) p[i] = (2 * p[i] - 1) / Math.Sqrt(3);
        var sv = new SphericalVoronoi(new NdArray(p, 8, 3));
        Assert.Equal(12, sv.Vertices.Shape[0]); // 6 faces x 2 triangles, as scipy
        Assert.All(sv.CalculateAreas(), a => Close(4 * Math.PI / 8, a, 1e-12, "octant"));
    }

    [Fact]
    public void RejectsOffSphereAndDuplicates()
    {
        var p = SpherePoints(10, 2, 1, [0, 0, 0]);
        var off = (double[])p.Clone(); off[0] *= 1.1;
        Assert.Throws<ArgumentException>(() => new SphericalVoronoi(new NdArray(off, 10, 3)));
        var dup = new double[33]; Array.Copy(p, dup, 30); Array.Copy(p, 0, dup, 30, 3);
        Assert.Throws<ArgumentException>(() => new SphericalVoronoi(new NdArray(dup, 11, 3)));
    }
}
