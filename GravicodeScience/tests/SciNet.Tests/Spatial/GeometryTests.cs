using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

public class ConvexHullTests
{
    [Fact]
    public void Square_WithInteriorAndEdgePoints_HasFourCcwVertices()
    {
        var pts = new double[,] { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0.5, 0.5 }, { 0.5, 0 }, { 0.2, 0.7 }, { 1, 0.5 } };
        var h = new ConvexHull(pts);
        Assert.Equal(new[] { 0, 1, 2, 3 }, h.Vertices);
        Close(4.0, h.Area, 1e-15, "perimeter");
        Close(1.0, h.Volume, 1e-15, "area");
        var eq = h.Equations.ToArray();
        for (var f = 0; f < h.Equations.Shape[0]; f++)
            for (var i = 0; i < 8; i++)
                Assert.True(eq[3 * f] * pts[i, 0] + eq[3 * f + 1] * pts[i, 1] + eq[3 * f + 2] <= 1e-15);
    }

    [Fact]
    public void Random2D_MatchesScipy()
    {
        // scipy 1.17.1: ConvexHull(lcg_points(60, 2, 7))
        var h = new ConvexHull(LcgNd(60, 2, 7));
        Assert.Equal(new[] { 10, 16, 24, 29, 35, 36, 41, 42, 52, 55 }, h.Vertices.OrderBy(v => v).ToArray());
        Close(3.4806601299657136, h.Area, 1e-12, "perimeter");
        Close(0.845708698424943, h.Volume, 1e-12, "area");
        // Counter-clockwise: positive signed area of consecutive vertices.
        var p = LcgNd(60, 2, 7).ToArray();
        for (var k = 0; k < h.Vertices.Length; k++)
        {
            int a = h.Vertices[k], b = h.Vertices[(k + 1) % h.Vertices.Length], c = h.Vertices[(k + 2) % h.Vertices.Length];
            var cross = (p[2 * b] - p[2 * a]) * (p[2 * c + 1] - p[2 * a + 1]) - (p[2 * b + 1] - p[2 * a + 1]) * (p[2 * c] - p[2 * a]);
            Assert.True(cross > 0);
        }
    }

    [Fact]
    public void Collinear2D_Throws()
        => Assert.Throws<ArgumentException>(() => new ConvexHull(new double[,] { { 0, 0 }, { 1, 1 }, { 2, 2 }, { 3, 3 } }));

    [Fact]
    public void Cube_WithInteriorAndFacePoints()
    {
        var list = new List<double[]>();
        for (var i = 0; i < 8; i++) list.Add([i & 1, (i >> 1) & 1, (i >> 2) & 1]);
        list.Add([0.5, 0.5, 0.5]); list.Add([0.5, 0.5, 0]); list.Add([1, 0.3, 0.6]); list.Add([0.5, 0, 0]);
        var h = new ConvexHull(list.ToArray());
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 5, 6, 7 }, h.Vertices);
        Close(6.0, h.Area, 1e-14, "area");
        Close(1.0, h.Volume, 1e-14, "volume");
        CheckClosedSurface(h, list.Select(r => r).ToArray());
    }

    [Fact]
    public void Random3D_MatchesScipy()
    {
        // scipy 1.17.1: ConvexHull(lcg_points(80, 3, 11))
        var pts = LcgNd(80, 3, 11);
        var h = new ConvexHull(pts);
        Assert.Equal(new[] { 2, 8, 10, 11, 12, 14, 17, 20, 24, 29, 33, 37, 42, 46, 50, 51, 54, 56, 57, 58, 59, 64, 70, 72, 73, 75, 77, 78 }, h.Vertices);
        Close(3.888563956242059, h.Area, 1e-12, "area");
        Close(0.6245388120261417, h.Volume, 1e-12, "volume");
        Assert.Equal(52, h.Simplices.GetLength(0));
        var p = pts.ToArray();
        CheckClosedSurface(h, Enumerable.Range(0, 80).Select(i => p.AsSpan(3 * i, 3).ToArray()).ToArray());
    }

    [Fact]
    public void LatticePoints_ManyCoplanarDegeneracies()
    {
        var list = new List<double[]>();
        for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) for (var k = 0; k < 4; k++) list.Add([i, j, k]);
        var h = new ConvexHull(list.ToArray());
        Assert.Equal(8, h.Vertices.Length);
        Close(54.0, h.Area, 1e-12, "area");
        Close(27.0, h.Volume, 1e-12, "volume");
        CheckClosedSurface(h, list.ToArray());
    }

    [Fact]
    public void Coplanar3D_Throws()
        => Assert.Throws<ArgumentException>(() => new ConvexHull(new double[,] { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 2, 3, 0 } }));

    private static void CheckClosedSurface(ConvexHull h, double[][] pts)
    {
        var f = h.Simplices.GetLength(0);
        var eq = h.Equations.ToArray();
        // Euler: a closed triangulated sphere has F = 2V - 4.
        Assert.Equal(2 * h.Vertices.Length - 4, f);
        for (var i = 0; i < f; i++)
        {
            var n = Math.Sqrt(eq[4 * i] * eq[4 * i] + eq[4 * i + 1] * eq[4 * i + 1] + eq[4 * i + 2] * eq[4 * i + 2]);
            Close(1.0, n, 1e-12, "unit normal");
            foreach (var p in pts)
                Assert.True(eq[4 * i] * p[0] + eq[4 * i + 1] * p[1] + eq[4 * i + 2] * p[2] + eq[4 * i + 3] <= 1e-12, "point outside a facet");
            for (var k = 0; k < 3; k++)
            {
                var nb = h.Neighbors[i, k];
                // The neighbour shares the edge opposite vertex k, and points back.
                int a = h.Simplices[i, (k + 1) % 3], b = h.Simplices[i, (k + 2) % 3];
                var verts = new[] { h.Simplices[nb, 0], h.Simplices[nb, 1], h.Simplices[nb, 2] };
                Assert.Contains(a, verts); Assert.Contains(b, verts);
                Assert.Contains(i, new[] { h.Neighbors[nb, 0], h.Neighbors[nb, 1], h.Neighbors[nb, 2] });
            }
        }
    }
}

public class DelaunayTests
{
    private static double Orient(double[] p, int a, int b, int c)
        => (p[2 * b] - p[2 * a]) * (p[2 * c + 1] - p[2 * a + 1]) - (p[2 * b + 1] - p[2 * a + 1]) * (p[2 * c] - p[2 * a]);

    /// <summary>Naive empty-circumcircle check with a relative tolerance for cocircular points.</summary>
    private static void CheckDelaunay(Delaunay tri, double[] p, int n)
    {
        var s = tri.Simplices;
        double area = 0;
        for (var t = 0; t < tri.NSimplex; t++)
        {
            int a = s[t, 0], b = s[t, 1], c = s[t, 2];
            var o = Orient(p, a, b, c);
            Assert.True(o > 0, "triangle not counter-clockwise");
            area += o / 2;
            // Circumcentre and radius.
            double ax = p[2 * a], ay = p[2 * a + 1], bx = p[2 * b], by = p[2 * b + 1], cx = p[2 * c], cy = p[2 * c + 1];
            var d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
            var ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
            var uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
            var r2 = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy);
            for (var i = 0; i < n; i++)
            {
                var d2 = (p[2 * i] - ux) * (p[2 * i] - ux) + (p[2 * i + 1] - uy) * (p[2 * i + 1] - uy);
                Assert.True(d2 >= r2 * (1 - 1e-9), $"point {i} inside circumcircle of {t}");
            }
            for (var k = 0; k < 3; k++)
            {
                var nb = tri.Neighbors[t, k];
                if (nb < 0) continue;
                Assert.Contains(t, new[] { tri.Neighbors[nb, 0], tri.Neighbors[nb, 1], tri.Neighbors[nb, 2] });
            }
        }
        // Total area equals the hull area.
        var hull = new ConvexHull(new NdArray((double[])p.Clone(), n, 2));
        Close(hull.Volume, area, 1e-9 * Math.Max(1, hull.Volume), "triangulated area");
    }

    [Fact]
    public void RandomPoints_AreDelaunayAndSatisfyEuler()
    {
        var p = Lcg(300, 2, 3);
        var tri = new Delaunay(new NdArray(p, 300, 2));
        CheckDelaunay(tri, p, 300);
        var hull = new ConvexHull(new NdArray((double[])p.Clone(), 300, 2));
        // No three of these points are collinear on the hull, so T = 2n - 2 - h.
        Assert.Equal(2 * 300 - 2 - hull.Vertices.Length, tri.NSimplex);
        Assert.Equal(hull.Vertices.Length, tri.ConvexHull.GetLength(0));
    }

    [Fact]
    public void Grid_CocircularInput_TriangulatesCleanly()
    {
        const int m = 12;
        var p = new double[m * m * 2];
        for (var i = 0; i < m; i++) for (var j = 0; j < m; j++) { p[2 * (i * m + j)] = i; p[2 * (i * m + j) + 1] = j; }
        var tri = new Delaunay(new NdArray(p, m * m, 2));
        Assert.Equal(2 * (m - 1) * (m - 1), tri.NSimplex);
        CheckDelaunay(tri, p, m * m);
        // Every hull edge is a unit lattice step: 4(m-1) of them.
        Assert.Equal(4 * (m - 1), tri.ConvexHull.GetLength(0));
    }

    [Fact]
    public void CocircularRing_AndCentre()
    {
        const int k = 16;
        var p = new double[(k + 1) * 2];
        for (var i = 0; i < k; i++) { p[2 * i] = Math.Round(Math.Cos(2 * Math.PI * i / k), 12); p[2 * i + 1] = Math.Round(Math.Sin(2 * Math.PI * i / k), 12); }
        var tri = new Delaunay(new NdArray(p, k + 1, 2));
        Assert.Equal(k, tri.NSimplex);
        CheckDelaunay(tri, p, k + 1);
    }

    [Fact]
    public void Duplicates_GoToCoplanar_AndCollinearThrows()
    {
        var tri = new Delaunay(new double[,] { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
        Assert.Equal(2, tri.NSimplex);
        Assert.Equal(1, tri.Coplanar.GetLength(0));
        Assert.Equal(3, tri.Coplanar[0, 0]);
        Assert.Equal(1, tri.Coplanar[0, 2]);
        Assert.Equal(-1, tri.VertexToSimplex[3]);
        Assert.Throws<ArgumentException>(() => new Delaunay(new double[,] { { 0, 0 }, { 1, 1 }, { 2, 2 }, { 0, 0 } }));
    }

    [Fact]
    public void FindSimplex_AgreesWithBarycentricBruteForce()
    {
        var p = Lcg(200, 2, 5);
        var tri = new Delaunay(new NdArray(p, 200, 2));
        var q = Lcg(1000, 2, 99);
        for (var i = 0; i < q.Length; i++) q[i] = q[i] * 1.2 - 0.1; // some outside the hull
        var found = tri.FindSimplex(new NdArray(q, 1000, 2));
        var brute = tri.FindSimplex(new NdArray(q, 1000, 2), bruteForce: true);
        for (var i = 0; i < 1000; i++)
        {
            if (found[i] < 0) { Assert.Equal(-1, brute[i]); continue; }
            var c = tri.Barycentric(found[i], q[2 * i], q[2 * i + 1]);
            Assert.True(c.All(v => v >= -1e-12), $"query {i} not inside its simplex");
            Assert.True(brute[i] >= 0);
        }
        // Vertices themselves are found.
        Assert.True(tri.FindSimplex(p[0], p[1]) >= 0);
        Assert.Equal(-1, tri.FindSimplex(5.0, 5.0));
    }

    [Fact]
    public void Transform_MatchesScipyLayout()
    {
        var tri = new Delaunay(new double[,] { { 0, 0 }, { 2, 0 }, { 0, 2 } });
        var t = tri.Transform.ToArray();
        // c = T^-1 (x - r): the last vertex maps to (0, 0).
        var s = tri.Simplices;
        var last = s[0, 2];
        var pts = new double[,] { { 0, 0 }, { 2, 0 }, { 0, 2 } };
        Close(pts[last, 0], t[4], 0, "r_x");
        Close(pts[last, 1], t[5], 0, "r_y");
        var c = tri.Barycentric(0, pts[s[0, 0], 0], pts[s[0, 0], 1]);
        Close(new double[] { 1, 0, 0 }, c, 1e-15, "vertex 0 barycentric");
    }

    [Fact]
    public void VertexNeighborVertices_IsSymmetric()
    {
        var p = Lcg(50, 2, 8);
        var tri = new Delaunay(new NdArray(p, 50, 2));
        var (indptr, indices) = tri.VertexNeighborVertices();
        for (var v = 0; v < 50; v++)
            for (var k = indptr[v]; k < indptr[v + 1]; k++)
            {
                var w = indices[k];
                Assert.Contains(v, indices[indptr[w]..indptr[w + 1]]);
            }
        // Sum of degrees = 2 * edges = 2 * (3T + h)/2.
        Assert.Equal(3 * tri.NSimplex + tri.ConvexHull.GetLength(0), indices.Length);
    }
}
