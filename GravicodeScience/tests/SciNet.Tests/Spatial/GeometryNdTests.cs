using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SpatialAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>Shared naive geometry for the N-d hull and Delaunay tests.</summary>
internal static class NdGeometry
{
    /// <summary>Determinant by Gaussian elimination with partial pivoting.</summary>
    public static double Det(double[,] a)
    {
        var n = a.GetLength(0);
        var m = (double[,])a.Clone();
        double det = 1;
        for (var c = 0; c < n; c++)
        {
            var piv = c;
            for (var r = c + 1; r < n; r++) if (Math.Abs(m[r, c]) > Math.Abs(m[piv, c])) piv = r;
            if (m[piv, c] == 0) return 0;
            if (piv != c) { for (var j = 0; j < n; j++) (m[c, j], m[piv, j]) = (m[piv, j], m[c, j]); det = -det; }
            det *= m[c, c];
            for (var r = c + 1; r < n; r++)
            {
                var f = m[r, c] / m[c, c];
                for (var j = c; j < n; j++) m[r, j] -= f * m[c, j];
            }
        }
        return det;
    }

    /// <summary>det[v_1 − v_0, …, v_d − v_0] for the simplex's vertices.</summary>
    public static double SignedVolumeTimesFactorial(double[] p, int d, int[] simplex)
    {
        var a = new double[d, d];
        for (var i = 0; i < d; i++)
            for (var j = 0; j < d; j++) a[i, j] = p[simplex[i + 1] * d + j] - p[simplex[0] * d + j];
        return Det(a);
    }

    /// <summary>Circumcentre by solving 2(v_i − v_0)·c = |v_i|² − |v_0|² (Cramer-free elimination).</summary>
    public static double[] Circumcentre(double[] p, int d, int[] s)
    {
        var a = new double[d, d + 1];
        for (var i = 0; i < d; i++)
        {
            double rhs = 0;
            for (var j = 0; j < d; j++)
            {
                var e = p[s[i + 1] * d + j] - p[s[0] * d + j];
                a[i, j] = 2 * e;
                rhs += e * e;
            }
            a[i, d] = rhs; // relative to v_0
        }
        for (var c = 0; c < d; c++)
        {
            var piv = c;
            for (var r = c + 1; r < d; r++) if (Math.Abs(a[r, c]) > Math.Abs(a[piv, c])) piv = r;
            for (var j = 0; j <= d; j++) (a[c, j], a[piv, j]) = (a[piv, j], a[c, j]);
            for (var r = 0; r < d; r++)
            {
                if (r == c) continue;
                var f = a[r, c] / a[c, c];
                for (var j = c; j <= d; j++) a[r, j] -= f * a[c, j];
            }
        }
        var x = new double[d];
        for (var i = 0; i < d; i++) x[i] = a[i, d] / a[i, i] + p[s[0] * d + i];
        return x;
    }

    public static long Checksum(int[,] simplices, int n)
    {
        var rows = new List<int[]>();
        for (var i = 0; i < simplices.GetLength(0); i++)
        {
            var r = new int[simplices.GetLength(1)];
            for (var k = 0; k < r.Length; k++) r[k] = simplices[i, k];
            Array.Sort(r);
            rows.Add(r);
        }
        rows.Sort((a, b) => { for (var k = 0; k < a.Length; k++) { var c = a[k].CompareTo(b[k]); if (c != 0) return c; } return 0; });
        long t = 0;
        foreach (var r in rows)
        {
            long h = 0;
            foreach (var v in r) h = h * n + v;
            t = (t * 31 + h) % 2147483647;
        }
        return t;
    }

    public static int[] Row(int[,] a, int i)
    {
        var r = new int[a.GetLength(1)];
        for (var k = 0; k < r.Length; k++) r[k] = a[i, k];
        return r;
    }

    public static double[] Grid(int side, int d, double step = 1)
    {
        var total = (int)Math.Pow(side, d);
        var r = new double[total * d];
        for (var i = 0; i < total; i++)
        {
            var t = i;
            for (var j = 0; j < d; j++) { r[i * d + j] = step * (t % side); t /= side; }
        }
        return r;
    }
}

public class ConvexHullNdTests
{
    private static void CheckHull(ConvexHull h, double[] p, int n, int d, double tol = 1e-12)
    {
        var f = h.Simplices.GetLength(0);
        var eq = h.Equations.ToArray();
        for (var i = 0; i < f; i++)
        {
            double len = 0;
            for (var j = 0; j < d; j++) len += eq[i * (d + 1) + j] * eq[i * (d + 1) + j];
            Close(1.0, Math.Sqrt(len), 1e-12, "unit normal");
            for (var q = 0; q < n; q++)
            {
                var s = eq[i * (d + 1) + d];
                for (var j = 0; j < d; j++) s += eq[i * (d + 1) + j] * p[q * d + j];
                Assert.True(s <= tol, $"point {q} is outside facet {i} by {s}");
            }
            for (var k = 0; k < d; k++)
            {
                var nb = h.Neighbors[i, k];
                var mine = NdGeometry.Row(h.Simplices, i);
                var theirs = NdGeometry.Row(h.Simplices, nb);
                for (var j = 0; j < d; j++) if (j != k) Assert.Contains(mine[j], theirs);
                Assert.DoesNotContain(mine[k], theirs);
                Assert.Contains(i, NdGeometry.Row(h.Neighbors, nb));
            }
        }
    }

    [Theory]
    // scipy 1.17.1: ConvexHull(lcg_points(n, d, seed)) -> volume, area, facet count
    [InlineData(60, 4, 5, 0.310198287107908, 2.7786431258047344, 201)]
    [InlineData(40, 5, 9, 0.08360205184120523, 1.1583818036883298, 392)]
    public void RandomNd_MatchesScipy(int n, int d, int seed, double volume, double area, int facets)
    {
        var pts = LcgNd(n, d, seed);
        var h = new ConvexHull(pts);
        Assert.Equal(d, h.NDim);
        Close(volume, h.Volume, 1e-12, "volume");
        Close(area, h.Area, 1e-12, "area");
        Assert.Equal(facets, h.Simplices.GetLength(0));
        CheckHull(h, pts.ToArray(), n, d);
    }

    [Fact]
    public void Random4D_VerticesMatchScipy()
    {
        // scipy 1.17.1: ConvexHull(lcg_points(60, 4, 5)).vertices
        int[] expected = [0, 1, 2, 3, 7, 8, 9, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 42, 43, 46, 48, 49, 51, 52, 53, 55, 56, 58, 59];
        Assert.Equal(expected, new ConvexHull(LcgNd(60, 4, 5)).Vertices);
    }

    [Fact]
    public void General3D_AgreesWithSpecialised3D()
    {
        var pts = LcgNd(80, 3, 11);
        var a = new ConvexHull(pts);
        var b = ConvexHull.GeneralNd(pts);
        Assert.Equal(a.Vertices, b.Vertices);
        Close(a.Volume, b.Volume, 1e-13, "volume");
        Close(a.Area, b.Area, 1e-13, "area");
        Assert.Equal(NdGeometry.Checksum(a.Simplices, 80), NdGeometry.Checksum(b.Simplices, 80));
        CheckHull(b, pts.ToArray(), 80, 3);
    }

    [Fact]
    public void Tesseract_WithInteriorAndFacePoints()
    {
        var p = new List<double>(NdGeometry.Grid(2, 4));
        p.AddRange([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 1, 0.2, 0.3, 0.9]);
        var arr = p.ToArray();
        var h = new ConvexHull(new NdArray(arr, 19, 4));
        Assert.Equal(Enumerable.Range(0, 16).ToArray(), h.Vertices);
        Close(1.0, h.Volume, 1e-14, "volume");
        Close(8.0, h.Area, 1e-14, "boundary 3-volume");
        CheckHull(h, arr, 19, 4);
    }

    [Fact]
    public void Lattice4D_ManyDegeneracies()
    {
        var p = NdGeometry.Grid(3, 4);
        var h = new ConvexHull(new NdArray(p, 81, 4));
        Assert.Equal(16, h.Vertices.Length);
        Close(16.0, h.Volume, 1e-12, "volume");
        Close(64.0, h.Area, 1e-12, "area");
        CheckHull(h, p, 81, 4);
    }

    [Fact]
    public void StandardSimplex4D()
    {
        double[] p = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.1, 0.1, 0.1, 0.1];
        var h = new ConvexHull(new NdArray(p, 6, 4));
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, h.Vertices);
        Assert.Equal(5, h.Simplices.GetLength(0));
        Close(1.0 / 24, h.Volume, 1e-15, "volume");
        // Four coordinate facets of 3-volume 1/6 and the slanted one, sqrt(4)/6.
        Close(4.0 / 6 + 2.0 / 6, h.Area, 1e-15, "area");
    }

    [Fact]
    public void SphereSample4D_MatchesScipy()
    {
        // scipy 1.17.1: p = lcg_points(200, 4, 7) - 0.5; p /= norm(p, axis=1); ConvexHull(p)
        var p = Lcg(200, 4, 7);
        for (var i = 0; i < 200; i++)
        {
            double s = 0;
            for (var j = 0; j < 4; j++) { p[i * 4 + j] -= 0.5; s += p[i * 4 + j] * p[i * 4 + j]; }
            s = Math.Sqrt(s);
            for (var j = 0; j < 4; j++) p[i * 4 + j] /= s;
        }
        var h = new ConvexHull(new NdArray(p, 200, 4));
        Close(3.6674811318117784, h.Volume, 1e-12, "volume");
        Close(16.32057331910559, h.Area, 1e-12, "area");
        Assert.Equal(200, h.Vertices.Length);
        // Inscribed in the unit 3-sphere, whose 4-volume is π²/2 ≈ 4.93.
        Assert.True(h.Volume < Math.PI * Math.PI / 2);
    }

    [Fact]
    public void Flat4D_Throws()
    {
        // Every point has x3 = x0: a 3-flat in 4-D. (x3 = x0 + x1 would not do: the rounded sums
        // are not exactly coplanar, and the exact predicates build that thin hull correctly.)
        var p = Lcg(30, 4, 3);
        for (var i = 0; i < 30; i++) p[i * 4 + 3] = p[i * 4];
        Assert.Throws<ArgumentException>(() => new ConvexHull(new NdArray(p, 30, 4)));
    }
}

public class DelaunayNdTests
{
    /// <summary>
    /// Brute-force Delaunay check: positive orientation, reciprocal neighbours sharing a face,
    /// no input point strictly inside any circumsphere, and simplex volumes summing to the
    /// hull volume.
    /// </summary>
    internal static void CheckDelaunay(Delaunay tri, double[] p, int n, int d, double relTol = 1e-9)
    {
        var m = d + 1;
        double fact = 1; for (var k = 2; k <= d; k++) fact *= k;
        double vol = 0;
        for (var s = 0; s < tri.NSimplex; s++)
        {
            var simplex = NdGeometry.Row(tri.Simplices, s);
            var v = NdGeometry.SignedVolumeTimesFactorial(p, d, simplex);
            Assert.True(v > 0, $"simplex {s} not positively oriented ({v})");
            vol += v / fact;
            var c = NdGeometry.Circumcentre(p, d, simplex);
            double r2 = 0;
            for (var j = 0; j < d; j++) r2 += (p[simplex[0] * d + j] - c[j]) * (p[simplex[0] * d + j] - c[j]);
            for (var i = 0; i < n; i++)
            {
                double d2 = 0;
                for (var j = 0; j < d; j++) d2 += (p[i * d + j] - c[j]) * (p[i * d + j] - c[j]);
                Assert.True(d2 >= r2 * (1 - relTol), $"point {i} inside the circumsphere of simplex {s}");
            }
            for (var k = 0; k < m; k++)
            {
                var nb = tri.Neighbors[s, k];
                if (nb < 0) continue;
                var theirs = NdGeometry.Row(tri.Simplices, nb);
                for (var j = 0; j < m; j++) if (j != k) Assert.Contains(simplex[j], theirs);
                Assert.DoesNotContain(simplex[k], theirs);
                Assert.Contains(s, NdGeometry.Row(tri.Neighbors, nb));
            }
        }
        var hull = new ConvexHull(new NdArray((double[])p.Clone(), n, d));
        Close(hull.Volume, vol, 1e-11 * Math.Max(1, hull.Volume), "simplex volumes vs hull volume");
    }

    [Theory]
    // scipy 1.17.1: Delaunay(lcg_points(n, d, seed)) -> nsimplex, len(convex_hull), and a
    // checksum of the sorted simplices / hull facets. Points in general position have a unique
    // Delaunay triangulation, so the simplices themselves must agree.
    [InlineData(40, 3, 3, 157, 40, 2125416190L, 1732183082L)]
    [InlineData(25, 4, 4, 214, 48, 1925482872L, 875099758L)]
    [InlineData(300, 3, 17, 1768, 86, 2069365075L, 1607181781L)]
    public void RandomNd_MatchesScipyAndIsDelaunay(int n, int d, int seed, int nsimplex, int nhull, long checksum, long hullChecksum)
    {
        var pts = LcgNd(n, d, seed);
        var tri = new Delaunay(pts);
        Assert.Equal(d, tri.NDim);
        Assert.Equal(nsimplex, tri.NSimplex);
        Assert.Equal(nhull, tri.ConvexHull.GetLength(0));
        Assert.Equal(checksum, NdGeometry.Checksum(tri.Simplices, n));
        Assert.Equal(hullChecksum, NdGeometry.Checksum(tri.ConvexHull, n));
        if (n <= 60) CheckDelaunay(tri, pts.ToArray(), n, d);
    }

    [Fact]
    public void CubeCorners_AllCospherical()
    {
        // Every corner of a cube lies on one sphere, so the lifted points are flat; this is the
        // case Qhull's Qz option (and the extra point here) exists for.
        var p = NdGeometry.Grid(2, 3);
        var tri = new Delaunay(new NdArray(p, 8, 3));
        CheckDelaunay(tri, p, 8, 3);
        Assert.Equal(12, tri.ConvexHull.GetLength(0)); // two triangles per square face
    }

    [Fact]
    public void Lattice3D_IsDelaunayAndUsesEveryPoint()
    {
        var p = NdGeometry.Grid(4, 3);
        var tri = new Delaunay(new NdArray(p, 64, 3));
        CheckDelaunay(tri, p, 64, 3);
        Assert.All(tri.VertexToSimplex, s => Assert.True(s >= 0));
        Assert.Equal(108, tri.ConvexHull.GetLength(0)); // 6 sides x 9 squares x 2 triangles
    }

    [Fact]
    public void Lattice4D_IsDelaunay()
    {
        var p = NdGeometry.Grid(3, 4);
        var tri = new Delaunay(new NdArray(p, 81, 4));
        CheckDelaunay(tri, p, 81, 4);
    }

    [Fact]
    public void Duplicates_AreCoplanar()
    {
        var p = new List<double>(Lcg(20, 3, 8));
        p.AddRange([p[3], p[4], p[5]]);       // duplicate of point 1
        p.AddRange([p[30], p[31], p[32]]);    // duplicate of point 10
        var tri = new Delaunay(new NdArray(p.ToArray(), 22, 3));
        Assert.Equal(2, tri.Coplanar.GetLength(0));
        Assert.Equal(20, tri.Coplanar[0, 0]); Assert.Equal(1, tri.Coplanar[0, 2]);
        Assert.Equal(21, tri.Coplanar[1, 0]); Assert.Equal(10, tri.Coplanar[1, 2]);
        Assert.Equal(-1, tri.VertexToSimplex[20]);
        Assert.Contains(1, NdGeometry.Row(tri.Simplices, tri.Coplanar[0, 1]));
    }

    [Fact]
    public void Coplanar3D_Throws()
    {
        var p = Lcg(20, 3, 2);
        for (var i = 0; i < 20; i++) p[i * 3 + 2] = p[i * 3]; // the plane z = x, exactly
        Assert.Throws<ArgumentException>(() => new Delaunay(new NdArray(p, 20, 3)));
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void FindSimplex_AgreesWithBruteForce(int d)
    {
        var n = d == 3 ? 120 : 60;
        var tri = new Delaunay(LcgNd(n, d, 31));
        // Queries inside and outside the hull, plus every input point exactly.
        var q = Lcg(600, d, 77);
        for (var i = 0; i < q.Length; i++) q[i] = 1.3 * q[i] - 0.15;
        var walk = tri.FindSimplex(new NdArray(q, 600, d));
        var brute = tri.FindSimplex(new NdArray(q, 600, d), bruteForce: true);
        var tf = tri.Transform;
        Assert.Equal(new[] { tri.NSimplex, d + 1, d }, tf.Shape.ToArray());
        for (var i = 0; i < 600; i++)
        {
            var x = q.AsSpan(i * d, d);
            Assert.Equal(brute[i] < 0, walk[i] < 0);
            if (walk[i] < 0) continue;
            var c = tri.Barycentric(walk[i], x);
            Close(1.0, c.Sum(), 1e-12, "barycentric sum");
            Assert.All(c, v => Assert.True(v >= -1e-12, $"query {i}: barycentric {v}"));
            // Reconstruct x from the barycentric coordinates.
            for (var j = 0; j < d; j++)
            {
                double s = 0;
                for (var k = 0; k <= d; k++) s += c[k] * tri.PointBuffer[tri.Simplices[walk[i], k] * d + j];
                Close(x[j], s, 1e-12, "reconstruction");
            }
        }
        var pts = tri.PointBuffer;
        var at = tri.FindSimplex(new NdArray((double[])pts.Clone(), n, d));
        for (var i = 0; i < n; i++)
            Assert.Contains(i, NdGeometry.Row(tri.Simplices, at[i]));
        Assert.Equal(-1, tri.FindSimplex(Enumerable.Repeat(5.0, d).ToArray()));
    }

    [Fact]
    public void VertexNeighborVertices_IsSymmetricAndMatchesEdges()
    {
        var tri = new Delaunay(LcgNd(50, 3, 12));
        var (indptr, indices) = tri.VertexNeighborVertices();
        var edges = new HashSet<(int, int)>();
        for (var s = 0; s < tri.NSimplex; s++)
            for (var a = 0; a < 4; a++)
                for (var b = a + 1; b < 4; b++)
                {
                    int u = tri.Simplices[s, a], v = tri.Simplices[s, b];
                    edges.Add((Math.Min(u, v), Math.Max(u, v)));
                }
        Assert.Equal(2 * edges.Count, indices.Length);
        for (var i = 0; i < 50; i++)
            for (var k = indptr[i]; k < indptr[i + 1]; k++)
                Assert.Contains((Math.Min(i, indices[k]), Math.Max(i, indices[k])), edges);
    }

    [Fact]
    public void Tsearch_IsFindSimplex()
    {
        var tri = new Delaunay(LcgNd(40, 3, 5));
        var q = LcgNd(50, 3, 6);
        Assert.Equal(tri.FindSimplex(q), SpatialFunctions.Tsearch(tri, q));
    }
}
