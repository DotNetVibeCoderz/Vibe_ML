using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// Voronoi diagram of points on a sphere (scipy.spatial.SphericalVoronoi), in 2-D (a circle)
/// and above.
/// </summary>
/// <remarks>
/// <para>As in scipy, the diagram comes from the convex hull of the generators: each hull facet
/// is a Voronoi vertex, at <c>center + radius · n</c> with n the facet's unit outward normal
/// (the direction of the facet's circumcentre from the centre), and a generator's region is the
/// set of facets that contain it. The hull is triangulated, so where four or more generators
/// are cocircular (a cube's corners) several facets give the same vertex — scipy keeps those
/// duplicates too, so vertex counts agree with scipy's. Vertex numbering follows the hull's
/// facet order and differs from scipy's; compare coordinates.</para>
/// <para><see cref="CalculateAreas"/> sums Van Oosterom–Strackee spherical triangles (generator,
/// v_i, v_{i+1}) in 3-D, which stays accurate for tiny regions where L'Huilier's formula loses
/// digits; in 2-D the "areas" are arc lengths. Both sum to the sphere's measure.</para>
/// <para><b>2-D differs from scipy on purpose.</b> scipy 1.17.1's 2-D <c>calculate_areas</c>
/// returns the arc of every hull edge (between two consecutive generators) in Qhull's facet
/// order, although it documents region areas. Here element i is the length of generator i's
/// region, half the gap on either side; the two lists have the same sum, 2πr.</para>
/// </remarks>
public sealed class SphericalVoronoi
{
    private readonly double[] _p;
    private readonly double[] _v;

    /// <summary>The generators, shape (n, d).</summary>
    public NdArray Points => new((double[])_p.Clone(), NPoints, NDim);

    /// <summary>Number of generators.</summary>
    public int NPoints { get; }

    /// <summary>Dimension.</summary>
    public int NDim { get; }

    /// <summary>Sphere radius.</summary>
    public double Radius { get; }

    /// <summary>Sphere centre.</summary>
    public double[] Center { get; }

    /// <summary>Voronoi vertices on the sphere, shape (nvertices, d).</summary>
    public NdArray Vertices => new((double[])_v.Clone(), _v.Length / NDim, NDim);

    /// <summary>Each generator's region as vertex indices (in facet order until
    /// <see cref="SortVerticesOfRegions"/> is called).</summary>
    public int[][] Regions { get; }

    /// <summary>Builds the diagram.</summary>
    /// <param name="points">Generators on the sphere, shape (n, d), d ≥ 2.</param>
    /// <param name="radius">Sphere radius.</param>
    /// <param name="center">Sphere centre (default the origin).</param>
    /// <param name="threshold">Relative tolerance for "on the sphere" and for duplicates, as scipy.</param>
    public SphericalVoronoi(NdArray points, double radius = 1, double[]? center = null, double threshold = 1e-6)
    {
        _p = SpatialInput.Copy(points, out var n, out var d, nameof(points));
        if (d < 2) throw new ArgumentException("SphericalVoronoi needs points of dimension 2 or more.");
        SpatialInput.RequireFinite(_p, "Points");
        if (!(radius > 0) || !double.IsFinite(radius)) throw new ArgumentException("The radius must be positive and finite.");
        NPoints = n; NDim = d; Radius = radius;
        Center = center is null ? new double[d] : (double[])center.Clone();
        if (Center.Length != d) throw new ArgumentException($"The centre must have {d} coordinates.");
        if (n < d + 1) throw new ArgumentException($"SphericalVoronoi needs at least {d + 1} points in {d}-D.");
        for (var i = 0; i < n; i++)
        {
            double r2 = 0;
            for (var j = 0; j < d; j++) { var t = _p[i * d + j] - Center[j]; r2 += t * t; }
            if (Math.Abs(Math.Sqrt(r2) - radius) > threshold * radius)
                throw new ArgumentException("Radius inconsistent with generators.");
        }
        // Duplicates within threshold (scipy uses a k-d tree pair query for the same check).
        var tree = new KDTree(new NdArray((double[])_p.Clone(), n, d));
        if (tree.QueryPairs(threshold * radius).Count > 0) throw new ArgumentException("Duplicate generators present.");

        var hull = new ConvexHull(new NdArray((double[])_p.Clone(), n, d));
        var f = hull.Simplices.GetLength(0);
        var eq = hull.Equations.ToArray();
        _v = new double[f * d];
        for (var i = 0; i < f; i++)
            for (var j = 0; j < d; j++) _v[i * d + j] = radius * eq[i * (d + 1) + j] + Center[j];
        // A generator inside the hull (not a hull vertex) has no region: scipy rejects the
        // input as not all on the sphere, which the radius check above has already enforced.
        var lists = new List<int>[n];
        for (var i = 0; i < n; i++) lists[i] = [];
        for (var i = 0; i < f; i++)
            for (var k = 0; k < d; k++) lists[hull.Simplices[i, k]].Add(i);
        Regions = [.. lists.Select(l => l.ToArray())];
    }

    /// <summary>Builds from a rectangular array of generators.</summary>
    public SphericalVoronoi(double[,] points, double radius = 1, double[]? center = null, double threshold = 1e-6)
        : this(SpatialInput.ToNd(points), radius, center, threshold) { }

    /// <summary>
    /// Orders each region's vertices counter-clockwise as seen from outside the sphere (scipy's
    /// <c>sort_vertices_of_regions</c>, 3-D only).
    /// </summary>
    public void SortVerticesOfRegions()
    {
        if (NDim != 3) throw new InvalidOperationException("Only supported for 3-D (scipy raises TypeError otherwise).");
        for (var i = 0; i < NPoints; i++)
        {
            var region = Regions[i];
            if (region.Length < 3) continue;
            // Tangent-plane basis at the generator: u ⟂ g, w = g × u; angle of each vertex.
            double gx = _p[3 * i] - Center[0], gy = _p[3 * i + 1] - Center[1], gz = _p[3 * i + 2] - Center[2];
            var gl = Math.Sqrt(gx * gx + gy * gy + gz * gz);
            gx /= gl; gy /= gl; gz /= gl;
            double ux, uy, uz;
            if (Math.Abs(gx) < 0.9) { ux = 0; uy = gz; uz = -gy; } else { ux = -gz; uy = 0; uz = gx; }
            var ul = Math.Sqrt(ux * ux + uy * uy + uz * uz);
            ux /= ul; uy /= ul; uz /= ul;
            double wx = gy * uz - gz * uy, wy = gz * ux - gx * uz, wz = gx * uy - gy * ux;
            var ang = new double[region.Length];
            for (var k = 0; k < region.Length; k++)
            {
                var v = region[k];
                double x = _v[3 * v] - Center[0], y = _v[3 * v + 1] - Center[1], z = _v[3 * v + 2] - Center[2];
                ang[k] = Math.Atan2(x * wx + y * wy + z * wz, x * ux + y * uy + z * uz);
            }
            Array.Sort(ang, region);
        }
    }

    /// <summary>
    /// Area of every region on the sphere (scipy's <c>calculate_areas</c>): solid-angle areas
    /// in 3-D (regions are sorted first), arc lengths in 2-D.
    /// </summary>
    public double[] CalculateAreas()
    {
        var areas = new double[NPoints];
        if (NDim == 2)
        {
            for (var i = 0; i < NPoints; i++)
            {
                var r = Regions[i];
                if (r.Length != 2) throw new InvalidOperationException("A 2-D region must have two vertices.");
                double ax = _v[2 * r[0]] - Center[0], ay = _v[2 * r[0] + 1] - Center[1];
                double bx = _v[2 * r[1]] - Center[0], by = _v[2 * r[1] + 1] - Center[1];
                var angle = Math.Atan2(Math.Abs(ax * by - ay * bx), ax * bx + ay * by);
                areas[i] = Radius * angle;
            }
            return areas;
        }
        if (NDim != 3) throw new InvalidOperationException("Only supported for 2-D and 3-D.");
        SortVerticesOfRegions();
        for (var i = 0; i < NPoints; i++)
        {
            var r = Regions[i];
            double gx = (_p[3 * i] - Center[0]) / Radius, gy = (_p[3 * i + 1] - Center[1]) / Radius, gz = (_p[3 * i + 2] - Center[2]) / Radius;
            double solid = 0;
            for (var k = 0; k < r.Length; k++)
            {
                int a = r[k], b = r[(k + 1) % r.Length];
                double ax = (_v[3 * a] - Center[0]) / Radius, ay = (_v[3 * a + 1] - Center[1]) / Radius, az = (_v[3 * a + 2] - Center[2]) / Radius;
                double bx = (_v[3 * b] - Center[0]) / Radius, by = (_v[3 * b + 1] - Center[1]) / Radius, bz = (_v[3 * b + 2] - Center[2]) / Radius;
                var triple = gx * (ay * bz - az * by) + gy * (az * bx - ax * bz) + gz * (ax * by - ay * bx);
                var denom = 1 + (gx * ax + gy * ay + gz * az) + (ax * bx + ay * by + az * bz) + (bx * gx + by * gy + bz * gz);
                solid += 2 * Math.Atan2(triple, denom);
            }
            areas[i] = Math.Abs(solid) * Radius * Radius;
        }
        return areas;
    }
}
