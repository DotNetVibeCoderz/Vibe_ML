using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// Intersection of halfspaces <c>A x + b ≤ 0</c> in any dimension ≥ 2
/// (scipy.spatial.HalfspaceIntersection), computed through the dual convex hull, as Qhull's
/// <c>H</c> mode does.
/// </summary>
/// <remarks>
/// <para><b>Duality.</b> With c a strictly interior point, halfspace i maps to the dual point
/// <c>a_i / −(a_i·c + b_i)</c>. The polytope's vertices are the facets of the dual points'
/// convex hull: a facet with unit outward normal n and offset o (n·y + o = 0) is the vertex
/// <c>c + n / −o</c>; the halfspaces whose dual points span the facet are the ones active at
/// that vertex (<see cref="DualFacets"/>), and redundant halfspaces are exactly the dual points
/// that are not hull vertices.</para>
/// <para><b>Non-simple vertices.</b> The hull here is triangulated, so a vertex where more than d
/// halfspaces meet (the apex of a square pyramid) is the shared point of several coplanar dual
/// facets. Adjacent dual facets are merged when the exact predicate finds them coplanar, or when
/// their floating hyperplanes agree to 1e-12 — the dual points are rounded quotients, so exact
/// coplanarity of the primal is usually lost in them; this is the one tolerance in the class,
/// playing the role of Qhull's facet merging. Each merged group is one intersection.</para>
/// <para>An interior point that is not strictly inside every halfspace throws, as scipy does.
/// An unbounded intersection has no finite dual: its dual hull has the origin on the boundary,
/// and the corresponding intersections come out infinite rather than wrong.</para>
/// </remarks>
public sealed class HalfspaceIntersection
{
    private readonly double[] _h;
    private readonly double[] _x;

    /// <summary>The halfspaces [A, b], shape (nineq, d + 1), each meaning <c>A x + b ≤ 0</c>.</summary>
    public NdArray Halfspaces => new((double[])_h.Clone(), NIneq, NDim + 1);

    /// <summary>The interior point used for the dual transform.</summary>
    public double[] InteriorPoint => (double[])_x.Clone();

    /// <summary>Dimension of the space.</summary>
    public int NDim { get; }

    /// <summary>Number of halfspaces.</summary>
    public int NIneq { get; }

    /// <summary>Vertices of the intersection, shape (nvertices, d).</summary>
    public NdArray Intersections { get; }

    /// <summary>The dual points, shape (nineq, d).</summary>
    public NdArray DualPoints { get; }

    /// <summary>For each intersection, the indices of the halfspaces active there (ascending).</summary>
    public int[][] DualFacets { get; }

    /// <summary>Indices of the non-redundant halfspaces (the dual hull's vertices, ascending).</summary>
    public int[] DualVertices { get; }

    /// <summary>The dual hull's facet equations (triangulated), shape (nfacets, d + 1).</summary>
    public NdArray DualEquations { get; }

    /// <summary>Surface (d−1)-volume of the dual hull.</summary>
    public double DualArea { get; }

    /// <summary>Volume of the dual hull.</summary>
    public double DualVolume { get; }

    /// <summary>The dual convex hull itself.</summary>
    public ConvexHull DualHull { get; }

    /// <summary>Intersects the halfspaces <c>A x + b ≤ 0</c>.</summary>
    /// <param name="halfspaces">Rows [A, b], shape (nineq, d + 1).</param>
    /// <param name="interiorPoint">A point strictly inside every halfspace (d values); scipy
    /// leaves finding it to the caller, e.g. by the Chebyshev-centre linear program.</param>
    public HalfspaceIntersection(NdArray halfspaces, double[] interiorPoint)
    {
        ArgumentNullException.ThrowIfNull(halfspaces);
        ArgumentNullException.ThrowIfNull(interiorPoint);
        if (halfspaces.Rank != 2 || halfspaces.Shape[1] < 3) throw new ArgumentException("halfspaces must have shape (nineq, ndim + 1) with ndim ≥ 2.");
        NIneq = halfspaces.Shape[0];
        NDim = halfspaces.Shape[1] - 1;
        var d = NDim;
        if (interiorPoint.Length != d) throw new ArgumentException($"The interior point must have {d} coordinates.");
        _h = halfspaces.AsContiguous().ToArray();
        _x = (double[])interiorPoint.Clone();
        SpatialInput.RequireFinite(_h, "Halfspaces");
        SpatialInput.RequireFinite(_x, "The interior point");

        var dual = new double[NIneq * d];
        for (var i = 0; i < NIneq; i++)
        {
            var s = _h[i * (d + 1) + d];
            for (var j = 0; j < d; j++) s += _h[i * (d + 1) + j] * _x[j];
            if (!(s < 0)) throw new ArgumentException($"The interior point is not strictly inside halfspace {i} (A x + b = {s}).");
            for (var j = 0; j < d; j++) dual[i * d + j] = _h[i * (d + 1) + j] / -s;
        }
        DualPoints = new NdArray((double[])dual.Clone(), NIneq, d);
        var hull = new ConvexHull(new NdArray(dual, NIneq, d));
        DualHull = hull;
        DualVertices = [.. hull.Vertices.OrderBy(v => v)];
        DualEquations = hull.Equations;
        DualArea = hull.Area;
        DualVolume = hull.Volume;

        // Merge coplanar adjacent dual facets (see remarks), then one vertex per group.
        var f = hull.Simplices.GetLength(0);
        var eq = hull.Equations.ToArray();
        var parent = new int[f];
        for (var i = 0; i < f; i++) parent[i] = i;
        int Find(int a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; }
        var face = new int[d];
        for (var i = 0; i < f; i++)
            for (var k = 0; k < d; k++)
            {
                var g = hull.Neighbors[i, k];
                if (g <= i) continue;
                var same = true;
                for (var j = 0; j <= d && same; j++) same = Math.Abs(eq[i * (d + 1) + j] - eq[g * (d + 1) + j]) <= 1e-12;
                if (!same && d >= 3)
                {
                    // The vertex of g not on i, against i's hyperplane, exactly.
                    for (var j = 0; j < d; j++) face[j] = hull.Simplices[i, j];
                    var apex = -1;
                    for (var j = 0; j < d; j++)
                    {
                        var v = hull.Simplices[g, j];
                        if (Array.IndexOf(face, v) < 0) { apex = v; break; }
                    }
                    same = apex >= 0 && SpatialPredicates.OrientNd(dual, d, false, null, face, dual.AsSpan(apex * d, d), 0) == 0;
                }
                if (same)
                {
                    int ra = Find(i), rb = Find(g);
                    if (ra != rb) parent[Math.Max(ra, rb)] = Math.Min(ra, rb);
                }
            }
        var groupOf = new Dictionary<int, int>();
        var members = new List<SortedSet<int>>();
        var verts = new List<double>();
        for (var i = 0; i < f; i++)
        {
            var r = Find(i);
            if (!groupOf.TryGetValue(r, out var gi))
            {
                gi = members.Count;
                groupOf[r] = gi;
                members.Add([]);
                var off = eq[r * (d + 1) + d];
                for (var j = 0; j < d; j++) verts.Add(eq[r * (d + 1) + j] / -off + _x[j]);
            }
            for (var j = 0; j < d; j++) members[gi].Add(hull.Simplices[i, j]);
        }
        Intersections = new NdArray(verts.ToArray(), members.Count, d);
        DualFacets = [.. members.Select(s => s.ToArray())];
    }

    /// <summary>Intersects the halfspaces given as a rectangular array [A, b].</summary>
    public HalfspaceIntersection(double[,] halfspaces, double[] interiorPoint) : this(SpatialInput.ToNd(halfspaces), interiorPoint) { }
}
