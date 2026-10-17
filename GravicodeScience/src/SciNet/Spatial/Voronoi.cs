using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// Voronoi diagram in any dimension ≥ 2, built as the dual of <see cref="Delaunay"/>
/// (scipy.spatial.Voronoi).
/// </summary>
/// <remarks>
/// <para>Voronoi vertices are circumcentres of Delaunay triangles. Adjacent triangles whose
/// four points are exactly cocircular (decided with the exact in-circle predicate) have the
/// same circumcentre and are merged into one vertex, as Qhull merges them — otherwise a lattice
/// would produce pairs of coincident vertices joined by zero-length ridges.</para>
/// <para><b>Numbering differs from scipy.</b> Qhull's vertex and region order is an artefact of
/// its facet lists. Here vertex v is the v-th merged triangle group in triangle order,
/// <c>Regions[i]</c> belongs to input point i (so <see cref="PointRegion"/> is the identity, except
/// that a duplicate point shares the region of the point it duplicates), and each region lists
/// its vertices counter-clockwise with -1 first when the region is unbounded. Compare with
/// scipy through coordinates and point pairs, not through indices.</para>
/// <para><b>3-D and above.</b> Vertices are circumcentres of the Delaunay simplices, merged
/// across faces whose d + 2 points are exactly cospherical (exact lifted predicate). A ridge is
/// reported for a Delaunay edge (i, j) when it has at least d entries counting -1 — Qhull's own
/// rule, which drops the diagonals of cospherical cells (all their simplices merge into one
/// vertex) exactly as scipy does. In 3-D a ridge's vertices are listed cyclically around the
/// edge, with -1 first for an unbounded ridge, the finite chain starting next to the opening;
/// above 3-D they are ascending. Regions list their vertices ascending, -1 first when
/// unbounded (scipy's order there is Qhull's, not meaningful). <see cref="RidgeVertices"/> is
/// a 2-column array and so 2-D only; use <see cref="RidgeVertexLists"/> in any dimension.
/// <see cref="FinitePolygons"/> is 2-D only.</para>
/// </remarks>
public sealed class Voronoi
{
    private readonly double[] _p;
    private readonly double[] _v;
    private readonly int[] _triVertex;
    private readonly int _d;
    private readonly int[,]? _ridgeVertices;

    /// <summary>The input points, shape (n, d).</summary>
    public NdArray Points => new((double[])_p.Clone(), _p.Length / _d, _d);

    /// <summary>Dimension of the points.</summary>
    public int NDim => _d;

    /// <summary>Voronoi vertices, shape (nvertices, d).</summary>
    public NdArray Vertices => new((double[])_v.Clone(), _v.Length / _d, _d);

    /// <summary>Pairs of input points separated by each ridge, [nridges, 2].</summary>
    public int[,] RidgePoints { get; }

    /// <summary>Voronoi vertices of each ridge, [nridges, 2]; -1 marks the vertex at infinity.
    /// 2-D only (a ridge has more vertices above 2-D): use <see cref="RidgeVertexLists"/>.</summary>
    public int[,] RidgeVertices => _ridgeVertices ?? throw new InvalidOperationException(
        $"RidgeVertices is a 2-column array and only describes 2-D diagrams; this one is {_d}-D, use RidgeVertexLists.");

    /// <summary>Voronoi vertices of each ridge, in any dimension (scipy's <c>ridge_vertices</c>);
    /// -1 marks the vertex at infinity.</summary>
    public int[][] RidgeVertexLists { get; }

    /// <summary>Each region's vertices, counter-clockwise; -1 (listed first) when unbounded.</summary>
    public int[][] Regions { get; }

    /// <summary>Index into <see cref="Regions"/> for each input point.</summary>
    public int[] PointRegion { get; }

    /// <summary>The underlying triangulation.</summary>
    public Delaunay Triangulation { get; }

    /// <summary>Builds the diagram of the rows of <paramref name="points"/>, shape (n, d).</summary>
    public Voronoi(NdArray points) : this(new Delaunay(points)) { }

    /// <summary>Builds the diagram of a rectangular array of points.</summary>
    public Voronoi(double[,] points) : this(new Delaunay(points)) { }

    private Voronoi(Delaunay tri)
    {
        Triangulation = tri;
        _p = tri.PointBuffer;
        _d = tri.NDim;
        if (_d != 2)
        {
            (_v, _triVertex, var ndRp, var ndRv, var ndRegions) = BuildNd(tri);
            RidgePoints = ndRp;
            RidgeVertexLists = ndRv;
            Regions = ndRegions;
            PointRegion = new int[tri.NPoints];
            for (var i = 0; i < tri.NPoints; i++) PointRegion[i] = i;
            for (var r = 0; r < tri.Coplanar.GetLength(0); r++) PointRegion[tri.Coplanar[r, 0]] = tri.Coplanar[r, 2];
            return;
        }
        var t = tri.TriangleBuffer;
        var nb = tri.NeighborBuffer;
        var ns = tri.NSimplex;
        var n = tri.NPoints;

        // Union triangles across edges whose four points are cocircular.
        var parent = new int[ns];
        for (var i = 0; i < ns; i++) parent[i] = i;
        int Find(int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
        for (var s = 0; s < ns; s++)
            for (var k = 0; k < 3; k++)
            {
                var o = nb[3 * s + k];
                if (o < s) continue; // each edge once; -1 (hull) skipped too
                var opp = 0;
                while (nb[3 * o + opp] != s) opp++;
                var q = t[3 * o + opp];
                int a = t[3 * s], b = t[3 * s + 1], c = t[3 * s + 2];
                if (SpatialPredicates.InCircle(_p[2 * a], _p[2 * a + 1], _p[2 * b], _p[2 * b + 1], _p[2 * c], _p[2 * c + 1], _p[2 * q], _p[2 * q + 1]) == 0)
                {
                    int ra = Find(s), rb = Find(o);
                    if (ra != rb) parent[Math.Max(ra, rb)] = Math.Min(ra, rb);
                }
            }
        _triVertex = new int[ns];
        var groupVertex = new int[ns];
        Array.Fill(groupVertex, -1);
        var verts = new List<double>();
        for (var s = 0; s < ns; s++)
        {
            var r = Find(s);
            if (groupVertex[r] < 0)
            {
                groupVertex[r] = verts.Count / 2;
                var (x, y) = Circumcentre(t[3 * r], t[3 * r + 1], t[3 * r + 2]);
                verts.Add(x); verts.Add(y);
            }
            _triVertex[s] = groupVertex[r];
        }
        _v = [.. verts];

        // Ridges: one per Delaunay edge whose two sides are different vertices.
        var rp = new List<(int, int)>();
        var rv = new List<(int, int)>();
        for (var s = 0; s < ns; s++)
            for (var k = 0; k < 3; k++)
            {
                var o = nb[3 * s + k];
                int a = t[3 * s + (k + 1) % 3], b = t[3 * s + (k + 2) % 3];
                if (o < 0) { rp.Add((Math.Min(a, b), Math.Max(a, b))); rv.Add((-1, _triVertex[s])); continue; }
                if (o < s || _triVertex[o] == _triVertex[s]) continue;
                rp.Add((Math.Min(a, b), Math.Max(a, b)));
                rv.Add((Math.Min(_triVertex[s], _triVertex[o]), Math.Max(_triVertex[s], _triVertex[o])));
            }
        RidgePoints = new int[rp.Count, 2];
        _ridgeVertices = new int[rv.Count, 2];
        RidgeVertexLists = new int[rv.Count][];
        for (var i = 0; i < rp.Count; i++)
        {
            RidgePoints[i, 0] = rp[i].Item1; RidgePoints[i, 1] = rp[i].Item2;
            _ridgeVertices[i, 0] = rv[i].Item1; _ridgeVertices[i, 1] = rv[i].Item2;
            RidgeVertexLists[i] = [rv[i].Item1, rv[i].Item2];
        }

        // Regions: walk the fan of triangles around each vertex counter-clockwise.
        Regions = new int[n][];
        PointRegion = new int[n];
        for (var i = 0; i < n; i++) PointRegion[i] = i;
        for (var i = 0; i < n; i++)
        {
            var s0 = tri.VertexToSimplex[i];
            if (s0 < 0) { Regions[i] = []; continue; }
            // Rotate clockwise until the hull (or back to the start) so the CCW walk covers the fan.
            var s = s0;
            var bounded = true;
            while (true)
            {
                var k = Slot(t, s, i);
                var prev = nb[3 * s + (k + 2) % 3]; // across edge (i, t[k+1]): the clockwise neighbour
                if (prev < 0) { bounded = false; break; }
                s = prev;
                if (s == s0) break;
            }
            var list = new List<int>();
            if (!bounded) list.Add(-1);
            var first = s;
            while (true)
            {
                var v = _triVertex[s];
                if (list.Count == 0 || list[^1] != v) list.Add(v);
                var k = Slot(t, s, i);
                var next = nb[3 * s + (k + 1) % 3]; // across edge (t[k+2], i): the counter-clockwise neighbour
                if (next < 0 || next == first) break;
                s = next;
            }
            if (bounded && list.Count > 1 && list[0] == list[^1]) list.RemoveAt(list.Count - 1);
            Regions[i] = [.. list];
        }
        for (var r = 0; r < tri.Coplanar.GetLength(0); r++) PointRegion[tri.Coplanar[r, 0]] = tri.Coplanar[r, 2];
    }

    // ================================================================== N-d

    private static (double[] V, int[] TriVertex, int[,] RidgePoints, int[][] RidgeVertices, int[][] Regions) BuildNd(Delaunay tri)
    {
        var p = tri.PointBuffer;
        var t = tri.TriangleBuffer;
        var nb = tri.NeighborBuffer;
        var d = tri.NDim; var m = d + 1;
        var ns = tri.NSimplex;
        var n = tri.NPoints;

        // Union simplices across faces whose d + 2 points are exactly cospherical.
        var parent = new int[ns];
        for (var i = 0; i < ns; i++) parent[i] = i;
        int Find(int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
        for (var s = 0; s < ns; s++)
            for (var k = 0; k < m; k++)
            {
                var o = nb[m * s + k];
                if (o < s) continue;
                var opp = 0;
                while (nb[m * o + opp] != s) opp++;
                var q = t[m * o + opp];
                if (SpatialPredicates.OrientNd(p, d, true, null, t.AsSpan(m * s, m), p.AsSpan(q * d, d), 0) == 0)
                {
                    int ra = Find(s), rb = Find(o);
                    if (ra != rb) parent[Math.Max(ra, rb)] = Math.Min(ra, rb);
                }
            }
        var triVertex = new int[ns];
        var groupVertex = new int[ns];
        Array.Fill(groupVertex, -1);
        var verts = new List<double>();
        for (var s = 0; s < ns; s++)
        {
            var r = Find(s);
            if (groupVertex[r] < 0)
            {
                groupVertex[r] = verts.Count / d;
                verts.AddRange(CircumcentreNd(p, d, t.AsSpan(m * r, m)));
            }
            triVertex[s] = groupVertex[r];
        }
        var v = verts.ToArray();

        // Simplices around every Delaunay edge, and which points lie on the hull.
        var edgeSimplices = new Dictionary<(int, int), List<int>>();
        var onHull = new bool[n];
        for (var s = 0; s < ns; s++)
        {
            for (var a = 0; a < m; a++)
                for (var b = a + 1; b < m; b++)
                {
                    var key = (Math.Min(t[m * s + a], t[m * s + b]), Math.Max(t[m * s + a], t[m * s + b]));
                    if (!edgeSimplices.TryGetValue(key, out var list)) edgeSimplices[key] = list = [];
                    list.Add(s);
                }
            for (var k = 0; k < m; k++)
            {
                if (nb[m * s + k] >= 0) continue;
                var facet = new int[d];
                var w = 0;
                for (var j = 0; j < m; j++) if (j != k) facet[w++] = t[m * s + j];
                foreach (var f in facet) onHull[f] = true;
            }
        }

        var ridgePoints = new List<(int, int)>();
        var ridgeVerts = new List<int[]>();
        foreach (var (edge, simplices) in edgeSimplices)
        {
            var groups = new SortedSet<int>();
            foreach (var s in simplices) groups.Add(triVertex[s]);
            // Unbounded iff the edge lies on the hull: some simplex around it has no neighbour
            // across a face that contains the edge.
            var unbounded = false;
            foreach (var s in simplices)
                for (var k = 0; k < m && !unbounded; k++)
                    if (nb[m * s + k] < 0 && t[m * s + k] != edge.Item1 && t[m * s + k] != edge.Item2) unbounded = true;
            // Qhull's rule: a ridge needs at least d vertices, counting the one at infinity.
            if (groups.Count + (unbounded ? 1 : 0) < d) continue;
            int[] list;
            if (d == 3) list = OrderAroundEdge(t, nb, triVertex, edge, simplices[0], unbounded);
            else
            {
                list = new int[groups.Count + (unbounded ? 1 : 0)];
                var w = 0;
                if (unbounded) list[w++] = -1;
                foreach (var g in groups) list[w++] = g;
            }
            ridgePoints.Add(edge);
            ridgeVerts.Add(list);
        }
        var rp = new int[ridgePoints.Count, 2];
        for (var i = 0; i < ridgePoints.Count; i++) { rp[i, 0] = ridgePoints[i].Item1; rp[i, 1] = ridgePoints[i].Item2; }

        var regionSets = new SortedSet<int>?[n];
        for (var s = 0; s < ns; s++)
            for (var k = 0; k < m; k++) (regionSets[t[m * s + k]] ??= []).Add(triVertex[s]);
        var regions = new int[n][];
        for (var i = 0; i < n; i++)
        {
            if (regionSets[i] is not { } set) { regions[i] = []; continue; }
            var r = new List<int>();
            if (onHull[i]) r.Add(-1);
            r.AddRange(set);
            regions[i] = [.. r];
        }
        return (v, triVertex, rp, [.. ridgeVerts], regions);
    }

    /// <summary>Circumcentre of a d-simplex: solves 2(v_i − v₀)·c' = |v_i − v₀|² for c' = c − v₀.</summary>
    internal static double[] CircumcentreNd(double[] p, int d, ReadOnlySpan<int> s)
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
            a[i, d] = rhs;
        }
        for (var c = 0; c < d; c++)
        {
            var piv = c;
            for (var r = c + 1; r < d; r++) if (Math.Abs(a[r, c]) > Math.Abs(a[piv, c])) piv = r;
            if (piv != c) for (var j = 0; j <= d; j++) (a[c, j], a[piv, j]) = (a[piv, j], a[c, j]);
            for (var r = c + 1; r < d; r++)
            {
                var f = a[r, c] / a[c, c];
                if (f == 0) continue;
                for (var j = c; j <= d; j++) a[r, j] -= f * a[c, j];
            }
        }
        var x = new double[d];
        for (var i = d - 1; i >= 0; i--)
        {
            var sum = a[i, d];
            for (var j = i + 1; j < d; j++) sum -= a[i, j] * x[j];
            x[i] = sum / a[i, i];
        }
        for (var i = 0; i < d; i++) x[i] += p[s[0] * d + i];
        return x;
    }

    /// <summary>
    /// A 3-D ridge's vertices in cyclic order: the tetrahedra around Delaunay edge (a, b) are
    /// walked through their shared faces, as Qhull orders them — purely combinatorial, so no
    /// angle is ever compared (the edge need not even cross its ridge). An unbounded ridge is an
    /// open chain, walked from one hull end to the other, with -1 first.
    /// </summary>
    private static int[] OrderAroundEdge(int[] t, int[] nb, int[] triVertex, (int A, int B) edge, int start, bool unbounded)
    {
        // Across the face opposite `drop` stays on the edge; the vertex of the next tetrahedron
        // that is neither a, b nor the kept one is the next to drop.
        List<int> Walk(int s0, int drop)
        {
            var seq = new List<int> { s0 };
            var s = s0;
            while (true)
            {
                var k = 0;
                while (t[4 * s + k] != drop) k++;
                var keep = -1;
                for (var j = 0; j < 4; j++)
                {
                    var x = t[4 * s + j];
                    if (x != edge.A && x != edge.B && x != drop) keep = x;
                }
                var next = nb[4 * s + k];
                if (next < 0 || next == s0) return seq;
                seq.Add(next);
                drop = keep;
                s = next;
            }
        }
        int other1 = -1, other2 = -1;
        for (var j = 0; j < 4; j++)
        {
            var x = t[4 * start + j];
            if (x == edge.A || x == edge.B) continue;
            if (other1 < 0) other1 = x; else other2 = x;
        }
        List<int> chain;
        if (!unbounded) chain = Walk(start, other1);
        else
        {
            // Walk to one end, then back across the whole chain from there.
            var toEnd = Walk(start, other1);
            var end = toEnd[^1];
            var free = -1;
            for (var k = 0; k < 4; k++)
            {
                var x = t[4 * end + k];
                if (x != edge.A && x != edge.B && nb[4 * end + k] < 0) { free = x; break; }
            }
            // At the end, the face opposite `free` is a hull face; walk away from it.
            var keepAtEnd = -1;
            for (var k = 0; k < 4; k++)
            {
                var x = t[4 * end + k];
                if (x != edge.A && x != edge.B && x != free) keepAtEnd = x;
            }
            chain = Walk(end, keepAtEnd);
        }
        var result = new List<int>();
        if (unbounded) result.Add(-1);
        foreach (var s in chain)
        {
            var g = triVertex[s];
            if (result.Count == 0 || result[^1] != g) result.Add(g);
        }
        // Merged (cospherical) runs can wrap around the start of a closed cycle.
        if (!unbounded && result.Count > 1 && result[0] == result[^1]) result.RemoveAt(result.Count - 1);
        return [.. result];
    }

    private static int Slot(int[] t, int s, int v)
    {
        for (var k = 0; k < 3; k++) if (t[3 * s + k] == v) return k;
        throw new InvalidOperationException("Vertex not in triangle.");
    }

    private (double X, double Y) Circumcentre(int a, int b, int c)
    {
        // Relative to a, for accuracy away from the origin.
        double bx = _p[2 * b] - _p[2 * a], by = _p[2 * b + 1] - _p[2 * a + 1];
        double cx = _p[2 * c] - _p[2 * a], cy = _p[2 * c + 1] - _p[2 * a + 1];
        var d = 2 * (bx * cy - by * cx);
        double b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
        return (_p[2 * a] + (cy * b2 - by * c2) / d, _p[2 * a + 1] + (bx * c2 - cx * b2) / d);
    }

    /// <summary>
    /// Every region as a finite polygon clipped to the box, for plotting: unbounded regions are
    /// closed by points far along their two infinite ridges (directed along the outward
    /// normal of the corresponding hull edge) and along the bisector between them, then clipped
    /// with Sutherland-Hodgman. Element i is point i's polygon, counter-clockwise, shape (k, 2)
    /// (possibly empty if the region misses the box).
    /// </summary>
    public NdArray[] FinitePolygons(double xmin, double xmax, double ymin, double ymax)
    {
        if (_d != 2) throw new InvalidOperationException("FinitePolygons is for 2-D diagrams.");
        if (!(xmax > xmin && ymax > ymin)) throw new ArgumentException("The box must have positive width and height.");
        var tri = Triangulation;
        var t = tri.TriangleBuffer; var nb = tri.NeighborBuffer;
        var n = tri.NPoints;
        // A radius comfortably beyond both the box and every vertex.
        double cx = 0.5 * (xmin + xmax), cy = 0.5 * (ymin + ymax);
        var far = Math.Sqrt((xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin));
        for (var i = 0; i < _v.Length / 2; i++) far = Math.Max(far, Math.Abs(_v[2 * i] - cx) + Math.Abs(_v[2 * i + 1] - cy));
        for (var i = 0; i < n; i++) far = Math.Max(far, Math.Abs(_p[2 * i] - cx) + Math.Abs(_p[2 * i + 1] - cy));
        far *= 4;

        var result = new NdArray[n];
        for (var i = 0; i < n; i++)
        {
            var region = Regions[PointRegion[i]];
            var poly = new List<(double, double)>();
            foreach (var v in region) if (v >= 0) poly.Add((_v[2 * v], _v[2 * v + 1]));
            if (region.Length > 0 && region[0] == -1)
            {
                // The two hull edges at point i, from the triangles at either end of its fan.
                var dirs = new List<(double X, double Y, double Ox, double Oy)>();
                for (var s = 0; s < tri.NSimplex; s++)
                    for (var k = 0; k < 3; k++)
                    {
                        if (nb[3 * s + k] >= 0) continue;
                        int a = t[3 * s + (k + 1) % 3], b = t[3 * s + (k + 2) % 3];
                        if (a != i && b != i) continue;
                        // CCW triangle: the outside of edge a->b is on its right.
                        double ex = _p[2 * b] - _p[2 * a], ey = _p[2 * b + 1] - _p[2 * a + 1];
                        var len = Math.Sqrt(ex * ex + ey * ey);
                        var vtx = _triVertex[s];
                        dirs.Add((ey / len, -ex / len, _v[2 * vtx], _v[2 * vtx + 1]));
                    }
                (double X, double Y) mid = (0, 0);
                foreach (var d in dirs)
                {
                    poly.Add((d.Ox + far * d.X, d.Oy + far * d.Y));
                    mid = (mid.X + d.X, mid.Y + d.Y);
                }
                var ml = Math.Sqrt(mid.X * mid.X + mid.Y * mid.Y);
                if (dirs.Count == 2 && ml < 1e-12) mid = (-dirs[0].Y, dirs[0].X); // opposite rays: perpendicular
                ml = Math.Sqrt(mid.X * mid.X + mid.Y * mid.Y);
                if (dirs.Count > 0 && ml > 0)
                {
                    // Either perpendicular may be the outward one; keep the one away from the site.
                    double mx = mid.X / ml, my = mid.Y / ml;
                    if (dirs.Count == 2 && mx * dirs[0].X + my * dirs[0].Y < -1e-12) { mx = -mx; my = -my; }
                    poly.Add((dirs[0].Ox + 2 * far * mx, dirs[0].Oy + 2 * far * my));
                }
            }
            var hull = ConvexPolygon(poly);
            var clipped = Clip(hull, xmin, xmax, ymin, ymax);
            var data = new double[clipped.Count * 2];
            for (var k = 0; k < clipped.Count; k++) { data[2 * k] = clipped[k].Item1; data[2 * k + 1] = clipped[k].Item2; }
            result[i] = new NdArray(data, clipped.Count, 2);
        }
        return result;
    }

    private static List<(double, double)> ConvexPolygon(List<(double X, double Y)> pts)
    {
        var s = pts.Distinct().OrderBy(q => q.X).ThenBy(q => q.Y).ToList();
        if (s.Count < 3) return s.Select(q => (q.X, q.Y)).ToList();
        var h = new List<(double X, double Y)>();
        static double Cross((double X, double Y) o, (double X, double Y) a, (double X, double Y) b)
            => (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
        foreach (var q in s) { while (h.Count >= 2 && Cross(h[^2], h[^1], q) <= 0) h.RemoveAt(h.Count - 1); h.Add(q); }
        var lower = h.Count + 1;
        for (var i = s.Count - 2; i >= 0; i--) { var q = s[i]; while (h.Count >= lower && Cross(h[^2], h[^1], q) <= 0) h.RemoveAt(h.Count - 1); h.Add(q); }
        h.RemoveAt(h.Count - 1);
        return h.Select(q => (q.X, q.Y)).ToList();
    }

    private static List<(double, double)> Clip(List<(double, double)> poly, double xmin, double xmax, double ymin, double ymax)
    {
        poly = ClipEdge(poly, q => q.Item1 >= xmin, (a, b) => Lerp(a, b, (xmin - a.Item1) / (b.Item1 - a.Item1)));
        poly = ClipEdge(poly, q => q.Item1 <= xmax, (a, b) => Lerp(a, b, (xmax - a.Item1) / (b.Item1 - a.Item1)));
        poly = ClipEdge(poly, q => q.Item2 >= ymin, (a, b) => Lerp(a, b, (ymin - a.Item2) / (b.Item2 - a.Item2)));
        poly = ClipEdge(poly, q => q.Item2 <= ymax, (a, b) => Lerp(a, b, (ymax - a.Item2) / (b.Item2 - a.Item2)));
        return poly;
    }

    private static (double, double) Lerp((double, double) a, (double, double) b, double s)
        => (a.Item1 + s * (b.Item1 - a.Item1), a.Item2 + s * (b.Item2 - a.Item2));

    private static List<(double, double)> ClipEdge(List<(double, double)> poly, Func<(double, double), bool> inside,
        Func<(double, double), (double, double), (double, double)> cut)
    {
        var output = new List<(double, double)>();
        for (var i = 0; i < poly.Count; i++)
        {
            var cur = poly[i]; var prev = poly[(i + poly.Count - 1) % poly.Count];
            var ci = inside(cur); var pi = inside(prev);
            if (ci)
            {
                if (!pi) output.Add(cut(prev, cur));
                output.Add(cur);
            }
            else if (pi) output.Add(cut(prev, cur));
        }
        return output;
    }
}
