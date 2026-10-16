using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// 2-D Voronoi diagram, built as the dual of <see cref="Delaunay"/> (scipy.spatial.Voronoi).
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
/// </remarks>
public sealed class Voronoi
{
    private readonly double[] _p;
    private readonly double[] _v;
    private readonly int[] _triVertex;

    /// <summary>The input points, shape (n, 2).</summary>
    public NdArray Points => new((double[])_p.Clone(), _p.Length / 2, 2);

    /// <summary>Voronoi vertices, shape (nvertices, 2).</summary>
    public NdArray Vertices => new((double[])_v.Clone(), _v.Length / 2, 2);

    /// <summary>Pairs of input points separated by each ridge, [nridges, 2].</summary>
    public int[,] RidgePoints { get; }

    /// <summary>Voronoi vertices of each ridge, [nridges, 2]; -1 marks the vertex at infinity.</summary>
    public int[,] RidgeVertices { get; }

    /// <summary>Each region's vertices, counter-clockwise; -1 (listed first) when unbounded.</summary>
    public int[][] Regions { get; }

    /// <summary>Index into <see cref="Regions"/> for each input point.</summary>
    public int[] PointRegion { get; }

    /// <summary>The underlying triangulation.</summary>
    public Delaunay Triangulation { get; }

    /// <summary>Builds the diagram of the rows of <paramref name="points"/>, shape (n, 2).</summary>
    public Voronoi(NdArray points) : this(new Delaunay(points)) { }

    /// <summary>Builds the diagram of a rectangular array of points.</summary>
    public Voronoi(double[,] points) : this(new Delaunay(points)) { }

    private Voronoi(Delaunay tri)
    {
        Triangulation = tri;
        _p = tri.PointBuffer;
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
        RidgeVertices = new int[rv.Count, 2];
        for (var i = 0; i < rp.Count; i++)
        {
            RidgePoints[i, 0] = rp[i].Item1; RidgePoints[i, 1] = rp[i].Item2;
            RidgeVertices[i, 0] = rv[i].Item1; RidgeVertices[i, 1] = rv[i].Item2;
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
