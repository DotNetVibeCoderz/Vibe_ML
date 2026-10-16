using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// Delaunay triangulation of a 2-D point set (scipy.spatial.Delaunay, 2-D only).
/// </summary>
/// <remarks>
/// <para><b>Algorithm.</b> Incremental Bowyer–Watson insertion with <em>ghost triangles</em>:
/// every hull edge carries a triangle joined to a vertex at infinity, so a point outside the
/// current hull is inserted exactly like an interior one, and there is no "super triangle" whose
/// finite corners can distort the hull. Points are inserted in Hilbert-curve order and located by
/// a visibility walk from the last triangle created, which makes location close to O(1) per
/// point.</para>
/// <para><b>Robustness.</b> Orientation and in-circle tests are exact in sign
/// (<c>SpatialPredicates</c>). A point exactly on a circumcircle does not enter the cavity, so a
/// lattice — where every unit square has four cocircular corners — triangulates cleanly; which
/// diagonal each square gets is arbitrary, as it is in Qhull (scipy adds the <c>Qt</c> option and
/// also picks one), so simplices of degenerate inputs differ from scipy's while both are valid
/// Delaunay triangulations.</para>
/// <para><b>Degenerate input.</b> Exact duplicates are not inserted; they are listed in
/// <see cref="Coplanar"/> as scipy does. All-collinear input throws, as Qhull does.</para>
/// <para><b>Orientation.</b> Every simplex is counter-clockwise. scipy's orientation follows
/// Qhull's facets and is not consistent, so code ported from scipy that relies on a sign should
/// be checked.</para>
/// </remarks>
public sealed class Delaunay
{
    private const int Ghost = -1;

    private readonly double[] _p;
    private readonly int[] _tri;   // 3 per simplex, CCW
    private readonly int[] _nbr;   // neighbour opposite each vertex, -1 at the hull
    private readonly int[] _hullTriangles;
    private double[]? _transform;

    /// <summary>The input points, shape (n, 2).</summary>
    public NdArray Points => new((double[])_p.Clone(), NPoints, 2);

    /// <summary>Number of input points.</summary>
    public int NPoints { get; }

    /// <summary>Number of triangles.</summary>
    public int NSimplex { get; }

    /// <summary>Triangles as point indices, [nsimplex, 3], each counter-clockwise.</summary>
    public int[,] Simplices { get; }

    /// <summary><c>Neighbors[s, k]</c> is the triangle opposite vertex k of triangle s, -1 on the hull.</summary>
    public int[,] Neighbors { get; }

    /// <summary>
    /// Input points that were not inserted because they duplicate an earlier point:
    /// rows [point index, a triangle containing it, the index of the point it duplicates].
    /// </summary>
    public int[,] Coplanar { get; }

    /// <summary>For each point, one triangle that has it as a vertex (-1 for duplicates).</summary>
    public int[] VertexToSimplex { get; }

    /// <summary>Hull edges as point pairs [nedges, 2], each directed counter-clockwise around the hull.</summary>
    public int[,] ConvexHull { get; }

    internal int[] TriangleBuffer => _tri;
    internal int[] NeighborBuffer => _nbr;
    internal double[] PointBuffer => _p;

    /// <summary>Triangulates the rows of <paramref name="points"/>, shape (n, 2).</summary>
    public Delaunay(NdArray points) : this(SpatialInput.Copy(points, out var n, out var d, nameof(points)), n, d) { }

    /// <summary>Triangulates a rectangular array of points.</summary>
    public Delaunay(double[,] points) : this(SpatialInput.Copy(points, out var n, out var d), n, d) { }

    /// <summary>Builds from a jagged array of points.</summary>
    public Delaunay(double[][] points) : this(SpatialInput.Copy(points, out var n, out var d), n, d) { }

    private Delaunay(double[] p, int n, int d)
    {
        if (d != 2) throw new ArgumentException($"Delaunay is implemented for 2-D points only, got {d}-D.");
        if (n < 3) throw new ArgumentException("A 2-D Delaunay triangulation needs at least 3 points.");
        SpatialInput.RequireFinite(p, "Delaunay points");
        _p = p; NPoints = n;

        var builder = new Builder(p, n);
        builder.Run();
        (_tri, _nbr) = builder.Extract();
        NSimplex = _tri.Length / 3;

        Simplices = new int[NSimplex, 3];
        Neighbors = new int[NSimplex, 3];
        var hull = new List<int>();
        var hullTris = new List<int>();
        VertexToSimplex = new int[n];
        Array.Fill(VertexToSimplex, -1);
        for (var s = 0; s < NSimplex; s++)
        {
            var onHull = false;
            for (var k = 0; k < 3; k++)
            {
                Simplices[s, k] = _tri[3 * s + k];
                Neighbors[s, k] = _nbr[3 * s + k];
                if (VertexToSimplex[_tri[3 * s + k]] < 0) VertexToSimplex[_tri[3 * s + k]] = s;
                if (_nbr[3 * s + k] < 0) { hull.Add(_tri[3 * s + (k + 1) % 3]); hull.Add(_tri[3 * s + (k + 2) % 3]); onHull = true; }
            }
            if (onHull) hullTris.Add(s);
        }
        ConvexHull = new int[hull.Count / 2, 2];
        for (var i = 0; i < hull.Count / 2; i++) { ConvexHull[i, 0] = hull[2 * i]; ConvexHull[i, 1] = hull[2 * i + 1]; }

        // Triangles touching the hull by a vertex or an edge: the only candidates for a query
        // point that lies outside the hull but within the barycentric tolerance.
        var hullVertex = new bool[n];
        foreach (var v in hull) hullVertex[v] = true;
        var near = new List<int>();
        for (var s = 0; s < NSimplex; s++)
            if (hullVertex[_tri[3 * s]] || hullVertex[_tri[3 * s + 1]] || hullVertex[_tri[3 * s + 2]]) near.Add(s);
        _hullTriangles = [.. near];

        Coplanar = new int[builder.Duplicates.Count, 3];
        for (var i = 0; i < builder.Duplicates.Count; i++)
        {
            var (dup, orig) = builder.Duplicates[i];
            Coplanar[i, 0] = dup; Coplanar[i, 1] = VertexToSimplex[orig]; Coplanar[i, 2] = orig;
        }
    }

    // ================================================================== queries

    /// <summary>
    /// Barycentric transforms like scipy's <c>transform</c>, shape (nsimplex, 3, 2): rows 0-1 hold
    /// <c>T⁻¹</c> and row 2 the last vertex <c>r</c>, so the barycentric coordinates of x are
    /// <c>c = T⁻¹(x - r)</c> and <c>1 - c₀ - c₁</c>.
    /// </summary>
    public NdArray Transform => new((double[])TransformBuffer.Clone(), NSimplex, 3, 2);

    private double[] TransformBuffer
    {
        get
        {
            if (_transform is not null) return _transform;
            var t = new double[NSimplex * 6];
            for (var s = 0; s < NSimplex; s++)
            {
                int a = _tri[3 * s], b = _tri[3 * s + 1], c = _tri[3 * s + 2];
                double rx = _p[2 * c], ry = _p[2 * c + 1];
                double t00 = _p[2 * a] - rx, t01 = _p[2 * b] - rx;
                double t10 = _p[2 * a + 1] - ry, t11 = _p[2 * b + 1] - ry;
                var det = t00 * t11 - t01 * t10;
                var o = 6 * s;
                if (det == 0)
                {
                    for (var k = 0; k < 4; k++) t[o + k] = double.NaN;
                }
                else
                {
                    t[o] = t11 / det; t[o + 1] = -t01 / det;
                    t[o + 2] = -t10 / det; t[o + 3] = t00 / det;
                }
                t[o + 4] = rx; t[o + 5] = ry;
            }
            Interlocked.CompareExchange(ref _transform, t, null);
            return _transform;
        }
    }

    /// <summary>Barycentric coordinates of (x, y) in triangle s (three values summing to 1).</summary>
    public double[] Barycentric(int s, double x, double y)
    {
        var t = TransformBuffer;
        var o = 6 * s;
        double dx = x - t[o + 4], dy = y - t[o + 5];
        var c0 = t[o] * dx + t[o + 1] * dy;
        var c1 = t[o + 2] * dx + t[o + 3] * dy;
        return [c0, c1, 1 - c0 - c1];
    }

    /// <summary>
    /// The triangle containing each query row, or -1 outside the triangulation (scipy's
    /// <c>find_simplex</c>).
    /// </summary>
    /// <param name="xi">Query points, shape (q, 2), or one point of shape (2).</param>
    /// <param name="bruteForce">Test every triangle instead of walking.</param>
    /// <param name="tol">Barycentric tolerance for points just outside a triangle; default
    /// 100 × machine epsilon, scipy's.</param>
    /// <remarks>A walk using exact orientation tests finds the triangle of any point inside
    /// the hull (a point on a shared edge gets either neighbour). Only points that the walk
    /// places outside the hull are retried against the hull triangles with the tolerance.</remarks>
    public int[] FindSimplex(NdArray xi, bool bruteForce = false, double? tol = null)
    {
        var q = SpatialInput.Copy(xi, out var nq, out var dq, nameof(xi));
        if (xi.Rank == 1) { if (q.Length != 2) throw new ArgumentException("A query point must have 2 coordinates."); nq = 1; }
        else if (dq != 2) throw new ArgumentException("Query points must have 2 coordinates.", nameof(xi));
        var eps = tol ?? 100 * 2.220446049250313e-16;
        var result = new int[nq];
        _ = TransformBuffer;
        if (nq >= 512)
            Parallel.For(0, nq, () => 0, (i, _, start) => { result[i] = Locate(q[2 * i], q[2 * i + 1], bruteForce, eps, ref start); return start; }, _ => { });
        else
        {
            var start = 0;
            for (var i = 0; i < nq; i++) result[i] = Locate(q[2 * i], q[2 * i + 1], bruteForce, eps, ref start);
        }
        return result;
    }

    /// <summary>The triangle containing (x, y), or -1.</summary>
    public int FindSimplex(double x, double y, double? tol = null)
    {
        var start = 0;
        return Locate(x, y, false, tol ?? 100 * 2.220446049250313e-16, ref start);
    }

    private bool InsideTol(int s, double x, double y, double eps)
    {
        var c = Barycentric(s, x, y);
        return c[0] >= -eps && c[1] >= -eps && c[2] >= -eps && c[0] <= 1 + eps && c[1] <= 1 + eps && c[2] <= 1 + eps;
    }

    private int Locate(double x, double y, bool brute, double eps, ref int start)
    {
        if (!double.IsFinite(x) || !double.IsFinite(y)) return -1;
        if (brute)
        {
            for (var s = 0; s < NSimplex; s++) if (InsideTol(s, x, y, eps)) return s;
            return -1;
        }
        var t = start < NSimplex ? start : 0;
        var steps = 0;
        var rot = 0;
        while (true)
        {
            var moved = false;
            rot = (rot + 1) % 3;
            for (var kk = 0; kk < 3; kk++)
            {
                var k = (kk + rot) % 3;
                int a = _tri[3 * t + (k + 1) % 3], b = _tri[3 * t + (k + 2) % 3];
                if (SpatialPredicates.Orient2D(_p[2 * a], _p[2 * a + 1], _p[2 * b], _p[2 * b + 1], x, y) < 0)
                {
                    var nb = _nbr[3 * t + k];
                    if (nb < 0) { t = -1; break; }
                    t = nb; moved = true; break;
                }
            }
            if (t < 0 || !moved) break;
            if (++steps > 4 * NSimplex + 16) { t = -1; break; } // cannot happen with exact predicates
        }
        if (t >= 0) { start = t; return t; }
        foreach (var s in _hullTriangles) if (InsideTol(s, x, y, eps)) { start = s; return s; }
        return -1;
    }

    /// <summary>
    /// The Delaunay neighbours of every vertex in CSR form (scipy's
    /// <c>vertex_neighbor_vertices</c>): the neighbours of k are
    /// <c>Indices[Indptr[k]..Indptr[k+1]]</c>, here sorted ascending.
    /// </summary>
    public (int[] Indptr, int[] Indices) VertexNeighborVertices()
    {
        var sets = new SortedSet<int>[NPoints];
        for (var i = 0; i < NPoints; i++) sets[i] = [];
        for (var s = 0; s < NSimplex; s++)
            for (var k = 0; k < 3; k++)
            {
                int a = _tri[3 * s + k], b = _tri[3 * s + (k + 1) % 3];
                sets[a].Add(b); sets[b].Add(a);
            }
        var indptr = new int[NPoints + 1];
        for (var i = 0; i < NPoints; i++) indptr[i + 1] = indptr[i] + sets[i].Count;
        var indices = new int[indptr[NPoints]];
        for (var i = 0; i < NPoints; i++) sets[i].CopyTo(indices, indptr[i]);
        return (indptr, indices);
    }

    // ================================================================== construction

    private sealed class Builder(double[] p, int n)
    {
        private readonly List<int> _v = new(6 * n + 16);
        private readonly List<int> _nb = new(6 * n + 16);
        private readonly List<bool> _alive = new(2 * n + 8);
        private readonly Stack<int> _free = new();
        private int _last;
        private int _walkRot;

        public readonly List<(int Dup, int Orig)> Duplicates = [];

        private double X(int v) => p[2 * v];
        private double Y(int v) => p[2 * v + 1];

        private int NewTriangle(int a, int b, int c)
        {
            int t;
            if (_free.Count > 0)
            {
                t = _free.Pop();
                _v[3 * t] = a; _v[3 * t + 1] = b; _v[3 * t + 2] = c;
                _alive[t] = true;
            }
            else
            {
                t = _alive.Count;
                _v.Add(a); _v.Add(b); _v.Add(c);
                _nb.Add(-1); _nb.Add(-1); _nb.Add(-1);
                _alive.Add(true);
            }
            return t;
        }

        private bool IsGhost(int t) => _v[3 * t + 2] == Ghost;

        private double Orient(int a, int b, double x, double y) => SpatialPredicates.Orient2D(X(a), Y(a), X(b), Y(b), x, y);

        /// <summary>Bowyer-Watson cavity test, with the ghost-triangle rule for the hull.</summary>
        private bool InCavity(int t, double x, double y)
        {
            int a = _v[3 * t], b = _v[3 * t + 1], c = _v[3 * t + 2];
            if (c == Ghost)
            {
                var o = Orient(a, b, x, y);
                if (o > 0) return true;
                if (o < 0) return false;
                // On the hull line: inside the cavity only on the open segment ab.
                double ax = X(a), ay = Y(a), bx = X(b), by = Y(b);
                var dot = (x - ax) * (bx - ax) + (y - ay) * (by - ay);
                var len2 = (bx - ax) * (bx - ax) + (by - ay) * (by - ay);
                return dot > 0 && dot < len2 && !(x == bx && y == by) && !(x == ax && y == ay);
            }
            return SpatialPredicates.InCircle(X(a), Y(a), X(b), Y(b), X(c), Y(c), x, y) > 0;
        }

        public void Run()
        {
            // Exact duplicates are recorded and skipped.
            var seen = new Dictionary<(double, double), int>();
            var unique = new List<int>(n);
            for (var i = 0; i < n; i++)
            {
                var key = (p[2 * i] + 0.0, p[2 * i + 1] + 0.0); // + 0.0 folds -0 into +0
                if (seen.TryGetValue(key, out var orig)) Duplicates.Add((i, orig));
                else { seen[key] = i; unique.Add(i); }
            }
            var order = HilbertOrder(unique);

            // First non-degenerate triangle in insertion order.
            var i0 = order[0];
            var i1 = order[1];
            var k2 = -1;
            for (var k = 2; k < order.Length; k++)
                if (Orient(i0, i1, X(order[k]), Y(order[k])) != 0) { k2 = k; break; }
            if (k2 < 0) throw new ArgumentException("Initial simplex is flat: all points are collinear (Qhull QH6154).");
            var i2 = order[k2];
            if (Orient(i0, i1, X(i2), Y(i2)) < 0) (i1, i2) = (i2, i1);

            var t0 = NewTriangle(i0, i1, i2);
            var g0 = NewTriangle(i2, i1, Ghost); // opposite i0: edge (i1, i2) reversed
            var g1 = NewTriangle(i0, i2, Ghost); // edge (i2, i0) reversed
            var g2 = NewTriangle(i1, i0, Ghost); // edge (i0, i1) reversed
            SetN(t0, g0, g1, g2);
            // Ghost (a, b, G): N[2] = real across (a,b); N[0] = across (b,G) = ghost starting at b;
            // N[1] = across (G,a) = ghost ending at a.
            SetN(g0, g2, g1, t0); // (i2,i1,G): next ghost starts at i1 -> g2; ghost ending at i2 -> g1
            SetN(g1, g0, g2, t0); // (i0,i2,G): starts at i2 -> g0; ends at i0 -> g2
            SetN(g2, g1, g0, t0); // (i1,i0,G): starts at i0 -> g1; ends at i1 -> g0
            _last = t0;

            var cavity = new List<int>();
            var stack = new Stack<int>();
            var mark = new List<int>();
            var boundary = new List<(int U, int V, int Out, int OutSlot)>();
            var startOf = new Dictionary<int, int>();
            var endOf = new Dictionary<int, int>();
            var stamp = 0;
            for (var k = 2; k < order.Length; k++)
            {
                if (k == k2) continue;
                var v = order[k];
                double x = X(v), y = Y(v);
                var start = Locate(x, y);

                stamp++;
                while (mark.Count < _alive.Count) mark.Add(0);
                cavity.Clear(); boundary.Clear();
                stack.Push(start); mark[start] = stamp;
                while (stack.Count > 0)
                {
                    var t = stack.Pop();
                    cavity.Add(t);
                    for (var e = 0; e < 3; e++)
                    {
                        var nb = _nb[3 * t + e];
                        if (mark[nb] == stamp) continue;
                        if (InCavity(nb, x, y)) { mark[nb] = stamp; stack.Push(nb); }
                    }
                }
                foreach (var t in cavity)
                    for (var e = 0; e < 3; e++)
                    {
                        var nb = _nb[3 * t + e];
                        if (mark[nb] == stamp) continue;
                        var slot = 0;
                        while (_nb[3 * nb + slot] != t) slot++;
                        boundary.Add((_v[3 * t + (e + 1) % 3], _v[3 * t + (e + 2) % 3], nb, slot));
                    }
                foreach (var t in cavity) { _alive[t] = false; _free.Push(t); }

                startOf.Clear(); endOf.Clear();
                var created = new int[boundary.Count];
                for (var bi = 0; bi < boundary.Count; bi++)
                {
                    var (u, w, _, _) = boundary[bi];
                    var t = NewTriangle(u, w, v);
                    created[bi] = t;
                    startOf[u] = t; endOf[w] = t;
                }
                while (mark.Count < _alive.Count) mark.Add(0);
                for (var bi = 0; bi < boundary.Count; bi++)
                {
                    var (u, w, outside, slot) = boundary[bi];
                    var t = created[bi];
                    // Unrotated (u, w, v): N[0] across (w, v), N[1] across (v, u), N[2] across (u, w).
                    SetN(t, startOf[w], endOf[u], outside);
                    _nb[3 * outside + slot] = t;
                }
                // Rotate ghosts so the vertex at infinity sits in slot 2 (rotation keeps N[k]
                // opposite V[k]).
                foreach (var t in created)
                {
                    if (_v[3 * t] == Ghost) Rotate(t, 1);
                    else if (_v[3 * t + 1] == Ghost) Rotate(t, 2);
                }
                foreach (var t in created) if (!IsGhost(t)) { _last = t; break; }
            }
        }

        private void Rotate(int t, int shift)
        {
            Span<int> v = [_v[3 * t], _v[3 * t + 1], _v[3 * t + 2]];
            Span<int> nb = [_nb[3 * t], _nb[3 * t + 1], _nb[3 * t + 2]];
            for (var k = 0; k < 3; k++)
            {
                _v[3 * t + k] = v[(k + shift) % 3];
                _nb[3 * t + k] = nb[(k + shift) % 3];
            }
        }

        private void SetN(int t, int a, int b, int c)
        {
            _nb[3 * t] = a; _nb[3 * t + 1] = b; _nb[3 * t + 2] = c;
        }

        /// <summary>Visibility walk; returns a triangle (real or ghost) whose cavity test holds for (x, y).</summary>
        private int Locate(double x, double y)
        {
            var t = _last;
            if (!_alive[t]) for (t = 0; !_alive[t]; t++) { }
            var guard = 0;
            while (true)
            {
                if (IsGhost(t))
                {
                    if (InCavity(t, x, y)) return t;
                    t = _nb[3 * t + 2];
                    continue;
                }
                _walkRot = (_walkRot + 1) % 3;
                var moved = false;
                for (var kk = 0; kk < 3; kk++)
                {
                    var k = (kk + _walkRot) % 3;
                    int a = _v[3 * t + (k + 1) % 3], b = _v[3 * t + (k + 2) % 3];
                    if (Orient(a, b, x, y) < 0) { t = _nb[3 * t + k]; moved = true; break; }
                }
                if (!moved) return t;
                if (++guard > 8 * _alive.Count + 64)
                    throw new InvalidOperationException("Delaunay point location did not terminate (corrupted mesh).");
            }
        }

        private int[] HilbertOrder(List<int> pts)
        {
            double minX = double.PositiveInfinity, minY = double.PositiveInfinity, maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;
            foreach (var i in pts)
            {
                minX = Math.Min(minX, X(i)); maxX = Math.Max(maxX, X(i));
                minY = Math.Min(minY, Y(i)); maxY = Math.Max(maxY, Y(i));
            }
            var span = Math.Max(maxX - minX, maxY - minY);
            if (span == 0) span = 1;
            const int side = 1 << 15; // keys below 2^30, so key * 2^31 + index fits a long
            var keys = new long[pts.Count];
            var idx = pts.ToArray();
            for (var k = 0; k < idx.Length; k++)
            {
                var hx = (int)Math.Min(side - 1, (X(idx[k]) - minX) / span * (side - 1));
                var hy = (int)Math.Min(side - 1, (Y(idx[k]) - minY) / span * (side - 1));
                keys[k] = (HilbertD(side, hx, hy) << 31) + idx[k];
            }
            Array.Sort(keys, idx);
            return idx;
        }

        private static long HilbertD(int side, int x, int y)
        {
            long d = 0;
            for (var s = side / 2; s > 0; s /= 2)
            {
                var rx = (x & s) > 0 ? 1 : 0;
                var ry = (y & s) > 0 ? 1 : 0;
                d += (long)s * s * ((3 * rx) ^ ry);
                if (ry == 0)
                {
                    if (rx == 1) { x = s - 1 - x; y = s - 1 - y; }
                    (x, y) = (y, x);
                }
            }
            return d;
        }

        public (int[] Tri, int[] Nbr) Extract()
        {
            var map = new int[_alive.Count];
            var count = 0;
            for (var t = 0; t < _alive.Count; t++)
                map[t] = _alive[t] && !IsGhost(t) ? count++ : -1;
            var tri = new int[3 * count];
            var nbr = new int[3 * count];
            for (var t = 0; t < _alive.Count; t++)
            {
                if (map[t] < 0) continue;
                for (var k = 0; k < 3; k++)
                {
                    tri[3 * map[t] + k] = _v[3 * t + k];
                    nbr[3 * map[t] + k] = map[_nb[3 * t + k]];
                }
            }
            return (tri, nbr);
        }
    }
}
