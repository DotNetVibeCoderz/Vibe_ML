using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// Delaunay triangulation of a point set in any dimension ≥ 2 (scipy.spatial.Delaunay).
/// </summary>
/// <remarks>
/// <para><b>2-D algorithm.</b> Incremental Bowyer–Watson insertion with <em>ghost triangles</em>:
/// every hull edge carries a triangle joined to a vertex at infinity, so a point outside the
/// current hull is inserted exactly like an interior one, and there is no "super triangle" whose
/// finite corners can distort the hull. Points are inserted in Hilbert-curve order and located by
/// a visibility walk from the last triangle created, which makes location close to O(1) per
/// point.</para>
/// <para><b>3-D and above</b> follow Qhull: the points are lifted onto the paraboloid
/// <c>x_{d+1} = |x|²</c>, the convex hull of the lifted set is built by quickhull in d + 1
/// dimensions (<c>QuickhullNd</c>), and the facets whose outward normal points down are the
/// simplices. As with Qhull's <c>Qz</c> option an extra point is placed far above the
/// paraboloid, so that cospherical inputs (the corners of a cube lift to a flat set) still have
/// a full-dimensional hull; it only ever appears in upper facets. Every visibility test is an
/// exact lifted-orientation (in-sphere) determinant, so there is nothing to "joggle" (Qhull's
/// <c>QJ</c>) and no facet merging: cospherical cells come out triangulated (Qhull's <c>Qt</c>),
/// split differently from scipy's but always into valid Delaunay simplices.</para>
/// <para><b>Robustness.</b> Orientation and in-circle / in-sphere tests are exact in sign
/// (<c>SpatialPredicates</c>). A point exactly on a circumsphere does not enter the cavity /
/// is not beyond the lifted facet, so a lattice triangulates cleanly; which diagonal each cell
/// gets is arbitrary, as it is in Qhull, so simplices of degenerate inputs differ from scipy's
/// while both are valid Delaunay triangulations.</para>
/// <para><b>Degenerate input.</b> Exact duplicates are not inserted; they are listed in
/// <see cref="Coplanar"/> as scipy does. Input lying in a lower-dimensional affine subspace
/// (collinear in 2-D, coplanar in 3-D) throws, as Qhull does.</para>
/// <para><b>Orientation.</b> Every simplex is positively oriented (counter-clockwise in 2-D,
/// <c>det[v₁−v₀, …, v_d−v₀] &gt; 0</c> in general). scipy's orientation follows Qhull's facets and
/// is not consistent, so code ported from scipy that relies on a sign should be checked.</para>
/// </remarks>
public sealed class Delaunay
{
    private const int Ghost = -1;

    private readonly double[] _p;
    private readonly int _d;
    private readonly int[] _tri;   // d + 1 per simplex, positively oriented
    private readonly int[] _nbr;   // neighbour opposite each vertex, -1 at the hull
    private readonly int[] _hullTriangles;
    private double[]? _transform;

    /// <summary>The input points, shape (n, d).</summary>
    public NdArray Points => new((double[])_p.Clone(), NPoints, _d);

    /// <summary>Number of input points.</summary>
    public int NPoints { get; }

    /// <summary>Dimension of the points.</summary>
    public int NDim => _d;

    /// <summary>Number of simplices.</summary>
    public int NSimplex { get; }

    /// <summary>Simplices as point indices, [nsimplex, d + 1], each positively oriented
    /// (counter-clockwise in 2-D).</summary>
    public int[,] Simplices { get; }

    /// <summary><c>Neighbors[s, k]</c> is the simplex opposite vertex k of simplex s, -1 on the hull.</summary>
    public int[,] Neighbors { get; }

    /// <summary>
    /// Input points that were not inserted because they duplicate an earlier point:
    /// rows [point index, a simplex containing it, the index of the point it duplicates].
    /// </summary>
    public int[,] Coplanar { get; }

    /// <summary>For each point, one simplex that has it as a vertex (-1 for duplicates).</summary>
    public int[] VertexToSimplex { get; }

    /// <summary>Hull facets as point indices [nfacets, d]: in 2-D edges directed counter-clockwise
    /// around the hull; above, the vertices of each simplex face that has no neighbour.</summary>
    public int[,] ConvexHull { get; }

    internal int[] TriangleBuffer => _tri;
    internal int[] NeighborBuffer => _nbr;
    internal double[] PointBuffer => _p;

    /// <summary>Triangulates the rows of <paramref name="points"/>, shape (n, d), d ≥ 2.</summary>
    public Delaunay(NdArray points) : this(SpatialInput.Copy(points, out var n, out var d, nameof(points)), n, d) { }

    /// <summary>Triangulates a rectangular array of points.</summary>
    public Delaunay(double[,] points) : this(SpatialInput.Copy(points, out var n, out var d), n, d) { }

    /// <summary>Builds from a jagged array of points.</summary>
    public Delaunay(double[][] points) : this(SpatialInput.Copy(points, out var n, out var d), n, d) { }

    private Delaunay(double[] p, int n, int d)
    {
        if (d < 2) throw new ArgumentException($"Delaunay needs points of dimension 2 or more, got {d}-D.");
        if (n < d + 1) throw new ArgumentException($"A {d}-D Delaunay triangulation needs at least {d + 1} points.");
        if (d + 1 > SpatialPredicates.MaxNdDimension) throw new ArgumentException($"Delaunay supports at most {SpatialPredicates.MaxNdDimension - 1} dimensions, got {d}.");
        SpatialInput.RequireFinite(p, "Delaunay points");
        _p = p; NPoints = n; _d = d;
        var m = d + 1;

        List<(int Dup, int Orig)> duplicates;
        if (d == 2)
        {
            var builder = new Builder(p, n);
            builder.Run();
            (_tri, _nbr) = builder.Extract();
            duplicates = builder.Duplicates;
        }
        else (_tri, _nbr, duplicates) = BuildLifted(p, n, d);
        NSimplex = _tri.Length / m;

        Simplices = new int[NSimplex, m];
        Neighbors = new int[NSimplex, m];
        var hull = new List<int>();
        VertexToSimplex = new int[n];
        Array.Fill(VertexToSimplex, -1);
        for (var s = 0; s < NSimplex; s++)
        {
            for (var k = 0; k < m; k++)
            {
                Simplices[s, k] = _tri[m * s + k];
                Neighbors[s, k] = _nbr[m * s + k];
                if (VertexToSimplex[_tri[m * s + k]] < 0) VertexToSimplex[_tri[m * s + k]] = s;
                if (_nbr[m * s + k] >= 0) continue;
                if (d == 2) { hull.Add(_tri[3 * s + (k + 1) % 3]); hull.Add(_tri[3 * s + (k + 2) % 3]); }
                else for (var j = 0; j < m; j++) if (j != k) hull.Add(_tri[m * s + j]);
            }
        }
        ConvexHull = new int[hull.Count / d, d];
        for (var i = 0; i < hull.Count / d; i++) for (var j = 0; j < d; j++) ConvexHull[i, j] = hull[d * i + j];

        // Simplices touching the hull by a vertex or a face: the only candidates for a query
        // point that lies outside the hull but within the barycentric tolerance.
        var hullVertex = new bool[n];
        foreach (var v in hull) hullVertex[v] = true;
        var near = new List<int>();
        for (var s = 0; s < NSimplex; s++)
            for (var k = 0; k < m; k++)
                if (hullVertex[_tri[m * s + k]]) { near.Add(s); break; }
        _hullTriangles = [.. near];

        Coplanar = new int[duplicates.Count, 3];
        for (var i = 0; i < duplicates.Count; i++)
        {
            var (dup, orig) = duplicates[i];
            Coplanar[i, 0] = dup; Coplanar[i, 1] = VertexToSimplex[orig]; Coplanar[i, 2] = orig;
        }
    }

    // ================================================================== queries

    /// <summary>
    /// Barycentric transforms like scipy's <c>transform</c>, shape (nsimplex, d + 1, d): rows
    /// 0..d−1 hold <c>T⁻¹</c> and row d the last vertex <c>r</c>, so the barycentric coordinates of
    /// x are <c>c = T⁻¹(x − r)</c> and <c>1 − Σc</c>. A degenerate simplex gets NaN, as in scipy.
    /// </summary>
    public NdArray Transform => new((double[])TransformBuffer.Clone(), NSimplex, _d + 1, _d);

    internal double[] TransformBuffer
    {
        get
        {
            if (_transform is not null) return _transform;
            var d = _d; var m = d + 1;
            var t = new double[NSimplex * m * d];
            if (d == 2)
            {
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
            }
            else
            {
                Parallel.For(0, NSimplex, new ParallelOptions { MaxDegreeOfParallelism = NSimplex > 2048 ? -1 : 1 }, s =>
                {
                    Span<double> a = stackalloc double[d * d];
                    Span<double> inv = stackalloc double[d * d];
                    var r = _tri[m * s + d];
                    // T[i, j] = v_j[i] − r[i].
                    for (var j = 0; j < d; j++)
                    {
                        var vj = _tri[m * s + j];
                        for (var i = 0; i < d; i++) a[i * d + j] = _p[vj * d + i] - _p[r * d + i];
                    }
                    var o = s * m * d;
                    if (InvertInPlace(a, inv, d)) inv.CopyTo(t.AsSpan(o, d * d));
                    else t.AsSpan(o, d * d).Fill(double.NaN);
                    for (var i = 0; i < d; i++) t[o + d * d + i] = _p[r * d + i];
                });
            }
            Interlocked.CompareExchange(ref _transform, t, null);
            return _transform;
        }
    }

    /// <summary>Gauss–Jordan inverse with partial pivoting; false when singular.</summary>
    private static bool InvertInPlace(Span<double> a, Span<double> inv, int d)
    {
        inv.Clear();
        for (var i = 0; i < d; i++) inv[i * d + i] = 1;
        for (var c = 0; c < d; c++)
        {
            var piv = c; var best = Math.Abs(a[c * d + c]);
            for (var r = c + 1; r < d; r++) { var v = Math.Abs(a[r * d + c]); if (v > best) { best = v; piv = r; } }
            if (best == 0) return false;
            if (piv != c)
                for (var j = 0; j < d; j++)
                {
                    (a[c * d + j], a[piv * d + j]) = (a[piv * d + j], a[c * d + j]);
                    (inv[c * d + j], inv[piv * d + j]) = (inv[piv * d + j], inv[c * d + j]);
                }
            var pv = a[c * d + c];
            for (var j = 0; j < d; j++) { a[c * d + j] /= pv; inv[c * d + j] /= pv; }
            for (var r = 0; r < d; r++)
            {
                if (r == c) continue;
                var f = a[r * d + c];
                if (f == 0) continue;
                for (var j = 0; j < d; j++) { a[r * d + j] -= f * a[c * d + j]; inv[r * d + j] -= f * inv[c * d + j]; }
            }
        }
        return true;
    }

    /// <summary>Barycentric coordinates of (x, y) in triangle s (three values summing to 1); 2-D only.</summary>
    public double[] Barycentric(int s, double x, double y)
    {
        if (_d != 2) throw new InvalidOperationException("Barycentric(s, x, y) is for 2-D triangulations; pass the point as a span.");
        var t = TransformBuffer;
        var o = 6 * s;
        double dx = x - t[o + 4], dy = y - t[o + 5];
        var c0 = t[o] * dx + t[o + 1] * dy;
        var c1 = t[o + 2] * dx + t[o + 3] * dy;
        return [c0, c1, 1 - c0 - c1];
    }

    /// <summary>Barycentric coordinates of <paramref name="x"/> (d values) in simplex s: d + 1
    /// values summing to 1, in the order of <see cref="Simplices"/>.</summary>
    public double[] Barycentric(int s, ReadOnlySpan<double> x)
    {
        if (x.Length != _d) throw new ArgumentException($"The point must have {_d} coordinates.", nameof(x));
        var c = new double[_d + 1];
        BarycentricInto(TransformBuffer, s, x, c);
        return c;
    }

    internal void BarycentricInto(double[] t, int s, ReadOnlySpan<double> x, Span<double> c)
    {
        var d = _d;
        var o = s * (d + 1) * d;
        double last = 1;
        for (var i = 0; i < d; i++)
        {
            double v = 0;
            for (var j = 0; j < d; j++) v += t[o + i * d + j] * (x[j] - t[o + d * d + j]);
            c[i] = v;
            last -= v;
        }
        c[d] = last;
    }

    /// <summary>
    /// The simplex containing each query row, or -1 outside the triangulation (scipy's
    /// <c>find_simplex</c>).
    /// </summary>
    /// <param name="xi">Query points, shape (q, d), or one point of shape (d).</param>
    /// <param name="bruteForce">Test every simplex instead of walking.</param>
    /// <param name="tol">Barycentric tolerance for points just outside a simplex; default
    /// 100 × machine epsilon, scipy's.</param>
    /// <remarks>A walk using exact orientation tests finds the simplex of any point inside
    /// the hull (a point on a shared face gets either neighbour). Only points that the walk
    /// places outside the hull are retried against the hull simplices with the tolerance.</remarks>
    public int[] FindSimplex(NdArray xi, bool bruteForce = false, double? tol = null)
    {
        var d = _d;
        var q = SpatialInput.Copy(xi, out var nq, out var dq, nameof(xi));
        if (xi.Rank == 1) { if (q.Length != d) throw new ArgumentException($"A query point must have {d} coordinates."); nq = 1; }
        else if (dq != d) throw new ArgumentException($"Query points must have {d} coordinates.", nameof(xi));
        var eps = tol ?? 100 * 2.220446049250313e-16;
        var result = new int[nq];
        _ = TransformBuffer;
        // Queries are walked in Morton (Z-curve) order, each walk starting from the previous
        // answer, so consecutive walks are short; results go back to their original rows.
        var order = nq >= 64 && !bruteForce ? MortonOrder(q, nq, d) : null;
        const int chunk = 256;
        var chunks = (nq + chunk - 1) / chunk;
        void Run(int c)
        {
            var start = 0;
            for (var t = c * chunk; t < Math.Min(nq, (c + 1) * chunk); t++)
            {
                var i = order is null ? t : order[t];
                result[i] = Locate(q.AsSpan(d * i, d), bruteForce, eps, ref start);
            }
        }
        if (nq >= 512) Parallel.For(0, chunks, Run);
        else for (var c = 0; c < chunks; c++) Run(c);
        return result;
    }

    private static int[] MortonOrder(double[] q, int nq, int d)
    {
        var bits = Math.Min(21, 63 / d);
        var lo = new double[d]; var hi = new double[d];
        Array.Fill(lo, double.PositiveInfinity); Array.Fill(hi, double.NegativeInfinity);
        for (var i = 0; i < nq; i++)
            for (var j = 0; j < d; j++)
            {
                var v = q[i * d + j];
                if (!double.IsFinite(v)) continue;
                lo[j] = Math.Min(lo[j], v); hi[j] = Math.Max(hi[j], v);
            }
        var keys = new ulong[nq];
        var order = new int[nq];
        var cells = (1UL << bits) - 1;
        for (var i = 0; i < nq; i++)
        {
            ulong key = 0;
            for (var j = 0; j < d; j++)
            {
                var span = hi[j] - lo[j];
                var v = q[i * d + j];
                var c = span > 0 && double.IsFinite(v) ? (ulong)Math.Clamp((v - lo[j]) / span * cells, 0, cells) : 0;
                for (var b = 0; b < bits; b++) key |= ((c >> b) & 1UL) << (b * d + j);
            }
            keys[i] = key;
            order[i] = i;
        }
        Array.Sort(keys, order);
        return order;
    }

    /// <summary>The triangle containing (x, y), or -1 (2-D).</summary>
    public int FindSimplex(double x, double y, double? tol = null)
    {
        if (_d != 2) throw new InvalidOperationException("FindSimplex(x, y) is for 2-D triangulations.");
        var start = 0;
        return Locate([x, y], false, tol ?? 100 * 2.220446049250313e-16, ref start);
    }

    /// <summary>The simplex containing the point <paramref name="x"/> (d values), or -1.</summary>
    public int FindSimplex(ReadOnlySpan<double> x, double? tol = null)
    {
        if (x.Length != _d) throw new ArgumentException($"The point must have {_d} coordinates.", nameof(x));
        _ = TransformBuffer;
        var start = 0;
        return Locate(x, false, tol ?? 100 * 2.220446049250313e-16, ref start);
    }

    private bool InsideTol(int s, ReadOnlySpan<double> x, double eps)
    {
        Span<double> c = stackalloc double[_d + 1];
        BarycentricInto(TransformBuffer, s, x, c);
        foreach (var v in c) if (!(v >= -eps && v <= 1 + eps)) return false;
        return true;
    }

    private int Locate(ReadOnlySpan<double> x, bool brute, double eps, ref int start)
    {
        foreach (var v in x) if (!double.IsFinite(v)) return -1;
        if (brute)
        {
            for (var s = 0; s < NSimplex; s++) if (InsideTol(s, x, eps)) return s;
            return -1;
        }
        var t = start < NSimplex ? start : 0;
        t = _d == 2 ? Walk2D(x[0], x[1], t) : WalkNd(x, t);
        if (t >= 0) { start = t; return t; }
        foreach (var s in _hullTriangles) if (InsideTol(s, x, eps)) { start = s; return s; }
        return -1;
    }

    private int Walk2D(double x, double y, int t)
    {
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
                    if (nb < 0) return -1;
                    t = nb; moved = true; break;
                }
            }
            if (!moved) return t;
            if (++steps > 4 * NSimplex + 16) return -1; // cannot happen with exact predicates
        }
    }

    /// <summary>
    /// Visibility walk with exact signs. For a positively oriented simplex the barycentric
    /// coordinate λ_k(x) has the sign of <c>(−1)^k · det[F_k − x]</c>, F_k being the simplex
    /// without vertex k in order: at x = v_k that determinant is <c>(−1)^k det[v_i − v₀]</c>.
    /// </summary>
    private int WalkNd(ReadOnlySpan<double> x, int t)
    {
        var d = _d; var m = d + 1;
        Span<int> face = stackalloc int[d];
        var steps = 0;
        var rot = 0;
        while (true)
        {
            var moved = false;
            rot = (rot + 1) % m;
            for (var kk = 0; kk < m; kk++)
            {
                var k = (kk + rot) % m;
                var w = 0;
                for (var j = 0; j < m; j++) if (j != k) face[w++] = _tri[m * t + j];
                var o = SpatialPredicates.OrientNd(_p, d, false, null, face, x, 0);
                if ((k & 1) == 1) o = -o;
                if (o < 0)
                {
                    var nb = _nbr[m * t + k];
                    if (nb < 0) return -1;
                    t = nb; moved = true; break;
                }
            }
            if (!moved) return t;
            if (++steps > 4 * NSimplex + 16) return -1; // cannot happen with exact predicates
        }
    }

    /// <summary>
    /// The Delaunay neighbours of every vertex in CSR form (scipy's
    /// <c>vertex_neighbor_vertices</c>): the neighbours of k are
    /// <c>Indices[Indptr[k]..Indptr[k+1]]</c>, here sorted ascending.
    /// </summary>
    public (int[] Indptr, int[] Indices) VertexNeighborVertices()
    {
        var m = _d + 1;
        var sets = new SortedSet<int>[NPoints];
        for (var i = 0; i < NPoints; i++) sets[i] = [];
        for (var s = 0; s < NSimplex; s++)
            for (var k = 0; k < m; k++)
                for (var j = k + 1; j < m; j++)
                {
                    int a = _tri[m * s + k], b = _tri[m * s + j];
                    sets[a].Add(b); sets[b].Add(a);
                }
        var indptr = new int[NPoints + 1];
        for (var i = 0; i < NPoints; i++) indptr[i + 1] = indptr[i] + sets[i].Count;
        var indices = new int[indptr[NPoints]];
        for (var i = 0; i < NPoints; i++) sets[i].CopyTo(indices, indptr[i]);
        return (indptr, indices);
    }

    /// <summary>
    /// The lifted points <c>(x, |x|²)</c>, shape (n, d + 1) (scipy's <c>lift_points</c> with
    /// unit scale and zero shift; Qhull's own scale and shift are not reproduced).
    /// </summary>
    public NdArray LiftPoints(NdArray x)
    {
        var q = SpatialInput.Copy(x, out var nq, out var dq, nameof(x));
        if (x.Rank == 1) { nq = 1; dq = q.Length; }
        if (dq != _d) throw new ArgumentException($"Points must have {_d} coordinates.", nameof(x));
        var r = new double[nq * (_d + 1)];
        for (var i = 0; i < nq; i++)
        {
            double s = 0;
            for (var j = 0; j < _d; j++) { var v = q[i * _d + j]; r[i * (_d + 1) + j] = v; s += v * v; }
            r[i * (_d + 1) + _d] = s;
        }
        return new NdArray(r, nq, _d + 1);
    }

    // ================================================================== N-d construction

    private static (int[] Tri, int[] Nbr, List<(int, int)> Duplicates) BuildLifted(double[] p, int n, int d)
    {
        var m = d + 1;
        // Exact duplicates are recorded and skipped (+ 0.0 folds -0 into +0).
        var seen = new Dictionary<RowKey, int>();
        var duplicates = new List<(int, int)>();
        var unique = new List<int>(n);
        for (var i = 0; i < n; i++)
        {
            var key = new RowKey(p, i, d);
            if (seen.TryGetValue(key, out var orig)) duplicates.Add((i, orig));
            else { seen[key] = i; unique.Add(i); }
        }
        if (unique.Count < d + 1) throw new ArgumentException($"A {d}-D Delaunay triangulation needs at least {d + 1} distinct points.");

        // The Qz point: centre of the bounding box, raised by Δ above the paraboloid. Δ is a power
        // of two (exact) at least 8× every |x − top|², so the point is strictly above the upper
        // hull of the lifted input and can never be a vertex of a lower facet.
        var lo = new double[d]; var hi = new double[d];
        Array.Fill(lo, double.PositiveInfinity); Array.Fill(hi, double.NegativeInfinity);
        foreach (var i in unique)
            for (var j = 0; j < d; j++) { lo[j] = Math.Min(lo[j], p[i * d + j]); hi[j] = Math.Max(hi[j], p[i * d + j]); }
        var p2 = new double[(n + 1) * d];
        Array.Copy(p, p2, n * d);
        double span = 0;
        for (var j = 0; j < d; j++) { p2[n * d + j] = 0.5 * (lo[j] + hi[j]); span = Math.Max(span, hi[j] - lo[j]); }
        double maxR2 = 0;
        foreach (var i in unique)
        {
            double r2 = 0;
            for (var j = 0; j < d; j++) { var t = p[i * d + j] - p2[n * d + j]; r2 += t * t; }
            maxR2 = Math.Max(maxR2, r2);
        }
        var delta = Math.Pow(2, Math.Ceiling(Math.Log2(8 * maxR2 + double.Epsilon)));
        if (!double.IsFinite(delta) || delta == 0) throw new ArgumentException("Delaunay points span too large or too small a range.");
        var excess = new double[n + 1];
        excess[n] = delta;

        // Floating coordinates for planes and ranking: centred on the Qz point and scaled; the
        // lifted coordinate is |x − top|² + e, a shear of |x|², which keeps lower facets lower.
        var D = d + 1;
        var scale = span > 0 ? 1 / span : 1;
        var fc = new double[(n + 1) * D];
        for (var i = 0; i <= n; i++)
        {
            double r2 = 0;
            for (var j = 0; j < d; j++)
            {
                var t = (p2[i * d + j] - p2[n * d + j]) * scale;
                fc[i * D + j] = t; r2 += t * t;
            }
            fc[i * D + d] = r2 + excess[i] * scale * scale;
        }
        var cand = new List<int>(unique) { n };
        var qh = new QuickhullNd(p2, n + 1, d, lifted: true, excess, fc);
        qh.Run(cand);

        // Lower facets: outward normal points down. With E the projected edge matrix
        // [v_i − v₀], σ·(−1)^d·sign(det E) > 0 exactly when that holds (see QuickhullNd remarks
        // for σ); det E = 0 is a vertical facet over a flat piece of the input's hull.
        var F = qh.FacetCount;
        var map = new int[F];
        var flip = new bool[F];
        var count = 0;
        Span<int> rest = stackalloc int[d];
        for (var f = 0; f < F; f++)
        {
            map[f] = -1;
            var hasTop = false;
            for (var k = 0; k < m; k++) if (qh.Vertices[f * m + k] == n) { hasTop = true; break; }
            if (hasTop) continue;
            for (var k = 1; k < m; k++) rest[k - 1] = qh.Vertices[f * m + k];
            var v0 = qh.Vertices[f * m];
            var sE = Math.Sign(SpatialPredicates.OrientNd(p2, d, false, null, rest, p2.AsSpan(v0 * d, d), 0));
            if (sE == 0) continue;
            var sign = qh.Sigma[f] * sE * ((d & 1) == 0 ? 1 : -1);
            if (sign <= 0) continue;
            map[f] = count++;
            flip[f] = sE < 0;
        }
        if (count == 0) throw new ArgumentException("Initial simplex is flat: the points are not full-dimensional (Qhull QH6154).");
        var tri = new int[count * m];
        var nbr = new int[count * m];
        for (var f = 0; f < F; f++)
        {
            var s = map[f];
            if (s < 0) continue;
            for (var k = 0; k < m; k++)
            {
                tri[s * m + k] = qh.Vertices[f * m + k];
                var g = qh.Neighbors[f * m + k];
                nbr[s * m + k] = map[g];
            }
            if (flip[f])
            {
                (tri[s * m], tri[s * m + 1]) = (tri[s * m + 1], tri[s * m]);
                (nbr[s * m], nbr[s * m + 1]) = (nbr[s * m + 1], nbr[s * m]);
            }
        }
        var used = new bool[n];
        foreach (var v in tri) used[v] = true;
        foreach (var i in unique)
            if (!used[i]) throw new InvalidOperationException($"Delaunay: point {i} is not a vertex of the triangulation (internal error).");
        return (tri, nbr, duplicates);
    }

    /// <summary>One point's coordinates as a dictionary key (exact equality, −0 = +0).</summary>
    private readonly struct RowKey(double[] p, int i, int d) : IEquatable<RowKey>
    {
        public bool Equals(RowKey o)
        {
            for (var j = 0; j < d; j++) if (p[i * d + j] != o.Row(j)) return false;
            return true;
        }

        private double Row(int j) => p[i * d + j];

        public override bool Equals(object? obj) => obj is RowKey r && Equals(r);

        public override int GetHashCode()
        {
            var h = new HashCode();
            for (var j = 0; j < d; j++) h.Add(p[i * d + j] + 0.0);
            return h.ToHashCode();
        }
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
