namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// Quickhull in any dimension D ≥ 2 (Barber, Dobkin and Huhdanpaa 1996 — the algorithm inside
/// Qhull), on plain points or on points lifted onto the paraboloid for Delaunay.
/// </summary>
/// <remarks>
/// <para><b>Exact visibility.</b> Every decision that shapes the hull — is point q strictly
/// beyond facet f — goes through <see cref="SpatialPredicates.OrientNd"/>, whose sign is exact.
/// Floating-point planes are kept only to <em>rank</em> conflict points (choosing the furthest
/// one as the next eye) and to report <c>equations</c>; a rounding error there changes the
/// insertion order, never the combinatorics. That is the alternative to Qhull's "joggle"
/// (<c>QJ</c>) and facet-merging strategies: with exact signs there is no imprecision to merge
/// away.</para>
/// <para><b>Degeneracies.</b> A point exactly on a facet's hyperplane is not beyond it. That
/// keeps the visible region and its horizon well defined, and rules out degenerate new facets:
/// an eye in the affine span of a horizon ridge would lie on the hyperplanes of both facets
/// sharing that ridge, so neither would be strictly visible and the ridge would not be on the
/// horizon. Coplanar neighbouring facets are therefore allowed and expected — a cube side is two
/// coplanar triangles, a lattice cell in a Delaunay triangulation is several cospherical
/// simplices — and the output is always simplicial, like Qhull with <c>Qt</c>.</para>
/// <para><b>Orientation bookkeeping.</b> Facet f stores a sign σ such that q is strictly beyond f
/// iff σ·det[V_i − q] &gt; 0. A new facet copies its visible parent's vertex list with the vertex
/// opposite the horizon ridge replaced by the eye, which preserves σ; the outward normal is then
/// σ·(−1)^D times the cofactor vector of the edge matrix, with no interior reference point.</para>
/// </remarks>
internal sealed class QuickhullNd
{
    private readonly double[] _p;
    private readonly int _d;
    private readonly bool _lifted;
    private readonly double[]? _excess;
    private readonly int _n;
    private readonly double[] _fc; // floating coordinates, n × D, used for planes only

    /// <summary>Hull dimension (d, or d + 1 when lifted).</summary>
    public readonly int D;

    private int[] _v = [];
    private int[] _nb = [];
    private sbyte[] _sigma = [];
    private double[] _plane = []; // D + 1 per facet: outward normal (unnormalised), offset
    private bool[] _alive = [];
    private int[] _mark = [];
    private List<int>?[] _conf = [];
    private int _count;

    /// <summary>Number of alive facets after <see cref="Run"/>.</summary>
    public int FacetCount { get; private set; }

    /// <summary>Alive facets' vertices, D per facet.</summary>
    public int[] Vertices { get; private set; } = [];

    /// <summary>Alive facets' neighbours (index into the alive list), opposite each vertex slot.</summary>
    public int[] Neighbors { get; private set; } = [];

    /// <summary>Alive facets' orientation signs.</summary>
    public sbyte[] Sigma { get; private set; } = [];

    /// <summary>Alive facets' floating planes (outward unnormalised normal, offset), D + 1 per facet.</summary>
    public double[] Planes { get; private set; } = [];

    /// <param name="p">Point buffer, n × d.</param>
    /// <param name="n">Number of points.</param>
    /// <param name="d">Input dimension.</param>
    /// <param name="lifted">Lift onto the paraboloid (hull dimension d + 1).</param>
    /// <param name="excess">Per-point excess height above the paraboloid (lifted only).</param>
    /// <param name="floatCoords">Floating coordinates n × D for planes; may be any translate,
    /// scaling or shear of the true ones (only used for ranking and for normals in the caller's
    /// frame).</param>
    public QuickhullNd(double[] p, int n, int d, bool lifted, double[]? excess, double[] floatCoords)
    {
        _p = p; _n = n; _d = d; _lifted = lifted; _excess = excess; _fc = floatCoords;
        D = lifted ? d + 1 : d;
        if (D > SpatialPredicates.MaxNdDimension) throw new ArgumentException($"Hulls are limited to {SpatialPredicates.MaxNdDimension} dimensions.");
    }

    private double Excess(int q) => _excess is null ? 0 : _excess[q];

    /// <summary>Exact orientation of facet vertex list against point q.</summary>
    private double Orient(ReadOnlySpan<int> verts, int q)
        => SpatialPredicates.OrientNd(_p, _d, _lifted, _excess, verts, _p.AsSpan(q * _d, _d), Excess(q));

    private bool Visible(int f, int q) => _sigma[f] * Math.Sign(Orient(_v.AsSpan(f * D, D), q)) > 0;

    private int NewFacet()
    {
        if (_count == _alive.Length)
        {
            var cap = Math.Max(64, 2 * _count);
            Array.Resize(ref _v, cap * D);
            Array.Resize(ref _nb, cap * D);
            Array.Resize(ref _sigma, cap);
            Array.Resize(ref _plane, cap * (D + 1));
            Array.Resize(ref _alive, cap);
            Array.Resize(ref _mark, cap);
            Array.Resize(ref _conf, cap);
        }
        var f = _count++;
        _alive[f] = true;
        _mark[f] = 0;
        _conf[f] = null;
        return f;
    }

    /// <summary>Cofactor normal of facet f in floating coordinates, oriented outward by σ.</summary>
    private void ComputePlane(int f)
    {
        if (D == 4) { ComputePlane4(f); return; }
        var Dm = D - 1;
        Span<double> e = stackalloc double[Math.Max(1, Dm * D)];
        var v0 = _v[f * D];
        for (var i = 1; i < D; i++)
        {
            var vi = _v[f * D + i];
            for (var j = 0; j < D; j++) e[(i - 1) * D + j] = _fc[vi * D + j] - _fc[v0 * D + j];
        }
        var size = 1 << D;
        Span<double> minor = size <= 1024 ? stackalloc double[size] : new double[size];
        minor[0] = 1;
        for (var mask = 1; mask < size; mask++)
        {
            var k = System.Numerics.BitOperations.PopCount((uint)mask) - 1;
            if (k >= Dm) { minor[mask] = 0; continue; }
            double s = 0; var c = 0;
            for (var bits = mask; bits != 0; bits &= bits - 1, c++)
            {
                var j = System.Numerics.BitOperations.TrailingZeroCount(bits);
                var t = e[k * D + j] * minor[mask & ~(1 << j)];
                s += ((k + c) & 1) == 0 ? t : -t;
            }
            minor[mask] = s;
        }
        // det[E; x] = Σ_j x_j (−1)^(D−1+j) minor(all but j); outward = σ(−1)^D times that vector.
        var o = f * (D + 1);
        double off = 0;
        for (var j = 0; j < D; j++)
        {
            var cof = (((D - 1 + j) & 1) == 0 ? 1.0 : -1.0) * minor[(size - 1) & ~(1 << j)];
            var nj = _sigma[f] * ((D & 1) == 0 ? cof : -cof);
            _plane[o + j] = nj;
            off -= nj * _fc[v0 * D + j];
        }
        _plane[o + D] = off;
    }

    /// <summary><see cref="ComputePlane"/> unrolled for D = 4 (3-D Delaunay, 4-D hulls).</summary>
    private void ComputePlane4(int f)
    {
        var fc = _fc;
        int v0 = _v[f * 4] * 4, v1 = _v[f * 4 + 1] * 4, v2 = _v[f * 4 + 2] * 4, v3 = _v[f * 4 + 3] * 4;
        double a0 = fc[v1] - fc[v0], a1 = fc[v1 + 1] - fc[v0 + 1], a2 = fc[v1 + 2] - fc[v0 + 2], a3 = fc[v1 + 3] - fc[v0 + 3];
        double b0 = fc[v2] - fc[v0], b1 = fc[v2 + 1] - fc[v0 + 1], b2 = fc[v2 + 2] - fc[v0 + 2], b3 = fc[v2 + 3] - fc[v0 + 3];
        double c0 = fc[v3] - fc[v0], c1 = fc[v3 + 1] - fc[v0 + 1], c2 = fc[v3 + 2] - fc[v0 + 2], c3 = fc[v3 + 3] - fc[v0 + 3];
        // 2×2 minors of rows (b, c), then 3×3 minors of (a, b, c) omitting one column each.
        double m01 = b0 * c1 - b1 * c0, m02 = b0 * c2 - b2 * c0, m03 = b0 * c3 - b3 * c0;
        double m12 = b1 * c2 - b2 * c1, m13 = b1 * c3 - b3 * c1, m23 = b2 * c3 - b3 * c2;
        var M0 = a1 * m23 - a2 * m13 + a3 * m12; // columns 1,2,3
        var M1 = a0 * m23 - a2 * m03 + a3 * m02; // columns 0,2,3
        var M2 = a0 * m13 - a1 * m03 + a3 * m01; // columns 0,1,3
        var M3 = a0 * m12 - a1 * m02 + a2 * m01; // columns 0,1,2
        // Cofactors (−1)^(3+j) M_j, then outward = σ(−1)^4 = σ times them.
        double s = _sigma[f];
        double n0 = -s * M0, n1 = s * M1, n2 = -s * M2, n3 = s * M3;
        var o = f * 5;
        _plane[o] = n0; _plane[o + 1] = n1; _plane[o + 2] = n2; _plane[o + 3] = n3;
        _plane[o + 4] = -(n0 * fc[v0] + n1 * fc[v0 + 1] + n2 * fc[v0 + 2] + n3 * fc[v0 + 3]);
    }

    private double Height(int f, int q)
    {
        var o = f * (D + 1);
        var h = _plane[o + D];
        for (var j = 0; j < D; j++) h += _plane[o + j] * _fc[q * D + j];
        return h;
    }

    /// <summary>Builds the hull. Throws <see cref="ArgumentException"/> when the input is flat.</summary>
    public void Run(IReadOnlyList<int> candidates)
    {
        var simplex = InitialSimplex(candidates);
        // Facet j omits simplex vertex j.
        var first = new int[D + 1];
        for (var j = 0; j <= D; j++)
        {
            var f = NewFacet();
            first[j] = f;
            var w = 0;
            for (var k = 0; k <= D; k++) if (k != j) _v[f * D + w++] = simplex[k];
            var o = Orient(_v.AsSpan(f * D, D), simplex[j]);
            if (o == 0) throw new ArgumentException("Initial simplex is flat (Qhull QH6154).");
            _sigma[f] = (sbyte)(o > 0 ? -1 : 1);
        }
        for (var j = 0; j <= D; j++)
        {
            var f = first[j];
            for (var s = 0; s < D; s++)
            {
                var vert = _v[f * D + s];
                var m = Array.IndexOf(simplex, vert);
                _nb[f * D + s] = first[m];
            }
            ComputePlane(f);
        }
        var inSimplex = new HashSet<int>(simplex);
        foreach (var q in candidates)
        {
            if (inSimplex.Contains(q)) continue;
            foreach (var f in first)
                if (Visible(f, q)) { (_conf[f] ??= []).Add(q); break; }
        }

        var pending = new Queue<int>(first);
        var stamp = 0;
        var stack = new Stack<int>();
        var visible = new List<int>();
        var orphans = new List<int>();
        var pool = new Stack<List<int>>();
        // New facets meet each other across ridges through the eye; such a ridge is named by
        // its other D − 2 vertices. Up to two of them pack exactly into a ulong (D ≤ 4, the
        // 3-D Delaunay and 4-D hull cases), which avoids allocating a key per ridge.
        var packed = D <= 4;
        var ridgesPacked = new Dictionary<ulong, int>();
        var ridges = new Dictionary<RidgeKey, int>();
        Span<int> tmp = stackalloc int[D];
        while (pending.Count > 0)
        {
            var fi = pending.Dequeue();
            if (!_alive[fi] || _conf[fi] is not { Count: > 0 } conf) continue;
            var eye = conf[0]; var best = Height(fi, eye);
            for (var t = 1; t < conf.Count; t++) { var h = Height(fi, conf[t]); if (h > best) { best = h; eye = conf[t]; } }

            stamp++;
            visible.Clear();
            stack.Push(fi); _mark[fi] = stamp;
            while (stack.Count > 0)
            {
                var g = stack.Pop();
                visible.Add(g);
                for (var s = 0; s < D; s++)
                {
                    var nb = _nb[g * D + s];
                    if (_mark[nb] == stamp || _mark[nb] == -stamp) continue;
                    if (Visible(nb, eye)) { _mark[nb] = stamp; stack.Push(nb); }
                    else _mark[nb] = -stamp; // known invisible this round
                }
            }
            ridgesPacked.Clear(); ridges.Clear();
            var c0 = _count; // new facets are allocated consecutively: [c0, _count)
            foreach (var g in visible)
                for (var s = 0; s < D; s++)
                {
                    var nb = _nb[g * D + s];
                    if (_mark[nb] == stamp) continue;
                    var f = NewFacet();
                    for (var k = 0; k < D; k++) _v[f * D + k] = _v[g * D + k];
                    _v[f * D + s] = eye;
                    _sigma[f] = _sigma[g];
                    _nb[f * D + s] = nb;
                    for (var k = 0; k < D; k++) if (_nb[nb * D + k] == g) { _nb[nb * D + k] = f; break; }
                    // Ridges through the eye: the facet minus one of its other vertices.
                    for (var k = 0; k < D; k++)
                    {
                        if (k == s) continue;
                        var w = 0;
                        for (var j = 0; j < D; j++) if (j != k && j != s) tmp[w++] = _v[f * D + j];
                        int other;
                        bool found;
                        if (packed)
                        {
                            ulong key = w == 0 ? 0UL : w == 1 ? (ulong)(uint)tmp[0]
                                : (ulong)(uint)Math.Min(tmp[0], tmp[1]) << 32 | (uint)Math.Max(tmp[0], tmp[1]);
                            found = ridgesPacked.Remove(key, out other);
                            if (!found) ridgesPacked[key] = f * D + k;
                        }
                        else
                        {
                            var key = new RidgeKey(tmp[..w]);
                            found = ridges.Remove(key, out other);
                            if (!found) ridges[key] = f * D + k;
                        }
                        if (found)
                        {
                            _nb[f * D + k] = other / D;
                            _nb[other] = f;
                        }
                    }
                }
            if (ridges.Count != 0 || ridgesPacked.Count != 0) throw new InvalidOperationException("Quickhull horizon is not closed (internal error).");
            var c1 = _count;
            orphans.Clear();
            foreach (var g in visible)
            {
                _alive[g] = false;
                if (_conf[g] is { } c)
                {
                    foreach (var q in c) if (q != eye) orphans.Add(q);
                    c.Clear();
                    pool.Push(c);
                }
                _conf[g] = null;
            }
            for (var f = c0; f < c1; f++) ComputePlane(f);
            // Each orphan goes to a new facet it is strictly beyond. The floating heights pick
            // the most likely one (the furthest, as Qhull assigns), so usually a single exact
            // test decides; the rest are scanned only when that guess is refuted.
            var planes = _plane;
            var stride = D + 1;
            foreach (var q in orphans)
            {
                var qc = _fc.AsSpan(q * D, D);
                var bestF = -1; var bestH = double.NegativeInfinity;
                for (var f = c0; f < c1; f++)
                {
                    var pl = planes.AsSpan(f * stride, stride);
                    var h = pl[D];
                    for (var j = 0; j < qc.Length; j++) h += pl[j] * qc[j];
                    if (h > bestH) { bestH = h; bestF = f; }
                }
                var target = -1;
                if (bestF >= 0 && Visible(bestF, q)) target = bestF;
                else
                    for (var f = c0; f < c1; f++)
                        if (f != bestF && Visible(f, q)) { target = f; break; }
                if (target >= 0) (_conf[target] ??= pool.Count > 0 ? pool.Pop() : []).Add(q);
            }
            for (var f = c0; f < c1; f++) pending.Enqueue(f);
        }

        var remap = new int[_count];
        var alive = 0;
        for (var f = 0; f < _count; f++) remap[f] = _alive[f] ? alive++ : -1;
        FacetCount = alive;
        Vertices = new int[alive * D];
        Neighbors = new int[alive * D];
        Sigma = new sbyte[alive];
        Planes = new double[alive * (D + 1)];
        for (var f = 0; f < _count; f++)
        {
            var r = remap[f];
            if (r < 0) continue;
            for (var k = 0; k < D; k++)
            {
                Vertices[r * D + k] = _v[f * D + k];
                Neighbors[r * D + k] = remap[_nb[f * D + k]];
            }
            Sigma[r] = _sigma[f];
            Array.Copy(_plane, f * (D + 1), Planes, r * (D + 1), D + 1);
        }
    }

    /// <summary>
    /// D + 1 points spanning a full-dimensional simplex, chosen greedily (each new point
    /// furthest from the affine span of the previous ones, by Gram–Schmidt in floating point),
    /// then certified non-flat by the exact predicate.
    /// </summary>
    private int[] InitialSimplex(IReadOnlyList<int> cand)
    {
        if (cand.Count < D + 1) throw new ArgumentException($"A {D}-dimensional hull needs at least {D + 1} points (Qhull QH6214).");
        var chosen = new List<int>();
        int lo = cand[0], hi = cand[0];
        foreach (var q in cand)
        {
            if (_fc[q * D] < _fc[lo * D]) lo = q;
            if (_fc[q * D] > _fc[hi * D]) hi = q;
        }
        chosen.Add(lo);
        var basis = new List<double[]>();
        var r = new double[D];
        while (chosen.Count < D + 1)
        {
            var bestQ = -1; var bestN = 0.0;
            foreach (var q in cand)
            {
                for (var j = 0; j < D; j++) r[j] = _fc[q * D + j] - _fc[lo * D + j];
                foreach (var b in basis)
                {
                    double dot = 0;
                    for (var j = 0; j < D; j++) dot += r[j] * b[j];
                    for (var j = 0; j < D; j++) r[j] -= dot * b[j];
                }
                double n2 = 0;
                for (var j = 0; j < D; j++) n2 += r[j] * r[j];
                if (n2 > bestN) { bestN = n2; bestQ = q; }
            }
            if (bestQ < 0 || chosen.Contains(bestQ))
                throw new ArgumentException($"Initial simplex is flat: the points lie in a {chosen.Count - 1}-dimensional subspace (Qhull QH6154).");
            var nb = new double[D];
            for (var j = 0; j < D; j++) nb[j] = _fc[bestQ * D + j] - _fc[lo * D + j];
            foreach (var b in basis)
            {
                double dot = 0;
                for (var j = 0; j < D; j++) dot += nb[j] * b[j];
                for (var j = 0; j < D; j++) nb[j] -= dot * b[j];
            }
            var len = 0.0;
            for (var j = 0; j < D; j++) len += nb[j] * nb[j];
            len = Math.Sqrt(len);
            for (var j = 0; j < D; j++) nb[j] /= len;
            basis.Add(nb);
            chosen.Add(bestQ);
        }
        var s = chosen.ToArray();
        // Exact certification: the floating choice can be flat by a rounding error.
        if (Orient(s.AsSpan(0, D), s[D]) == 0)
        {
            var fixedUp = false;
            foreach (var q in cand)
            {
                if (Array.IndexOf(s, q) >= 0) continue;
                if (Orient(s.AsSpan(0, D), q) != 0) { s[D] = q; fixedUp = true; break; }
            }
            if (!fixedUp) throw new ArgumentException("Initial simplex is flat: the points are not full-dimensional (Qhull QH6154).");
        }
        return s;
    }

    /// <summary>A sorted vertex set identifying a ridge, compared by value.</summary>
    private readonly struct RidgeKey : IEquatable<RidgeKey>
    {
        private readonly int[] _k;
        private readonly int _hash;

        public RidgeKey(ReadOnlySpan<int> verts)
        {
            _k = verts.ToArray();
            Array.Sort(_k);
            var h = new HashCode();
            foreach (var v in _k) h.Add(v);
            _hash = h.ToHashCode();
        }

        public bool Equals(RidgeKey other) => _hash == other._hash && _k.AsSpan().SequenceEqual(other._k);
        public override bool Equals(object? obj) => obj is RidgeKey r && Equals(r);
        public override int GetHashCode() => _hash;
    }
}
