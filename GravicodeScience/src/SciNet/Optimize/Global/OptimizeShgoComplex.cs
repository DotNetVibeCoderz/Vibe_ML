namespace Gravicode.Science.SciNet;

/// <summary>A vertex of SHGO's simplicial complex: a point, its objective value and its neighbours.</summary>
internal sealed class ShgoVertex
{
    public readonly double[] X;
    public readonly int Index;
    public readonly HashSet<ShgoVertex> Nn = [];
    public double F = double.PositiveInfinity;
    public bool Feasible = true;
    public bool Evaluated;

    public ShgoVertex(double[] x, int index) { X = x; Index = index; }

    public void Connect(ShgoVertex v)
    {
        if (ReferenceEquals(v, this) || Nn.Contains(v)) return;
        Nn.Add(v);
        v.Nn.Add(this);
    }

    public void Disconnect(ShgoVertex v)
    {
        if (!Nn.Remove(v)) return;
        v.Nn.Remove(this);
    }

    /// <summary>
    /// The star-domain test: a vertex is in the minimiser pool when its value is strictly below every
    /// neighbour's (an isolated vertex qualifies, as in SciPy).
    /// </summary>
    public bool Minimiser()
    {
        foreach (var v in Nn) if (!(F < v.F)) return false;
        return true;
    }
}

/// <summary>Value equality on coordinate arrays (Python tuple semantics: 0.0 == −0.0).</summary>
internal sealed class ShgoKeyComparer : IEqualityComparer<double[]>
{
    public static readonly ShgoKeyComparer Instance = new();

    public bool Equals(double[]? a, double[]? b)
    {
        if (ReferenceEquals(a, b)) return true;
        if (a is null || b is null || a.Length != b.Length) return false;
        for (var i = 0; i < a.Length; i++) if (!(a[i] == b[i])) return false;
        return true;
    }

    public int GetHashCode(double[] a)
    {
        var h = new HashCode();
        foreach (var v in a) h.Add(v == 0 ? 0.0 : v);
        return h.ToHashCode();
    }
}

/// <summary>Vertex cache in insertion order (SciPy's <c>VertexCacheField</c>, an OrderedDict).</summary>
internal sealed class ShgoVertexCache
{
    private readonly Dictionary<double[], ShgoVertex> _map = new(ShgoKeyComparer.Instance);
    public readonly List<ShgoVertex> Ordered = [];
    public readonly List<ShgoVertex> Pending = [];

    public int Count => Ordered.Count;

    public bool Contains(double[] x) => _map.ContainsKey(x);

    public ShgoVertex this[double[] x]
    {
        get
        {
            if (_map.TryGetValue(x, out var v)) return v;
            var key = (double[])x.Clone();
            v = new ShgoVertex(key, Ordered.Count);
            _map[key] = v;
            Ordered.Add(v);
            Pending.Add(v);
            return v;
        }
    }
}

/// <summary>
/// Port of SciPy's <c>_shgo_lib._complex.Complex</c> hypercube triangulation: the initial
/// triangulation of the box (<c>cyclic_product</c>: every corner plus the centroid, connected into a
/// triangulation of the cube) and its uniform refinement sub-cube by sub-cube
/// (<c>refine_local_space</c>). SciPy writes both as generators that yield once per new vertex so a
/// refinement can stop after exactly n new points; C# iterators reproduce that, including the
/// partially-refined state carried between calls.
/// </summary>
/// <remarks>
/// SciPy iterates a Python <c>set</c> of neighbours when queuing sub-cubes; its order depends on
/// CPython's hash table. Here that set is visited in vertex-creation order. Complete refinements are
/// unaffected; a partial refinement (n not a whole number of sub-cubes) may refine a different
/// sub-cube first than SciPy does.
/// </remarks>
internal sealed class ShgoComplex
{
    private readonly int _dim;
    private readonly double[] _origin, _supremum;
    public readonly ShgoVertexCache V = new();
    private readonly Dictionary<(ShgoVertex, ShgoVertex), ShgoVertex> _splitMemo = [];
    private IEnumerator<bool>? _cp;
    private IEnumerator<bool>? _rls;
    public List<(double[] O, double[] S)>? TriangulatedVectors;

    public ShgoComplex(int dim, double[] lb, double[] ub)
    {
        _dim = dim; _origin = (double[])lb.Clone(); _supremum = (double[])ub.Clone();
    }

    /// <summary>Midpoint vertex of an edge (memoised on the ordered pair, as SciPy's <c>functools.cache</c>).</summary>
    public ShgoVertex SplitEdge(double[] x1, double[] x2)
    {
        var v1 = V[x1]; var v2 = V[x2];
        if (_splitMemo.TryGetValue((v1, v2), out var memo)) return memo;
        v1.Disconnect(v2);
        var c = new double[_dim];
        for (var i = 0; i < _dim; i++) c[i] = (v2.X[i] - v1.X[i]) / 2.0 + v1.X[i];
        var vc = V[c];
        vc.Connect(v1);
        vc.Connect(v2);
        _splitMemo[(v1, v2)] = vc;
        return vc;
    }

    private static double[] With(double[] x, int i, double value)
    {
        var r = (double[])x.Clone();
        r[i] = value;
        return r;
    }

    private IEnumerable<bool> CyclicProduct()
    {
        var vot = _origin; var vut = _supremum;
        var vo = V[vot];
        yield return true;
        V[vut].Connect(V[vot]);
        yield return true;
        var c0x = new List<List<ShgoVertex>> { new() { V[vot] } };
        var aVo = V[With(_origin, 0, vut[0])];
        V[vot].Connect(aVo);
        yield return true;
        var c1x = new List<List<ShgoVertex>> { new() { aVo } };
        var abC = new List<(ShgoVertex, ShgoVertex)>();
        for (var i = 0; i < _dim - 1; i++)
        {
            c0x.Add([]); c1x.Add([]);
            var cC0x = c0x.Take(i + 1).Select(l => l.ToList()).ToList();
            var cC1x = c1x.Take(i + 1).Select(l => l.ToList()).ToList();
            for (var j = 0; j < cC0x.Count; j++)
            {
                var vlList = cC0x[j]; var vuList = cC1x[j];
                for (var k = 0; k < Math.Min(vlList.Count, vuList.Count); k++)
                {
                    var vl = vlList[k]; var vu = vuList[k];
                    var aVl = V[With(vl.X, i + 1, vut[i + 1])];
                    vl.Connect(aVl);
                    yield return true;
                    var aVu = V[With(vu.X, i + 1, vut[i + 1])];
                    vu.Connect(aVu);
                    aVl.Connect(aVu);
                    vl.Connect(aVu);
                    abC.Add((vl, aVu));
                    c0x[i + 1].Add(vl); c0x[i + 1].Add(vu);
                    c1x[i + 1].Add(aVl); c1x[i + 1].Add(aVu);
                    c0x[j].Add(aVl);
                    c1x[j].Add(aVu);
                    yield return true;
                }
            }
            var abCc = abC.ToList();
            foreach (var (p0, p1) in abCc)
            {
                var bV = V[With(p0.X, i + 1, vut[i + 1])];
                var abV = V[With(p1.X, i + 1, vut[i + 1])];
                p0.Connect(abV);
                bV.Connect(abV);
                abC.Add((p0, abV));
                abC.Add((bV, abV));
            }
        }
        // Centroid: replace the main diagonal by a vertex joined to everything the origin touches.
        vo = V[vot];
        var vs = V[vut];
        vo.Disconnect(vs);
        var vc = SplitEdge(vot, vut);
        foreach (var v in vo.Nn.ToList()) v.Connect(vc);
        yield return true;
    }

    /// <summary>Consumes the initial triangulation until the cache holds n vertices (all of it when n is null).</summary>
    public void Triangulate(int? n)
    {
        if (n is null)
        {
            foreach (var _ in CyclicProduct()) { }
            (TriangulatedVectors ??= []).Add((_origin, _supremum));
            return;
        }
        _cp ??= CyclicProduct().GetEnumerator();
        while (V.Count < n)
        {
            if (!_cp.MoveNext())
            {
                (TriangulatedVectors ??= []).Add((_origin, _supremum));
                break;
            }
        }
    }

    /// <summary>Adds n vertices by continuing the current refinement (SciPy's <c>Complex.refine</c>).</summary>
    public void Refine(int n)
    {
        var nt = V.Count + n;
        while (V.Count < nt)
        {
            if (TriangulatedVectors is null) { Triangulate(nt); continue; }
            if (_rls is not null && _rls.MoveNext()) continue;
            var vp = TriangulatedVectors[0];
            _rls = RefineLocalSpace(vp.O, vp.S).GetEnumerator();
            _rls.MoveNext();
        }
    }

    /// <summary>Refines every queued sub-cube completely (SciPy's <c>refine_all</c>).</summary>
    public void RefineAll()
    {
        if (TriangulatedVectors is null) { Triangulate(null); return; }
        foreach (var vp in TriangulatedVectors.ToList())
            foreach (var _ in RefineLocalSpace(vp.O, vp.S)) { }
    }

    private HashSet<ShgoVertex> VPool(double[] origin, double[] supremum)
    {
        var vo = V[origin]; var vs = V[supremum];
        var bl = (double[])origin.Clone(); var bu = (double[])supremum.Clone();
        for (var i = 0; i < _dim; i++)
        {
            if (bl[i] > supremum[i]) bl[i] = supremum[i];
            if (bu[i] < origin[i]) bu[i] = origin[i];
        }
        var pool = new HashSet<ShgoVertex>(vo.Nn);
        pool.UnionWith(vs.Nn);
        pool.RemoveWhere(vn =>
        {
            for (var i = 0; i < _dim; i++) if (!(bl[i] <= vn.X[i] && vn.X[i] <= bu[i])) return true;
            return false;
        });
        return pool;
    }

    private IEnumerable<bool> RefineLocalSpace(double[] origin, double[] supremum)
    {
        var originC = origin; var supremumC = supremum;
        var sOrigin = (double[])origin.Clone();
        var sSupremum = (double[])supremum.Clone();
        for (var i = 0; i < _dim; i++)
            if (origin[i] > supremum[i]) { sOrigin[i] = supremum[i]; sSupremum[i] = origin[i]; }
        var vot = sOrigin; var vut = sSupremum;
        // Function-scope variables: SciPy reads some of them after the loop that set them ended.
        ShgoVertex? vl = null, vu = null, aVu = null;
        var vo = V[vot];
        var vs = V[vut];
        var vco = SplitEdge(vo.X, vs.X);
        var supSet = vco.Nn.OrderBy(v => v.Index).ToList();
        var aVlKey = With(vot, 0, vut[0]);
        ShgoVertex aVl = V[aVlKey];
        var cV = SplitEdge(vo.X, aVl.X);
        cV.Connect(vco);
        yield return true;
        var cox = new List<List<ShgoVertex>> { new() { vo } };
        var ccx = new List<List<ShgoVertex>> { new() { cV } };
        var cux = new List<List<ShgoVertex>> { new() { aVl } };
        var abC = new List<ShgoVertex[]>();
        var sAbC = new List<ShgoVertex[]>();
        for (var i = 0; i < _dim - 1; i++)
        {
            cox.Add([]); ccx.Add([]); cux.Add([]);
            var tAVl = With(vot, i + 1, vut[i + 1]);
            var cCox = cox.Take(i + 1).Select(l => l.ToList()).ToList();
            var cCcx = ccx.Take(i + 1).Select(l => l.ToList()).ToList();
            var cCux = cux.Take(i + 1).Select(l => l.ToList()).ToList();
            var abCc = abC.ToList();
            var sAbCc = sAbC.ToList();
            var y = vut[i + 1];
            if (V.Contains(tAVl))
            {
                foreach (var vec in sAbCc)
                {
                    var bcVc = V[With(vec[0].X, i + 1, y)];
                    bcVc.Connect(vco);
                    yield return true;
                    var dBcVc = SplitEdge(vec[0].X, bcVc.X);
                    dBcVc.Connect(bcVc);
                    dBcVc.Connect(vec[1]); dBcVc.Connect(vec[2]); dBcVc.Connect(vec[3]);
                    yield return true;
                    var bVl = V[With(vec[1].X, i + 1, y)];
                    bcVc.Connect(bVl); dBcVc.Connect(bVl);
                    yield return true;
                    var bVu = V[With(vec[2].X, i + 1, y)];
                    bcVc.Connect(bVu); dBcVc.Connect(bVu);
                    var bVlC = SplitEdge(bVu.X, bVl.X);
                    bcVc.Connect(bVlC);
                    yield return true;
                    var baVu = V[With(vec[3].X, i + 1, y)];
                    bcVc.Connect(baVu); dBcVc.Connect(baVu);
                    SplitEdge(vec[1].X, baVu.X);
                    SplitEdge(bVl.X, baVu.X);
                    var bVuC = SplitEdge(bVu.X, baVu.X);
                    bcVc.Connect(bVuC);
                    yield return true;
                    yield return true;
                    yield return true;
                    dBcVc = SplitEdge(vec[0].X, bcVc.X);
                    dBcVc.Connect(vco);
                    yield return true;
                    var dBVl = SplitEdge(vec[1].X, bVl.X);
                    dBcVc.Connect(vco); dBcVc.Connect(dBVl);
                    yield return true;
                    var dBVu = SplitEdge(vec[2].X, bVu.X);
                    dBcVc.Connect(vco); dBcVc.Connect(dBVu);
                    yield return true;
                    var dBaVu = SplitEdge(vec[3].X, baVu.X);
                    dBcVc.Connect(vco); dBcVc.Connect(dBaVu);
                    yield return true;
                    if (vl is null || vu is null || aVu is null)
                        throw new InvalidOperationException("SHGO refinement reached an unreachable state.");
                    SplitAllPairs([vl, vu, aVu, bVl, bVu, baVu]);
                    abC.Add([dBcVc, vec[1], bVl, aVu, baVu]);
                    abC.Add([dBcVc, vl, bVl, aVu, baVu]);
                }
                foreach (var vec in abCc)
                {
                    var bcVc = V[With(vec[0].X, i + 1, y)];
                    bcVc.Connect(vco);
                    yield return true;
                    var dBcVc = SplitEdge(vec[0].X, bcVc.X);
                    dBcVc.Connect(bcVc);
                    dBcVc.Connect(vec[1]); dBcVc.Connect(vec[2]); dBcVc.Connect(vec[3]); dBcVc.Connect(vec[4]);
                    yield return true;
                    var bVl = V[With(vec[1].X, i + 1, y)];
                    bcVc.Connect(bVl); dBcVc.Connect(bVl);
                    yield return true;
                    var bVu = V[With(vec[2].X, i + 1, y)];
                    bcVc.Connect(bVu); dBcVc.Connect(bVu);
                    yield return true;
                    var baVl = V[With(vec[3].X, i + 1, y)];
                    bcVc.Connect(baVl); dBcVc.Connect(baVl);
                    SplitEdge(bVu.X, baVl.X);
                    yield return true;
                    var baVu = V[With(vec[4].X, i + 1, y)];
                    bcVc.Connect(baVu); dBcVc.Connect(baVu);
                    SplitEdge(vec[1].X, baVu.X);
                    SplitEdge(bVl.X, baVu.X);
                    yield return true;
                    yield return true;
                    yield return true;
                    dBcVc = SplitEdge(vec[0].X, bcVc.X);
                    dBcVc.Connect(vco);
                    yield return true;
                    var dBVl = SplitEdge(vec[1].X, bVl.X);
                    dBcVc.Connect(vco); dBcVc.Connect(dBVl);
                    yield return true;
                    var dBVu = SplitEdge(vec[2].X, bVu.X);
                    dBcVc.Connect(vco); dBcVc.Connect(dBVu);
                    yield return true;
                    var dBaVl = SplitEdge(vec[3].X, baVl.X);
                    dBcVc.Connect(vco); dBcVc.Connect(dBaVl);
                    yield return true;
                    var dBaVu = SplitEdge(vec[4].X, baVu.X);
                    dBcVc.Connect(vco); dBcVc.Connect(dBaVu);
                    yield return true;
                    vl = vec[1]; vu = vec[2]; aVl = vec[3]; aVu = vec[4];
                    SplitAllPairs([vl, vu, aVl, aVu, bVl, bVu, baVl, baVu]);
                    abC.Add([bcVc, bVl, bVu, baVl, baVu]);
                    abC.Add([dBcVc, dBVl, dBVu, dBaVl, dBaVu]);
                    abC.Add([dBcVc, vec[1], bVl, aVu, baVu]);
                    abC.Add([dBcVc, vu, bVu, aVl, baVl]);
                }
                for (var j = 0; j < cCox.Count; j++)
                {
                    var lVL = cCox[j]; var lVC = cCcx[j]; var lVU = cCux[j];
                    var cnt = Math.Min(lVL.Count, Math.Min(lVC.Count, lVU.Count));
                    for (var k = 0; k < cnt; k++)
                    {
                        vl = lVL[k]; var vc = lVC[k]; vu = lVU[k];
                        aVl = V[With(vl.X, i + 1, y)];
                        aVu = V[With(vu.X, i + 1, y)];
                        var cVc = SplitEdge(vl.X, aVu.X);
                        SplitEdge(vl.X, vu.X);
                        cVc.Connect(vco); cVc.Connect(vc); cVc.Connect(vl); cVc.Connect(vu); cVc.Connect(aVl); cVc.Connect(aVu);
                        yield return true;
                        var cVl = SplitEdge(vl.X, aVl.X);
                        cVl.Connect(vco); cVc.Connect(cVl);
                        yield return true;
                        var cVu = SplitEdge(vu.X, aVu.X);
                        cVu.Connect(vco); cVc.Connect(cVu);
                        yield return true;
                        var aVc = SplitEdge(aVl.X, aVu.X);
                        aVc.Connect(vco); aVc.Connect(cVc);
                        abC.Add([cVc, vl, vu, aVl, aVu]);
                        cox[i + 1].Add(vl); cox[i + 1].Add(vc); cox[i + 1].Add(vu);
                        ccx[i + 1].Add(cVl); ccx[i + 1].Add(cVc); ccx[i + 1].Add(cVu);
                        cux[i + 1].Add(aVl); cux[i + 1].Add(aVc); cux[i + 1].Add(aVu);
                        cox[j].Add(cVl); cox[j].Add(aVl);
                        ccx[j].Add(cVc); ccx[j].Add(aVc);
                        cux[j].Add(cVu); cux[j].Add(aVu);
                        yield return true;
                    }
                }
            }
            else
            {
                foreach (var vec in abCc)
                {
                    var baVu = V[With(vec[4].X, i + 1, y)];
                    yield return true;
                    var dBcVc = SplitEdge(vec[1].X, baVu.X);
                    yield return true;
                    dBcVc.Connect(vec[1]); dBcVc.Connect(vec[2]); dBcVc.Connect(vec[3]); dBcVc.Connect(vec[4]);
                    yield return true;
                    var baVl = V[With(vec[3].X, i + 1, y)];
                    yield return true;
                    var dBaVl = SplitEdge(vec[3].X, baVl.X);
                    var dBaVu = SplitEdge(vec[4].X, baVu.X);
                    SplitEdge(dBaVl.X, dBaVu.X);
                    yield return true;
                    yield return true;
                    yield return true;
                    vl = vec[1]; vu = vec[2]; aVl = vec[3]; aVu = vec[4];
                    SplitAllPairs([vl, vu, aVl, aVu, baVl, baVu]);
                }
                var lVL = cox[i]; var lVC = ccx[i]; var lVU = cux[i];
                var cnt = Math.Min(lVL.Count, Math.Min(lVC.Count, lVU.Count));
                for (var k = 0; k < cnt; k++)
                {
                    vl = lVL[k]; var vc = lVC[k]; vu = lVU[k];
                    aVu = V[With(vu.X, i + 1, y)];
                    yield return true;
                    var cVc = SplitEdge(vl.X, aVu.X);
                    SplitEdge(vl.X, vu.X);
                    cVc.Connect(vco); cVc.Connect(vc); cVc.Connect(vl); cVc.Connect(vu); cVc.Connect(aVu);
                    yield return true;
                    var cVu = SplitEdge(vu.X, aVu.X);
                    cVu.Connect(vco); cVc.Connect(cVu);
                    yield return true;
                    cox[i + 1].Add(vu);
                    ccx[i + 1].Add(cVu);
                    cux[i + 1].Add(aVu);
                    sAbC.Add([cVc, vl, vu, aVu]);
                    yield return true;
                }
            }
        }

        var tv = TriangulatedVectors!;
        var idx = tv.FindIndex(p => ShgoKeyComparer.Instance.Equals(p.O, originC) && ShgoKeyComparer.Instance.Equals(p.S, supremumC));
        if (idx >= 0) tv.RemoveAt(idx);
        foreach (var s in supSet) tv.Add((vco.X, s.X));
        var cNnLists = supSet.Select(s => VPool(vco.X, s.X)).ToList();
        for (var k = 0; k < supSet.Count; k++)
        {
            var vcn = SplitEdge(vco.X, supSet[k].X);
            foreach (var vnn in cNnLists[k]) vcn.Connect(vnn);
            yield return true;
        }
        yield return true;
    }

    private void SplitAllPairs(ShgoVertex[] comb)
    {
        for (var a = 0; a < comb.Length; a++)
            for (var b = a + 1; b < comb.Length; b++) SplitEdge(comb[a].X, comb[b].X);
    }
}

/// <summary>
/// Bowyer–Watson Delaunay triangulation in any dimension, for SHGO's Sobol/Halton sampling (SciPy
/// uses Qhull). Points are scaled uniformly (which preserves the Delaunay property) and nudged by a
/// relative 1e-10 deterministic jitter so that the cospherical configurations unscrambled Sobol
/// points produce have a definite answer; a large enclosing simplex is removed at the end.
/// </summary>
/// <remarks>
/// Not a port of Qhull: on degenerate (cospherical) inputs the tie is broken differently, so the
/// neighbour graph — and with it the minimiser pool — can differ from SciPy's in those places.
/// </remarks>
internal static class ShgoDelaunay
{
    private sealed class Simplex(int[] v, double[] c, double r2)
    {
        public readonly int[] V = v;
        public readonly double[] C = c;
        public readonly double R2 = r2;
        public bool Alive = true;
    }

    public static List<int[]> Triangulate(IReadOnlyList<double[]> points, int d)
    {
        var n = points.Count;
        var lo = new double[d]; var hi = new double[d];
        Array.Fill(lo, double.PositiveInfinity); Array.Fill(hi, double.NegativeInfinity);
        foreach (var p in points)
            for (var k = 0; k < d; k++) { lo[k] = Math.Min(lo[k], p[k]); hi[k] = Math.Max(hi[k], p[k]); }
        double span = 0;
        for (var k = 0; k < d; k++) span = Math.Max(span, hi[k] - lo[k]);
        if (span == 0) span = 1;
        var rng = new Gravicode.Science.GraviNum.GraviRandom(20240521);
        var q = new List<double[]>(n + d + 1);
        foreach (var p in points)
        {
            var t = new double[d];
            for (var k = 0; k < d; k++) t[k] = ((p[k] - lo[k]) / span - 0.5) + 1e-10 * (2 * rng.NextDouble() - 1);
            q.Add(t);
        }
        // Super-simplex {x_k ≥ −M, Σx_k ≤ M} contains the unit cube centred at 0 for M ≥ d.
        var m = 50.0 * d;
        var s0 = new double[d]; Array.Fill(s0, -m);
        q.Add(s0);
        for (var k = 0; k < d; k++)
        {
            var s = new double[d]; Array.Fill(s, -m); s[k] = m + (d - 1) * m;
            q.Add(s);
        }
        var simplices = new List<Simplex>();
        var first = Enumerable.Range(n, d + 1).ToArray();
        simplices.Add(Make(first, q, d));

        var facetCount = new Dictionary<string, (int[] Facet, int Count)>();
        for (var pi = 0; pi < n; pi++)
        {
            var p = q[pi];
            var bad = new List<Simplex>();
            foreach (var s in simplices)
            {
                if (!s.Alive) continue;
                double dist = 0;
                for (var k = 0; k < d; k++) { var t = p[k] - s.C[k]; dist += t * t; }
                if (dist < s.R2) bad.Add(s);
            }
            facetCount.Clear();
            foreach (var s in bad)
            {
                s.Alive = false;
                for (var skip = 0; skip <= d; skip++)
                {
                    var f = new int[d];
                    var w = 0;
                    for (var k = 0; k <= d; k++) if (k != skip) f[w++] = s.V[k];
                    Array.Sort(f);
                    var key = string.Join(',', f);
                    facetCount[key] = facetCount.TryGetValue(key, out var e) ? (e.Facet, e.Count + 1) : (f, 1);
                }
            }
            foreach (var (facet, count) in facetCount.Values)
            {
                if (count != 1) continue;
                simplices.Add(Make([.. facet, pi], q, d));
            }
            if (simplices.Count > 4 * (bad.Count + 16) && simplices.Count(s => !s.Alive) > simplices.Count / 2)
                simplices.RemoveAll(s => !s.Alive);
        }
        var result = new List<int[]>();
        foreach (var s in simplices)
            if (s.Alive && s.V.All(v => v < n)) result.Add(s.V);
        return result;
    }

    private static Simplex Make(int[] v, List<double[]> q, int d)
    {
        // Circumcentre: 2 (p_i − p_0) · c = |p_i|² − |p_0|².
        var a = new double[d * d];
        var b = new double[d];
        var p0 = q[v[0]];
        double n0 = 0;
        for (var k = 0; k < d; k++) n0 += p0[k] * p0[k];
        for (var i = 0; i < d; i++)
        {
            var pi = q[v[i + 1]];
            double ni = 0;
            for (var k = 0; k < d; k++) { a[i * d + k] = 2 * (pi[k] - p0[k]); ni += pi[k] * pi[k]; }
            b[i] = ni - n0;
        }
        var c = OptimizeDense.Solve(a, d, b);
        if (c is null || c.Any(t => !double.IsFinite(t)))
            return new Simplex(v, new double[d], double.PositiveInfinity); // flat: removed by the next insertion
        double r2 = 0;
        for (var k = 0; k < d; k++) { var t = c[k] - p0[k]; r2 += t * t; }
        return new Simplex(v, c, r2);
    }
}
