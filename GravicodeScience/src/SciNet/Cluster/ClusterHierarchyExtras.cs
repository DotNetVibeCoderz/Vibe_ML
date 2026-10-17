using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

public static partial class Cluster
{
    /// <summary>
    /// The root node of every flat cluster (scipy's <c>leaders</c>): <c>L[j]</c> is the id of the
    /// largest linkage node whose leaves all carry label <c>M[j]</c>.
    /// </summary>
    /// <remarks>Nodes are visited in scipy's order — post-order, left subtree first, a leader
    /// recorded when its parent turns out to be mixed — so <c>L</c> and <c>M</c> come out in the
    /// same order as scipy's. A flat assignment that is not a union of whole subtrees (one label
    /// on two separate branches) throws, where scipy raises.</remarks>
    /// <param name="z">Linkage matrix.</param>
    /// <param name="t">Flat cluster label of every observation (e.g. from <see cref="Fcluster"/>).</param>
    public static (int[] L, int[] M) Leaders(NdArray z, int[] t)
    {
        ArgumentNullException.ThrowIfNull(t);
        var a = ZBuffer(z, out var n);
        if (t.Length != n) throw new ArgumentException("Mismatch: len(T)!=Z.shape[0] + 1.");
        var fid = new int[n - 1];
        var visited = new bool[2 * n - 1];
        var seen = new HashSet<int>();
        var l = new List<int>(); var m = new List<int>();
        var stack = new int[n];
        var k = 0;
        stack[0] = 2 * n - 2;
        void Emit(int node, int label)
        {
            if (!seen.Add(label)) throw new ArgumentException($"T is not a valid assignment vector. Error found when examining linkage node {node} (< 2n-1).");
            l.Add(node); m.Add(label);
        }
        while (k >= 0)
        {
            var root = stack[k] - n;
            int lc = (int)a[4 * root], rc = (int)a[4 * root + 1];
            if (lc >= n && !visited[lc]) { visited[lc] = true; stack[++k] = lc; continue; }
            if (rc >= n && !visited[rc]) { visited[rc] = true; stack[++k] = rc; continue; }
            var fl = lc >= n ? fid[lc - n] : t[lc];
            var fr = rc >= n ? fid[rc - n] : t[rc];
            if (fl == fr && fl != -1) fid[root] = fl;
            else
            {
                fid[root] = -1;
                if (fl != -1) Emit(lc, fl);
                if (fr != -1) Emit(rc, fr);
            }
            k--;
        }
        if (fid[n - 2] != -1) Emit(2 * n - 2, fid[n - 2]);
        return ([.. l], [.. m]);
    }

    /// <summary>
    /// Whether two flat clusterings are the same partition up to relabelling (scipy's
    /// <c>is_isomorphic</c>): there is a bijection between the labels of <paramref name="t1"/>
    /// and <paramref name="t2"/>.
    /// </summary>
    public static bool IsIsomorphic(int[] t1, int[] t2)
    {
        ArgumentNullException.ThrowIfNull(t1);
        ArgumentNullException.ThrowIfNull(t2);
        if (t1.Length != t2.Length) throw new ArgumentException("T1 and T2 must have the same number of elements.");
        var forward = new Dictionary<int, int>();
        var backward = new Dictionary<int, int>();
        for (var i = 0; i < t1.Length; i++)
        {
            if (forward.TryGetValue(t1[i], out var b)) { if (b != t2[i]) return false; }
            else forward[t1[i]] = t2[i];
            if (backward.TryGetValue(t2[i], out var f)) { if (f != t1[i]) return false; }
            else backward[t2[i]] = t1[i];
        }
        return true;
    }

    /// <summary>
    /// Whether <paramref name="r"/> is a valid inconsistency matrix (scipy's <c>is_valid_im</c>):
    /// shape (m, 4), non-negative standard deviations and link counts.
    /// </summary>
    public static bool IsValidIm(NdArray r, bool throwOnError = false)
    {
        string? error = null;
        if (r is null || r.Rank != 2) error = "Inconsistency matrix must have shape=2 (i.e. be two-dimensional).";
        else if (r.Shape[1] != 4) error = "Inconsistency matrix must have 4 columns.";
        else if (r.Shape[0] < 1) error = "Inconsistency matrix must have at least one row.";
        else
        {
            var a = r.AsContiguous().ToArray();
            for (var i = 0; i < r.Shape[0] && error is null; i++)
            {
                if (a[4 * i] < 0) error = "Inconsistency matrix contains negative link height means.";
                else if (a[4 * i + 1] < 0) error = "Inconsistency matrix contains negative link height standard deviations.";
                else if (a[4 * i + 2] < 0) error = "Inconsistency matrix contains negative link counts.";
            }
        }
        if (error is not null && throwOnError) throw new ArgumentException(error, nameof(r));
        return error is null;
    }

    /// <summary>
    /// Converts a scipy linkage matrix to MATLAB's format (scipy's <c>to_mlab_linkage</c>):
    /// the first three columns, with 1-based cluster indices.
    /// </summary>
    public static NdArray ToMlabLinkage(NdArray z)
    {
        var a = ZBuffer(z, out var n);
        var r = new double[(n - 1) * 3];
        for (var i = 0; i < n - 1; i++)
        {
            r[3 * i] = a[4 * i] + 1; r[3 * i + 1] = a[4 * i + 1] + 1; r[3 * i + 2] = a[4 * i + 2];
        }
        return new NdArray(r, n - 1, 3);
    }

    /// <summary>
    /// Converts a MATLAB linkage matrix (m, 3), 1-based, to scipy's format (scipy's
    /// <c>from_mlab_linkage</c>): 0-based indices and the cluster-size column recomputed.
    /// </summary>
    public static NdArray FromMlabLinkage(NdArray z)
    {
        ArgumentNullException.ThrowIfNull(z);
        if (z.Rank != 2 || z.Shape[1] != 3) throw new ArgumentException("The MATLAB linkage matrix must have shape (m, 3).");
        var m = z.Shape[0];
        var a = z.AsContiguous().ToArray();
        if (m == 0) return new NdArray(new double[0], 0, 4);
        var n = m + 1;
        var r = new double[m * 4];
        var count = new double[m];
        for (var i = 0; i < m; i++)
        {
            double c0 = a[3 * i] - 1, c1 = a[3 * i + 1] - 1;
            if (c0 < 0 || c1 < 0 || c0 >= n + i || c1 >= n + i) throw new ArgumentException("The format of the indices is not 1..N.");
            r[4 * i] = c0; r[4 * i + 1] = c1; r[4 * i + 2] = a[3 * i + 2];
            count[i] = (c0 < n ? 1 : count[(int)c0 - n]) + (c1 < n ? 1 : count[(int)c1 - n]);
            r[4 * i + 3] = count[i];
        }
        return new NdArray(r, m, 4);
    }
}

/// <summary>
/// Disjoint-set (union–find) data structure over arbitrary elements
/// (scipy.cluster.hierarchy.DisjointSet), with path halving and union by size.
/// </summary>
/// <remarks>Like scipy's, the root of a merged set is the root of the larger one, and on a tie
/// the root of the set that was added first, so <see cref="this[T]"/> returns the same
/// representative scipy does. Elements are enumerated in insertion order.</remarks>
public sealed class ClusterDisjointSet<T> where T : notnull
{
    private readonly Dictionary<T, int> _index = [];
    private readonly List<T> _elements = [];
    private readonly List<int> _parent = [];
    private readonly List<int> _size = [];

    /// <summary>Creates an empty structure, or one singleton subset per element.</summary>
    public ClusterDisjointSet(IEnumerable<T>? elements = null)
    {
        if (elements is not null) foreach (var e in elements) Add(e);
    }

    /// <summary>Number of subsets.</summary>
    public int NSubsets { get; private set; }

    /// <summary>Number of elements.</summary>
    public int Count => _elements.Count;

    /// <summary>Whether <paramref name="x"/> is an element.</summary>
    public bool Contains(T x) => _index.ContainsKey(x);

    /// <summary>Adds <paramref name="x"/> as a singleton subset (no-op if present).</summary>
    public void Add(T x)
    {
        if (_index.ContainsKey(x)) return;
        _index[x] = _elements.Count;
        _elements.Add(x); _parent.Add(_elements.Count - 1); _size.Add(1);
        NSubsets++;
    }

    private int Find(int i)
    {
        while (_parent[i] != i) { _parent[i] = _parent[_parent[i]]; i = _parent[i]; }
        return i;
    }

    private int IndexOf(T x) => _index.TryGetValue(x, out var i) ? i : throw new KeyNotFoundException($"{x} is not an element.");

    /// <summary>The root element of the subset containing <paramref name="x"/>.</summary>
    public T this[T x] => _elements[Find(IndexOf(x))];

    /// <summary>Merges the subsets of x and y; false when they were already one subset.</summary>
    public bool Merge(T x, T y)
    {
        int a = Find(IndexOf(x)), b = Find(IndexOf(y));
        if (a == b) return false;
        // scipy: the larger set's root wins; on equal sizes, the one inserted earlier.
        if (_size[a] < _size[b] || (_size[a] == _size[b] && b < a)) (a, b) = (b, a);
        _parent[b] = a;
        _size[a] += _size[b];
        NSubsets--;
        return true;
    }

    /// <summary>Whether x and y are in the same subset.</summary>
    public bool Connected(T x, T y) => Find(IndexOf(x)) == Find(IndexOf(y));

    /// <summary>The subset containing <paramref name="x"/>, in insertion order.</summary>
    public HashSet<T> Subset(T x)
    {
        var r = Find(IndexOf(x));
        var s = new HashSet<T>();
        for (var i = 0; i < _elements.Count; i++) if (Find(i) == r) s.Add(_elements[i]);
        return s;
    }

    /// <summary>Size of the subset containing <paramref name="x"/>.</summary>
    public int SubsetSize(T x) => _size[Find(IndexOf(x))];

    /// <summary>All subsets, ordered by their first-inserted element.</summary>
    public List<HashSet<T>> Subsets()
    {
        var byRoot = new Dictionary<int, HashSet<T>>();
        var result = new List<HashSet<T>>();
        for (var i = 0; i < _elements.Count; i++)
        {
            var r = Find(i);
            if (!byRoot.TryGetValue(r, out var s)) { byRoot[r] = s = []; result.Add(s); }
            s.Add(_elements[i]);
        }
        return result;
    }
}
