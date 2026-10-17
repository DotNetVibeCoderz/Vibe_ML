using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Spatial;

namespace Gravicode.Science.SciNet;

/// <summary>Agglomeration rule of <see cref="Cluster.Linkage(double[], LinkageMethod, bool)"/>.</summary>
public enum LinkageMethod
{
    /// <summary>Nearest point (minimum spanning tree).</summary>
    Single,
    /// <summary>Farthest point.</summary>
    Complete,
    /// <summary>UPGMA: mean of all pairwise distances.</summary>
    Average,
    /// <summary>WPGMA: mean of the two merged clusters' distances.</summary>
    Weighted,
    /// <summary>UPGMC: distance between centroids (Euclidean only).</summary>
    Centroid,
    /// <summary>WPGMC: distance between merged-midpoint centroids (Euclidean only).</summary>
    Median,
    /// <summary>Ward's minimum-variance criterion (Euclidean only).</summary>
    Ward,
}

/// <summary>Flattening rule of <see cref="Cluster.Fcluster"/>.</summary>
public enum FclusterCriterion
{
    /// <summary>Inconsistency coefficient of every node in the cluster at most t.</summary>
    Inconsistent,
    /// <summary>Cophenetic distance within a cluster at most t.</summary>
    Distance,
    /// <summary>At most t clusters, with the smallest distance threshold achieving it.</summary>
    Maxclust,
    /// <summary>Threshold a caller-supplied monotone statistic at t.</summary>
    Monocrit,
    /// <summary>At most t clusters, thresholding a caller-supplied monotone statistic.</summary>
    MaxclustMonocrit,
}

/// <summary>
/// A node of a hierarchical clustering tree (scipy.cluster.hierarchy.ClusterNode).
/// </summary>
public sealed class ClusterNode : IComparable<ClusterNode>
{
    /// <summary>Node id: 0..n-1 for observations, n + row for the link in row <c>row</c> of Z.</summary>
    public int Id { get; }

    /// <summary>Left child, null for a leaf.</summary>
    public ClusterNode? Left { get; }

    /// <summary>Right child, null for a leaf.</summary>
    public ClusterNode? Right { get; }

    /// <summary>Merge distance (0 for a leaf).</summary>
    public double Dist { get; }

    /// <summary>Number of observations below this node.</summary>
    public int Count { get; }

    /// <summary>True for an original observation.</summary>
    public bool IsLeaf => Left is null;

    /// <summary>Creates a node; a leaf when both children are null.</summary>
    public ClusterNode(int id, ClusterNode? left = null, ClusterNode? right = null, double dist = 0, int count = 1)
    {
        if (id < 0) throw new ArgumentException("The id must be non-negative.");
        if (dist < 0) throw new ArgumentException("The distance must be non-negative.");
        if ((left is null) != (right is null)) throw new ArgumentException("Only full or proper binary trees are permitted. This node has one child.");
        if (count < 1) throw new ArgumentException("A cluster must contain at least one original observation.");
        Id = id; Left = left; Right = right; Dist = dist;
        Count = left is null ? count : left.Count + right!.Count;
    }

    /// <summary>Orders nodes by merge distance, as scipy's <c>ClusterNode.__lt__</c> does.</summary>
    public int CompareTo(ClusterNode? other) => other is null ? 1 : Dist.CompareTo(other.Dist);

    /// <summary>Leaf ids from left to right.</summary>
    public List<int> PreOrder() => PreOrder(n => n.Id);

    /// <summary><paramref name="func"/> applied to each leaf from left to right.</summary>
    public List<T> PreOrder<T>(Func<ClusterNode, T> func)
    {
        var result = new List<T>();
        var stack = new Stack<ClusterNode>();
        stack.Push(this);
        while (stack.Count > 0)
        {
            var nd = stack.Pop();
            if (nd.IsLeaf) { result.Add(func(nd)); continue; }
            stack.Push(nd.Right!);
            stack.Push(nd.Left!);
        }
        return result;
    }
}

/// <summary>
/// Clustering (scipy.cluster): hierarchical agglomeration and flat clusters
/// (<c>hierarchy</c>), and vector quantisation / k-means (<c>vq</c>).
/// </summary>
/// <remarks>
/// <para><b>Linkage matrices are scipy's, bit for bit in structure.</b> Row k of Z (n-1 × 4)
/// merges clusters Z[k,0] &lt; Z[k,1] at distance Z[k,2] into a cluster of Z[k,3] observations
/// with id n + k. The algorithms are the ones scipy runs (Müllner 2011, "Modern hierarchical,
/// agglomerative clustering algorithms"): Prim's minimum spanning tree for single linkage, the
/// nearest-neighbour chain for complete/average/weighted/Ward, and the generic algorithm with a
/// heap of lower bounds for centroid/median (which can produce inversions, so the NN-chain is
/// invalid for them). MST and NN-chain rows are then stably sorted by distance and relabelled
/// with a union-find, exactly as scipy does, so ties come out in scipy's order.</para>
/// <para>Each update reads distances through condensed indices into one <c>double[]</c>; the
/// whole thing is O(n²) memory, like scipy.</para>
/// </remarks>
public static partial class Cluster
{
    // ================================================================== linkage

    private static long Ci(int n, int i, int j)
    {
        if (i > j) (i, j) = (j, i);
        return (long)n * i - (long)i * (i + 1) / 2 + (j - i - 1);
    }

    /// <summary>Condensed index of (i, j) for i &lt; j, without the ordering test.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long CiOrdered(int n, int i, int j) => (long)n * i - (long)i * (i + 1) / 2 + (j - i - 1);

    /// <summary>
    /// A Lance–Williams update as a struct, so each method's loop is compiled with its formula
    /// inlined instead of calling a delegate per element.
    /// </summary>
    /// <remarks>
    /// The expressions are exactly the ones the delegate version used, operand order included:
    /// Linkage is pinned bit-for-bit against scipy's Z, and reassociating a single product would
    /// be enough to flip a tie between two merges.
    /// </remarks>
    private interface ILinkageRule
    {
        double Update(double dxi, double dyi, double dxy, int nx, int ny, int ni);
    }

    private readonly struct CompleteRule : ILinkageRule
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Update(double dxi, double dyi, double dxy, int nx, int ny, int ni) => Math.Max(dxi, dyi);
    }

    private readonly struct AverageRule : ILinkageRule
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Update(double dxi, double dyi, double dxy, int nx, int ny, int ni) => (nx * dxi + ny * dyi) / (nx + ny);
    }

    private readonly struct WeightedRule : ILinkageRule
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Update(double dxi, double dyi, double dxy, int nx, int ny, int ni) => 0.5 * (dxi + dyi);
    }

    private readonly struct CentroidRule : ILinkageRule
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Update(double dxi, double dyi, double dxy, int nx, int ny, int ni)
            => Math.Sqrt((((nx * dxi * dxi) + (ny * dyi * dyi)) - (nx * ny * dxy * dxy) / (nx + ny)) / (nx + ny));
    }

    private readonly struct MedianRule : ILinkageRule
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Update(double dxi, double dyi, double dxy, int nx, int ny, int ni)
            => Math.Sqrt(0.5 * (dxi * dxi + dyi * dyi) - 0.25 * dxy * dxy);
    }

    private readonly struct WardRule : ILinkageRule
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Update(double dxi, double dyi, double dxy, int nx, int ny, int ni)
        {
            var t = 1.0 / (nx + ny + ni);
            return Math.Sqrt((ni + nx) * t * dxi * dxi + (ni + ny) * t * dyi * dyi - ni * t * dxy * dxy);
        }
    }

    /// <summary>
    /// Hierarchical clustering of a condensed distance matrix (length n(n-1)/2, as from
    /// <see cref="Distance.Pdist(NdArray, DistanceMetric, double, double[], double[], NdArray, double?, int)"/>).
    /// </summary>
    /// <param name="y">Condensed distances.</param>
    /// <param name="method">Agglomeration rule.</param>
    /// <param name="optimalOrdering">Reorder children to minimise the distance between adjacent
    /// leaves (<see cref="OptimalLeafOrdering"/>); slower, O(n³) in the worst case.</param>
    /// <returns>The (n-1) × 4 linkage matrix Z.</returns>
    public static NdArray Linkage(double[] y, LinkageMethod method = LinkageMethod.Single, bool optimalOrdering = false)
    {
        ArgumentNullException.ThrowIfNull(y);
        return LinkageCore(y, method, optimalOrdering, owned: false);
    }

    /// <param name="y">Condensed distances.</param>
    /// <param name="method">Agglomeration rule.</param>
    /// <param name="optimalOrdering">Reorder the leaves optimally.</param>
    /// <param name="owned">The buffer is private to this call (from pdist) and may be overwritten,
    /// which saves copying n(n−1)/2 doubles — 16 MB at n = 2000.</param>
    private static NdArray LinkageCore(double[] y, LinkageMethod method, bool optimalOrdering, bool owned)
    {
        var n = Distance.NumObsY(y.LongLength);
        if (!AllFinite(y)) throw new ArgumentException("The condensed distance matrix must contain only finite values.");
        if (n < 2) throw new ArgumentException("Linkage needs at least two observations.");
        var work = owned && !optimalOrdering ? y : null;
        var z = method switch
        {
            LinkageMethod.Single => MstSingleLinkage(y, n),
            LinkageMethod.Complete => NnChain<CompleteRule>(work ?? (double[])y.Clone(), n),
            LinkageMethod.Average => NnChain<AverageRule>(work ?? (double[])y.Clone(), n),
            LinkageMethod.Weighted => NnChain<WeightedRule>(work ?? (double[])y.Clone(), n),
            LinkageMethod.Ward => NnChain<WardRule>(work ?? (double[])y.Clone(), n),
            LinkageMethod.Centroid => FastLinkage<CentroidRule>(work ?? (double[])y.Clone(), n),
            LinkageMethod.Median => FastLinkage<MedianRule>(work ?? (double[])y.Clone(), n),
            _ => throw new ArgumentException($"Unknown linkage method {method}."),
        };
        var zz = new NdArray(z, n - 1, 4);
        return optimalOrdering ? OptimalLeafOrdering(zz, y) : zz;
    }

    private static bool AllFinite(double[] y)
    {
        // x - x is 0 for finite x and NaN for ±inf or NaN, so one vector compare per lane suffices.
        var i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var bad = Vector256<double>.Zero;
            ref var start = ref MemoryMarshal.GetArrayDataReference(y);
            for (; i <= y.Length - 4; i += 4)
            {
                var v = Vector256.LoadUnsafe(ref start, (nuint)i);
                bad |= ~Vector256.Equals(v - v, Vector256<double>.Zero);
            }
            if (bad != Vector256<double>.Zero) return false;
        }
        for (; i < y.Length; i++) if (!double.IsFinite(y[i])) return false;
        return true;
    }

    /// <summary>
    /// Hierarchical clustering of observations (rows of <paramref name="x"/>), using
    /// <paramref name="metric"/> for the pairwise distances. A rank-1 array is taken as a
    /// condensed distance matrix, as in scipy.
    /// </summary>
    public static NdArray Linkage(NdArray x, LinkageMethod method = LinkageMethod.Single,
        DistanceMetric metric = DistanceMetric.Euclidean, bool optimalOrdering = false)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Rank == 1) return LinkageCore(x.ToArray(), method, optimalOrdering, owned: true);
        if (x.Rank != 2) throw new ArgumentException("`y` must be 1 or 2 dimensional.");
        if (method is LinkageMethod.Centroid or LinkageMethod.Median or LinkageMethod.Ward && metric != DistanceMetric.Euclidean)
            throw new ArgumentException($"`method={method}` requires the distance metric to be Euclidean.");
        return LinkageCore(Distance.Pdist(x, metric), method, optimalOrdering, owned: true);
    }

    private static double[] MstSingleLinkage(double[] dists, int n)
    {
        var z = new double[(n - 1) * 4];
        var merged = new bool[n];
        var d = new double[n];
        Array.Fill(d, double.PositiveInfinity);
        var x = 0;
        for (var k = 0; k < n - 1; k++)
        {
            var currentMin = double.PositiveInfinity;
            merged[x] = true;
            var y = -1;
            // Column part of row x (i < x) is strided in condensed storage, the rest contiguous;
            // both walked incrementally instead of recomputing the index per element.
            var idx = (long)x - 1;          // Ci(n, 0, x)
            for (var i = 0; i < x; i++)
            {
                if (!merged[i])
                {
                    var dist = dists[idx];
                    if (d[i] > dist) d[i] = dist;
                    if (d[i] < currentMin) { y = i; currentMin = d[i]; }
                }
                idx += n - i - 2;
            }
            var row = x + 1 < n ? CiOrdered(n, x, x + 1) - (x + 1) : 0;
            for (var i = x + 1; i < n; i++)
            {
                if (merged[i]) continue;
                var dist = dists[row + i];
                if (d[i] > dist) d[i] = dist;
                if (d[i] < currentMin) { y = i; currentMin = d[i]; }
            }
            z[4 * k] = x; z[4 * k + 1] = y; z[4 * k + 2] = currentMin;
            x = y;
        }
        return SortAndLabel(z, n);
    }

    /// <summary>
    /// Nearest-neighbour chain (Müllner 2011, as scipy's <c>nn_chain</c>) for the reducible
    /// methods, on a condensed matrix that it overwrites.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Live clusters are kept in an ascending doubly-linked list, so scans skip merged slots
    /// instead of testing a size per index — by the end almost every slot is dead. Within a scan
    /// the condensed index advances incrementally: down a column for <c>i &lt; x</c> (stride
    /// <c>n − i − 2</c>), along the row for <c>i &gt; x</c>.
    /// </para>
    /// <para>
    /// Order of visits and comparisons is unchanged (ascending i, strict &lt;), so ties resolve
    /// exactly as before and Z is bit-identical.
    /// </para>
    /// </remarks>
    private static double[] NnChain<TRule>(double[] d, int n) where TRule : struct, ILinkageRule
    {
        var rule = default(TRule);
        var z = new double[(n - 1) * 4];
        var size = new int[n];
        Array.Fill(size, 1);
        var next = new int[n + 1];
        var prev = new int[n + 1];
        // Sentinel n terminates the list; head is next[n].
        for (var i = 0; i < n; i++) { next[i] = i + 1; prev[i] = i - 1 < 0 ? n : i - 1; }
        next[n] = 0; prev[n] = n - 1;
        var chain = new int[n];
        var chainLength = 0;
        for (var k = 0; k < n - 1; k++)
        {
            if (chainLength == 0)
            {
                chainLength = 1;
                chain[0] = next[n];
            }
            int x, y = 0;
            double currentMin;
            while (true)
            {
                x = chain[chainLength - 1];
                // Prefer the previous chain element on ties, or the chain can cycle.
                if (chainLength > 1)
                {
                    y = chain[chainLength - 2];
                    currentMin = d[Ci(n, x, y)];
                }
                else currentMin = double.PositiveInfinity;

                var rowX = x + 1 < n ? CiOrdered(n, x, x + 1) - (x + 1) : 0;
                for (var i = next[n]; i != n; i = next[i])
                {
                    if (i == x) continue;
                    var dist = i < x ? d[CiOrdered(n, i, x)] : d[rowX + i];
                    if (dist < currentMin) { currentMin = dist; y = i; }
                }
                if (chainLength > 1 && y == chain[chainLength - 2]) break;
                chain[chainLength++] = y;
            }
            chainLength -= 2;
            if (x > y) (x, y) = (y, x);
            int nx = size[x], ny = size[y];
            z[4 * k] = x; z[4 * k + 1] = y; z[4 * k + 2] = currentMin; z[4 * k + 3] = nx + ny;
            size[x] = 0;
            size[y] = nx + ny;
            // Unlink x.
            next[prev[x]] = next[x];
            prev[next[x]] = prev[x];

            var rowx = CiOrdered(n, x, x + 1) - (x + 1);
            var rowy = y + 1 < n ? CiOrdered(n, y, y + 1) - (y + 1) : 0;
            for (var i = next[n]; i != n; i = next[i])
            {
                if (i == y) continue;
                var ni = size[i];
                long ix = i < x ? CiOrdered(n, i, x) : rowx + i;
                long iy = i < y ? CiOrdered(n, i, y) : rowy + i;
                d[iy] = rule.Update(d[ix], d[iy], currentMin, nx, ny, ni);
            }
        }
        return SortAndLabel(z, n);
    }


    /// <summary>Stable sort by distance, then relabel with a union-find (scipy's label()).</summary>
    private static double[] SortAndLabel(double[] z, int n)
    {
        var m = n - 1;
        var order = Enumerable.Range(0, m).OrderBy(i => z[4 * i + 2]).ToArray(); // OrderBy is stable, like mergesort
        var s = new double[m * 4];
        for (var i = 0; i < m; i++) Array.Copy(z, 4 * order[i], s, 4 * i, 4);
        var parent = new int[2 * n - 1];
        var size = new int[2 * n - 1];
        for (var i = 0; i < 2 * n - 1; i++) { parent[i] = i; size[i] = 1; }
        var next = n;
        int Find(int x)
        {
            var p = x;
            while (parent[x] != x) x = parent[x];
            while (parent[p] != x) { var t = parent[p]; parent[p] = x; p = t; }
            return x;
        }
        for (var i = 0; i < m; i++)
        {
            int xr = Find((int)s[4 * i]), yr = Find((int)s[4 * i + 1]);
            if (xr < yr) { s[4 * i] = xr; s[4 * i + 1] = yr; }
            else { s[4 * i] = yr; s[4 * i + 1] = xr; }
            parent[xr] = next; parent[yr] = next;
            size[next] = size[xr] + size[yr];
            s[4 * i + 3] = size[next];
            next++;
        }
        return s;
    }

    /// <summary>scipy's binary heap keyed by cluster index, used by the generic algorithm.</summary>
    private sealed class LinkageHeap
    {
        private readonly int[] _indexByKey, _keyByIndex;
        private readonly double[] _values;
        private int _size;

        public LinkageHeap(double[] values)
        {
            _size = values.Length;
            _indexByKey = new int[_size]; _keyByIndex = new int[_size];
            _values = (double[])values.Clone();
            for (var i = 0; i < _size; i++) { _indexByKey[i] = i; _keyByIndex[i] = i; }
            for (var i = _size / 2 - 1; i >= 0; i--) SiftDown(i);
        }

        public (int Key, double Value) GetMin() => (_keyByIndex[0], _values[0]);

        public void RemoveMin()
        {
            Swap(0, _size - 1);
            _size--;
            SiftDown(0);
        }

        public void ChangeValue(int key, double value)
        {
            var index = _indexByKey[key];
            var old = _values[index];
            _values[index] = value;
            if (value < old) SiftUp(index); else SiftDown(index);
        }

        private void SiftUp(int index)
        {
            var parent = (index - 1) >> 1;
            while (index > 0 && _values[parent] > _values[index])
            {
                Swap(index, parent);
                index = parent;
                parent = (index - 1) >> 1;
            }
        }

        private void SiftDown(int index)
        {
            var child = 2 * index + 1;
            while (child < _size)
            {
                if (child + 1 < _size && _values[child + 1] < _values[child]) child++;
                if (_values[index] > _values[child])
                {
                    Swap(index, child);
                    index = child;
                    child = 2 * index + 1;
                }
                else break;
            }
        }

        private void Swap(int i, int j)
        {
            (_values[i], _values[j]) = (_values[j], _values[i]);
            int ki = _keyByIndex[i], kj = _keyByIndex[j];
            _keyByIndex[i] = kj; _keyByIndex[j] = ki;
            _indexByKey[ki] = j; _indexByKey[kj] = i;
        }
    }

    /// <summary>Nearest live cluster above x: a contiguous walk along row x of the condensed matrix.</summary>
    private static (int Key, double Value) FindMinDist(int n, double[] d, int[] next, int x)
    {
        var currentMin = double.PositiveInfinity;
        var y = -1;
        var row = x + 1 < n ? CiOrdered(n, x, x + 1) - (x + 1) : 0;
        for (var i = next[x]; i != n; i = next[i])
        {
            var dist = d[row + i];
            if (dist < currentMin) { currentMin = dist; y = i; }
        }
        return (y, currentMin);
    }

    /// <summary>
    /// scipy's generic <c>fast_linkage</c> (Müllner's heap-of-nearest-neighbours algorithm) for the
    /// non-reducible methods (centroid, median), on a condensed matrix that it overwrites.
    /// </summary>
    private static double[] FastLinkage<TRule>(double[] d, int n) where TRule : struct, ILinkageRule
    {
        var rule = default(TRule);
        var z = new double[(n - 1) * 4];
        var size = new int[n];
        Array.Fill(size, 1);
        var next = new int[n + 1];
        var prev = new int[n + 1];
        for (var i = 0; i < n; i++) { next[i] = i + 1; prev[i] = i - 1 < 0 ? n : i - 1; }
        next[n] = 0; prev[n] = n - 1;
        var clusterId = new int[n];
        for (var i = 0; i < n; i++) clusterId[i] = i;
        var neighbor = new int[n - 1];
        var minDist = new double[n - 1];
        for (var x0 = 0; x0 < n - 1; x0++)
        {
            var (key, value) = FindMinDist(n, d, next, x0);
            neighbor[x0] = key; minDist[x0] = value;
        }
        var heap = new LinkageHeap(minDist);
        for (var k = 0; k < n - 1; k++)
        {
            int x = 0, y = 0;
            double dist = 0;
            for (var i = 0; i < n - k; i++)
            {
                (x, dist) = heap.GetMin();
                y = neighbor[x];
                if (dist == d[Ci(n, x, y)]) break;
                (y, dist) = FindMinDist(n, d, next, x);
                neighbor[x] = y; minDist[x] = dist;
                heap.ChangeValue(x, dist);
            }
            heap.RemoveMin();
            int idX = clusterId[x], idY = clusterId[y];
            int nx = size[x], ny = size[y];
            if (idX > idY) (idX, idY) = (idY, idX);
            z[4 * k] = idX; z[4 * k + 1] = idY; z[4 * k + 2] = dist; z[4 * k + 3] = nx + ny;
            size[x] = 0;
            size[y] = nx + ny;
            clusterId[y] = n + k;
            next[prev[x]] = next[x];
            prev[next[x]] = prev[x];

            // x < y always here: x is the heap key, y its neighbour above it.
            var rowx = CiOrdered(n, x, x + 1) - (x + 1);
            var rowy = y + 1 < n ? CiOrdered(n, y, y + 1) - (y + 1) : 0;
            for (var w = next[n]; w != n; w = next[w])
            {
                if (w == y) continue;
                long iw = w < x ? CiOrdered(n, w, x) : rowx + w;
                long iy = w < y ? CiOrdered(n, w, y) : rowy + w;
                d[iy] = rule.Update(d[iw], d[iy], dist, nx, ny, size[w]);
            }
            for (var w = next[n]; w < x; w = next[w])
                if (neighbor[w] == x) neighbor[w] = y;
            for (var w = next[n]; w < y; w = next[w])
            {
                var dd = d[CiOrdered(n, w, y)];
                if (dd < minDist[w])
                {
                    neighbor[w] = y; minDist[w] = dd;
                    heap.ChangeValue(w, dd);
                }
            }
            if (y < n - 1)
            {
                var (zk, zd) = FindMinDist(n, d, next, y);
                if (zk != -1)
                {
                    neighbor[y] = zk; minDist[y] = zd;
                    heap.ChangeValue(y, zd);
                }
            }
        }
        return z;
    }

    // ================================================================== validation

    private static double[] ZBuffer(NdArray z, out int n)
    {
        IsValidLinkage(z, throwOnError: true);
        n = z.Shape[0] + 1;
        return z.AsContiguous().ToArray();
    }

    /// <summary>
    /// Whether <paramref name="z"/> is a well-formed linkage matrix: (m, 4), non-negative
    /// indices, distances and counts, no cluster used before it is formed or used twice.
    /// </summary>
    public static bool IsValidLinkage(NdArray z, bool throwOnError = false)
    {
        string? error = null;
        if (z is null || z.Rank != 2) error = "Linkage matrix must have shape=2 (i.e. be two-dimensional).";
        else if (z.Shape[1] != 4) error = "Linkage matrix must have 4 columns.";
        else if (z.Shape[0] == 0) error = "Linkage must be computed on at least two observations.";
        else
        {
            var m = z.Shape[0];
            var a = z.AsContiguous().ToArray();
            var seen = new HashSet<double>();
            for (var i = 0; i < m && error is null; i++)
            {
                if (a[4 * i] < 0 || a[4 * i + 1] < 0) error = "Linkage contains negative indices.";
                else if (a[4 * i + 2] < 0) error = "Linkage contains negative distances.";
                else if (a[4 * i + 3] < 0) error = "Linkage contains negative counts.";
                else if (a[4 * i + 3] > m + 1) error = "Linkage contains excessive observations in a cluster";
                else if (Math.Max(a[4 * i], a[4 * i + 1]) >= m + 1 + i) error = "Linkage uses non-singleton cluster before it is formed.";
                else if (!seen.Add(a[4 * i]) | !seen.Add(a[4 * i + 1])) error = "Linkage uses the same cluster more than once.";
            }
        }
        if (error is not null && throwOnError) throw new ArgumentException(error, nameof(z));
        return error is null;
    }

    /// <summary>Whether merge distances never decrease down the rows of Z (no inversions).</summary>
    public static bool IsMonotonic(NdArray z)
    {
        var a = ZBuffer(z, out var n);
        for (var i = 1; i < n - 1; i++) if (a[4 * i + 2] < a[4 * (i - 1) + 2]) return false;
        return true;
    }

    /// <summary>Number of observations of a linkage matrix (rows + 1).</summary>
    public static int NumObsLinkage(NdArray z) { ZBuffer(z, out var n); return n; }

    /// <summary>Whether a linkage matrix and a condensed distance matrix describe the same number of observations.</summary>
    public static bool Correspond(NdArray z, double[] y) => Distance.NumObsY(y.LongLength) == NumObsLinkage(z);

    // ================================================================== trees and orderings

    /// <summary>Converts Z into a tree; returns the root.</summary>
    public static ClusterNode ToTree(NdArray z) => ToTree(z, out _);

    /// <summary>Converts Z into a tree; <paramref name="nodes"/> holds every node by id (scipy's <c>rd=True</c>).</summary>
    public static ClusterNode ToTree(NdArray z, out ClusterNode[] nodes)
    {
        var a = ZBuffer(z, out var n);
        nodes = new ClusterNode[2 * n - 1];
        for (var i = 0; i < n; i++) nodes[i] = new ClusterNode(i);
        for (var i = 0; i < n - 1; i++)
        {
            int fi = (int)a[4 * i], fj = (int)a[4 * i + 1];
            if (fi > i + n) throw new ArgumentException($"Corrupt matrix Z. Index to derivative cluster is used before it is formed. See row {fi}, column 0");
            if (fj > i + n) throw new ArgumentException($"Corrupt matrix Z. Index to derivative cluster is used before it is formed. See row {fj}, column 1");
            var nd = new ClusterNode(i + n, nodes[fi], nodes[fj], a[4 * i + 2]);
            if ((int)a[4 * i + 3] != nd.Count) throw new ArgumentException($"Corrupt matrix Z. The count Z[{i},3] is incorrect.");
            nodes[n + i] = nd;
        }
        return nodes[2 * n - 2];
    }

    /// <summary>Observation ids in the left-to-right order of the dendrogram leaves.</summary>
    public static int[] LeavesList(NdArray z)
    {
        var a = ZBuffer(z, out var n);
        var result = new int[n];
        var k = 0;
        var stack = new Stack<int>();
        stack.Push(2 * n - 2);
        while (stack.Count > 0)
        {
            var id = stack.Pop();
            if (id < n) { result[k++] = id; continue; }
            stack.Push((int)a[4 * (id - n) + 1]);
            stack.Push((int)a[4 * (id - n)]);
        }
        return result;
    }

    private static List<int>[] Members(double[] a, int n)
    {
        var members = new List<int>[2 * n - 1];
        for (var i = 0; i < n; i++) members[i] = [i];
        for (var i = 0; i < n - 1; i++)
        {
            var l = members[(int)a[4 * i]]; var r = members[(int)a[4 * i + 1]];
            var m = new List<int>(l.Count + r.Count);
            m.AddRange(l); m.AddRange(r);
            members[n + i] = m;
        }
        return members;
    }

    /// <summary>
    /// Cophenetic distances (condensed): the height of the link at which each pair of
    /// observations first joins.
    /// </summary>
    public static double[] Cophenet(NdArray z)
    {
        var a = ZBuffer(z, out var n);
        var zz = new double[(long)n * (n - 1) / 2];
        var members = Members(a, n);
        for (var i = 0; i < n - 1; i++)
        {
            var h = a[4 * i + 2];
            foreach (var p in members[(int)a[4 * i]])
                foreach (var q in members[(int)a[4 * i + 1]])
                    zz[Ci(n, p, q)] = h;
        }
        return zz;
    }

    /// <summary>
    /// Cophenetic distances and the cophenetic correlation coefficient with the original
    /// distances <paramref name="y"/> (Pearson correlation of the two condensed vectors).
    /// </summary>
    public static (double C, double[] CophDists) Cophenet(NdArray z, double[] y)
    {
        ArgumentNullException.ThrowIfNull(y);
        var zz = Cophenet(z);
        if (y.Length != zz.Length) throw new ArgumentException("Y does not correspond to Z.", nameof(y));
        double my = y.Average(), mz = zz.Average();
        double num = 0, da = 0, db = 0;
        for (var i = 0; i < y.Length; i++)
        {
            var yy = y[i] - my; var zv = zz[i] - mz;
            num += yy * zv; da += yy * yy; db += zv * zv;
        }
        return (num / Math.Sqrt(da * db), zz);
    }

    /// <summary>
    /// Inconsistency statistics, (n-1) × 4: for each link, the mean and standard deviation
    /// (ddof 1) of the heights of the links within <paramref name="d"/> levels below it
    /// (itself included), their count, and the inconsistency coefficient (height - mean)/std
    /// (0 when std is 0).
    /// </summary>
    public static NdArray Inconsistent(NdArray z, int d = 2)
    {
        if (d < 0) throw new ArgumentException("The second argument d must be a nonnegative integer value.");
        var a = ZBuffer(z, out var n);
        var r = new double[(n - 1) * 4];
        var stack = new int[n];
        var visited = new bool[2 * n];
        for (var i = 0; i < n - 1; i++)
        {
            var k = 0;
            double levelSum = 0, levelStdSum = 0;
            var levelCount = 0;
            Array.Clear(visited);
            stack[0] = i;
            while (k >= 0)
            {
                var root = stack[k];
                if (k < d - 1)
                {
                    var lc = (int)a[4 * root];
                    if (lc >= n && !visited[lc]) { visited[lc] = true; stack[++k] = lc - n; continue; }
                    var rc = (int)a[4 * root + 1];
                    if (rc >= n && !visited[rc]) { visited[rc] = true; stack[++k] = rc - n; continue; }
                }
                var dist = a[4 * root + 2];
                levelCount++;
                levelSum += dist;
                levelStdSum += dist * dist;
                k--;
            }
            r[4 * i + 2] = levelCount;
            r[4 * i] = levelSum / levelCount;
            var levelStd = levelCount < 2
                ? (levelStdSum - levelSum * levelSum) / levelCount
                : (levelStdSum - levelSum * levelSum / levelCount) / (levelCount - 1);
            if (levelStd > 0)
            {
                levelStd = Math.Sqrt(levelStd);
                r[4 * i + 1] = levelStd;
                r[4 * i + 3] = (a[4 * i + 2] - r[4 * i]) / levelStd;
            }
        }
        return new NdArray(r, n - 1, 4);
    }

    /// <summary>Maximum of a per-link statistic over each link's subtree (post-order).</summary>
    private static double[] MaxOverSubtree(double[] a, int n, Func<int, double> value)
    {
        var md = new double[n - 1];
        // Rows are in formation order, so children come before parents.
        for (var i = 0; i < n - 1; i++)
        {
            var m = value(i);
            int lc = (int)a[4 * i], rc = (int)a[4 * i + 1];
            if (lc >= n) m = Math.Max(m, md[lc - n]);
            if (rc >= n) m = Math.Max(m, md[rc - n]);
            md[i] = m;
        }
        return md;
    }

    /// <summary>Maximum merge distance in each link's subtree.</summary>
    public static double[] Maxdists(NdArray z)
    {
        var a = ZBuffer(z, out var n);
        return MaxOverSubtree(a, n, i => a[4 * i + 2]);
    }

    /// <summary>Maximum inconsistency coefficient (column 3 of R) in each link's subtree.</summary>
    public static double[] Maxinconsts(NdArray z, NdArray r) => MaxRstat(z, r, 3);

    /// <summary>Maximum of column <paramref name="i"/> of the inconsistency matrix in each link's subtree.</summary>
    public static double[] MaxRstat(NdArray z, NdArray r, int i)
    {
        if (i < 0 || i > 3) throw new ArgumentException("i must be an integer between 0 and 3 inclusive.");
        var a = ZBuffer(z, out var n);
        ArgumentNullException.ThrowIfNull(r);
        if (r.Rank != 2 || r.Shape[0] != n - 1 || r.Shape[1] != 4) throw new ArgumentException("The inconsistency matrix and linkage matrix each have a different number of rows.");
        var rv = r.AsContiguous().ToArray();
        return MaxOverSubtree(a, n, row => rv[4 * row + i]);
    }

    // ================================================================== flat clusters

    /// <summary>
    /// Flat clusters from a hierarchy (scipy's <c>fcluster</c>); returns 1-based cluster numbers
    /// in scipy's numbering (clusters are numbered in a left-to-right traversal of the tree).
    /// </summary>
    /// <param name="z">Linkage matrix.</param>
    /// <param name="t">Threshold, or the maximum number of clusters for the Maxclust criteria.</param>
    /// <param name="criterion">Flattening rule.</param>
    /// <param name="depth">Depth of the inconsistency calculation (Inconsistent criterion).</param>
    /// <param name="r">Precomputed inconsistency matrix (Inconsistent criterion).</param>
    /// <param name="monocrit">Per-link monotone statistic (the Monocrit criteria).</param>
    public static int[] Fcluster(NdArray z, double t, FclusterCriterion criterion = FclusterCriterion.Inconsistent,
        int depth = 2, NdArray? r = null, double[]? monocrit = null)
    {
        var a = ZBuffer(z, out var n);
        var result = new int[n];
        switch (criterion)
        {
            case FclusterCriterion.Inconsistent:
                ClusterMonocrit(a, n, Maxinconsts(z, r ?? Inconsistent(z, depth)), t, result);
                break;
            case FclusterCriterion.Distance:
                ClusterMonocrit(a, n, Maxdists(z), t, result);
                break;
            case FclusterCriterion.Maxclust:
                ClusterMaxclustMonocrit(a, n, Maxdists(z), (int)t, result);
                break;
            case FclusterCriterion.Monocrit:
                ClusterMonocrit(a, n, CheckMonocrit(monocrit, n), t, result);
                break;
            case FclusterCriterion.MaxclustMonocrit:
                ClusterMaxclustMonocrit(a, n, CheckMonocrit(monocrit, n), (int)t, result);
                break;
            default:
                throw new ArgumentException($"Invalid cluster formation criterion: {criterion}");
        }
        return result;
    }

    private static double[] CheckMonocrit(double[]? mc, int n)
    {
        if (mc is null) throw new ArgumentException("This criterion needs a monocrit array.");
        if (mc.Length != n - 1) throw new ArgumentException("monocrit must have one value per link (n - 1).");
        return mc;
    }

    /// <summary>scipy's cluster_monocrit: a node forms a cluster when its statistic is at most the cutoff.</summary>
    private static void ClusterMonocrit(double[] z, int n, double[] mc, double cutoff, int[] t)
    {
        var currNode = new int[n];
        var visited = new bool[2 * n];
        var clusterLeader = -1;
        var nCluster = 0;
        var k = 0;
        currNode[0] = 2 * n - 2;
        while (k >= 0)
        {
            var root = currNode[k] - n;
            int lc = (int)z[4 * root], rc = (int)z[4 * root + 1];
            if (clusterLeader == -1 && mc[root] <= cutoff)
            {
                clusterLeader = root;
                nCluster++;
            }
            if (lc >= n && !visited[lc]) { visited[lc] = true; currNode[++k] = lc; continue; }
            if (rc >= n && !visited[rc]) { visited[rc] = true; currNode[++k] = rc; continue; }
            if (lc < n)
            {
                if (clusterLeader == -1) nCluster++;
                t[lc] = nCluster;
            }
            if (rc < n)
            {
                if (clusterLeader == -1) nCluster++;
                t[rc] = nCluster;
            }
            if (clusterLeader == root) clusterLeader = -1;
            k--;
        }
    }

    /// <summary>
    /// scipy's cluster_maxclust_monocrit: bisect over the row positions of the statistic for
    /// the first threshold <c>mc[i]</c> giving at most maxNc clusters, then
    /// <see cref="ClusterMonocrit"/> at that threshold.
    /// </summary>
    /// <remarks>The bisection runs over <em>row positions</em>, not sorted values, which is only
    /// a true minimum when the statistic increases down the rows. For non-monotone linkages
    /// (centroid, median) or a maxinconsts vector it is not, and the threshold scipy picks can
    /// over-merge; this reproduces scipy's choice rather than "fixing" it, because an fcluster
    /// that disagrees with scipy's labels would be more surprising than one that matches them.
    /// The bounds (-1, n-1) were established by replaying 2400 scipy 1.17.1 calls.</remarks>
    private static void ClusterMaxclustMonocrit(double[] z, int n, double[] mc, int maxNc, int[] t)
    {
        if (maxNc < 1) throw new ArgumentException("The maximum number of clusters must be at least 1.");
        if (maxNc >= n)
        {
            // Every observation alone: scipy numbers them in observation order here.
            for (var i = 0; i < n; i++) t[i] = i + 1;
            return;
        }
        int lower = -1, upper = n - 1;
        var stack = new Stack<int>();
        while (upper - lower > 1)
        {
            var i = (lower + upper) >> 1;
            var thresh = mc[i];
            var nc = 0;
            stack.Clear();
            stack.Push(2 * n - 2);
            while (stack.Count > 0 && nc <= maxNc)
            {
                var node = stack.Pop();
                if (node < n) { nc++; continue; }
                var r = node - n;
                if (mc[r] <= thresh) { nc++; continue; }
                stack.Push((int)z[4 * r]); stack.Push((int)z[4 * r + 1]);
            }
            if (nc > maxNc) lower = i; else upper = i;
        }
        ClusterMonocrit(z, n, mc, upper < n - 1 ? mc[upper] : double.PositiveInfinity, t);
    }

    /// <summary>
    /// Clusters observations directly (scipy's <c>fclusterdata</c>): pdist, linkage, fcluster.
    /// </summary>
    public static int[] FclusterData(NdArray x, double t, FclusterCriterion criterion = FclusterCriterion.Inconsistent,
        DistanceMetric metric = DistanceMetric.Euclidean, int depth = 2, LinkageMethod method = LinkageMethod.Single, NdArray? r = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Rank != 2) throw new ArgumentException("The observation matrix X must be an n by m array.");
        var z = Linkage(Distance.Pdist(x, metric), method);
        return Fcluster(z, t, criterion, depth, r ?? Inconsistent(z, depth));
    }

    /// <summary>
    /// Cluster membership after each merge (scipy's <c>cut_tree</c>), shape [n, columns]: with
    /// neither argument, column c is the partition into n - c clusters; otherwise one column per
    /// requested cluster count or height. Labels are 0-based and renumbered so they stay
    /// consecutive.
    /// </summary>
    public static int[,] CutTree(NdArray z, int[]? nClusters = null, double[]? height = null)
    {
        if (nClusters is not null && height is not null) throw new ArgumentException("At least one of either height or n_clusters must be None.");
        var root = ToTree(z);
        var nobs = root.Count;
        // scipy's _order_cluster_tree: breadth-first, inserted with bisect.insort_left by dist.
        var nodes = new List<ClusterNode>();
        var q = new Queue<ClusterNode>();
        q.Enqueue(root);
        while (q.Count > 0)
        {
            var node = q.Dequeue();
            if (node.IsLeaf) continue;
            var lo = 0; var hi = nodes.Count;
            while (lo < hi) { var mid = (lo + hi) / 2; if (nodes[mid].Dist < node.Dist) lo = mid + 1; else hi = mid; }
            nodes.Insert(lo, node);
            q.Enqueue(node.Right!);
            q.Enqueue(node.Left!);
        }
        int[] cols;
        if (height is not null)
        {
            var heights = nodes.Select(x => x.Dist).ToArray();
            cols = height.Select(h => { var lo = 0; var hi = heights.Length; while (lo < hi) { var mid = (lo + hi) / 2; if (heights[mid] < h) lo = mid + 1; else hi = mid; } return lo; }).ToArray();
        }
        else if (nClusters is not null)
            cols = nClusters.Select(c => nobs - Math.Clamp(c, 0, nobs)).ToArray();
        else cols = Enumerable.Range(0, nobs).ToArray();

        var groups = new int[cols.Length][];
        var last = Enumerable.Range(0, nobs).ToArray();
        for (var c = 0; c < cols.Length; c++) if (cols[c] == 0) groups[c] = (int[])last.Clone();
        for (var i = 0; i < nodes.Count; i++)
        {
            var idx = nodes[i].PreOrder();
            var cur = (int[])last.Clone();
            var min = idx.Min(j => last[j]);
            var max = idx.Max(j => last[j]);
            foreach (var j in idx) cur[j] = min;
            for (var j = 0; j < nobs; j++) if (cur[j] > max) cur[j]--;
            for (var c = 0; c < cols.Length; c++) if (cols[c] == i + 1) groups[c] = (int[])cur.Clone();
            last = cur;
        }
        var result = new int[nobs, cols.Length];
        for (var c = 0; c < cols.Length; c++)
            for (var j = 0; j < nobs; j++) result[j, c] = groups[c]?[j] ?? 0;
        return result;
    }

    // ================================================================== optimal leaf ordering

    /// <summary>
    /// Reorders the children of each link so that the sum of distances between adjacent leaves
    /// is minimal (Bar-Joseph, Gifford and Jaakkola 2001), returning a new Z whose rows and
    /// heights are unchanged and whose column 0/1 entries may be swapped.
    /// </summary>
    /// <remarks>Dynamic programming over a single n × n table: every pair of leaves is
    /// separated at exactly one link, which owns that pair's entry. Worst case O(n³). When
    /// several orderings are optimal, the one chosen can differ from scipy's (ties are broken
    /// by first occurrence in index order here).</remarks>
    public static NdArray OptimalLeafOrdering(NdArray z, double[] y)
    {
        ArgumentNullException.ThrowIfNull(y);
        var a = ZBuffer(z, out var n);
        if (Distance.NumObsY(y.LongLength) != n) throw new ArgumentException("y does not correspond to Z.");
        var members = Members(a, n);
        var m = new double[n * n];
        var part = new int[n * n];      // partner of i inside i's side, for the pair (i, j)
        var childOf = new int[2 * n - 1][];
        for (var i = 0; i < n - 1; i++) childOf[n + i] = [(int)a[4 * i], (int)a[4 * i + 1]];
        double D(int p, int q) => p == q ? 0 : y[Ci(n, p, q)];

        // For leaf i inside subtree s, the leaves of s's child that does not contain i.
        List<int> Opposite(int s, int i)
        {
            if (s < n) return [i];
            var c = childOf[s];
            return members[c[0]].Contains(i) ? members[c[1]] : members[c[0]];
        }

        for (var row = 0; row < n - 1; row++)
        {
            int A = (int)a[4 * row], B = (int)a[4 * row + 1];
            var la = members[A]; var lb = members[B];
            // T[i, l] = min_k M[i,k] + D[k,l], k on the far side of i within A.
            var tVal = new double[la.Count * lb.Count];
            var tArg = new int[la.Count * lb.Count];
            for (var ii = 0; ii < la.Count; ii++)
            {
                var i = la[ii];
                var ks = Opposite(A, i);
                for (var ll = 0; ll < lb.Count; ll++)
                {
                    var l = lb[ll];
                    double best = double.PositiveInfinity; var arg = -1;
                    foreach (var k in ks)
                    {
                        var v = (k == i ? 0 : m[i * n + k]) + D(k, l);
                        if (v < best) { best = v; arg = k; }
                    }
                    tVal[ii * lb.Count + ll] = best; tArg[ii * lb.Count + ll] = arg;
                }
            }
            var lIndex = new Dictionary<int, int>();
            for (var ll = 0; ll < lb.Count; ll++) lIndex[lb[ll]] = ll;
            for (var jj = 0; jj < lb.Count; jj++)
            {
                var j = lb[jj];
                var ls = Opposite(B, j);
                for (var ii = 0; ii < la.Count; ii++)
                {
                    var i = la[ii];
                    double best = double.PositiveInfinity; int argL = -1;
                    foreach (var l in ls)
                    {
                        var v = tVal[ii * lb.Count + lIndex[l]] + (l == j ? 0 : m[l * n + j]);
                        if (v < best) { best = v; argL = l; }
                    }
                    m[i * n + j] = best; m[j * n + i] = best;
                    part[i * n + j] = tArg[ii * lb.Count + lIndex[argL]];
                    part[j * n + i] = argL;
                }
            }
        }

        // Best end points at the root.
        var rootRow = n - 2;
        int ra = (int)a[4 * rootRow], rb = (int)a[4 * rootRow + 1];
        double bestRoot = double.PositiveInfinity; int first = -1, last = -1;
        foreach (var i in members[ra])
            foreach (var j in members[rb])
                if (m[i * n + j] < bestRoot) { bestRoot = m[i * n + j]; first = i; last = j; }

        var outZ = (double[])a.Clone();
        var stack = new Stack<(int Node, int First, int Last)>();
        stack.Push((2 * n - 2, first, last));
        while (stack.Count > 0)
        {
            var (node, f, l) = stack.Pop();
            if (node < n) continue;
            var r = node - n;
            int c0 = (int)a[4 * r], c1 = (int)a[4 * r + 1];
            if (!members[c0].Contains(f))
            {
                (c0, c1) = (c1, c0);
                outZ[4 * r] = c0; outZ[4 * r + 1] = c1;
            }
            stack.Push((c0, f, c0 < n ? f : part[f * n + l]));
            stack.Push((c1, c1 < n ? l : part[l * n + f], l));
        }
        return new NdArray(outZ, n - 1, 4);
    }
}
