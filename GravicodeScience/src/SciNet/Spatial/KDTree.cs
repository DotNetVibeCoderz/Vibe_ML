using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Spatial;

/// <summary>
/// k-d tree for nearest-neighbour and fixed-radius queries (scipy.spatial.KDTree / cKDTree).
/// </summary>
/// <remarks>
/// <para><b>Layout.</b> The points are copied once into a contiguous row-major buffer, then a
/// second time in tree order, so a leaf scan reads consecutive memory. <see cref="Indices"/> maps
/// tree position back to input row, exactly like cKDTree's <c>indices</c>.</para>
/// <para><b>Splits.</b> <c>balancedTree: true</c> (scipy's default) splits at the median of the
/// widest dimension; otherwise the sliding-midpoint rule of Maneewongvatana and Mount (1999)
/// is used, which builds faster and copes better with clustered data. The split dimension is
/// always chosen from the node's actual data spread, so piles of duplicate points become a leaf
/// instead of an O(n)-deep chain.</para>
/// <para><b>Boxes.</b> With <c>compactNodes: true</c> each node stores the tight bounding box of
/// its points; otherwise the box inherited from the splits. Both are valid for pruning; tight
/// boxes prune more and cost one pass over each level at build time.</para>
/// <para><b>Ties.</b> Neighbours are ordered by (distance, index). A candidate at exactly the
/// current k-th distance replaces it only if its index is smaller, so results are the k smallest
/// pairs under that order whatever the tree shape or thread count. cKDTree breaks ties by
/// traversal order instead, so on data with equal distances the indices may differ from scipy's
/// while the distances agree.</para>
/// <para><b>Not supported:</b> periodic boundaries (scipy's <c>boxsize</c>) and the
/// <c>weights</c> argument of <c>count_neighbors</c>.</para>
/// </remarks>
public sealed class KDTree
{
    private const int ParallelThreshold = 256;

    private readonly double[] _data;   // input order
    private readonly double[] _tree;   // tree order
    private readonly int[] _index;     // tree position -> input row

    private int[] _start = [], _end = [], _left = [], _right = [];
    private double[] _lo = [], _hi = [];
    private int _nodes;

    /// <summary>Number of points.</summary>
    public int N { get; }

    /// <summary>Dimension of the points.</summary>
    public int M { get; }

    /// <summary>Maximum number of points in a leaf.</summary>
    public int LeafSize { get; }

    /// <summary>Number of tree nodes (internal and leaves).</summary>
    public int NodeCount => _nodes;

    /// <summary>A copy of the indexed points, shape (N, M).</summary>
    public NdArray Data => new((double[])_data.Clone(), N, M);

    /// <summary>Per-dimension minimum of the data.</summary>
    public double[] Mins { get; }

    /// <summary>Per-dimension maximum of the data.</summary>
    public double[] Maxes { get; }

    /// <summary>Tree order: <c>Indices[t]</c> is the input row stored at tree position t.</summary>
    public int[] Indices => (int[])_index.Clone();

    /// <summary>Builds a tree over the rows of an (n, m) array.</summary>
    /// <param name="data">Points, shape (n, m). A rank-1 array is read as n points in 1-D.</param>
    /// <param name="leafSize">Maximum points per leaf (scipy's KDTree default is 10).</param>
    /// <param name="compactNodes">Shrink each node's box to its data (scipy default true).</param>
    /// <param name="balancedTree">Median splits when true, sliding midpoint when false.</param>
    public KDTree(NdArray data, int leafSize = 10, bool compactNodes = true, bool balancedTree = true)
        : this(SpatialInput.Copy(data, out var n, out var m, nameof(data)), n, m, leafSize, compactNodes, balancedTree) { }

    /// <summary>Builds a tree over the rows of a rectangular array.</summary>
    public KDTree(double[,] data, int leafSize = 10, bool compactNodes = true, bool balancedTree = true)
        : this(SpatialInput.Copy(data, out var n, out var m), n, m, leafSize, compactNodes, balancedTree) { }

    /// <summary>Builds a tree over a jagged array of points.</summary>
    public KDTree(double[][] data, int leafSize = 10, bool compactNodes = true, bool balancedTree = true)
        : this(SpatialInput.Copy(data, out var n, out var m), n, m, leafSize, compactNodes, balancedTree) { }

    private KDTree(double[] data, int n, int m, int leafSize, bool compact, bool balanced)
    {
        if (leafSize < 1) throw new ArgumentException("leafSize must be at least 1.", nameof(leafSize));
        if (m < 1 && n > 0) throw new ArgumentException("Points must have at least one coordinate.");
        SpatialInput.RequireFinite(data, "KDTree data");
        _data = data; N = n; M = m; LeafSize = leafSize;
        Mins = new double[m]; Maxes = new double[m];
        _index = new int[n];
        for (var i = 0; i < n; i++) _index[i] = i;
        if (n > 0)
        {
            Bounds(0, n, Mins, Maxes);
            var cap = Math.Max(1, 4 * (n / Math.Max(1, leafSize / 2) + 1));
            Grow(cap);
            Build(0, n, (double[])Mins.Clone(), (double[])Maxes.Clone(), compact, balanced);
        }
        _tree = new double[n * m];
        for (var t = 0; t < n; t++) Array.Copy(_data, _index[t] * m, _tree, t * m, m);
    }

    // ------------------------------------------------------------------ construction

    private void Grow(int capacity)
    {
        Array.Resize(ref _start, capacity); Array.Resize(ref _end, capacity);
        Array.Resize(ref _left, capacity); Array.Resize(ref _right, capacity);
        Array.Resize(ref _lo, capacity * M); Array.Resize(ref _hi, capacity * M);
    }

    private void Bounds(int start, int end, Span<double> lo, Span<double> hi)
    {
        var m = M;
        lo.Fill(double.PositiveInfinity); hi.Fill(double.NegativeInfinity);
        for (var t = start; t < end; t++)
        {
            var row = _data.AsSpan(_index[t] * m, m);
            for (var j = 0; j < m; j++)
            {
                var v = row[j];
                if (v < lo[j]) lo[j] = v;
                if (v > hi[j]) hi[j] = v;
            }
        }
    }

    private int Build(int start, int end, double[] boxLo, double[] boxHi, bool compact, bool balanced)
    {
        var m = M;
        if (_nodes == _start.Length) Grow(_start.Length * 2);
        var node = _nodes++;
        _start[node] = start; _end[node] = end; _left[node] = -1; _right[node] = -1;

        var dataLo = new double[m]; var dataHi = new double[m];
        Bounds(start, end, dataLo, dataHi);
        var lo = compact ? dataLo : boxLo;
        var hi = compact ? dataHi : boxHi;
        lo.CopyTo(_lo, node * m); hi.CopyTo(_hi, node * m);

        if (end - start <= LeafSize) return node;

        var dim = 0; var spread = -1.0;
        for (var j = 0; j < m; j++)
        {
            var s = dataHi[j] - dataLo[j];
            if (s > spread) { spread = s; dim = j; }
        }
        if (spread <= 0) return node; // all points identical: a leaf, however large

        int mid; double split;
        if (balanced)
        {
            mid = start + (end - start) / 2;
            Select(start, end, mid, dim);
            split = _data[_index[mid] * m + dim];
        }
        else
        {
            split = 0.5 * (dataLo[dim] + dataHi[dim]);
            mid = Partition(start, end, dim, split);
            if (mid == start)
            {
                // Midpoint rounded onto the minimum (adjacent doubles): slide to the minimum point.
                var best = start;
                for (var t = start + 1; t < end; t++)
                    if (_data[_index[t] * m + dim] < _data[_index[best] * m + dim]) best = t;
                (_index[start], _index[best]) = (_index[best], _index[start]);
                mid = start + 1;
                split = _data[_index[start] * m + dim];
            }
            else if (mid == end)
            {
                var best = start;
                for (var t = start + 1; t < end; t++)
                    if (_data[_index[t] * m + dim] > _data[_index[best] * m + dim]) best = t;
                (_index[end - 1], _index[best]) = (_index[best], _index[end - 1]);
                mid = end - 1;
                split = _data[_index[end - 1] * m + dim];
            }
        }

        var leftHi = (double[])hi.Clone(); leftHi[dim] = Math.Min(leftHi[dim], split);
        var rightLo = (double[])lo.Clone(); rightLo[dim] = Math.Max(rightLo[dim], split);
        var l = Build(start, mid, (double[])lo.Clone(), leftHi, compact, balanced);
        var r = Build(mid, end, rightLo, (double[])hi.Clone(), compact, balanced);
        _left[node] = l; _right[node] = r;
        return node;
    }

    /// <summary>Moves rows with coordinate &lt; split before the rest; returns the boundary.</summary>
    private int Partition(int start, int end, int dim, double split)
    {
        var m = M;
        int i = start, j = end - 1;
        while (i <= j)
        {
            if (_data[_index[i] * m + dim] < split) i++;
            else { (_index[i], _index[j]) = (_index[j], _index[i]); j--; }
        }
        return i;
    }

    /// <summary>Quickselect (median-of-three, Hoare partition) on the index range.</summary>
    private void Select(int start, int end, int k, int dim)
    {
        var m = M;
        int lo = start, hi = end - 1;
        while (hi > lo)
        {
            var midI = lo + (hi - lo) / 2;
            double a = _data[_index[lo] * m + dim], b = _data[_index[midI] * m + dim], c = _data[_index[hi] * m + dim];
            var pivot = Math.Max(Math.Min(a, b), Math.Min(Math.Max(a, b), c));
            int i = lo, j = hi;
            while (i <= j)
            {
                while (_data[_index[i] * m + dim] < pivot) i++;
                while (_data[_index[j] * m + dim] > pivot) j--;
                if (i <= j) { (_index[i], _index[j]) = (_index[j], _index[i]); i++; j--; }
            }
            if (k <= j) hi = j;
            else if (k >= i) lo = i;
            else return;
        }
    }

    // ------------------------------------------------------------------ geometry helpers

    private double BoxMin<TNorm>(TNorm norm, int node, ReadOnlySpan<double> x) where TNorm : struct, ISpatialNorm
    {
        var m = M; var o = node * m;
        var acc = 0.0;
        for (var j = 0; j < m; j++)
        {
            var v = x[j];
            var g = v < _lo[o + j] ? _lo[o + j] - v : v > _hi[o + j] ? v - _hi[o + j] : 0.0;
            acc = norm.Add(acc, g);
        }
        return acc;
    }

    private double BoxMax<TNorm>(TNorm norm, int node, ReadOnlySpan<double> x) where TNorm : struct, ISpatialNorm
    {
        var m = M; var o = node * m;
        var acc = 0.0;
        for (var j = 0; j < m; j++)
        {
            var v = x[j];
            acc = norm.Add(acc, Math.Max(Math.Abs(v - _lo[o + j]), Math.Abs(_hi[o + j] - v)));
        }
        return acc;
    }

    private static void CheckP(double p)
    {
        if (!(p >= 1)) throw new ArgumentException("p must satisfy 1 <= p <= infinity.", nameof(p));
    }

    private static double EpsFactor(double p, double eps)
    {
        if (eps < 0) throw new ArgumentException("eps must be non-negative.", nameof(eps));
        if (eps == 0) return 1.0;
        if (double.IsPositiveInfinity(p)) return 1.0 / (1.0 + eps);
        return 1.0 / Math.Pow(1.0 + eps, p);
    }

    private static int Workers(int workers) => workers == -1 ? Environment.ProcessorCount : Math.Max(1, workers);

    private double[] QueryBuffer(NdArray x, out int q)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Rank == 1)
        {
            if (x.Shape[0] != M) throw new ArgumentException($"x must have shape (..., {M}).", nameof(x));
            q = 1;
        }
        else if (x.Rank == 2)
        {
            if (x.Shape[1] != M) throw new ArgumentException($"x must have shape (..., {M}).", nameof(x));
            q = x.Shape[0];
        }
        else throw new ArgumentException("x must be rank 1 or 2.", nameof(x));
        var buf = new double[q * M];
        x.CopyTo(buf);
        return buf;
    }

    private void CheckPoint(ReadOnlySpan<double> x)
    {
        if (x.Length != M) throw new ArgumentException($"x must have length {M}.", nameof(x));
    }

    // ------------------------------------------------------------------ k nearest neighbours

    /// <summary>Bounded max-heap of (power distance, index) ordered lexicographically.</summary>
    private sealed class KnnHeap(int k)
    {
        public readonly double[] D = new double[k];
        public readonly int[] I = new int[k];
        public int Count;
        public readonly int K = k;

        public double Bound(double ub) => Count < K ? ub : D[0];

        private bool Greater(int a, int b) => D[a] > D[b] || (D[a] == D[b] && I[a] > I[b]);

        public void Offer(double d, int idx, double ub)
        {
            if (Count < K)
            {
                if (!(d < ub)) return;
                var c = Count++;
                D[c] = d; I[c] = idx;
                while (c > 0)
                {
                    var p = (c - 1) >> 1;
                    if (!Greater(c, p)) break;
                    Swap(c, p); c = p;
                }
                return;
            }
            if (d > D[0] || (d == D[0] && idx >= I[0])) return;
            D[0] = d; I[0] = idx;
            var i = 0;
            while (true)
            {
                var l = 2 * i + 1;
                if (l >= Count) break;
                var r = l + 1;
                var big = r < Count && Greater(r, l) ? r : l;
                if (!Greater(big, i)) break;
                Swap(big, i); i = big;
            }
        }

        private void Swap(int a, int b)
        {
            (D[a], D[b]) = (D[b], D[a]);
            (I[a], I[b]) = (I[b], I[a]);
        }

        /// <summary>Sorts ascending in place (destroys the heap).</summary>
        public void SortAscending()
        {
            Array.Sort(D, I, 0, Count);
            // Array.Sort is unstable on equal keys; restore index order within ties.
            var s = 0;
            while (s < Count)
            {
                var e = s + 1;
                while (e < Count && D[e] == D[s]) e++;
                if (e - s > 1) Array.Sort(I, s, e - s);
                s = e;
            }
        }
    }

    private void KnnNode<TNorm>(TNorm norm, int node, ReadOnlySpan<double> x, KnnHeap heap, double epsFac, double ub)
        where TNorm : struct, ISpatialNorm
    {
        var l = _left[node];
        if (l < 0)
        {
            var m = M;
            for (var t = _start[node]; t < _end[node]; t++)
                heap.Offer(norm.Point(x, _tree.AsSpan(t * m, m)), _index[t], ub);
            return;
        }
        var r = _right[node];
        var dl = BoxMin(norm, l, x);
        var dr = BoxMin(norm, r, x);
        int near = l, far = r; double dn = dl, df = dr;
        if (dr < dl) { near = r; far = l; dn = dr; df = dl; }
        if (dn <= heap.Bound(ub) * epsFac) KnnNode(norm, near, x, heap, epsFac, ub);
        if (df <= heap.Bound(ub) * epsFac) KnnNode(norm, far, x, heap, epsFac, ub);
    }

    private void KnnOne<TNorm>(TNorm norm, ReadOnlySpan<double> x, KnnHeap heap, double epsFac, double ubPow,
        Span<double> distances, Span<int> indices) where TNorm : struct, ISpatialNorm
    {
        heap.Count = 0;
        if (N > 0 && BoxMin(norm, 0, x) <= ubPow) KnnNode(norm, 0, x, heap, epsFac, ubPow);
        heap.SortAscending();
        for (var i = 0; i < heap.K; i++)
        {
            if (i < heap.Count) { distances[i] = norm.FromPower(heap.D[i]); indices[i] = heap.I[i]; }
            else { distances[i] = double.PositiveInfinity; indices[i] = N; }
        }
    }

    private void KnnBatch<TNorm>(TNorm norm, double[] queries, int q, int k, double eps, double p,
        double upperBound, double[] dist, int[] idx, int workers) where TNorm : struct, ISpatialNorm
    {
        var m = M;
        var epsFac = EpsFactor(p, eps);
        var ubPow = norm.ToPower(upperBound);
        var w = Workers(workers);
        if (w == 1 || q < ParallelThreshold)
        {
            var heap = new KnnHeap(k);
            for (var i = 0; i < q; i++)
                KnnOne(norm, queries.AsSpan(i * m, m), heap, epsFac, ubPow, dist.AsSpan(i * k, k), idx.AsSpan(i * k, k));
            return;
        }
        Parallel.For(0, q, new ParallelOptions { MaxDegreeOfParallelism = w },
            () => new KnnHeap(k),
            (i, _, heap) =>
            {
                KnnOne(norm, queries.AsSpan(i * m, m), heap, epsFac, ubPow, dist.AsSpan(i * k, k), idx.AsSpan(i * k, k));
                return heap;
            }, _ => { });
    }

    private void KnnDispatch(double[] queries, int q, int k, double eps, double p, double ub, double[] dist, int[] idx, int workers)
    {
        CheckP(p);
        if (k < 1) throw new ArgumentException("k must be at least 1.", nameof(k));
        if (double.IsNaN(ub) || ub < 0) throw new ArgumentException("distanceUpperBound must be non-negative.");
        if (p == 2) KnnBatch(new SpatialNormL2(), queries, q, k, eps, p, ub, dist, idx, workers);
        else if (p == 1) KnnBatch(new SpatialNormL1(), queries, q, k, eps, p, ub, dist, idx, workers);
        else if (double.IsPositiveInfinity(p)) KnnBatch(new SpatialNormLInf(), queries, q, k, eps, p, ub, dist, idx, workers);
        else KnnBatch(new SpatialNormLp(p), queries, q, k, eps, p, ub, dist, idx, workers);
    }

    /// <summary>The nearest neighbour of one point.</summary>
    /// <returns>The distance and input row of the nearest point; (+inf, N) if none lies
    /// strictly within <paramref name="distanceUpperBound"/>.</returns>
    public (double Distance, int Index) Query(ReadOnlySpan<double> x, double p = 2, double distanceUpperBound = double.PositiveInfinity)
    {
        CheckPoint(x);
        var (d, i) = Query(x, 1, 0, p, distanceUpperBound);
        return (d[0], i[0]);
    }

    /// <summary>The k nearest neighbours of one point, nearest first.</summary>
    /// <param name="x">Query point of length M.</param>
    /// <param name="k">Number of neighbours.</param>
    /// <param name="eps">Approximation: the k-th returned neighbour is no further than (1+eps)
    /// times the true k-th neighbour.</param>
    /// <param name="p">Minkowski p, 1 &lt;= p &lt;= infinity.</param>
    /// <param name="distanceUpperBound">Only neighbours strictly closer than this are returned
    /// (scipy's convention); missing ones are reported as distance +inf, index N.</param>
    public (double[] Distances, int[] Indices) Query(ReadOnlySpan<double> x, int k, double eps = 0, double p = 2,
        double distanceUpperBound = double.PositiveInfinity)
    {
        CheckPoint(x);
        var dist = new double[k];
        var idx = new int[k];
        KnnDispatch(x.ToArray(), 1, k, eps, p, distanceUpperBound, dist, idx, 1);
        return (dist, idx);
    }

    /// <summary>The k nearest neighbours of every row of <paramref name="x"/>.</summary>
    /// <returns>Distances shape (q, k) and indices [q, k]. Unlike scipy, the k axis is kept
    /// when k = 1 so the result type does not depend on a value.</returns>
    /// <param name="x">Query points, shape (q, M), or a single point of shape (M).</param>
    /// <param name="k">Number of neighbours.</param>
    /// <param name="eps">Approximation factor (see the single-point overload).</param>
    /// <param name="p">Minkowski p, 1 &lt;= p &lt;= infinity.</param>
    /// <param name="distanceUpperBound">Strict upper bound on returned distances.</param>
    /// <param name="workers">Threads: -1 for all cores (default; results are identical for any
    /// value), 1 for serial. Small batches always run serially.</param>
    public (NdArray Distances, int[,] Indices) Query(NdArray x, int k = 1, double eps = 0, double p = 2,
        double distanceUpperBound = double.PositiveInfinity, int workers = -1)
    {
        var buf = QueryBuffer(x, out var q);
        var dist = new double[q * k];
        var idx = new int[q * k];
        KnnDispatch(buf, q, k, eps, p, distanceUpperBound, dist, idx, workers);
        var outIdx = new int[q, k];
        Buffer.BlockCopy(idx, 0, outIdx, 0, idx.Length * sizeof(int));
        return (new NdArray(dist, q, k), outIdx);
    }

    /// <summary>Batch query with a two-dimensional array of points.</summary>
    public (NdArray Distances, int[,] Indices) Query(double[,] x, int k = 1, double eps = 0, double p = 2,
        double distanceUpperBound = double.PositiveInfinity, int workers = -1)
        => Query(SpatialInput.ToNd(x), k, eps, p, distanceUpperBound, workers);

    /// <summary>
    /// Selected neighbour ranks for every query point, like scipy's <c>k=[1, 3]</c>: column j is
    /// the <c>ks[j]</c>-th nearest neighbour (1-based).
    /// </summary>
    public (NdArray Distances, int[,] Indices) Query(NdArray x, int[] ks, double eps = 0, double p = 2,
        double distanceUpperBound = double.PositiveInfinity, int workers = -1)
    {
        ArgumentNullException.ThrowIfNull(ks);
        if (ks.Length == 0) throw new ArgumentException("ks must not be empty.", nameof(ks));
        var kmax = 0;
        foreach (var kk in ks)
        {
            if (kk < 1) throw new ArgumentException("Neighbour ranks are 1-based and must be positive.", nameof(ks));
            kmax = Math.Max(kmax, kk);
        }
        var (d, i) = Query(x, kmax, eps, p, distanceUpperBound, workers);
        var q = d.Shape[0];
        var src = d.AsSpan();
        var outD = new double[q * ks.Length];
        var outI = new int[q, ks.Length];
        for (var r = 0; r < q; r++)
            for (var c = 0; c < ks.Length; c++)
            {
                outD[r * ks.Length + c] = src[r * kmax + ks[c] - 1];
                outI[r, c] = i[r, ks[c] - 1];
            }
        return (new NdArray(outD, q, ks.Length), outI);
    }

    // ------------------------------------------------------------------ fixed radius

    private void AddAll(int node, List<int> result)
    {
        for (var t = _start[node]; t < _end[node]; t++) result.Add(_index[t]);
    }

    private void BallNode<TNorm>(TNorm norm, int node, ReadOnlySpan<double> x, double rPow, double epsFac, List<int> result)
        where TNorm : struct, ISpatialNorm
    {
        var l = _left[node];
        if (l < 0)
        {
            var m = M;
            for (var t = _start[node]; t < _end[node]; t++)
                if (norm.Point(x, _tree.AsSpan(t * m, m)) <= rPow) result.Add(_index[t]);
            return;
        }
        for (var side = 0; side < 2; side++)
        {
            var c = side == 0 ? l : _right[node];
            if (BoxMin(norm, c, x) > rPow * epsFac) continue;
            if (BoxMax(norm, c, x) * epsFac <= rPow) { AddAll(c, result); continue; }
            BallNode(norm, c, x, rPow, epsFac, result);
        }
    }

    private long BallCountNode<TNorm>(TNorm norm, int node, ReadOnlySpan<double> x, double rPow, double epsFac)
        where TNorm : struct, ISpatialNorm
    {
        var l = _left[node];
        if (l < 0)
        {
            var m = M; long c = 0;
            for (var t = _start[node]; t < _end[node]; t++)
                if (norm.Point(x, _tree.AsSpan(t * m, m)) <= rPow) c++;
            return c;
        }
        long total = 0;
        for (var side = 0; side < 2; side++)
        {
            var c = side == 0 ? l : _right[node];
            if (BoxMin(norm, c, x) > rPow * epsFac) continue;
            if (BoxMax(norm, c, x) * epsFac <= rPow) { total += _end[c] - _start[c]; continue; }
            total += BallCountNode(norm, c, x, rPow, epsFac);
        }
        return total;
    }

    private void BallRoot<TNorm>(TNorm norm, ReadOnlySpan<double> x, double r, double eps, double p, List<int> result)
        where TNorm : struct, ISpatialNorm
    {
        if (N == 0) return;
        var rPow = norm.ToPower(r);
        var epsFac = EpsFactor(p, eps);
        if (BoxMin(norm, 0, x) > rPow * epsFac) return;
        if (BoxMax(norm, 0, x) * epsFac <= rPow) { AddAll(0, result); return; }
        BallNode(norm, 0, x, rPow, epsFac, result);
    }

    private long BallCountRoot<TNorm>(TNorm norm, ReadOnlySpan<double> x, double r, double eps, double p)
        where TNorm : struct, ISpatialNorm
    {
        if (N == 0) return 0;
        var rPow = norm.ToPower(r);
        var epsFac = EpsFactor(p, eps);
        if (BoxMin(norm, 0, x) > rPow * epsFac) return 0;
        if (BoxMax(norm, 0, x) * epsFac <= rPow) return N;
        return BallCountNode(norm, 0, x, rPow, epsFac);
    }

    private void Ball(ReadOnlySpan<double> x, double r, double p, double eps, List<int> result)
    {
        if (p == 2) BallRoot(new SpatialNormL2(), x, r, eps, p, result);
        else if (p == 1) BallRoot(new SpatialNormL1(), x, r, eps, p, result);
        else if (double.IsPositiveInfinity(p)) BallRoot(new SpatialNormLInf(), x, r, eps, p, result);
        else BallRoot(new SpatialNormLp(p), x, r, eps, p, result);
    }

    private long BallCount(ReadOnlySpan<double> x, double r, double p, double eps)
    {
        if (p == 2) return BallCountRoot(new SpatialNormL2(), x, r, eps, p);
        if (p == 1) return BallCountRoot(new SpatialNormL1(), x, r, eps, p);
        if (double.IsPositiveInfinity(p)) return BallCountRoot(new SpatialNormLInf(), x, r, eps, p);
        return BallCountRoot(new SpatialNormLp(p), x, r, eps, p);
    }

    private static void CheckRadius(double r)
    {
        if (double.IsNaN(r) || r < 0) throw new ArgumentException("The radius must be non-negative.", nameof(r));
    }

    /// <summary>All input rows within distance <paramref name="r"/> (inclusive) of one point.</summary>
    /// <param name="x">Query point of length M.</param>
    /// <param name="r">Radius.</param>
    /// <param name="p">Minkowski p.</param>
    /// <param name="eps">Approximation: branches whose nearest point is further than r/(1+eps)
    /// are skipped and those whose furthest point is within r(1+eps) are taken whole.</param>
    /// <param name="returnSorted">Sort the indices. scipy's default (null) leaves a single-point
    /// result in traversal order; that order is implementation-defined and differs from scipy's.</param>
    public int[] QueryBallPoint(ReadOnlySpan<double> x, double r, double p = 2, double eps = 0, bool? returnSorted = null)
    {
        CheckPoint(x); CheckP(p); CheckRadius(r);
        var list = new List<int>();
        Ball(x, r, p, eps, list);
        var result = list.ToArray();
        if (returnSorted == true) Array.Sort(result);
        return result;
    }

    /// <summary>Ball query for each row of <paramref name="x"/> with a common radius.</summary>
    /// <param name="x">Query points (q, M).</param>
    /// <param name="r">Radius.</param>
    /// <param name="p">Minkowski p.</param>
    /// <param name="eps">Approximation factor.</param>
    /// <param name="returnSorted">null or true sorts each list (scipy sorts multi-point results
    /// by default).</param>
    /// <param name="workers">Threads, -1 for all cores.</param>
    public int[][] QueryBallPoint(NdArray x, double r, double p = 2, double eps = 0, bool? returnSorted = null, int workers = -1)
    {
        var buf = QueryBuffer(x, out var q);
        var radii = new double[q];
        Array.Fill(radii, r);
        return BallMany(buf, q, radii, p, eps, returnSorted != false, workers);
    }

    /// <summary>Ball query with one radius per query row.</summary>
    public int[][] QueryBallPoint(NdArray x, double[] r, double p = 2, double eps = 0, bool? returnSorted = null, int workers = -1)
    {
        ArgumentNullException.ThrowIfNull(r);
        var buf = QueryBuffer(x, out var q);
        if (r.Length != q) throw new ArgumentException("r must have one radius per query point.", nameof(r));
        return BallMany(buf, q, r, p, eps, returnSorted != false, workers);
    }

    private int[][] BallMany(double[] buf, int q, double[] radii, double p, double eps, bool sort, int workers)
    {
        CheckP(p);
        foreach (var rr in radii) CheckRadius(rr);
        var m = M;
        var result = new int[q][];
        var w = Workers(workers);
        if (w == 1 || q < ParallelThreshold)
        {
            var list = new List<int>();
            for (var i = 0; i < q; i++) Body(i, list);
        }
        else
        {
            Parallel.For(0, q, new ParallelOptions { MaxDegreeOfParallelism = w }, () => new List<int>(),
                (i, _, list) => { Body(i, list); return list; }, _ => { });
        }
        return result;

        void Body(int i, List<int> list)
        {
            list.Clear();
            Ball(buf.AsSpan(i * m, m), radii[i], p, eps, list);
            var a = list.ToArray();
            if (sort) Array.Sort(a);
            result[i] = a;
        }
    }

    /// <summary>Number of rows within <paramref name="r"/> of one point (scipy's <c>return_length=True</c>).</summary>
    public long QueryBallPointLength(ReadOnlySpan<double> x, double r, double p = 2, double eps = 0)
    {
        CheckPoint(x); CheckP(p); CheckRadius(r);
        return BallCount(x, r, p, eps);
    }

    /// <summary>Neighbour counts within <paramref name="r"/> for each query row.</summary>
    public long[] QueryBallPointLength(NdArray x, double r, double p = 2, double eps = 0, int workers = -1)
    {
        CheckP(p); CheckRadius(r);
        var buf = QueryBuffer(x, out var q);
        var m = M;
        var result = new long[q];
        var w = Workers(workers);
        if (w == 1 || q < ParallelThreshold)
            for (var i = 0; i < q; i++) result[i] = BallCount(buf.AsSpan(i * m, m), r, p, eps);
        else
            Parallel.For(0, q, new ParallelOptions { MaxDegreeOfParallelism = w },
                i => result[i] = BallCount(buf.AsSpan(i * m, m), r, p, eps));
        return result;
    }

    /// <summary>
    /// For each point of this tree, the sorted rows of <paramref name="other"/> within
    /// <paramref name="r"/> (scipy's <c>query_ball_tree</c>).
    /// </summary>
    public int[][] QueryBallTree(KDTree other, double r, double p = 2, double eps = 0)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (other.M != M) throw new ArgumentException("Both trees must have the same dimension.", nameof(other));
        var radii = new double[N];
        Array.Fill(radii, r);
        return other.BallMany(_data, N, radii, p, eps, true, -1);
    }

    /// <summary>All pairs (i, j), i &lt; j, of points of this tree within distance r.</summary>
    public HashSet<(int I, int J)> QueryPairs(double r, double p = 2, double eps = 0)
    {
        var set = new HashSet<(int, int)>();
        var arr = QueryPairsArray(r, p, eps);
        for (var i = 0; i < arr.GetLength(0); i++) set.Add((arr[i, 0], arr[i, 1]));
        return set;
    }

    /// <summary>The pairs of <see cref="QueryPairs"/> as an [npairs, 2] array sorted by (i, j)
    /// (scipy's <c>output_type='ndarray'</c>).</summary>
    public int[,] QueryPairsArray(double r, double p = 2, double eps = 0)
    {
        var radii = new double[N];
        Array.Fill(radii, r);
        var lists = BallMany(_data, N, radii, p, eps, true, -1);
        var count = 0;
        for (var i = 0; i < N; i++)
            foreach (var j in lists[i]) if (j > i) count++;
        var result = new int[count, 2];
        var k = 0;
        for (var i = 0; i < N; i++)
            foreach (var j in lists[i])
                if (j > i) { result[k, 0] = i; result[k, 1] = j; k++; }
        return result;
    }

    /// <summary>
    /// The pairs (i in this, j in other) within <paramref name="maxDistance"/> and their
    /// distances, sorted by (i, j) — scipy's <c>sparse_distance_matrix</c> as a coordinate list.
    /// Zero distances are kept (a dok_matrix would drop them).
    /// </summary>
    public List<(int I, int J, double Distance)> SparseDistanceMatrix(KDTree other, double maxDistance, double p = 2)
    {
        ArgumentNullException.ThrowIfNull(other);
        var lists = QueryBallTree(other, maxDistance, p);
        var result = new List<(int, int, double)>();
        var m = M;
        for (var i = 0; i < N; i++)
        {
            var a = _data.AsSpan(i * m, m);
            foreach (var j in lists[i])
                result.Add((i, j, Minkowski(a, other._data.AsSpan(j * m, m), p)));
        }
        return result;
    }

    /// <summary>The pairs of <see cref="SparseDistanceMatrix"/> as a dictionary (scipy's <c>output_type='dict'</c>).</summary>
    public Dictionary<(int I, int J), double> SparseDistanceMatrixDictionary(KDTree other, double maxDistance, double p = 2)
    {
        var d = new Dictionary<(int, int), double>();
        foreach (var (i, j, v) in SparseDistanceMatrix(other, maxDistance, p)) d[(i, j)] = v;
        return d;
    }

    internal static double Minkowski(ReadOnlySpan<double> a, ReadOnlySpan<double> b, double p)
    {
        if (p == 2) return Math.Sqrt(new SpatialNormL2().Point(a, b));
        if (p == 1) return new SpatialNormL1().Point(a, b);
        if (double.IsPositiveInfinity(p)) return new SpatialNormLInf().Point(a, b);
        var n = new SpatialNormLp(p);
        return n.FromPower(n.Point(a, b));
    }

    // ------------------------------------------------------------------ count_neighbors

    /// <summary>Number of pairs (x in this, y in other) with distance(x, y) &lt;= r.</summary>
    public long CountNeighbors(KDTree other, double r, double p = 2) => CountNeighbors(other, [r], p)[0];

    /// <summary>
    /// Pair counts for several radii (scipy's <c>count_neighbors</c>, a two-point correlation).
    /// </summary>
    /// <param name="other">Tree whose points are paired with this tree's.</param>
    /// <param name="r">Radii, any order.</param>
    /// <param name="p">Minkowski p.</param>
    /// <param name="cumulative">True: element i counts pairs with d &lt;= r[i]. False: pairs
    /// in the bin (r[i-1], r[i]] of the radii as given, the first bin being d &lt;= r[0] —
    /// scipy's semantics, which assume r is increasing in that mode.</param>
    /// <remarks>Each query point walks the other tree once; a whole node is credited to every
    /// radius beyond its furthest corner and skipped for every radius short of its nearest, so
    /// only the radii that actually cut a node cost per-point work.</remarks>
    public long[] CountNeighbors(KDTree other, double[] r, double p = 2, bool cumulative = true)
    {
        ArgumentNullException.ThrowIfNull(other);
        ArgumentNullException.ThrowIfNull(r);
        CheckP(p);
        if (other.M != M) throw new ArgumentException("Both trees must have the same dimension.", nameof(other));
        foreach (var rr in r) if (double.IsNaN(rr)) throw new ArgumentException("Radii must not be NaN.", nameof(r));
        var nr = r.Length;
        var order = new int[nr];
        for (var i = 0; i < nr; i++) order[i] = i;
        var sorted = (double[])r.Clone();
        Array.Sort(sorted, order);

        long[] diff;
        if (p == 2) diff = other.CountAll(new SpatialNormL2(), _data, N, sorted);
        else if (p == 1) diff = other.CountAll(new SpatialNormL1(), _data, N, sorted);
        else if (double.IsPositiveInfinity(p)) diff = other.CountAll(new SpatialNormLInf(), _data, N, sorted);
        else diff = other.CountAll(new SpatialNormLp(p), _data, N, sorted);

        var cum = new long[nr];
        long run = 0;
        for (var i = 0; i < nr; i++) { run += diff[i]; cum[i] = run; }
        var result = new long[nr];
        for (var i = 0; i < nr; i++) result[order[i]] = cum[i];
        if (!cumulative)
        {
            var bins = new long[nr];
            for (var i = 0; i < nr; i++) bins[i] = i == 0 ? result[0] : result[i] - result[i - 1];
            return bins;
        }
        return result;
    }

    private long[] CountAll<TNorm>(TNorm norm, double[] queries, int q, double[] sortedR) where TNorm : struct, ISpatialNorm
    {
        var nr = sortedR.Length;
        var rp = new double[nr];
        for (var i = 0; i < nr; i++) rp[i] = sortedR[i] < 0 ? -1 : norm.ToPower(sortedR[i]);
        var total = new long[nr + 1];
        if (N == 0 || q == 0) return total;
        var m = M;
        var gate = new object();
        Parallel.For(0, q, () => new long[nr + 1],
            (i, _, diff) => { CountNode(norm, 0, queries.AsSpan(i * m, m), rp, 0, nr, diff); return diff; },
            diff => { lock (gate) for (var j = 0; j <= nr; j++) total[j] += diff[j]; });
        return total;
    }

    private void CountNode<TNorm>(TNorm norm, int node, ReadOnlySpan<double> x, double[] rp, int a, int b, long[] diff)
        where TNorm : struct, ISpatialNorm
    {
        var dmin = BoxMin(norm, node, x);
        var dmax = BoxMax(norm, node, x);
        while (a < b && rp[a] < dmin) a++;
        var bb = a;
        while (bb < b && rp[bb] < dmax) bb++;
        if (bb < b) { long size = _end[node] - _start[node]; diff[bb] += size; diff[b] -= size; }
        if (a >= bb) return;
        var l = _left[node];
        if (l < 0)
        {
            var m = M;
            for (var t = _start[node]; t < _end[node]; t++)
            {
                var d = norm.Point(x, _tree.AsSpan(t * m, m));
                var j = a;
                while (j < bb && rp[j] < d) j++;
                if (j < bb) { diff[j]++; diff[bb]--; }
            }
            return;
        }
        CountNode(norm, l, x, rp, a, bb, diff);
        CountNode(norm, _right[node], x, rp, a, bb, diff);
    }
}
