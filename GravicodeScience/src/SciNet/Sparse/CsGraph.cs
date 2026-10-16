using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet.Sparse;

/// <summary>Algorithm for <see cref="CsGraph.ShortestPath"/>.</summary>
public enum ShortestPathMethod
{
    /// <summary>Johnson when there are negative weights and the graph is sparse, Dijkstra when sparse, Floyd–Warshall when dense.</summary>
    Auto,

    /// <summary>All pairs by dynamic programming, O(n³); handles negative weights.</summary>
    FloydWarshall,

    /// <summary>Binary-heap Dijkstra from each source; non-negative weights only.</summary>
    Dijkstra,

    /// <summary>Bellman–Ford from each source; handles negative weights, O(n·E) per source.</summary>
    BellmanFord,

    /// <summary>Bellman–Ford reweighting then Dijkstra; best for sparse graphs with negative weights.</summary>
    Johnson,
}

/// <summary>Connectivity notion for directed graphs in <see cref="CsGraph.ConnectedComponents"/>.</summary>
public enum ComponentConnection
{
    /// <summary>Connected ignoring edge direction.</summary>
    Weak,

    /// <summary>Mutually reachable.</summary>
    Strong,
}

/// <summary>Algorithm for <see cref="CsGraph.MaximumFlow"/>.</summary>
public enum MaximumFlowMethod
{
    /// <summary>Dinic's blocking-flow algorithm (SciPy's default), O(V²E).</summary>
    Dinic,

    /// <summary>Edmonds–Karp shortest augmenting paths, O(VE²).</summary>
    EdmondsKarp,
}

/// <summary>Which side <see cref="CsGraph.MaximumBipartiteMatching"/> reports (SciPy's <c>perm_type</c>).</summary>
public enum MatchingPermutation
{
    /// <summary>One entry per column: the row matched to it (SciPy's default <c>'row'</c>).</summary>
    Row,

    /// <summary>One entry per row: the column matched to it.</summary>
    Column,
}

/// <summary>Shortest-path distances and, if requested, predecessors.</summary>
/// <param name="Distances">(sources × n) distances; unreachable is +∞.</param>
/// <param name="Predecessors">(sources × n) predecessor of each node on its shortest path, <see cref="CsGraph.NoPredecessor"/> where none.</param>
public sealed record ShortestPathResult(NdArray Distances, int[,]? Predecessors);

/// <summary>Maximum flow value and the flow itself.</summary>
/// <param name="FlowValue">Total flow from source to sink.</param>
/// <param name="Flow">Antisymmetric net flow: <c>Flow[u, v] = −Flow[v, u]</c>, as SciPy returns it.</param>
public sealed record MaximumFlowResult(double FlowValue, CsrMatrix Flow);

/// <summary>A graph Laplacian with its diagonal information.</summary>
/// <param name="Laplacian">The Laplacian matrix.</param>
/// <param name="Diagonal">Degrees (unnormalised) or square-rooted degrees with zeros replaced by 1 (normalised), as SciPy's <c>return_diag</c>.</param>
public sealed record LaplacianResult(CsrMatrix Laplacian, double[] Diagonal);

/// <summary>Thrown when a shortest-path computation meets a negative cycle (SciPy's <c>NegativeCycleError</c>).</summary>
public sealed class NegativeCycleException(string message) : Exception(message);

/// <summary>
/// Compressed sparse graph algorithms — <c>scipy.sparse.csgraph</c>. A graph is an n×n sparse
/// matrix whose stored entry <c>(i, j)</c> is an edge i→j with that weight.
/// </summary>
/// <remarks>
/// As in SciPy, <em>explicitly stored zeros are edges of weight zero</em>; use
/// <see cref="CsgraphFromDense"/> to turn a dense matrix with a "no edge" value into a graph. With
/// <c>directed: false</c> every edge can be walked both ways and, where both <c>(i, j)</c> and
/// <c>(j, i)</c> are stored, the smaller weight is used. Everything is iterative: no algorithm here
/// recurses, so path graphs with millions of nodes do not overflow the stack.
/// </remarks>
public static class CsGraph
{
    /// <summary>SciPy's marker for "no predecessor".</summary>
    public const int NoPredecessor = -9999;

    // ------------------------------------------------------------------ graph preparation

    private sealed class Adjacency(int n, int[] ptr, int[] idx, double[] w)
    {
        public readonly int N = n;
        public readonly int[] Ptr = ptr;
        public readonly int[] Idx = idx;
        public readonly double[] W = w;
    }

    private static CsrMatrix Validate(SparseMatrixBase graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        if (graph.Rows != graph.Columns) throw new ArgumentException($"A graph must be square, got {graph.Rows}x{graph.Columns}.");
        return graph.ToCsr().SumDuplicates();
    }

    /// <summary>Out-edges; for undirected graphs the union with the reversed edges, minimum weight on duplicates.</summary>
    private static Adjacency Build(CsrMatrix g, bool directed, bool unweighted)
    {
        var n = g.Rows;
        if (directed)
        {
            var w = g.Data.ToArray();
            if (unweighted) Array.Fill(w, 1.0);
            return new Adjacency(n, g.IndptrArray, g.IndicesArray, w);
        }
        var nnz = g.Nnz;
        var rows = new int[2 * nnz]; var cols = new int[2 * nnz]; var vals = new double[2 * nnz];
        var k = 0;
        foreach (var (i, j, v) in g.EnumerateEntries())
        {
            rows[k] = i; cols[k] = j; vals[k++] = v;
            rows[k] = j; cols[k] = i; vals[k++] = v;
        }
        var ptr = new int[n + 1];
        for (var t = 0; t < k; t++) ptr[rows[t] + 1]++;
        for (var i = 0; i < n; i++) ptr[i + 1] += ptr[i];
        var next = (int[])ptr.Clone();
        var idx = new int[k]; var wv = new double[k];
        for (var t = 0; t < k; t++) { var d = next[rows[t]]++; idx[d] = cols[t]; wv[d] = vals[t]; }
        // Sort each row and keep the minimum weight among duplicates.
        var outPtr = new int[n + 1];
        var write = 0;
        for (var i = 0; i < n; i++)
        {
            Array.Sort(idx, wv, ptr[i], ptr[i + 1] - ptr[i]);
            var p = ptr[i];
            while (p < ptr[i + 1])
            {
                var j = idx[p];
                var best = wv[p++];
                while (p < ptr[i + 1] && idx[p] == j) best = Math.Min(best, wv[p++]);
                idx[write] = j;
                wv[write++] = best;
            }
            outPtr[i + 1] = write;
        }
        Array.Resize(ref idx, write);
        Array.Resize(ref wv, write);
        if (unweighted) Array.Fill(wv, 1.0);
        return new Adjacency(n, outPtr, idx, wv);
    }

    private static int[] Sources(int n, int[]? indices)
    {
        if (indices is null) return Enumerable.Range(0, n).ToArray();
        foreach (var s in indices) if ((uint)s >= (uint)n) throw new ArgumentException($"Index {s} out of range for {n} nodes.");
        return indices;
    }

    // ------------------------------------------------------------------ shortest paths

    /// <summary>
    /// Shortest-path distances between nodes — <c>scipy.sparse.csgraph.shortest_path</c>.
    /// </summary>
    /// <param name="graph">Square weighted adjacency matrix.</param>
    /// <param name="method">Algorithm (see <see cref="ShortestPathMethod"/>).</param>
    /// <param name="directed">Respect edge direction.</param>
    /// <param name="unweighted">Treat every edge as weight 1 (hop counts).</param>
    /// <param name="indices">Sources (default: all nodes).</param>
    /// <param name="returnPredecessors">Also return the predecessor matrix.</param>
    public static ShortestPathResult ShortestPath(SparseMatrixBase graph, ShortestPathMethod method = ShortestPathMethod.Auto,
        bool directed = true, bool unweighted = false, int[]? indices = null, bool returnPredecessors = false)
    {
        var g = Validate(graph);
        var n = g.Rows;
        if (method == ShortestPathMethod.Auto)
        {
            var negative = !unweighted && g.Data.ToArray().Any(v => v < 0);
            if (g.Nnz < (long)n * n / 4) method = negative ? ShortestPathMethod.Johnson : ShortestPathMethod.Dijkstra;
            else method = ShortestPathMethod.FloydWarshall;
        }
        return method switch
        {
            ShortestPathMethod.FloydWarshall => FloydWarshall(g, directed, unweighted, returnPredecessors, indices),
            ShortestPathMethod.Dijkstra => Dijkstra(g, directed, indices, returnPredecessors, unweighted),
            ShortestPathMethod.BellmanFord => BellmanFord(g, directed, indices, returnPredecessors, unweighted),
            _ => Johnson(g, directed, indices, returnPredecessors, unweighted),
        };
    }

    /// <summary>
    /// All-pairs shortest paths by Floyd–Warshall — <c>floyd_warshall</c>. Rows are relaxed in
    /// parallel for each pivot (row k does not change during step k, so this is race-free).
    /// Throws <see cref="NegativeCycleException"/> if a negative cycle exists.
    /// </summary>
    public static ShortestPathResult FloydWarshall(SparseMatrixBase graph, bool directed = true, bool unweighted = false,
        bool returnPredecessors = false, int[]? indices = null)
    {
        var g = Validate(graph);
        var adj = Build(g, directed, unweighted);
        var n = adj.N;
        var d = new double[n * n];
        Array.Fill(d, double.PositiveInfinity);
        var pred = new int[n * n];
        Array.Fill(pred, NoPredecessor);
        for (var i = 0; i < n; i++)
            for (var p = adj.Ptr[i]; p < adj.Ptr[i + 1]; p++)
            {
                var j = adj.Idx[p];
                if (i == j) { if (adj.W[p] < 0) throw new NegativeCycleException($"Negative self-loop at node {i}."); continue; }
                if (adj.W[p] < d[i * n + j]) { d[i * n + j] = adj.W[p]; pred[i * n + j] = i; }
            }
        for (var i = 0; i < n; i++) d[i * n + i] = 0;
        for (var k = 0; k < n; k++)
        {
            var kk = k;
            void Row(int i)
            {
                var dik = d[i * n + kk];
                if (double.IsPositiveInfinity(dik) || i == kk) return;
                var rowI = i * n;
                var rowK = kk * n;
                for (var j = 0; j < n; j++)
                {
                    var cand = dik + d[rowK + j];
                    if (cand < d[rowI + j]) { d[rowI + j] = cand; pred[rowI + j] = pred[rowK + j]; }
                }
            }
            if (n >= 192) Parallel.For(0, n, Row);
            else for (var i = 0; i < n; i++) Row(i);
        }
        for (var i = 0; i < n; i++) if (d[i * n + i] < 0) throw new NegativeCycleException("Negative cycle detected.");
        var src = Sources(n, indices);
        var outD = new double[src.Length * n];
        int[,]? outP = returnPredecessors ? new int[src.Length, n] : null;
        for (var s = 0; s < src.Length; s++)
        {
            Array.Copy(d, src[s] * n, outD, s * n, n);
            if (outP is not null) for (var j = 0; j < n; j++) outP[s, j] = pred[src[s] * n + j];
        }
        return new ShortestPathResult(new NdArray(outD, src.Length, n), outP);
    }

    /// <summary>
    /// Dijkstra's algorithm from each source — <c>dijkstra</c>. Sources run in parallel.
    /// </summary>
    /// <param name="graph">Square adjacency matrix with non-negative weights.</param>
    /// <param name="directed">Respect edge direction.</param>
    /// <param name="indices">Sources (default: all nodes).</param>
    /// <param name="returnPredecessors">Also return predecessors.</param>
    /// <param name="unweighted">Treat every edge as weight 1.</param>
    /// <param name="limit">Do not explore paths longer than this; farther nodes report +∞.</param>
    public static ShortestPathResult Dijkstra(SparseMatrixBase graph, bool directed = true, int[]? indices = null,
        bool returnPredecessors = false, bool unweighted = false, double limit = double.PositiveInfinity)
    {
        var g = Validate(graph);
        var adj = Build(g, directed, unweighted);
        foreach (var w in adj.W) if (w < 0) throw new ArgumentException("Dijkstra needs non-negative weights; use BellmanFord or Johnson.");
        return RunDijkstra(adj, Sources(adj.N, indices), returnPredecessors, limit, null);
    }

    /// <summary>
    /// Multi-source Dijkstra (<c>dijkstra(..., min_only=True)</c>): the distance from each node to
    /// the nearest of the sources, the predecessor on that path, and which source it was.
    /// </summary>
    public static (double[] Distances, int[] Predecessors, int[] Sources) DijkstraMinOnly(SparseMatrixBase graph, int[] indices,
        bool directed = true, bool unweighted = false, double limit = double.PositiveInfinity)
    {
        var g = Validate(graph);
        var adj = Build(g, directed, unweighted);
        foreach (var w in adj.W) if (w < 0) throw new ArgumentException("Dijkstra needs non-negative weights.");
        var n = adj.N;
        var dist = new double[n];
        var pred = new int[n];
        var source = new int[n];
        SingleDijkstra(adj, Sources(n, indices), dist, pred, source, limit);
        return (dist, pred, source);
    }

    private static ShortestPathResult RunDijkstra(Adjacency adj, int[] src, bool returnPredecessors, double limit, double[]? potential)
    {
        var n = adj.N;
        var outD = new double[src.Length * n];
        int[,]? outP = returnPredecessors ? new int[src.Length, n] : null;
        void One(int s)
        {
            var dist = new double[n];
            var pred = new int[n];
            SingleDijkstra(adj, [src[s]], dist, pred, null, limit);
            if (potential is not null)
                for (var j = 0; j < n; j++)
                    if (!double.IsPositiveInfinity(dist[j])) dist[j] = dist[j] - potential[src[s]] + potential[j];
            Array.Copy(dist, 0, outD, s * n, n);
            if (outP is not null) lock (outP) for (var j = 0; j < n; j++) outP[s, j] = pred[j];
        }
        if (src.Length > 4 && (long)src.Length * adj.Idx.Length > 20_000) Parallel.For(0, src.Length, One);
        else for (var s = 0; s < src.Length; s++) One(s);
        return new ShortestPathResult(new NdArray(outD, src.Length, n), outP);
    }

    private static void SingleDijkstra(Adjacency adj, int[] sources, double[] dist, int[] pred, int[]? origin, double limit)
    {
        Array.Fill(dist, double.PositiveInfinity);
        Array.Fill(pred, NoPredecessor);
        if (origin is not null) Array.Fill(origin, NoPredecessor);
        var done = new bool[adj.N];
        var pq = new PriorityQueue<int, double>();
        foreach (var s in sources)
        {
            dist[s] = 0;
            if (origin is not null) origin[s] = s;
            pq.Enqueue(s, 0);
        }
        while (pq.TryDequeue(out var u, out var du))
        {
            if (done[u] || du > dist[u]) continue;
            done[u] = true;
            for (var p = adj.Ptr[u]; p < adj.Ptr[u + 1]; p++)
            {
                var v = adj.Idx[p];
                var nd = du + adj.W[p];
                if (nd > limit) continue;
                if (nd < dist[v])
                {
                    dist[v] = nd;
                    pred[v] = u;
                    if (origin is not null) origin[v] = origin[u];
                    pq.Enqueue(v, nd);
                }
            }
        }
    }

    /// <summary>
    /// Bellman–Ford from each source — <c>bellman_ford</c>. Throws <see cref="NegativeCycleException"/>
    /// when a negative cycle is reachable (for an undirected graph any negative edge is one).
    /// </summary>
    public static ShortestPathResult BellmanFord(SparseMatrixBase graph, bool directed = true, int[]? indices = null,
        bool returnPredecessors = false, bool unweighted = false)
    {
        var g = Validate(graph);
        var adj = Build(g, directed, unweighted);
        var n = adj.N;
        var src = Sources(n, indices);
        var outD = new double[src.Length * n];
        int[,]? outP = returnPredecessors ? new int[src.Length, n] : null;
        var dist = new double[n];
        var pred = new int[n];
        for (var s = 0; s < src.Length; s++)
        {
            Array.Fill(dist, double.PositiveInfinity);
            Array.Fill(pred, NoPredecessor);
            dist[src[s]] = 0;
            for (var round = 0; round < n; round++)
            {
                var changed = false;
                for (var u = 0; u < n; u++)
                {
                    var du = dist[u];
                    if (double.IsPositiveInfinity(du)) continue;
                    for (var p = adj.Ptr[u]; p < adj.Ptr[u + 1]; p++)
                    {
                        var v = adj.Idx[p];
                        var nd = du + adj.W[p];
                        if (nd < dist[v]) { dist[v] = nd; pred[v] = u; changed = true; }
                    }
                }
                if (!changed) break;
                if (round == n - 1) throw new NegativeCycleException("Negative cycle detected.");
            }
            Array.Copy(dist, 0, outD, s * n, n);
            if (outP is not null) for (var j = 0; j < n; j++) outP[s, j] = pred[j];
        }
        return new ShortestPathResult(new NdArray(outD, src.Length, n), outP);
    }

    /// <summary>
    /// Johnson's algorithm — <c>johnson</c>: a Bellman–Ford pass from a virtual node gives
    /// potentials <c>h</c> that make every reweighted edge <c>w + h(u) − h(v)</c> non-negative, then
    /// Dijkstra runs from each source. Throws <see cref="NegativeCycleException"/> on a negative cycle.
    /// </summary>
    public static ShortestPathResult Johnson(SparseMatrixBase graph, bool directed = true, int[]? indices = null,
        bool returnPredecessors = false, bool unweighted = false)
    {
        var g = Validate(graph);
        var adj = Build(g, directed, unweighted);
        var n = adj.N;
        if (!directed) foreach (var w in adj.W) if (w < 0) throw new NegativeCycleException("Negative edge in an undirected graph is a negative cycle.");
        var h = new double[n];
        for (var round = 0; ; round++)
        {
            var changed = false;
            for (var u = 0; u < n; u++)
                for (var p = adj.Ptr[u]; p < adj.Ptr[u + 1]; p++)
                {
                    var v = adj.Idx[p];
                    var nd = h[u] + adj.W[p];
                    if (nd < h[v]) { h[v] = nd; changed = true; }
                }
            if (!changed) break;
            if (round >= n) throw new NegativeCycleException("Negative cycle detected.");
        }
        var rw = new double[adj.W.Length];
        for (var u = 0; u < n; u++)
            for (var p = adj.Ptr[u]; p < adj.Ptr[u + 1]; p++)
                rw[p] = Math.Max(0.0, adj.W[p] + h[u] - h[adj.Idx[p]]);
        var re = new Adjacency(n, adj.Ptr, adj.Idx, rw);
        return RunDijkstra(re, Sources(n, indices), returnPredecessors, double.PositiveInfinity, h);
    }

    /// <summary>
    /// The <paramref name="k"/> shortest loopless paths from <paramref name="source"/> to
    /// <paramref name="sink"/> by Yen's algorithm (SciPy's <c>yen</c>), shortest first.
    /// Non-negative weights only. Fewer than k are returned when fewer exist.
    /// </summary>
    public static (double[] Distances, int[][] Paths) YenKShortest(SparseMatrixBase graph, int source, int sink, int k,
        bool directed = true, bool unweighted = false)
    {
        var g = Validate(graph);
        var adj = Build(g, directed, unweighted);
        var n = adj.N;
        if ((uint)source >= (uint)n || (uint)sink >= (uint)n) throw new ArgumentException("source/sink out of range.");
        foreach (var w in adj.W) if (w < 0) throw new ArgumentException("YenKShortest needs non-negative weights.");
        (double Cost, int[] Path)? Shortest(int s, HashSet<long> bannedEdges, bool[] bannedNodes)
        {
            var dist = new double[n]; var pred = new int[n];
            Array.Fill(dist, double.PositiveInfinity); Array.Fill(pred, -1);
            var pq = new PriorityQueue<int, double>();
            dist[s] = 0; pq.Enqueue(s, 0);
            while (pq.TryDequeue(out var u, out var du))
            {
                if (du > dist[u]) continue;
                if (u == sink) break;
                for (var p = adj.Ptr[u]; p < adj.Ptr[u + 1]; p++)
                {
                    var v = adj.Idx[p];
                    if (bannedNodes[v] || bannedEdges.Contains((long)u * n + v)) continue;
                    var nd = du + adj.W[p];
                    if (nd < dist[v]) { dist[v] = nd; pred[v] = u; pq.Enqueue(v, nd); }
                }
            }
            if (double.IsPositiveInfinity(dist[sink])) return null;
            var path = new List<int>();
            for (var v = sink; v != -1; v = pred[v]) path.Add(v);
            path.Reverse();
            return (dist[sink], path.ToArray());
        }
        double EdgeWeight(int u, int v)
        {
            var best = double.PositiveInfinity;
            for (var p = adj.Ptr[u]; p < adj.Ptr[u + 1]; p++) if (adj.Idx[p] == v) best = Math.Min(best, adj.W[p]);
            return best;
        }
        var found = new List<(double, int[])>();
        var first = Shortest(source, [], new bool[n]);
        if (first is null || k <= 0) return ([], []);
        found.Add(first.Value);
        var candidates = new List<(double Cost, int[] Path)>();
        while (found.Count < k)
        {
            var last = found[^1].Item2;
            for (var i = 0; i < last.Length - 1; i++)
            {
                var spur = last[i];
                var root = last.Take(i + 1).ToArray();
                var bannedEdges = new HashSet<long>();
                foreach (var (_, p) in found)
                    if (p.Length > i + 1 && p.Take(i + 1).SequenceEqual(root)) bannedEdges.Add((long)p[i] * n + p[i + 1]);
                var bannedNodes = new bool[n];
                for (var t = 0; t < i; t++) bannedNodes[root[t]] = true;
                var spurPath = Shortest(spur, bannedEdges, bannedNodes);
                if (spurPath is null) continue;
                var rootCost = 0.0;
                for (var t = 0; t < i; t++) rootCost += EdgeWeight(root[t], root[t + 1]);
                var total = root.Take(i).Concat(spurPath.Value.Path).ToArray();
                if (!candidates.Any(c => c.Path.SequenceEqual(total)) && !found.Any(f => f.Item2.SequenceEqual(total)))
                    candidates.Add((rootCost + spurPath.Value.Cost, total));
            }
            if (candidates.Count == 0) break;
            var bestIdx = 0;
            for (var c = 1; c < candidates.Count; c++)
                if (candidates[c].Cost < candidates[bestIdx].Cost) bestIdx = c;
            found.Add(candidates[bestIdx]);
            candidates.RemoveAt(bestIdx);
        }
        return (found.Select(f => f.Item1).ToArray(), found.Select(f => f.Item2).ToArray());
    }

    // ------------------------------------------------------------------ traversal

    private static (Adjacency Out, Adjacency? In) TraversalGraph(CsrMatrix g, bool directed)
    {
        var outAdj = new Adjacency(g.Rows, g.IndptrArray, g.IndicesArray, g.DataArray);
        if (directed) return (outAdj, null);
        var t = g.ToCsc().Inner; // Gᵀ as CSR: in-edges, sorted
        return (outAdj, new Adjacency(t.Rows, t.IndptrArray, t.IndicesArray, t.DataArray));
    }

    /// <summary>
    /// Breadth-first order from <paramref name="start"/> — <c>breadth_first_order</c>. Neighbours are
    /// visited in stored order (for undirected graphs, out-edges then in-edges), matching SciPy.
    /// </summary>
    public static (int[] Order, int[] Predecessors) BreadthFirstOrder(SparseMatrixBase graph, int start, bool directed = true)
    {
        var g = Validate(graph);
        var n = g.Rows;
        if ((uint)start >= (uint)n) throw new ArgumentException("start out of range.");
        var (o, i) = TraversalGraph(g, directed);
        var pred = new int[n];
        Array.Fill(pred, NoPredecessor);
        var seen = new bool[n];
        var order = new int[n];
        var head = 0;
        var tail = 0;
        order[tail++] = start;
        seen[start] = true;
        while (head < tail)
        {
            var u = order[head++];
            foreach (var adj in i is null ? [o] : new[] { o, i })
                for (var p = adj.Ptr[u]; p < adj.Ptr[u + 1]; p++)
                {
                    var v = adj.Idx[p];
                    if (seen[v]) continue;
                    seen[v] = true;
                    pred[v] = u;
                    order[tail++] = v;
                }
        }
        return (order[..tail], pred);
    }

    /// <summary>
    /// Depth-first (pre-)order from <paramref name="start"/> — <c>depth_first_order</c>, with an
    /// explicit stack and SciPy's neighbour order.
    /// </summary>
    public static (int[] Order, int[] Predecessors) DepthFirstOrder(SparseMatrixBase graph, int start, bool directed = true)
    {
        var g = Validate(graph);
        var n = g.Rows;
        if ((uint)start >= (uint)n) throw new ArgumentException("start out of range.");
        var (o, inAdj) = TraversalGraph(g, directed);
        var pred = new int[n];
        Array.Fill(pred, NoPredecessor);
        var seen = new bool[n];
        var order = new List<int>(n) { start };
        seen[start] = true;
        var stack = new List<int> { start };
        while (stack.Count > 0)
        {
            var u = stack[^1];
            var next = -1;
            for (var p = o.Ptr[u]; p < o.Ptr[u + 1] && next < 0; p++) if (!seen[o.Idx[p]]) next = o.Idx[p];
            if (next < 0 && inAdj is not null)
                for (var p = inAdj.Ptr[u]; p < inAdj.Ptr[u + 1] && next < 0; p++) if (!seen[inAdj.Idx[p]]) next = inAdj.Idx[p];
            if (next < 0) { stack.RemoveAt(stack.Count - 1); continue; }
            seen[next] = true;
            pred[next] = u;
            order.Add(next);
            stack.Add(next);
        }
        return (order.ToArray(), pred);
    }

    /// <summary>The breadth-first tree from <paramref name="start"/> as a sparse matrix of the edges used — <c>breadth_first_tree</c>.</summary>
    public static CsrMatrix BreadthFirstTree(SparseMatrixBase graph, int start, bool directed = true)
        => ReconstructPath(graph, BreadthFirstOrder(graph, start, directed).Predecessors, directed);

    /// <summary>The depth-first tree from <paramref name="start"/> — <c>depth_first_tree</c>.</summary>
    public static CsrMatrix DepthFirstTree(SparseMatrixBase graph, int start, bool directed = true)
        => ReconstructPath(graph, DepthFirstOrder(graph, start, directed).Predecessors, directed);

    /// <summary>
    /// The tree defined by a predecessor array as a sparse matrix — <c>reconstruct_path</c>. Entry
    /// <c>(pred[j], j)</c> carries the edge weight (for undirected graphs the smaller of the two directions).
    /// </summary>
    public static CsrMatrix ReconstructPath(SparseMatrixBase graph, int[] predecessors, bool directed = true)
    {
        var g = Validate(graph);
        var n = g.Rows;
        if (predecessors.Length != n) throw new ArgumentException("predecessors must have one entry per node.");
        var r = new List<int>(); var c = new List<int>(); var v = new List<double>();
        for (var j = 0; j < n; j++)
        {
            var p = predecessors[j];
            if (p < 0) continue;
            r.Add(p); c.Add(j); v.Add(EdgeWeight(g, p, j, directed));
        }
        return CsrMatrix.FromTriplets(n, n, r.ToArray(), c.ToArray(), v.ToArray());
    }

    private static double EdgeWeight(CsrMatrix g, int u, int v, bool directed)
    {
        var best = double.PositiveInfinity;
        for (var p = g.IndptrArray[u]; p < g.IndptrArray[u + 1]; p++) if (g.IndicesArray[p] == v) best = Math.Min(best, g.DataArray[p]);
        if (!directed)
            for (var p = g.IndptrArray[v]; p < g.IndptrArray[v + 1]; p++) if (g.IndicesArray[p] == u) best = Math.Min(best, g.DataArray[p]);
        if (double.IsPositiveInfinity(best)) throw new ArgumentException($"No edge {u}→{v} in the graph.");
        return best;
    }

    /// <summary>
    /// Distance matrix from a predecessor matrix — <c>construct_dist_matrix</c>: walks each path
    /// back to its source summing edge weights. Unreachable pairs get <paramref name="nullValue"/>.
    /// </summary>
    public static NdArray ConstructDistMatrix(SparseMatrixBase graph, int[,] predecessors, bool directed = true, double nullValue = double.PositiveInfinity)
    {
        var g = Validate(graph);
        var n = g.Rows;
        if (predecessors.GetLength(0) != n || predecessors.GetLength(1) != n)
            throw new ArgumentException("predecessors must be n×n, row i holding the paths from source i.");
        var d = new double[n * n];
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                if (i == j) continue;
                if (predecessors[i, j] < 0) { d[i * n + j] = nullValue; continue; }
                var sum = 0.0;
                var v = j;
                var guard = 0;
                while (v != i)
                {
                    var u = predecessors[i, v];
                    if (u < 0 || ++guard > n) throw new ArgumentException($"predecessors row {i} does not lead back to its source.");
                    sum += EdgeWeight(g, u, v, directed);
                    v = u;
                }
                d[i * n + j] = sum;
            }
        return new NdArray(d, n, n);
    }

    // ------------------------------------------------------------------ components

    /// <summary>
    /// Connected components — <c>connected_components</c>. Labels are numbered in SciPy's order:
    /// weak components by their lowest node, strong components in the order Tarjan/Pearce's
    /// depth-first search completes them (sinks first).
    /// </summary>
    /// <param name="graph">Square adjacency matrix.</param>
    /// <param name="directed">Whether edges are directed (undirected implies weak).</param>
    /// <param name="connection">Weak or strong connectivity for directed graphs.</param>
    /// <remarks>The strong case is an iterative Tarjan with an explicit call stack — no recursion depth limit.</remarks>
    public static (int Count, int[] Labels) ConnectedComponents(SparseMatrixBase graph, bool directed = true,
        ComponentConnection connection = ComponentConnection.Weak)
    {
        var g = Validate(graph);
        var n = g.Rows;
        var labels = new int[n];
        Array.Fill(labels, -1);
        if (!directed || connection == ComponentConnection.Weak)
        {
            var (o, inAdj) = TraversalGraph(g, directed: false);
            var count = 0;
            var queue = new int[n];
            for (var s = 0; s < n; s++)
            {
                if (labels[s] >= 0) continue;
                int head = 0, tail = 0;
                queue[tail++] = s;
                labels[s] = count;
                while (head < tail)
                {
                    var u = queue[head++];
                    foreach (var adj in new[] { o, inAdj! })
                        for (var p = adj.Ptr[u]; p < adj.Ptr[u + 1]; p++)
                        {
                            var v = adj.Idx[p];
                            if (labels[v] >= 0) continue;
                            labels[v] = count;
                            queue[tail++] = v;
                        }
                }
                count++;
            }
            return (count, labels);
        }
        // Iterative Tarjan.
        var ptr = g.IndptrArray; var idx = g.IndicesArray;
        var index = new int[n];
        var low = new int[n];
        Array.Fill(index, -1);
        var onStack = new bool[n];
        var stack = new Stack<int>();
        var call = new Stack<(int Node, int Edge)>();
        int counter = 0, comps = 0;
        for (var s = 0; s < n; s++)
        {
            if (index[s] >= 0) continue;
            call.Push((s, ptr[s]));
            index[s] = low[s] = counter++;
            stack.Push(s);
            onStack[s] = true;
            while (call.Count > 0)
            {
                var (u, e) = call.Pop();
                var descended = false;
                for (; e < ptr[u + 1]; e++)
                {
                    var v = idx[e];
                    if (index[v] < 0)
                    {
                        call.Push((u, e + 1));
                        index[v] = low[v] = counter++;
                        stack.Push(v);
                        onStack[v] = true;
                        call.Push((v, ptr[v]));
                        descended = true;
                        break;
                    }
                    if (onStack[v]) low[u] = Math.Min(low[u], index[v]);
                }
                if (descended) continue;
                if (low[u] == index[u])
                {
                    int w;
                    do
                    {
                        w = stack.Pop();
                        onStack[w] = false;
                        labels[w] = comps;
                    } while (w != u);
                    comps++;
                }
                if (call.Count > 0)
                {
                    var parent = call.Peek().Node;
                    low[parent] = Math.Min(low[parent], low[u]);
                }
            }
        }
        return (comps, labels);
    }

    // ------------------------------------------------------------------ spanning trees

    /// <summary>
    /// Minimum spanning tree (forest) of the undirected graph — <c>minimum_spanning_tree</c>, by
    /// Kruskal with union–find (path halving, union by size). Edges keep the orientation in which
    /// they were stored; ties are broken by stored (row-major) order.
    /// </summary>
    public static CsrMatrix MinimumSpanningTree(SparseMatrixBase graph)
    {
        var g = Validate(graph).EliminateZeros();
        var n = g.Rows;
        var entries = g.EnumerateEntries().Where(e => e.Row != e.Column).ToArray();
        var order = Enumerable.Range(0, entries.Length).OrderBy(i => entries[i].Value).ToArray(); // LINQ OrderBy is stable
        var parent = new int[n];
        var size = new int[n];
        for (var i = 0; i < n; i++) { parent[i] = i; size[i] = 1; }
        int Find(int x)
        {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        }
        var r = new List<int>(); var c = new List<int>(); var v = new List<double>();
        foreach (var t in order)
        {
            var (i, j, w) = entries[t];
            int a = Find(i), b = Find(j);
            if (a == b) continue;
            if (size[a] < size[b]) (a, b) = (b, a);
            parent[b] = a;
            size[a] += size[b];
            r.Add(i); c.Add(j); v.Add(w);
            if (r.Count == n - 1) break;
        }
        return CsrMatrix.FromTriplets(n, n, r.ToArray(), c.ToArray(), v.ToArray());
    }

    // ------------------------------------------------------------------ flows and matchings

    /// <summary>
    /// Maximum flow from <paramref name="source"/> to <paramref name="sink"/> — <c>maximum_flow</c>.
    /// Stored entries are capacities (must be non-negative). SciPy insists on integer capacities;
    /// real capacities are accepted here, but only integer ones give exact results.
    /// </summary>
    public static MaximumFlowResult MaximumFlow(SparseMatrixBase graph, int source, int sink, MaximumFlowMethod method = MaximumFlowMethod.Dinic)
    {
        var g = Validate(graph);
        var n = g.Rows;
        if ((uint)source >= (uint)n || (uint)sink >= (uint)n) throw new ArgumentException("source/sink out of range.");
        if (source == sink) throw new ArgumentException("source and sink must differ.");
        // Residual graph: each stored edge plus a zero-capacity reverse twin.
        var m = g.Nnz;
        var head = new int[2 * m]; var cap = new double[2 * m]; var from = new int[2 * m];
        var deg = new int[n + 1];
        foreach (var (i, j, w) in g.EnumerateEntries())
        {
            if (w < 0) throw new ArgumentException("Capacities must be non-negative.");
            deg[i + 1]++; deg[j + 1]++;
        }
        for (var i = 0; i < n; i++) deg[i + 1] += deg[i];
        var next = (int[])deg.Clone();
        var twin = new int[2 * m];
        foreach (var (i, j, w) in g.EnumerateEntries())
        {
            var a = next[i]++;
            var b = next[j]++;
            head[a] = j; cap[a] = w; from[a] = i; twin[a] = b;
            head[b] = i; cap[b] = 0; from[b] = j; twin[b] = a;
        }
        var original = (double[])cap.Clone();
        double total = 0;
        if (method == MaximumFlowMethod.EdmondsKarp)
        {
            var predEdge = new int[n];
            var queue = new int[n];
            while (true)
            {
                Array.Fill(predEdge, -1);
                int qh = 0, qt = 0;
                queue[qt++] = source;
                predEdge[source] = -2;
                while (qh < qt && predEdge[sink] == -1)
                {
                    var u = queue[qh++];
                    for (var e = deg[u]; e < deg[u + 1]; e++)
                        if (cap[e] > 0 && predEdge[head[e]] == -1) { predEdge[head[e]] = e; queue[qt++] = head[e]; }
                }
                if (predEdge[sink] == -1) break;
                var push = double.PositiveInfinity;
                for (var v = sink; v != source; v = from[predEdge[v]]) push = Math.Min(push, cap[predEdge[v]]);
                for (var v = sink; v != source; v = from[predEdge[v]]) { cap[predEdge[v]] -= push; cap[twin[predEdge[v]]] += push; }
                total += push;
            }
        }
        else
        {
            var level = new int[n];
            var it = new int[n];
            var queue = new int[n];
            var pathEdges = new int[n];
            while (true)
            {
                Array.Fill(level, -1);
                int qh = 0, qt = 0;
                queue[qt++] = source;
                level[source] = 0;
                while (qh < qt)
                {
                    var u = queue[qh++];
                    for (var e = deg[u]; e < deg[u + 1]; e++)
                        if (cap[e] > 0 && level[head[e]] < 0) { level[head[e]] = level[u] + 1; queue[qt++] = head[e]; }
                }
                if (level[sink] < 0) break;
                Array.Copy(deg, it, n);
                // Iterative blocking-flow DFS with current-arc pointers.
                while (true)
                {
                    var depth = 0;
                    var u = source;
                    var found = false;
                    while (true)
                    {
                        if (u == sink) { found = true; break; }
                        var advanced = false;
                        for (; it[u] < deg[u + 1]; it[u]++)
                        {
                            var e = it[u];
                            var v = head[e];
                            if (cap[e] > 0 && level[v] == level[u] + 1)
                            {
                                pathEdges[depth++] = e;
                                u = v;
                                advanced = true;
                                break;
                            }
                        }
                        if (advanced) continue;
                        if (depth == 0) break;
                        level[u] = -1; // dead end: prune
                        u = from[pathEdges[--depth]];
                        it[u]++;
                    }
                    if (!found) break;
                    var push = double.PositiveInfinity;
                    for (var t = 0; t < depth; t++) push = Math.Min(push, cap[pathEdges[t]]);
                    for (var t = 0; t < depth; t++) { cap[pathEdges[t]] -= push; cap[twin[pathEdges[t]]] += push; }
                    total += push;
                }
            }
        }
        // Net flow on each original edge = capacity − residual; antisymmetrise.
        var fr = new List<int>(); var fc = new List<int>(); var fv = new List<double>();
        for (var e = 0; e < 2 * m; e++)
        {
            var flow = original[e] - cap[e];
            if (original[e] == 0 && flow <= 0) continue; // reverse twins are covered by their forward edge
            if (flow == 0) continue;
            fr.Add(from[e]); fc.Add(head[e]); fv.Add(flow);
            fr.Add(head[e]); fc.Add(from[e]); fv.Add(-flow);
        }
        var flowMatrix = CsrMatrix.FromTriplets(n, n, fr.ToArray(), fc.ToArray(), fv.ToArray()).EliminateZeros();
        return new MaximumFlowResult(total, flowMatrix);
    }

    /// <summary>
    /// Maximum cardinality matching of a bipartite graph given as its m×n biadjacency matrix —
    /// <c>maximum_bipartite_matching</c>, by Hopcroft–Karp (O(E√V)). Unmatched entries are −1.
    /// Any maximum matching is correct; it may differ from SciPy's when several exist.
    /// </summary>
    public static int[] MaximumBipartiteMatching(SparseMatrixBase graph, MatchingPermutation permType = MatchingPermutation.Row)
    {
        var (rowMatch, colMatch) = HopcroftKarp(graph.ToCsr().SumDuplicates());
        return permType == MatchingPermutation.Row ? colMatch : rowMatch;
    }

    private static (int[] RowMatch, int[] ColMatch) HopcroftKarp(CsrMatrix g)
    {
        int m = g.Rows, n = g.Columns;
        var ptr = g.IndptrArray; var idx = g.IndicesArray;
        var rowMatch = new int[m]; var colMatch = new int[n];
        Array.Fill(rowMatch, -1); Array.Fill(colMatch, -1);
        var dist = new int[m];
        var queue = new int[m];
        var it = new int[m];
        var stack = new int[m + 1];
        while (true)
        {
            int qh = 0, qt = 0;
            var found = false;
            for (var i = 0; i < m; i++)
            {
                if (rowMatch[i] < 0) { dist[i] = 0; queue[qt++] = i; }
                else dist[i] = -1;
            }
            while (qh < qt)
            {
                var u = queue[qh++];
                for (var p = ptr[u]; p < ptr[u + 1]; p++)
                {
                    var w = colMatch[idx[p]];
                    if (w < 0) found = true;
                    else if (dist[w] < 0) { dist[w] = dist[u] + 1; queue[qt++] = w; }
                }
            }
            if (!found) break;
            for (var i = 0; i < m; i++) it[i] = ptr[i];
            for (var r = 0; r < m; r++)
            {
                if (rowMatch[r] >= 0) continue;
                // Iterative layered DFS from free row r.
                var top = 0;
                stack[top++] = r;
                var augmented = false;
                while (top > 0 && !augmented)
                {
                    var u = stack[top - 1];
                    var advanced = false;
                    for (; it[u] < ptr[u + 1]; it[u]++)
                    {
                        var c = idx[it[u]];
                        var w = colMatch[c];
                        if (w < 0)
                        {
                            // Augment along the stack.
                            for (var t = top - 1; t >= 0; t--)
                            {
                                var row = stack[t];
                                var col = idx[it[row]];
                                colMatch[col] = row;
                                rowMatch[row] = col;
                            }
                            augmented = true;
                            break;
                        }
                        if (dist[w] == dist[u] + 1)
                        {
                            stack[top++] = w;
                            advanced = true;
                            break;
                        }
                    }
                    if (augmented || advanced) continue;
                    dist[u] = -1;
                    top--;
                    if (top > 0) it[stack[top - 1]]++;
                }
            }
        }
        return (rowMatch, colMatch);
    }

    /// <summary>
    /// Structural rank: the size of a maximum matching between rows and columns of the pattern —
    /// <c>structural_rank</c>. An upper bound on the numerical rank.
    /// </summary>
    public static int StructuralRank(SparseMatrixBase graph)
    {
        var (rowMatch, _) = HopcroftKarp(graph.ToCsr().SumDuplicates());
        return rowMatch.Count(c => c >= 0);
    }

    /// <summary>
    /// Minimum-weight full bipartite matching — <c>min_weight_full_bipartite_matching</c>. Every
    /// row of the m×n biadjacency matrix (m ≤ n; a wide input is solved transposed) is matched to a
    /// distinct column, minimising (or maximising) total weight. Throws when no full matching exists.
    /// </summary>
    /// <remarks>
    /// Successive shortest augmenting paths with row/column potentials (the Hungarian method in
    /// Jonker–Volgenant form, as SciPy's LAPJVsp), each path found by Dijkstra over the sparse
    /// edges only. Stored zeros are edges of weight zero.
    /// </remarks>
    public static (int[] RowIndices, int[] ColumnIndices) MinWeightFullBipartiteMatching(SparseMatrixBase graph, bool maximize = false)
    {
        var g = graph.ToCsr().SumDuplicates();
        if (g.Rows > g.Columns)
        {
            var (r, c) = MinWeightFullBipartiteMatching(g.ToCsc().Inner, maximize);
            // r indexes original columns, c original rows: report sorted by row.
            var pairs = r.Zip(c).OrderBy(p => p.Second).ToArray();
            return (pairs.Select(p => p.Second).ToArray(), pairs.Select(p => p.First).ToArray());
        }
        int m = g.Rows, n = g.Columns;
        var ptr = g.IndptrArray; var idx = g.IndicesArray;
        var cost = g.Data.ToArray();
        if (maximize) for (var t = 0; t < cost.Length; t++) cost[t] = -cost[t];
        foreach (var v in cost) if (!double.IsFinite(v)) throw new ArgumentException("Weights must be finite.");
        // Potentials: rc(i,j) = c(i,j) − pr[i] − pc[j] ≥ 0 throughout, = 0 on matched edges.
        // Columns start at pc = 0 and only matched columns ever move: with more columns than rows
        // the dual needs pc ≤ 0 and pc = 0 on unmatched columns, so the reduction goes on rows.
        var pr = new double[m];
        var pc = new double[n];
        for (var i = 0; i < m; i++)
        {
            var best = double.PositiveInfinity;
            for (var p = ptr[i]; p < ptr[i + 1]; p++) best = Math.Min(best, cost[p]);
            pr[i] = double.IsPositiveInfinity(best) ? 0 : best;
        }
        var rowMatch = new int[m]; var colMatch = new int[n];
        Array.Fill(rowMatch, -1); Array.Fill(colMatch, -1);
        var dist = new double[n];
        var prevRow = new int[n];
        var final = new bool[n];
        var finalized = new List<int>();
        for (var root = 0; root < m; root++)
        {
            Array.Fill(dist, double.PositiveInfinity);
            Array.Fill(final, false);
            finalized.Clear();
            var pq = new PriorityQueue<int, double>();
            void Relax(int i, double di)
            {
                for (var p = ptr[i]; p < ptr[i + 1]; p++)
                {
                    var j = idx[p];
                    if (final[j]) continue;
                    var nd = di + Math.Max(0, cost[p] - pr[i] - pc[j]);
                    if (nd < dist[j]) { dist[j] = nd; prevRow[j] = i; pq.Enqueue(j, nd); }
                }
            }
            Relax(root, 0);
            var freeCol = -1;
            double dFree = 0;
            while (pq.TryDequeue(out var j, out var dj))
            {
                if (final[j] || dj > dist[j]) continue;
                final[j] = true;
                if (colMatch[j] < 0) { freeCol = j; dFree = dj; break; }
                finalized.Add(j);
                Relax(colMatch[j], dj);
            }
            if (freeCol < 0) throw new ArgumentException("no full matching exists");
            pr[root] += dFree;
            foreach (var j in finalized)
            {
                pc[j] -= dFree - dist[j];
                pr[colMatch[j]] += dFree - dist[j];
            }
            for (var j = freeCol; j >= 0;)
            {
                var i = prevRow[j];
                var prevCol = rowMatch[i];
                rowMatch[i] = j;
                colMatch[j] = i;
                j = i == root ? -1 : prevCol;
            }
        }
        return (Enumerable.Range(0, m).ToArray(), rowMatch);
    }

    // ------------------------------------------------------------------ orderings and Laplacians

    /// <summary>
    /// Reverse Cuthill–McKee ordering — <c>reverse_cuthill_mckee</c>: a permutation that pulls the
    /// non-zeros towards the diagonal (small bandwidth). A port of SciPy's algorithm, including its
    /// degree definition (a stored diagonal counts twice) and level-wise insertion sort, so the
    /// orderings are identical whenever no two nodes tie on degree in a way NumPy's unstable
    /// argsort would break differently (ties here go to the lower index).
    /// </summary>
    /// <param name="graph">Square matrix.</param>
    /// <param name="symmetricMode">Assume the pattern is symmetric; otherwise <c>A + Aᵀ</c>'s pattern is used.</param>
    public static int[] ReverseCuthillMcKee(SparseMatrixBase graph, bool symmetricMode = false)
    {
        var g = Validate(graph);
        var n = g.Rows;
        CsrMatrix s;
        if (symmetricMode) s = g;
        else
        {
            var pattern = g.Map(_ => 1.0).Abs();
            s = pattern.Add(pattern.ToCsc().Inner);
        }
        var ptr = s.IndptrArray; var ind = s.IndicesArray;
        var degree = new int[n];
        for (var i = 0; i < n; i++)
        {
            degree[i] = ptr[i + 1] - ptr[i];
            for (var p = ptr[i]; p < ptr[i + 1]; p++) if (ind[p] == i) { degree[i]++; break; }
        }
        var inds = Enumerable.Range(0, n).OrderBy(i => degree[i]).ToArray();
        var revInds = new int[n];
        for (var i = 0; i < n; i++) revInds[inds[i]] = i;
        var order = new int[n];
        var temp = new int[Math.Max(1, degree.DefaultIfEmpty(0).Max())];
        var count = 0;
        for (var zz = 0; zz < n; zz++)
        {
            if (inds[zz] == -1) continue;
            var seed = inds[zz];
            order[count++] = seed;
            inds[revInds[seed]] = -1;
            int levelStart = count - 1, levelEnd = count;
            while (levelStart < levelEnd)
            {
                for (var ii = levelStart; ii < levelEnd; ii++)
                {
                    var i = order[ii];
                    var nOld = count;
                    for (var jj = ptr[i]; jj < ptr[i + 1]; jj++)
                    {
                        var j = ind[jj];
                        if (inds[revInds[j]] != -1)
                        {
                            inds[revInds[j]] = -1;
                            order[count++] = j;
                        }
                    }
                    var levelLen = 0;
                    if (count - nOld > temp.Length) Array.Resize(ref temp, count - nOld);
                    for (var kk = nOld; kk < count; kk++) temp[levelLen++] = degree[order[kk]];
                    for (var kk = 1; kk < levelLen; kk++)
                    {
                        var t = temp[kk];
                        var t2 = order[nOld + kk];
                        var ll = kk;
                        while (ll > 0 && t < temp[ll - 1])
                        {
                            temp[ll] = temp[ll - 1];
                            order[nOld + ll] = order[nOld + ll - 1];
                            ll--;
                        }
                        temp[ll] = t;
                        order[nOld + ll] = t2;
                    }
                }
                levelStart = levelEnd;
                levelEnd = count;
            }
            if (count == n) break;
        }
        Array.Reverse(order);
        return order;
    }

    /// <summary>
    /// Graph Laplacian — <c>laplacian</c>: <c>L = D − A</c> with the diagonal of A ignored, or
    /// <c>I − D^{-1/2} A D^{-1/2}</c> when <paramref name="normed"/> (isolated nodes get a zero row).
    /// Degrees are column sums (in-degree) unless <paramref name="useOutDegree"/>, as in SciPy.
    /// </summary>
    public static CsrMatrix Laplacian(SparseMatrixBase graph, bool normed = false, bool useOutDegree = false)
        => LaplacianWithDiagonal(graph, normed, useOutDegree).Laplacian;

    /// <summary><see cref="Laplacian"/> plus the diagonal SciPy returns with <c>return_diag=True</c>.</summary>
    public static LaplacianResult LaplacianWithDiagonal(SparseMatrixBase graph, bool normed = false, bool useOutDegree = false)
    {
        var g = Validate(graph);
        var n = g.Rows;
        var off = g.Filter(static (i, j, _) => i == j);
        var deg = off.SumAxis(useOutDegree ? 1 : 0);
        if (!normed)
        {
            var l = SparseMatrices.Diags(deg, 0).Subtract(off);
            return new LaplacianResult(l, deg);
        }
        var d = deg.Select(x => x == 0 ? 1.0 : Math.Sqrt(x)).ToArray();
        var r = new List<int>(); var c = new List<int>(); var v = new List<double>();
        foreach (var (i, j, w) in off.EnumerateEntries()) { r.Add(i); c.Add(j); v.Add(-w / (d[i] * d[j])); }
        for (var i = 0; i < n; i++) if (deg[i] != 0) { r.Add(i); c.Add(i); v.Add(1.0); }
        return new LaplacianResult(CsrMatrix.FromTriplets(n, n, r.ToArray(), c.ToArray(), v.ToArray()), d);
    }

    // ------------------------------------------------------------------ conversions

    /// <summary>
    /// A graph from a dense matrix — <c>csgraph_from_dense</c>: entries equal to
    /// <paramref name="nullValue"/> (and NaN/∞ if flagged) are non-edges; everything else, zeros
    /// included when <paramref name="nullValue"/> is not zero, is stored.
    /// </summary>
    public static CsrMatrix CsgraphFromDense(NdArray dense, double nullValue = 0.0, bool nanNull = true, bool infNull = true)
    {
        if (dense.Rank != 2 || dense.Shape[0] != dense.Shape[1]) throw new ArgumentException("A dense graph must be a square rank 2 array.");
        var n = dense.Shape[0];
        var src = dense.AsContiguous().AsSpan();
        var r = new List<int>(); var c = new List<int>(); var v = new List<double>();
        for (var i = 0; i < n; i++)
            for (var j = 0; j < n; j++)
            {
                var x = src[i * n + j];
                if (x == nullValue || (nanNull && double.IsNaN(x)) || (infNull && double.IsInfinity(x))) continue;
                r.Add(i); c.Add(j); v.Add(x);
            }
        // Built directly (not via FromTriplets' summing) so explicit zeros survive as edges.
        var ptr = new int[n + 1];
        foreach (var i in r) ptr[i + 1]++;
        for (var i = 0; i < n; i++) ptr[i + 1] += ptr[i];
        return new CsrMatrix(v.ToArray(), c.ToArray(), ptr, n, n, copy: false);
    }

    /// <summary>
    /// A dense matrix from a graph — <c>csgraph_to_dense</c>: non-edges become
    /// <paramref name="nullValue"/>; parallel edges collapse to their minimum weight.
    /// </summary>
    public static NdArray CsgraphToDense(SparseMatrixBase graph, double nullValue = 0.0)
    {
        ArgumentNullException.ThrowIfNull(graph);
        int m = graph.Rows, n = graph.Columns;
        var d = new double[m * n];
        var set = new bool[m * n];
        foreach (var (i, j, w) in graph.ToCsr().EnumerateEntries())
        {
            var k = i * n + j;
            d[k] = set[k] ? Math.Min(d[k], w) : w;
            set[k] = true;
        }
        for (var k = 0; k < d.Length; k++) if (!set[k]) d[k] = nullValue;
        return new NdArray(d, m, n);
    }
}
