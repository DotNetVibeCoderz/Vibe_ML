using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SparseAssert;

namespace Gravicode.Science.Tests.SciNet;

public class CsGraphTests
{
    private const double Inf = double.PositiveInfinity;

    // Directed 5-node example used for the SciPy reference values below.
    private static CsrMatrix G5 => CsrMatrix.FromDense(new NdArray([
        0, 4, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 2, 0, 5, 0,
        0, 0, 0, 0, 3,
        1, 0, 0, 0, 0.0], 5, 5));

    private static void CloseInf(double[] expected, double[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            if (double.IsPositiveInfinity(expected[i])) Assert.True(double.IsPositiveInfinity(actual[i]), $"{what}[{i}] should be ∞, got {actual[i]}");
            else Close(expected[i], actual[i], tol, $"{what}[{i}]");
        }
    }

    [Fact]
    public void ShortestPath_MatchesScipy_ForEveryMethod()
    {
        // scipy 1.17.1: shortest_path(G5, return_predecessors=True)
        double[] expected = [0, 3, 1, 4, 7, 5, 0, 6, 1, 4, 7, 2, 0, 3, 6, 4, 7, 5, 0, 3, 1, 4, 2, 5, 0];
        int[] pred = [-9999, 2, 0, 1, 3, 4, -9999, 0, 1, 3, 4, 2, -9999, 1, 3, 4, 2, 0, -9999, 3, 4, 2, 0, 1, -9999];
        foreach (var method in Enum.GetValues<ShortestPathMethod>())
        {
            var r = CsGraph.ShortestPath(G5, method, returnPredecessors: true);
            Close(expected, r.Distances.ToArray(), 0, method.ToString());
            for (var i = 0; i < 5; i++)
                for (var j = 0; j < 5; j++) Assert.Equal(pred[i * 5 + j], r.Predecessors![i, j]);
        }
    }

    [Fact]
    public void ShortestPath_UndirectedAndUnweighted_MatchScipy()
    {
        // scipy 1.17.1: shortest_path(G5, directed=False)
        double[] undirected = [0, 3, 1, 4, 1, 3, 0, 2, 1, 4, 1, 2, 0, 3, 2, 4, 1, 3, 0, 3, 1, 4, 2, 3, 0];
        // scipy 1.17.1: shortest_path(G5, unweighted=True)
        double[] hops = [0, 1, 1, 2, 3, 3, 0, 4, 1, 2, 3, 1, 0, 1, 2, 2, 3, 3, 0, 1, 1, 2, 2, 3, 0];
        foreach (var method in Enum.GetValues<ShortestPathMethod>())
        {
            var r = CsGraph.ShortestPath(G5, method, directed: false, returnPredecessors: true);
            Close(undirected, r.Distances.ToArray(), 0, "undirected " + method);
            // Ties make predecessors non-unique; they must still rebuild the same distances.
            Close(undirected, CsGraph.ConstructDistMatrix(G5, r.Predecessors!, directed: false).ToArray(), 0, "rebuilt");
            Close(hops, CsGraph.ShortestPath(G5, method, unweighted: true).Distances.ToArray(), 0, "unweighted " + method);
        }
        // scipy 1.17.1: dijkstra(G5, indices=[0], limit=3) -> [0, 3, 1, inf, inf]
        CloseInf([0, 3, 1, Inf, Inf], CsGraph.Dijkstra(G5, indices: [0], limit: 3).Distances.ToArray(), 0, "limit");
    }

    [Fact]
    public void ExplicitZeros_AreEdges_AsInScipy()
    {
        // csr with a stored zero 0→1 and 1→2 of weight 1: scipy gives d[0] = [0, 0, 1].
        var g = new CsrMatrix([0.0, 1.0], [1, 2], [0, 1, 2, 2], 3, 3);
        CloseInf([0, 0, 1, Inf, 0, 1, Inf, Inf, 0], CsGraph.ShortestPath(g, ShortestPathMethod.Dijkstra).Distances.ToArray(), 0, "zeros");
        Assert.Equal(1, CsGraph.ConnectedComponents(g).Count);
        // ...and CsgraphFromDense with a non-zero null value keeps zeros as edges.
        var fromDense = CsGraph.CsgraphFromDense(new NdArray([Inf, 0, Inf, Inf, Inf, 1, Inf, Inf, Inf], 3, 3), nullValue: Inf);
        Assert.Equal(2, fromDense.Nnz);
        Close(new NdArray([Inf, 0, Inf, Inf, Inf, 1, Inf, Inf, Inf], 3, 3), CsGraph.CsgraphToDense(fromDense, Inf), 0);
    }

    [Fact]
    public void NegativeWeights_BellmanFordAndJohnson_MatchScipy_AndDetectCycles()
    {
        var d = G5.ToDense().ToArray();
        d[2 * 5 + 1] = -2; d[1 * 5 + 3] = 1;
        var g = CsrMatrix.FromDense(new NdArray(d, 5, 5));
        // scipy 1.17.1: bellman_ford(g) == johnson(g)
        double[] expected = [0, -1, 1, 0, 3, 5, 0, 6, 1, 4, 3, -2, 0, -1, 2, 4, 3, 5, 0, 3, 1, 0, 2, 1, 0];
        Close(expected, CsGraph.BellmanFord(g).Distances.ToArray(), 1e-15, "bellman-ford");
        Close(expected, CsGraph.Johnson(g).Distances.ToArray(), 1e-12, "johnson");
        Close(expected, CsGraph.FloydWarshall(g).Distances.ToArray(), 1e-15, "floyd-warshall");
        Close(expected, CsGraph.ShortestPath(g).Distances.ToArray(), 1e-12, "auto");
        Assert.Throws<ArgumentException>(() => CsGraph.Dijkstra(g));
        // 0→2 (1), 2→0 (−3): a negative cycle.
        var cyc = CsrMatrix.FromDense(new NdArray([0, 0, 1, 0, 0, 0, -3, 0, 0.0], 3, 3));
        Assert.Throws<NegativeCycleException>(() => CsGraph.BellmanFord(cyc));
        Assert.Throws<NegativeCycleException>(() => CsGraph.Johnson(cyc));
        Assert.Throws<NegativeCycleException>(() => CsGraph.FloydWarshall(cyc));
        // An undirected negative edge is a negative cycle.
        var neg = CsrMatrix.FromDense(new NdArray([0, -1, 0, 0.0], 2, 2));
        Assert.Throws<NegativeCycleException>(() => CsGraph.BellmanFord(neg, directed: false));
    }

    [Fact]
    public void ShortestPath_AllMethodsAgree_OnRandomGraphs()
    {
        for (var seed = 0; seed < 4; seed++)
        {
            var g = SparseMatrices.RandomArray(60, 60, 0.06, seed, r => r.Uniform(0.1, 5));
            var fw = CsGraph.FloydWarshall(g).Distances.ToArray();
            CloseInf(fw, CsGraph.Dijkstra(g).Distances.ToArray(), 1e-12, "dijkstra");
            CloseInf(fw, CsGraph.BellmanFord(g).Distances.ToArray(), 1e-12, "bellman-ford");
            CloseInf(fw, CsGraph.Johnson(g).Distances.ToArray(), 1e-12, "johnson");
            var und = CsGraph.FloydWarshall(g, directed: false).Distances.ToArray();
            CloseInf(und, CsGraph.Dijkstra(g, directed: false).Distances.ToArray(), 1e-12, "undirected");
            // min_only: distance to the nearest of two sources is the minimum of their rows.
            var (dist, _, src) = CsGraph.DijkstraMinOnly(g, [3, 17]);
            for (var j = 0; j < 60; j++)
            {
                var best = Math.Min(fw[3 * 60 + j], fw[17 * 60 + j]);
                if (double.IsPositiveInfinity(best)) { Assert.True(double.IsPositiveInfinity(dist[j])); Assert.Equal(CsGraph.NoPredecessor, src[j]); }
                else { Close(best, dist[j], 1e-12, "min only"); Close(best, fw[src[j] * 60 + j], 1e-12, "source"); }
            }
        }
    }

    [Fact]
    public void Yen_MatchesScipy()
    {
        // scipy 1.17.1: yen(G5, 0, 4, 3) -> [7, 8, 9]
        var (dist, paths) = CsGraph.YenKShortest(G5, 0, 4, 3);
        Close([7.0, 8, 9], dist, 1e-15);
        Assert.Equal([0, 2, 1, 3, 4], paths[0]);
        Assert.All(paths, p => Assert.Equal(p.Length, p.Distinct().Count()));
    }

    [Fact]
    public void Traversals_MatchScipyOrder()
    {
        var g = CsrMatrix.FromDense(new NdArray([0, 1, 2, 0, 0, 0, 0, 1, 2, 0, 0, 3, 0, 0, 0, 0.0], 4, 4));
        // scipy 1.17.1: depth_first_order(g, 0, directed=False) / breadth_first_order(g, 0, directed=False)
        var (dfo, dfp) = CsGraph.DepthFirstOrder(g, 0, directed: false);
        Assert.Equal([0, 1, 3, 2], dfo);
        Assert.Equal([-9999, 0, 3, 1], dfp);
        var (bfo, bfp) = CsGraph.BreadthFirstOrder(g, 0, directed: false);
        Assert.Equal([0, 1, 2, 3], bfo);
        Assert.Equal([-9999, 0, 0, 1], bfp);
        // scipy 1.17.1: breadth_first_tree(g, 0, directed=True).toarray()
        Close(new NdArray([0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.0], 4, 4), CsGraph.BreadthFirstTree(g, 0).ToDense(), 0);
        var dft = CsGraph.DepthFirstTree(g, 0);
        Assert.Equal(3, dft.Nnz);
        var (directedOrder, _) = CsGraph.DepthFirstOrder(g, 3);
        Assert.Equal([3], directedOrder); // node 3 has no out-edges
    }

    [Fact]
    public void Traversal_OnALongPath_DoesNotOverflowTheStack()
    {
        const int n = 200_000;
        var rows = Enumerable.Range(0, n - 1).ToArray();
        var cols = Enumerable.Range(1, n - 1).ToArray();
        var path = CsrMatrix.FromTriplets(n, n, rows, cols, Enumerable.Repeat(1.0, n - 1).ToArray());
        var (order, _) = CsGraph.DepthFirstOrder(path, 0);
        Assert.Equal(n, order.Length);
        var (count, labels) = CsGraph.ConnectedComponents(path, connection: ComponentConnection.Strong);
        Assert.Equal(n, count);
        Assert.Equal(0, labels[n - 1]); // the sink completes first
    }

    [Fact]
    public void ConnectedComponents_MatchScipyLabels()
    {
        // scipy 1.17.1 strong: (2, [0, 0, 0, 1]) and (4, [2, 1, 0, 3, 3]); weak: (2, [0, 0, 0, 1, 1])
        var g1 = CsrMatrix.FromDense(new NdArray([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0.0], 4, 4));
        var (c1, l1) = CsGraph.ConnectedComponents(g1, connection: ComponentConnection.Strong);
        Assert.Equal(2, c1);
        Assert.Equal([0, 0, 0, 1], l1);
        var d2 = new double[25];
        d2[0 * 5 + 1] = 1; d2[1 * 5 + 2] = 1; d2[3 * 5 + 4] = 1; d2[4 * 5 + 3] = 1;
        var g2 = CsrMatrix.FromDense(new NdArray(d2, 5, 5));
        var (c2, l2) = CsGraph.ConnectedComponents(g2, connection: ComponentConnection.Strong);
        Assert.Equal(4, c2);
        Assert.Equal([2, 1, 0, 3, 3], l2);
        var (c3, l3) = CsGraph.ConnectedComponents(g2);
        Assert.Equal(2, c3);
        Assert.Equal([0, 0, 0, 1, 1], l3);
    }

    [Fact]
    public void StrongComponents_AgreeWithBruteForceReachability()
    {
        var g = SparseMatrices.Random(40, 40, 0.04, seed: 7);
        var fw = CsGraph.FloydWarshall(g, unweighted: true).Distances.ToArray();
        var (_, labels) = CsGraph.ConnectedComponents(g, connection: ComponentConnection.Strong);
        for (var i = 0; i < 40; i++)
            for (var j = 0; j < 40; j++)
            {
                var mutual = !double.IsPositiveInfinity(fw[i * 40 + j]) && !double.IsPositiveInfinity(fw[j * 40 + i]);
                Assert.Equal(mutual, labels[i] == labels[j]);
            }
        var (_, weak) = CsGraph.ConnectedComponents(g, directed: false);
        var und = CsGraph.FloydWarshall(g, directed: false, unweighted: true).Distances.ToArray();
        for (var i = 0; i < 40; i++)
            for (var j = 0; j < 40; j++)
                Assert.Equal(!double.IsPositiveInfinity(und[i * 40 + j]), weak[i] == weak[j]);
    }

    [Fact]
    public void MinimumSpanningTree_MatchesScipy_AndBruteForcePrim()
    {
        // scipy 1.17.1 docs example: MST keeps (0,3)=3, (1,2)=2, (1,3)=5
        var x = CsrMatrix.FromDense(new NdArray([0, 8, 0, 3, 0, 0, 2, 5, 0, 0, 0, 6, 0, 0, 0, 0.0], 4, 4));
        Close(new NdArray([0, 0, 0, 3, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0.0], 4, 4), CsGraph.MinimumSpanningTree(x).ToDense(), 0);
        // Orientation follows storage: the lower-triangle version comes back lower.
        Close(new NdArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 5, 0, 0.0], 4, 4), CsGraph.MinimumSpanningTree(x.ToCsc().Transpose()).ToDense(), 0);

        var g = SparseMatrices.RandomArray(30, 30, 0.2, 5, r => r.Uniform(1, 10));
        var mst = CsGraph.MinimumSpanningTree(g);
        Assert.Equal(PrimWeight(g), mst.Sum(), 1e-12);
    }

    private static double PrimWeight(CsrMatrix g)
    {
        // Dense Prim on the symmetrised minimum weights, summed over all components.
        var n = g.Rows;
        var w = new double[n, n];
        for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) w[i, j] = Inf;
        foreach (var (i, j, v) in g.EnumerateEntries()) if (i != j) { w[i, j] = Math.Min(w[i, j], v); w[j, i] = Math.Min(w[j, i], v); }
        var inTree = new bool[n];
        var total = 0.0;
        for (var root = 0; root < n; root++)
        {
            if (inTree[root]) continue;
            var best = new double[n];
            Array.Fill(best, Inf);
            best[root] = 0;
            while (true)
            {
                var u = -1;
                for (var i = 0; i < n; i++) if (!inTree[i] && !double.IsPositiveInfinity(best[i]) && (u < 0 || best[i] < best[u])) u = i;
                if (u < 0) break;
                inTree[u] = true;
                total += best[u];
                for (var v = 0; v < n; v++) if (!inTree[v]) best[v] = Math.Min(best[v], w[u, v]);
            }
        }
        return total;
    }

    [Fact]
    public void MaximumFlow_MatchesScipy_AndIsAValidFlow()
    {
        // CLRS network; scipy 1.17.1 maximum_flow(F, 0, 5).flow_value == 23
        var f = CsrMatrix.FromDense(new NdArray([
            0, 16, 13, 0, 0, 0,
            0, 0, 10, 12, 0, 0,
            0, 4, 0, 0, 14, 0,
            0, 0, 9, 0, 0, 20,
            0, 0, 0, 7, 0, 4,
            0, 0, 0, 0, 0, 0.0], 6, 6));
        foreach (var method in Enum.GetValues<MaximumFlowMethod>())
        {
            var r = CsGraph.MaximumFlow(f, 0, 5, method);
            Assert.Equal(23.0, r.FlowValue);
            AssertValidFlow(f, r, 0, 5);
        }
        for (var seed = 0; seed < 5; seed++)
        {
            var g = SparseMatrices.RandomArray(12, 12, 0.3, seed, r => r.Next(1, 10));
            var dinic = CsGraph.MaximumFlow(g, 0, 11);
            var ek = CsGraph.MaximumFlow(g, 0, 11, MaximumFlowMethod.EdmondsKarp);
            Assert.Equal(dinic.FlowValue, ek.FlowValue);
            Assert.Equal(BruteForceMinCut(g, 0, 11), dinic.FlowValue);
            AssertValidFlow(g, dinic, 0, 11);
        }
    }

    private static void AssertValidFlow(CsrMatrix cap, MaximumFlowResult r, int s, int t)
    {
        var n = cap.Rows;
        var flow = r.Flow.ToDense().To2DArray();
        var c = cap.ToDense().To2DArray();
        for (var i = 0; i < n; i++)
        {
            var net = 0.0;
            for (var j = 0; j < n; j++)
            {
                Assert.Equal(-flow[j, i], flow[i, j]);
                Assert.True(flow[i, j] <= c[i, j] + 1e-12, $"flow {flow[i, j]} exceeds capacity {c[i, j]} on {i}->{j}");
                net += flow[i, j];
            }
            if (i == s) Assert.Equal(r.FlowValue, net, 1e-12);
            else if (i == t) Assert.Equal(-r.FlowValue, net, 1e-12);
            else Assert.Equal(0.0, net, 1e-12);
        }
    }

    private static double BruteForceMinCut(CsrMatrix g, int s, int t)
    {
        // Enumerate every s–t cut of the 12-node graph (2^10 subsets).
        var n = g.Rows;
        var best = Inf;
        var others = Enumerable.Range(0, n).Where(v => v != s && v != t).ToArray();
        for (var mask = 0; mask < 1 << others.Length; mask++)
        {
            var side = new bool[n];
            side[s] = true;
            for (var b = 0; b < others.Length; b++) if ((mask >> b & 1) == 1) side[others[b]] = true;
            var cut = 0.0;
            foreach (var (i, j, v) in g.EnumerateEntries()) if (side[i] && !side[j]) cut += v;
            best = Math.Min(best, cut);
        }
        return best;
    }

    [Fact]
    public void BipartiteMatchings_AreMaximum_AndOptimal()
    {
        for (var seed = 0; seed < 6; seed++)
        {
            var g = SparseMatrices.RandomArray(7, 8, 0.35, seed, r => r.Uniform(1, 9));
            var colToRow = CsGraph.MaximumBipartiteMatching(g);
            var rowToCol = CsGraph.MaximumBipartiteMatching(g, MatchingPermutation.Column);
            Assert.Equal(8, colToRow.Length);
            Assert.Equal(7, rowToCol.Length);
            var size = 0;
            for (var j = 0; j < 8; j++)
            {
                if (colToRow[j] < 0) continue;
                size++;
                Assert.NotEqual(0.0, g[colToRow[j], j]);
            }
            Assert.Equal(BruteMatchingSize(g), size);
            Assert.Equal(size, rowToCol.Count(c => c >= 0));
            Assert.Equal(size, CsGraph.StructuralRank(g));
        }
        // scipy docs: maximum_bipartite_matching([[0,1,1],[1,0,0],[0,0,1]]) has size 3
        var docs = CsrMatrix.FromDense(new NdArray([0, 1, 1, 1, 0, 0, 0, 0, 1.0], 3, 3));
        Assert.Equal(3, CsGraph.StructuralRank(docs));
    }

    private static int BruteMatchingSize(CsrMatrix g)
    {
        int best = 0, m = g.Rows, n = g.Columns;
        var usedCol = new bool[n];
        void Go(int row, int size)
        {
            if (row == m) { best = Math.Max(best, size); return; }
            Go(row + 1, size);
            for (var p = g.Indptr[row]; p < g.Indptr[row + 1]; p++)
            {
                var c = g.Indices[p];
                if (usedCol[c]) continue;
                usedCol[c] = true;
                Go(row + 1, size + 1);
                usedCol[c] = false;
            }
        }
        Go(0, 0);
        return best;
    }

    [Fact]
    public void MinWeightFullBipartiteMatching_MatchesPermutationBruteForce()
    {
        // scipy 1.17.1: min_weight_full_bipartite_matching(csr([[5,8,7],[9,6,1],[3,4,8]])) -> ([0,1,2], [0,2,1])
        var docs = CsrMatrix.FromDense(new NdArray([5, 8, 7, 9, 6, 1, 3, 4, 8.0], 3, 3));
        var (r0, c0) = CsGraph.MinWeightFullBipartiteMatching(docs);
        Assert.Equal([0, 1, 2], r0);
        Assert.Equal([0, 2, 1], c0); // 5 + 1 + 4 = 10 is the unique minimum
        for (var seed = 0; seed < 6; seed++)
        {
            var dense = RandomDense(5, 6, 0.75, seed + 40, -3, 9);
            var g = CsrMatrix.FromDense(dense);
            double? brute = BruteMinAssignment(g, maximize: false);
            if (brute is null)
            {
                Assert.Throws<ArgumentException>(() => CsGraph.MinWeightFullBipartiteMatching(g));
                continue;
            }
            var (rows, cols) = CsGraph.MinWeightFullBipartiteMatching(g);
            Assert.Equal(cols.Length, cols.Distinct().Count());
            Assert.Equal(brute.Value, rows.Zip(cols, (r, c) => g[r, c]).Sum(), 1e-12);
            var (rows2, cols2) = CsGraph.MinWeightFullBipartiteMatching(g, maximize: true);
            Assert.Equal(BruteMinAssignment(g, maximize: true)!.Value, rows2.Zip(cols2, (r, c) => g[r, c]).Sum(), 1e-12);
        }
    }

    private static double? BruteMinAssignment(CsrMatrix g, bool maximize)
    {
        int m = g.Rows, n = g.Columns;
        double? best = null;
        var used = new bool[n];
        void Go(int row, double total)
        {
            if (row == m) { if (best is null || (maximize ? total > best : total < best)) best = total; return; }
            for (var p = g.Indptr[row]; p < g.Indptr[row + 1]; p++)
            {
                var c = g.Indices[p];
                if (used[c]) continue;
                used[c] = true;
                Go(row + 1, total + g.Data[p]);
                used[c] = false;
            }
        }
        Go(0, 0);
        return best;
    }

    [Fact]
    public void ReverseCuthillMcKee_MatchesScipy()
    {
        // scipy 1.17.1: reverse_cuthill_mckee on the docs graph -> [3, 2, 1, 0]; symmetric_mode=True -> [2, 0, 1, 3]
        var g1 = CsrMatrix.FromDense(new NdArray([0, 1, 2, 0, 0, 0, 0, 1, 2, 0, 0, 3, 0, 0, 0, 0.0], 4, 4));
        Assert.Equal([3, 2, 1, 0], CsGraph.ReverseCuthillMcKee(g1));
        Assert.Equal([2, 0, 1, 3], CsGraph.ReverseCuthillMcKee(g1, symmetricMode: true));

        // A 4×3 grid Laplacian (zeros eliminated) under a fixed permutation.
        int[] p = [9, 2, 7, 4, 5, 11, 0, 3, 6, 10, 8, 1];
        var grid = Poisson2D(4, 3).GetRows(p).GetColumns(p);
        // NumPy's argsort is not stable (it breaks the many degree ties of a grid in SIMD-sort
        // order), so SciPy gives [5, 2, 9, 7, 8, 0, 1, 4, 10, 11, 3, 6] here and only the bandwidth
        // is comparable: ours must be no wider than SciPy's (scipy 1.17.1).
        var gridOrder = CsGraph.ReverseCuthillMcKee(grid, symmetricMode: true);
        int[] scipyOrder = [5, 2, 9, 7, 8, 0, 1, 4, 10, 11, 3, 6];
        Assert.Equal(12, gridOrder.Distinct().Count());
        Assert.True(Bandwidth(grid.GetRows(gridOrder).GetColumns(gridOrder)) <= Bandwidth(grid.GetRows(scipyOrder).GetColumns(scipyOrder)));

        var b = new double[64];
        foreach (var (i, j) in new[] { (0, 3), (3, 5), (5, 7), (1, 2), (2, 6), (6, 4), (4, 0), (7, 1), (2, 2) }) b[i * 8 + j] = 1;
        Assert.Equal([1, 2, 7, 6, 5, 4, 3, 0], CsGraph.ReverseCuthillMcKee(CsrMatrix.FromDense(new NdArray(b, 8, 8))));
        var c = new double[100];
        foreach (var (i, j) in new[] { (0, 4), (4, 7), (7, 2), (2, 9), (9, 1), (1, 5), (5, 3), (3, 8), (8, 6), (0, 6), (2, 5), (4, 4) }) c[i * 10 + j] = 1;
        Assert.Equal([1, 9, 5, 2, 3, 7, 8, 4, 6, 0], CsGraph.ReverseCuthillMcKee(CsrMatrix.FromDense(new NdArray(c, 10, 10))));
    }

    [Fact]
    public void ReverseCuthillMcKee_ShrinksBandwidth()
    {
        var p = Poisson2D(20, 20);
        var perm = new GraviRandom(3).Permutation(400);
        var shuffled = p.GetRows(perm).GetColumns(perm);
        var order = CsGraph.ReverseCuthillMcKee(shuffled, symmetricMode: true);
        Assert.Equal(400, order.Distinct().Count());
        var reordered = shuffled.GetRows(order).GetColumns(order);
        Assert.True(Bandwidth(reordered) <= 25, $"bandwidth {Bandwidth(reordered)} vs shuffled {Bandwidth(shuffled)}");
    }

    private static int Bandwidth(CsrMatrix a) => a.EnumerateEntries().Max(e => Math.Abs(e.Row - e.Column));

    [Fact]
    public void Laplacian_MatchesScipy()
    {
        var a = new NdArray([0, 2, 0, 1, 2, 0, 3, 0, 0, 3, 5, 0, 1, 0, 0, 0.0], 4, 4); // A[2,2] = 5 is ignored
        // scipy 1.17.1: laplacian(A)
        Close(new NdArray([3, -2, 0, -1, -2, 5, -3, 0, 0, -3, 3, 0, -1, 0, 0, 1.0], 4, 4), CsGraph.Laplacian(CsrMatrix.FromDense(a)).ToDense(), 0);
        // scipy 1.17.1: laplacian(A, normed=True, return_diag=True)
        var normed = CsGraph.LaplacianWithDiagonal(CsrMatrix.FromDense(a), normed: true);
        Close([1.7320508075688772, 2.23606797749979, 1.7320508075688772, 1.0], normed.Diagonal, 1e-15, "diag");
        Close(new NdArray([
            1, -0.5163977794943222, 0, -0.5773502691896258,
            -0.5163977794943222, 1, -0.7745966692414834, 0,
            0, -0.7745966692414834, 1, 0,
            -0.5773502691896258, 0, 0, 1.0], 4, 4), normed.Laplacian.ToDense(), 1e-15, "normed");
        // Directed: in-degree by default, out-degree on request; isolated nodes get zero rows.
        var b = CsrMatrix.FromDense(new NdArray([0, 2, 0, 0, 0, 0, 0, 0, 0.0], 3, 3));
        var lin = CsGraph.LaplacianWithDiagonal(b);
        Close([0.0, 2, 0], lin.Diagonal, 0);
        Close(new NdArray([0, -2, 0, 0, 2, 0, 0, 0, 0.0], 3, 3), lin.Laplacian.ToDense(), 0);
        Close([2.0, 0, 0], CsGraph.LaplacianWithDiagonal(b, useOutDegree: true).Diagonal, 0);
        var bn = CsGraph.LaplacianWithDiagonal(b, normed: true);
        Close([1.0, Math.Sqrt(2), 1], bn.Diagonal, 1e-15);
        Close(new NdArray([0, -Math.Sqrt(2), 0, 0, 1, 0, 0, 0, 0.0], 3, 3), bn.Laplacian.ToDense(), 1e-15);
    }

    [Fact]
    public void ReconstructPath_AndDistMatrix()
    {
        var r = CsGraph.ShortestPath(G5, returnPredecessors: true);
        var rebuilt = CsGraph.ConstructDistMatrix(G5, r.Predecessors!);
        Close(r.Distances, rebuilt, 0);
        var row0 = Enumerable.Range(0, 5).Select(j => r.Predecessors![0, j]).ToArray();
        var tree = CsGraph.ReconstructPath(G5, row0);
        Assert.Equal(4, tree.Nnz);
        Assert.Equal(7.0, tree.Sum()); // edges 0→2 (1), 2→1 (2), 1→3 (1), 3→4 (3)
    }
}
