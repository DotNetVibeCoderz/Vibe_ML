using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviGraph.Algorithms;

/// <summary>Traversal, shortest paths, ranking and centrality.</summary>
public static class GraphAlgorithms
{
    /// <summary>Nodes reachable from <paramml name="start"/>, in breadth-first order.</summary>
    /// <param name="graph">The graph to walk.</param>
    /// <param name="start">Node to start from.</param>
    public static IReadOnlyList<int> BreadthFirstSearch(Graph graph, int start)
    {
        var visited = new bool[graph.NodeCount];
        var order = new List<int>();
        var queue = new Queue<int>();

        visited[start] = true;
        queue.Enqueue(start);

        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            order.Add(node);
            foreach (var (target, _) in graph.Neighbors(node))
                if (!visited[target]) { visited[target] = true; queue.Enqueue(target); }
        }
        return order;
    }

    /// <summary>Nodes reachable from <paramref name="start"/>, in depth-first order.</summary>
    public static IReadOnlyList<int> DepthFirstSearch(Graph graph, int start)
    {
        var visited = new bool[graph.NodeCount];
        var order = new List<int>();
        var stack = new Stack<int>();
        stack.Push(start);

        while (stack.Count > 0)
        {
            var node = stack.Pop();
            if (visited[node]) continue;
            visited[node] = true;
            order.Add(node);

            // Pushing in reverse keeps the visit order matching the adjacency order.
            var neighbours = graph.Neighbors(node);
            for (var i = neighbours.Count - 1; i >= 0; i--)
                if (!visited[neighbours[i].Target]) stack.Push(neighbours[i].Target);
        }
        return order;
    }

    /// <summary>Hop counts from <paramref name="start"/>; unreachable nodes get -1.</summary>
    public static int[] HopDistances(Graph graph, int start)
    {
        var distance = new int[graph.NodeCount];
        Array.Fill(distance, -1);
        distance[start] = 0;

        var queue = new Queue<int>();
        queue.Enqueue(start);
        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            foreach (var (target, _) in graph.Neighbors(node))
                if (distance[target] < 0)
                {
                    distance[target] = distance[node] + 1;
                    queue.Enqueue(target);
                }
        }
        return distance;
    }

    /// <summary>
    /// Weighted shortest paths from one node, by Dijkstra's algorithm.
    /// </summary>
    /// <remarks>
    /// A binary heap keeps the cost at <c>O((V + E) log V)</c>; the same relaxation over a linear
    /// scan would be quadratic, which matters on the 100k-node graphs the benchmarks target.
    /// Negative weights are rejected rather than silently producing wrong answers.
    /// </remarks>
    public static (double[] Distances, int[] Previous) ShortestPaths(Graph graph, int start)
    {
        var distance = new double[graph.NodeCount];
        var previous = new int[graph.NodeCount];
        Array.Fill(distance, double.PositiveInfinity);
        Array.Fill(previous, -1);
        distance[start] = 0;

        var queue = new PriorityQueue<int, double>();
        queue.Enqueue(start, 0);
        var settled = new bool[graph.NodeCount];

        while (queue.TryDequeue(out var node, out var priority))
        {
            if (settled[node]) continue;
            settled[node] = true;

            foreach (var (target, weight) in graph.Neighbors(node))
            {
                if (weight < 0)
                    throw new InvalidOperationException("Dijkstra requires non-negative edge weights.");
                var candidate = priority + weight;
                if (candidate >= distance[target]) continue;
                distance[target] = candidate;
                previous[target] = node;
                queue.Enqueue(target, candidate);
            }
        }
        return (distance, previous);
    }

    /// <summary>The shortest path between two nodes, or an empty list when none exists.</summary>
    public static IReadOnlyList<int> ShortestPath(Graph graph, int start, int end)
    {
        var (distance, previous) = ShortestPaths(graph, start);
        if (double.IsPositiveInfinity(distance[end])) return [];

        var path = new List<int>();
        for (var node = end; node >= 0; node = previous[node]) path.Add(node);
        path.Reverse();
        return path;
    }

    /// <summary>
    /// PageRank: the stationary distribution of a random surfer who follows edges with
    /// probability <paramref name="damping"/> and teleports otherwise.
    /// </summary>
    /// <remarks>
    /// Dangling nodes - those with no outgoing edges - would leak probability mass out of the
    /// system and stop the scores summing to one, so their mass is redistributed uniformly each
    /// iteration. That detail is the difference between a correct implementation and one whose
    /// scores quietly shrink.
    /// </remarks>
    public static NdArray PageRank(Graph graph, double damping = 0.85, int maxIterations = 100,
        double tolerance = 1e-8)
    {
        var n = graph.NodeCount;
        if (n == 0) return NdArray.Zeros(0);

        var rank = NdArray.Full(1.0 / n, n);
        var outWeight = new double[n];
        for (var i = 0; i < n; i++) outWeight[i] = graph.WeightedDegree(i);

        for (var iteration = 0; iteration < maxIterations; iteration++)
        {
            var next = new double[n];
            var dangling = 0.0;

            for (var i = 0; i < n; i++)
            {
                if (outWeight[i] <= 0) { dangling += rank.At(i); continue; }
                var share = rank.At(i) / outWeight[i];
                foreach (var (target, weight) in graph.Neighbors(i)) next[target] += share * weight;
            }

            var teleport = (1.0 - damping) / n + damping * dangling / n;
            var delta = 0.0;
            for (var i = 0; i < n; i++)
            {
                var updated = teleport + damping * next[i];
                delta += Math.Abs(updated - rank.At(i));
                next[i] = updated;
            }

            for (var i = 0; i < n; i++) rank.SetAt(i, next[i]);
            if (delta < tolerance) break;
        }

        return rank;
    }

    /// <summary>PageRank biased toward a set of seed nodes.</summary>
    public static NdArray PersonalizedPageRank(Graph graph, IReadOnlyList<int> seeds,
        double damping = 0.85, int maxIterations = 100, double tolerance = 1e-8)
    {
        var n = graph.NodeCount;
        var seedSet = new HashSet<int>(seeds);
        var restart = new double[n];
        foreach (var seed in seedSet) restart[seed] = 1.0 / seedSet.Count;

        var rank = new double[n];
        Array.Copy(restart, rank, n);

        for (var iteration = 0; iteration < maxIterations; iteration++)
        {
            var next = new double[n];
            var dangling = 0.0;
            for (var i = 0; i < n; i++)
            {
                var degree = graph.WeightedDegree(i);
                if (degree <= 0) { dangling += rank[i]; continue; }
                var share = rank[i] / degree;
                foreach (var (target, weight) in graph.Neighbors(i)) next[target] += share * weight;
            }

            var delta = 0.0;
            for (var i = 0; i < n; i++)
            {
                var updated = (1 - damping) * restart[i] + damping * (next[i] + dangling * restart[i]);
                delta += Math.Abs(updated - rank[i]);
                next[i] = updated;
            }
            Array.Copy(next, rank, n);
            if (delta < tolerance) break;
        }
        return new NdArray(rank, n);
    }

    /// <summary>Degree centrality: degree normalised by the largest possible degree.</summary>
    public static NdArray DegreeCentrality(Graph graph)
    {
        var n = graph.NodeCount;
        var result = NdArray.Zeros(n);
        var denominator = Math.Max(1, n - 1);
        for (var i = 0; i < n; i++) result.SetAt(i, (double)graph.Degree(i) / denominator);
        return result;
    }

    /// <summary>
    /// Closeness centrality: the reciprocal of the mean distance to every reachable node.
    /// </summary>
    public static NdArray ClosenessCentrality(Graph graph)
    {
        var n = graph.NodeCount;
        var result = NdArray.Zeros(n);

        Parallel.For(0, n, i =>
        {
            var (distance, _) = ShortestPaths(graph, i);
            double total = 0;
            var reachable = 0;
            for (var j = 0; j < n; j++)
            {
                if (i == j || double.IsPositiveInfinity(distance[j])) continue;
                total += distance[j];
                reachable++;
            }
            // Scaling by the reachable fraction keeps scores comparable across components.
            result.SetAt(i, total <= 0 ? 0.0 : reachable / total * ((double)reachable / Math.Max(1, n - 1)));
        });
        return result;
    }

    /// <summary>
    /// Betweenness centrality by Brandes' algorithm: how often a node sits on a shortest path
    /// between two others.
    /// </summary>
    /// <remarks>
    /// Brandes runs in <c>O(VE)</c> rather than the <c>O(V³)</c> of accumulating all-pairs paths
    /// explicitly, by back-propagating dependency accumulations from each source in one pass.
    /// This implementation is unweighted (BFS-based).
    /// </remarks>
    public static NdArray BetweennessCentrality(Graph graph, bool normalize = true)
    {
        var n = graph.NodeCount;
        var centrality = new double[n];
        var padlock = new object();

        Parallel.For(0, n, source =>
        {
            var stack = new Stack<int>();
            var predecessors = new List<int>[n];
            var pathCount = new double[n];
            var distance = new int[n];
            var dependency = new double[n];

            for (var i = 0; i < n; i++) { predecessors[i] = []; distance[i] = -1; }
            pathCount[source] = 1;
            distance[source] = 0;

            var queue = new Queue<int>();
            queue.Enqueue(source);
            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                stack.Push(node);
                foreach (var (target, _) in graph.Neighbors(node))
                {
                    if (distance[target] < 0)
                    {
                        distance[target] = distance[node] + 1;
                        queue.Enqueue(target);
                    }
                    if (distance[target] == distance[node] + 1)
                    {
                        pathCount[target] += pathCount[node];
                        predecessors[target].Add(node);
                    }
                }
            }

            // Unwind in reverse BFS order so each node's dependency is complete when used.
            while (stack.Count > 0)
            {
                var node = stack.Pop();
                foreach (var predecessor in predecessors[node])
                    dependency[predecessor] += pathCount[predecessor] / pathCount[node] * (1 + dependency[node]);
                if (node != source)
                    lock (padlock) { centrality[node] += dependency[node]; }
            }
        });

        var result = new NdArray(centrality, n);
        if (!normalize || n <= 2) return graph.Directed ? result : result * 0.5;

        var scale = graph.Directed ? 1.0 / ((n - 1.0) * (n - 2.0)) : 1.0 / ((n - 1.0) * (n - 2.0));
        return result * (graph.Directed ? scale : scale);
    }

    /// <summary>
    /// Eigenvector centrality by power iteration: a node is important when its neighbours are.
    /// </summary>
    public static NdArray EigenvectorCentrality(Graph graph, int maxIterations = 200, double tolerance = 1e-10)
    {
        var n = graph.NodeCount;
        var x = NdArray.Full(1.0 / Math.Sqrt(n), n);

        for (var iteration = 0; iteration < maxIterations; iteration++)
        {
            var next = new double[n];
            for (var i = 0; i < n; i++)
                foreach (var (target, weight) in graph.Neighbors(i)) next[target] += x.At(i) * weight;

            var norm = Math.Sqrt(next.Sum(v => v * v));
            if (norm < 1e-300) break;

            var delta = 0.0;
            for (var i = 0; i < n; i++)
            {
                next[i] /= norm;
                delta += Math.Abs(next[i] - x.At(i));
            }
            for (var i = 0; i < n; i++) x.SetAt(i, next[i]);
            if (delta < tolerance) break;
        }
        return x;
    }

    /// <summary>
    /// Connected components; the returned array holds a component id per node.
    /// </summary>
    /// <remarks>
    /// On a directed graph this reports <em>weakly</em> connected components - edges are followed
    /// in both directions. That is what "connected components" means for a citation or follower
    /// network: following only out-edges would split Cora into more than a thousand fragments
    /// simply because citations point one way. Use <see cref="StronglyConnectedComponents"/> when
    /// mutual reachability is what matters.
    /// </remarks>
    public static (int Count, int[] Component) ConnectedComponents(Graph graph)
    {
        var component = new int[graph.NodeCount];
        Array.Fill(component, -1);
        var count = 0;

        for (var start = 0; start < graph.NodeCount; start++)
        {
            if (component[start] >= 0) continue;

            var queue = new Queue<int>();
            queue.Enqueue(start);
            component[start] = count;

            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                foreach (var (target, _) in graph.Neighbors(node))
                    if (component[target] < 0) { component[target] = count; queue.Enqueue(target); }

                // Following predecessors as well is what makes this weak connectivity; on an
                // undirected graph the two lists hold the same neighbours, so nothing changes.
                if (graph.Directed)
                    foreach (var (source, _) in graph.Predecessors(node))
                        if (component[source] < 0) { component[source] = count; queue.Enqueue(source); }
            }
            count++;
        }
        return (count, component);
    }

    /// <summary>
    /// Strongly connected components of a directed graph by Kosaraju's two-pass algorithm:
    /// nodes in the same component can all reach each other following edge directions.
    /// </summary>
    public static (int Count, int[] Component) StronglyConnectedComponents(Graph graph)
    {
        var n = graph.NodeCount;
        var visited = new bool[n];
        var order = new List<int>(n);

        // First pass: record nodes by finish time on the forward graph.
        for (var start = 0; start < n; start++)
        {
            if (visited[start]) continue;
            var stack = new Stack<(int Node, bool Expanded)>();
            stack.Push((start, false));

            while (stack.Count > 0)
            {
                var (node, expanded) = stack.Pop();
                if (expanded) { order.Add(node); continue; }
                if (visited[node]) continue;
                visited[node] = true;

                stack.Push((node, true));
                foreach (var (target, _) in graph.Neighbors(node))
                    if (!visited[target]) stack.Push((target, false));
            }
        }

        // Second pass: walk the reverse graph in decreasing finish time.
        var component = new int[n];
        Array.Fill(component, -1);
        var count = 0;

        for (var i = order.Count - 1; i >= 0; i--)
        {
            var start = order[i];
            if (component[start] >= 0) continue;

            var queue = new Queue<int>();
            queue.Enqueue(start);
            component[start] = count;

            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                foreach (var (source, _) in graph.Predecessors(node))
                    if (component[source] < 0) { component[source] = count; queue.Enqueue(source); }
            }
            count++;
        }
        return (count, component);
    }

    /// <summary>Number of triangles each node participates in.</summary>
    public static int[] TriangleCounts(Graph graph)
    {
        var neighbours = new HashSet<int>[graph.NodeCount];
        for (var i = 0; i < graph.NodeCount; i++)
            neighbours[i] = [.. graph.Neighbors(i).Select(t => t.Target).Where(t => t != i)];

        var counts = new int[graph.NodeCount];
        for (var i = 0; i < graph.NodeCount; i++)
            foreach (var j in neighbours[i])
            {
                if (j <= i) continue;
                foreach (var k in neighbours[j])
                {
                    if (k <= j || !neighbours[i].Contains(k)) continue;
                    counts[i]++; counts[j]++; counts[k]++;
                }
            }
        return counts;
    }

    /// <summary>
    /// Local clustering coefficient: how close each node's neighbourhood is to being a clique.
    /// </summary>
    public static NdArray ClusteringCoefficients(Graph graph)
    {
        var triangles = TriangleCounts(graph);
        var result = NdArray.Zeros(graph.NodeCount);
        for (var i = 0; i < graph.NodeCount; i++)
        {
            var degree = graph.Neighbors(i).Select(t => t.Target).Where(t => t != i).Distinct().Count();
            var possible = degree * (degree - 1) / 2.0;
            result.SetAt(i, possible <= 0 ? 0.0 : triangles[i] / possible);
        }
        return result;
    }

    /// <summary>The average of the local clustering coefficients.</summary>
    public static double AverageClusteringCoefficient(Graph graph)
        => Statistics.Mean(ClusteringCoefficients(graph));

    /// <summary>Topological order of a directed acyclic graph; empty when a cycle exists.</summary>
    public static IReadOnlyList<int> TopologicalSort(Graph graph)
    {
        var inDegree = new int[graph.NodeCount];
        for (var i = 0; i < graph.NodeCount; i++)
            foreach (var (target, _) in graph.Neighbors(i)) inDegree[target]++;

        var queue = new Queue<int>();
        for (var i = 0; i < graph.NodeCount; i++) if (inDegree[i] == 0) queue.Enqueue(i);

        var order = new List<int>();
        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            order.Add(node);
            foreach (var (target, _) in graph.Neighbors(node))
                if (--inDegree[target] == 0) queue.Enqueue(target);
        }
        return order.Count == graph.NodeCount ? order : [];
    }

    /// <summary>
    /// Community detection by label propagation: each node repeatedly adopts the most common
    /// label among its neighbours until the assignment stabilises.
    /// </summary>
    public static int[] LabelPropagation(Graph graph, int maxIterations = 100, int seed = 42)
    {
        var labels = Enumerable.Range(0, graph.NodeCount).ToArray();
        var rng = new GraviRandom(seed);

        for (var iteration = 0; iteration < maxIterations; iteration++)
        {
            var order = rng.Permutation(graph.NodeCount);
            var changed = false;

            foreach (var node in order)
            {
                var counts = new Dictionary<int, double>();
                foreach (var (target, weight) in graph.Neighbors(node))
                {
                    counts.TryGetValue(labels[target], out var c);
                    counts[labels[target]] = c + weight;
                }
                if (counts.Count == 0) continue;

                var best = counts.MaxBy(kv => kv.Value).Key;
                if (best != labels[node]) { labels[node] = best; changed = true; }
            }
            if (!changed) break;
        }

        // Renumber so the ids are contiguous from zero.
        var map = labels.Distinct().OrderBy(l => l).Select((l, i) => (l, i)).ToDictionary(p => p.l, p => p.i);
        return labels.Select(l => map[l]).ToArray();
    }

    /// <summary>Modularity of a partition; higher means denser communities than chance predicts.</summary>
    public static double Modularity(Graph graph, IReadOnlyList<int> communities)
    {
        var totalWeight = graph.Edges().Sum(e => e.Weight);
        if (totalWeight <= 0) return 0.0;
        var m2 = 2.0 * totalWeight;

        var degree = new double[graph.NodeCount];
        for (var i = 0; i < graph.NodeCount; i++) degree[i] = graph.WeightedDegree(i);

        var modularity = 0.0;
        for (var i = 0; i < graph.NodeCount; i++)
            for (var j = 0; j < graph.NodeCount; j++)
            {
                if (communities[i] != communities[j]) continue;
                var a = graph.Neighbors(i).Where(n => n.Target == j).Sum(n => n.Weight);
                modularity += a - degree[i] * degree[j] / m2;
            }
        return modularity / m2;
    }
}
