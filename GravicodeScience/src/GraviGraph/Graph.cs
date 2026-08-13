using System.Text.Json;
using System.Text.Json.Serialization;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviGraph;

/// <summary>An edge between two nodes.</summary>
/// <param name="Source">Origin node index.</param>
/// <param name="Target">Destination node index.</param>
/// <param name="Weight">Edge weight; 1 for an unweighted graph.</param>
public readonly record struct Edge(int Source, int Target, double Weight = 1.0);

/// <summary>
/// A graph stored as an adjacency list, with optional node features and labels.
/// </summary>
/// <remarks>
/// The adjacency list is the primary structure because graph work is overwhelmingly
/// neighbourhood-local: PageRank, BFS and message passing all iterate a node's neighbours, which
/// costs <c>O(degree)</c> here and <c>O(nodes)</c> on a dense matrix. Real networks are sparse -
/// Cora has 2708 nodes and 5429 edges, so a dense matrix would be 99.9% zeros - which is also why
/// <see cref="ToSparseAdjacency"/> exists and <see cref="ToDenseAdjacency"/> carries a warning.
/// </remarks>
public sealed class Graph
{
    private readonly List<List<(int Target, double Weight)>> _adjacency;
    private readonly List<List<(int Source, double Weight)>> _reverse;

    /// <summary>Creates an empty graph.</summary>
    public Graph(bool directed = false)
    {
        Directed = directed;
        _adjacency = [];
        _reverse = [];
        NodeLabels = [];
        NodeNames = [];
    }

    /// <summary>Creates a graph with <paramref name="nodeCount"/> isolated nodes.</summary>
    public Graph(int nodeCount, bool directed = false) : this(directed)
    {
        for (var i = 0; i < nodeCount; i++) AddNode();
    }

    /// <summary>True when edges have a direction.</summary>
    public bool Directed { get; }

    /// <summary>Number of nodes.</summary>
    public int NodeCount => _adjacency.Count;

    /// <summary>Number of edges. Undirected edges are counted once.</summary>
    public int EdgeCount { get; private set; }

    /// <summary>Optional class label per node, or -1 when unlabelled.</summary>
    public List<int> NodeLabels { get; }

    /// <summary>Optional display name per node.</summary>
    public List<string> NodeNames { get; }

    /// <summary>Optional node feature matrix, one row per node.</summary>
    public NdArray? NodeFeatures { get; set; }

    /// <summary>Distinct class names, when the graph came with labels.</summary>
    public IReadOnlyList<string> Classes { get; set; } = [];

    /// <summary>Adds a node and returns its index.</summary>
    public int AddNode(string? name = null, int label = -1)
    {
        _adjacency.Add([]);
        _reverse.Add([]);
        NodeLabels.Add(label);
        NodeNames.Add(name ?? _adjacency.Count.ToString());
        return _adjacency.Count - 1;
    }

    /// <summary>Adds an edge, growing the node set if an endpoint does not exist yet.</summary>
    public Graph AddEdge(int source, int target, double weight = 1.0)
    {
        while (NodeCount <= Math.Max(source, target)) AddNode();

        _adjacency[source].Add((target, weight));
        _reverse[target].Add((source, weight));
        if (!Directed && source != target)
        {
            _adjacency[target].Add((source, weight));
            _reverse[source].Add((target, weight));
        }
        EdgeCount++;
        return this;
    }

    /// <summary>Outgoing neighbours of a node.</summary>
    public IReadOnlyList<(int Target, double Weight)> Neighbors(int node) => _adjacency[node];

    /// <summary>Incoming neighbours of a node; the same as <see cref="Neighbors"/> when undirected.</summary>
    public IReadOnlyList<(int Source, double Weight)> Predecessors(int node) => _reverse[node];

    /// <summary>Number of outgoing edges.</summary>
    public int OutDegree(int node) => _adjacency[node].Count;

    /// <summary>Number of incoming edges.</summary>
    public int InDegree(int node) => _reverse[node].Count;

    /// <summary>Degree of a node; out-degree for a directed graph.</summary>
    public int Degree(int node) => Directed ? OutDegree(node) : _adjacency[node].Count;

    /// <summary>Sum of the weights on a node's outgoing edges.</summary>
    public double WeightedDegree(int node) => _adjacency[node].Sum(n => n.Weight);

    /// <summary>True when an edge exists from <paramref name="source"/> to <paramref name="target"/>.</summary>
    public bool HasEdge(int source, int target) => _adjacency[source].Any(n => n.Target == target);

    /// <summary>Every edge, each undirected edge appearing once.</summary>
    public IEnumerable<Edge> Edges()
    {
        for (var i = 0; i < NodeCount; i++)
            foreach (var (target, weight) in _adjacency[i])
                if (Directed || i <= target)
                    yield return new Edge(i, target, weight);
    }

    /// <summary>Degree of every node, as an array.</summary>
    public NdArray Degrees()
    {
        var result = NdArray.Zeros(NodeCount);
        for (var i = 0; i < NodeCount; i++) result.SetAt(i, Degree(i));
        return result;
    }

    /// <summary>Fraction of possible edges that exist.</summary>
    public double Density
    {
        get
        {
            if (NodeCount < 2) return 0.0;
            var possible = (double)NodeCount * (NodeCount - 1);
            if (!Directed) possible /= 2;
            return EdgeCount / possible;
        }
    }

    /// <summary>
    /// The adjacency matrix in sparse form; this is the representation the GNN layers multiply by.
    /// </summary>
    public SparseMatrix ToSparseAdjacency(bool addSelfLoops = false, bool symmetricNormalize = false)
    {
        var triplets = new List<(int, int, double)>(EdgeCount * 2 + NodeCount);
        for (var i = 0; i < NodeCount; i++)
            foreach (var (target, weight) in _adjacency[i])
                triplets.Add((i, target, weight));

        if (addSelfLoops)
            for (var i = 0; i < NodeCount; i++) triplets.Add((i, i, 1.0));

        if (!symmetricNormalize) return SparseMatrix.FromTriplets(NodeCount, NodeCount, triplets);

        // D^-1/2 (A + I) D^-1/2, the propagation matrix from the GCN paper. Normalising keeps a
        // high-degree node's messages from swamping its neighbours and keeps activations stable
        // across layers.
        var degree = new double[NodeCount];
        foreach (var (row, _, weight) in triplets) degree[row] += weight;

        var normalized = triplets
            .Select(t => (t.Item1, t.Item2, t.Item3 / Math.Sqrt(Math.Max(degree[t.Item1], 1e-12) * Math.Max(degree[t.Item2], 1e-12))))
            .ToList();
        return SparseMatrix.FromTriplets(NodeCount, NodeCount, normalized);
    }

    /// <summary>
    /// The adjacency matrix in dense form. Costs <c>NodeCount²</c> doubles, so this is for small
    /// graphs and visualisation only.
    /// </summary>
    public NdArray ToDenseAdjacency()
    {
        if ((long)NodeCount * NodeCount > 50_000_000)
            throw new InvalidOperationException(
                $"A dense adjacency matrix for {NodeCount} nodes needs {(long)NodeCount * NodeCount * 8 / 1_048_576} MB. Use ToSparseAdjacency instead.");

        var matrix = NdArray.Zeros(NodeCount, NodeCount);
        for (var i = 0; i < NodeCount; i++)
            foreach (var (target, weight) in _adjacency[i]) matrix[i, target] = weight;
        return matrix;
    }

    /// <summary>The graph Laplacian <c>D - A</c>.</summary>
    public NdArray Laplacian()
    {
        var laplacian = ToDenseAdjacency() * -1.0;
        for (var i = 0; i < NodeCount; i++) laplacian[i, i] += WeightedDegree(i);
        return laplacian;
    }

    /// <summary>
    /// An undirected copy in which every edge can be traversed both ways.
    /// </summary>
    /// <remarks>
    /// Random-walk embeddings and most community-detection methods assume they can move freely
    /// between related nodes. On a citation graph, walking only along the direction of citation
    /// leaves most walks stranded after a step or two, so DeepWalk and node2vec are conventionally
    /// run on the undirected view.
    /// </remarks>
    public Graph AsUndirected()
    {
        if (!Directed) return this;

        var result = new Graph(NodeCount, directed: false)
        {
            NodeFeatures = NodeFeatures,
            Classes = Classes,
        };
        for (var i = 0; i < NodeCount; i++)
        {
            result.NodeNames[i] = NodeNames[i];
            result.NodeLabels[i] = NodeLabels[i];
        }

        var seen = new HashSet<(int, int)>();
        for (var i = 0; i < NodeCount; i++)
            foreach (var (target, weight) in _adjacency[i])
            {
                var key = i <= target ? (i, target) : (target, i);
                if (seen.Add(key)) result.AddEdge(key.Item1, key.Item2, weight);
            }
        return result;
    }

    /// <summary>The subgraph induced by a set of nodes, renumbered from zero.</summary>
    public Graph Subgraph(IReadOnlyList<int> nodes)
    {
        var map = nodes.Select((n, i) => (n, i)).ToDictionary(p => p.n, p => p.i);
        var result = new Graph(nodes.Count, Directed);

        for (var i = 0; i < nodes.Count; i++)
        {
            result.NodeNames[i] = NodeNames[nodes[i]];
            result.NodeLabels[i] = NodeLabels[nodes[i]];
        }

        foreach (var node in nodes)
            foreach (var (target, weight) in _adjacency[node])
                if (map.TryGetValue(target, out var mapped) && (Directed || map[node] <= mapped))
                    result.AddEdge(map[node], mapped, weight);

        if (NodeFeatures is not null) result.NodeFeatures = NodeFeatures.Take(nodes);
        result.Classes = Classes;
        return result;
    }

    // ---------------------------------------------------------------- generators

    /// <summary>An Erdos-Renyi random graph where each possible edge exists with probability <paramref name="p"/>.</summary>
    public static Graph Random(int nodes, double p, int seed = 42, bool directed = false)
    {
        var rng = new GraviRandom(seed);
        var graph = new Graph(nodes, directed);
        for (var i = 0; i < nodes; i++)
            for (var j = directed ? 0 : i + 1; j < nodes; j++)
            {
                if (i == j) continue;
                if (rng.NextDouble() < p) graph.AddEdge(i, j);
            }
        return graph;
    }

    /// <summary>
    /// A Barabasi-Albert graph, grown by preferential attachment so the degree distribution
    /// follows a power law - the shape most real networks actually have.
    /// </summary>
    public static Graph ScaleFree(int nodes, int edgesPerNode = 2, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var graph = new Graph(Math.Min(edgesPerNode + 1, nodes));

        // Seed with a small complete graph so there is something to attach to.
        for (var i = 0; i < graph.NodeCount; i++)
            for (var j = i + 1; j < graph.NodeCount; j++) graph.AddEdge(i, j);

        var targets = new List<int>();
        for (var i = 0; i < graph.NodeCount; i++)
            for (var d = 0; d < graph.Degree(i); d++) targets.Add(i);

        for (var n = graph.NodeCount; n < nodes; n++)
        {
            var node = graph.AddNode();
            var chosen = new HashSet<int>();
            while (chosen.Count < Math.Min(edgesPerNode, targets.Count))
            {
                // Sampling from the repeated-endpoint list is sampling proportional to degree.
                var candidate = targets[rng.Next(targets.Count)];
                if (candidate != node) chosen.Add(candidate);
            }
            foreach (var target in chosen)
            {
                graph.AddEdge(node, target);
                targets.Add(node);
                targets.Add(target);
            }
        }
        return graph;
    }

    /// <summary>A ring of <paramref name="nodes"/> nodes, each joined to the next.</summary>
    public static Graph Cycle(int nodes)
    {
        var graph = new Graph(nodes);
        for (var i = 0; i < nodes; i++) graph.AddEdge(i, (i + 1) % nodes);
        return graph;
    }

    /// <summary>A complete graph on <paramref name="nodes"/> nodes.</summary>
    public static Graph Complete(int nodes)
    {
        var graph = new Graph(nodes);
        for (var i = 0; i < nodes; i++)
            for (var j = i + 1; j < nodes; j++) graph.AddEdge(i, j);
        return graph;
    }

    /// <summary>
    /// Disconnected communities with dense internal and sparse external connections, useful for
    /// testing clustering and embedding methods against a known answer.
    /// </summary>
    public static Graph Communities(int communities, int sizePerCommunity,
        double internalP = 0.4, double externalP = 0.01, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var total = communities * sizePerCommunity;
        var graph = new Graph(total);

        for (var i = 0; i < total; i++)
        {
            graph.NodeLabels[i] = i / sizePerCommunity;
            for (var j = i + 1; j < total; j++)
            {
                var sameCommunity = i / sizePerCommunity == j / sizePerCommunity;
                if (rng.NextDouble() < (sameCommunity ? internalP : externalP)) graph.AddEdge(i, j);
            }
        }
        graph.Classes = Enumerable.Range(0, communities).Select(c => $"community{c}").ToArray();
        return graph;
    }

    // ---------------------------------------------------------------- IO

    /// <summary>The JSON shape read and written by <see cref="Load"/> and <see cref="Save"/>.</summary>
    private sealed record GraphPayload(
        string? Name,
        string? Description,
        bool Directed,
        int FeatureDimension,
        string[]? Classes,
        NodePayload[] Nodes,
        int[][] Edges);

    private sealed record NodePayload(
        int Id,
        [property: JsonPropertyName("paperId")] string? PaperId,
        int Label,
        int[]? Features);

    /// <summary>
    /// Reads a graph from JSON.
    /// </summary>
    /// <remarks>
    /// Node features are stored as the indices of the non-zero entries rather than as a full
    /// vector. Cora's features are 1433-dimensional but average about eighteen non-zeros, so the
    /// sparse form is roughly eighty times smaller and loads correspondingly faster.
    /// </remarks>
    public static Graph Load(string path)
    {
        var payload = JsonSerializer.Deserialize<GraphPayload>(File.ReadAllText(path),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidDataException($"'{path}' does not contain a graph.");

        var graph = new Graph(payload.Nodes.Length, payload.Directed)
        {
            Classes = payload.Classes ?? [],
        };

        for (var i = 0; i < payload.Nodes.Length; i++)
        {
            graph.NodeLabels[i] = payload.Nodes[i].Label;
            graph.NodeNames[i] = payload.Nodes[i].PaperId ?? i.ToString();
        }

        foreach (var edge in payload.Edges)
            if (edge.Length >= 2) graph.AddEdge(edge[0], edge[1], edge.Length > 2 ? edge[2] : 1.0);

        if (payload.FeatureDimension > 0)
        {
            var features = NdArray.Zeros(payload.Nodes.Length, payload.FeatureDimension);
            for (var i = 0; i < payload.Nodes.Length; i++)
                foreach (var index in payload.Nodes[i].Features ?? [])
                    if (index >= 0 && index < payload.FeatureDimension) features[i, index] = 1.0;
            graph.NodeFeatures = features;
        }

        return graph;
    }

    /// <summary>Writes the graph as JSON in the same format <see cref="Load"/> reads.</summary>
    public void Save(string path, string? name = null, string? description = null)
    {
        var nodes = new NodePayload[NodeCount];
        for (var i = 0; i < NodeCount; i++)
        {
            int[]? features = null;
            if (NodeFeatures is not null)
                features = Enumerable.Range(0, NodeFeatures.Shape[1])
                    .Where(j => NodeFeatures[i, j] != 0)
                    .ToArray();
            nodes[i] = new NodePayload(i, NodeNames[i], NodeLabels[i], features);
        }

        var payload = new GraphPayload(
            name, description, Directed,
            NodeFeatures?.Shape[1] ?? 0,
            Classes.ToArray(),
            nodes,
            Edges().Select(e => new[] { e.Source, e.Target }).ToArray());

        File.WriteAllText(path, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = false }));
    }

    /// <summary>Reads a whitespace or comma separated edge list.</summary>
    public static Graph LoadEdgeList(string path, bool directed = false, char separator = ' ')
    {
        var graph = new Graph(directed);
        foreach (var line in File.ReadLines(path))
        {
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var parts = line.Split(separator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (parts.Length < 2) continue;
            graph.AddEdge(int.Parse(parts[0]), int.Parse(parts[1]),
                parts.Length > 2 ? double.Parse(parts[2]) : 1.0);
        }
        return graph;
    }

    /// <inheritdoc />
    public override string ToString()
        => $"Graph({(Directed ? "directed" : "undirected")}, nodes={NodeCount}, edges={EdgeCount}, density={Density:P3})";
}
