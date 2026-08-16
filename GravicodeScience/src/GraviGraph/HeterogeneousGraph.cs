using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviGraph;

/// <summary>
/// An edge type, named by what it connects and how.
/// </summary>
/// <param name="Source">Node type an edge of this kind starts at.</param>
/// <param name="Relation">The relation's name.</param>
/// <param name="Target">Node type it ends at.</param>
/// <remarks>
/// The triple, not the relation name alone, is what identifies an edge type — <c>(user, rates,
/// film)</c> and <c>(critic, rates, film)</c> are different relations that happen to share a verb,
/// and a model that conflates them learns one set of weights for two behaviours.
/// </remarks>
public readonly record struct EdgeType(string Source, string Relation, string Target)
{
    /// <inheritdoc />
    public override string ToString() => $"{Source}--{Relation}->{Target}";
}

/// <summary>
/// A graph whose nodes and edges have types, each type carrying its own features.
/// </summary>
/// <remarks>
/// <para>
/// Most real graphs are not homogeneous. A recommendation graph has users and items; a citation
/// graph has papers, authors and venues; a knowledge graph has dozens of entity types and hundreds
/// of relations. Flattening them into one node set loses the thing that made them informative: that
/// "user 3 bought item 7" and "item 7 is in category 2" are different kinds of evidence and should
/// not be averaged together.
/// </para>
/// <para>
/// The structure here keeps one <see cref="Graph"/>-like adjacency per <see cref="EdgeType"/>, and
/// one feature matrix per node type. Node indices are <em>local to their type</em> — user 0 and
/// item 0 are different nodes — which is what allows each type to have a different feature
/// dimension. That is the whole reason a heterogeneous graph is not just a graph with labels.
/// </para>
/// <para>
/// Message passing over this is <see cref="RelationalConvolution"/>: one weight matrix per relation,
/// summed at the destination. That is R-GCN, and the per-relation weights are exactly what a
/// homogeneous GCN cannot express.
/// </para>
/// </remarks>
public sealed class HeterogeneousGraph
{
    private readonly Dictionary<string, int> _nodeCounts = new(StringComparer.Ordinal);
    private readonly Dictionary<string, NdArray> _features = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int[]> _labels = new(StringComparer.Ordinal);
    private readonly Dictionary<EdgeType, List<(int Source, int Target, double Weight)>> _edges = [];
    private readonly Dictionary<EdgeType, NdArray> _edgeFeatures = [];

    /// <summary>The node types present.</summary>
    public IReadOnlyCollection<string> NodeTypes => _nodeCounts.Keys;

    /// <summary>The edge types present.</summary>
    public IReadOnlyCollection<EdgeType> EdgeTypes => _edges.Keys;

    /// <summary>Total nodes across every type.</summary>
    public int NodeCount => _nodeCounts.Values.Sum();

    /// <summary>Total edges across every type.</summary>
    public int EdgeCount => _edges.Values.Sum(e => e.Count);

    /// <summary>Declares a node type and how many nodes it has.</summary>
    public HeterogeneousGraph AddNodeType(string type, int count)
    {
        ArgumentException.ThrowIfNullOrEmpty(type);
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));

        _nodeCounts[type] = Math.Max(_nodeCounts.GetValueOrDefault(type), count);
        return this;
    }

    /// <summary>How many nodes a type has.</summary>
    public int CountOf(string type) => _nodeCounts.GetValueOrDefault(type);

    /// <summary>Attaches a feature matrix to a node type, one row per node.</summary>
    /// <remarks>
    /// Each type may have a different feature dimension, which is the point: a user's features and a
    /// film's features describe different things and there is no reason for them to line up.
    /// </remarks>
    public HeterogeneousGraph SetFeatures(string type, NdArray features)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Rank != 2) throw new ArgumentException("Features must be a rank 2 array.", nameof(features));

        AddNodeType(type, features.Shape[0]);

        if (features.Shape[0] != _nodeCounts[type])
            throw new ArgumentException(
                $"Type '{type}' has {_nodeCounts[type]} nodes but {features.Shape[0]} feature rows.",
                nameof(features));

        _features[type] = features;
        return this;
    }

    /// <summary>The feature matrix of a node type.</summary>
    public NdArray? Features(string type) => _features.GetValueOrDefault(type);

    /// <summary>Attaches class labels to a node type.</summary>
    public HeterogeneousGraph SetLabels(string type, int[] labels)
    {
        AddNodeType(type, labels.Length);
        _labels[type] = labels;
        return this;
    }

    /// <summary>The labels of a node type.</summary>
    public int[]? Labels(string type) => _labels.GetValueOrDefault(type);

    /// <summary>Adds an edge of the given type.</summary>
    /// <remarks>
    /// The endpoints are indices within their own node type. Growing the type on demand matches how
    /// <see cref="Graph.AddEdge"/> behaves, so a graph can be built edge by edge without declaring
    /// its size first.
    /// </remarks>
    public HeterogeneousGraph AddEdge(EdgeType type, int source, int target, double weight = 1.0)
    {
        if (source < 0 || target < 0) throw new ArgumentOutOfRangeException(nameof(source));

        AddNodeType(type.Source, source + 1);
        AddNodeType(type.Target, target + 1);

        if (!_edges.TryGetValue(type, out var list)) _edges[type] = list = [];
        list.Add((source, target, weight));

        return this;
    }

    /// <summary>Adds an edge by naming its type inline.</summary>
    public HeterogeneousGraph AddEdge(string sourceType, string relation, string targetType,
        int source, int target, double weight = 1.0)
        => AddEdge(new EdgeType(sourceType, relation, targetType), source, target, weight);

    /// <summary>The edges of one type.</summary>
    public IReadOnlyList<(int Source, int Target, double Weight)> Edges(EdgeType type)
        => _edges.GetValueOrDefault(type) ?? [];

    /// <summary>
    /// Attaches a feature vector to every edge of a type, in the order the edges were added.
    /// </summary>
    /// <remarks>
    /// Edge features carry what the endpoints cannot: a rating's score, a transaction's amount, a
    /// timestamp. A weight is the one-dimensional case, and once there is more than one number to
    /// say about an edge, folding it into a scalar weight throws the rest away.
    /// </remarks>
    public HeterogeneousGraph SetEdgeFeatures(EdgeType type, NdArray features)
    {
        ArgumentNullException.ThrowIfNull(features);

        var count = Edges(type).Count;
        if (features.Rank != 2 || features.Shape[0] != count)
            throw new ArgumentException(
                $"Edge type {type} has {count} edges but {features.Shape[0]} feature rows.", nameof(features));

        _edgeFeatures[type] = features;
        return this;
    }

    /// <summary>The edge feature matrix of a type.</summary>
    public NdArray? EdgeFeatures(EdgeType type) => _edgeFeatures.GetValueOrDefault(type);

    /// <summary>Every edge type whose destination is the given node type.</summary>
    /// <remarks>What message passing needs: the relations that deliver messages into a type.</remarks>
    public IReadOnlyList<EdgeType> IncomingTypes(string nodeType)
        => [.. _edges.Keys.Where(t => t.Target == nodeType)];

    /// <summary>
    /// The reverse of an edge type, with its edges flipped.
    /// </summary>
    /// <remarks>
    /// Message passing only moves along edge direction, so a bipartite graph with edges only from
    /// users to films gives films no way to inform users. Adding the reverse relation is how
    /// information flows both ways — and it is added as a <em>separate</em> relation with its own
    /// name, because "user rates film" and "film is rated by user" deserve different weights.
    /// </remarks>
    public HeterogeneousGraph AddReverseEdges(EdgeType type, string? reverseName = null)
    {
        var reverse = new EdgeType(type.Target, reverseName ?? $"rev_{type.Relation}", type.Source);

        foreach (var (source, target, weight) in Edges(type))
            AddEdge(reverse, target, source, weight);

        return this;
    }

    /// <summary>
    /// Collapses everything into one homogeneous graph, forgetting the types.
    /// </summary>
    /// <param name="offsets">Where each node type's block starts in the flattened index space.</param>
    /// <remarks>
    /// The escape hatch for running an algorithm that does not understand types — PageRank,
    /// connected components, a plain GCN. It is lossy by construction: after this, a
    /// <c>bought</c> edge and a <c>viewed</c> edge are indistinguishable, which is exactly the
    /// information a heterogeneous model exists to keep.
    /// </remarks>
    public Graph ToHomogeneous(out IReadOnlyDictionary<string, int> offsets)
    {
        var starts = new Dictionary<string, int>(StringComparer.Ordinal);
        var running = 0;

        foreach (var type in _nodeCounts.Keys.OrderBy(t => t, StringComparer.Ordinal))
        {
            starts[type] = running;
            running += _nodeCounts[type];
        }

        var graph = new Graph(running, directed: true);

        foreach (var (type, edges) in _edges)
            foreach (var (source, target, weight) in edges)
                graph.AddEdge(starts[type.Source] + source, starts[type.Target] + target, weight);

        offsets = starts;
        return graph;
    }

    /// <inheritdoc />
    public override string ToString()
        => $"HeterogeneousGraph({_nodeCounts.Count} node types, {_edges.Count} edge types, " +
           $"{NodeCount} nodes, {EdgeCount} edges)";
}

/// <summary>
/// Relational graph convolution: one weight matrix per relation, summed at the destination.
/// </summary>
/// <remarks>
/// <para>
/// The heterogeneous counterpart of a GCN layer, and the R-GCN of Schlichtkrull et al. Each relation
/// gets its own transformation, so the model can learn that a <c>bought</c> edge means something
/// different from a <c>viewed</c> edge instead of averaging them into one notion of neighbourhood.
/// </para>
/// <para>
/// Normalising by in-degree per relation rather than overall is deliberate. A node with a thousand
/// <c>viewed</c> edges and three <c>bought</c> edges would otherwise have the purchases drowned out
/// entirely — and purchases are the informative signal. Per-relation normalisation gives each
/// relation an equal voice regardless of how many edges it contributes.
/// </para>
/// </remarks>
public sealed class RelationalConvolution
{
    private readonly Dictionary<EdgeType, NdArray> _relationWeights = [];
    private readonly Dictionary<string, NdArray> _selfWeights = new(StringComparer.Ordinal);

    /// <summary>Creates a layer with random weights for a graph's relations.</summary>
    /// <param name="graph">The graph whose relations to build weights for.</param>
    /// <param name="inputSizes">Feature dimension of each node type.</param>
    /// <param name="outputSize">Width of the layer's output, shared across node types.</param>
    /// <param name="rng">Source of randomness.</param>
    public RelationalConvolution(HeterogeneousGraph graph, IReadOnlyDictionary<string, int> inputSizes,
        int outputSize, GraviRandom? rng = null)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(inputSizes);

        rng ??= new GraviRandom(42);
        OutputSize = outputSize;

        foreach (var type in graph.EdgeTypes)
        {
            if (!inputSizes.TryGetValue(type.Source, out var inputSize))
                throw new ArgumentException($"No input size given for node type '{type.Source}'.", nameof(inputSizes));

            _relationWeights[type] = Glorot(inputSize, outputSize, rng);
        }

        // A self-loop weight per node type, so a node's own features survive a layer even when it
        // has no incoming edges at all. Without it an isolated node's representation is zero.
        foreach (var (type, size) in inputSizes) _selfWeights[type] = Glorot(size, outputSize, rng);
    }

    /// <summary>Width of the layer's output.</summary>
    public int OutputSize { get; }

    /// <summary>The weight matrix of one relation.</summary>
    public NdArray Weights(EdgeType type) => _relationWeights[type];

    /// <summary>
    /// Runs one layer of message passing.
    /// </summary>
    /// <param name="graph">The graph to pass messages over.</param>
    /// <param name="inputs">Current representations, one matrix per node type.</param>
    /// <param name="activation">Applied to each output element. Defaults to ReLU.</param>
    /// <returns>New representations, one matrix per node type.</returns>
    public IReadOnlyDictionary<string, NdArray> Forward(HeterogeneousGraph graph,
        IReadOnlyDictionary<string, NdArray> inputs, Func<double, double>? activation = null)
    {
        activation ??= x => Math.Max(0, x);

        var outputs = new Dictionary<string, NdArray>(StringComparer.Ordinal);

        // Start from each node's own transformed features, then add what arrives from neighbours.
        foreach (var type in graph.NodeTypes)
        {
            if (!inputs.TryGetValue(type, out var features)) continue;
            outputs[type] = LinAlg.Dot(features, _selfWeights[type]);
        }

        foreach (var edgeType in graph.EdgeTypes)
        {
            if (!inputs.TryGetValue(edgeType.Source, out var source)) continue;
            if (!outputs.TryGetValue(edgeType.Target, out var destination)) continue;

            var messages = LinAlg.Dot(source, _relationWeights[edgeType]);
            var edges = graph.Edges(edgeType);

            // In-degree within this relation only, so a sparse relation is not drowned out by a
            // dense one at the same destination.
            var inDegree = new int[destination.Shape[0]];
            foreach (var (_, target, _) in edges)
                if (target < inDegree.Length) inDegree[target]++;

            foreach (var (from, to, weight) in edges)
            {
                if (from >= messages.Shape[0] || to >= destination.Shape[0]) continue;

                var scale = weight / Math.Max(1, inDegree[to]);
                for (var d = 0; d < OutputSize; d++)
                    destination[to, d] += scale * messages[from, d];
            }
        }

        foreach (var matrix in outputs.Values)
            for (var i = 0; i < matrix.Size; i++)
                matrix.SetAt(i, activation(matrix.At(i)));

        return outputs;
    }

    /// <summary>Glorot initialisation, scaled to keep activation variance stable across a layer.</summary>
    private static NdArray Glorot(int inputSize, int outputSize, GraviRandom rng)
    {
        var limit = Math.Sqrt(6.0 / (inputSize + outputSize));
        var weights = NdArray.Zeros(inputSize, outputSize);

        for (var i = 0; i < weights.Size; i++)
            weights.SetAt(i, (rng.NextDouble() * 2 - 1) * limit);

        return weights;
    }
}
