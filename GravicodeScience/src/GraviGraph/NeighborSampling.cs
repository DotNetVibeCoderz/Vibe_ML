using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviGraph;

/// <summary>
/// A sampled computation graph for a batch of target nodes.
/// </summary>
/// <param name="Nodes">
/// Every node needed to compute the batch, ordered so that a node's neighbours precede it in the
/// layer ordering.
/// </param>
/// <param name="Layers">
/// One edge list per hop, outermost first. Layer 0 delivers messages into the frontier furthest from
/// the targets.
/// </param>
/// <param name="TargetPositions">Where each requested target sits in <paramref name="Nodes"/>.</param>
public readonly record struct SampledBlock(
    int[] Nodes,
    IReadOnlyList<(int Source, int Target)[]> Layers,
    int[] TargetPositions)
{
    /// <summary>How many nodes the batch touches.</summary>
    public int NodeCount => Nodes.Length;

    /// <summary>How many edges the batch touches.</summary>
    public int EdgeCount => Layers.Sum(l => l.Length);
}

/// <summary>
/// Neighbourhood sampling: training a GNN on a graph too large to hold in memory at once.
/// </summary>
/// <remarks>
/// <para>
/// Full-batch message passing computes every node's representation in every layer, so one step needs
/// the whole graph. That is fine for Cora and impossible for a social network. Sampling replaces it
/// with per-batch computation: to update a hundred nodes, take a bounded sample of their
/// neighbours, then a bounded sample of <em>those</em> neighbours, and compute only that.
/// </para>
/// <para>
/// The problem it actually solves is not memory but <b>neighbourhood explosion</b>. A two-layer GNN
/// on a graph with average degree 100 touches ten thousand nodes per target; three layers touches a
/// million. Capping the fan-out per hop — GraphSAGE's contribution — makes the cost per target
/// bounded and independent of the graph's size, which is what turns an intractable model into a
/// trainable one.
/// </para>
/// <para>
/// <b>Sampling changes the estimator, not just the speed.</b> Each node's aggregate is now a
/// stochastic estimate of the full-neighbourhood one, unbiased for a mean aggregator and noisier for
/// a small fan-out. Very small samples make training unstable rather than merely approximate, which
/// is the trade-off <see cref="Sample"/>'s fan-out parameter controls.
/// </para>
/// </remarks>
public static class NeighborSampler
{
    /// <summary>
    /// Builds the computation graph for a batch of target nodes.
    /// </summary>
    /// <param name="graph">The graph to sample from.</param>
    /// <param name="targets">The nodes whose representations are wanted.</param>
    /// <param name="fanOut">
    /// How many neighbours to keep per node at each hop, outermost first. Its length is the number
    /// of layers.
    /// </param>
    /// <param name="rng">Source of randomness.</param>
    /// <param name="replace">
    /// Whether to sample with replacement. With replacement every node gets exactly the requested
    /// fan-out, which keeps tensors rectangular; without, a low-degree node contributes fewer.
    /// </param>
    /// <remarks>
    /// Built outward from the targets and then reversed, because the layer that must run first is
    /// the one furthest from them — a node's representation at layer <c>k</c> needs its neighbours'
    /// representations at layer <c>k−1</c>.
    /// </remarks>
    public static SampledBlock Sample(Graph graph, IReadOnlyList<int> targets,
        IReadOnlyList<int> fanOut, GraviRandom? rng = null, bool replace = false)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(targets);
        ArgumentNullException.ThrowIfNull(fanOut);

        if (fanOut.Count == 0) throw new ArgumentException("At least one hop is needed.", nameof(fanOut));
        rng ??= new GraviRandom(42);

        // Local indices, so the block can be computed against a compact feature matrix rather than
        // against the whole graph's.
        var index = new Dictionary<int, int>();
        var nodes = new List<int>();

        int Local(int node)
        {
            if (index.TryGetValue(node, out var existing)) return existing;

            index[node] = nodes.Count;
            nodes.Add(node);
            return nodes.Count - 1;
        }

        var frontier = new List<int>();
        foreach (var target in targets)
        {
            if (target < 0 || target >= graph.NodeCount)
                throw new ArgumentOutOfRangeException(nameof(targets), $"Node {target} is not in the graph.");

            Local(target);
            frontier.Add(target);
        }

        var targetPositions = targets.Select(t => index[t]).ToArray();
        var layers = new List<(int Source, int Target)[]>();

        // Walk outward: the current frontier's neighbours become the next frontier.
        foreach (var hop in fanOut)
        {
            var edges = new List<(int, int)>();
            var next = new HashSet<int>();

            foreach (var node in frontier)
            {
                foreach (var neighbour in Choose(graph.Neighbors(node), hop, rng, replace))
                {
                    edges.Add((Local(neighbour), index[node]));
                    next.Add(neighbour);
                }
            }

            layers.Add([.. edges]);
            frontier = [.. next];

            // Nothing further out to sample; stopping early is honest about the graph's extent.
            if (frontier.Count == 0) break;
        }

        // Outermost layer first, because that is the order the layers must be evaluated in.
        layers.Reverse();

        return new SampledBlock([.. nodes], layers, targetPositions);
    }

    /// <summary>
    /// Splits the nodes into shuffled mini-batches.
    /// </summary>
    /// <remarks>
    /// Shuffling matters more here than in ordinary mini-batching. Node ids in a real graph are
    /// rarely arbitrary — they often follow crawl order, so consecutive ids are neighbours — and an
    /// unshuffled batch is then a single dense region rather than a sample of the graph.
    /// </remarks>
    public static IEnumerable<int[]> Batches(IReadOnlyList<int> nodes, int batchSize, GraviRandom? rng = null)
    {
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));

        var order = nodes.ToArray();

        if (rng is not null)
            for (var i = order.Length - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

        for (var start = 0; start < order.Length; start += batchSize)
            yield return order[start..Math.Min(start + batchSize, order.Length)];
    }

    /// <summary>
    /// Runs mean aggregation over a sampled block.
    /// </summary>
    /// <param name="block">The sampled computation graph.</param>
    /// <param name="features">Features for the block's nodes, one row per entry in its node list.</param>
    /// <param name="weights">One weight matrix per layer.</param>
    /// <param name="activation">Applied element-wise. Defaults to ReLU.</param>
    /// <returns>Representations for the block's target nodes only.</returns>
    /// <remarks>
    /// GraphSAGE's mean aggregator: a node's new representation combines its own features with the
    /// average of its sampled neighbours'. Keeping the node's own contribution separate from the
    /// neighbourhood average — rather than including it in the mean — is what lets the model tell a
    /// node apart from its surroundings, and it is the difference between SAGE and a plain GCN.
    /// </remarks>
    public static NdArray Aggregate(SampledBlock block, NdArray features,
        IReadOnlyList<NdArray> weights, Func<double, double>? activation = null)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(weights);

        if (weights.Count != block.Layers.Count)
            throw new ArgumentException(
                $"The block has {block.Layers.Count} layers but {weights.Count} weight matrices were given.",
                nameof(weights));

        activation ??= x => Math.Max(0, x);
        var x = features.Copy();

        for (var layer = 0; layer < block.Layers.Count; layer++)
        {
            var width = weights[layer].Shape[1];
            var aggregated = NdArray.Zeros(x.Shape[0], x.Shape[1]);
            var counts = new int[x.Shape[0]];

            foreach (var (source, target) in block.Layers[layer])
            {
                if (source >= x.Shape[0] || target >= x.Shape[0]) continue;

                for (var d = 0; d < x.Shape[1]; d++) aggregated[target, d] += x[source, d];
                counts[target]++;
            }

            // Concatenating self and neighbourhood, then projecting. A node with no sampled
            // neighbours keeps only its own half, which is the right answer rather than a zero.
            var combined = NdArray.Zeros(x.Shape[0], x.Shape[1] * 2);

            for (var node = 0; node < x.Shape[0]; node++)
                for (var d = 0; d < x.Shape[1]; d++)
                {
                    combined[node, d] = x[node, d];
                    combined[node, x.Shape[1] + d] = counts[node] > 0 ? aggregated[node, d] / counts[node] : 0.0;
                }

            if (weights[layer].Shape[0] != combined.Shape[1])
                throw new ArgumentException(
                    $"Layer {layer} expects a {weights[layer].Shape[0]}-wide input but the concatenation " +
                    $"is {combined.Shape[1]} wide. A SAGE layer takes twice its feature width.", nameof(weights));

            var next = LinAlg.Dot(combined, weights[layer]);
            for (var i = 0; i < next.Size; i++) next.SetAt(i, activation(next.At(i)));

            x = next;
        }

        var result = NdArray.Zeros(block.TargetPositions.Length, x.Shape[1]);
        for (var i = 0; i < block.TargetPositions.Length; i++)
            for (var d = 0; d < x.Shape[1]; d++)
                result[i, d] = x[block.TargetPositions[i], d];

        return result;
    }

    /// <summary>Gathers the rows of a full feature matrix that a block needs.</summary>
    /// <remarks>
    /// The point of the whole exercise: only these rows have to be in memory, and there are at most
    /// <c>batch × ∏ fanOut</c> of them however large the graph is.
    /// </remarks>
    public static NdArray GatherFeatures(SampledBlock block, NdArray features)
    {
        var result = NdArray.Zeros(block.Nodes.Length, features.Shape[1]);

        for (var i = 0; i < block.Nodes.Length; i++)
            for (var d = 0; d < features.Shape[1]; d++)
                result[i, d] = features[block.Nodes[i], d];

        return result;
    }

    /// <summary>Picks up to <paramref name="count"/> neighbours.</summary>
    private static IEnumerable<int> Choose(IReadOnlyList<(int Target, double Weight)> neighbours,
        int count, GraviRandom rng, bool replace)
    {
        if (neighbours.Count == 0) yield break;

        if (count <= 0 || (!replace && count >= neighbours.Count))
        {
            // Fewer neighbours than the cap: take them all, and sample nothing.
            foreach (var (target, _) in neighbours) yield return target;
            yield break;
        }

        if (replace)
        {
            for (var i = 0; i < count; i++) yield return neighbours[rng.Next(neighbours.Count)].Target;
            yield break;
        }

        // Without replacement: a partial Fisher-Yates over the indices, so no neighbour repeats and
        // the cost is the sample size rather than the degree.
        var order = Enumerable.Range(0, neighbours.Count).ToArray();

        for (var i = 0; i < count; i++)
        {
            var j = i + rng.Next(order.Length - i);
            (order[i], order[j]) = (order[j], order[i]);
            yield return neighbours[order[i]].Target;
        }
    }
}
