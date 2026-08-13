using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviGraph;

/// <summary>An edge that exists at a point in time.</summary>
/// <param name="Source">Origin node.</param>
/// <param name="Target">Destination node.</param>
/// <param name="Time">When the interaction happened.</param>
/// <param name="Weight">Edge weight.</param>
public readonly record struct TemporalEdge(int Source, int Target, double Time, double Weight = 1.0);

/// <summary>
/// A graph whose edges are timestamped events rather than standing facts.
/// </summary>
/// <remarks>
/// <para>
/// Most graphs people call static are nothing of the sort — a transaction network, a message log, a
/// citation record are all sequences of events, and collapsing them into one adjacency matrix
/// destroys the ordering. That matters more than it looks: in a static graph an edge <c>a→b</c> and
/// an edge <c>b→c</c> imply a path from <c>a</c> to <c>c</c>, but if <c>b→c</c> happened
/// <em>before</em> <c>a→b</c>, nothing could have travelled that way. Information, money and disease
/// all obey that ordering, and a static analysis systematically overstates what is reachable.
/// </para>
/// <para>
/// <see cref="TemporallyReachable"/> respects it; <see cref="Snapshot"/> and
/// <see cref="Collapse"/> deliberately do not, and exist for when a static view is what is wanted.
/// </para>
/// </remarks>
public sealed class TemporalGraph
{
    private readonly List<TemporalEdge> _edges = [];
    private bool _sorted = true;

    /// <summary>Creates an empty temporal graph.</summary>
    public TemporalGraph(bool directed = true) => Directed = directed;

    /// <summary>True when edges have a direction.</summary>
    public bool Directed { get; }

    /// <summary>Number of nodes, taken as one more than the largest index used.</summary>
    public int NodeCount { get; private set; }

    /// <summary>Number of timestamped edges.</summary>
    public int EdgeCount => _edges.Count;

    /// <summary>The edges, in time order.</summary>
    public IReadOnlyList<TemporalEdge> Edges
    {
        get
        {
            EnsureSorted();
            return _edges;
        }
    }

    /// <summary>The earliest and latest times any edge occurs.</summary>
    public (double Start, double End) TimeRange
        => _edges.Count == 0 ? (0, 0) : (_edges.Min(e => e.Time), _edges.Max(e => e.Time));

    /// <summary>Adds a timestamped edge.</summary>
    public TemporalGraph AddEdge(int source, int target, double time, double weight = 1.0)
    {
        if (source < 0 || target < 0) throw new ArgumentOutOfRangeException(nameof(source));
        if (double.IsNaN(time)) throw new ArgumentException("An edge needs a time.", nameof(time));

        _edges.Add(new TemporalEdge(source, target, time, weight));
        NodeCount = Math.Max(NodeCount, Math.Max(source, target) + 1);
        _sorted = false;

        return this;
    }

    /// <summary>
    /// The graph as it stood over a time window.
    /// </summary>
    /// <param name="from">Inclusive start.</param>
    /// <param name="to">Inclusive end.</param>
    /// <remarks>
    /// The usual way to apply a static algorithm to a dynamic graph: slice it into windows and run
    /// the algorithm on each. It works, and it silently assumes that everything inside a window
    /// happened simultaneously — which is only harmless when the window is short relative to how
    /// fast the graph changes.
    /// </remarks>
    public Graph Snapshot(double from, double to)
    {
        var graph = new Graph(NodeCount, Directed);

        foreach (var edge in _edges)
            if (edge.Time >= from && edge.Time <= to)
                graph.AddEdge(edge.Source, edge.Target, edge.Weight);

        return graph;
    }

    /// <summary>Every edge up to and including <paramref name="time"/>.</summary>
    /// <remarks>
    /// What a model may see when predicting at that moment. Training a link predictor on a static
    /// graph and testing it on later edges leaks the answer unless the training graph is cut here.
    /// </remarks>
    public Graph SnapshotUpTo(double time) => Snapshot(double.NegativeInfinity, time);

    /// <summary>A sequence of snapshots covering the whole time range.</summary>
    public IReadOnlyList<Graph> Windows(int count)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));

        var (start, end) = TimeRange;
        var width = (end - start) / count;
        if (width <= 0) return [Snapshot(start, end)];

        var snapshots = new List<Graph>(count);
        for (var i = 0; i < count; i++)
        {
            var from = start + i * width;
            // The last window's upper bound is inclusive of the final edge, which a half-open
            // interval would drop.
            var to = i == count - 1 ? end : from + width;
            snapshots.Add(Snapshot(from, to));
        }

        return snapshots;
    }

    /// <summary>Every edge as one static graph, ignoring time entirely.</summary>
    public Graph Collapse() => Snapshot(double.NegativeInfinity, double.PositiveInfinity);

    /// <summary>
    /// The nodes reachable from a source along a path whose edge times increase.
    /// </summary>
    /// <param name="source">Where to start.</param>
    /// <param name="startTime">The earliest edge that may be used.</param>
    /// <param name="maxGap">
    /// The longest wait allowed between consecutive edges on a path. Infinite means no limit.
    /// </param>
    /// <returns>Each reachable node with the earliest time it can be reached.</returns>
    /// <remarks>
    /// <para>
    /// A time-respecting path is one whose edges occur in non-decreasing time order. This is
    /// strictly narrower than static reachability, and often dramatically so — a real contact
    /// network can be fully connected statically while most pairs cannot reach each other at all
    /// once the ordering is enforced.
    /// </para>
    /// <para>
    /// Computed by one pass over the time-sorted edges, relaxing each in turn. Because the edges are
    /// processed in time order, any edge that can extend a path has already had its source's
    /// earliest arrival finalised — which is what makes a single pass sufficient where a static
    /// graph would need a traversal.
    /// </para>
    /// </remarks>
    public IReadOnlyDictionary<int, double> TemporallyReachable(int source,
        double startTime = double.NegativeInfinity, double maxGap = double.PositiveInfinity)
    {
        EnsureSorted();

        var arrival = new Dictionary<int, double> { [source] = startTime };

        foreach (var edge in _edges)
        {
            if (edge.Time < startTime) continue;

            Relax(edge.Source, edge.Target, edge.Time);
            if (!Directed) Relax(edge.Target, edge.Source, edge.Time);
        }

        return arrival;

        void Relax(int from, int to, double time)
        {
            if (!arrival.TryGetValue(from, out var reachedAt)) return;

            // The edge must not precede our arrival, and must not come so long after it that the
            // caller considers the connection broken.
            if (time < reachedAt || time - reachedAt > maxGap) return;

            if (!arrival.TryGetValue(to, out var existing) || time < existing) arrival[to] = time;
        }
    }

    /// <summary>
    /// How much smaller temporal reachability is than static reachability.
    /// </summary>
    /// <remarks>
    /// A single number for how much a static analysis would overstate. Ratios well below one are
    /// normal on real event data, and are the argument for keeping the timestamps.
    /// </remarks>
    public double TemporalEfficiency()
    {
        if (NodeCount == 0) return 0.0;

        var collapsed = new Graph(NodeCount, Directed);
        foreach (var edge in _edges) collapsed.AddEdge(edge.Source, edge.Target, edge.Weight);

        var temporal = 0;
        var statik = 0;

        for (var node = 0; node < NodeCount; node++)
        {
            temporal += TemporallyReachable(node).Count - 1;
            statik += StaticReachable(collapsed, node) - 1;
        }

        return statik > 0 ? (double)temporal / statik : 0.0;
    }

    /// <summary>
    /// Node representations that fade with time, as an exponentially weighted neighbour average.
    /// </summary>
    /// <param name="features">Initial node features.</param>
    /// <param name="asOf">The moment to compute the representation at.</param>
    /// <param name="halfLife">How long an interaction takes to lose half its influence.</param>
    /// <remarks>
    /// The cheapest useful temporal embedding, and the right first thing to try. A recent
    /// interaction should say more about a node than one from a year ago, and a static aggregation
    /// weights them identically — which is why a model trained on a collapsed graph keeps
    /// recommending what someone liked once, long ago.
    /// </remarks>
    public NdArray TimeDecayedFeatures(NdArray features, double asOf, double halfLife)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (halfLife <= 0) throw new ArgumentOutOfRangeException(nameof(halfLife));

        var width = features.Shape[1];
        var result = features.Copy();
        var totalWeight = new double[NodeCount];

        var decay = Math.Log(2) / halfLife;

        foreach (var edge in _edges)
        {
            if (edge.Time > asOf) continue;      // no looking forward

            var weight = edge.Weight * Math.Exp(-decay * (asOf - edge.Time));

            Accumulate(edge.Target, edge.Source, weight);
            if (!Directed) Accumulate(edge.Source, edge.Target, weight);
        }

        // A weighted average in which the node's own features carry weight 1 and each neighbour
        // carries its decayed edge weight. Nodes that received nothing keep their own features
        // unchanged, which is what the division by (1 + 0) already gives.
        for (var node = 0; node < NodeCount && node < result.Shape[0]; node++)
            for (var d = 0; d < width; d++)
                result[node, d] /= 1 + totalWeight[node];

        return result;

        void Accumulate(int to, int from, double weight)
        {
            if (to >= result.Shape[0] || from >= features.Shape[0]) return;

            for (var d = 0; d < width; d++) result[to, d] += weight * features[from, d];
            totalWeight[to] += weight;
        }
    }

    /// <summary>Loads timestamped edges from a CSV of <c>source,target,time[,weight]</c>.</summary>
    public static TemporalGraph LoadCsv(string path, bool directed = true, bool hasHeader = true)
    {
        var graph = new TemporalGraph(directed);
        var first = true;

        foreach (var line in File.ReadLines(path))
        {
            if (first && hasHeader) { first = false; continue; }
            first = false;

            var parts = line.Split(',');
            if (parts.Length < 3) continue;

            graph.AddEdge(
                int.Parse(parts[0]), int.Parse(parts[1]),
                double.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture),
                parts.Length > 3
                    ? double.Parse(parts[3], System.Globalization.CultureInfo.InvariantCulture)
                    : 1.0);
        }

        return graph;
    }

    /// <inheritdoc />
    public override string ToString()
        => $"TemporalGraph({NodeCount} nodes, {EdgeCount} events, t in [{TimeRange.Start:G4}, {TimeRange.End:G4}])";

    private void EnsureSorted()
    {
        if (_sorted) return;

        _edges.Sort((a, b) => a.Time.CompareTo(b.Time));
        _sorted = true;
    }

    private static int StaticReachable(Graph graph, int source)
    {
        var seen = new HashSet<int> { source };
        var queue = new Queue<int>();
        queue.Enqueue(source);

        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            foreach (var (target, _) in graph.Neighbors(node))
                if (seen.Add(target)) queue.Enqueue(target);
        }

        return seen.Count;
    }
}
