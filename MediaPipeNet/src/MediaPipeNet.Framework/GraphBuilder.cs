using Microsoft.Extensions.Logging;

namespace MediaPipeNet.Framework;

/// <summary>What happens when a bounded input queue is full.</summary>
public enum QueueOverflowPolicy
{
    /// <summary>The oldest queued packet is dropped to make room (lowest latency for live streams).</summary>
    DropOldest = 0,
    /// <summary>The incoming packet is dropped.</summary>
    DropNewest = 1,
}

/// <summary>Options of a graph input stream.</summary>
/// <param name="MaxQueueSize">Maximum packets waiting at each consumer of the stream (0 = unbounded).</param>
/// <param name="OverflowPolicy">Which packet is dropped when the queue is full.</param>
public sealed record GraphInputOptions(int MaxQueueSize = 0, QueueOverflowPolicy OverflowPolicy = QueueOverflowPolicy.DropOldest);

/// <summary>Graph-wide options.</summary>
public sealed record GraphOptions
{
    /// <summary>
    /// Maximum number of input timestamps in flight (entered but not yet reflected on every graph
    /// output). Further packets are dropped by <see cref="CalculatorGraph.AddPacket"/> — the
    /// behaviour of MediaPipe's <c>FlowLimiterCalculator</c>. 0 disables the limit.
    /// </summary>
    public int MaxInFlight { get; init; }

    /// <summary>Logger used by the graph and passed to nodes.</summary>
    public ILogger? Logger { get; init; }

    /// <summary>
    /// Records a timeline of node executions (open, process per timestamp, close) that can be exported with
    /// <see cref="CalculatorGraph.WriteChromeTrace"/> and viewed in <c>chrome://tracing</c> or Perfetto. Default false.
    /// </summary>
    public bool EnableTracing { get; init; }

    /// <summary>Maximum trace events kept (oldest are discarded). Default 100 000.</summary>
    public int MaxTraceEvents { get; init; } = 100_000;
}

/// <summary>A node declared in a <see cref="GraphBuilder"/>.</summary>
public sealed class NodeBuilder
{
    internal NodeBuilder(GraphBuilder graph, string name, ICalculatorNode node)
    {
        Graph = graph;
        Name = name;
        Node = node;
    }

    /// <summary>The graph being built (for fluent chaining).</summary>
    public GraphBuilder Graph { get; }

    /// <summary>Node name.</summary>
    public string Name { get; }

    /// <summary>Node instance.</summary>
    public ICalculatorNode Node { get; }

    internal Dictionary<string, string> Inputs { get; } = new(StringComparer.Ordinal);
    internal Dictionary<string, string> Outputs { get; } = new(StringComparer.Ordinal);
    internal Dictionary<string, string> SidePackets { get; } = new(StringComparer.Ordinal);
    internal Dictionary<string, GraphInputOptions> QueueOptions { get; } = new(StringComparer.Ordinal);
    internal HashSet<string> BackEdges { get; } = new(StringComparer.Ordinal);
    internal string? Executor { get; private set; }

    /// <summary>
    /// Connects input <paramref name="tag"/> to stream <paramref name="stream"/>. A <paramref name="backEdge"/>
    /// input closes a loop (its stream is produced downstream of this node, e.g. a loopback of the previous
    /// result, as MediaPipe's <c>back_edge: true</c>): it is ignored by cycle detection, never holds the node
    /// back, and delivers the newest packet at or before the timestamp being processed.
    /// </summary>
    public NodeBuilder In(string tag, string stream, bool backEdge = false)
    {
        Inputs[tag] = stream;
        if (backEdge) BackEdges.Add(tag);
        else BackEdges.Remove(tag);
        return this;
    }

    /// <summary>Runs this node on the named executor (see <see cref="GraphBuilder.AddExecutor"/>) instead of the thread pool.</summary>
    public NodeBuilder OnExecutor(string executor)
    {
        ArgumentException.ThrowIfNullOrEmpty(executor);
        Executor = executor;
        return this;
    }

    /// <summary>Connects output <paramref name="tag"/> to stream <paramref name="stream"/>.</summary>
    public NodeBuilder Out(string tag, string stream)
    {
        Outputs[tag] = stream;
        return this;
    }

    /// <summary>Binds side packet <paramref name="tag"/> to the graph side packet <paramref name="sidePacket"/>.</summary>
    public NodeBuilder Side(string tag, string sidePacket)
    {
        SidePackets[tag] = sidePacket;
        return this;
    }

    /// <summary>Bounds the queue of input <paramref name="tag"/>.</summary>
    public NodeBuilder LimitQueue(string tag, int maxQueueSize, QueueOverflowPolicy policy = QueueOverflowPolicy.DropOldest)
    {
        QueueOptions[tag] = new GraphInputOptions(maxQueueSize, policy);
        return this;
    }
}

/// <summary>
/// Fluent builder for a <see cref="CalculatorGraph"/>: declare graph inputs, nodes and their stream
/// connections, graph outputs and side packets, then <see cref="Build"/> to validate
/// (single producer per stream, type compatibility, no cycles) and create the graph.
/// </summary>
/// <example>
/// <code>
/// var graph = new GraphBuilder()
///     .AddInputStream&lt;int&gt;("numbers")
///     .AddNode("square", new LambdaNode&lt;int, int&gt;(x =&gt; x * x)).In("IN", "numbers").Out("OUT", "squares").Graph
///     .AddOutputStream("squares")
///     .Build();
/// </code>
/// </example>
public sealed class GraphBuilder
{
    internal List<(string Name, Type Type, GraphInputOptions Options)> InputStreams { get; } = [];
    internal List<string> OutputStreams { get; } = [];
    internal Dictionary<string, Type> SidePackets { get; } = new(StringComparer.Ordinal);
    internal List<NodeBuilder> Nodes { get; } = [];
    internal Dictionary<string, int> Executors { get; } = new(StringComparer.Ordinal);
    private bool _expanded;

    /// <summary>Graph-wide options.</summary>
    public GraphOptions Options { get; set; } = new();

    /// <summary>Declares a graph input stream fed with <see cref="CalculatorGraph.AddPacket"/>.</summary>
    public GraphBuilder AddInputStream<T>(string name, GraphInputOptions? options = null) => AddInputStream(name, typeof(T), options);

    /// <summary>Declares a graph input stream with a runtime type.</summary>
    public GraphBuilder AddInputStream(string name, Type type, GraphInputOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        if (InputStreams.Exists(s => s.Name == name)) throw new ArgumentException($"Input stream '{name}' already declared.");
        InputStreams.Add((name, type, options ?? new GraphInputOptions()));
        return this;
    }

    /// <summary>Declares a stream whose packets are exposed to the application (observers/pollers).</summary>
    public GraphBuilder AddOutputStream(string name)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        if (!OutputStreams.Contains(name)) OutputStreams.Add(name);
        return this;
    }

    /// <summary>Declares a graph side packet supplied to <see cref="CalculatorGraph.StartAsync"/>.</summary>
    public GraphBuilder AddSidePacket<T>(string name)
    {
        SidePackets[name] = typeof(T);
        return this;
    }

    /// <summary>Adds a node instance.</summary>
    public NodeBuilder AddNode(string name, ICalculatorNode node)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(node);
        if (Nodes.Exists(n => n.Name == name)) throw new ArgumentException($"Node '{name}' already exists.");
        var nb = new NodeBuilder(this, name, node);
        Nodes.Add(nb);
        return nb;
    }

    /// <summary>Adds a node created with its parameterless constructor.</summary>
    public NodeBuilder AddNode<TNode>(string name) where TNode : ICalculatorNode, new() => AddNode(name, new TNode());

    /// <summary>
    /// Declares an executor: <paramref name="threads"/> dedicated threads that run the nodes assigned to it
    /// with <see cref="NodeBuilder.OnExecutor"/> (MediaPipe's <c>executor</c> / <c>ThreadPoolExecutor</c>).
    /// Use one thread to serialize nodes that share a non-thread-safe resource (a GPU context, a camera), or
    /// a separate pool to keep heavy nodes from starving the rest of the graph.
    /// </summary>
    public GraphBuilder AddExecutor(string name, int threads = 1)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(threads);
        Executors[name] = threads;
        return this;
    }

    /// <summary>
    /// Inlines <paramref name="subgraph"/> as a reusable block (MediaPipe's subgraphs). Its nodes are added
    /// as <c>{name}/{node}</c>; its input streams are connected to the streams named in
    /// <paramref name="inputs"/> (subgraph input → outer stream), its streams listed in
    /// <paramref name="outputs"/> (subgraph stream → outer stream) are renamed, and every other internal
    /// stream becomes <c>{name}/{stream}</c>. Side packets keep their names. A subgraph builder can be inlined once
    /// (its node instances hold state); build a fresh one for each use.
    /// </summary>
    public GraphBuilder AddSubgraph(string name, GraphBuilder subgraph, IReadOnlyDictionary<string, string> inputs, IReadOnlyDictionary<string, string> outputs)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(subgraph);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputs);
        if (ReferenceEquals(subgraph, this)) throw new ArgumentException("A graph cannot contain itself.", nameof(subgraph));
        if (subgraph._expanded) throw new InvalidOperationException("This subgraph builder was already inlined; create a new one for each use.");
        foreach (var (input, _, _) in subgraph.InputStreams)
            if (!inputs.ContainsKey(input)) throw new GraphValidationException($"Subgraph '{name}': input stream '{input}' is not connected.");
        foreach (var output in subgraph.OutputStreams)
            if (!outputs.ContainsKey(output)) throw new GraphValidationException($"Subgraph '{name}': output stream '{output}' is not connected.");
        string Map(string stream) =>
            inputs.TryGetValue(stream, out var outer) ? outer : outputs.TryGetValue(stream, out outer) ? outer : $"{name}/{stream}";
        foreach (var n in subgraph.Nodes)
        {
            var nb = AddNode($"{name}/{n.Name}", n.Node);
            foreach (var (tag, stream) in n.Inputs) nb.In(tag, Map(stream), n.BackEdges.Contains(tag));
            foreach (var (tag, stream) in n.Outputs) nb.Out(tag, Map(stream));
            foreach (var (tag, side) in n.SidePackets) nb.Side(tag, side);
            foreach (var (tag, q) in n.QueueOptions) nb.QueueOptions[tag] = q;
            if (n.Executor is not null) nb.OnExecutor(n.Executor);
        }
        foreach (var (side, type) in subgraph.SidePackets) SidePackets.TryAdd(side, type);
        foreach (var (executor, threads) in subgraph.Executors) Executors.TryAdd(executor, threads);
        subgraph._expanded = true;
        return this;
    }

    /// <summary>Validates the declaration and creates the graph.</summary>
    /// <exception cref="GraphValidationException">The graph is invalid.</exception>
    public CalculatorGraph Build() => new(this);
}

/// <summary>Raised when a graph declaration is invalid.</summary>
public sealed class GraphValidationException : MediaPipeException
{
    /// <summary>Creates the exception.</summary>
    public GraphValidationException() { }

    /// <summary>Creates the exception with a message.</summary>
    public GraphValidationException(string message) : base(message) { }

    /// <summary>Creates the exception with a message and an inner exception.</summary>
    public GraphValidationException(string message, Exception innerException) : base(message, innerException) { }
}
