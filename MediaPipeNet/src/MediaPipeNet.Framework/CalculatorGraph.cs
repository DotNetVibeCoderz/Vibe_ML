using System.Diagnostics;
using System.Text;
using System.Threading.Channels;
using MediaPipeNet.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace MediaPipeNet.Framework;

/// <summary>Lifecycle state of a <see cref="CalculatorGraph"/>.</summary>
public enum GraphState
{
    /// <summary>Built, not started.</summary>
    Created,
    /// <summary>Running and accepting packets.</summary>
    Running,
    /// <summary>All inputs closed and every node closed.</summary>
    Done,
    /// <summary>A node failed; see <see cref="CalculatorGraph.Error"/>.</summary>
    Failed,
    /// <summary>Cancelled by the application.</summary>
    Cancelled,
}

/// <summary>
/// A running pipeline of <see cref="ICalculatorNode"/>s connected by packet streams — MediaPipe's
/// <c>CalculatorGraph</c>.
/// </summary>
/// <remarks>
/// <para>
/// Scheduling: each node processes one timestamp at a time, but different nodes run concurrently on
/// the thread pool, so consecutive frames are pipelined through the graph. A node with the
/// <see cref="InputPolicy.Synchronized"/> policy runs for timestamp T once every input has either
/// delivered its packet for T or advanced its timestamp bound beyond T. After processing T, outputs
/// that emitted nothing have their bound advanced to T+1 automatically, so downstream nodes never stall.
/// </para>
/// <para>
/// Live streams: bound graph inputs with <see cref="GraphInputOptions.MaxQueueSize"/> (drop oldest) or
/// limit in-flight frames with <see cref="GraphOptions.MaxInFlight"/> (drop newest while busy).
/// </para>
/// </remarks>
public sealed class CalculatorGraph : IAsyncDisposable
{
    private readonly Lock _gate = new();
    private readonly Dictionary<string, OutputPort> _streams = new(StringComparer.Ordinal);
    private readonly Dictionary<string, OutputPort> _graphInputs = new(StringComparer.Ordinal);
    private readonly Dictionary<string, GraphInputOptions> _graphInputOptions = new(StringComparer.Ordinal);
    private readonly List<OutputPort> _graphOutputPorts = [];
    private readonly Dictionary<string, Type> _declaredSidePackets;
    private readonly List<NodeRuntime> _nodes;
    private readonly SortedSet<Timestamp> _inFlight = [];
    private readonly List<TaskCompletionSource> _idleWaiters = [];
    private readonly TaskCompletionSource _done = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly CancellationTokenSource _cts = new();
    private readonly ILogger _logger;
    private readonly int _maxInFlight;
    private int _pending;
    private int _openNodes;
    private long _droppedPackets;
    private Exception? _error;

    internal CalculatorGraph(GraphBuilder builder)
    {
        _logger = builder.Options.Logger ?? NullLogger.Instance;
        _maxInFlight = builder.Options.MaxInFlight;
        _declaredSidePackets = new Dictionary<string, Type>(builder.SidePackets, StringComparer.Ordinal);

        // Graph inputs are streams produced by the application.
        foreach (var (name, type, options) in builder.InputStreams)
        {
            var port = new OutputPort(name, new PortSpec(name, type), producer: null);
            _streams.Add(name, port);
            _graphInputs.Add(name, port);
            _graphInputOptions.Add(name, options);
        }

        // Node outputs.
        var nodes = new List<NodeRuntime>();
        foreach (var nb in builder.Nodes)
        {
            var contract = new CalculatorContract();
            nb.Node.GetContract(contract);
            var node = new NodeRuntime(this, nb.Name, nb.Node, contract);
            foreach (var tag in nb.Outputs.Keys.Concat(nb.Inputs.Keys).Concat(nb.SidePackets.Keys))
            {
                if (!contract.Outputs.Any(p => p.Tag == tag) && !contract.Inputs.Any(p => p.Tag == tag) && !contract.InputSidePackets.Any(p => p.Tag == tag))
                    throw new GraphValidationException($"Node '{nb.Name}' ({nb.Node.GetType().Name}) has no port '{tag}'.");
            }
            for (int i = 0; i < contract.Outputs.Count; i++)
            {
                var spec = contract.Outputs[i];
                if (!nb.Outputs.TryGetValue(spec.Tag, out var stream)) continue;
                if (_streams.ContainsKey(stream))
                    throw new GraphValidationException($"Stream '{stream}' has more than one producer (node '{nb.Name}').");
                var port = new OutputPort(stream, spec, node);
                _streams.Add(stream, port);
                node.Outputs[i] = port;
            }
            nodes.Add(node);
        }

        // Node inputs and side packets.
        foreach (var (nb, node) in builder.Nodes.Zip(nodes))
        {
            var contract = node.Contract;
            for (int i = 0; i < contract.Inputs.Count; i++)
            {
                var spec = contract.Inputs[i];
                if (!nb.Inputs.TryGetValue(spec.Tag, out var stream))
                {
                    if (!spec.Optional) throw new GraphValidationException($"Node '{nb.Name}': required input '{spec.Tag}' is not connected.");
                    node.Inputs[i] = InputQueue.Unconnected(node, i, spec);
                    continue;
                }
                if (!_streams.TryGetValue(stream, out var producer))
                    throw new GraphValidationException($"Node '{nb.Name}': input '{spec.Tag}' refers to unknown stream '{stream}'.");
                if (!AreCompatible(producer.Spec.Type, spec.Type))
                    throw new GraphValidationException(
                        $"Node '{nb.Name}': input '{spec.Tag}' expects {spec.Type.Name} but stream '{stream}' carries {producer.Spec.Type.Name}.");
                var options = nb.QueueOptions.GetValueOrDefault(spec.Tag)
                              ?? (producer.Producer is null ? _graphInputOptions[stream] : new GraphInputOptions());
                var queue = new InputQueue(node, i, spec, stream, options);
                producer.Consumers.Add(queue);
                node.Inputs[i] = queue;
            }
            if (contract.Inputs.Count == 0)
                throw new GraphValidationException($"Node '{nb.Name}' has no inputs; source nodes are fed through graph input streams.");
            foreach (var side in contract.InputSidePackets)
            {
                if (nb.SidePackets.TryGetValue(side.Tag, out var name))
                {
                    if (!_declaredSidePackets.ContainsKey(name)) _declaredSidePackets[name] = side.Type;
                    node.SidePacketBindings[side.Tag] = name;
                }
                else if (!side.Optional)
                {
                    throw new GraphValidationException($"Node '{nb.Name}': required side packet '{side.Tag}' is not bound.");
                }
            }
        }

        foreach (var name in builder.OutputStreams)
        {
            if (!_streams.TryGetValue(name, out var port))
                throw new GraphValidationException($"Graph output stream '{name}' is not produced by any node or input.");
            port.IsGraphOutput = true;
            _graphOutputPorts.Add(port);
        }

        _nodes = TopologicalSort(nodes);
        State = GraphState.Created;
    }

    /// <summary>Current state.</summary>
    public GraphState State { get; private set; }

    /// <summary>The first error raised by a node, if any.</summary>
    public Exception? Error => _error;

    /// <summary>Node names in topological order.</summary>
    public IReadOnlyList<string> NodeNames => _nodes.Select(n => n.Name).ToArray();

    /// <summary>Names of all streams.</summary>
    public IReadOnlyCollection<string> StreamNames => _streams.Keys;

    /// <summary>Packets dropped by queue limits or in-flight limiting.</summary>
    public long DroppedPackets => Interlocked.Read(ref _droppedPackets);

    /// <summary>Input timestamps currently in flight (tracked when <see cref="GraphOptions.MaxInFlight"/> is set).</summary>
    public int InFlightCount
    {
        get { lock (_gate) return _inFlight.Count; }
    }

    /// <summary>Raised (on a worker thread) when a node fails.</summary>
    public event EventHandler<Exception>? Failed;

    // ====================================================================== public API

    /// <summary>
    /// Registers a callback invoked for every packet of output stream <paramref name="stream"/>.
    /// Callbacks run synchronously on the producing node's thread; keep them short.
    /// </summary>
    public void ObserveOutputStream<T>(string stream, Action<Packet<T>> callback)
    {
        ArgumentNullException.ThrowIfNull(callback);
        var port = GetObservablePort(stream);
        lock (_gate) port.Observers.Add(p => callback(p.As<T>()));
    }

    /// <summary>Creates a pull-based reader for output stream <paramref name="stream"/>.</summary>
    /// <param name="stream">Stream name.</param>
    /// <param name="capacity">Maximum buffered packets (0 = unbounded); when full the oldest is dropped.</param>
    public ChannelReader<Packet<T>> CreateOutputStreamPoller<T>(string stream, int capacity = 0)
    {
        var port = GetObservablePort(stream);
        var channel = capacity > 0
            ? Channel.CreateBounded<Packet<T>>(new BoundedChannelOptions(capacity) { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true })
            : Channel.CreateUnbounded<Packet<T>>(new UnboundedChannelOptions { SingleReader = true });
        lock (_gate)
        {
            port.Observers.Add(p => channel.Writer.TryWrite(p.As<T>()));
            port.OnClosed.Add(() => channel.Writer.TryComplete(_error));
        }
        return channel.Reader;
    }

    /// <summary>Opens every node (in topological order) and starts accepting packets.</summary>
    /// <param name="sidePackets">Values of the graph's side packets.</param>
    /// <param name="cancellationToken">Cancels the whole run.</param>
    public async Task StartAsync(IReadOnlyDictionary<string, object?>? sidePackets = null, CancellationToken cancellationToken = default)
    {
        if (State != GraphState.Created) throw new InvalidOperationException("The graph was already started.");
        sidePackets ??= new Dictionary<string, object?>();
        foreach (var (name, type) in _declaredSidePackets)
        {
            if (sidePackets.TryGetValue(name, out var v) && v is not null && !type.IsInstanceOfType(v))
                throw new GraphValidationException($"Side packet '{name}' must be a {type.Name} but is a {v.GetType().Name}.");
        }
        if (cancellationToken.CanBeCanceled) cancellationToken.Register(Cancel);

        foreach (var node in _nodes)
        {
            foreach (var (tag, name) in node.SidePacketBindings)
            {
                if (sidePackets.TryGetValue(name, out var value)) node.SidePacketValues[tag] = value;
                else if (!node.Contract.InputSidePackets.First(p => p.Tag == tag).Optional)
                    throw new GraphValidationException($"Node '{node.Name}': side packet '{name}' was not provided.");
            }
            node.Context = new CalculatorContext(node, _logger, _cts.Token);
            await node.Node.OpenAsync(node.Context, _cts.Token).ConfigureAwait(false);
            node.Opened = true;
        }
        _openNodes = _nodes.Count;
        lock (_gate) State = GraphState.Running;
        _logger.LogDebug("Graph started with {Count} nodes", _nodes.Count);
    }

    /// <summary>
    /// Adds a packet to graph input <paramref name="stream"/>. Timestamps must strictly increase per stream.
    /// </summary>
    /// <returns>False when the packet was dropped by flow limiting.</returns>
    public bool AddPacket(string stream, Packet packet)
    {
        ArgumentNullException.ThrowIfNull(packet);
        if (!_graphInputs.TryGetValue(stream, out var port))
            throw new ArgumentException($"'{stream}' is not a graph input stream.", nameof(stream));
        if (!packet.Timestamp.IsRangeValue)
            throw new ArgumentException($"Packet timestamp {packet.Timestamp} is not a regular timestamp.", nameof(packet));
        if (!AreCompatible(packet.PayloadType, port.Spec.Type) && !port.Spec.Type.IsInstanceOfType(packet.UntypedValue))
            throw new ArgumentException($"Stream '{stream}' carries {port.Spec.Type.Name}, not {packet.PayloadType.Name}.", nameof(packet));

        List<NodeRuntime> toSchedule;
        List<Action<Packet>>? observers = null;
        bool accepted;
        lock (_gate)
        {
            ThrowIfNotRunning();
            if (port.Closed) throw new InvalidOperationException($"Input stream '{stream}' is closed.");
            if (packet.Timestamp < port.NextBound)
                throw new ArgumentException(
                    $"Timestamps on '{stream}' must strictly increase: got {packet.Timestamp}, expected ≥ {port.NextBound}.", nameof(packet));

            bool limited = _maxInFlight > 0 && _graphOutputPorts.Count > 0;
            if (limited && _inFlight.Count >= _maxInFlight && !_inFlight.Contains(packet.Timestamp))
            {
                // Flow limiting: drop the new packet but advance the bound so downstream nodes never wait for it.
                _droppedPackets++;
                MediaPipeTelemetry.FramesDropped.Add(1);
                toSchedule = SetBoundLocked(port, packet.Timestamp.NextAllowedInStream());
                accepted = false;
            }
            else
            {
                if (limited) _inFlight.Add(packet.Timestamp);
                toSchedule = EmitLocked(port, packet, out observers);
                accepted = true;
            }
        }
        if (observers is not null) Notify(observers, packet);
        ScheduleAll(toSchedule);
        return accepted;
    }
    /// <summary>Adds a value at a millisecond timestamp. See <see cref="AddPacket"/>.</summary>
    public bool AddPacket<T>(string stream, T value, long timestampMs) => AddPacket(stream, Packet.Create(value, timestampMs));

    /// <summary>Closes a graph input stream: its consumers will receive no more packets.</summary>
    public void CloseInputStream(string stream)
    {
        if (!_graphInputs.TryGetValue(stream, out var port))
            throw new ArgumentException($"'{stream}' is not a graph input stream.", nameof(stream));
        List<NodeRuntime> toSchedule;
        lock (_gate)
        {
            if (port.Closed) return;
            toSchedule = SetBoundLocked(port, Timestamp.Done);
        }
        ScheduleAll(toSchedule);
        CompleteIfFinished();
    }

    /// <summary>Closes every graph input stream.</summary>
    public void CloseAllInputStreams()
    {
        foreach (var name in _graphInputs.Keys) CloseInputStream(name);
    }

    /// <summary>Completes when no node is running or runnable.</summary>
    public Task WaitUntilIdleAsync(CancellationToken cancellationToken = default)
    {
        TaskCompletionSource tcs;
        lock (_gate)
        {
            if (_error is not null) return Task.FromException(_error);
            if (_pending == 0) return Task.CompletedTask;
            tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            _idleWaiters.Add(tcs);
        }
        return tcs.Task.WaitAsync(cancellationToken);
    }

    /// <summary>Completes when every node has closed (call <see cref="CloseAllInputStreams"/> first).</summary>
    /// <exception cref="MediaPipeException">A node failed.</exception>
    public Task WaitUntilDoneAsync(CancellationToken cancellationToken = default) => _done.Task.WaitAsync(cancellationToken);

    /// <summary>Closes all inputs and waits for the graph to finish.</summary>
    public async Task CloseAsync(CancellationToken cancellationToken = default)
    {
        if (State == GraphState.Running) CloseAllInputStreams();
        if (State is GraphState.Running or GraphState.Done) await WaitUntilDoneAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Cancels the run; pending packets are discarded.</summary>
    public void Cancel()
    {
        lock (_gate)
        {
            if (State is GraphState.Done or GraphState.Failed or GraphState.Cancelled) return;
            State = GraphState.Cancelled;
        }
        _cts.Cancel();
        _done.TrySetCanceled();
        ReleaseIdleWaiters(null);
    }

    /// <summary>
    /// The graph's connections for visualization: <c>From</c> is a node name or <c>"input:&lt;stream&gt;"</c>,
    /// <c>To</c> is a node name or <c>"output:&lt;stream&gt;"</c>.
    /// </summary>
    public IReadOnlyList<(string From, string To, string Stream)> GetEdges()
    {
        var edges = new List<(string, string, string)>();
        foreach (var (name, port) in _streams)
        {
            string from = port.Producer?.Name ?? "input:" + name;
            foreach (var c in port.Consumers) edges.Add((from, c.Node.Name, name));
            if (port.IsGraphOutput) edges.Add((from, "output:" + name, name));
        }
        return edges;
    }

    /// <summary>Type name of a node's calculator.</summary>
    public string GetNodeType(string nodeName) => _nodes.First(n => n.Name == nodeName).Node.GetType().Name.Split('`')[0];

    /// <summary>Renders the graph as a Mermaid flowchart (for documentation and debugging).</summary>
    public string ToMermaid()
    {
        var sb = new StringBuilder("flowchart LR\n");
        static string Id(string s) => new(s.Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray());
        foreach (var input in _graphInputs.Keys) sb.Append("    in_").Append(Id(input)).Append("[/\"").Append(input).Append("\"/]\n");
        foreach (var node in _nodes) sb.Append("    n_").Append(Id(node.Name)).Append("(\"").Append(node.Name).Append("<br/><small>").Append(node.Node.GetType().Name.Split('`')[0]).Append("</small>\")\n");
        foreach (var (name, port) in _streams)
        {
            string from = port.Producer is null ? "in_" + Id(name) : "n_" + Id(port.Producer.Name);
            foreach (var c in port.Consumers)
                sb.Append("    ").Append(from).Append(" -->|").Append(name).Append("| n_").Append(Id(c.Node.Name)).Append('\n');
            if (port.IsGraphOutput)
                sb.Append("    ").Append(from).Append(" -->|").Append(name).Append("| out_").Append(Id(name)).Append("[/\"").Append(name).Append("\"/]\n");
        }
        return sb.ToString();
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (State == GraphState.Running)
        {
            try { await CloseAsync().WaitAsync(TimeSpan.FromSeconds(5)).ConfigureAwait(false); }
            catch (Exception e) when (e is TimeoutException or MediaPipeException or OperationCanceledException) { Cancel(); }
        }
        _cts.Dispose();
    }

    // ====================================================================== internals

    internal void Emit(OutputPort port, Packet packet, string nodeName)
    {
        List<NodeRuntime> toSchedule;
        List<Action<Packet>>? observers;
        lock (_gate)
        {
            if (packet.Timestamp < port.NextBound)
                throw new InvalidOperationException(
                    $"Node '{nodeName}': packet timestamp {packet.Timestamp} on '{port.Name}' is not greater than the previous bound {port.NextBound}.");
            toSchedule = EmitLocked(port, packet, out observers);
        }
        MediaPipeTelemetry.GraphPackets.Add(1, new KeyValuePair<string, object?>("node", nodeName));
        if (observers is not null) Notify(observers, packet);
        ScheduleAll(toSchedule);
    }

    internal void AdvanceBound(OutputPort port, Timestamp bound)
    {
        List<NodeRuntime> toSchedule;
        lock (_gate) toSchedule = SetBoundLocked(port, bound);
        ScheduleAll(toSchedule);
    }

    private List<NodeRuntime> EmitLocked(OutputPort port, Packet packet, out List<Action<Packet>>? observers)
    {
        var toSchedule = new List<NodeRuntime>(port.Consumers.Count);
        port.NextBound = packet.Timestamp.NextAllowedInStream();
        foreach (var q in port.Consumers)
        {
            if (q.Options.MaxQueueSize > 0 && q.Packets.Count >= q.Options.MaxQueueSize)
            {
                _droppedPackets++;
                MediaPipeTelemetry.FramesDropped.Add(1);
                if (q.Options.OverflowPolicy == QueueOverflowPolicy.DropNewest)
                {
                    q.Bound = Timestamp.Maximum(q.Bound, port.NextBound);
                    toSchedule.Add(q.Node);
                    continue;
                }
                q.Packets.Dequeue();
            }
            q.Packets.Enqueue(packet);
            q.Bound = Timestamp.Maximum(q.Bound, port.NextBound);
            toSchedule.Add(q.Node);
        }
        observers = port.Observers.Count > 0 ? port.Observers : null;
        if (port.IsGraphOutput) PruneInFlight();
        return toSchedule;
    }

    private List<NodeRuntime> SetBoundLocked(OutputPort port, Timestamp bound)
    {
        var toSchedule = new List<NodeRuntime>();
        if (bound <= port.NextBound) return toSchedule;
        port.NextBound = bound;
        foreach (var q in port.Consumers)
        {
            if (bound > q.Bound)
            {
                q.Bound = bound;
                toSchedule.Add(q.Node);
            }
        }
        if (bound == Timestamp.Done && !port.Closed)
        {
            port.Closed = true;
            foreach (var onClosed in port.OnClosed) onClosed();
        }
        if (port.IsGraphOutput) PruneInFlight();
        return toSchedule;
    }

    private void PruneInFlight()
    {
        if (_inFlight.Count == 0 || _graphOutputPorts.Count == 0) return;
        var min = _graphOutputPorts.Min(p => p.NextBound);
        while (_inFlight.Count > 0 && _inFlight.Min < min) _inFlight.Remove(_inFlight.Min);
    }

    private void Notify(List<Action<Packet>> observers, Packet packet)
    {
        foreach (var observer in observers.ToArray())
        {
            try { observer(packet); }
            catch (Exception e) { Fail(new MediaPipeException($"Output observer failed: {e.Message}", e)); }
        }
    }

    private void ScheduleAll(List<NodeRuntime> nodes)
    {
        foreach (var node in nodes) TrySchedule(node);
    }

    private void TrySchedule(NodeRuntime node)
    {
        lock (_gate)
        {
            if (State != GraphState.Running || !node.Opened || node.Running || node.Closed) return;
            if (!node.IsRunnable()) return;
            node.Running = true;
            _pending++;
        }
        ThreadPool.UnsafeQueueUserWorkItem(static n => _ = n.Graph.RunNodeAsync(n), node, preferLocal: false);
    }

    private async Task RunNodeAsync(NodeRuntime node)
    {
        var ctx = node.Context!;
        try
        {
            while (true)
            {
                bool close = false;
                Timestamp ts;
                lock (_gate)
                {
                    if (State != GraphState.Running && State != GraphState.Done)
                    {
                        node.Running = false;
                        break;
                    }
                    if (!node.TryPop(ctx.InputPackets, out ts))
                    {
                        if (node.InputsDone())
                        {
                            close = true;
                            node.Closed = true;
                        }
                        else
                        {
                            node.Running = false;
                            break;
                        }
                    }
                }

                if (close)
                {
                    ctx.InputTimestamp = Timestamp.Done;
                    Array.Clear(ctx.InputPackets);
                    await node.Node.CloseAsync(ctx, _cts.Token).ConfigureAwait(false);
                    List<NodeRuntime> downstream = [];
                    lock (_gate)
                    {
                        foreach (var port in node.Outputs)
                            if (port is not null) downstream.AddRange(SetBoundLocked(port, Timestamp.Done));
                        node.Running = false;
                    }
                    ScheduleAll(downstream);
                    if (Interlocked.Decrement(ref _openNodes) == 0) CompleteIfFinished();
                    break;
                }

                ctx.InputTimestamp = ts;
                using (var activity = MediaPipeTelemetry.ActivitySource.StartActivity(node.Name, ActivityKind.Internal))
                {
                    activity?.SetTag("timestamp", ts.Value);
                    await node.Node.ProcessAsync(ctx, _cts.Token).ConfigureAwait(false);
                }
                Array.Clear(ctx.InputPackets);

                if (node.Contract.PropagateTimestampBounds)
                {
                    List<NodeRuntime> downstream = [];
                    lock (_gate)
                    {
                        var next = ts.NextAllowedInStream();
                        foreach (var port in node.Outputs)
                            if (port is not null) downstream.AddRange(SetBoundLocked(port, next));
                    }
                    ScheduleAll(downstream);
                }
            }
        }
        catch (OperationCanceledException) when (_cts.IsCancellationRequested)
        {
            lock (_gate) node.Running = false;
        }
        catch (Exception e)
        {
            lock (_gate) node.Running = false;
            Fail(new MediaPipeException($"Node '{node.Name}' failed: {e.Message}", e));
        }
        finally
        {
            bool idle;
            lock (_gate) idle = --_pending == 0;
            if (idle) ReleaseIdleWaiters(null);
            // A packet may have arrived between the last TryPop and Running=false being observed.
            if (State == GraphState.Running) TrySchedule(node);
        }
    }

    private void CompleteIfFinished()
    {
        lock (_gate)
        {
            if (State != GraphState.Running || _openNodes > 0) return;
            if (_graphInputs.Values.Any(p => !p.Closed)) return;
            State = GraphState.Done;
        }
        // Graph input streams that feed graph outputs directly.
        _done.TrySetResult();
        _logger.LogDebug("Graph done");
    }

    private void Fail(Exception error)
    {
        lock (_gate)
        {
            if (_error is not null) return;
            _error = error;
            State = GraphState.Failed;
        }
        _logger.LogError(error, "Graph failed");
        _cts.Cancel();
        Failed?.Invoke(this, error);
        foreach (var port in _streams.Values)
            foreach (var onClosed in port.OnClosed) onClosed();
        _done.TrySetException(error);
        ReleaseIdleWaiters(error);
    }

    private void ReleaseIdleWaiters(Exception? error)
    {
        TaskCompletionSource[] waiters;
        lock (_gate)
        {
            waiters = [.. _idleWaiters];
            _idleWaiters.Clear();
        }
        foreach (var w in waiters)
        {
            if (error is null) w.TrySetResult();
            else w.TrySetException(error);
        }
    }

    private void ThrowIfNotRunning()
    {
        if (State == GraphState.Running) return;
        if (_error is not null) throw new MediaPipeException("The graph has failed.", _error);
        throw new InvalidOperationException($"The graph is not running (state: {State}).");
    }

    private OutputPort GetObservablePort(string stream)
    {
        if (State != GraphState.Created) throw new InvalidOperationException("Observers must be registered before StartAsync.");
        if (!_streams.TryGetValue(stream, out var port) || !port.IsGraphOutput)
            throw new ArgumentException($"'{stream}' is not a graph output stream (declare it with AddOutputStream).", nameof(stream));
        return port;
    }

    internal static bool AreCompatible(Type produced, Type consumed) =>
        consumed == typeof(object) || produced == typeof(object) || consumed.IsAssignableFrom(produced);

    private static List<NodeRuntime> TopologicalSort(List<NodeRuntime> nodes)
    {
        var indegree = nodes.ToDictionary(n => n, n => n.Inputs.Count(q => q?.Stream is not null && IsNodeProduced(nodes, q.Stream)));
        var ordered = new List<NodeRuntime>(nodes.Count);
        var ready = new Queue<NodeRuntime>(nodes.Where(n => indegree[n] == 0));
        while (ready.Count > 0)
        {
            var n = ready.Dequeue();
            ordered.Add(n);
            foreach (var port in n.Outputs)
            {
                if (port is null) continue;
                foreach (var q in port.Consumers)
                    if (--indegree[q.Node] == 0) ready.Enqueue(q.Node);
            }
        }
        if (ordered.Count != nodes.Count)
        {
            var cyclic = nodes.Except(ordered).Select(n => n.Name);
            throw new GraphValidationException($"The graph contains a cycle involving: {string.Join(", ", cyclic)}.");
        }
        return ordered;
    }

    private static bool IsNodeProduced(List<NodeRuntime> nodes, string stream) =>
        nodes.Any(n => n.Outputs.Any(o => o?.Name == stream));

    // ====================================================================== runtime types

    internal sealed class OutputPort(string name, PortSpec spec, NodeRuntime? producer)
    {
        public string Name { get; } = name;
        public PortSpec Spec { get; } = spec;
        public NodeRuntime? Producer { get; } = producer;
        public List<InputQueue> Consumers { get; } = [];
        public List<Action<Packet>> Observers { get; } = [];
        public List<Action> OnClosed { get; } = [];
        public Timestamp NextBound { get; set; } = Timestamp.PreStream;
        public bool Closed { get; set; }
        public bool IsGraphOutput { get; set; }
    }

    internal sealed class InputQueue(NodeRuntime node, int index, PortSpec spec, string? stream, GraphInputOptions options)
    {
        public NodeRuntime Node { get; } = node;
        public int Index { get; } = index;
        public PortSpec Spec { get; } = spec;
        public string? Stream { get; } = stream;
        public GraphInputOptions Options { get; } = options;
        public Queue<Packet> Packets { get; } = new();
        public Timestamp Bound { get; set; } = Timestamp.PreStream;

        public static InputQueue Unconnected(NodeRuntime node, int index, PortSpec spec) =>
            new(node, index, spec, null, new GraphInputOptions()) { Bound = Timestamp.Done };

        /// <summary>Earliest timestamp this input can still deliver.</summary>
        public Timestamp Candidate => Packets.Count > 0 ? Packets.Peek().Timestamp : Bound;
    }

    internal sealed class NodeRuntime
    {
        public NodeRuntime(CalculatorGraph graph, string name, ICalculatorNode node, CalculatorContract contract)
        {
            Graph = graph;
            Name = name;
            Node = node;
            Contract = contract;
            Inputs = new InputQueue[contract.Inputs.Count];
            Outputs = new OutputPort?[contract.Outputs.Count];
        }

        public CalculatorGraph Graph { get; }
        public string Name { get; }
        public ICalculatorNode Node { get; }
        public CalculatorContract Contract { get; }
        public InputQueue[] Inputs { get; }
        public OutputPort?[] Outputs { get; }
        public Dictionary<string, string> SidePacketBindings { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, object?> SidePacketValues { get; } = new(StringComparer.Ordinal);
        public CalculatorContext? Context { get; set; }
        public bool Opened { get; set; }
        public bool Running { get; set; }
        public bool Closed { get; set; }

        public bool InputsDone() => Inputs.All(q => q.Packets.Count == 0 && q.Bound == Timestamp.Done);

        public bool IsRunnable() => TryGetReadyTimestamp(out _) || InputsDone();

        private bool TryGetReadyTimestamp(out Timestamp ts)
        {
            ts = Timestamp.Done;
            if (Contract.InputPolicy == InputPolicy.Immediate)
            {
                foreach (var q in Inputs)
                    if (q.Packets.Count > 0 && q.Packets.Peek().Timestamp < ts) ts = q.Packets.Peek().Timestamp;
                return ts != Timestamp.Done;
            }
            foreach (var q in Inputs)
            {
                var c = q.Candidate;
                if (c < ts) ts = c;
            }
            if (ts == Timestamp.Done) return false;
            bool any = false;
            foreach (var q in Inputs)
            {
                if (q.Candidate != ts) continue;
                if (q.Packets.Count == 0) return false; // bound == ts: a packet at ts may still arrive
                any = true;
            }
            return any;
        }

        public bool TryPop(Packet?[] into, out Timestamp ts)
        {
            if (!TryGetReadyTimestamp(out ts)) return false;
            for (int i = 0; i < Inputs.Length; i++)
            {
                var q = Inputs[i];
                into[i] = q.Packets.Count > 0 && q.Packets.Peek().Timestamp == ts ? q.Packets.Dequeue() : null;
            }
            return true;
        }
    }
}
