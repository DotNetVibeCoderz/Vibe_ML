using Microsoft.Extensions.Logging;

namespace MediaPipeNet.Framework;

/// <summary>
/// Everything a node sees during <see cref="ICalculatorNode.ProcessAsync"/>: the input packets of
/// the current timestamp, its output streams, its side packets and a logger. One context instance
/// belongs to one node and is reused across invocations.
/// </summary>
public sealed class CalculatorContext
{
    private readonly CalculatorGraph.NodeRuntime _node;
    private readonly Dictionary<string, int> _inputIndex;
    private readonly Dictionary<string, int> _outputIndex;
    internal readonly Packet?[] InputPackets;

    internal CalculatorContext(CalculatorGraph.NodeRuntime node, ILogger logger, CancellationToken cancellationToken)
    {
        _node = node;
        Logger = logger;
        CancellationToken = cancellationToken;
        _inputIndex = node.Contract.Inputs.Select((p, i) => (p.Tag, i)).ToDictionary(x => x.Tag, x => x.i, StringComparer.Ordinal);
        _outputIndex = node.Contract.Outputs.Select((p, i) => (p.Tag, i)).ToDictionary(x => x.Tag, x => x.i, StringComparer.Ordinal);
        InputPackets = new Packet?[node.Contract.Inputs.Count];
    }

    /// <summary>The node's name in the graph.</summary>
    public string NodeName => _node.Name;

    /// <summary>The timestamp being processed.</summary>
    public Timestamp InputTimestamp { get; internal set; } = Timestamp.Unset;

    /// <summary>Logger scoped to the graph.</summary>
    public ILogger Logger { get; }

    /// <summary>Signalled when the graph is cancelled or fails.</summary>
    public CancellationToken CancellationToken { get; }

    /// <summary>Input tags declared by the node.</summary>
    public IEnumerable<string> InputTags => _inputIndex.Keys;

    /// <summary>Output tags declared by the node.</summary>
    public IEnumerable<string> OutputTags => _outputIndex.Keys;

    // ------------------------------------------------------------------ inputs

    /// <summary>True when input <paramref name="tag"/> has a packet at <see cref="InputTimestamp"/>.</summary>
    public bool HasInput(string tag) => InputPackets[InputIndex(tag)] is not null;

    /// <summary>The raw packet of input <paramref name="tag"/>, or null when it has none at this timestamp.</summary>
    public Packet? GetInputPacket(string tag) => InputPackets[InputIndex(tag)];

    /// <summary>The payload of input <paramref name="tag"/>.</summary>
    /// <exception cref="InvalidOperationException">The input has no packet at this timestamp.</exception>
    public T GetInput<T>(string tag) =>
        TryGetInput<T>(tag, out var value)
            ? value
            : throw new InvalidOperationException($"Node '{NodeName}': input '{tag}' has no packet at {InputTimestamp}.");

    /// <summary>Gets the payload of input <paramref name="tag"/> when present.</summary>
    public bool TryGetInput<T>(string tag, out T value)
    {
        var p = InputPackets[InputIndex(tag)];
        if (p is null)
        {
            value = default!;
            return false;
        }
        value = p is Packet<T> typed ? typed.Value : p.As<T>().Value;
        return true;
    }

    /// <summary>The payload of input <paramref name="tag"/>, or <paramref name="fallback"/> when absent.</summary>
    public T GetInputOrDefault<T>(string tag, T fallback = default!) => TryGetInput<T>(tag, out var v) ? v : fallback;

    // ------------------------------------------------------------------ outputs

    /// <summary>True when output <paramref name="tag"/> is connected to a stream.</summary>
    public bool IsOutputConnected(string tag) => _node.Outputs[OutputIndex(tag)] is not null;

    /// <summary>Sends a value on output <paramref name="tag"/> at <see cref="InputTimestamp"/>.</summary>
    public void Send<T>(string tag, T value) => SendPacket(tag, new Packet<T>(value, InputTimestamp));

    /// <summary>Sends a value on output <paramref name="tag"/> at an explicit timestamp.</summary>
    public void Send<T>(string tag, T value, Timestamp timestamp) => SendPacket(tag, new Packet<T>(value, timestamp));

    /// <summary>Sends a packet on output <paramref name="tag"/>. Unconnected outputs ignore it.</summary>
    public void SendPacket(string tag, Packet packet)
    {
        ArgumentNullException.ThrowIfNull(packet);
        var port = _node.Outputs[OutputIndex(tag)];
        if (port is null) return;
        _node.Graph.Emit(port, packet, _node.Name);
    }

    /// <summary>
    /// Promises that output <paramref name="tag"/> will carry no packet earlier than <paramref name="bound"/>,
    /// letting downstream nodes proceed without waiting.
    /// </summary>
    public void SetNextTimestampBound(string tag, Timestamp bound)
    {
        var port = _node.Outputs[OutputIndex(tag)];
        if (port is not null) _node.Graph.AdvanceBound(port, bound);
    }

    // ------------------------------------------------------------------ side packets

    /// <summary>The value of side packet <paramref name="tag"/>.</summary>
    public T GetSidePacket<T>(string tag) =>
        TryGetSidePacket<T>(tag, out var v)
            ? v
            : throw new InvalidOperationException($"Node '{NodeName}': side packet '{tag}' was not provided.");

    /// <summary>Gets the value of side packet <paramref name="tag"/> when provided.</summary>
    public bool TryGetSidePacket<T>(string tag, out T value)
    {
        if (_node.SidePacketValues.TryGetValue(tag, out var raw) && raw is T typed)
        {
            value = typed;
            return true;
        }
        value = default!;
        return false;
    }

    private int InputIndex(string tag) =>
        _inputIndex.TryGetValue(tag, out int i) ? i : throw new ArgumentException($"Node '{NodeName}' has no input '{tag}'.", nameof(tag));

    private int OutputIndex(string tag) =>
        _outputIndex.TryGetValue(tag, out int i) ? i : throw new ArgumentException($"Node '{NodeName}' has no output '{tag}'.", nameof(tag));
}
