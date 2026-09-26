namespace MediaPipeNet.Framework;

/// <summary>How a node's inputs are grouped into invocations of <see cref="ICalculatorNode.ProcessAsync"/>.</summary>
public enum InputPolicy
{
    /// <summary>
    /// MediaPipe's default input stream handler: the node runs once per timestamp, after every input
    /// either delivered its packet for that timestamp or advanced its timestamp bound past it.
    /// </summary>
    Synchronized = 0,

    /// <summary>
    /// Every packet is processed as soon as it arrives, independently of the other inputs
    /// (MediaPipe's <c>ImmediateInputStreamHandler</c>). Useful for aggregators and flow limiters.
    /// The same timestamp can reach the node once per input, so emit at most one packet per timestamp.
    /// </summary>
    Immediate = 1,
}

/// <summary>A named, typed port of a node or a side packet.</summary>
/// <param name="Tag">Port tag, e.g. <c>IMAGE</c>.</param>
/// <param name="Type">Payload type; <see cref="object"/> accepts anything.</param>
/// <param name="Optional">Whether the port may be left unconnected.</param>
public sealed record PortSpec(string Tag, Type Type, bool Optional = false);

/// <summary>
/// Declares a node's ports: its input and output streams, the side packets it consumes and the
/// input policy. Filled by <see cref="ICalculatorNode.GetContract"/> and validated when the graph is built.
/// </summary>
public sealed class CalculatorContract
{
    private readonly List<PortSpec> _inputs = [];
    private readonly List<PortSpec> _outputs = [];
    private readonly List<PortSpec> _sidePackets = [];

    /// <summary>Declared inputs.</summary>
    public IReadOnlyList<PortSpec> Inputs => _inputs;

    /// <summary>Declared outputs.</summary>
    public IReadOnlyList<PortSpec> Outputs => _outputs;

    /// <summary>Declared input side packets.</summary>
    public IReadOnlyList<PortSpec> InputSidePackets => _sidePackets;

    /// <summary>The input policy (default <see cref="InputPolicy.Synchronized"/>).</summary>
    public InputPolicy InputPolicy { get; set; }

    /// <summary>
    /// When true, a node whose outputs receive no packet for a processed timestamp automatically
    /// advances those outputs' bound past it (MediaPipe's <c>SetOffset(0)</c>). Default true.
    /// </summary>
    public bool PropagateTimestampBounds { get; set; } = true;

    /// <summary>Declares an input stream.</summary>
    public CalculatorContract AddInput<T>(string tag, bool optional = false) => Add(_inputs, new PortSpec(tag, typeof(T), optional));

    /// <summary>Declares an input stream with a runtime type.</summary>
    public CalculatorContract AddInput(string tag, Type type, bool optional = false) => Add(_inputs, new PortSpec(tag, type, optional));

    /// <summary>Declares an output stream.</summary>
    public CalculatorContract AddOutput<T>(string tag, bool optional = true) => Add(_outputs, new PortSpec(tag, typeof(T), optional));

    /// <summary>Declares an output stream with a runtime type.</summary>
    public CalculatorContract AddOutput(string tag, Type type, bool optional = true) => Add(_outputs, new PortSpec(tag, type, optional));

    /// <summary>Declares an input side packet.</summary>
    public CalculatorContract AddInputSidePacket<T>(string tag, bool optional = false) => Add(_sidePackets, new PortSpec(tag, typeof(T), optional));

    private CalculatorContract Add(List<PortSpec> list, PortSpec spec)
    {
        ArgumentException.ThrowIfNullOrEmpty(spec.Tag);
        if (list.Exists(p => p.Tag == spec.Tag))
            throw new ArgumentException($"Port '{spec.Tag}' is declared twice.");
        list.Add(spec);
        return this;
    }
}
