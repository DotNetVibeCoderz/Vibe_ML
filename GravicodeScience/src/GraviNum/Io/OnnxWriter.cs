using System.Buffers.Binary;
using System.Text;

namespace Gravicode.Science.GraviNum.Io;

/// <summary>
/// Builds an ONNX graph and writes it out, without a protobuf dependency.
/// </summary>
/// <remarks>
/// <para>
/// The mirror of <see cref="OnnxReader"/>, and dependency-free for the same reason: the format is
/// protobuf, only a handful of its fields are needed, and pulling in a code generator or ONNX
/// Runtime to emit a few hundred bytes would be a poor trade.
/// </para>
/// <para>
/// This exists so a model fitted here can be served anywhere — ONNX Runtime, Python, a browser —
/// which is the other half of the interop story from reading weights in. The graph is built from
/// <b>core</b> operators only (<c>Sub</c>, <c>Div</c>, <c>MatMul</c>, <c>Add</c>, <c>ArgMax</c>)
/// rather than the <c>ai.onnx.ml</c> set, because core operators are implemented by every runtime
/// while the ML ones are not.
/// </para>
/// <para>
/// Everything is written as <c>float</c>. ONNX runtimes universally support float32 tensors and
/// only patchily support float64, so exporting doubles would produce a file many tools refuse to
/// load. That is a real loss of precision at the boundary, and it is the caller's to know about.
/// </para>
/// </remarks>
public sealed class OnnxGraphBuilder
{
    private const int FloatType = 1;
    private const int Int64Type = 7;

    private readonly List<byte[]> _nodes = [];
    private readonly List<byte[]> _initializers = [];
    private int _counter;

    /// <summary>Creates a builder for a graph with one input and one output.</summary>
    /// <param name="inputName">Name of the graph input.</param>
    /// <param name="features">Number of input columns; the row count stays symbolic.</param>
    public OnnxGraphBuilder(string inputName = "input", int features = 0)
    {
        InputName = inputName;
        Features = features;
    }

    /// <summary>Name of the graph input.</summary>
    public string InputName { get; }

    /// <summary>Number of input columns.</summary>
    public int Features { get; }

    /// <summary>A fresh intermediate value name.</summary>
    private string NextName(string prefix) => $"{prefix}_{_counter++}";

    /// <summary>Adds a constant tensor to the graph and returns its name.</summary>
    public string AddInitializer(string name, NdArray values, params int[] shape)
    {
        _initializers.Add(Tensor(name, shape.Length > 0 ? shape : [.. values.Shape], values));
        return name;
    }

    /// <summary>Appends a node computing <c>output = op(inputs...)</c>.</summary>
    public string AddNode(string opType, string[] inputs, string? outputName = null,
        params byte[][] attributes)
    {
        var output = outputName ?? NextName(opType.ToLowerInvariant());
        var body = new List<byte>();

        foreach (var input in inputs) WriteString(body, 1, input);
        WriteString(body, 2, output);
        WriteString(body, 3, NextName("node"));
        WriteString(body, 4, opType);
        foreach (var attribute in attributes) WriteBytes(body, 5, attribute);

        _nodes.Add([.. body]);
        return output;
    }

    /// <summary>An integer attribute, for operators such as <c>ArgMax</c>.</summary>
    public static byte[] IntAttribute(string name, long value)
    {
        var body = new List<byte>();
        WriteString(body, 1, name);
        WriteVarintField(body, 3, (ulong)value);   // i
        WriteVarintField(body, 20, 2);             // type = INT
        return [.. body];
    }

    /// <summary>Serialises the finished model.</summary>
    /// <param name="outputName">Name of the graph output produced by the last node.</param>
    /// <param name="outputColumns">Columns the output has, or 0 to leave it symbolic.</param>
    /// <param name="outputIsInt64">True when the output is a class index rather than a score.</param>
    public byte[] Build(string outputName, int outputColumns = 0, bool outputIsInt64 = false)
    {
        var graph = new List<byte>();

        foreach (var node in _nodes) WriteBytes(graph, 1, node);
        WriteString(graph, 2, "gravicode");
        foreach (var initializer in _initializers) WriteBytes(graph, 5, initializer);

        WriteBytes(graph, 11, ValueInfo(InputName, FloatType, Features));
        WriteBytes(graph, 12, ValueInfo(outputName, outputIsInt64 ? Int64Type : FloatType, outputColumns));

        var model = new List<byte>();
        WriteVarintField(model, 1, 8);                              // ir_version
        WriteString(model, 2, "Gravicode.Science");                 // producer_name
        WriteBytes(model, 7, [.. graph]);                           // graph
        WriteBytes(model, 8, OperatorSet());                        // opset_import
        return [.. model];
    }

    /// <summary>Writes the model to disk.</summary>
    public void Save(string path, string outputName, int outputColumns = 0, bool outputIsInt64 = false)
        => File.WriteAllBytes(path, Build(outputName, outputColumns, outputIsInt64));

    // ---------------------------------------------------------------- protobuf pieces

    /// <summary>The default operator set. 13 is old enough to be universal and new enough to matter.</summary>
    private static byte[] OperatorSet()
    {
        var body = new List<byte>();
        WriteString(body, 1, "");        // domain: the default one
        WriteVarintField(body, 2, 13);   // version
        return [.. body];
    }

    private static byte[] Tensor(string name, int[] shape, NdArray values)
    {
        var body = new List<byte>();

        foreach (var d in shape) WriteVarintField(body, 1, (ulong)d);
        WriteVarintField(body, 2, FloatType);
        WriteString(body, 8, name);

        var flat = values.AsContiguous().ToArray();
        var raw = new byte[flat.Length * 4];
        for (var i = 0; i < flat.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan(i * 4), (float)flat[i]);

        WriteBytes(body, 9, raw);
        return [.. body];
    }

    /// <summary>
    /// A graph input or output declaration.
    /// </summary>
    /// <remarks>
    /// The row dimension is written as a <em>symbolic</em> name rather than a number, so the
    /// exported model accepts any batch size. Pinning it to whatever the training set happened to
    /// have is a common export bug and makes the model useless for single-row inference.
    /// </remarks>
    private static byte[] ValueInfo(string name, int elementType, int columns)
    {
        var dims = new List<byte>();

        var batch = new List<byte>();
        WriteString(batch, 2, "batch");                 // dim_param
        WriteBytes(dims, 1, [.. batch]);

        if (columns > 0)
        {
            var fixedDim = new List<byte>();
            WriteVarintField(fixedDim, 1, (ulong)columns);   // dim_value
            WriteBytes(dims, 1, [.. fixedDim]);
        }

        var shape = new List<byte>();
        shape.AddRange(dims);

        var tensorType = new List<byte>();
        WriteVarintField(tensorType, 1, (ulong)elementType);
        WriteBytes(tensorType, 2, [.. shape]);

        var type = new List<byte>();
        WriteBytes(type, 1, [.. tensorType]);

        var info = new List<byte>();
        WriteString(info, 1, name);
        WriteBytes(info, 2, [.. type]);
        return [.. info];
    }

    private static void WriteVarint(List<byte> to, ulong value)
    {
        while (value >= 0x80)
        {
            to.Add((byte)(value | 0x80));
            value >>= 7;
        }
        to.Add((byte)value);
    }

    private static void WriteVarintField(List<byte> to, int field, ulong value)
    {
        WriteVarint(to, (ulong)(field << 3));
        WriteVarint(to, value);
    }

    private static void WriteBytes(List<byte> to, int field, ReadOnlySpan<byte> payload)
    {
        WriteVarint(to, (ulong)((field << 3) | 2));
        WriteVarint(to, (ulong)payload.Length);
        to.AddRange(payload.ToArray());
    }

    private static void WriteString(List<byte> to, int field, string value)
        => WriteBytes(to, field, Encoding.UTF8.GetBytes(value));
}
