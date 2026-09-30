using System.Buffers.Binary;
using System.Text;

namespace Gravicode.HFNet.GraviOptimum;

/// <summary>One node of an ONNX graph: what it computes and what it reads and writes.</summary>
/// <param name="Name">The node's name. Exporters from PyTorch write the module path here, such as
/// <c>/down_blocks.0/attentions.0/transformer_blocks.0/attn1/to_q/MatMul</c>.</param>
/// <param name="OpType">The operator, such as <c>MatMul</c>.</param>
/// <param name="Inputs">Input value names, in order.</param>
/// <param name="Outputs">Output value names, in order.</param>
/// <param name="Attributes">Attribute names.</param>
public sealed record OnnxNode(string Name, string OpType, IReadOnlyList<string> Inputs, IReadOnlyList<string> Outputs, IReadOnlyList<string> Attributes);

/// <summary>A constant tensor stored in an ONNX model - a weight, usually.</summary>
/// <param name="Name">The value name nodes refer to it by.</param>
/// <param name="Dimensions">Its shape.</param>
/// <param name="DataType">ONNX's element type code: 1 is float32, 10 float16.</param>
public sealed record OnnxInitializer(string Name, IReadOnlyList<long> Dimensions, int DataType)
{
    /// <summary>Where the bytes are: an offset into the model file, or a separate file.</summary>
    internal string File { get; init; } = "";

    internal long Offset { get; init; }

    internal long Length { get; init; }

    /// <summary>Values stored in <c>float_data</c> rather than as raw bytes.</summary>
    internal float[]? Inline { get; init; }

    /// <summary>How many elements the shape holds.</summary>
    public long Count => Dimensions.Aggregate(1L, (a, b) => a * b);
}

/// <summary>
/// Reads the structure of an ONNX model - nodes and initializers - without loading its weights.
/// </summary>
/// <remarks>
/// <para>
/// An ONNX file is a protobuf <c>ModelProto</c>. This walks it field by field, reading names and
/// shapes and seeking past tensor bytes, so listing a 1.7 GB UNet costs its metadata, not its
/// size. A weight is read only when asked for, from the model file or from the external data file
/// a large export keeps beside it.
/// </para>
/// <para>
/// It exists for two jobs ONNX Runtime has no API for: finding the weight behind a named layer, so
/// a LoRA adapter can be merged into it through <c>SessionOptions.AddInitializer</c>, and rewriting
/// one attribute of a small graph.
/// </para>
/// </remarks>
public sealed class OnnxModelFile
{
    private OnnxModelFile(string path, List<OnnxNode> nodes, Dictionary<string, OnnxInitializer> initializers)
    {
        Path = path;
        Nodes = nodes;
        Initializers = initializers;
    }

    /// <summary>The model file.</summary>
    public string Path { get; }

    /// <summary>The graph's nodes, in file order.</summary>
    public IReadOnlyList<OnnxNode> Nodes { get; }

    /// <summary>The graph's initializers, by name.</summary>
    public IReadOnlyDictionary<string, OnnxInitializer> Initializers { get; }

    /// <summary>Reads a model's structure.</summary>
    /// <param name="path">A <c>.onnx</c> file.</param>
    public static OnnxModelFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);

        var nodes = new List<OnnxNode>();
        var initializers = new Dictionary<string, OnnxInitializer>(StringComparer.Ordinal);
        var directory = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(path)) ?? ".";

        var end = stream.Length;
        while (stream.Position < end)
        {
            var (field, wire) = ReadTag(stream);
            if (field == 7 && wire == 2)
            {
                var length = (long)ReadVarint(stream);
                ReadGraph(stream, stream.Position + length, path, directory, nodes, initializers);
            }
            else
            {
                Skip(stream, wire);
            }
        }

        return new OnnxModelFile(path, nodes, initializers);
    }

    /// <summary>An initializer's values as float32, widening float16.</summary>
    /// <exception cref="NotSupportedException">The tensor is neither float32 nor float16.</exception>
    public float[] ReadFloats(OnnxInitializer initializer)
    {
        ArgumentNullException.ThrowIfNull(initializer);
        if (initializer.Inline is { } inline) return inline;

        var bytes = new byte[initializer.Length];
        using (var stream = new FileStream(initializer.File, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            stream.Position = initializer.Offset;
            stream.ReadExactly(bytes);
        }

        switch (initializer.DataType)
        {
            case 1:
            {
                var values = new float[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, values, 0, values.Length * 4);
                return values;
            }

            case 10:
            {
                var values = new float[bytes.Length / 2];
                for (var i = 0; i < values.Length; i++) values[i] = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(i * 2)));
                return values;
            }

            default:
                throw new NotSupportedException(
                    $"Initializer '{initializer.Name}' has ONNX data type {initializer.DataType}; only float32 (1) and float16 (10) are read.");
        }
    }

    /// <summary>
    /// A copy of a small model with one float attribute set on every node of an operator type.
    /// </summary>
    /// <param name="path">The model; must be under 512 MB, since it is rewritten in memory.</param>
    /// <param name="opType">Which nodes to change, such as <c>RandomNormalLike</c>.</param>
    /// <param name="attribute">The attribute to set, such as <c>scale</c>.</param>
    /// <param name="value">Its new value.</param>
    /// <returns>The rewritten model and how many nodes changed.</returns>
    public static (byte[] Model, int Changed) SetFloatAttribute(string path, string opType, string attribute, float value)
    {
        var info = new FileInfo(path);
        if (info.Length > 512L << 20)
        {
            throw new NotSupportedException($"'{path}' is {info.Length >> 20} MB; attributes are only rewritten in models under 512 MB.");
        }

        var bytes = File.ReadAllBytes(path);
        var changed = 0;

        var model = new MemoryStream();
        var position = 0;
        while (position < bytes.Length)
        {
            var start = position;
            var (field, wire) = ReadTag(bytes, ref position);
            if (field == 7 && wire == 2)
            {
                var length = (int)ReadVarint(bytes, ref position);
                var graph = RewriteGraph(bytes.AsSpan(position, length), opType, attribute, value, ref changed);
                position += length;
                WriteTag(model, 7, 2);
                WriteVarint(model, (ulong)graph.Length);
                model.Write(graph);
            }
            else
            {
                SkipBytes(bytes, ref position, wire);
                model.Write(bytes.AsSpan(start, position - start));
            }
        }

        return (model.ToArray(), changed);
    }

    // ------------------------------------------------------------------ graph reading

    private static void ReadGraph(
        Stream stream, long end, string path, string directory, List<OnnxNode> nodes, Dictionary<string, OnnxInitializer> initializers)
    {
        while (stream.Position < end)
        {
            var (field, wire) = ReadTag(stream);

            if (field == 1 && wire == 2)
            {
                var length = (int)ReadVarint(stream);
                var buffer = new byte[length];
                stream.ReadExactly(buffer);
                nodes.Add(ParseNode(buffer));
            }
            else if (field == 5 && wire == 2)
            {
                var length = (long)ReadVarint(stream);
                var initializer = ReadTensor(stream, stream.Position + length, path, directory);
                initializers[initializer.Name] = initializer;
            }
            else
            {
                Skip(stream, wire);
            }
        }
    }

    private static OnnxNode ParseNode(byte[] bytes)
    {
        var inputs = new List<string>();
        var outputs = new List<string>();
        var attributes = new List<string>();
        var name = "";
        var op = "";

        var position = 0;
        while (position < bytes.Length)
        {
            var (field, wire) = ReadTag(bytes, ref position);
            if (wire == 2)
            {
                var length = (int)ReadVarint(bytes, ref position);
                var span = bytes.AsSpan(position, length);
                switch (field)
                {
                    case 1: inputs.Add(Encoding.UTF8.GetString(span)); break;
                    case 2: outputs.Add(Encoding.UTF8.GetString(span)); break;
                    case 3: name = Encoding.UTF8.GetString(span); break;
                    case 4: op = Encoding.UTF8.GetString(span); break;
                    case 5: attributes.Add(AttributeName(span)); break;
                }

                position += length;
            }
            else
            {
                SkipBytes(bytes, ref position, wire);
            }
        }

        return new OnnxNode(name, op, inputs, outputs, attributes);
    }

    private static string AttributeName(ReadOnlySpan<byte> attribute)
    {
        var position = 0;
        var bytes = attribute.ToArray();
        while (position < bytes.Length)
        {
            var (field, wire) = ReadTag(bytes, ref position);
            if (field == 1 && wire == 2)
            {
                var length = (int)ReadVarint(bytes, ref position);
                return Encoding.UTF8.GetString(bytes, position, length);
            }

            SkipBytes(bytes, ref position, wire);
        }

        return "";
    }

    private static OnnxInitializer ReadTensor(Stream stream, long end, string path, string directory)
    {
        var dims = new List<long>();
        var dataType = 0;
        var name = "";
        long rawOffset = -1, rawLength = 0;
        float[]? floats = null;
        var external = new Dictionary<string, string>(StringComparer.Ordinal);

        while (stream.Position < end)
        {
            var (field, wire) = ReadTag(stream);
            switch (field)
            {
                case 1 when wire == 0:
                    dims.Add((long)ReadVarint(stream));
                    break;

                case 1 when wire == 2:
                {
                    var packedEnd = (long)ReadVarint(stream) + stream.Position;
                    while (stream.Position < packedEnd) dims.Add((long)ReadVarint(stream));
                    break;
                }

                case 2 when wire == 0:
                    dataType = (int)ReadVarint(stream);
                    break;

                case 4 when wire == 2:
                {
                    var length = (int)ReadVarint(stream);
                    var buffer = new byte[length];
                    stream.ReadExactly(buffer);
                    floats = new float[length / 4];
                    Buffer.BlockCopy(buffer, 0, floats, 0, floats.Length * 4);
                    break;
                }

                case 8 when wire == 2:
                    name = ReadString(stream);
                    break;

                case 9 when wire == 2:
                    rawLength = (long)ReadVarint(stream);
                    rawOffset = stream.Position;
                    stream.Position += rawLength;
                    break;

                case 13 when wire == 2:
                {
                    var entryEnd = (long)ReadVarint(stream) + stream.Position;
                    string key = "", value = "";
                    while (stream.Position < entryEnd)
                    {
                        var (entryField, entryWire) = ReadTag(stream);
                        if (entryField == 1 && entryWire == 2) key = ReadString(stream);
                        else if (entryField == 2 && entryWire == 2) value = ReadString(stream);
                        else Skip(stream, entryWire);
                    }

                    external[key] = value;
                    break;
                }

                default:
                    Skip(stream, wire);
                    break;
            }
        }

        if (external.TryGetValue("location", out var location))
        {
            var bytesPerElement = dataType == 10 ? 2 : 4;
            var count = dims.Aggregate(1L, (a, b) => a * b);
            return new OnnxInitializer(name, dims, dataType)
            {
                File = System.IO.Path.Combine(directory, location),
                Offset = external.TryGetValue("offset", out var offset) ? long.Parse(offset) : 0,
                Length = external.TryGetValue("length", out var length) ? long.Parse(length) : count * bytesPerElement,
            };
        }

        return new OnnxInitializer(name, dims, dataType)
        {
            File = path,
            Offset = rawOffset,
            Length = rawLength,
            Inline = rawOffset < 0 ? floats ?? [] : null,
        };
    }

    // ------------------------------------------------------------------ graph rewriting

    private static byte[] RewriteGraph(ReadOnlySpan<byte> graph, string opType, string attribute, float value, ref int changed)
    {
        var bytes = graph.ToArray();
        var output = new MemoryStream();
        var position = 0;

        while (position < bytes.Length)
        {
            var start = position;
            var (field, wire) = ReadTag(bytes, ref position);

            if (field == 1 && wire == 2)
            {
                var length = (int)ReadVarint(bytes, ref position);
                var node = bytes.AsSpan(position, length).ToArray();
                position += length;

                var parsed = ParseNode(node);
                var rewritten = parsed.OpType == opType ? WithFloatAttribute(node, attribute, value) : node;
                if (parsed.OpType == opType) changed++;

                WriteTag(output, 1, 2);
                WriteVarint(output, (ulong)rewritten.Length);
                output.Write(rewritten);
            }
            else
            {
                SkipBytes(bytes, ref position, wire);
                output.Write(bytes.AsSpan(start, position - start));
            }
        }

        return output.ToArray();
    }

    /// <summary>A node's bytes with the named attribute replaced by, or given, a float value.</summary>
    private static byte[] WithFloatAttribute(byte[] node, string attribute, float value)
    {
        var output = new MemoryStream();
        var position = 0;

        while (position < node.Length)
        {
            var start = position;
            var (field, wire) = ReadTag(node, ref position);
            if (field == 5 && wire == 2)
            {
                var length = (int)ReadVarint(node, ref position);
                var name = AttributeName(node.AsSpan(position, length));
                position += length;
                if (name == attribute) continue;   // dropped; re-added below with the new value
                output.Write(node.AsSpan(start, position - start));
            }
            else
            {
                SkipBytes(node, ref position, wire);
                output.Write(node.AsSpan(start, position - start));
            }
        }

        // AttributeProto: name (1), f (2, fixed32), type (20) = FLOAT (1).
        var attributeBytes = new MemoryStream();
        var nameBytes = Encoding.UTF8.GetBytes(attribute);
        WriteTag(attributeBytes, 1, 2);
        WriteVarint(attributeBytes, (ulong)nameBytes.Length);
        attributeBytes.Write(nameBytes);
        WriteTag(attributeBytes, 2, 5);
        Span<byte> single = stackalloc byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(single, value);
        attributeBytes.Write(single);
        WriteTag(attributeBytes, 20, 0);
        WriteVarint(attributeBytes, 1);

        WriteTag(output, 5, 2);
        WriteVarint(output, (ulong)attributeBytes.Length);
        output.Write(attributeBytes.ToArray());

        return output.ToArray();
    }

    // ------------------------------------------------------------------ protobuf primitives

    private static (int Field, int Wire) ReadTag(Stream stream)
    {
        var tag = ReadVarint(stream);
        return ((int)(tag >> 3), (int)(tag & 7));
    }

    private static (int Field, int Wire) ReadTag(byte[] bytes, ref int position)
    {
        var tag = ReadVarint(bytes, ref position);
        return ((int)(tag >> 3), (int)(tag & 7));
    }

    private static ulong ReadVarint(Stream stream)
    {
        ulong result = 0;
        for (var shift = 0; shift < 64; shift += 7)
        {
            var b = stream.ReadByte();
            if (b < 0) throw new InvalidDataException("The ONNX file ends inside a varint.");
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
        }

        throw new InvalidDataException("A varint in the ONNX file is longer than 64 bits.");
    }

    private static ulong ReadVarint(byte[] bytes, ref int position)
    {
        ulong result = 0;
        for (var shift = 0; shift < 64; shift += 7)
        {
            var b = bytes[position++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
        }

        throw new InvalidDataException("A varint in the ONNX file is longer than 64 bits.");
    }

    private static string ReadString(Stream stream)
    {
        var length = (int)ReadVarint(stream);
        var buffer = new byte[length];
        stream.ReadExactly(buffer);
        return Encoding.UTF8.GetString(buffer);
    }

    private static void Skip(Stream stream, int wire)
    {
        switch (wire)
        {
            case 0: ReadVarint(stream); break;
            case 1: stream.Position += 8; break;
            case 2:
            {
                // Read the length before touching Position: "Position += ReadVarint(...)" reads
                // Position first, then the varint moves it, and the sum lands short by the varint.
                var length = (long)ReadVarint(stream);
                stream.Position += length;
                break;
            }

            case 5: stream.Position += 4; break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wire} in the ONNX file.");
        }
    }

    private static void SkipBytes(byte[] bytes, ref int position, int wire)
    {
        switch (wire)
        {
            case 0: ReadVarint(bytes, ref position); break;
            case 1: position += 8; break;
            case 2:
            {
                var length = (int)ReadVarint(bytes, ref position);
                position += length;
                break;
            }

            case 5: position += 4; break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wire} in the ONNX file.");
        }
    }

    private static void WriteTag(Stream stream, int field, int wire) => WriteVarint(stream, (ulong)((field << 3) | wire));

    private static void WriteVarint(Stream stream, ulong value)
    {
        while (value >= 0x80)
        {
            stream.WriteByte((byte)(value | 0x80));
            value >>= 7;
        }

        stream.WriteByte((byte)value);
    }
}
