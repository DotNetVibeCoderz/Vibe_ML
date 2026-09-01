using System.Buffers.Binary;

namespace Gravicode.Science.GraviNum.Io;

/// <summary>One named tensor read from an ONNX file.</summary>
/// <param name="Name">The initializer's name, as the exporting framework wrote it.</param>
/// <param name="Shape">Dimensions, outermost first.</param>
/// <param name="Values">Values in row-major order, widened to <see cref="double"/>.</param>
public sealed record OnnxTensor(string Name, int[] Shape, double[] Values)
{
    /// <summary>The tensor as an <see cref="NdArray"/>.</summary>
    public NdArray ToNdArray() => new(Values, Shape);

    /// <summary>Total number of elements.</summary>
    public long Count => Values.LongLength;

    /// <inheritdoc />
    public override string ToString() => $"{Name} [{string.Join("x", Shape)}]";
}

/// <summary>
/// Reads the weights out of an ONNX file, without a protobuf or ONNX Runtime dependency.
/// </summary>
/// <remarks>
/// <para>
/// This is a weight reader, not a runtime. It extracts the graph's <em>initializers</em> — the
/// named constant tensors that hold trained parameters — and stops there. It does not execute a
/// graph, so it cannot run an arbitrary model; what it can do is give this library's own layers
/// real trained weights, which is the gap that matters most, since a freshly built
/// <c>TransformerModel</c> is otherwise random for ever.
/// </para>
/// <para>
/// Deliberately dependency-free. ONNX Runtime is a large platform-specific native package, and
/// pulling it in to read a few arrays would cost every user of this library megabytes they did not
/// ask for. Only the protobuf wire format is needed, and only a handful of its fields.
/// </para>
/// <para>
/// Everything widens to <c>double</c> on the way in, because that is what the rest of the library
/// speaks. Weights are usually stored as float32, so this costs memory but loses nothing: every
/// float32 is exactly representable as a double.
/// </para>
/// </remarks>
public static class OnnxReader
{
    // Protobuf wire types.
    private const int Varint = 0;
    private const int Fixed64 = 1;
    private const int LengthDelimited = 2;
    private const int Fixed32 = 5;

    // ONNX TensorProto.DataType values, from the specification.
    private const int Float = 1;
    private const int Int8 = 3;
    private const int Int16 = 5;
    private const int Int32 = 6;
    private const int Int64 = 7;
    private const int Float16 = 10;
    private const int Double = 11;

    /// <summary>Reads every initializer from an ONNX file.</summary>
    /// <param name="path">Path to a <c>.onnx</c> file.</param>
    public static IReadOnlyList<OnnxTensor> ReadWeights(string path)
        => ReadWeights(File.ReadAllBytes(path));

    /// <summary>Reads every initializer from ONNX file contents.</summary>
    public static IReadOnlyList<OnnxTensor> ReadWeights(ReadOnlySpan<byte> content)
    {
        var tensors = new List<OnnxTensor>();

        // ModelProto field 7 is the graph. Everything else — version, producer, metadata — is
        // skipped, which is also what makes this robust to ONNX versions it has never seen.
        var model = new Cursor(content);
        while (model.TryReadTag(out var field, out var wire))
        {
            if (field == 7 && wire == LengthDelimited) ReadGraph(model.ReadBytes(), tensors);
            else model.Skip(wire);
        }

        return tensors;
    }

    /// <summary>Reads the initializers of the model at <paramref name="path"/>, keyed by name.</summary>
    public static Dictionary<string, OnnxTensor> ReadWeightsByName(string path)
        => ReadWeights(path).ToDictionary(t => t.Name, StringComparer.Ordinal);

    private static void ReadGraph(ReadOnlySpan<byte> graph, List<OnnxTensor> tensors)
    {
        // GraphProto field 5 is the repeated initializer list.
        var cursor = new Cursor(graph);
        while (cursor.TryReadTag(out var field, out var wire))
        {
            if (field == 5 && wire == LengthDelimited)
            {
                var tensor = ReadTensor(cursor.ReadBytes());
                if (tensor is not null) tensors.Add(tensor);
            }
            else cursor.Skip(wire);
        }
    }

    private static OnnxTensor? ReadTensor(ReadOnlySpan<byte> body)
    {
        var dims = new List<int>();
        var dataType = 0;
        var name = string.Empty;
        ReadOnlySpan<byte> raw = default;
        double[]? typed = null;

        var cursor = new Cursor(body);
        while (cursor.TryReadTag(out var field, out var wire))
        {
            switch (field, wire)
            {
                // dims, packed or one at a time depending on the exporter.
                case (1, Varint):
                    dims.Add((int)cursor.ReadVarint());
                    break;
                case (1, LengthDelimited):
                    var packed = new Cursor(cursor.ReadBytes());
                    while (!packed.AtEnd) dims.Add((int)packed.ReadVarint());
                    break;

                case (2, Varint): dataType = (int)cursor.ReadVarint(); break;
                case (8, LengthDelimited): name = cursor.ReadString(); break;
                case (9, LengthDelimited): raw = cursor.ReadBytes(); break;

                // float_data and double_data: used when the exporter did not pack into raw_data.
                case (4, LengthDelimited): typed = ReadPackedFloats(cursor.ReadBytes()); break;
                case (10, LengthDelimited): typed = ReadPackedDoubles(cursor.ReadBytes()); break;

                default: cursor.Skip(wire); break;
            }
        }

        var shape = dims.Count > 0 ? dims.ToArray() : [1];
        var values = typed ?? (raw.IsEmpty ? null : Decode(raw, dataType));

        // A tensor whose type this reader does not decode is skipped rather than guessed at:
        // returning wrong numbers silently is far worse than returning fewer of them.
        if (values is null) return null;

        var expected = shape.Aggregate(1L, (a, b) => a * b);
        if (values.LongLength != expected) return null;

        return new OnnxTensor(name, shape, values);
    }

    /// <summary>Widens raw little-endian tensor bytes to <see cref="double"/>.</summary>
    private static double[]? Decode(ReadOnlySpan<byte> raw, int dataType)
    {
        switch (dataType)
        {
            case Float:
            {
                var result = new double[raw.Length / 4];
                for (var i = 0; i < result.Length; i++)
                    result[i] = BinaryPrimitives.ReadSingleLittleEndian(raw[(i * 4)..]);
                return result;
            }
            case Double:
            {
                var result = new double[raw.Length / 8];
                for (var i = 0; i < result.Length; i++)
                    result[i] = BinaryPrimitives.ReadDoubleLittleEndian(raw[(i * 8)..]);
                return result;
            }
            case Float16:
            {
                var result = new double[raw.Length / 2];
                for (var i = 0; i < result.Length; i++)
                    result[i] = (double)BinaryPrimitives.ReadHalfLittleEndian(raw[(i * 2)..]);
                return result;
            }
            case Int64:
            {
                var result = new double[raw.Length / 8];
                for (var i = 0; i < result.Length; i++)
                    result[i] = BinaryPrimitives.ReadInt64LittleEndian(raw[(i * 8)..]);
                return result;
            }
            case Int32:
            {
                var result = new double[raw.Length / 4];
                for (var i = 0; i < result.Length; i++)
                    result[i] = BinaryPrimitives.ReadInt32LittleEndian(raw[(i * 4)..]);
                return result;
            }
            case Int16:
            {
                var result = new double[raw.Length / 2];
                for (var i = 0; i < result.Length; i++)
                    result[i] = BinaryPrimitives.ReadInt16LittleEndian(raw[(i * 2)..]);
                return result;
            }
            case Int8:
            {
                var result = new double[raw.Length];
                for (var i = 0; i < result.Length; i++) result[i] = (sbyte)raw[i];
                return result;
            }
            default:
                return null;
        }
    }

    private static double[] ReadPackedFloats(ReadOnlySpan<byte> bytes)
    {
        var result = new double[bytes.Length / 4];
        for (var i = 0; i < result.Length; i++)
            result[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes[(i * 4)..]);
        return result;
    }

    private static double[] ReadPackedDoubles(ReadOnlySpan<byte> bytes)
    {
        var result = new double[bytes.Length / 8];
        for (var i = 0; i < result.Length; i++)
            result[i] = BinaryPrimitives.ReadDoubleLittleEndian(bytes[(i * 8)..]);
        return result;
    }

    /// <summary>A forward-only reader over protobuf wire format.</summary>
    /// <remarks>
    /// Unknown fields are skipped by wire type rather than treated as an error. That is what the
    /// format is designed for, and it is why this keeps working against ONNX versions that add
    /// fields it has never heard of.
    /// </remarks>
    private ref struct Cursor(ReadOnlySpan<byte> data)
    {
        private readonly ReadOnlySpan<byte> _data = data;
        private int _position;

        public readonly bool AtEnd => _position >= _data.Length;

        public bool TryReadTag(out int field, out int wireType)
        {
            field = 0;
            wireType = 0;
            if (AtEnd) return false;

            var key = ReadVarint();
            field = (int)(key >> 3);
            wireType = (int)(key & 7);
            return true;
        }

        public ulong ReadVarint()
        {
            ulong value = 0;
            var shift = 0;

            while (_position < _data.Length)
            {
                var b = _data[_position++];
                value |= (ulong)(b & 0x7F) << shift;
                if ((b & 0x80) == 0) return value;

                shift += 7;
                if (shift > 63) throw new InvalidDataException("Malformed varint in ONNX file.");
            }

            throw new InvalidDataException("ONNX file ended inside a varint.");
        }

        public ReadOnlySpan<byte> ReadBytes()
        {
            var length = (int)ReadVarint();
            if (length < 0 || _position + length > _data.Length)
                throw new InvalidDataException("ONNX file declares a field longer than the file.");

            var slice = _data.Slice(_position, length);
            _position += length;
            return slice;
        }

        public string ReadString() => System.Text.Encoding.UTF8.GetString(ReadBytes());

        public void Skip(int wireType)
        {
            switch (wireType)
            {
                case Varint: ReadVarint(); break;
                case Fixed64: _position += 8; break;
                case LengthDelimited: ReadBytes(); break;
                case Fixed32: _position += 4; break;
                default: throw new InvalidDataException($"Unknown protobuf wire type {wireType}.");
            }
        }
    }
}
