using Gravicode.Science.GraviNum.Io;
using Xunit;

namespace GraviNum.Tests;

/// <summary>
/// Tests for the ONNX weight reader.
/// </summary>
/// <remarks>
/// The fixtures are built here by writing protobuf wire format directly. That is deliberate: the
/// reader was also checked against a file produced by the official Python <c>onnx</c> library, and
/// these encode the same layout so the suite keeps testing the real format without needing Python
/// or a checked-in binary.
/// </remarks>
public class OnnxReaderTests
{
    // ---------------------------------------------------------------- protobuf writing helpers

    private static void WriteVarint(List<byte> to, ulong value)
    {
        while (value >= 0x80)
        {
            to.Add((byte)(value | 0x80));
            value >>= 7;
        }
        to.Add((byte)value);
    }

    private static void WriteTag(List<byte> to, int field, int wireType)
        => WriteVarint(to, (ulong)((field << 3) | wireType));

    private static void WriteLengthDelimited(List<byte> to, int field, ReadOnlySpan<byte> payload)
    {
        WriteTag(to, field, 2);
        WriteVarint(to, (ulong)payload.Length);
        to.AddRange(payload.ToArray());
    }

    /// <summary>Builds a TensorProto: dims, data type, name and raw data.</summary>
    private static byte[] Tensor(string name, int[] dims, int dataType, byte[] raw)
    {
        var body = new List<byte>();

        foreach (var d in dims)
        {
            WriteTag(body, 1, 0);
            WriteVarint(body, (ulong)d);
        }

        WriteTag(body, 2, 0);
        WriteVarint(body, (ulong)dataType);

        WriteLengthDelimited(body, 8, System.Text.Encoding.UTF8.GetBytes(name));
        WriteLengthDelimited(body, 9, raw);

        return [.. body];
    }

    /// <summary>Wraps tensors in a GraphProto inside a ModelProto, as a real file does.</summary>
    private static byte[] Model(params byte[][] tensors)
    {
        var graph = new List<byte>();

        // An unknown field first, to prove skipping works — real files are full of them.
        WriteLengthDelimited(graph, 2, System.Text.Encoding.UTF8.GetBytes("graph-name"));
        foreach (var tensor in tensors) WriteLengthDelimited(graph, 5, tensor);

        var model = new List<byte>();
        WriteTag(model, 1, 0);              // ir_version, a varint field to skip
        WriteVarint(model, 8);
        WriteLengthDelimited(model, 7, graph.ToArray());
        return [.. model];
    }

    private static byte[] Float32Bytes(params float[] values)
    {
        var bytes = new byte[values.Length * 4];
        for (var i = 0; i < values.Length; i++)
            BitConverter.GetBytes(values[i]).CopyTo(bytes, i * 4);
        return bytes;
    }

    // ---------------------------------------------------------------- tests

    [Fact]
    public void ReadsAFloat32InitializerWithItsShape()
    {
        var content = Model(Tensor("encoder.weight", [2, 3], 1,
            Float32Bytes(1f, 2f, 3f, 4f, 5f, 6f)));

        var tensors = OnnxReader.ReadWeights(content);

        var tensor = Assert.Single(tensors);
        Assert.Equal("encoder.weight", tensor.Name);
        Assert.Equal([2, 3], tensor.Shape);
        Assert.Equal([1, 2, 3, 4, 5, 6], tensor.Values);

        var array = tensor.ToNdArray();
        Assert.Equal(6.0, array[1, 2], 10);
    }

    [Fact]
    public void ReadsFloat64AndInt64Initializers()
    {
        var doubles = new byte[16];
        BitConverter.GetBytes(1.5).CopyTo(doubles, 0);
        BitConverter.GetBytes(-2.25).CopyTo(doubles, 8);

        var longs = new byte[16];
        BitConverter.GetBytes(7L).CopyTo(longs, 0);
        BitConverter.GetBytes(-3L).CopyTo(longs, 8);

        var content = Model(
            Tensor("bias", [2], 11, doubles),
            Tensor("shape", [2], 7, longs));

        var byName = OnnxReader.ReadWeights(content).ToDictionary(t => t.Name);

        Assert.Equal([1.5, -2.25], byName["bias"].Values);
        Assert.Equal([7.0, -3.0], byName["shape"].Values);
    }

    [Fact]
    public void UnknownFieldsAreSkippedRatherThanFailing()
    {
        // Protobuf is designed so a reader can ignore fields it has never seen, and that is what
        // keeps this working against future ONNX versions. The fixture includes an unknown
        // varint field on the model and an unknown string field on the graph.
        var content = Model(Tensor("w", [1], 1, Float32Bytes(42f)));

        var tensor = Assert.Single(OnnxReader.ReadWeights(content));
        Assert.Equal(42.0, tensor.Values[0], 6);
    }

    [Fact]
    public void ATensorOfAnUndecodableTypeIsSkippedNotGuessed()
    {
        // Type 14 is COMPLEX64. Returning wrong numbers silently would be far worse than
        // returning fewer of them.
        var content = Model(
            Tensor("complex", [2], 14, new byte[16]),
            Tensor("real", [1], 1, Float32Bytes(1f)));

        var tensor = Assert.Single(OnnxReader.ReadWeights(content));
        Assert.Equal("real", tensor.Name);
    }

    [Fact]
    public void ATensorWhoseDataDoesNotMatchItsShapeIsSkipped()
    {
        // Claims six elements, supplies two. A truncated file should not produce a short tensor
        // that later reads as garbage.
        var content = Model(Tensor("truncated", [2, 3], 1, Float32Bytes(1f, 2f)));
        Assert.Empty(OnnxReader.ReadWeights(content));
    }

    [Fact]
    public void ADeclaredLengthPastTheEndOfTheFileIsRejected()
    {
        var content = new List<byte>();
        WriteTag(content, 7, 2);
        WriteVarint(content, 500);      // a graph far longer than what follows
        content.AddRange(new byte[4]);

        Assert.Throws<InvalidDataException>(() => OnnxReader.ReadWeights(content.ToArray()));
    }

    [Fact]
    public void WeightsCanBeLookedUpByName()
    {
        var path = Path.Combine(Path.GetTempPath(), $"gravicode-{Guid.NewGuid():N}.onnx");
        try
        {
            File.WriteAllBytes(path, Model(
                Tensor("a", [1], 1, Float32Bytes(1f)),
                Tensor("b", [1], 1, Float32Bytes(2f))));

            var byName = OnnxReader.ReadWeightsByName(path);
            Assert.Equal(2.0, byName["b"].Values[0], 6);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
