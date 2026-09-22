using System.Buffers.Binary;
using System.Text;
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviHub.Io;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviHub.Tests;

/// <summary>
/// Tests for the safetensors reader and writer.
/// </summary>
/// <remarks>
/// The decode tests pin individual byte patterns against values worked out by hand from each
/// format's definition, rather than against what this implementation happens to produce. A
/// round-trip test alone would pass with a consistently wrong exponent bias.
/// </remarks>
public sealed class SafeTensorsTests : IDisposable
{
    private readonly string _directory = Path.Combine(
        Path.GetTempPath(), "hfnet-tests", Guid.NewGuid().ToString("N"));

    public SafeTensorsTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    private string Path_(string name) => System.IO.Path.Combine(_directory, name);

    [Fact]
    public void RoundTripsShapesAndValuesInF64()
    {
        var tensors = new Dictionary<string, NdArray>
        {
            ["weight"] = new NdArray([1.5, -2.25, 3.125, 4.0, 5.5, -6.75], 2, 3),
            ["bias"] = new NdArray([0.5, -0.5], 2),
        };

        var path = Path_("round-trip.safetensors");
        SafeTensors.Write(path, tensors, SafeTensorDType.F64);

        var read = SafeTensors.ReadAll(path);

        Assert.Equal(2, read.Count);
        Assert.Equal([2, 3], read["weight"].Shape.ToArray());
        Assert.Equal([2], read["bias"].Shape.ToArray());

        // F64 is exact, so this is equality rather than a tolerance.
        Assert.Equal(tensors["weight"].ToArray(), read["weight"].ToArray());
        Assert.Equal(tensors["bias"].ToArray(), read["bias"].ToArray());
    }

    [Fact]
    public void WritesTensorsInSortedOrderSoOutputIsDeterministic()
    {
        var first = Path_("a.safetensors");
        var second = Path_("b.safetensors");

        // Same tensors, opposite insertion order. The bytes must still match.
        SafeTensors.Write(first, new Dictionary<string, NdArray>
        {
            ["zeta"] = new NdArray([1.0], 1),
            ["alpha"] = new NdArray([2.0], 1),
        });

        SafeTensors.Write(second, new Dictionary<string, NdArray>
        {
            ["alpha"] = new NdArray([2.0], 1),
            ["zeta"] = new NdArray([1.0], 1),
        });

        Assert.Equal(File.ReadAllBytes(first), File.ReadAllBytes(second));
    }

    [Fact]
    public void ReadsOneTensorWithoutReadingTheRest()
    {
        var path = Path_("lazy.safetensors");
        SafeTensors.Write(path, new Dictionary<string, NdArray>
        {
            ["small"] = new NdArray([1.0, 2.0], 2),
            ["large"] = NdArray.Zeros(64, 64),
        });

        using var reader = SafeTensors.Open(path);

        Assert.Equal(2, reader.Tensors.Count);
        Assert.True(reader.Contains("small"));
        Assert.False(reader.Contains("absent"));
        Assert.Equal([1.0, 2.0], reader.Read("small").ToArray());
        Assert.Equal(4096, reader.Info("large")!.Value.ElementCount);
    }

    [Fact]
    public void PreservesMetadata()
    {
        var path = Path_("meta.safetensors");
        SafeTensors.Write(
            path,
            new Dictionary<string, NdArray> { ["w"] = new NdArray([1.0], 1) },
            SafeTensorDType.F32,
            new Dictionary<string, string> { ["format"] = "pt", ["author"] = "Gravicode Studios" });

        var metadata = SafeTensors.ReadMetadata(path);

        Assert.Equal("pt", metadata["format"]);
        Assert.Equal("Gravicode Studios", metadata["author"]);
    }

    [Fact]
    public void PerTensorDtypeKeepsIndexBuffersExact()
    {
        // 511 is representable in F32 and not in BF16, which has eight mantissa bits.
        var tensors = new Dictionary<string, NdArray>
        {
            ["weights"] = new NdArray([0.1, 0.2, 0.3], 3),
            ["position_ids"] = new NdArray([509, 510, 511], 3),
        };

        var path = Path_("mixed.safetensors");
        SafeTensors.Write(path, tensors,
            name => name == "position_ids" ? SafeTensorDType.F32 : SafeTensorDType.BF16);

        var read = SafeTensors.ReadAll(path);

        Assert.Equal([509.0, 510.0, 511.0], read["position_ids"].ToArray());
        Assert.All(read["weights"].ToArray().Zip(tensors["weights"].ToArray()),
            pair => Assert.True(Math.Abs(pair.First - pair.Second) < 0.01));
    }

    [Theory]
    // 1.0 is sign 0, exponent 127 (0x3F80 as the top 16 bits of a float).
    [InlineData(0x3F80, 1.0)]
    [InlineData(0xBF80, -1.0)]
    [InlineData(0x4000, 2.0)]
    [InlineData(0x0000, 0.0)]
    [InlineData(0x3F00, 0.5)]
    public void DecodesBFloat16AsTheTopHalfOfAFloat(int bits, double expected)
    {
        var bytes = new byte[2];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes, (ushort)bits);

        var decoded = new double[1];
        SafeTensors.DecodeBlock(bytes, decoded, SafeTensorDType.BF16);

        Assert.Equal(expected, decoded[0]);
    }

    [Theory]
    // float8_e4m3fn: value = 2^(exp-7) * (1 + mantissa/8) for a normal exponent.
    [InlineData(0b0_0111_000, 1.0)]      // exp 7 -> 2^0, mantissa 0
    [InlineData(0b0_1000_000, 2.0)]      // exp 8 -> 2^1
    [InlineData(0b1_0111_000, -1.0)]     // sign set
    [InlineData(0b0_0111_100, 1.5)]      // mantissa 4/8
    [InlineData(0b0_0000_000, 0.0)]
    public void DecodesFloat8E4M3(int raw, double expected)
    {
        var decoded = new double[1];
        SafeTensors.DecodeBlock([(byte)raw], decoded, SafeTensorDType.F8E4M3);

        Assert.Equal(expected, decoded[0]);
    }

    [Fact]
    public void Float8E4M3TreatsTheTopExponentAsFiniteExceptForNaN()
    {
        var decoded = new double[2];

        // 0b0_1111_110 is 448, the format's maximum - not an infinity.
        SafeTensors.DecodeBlock([0b0_1111_110], decoded.AsSpan(0, 1), SafeTensorDType.F8E4M3);
        Assert.Equal(448.0, decoded[0]);

        // Only the all-ones mantissa alongside it is NaN.
        SafeTensors.DecodeBlock([0b0_1111_111], decoded.AsSpan(1, 1), SafeTensorDType.F8E4M3);
        Assert.True(double.IsNaN(decoded[1]));
    }

    [Theory]
    // float8_e5m2 is IEEE-shaped: exponent bias 15, and the top exponent is infinity or NaN.
    [InlineData(0b0_01111_00, 1.0)]
    [InlineData(0b0_10000_00, 2.0)]
    [InlineData(0b1_01111_00, -1.0)]
    [InlineData(0b0_01111_01, 1.25)]
    public void DecodesFloat8E5M2(int raw, double expected)
    {
        var decoded = new double[1];
        SafeTensors.DecodeBlock([(byte)raw], decoded, SafeTensorDType.F8E5M2);

        Assert.Equal(expected, decoded[0]);
    }

    [Fact]
    public void Float8E5M2HasRealInfinities()
    {
        var decoded = new double[1];
        SafeTensors.DecodeBlock([0b0_11111_00], decoded, SafeTensorDType.F8E5M2);

        Assert.True(double.IsPositiveInfinity(decoded[0]));
    }

    [Fact]
    public void BFloat16WriteRoundsToNearestRatherThanTruncating()
    {
        // A value exactly between two bfloat16 neighbours must round to the even one, not down.
        // Truncation would bias an entire checkpoint towards zero.
        var tensors = new Dictionary<string, NdArray>
        {
            ["w"] = new NdArray([1.0 + Math.Pow(2, -9), 1.0 - Math.Pow(2, -9)], 2),
        };

        var path = Path_("rounding.safetensors");
        SafeTensors.Write(path, tensors, SafeTensorDType.BF16);
        var read = SafeTensors.ReadAll(path)["w"].ToArray();

        // Truncation would send both towards zero; rounding sends them in opposite directions.
        Assert.True(read[0] >= 1.0, $"expected >= 1.0, got {read[0]}");
        Assert.True(read[1] <= 1.0, $"expected <= 1.0, got {read[1]}");
    }

    [Fact]
    public void RejectsAFileWhoseHeaderLengthIsImplausible()
    {
        var path = Path_("corrupt.safetensors");

        var bytes = new byte[32];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, ulong.MaxValue);
        File.WriteAllBytes(path, bytes);

        var exception = Assert.Throws<InvalidDataException>(() => SafeTensors.Inspect(path));
        Assert.Contains("not plausible", exception.Message);
    }

    [Fact]
    public void RejectsATensorWhoseByteRangeDoesNotMatchItsShape()
    {
        // A header claiming a 2x3 F32 tensor needs 24 bytes; this one reserves 8.
        var header = """{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,8]}}""";
        var headerBytes = Encoding.UTF8.GetBytes(header);

        var file = new byte[8 + headerBytes.Length + 8];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)headerBytes.Length);
        headerBytes.CopyTo(file, 8);

        var path = Path_("mismatched.safetensors");
        File.WriteAllBytes(path, file);

        var exception = Assert.Throws<InvalidDataException>(() => SafeTensors.Inspect(path));
        Assert.Contains("24 bytes", exception.Message);
    }

    [Fact]
    public void ZeroRankTensorBecomesALengthOneVector()
    {
        var header = """{"scalar":{"dtype":"F64","shape":[],"data_offsets":[0,8]}}""";
        var headerBytes = Encoding.UTF8.GetBytes(header);

        var file = new byte[8 + headerBytes.Length + 8];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)headerBytes.Length);
        headerBytes.CopyTo(file, 8);
        BinaryPrimitives.WriteDoubleLittleEndian(file.AsSpan(8 + headerBytes.Length), 42.0);

        var path = Path_("scalar.safetensors");
        File.WriteAllBytes(path, file);

        using var reader = SafeTensors.Open(path);
        var tensor = reader.Read("scalar");

        Assert.Equal(1, tensor.Size);
        Assert.Equal(42.0, tensor.At(0));
    }

    [Fact]
    public void InspectOrdersTensorsByName()
    {
        var path = Path_("ordered.safetensors");
        SafeTensors.Write(path, new Dictionary<string, NdArray>
        {
            ["c"] = new NdArray([1.0], 1),
            ["a"] = new NdArray([1.0], 1),
            ["b"] = new NdArray([1.0], 1),
        });

        Assert.Equal(["a", "b", "c"], SafeTensors.Inspect(path).Select(t => t.Name));
    }

    [Fact]
    public void RefusesToWriteAnIntegerDtype()
    {
        var exception = Assert.Throws<NotSupportedException>(() => SafeTensors.Write(
            Path_("bad.safetensors"),
            new Dictionary<string, NdArray> { ["w"] = new NdArray([1.0], 1) },
            SafeTensorDType.I32));

        Assert.Contains("not supported", exception.Message);
    }
}
