using Gravicode.HFNet.GraviHub.Io;
using Gravicode.HFNet.GraviOptimum;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviOptimum.Tests;

/// <summary>Tests for parsing an execution target name.</summary>
public sealed class ExecutionTargetTests
{
    [Theory]
    [InlineData("cpu", ExecutionTarget.Cpu)]
    [InlineData("CPU", ExecutionTarget.Cpu)]
    [InlineData("cuda", ExecutionTarget.Cuda)]
    [InlineData("CUDA", ExecutionTarget.Cuda)]
    [InlineData("gpu", ExecutionTarget.Cuda)]
    [InlineData("nvidia", ExecutionTarget.Cuda)]
    [InlineData("directml", ExecutionTarget.DirectML)]
    [InlineData("dml", ExecutionTarget.DirectML)]
    [InlineData("oneDNN", ExecutionTarget.OneDnn)]
    [InlineData("mkl", ExecutionTarget.OneDnn)]
    [InlineData("auto", ExecutionTarget.Auto)]
    [InlineData("", ExecutionTarget.Auto)]
    [InlineData("  ", ExecutionTarget.Auto)]
    public void AcceptsTheSpellingsPeopleActuallyUse(string name, ExecutionTarget expected)
        => Assert.Equal(expected, Optimum.ParseTarget(name));

    [Fact]
    public void RejectsAnUnknownNameRatherThanFallingBackSilently()
    {
        // Falling back to the CPU here is how a "TPU" deployment runs for months without anyone
        // noticing the flag was never honoured.
        var exception = Assert.Throws<ArgumentException>(() => Optimum.ParseTarget("tpu"));

        Assert.Contains("CUDA", exception.Message);
    }
}

/// <summary>Tests for weight quantisation.</summary>
public sealed class QuantizationTests : IDisposable
{
    private readonly string _directory = Path.Combine(
        Path.GetTempPath(), "hfnet-quant", Guid.NewGuid().ToString("N"));

    public QuantizationTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    private string WriteSource(SafeTensorDType dtype = SafeTensorDType.F32)
    {
        var random = new GraviRandom(5);
        var weights = NdArray.Zeros(64, 64);
        for (var i = 0; i < weights.Size; i++) weights.SetAt(i, random.Normal(0, 0.02));

        var path = Path.Combine(_directory, $"source-{dtype}.safetensors");
        SafeTensors.Write(path, new Dictionary<string, NdArray>
        {
            ["encoder.weight"] = weights,
            ["encoder.bias"] = NdArray.Zeros(64),
        }, dtype);

        return path;
    }

    [Fact]
    public void HalvesTheFileGoingFromF32ToBFloat16()
    {
        var source = WriteSource();
        var destination = Path.Combine(_directory, "bf16.safetensors");

        var report = Optimum.Quantize(source, destination, QuantizationLevel.BFloat16);

        Assert.True(report.SizeRatio is > 0.45 and < 0.55, $"ratio was {report.SizeRatio}");
    }

    [Fact]
    public void Float16IsMorePreciseThanBFloat16OnSmallWeights()
    {
        // Both are sixteen bits. bfloat16 spends more of them on the exponent, so within the range
        // where float16 does not underflow it has the smaller rounding error.
        var source = WriteSource();

        var bf16 = Optimum.Quantize(source, Path.Combine(_directory, "a.safetensors"), QuantizationLevel.BFloat16);
        var f16 = Optimum.Quantize(source, Path.Combine(_directory, "b.safetensors"), QuantizationLevel.Float16);

        Assert.True(f16.MaxAbsoluteError < bf16.MaxAbsoluteError,
            $"f16 {f16.MaxAbsoluteError} should beat bf16 {bf16.MaxAbsoluteError}");
    }

    [Fact]
    public void NoneCopiesWithoutError()
    {
        var source = WriteSource();
        var destination = Path.Combine(_directory, "same.safetensors");

        var report = Optimum.Quantize(source, destination, QuantizationLevel.None);

        Assert.Equal(0, report.MaxAbsoluteError);
        Assert.Equal(1.0, report.SizeRatio);
        Assert.Equal(File.ReadAllBytes(source), File.ReadAllBytes(destination));
    }

    [Fact]
    public void IntegerTensorsInTheSourceKeepFullPrecision()
    {
        // 511 is not representable in bfloat16's eight mantissa bits. An index buffer quantised
        // along with the weights reports an error of 1.0 that has nothing to do with the model.
        // Written by hand: the writer only emits float types, so an integer-typed source can
        // only come from a real checkpoint or from bytes laid out here.
        var path = Path.Combine(_directory, "with-indices.safetensors");
        WriteWithInt32Indices(path);

        var destination = Path.Combine(_directory, "quantised.safetensors");
        var report = Optimum.Quantize(path, destination, QuantizationLevel.BFloat16);

        Assert.True(report.MaxAbsoluteError < 0.01, $"error was {report.MaxAbsoluteError}");
        Assert.Equal([510.0, 511.0], SafeTensors.ReadAll(destination)["position_ids"].ToArray());
    }

    /// <summary>Emits a two-tensor file where position_ids is stored as I32.</summary>
    private static void WriteWithInt32Indices(string path)
    {
        const string Header =
            """{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"position_ids":{"dtype":"I32","shape":[2],"data_offsets":[8,16]}}""";

        var headerBytes = System.Text.Encoding.UTF8.GetBytes(Header);
        var file = new byte[8 + headerBytes.Length + 16];

        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)headerBytes.Length);
        headerBytes.CopyTo(file, 8);

        var data = file.AsSpan(8 + headerBytes.Length);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(data[..4], 0.1f);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(data[4..8], 0.2f);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(data[8..12], 510);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(data[12..16], 511);

        File.WriteAllBytes(path, file);
    }

    [Fact]
    public void TheCallerCanNameExtraTensorsToKeepExact()
    {
        var path = Path.Combine(_directory, "all-float.safetensors");
        SafeTensors.Write(path, new Dictionary<string, NdArray>
        {
            ["weight"] = new NdArray([0.1, 0.2], 2),
            ["position_ids"] = new NdArray([510, 511], 2),
        }, SafeTensorDType.F32);

        var destination = Path.Combine(_directory, "kept.safetensors");
        var report = Optimum.Quantize(
            path, destination, QuantizationLevel.BFloat16,
            keepFullPrecision: new HashSet<string> { "position_ids" });

        Assert.Equal([510.0, 511.0], SafeTensors.ReadAll(destination)["position_ids"].ToArray());
        Assert.True(report.MaxAbsoluteError < 0.01);
    }

    [Fact]
    public void WithoutThatHintBFloat16ManglesAnIndexBuffer()
    {
        // The counterpart of the test above: this is the failure the parameter exists to prevent.
        var path = Path.Combine(_directory, "unhinted.safetensors");
        SafeTensors.Write(path, new Dictionary<string, NdArray>
        {
            ["position_ids"] = new NdArray([511], 1),
        }, SafeTensorDType.F32);

        var report = Optimum.Quantize(
            path, Path.Combine(_directory, "mangled.safetensors"), QuantizationLevel.BFloat16);

        Assert.True(report.MaxAbsoluteError >= 1.0, $"expected a whole-number error, got {report.MaxAbsoluteError}");
    }

    [Fact]
    public void Int8IsRefusedWithAReasonRatherThanApproximated()
    {
        var exception = Assert.Throws<NotSupportedException>(() => Optimum.Quantize(
            WriteSource(), Path.Combine(_directory, "int8.safetensors"), QuantizationLevel.Int8));

        Assert.Contains("scales", exception.Message);
    }

    [Fact]
    public void ReportFormatsBothSizesAndTheError()
    {
        var report = new QuantizationReport(2_000_000, 1_000_000, 1e-3, 1e-5);

        Assert.Equal(0.5, report.SizeRatio);
        Assert.Contains("50", report.ToString());
    }
}
