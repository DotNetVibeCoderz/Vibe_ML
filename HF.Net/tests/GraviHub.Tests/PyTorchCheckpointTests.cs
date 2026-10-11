using Gravicode.HFNet.GraviHub.Io;
using Xunit;

namespace Gravicode.HFNet.GraviHub.Tests;

/// <summary>
/// Tests for the PyTorch checkpoint reader, in both of torch.save's layouts.
/// </summary>
/// <remarks>
/// <c>Fixtures/make_checkpoints.py</c> wrote the same state dict as <c>zip.bin</c> (torch 1.6+) and
/// <c>legacy.bin</c> (the bare-pickle layout <c>ProsusAI/finbert</c> still ships). The expected values
/// below are written out by hand from that script, not read back from either file, and every one is
/// exact in its dtype - so the assertions are equalities.
/// </remarks>
public sealed class PyTorchCheckpointTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    public static TheoryData<string> Layouts => new() { "legacy.bin", "zip.bin" };

    [Theory]
    [MemberData(nameof(Layouts))]
    public void ListsEveryTensorWithItsDTypeAndShape(string file)
    {
        var tensors = PyTorchCheckpoint.Inspect(Path.Combine(Fixtures, file)).ToDictionary(t => t.Name);

        Assert.Equal(
            ["encoder.bias", "encoder.weight", "encoder.weight_t", "half", "ids", "scalar", "tail"],
            tensors.Keys.Order(StringComparer.Ordinal));
        Assert.Equal([2, 3], tensors["encoder.weight"].Shape);
        Assert.Equal([3, 2], tensors["encoder.weight_t"].Shape);
        Assert.Equal(SafeTensorDType.F16, tensors["half"].DType);
        Assert.Equal(SafeTensorDType.I64, tensors["ids"].DType);
        Assert.Equal(SafeTensorDType.F64, tensors["scalar"].DType);
        Assert.Equal([1], tensors["scalar"].Shape); // rank 0 becomes length one
    }

    [Theory]
    [MemberData(nameof(Layouts))]
    public void ReadsValuesIncludingViewsOffsetsAndNarrowTypes(string file)
    {
        using var checkpoint = PyTorchCheckpoint.Open(Path.Combine(Fixtures, file));

        Assert.Equal([1.5, -2.25, 3.125, 4.0, 5.5, -6.75], checkpoint.Read("encoder.weight").ToArray());
        // The transpose shares the weight's storage with strides (1, 3).
        Assert.Equal([1.5, 4.0, -2.25, 5.5, 3.125, -6.75], checkpoint.Read("encoder.weight_t").ToArray());
        Assert.Equal([0.5, -0.5], checkpoint.Read("encoder.bias").ToArray());
        Assert.Equal([0.25, -1.0, 2048.0], checkpoint.Read("half").ToArray());
        Assert.Equal([3.0, -7.0, 1099511627776.0], checkpoint.Read("ids").ToArray());
        Assert.Equal([42.0], checkpoint.Read("scalar").ToArray());
        // arange(10)[6:]: starts six elements into its storage.
        Assert.Equal([6.0, 7.0, 8.0, 9.0], checkpoint.Read("tail").ToArray());
    }

    [Fact]
    public void RejectsAFileThatIsNeitherLayout()
    {
        var path = Path.Combine(Path.GetTempPath(), $"hfnet-not-a-checkpoint-{Guid.NewGuid():N}.bin");
        // A valid pickle (the integer 7) where the legacy layout expects its magic number, then 8.
        File.WriteAllBytes(path, [0x80, 0x02, 0x4B, 0x07, 0x2E, 0x80, 0x02, 0x4B, 0x08, 0x2E]);
        try
        {
            var error = Assert.Throws<InvalidDataException>(() => PyTorchCheckpoint.Open(path));
            Assert.Contains("1001", error.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
