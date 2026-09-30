using Gravicode.HFNet.GraviOptimum;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviOptimum.Tests;

/// <summary>
/// Half-precision graphs, reading a model file's structure, and patching a node attribute - against
/// the tiny graphs <c>Fixtures/make_fixtures.py</c> writes.
/// </summary>
public sealed class OnnxFileTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    [Fact]
    public void A_float16_graph_runs_and_matches_onnx_runtime_in_Python()
    {
        // Expected values are the Python session's output for the same input, printed by the
        // fixture script: the sums are rounded to float16, which is why 1 * 0.5 + 0.1 is 0.60009765625.
        using var session = OnnxSession.Open(Path.Combine(Fixtures, "half.onnx"), ExecutionTarget.Cpu);
        var output = session.Run(new NdArray([1, 2, 3, -4, 0.5, 8], 2, 3));

        Assert.Equal([0.60009765625, -2.30078125, 9.296875, -1.900390625, -0.425048828125, 24.296875], output.ToArray());
        Assert.Equal("Float16", session.Inputs[0].ElementType);
    }

    [Fact]
    public void The_structure_and_float16_weights_are_read_from_the_file()
    {
        var model = OnnxModelFile.Read(Path.Combine(Fixtures, "half.onnx"));

        Assert.Equal(["/scale/Mul", "/shift/Add"], model.Nodes.Select(n => n.Name));
        Assert.Equal(["x", "w"], model.Nodes[0].Inputs);
        Assert.Equal([3L], model.Initializers["w"].Dimensions);
        Assert.Equal([0.5f, -1.25f, 3.0f], model.ReadFloats(model.Initializers["w"]));

        // 0.1 is not a float16 value; the stored one is the nearest, 0.0999755859375.
        Assert.Equal((float)(Half)0.1f, model.ReadFloats(model.Initializers["b"])[0]);
    }

    [Fact]
    public void Setting_a_sampler_s_scale_to_zero_makes_the_graph_deterministic()
    {
        var path = Path.Combine(Fixtures, "random.onnx");
        var input = new NdArray(Enumerable.Range(0, 64).Select(i => i / 8.0).ToArray(), 1, 64);

        using (var session = OnnxSession.Open(path, ExecutionTarget.Cpu))
        {
            // Unpatched, the node has no seed: two runs of the same session differ.
            Assert.NotEqual(session.Run(input).ToArray(), session.Run(input).ToArray());
        }

        // The node relies on the default scale of 1, so the attribute has to be added, not edited.
        var (bytes, changed) = OnnxModelFile.SetFloatAttribute(path, "RandomNormalLike", "scale", 0f);
        Assert.Equal(1, changed);

        var patched = Path.Combine(Path.GetTempPath(), $"hfnet-random-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(patched, bytes);
        try
        {
            using var session = OnnxSession.Open(patched, ExecutionTarget.Cpu);
            Assert.Equal(input.ToArray(), session.Run(input).ToArray());
            Assert.Contains("scale", OnnxModelFile.Read(patched).Nodes[0].Attributes);
        }
        finally
        {
            File.Delete(patched);
        }
    }
}
