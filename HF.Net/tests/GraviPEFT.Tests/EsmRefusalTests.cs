using Gravicode.HFNet.GraviPEFT;
using Gravicode.HFNet.GraviTransformers;
using Xunit;

namespace Gravicode.HFNet.GraviPEFT.Tests;

/// <summary>
/// LoRA and prefix tuning train through the BERT-family encoder's own forward and backward pass. ESM's
/// pre-norm, rotary blocks are a different model, so both refuse it rather than train the wrong function.
/// </summary>
public sealed class EsmRefusalTests
{
    private static readonly string Model = Path.Combine(AppContext.BaseDirectory, "Fixtures", "esm-rotary");

    [Fact]
    public void LoRA_refuses_an_ESM_model_and_names_the_alternative()
    {
        using var model = TransformerModel.Open(Model);

        var error = Assert.Throws<NotSupportedException>(() => PEFT.ApplyLoRA(model));
        Assert.Contains("LoRA", error.Message);
        Assert.Contains("Hidden", error.Message);
    }

    [Fact]
    public void Prefix_tuning_refuses_an_ESM_model()
    {
        using var model = TransformerModel.Open(Model);

        var error = Assert.Throws<NotSupportedException>(() => PEFT.ApplyPrefixTuning(model));
        Assert.Contains("Prefix tuning", error.Message);
    }
}
