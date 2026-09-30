using Gravicode.HFNet.GraviDiffusers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace Gravicode.HFNet.GraviDiffusers.Tests;

/// <summary>
/// The pipeline against diffusers' own ONNX pipelines, image for image.
/// </summary>
/// <remarks>
/// <para>
/// The models under <c>Fixtures/</c> are tiny random Stable Diffusion components exported from torch
/// by <c>Fixtures/make_fixtures.py</c>, which also wrote the <c>ref-*.png</c> images by running
/// <c>OnnxStableDiffusionPipeline</c>, <c>...Img2ImgPipeline</c> and <c>...InpaintPipeline</c> on them
/// with a NumPy seed. The VAE is factor 8, as Stable Diffusion's is, and the VAE encoder samples inside
/// the graph, as real exports do; the reference ran it with that sampling turned into the mean.
/// </para>
/// <para>
/// Random weights make the pictures noise, which is what makes the comparison strict: nothing about
/// a noise image looks right by accident, and every pixel has to agree.
/// </para>
/// </remarks>
public sealed class PipelineParityTests
{
    private const string Prompt = "a red lighthouse at dawn";

    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    private static readonly GenerationOptions TextToImage = new(Steps: 4, GuidanceScale: 7.5, Width: 64, Height: 64, Seed: 3);

    private static DiffusionPipeline Open(string name, Sampler sampler = Sampler.Ddim)
    {
        // The mean-patched encoder is written beside the original; a private copy keeps parallel
        // test runs from racing on that file.
        var copy = Path.Combine(Path.GetTempPath(), "hfnet-sd8-" + Guid.NewGuid().ToString("N"), name);
        foreach (var file in Directory.GetFiles(Path.Combine(Fixtures, name), "*", SearchOption.AllDirectories))
        {
            var target = Path.Combine(copy, Path.GetRelativePath(Path.Combine(Fixtures, name), file));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(file, target);
        }

        return DiffusionPipeline.FromDirectory(copy, sampler: sampler);
    }

    private static Image<Rgb24> Fixture(string file) => Image.Load<Rgb24>(Path.Combine(Fixtures, file));

    /// <summary>The largest per-channel difference and how many channel values differ at all.</summary>
    private static (int Max, int Differing) Compare(Image<Rgb24> actual, Image<Rgb24> expected)
    {
        Assert.Equal(expected.Width, actual.Width);
        Assert.Equal(expected.Height, actual.Height);

        int max = 0, differing = 0;
        for (var y = 0; y < actual.Height; y++)
        {
            for (var x = 0; x < actual.Width; x++)
            {
                var a = actual[x, y];
                var e = expected[x, y];
                foreach (var d in new[] { Math.Abs(a.R - e.R), Math.Abs(a.G - e.G), Math.Abs(a.B - e.B) })
                {
                    max = Math.Max(max, d);
                    if (d > 0) differing++;
                }
            }
        }

        return (max, differing);
    }

    private static void AssertIdentical(Image<Rgb24> actual, string reference)
    {
        using var expected = Fixture(reference);
        var (max, differing) = Compare(actual, expected);
        Assert.True(differing == 0, $"{differing} channel values differ from {reference}, by up to {max}.");
    }

    /// <summary>
    /// Identical but for pixels on a rounding boundary. Paths through the VAE encoder leave a couple
    /// of dozen of the 12,288 values within 1e-3 of a half level, and float32 convolution noise
    /// between two ONNX Runtime sessions can tip one of them.
    /// </summary>
    private static void AssertIdenticalButRounding(Image<Rgb24> actual, string reference)
    {
        using var expected = Fixture(reference);
        var (max, differing) = Compare(actual, expected);
        Assert.True(max <= 1 && differing <= 4, $"{differing} channel values differ from {reference}, by up to {max}.");
    }

    [Fact]
    public void Geometry_comes_from_the_configs()
    {
        using var pipeline = Open("sd8");
        using var inpainting = Open("sd8-inpaint");

        Assert.Equal(8, pipeline.VaeScaleFactor);
        Assert.Equal(4, pipeline.LatentChannels);
        Assert.Equal(0.18215, pipeline.LatentScaling);
        Assert.Equal(4, pipeline.UnetInputChannels);
        Assert.Equal(9, inpainting.UnetInputChannels);
        Assert.Equal(1, pipeline.SchedulerConfig.StepsOffset);
    }

    [Fact]
    public void Text_to_image_with_DDIM_is_diffusers_image()
    {
        using var pipeline = Open("sd8");
        using var image = pipeline.Generate(Prompt, TextToImage);

        AssertIdentical(image, "ref-t2i-ddim.png");
    }

    [Fact]
    public void Text_to_image_with_Euler_is_diffusers_image()
    {
        using var pipeline = Open("sd8", Sampler.Euler);
        using var image = pipeline.Generate(Prompt, TextToImage);

        AssertIdentical(image, "ref-t2i-euler.png");
    }

    [Fact]
    public void Image_to_image_is_diffusers_image()
    {
        using var pipeline = Open("sd8");
        using var init = Fixture("init.png");
        using var image = pipeline.ImageToImage(Prompt, init, strength: 0.6, TextToImage with { Steps = 5, Seed = 11 });

        AssertIdenticalButRounding(image, "ref-img2img.png");
    }

    [Fact]
    public void The_VAE_encoder_is_deterministic()
    {
        // The export samples with an unseeded RandomNormalLike; unpatched, two runs differ.
        using var pipeline = Open("sd8");
        using var init = Fixture("init.png");
        using var first = pipeline.ImageToImage(Prompt, init, 0.6, TextToImage with { Steps = 5, Seed = 11 });
        using var second = pipeline.ImageToImage(Prompt, init, 0.6, TextToImage with { Steps = 5, Seed = 11 });

        Assert.Equal((0, 0), Compare(first, second));
    }

    [Fact]
    public void Inpainting_with_a_nine_channel_UNet_is_diffusers_image()
    {
        using var pipeline = Open("sd8-inpaint");
        using var init = Fixture("init.png");
        using var mask = Fixture("mask.png");
        using var image = pipeline.Inpaint(Prompt, init, mask, TextToImage with { Seed = 9 });

        AssertIdenticalButRounding(image, "ref-inpaint.png");
    }

    [Fact]
    public void An_inpainting_UNet_refuses_text_to_image()
    {
        using var pipeline = Open("sd8-inpaint");

        var error = Assert.Throws<InvalidOperationException>(() => pipeline.Generate(Prompt, TextToImage));
        Assert.Contains("Inpaint", error.Message);
    }

    [Fact]
    public void Blended_inpainting_runs_on_an_ordinary_UNet()
    {
        using var pipeline = Open("sd8");
        using var init = Fixture("init.png");
        using var mask = Fixture("mask.png");
        using var image = pipeline.Inpaint(Prompt, init, mask, TextToImage with { Seed = 9 });

        Assert.Equal(64, image.Width);
    }

    [Fact]
    public void A_size_off_the_latent_grid_is_refused()
    {
        using var pipeline = Open("sd8");

        var error = Assert.Throws<ArgumentException>(() => pipeline.Generate(Prompt, TextToImage with { Width = 60 }));
        Assert.Contains("multiples of 8", error.Message);
    }

    [Theory]
    [InlineData("lora-peft.safetensors")]
    [InlineData("lora-kohya.safetensors")]
    public void A_LoRA_gives_the_image_of_the_model_it_was_fused_into(string file)
    {
        // The reference ran a UNet and text encoder with the adapter fused by PEFT in float32 and
        // exported. HF.Net adds the same product to the ONNX weights; the two roundings of
        // W + scale * B A differ in the last bit here and there, which moves a handful of pixels one level.
        using var pipeline = Open("sd8");
        var report = pipeline.LoadLora(Path.Combine(Fixtures, file));

        Assert.True(report.UnetLayers > 0 && report.TextEncoderLayers > 0, report.ToString());
        Assert.Empty(report.Unmatched);
        Assert.Equal(Classified(file), report.UnetLayers + report.TextEncoderLayers);

        using var image = pipeline.Generate(Prompt, TextToImage);
        using var expected = Fixture("ref-lora.png");
        var (max, differing) = Compare(image, expected);

        Assert.True(max <= 1 && differing < 40, $"{differing} channel values differ from the fused model, by up to {max}.");
    }

    /// <summary>How many adapter pairs a file holds: one down matrix each, in any of the layouts.</summary>
    private static int Classified(string file)
    {
        using var reader = Gravicode.HFNet.GraviHub.Io.SafeTensors.Open(Path.Combine(Fixtures, file));
        return reader.Tensors.Count(t => t.Name.EndsWith(".lora_A.weight", StringComparison.Ordinal)
            || t.Name.EndsWith(".lora_down.weight", StringComparison.Ordinal));
    }

    [Fact]
    public void Unloading_a_LoRA_restores_the_base_model_exactly()
    {
        using var pipeline = Open("sd8");
        pipeline.LoadLora(Path.Combine(Fixtures, "lora-peft.safetensors"));
        pipeline.UnloadLoras();

        using var image = pipeline.Generate(Prompt, TextToImage);
        AssertIdentical(image, "ref-t2i-ddim.png");
    }

    [Fact]
    public void Weighting_at_unit_weight_is_the_plain_prompt()
    {
        // The weighted path tokenizes the pieces itself and assembles start, end and padding; with
        // every weight 1 that must be the plain path's sequence exactly.
        using var pipeline = Open("sd8");
        pipeline.PromptWeighting = true;

        using var image = pipeline.Generate("(a red lighthouse at dawn:1.0)", TextToImage);
        AssertIdentical(image, "ref-t2i-ddim.png");
    }

    [Fact]
    public void Emphasis_changes_the_image_only_when_weighting_is_on()
    {
        using var pipeline = Open("sd8");
        using var plain = pipeline.Generate("a (red:1.8) lighthouse at dawn", TextToImage);

        pipeline.PromptWeighting = true;
        using var weighted = pipeline.Generate("a (red:1.8) lighthouse at dawn", TextToImage);

        Assert.NotEqual(0, Compare(plain, weighted).Differing);
    }
}
