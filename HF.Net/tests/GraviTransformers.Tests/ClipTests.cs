using System.Text.Json;
using Gravicode.HFNet.GraviTransformers.Vision;
using Gravicode.Science.GraviNum;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>
/// CLIP against transformers in float64, and the image processors against the PIL ones - on the
/// fixtures <c>Fixtures/make_clip.py</c> writes.
/// </summary>
public sealed class ClipTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    private static readonly JsonElement Reference =
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "clip-reference.json"))).RootElement;

    private static double[] Values(string name) => [.. Reference.GetProperty(name).EnumerateArray().Select(v => v.GetDouble())];

    private static int[] Ids(string name) => [.. Reference.GetProperty(name).EnumerateArray().Select(v => v.GetInt32())];

    private static NdArray Pixels() => new(Values("pixels"), 3, 32, 32);

    private static void Close(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        var worst = expected.Zip(actual).Max(p => Math.Abs(p.First - p.Second));
        Assert.True(worst < tolerance, $"{what}: largest difference {worst:E2}");
    }

    [Theory]
    [InlineData("clip-tiny", "new")]
    [InlineData("clip-tiny-legacy", "legacy")]
    public void The_text_tower_matches_transformers(string folder, string rule)
    {
        using var model = ClipModel.Open(Path.Combine(Fixtures, folder));
        var ids = Ids($"{rule}_ids");

        Close(Values($"{rule}_hidden"), model.TextHidden(ids), 1e-12, "last_hidden_state");
        Close(Values($"{rule}_features"), model.EmbedTokens(ids).ToArray(), 1e-12, "text features");
    }

    [Fact]
    public void The_two_pooling_rules_read_different_positions()
    {
        // [96, 5, 98, 17, 98, 1]: the first end token is at 2. [0, 5, 97, 42, 2, 11]: the highest id
        // is at 2 too, but under the new rule there is no 98 at all, so it falls back to the last.
        using var current = ClipModel.Open(Path.Combine(Fixtures, "clip-tiny"));
        using var legacy = ClipModel.Open(Path.Combine(Fixtures, "clip-tiny-legacy"));

        Assert.Equal(2, current.EndPosition(Ids("new_ids")));
        Assert.Equal(2, legacy.EndPosition(Ids("legacy_ids")));
        Assert.Equal(2, legacy.EndPosition([0, 5, 97, 42, 2, 11]));
        Assert.Equal(5, current.EndPosition([0, 5, 97, 42, 2, 11]));
    }

    [Fact]
    public void Attention_in_the_text_tower_is_causal()
    {
        // Changing a later token must leave every earlier position's hidden state untouched.
        using var model = ClipModel.Open(Path.Combine(Fixtures, "clip-tiny"));
        var first = model.TextHidden([96, 5, 17, 23, 98]);
        var second = model.TextHidden([96, 5, 17, 60, 98]);

        Assert.Equal(first.AsSpan(0, 3 * 32).ToArray(), second.AsSpan(0, 3 * 32).ToArray());
        Assert.NotEqual(first.AsSpan(3 * 32, 32).ToArray(), second.AsSpan(3 * 32, 32).ToArray());
    }

    [Fact]
    public void The_vision_tower_matches_transformers()
    {
        using var model = ClipModel.Open(Path.Combine(Fixtures, "clip-tiny"));

        Close(Values("vision_hidden"), model.VisionHidden(Pixels()), 1e-12, "last_hidden_state");
        Close(Values("image_features"), model.EmbedImage(Pixels()).ToArray(), 1e-12, "image features");
    }

    [Fact]
    public void Image_text_logits_match_transformers()
    {
        using var model = ClipModel.Open(Path.Combine(Fixtures, "clip-tiny"));
        var image = model.EmbedImage(Pixels()).ToArray();
        var norm = Math.Sqrt(image.Sum(v => v * v));

        var texts = Reference.GetProperty("logit_texts").EnumerateArray()
            .Select(t => t.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();

        var logits = texts.Select(ids =>
        {
            var text = model.EmbedTokens(ids).ToArray();
            var textNorm = Math.Sqrt(text.Sum(v => v * v));
            return Math.Exp(model.LogitScale) * image.Zip(text).Sum(p => p.First * p.Second) / (norm * textNorm);
        }).ToArray();

        Close(Values("logits"), logits, 1e-10, "logits_per_image");
    }

    [Fact]
    public void A_directory_without_tokenizer_files_refuses_text_by_name()
    {
        using var model = ClipModel.Open(Path.Combine(Fixtures, "clip-tiny"));

        Assert.Null(model.Tokenizer);
        var error = Assert.Throws<InvalidOperationException>(() => model.EmbedText("a cat"));
        Assert.Contains("EmbedTokens", error.Message);
    }

    [Theory]
    [InlineData("clip")]
    [InlineData("vit")]
    public void The_image_processor_is_the_PIL_processor_to_the_last_bit(string name)
    {
        // transformers' slow processors: PIL's resize, an integer centre crop, float32 rescale and
        // normalise. Bicubic shortest-edge with a crop for CLIP, bilinear stretch for ViT.
        var processor = ImageProcessor.Load(Path.Combine(Fixtures, $"processor-{name}", "preprocessor_config.json"));
        using var image = Image.Load<Rgb24>(Path.Combine(Fixtures, "photo.png"));

        var bytes = File.ReadAllBytes(Path.Combine(Fixtures, $"processor-{name}.bin"));
        var expected = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, expected, 0, bytes.Length);

        var actual = processor.Convert(image).ToArray();

        Assert.Equal(expected.Length, actual.Length);
        var differing = expected.Zip(actual).Count(p => p.First != (float)p.Second);
        Assert.True(differing == 0, $"{differing} of {expected.Length} values differ; {processor}");
    }

    [Fact]
    public void The_CLIP_processor_config_is_read_as_shortest_edge_bicubic_with_a_crop()
    {
        var processor = ImageProcessor.Load(Path.Combine(Fixtures, "processor-clip", "preprocessor_config.json"));

        Assert.True(processor.ShortestEdge);
        Assert.Equal(48, processor.Size);
        Assert.Equal(40, processor.CropSize);
        Assert.Equal(Resample.Bicubic, processor.Resample);
        Assert.Equal(40, processor.OutputSize);
    }
}
