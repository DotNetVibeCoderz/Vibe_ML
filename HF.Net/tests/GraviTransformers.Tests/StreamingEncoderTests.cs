using System.Text.Json;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>
/// The layer-at-a-time encoder against transformers in float64 and against the whole-model encoder,
/// on the checkpoints <c>Fixtures/make_bert.py</c> writes - one file and three shards.
/// </summary>
public sealed class StreamingEncoderTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    private static readonly JsonElement Reference =
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "bert-reference.json"))).RootElement;

    private static string Text => Reference.GetProperty("text").GetString()!;

    [Theory]
    [InlineData("bert-small")]
    [InlineData("bert-small-sharded")]
    public void Hidden_states_match_transformers(string folder)
    {
        using var encoder = StreamingEncoder.Open(Path.Combine(Fixtures, folder));
        var expected = Reference.GetProperty("hidden").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var actual = encoder.Hidden(Text).ToArray();

        Assert.Equal(expected.Length, actual.Length);
        var worst = expected.Zip(actual).Max(p => Math.Abs(p.First - p.Second));
        Assert.True(worst < 1e-12, $"largest difference {worst:E2}");
    }

    [Fact]
    public void Streaming_is_the_whole_model_encoder_to_the_bit()
    {
        // Same kernels, same float32 weights, same order of addition in the embeddings: the two must
        // not differ at all, which is what lets a caller switch between them freely.
        using var whole = TransformerModel.Open(Path.Combine(Fixtures, "bert-small"));
        using var streaming = StreamingEncoder.Open(Path.Combine(Fixtures, "bert-small-sharded"));

        Assert.Equal(whole.Hidden(Text).ToArray(), streaming.Hidden(Text).ToArray());
        Assert.Equal(whole.Embed(Text).ToArray(), streaming.Embed(Text).ToArray());
    }

    [Fact]
    public void A_directory_without_weights_is_refused_and_safetensors_are_mapped()
    {
        using (var store = WeightStore.Open(Path.Combine(Fixtures, "bert-small")))
        {
            Assert.True(store.IsMapped("embeddings.word_embeddings.weight"));
            Assert.Equal([25, 16], store.ShapeOf("embeddings.word_embeddings.weight"));
        }

        var directory = Path.Combine(Path.GetTempPath(), "hfnet-empty-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            File.Copy(Path.Combine(Fixtures, "bert-small", "config.json"), Path.Combine(directory, "config.json"));
            Assert.ThrowsAny<IOException>(() => StreamingEncoder.Open(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
