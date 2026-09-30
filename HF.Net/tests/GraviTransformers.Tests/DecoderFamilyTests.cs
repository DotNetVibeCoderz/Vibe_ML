using System.Text.Json;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>
/// Llama, Mistral, Qwen2, Qwen3 and GPT-NeoX against transformers in float64, on the checkpoints
/// <c>Fixtures/make_decoders.py</c> writes - one per feature.
/// </summary>
public sealed class DecoderFamilyTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    private static readonly JsonElement Reference =
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "decoders-reference.json"))).RootElement;

    private static int[] Ids => [.. Reference.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32())];

    private static CausalLanguageModel Open(string name) => CausalLanguageModel.Open(Path.Combine(Fixtures, "decoders", name));

    public static TheoryData<string> Cases =>
    [
        "llama",            // grouped-query attention: 4 query heads, 2 key/value heads
        "llama-linear",     // linear rotary scaling
        "llama3",           // Llama 3.1's rotary scaling, tied embeddings
        "mistral",          // a sliding window of 4, shorter than the 12-token input
        "qwen2",            // query/key/value biases
        "qwen3",            // per-head query/key RMSNorm, head_dim wider than hidden / heads
        "neox",             // a quarter of each head rotated, parallel residual
        "neox-sequential",  // half of each head rotated, sequential residual
    ];

    [Theory]
    [MemberData(nameof(Cases))]
    public void Logits_at_every_position_match_transformers(string name)
    {
        using var model = Open(name);
        var expected = Reference.GetProperty(name).GetProperty("logits").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var actual = model.Logits(Ids);

        Assert.Equal(expected.Length, actual.Length);
        var worst = expected.Zip(actual).Max(p => Math.Abs(p.First - p.Second));
        Assert.True(worst < 1e-10, $"{name}: largest logit difference {worst:E2}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Greedy_generation_through_the_cache_is_transformers_generate(string name)
    {
        using var model = Open(name);
        var greedy = Reference.GetProperty(name).GetProperty("greedy").EnumerateArray().Select(v => v.GetInt32()).ToArray();

        var continuation = model.GenerateIds(greedy[..4], new GenerationSettings(MaxNewTokens: 10, StopAtEndToken: false)).ToArray();

        Assert.Equal(greedy[4..], continuation);
    }

    [Fact]
    public void The_config_is_read_as_each_family_declares_it()
    {
        using var llama = Open("llama");
        using var qwen3 = Open("qwen3");
        using var mistral = Open("mistral");
        using var neox = Open("neox");

        Assert.Equal((DecoderFamily.Llama, 4, 2, 8), (llama.Config.Family, llama.Config.Heads, llama.Config.KeyValueHeads, llama.Config.HeadSize));
        Assert.Equal((16, true), (qwen3.Config.HeadSize, qwen3.Config.QueryKeyNorm));
        Assert.Equal(4, mistral.Config.SlidingWindow);
        Assert.Equal((DecoderFamily.GptNeoX, 2, true), (neox.Config.Family, neox.Config.RotaryDimensions, neox.Config.ParallelResidual));
        Assert.Equal([98], llama.Config.EndTokenIds);
    }

    [Fact]
    public void The_sliding_window_really_limits_what_a_token_sees()
    {
        // A window of 4 lets each layer look 3 positions back, so two layers reach 6: the logits at
        // position 11 depend on positions 5-11. Changing position 2 must leave them exactly alone;
        // changing position 8 must not.
        using var model = Open("mistral");
        var vocab = model.Config.VocabularySize;
        int[] ids = [1, 5, 17, 42, 3, 77, 60, 11, 23, 90, 8, 31];

        double[] At11(int[] tokens) => model.Logits(tokens).AsSpan(11 * vocab, vocab).ToArray();
        var original = At11(ids);

        var farChange = (int[])ids.Clone();
        farChange[2] = 50;
        var nearChange = (int[])ids.Clone();
        nearChange[8] = 50;

        Assert.Equal(original, At11(farChange));
        Assert.NotEqual(original, At11(nearChange));
    }

    [Fact]
    public void A_rotary_scaling_this_does_not_implement_is_refused_by_name()
    {
        var directory = Path.Combine(Path.GetTempPath(), "hfnet-yarn-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var config = File.ReadAllText(Path.Combine(Fixtures, "decoders", "llama", "config.json"))
                .Replace("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"", StringComparison.Ordinal);
            File.WriteAllText(Path.Combine(directory, "config.json"), config);

            var error = Assert.Throws<NotSupportedException>(() => CausalLanguageModelConfig.Load(Path.Combine(directory, "config.json")));
            Assert.Contains("yarn", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
