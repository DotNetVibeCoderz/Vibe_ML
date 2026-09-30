using System.Text.Json;
using Xunit;

namespace Gravicode.HFNet.GraviTransformers.Tests;

/// <summary>GPT-2 against transformers in float64, on the checkpoint <c>Fixtures/make_gpt2.py</c> writes.</summary>
public sealed class CausalLanguageModelTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    private static readonly JsonElement Reference =
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "gpt2-reference.json"))).RootElement;

    private static int[] Ints(string name) => [.. Reference.GetProperty(name).EnumerateArray().Select(v => v.GetInt32())];

    private static CausalLanguageModel Open() => CausalLanguageModel.Open(Path.Combine(Fixtures, "gpt2-tiny"));

    [Fact]
    public void Logits_at_every_position_match_transformers()
    {
        using var model = Open();
        var expected = Reference.GetProperty("logits").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var actual = model.Logits(Ints("ids"));

        var worst = expected.Zip(actual).Max(p => Math.Abs(p.First - p.Second));
        Assert.True(worst < 1e-11, $"largest logit difference {worst:E2}");
    }

    [Fact]
    public void Greedy_generation_is_transformers_generate_token_for_token()
    {
        using var model = Open();
        var greedy = Ints("greedy");

        var continuation = model.GenerateIds(greedy[..3], new GenerationSettings(MaxNewTokens: 12, StopAtEndToken: false)).ToArray();

        Assert.Equal(greedy[3..], continuation);
    }

    [Fact]
    public void Repetition_penalty_matches_transformers()
    {
        using var model = Open();
        var penalised = Ints("penalised");

        var continuation = model.GenerateIds(
            penalised[..3], new GenerationSettings(MaxNewTokens: 12, RepetitionPenalty: 1.3, StopAtEndToken: false)).ToArray();

        Assert.Equal(penalised[3..], continuation);
    }

    [Fact]
    public void The_cache_gives_what_recomputing_the_whole_text_gives()
    {
        // Each cached step must equal the last row of a full forward pass over the text so far.
        using var model = Open();
        var ids = new List<int> { 5, 17, 42 };

        foreach (var next in model.GenerateIds(ids.ToArray(), new GenerationSettings(MaxNewTokens: 8, StopAtEndToken: false)))
        {
            var logits = model.Logits(ids);
            var last = logits.AsSpan((ids.Count - 1) * model.Config.VocabularySize, model.Config.VocabularySize).ToArray();
            var best = Array.IndexOf(last, last.Max());

            Assert.Equal(best, next);
            ids.Add(next);
        }
    }

    [Fact]
    public void Attention_is_causal()
    {
        using var model = Open();
        var vocab = model.Config.VocabularySize;
        var first = model.Logits([5, 17, 42, 3]);
        var second = model.Logits([5, 17, 42, 60]);

        Assert.Equal(first.AsSpan(0, 3 * vocab).ToArray(), second.AsSpan(0, 3 * vocab).ToArray());
    }

    [Fact]
    public void Sampling_with_a_seed_repeats_and_top_k_of_one_is_greedy()
    {
        using var model = Open();
        int[] prompt = [5, 17, 42];

        var sampled = new GenerationSettings(MaxNewTokens: 10, Sample: true, Temperature: 0.8, TopK: 20, TopP: 0.9, Seed: 7, StopAtEndToken: false);
        Assert.Equal(model.GenerateIds(prompt, sampled).ToArray(), model.GenerateIds(prompt, sampled).ToArray());

        var topOne = sampled with { TopK = 1, TopP = 1.0 };
        Assert.Equal(Ints("greedy")[3..13], model.GenerateIds(prompt, topOne).ToArray());
    }

    [Fact]
    public void Generation_stops_when_the_context_is_full()
    {
        using var model = Open();
        var produced = model.GenerateIds([5, 17, 42], new GenerationSettings(MaxNewTokens: 100, StopAtEndToken: false)).Count();

        Assert.Equal(model.Config.MaxPositions - 3, produced);
    }

    [Fact]
    public void An_encoder_loader_points_a_decoder_at_this_class()
    {
        var error = Assert.Throws<NotSupportedException>(() => CausalLanguageModelConfig.Load(Path.Combine(Fixtures, "clip-tiny", "config.json")));
        Assert.Contains("GPT-2", error.Message);
    }
}
