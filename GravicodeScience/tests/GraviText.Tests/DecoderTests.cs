using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Generation;
using Gravicode.Science.GraviText.Tokenization;
using Gravicode.Science.GraviText.Transformers;
using Xunit;

namespace Gravicode.Science.Tests.GraviText;

/// <summary>
/// Tests for the decoder stack.
/// </summary>
/// <remarks>
/// The property that matters most is causality, and it is the one that cannot be seen by looking at
/// the output: a decoder that peeks at the future generates plausible text and is useless. So the
/// central test changes a later token and asserts that no earlier hidden state moved — which fails
/// immediately if the mask is missing, and cannot be faked.
/// </remarks>
public class DecoderTests
{
    private static Vocabulary BuildVocabulary(int extra = 20)
    {
        var vocabulary = new Vocabulary();
        for (var i = 0; i < extra; i++) vocabulary.Add($"tok{i}");
        return vocabulary;
    }

    private static TransformerDecoder Decoder(Vocabulary? vocabulary = null, int seed = 5)
    {
        vocabulary ??= BuildVocabulary();
        var config = new TransformerConfig(
            VocabularySize: vocabulary.Count, HiddenSize: 16, Layers: 2, Heads: 4,
            IntermediateSize: 32, MaxPositions: 24);

        return new TransformerDecoder(config, vocabulary, new GraviRandom(seed));
    }

    [Fact]
    public void ChangingALaterTokenLeavesEveryEarlierStateUntouched()
    {
        // The definition of causality, and the one thing no amount of inspecting the output would
        // reveal. Without the mask every position shifts and the model is unusable for generation.
        var decoder = Decoder();

        int[] first = [5, 6, 7, 8, 9];
        int[] second = [5, 6, 7, 18, 19];

        var a = decoder.Forward(first);
        var b = decoder.Forward(second);

        for (var t = 0; t < 3; t++)
            for (var d = 0; d < a.Shape[1]; d++)
                Assert.True(Math.Abs(a[t, d] - b[t, d]) < 1e-12,
                    $"position {t} moved when a later token changed: {a[t, d]:R} against {b[t, d]:R}");

        // And the positions at and after the change must differ, or the test proves nothing.
        var moved = false;
        for (var d = 0; d < a.Shape[1] && !moved; d++) moved = Math.Abs(a[3, d] - b[3, d]) > 1e-9;
        Assert.True(moved, "the changed position did not move, so this test would pass vacuously");
    }

    [Fact]
    public void AttentionIsStrictlyLowerTriangular()
    {
        var decoder = Decoder();
        decoder.Forward([5, 6, 7, 8]);

        var attention = new CausalSelfAttention(
            new TransformerConfig(20, 16, 2, 4, 32, 24), new GraviRandom(3));

        attention.Forward(NdArray.Zeros(5, 16));

        foreach (var head in attention.LastAttention)
            for (var i = 0; i < 5; i++)
                for (var j = i + 1; j < 5; j++)
                    Assert.True(head[i, j] < 1e-9,
                        $"position {i} attended {head[i, j]:E3} to the future position {j}");
    }

    [Fact]
    public void EachRowOfAttentionSumsToOne()
    {
        var attention = new CausalSelfAttention(
            new TransformerConfig(20, 16, 2, 4, 32, 24), new GraviRandom(7));

        attention.Forward(new GraviRandom(9).StandardNormal(6, 16));

        foreach (var head in attention.LastAttention)
            for (var i = 0; i < 6; i++)
            {
                var total = 0.0;
                for (var j = 0; j < 6; j++) total += head[i, j];
                Assert.Equal(1.0, total, 9);
            }
    }

    [Fact]
    public void TheFirstPositionAttendsOnlyToItself()
    {
        // It has no past, so the causal mask leaves exactly one reachable position.
        var attention = new CausalSelfAttention(
            new TransformerConfig(20, 16, 2, 4, 32, 24), new GraviRandom(11));

        attention.Forward(new GraviRandom(13).StandardNormal(5, 16));

        foreach (var head in attention.LastAttention)
            Assert.Equal(1.0, head[0, 0], 9);
    }

    [Fact]
    public void LogitsAreProducedForEveryPosition()
    {
        // One prediction per position from a single pass, which is what the causal mask makes safe
        // and is the whole reason language-model training is affordable.
        var vocabulary = BuildVocabulary();
        var decoder = Decoder(vocabulary);

        var logits = decoder.Logits([5, 6, 7, 8]);

        Assert.Equal([4, vocabulary.Count], logits.Shape.ToArray());
    }

    [Fact]
    public void GreedyGenerationIsDeterministic()
    {
        var decoder = Decoder();

        var first = decoder.Generate([5, 6], maxNewTokens: 8, options: SamplingOptions.Greedy);
        var second = decoder.Generate([5, 6], maxNewTokens: 8, options: SamplingOptions.Greedy);

        Assert.Equal(first, second);
    }

    [Fact]
    public void GenerationKeepsThePromptAndAddsTheRequestedNumberOfTokens()
    {
        var decoder = Decoder();
        var generated = decoder.Generate([5, 6, 7], maxNewTokens: 10, options: SamplingOptions.Greedy);

        Assert.Equal(13, generated.Length);
        Assert.Equal([5, 6, 7], generated.Take(3));
    }

    [Fact]
    public void SamplingProducesDifferentContinuationsFromTheSamePrompt()
    {
        var decoder = Decoder();
        var rng = new GraviRandom(17);

        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 0; i < 20; i++)
            seen.Add(string.Join(',', decoder.Generate([5, 6], maxNewTokens: 6,
                options: new SamplingOptions(Temperature: 1.5), rng: rng)));

        Assert.True(seen.Count > 1, "sampling produced only one continuation");
    }

    [Fact]
    public void TopKRestrictsGenerationToTheAllowedTokens()
    {
        // With k of 1 the choice is forced, so sampling must agree with greedy decoding exactly.
        var decoder = Decoder();
        var rng = new GraviRandom(19);

        var greedy = decoder.Generate([5, 6], maxNewTokens: 6, options: SamplingOptions.Greedy);
        var topOne = decoder.Generate([5, 6], maxNewTokens: 6,
            options: new SamplingOptions(Temperature: 1.0, TopK: 1), rng: rng);

        Assert.Equal(greedy, topOne);
    }

    [Fact]
    public void ATinyTopPStillChoosesSomething()
    {
        // The token that crosses the threshold is kept, so the nucleus is never empty — which it
        // would be for any p below the top token's probability if the check came first.
        var decoder = Decoder();
        var generated = decoder.Generate([5, 6], maxNewTokens: 4,
            options: new SamplingOptions(Temperature: 1.0, TopP: 1e-6), rng: new GraviRandom(23));

        Assert.Equal(6, generated.Length);
    }

    [Fact]
    public void ARepetitionPenaltyDiscouragesRepeatingTokens()
    {
        // The sign matters: dividing a negative logit by the penalty raises it, which is the
        // opposite of a penalty. A model that repeats more with the penalty on has that bug.
        var decoder = Decoder();

        int Repeats(double penalty)
        {
            var generated = decoder.Generate([5, 5, 5], maxNewTokens: 12,
                options: new SamplingOptions(Temperature: 0.5, RepetitionPenalty: penalty),
                rng: new GraviRandom(29));

            return generated.Skip(3).Count(t => t == generated[3]);
        }

        Assert.True(Repeats(2.0) <= Repeats(1.0),
            "the penalty did not reduce repetition");
    }

    [Fact]
    public void AStopTokenEndsGenerationEarly()
    {
        var decoder = Decoder();

        // Take one greedy step to learn what the model would produce, then declare it a stop token.
        var unbounded = decoder.Generate([5, 6], maxNewTokens: 10, options: SamplingOptions.Greedy);
        var stopped = decoder.Generate([5, 6], maxNewTokens: 10, options: SamplingOptions.Greedy,
            stopTokens: new HashSet<int> { unbounded[2] });

        Assert.Equal(3, stopped.Length);
        Assert.Equal(unbounded[2], stopped[^1]);
    }

    [Fact]
    public void GenerationBeyondTheContextWindowKeepsTheRecentTokens()
    {
        // The window is finite, and dropping the oldest tokens is better than failing — the next
        // token depends on the recent context, not the start of the sequence.
        var decoder = Decoder();
        var prompt = Enumerable.Range(5, 20).ToArray();

        var generated = decoder.Generate(prompt, maxNewTokens: 10, options: SamplingOptions.Greedy);
        Assert.Equal(30, generated.Length);
    }

    [Fact]
    public void CrossEntropyIsPositiveAndPerplexityIsItsExponential()
    {
        var decoder = Decoder();
        int[] sequence = [5, 6, 7, 8, 9];

        var entropy = decoder.CrossEntropy(sequence);

        Assert.True(entropy > 0, $"cross entropy came out at {entropy}");
        Assert.Equal(Math.Exp(entropy), decoder.Perplexity(sequence), 9);
    }

    [Fact]
    public void AnUntrainedModelsPerplexityIsAboutTheVocabularySize()
    {
        // With random weights every token is roughly equally likely, so the perplexity should sit
        // near the vocabulary size. Far below it would mean the model is already predicting, which
        // for random weights means something is leaking.
        var vocabulary = BuildVocabulary(extra: 45);
        var decoder = Decoder(vocabulary);

        var perplexity = decoder.Perplexity([5, 6, 7, 8, 9, 10, 11, 12]);

        Assert.True(perplexity > vocabulary.Count / 4.0,
            $"perplexity {perplexity:F2} was suspiciously low for {vocabulary.Count} random-weight tokens");
    }

    [Fact]
    public void GeneratingFromTextRoundTripsThroughTheTokenizer()
    {
        var vocabulary = new Vocabulary();
        foreach (var word in new[] { "the", "cat", "sat", "on", "mat" }) vocabulary.Add(word);

        var decoder = Decoder(vocabulary);
        var text = decoder.Generate("the cat", new WhitespaceTokenizer(),
            maxNewTokens: 4, options: SamplingOptions.Greedy);

        Assert.False(string.IsNullOrWhiteSpace(text));
        Assert.StartsWith("the cat", text, StringComparison.Ordinal);
    }

    [Fact]
    public void MalformedInputsAreRejected()
    {
        var decoder = Decoder();

        Assert.Throws<ArgumentException>(() => decoder.Forward([]));
        Assert.Throws<ArgumentException>(() => decoder.Generate([], maxNewTokens: 4));
        Assert.Throws<ArgumentException>(() => decoder.CrossEntropy([5]));
        Assert.Throws<ArgumentOutOfRangeException>(() => decoder.Generate([5], maxNewTokens: -1));

        // Longer than the context window in a single forward pass.
        Assert.Throws<ArgumentException>(() => decoder.Forward(Enumerable.Range(0, 100).ToArray()));
    }
}
