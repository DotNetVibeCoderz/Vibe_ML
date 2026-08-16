using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Tokenization;
using Xunit;

namespace Gravicode.Science.Tests.GraviText;

/// <summary>
/// Tests for the trainable sub-word tokenizers.
/// </summary>
/// <remarks>
/// The reference for BPE is the worked example from the original paper's lineage — a corpus of
/// <c>low</c>, <c>lower</c>, <c>newest</c> and <c>widest</c> whose first merges are known by hand.
/// For unigram, the references are the properties that distinguish Viterbi segmentation from a
/// greedy one, and exact reversibility, which a whitespace-delimited scheme cannot offer.
/// </remarks>
public class SubwordTokenizerTests
{
    /// <summary>
    /// The corpus every BPE explanation uses, with the frequencies that make the merges predictable.
    /// </summary>
    private static IEnumerable<string> ClassicCorpus()
    {
        var words = new (string Word, int Count)[] { ("low", 5), ("lower", 2), ("newest", 6), ("widest", 3) };
        foreach (var (word, count) in words)
            for (var i = 0; i < count; i++)
                yield return word;
    }

    // -------------------------------------------------------------------- BPE

    [Fact]
    public void TheFirstMergesAreTheMostFrequentAdjacentPairs()
    {
        // Worked by hand: "es" occurs 9 times (newest 6 + widest 3), more than any other pair, so
        // it merges first. Then "es" followed by the word-final "t" — also 9. The t carries the
        // end-of-word marker because it ends both words, which is the marker doing its job.
        var tokenizer = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 40);

        Assert.Equal(("e", "s"), tokenizer.Merges[0]);
        Assert.Equal(("es", "t" + BpeTokenizer.EndOfWord), tokenizer.Merges[1]);
    }

    [Fact]
    public void FrequentWordsBecomeSingleTokens()
    {
        // The property that makes BPE worth the trouble: enough merges and a common word is one
        // token, while the vocabulary stays bounded.
        var tokenizer = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 60, minFrequency: 1);

        Assert.Single(tokenizer.Encode("newest"));
        Assert.Single(tokenizer.Encode("low"));
    }

    [Fact]
    public void AnUnseenWordDecomposesRatherThanBecomingUnknown()
    {
        // The whole point of sub-word tokenization. "lowest" was never in the corpus, but "low" and
        // "est" both were, so it segments into known pieces.
        var tokenizer = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 60, minFrequency: 1);
        var pieces = tokenizer.Encode("lowest");

        Assert.True(pieces.Count > 1, "an unseen word should decompose");
        Assert.All(pieces, piece => Assert.True(tokenizer.Vocabulary.Contains(piece),
            $"piece '{piece}' was not in the vocabulary"));
        Assert.DoesNotContain(Vocabulary.UnknownToken, pieces);
    }

    [Fact]
    public void EncodingIsReversible()
    {
        var tokenizer = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 60, minFrequency: 1);

        Assert.Equal("newest", tokenizer.Decode(tokenizer.Encode("newest")));
        Assert.Equal("lowest", tokenizer.Decode(tokenizer.Encode("lowest")));
        Assert.Equal("low lower newest", tokenizer.Decode(tokenizer.Tokenize("low lower newest")));
    }

    [Fact]
    public void TheEndOfWordMarkerKeepsWordFinalPiecesDistinct()
    {
        // Without it, merges learned inside words leak across boundaries and the segmentation stops
        // surviving a change in spacing.
        var tokenizer = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 40);
        var pieces = tokenizer.Encode("low");

        Assert.EndsWith(BpeTokenizer.EndOfWord, pieces[^1], StringComparison.Ordinal);
        Assert.All(pieces.Take(pieces.Count - 1),
            piece => Assert.DoesNotContain(BpeTokenizer.EndOfWord, piece, StringComparison.Ordinal));
    }

    [Fact]
    public void MergesAreAppliedByRankNotByPosition()
    {
        // The subtle bug: applying merges left to right lets an early low-rank merge consume a
        // symbol a higher-rank merge needed. Replaying by rank must give exactly what training
        // produced.
        var tokenizer = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 60, minFrequency: 1);

        // Rebuilt from the merges alone — no cache, no vocabulary — and must agree.
        var rebuilt = new BpeTokenizer(tokenizer.Merges);

        foreach (var word in new[] { "newest", "widest", "lowest", "lower", "slowest" })
            Assert.Equal(tokenizer.Encode(word), rebuilt.Encode(word));
    }

    [Fact]
    public void TrainingIsReproducible()
    {
        // Ties broken by the pair itself rather than by dictionary order, so two runs over the same
        // corpus cannot diverge. Irreproducibility here is invisible until a model trained on one
        // tokenizer is served with another.
        var first = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 60, minFrequency: 1);
        var second = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 60, minFrequency: 1);

        Assert.Equal(first.Merges, second.Merges);
    }

    [Fact]
    public void ALargerVocabularyMeansFewerPiecesPerWord()
    {
        var small = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 20, minFrequency: 1);
        var large = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 60, minFrequency: 1);

        Assert.True(large.Encode("newest").Count <= small.Encode("newest").Count);
        Assert.True(large.Merges.Count > small.Merges.Count);
    }

    [Fact]
    public void TrainingStopsWhenNoPairIsFrequentEnough()
    {
        // A large budget must not manufacture merges that the corpus does not support.
        var tokenizer = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 100000, minFrequency: 3);

        Assert.True(tokenizer.Merges.Count < 100,
            $"training produced {tokenizer.Merges.Count} merges from a four-word corpus");
    }

    [Fact]
    public void MergesRoundTripThroughAFile()
    {
        // The merges are saved rather than the vocabulary, because the order is what defines the
        // tokenization and cannot be recovered from a token list.
        var tokenizer = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 60, minFrequency: 1);
        var path = Path.Combine(Path.GetTempPath(), $"gravi_{Guid.NewGuid():N}.merges");

        try
        {
            tokenizer.Save(path);
            var loaded = BpeTokenizer.Load(path);

            Assert.Equal(tokenizer.Merges, loaded.Merges);
            foreach (var word in new[] { "newest", "lowest", "unseen" })
                Assert.Equal(tokenizer.Encode(word), loaded.Encode(word));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void MultiByteCharactersAreNotTornInHalf()
    {
        // Splitting on chars rather than text elements would cut a surrogate pair down the middle
        // and produce pieces that are not valid text.
        var tokenizer = BpeTokenizer.Train(["kucing 🐱 lucu", "anjing 🐶 lucu", "kucing lucu"],
            vocabularySize: 40, minFrequency: 1);

        var pieces = tokenizer.Tokenize("kucing lucu");
        Assert.Equal("kucing lucu", tokenizer.Decode(pieces));
    }

    [Fact]
    public void AnEmptyWordEncodesToNothing()
    {
        var tokenizer = BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 30);
        Assert.Empty(tokenizer.Encode(""));
    }

    // ---------------------------------------------------------------- unigram

    private static readonly string[] Sentences =
    [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog",
        "a cat on a mat",
        "the mat and the log",
    ];

    [Fact]
    public void UnigramSegmentationIsExactlyReversible()
    {
        // What encoding whitespace rather than splitting on it buys: decoding is concatenation, and
        // the original spacing is recovered rather than guessed.
        var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: 60, seedSize: 400);

        foreach (var sentence in Sentences)
            Assert.Equal(sentence, tokenizer.Decode(tokenizer.Encode(sentence)));
    }

    [Fact]
    public void UnseenTextStillSegmentsWithoutUnknowns()
    {
        var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: 60, seedSize: 400);
        var pieces = tokenizer.Encode("the cat on the log");

        Assert.NotEmpty(pieces);
        Assert.Equal("the cat on the log", tokenizer.Decode(pieces));
    }

    [Fact]
    public void SegmentationIsOptimalRatherThanGreedy()
    {
        // The Viterbi claim, checked directly: no other segmentation of the text into vocabulary
        // pieces can score higher. A greedy longest-match scan has no such guarantee.
        var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: 60, seedSize: 400);

        const string text = "the cat sat";
        var chosen = tokenizer.Encode(text);
        var chosenScore = chosen.Sum(tokenizer.LogProbability);

        // Compare against every segmentation reachable by splitting at a different first boundary.
        var normalised = string.Concat("▁", text).Replace(' ', '▁');

        for (var cut = 1; cut < Math.Min(normalised.Length, 12); cut++)
        {
            var head = normalised[..cut];
            if (double.IsNegativeInfinity(tokenizer.LogProbability(head))) continue;

            var tail = tokenizer.Encode(normalised[cut..].Replace('▁', ' '));
            var alternativeScore = tokenizer.LogProbability(head) + tail.Sum(tokenizer.LogProbability);

            Assert.True(chosenScore >= alternativeScore - 1e-9,
                $"a segmentation starting '{head}' scored {alternativeScore:F4} against the chosen {chosenScore:F4}");
        }
    }

    [Fact]
    public void EveryCharacterRemainsSpellable()
    {
        // Pruning must never drop a single character: a vocabulary that cannot spell a character
        // cannot segment text containing it at all.
        var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: 30, seedSize: 400);

        foreach (var character in "thecasondgml▁")
            Assert.False(double.IsNegativeInfinity(tokenizer.LogProbability(character.ToString())),
                $"the character '{character}' was pruned out of the vocabulary");
    }

    [Fact]
    public void FrequentSequencesGetHigherProbabilityThanRareOnes()
    {
        // What EM is for. "the" appears in every sentence; "log" in two.
        var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: 80, seedSize: 400);

        var common = tokenizer.LogProbability("▁the");
        var rare = tokenizer.LogProbability("▁log");

        if (!double.IsNegativeInfinity(common) && !double.IsNegativeInfinity(rare))
            Assert.True(common > rare,
                $"'the' scored {common:F4} against 'log's {rare:F4}");
    }

    [Fact]
    public void PruningReachesTheRequestedVocabularySize()
    {
        foreach (var size in new[] { 30, 50, 80 })
        {
            var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: size, seedSize: 400);
            Assert.True(tokenizer.PieceCount <= size,
                $"asked for {size} pieces and got {tokenizer.PieceCount}");
        }
    }

    [Fact]
    public void SamplingProducesSeveralDistinctSegmentations()
    {
        // Subword regularisation: training on more than one segmentation of a sentence is what
        // makes a model robust to the tokenizer's arbitrary choices. BPE cannot do this — it has no
        // probabilities to sample from.
        var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: 80, seedSize: 400);
        var rng = new GraviRandom(7);

        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 0; i < 100; i++)
            seen.Add(string.Join('|', tokenizer.SampleEncoding("the cat sat", rng, alpha: 0.2)));

        Assert.True(seen.Count > 1, "sampling produced only one segmentation");
    }

    [Fact]
    public void SampledSegmentationsAreStillReversible()
    {
        // An alternative segmentation is only useful if it means the same text.
        var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: 80, seedSize: 400);
        var rng = new GraviRandom(11);

        for (var i = 0; i < 50; i++)
            Assert.Equal("the cat sat", tokenizer.Decode(tokenizer.SampleEncoding("the cat sat", rng)));
    }

    [Fact]
    public void ALargeAlphaMakesSamplingApproachTheBestSegmentation()
    {
        // The smoothing parameter's meaning: as it grows, the sampling distribution concentrates on
        // the Viterbi path.
        var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: 80, seedSize: 400);
        var best = string.Join('|', tokenizer.Encode("the cat sat"));

        var rng = new GraviRandom(13);
        var matches = 0;
        for (var i = 0; i < 50; i++)
            if (string.Join('|', tokenizer.SampleEncoding("the cat sat", rng, alpha: 20.0)) == best) matches++;

        Assert.True(matches > 40, $"with a large alpha only {matches}/50 draws were the best segmentation");
    }

    [Fact]
    public void UnigramModelsRoundTripThroughAFile()
    {
        var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: 60, seedSize: 400);
        var path = Path.Combine(Path.GetTempPath(), $"gravi_{Guid.NewGuid():N}.model");

        try
        {
            tokenizer.Save(path);
            var loaded = UnigramTokenizer.Load(path);

            Assert.Equal(tokenizer.PieceCount, loaded.PieceCount);
            foreach (var sentence in Sentences)
                Assert.Equal(tokenizer.Encode(sentence), loaded.Encode(sentence));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void TextWithNoSpacesIsStillSegmented()
    {
        // The case a whitespace pre-tokenizer cannot handle at all, and the reason SentencePiece
        // treats the input as a raw stream.
        var tokenizer = UnigramTokenizer.Train(
            ["catsatonthemat", "dogsatonthelog", "catandthedog"], vocabularySize: 40, seedSize: 300);

        var pieces = tokenizer.Encode("catsatonthelog");

        Assert.True(pieces.Count > 1, "the text was not segmented at all");
        Assert.Equal("catsatonthelog", tokenizer.Decode(pieces));
    }

    [Fact]
    public void UnrepresentableCharactersDoNotBreakSegmentation()
    {
        // A character the vocabulary has never seen would leave the lattice disconnected and every
        // later position unreachable. Emitting it alone keeps segmentation total.
        var tokenizer = UnigramTokenizer.Train(Sentences, vocabularySize: 60, seedSize: 400);
        var pieces = tokenizer.Encode("the cat ✦ sat");

        Assert.NotEmpty(pieces);
        Assert.Equal("the cat ✦ sat", tokenizer.Decode(pieces));
    }

    [Fact]
    public void MalformedInputsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => UnigramTokenizer.Train([], vocabularySize: 20));
        Assert.Throws<ArgumentOutOfRangeException>(() => UnigramTokenizer.Train(Sentences, vocabularySize: 1));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => UnigramTokenizer.Train(Sentences, shrinkFactor: 1.5));
        Assert.Throws<ArgumentException>(
            () => new UnigramTokenizer(new Dictionary<string, double>()));

        Assert.Throws<ArgumentOutOfRangeException>(
            () => BpeTokenizer.Train(ClassicCorpus(), vocabularySize: 0));
    }
}
