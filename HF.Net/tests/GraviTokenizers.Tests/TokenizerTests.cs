using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTokenizers.Components;
using Xunit;

namespace Gravicode.HFNet.GraviTokenizers.Tests;

/// <summary>
/// Tests for the tokenizer pipeline, built from vocabularies defined in the test rather than
/// downloaded, so they pin behaviour without a network.
/// </summary>
public sealed class WordPieceTests
{
    private static HfTokenizer Build(bool lowercase = true)
    {
        // A vocabulary small enough to reason about entirely. Ids are the list position.
        string[] tokens =
        [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "hello", "world", "token", "##izer", "##s", "##ing", "the", "cat", ",", "!", "un", "##believable",
        ];

        var vocabulary = tokens.Select((t, i) => (t, i)).ToDictionary(p => p.t, p => p.i, StringComparer.Ordinal);

        return new HfTokenizer(
            new WordPieceModel(vocabulary),
            new BertPreTokenizer(),
            new BertNormalizer(lowercase),
            new WordPieceDecoder(),
            PostProcessor.Bert(2, 3),
            [new AddedToken(0, "[PAD]", true), new AddedToken(4, "[MASK]", true)]);
    }

    [Fact]
    public void SegmentsAWordIntoLongestMatchingPieces()
    {
        var encoding = Build().Encode("tokenizers", addSpecialTokens: false);

        Assert.Equal(["token", "##izer", "##s"], encoding.Tokens);
        Assert.Equal([7, 8, 9], encoding.Ids);
    }

    [Fact]
    public void WrapsWithClassifierAndSeparator()
    {
        var encoding = Build().Encode("hello world");

        Assert.Equal(["[CLS]", "hello", "world", "[SEP]"], encoding.Tokens);
        Assert.Equal([2, 5, 6, 3], encoding.Ids);
        Assert.Equal([1, 0, 0, 1], encoding.SpecialTokensMask);
    }

    [Fact]
    public void SplitsPunctuationIntoItsOwnToken()
    {
        var encoding = Build().Encode("hello, world!", addSpecialTokens: false);

        Assert.Equal(["hello", ",", "world", "!"], encoding.Tokens);
    }

    [Fact]
    public void AWordWithNoSegmentationBecomesEntirelyUnknown()
    {
        // "quixotic" shares no piece with the vocabulary. The reference implementation makes the
        // whole word unknown rather than emitting the prefixes it did match.
        var encoding = Build().Encode("quixotic", addSpecialTokens: false);

        Assert.Equal(["[UNK]"], encoding.Tokens);
    }

    [Fact]
    public void OffsetsPointBackIntoTheOriginalCasing()
    {
        const string Text = "Tokenizers";
        var encoding = Build().Encode(Text, addSpecialTokens: false);

        Assert.Equal("Token", encoding.Span(Text, 0));
        Assert.Equal("izer", encoding.Span(Text, 1));
        Assert.Equal("s", encoding.Span(Text, 2));

        // The whole run recovers the source substring, capital and all.
        Assert.Equal("Tokenizers", encoding.Span(Text, 0, 2));
    }

    [Fact]
    public void SpecialTokensCoverNothing()
    {
        const string Text = "hello world";
        var encoding = Build().Encode(Text);

        Assert.Equal((0, 0), encoding.Offsets[0]);
        Assert.Equal("", encoding.Span(Text, 0));
    }

    [Fact]
    public void DecodeDropsSpecialTokensAndRejoinsSubwords()
    {
        var tokenizer = Build();
        var encoding = tokenizer.Encode("tokenizers");

        Assert.Equal("tokenizers", tokenizer.Decode(encoding.Ids));
    }

    [Fact]
    public void DecodeCanKeepSpecialTokens()
    {
        var tokenizer = Build();
        var encoding = tokenizer.Encode("hello");

        Assert.Contains("[CLS]", tokenizer.Decode(encoding.Ids, skipSpecialTokens: false));
    }

    [Fact]
    public void AnAddedTokenIsMatchedVerbatimRatherThanSplit()
    {
        var tokenizer = Build();
        var encoding = tokenizer.Encode("hello [MASK] world", addSpecialTokens: false);

        Assert.Equal(["hello", "[MASK]", "world"], encoding.Tokens);
    }

    [Fact]
    public void PairEncodingNumbersTheSecondSegmentOne()
    {
        var encoding = Build().Encode("hello", "world");

        Assert.Equal(["[CLS]", "hello", "[SEP]", "world", "[SEP]"], encoding.Tokens);
        Assert.Equal([0, 0, 0, 1, 1], encoding.TypeIds);
    }

    [Fact]
    public void BatchPadsToTheLongestAndMasksThePadding()
    {
        var batch = Build().EncodeBatch(["hello", "hello world tokenizers"]);

        Assert.Equal(2, batch.Count);
        Assert.All(batch.Encodings, e => Assert.Equal(batch.Width, e.Length));

        // The short input's real tokens are [CLS] hello [SEP]; the rest is padding.
        Assert.Equal(3, batch.Encodings[0].AttentionMask.Sum());
        Assert.Equal(batch.Width, batch.Encodings[1].AttentionMask.Sum());
    }

    [Fact]
    public void BatchTruncatesToAFixedWidth()
    {
        var batch = Build().EncodeBatch(
            ["hello world tokenizers the cat"], BatchOptions.Exactly(4));

        Assert.Equal(4, batch.Width);
        Assert.Equal(4, batch.Encodings[0].Length);
    }

    [Fact]
    public void DefaultBatchOptionsActuallyPad()
    {
        // BatchOptions is a record struct, so `new()` would zero the padding strategy rather than
        // running the primary constructor's defaults. Default must not be spelled that way.
        Assert.Equal(PaddingStrategy.Longest, BatchOptions.Default.Padding);
        Assert.True(BatchOptions.Default.MaxLength > 0);
    }

    [Fact]
    public void LowercasingCanBeTurnedOff()
    {
        // With case preserved, "hello" is no longer in the vocabulary as written.
        var encoding = Build(lowercase: false).Encode("Hello", addSpecialTokens: false);

        Assert.Equal(["[UNK]"], encoding.Tokens);
    }

    [Fact]
    public void SpecialTokenIdsAreResolved()
    {
        var tokenizer = Build();

        Assert.Equal(0, tokenizer.PadId);
        Assert.Equal(2, tokenizer.ClassifierId);
        Assert.Equal(3, tokenizer.SeparatorId);
        Assert.Equal(4, tokenizer.MaskId);
    }
}

/// <summary>Tests for byte pair encoding and the byte alphabet it runs over.</summary>
public sealed class BpeTests
{
    [Fact]
    public void MergesAreAppliedByRankNotByPosition()
    {
        // "abcd" with merges ranked: (c,d) first, then (a,b), then (ab,cd).
        // Taking the leftmost mergeable pair each time would merge (a,b) first and reach the same
        // place here, so the ranks are ordered to distinguish: (b,c) is ranked last and must lose.
        var vocabulary = new[] { "a", "b", "c", "d", "cd", "ab", "abcd", "bc" }
            .Select((t, i) => (t, i)).ToDictionary(p => p.t, p => p.i, StringComparer.Ordinal);

        var model = new BpeModel(vocabulary, [("c", "d"), ("a", "b"), ("ab", "cd"), ("b", "c")]);

        Assert.Equal(["abcd"], model.Tokenize("abcd"));
    }

    [Fact]
    public void ByteAlphabetRoundTripsEveryByte()
    {
        foreach (var text in (string[])["hello", "Ünïcodé", "日本語", "emoji 🎉", " leading space"])
        {
            Assert.Equal(text, ByteAlphabet.Decode(ByteAlphabet.Encode(text)));
        }
    }

    [Fact]
    public void ByteAlphabetMapsSpaceToAVisibleCharacter()
    {
        // A space must not survive as a space, or the pre-tokenizer's own splitting would destroy
        // it and the encoding would stop being lossless.
        var encoded = ByteAlphabet.Encode(" ");

        Assert.Single(encoded);
        Assert.NotEqual(' ', encoded[0]);
    }

    [Fact]
    public void ByteLevelPreTokenizerKeepsALeadingSpaceWithItsWord()
    {
        var pieces = new ByteLevelPreTokenizer().Split("hello world");

        Assert.Equal(2, pieces.Count);

        // The second piece carries the space, which is what the GPT-2 regex is for.
        Assert.Equal("world", ByteAlphabet.Decode(pieces[1].Word).TrimStart());
        Assert.StartsWith(" ", ByteAlphabet.Decode(pieces[1].Word));
    }
}

/// <summary>Tests for Unigram Viterbi segmentation.</summary>
public sealed class UnigramTests
{
    [Fact]
    public void PicksTheHighestProbabilitySegmentationNotTheGreedyOne()
    {
        // Greedy longest-match would take "ab" then be forced into "c" alone.
        // The better total is "a" + "bc", and only a global search finds it.
        var model = new UnigramModel(
        [
            ("a", -1.0),
            ("b", -5.0),
            ("c", -5.0),
            ("ab", -3.0),
            ("bc", -1.0),
        ]);

        Assert.Equal(["a", "bc"], model.Tokenize("abc"));
    }

    [Fact]
    public void AnUnknownCharacterStaysCrossable()
    {
        // Without a finite penalty for unknown single characters the whole word would have no path.
        var model = new UnigramModel([("a", -1.0), ("b", -1.0)]);

        var pieces = model.Tokenize("azb");

        Assert.Equal(3, pieces.Count);
        Assert.Equal("z", pieces[1]);
    }
}

/// <summary>Tests for the normalizers.</summary>
public sealed class NormalizerTests
{
    [Fact]
    public void StripsAccentsWithoutStringNormalize()
    {
        // The libraries build with InvariantGlobalization, under which String.Normalize is a no-op,
        // so this has to come from the explicit folding table.
        var normalizer = new BertNormalizer(Lowercase: true, StripAccents: true);

        Assert.Equal("cafe", normalizer.Normalize("café"));
        Assert.Equal("naive", normalizer.Normalize("naïve"));
        Assert.Equal("uber", normalizer.Normalize("Über"));
    }

    [Fact]
    public void LowercasesWithoutStrippingWhenAsked()
    {
        Assert.Equal("café", new BertNormalizer(Lowercase: true, StripAccents: false).Normalize("Café"));
    }

    [Fact]
    public void SequenceAppliesInOrder()
    {
        var normalizer = new SequenceNormalizer(
            [new ReplaceNormalizer("-", " "), new LowercaseNormalizer(), new StripNormalizer()]);

        Assert.Equal("a b", normalizer.Normalize("  A-B  "));
    }
}

/// <summary>Tests for pre-tokenization.</summary>
public sealed class PreTokenizerTests
{
    [Fact]
    public void BertSplitsOnWhitespaceAndPunctuation()
    {
        var pieces = new BertPreTokenizer().Split("don't stop!");

        Assert.Equal(["don", "'", "t", "stop", "!"], pieces.Select(p => p.Word));
    }

    [Fact]
    public void BertTreatsAsciiSymbolsAsPunctuation()
    {
        // $ and + are Unicode symbols, not punctuation, but BERT splits on them.
        var pieces = new BertPreTokenizer().Split("$5+3");

        Assert.Equal(["$", "5", "+", "3"], pieces.Select(p => p.Word));
    }

    [Fact]
    public void OffsetsAreRelativeToTheWholeInput()
    {
        const string Text = "one two three";
        var pieces = new WhitespacePreTokenizer().Split(Text);

        Assert.Equal(3, pieces.Count);
        foreach (var piece in pieces)
        {
            Assert.Equal(piece.Word, Text[piece.Start..piece.End]);
        }
    }

    [Fact]
    public void SequenceRebasesOffsetsOntoTheOriginal()
    {
        const string Text = "alpha beta-gamma";

        var pieces = new SequencePreTokenizer(
            [new WhitespacePreTokenizer(), new SplitPreTokenizer("-")]).Split(Text);

        Assert.Equal(["alpha", "beta", "gamma"], pieces.Select(p => p.Word));
        foreach (var piece in pieces)
        {
            Assert.Equal(piece.Word, Text[piece.Start..piece.End]);
        }
    }

    [Fact]
    public void MetaspaceMarksWordBoundaries()
    {
        var pieces = new MetaspacePreTokenizer().Split("hello world");

        Assert.All(pieces, p => Assert.StartsWith("▁", p.Word));
    }
}
