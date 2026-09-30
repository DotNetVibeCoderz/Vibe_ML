using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTokenizers.Components;
using Xunit;

namespace Gravicode.HFNet.GraviTokenizers.Tests;

/// <summary>
/// Behaviour pinned to the Rust <c>tokenizers</c> crate and to Transformers, after a parity sweep
/// over eight real tokenizers found six ways the two disagreed.
/// </summary>
/// <remarks>
/// Every expected value here was produced by the reference - Python's <c>CLIPTokenizer</c>, the
/// crate's <c>Precompiled</c> normalizer, or the crate's own documented split examples - never by
/// this library. The fixtures are real files: the CLIP vocabulary of
/// <c>optimum-internal-testing/tiny-stable-diffusion-onnx</c> and XLM-RoBERTa's charsmap.
/// </remarks>
public sealed class ReferenceParityTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    // ------------------------------------------------------------------ CLIP

    private static HfTokenizer Clip()
        => HfTokenizer.FromClipFiles(Fixture("clip-vocab.json"), Fixture("clip-merges.txt")).WithPadToken("<|endoftext|>");

    [Theory]
    [InlineData("a lighthouse at dawn, oil painting", new[] { 0, 197, 77, 537, 449, 392, 741, 434, 69, 66, 999, 279, 80, 714, 81, 1 })]
    [InlineData("café crème, 3 cats & 12 dogs — it's    fine", new[] { 0, 68, 66, 71, 126, 259, 68, 83, 126, 102, 78, 195, 279, 243, 68, 1 })]
    [InlineData("", new[] { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 })]
    public void Clip_matches_the_reference_tokenizer_padded_and_truncated_to_16(string prompt, int[] expected)
    {
        // Read as GPT-2 BPE these files give different ids for every prompt, which is what a
        // Stable Diffusion pipeline was conditioned on until this was fixed. The accented case
        // covers lowercasing before the byte alphabet; the truncated ones keep <|endoftext|> last.
        var ids = Clip().EncodeBatch([prompt], BatchOptions.Exactly(16)).Encodings[0].Ids;

        Assert.Equal(expected, ids);
    }

    [Fact]
    public void Truncating_a_long_prompt_keeps_its_closing_token()
    {
        var ids = Clip().EncodeBatch(["beautiful mountain landscape " + string.Concat(Enumerable.Repeat("beautiful mountain landscape ", 9))], BatchOptions.Exactly(16)).Encodings[0].Ids;

        Assert.Equal([0, 67, 471, 529, 524, 86, 201, 78, 553, 85, 675, 77, 609, 84, 68, 1], ids);
    }

    [Fact]
    public void A_CLIP_vocabulary_is_recognised_without_a_config()
        => Assert.True(HfTokenizer.IsClip(null, Fixture("clip-vocab.json")));

    // ------------------------------------------------------------------ Split, the crate's examples

    [Theory]
    [InlineData(SplitBehavior.Removed, new[] { "the", "final", "countdown" })]
    [InlineData(SplitBehavior.Isolated, new[] { "the", "-", "final", "-", "-", "countdown" })]
    [InlineData(SplitBehavior.MergedWithPrevious, new[] { "the-", "final-", "-", "countdown" })]
    [InlineData(SplitBehavior.MergedWithNext, new[] { "the", "-final", "-", "-countdown" })]
    [InlineData(SplitBehavior.Contiguous, new[] { "the", "-", "final", "--", "countdown" })]
    public void Split_behaves_as_the_tokenizers_crate_documents(SplitBehavior behavior, string[] expected)
    {
        // The example from the crate's documentation of SplitDelimiterBehavior.
        var pieces = new SplitPreTokenizer("-", Invert: false, behavior).Split("the-final--countdown");

        Assert.Equal(expected, pieces.Select(p => p.Word));
    }

    [Fact]
    public void An_inverted_removed_split_keeps_the_matches()
    {
        // CLIP's pre-tokenizer: the pattern describes words, and invert makes everything else the
        // delimiter. Reading only "Removed" dropped the words and kept the gaps.
        var pieces = new SplitPreTokenizer(@"\p{L}+", Invert: true, SplitBehavior.Removed).Split("ab, cd  ef");

        Assert.Equal(["ab", "cd", "ef"], pieces.Select(p => p.Word));
        Assert.Equal([(0, 2), (4, 6), (8, 10)], pieces.Select(p => (p.Start, p.End)));
    }

    // ------------------------------------------------------------------ model type inference

    [Fact]
    public void A_model_with_merges_and_no_type_is_BPE()
    {
        // GPT-2's own tokenizer.json has no "type". Defaulting to WordPiece ran it as greedy
        // longest-match: "abc" became "ab"+"c" here instead of the ranked merges' "a"+"bc".
        const string Json = """
            {
              "model": {
                "vocab": { "a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4 },
                "merges": [ "b c", "a b" ]
              },
              "pre_tokenizer": { "type": "Whitespace" }
            }
            """;

        var tokenizer = HfTokenizer.FromJson(Json);

        Assert.Equal(["a", "bc"], tokenizer.Encode("abc").Tokens);
    }

    // ------------------------------------------------------------------ BERT and CJK

    [Fact]
    public void Each_CJK_ideograph_is_a_word_of_its_own_but_kana_is_not()
    {
        // Transformers' handle_chinese_chars: 日本語 are three pieces; の and テキスト stay a word.
        var pieces = new BertPreTokenizer().Split("日本語のテキスト text");

        Assert.Equal(["日", "本", "語", "のテキスト", "text"], pieces.Select(p => p.Word));
    }

    // ------------------------------------------------------------------ SentencePiece normalization

    [Theory]
    [InlineData("™", "TM")]
    [InlineData("ﬁle", "file")]
    [InlineData("①②", "12")]
    [InlineData("ＡＢＣ", "ABC")]
    [InlineData("café", "café")]
    [InlineData("é", "é")]
    [InlineData("plain text", "plain text")]
    public void The_precompiled_charsmap_normalizes_as_the_reference_does(string input, string expected)
    {
        // XLM-RoBERTa's charsmap, read directly. .NET's own NFKC is a no-op under invariant
        // globalization, and skipping the step made "™" an unknown token.
        var normalizer = new PrecompiledNormalizer(File.ReadAllText(Fixture("xlmr-charsmap.txt")));

        Assert.Equal(expected, normalizer.Normalize(input));
    }

    // ------------------------------------------------------------------ Unigram

    [Fact]
    public void Consecutive_unknown_characters_are_one_unknown_piece()
    {
        // The reference's Unigram fuses a run of unknowns: T5 reads "©®" as one <unk>, not two.
        var model = new UnigramModel([("<unk>", 0), ("a", -1), ("b", -1)], unknownId: 0);

        Assert.Equal(["a", "©®", "b"], model.Tokenize("a©®b"));
        Assert.Equal(0, model.IdOf("©®"));
    }
}
