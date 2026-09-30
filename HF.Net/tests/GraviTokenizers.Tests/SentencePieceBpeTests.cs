using System.Text.Json;
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTokenizers.Components;
using Gravicode.HFNet.GraviTokenizers.Unicode;
using Xunit;

namespace Gravicode.HFNet.GraviTokenizers.Tests;

/// <summary>
/// Unicode normalization against Python's <c>unicodedata</c>, and the pieces Llama-style tokenizers
/// need - byte fallback, the decoder chain, digits - against the Rust crate's behaviour.
/// </summary>
public sealed class SentencePieceBpeTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    [Fact]
    public void All_four_normalization_forms_match_unicodedata()
    {
        // Fixtures/make_normalization.py: composed and decomposed Latin, stacked marks that need
        // reordering, Hangul, compatibility characters, singletons and exclusions, and random strings.
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "normalization.json")));
        var failures = new List<string>();

        foreach (var row in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            var text = row.GetProperty("text").GetString()!;
            foreach (var (name, form) in new[] { ("NFC", NormalizationForm.C), ("NFD", NormalizationForm.D), ("NFKC", NormalizationForm.KC), ("NFKD", NormalizationForm.KD) })
            {
                var expected = row.GetProperty(name).GetString();
                var actual = UnicodeNormalization.Normalize(text, form);
                if (actual != expected) failures.Add($"{name}({Escape(text)}) = {Escape(actual)}, expected {Escape(expected!)}");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(10)));
    }

    private static string Escape(string text) => string.Concat(text.EnumerateRunes().Select(r => r.Value < 0x80 ? r.ToString() : $"\\u{r.Value:X4}"));

    /// <summary>A tiny SentencePiece-style BPE: "▁" words, byte pieces for the rest.</summary>
    private static HfTokenizer Tiny()
    {
        var vocabulary = new Dictionary<string, int> { ["<unk>"] = 0, ["<s>"] = 1, ["</s>"] = 2 };
        foreach (var piece in (string[])["▁", "h", "i", "▁h", "▁hi", "t", "▁t"]) vocabulary[piece] = vocabulary.Count;
        for (var b = 0; b < 256; b++) vocabulary[$"<0x{b:X2}>"] = vocabulary.Count;

        var model = new BpeModel(vocabulary, [("▁", "h"), ("▁h", "i"), ("▁", "t")], "<unk>", byteFallback: true, fuseUnknown: true);
        var normalizer = new SequenceNormalizer([new PrependNormalizer("▁"), new ReplaceNormalizer(" ", "▁")]);
        var decoder = new SequenceDecoder([new ReplaceStep("▁", " "), new ByteFallbackStep(), new FuseStep(), new StripStep(' ', 1, 0)]);

        return new HfTokenizer(model, new WholeTextPreTokenizer(), normalizer, decoder,
            addedTokens: [new AddedToken(0, "<unk>", true), new AddedToken(1, "<s>", true), new AddedToken(2, "</s>", true)]);
    }

    [Fact]
    public void A_character_outside_the_vocabulary_becomes_its_utf8_bytes()
    {
        var tokenizer = Tiny();
        var encoding = tokenizer.Encode("hi é\nt");

        // "▁hi", "▁", then é as <0xC3><0xA9>, "\n" as <0x0A>, then "t" - whole text, one piece,
        // so the newline survives rather than being split away.
        Assert.Equal(["▁hi", "▁", "<0xC3>", "<0xA9>", "<0x0A>", "t"], encoding.Tokens);
    }

    [Fact]
    public void The_decoder_chain_turns_bytes_and_markers_back_into_the_text()
    {
        var tokenizer = Tiny();
        var ids = tokenizer.Encode("hi é\nt").Ids;

        Assert.Equal("hi é\nt", tokenizer.Decode(ids));
    }

    [Fact]
    public void Byte_pieces_that_are_not_valid_utf8_decode_to_one_replacement_per_byte()
    {
        var steps = new ByteFallbackStep().DecodeChain(["a", "<0xC3>", "<0x28>", "b"]);

        // 0xC3 0x28 is not UTF-8: the reference gives one U+FFFD per byte, not per sequence.
        Assert.Equal(["a", "�", "�", "b"], steps);
    }

    [Theory]
    [InlineData(true, new[] { "3", ".", "1", "4" })]
    [InlineData(false, new[] { "3", ".", "14" })]
    public void Digits_are_isolated_not_dropped(bool individual, string[] expected)
    {
        // Digits used to be split out and thrown away: "3.14" became ".".
        var json = """
            {"version":"1.0","added_tokens":[],"normalizer":null,
             "pre_tokenizer":{"type":"Digits","individual_digits":INDIVIDUAL},
             "post_processor":null,"decoder":null,
             "model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,
                      "vocab":{"[UNK]":0,"3":1,".":2,"1":3,"4":4,"14":5}}}
            """.Replace("INDIVIDUAL", individual ? "true" : "false");

        Assert.Equal(expected, HfTokenizer.FromJson(json).Encode("3.14", addSpecialTokens: false).Tokens);
    }

    [Fact]
    public void WordPiece_decoding_cleans_up_the_space_before_punctuation()
    {
        var step = new WordPieceStep();

        Assert.Equal("hello, world! it's", string.Concat(step.DecodeChain(["hello", ",", "world", "!", "it", "'", "s"])).Replace(" ' ", "'"));
        Assert.Equal("don't", WordPieceStep.CleanUp("do n't"));
    }
}
