using Gravicode.HFNet.GraviTokenizers;
using Xunit;

namespace Gravicode.HFNet.GraviTokenizers.Tests;

/// <summary>
/// transformers' <c>EsmTokenizer</c>, on ESM-2's own 33-token vocabulary (<c>Fixtures/esm</c>, written by
/// <c>GraviTransformers.Tests/Fixtures/make_esm.py</c>).
/// </summary>
/// <remarks>
/// Every expected id list below is what transformers 5.17's <c>EsmTokenizer</c> returned for the same
/// text, copied by hand from the script's output - not produced by HF.Net.
/// </remarks>
public sealed class EsmTokenizerTests
{
    private static readonly string Folder = Path.Combine(AppContext.BaseDirectory, "Fixtures", "esm");

    private static HfTokenizer Esm() => HfTokenizer.FromDirectory(Folder);

    public static TheoryData<string, int[]> Reference => new()
    {
        { "MKTAYIAKQR", [0, 20, 15, 11, 5, 19, 12, 5, 15, 16, 10, 2] },
        { "M K T A", [0, 20, 15, 11, 5, 2] },                          // whitespace is ignored
        { "MKT<mask>AY", [0, 20, 15, 11, 32, 5, 19, 2] },               // special tokens inside the text
        { "mkta", [0, 3, 2] },                                          // lowercase is not in the vocabulary: one <unk> for the run
        { "MKJJXB", [0, 20, 15, 3, 24, 25, 2] },                        // "JJ" is one <unk>, not two
        { "<cls>MK<eos>", [0, 0, 20, 15, 2, 2] },
        { "", [0, 2] },
        { "MK  <mask> J A<pad>", [0, 20, 15, 32, 3, 5, 1, 2] },
    };

    [Theory]
    [MemberData(nameof(Reference))]
    public void Ids_match_transformers_EsmTokenizer(string text, int[] expected)
    {
        Assert.Equal(expected, Esm().Encode(text).ToIdArray());
    }

    [Fact]
    public void A_bert_reading_of_the_same_vocab_would_have_been_one_unknown()
    {
        // What 0.10 did: read as WordPiece, the protein is one word with no ## pieces to split it into.
        var bert = HfTokenizer.FromBertVocabulary(Path.Combine(Folder, "vocab.txt"));
        Assert.NotEqual(Esm().Encode("MKTAYIAKQR").ToIdArray(), bert.Encode("MKTAYIAKQR").ToIdArray());
    }

    [Fact]
    public void Special_ids_resolve_to_ESM_s_tokens()
    {
        var tokenizer = Esm();

        Assert.Equal(0, tokenizer.ClassifierId);
        Assert.Equal(1, tokenizer.PadId);
        Assert.Equal(2, tokenizer.SeparatorId);
        Assert.Equal(3, tokenizer.UnknownId);
        Assert.Equal(32, tokenizer.MaskId);
        Assert.Equal(33, tokenizer.VocabularySize);
    }

    [Fact]
    public void Decoding_joins_with_spaces_and_skips_specials_including_unknown()
    {
        var tokenizer = Esm();

        // transformers: '<cls> M K <unk> X B <eos>' and, skipping specials, 'M K X B'.
        int[] ids = [0, 20, 15, 3, 24, 25, 2];
        Assert.Equal("<cls> M K <unk> X B <eos>", tokenizer.Decode(ids, skipSpecialTokens: false));
        Assert.Equal("M K X B", tokenizer.Decode(ids));
        Assert.Equal("M K T A Y", tokenizer.Decode([0, 20, 15, 11, 32, 5, 19, 2]));
    }

    [Fact]
    public void A_pair_is_cls_A_eos_B_eos()
    {
        Assert.Equal([0, 20, 15, 11, 2, 5, 19, 12, 2], Esm().Encode("MKT", "AYI").ToIdArray());
    }

    [Fact]
    public void Offsets_point_at_each_residue()
    {
        var encoding = Esm().Encode("MK TA");

        Assert.Equal((0, 1), encoding.Offsets[1]);
        Assert.Equal((1, 2), encoding.Offsets[2]);
        Assert.Equal((3, 4), encoding.Offsets[3]);
        Assert.Equal((4, 5), encoding.Offsets[4]);
    }

    [Fact]
    public void Without_a_config_the_vocabulary_gives_itself_away()
    {
        var folder = Directory.CreateTempSubdirectory("hfnet-esm-").FullName;
        try
        {
            File.Copy(Path.Combine(Folder, "vocab.txt"), Path.Combine(folder, "vocab.txt"));
            Assert.Equal([0, 20, 15, 2], HfTokenizer.FromDirectory(folder).Encode("MK").ToIdArray());
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void A_null_eos_token_leaves_the_sequence_open()
    {
        // The Nucleotide Transformer's tokenizer_config.json sets "eos_token": null.
        var folder = Directory.CreateTempSubdirectory("hfnet-esm-").FullName;
        try
        {
            File.Copy(Path.Combine(Folder, "vocab.txt"), Path.Combine(folder, "vocab.txt"));
            File.WriteAllText(Path.Combine(folder, "tokenizer_config.json"), """{"tokenizer_class": "EsmTokenizer", "eos_token": null}""");
            Assert.Equal([0, 20, 15], HfTokenizer.FromDirectory(folder).Encode("MK").ToIdArray());
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void Multi_character_entries_are_matched_longest_first()
    {
        // The Nucleotide Transformer v1 layout: 6-mers, with single nucleotides as the fallback.
        var folder = Directory.CreateTempSubdirectory("hfnet-esm-").FullName;
        try
        {
            File.WriteAllLines(Path.Combine(folder, "vocab.txt"), ["<unk>", "<pad>", "<mask>", "<cls>", "<eos>", "<bos>", "ACGTAC", "GGGGGG", "A", "C", "G", "T", "N"]);
            File.WriteAllText(Path.Combine(folder, "tokenizer_config.json"), """{"tokenizer_class": "EsmTokenizer", "eos_token": null}""");

            // ACGTAC | GGGGGG | T | A - and N alone.
            Assert.Equal([3, 6, 7, 11, 8, 12], HfTokenizer.FromDirectory(folder).Encode("ACGTACGGGGGGTAN").ToIdArray());
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void A_folder_without_tokenizer_files_is_null_from_TryFromDirectory_and_an_error_from_FromDirectory()
    {
        var folder = Directory.CreateTempSubdirectory("hfnet-none-").FullName;
        try
        {
            Assert.Null(HfTokenizer.TryFromDirectory(folder));
            var error = Assert.Throws<FileNotFoundException>(() => HfTokenizer.FromDirectory(folder));
            Assert.Contains("vocab.txt", error.Message);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
