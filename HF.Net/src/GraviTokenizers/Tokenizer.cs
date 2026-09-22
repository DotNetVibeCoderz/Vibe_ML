using Gravicode.HFNet.GraviTokenizers.Components;
using Gravicode.Science.GraviText.Tokenization;

namespace Gravicode.HFNet.GraviTokenizers;

/// <summary>
/// The one-line entry point: tokenize a string, or train a tokenizer on a corpus.
/// </summary>
/// <remarks>
/// <para>
/// The <see cref="BPE(string)"/>, <see cref="WordPiece(string)"/> and
/// <see cref="SentencePiece(string)"/> shortcuts each run a well-known pretrained tokenizer, which
/// they download once and then keep. They are for exploring and for samples; anything that has a
/// model already should use that model's own tokenizer via
/// <see cref="HfTokenizer.FromPretrained"/>, because a tokenizer that does not match its model
/// produces ids the model was never trained on.
/// </para>
/// <para>
/// The <c>Train</c> methods build a tokenizer from a corpus, delegating the learning to the
/// trainers in <c>Gravicode.Science.GraviText</c> and wrapping the result so it encodes through the
/// same pipeline as a loaded one.
/// </para>
/// </remarks>
public static class Tokenizer
{
    private static readonly Lazy<HfTokenizer> Gpt2 = new(() => HfTokenizer.FromPretrained("gpt2"));
    private static readonly Lazy<HfTokenizer> Bert = new(() => HfTokenizer.FromPretrained("bert-base-uncased"));
    private static readonly Lazy<HfTokenizer> Xlm = new(() => HfTokenizer.FromPretrained("xlm-roberta-base"));

    /// <summary>Tokenizes text with byte-level BPE, as GPT-2 does.</summary>
    /// <param name="text">The input.</param>
    /// <returns>The subword pieces.</returns>
    /// <remarks>Downloads the GPT-2 tokenizer on first use and caches it for the process.</remarks>
    public static IReadOnlyList<string> BPE(string text) => Gpt2.Value.Encode(text).Tokens;

    /// <summary>Tokenizes text with WordPiece, as BERT does.</summary>
    /// <param name="text">The input.</param>
    public static IReadOnlyList<string> WordPiece(string text) => Bert.Value.Encode(text).Tokens;

    /// <summary>Tokenizes text with a SentencePiece Unigram model, as XLM-RoBERTa does.</summary>
    /// <param name="text">The input.</param>
    public static IReadOnlyList<string> SentencePiece(string text) => Xlm.Value.Encode(text).Tokens;

    /// <summary>Loads the tokenizer belonging to a Hub model.</summary>
    /// <param name="repoId">A model id such as <c>bert-base-uncased</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static HfTokenizer FromPretrained(string repoId, string revision = "main")
        => HfTokenizer.FromPretrained(repoId, revision);

    /// <summary>Loads a tokenizer from a local <c>tokenizer.json</c>.</summary>
    public static HfTokenizer Load(string path) => HfTokenizer.Load(path);

    // ------------------------------------------------------------------ training

    /// <summary>Trains a byte pair encoding tokenizer on a corpus.</summary>
    /// <param name="corpus">The documents to learn from.</param>
    /// <param name="vocabularySize">How many pieces to end up with.</param>
    /// <param name="minFrequency">Ignore pairs seen fewer times than this.</param>
    /// <remarks>
    /// The merge order is the model. Two runs over the same corpus give the same merges here, which
    /// is what makes a tokenizer trained today still match text tokenized next month.
    /// </remarks>
    public static HfTokenizer TrainBpe(
        IEnumerable<string> corpus, int vocabularySize = 1000, int minFrequency = 2)
    {
        var trained = BpeTokenizer.Train(corpus, vocabularySize, minFrequency);

        var vocabulary = ToDictionary(trained.Vocabulary);
        var model = new BpeModel(
            vocabulary,
            trained.Merges,
            unknownToken: Vocabulary.UnknownToken,
            endOfWordSuffix: BpeTokenizer.EndOfWord);

        return new HfTokenizer(
            model,
            new WhitespacePreTokenizer(),
            new LowercaseNormalizer(),
            new WhitespaceDecoder(),
            BertPostProcessor(trained.Vocabulary),
            SpecialTokens(trained.Vocabulary));
    }

    /// <summary>Trains a WordPiece tokenizer on a corpus.</summary>
    /// <param name="corpus">The documents to learn from.</param>
    /// <param name="vocabularySize">How many pieces to end up with.</param>
    /// <param name="minFrequency">Ignore pieces seen fewer times than this.</param>
    public static HfTokenizer TrainWordPiece(
        IEnumerable<string> corpus, int vocabularySize = 5000, int minFrequency = 2)
    {
        var vocabulary = WordPieceTokenizer.Train(corpus, vocabularySize, minFrequency);

        return new HfTokenizer(
            new WordPieceModel(ToDictionary(vocabulary)),
            new BertPreTokenizer(),
            new BertNormalizer(),
            new WordPieceDecoder(),
            BertPostProcessor(vocabulary),
            SpecialTokens(vocabulary));
    }

    /// <summary>Trains a SentencePiece-style Unigram tokenizer on a corpus.</summary>
    /// <param name="corpus">The documents to learn from.</param>
    /// <param name="vocabularySize">How many pieces to end up with.</param>
    public static HfTokenizer TrainUnigram(IEnumerable<string> corpus, int vocabularySize = 1000)
    {
        var trained = UnigramTokenizer.Train(corpus, vocabularySize);
        var tokens = trained.Vocabulary.Tokens;

        var pieces = new List<(string, double)>(tokens.Count);
        foreach (var token in tokens) pieces.Add((token, trained.LogProbability(token)));

        return new HfTokenizer(
            new UnigramModel(pieces, trained.Vocabulary.UnknownId),
            new MetaspacePreTokenizer(),
            new LowercaseNormalizer(),
            new MetaspaceDecoder(),
            BertPostProcessor(trained.Vocabulary),
            SpecialTokens(trained.Vocabulary));
    }

    private static Dictionary<string, int> ToDictionary(Vocabulary vocabulary)
    {
        var map = new Dictionary<string, int>(vocabulary.Count, StringComparer.Ordinal);
        for (var id = 0; id < vocabulary.Count; id++) map[vocabulary[id]] = id;
        return map;
    }

    private static PostProcessor BertPostProcessor(Vocabulary vocabulary)
        => PostProcessor.Bert(vocabulary.ClassId, vocabulary.SeparatorId);

    private static List<AddedToken> SpecialTokens(Vocabulary vocabulary) =>
    [
        new(vocabulary.PadId, Vocabulary.PadToken, true),
        new(vocabulary.UnknownId, Vocabulary.UnknownToken, true),
        new(vocabulary.ClassId, Vocabulary.ClassToken, true),
        new(vocabulary.SeparatorId, Vocabulary.SeparatorToken, true),
    ];
}
