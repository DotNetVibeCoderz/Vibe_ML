using System.Collections.Concurrent;
using System.Text;

namespace Gravicode.HFNet.GraviTokenizers.Components;

/// <summary>
/// Greedy longest-match subword segmentation - the model BERT and its descendants use.
/// </summary>
/// <param name="vocabulary">Piece to id.</param>
/// <param name="unknownToken">The piece to emit when a word cannot be segmented at all.</param>
/// <param name="continuingPrefix">The marker on a piece that is not word-initial.</param>
/// <param name="maxCharsPerWord">Words longer than this become the unknown token without trying.</param>
/// <remarks>
/// The segmentation is greedy from the left and never backtracks across a successful match. That
/// is the reference behaviour, and it is why a single unknown character in the middle of a word
/// makes the <b>whole word</b> unknown rather than just that character.
/// </remarks>
public sealed class WordPieceModel(
    IReadOnlyDictionary<string, int> vocabulary,
    string unknownToken = "[UNK]",
    string continuingPrefix = "##",
    int maxCharsPerWord = 100) : ITokenizerModel
{
    private readonly Dictionary<string, int> _vocabulary = new(vocabulary, StringComparer.Ordinal);
    private readonly string[] _byId = BuildReverse(vocabulary);
    private readonly ConcurrentDictionary<string, string[]> _cache = new(StringComparer.Ordinal);

    /// <summary>The piece emitted for text the vocabulary cannot represent.</summary>
    public string UnknownToken { get; } = unknownToken;

    /// <summary>The marker prefixed to a piece that continues a word.</summary>
    public string ContinuingPrefix { get; } = continuingPrefix;

    /// <inheritdoc />
    public int VocabularySize => _vocabulary.Count;

    /// <inheritdoc />
    public int IdOf(string token)
        => _vocabulary.TryGetValue(token, out var id) ? id : _vocabulary.GetValueOrDefault(UnknownToken, 0);

    /// <inheritdoc />
    public string TokenOf(int id) => id >= 0 && id < _byId.Length ? _byId[id] : "";

    /// <inheritdoc />
    public IReadOnlyList<string> Tokenize(string word)
        => _cache.GetOrAdd(word, static (key, self) => self.Segment(key), this);

    private string[] Segment(string word)
    {
        if (word.Length == 0) return [];
        if (word.Length > maxCharsPerWord) return [UnknownToken];

        var pieces = new List<string>();
        var start = 0;

        while (start < word.Length)
        {
            var end = word.Length;
            string? match = null;

            while (start < end)
            {
                var candidate = start == 0
                    ? word[start..end]
                    : ContinuingPrefix + word[start..end];

                if (_vocabulary.ContainsKey(candidate))
                {
                    match = candidate;
                    break;
                }

                end--;
            }

            // No prefix of the remainder is in the vocabulary, so the word as a whole is unknown.
            // Emitting the pieces found so far would produce a token sequence the model was never
            // trained on, which is worse than admitting the word is out of vocabulary.
            if (match is null) return [UnknownToken];

            pieces.Add(match);
            start = end;
        }

        return [.. pieces];
    }

    internal static string[] BuildReverse(IReadOnlyDictionary<string, int> vocabulary)
    {
        var size = vocabulary.Count == 0 ? 0 : vocabulary.Values.Max() + 1;
        var byId = new string[size];
        foreach (var pair in vocabulary)
        {
            if (pair.Value >= 0 && pair.Value < size) byId[pair.Value] = pair.Key;
        }
        return byId;
    }
}

/// <summary>
/// Byte pair encoding: repeatedly merge the highest-priority adjacent pair.
/// </summary>
/// <remarks>
/// <para>
/// Merges are applied <b>by rank, not left to right</b>. Scanning for the first mergeable pair and
/// taking it lets an early low-rank merge consume a symbol that a higher-rank merge needed, which
/// produces a different segmentation from the reference tokenizer for a small fraction of words -
/// small enough to pass a smoke test and large enough to move a model's output.
/// </para>
/// <para>
/// Results are cached per word. Real text repeats words heavily, and the merge loop is the
/// expensive part of tokenization.
/// </para>
/// </remarks>
public sealed class BpeModel : ITokenizerModel
{
    private readonly Dictionary<string, int> _vocabulary;
    private readonly string[] _byId;
    private readonly Dictionary<(string Left, string Right), int> _ranks;
    private readonly ConcurrentDictionary<string, string[]> _cache = new(StringComparer.Ordinal);
    private readonly string? _unknownToken;
    private readonly string _continuingPrefix;
    private readonly string _endOfWordSuffix;

    /// <summary>Creates a BPE model.</summary>
    /// <param name="vocabulary">Piece to id.</param>
    /// <param name="merges">The merge list, in priority order - earliest is highest priority.</param>
    /// <param name="unknownToken">The piece for unrepresentable input, or null to drop it.</param>
    /// <param name="continuingPrefix">A marker for non-initial pieces, usually empty.</param>
    /// <param name="endOfWordSuffix">A marker for the final piece of a word, usually empty.</param>
    public BpeModel(
        IReadOnlyDictionary<string, int> vocabulary,
        IReadOnlyList<(string Left, string Right)> merges,
        string? unknownToken = null,
        string continuingPrefix = "",
        string endOfWordSuffix = "")
    {
        _vocabulary = new Dictionary<string, int>(vocabulary, StringComparer.Ordinal);
        _byId = WordPieceModel.BuildReverse(vocabulary);
        _unknownToken = unknownToken;
        _continuingPrefix = continuingPrefix;
        _endOfWordSuffix = endOfWordSuffix;

        _ranks = new Dictionary<(string, string), int>(merges.Count);
        for (var i = 0; i < merges.Count; i++) _ranks.TryAdd(merges[i], i);
    }

    /// <inheritdoc />
    public int VocabularySize => _vocabulary.Count;

    /// <inheritdoc />
    public int IdOf(string token)
    {
        if (_vocabulary.TryGetValue(token, out var id)) return id;
        return _unknownToken is not null ? _vocabulary.GetValueOrDefault(_unknownToken, 0) : 0;
    }

    /// <inheritdoc />
    public string TokenOf(int id) => id >= 0 && id < _byId.Length ? _byId[id] : "";

    /// <inheritdoc />
    public IReadOnlyList<string> Tokenize(string word)
        => _cache.GetOrAdd(word, static (key, self) => self.Merge(key), this);

    private string[] Merge(string word)
    {
        if (word.Length == 0) return [];

        // Start from characters, honouring the two optional markers a trained model may carry.
        var symbols = new List<string>(word.Length);
        var elements = System.Globalization.StringInfo.GetTextElementEnumerator(word);
        while (elements.MoveNext()) symbols.Add((string)elements.Current);

        if (symbols.Count == 0) return [];

        if (_continuingPrefix.Length > 0)
        {
            for (var i = 1; i < symbols.Count; i++) symbols[i] = _continuingPrefix + symbols[i];
        }

        if (_endOfWordSuffix.Length > 0)
        {
            symbols[^1] += _endOfWordSuffix;
        }

        while (symbols.Count > 1)
        {
            var bestRank = int.MaxValue;
            var bestIndex = -1;

            for (var i = 0; i < symbols.Count - 1; i++)
            {
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0) break;

            symbols[bestIndex] += symbols[bestIndex + 1];
            symbols.RemoveAt(bestIndex + 1);
        }

        if (_unknownToken is not null)
        {
            for (var i = 0; i < symbols.Count; i++)
            {
                if (!_vocabulary.ContainsKey(symbols[i])) symbols[i] = _unknownToken;
            }
        }

        return [.. symbols];
    }
}

/// <summary>
/// SentencePiece-style Unigram segmentation: the highest-probability split under a unigram model,
/// found by Viterbi.
/// </summary>
/// <param name="pieces">Each piece with its log probability.</param>
/// <param name="unknownId">The id to fall back to for a character no piece covers.</param>
/// <remarks>
/// Unlike BPE and WordPiece this is a global optimum over the whole word rather than a greedy
/// walk, which is the point of the model: a locally attractive long piece that forces a bad split
/// afterwards loses to the better overall segmentation.
/// </remarks>
public sealed class UnigramModel(IReadOnlyList<(string Piece, double LogProbability)> pieces, int unknownId = 0)
    : ITokenizerModel
{
    private readonly Dictionary<string, int> _ids = BuildIds(pieces);
    private readonly Dictionary<string, double> _scores = BuildScores(pieces);
    private readonly string[] _byId = [.. pieces.Select(p => p.Piece)];
    private readonly int _longestPiece = pieces.Count == 0 ? 1 : pieces.Max(p => p.Piece.Length);
    private readonly ConcurrentDictionary<string, string[]> _cache = new(StringComparer.Ordinal);

    /// <inheritdoc />
    public int VocabularySize => _byId.Length;

    /// <inheritdoc />
    public int IdOf(string token) => _ids.TryGetValue(token, out var id) ? id : unknownId;

    /// <inheritdoc />
    public string TokenOf(int id) => id >= 0 && id < _byId.Length ? _byId[id] : "";

    /// <inheritdoc />
    public IReadOnlyList<string> Tokenize(string word)
        => _cache.GetOrAdd(word, static (key, self) => self.Viterbi(key), this);

    private string[] Viterbi(string word)
    {
        if (word.Length == 0) return [];

        var best = new double[word.Length + 1];
        var from = new int[word.Length + 1];
        Array.Fill(best, double.NegativeInfinity);
        best[0] = 0;

        for (var end = 1; end <= word.Length; end++)
        {
            var earliest = Math.Max(0, end - _longestPiece);

            for (var start = earliest; start < end; start++)
            {
                if (double.IsNegativeInfinity(best[start])) continue;

                var piece = word[start..end];

                // A single character with no piece still has to be crossable, or a word containing
                // one unknown character has no path at all and the whole segmentation fails.
                var score = _scores.TryGetValue(piece, out var known)
                    ? known
                    : end - start == 1 ? UnknownPenalty : double.NegativeInfinity;

                if (double.IsNegativeInfinity(score)) continue;

                var total = best[start] + score;
                if (total > best[end])
                {
                    best[end] = total;
                    from[end] = start;
                }
            }
        }

        if (double.IsNegativeInfinity(best[word.Length])) return [word];

        var result = new List<string>();
        for (var at = word.Length; at > 0; at = from[at]) result.Add(word[from[at]..at]);
        result.Reverse();
        return [.. result];
    }

    /// <summary>
    /// The score for a single character the vocabulary does not contain.
    /// </summary>
    /// <remarks>
    /// Low enough that any real piece wins, finite so that a path across the character still
    /// exists. Infinity here would make an unknown character unsegmentable rather than expensive.
    /// </remarks>
    private const double UnknownPenalty = -20.0;

    private static Dictionary<string, int> BuildIds(IReadOnlyList<(string Piece, double LogProbability)> pieces)
    {
        var ids = new Dictionary<string, int>(pieces.Count, StringComparer.Ordinal);
        for (var i = 0; i < pieces.Count; i++) ids.TryAdd(pieces[i].Piece, i);
        return ids;
    }

    private static Dictionary<string, double> BuildScores(IReadOnlyList<(string Piece, double LogProbability)> pieces)
    {
        var scores = new Dictionary<string, double>(pieces.Count, StringComparer.Ordinal);
        foreach (var (piece, logProbability) in pieces) scores.TryAdd(piece, logProbability);
        return scores;
    }
}

/// <summary>A special token the tokenizer inserts or recognises verbatim.</summary>
/// <param name="Id">Its vocabulary id.</param>
/// <param name="Content">Its literal text.</param>
/// <param name="Special">Whether it is marked special, and so masked out of model attention targets.</param>
public readonly record struct AddedToken(int Id, string Content, bool Special)
{
    /// <inheritdoc />
    public override string ToString() => $"{Content} (#{Id})";
}

/// <summary>
/// Inserts the special tokens a model expects around the encoded sequence.
/// </summary>
/// <param name="SingleTemplate">
/// The template for one sequence, using <c>$A</c> for its tokens, for example
/// <c>[CLS] $A [SEP]</c>.
/// </param>
/// <param name="PairTemplate">The template for two, using <c>$A</c> and <c>$B</c>.</param>
/// <param name="SpecialTokens">The literal tokens the templates refer to, with their ids.</param>
/// <remarks>
/// The template is stored rather than hardcoded because the convention differs per family: BERT
/// wraps with <c>[CLS]</c>/<c>[SEP]</c>, RoBERTa with <c>&lt;s&gt;</c>/<c>&lt;/s&gt;</c> and
/// doubles the separator between a pair, and GPT-2 adds nothing at all. Getting this wrong shifts
/// every position by one and degrades output without raising anything.
/// </remarks>
public sealed record PostProcessor(
    IReadOnlyList<string> SingleTemplate,
    IReadOnlyList<string> PairTemplate,
    IReadOnlyDictionary<string, int> SpecialTokens)
{
    /// <summary>A processor that adds nothing, which is what GPT-2 style models want.</summary>
    public static PostProcessor None => new([ "$A" ], [ "$A", "$B" ], new Dictionary<string, int>());

    /// <summary>The BERT convention.</summary>
    public static PostProcessor Bert(int clsId, int sepId) => new(
        ["[CLS]", "$A", "[SEP]"],
        ["[CLS]", "$A", "[SEP]", "$B", "[SEP]"],
        new Dictionary<string, int>(StringComparer.Ordinal) { ["[CLS]"] = clsId, ["[SEP]"] = sepId });

    /// <summary>Whether this processor inserts anything at all.</summary>
    public bool IsIdentity => SingleTemplate.Count == 1 && SingleTemplate[0] == "$A";
}
