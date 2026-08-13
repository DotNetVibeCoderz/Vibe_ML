using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Gravicode.Science.GraviText.Tokenization;

/// <summary>Splits text into tokens.</summary>
public interface ITokenizer
{
    /// <summary>Splits one document into tokens.</summary>
    IReadOnlyList<string> Tokenize(string text);
}

/// <summary>Splits on runs of whitespace. The cheapest useful tokenizer.</summary>
public sealed class WhitespaceTokenizer : ITokenizer
{
    /// <inheritdoc />
    public IReadOnlyList<string> Tokenize(string text)
        => text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
}

/// <summary>
/// Splits on a Unicode-aware word pattern, which keeps letters and digits and drops punctuation.
/// </summary>
public sealed partial class RegexTokenizer(string? pattern = null, bool lowercase = true) : ITokenizer
{
    private readonly Regex _regex = pattern is null ? DefaultPattern() : new Regex(pattern, RegexOptions.Compiled);

    [GeneratedRegex(@"[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*", RegexOptions.Compiled)]
    private static partial Regex DefaultPattern();

    /// <inheritdoc />
    public IReadOnlyList<string> Tokenize(string text)
    {
        var result = new List<string>();
        foreach (Match match in _regex.Matches(text))
            result.Add(lowercase ? match.Value.ToLowerInvariant() : match.Value);
        return result;
    }
}

/// <summary>Splits into individual characters, useful for character-level models.</summary>
public sealed class CharacterTokenizer(bool keepWhitespace = false) : ITokenizer
{
    /// <inheritdoc />
    public IReadOnlyList<string> Tokenize(string text)
        => text.Where(c => keepWhitespace || !char.IsWhiteSpace(c))
            .Select(c => c.ToString())
            .ToList();
}

/// <summary>Splits text into sentences on terminal punctuation.</summary>
public static partial class SentenceSplitter
{
    [GeneratedRegex(@"(?<=[.!?])\s+(?=[\p{Lu}\p{N}])", RegexOptions.Compiled)]
    private static partial Regex Boundary();

    /// <summary>Splits <paramref name="text"/> into trimmed, non-empty sentences.</summary>
    public static IReadOnlyList<string> Split(string text)
        => Boundary().Split(text)
            .Select(s => s.Trim())
            .Where(s => s.Length > 0)
            .ToList();
}

/// <summary>
/// A token-to-id mapping with the special tokens transformer models expect.
/// </summary>
public sealed class Vocabulary
{
    private readonly Dictionary<string, int> _tokenToId = new(StringComparer.Ordinal);
    private readonly List<string> _idToToken = [];

    /// <summary>Creates a vocabulary containing only the special tokens.</summary>
    public Vocabulary()
    {
        foreach (var token in new[] { PadToken, UnknownToken, ClassToken, SeparatorToken, MaskToken })
            Add(token);
    }

    /// <summary>Padding token.</summary>
    public const string PadToken = "[PAD]";

    /// <summary>Token substituted for anything out of vocabulary.</summary>
    public const string UnknownToken = "[UNK]";

    /// <summary>Sequence classification token.</summary>
    public const string ClassToken = "[CLS]";

    /// <summary>Segment separator.</summary>
    public const string SeparatorToken = "[SEP]";

    /// <summary>Masked-language-model placeholder.</summary>
    public const string MaskToken = "[MASK]";

    /// <summary>Number of distinct tokens.</summary>
    public int Count => _idToToken.Count;

    /// <summary>Id of the padding token.</summary>
    public int PadId => _tokenToId[PadToken];

    /// <summary>Id of the unknown token.</summary>
    public int UnknownId => _tokenToId[UnknownToken];

    /// <summary>Id of the classification token.</summary>
    public int ClassId => _tokenToId[ClassToken];

    /// <summary>Id of the separator token.</summary>
    public int SeparatorId => _tokenToId[SeparatorToken];

    /// <summary>Every token, indexed by id.</summary>
    public IReadOnlyList<string> Tokens => _idToToken;

    /// <summary>Adds a token if it is new and returns its id.</summary>
    public int Add(string token)
    {
        if (_tokenToId.TryGetValue(token, out var existing)) return existing;
        var id = _idToToken.Count;
        _tokenToId[token] = id;
        _idToToken.Add(token);
        return id;
    }

    /// <summary>Maps a token to its id, or to <see cref="UnknownId"/>.</summary>
    public int this[string token] => _tokenToId.TryGetValue(token, out var id) ? id : UnknownId;

    /// <summary>Maps an id back to its token.</summary>
    public string this[int id] => id >= 0 && id < _idToToken.Count ? _idToToken[id] : UnknownToken;

    /// <summary>True when the token is present.</summary>
    public bool Contains(string token) => _tokenToId.ContainsKey(token);

    /// <summary>
    /// Builds a vocabulary from a corpus, keeping tokens seen at least
    /// <paramref name="minFrequency"/> times, capped at <paramref name="maxSize"/> by frequency.
    /// </summary>
    public static Vocabulary Build(IEnumerable<IReadOnlyList<string>> documents,
        int minFrequency = 1, int maxSize = 0)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var document in documents)
            foreach (var token in document)
            {
                counts.TryGetValue(token, out var c);
                counts[token] = c + 1;
            }

        var vocabulary = new Vocabulary();
        var ordered = counts
            .Where(kv => kv.Value >= minFrequency)
            .OrderByDescending(kv => kv.Value)
            .ThenBy(kv => kv.Key, StringComparer.Ordinal);

        foreach (var (token, _) in maxSize > 0 ? ordered.Take(maxSize) : ordered)
            vocabulary.Add(token);

        return vocabulary;
    }

    /// <summary>Encodes tokens as ids.</summary>
    public int[] Encode(IReadOnlyList<string> tokens) => tokens.Select(t => this[t]).ToArray();

    /// <summary>Decodes ids back into tokens.</summary>
    public string Decode(IReadOnlyList<int> ids, bool skipSpecial = true)
    {
        var tokens = ids.Select(id => this[id]);
        if (skipSpecial)
            tokens = tokens.Where(t => t is not (PadToken or ClassToken or SeparatorToken));
        return string.Join(' ', tokens).Replace(" ##", "");
    }

    /// <summary>Writes the vocabulary, one token per line.</summary>
    public void Save(string path) => File.WriteAllLines(path, _idToToken);

    /// <summary>Reads a vocabulary written by <see cref="Save"/>.</summary>
    public static Vocabulary Load(string path)
    {
        var vocabulary = new Vocabulary();
        vocabulary._tokenToId.Clear();
        vocabulary._idToToken.Clear();
        foreach (var line in File.ReadLines(path)) vocabulary.Add(line);
        return vocabulary;
    }
}

/// <summary>
/// WordPiece sub-word tokenization: the scheme BERT uses.
/// </summary>
/// <remarks>
/// A word is matched greedily against the vocabulary from the longest prefix down, and the
/// remainder is re-matched with a <c>##</c> continuation marker. This is what lets a fixed
/// 30k vocabulary cover an open-ended language: an unseen word decomposes into known pieces
/// rather than collapsing to <c>[UNK]</c>.
/// </remarks>
public sealed class WordPieceTokenizer(Vocabulary vocabulary, int maxCharsPerWord = 100) : ITokenizer
{
    private readonly ITokenizer _preTokenizer = new RegexTokenizer();

    /// <summary>The sub-word vocabulary.</summary>
    public Vocabulary Vocabulary { get; } = vocabulary;

    /// <inheritdoc />
    public IReadOnlyList<string> Tokenize(string text)
    {
        var result = new List<string>();
        foreach (var word in _preTokenizer.Tokenize(text)) result.AddRange(TokenizeWord(word));
        return result;
    }

    private IEnumerable<string> TokenizeWord(string word)
    {
        if (word.Length > maxCharsPerWord) return [Vocabulary.UnknownToken];

        var pieces = new List<string>();
        var start = 0;
        while (start < word.Length)
        {
            // Longest match wins; failing that, the whole word becomes unknown.
            var end = word.Length;
            string? found = null;
            while (start < end)
            {
                var candidate = start == 0 ? word[start..end] : "##" + word[start..end];
                if (Vocabulary.Contains(candidate)) { found = candidate; break; }
                end--;
            }

            if (found is null) return [Vocabulary.UnknownToken];
            pieces.Add(found);
            start = end;
        }
        return pieces;
    }

    /// <summary>Encodes text with <c>[CLS]</c> and <c>[SEP]</c>, padded or truncated to a fixed length.</summary>
    public (int[] Ids, int[] AttentionMask) Encode(string text, int maxLength = 128)
    {
        var tokens = Tokenize(text);
        var ids = new List<int> { Vocabulary.ClassId };
        ids.AddRange(tokens.Take(maxLength - 2).Select(t => Vocabulary[t]));
        ids.Add(Vocabulary.SeparatorId);

        var mask = new int[maxLength];
        var padded = new int[maxLength];
        for (var i = 0; i < maxLength; i++)
        {
            padded[i] = i < ids.Count ? ids[i] : Vocabulary.PadId;
            mask[i] = i < ids.Count ? 1 : 0;
        }
        return (padded, mask);
    }

    /// <summary>
    /// Learns a WordPiece vocabulary from a corpus by starting from characters and repeatedly
    /// merging the most frequent adjacent pair.
    /// </summary>
    public static Vocabulary Train(IEnumerable<string> corpus, int vocabularySize = 5000, int minFrequency = 2)
    {
        var pre = new RegexTokenizer();
        var wordCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var document in corpus)
            foreach (var word in pre.Tokenize(document))
            {
                wordCounts.TryGetValue(word, out var c);
                wordCounts[word] = c + 1;
            }

        // Each word starts as its characters, continuations marked with ##.
        var splits = wordCounts.Keys.ToDictionary(
            w => w,
            w => new List<string>(w.Select((c, i) => i == 0 ? c.ToString() : "##" + c)),
            StringComparer.Ordinal);

        var vocabulary = new Vocabulary();
        foreach (var pieces in splits.Values)
            foreach (var piece in pieces) vocabulary.Add(piece);

        while (vocabulary.Count < vocabularySize)
        {
            var pairCounts = new Dictionary<(string, string), int>();
            foreach (var (word, pieces) in splits)
            {
                var frequency = wordCounts[word];
                for (var i = 0; i + 1 < pieces.Count; i++)
                {
                    var key = (pieces[i], pieces[i + 1]);
                    pairCounts.TryGetValue(key, out var c);
                    pairCounts[key] = c + frequency;
                }
            }

            if (pairCounts.Count == 0) break;
            var (best, count) = pairCounts.MaxBy(kv => kv.Value);
            if (count < minFrequency) break;

            var merged = best.Item1 + best.Item2.Replace("##", "");
            vocabulary.Add(merged);

            foreach (var (word, pieces) in splits)
            {
                for (var i = 0; i + 1 < pieces.Count; i++)
                {
                    if (pieces[i] != best.Item1 || pieces[i + 1] != best.Item2) continue;
                    pieces[i] = merged;
                    pieces.RemoveAt(i + 1);
                }
                _ = word;
            }
        }

        return vocabulary;
    }
}

/// <summary>Case folding, accent stripping and punctuation handling.</summary>
public static class TextNormalizer
{
    /// <summary>Lowercases, strips accents and collapses whitespace.</summary>
    public static string Normalize(string text, bool lowercase = true, bool stripAccents = true,
        bool removePunctuation = false)
    {
        var value = text;
        if (lowercase) value = value.ToLowerInvariant();
        if (stripAccents) value = StripAccents(value);
        if (removePunctuation) value = RemovePunctuation(value);
        return CollapseWhitespace(value);
    }

    // Latin-1 Supplement and Latin Extended-A folded to ASCII. An explicit table is used rather
    // than FormD normalisation because the libraries build with InvariantGlobalization, where
    // Unicode normalisation is unavailable and silently returns the string unchanged.
    private static readonly Dictionary<char, string> AccentFolding = BuildFoldingTable();

    private static Dictionary<char, string> BuildFoldingTable()
    {
        var table = new Dictionary<char, string>();

        void Map(string accented, string plain)
        {
            foreach (var c in accented) table[c] = plain;
        }

        Map("ÀÁÂÃÄÅĀĂĄ", "A"); Map("àáâãäåāăą", "a");
        Map("ÇĆĈĊČ", "C"); Map("çćĉċč", "c");
        Map("ÐĎĐ", "D"); Map("ðďđ", "d");
        Map("ÈÉÊËĒĔĖĘĚ", "E"); Map("èéêëēĕėęě", "e");
        Map("ĜĞĠĢ", "G"); Map("ĝğġģ", "g");
        Map("ĤĦ", "H"); Map("ĥħ", "h");
        Map("ÌÍÎÏĨĪĬĮİ", "I"); Map("ìíîïĩīĭįı", "i");
        Map("Ĵ", "J"); Map("ĵ", "j");
        Map("Ķ", "K"); Map("ķ", "k");
        Map("ĹĻĽĿŁ", "L"); Map("ĺļľŀł", "l");
        Map("ÑŃŅŇ", "N"); Map("ñńņňŉ", "n");
        Map("ÒÓÔÕÖØŌŎŐ", "O"); Map("òóôõöøōŏő", "o");
        Map("ŔŖŘ", "R"); Map("ŕŗř", "r");
        Map("ŚŜŞŠ", "S"); Map("śŝşš", "s");
        Map("ŢŤŦ", "T"); Map("ţťŧ", "t");
        Map("ÙÚÛÜŨŪŬŮŰŲ", "U"); Map("ùúûüũūŭůűų", "u");
        Map("Ŵ", "W"); Map("ŵ", "w");
        Map("ÝŶŸ", "Y"); Map("ýÿŷ", "y");
        Map("ŹŻŽ", "Z"); Map("źżž", "z");

        table['Æ'] = "AE"; table['æ'] = "ae";
        table['Œ'] = "OE"; table['œ'] = "oe";
        table['ß'] = "ss";
        table['Þ'] = "TH"; table['þ'] = "th";

        return table;
    }

    /// <summary>Folds accented Latin characters onto their unaccented ASCII equivalents.</summary>
    public static string StripAccents(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var c in text)
            sb.Append(AccentFolding.TryGetValue(c, out var folded) ? folded : c.ToString());
        return sb.ToString();
    }

    /// <summary>Replaces punctuation with spaces.</summary>
    public static string RemovePunctuation(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var c in text) sb.Append(char.IsPunctuation(c) || char.IsSymbol(c) ? ' ' : c);
        return sb.ToString();
    }

    /// <summary>Collapses runs of whitespace into single spaces and trims.</summary>
    public static string CollapseWhitespace(string text)
    {
        var sb = new StringBuilder(text.Length);
        var previousWasSpace = false;
        foreach (var c in text)
        {
            var isSpace = char.IsWhiteSpace(c);
            if (isSpace && previousWasSpace) continue;
            sb.Append(isSpace ? ' ' : c);
            previousWasSpace = isSpace;
        }
        return sb.ToString().Trim();
    }
}
