using System.Text;

namespace Gravicode.Science.GraviText.Tokenization;

/// <summary>
/// Byte-pair encoding: a sub-word vocabulary learned from a corpus by repeated merging.
/// </summary>
/// <remarks>
/// <para>
/// The scheme GPT uses, and the one <see cref="WordPieceTokenizer"/> is a variant of. It starts
/// from characters and repeatedly merges the most frequent adjacent pair, recording each merge in
/// order. Applying it later means replaying those merges: common words end up as single tokens
/// because their characters got merged early, and a word never seen in training still decomposes
/// into known pieces rather than collapsing to <c>[UNK]</c>.
/// </para>
/// <para>
/// The difference from WordPiece is which pair gets merged. BPE takes the most <em>frequent</em>
/// pair; WordPiece takes the pair that most increases the corpus likelihood, which is frequency
/// divided by the product of the parts' frequencies. In practice the vocabularies look similar and
/// BPE is simpler to train, which is why it is what gets trained here — the existing WordPiece
/// tokenizer applies a vocabulary it is given.
/// </para>
/// <para>
/// <b>The merge order is the model.</b> The vocabulary alone is not enough to tokenize with: the
/// same set of tokens applied in a different order produces a different segmentation. That is why
/// <see cref="Save"/> writes the ranked merges rather than a token list, and why loading a
/// vocabulary without its merges cannot reproduce the original encoding.
/// </para>
/// <para>
/// Words are trained and encoded with an end-of-word marker, so that <c>"est"</c> ending a word is
/// a different token from <c>"est"</c> inside one. Without it, the tokenizer learns merges that
/// span word boundaries and produces segmentations that do not survive re-spacing the text.
/// </para>
/// </remarks>
public sealed class BpeTokenizer : ITokenizer
{
    /// <summary>Appended to the last symbol of a word so word-final pieces stay distinct.</summary>
    public const string EndOfWord = "</w>";

    private readonly Dictionary<(string Left, string Right), int> _ranks = new();
    private readonly Dictionary<string, string[]> _cache = new(StringComparer.Ordinal);
    private readonly ITokenizer _preTokenizer;

    /// <summary>Creates a tokenizer from learned merges, most important first.</summary>
    /// <param name="merges">The merge rules in the order they were learned.</param>
    /// <param name="vocabulary">The token vocabulary, if one was built alongside.</param>
    /// <param name="preTokenizer">How text is split into words before merging. Defaults to a regex tokenizer.</param>
    public BpeTokenizer(IReadOnlyList<(string Left, string Right)> merges,
        Vocabulary? vocabulary = null, ITokenizer? preTokenizer = null)
    {
        ArgumentNullException.ThrowIfNull(merges);

        for (var i = 0; i < merges.Count; i++) _ranks.TryAdd(merges[i], i);

        Merges = merges;
        Vocabulary = vocabulary ?? BuildVocabulary(merges);
        _preTokenizer = preTokenizer ?? new RegexTokenizer();
    }

    /// <summary>The merge rules, in learned order.</summary>
    public IReadOnlyList<(string Left, string Right)> Merges { get; }

    /// <summary>The token vocabulary.</summary>
    public Vocabulary Vocabulary { get; }

    /// <summary>
    /// Learns merges from a corpus.
    /// </summary>
    /// <param name="corpus">Documents to learn from.</param>
    /// <param name="vocabularySize">
    /// Target vocabulary size, including the base characters and the special tokens. Training stops
    /// early if no pair occurs more than <paramref name="minFrequency"/> times.
    /// </param>
    /// <param name="minFrequency">A pair must occur at least this often to be worth a merge.</param>
    /// <param name="preTokenizer">How to split text into words. Defaults to a regex tokenizer.</param>
    /// <remarks>
    /// <para>
    /// Training works on <em>word types</em> with their counts, not on the raw token stream. A word
    /// appearing ten thousand times is one entry weighted by ten thousand, which is what makes this
    /// tractable on a real corpus — the alternative re-scans every occurrence on every merge, and
    /// there are thousands of merges.
    /// </para>
    /// <para>
    /// Ties are broken deterministically by the pair itself rather than by dictionary order. Without
    /// that, two runs over the same corpus can produce different merge orders and therefore
    /// different tokenizations, which is exactly the kind of irreproducibility that is invisible
    /// until a model trained on one is served with the other.
    /// </para>
    /// </remarks>
    public static BpeTokenizer Train(IEnumerable<string> corpus, int vocabularySize = 1000,
        int minFrequency = 2, ITokenizer? preTokenizer = null)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        if (vocabularySize < 1) throw new ArgumentOutOfRangeException(nameof(vocabularySize));

        preTokenizer ??= new RegexTokenizer();

        // Word types and their counts, each split into characters with the end-of-word marker.
        var words = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var document in corpus)
            foreach (var word in preTokenizer.Tokenize(document))
            {
                words.TryGetValue(word, out var count);
                words[word] = count + 1;
            }

        var symbols = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var alphabet = new SortedSet<string>(StringComparer.Ordinal);

        foreach (var word in words.Keys)
        {
            var pieces = Split(word);
            symbols[word] = pieces;
            foreach (var piece in pieces) alphabet.Add(piece);
        }

        var vocabulary = new Vocabulary();
        foreach (var symbol in alphabet) vocabulary.Add(symbol);

        var merges = new List<(string, string)>();
        var budget = vocabularySize - vocabulary.Count;

        for (var step = 0; step < budget; step++)
        {
            var counts = new Dictionary<(string, string), int>();

            foreach (var (word, count) in words)
            {
                var pieces = symbols[word];
                for (var i = 0; i + 1 < pieces.Count; i++)
                {
                    var pair = (pieces[i], pieces[i + 1]);
                    counts.TryGetValue(pair, out var existing);
                    counts[pair] = existing + count;
                }
            }

            if (counts.Count == 0) break;

            // Most frequent, ties broken by the pair itself so training is reproducible.
            var best = counts
                .OrderByDescending(kv => kv.Value)
                .ThenBy(kv => kv.Key.Item1, StringComparer.Ordinal)
                .ThenBy(kv => kv.Key.Item2, StringComparer.Ordinal)
                .First();

            if (best.Value < minFrequency) break;

            merges.Add(best.Key);
            var merged = best.Key.Item1 + best.Key.Item2;
            vocabulary.Add(merged);

            foreach (var word in words.Keys) symbols[word] = ApplyMerge(symbols[word], best.Key);
        }

        return new BpeTokenizer(merges, vocabulary, preTokenizer);
    }

    /// <inheritdoc />
    public IReadOnlyList<string> Tokenize(string text)
    {
        var result = new List<string>();
        foreach (var word in _preTokenizer.Tokenize(text)) result.AddRange(Encode(word));
        return result;
    }

    /// <summary>Splits one word into sub-word pieces by replaying the merges.</summary>
    /// <remarks>
    /// Merges are applied by rank, not left to right: at each step the whole word is scanned for
    /// the highest-ranked applicable pair. Applying them in positional order instead gives a
    /// different — and wrong — segmentation, because an early low-rank merge can consume a symbol a
    /// higher-rank merge needed.
    /// </remarks>
    public IReadOnlyList<string> Encode(string word)
    {
        if (word.Length == 0) return [];
        if (_cache.TryGetValue(word, out var cached)) return cached;

        var pieces = Split(word);

        while (pieces.Count > 1)
        {
            var bestRank = int.MaxValue;
            (string, string) best = default;

            for (var i = 0; i + 1 < pieces.Count; i++)
            {
                var pair = (pieces[i], pieces[i + 1]);
                if (_ranks.TryGetValue(pair, out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    best = pair;
                }
            }

            if (bestRank == int.MaxValue) break;
            pieces = ApplyMerge(pieces, best);
        }

        var result = pieces.ToArray();
        _cache[word] = result;
        return result;
    }

    /// <summary>Encodes text as vocabulary ids.</summary>
    public int[] EncodeIds(string text) => Vocabulary.Encode(Tokenize(text));

    /// <summary>Reassembles text from pieces, undoing the end-of-word markers.</summary>
    public string Decode(IReadOnlyList<string> pieces)
    {
        var builder = new StringBuilder();

        foreach (var piece in pieces)
        {
            if (piece.EndsWith(EndOfWord, StringComparison.Ordinal))
            {
                builder.Append(piece.AsSpan(0, piece.Length - EndOfWord.Length)).Append(' ');
                continue;
            }

            builder.Append(piece);
        }

        return builder.ToString().TrimEnd();
    }

    /// <summary>Reassembles text from vocabulary ids.</summary>
    public string DecodeIds(IReadOnlyList<int> ids)
        => Decode([.. ids.Select(id => Vocabulary[id])]);

    /// <summary>Writes the merges, one pair per line, in rank order.</summary>
    /// <remarks>
    /// The merges rather than the vocabulary, because the order is what defines the tokenization —
    /// the vocabulary can be rebuilt from them, and cannot substitute for them.
    /// </remarks>
    public void Save(string path)
        => File.WriteAllLines(path, Merges.Select(m => $"{m.Left} {m.Right}"));

    /// <summary>Reads merges written by <see cref="Save"/>.</summary>
    public static BpeTokenizer Load(string path, ITokenizer? preTokenizer = null)
    {
        var merges = new List<(string, string)>();

        foreach (var line in File.ReadLines(path))
        {
            if (line.Length == 0) continue;

            var space = line.IndexOf(' ');
            if (space <= 0) continue;

            merges.Add((line[..space], line[(space + 1)..]));
        }

        return new BpeTokenizer(merges, preTokenizer: preTokenizer);
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>Splits a word into its starting symbols, marking the last one as word-final.</summary>
    private static List<string> Split(string word)
    {
        var pieces = new List<string>(word.Length);

        // Enumerated as text elements rather than chars, so a surrogate pair or a combining
        // sequence stays one symbol instead of being torn in half.
        var enumerator = System.Globalization.StringInfo.GetTextElementEnumerator(word);
        while (enumerator.MoveNext()) pieces.Add((string)enumerator.Current);

        if (pieces.Count > 0) pieces[^1] += EndOfWord;
        return pieces;
    }

    private static List<string> ApplyMerge(List<string> pieces, (string Left, string Right) pair)
    {
        var result = new List<string>(pieces.Count);

        for (var i = 0; i < pieces.Count; i++)
        {
            if (i + 1 < pieces.Count && pieces[i] == pair.Left && pieces[i + 1] == pair.Right)
            {
                result.Add(pair.Left + pair.Right);
                i++;
                continue;
            }

            result.Add(pieces[i]);
        }

        return result;
    }

    /// <summary>Rebuilds a vocabulary from merges when one was not supplied.</summary>
    private static Vocabulary BuildVocabulary(IReadOnlyList<(string Left, string Right)> merges)
    {
        var vocabulary = new Vocabulary();

        // The parts have to come before the results, or an id refers to a token built from tokens
        // that do not exist yet.
        foreach (var (left, right) in merges)
        {
            vocabulary.Add(left);
            vocabulary.Add(right);
            vocabulary.Add(left + right);
        }

        return vocabulary;
    }
}
