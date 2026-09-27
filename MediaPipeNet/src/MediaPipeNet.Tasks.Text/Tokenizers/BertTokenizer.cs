using System.Globalization;
using System.Text;

namespace MediaPipeNet.Tasks.Text.Tokenizers;

/// <summary>
/// The BERT WordPiece tokenizer of MediaPipe's <c>BertPreprocessorCalculator</c>: ASCII lower-casing,
/// splitting on whitespace (dropped) and on punctuation and CJK ideographs (kept as tokens), then greedy
/// longest-match-first WordPiece with the <c>##</c> continuation prefix and <c>[UNK]</c> for words that
/// cannot be covered.
/// </summary>
public sealed class BertTokenizer
{
    private const int MaxBytesPerToken = 100;
    private readonly Dictionary<string, int> _vocab;

    /// <summary>Creates a tokenizer over a vocabulary (one token per line; the line number is the id).</summary>
    public BertTokenizer(IReadOnlyList<string> vocabulary)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);
        _vocab = new Dictionary<string, int>(vocabulary.Count, StringComparer.Ordinal);
        for (int i = 0; i < vocabulary.Count; i++) _vocab.TryAdd(vocabulary[i], i);
        ClassifierId = Id("[CLS]");
        SeparatorId = Id("[SEP]");
        UnknownId = Id("[UNK]");
        PadId = _vocab.GetValueOrDefault("[PAD]");
    }

    /// <summary>Id of <c>[CLS]</c>.</summary>
    public int ClassifierId { get; }

    /// <summary>Id of <c>[SEP]</c>.</summary>
    public int SeparatorId { get; }

    /// <summary>Id of <c>[UNK]</c>.</summary>
    public int UnknownId { get; }

    /// <summary>Id of <c>[PAD]</c>.</summary>
    public int PadId { get; }

    /// <summary>Vocabulary size.</summary>
    public int VocabularySize => _vocab.Count;

    /// <summary>Splits text into WordPiece sub-words.</summary>
    public List<string> Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var subwords = new List<string>();
        foreach (var word in SplitWords(LowerAscii(text))) WordPiece(word, subwords);
        return subwords;
    }

    /// <summary>
    /// Encodes text as model inputs of length <paramref name="maxSequenceLength"/>:
    /// <c>[CLS] tokens… [SEP] 0 0 …</c>, the attention mask (1 for real tokens) and all-zero segment ids.
    /// Tokens that do not fit are truncated.
    /// </summary>
    public (int[] InputIds, int[] InputMask, int[] SegmentIds) Encode(string text, int maxSequenceLength)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxSequenceLength, 2);
        var subwords = Tokenize(text);
        int n = Math.Min(subwords.Count + 2, maxSequenceLength);
        var ids = new int[maxSequenceLength];
        var mask = new int[maxSequenceLength];
        ids[0] = ClassifierId;
        for (int i = 1; i < n - 1; i++) ids[i] = _vocab.TryGetValue(subwords[i - 1], out var id) ? id : UnknownId;
        ids[n - 1] = SeparatorId;
        for (int i = n; i < maxSequenceLength; i++) ids[i] = PadId;
        for (int i = 0; i < n; i++) mask[i] = 1;
        return (ids, mask, new int[maxSequenceLength]);
    }

    private int Id(string token) =>
        _vocab.TryGetValue(token, out var id) ? id : throw new ArgumentException($"The vocabulary has no {token} token.");

    private static string LowerAscii(string s)
    {
        var chars = s.ToCharArray();
        for (int i = 0; i < chars.Length; i++) if (chars[i] is >= 'A' and <= 'Z') chars[i] = (char)(chars[i] + 32);
        return new string(chars);
    }

    /// <summary>Whitespace separates words; punctuation and CJK ideographs are words of their own.</summary>
    private static IEnumerable<string> SplitWords(string text)
    {
        var current = new StringBuilder();
        foreach (var rune in text.EnumerateRunes())
        {
            int v = rune.Value;
            if (v is ' ' or '\t' or '\n' or '\r' or '\f')
            {
                if (current.Length > 0) { yield return current.ToString(); current.Clear(); }
            }
            else if (IsSplitCharacter(rune))
            {
                if (current.Length > 0) { yield return current.ToString(); current.Clear(); }
                yield return rune.ToString();
            }
            else
            {
                current.Append(rune.ToString());
            }
        }
        if (current.Length > 0) yield return current.ToString();
    }

    private static bool IsSplitCharacter(Rune rune)
    {
        int v = rune.Value;
        if (v is >= '!' and <= '/' or >= ':' and <= '@' or >= '[' and <= '`' or >= '{' and <= '~') return true;
        var category = Rune.GetUnicodeCategory(rune);
        if (category is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation or UnicodeCategory.OpenPunctuation
            or UnicodeCategory.ClosePunctuation or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation
            or UnicodeCategory.OtherPunctuation) return true;
        return v is >= 0x4E00 and <= 0x9FFF or >= 0x3400 and <= 0x4DBF or >= 0x20000 and <= 0x2A6DF or >= 0x2A700 and <= 0x2B73F
            or >= 0x2B740 and <= 0x2B81F or >= 0x2B820 and <= 0x2CEAF or >= 0xF900 and <= 0xFAFF or >= 0x2F800 and <= 0x2FA1F;
    }

    /// <summary>Greedy longest-match-first WordPiece; the whole word becomes [UNK] when any part is not in the vocabulary.</summary>
    private void WordPiece(string word, List<string> output)
    {
        if (Encoding.UTF8.GetByteCount(word) > MaxBytesPerToken)
        {
            output.Add("[UNK]");
            return;
        }
        int mark = output.Count;
        int start = 0;
        while (start < word.Length)
        {
            string? found = null;
            int end = word.Length;
            while (end > start)
            {
                if (end < word.Length && char.IsLowSurrogate(word[end])) { end--; continue; }
                var piece = start == 0 ? word[start..end] : "##" + word[start..end];
                if (_vocab.ContainsKey(piece))
                {
                    found = piece;
                    break;
                }
                end--;
            }
            if (found is null)
            {
                output.RemoveRange(mark, output.Count - mark);
                output.Add("[UNK]");
                return;
            }
            output.Add(found);
            start = end;
        }
    }
}
