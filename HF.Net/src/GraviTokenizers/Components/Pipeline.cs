using System.Text;
using System.Text.RegularExpressions;

namespace Gravicode.HFNet.GraviTokenizers.Components;

/// <summary>A word carved out of the input, with the range of the original text it covers.</summary>
/// <param name="Word">The text of the piece, before normalization.</param>
/// <param name="Start">Index of the first character in the source text.</param>
/// <param name="End">Index one past the last character in the source text.</param>
public readonly record struct PreToken(string Word, int Start, int End);

/// <summary>Cleans up text before it is looked up in a vocabulary.</summary>
/// <remarks>
/// Normalizers here run <b>per pre-token</b>, not over the whole string. Accent folding and case
/// folding can both change length, and running them over the whole input desynchronises every
/// offset after the first affected character. Applying them inside a word keeps the word's own
/// start and end - taken from the untouched source - correct.
/// </remarks>
public interface INormalizer
{
    /// <summary>Normalizes one piece of text.</summary>
    string Normalize(string text);
}

/// <summary>Splits text into the units the model's vocabulary is defined over.</summary>
public interface IPreTokenizer
{
    /// <summary>Splits text, carrying each piece's range in the original string.</summary>
    IReadOnlyList<PreToken> Split(string text);
}

/// <summary>Turns one word into vocabulary pieces.</summary>
public interface ITokenizerModel
{
    /// <summary>Segments a word into pieces that are present in the vocabulary.</summary>
    IReadOnlyList<string> Tokenize(string word);

    /// <summary>The id for a piece, or the unknown id when it is absent.</summary>
    int IdOf(string token);

    /// <summary>The piece for an id, or an empty string when the id is out of range.</summary>
    string TokenOf(int id);

    /// <summary>Number of entries in the vocabulary.</summary>
    int VocabularySize { get; }
}

/// <summary>Reassembles decoded pieces back into text.</summary>
public interface IDecoder
{
    /// <summary>Joins pieces into a string.</summary>
    string Decode(IReadOnlyList<string> tokens);
}

// ====================================================================== normalizers

/// <summary>Applies several normalizers in order.</summary>
/// <param name="Steps">The normalizers to run, first to last.</param>
public sealed record SequenceNormalizer(IReadOnlyList<INormalizer> Steps) : INormalizer
{
    /// <inheritdoc />
    public string Normalize(string text)
    {
        foreach (var step in Steps) text = step.Normalize(text);
        return text;
    }
}

/// <summary>Lowercases text.</summary>
public sealed class LowercaseNormalizer : INormalizer
{
    /// <inheritdoc />
    public string Normalize(string text) => text.ToLowerInvariant();
}

/// <summary>Trims leading and trailing whitespace.</summary>
/// <param name="Left">Whether to trim the start.</param>
/// <param name="Right">Whether to trim the end.</param>
public sealed record StripNormalizer(bool Left = true, bool Right = true) : INormalizer
{
    /// <inheritdoc />
    public string Normalize(string text)
    {
        if (Left && Right) return text.Trim();
        return Left ? text.TrimStart() : Right ? text.TrimEnd() : text;
    }
}

/// <summary>
/// SentencePiece's precompiled normalizer: the <c>precompiled_charsmap</c> a <c>Precompiled</c>
/// entry in <c>tokenizer.json</c> carries, read as the lookup table it is.
/// </summary>
/// <remarks>
/// <para>
/// The map is SentencePiece's NFKC-with-extras, compiled into a double-array trie from input bytes
/// to normalized strings. Reading it directly gives exactly the normalization the model was trained
/// with, on every platform: .NET's own <c>string.Normalize</c> is a no-op under invariant
/// globalization, which this stack builds with, and skipping the step turned characters such as
/// <c>™</c> into unknown tokens on XLM-RoBERTa and T5.
/// </para>
/// <para>
/// The layout is darts-clone's, as SentencePiece writes it and the Rust <c>tokenizers</c> crate reads
/// it: a little-endian <c>uint32</c> byte count, that many bytes of trie units, then the
/// NUL-separated replacement strings. Each grapheme shorter than six bytes is looked up whole; a
/// longer one, or one with no entry, character by character.
/// </para>
/// </remarks>
public sealed class PrecompiledNormalizer : INormalizer
{
    private readonly uint[] _trie;
    private readonly byte[] _normalized;

    /// <summary>Reads a base64 <c>precompiled_charsmap</c>.</summary>
    public PrecompiledNormalizer(string base64CharsMap)
    {
        var bytes = Convert.FromBase64String(base64CharsMap);
        if (bytes.Length < 4) throw new InvalidDataException("The precompiled_charsmap is too short to hold a trie.");

        var trieBytes = (int)BitConverter.ToUInt32(bytes, 0);
        if (trieBytes < 0 || 4 + trieBytes > bytes.Length || trieBytes % 4 != 0)
        {
            throw new InvalidDataException("The precompiled_charsmap declares a trie larger than itself.");
        }

        _trie = new uint[trieBytes / 4];
        Buffer.BlockCopy(bytes, 4, _trie, 0, trieBytes);
        _normalized = bytes[(4 + trieBytes)..];
    }

    /// <inheritdoc />
    public string Normalize(string text)
    {
        if (text.Length == 0 || _trie.Length == 0) return text;

        var result = new System.Text.StringBuilder(text.Length);
        var elements = System.Globalization.StringInfo.GetTextElementEnumerator(text);

        while (elements.MoveNext())
        {
            var grapheme = (string)elements.Current;

            if (System.Text.Encoding.UTF8.GetByteCount(grapheme) < 6 && Transform(grapheme) is { } whole)
            {
                result.Append(whole);
                continue;
            }

            foreach (var rune in grapheme.EnumerateRunes())
            {
                var part = rune.ToString();
                result.Append(Transform(part) ?? part);
            }
        }

        return result.ToString();
    }

    /// <summary>The replacement for <paramref name="chunk"/>, from its shortest prefix in the trie.</summary>
    private string? Transform(string chunk)
    {
        var key = System.Text.Encoding.UTF8.GetBytes(chunk);

        // darts-clone common prefix search; the first hit is the one the reference uses.
        uint position = 0;
        var unit = _trie[position];
        position ^= Offset(unit);

        foreach (var c in key)
        {
            if (c == 0) break;

            position ^= c;
            if (position >= _trie.Length) return null;

            unit = _trie[position];
            if (Label(unit) != c) return null;

            position ^= Offset(unit);
            if (HasLeaf(unit))
            {
                if (position >= _trie.Length) return null;
                var index = (int)Value(_trie[position]);
                return ReadString(index);
            }
        }

        return null;
    }

    private string ReadString(int index)
    {
        if (index < 0 || index >= _normalized.Length) return "";

        var end = index;
        while (end < _normalized.Length && _normalized[end] != 0) end++;
        return System.Text.Encoding.UTF8.GetString(_normalized, index, end - index);
    }

    private static bool HasLeaf(uint unit) => ((unit >> 8) & 1) == 1;

    private static uint Value(uint unit) => unit & ((1u << 31) - 1);

    private static uint Label(uint unit) => unit & ((1u << 31) | 0xFF);

    private static uint Offset(uint unit) => (unit >> 10) << (int)((unit & (1u << 9)) >> 6);
}

/// <summary>Replaces every occurrence of a pattern.</summary>
/// <param name="Pattern">The text or regular expression to look for.</param>
/// <param name="Replacement">What to put in its place.</param>
/// <param name="IsRegex">Whether <paramref name="Pattern"/> is a regular expression.</param>
public sealed record ReplaceNormalizer(string Pattern, string Replacement, bool IsRegex = false) : INormalizer
{
    /// <inheritdoc />
    public string Normalize(string text)
        => IsRegex ? Regex.Replace(text, Pattern, Replacement) : text.Replace(Pattern, Replacement);
}

/// <summary>
/// The normalizer BERT-family tokenizers use: optional case folding and accent stripping.
/// </summary>
/// <param name="Lowercase">Whether to lowercase.</param>
/// <param name="StripAccents">
/// Whether to fold accented characters to their base letter. When null it follows
/// <paramref name="Lowercase"/>, which is what the reference implementation does.
/// </param>
/// <remarks>
/// Accent stripping goes through an explicit folding table rather than Unicode normalization. The
/// libraries here build with <c>InvariantGlobalization</c>, under which <c>String.Normalize</c>
/// silently returns its input unchanged - so a decomposition-based implementation would look
/// correct, compile, and quietly do nothing.
/// </remarks>
public sealed record BertNormalizer(bool Lowercase = true, bool? StripAccents = null) : INormalizer
{
    /// <inheritdoc />
    public string Normalize(string text)
    {
        if (Lowercase) text = text.ToLowerInvariant();
        return (StripAccents ?? Lowercase) ? AccentFolding.Strip(text) : text;
    }
}

/// <summary>An accent-folding table, used because <c>String.Normalize</c> is a no-op here.</summary>
internal static class AccentFolding
{
    private const string Accented =
        "àáâãäåāăąèéêëēĕėęěìíîïĩīĭįıòóôõöøōŏőùúûüũūŭůűųñńņňçćĉċčÿýŷžźżšśŝşğĝģĥħďđťţŀłŕŗřßæœð";

    private static readonly string[] Replacements =
    [
        "a", "a", "a", "a", "a", "a", "a", "a", "a",
        "e", "e", "e", "e", "e", "e", "e", "e", "e",
        "i", "i", "i", "i", "i", "i", "i", "i", "i",
        "o", "o", "o", "o", "o", "o", "o", "o", "o",
        "u", "u", "u", "u", "u", "u", "u", "u", "u", "u",
        "n", "n", "n", "n",
        "c", "c", "c", "c", "c",
        "y", "y", "y",
        "z", "z", "z",
        "s", "s", "s", "s",
        "g", "g", "g",
        "h", "h",
        "d", "d",
        "t", "t",
        "l", "l",
        "r", "r", "r",
        "ss", "ae", "oe", "d",
    ];

    /// <summary>Folds accented Latin characters to their unaccented form.</summary>
    internal static string Strip(string text)
    {
        var builder = new StringBuilder(text.Length);
        var upperCaseShift = false;

        foreach (var character in text)
        {
            var lower = char.ToLowerInvariant(character);
            upperCaseShift = character != lower;

            var index = Accented.IndexOf(lower);
            if (index < 0)
            {
                builder.Append(character);
                continue;
            }

            var replacement = Replacements[index];
            builder.Append(upperCaseShift ? replacement.ToUpperInvariant() : replacement);
        }

        return builder.ToString();
    }
}

// ====================================================================== pre-tokenizers

/// <summary>Applies several pre-tokenizers in order, each splitting the previous one's output.</summary>
/// <param name="Steps">The pre-tokenizers to run, first to last.</param>
public sealed record SequencePreTokenizer(IReadOnlyList<IPreTokenizer> Steps) : IPreTokenizer
{
    /// <inheritdoc />
    public IReadOnlyList<PreToken> Split(string text)
    {
        IReadOnlyList<PreToken> current = [new PreToken(text, 0, text.Length)];

        foreach (var step in Steps)
        {
            var next = new List<PreToken>(current.Count);
            foreach (var token in current)
            {
                // Each step re-splits a piece, so its offsets are relative to that piece and have
                // to be rebased onto the original text before the next step sees them.
                foreach (var part in step.Split(token.Word))
                {
                    next.Add(new PreToken(part.Word, token.Start + part.Start, token.Start + part.End));
                }
            }
            current = next;
        }

        return current;
    }
}

/// <summary>Splits on whitespace, keeping each run of non-whitespace as one piece.</summary>
public sealed class WhitespacePreTokenizer : IPreTokenizer
{
    /// <inheritdoc />
    public IReadOnlyList<PreToken> Split(string text)
    {
        var tokens = new List<PreToken>();
        var start = -1;

        for (var i = 0; i < text.Length; i++)
        {
            if (char.IsWhiteSpace(text[i]))
            {
                if (start >= 0) tokens.Add(new PreToken(text[start..i], start, i));
                start = -1;
            }
            else if (start < 0)
            {
                start = i;
            }
        }

        if (start >= 0) tokens.Add(new PreToken(text[start..], start, text.Length));
        return tokens;
    }
}

/// <summary>Splits on whitespace and then splits punctuation into its own pieces.</summary>
/// <remarks>This is what BERT-family tokenizers use before WordPiece.</remarks>
public sealed class BertPreTokenizer : IPreTokenizer
{
    /// <inheritdoc />
    public IReadOnlyList<PreToken> Split(string text)
    {
        var tokens = new List<PreToken>();
        var start = -1;

        for (var i = 0; i < text.Length; i++)
        {
            var character = text[i];

            if (char.IsWhiteSpace(character))
            {
                if (start >= 0) tokens.Add(new PreToken(text[start..i], start, i));
                start = -1;
            }
            else if (IsPunctuation(character) || IsCjk(text, i, out _))
            {
                // Punctuation and CJK ideographs are pieces of their own. BERT's
                // handle_chinese_chars does this for CJK: without it a run of ideographs is one
                // "word", longer than any vocabulary entry, and the whole run becomes [UNK].
                var width = IsCjk(text, i, out var length) ? length : 1;
                if (start >= 0) tokens.Add(new PreToken(text[start..i], start, i));
                tokens.Add(new PreToken(text.Substring(i, width), i, i + width));
                i += width - 1;
                start = -1;
            }
            else if (start < 0)
            {
                start = i;
            }
        }

        if (start >= 0) tokens.Add(new PreToken(text[start..], start, text.Length));
        return tokens;
    }

    /// <summary>
    /// Whether a CJK ideograph starts at <paramref name="index"/>, and how many UTF-16 units it takes.
    /// </summary>
    /// <remarks>
    /// The ranges are Transformers' <c>_is_chinese_char</c>: the CJK Unified Ideographs and their
    /// extensions, and the compatibility ideographs. Hiragana, katakana and Hangul are not in them,
    /// and the reference does not isolate them either.
    /// </remarks>
    internal static bool IsCjk(string text, int index, out int length)
    {
        length = 1;
        int code = text[index];

        if (char.IsHighSurrogate(text[index]) && index + 1 < text.Length && char.IsLowSurrogate(text[index + 1]))
        {
            code = char.ConvertToUtf32(text[index], text[index + 1]);
            length = 2;
        }

        return code is (>= 0x4E00 and <= 0x9FFF) or (>= 0x3400 and <= 0x4DBF) or (>= 0x20000 and <= 0x2A6DF)
            or (>= 0x2A700 and <= 0x2B73F) or (>= 0x2B740 and <= 0x2B81F) or (>= 0x2B820 and <= 0x2CEAF)
            or (>= 0xF900 and <= 0xFAFF) or (>= 0x2F800 and <= 0x2FA1F);
    }

    /// <summary>
    /// Whether a character counts as punctuation for BERT, which is broader than
    /// <see cref="char.IsPunctuation(char)"/>.
    /// </summary>
    /// <remarks>
    /// The reference implementation treats every ASCII non-alphanumeric as punctuation, so
    /// <c>$</c>, <c>+</c> and <c>^</c> split even though Unicode files them as symbols. Using the
    /// Unicode category alone leaves those attached to the following word and produces tokens that
    /// are not in the vocabulary.
    /// </remarks>
    internal static bool IsPunctuation(char character)
    {
        if (character is (>= '!' and <= '/') or (>= ':' and <= '@') or (>= '[' and <= '`') or (>= '{' and <= '~'))
        {
            return true;
        }

        return char.IsPunctuation(character);
    }
}

/// <summary>Splits punctuation characters into their own pieces.</summary>
public sealed class PunctuationPreTokenizer : IPreTokenizer
{
    /// <inheritdoc />
    public IReadOnlyList<PreToken> Split(string text)
    {
        var tokens = new List<PreToken>();
        var start = 0;

        for (var i = 0; i < text.Length; i++)
        {
            if (!BertPreTokenizer.IsPunctuation(text[i])) continue;

            if (i > start) tokens.Add(new PreToken(text[start..i], start, i));
            tokens.Add(new PreToken(text[i].ToString(), i, i + 1));
            start = i + 1;
        }

        if (start < text.Length) tokens.Add(new PreToken(text[start..], start, text.Length));
        return tokens;
    }
}

/// <summary>What a split does with the text its pattern marks as a delimiter.</summary>
/// <remarks>The five behaviours of the Rust <c>tokenizers</c> crate, by the same names.</remarks>
public enum SplitBehavior
{
    /// <summary>Delimiters are dropped.</summary>
    Removed,

    /// <summary>Each delimiter becomes a piece of its own.</summary>
    Isolated,

    /// <summary>Each delimiter is appended to the piece before it.</summary>
    MergedWithPrevious,

    /// <summary>Each delimiter is prepended to the piece after it.</summary>
    MergedWithNext,

    /// <summary>Runs of consecutive delimiters become one piece.</summary>
    Contiguous,
}

/// <summary>Splits on a regular expression.</summary>
/// <param name="Pattern">The expression.</param>
/// <param name="Invert">
/// When false the matches are the delimiters; when true everything <i>between</i> matches is, so
/// with <see cref="SplitBehavior.Removed"/> the matches are the pieces kept.
/// </param>
/// <param name="Behavior">What becomes of the delimiters.</param>
/// <remarks>
/// <c>invert</c> and <c>behavior</c> are independent in <c>tokenizer.json</c>, and reading only one
/// of them is a quiet disaster: CLIP's pre-tokenizer is <c>Removed</c> with <c>invert: true</c> -
/// keep the words - and reading it as <c>Removed</c> alone drops every word and keeps the spaces.
/// </remarks>
public sealed partial record SplitPreTokenizer(string Pattern, bool Invert = false, SplitBehavior Behavior = SplitBehavior.Removed)
    : IPreTokenizer
{
    private Regex? _compiled;

    /// <inheritdoc />
    public IReadOnlyList<PreToken> Split(string text)
    {
        _compiled ??= new Regex(Pattern, RegexOptions.Compiled);

        // The text as alternating spans, each marked delimiter or not.
        var spans = new List<(int Start, int End, bool Delimiter)>();
        var cursor = 0;
        foreach (Match match in _compiled.Matches(text))
        {
            if (match.Length == 0) continue;
            if (match.Index > cursor) spans.Add((cursor, match.Index, Invert));
            spans.Add((match.Index, match.Index + match.Length, !Invert));
            cursor = match.Index + match.Length;
        }

        if (cursor < text.Length) spans.Add((cursor, text.Length, Invert));

        // Each behaviour below is the Rust tokenizers crate's fold, including its rule that a
        // delimiter only merges when the span before it (or after it) was not a delimiter too:
        // "the-final--countdown" split on "-" merged with the previous piece is
        // ["the-", "final-", "-", "countdown"], not ["the-", "final--", "countdown"].
        var pieces = new List<(int Start, int End)>();

        switch (Behavior)
        {
            case SplitBehavior.Removed:
                pieces.AddRange(spans.Where(s => !s.Delimiter).Select(s => (s.Start, s.End)));
                break;

            case SplitBehavior.Isolated:
                pieces.AddRange(spans.Select(s => (s.Start, s.End)));
                break;

            case SplitBehavior.Contiguous:
            {
                var previousDelimiter = false;
                foreach (var (start, end, delimiter) in spans)
                {
                    if (delimiter && previousDelimiter) pieces[^1] = (pieces[^1].Start, end);
                    else pieces.Add((start, end));
                    previousDelimiter = delimiter;
                }

                break;
            }

            case SplitBehavior.MergedWithPrevious:
            {
                var previousDelimiter = false;
                foreach (var (start, end, delimiter) in spans)
                {
                    if (delimiter && !previousDelimiter && pieces.Count > 0) pieces[^1] = (pieces[^1].Start, end);
                    else pieces.Add((start, end));
                    previousDelimiter = delimiter;
                }

                break;
            }

            case SplitBehavior.MergedWithNext:
            {
                var previousDelimiter = false;
                for (var i = spans.Count - 1; i >= 0; i--)
                {
                    var (start, end, delimiter) = spans[i];
                    if (delimiter && !previousDelimiter && pieces.Count > 0) pieces[^1] = (start, pieces[^1].End);
                    else pieces.Add((start, end));
                    previousDelimiter = delimiter;
                }

                pieces.Reverse();
                break;
            }
        }

        return [.. pieces.Select(p => new PreToken(text[p.Start..p.End], p.Start, p.End))];
    }
}

/// <summary>
/// The GPT-2 byte-level pre-tokenizer: split on a fixed regex, then re-encode each byte as a
/// printable character.
/// </summary>
/// <param name="AddPrefixSpace">Whether a leading space is added so the first word looks interior.</param>
/// <remarks>
/// <para>
/// The byte mapping is what makes byte-level BPE lossless: every one of the 256 byte values gets a
/// distinct printable character, so any input - emoji, invalid UTF-8, control characters - is
/// representable and round-trips exactly. A vocabulary built over raw text cannot say that.
/// </para>
/// <para>
/// The offsets reported here are in characters of the source string, while the tokens are in the
/// mapped alphabet. They are deliberately not the same thing: the mapped form has no position in
/// the original text, and reporting its indices would make every span wrong for non-ASCII input.
/// </para>
/// </remarks>
public sealed partial record ByteLevelPreTokenizer(bool AddPrefixSpace = true) : IPreTokenizer
{
    /// <summary>The GPT-2 splitting expression, which keeps a leading space with its word.</summary>
    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled)]
    private static partial Regex Splitter();

    /// <inheritdoc />
    public IReadOnlyList<PreToken> Split(string text)
    {
        var source = AddPrefixSpace && text.Length > 0 && text[0] != ' ' ? " " + text : text;
        var shift = source.Length - text.Length;

        var tokens = new List<PreToken>();
        foreach (Match match in Splitter().Matches(source))
        {
            if (match.Length == 0) continue;

            var start = Math.Max(0, match.Index - shift);
            var end = Math.Max(start, match.Index + match.Length - shift);
            tokens.Add(new PreToken(ByteAlphabet.Encode(match.Value), start, end));
        }

        return tokens;
    }
}

/// <summary>The GPT-2 byte-to-character alphabet, and its inverse.</summary>
public static class ByteAlphabet
{
    private static readonly char[] ByteToChar = new char[256];
    private static readonly Dictionary<char, byte> CharToByte = [];

    static ByteAlphabet()
    {
        // The printable ASCII range plus two Latin-1 runs map to themselves; everything left over
        // is pushed into the private-use area above 256 so that no byte maps to whitespace or to a
        // control character, either of which would be destroyed by a later normalization step.
        var assigned = new List<int>();
        for (var b = '!'; b <= '~'; b++) assigned.Add(b);
        for (var b = '¡'; b <= '¬'; b++) assigned.Add(b);
        for (var b = '®'; b <= 'ÿ'; b++) assigned.Add(b);

        var mapped = new List<int>(assigned);
        var extra = 0;

        for (var b = 0; b < 256; b++)
        {
            if (assigned.Contains(b)) continue;
            assigned.Add(b);
            mapped.Add(256 + extra);
            extra++;
        }

        for (var i = 0; i < assigned.Count; i++)
        {
            ByteToChar[assigned[i]] = (char)mapped[i];
            CharToByte[(char)mapped[i]] = (byte)assigned[i];
        }
    }

    /// <summary>Re-encodes text as one character per UTF-8 byte.</summary>
    public static string Encode(string text)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(text);
        var builder = new StringBuilder(bytes.Length);
        foreach (var b in bytes) builder.Append(ByteToChar[b]);
        return builder.ToString();
    }

    /// <summary>Recovers the original text from the byte alphabet.</summary>
    public static string Decode(string mapped)
    {
        var bytes = new List<byte>(mapped.Length);
        foreach (var character in mapped)
        {
            if (CharToByte.TryGetValue(character, out var b)) bytes.Add(b);
        }
        return System.Text.Encoding.UTF8.GetString([.. bytes]);
    }
}

/// <summary>
/// The SentencePiece-style pre-tokenizer: whitespace becomes a visible marker so that word
/// boundaries survive into the vocabulary.
/// </summary>
/// <param name="Replacement">The marker character, <c>U+2581</c> by convention.</param>
/// <param name="AddPrefixSpace">Whether to prepend a space before replacing.</param>
public sealed record MetaspacePreTokenizer(char Replacement = '▁', bool AddPrefixSpace = true) : IPreTokenizer
{
    /// <inheritdoc />
    public IReadOnlyList<PreToken> Split(string text)
    {
        var source = AddPrefixSpace && (text.Length == 0 || text[0] != ' ') ? " " + text : text;
        var shift = source.Length - text.Length;

        var tokens = new List<PreToken>();
        var builder = new StringBuilder();
        var start = 0;

        for (var i = 0; i < source.Length; i++)
        {
            var character = source[i] == ' ' ? Replacement : source[i];

            if (character == Replacement && builder.Length > 0)
            {
                tokens.Add(new PreToken(
                    builder.ToString(),
                    Math.Max(0, start - shift),
                    Math.Max(0, i - shift)));
                builder.Clear();
                start = i;
            }

            builder.Append(character);
        }

        if (builder.Length > 0)
        {
            tokens.Add(new PreToken(
                builder.ToString(),
                Math.Max(0, start - shift),
                Math.Max(0, source.Length - shift)));
        }

        return tokens;
    }
}

// ====================================================================== decoders

/// <summary>Joins WordPiece tokens, dropping the continuation prefix.</summary>
/// <param name="Prefix">The continuation marker, <c>##</c> by convention.</param>
public sealed record WordPieceDecoder(string Prefix = "##") : IDecoder
{
    /// <inheritdoc />
    public string Decode(IReadOnlyList<string> tokens)
    {
        var builder = new StringBuilder();

        foreach (var token in tokens)
        {
            if (token.StartsWith(Prefix, StringComparison.Ordinal))
            {
                builder.Append(token.AsSpan(Prefix.Length));
            }
            else
            {
                if (builder.Length > 0) builder.Append(' ');
                builder.Append(token);
            }
        }

        return builder.ToString();
    }
}

/// <summary>Reverses the byte alphabet, recovering the exact original bytes.</summary>
public sealed class ByteLevelDecoder : IDecoder
{
    /// <inheritdoc />
    public string Decode(IReadOnlyList<string> tokens) => ByteAlphabet.Decode(string.Concat(tokens));
}

/// <summary>Turns the SentencePiece marker back into spaces.</summary>
/// <param name="Replacement">The marker character.</param>
public sealed record MetaspaceDecoder(char Replacement = '▁') : IDecoder
{
    /// <inheritdoc />
    public string Decode(IReadOnlyList<string> tokens)
        => string.Concat(tokens).Replace(Replacement, ' ').TrimStart();
}

/// <summary>Joins tokens with spaces.</summary>
public sealed class WhitespaceDecoder : IDecoder
{
    /// <inheritdoc />
    public string Decode(IReadOnlyList<string> tokens) => string.Join(' ', tokens);
}
