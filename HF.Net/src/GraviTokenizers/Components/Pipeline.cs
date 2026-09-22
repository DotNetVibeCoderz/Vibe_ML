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
            else if (IsPunctuation(character))
            {
                if (start >= 0) tokens.Add(new PreToken(text[start..i], start, i));
                tokens.Add(new PreToken(text[i].ToString(), i, i + 1));
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

/// <summary>Splits on a regular expression.</summary>
/// <param name="Pattern">The expression.</param>
/// <param name="Invert">
/// When false the matches are the delimiters; when true the matches are the pieces to keep.
/// </param>
public sealed partial record SplitPreTokenizer(string Pattern, bool Invert = false) : IPreTokenizer
{
    private Regex? _compiled;

    /// <inheritdoc />
    public IReadOnlyList<PreToken> Split(string text)
    {
        _compiled ??= new Regex(Pattern, RegexOptions.Compiled);
        var tokens = new List<PreToken>();

        if (Invert)
        {
            foreach (Match match in _compiled.Matches(text))
            {
                if (match.Length > 0) tokens.Add(new PreToken(match.Value, match.Index, match.Index + match.Length));
            }
            return tokens;
        }

        var cursor = 0;
        foreach (Match match in _compiled.Matches(text))
        {
            if (match.Index > cursor)
            {
                tokens.Add(new PreToken(text[cursor..match.Index], cursor, match.Index));
            }
            cursor = match.Index + match.Length;
        }

        if (cursor < text.Length) tokens.Add(new PreToken(text[cursor..], cursor, text.Length));
        return tokens;
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
