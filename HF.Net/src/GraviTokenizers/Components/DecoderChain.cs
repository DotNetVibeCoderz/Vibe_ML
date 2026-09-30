using System.Text;
using System.Text.RegularExpressions;

namespace Gravicode.HFNet.GraviTokenizers.Components;

/// <summary>One stage of a decoder: tokens in, tokens out, as the Rust crate's <c>decode_chain</c> works.</summary>
/// <remarks>
/// A <c>tokenizer.json</c> decoder is often a sequence - Llama's is Replace, ByteFallback, Fuse,
/// Strip - and each stage sees the previous stage's list of tokens, not a joined string. Byte
/// fallback, for one, has to see the <c>&lt;0x..&gt;</c> pieces as pieces.
/// </remarks>
public interface IDecoderStep
{
    /// <summary>Transforms the token list.</summary>
    List<string> DecodeChain(List<string> tokens);
}

/// <summary>Runs decoder steps in order and joins what is left.</summary>
/// <param name="Steps">The stages, first to last.</param>
public sealed record SequenceDecoder(IReadOnlyList<IDecoderStep> Steps) : IDecoder
{
    /// <inheritdoc />
    public string Decode(IReadOnlyList<string> tokens)
    {
        var current = new List<string>(tokens);
        foreach (var step in Steps) current = step.DecodeChain(current);
        return string.Concat(current);
    }
}

/// <summary>Replaces a literal or a pattern in every token.</summary>
public sealed record ReplaceStep(string Pattern, string Content, bool IsRegex = false) : IDecoderStep
{
    /// <inheritdoc />
    public List<string> DecodeChain(List<string> tokens)
        => [.. tokens.Select(t => IsRegex ? Regex.Replace(t, Pattern, Content) : t.Replace(Pattern, Content, StringComparison.Ordinal))];
}

/// <summary>
/// Turns runs of <c>&lt;0xXX&gt;</c> byte pieces back into text: the bytes are decoded as UTF-8,
/// and a run that is not valid UTF-8 becomes one U+FFFD per byte, as the reference does.
/// </summary>
public sealed class ByteFallbackStep : IDecoderStep
{
    /// <inheritdoc />
    public List<string> DecodeChain(List<string> tokens)
    {
        var result = new List<string>(tokens.Count);
        var bytes = new List<byte>();

        void Flush()
        {
            if (bytes.Count == 0) return;
            try
            {
                result.Add(new UTF8Encoding(false, throwOnInvalidBytes: true).GetString([.. bytes]));
            }
            catch (DecoderFallbackException)
            {
                for (var i = 0; i < bytes.Count; i++) result.Add("�");
            }

            bytes.Clear();
        }

        foreach (var token in tokens)
        {
            if (TryByte(token, out var value))
            {
                bytes.Add(value);
                continue;
            }

            Flush();
            result.Add(token);
        }

        Flush();
        return result;
    }

    internal static bool TryByte(string token, out byte value)
    {
        value = 0;
        return token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token[5] == '>'
            && byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out value);
    }
}

/// <summary>Joins every token into one.</summary>
public sealed class FuseStep : IDecoderStep
{
    /// <inheritdoc />
    public List<string> DecodeChain(List<string> tokens) => [string.Concat(tokens)];
}

/// <summary>Removes up to <see cref="Start"/> leading and <see cref="Stop"/> trailing copies of a character from each token.</summary>
public sealed record StripStep(char Content, int Start, int Stop) : IDecoderStep
{
    /// <inheritdoc />
    public List<string> DecodeChain(List<string> tokens)
    {
        var result = new List<string>(tokens.Count);
        foreach (var token in tokens)
        {
            var begin = 0;
            while (begin < Start && begin < token.Length && token[begin] == Content) begin++;

            var end = token.Length;
            var removed = 0;
            while (removed < Stop && end > begin && token[end - 1] == Content)
            {
                end--;
                removed++;
            }

            result.Add(token[begin..end]);
        }

        return result;
    }
}

/// <summary>
/// SentencePiece's marker back to spaces; the first token loses its leading space unless the
/// tokenizer never prepends one.
/// </summary>
public sealed record MetaspaceStep(char Replacement = '▁', bool StripFirst = true) : IDecoderStep
{
    /// <inheritdoc />
    public List<string> DecodeChain(List<string> tokens)
    {
        var result = new List<string>(tokens.Count);
        for (var i = 0; i < tokens.Count; i++)
        {
            var text = tokens[i].Replace(Replacement, ' ');
            if (i == 0 && StripFirst && text.StartsWith(' ')) text = text[1..];
            result.Add(text);
        }

        return result;
    }
}

/// <summary>The byte alphabet back to bytes, then UTF-8 - all tokens at once, since a character may span two.</summary>
/// <remarks>
/// A token with any character outside the byte alphabet - an added token such as Pythia's runs of
/// spaces, stored as written - contributes its own UTF-8 bytes instead, as the reference does.
/// Dropping those characters lost every multiple space in decoded text.
/// </remarks>
public sealed class ByteLevelStep : IDecoderStep
{
    /// <inheritdoc />
    public List<string> DecodeChain(List<string> tokens)
    {
        var bytes = new List<byte>();
        foreach (var token in tokens)
        {
            var mapped = ByteAlphabet.TryDecodeBytes(token);
            bytes.AddRange(mapped ?? System.Text.Encoding.UTF8.GetBytes(token));
        }

        return [System.Text.Encoding.UTF8.GetString([.. bytes])];
    }
}

/// <summary>
/// WordPiece pieces back to words: continuation pieces lose their prefix, others gain a space, and
/// with <see cref="Cleanup"/> the spaces tokenization put before punctuation and contractions go.
/// </summary>
public sealed record WordPieceStep(string Prefix = "##", bool Cleanup = true) : IDecoderStep
{
    /// <inheritdoc />
    public List<string> DecodeChain(List<string> tokens)
    {
        var result = new List<string>(tokens.Count);
        for (var i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            if (i != 0)
            {
                token = token.StartsWith(Prefix, StringComparison.Ordinal) ? token[Prefix.Length..] : " " + token;
            }

            result.Add(Cleanup ? CleanUp(token) : token);
        }

        return result;
    }

    /// <summary>The reference's <c>clean_up_tokenization</c>, replacement for replacement.</summary>
    internal static string CleanUp(string text) => text
        .Replace(" .", ".", StringComparison.Ordinal)
        .Replace(" ?", "?", StringComparison.Ordinal)
        .Replace(" !", "!", StringComparison.Ordinal)
        .Replace(" ,", ",", StringComparison.Ordinal)
        .Replace(" ' ", "'", StringComparison.Ordinal)
        .Replace(" n't", "n't", StringComparison.Ordinal)
        .Replace(" 'm", "'m", StringComparison.Ordinal)
        .Replace(" do not", " don't", StringComparison.Ordinal)
        .Replace(" 's", "'s", StringComparison.Ordinal)
        .Replace(" 've", "'ve", StringComparison.Ordinal)
        .Replace(" 're", "'re", StringComparison.Ordinal);
}

/// <summary>Joins tokens with spaces - the fallback when a tokenizer names no decoder.</summary>
public sealed class WhitespaceStep : IDecoderStep
{
    /// <inheritdoc />
    public List<string> DecodeChain(List<string> tokens) => [string.Join(' ', tokens)];
}

/// <summary>Prepends a string to non-empty text - Llama's normalizer puts the SentencePiece marker in front.</summary>
/// <param name="Prefix">What to put in front.</param>
public sealed record PrependNormalizer(string Prefix) : INormalizer
{
    /// <inheritdoc />
    public string Normalize(string text) => text.Length == 0 ? text : Prefix + text;
}

/// <summary>No pre-tokenization: the whole text is one piece, which is what a null <c>pre_tokenizer</c> means.</summary>
public sealed class WholeTextPreTokenizer : IPreTokenizer
{
    /// <inheritdoc />
    public IReadOnlyList<PreToken> Split(string text) => text.Length == 0 ? [] : [new PreToken(text, 0, text.Length)];
}
