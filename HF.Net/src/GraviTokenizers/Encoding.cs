using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviTokenizers;

/// <summary>
/// What a tokenizer produces for one input: the ids a model consumes, plus everything needed to
/// map them back to the text they came from.
/// </summary>
/// <remarks>
/// The offsets are the part worth keeping. Without them a token classification result - an entity
/// span, an answer span - can only be reported as token indices, which are meaningless to anyone
/// who did not run the same tokenizer. With them the result is a substring of the original input.
/// </remarks>
public sealed class Encoding
{
    /// <summary>Creates an encoding.</summary>
    /// <param name="ids">The vocabulary ids.</param>
    /// <param name="tokens">The surface pieces the ids correspond to.</param>
    /// <param name="attentionMask">1 for a real token, 0 for padding.</param>
    /// <param name="typeIds">Segment ids, 0 for the first sequence and 1 for the second.</param>
    /// <param name="specialTokensMask">1 where the tokenizer inserted a special token.</param>
    /// <param name="offsets">Character ranges in the source text, empty for inserted tokens.</param>
    public Encoding(
        IReadOnlyList<int> ids,
        IReadOnlyList<string> tokens,
        IReadOnlyList<int> attentionMask,
        IReadOnlyList<int> typeIds,
        IReadOnlyList<int> specialTokensMask,
        IReadOnlyList<(int Start, int End)> offsets)
    {
        Ids = ids;
        Tokens = tokens;
        AttentionMask = attentionMask;
        TypeIds = typeIds;
        SpecialTokensMask = specialTokensMask;
        Offsets = offsets;
    }

    /// <summary>The vocabulary ids, which is what a model actually consumes.</summary>
    public IReadOnlyList<int> Ids { get; }

    /// <summary>The surface pieces, useful for debugging a tokenizer that is behaving oddly.</summary>
    public IReadOnlyList<string> Tokens { get; }

    /// <summary>1 for a real token, 0 for padding.</summary>
    public IReadOnlyList<int> AttentionMask { get; }

    /// <summary>Segment ids: 0 for the first sequence, 1 for the second in a pair.</summary>
    public IReadOnlyList<int> TypeIds { get; }

    /// <summary>1 where the token was inserted by the tokenizer rather than found in the text.</summary>
    public IReadOnlyList<int> SpecialTokensMask { get; }

    /// <summary>
    /// Character ranges in the source text. A special token has the empty range <c>(0, 0)</c>.
    /// </summary>
    public IReadOnlyList<(int Start, int End)> Offsets { get; }

    /// <summary>Number of tokens, padding included.</summary>
    public int Length => Ids.Count;

    /// <summary>The ids as an array, for handing to a model.</summary>
    public int[] ToIdArray() => [.. Ids];

    /// <summary>The attention mask as an array.</summary>
    public int[] ToMaskArray() => [.. AttentionMask];

    /// <summary>The substring of <paramref name="text"/> that token <paramref name="index"/> covers.</summary>
    /// <remarks>
    /// Returns an empty string for a special token, which by construction covers nothing.
    /// </remarks>
    public string Span(string text, int index)
    {
        var (start, end) = Offsets[index];
        return end > start && end <= text.Length ? text[start..end] : "";
    }

    /// <summary>
    /// The substring covering a run of tokens, from <paramref name="first"/> to
    /// <paramref name="last"/> inclusive.
    /// </summary>
    /// <remarks>
    /// Taken as one range from the first token's start to the last token's end rather than by
    /// joining the pieces: the original text keeps its whitespace and punctuation, and rebuilding
    /// it from subword pieces silently loses both.
    /// </remarks>
    public string Span(string text, int first, int last)
    {
        var start = Offsets[first].Start;
        var end = Offsets[last].End;
        return end > start && end <= text.Length ? text[start..end] : "";
    }

    /// <inheritdoc />
    public override string ToString()
        => $"Encoding({Length} tokens: {string.Join(' ', Tokens.Take(12))}{(Length > 12 ? " ..." : "")})";
}

/// <summary>A batch of encodings, padded to one width so it can go to a model as a matrix.</summary>
public sealed class EncodingBatch
{
    /// <summary>Creates a batch from encodings that are already the same length.</summary>
    /// <param name="encodings">The per-input encodings.</param>
    public EncodingBatch(IReadOnlyList<Encoding> encodings)
    {
        Encodings = encodings;
        Width = encodings.Count == 0 ? 0 : encodings[0].Length;

        foreach (var encoding in encodings)
        {
            if (encoding.Length != Width)
            {
                throw new ArgumentException(
                    $"A batch must be rectangular, but encodings of {Width} and {encoding.Length} tokens were mixed. "
                    + "Encode with padding enabled.",
                    nameof(encodings));
            }
        }
    }

    /// <summary>The individual encodings.</summary>
    public IReadOnlyList<Encoding> Encodings { get; }

    /// <summary>Number of inputs in the batch.</summary>
    public int Count => Encodings.Count;

    /// <summary>Tokens per input.</summary>
    public int Width { get; }

    /// <summary>The ids as a <c>[batch, width]</c> matrix.</summary>
    public NdArray Ids => Matrix(e => e.Ids);

    /// <summary>The attention mask as a <c>[batch, width]</c> matrix.</summary>
    public NdArray AttentionMask => Matrix(e => e.AttentionMask);

    /// <summary>The segment ids as a <c>[batch, width]</c> matrix.</summary>
    public NdArray TypeIds => Matrix(e => e.TypeIds);

    private NdArray Matrix(Func<Encoding, IReadOnlyList<int>> select)
    {
        var data = new double[Count * Width];
        for (var row = 0; row < Count; row++)
        {
            var values = select(Encodings[row]);
            for (var column = 0; column < Width; column++) data[(row * Width) + column] = values[column];
        }
        return new NdArray(data, Count, Width);
    }

    /// <inheritdoc />
    public override string ToString() => $"EncodingBatch({Count} x {Width})";
}

/// <summary>How a tokenizer should pad a batch.</summary>
public enum PaddingStrategy
{
    /// <summary>No padding; each input keeps its own length.</summary>
    None,

    /// <summary>Pad every input to the longest one in the batch.</summary>
    Longest,

    /// <summary>Pad every input to a fixed length.</summary>
    Fixed,
}

/// <summary>Padding and truncation settings for a batch.</summary>
/// <param name="Padding">How to pad.</param>
/// <param name="MaxLength">
/// The width for <see cref="PaddingStrategy.Fixed"/>, and the truncation limit when truncation is
/// on. Ignored by <see cref="PaddingStrategy.Longest"/> unless truncation applies first.
/// </param>
/// <param name="Truncate">Whether inputs longer than <paramref name="MaxLength"/> are cut.</param>
public readonly record struct BatchOptions(
    PaddingStrategy Padding = PaddingStrategy.Longest,
    int MaxLength = 512,
    bool Truncate = true)
{
    /// <summary>The settings a BERT-style encoder usually wants.</summary>
    /// <remarks>
    /// Spelled out rather than written <c>new()</c>. A record struct's parameterless constructor
    /// zeroes every field and does <b>not</b> run the primary constructor's default arguments, so
    /// <c>new BatchOptions()</c> means "no padding, truncate to zero" - which then fails on the
    /// first ragged batch with a message about rectangularity rather than about padding.
    /// </remarks>
    public static BatchOptions Default => new(PaddingStrategy.Longest, 512, Truncate: true);

    /// <summary>Pad and truncate to exactly <paramref name="length"/> tokens.</summary>
    public static BatchOptions Exactly(int length)
        => new(PaddingStrategy.Fixed, length, Truncate: true);
}
