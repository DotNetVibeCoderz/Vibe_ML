using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTransformers;

namespace Gravicode.HFNet.GraviPEFT;

/// <summary>
/// Moves word-level tags onto subword tokens, and back again, the way Transformers' token
/// classification examples do.
/// </summary>
/// <remarks>
/// Datasets such as CoNLL tag words; the model sees subwords. The convention this follows is
/// Transformers' <c>label_all_tokens=False</c>: the first piece of each word carries the word's tag
/// and every other piece is ignored in the loss. Tagging the continuation pieces too makes long
/// words count several times over, and makes a model trained here disagree with one trained in
/// Python on the same data.
/// </remarks>
internal static class TokenAlignment
{
    /// <summary>Joins words with single spaces, returning the text and where each word sits in it.</summary>
    internal static (string Text, (int Start, int End)[] Spans) Join(IReadOnlyList<string> words)
    {
        var spans = new (int Start, int End)[words.Count];
        var builder = new System.Text.StringBuilder();

        for (var w = 0; w < words.Count; w++)
        {
            if (w > 0) builder.Append(' ');
            spans[w] = (builder.Length, builder.Length + words[w].Length);
            builder.Append(words[w]);
        }

        return (builder.ToString(), spans);
    }

    /// <summary>One target per token: the word's tag on its first piece, ignored everywhere else.</summary>
    /// <param name="encoding">The tokenization of the joined text.</param>
    /// <param name="spans">Where each word sits in that text.</param>
    /// <param name="tags">Class index per word.</param>
    internal static int[] Targets(Encoding encoding, IReadOnlyList<(int Start, int End)> spans, IReadOnlyList<int> tags)
    {
        var targets = new int[encoding.Length];
        var labelled = new bool[spans.Count];

        for (var i = 0; i < targets.Length; i++)
        {
            targets[i] = LinearHead.Ignored;
            if (encoding.SpecialTokensMask[i] == 1) continue;

            var start = encoding.Offsets[i].Start;
            var word = WordAt(spans, start);
            if (word < 0 || labelled[word] || start != spans[word].Start) continue;

            targets[i] = tags[word];
            labelled[word] = true;
        }

        return targets;
    }

    private static int WordAt(IReadOnlyList<(int Start, int End)> spans, int position)
    {
        for (var w = 0; w < spans.Count; w++)
        {
            if (position >= spans[w].Start && position < spans[w].End) return w;
        }

        return -1;
    }

    /// <summary>
    /// For each token, the index of the token that starts its word - itself, unless it continues a
    /// word that an earlier token began.
    /// </summary>
    /// <remarks>
    /// Only a word's first piece was trained, so a prediction is read from there and applied to the
    /// whole word, which is Transformers' <c>aggregation_strategy="first"</c>. Free text has no word
    /// list, so a token continues the previous one when the two touch and the characters either side
    /// of the join are letters or digits. Punctuation that touches a word ("Paris,") stays apart,
    /// which is how the Transformers pre-tokenizers split it too.
    /// </remarks>
    internal static int[] WordStarts(string text, Encoding encoding)
    {
        var starts = new int[encoding.Length];

        for (var i = 0; i < starts.Length; i++)
        {
            starts[i] = i;
            if (i == 0 || encoding.SpecialTokensMask[i] == 1 || encoding.SpecialTokensMask[i - 1] == 1) continue;

            var (start, _) = encoding.Offsets[i];
            var (_, previousEnd) = encoding.Offsets[i - 1];

            var touches = start == previousEnd && start > 0 && start < text.Length;
            if (touches && char.IsLetterOrDigit(text[start]) && char.IsLetterOrDigit(text[start - 1]))
            {
                starts[i] = starts[i - 1];
            }
        }

        return starts;
    }

    /// <summary>Entity spans from per-token logits, one label per word.</summary>
    /// <param name="text">The input the encoding came from.</param>
    /// <param name="encoding">Its tokenization.</param>
    /// <param name="logits">Scores per token; rows past the end (truncation) are ignored.</param>
    /// <param name="labels">Label names by class index.</param>
    /// <remarks>
    /// A word takes the label of its first piece, and all its pieces join the span that label puts
    /// it in. Decoding piece by piece instead splits "Kartini" into "Ka", "rti" and "ni", because
    /// every piece carries the <c>B-</c> its first piece was trained on. A word tagged <c>B-X</c>
    /// starts an entity; <c>I-X</c> continues one of type X, or starts one when there is none to
    /// continue; <c>O</c> ends it.
    /// </remarks>
    internal static IReadOnlyList<Entity> Decode(
        string text, Encoding encoding, double[][] logits, IReadOnlyList<string> labels)
    {
        var starts = WordStarts(text, encoding);
        var entities = new List<Entity>();

        var type = "";
        var spanStart = 0;
        var spanEnd = 0;
        var scores = new List<double>();

        void Flush()
        {
            if (type.Length > 0 && spanEnd > spanStart && spanEnd <= text.Length)
            {
                entities.Add(new Entity(text[spanStart..spanEnd], type, scores.Average(), spanStart, spanEnd));
            }

            type = "";
            scores.Clear();
        }

        var count = Math.Min(encoding.Length, logits.Length);
        for (var i = 0; i < count; i++)
        {
            if (encoding.SpecialTokensMask[i] == 1) { Flush(); continue; }

            var (start, end) = encoding.Offsets[i];

            // A continuing piece only stretches the current span to cover itself.
            if (starts[i] != i)
            {
                if (type.Length > 0) spanEnd = end;
                continue;
            }

            var probabilities = LinearHead.Softmax(logits[i]);
            var best = Array.IndexOf(probabilities, probabilities.Max());
            var label = labels[best];

            if (label == "O") { Flush(); continue; }

            var beginning = label.StartsWith("B-", StringComparison.Ordinal);
            var kind = label.Length > 2 && label[1] == '-' ? label[2..] : label;

            if (type.Length == 0 || beginning || kind != type)
            {
                Flush();
                type = kind;
                spanStart = start;
            }

            spanEnd = end;
            scores.Add(probabilities[best]);
        }

        Flush();
        return entities;
    }

    /// <summary>
    /// The first and last token of an answer in the context, or <c>(0, 0)</c> when it is not wholly
    /// inside the first <paramref name="length"/> tokens.
    /// </summary>
    /// <param name="encoding">The question-and-context tokenization; context offsets are into the context.</param>
    /// <param name="answerStart">Index of the answer's first character in the context.</param>
    /// <param name="answerEnd">Index one past its last character.</param>
    /// <param name="length">How many tokens survive truncation.</param>
    /// <remarks>
    /// The start is the last context token that begins at or before the answer, and the end the
    /// first that finishes at or after it, which is how Transformers' question answering example
    /// maps characters to tokens. Pointing at <c>[CLS]</c> for an answer cut off by truncation is its
    /// convention too: the model learns that position means "not here".
    /// </remarks>
    internal static (int Start, int End) AnswerTokens(Encoding encoding, int answerStart, int answerEnd, int length)
    {
        var first = -1;
        var last = -1;
        var count = Math.Min(length, encoding.Length);

        for (var i = 0; i < count; i++)
        {
            if (encoding.TypeIds[i] != 1 || encoding.SpecialTokensMask[i] == 1) continue;

            var (start, end) = encoding.Offsets[i];
            if (start <= answerStart) first = i;
            if (last < 0 && end >= answerEnd) last = i;
        }

        return first < 0 || last < 0 || last < first ? (0, 0) : (first, last);
    }

    /// <summary>The first <paramref name="length"/> positions of an encoding.</summary>
    internal static Encoding Truncate(Encoding encoding, int length)
        => encoding.Length <= length
            ? encoding
            : new Encoding(
                [.. encoding.Ids.Take(length)],
                [.. encoding.Tokens.Take(length)],
                [.. encoding.AttentionMask.Take(length)],
                [.. encoding.TypeIds.Take(length)],
                [.. encoding.SpecialTokensMask.Take(length)],
                [.. encoding.Offsets.Take(length)]);
}
