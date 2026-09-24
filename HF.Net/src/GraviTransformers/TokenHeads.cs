using Gravicode.Science.GraviNum;
using Encoding = Gravicode.HFNet.GraviTokenizers.Encoding;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>One entity found in a text.</summary>
/// <param name="Text">The span as it appears in the input, not as pieces rejoined.</param>
/// <param name="Label">The entity type, with any <c>B-</c> or <c>I-</c> prefix removed.</param>
/// <param name="Score">Mean confidence across the tokens in the span.</param>
/// <param name="Start">Index of the first character in the input.</param>
/// <param name="End">Index one past the last character.</param>
public readonly record struct Entity(string Text, string Label, double Score, int Start, int End)
{
    /// <inheritdoc />
    public override string ToString() => $"{Label}: {Text} ({Score:P1})";
}

/// <summary>An extracted answer span.</summary>
/// <param name="Text">The answer as it appears in the context.</param>
/// <param name="Score">Joint confidence of the start and end positions.</param>
/// <param name="Start">Index of the first character in the context.</param>
/// <param name="End">Index one past the last character.</param>
public readonly record struct Answer(string Text, double Score, int Start, int End)
{
    /// <summary>Whether the model declined to answer by pointing at the classifier token.</summary>
    public bool IsEmpty => Text.Length == 0;

    /// <inheritdoc />
    public override string ToString()
        => IsEmpty ? "(no answer)" : $"{Text} ({Score:P1})";
}

/// <summary>
/// A linear head over the per-token hidden states: one label per token.
/// </summary>
/// <remarks>
/// The arithmetic is the easy part. What actually decides whether the output is usable is the
/// <i>decoding</i> - turning a label per subword into a span of the original text - which is what
/// <see cref="Decode"/> does and where the offsets earn their keep.
/// </remarks>
internal sealed class TokenClassificationHead
{
    private readonly NdArray _weight;
    private readonly NdArray _bias;

    private TokenClassificationHead(NdArray weight, NdArray bias)
    {
        _weight = weight;
        _bias = bias;
    }

    /// <summary>Loads the head, or returns null when the checkpoint is not a token classifier.</summary>
    /// <remarks>
    /// A sequence classifier and a token classifier both store <c>classifier.weight</c> with the
    /// same shape, so the weights alone cannot tell them apart. The architecture declared in
    /// <c>config.json</c> is the only thing that can, and guessing wrong gives a per-token model
    /// asked for one label, or the reverse - both of which run.
    /// </remarks>
    internal static TokenClassificationHead? TryLoad(WeightStore weights, PretrainedConfig config)
    {
        if (config.LabelCount == 0) return null;

        var declared = config.Architectures.Any(
            a => a.Contains("TokenClassification", StringComparison.OrdinalIgnoreCase));

        if (!declared) return null;
        if (!weights.TryRead("classifier.weight", out var weight)) return null;
        if (!weights.TryRead("classifier.bias", out var bias)) return null;

        return new TokenClassificationHead(weight, bias);
    }

    /// <summary>Scores every label for every position.</summary>
    internal double[][] Apply(NdArray hidden)
    {
        var rows = hidden.Shape[0];
        var logits = new double[rows][];

        for (var i = 0; i < rows; i++)
        {
            logits[i] = ClassificationHead.Linear(hidden.Row(i).ToArray(), _weight, _bias);
        }

        return logits;
    }

    /// <summary>
    /// Turns per-token labels into entity spans over the original text.
    /// </summary>
    /// <param name="text">The input the encoding came from.</param>
    /// <param name="encoding">The tokenization, for its offsets and special-token mask.</param>
    /// <param name="logits">One score vector per position.</param>
    /// <param name="labels">Class index to label name.</param>
    /// <param name="ignore">The label meaning "not an entity", usually <c>O</c>.</param>
    /// <remarks>
    /// <para>
    /// Tokens are merged into a span while they carry the same entity type, which handles both
    /// BIO tagging and the flat schemes some checkpoints use. A <c>B-</c> prefix starts a new span
    /// even against the same type, so two adjacent people do not become one.
    /// </para>
    /// <para>
    /// The span text is taken from the <b>source offsets</b>, never by rejoining subword pieces:
    /// rejoining loses the original casing and any punctuation inside the entity, and produces
    /// "##" artefacts that then have to be cleaned up by guesswork.
    /// </para>
    /// </remarks>
    internal static IReadOnlyList<Entity> Decode(
        string text,
        Encoding encoding,
        double[][] logits,
        IReadOnlyDictionary<int, string> labels,
        string ignore = "O")
    {
        var entities = new List<Entity>();

        var current = "";
        var start = 0;
        var end = 0;
        var scores = new List<double>();

        void Flush()
        {
            if (current.Length == 0) return;

            var span = end > start && end <= text.Length ? text[start..end] : "";
            if (span.Length > 0) entities.Add(new Entity(span, current, scores.Average(), start, end));

            current = "";
            scores.Clear();
        }

        for (var i = 0; i < encoding.Length && i < logits.Length; i++)
        {
            if (encoding.SpecialTokensMask[i] == 1) { Flush(); continue; }

            var probabilities = TransformerModel.Softmax(logits[i]);
            var best = Array.IndexOf(probabilities, probabilities.Max());
            var raw = labels.GetValueOrDefault(best, "O");

            if (raw == ignore) { Flush(); continue; }

            var beginning = raw.StartsWith("B-", StringComparison.Ordinal);
            var type = raw.Length > 2 && raw[1] == '-' ? raw[2..] : raw;

            var (tokenStart, tokenEnd) = encoding.Offsets[i];

            // A new span starts on a B- tag, on a change of type, or where the previous span ended
            // and this token does not continue it.
            if (current.Length == 0 || beginning || type != current)
            {
                Flush();
                current = type;
                start = tokenStart;
            }

            end = tokenEnd;
            scores.Add(probabilities[best]);
        }

        Flush();
        return entities;
    }
}

/// <summary>
/// The extractive question answering head: two scores per position, for the start and the end.
/// </summary>
/// <remarks>
/// The head is one linear layer, and everything interesting is in choosing the pair. The search is
/// constrained - the answer has to lie inside the context, end at or after start, and be shorter
/// than a limit - because an unconstrained argmax over each independently routinely produces an
/// end before its start, which is not a span at all.
/// </remarks>
internal sealed class QuestionAnsweringHead
{
    private readonly NdArray _weight;
    private readonly NdArray _bias;

    private QuestionAnsweringHead(NdArray weight, NdArray bias)
    {
        _weight = weight;
        _bias = bias;
    }

    /// <summary>Loads the head, or returns null when the checkpoint has none.</summary>
    internal static QuestionAnsweringHead? TryLoad(WeightStore weights)
    {
        if (!weights.TryReadAny(out var weight, "qa_outputs.weight", "qa_outputs.dense.weight"))
        {
            return null;
        }

        if (!weights.TryReadAny(out var bias, "qa_outputs.bias", "qa_outputs.dense.bias"))
        {
            return null;
        }

        // Two outputs: start and end. Anything else is a different head wearing the same name.
        if (weight.Shape[0] != 2) return null;

        return new QuestionAnsweringHead(weight, bias);
    }

    /// <summary>Produces the start and end score for every position.</summary>
    internal (double[] Start, double[] End) Apply(NdArray hidden)
    {
        var rows = hidden.Shape[0];
        var start = new double[rows];
        var end = new double[rows];

        for (var i = 0; i < rows; i++)
        {
            var pair = ClassificationHead.Linear(hidden.Row(i).ToArray(), _weight, _bias);
            start[i] = pair[0];
            end[i] = pair[1];
        }

        return (start, end);
    }

    /// <summary>
    /// Picks the best legal span and returns it as a substring of the context.
    /// </summary>
    /// <param name="context">The passage the answer must come from.</param>
    /// <param name="encoding">The question-and-context tokenization.</param>
    /// <param name="startLogits">Start score per position.</param>
    /// <param name="endLogits">End score per position.</param>
    /// <param name="maxAnswerTokens">Longest span to consider.</param>
    /// <param name="topK">How many candidates to return, best first.</param>
    /// <remarks>
    /// Only positions in segment 1 - the context - are eligible. A model is perfectly capable of
    /// pointing at a word in the question, and the answer to "who wrote it?" being "who" is the
    /// kind of output that looks like a model failure and is actually a decoding one.
    /// </remarks>
    internal static IReadOnlyList<Answer> Decode(
        string context,
        Encoding encoding,
        double[] startLogits,
        double[] endLogits,
        int maxAnswerTokens = 30,
        int topK = 1)
    {
        var startProbabilities = TransformerModel.Softmax(startLogits);
        var endProbabilities = TransformerModel.Softmax(endLogits);

        var candidates = new List<Answer>();

        for (var i = 0; i < encoding.Length; i++)
        {
            if (encoding.TypeIds[i] != 1 || encoding.SpecialTokensMask[i] == 1) continue;

            var limit = Math.Min(encoding.Length - 1, i + maxAnswerTokens - 1);

            for (var j = i; j <= limit; j++)
            {
                if (encoding.TypeIds[j] != 1 || encoding.SpecialTokensMask[j] == 1) continue;

                var start = encoding.Offsets[i].Start;
                var end = encoding.Offsets[j].End;

                if (end <= start || end > context.Length) continue;

                candidates.Add(new Answer(
                    context[start..end],
                    startProbabilities[i] * endProbabilities[j],
                    start,
                    end));
            }
        }

        if (candidates.Count == 0) return [new Answer("", 0, 0, 0)];

        return [.. candidates.OrderByDescending(c => c.Score).Take(Math.Max(1, topK))];
    }
}
