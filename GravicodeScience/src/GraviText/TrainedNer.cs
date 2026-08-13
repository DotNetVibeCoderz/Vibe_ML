using System.Text;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Sequence;

namespace Gravicode.Science.GraviText.Tasks;

/// <summary>One token with its BIO tag.</summary>
/// <param name="Token">The surface form.</param>
/// <param name="Tag">A BIO tag such as <c>O</c>, <c>B-PER</c> or <c>I-LOC</c>.</param>
public readonly record struct TaggedToken(string Token, string Tag);

/// <summary>A sentence annotated with entity tags.</summary>
public sealed class TaggedSentence(IReadOnlyList<TaggedToken> tokens)
{
    /// <summary>The tokens and their tags.</summary>
    public IReadOnlyList<TaggedToken> Tokens { get; } = tokens;

    /// <summary>Just the surface forms.</summary>
    public IReadOnlyList<string> Words => [.. Tokens.Select(t => t.Token)];

    /// <summary>Just the tags.</summary>
    public IReadOnlyList<string> Tags => [.. Tokens.Select(t => t.Tag)];

    /// <inheritdoc />
    public override string ToString() => string.Join(' ', Words);

    /// <summary>
    /// Reads sentences in the CoNLL column format: one token and tag per line, blank lines between
    /// sentences.
    /// </summary>
    public static IReadOnlyList<TaggedSentence> LoadConll(string path)
    {
        var sentences = new List<TaggedSentence>();
        var current = new List<TaggedToken>();

        foreach (var line in File.ReadLines(path))
        {
            var trimmed = line.Trim();

            if (trimmed.Length == 0 || trimmed.StartsWith('#'))
            {
                if (current.Count > 0) { sentences.Add(new TaggedSentence(current)); current = []; }
                continue;
            }

            var parts = trimmed.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;

            // The tag is the last column, so the format survives extra columns such as a POS tag.
            current.Add(new TaggedToken(parts[0], parts[^1]));
        }

        if (current.Count > 0) sentences.Add(new TaggedSentence(current));
        return sentences;
    }
}

/// <summary>Precision, recall and F1 over whole entities.</summary>
/// <param name="Precision">Of the entities predicted, the fraction that were right.</param>
/// <param name="Recall">Of the entities present, the fraction that were found.</param>
/// <param name="F1">Their harmonic mean.</param>
/// <param name="Predicted">How many entities were predicted.</param>
/// <param name="Actual">How many entities were present.</param>
public readonly record struct EntityScore(
    double Precision, double Recall, double F1, int Predicted, int Actual)
{
    /// <inheritdoc />
    public override string ToString()
        => $"P {Precision:P1}  R {Recall:P1}  F1 {F1:P1}  ({Predicted} predicted, {Actual} actual)";
}

/// <summary>
/// A trained named-entity recogniser: learned features scored per token, decoded by a CRF.
/// </summary>
/// <remarks>
/// <para>
/// The counterpart to the rule-based <see cref="NamedEntityRecognizer"/>, which matches gazetteers
/// and capitalisation patterns. This one learns from annotated text, which is what lets it
/// generalise to names it has never seen — the features describe the <em>shape</em> of a token and
/// its neighbours, not its identity, so a capitalised unknown word after "in" can be recognised as
/// a location on evidence rather than on a list.
/// </para>
/// <para>
/// The design is deliberately two-stage. A linear model scores each token independently against its
/// features, and a <see cref="LinearChainCrf"/> decodes the best <em>sequence</em> from those
/// scores. That split is what makes the BIO constraints enforceable: the per-token model cannot know
/// that an <c>I-PER</c> may not follow an <c>O</c>, and the CRF makes it structurally impossible
/// rather than merely unlikely.
/// </para>
/// <para>
/// <b>Evaluate on entities, not tokens.</b> Token accuracy on NER is dominated by the <c>O</c> tag —
/// a model that predicts <c>O</c> everywhere scores above 85% on most corpora — so it is close to
/// meaningless. <see cref="Evaluate"/> scores whole entities, which requires the boundaries and the
/// type all to be right.
/// </para>
/// </remarks>
public sealed class TrainedNer
{
    private readonly Dictionary<string, int> _featureIndex = new(StringComparer.Ordinal);
    private readonly List<string> _labels = [];
    private readonly Dictionary<string, int> _labelIndex = new(StringComparer.Ordinal);
    private double[,] _weights = new double[0, 0];
    private LinearChainCrf? _crf;

    /// <summary>The tag set, in the order the model indexes it.</summary>
    public IReadOnlyList<string> Labels => _labels;

    /// <summary>How many distinct features were learned.</summary>
    public int FeatureCount => _featureIndex.Count;

    /// <summary>True once <see cref="Fit"/> has run.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Trains on annotated sentences.
    /// </summary>
    /// <param name="sentences">Sentences with BIO tags.</param>
    /// <param name="epochs">Passes over the data for the per-token model.</param>
    /// <param name="learningRate">Step size for the per-token model.</param>
    /// <param name="crfEpochs">Passes for the transition model.</param>
    /// <remarks>
    /// Two fits in sequence: an averaged-perceptron-style update on the emission weights, then the
    /// CRF's transitions on the resulting emissions. Training them jointly would be better and needs
    /// the gradient to flow from the CRF back into the features, which is a tape-based model rather
    /// than this one; the staged version is what a feature-based tagger has always done and it is
    /// enough to demonstrate the architecture.
    /// </remarks>
    public TrainedNer Fit(IReadOnlyList<TaggedSentence> sentences, int epochs = 30,
        double learningRate = 0.1, int crfEpochs = 60)
    {
        ArgumentNullException.ThrowIfNull(sentences);
        if (sentences.Count == 0) throw new ArgumentException("There is nothing to train on.", nameof(sentences));

        // The label set, with O first so it is the natural default.
        foreach (var tag in sentences.SelectMany(s => s.Tags).Distinct().OrderBy(t => t == "O" ? "" : t,
            StringComparer.Ordinal))
            if (_labelIndex.TryAdd(tag, _labels.Count)) _labels.Add(tag);

        // The feature vocabulary, built once so weights can be a dense matrix.
        foreach (var sentence in sentences)
            for (var i = 0; i < sentence.Tokens.Count; i++)
                foreach (var feature in Features(sentence.Words, i))
                    _featureIndex.TryAdd(feature, _featureIndex.Count);

        _weights = new double[_featureIndex.Count, _labels.Count];

        // Perceptron updates: push the true label's weights up and the wrongly predicted label's
        // down. Only mistakes cause an update, which is what keeps it fast.
        for (var epoch = 0; epoch < epochs; epoch++)
            foreach (var sentence in sentences)
                for (var i = 0; i < sentence.Tokens.Count; i++)
                {
                    var active = ActiveFeatures(sentence.Words, i);
                    var truth = _labelIndex[sentence.Tokens[i].Tag];
                    var predicted = Predict(active);

                    if (predicted == truth) continue;

                    foreach (var feature in active)
                    {
                        _weights[feature, truth] += learningRate;
                        _weights[feature, predicted] -= learningRate;
                    }
                }

        _crf = new LinearChainCrf(_labels.Count);
        _crf.ApplyBioConstraints(_labels);

        var emissions = sentences.Select(s => Emissions(s.Words)).ToList();
        var tags = sentences.Select(s => s.Tags.Select(t => _labelIndex[t]).ToArray()).ToList();

        _crf.Fit(emissions, tags, crfEpochs, learningRate: 0.5);

        IsFitted = true;
        return this;
    }

    /// <summary>Tags a tokenised sentence.</summary>
    public IReadOnlyList<string> Tag(IReadOnlyList<string> words)
    {
        RequireFitted();
        if (words.Count == 0) return [];

        return [.. _crf!.Decode(Emissions(words)).Select(id => _labels[id])];
    }

    /// <summary>Finds the entities in a tokenised sentence.</summary>
    /// <remarks>
    /// Walks the BIO tags into spans. Because the CRF enforces the scheme, a continuation tag always
    /// has an opening tag before it and the walk cannot encounter a state it has no rule for.
    /// </remarks>
    public IReadOnlyList<Entity> Recognize(IReadOnlyList<string> words)
    {
        var tags = Tag(words);
        var entities = new List<Entity>();

        var start = -1;
        var type = "";
        var offset = 0;
        var startOffset = 0;

        for (var i = 0; i < words.Count; i++)
        {
            var tag = tags[i];

            if (tag.StartsWith("B-", StringComparison.Ordinal))
            {
                if (start >= 0) entities.Add(Build(words, start, i, type, startOffset, offset - 1));
                start = i;
                startOffset = offset;
                type = tag[2..];
            }
            else if (!tag.StartsWith("I-", StringComparison.Ordinal) && start >= 0)
            {
                entities.Add(Build(words, start, i, type, startOffset, offset - 1));
                start = -1;
            }

            offset += words[i].Length + 1;
        }

        if (start >= 0) entities.Add(Build(words, start, words.Count, type, startOffset, offset - 1));
        return entities;
    }

    /// <summary>Finds the entities in raw text, splitting on whitespace.</summary>
    public IReadOnlyList<Entity> Recognize(string text)
        => Recognize(text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    /// <summary>
    /// Scores predictions against gold annotations, entity by entity.
    /// </summary>
    /// <remarks>
    /// An entity counts as correct only when its start, its end and its type all match. Partial
    /// credit for overlapping spans is a defensible alternative and a far more forgiving one; the
    /// strict measure is what the CoNLL evaluation uses and what published figures mean.
    /// </remarks>
    public EntityScore Evaluate(IReadOnlyList<TaggedSentence> sentences)
    {
        RequireFitted();

        var correct = 0;
        var predicted = 0;
        var actual = 0;

        foreach (var sentence in sentences)
        {
            var gold = Spans(sentence.Tags);
            var found = Spans(Tag(sentence.Words));

            actual += gold.Count;
            predicted += found.Count;
            correct += found.Count(gold.Contains);
        }

        var precision = predicted > 0 ? (double)correct / predicted : 0.0;
        var recall = actual > 0 ? (double)correct / actual : 0.0;
        var f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

        return new EntityScore(precision, recall, f1, predicted, actual);
    }

    /// <summary>Per-token accuracy, reported for completeness rather than as a headline number.</summary>
    /// <remarks>
    /// Dominated by the <c>O</c> tag, so a model that predicts nothing scores well above 80% on most
    /// corpora. Use <see cref="Evaluate"/> to judge a tagger.
    /// </remarks>
    public double TokenAccuracy(IReadOnlyList<TaggedSentence> sentences)
    {
        RequireFitted();

        var correct = 0;
        var total = 0;

        foreach (var sentence in sentences)
        {
            var predicted = Tag(sentence.Words);
            for (var i = 0; i < predicted.Count; i++)
            {
                if (predicted[i] == sentence.Tags[i]) correct++;
                total++;
            }
        }

        return total > 0 ? (double)correct / total : 0.0;
    }

    // ------------------------------------------------------------------ features

    /// <summary>
    /// The features describing one token in its context.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Deliberately shape-based rather than identity-based. The word itself is one feature among
    /// many; the rest describe capitalisation, affixes, digit content and the neighbours — which is
    /// what lets an unseen name be recognised on evidence rather than on memorisation, and is the
    /// whole difference between this and a gazetteer.
    /// </para>
    /// <para>
    /// The word-shape feature does most of the work: mapping every capital to <c>X</c>, lower-case
    /// to <c>x</c> and digit to <c>d</c> turns "Jakarta" and "Bandung" into the same
    /// <c>Xxxxxxx</c>, so evidence about one transfers to the other.
    /// </para>
    /// </remarks>
    private static IEnumerable<string> Features(IReadOnlyList<string> words, int position)
    {
        var word = words[position];
        var lower = word.ToLowerInvariant();

        yield return "bias";
        yield return $"w={lower}";
        yield return $"shape={Shape(word)}";
        yield return $"upper={char.IsUpper(word[0])}";
        yield return $"allcaps={word.All(char.IsUpper) && word.Length > 1}";
        yield return $"hasdigit={word.Any(char.IsDigit)}";
        yield return $"first={position == 0}";

        if (lower.Length >= 2) yield return $"suf2={lower[^2..]}";
        if (lower.Length >= 3) yield return $"suf3={lower[^3..]}";
        if (lower.Length >= 2) yield return $"pre2={lower[..2]}";
        if (lower.Length >= 3) yield return $"pre3={lower[..3]}";

        // The neighbours, which is where most of the signal about entity boundaries lives — "in
        // Bandung" and "PT Bandung" say different things about the same word.
        yield return position > 0 ? $"-1w={words[position - 1].ToLowerInvariant()}" : "-1w=<s>";
        yield return position > 0 ? $"-1shape={Shape(words[position - 1])}" : "-1shape=<s>";
        yield return position + 1 < words.Count ? $"+1w={words[position + 1].ToLowerInvariant()}" : "+1w=</s>";
        yield return position + 1 < words.Count ? $"+1shape={Shape(words[position + 1])}" : "+1shape=</s>";
    }

    /// <summary>Collapses a word to its capitalisation and digit pattern.</summary>
    private static string Shape(string word)
    {
        var builder = new StringBuilder(word.Length);

        foreach (var c in word)
            builder.Append(char.IsUpper(c) ? 'X' : char.IsLower(c) ? 'x' : char.IsDigit(c) ? 'd' : c);

        return builder.ToString();
    }

    /// <summary>The indices of a token's features that the model has seen before.</summary>
    /// <remarks>
    /// Unseen features are dropped rather than hashed into a bucket. Hashing would let an unknown
    /// feature collide with a learned one and borrow its weight, which is worse than contributing
    /// nothing.
    /// </remarks>
    private int[] ActiveFeatures(IReadOnlyList<string> words, int position)
        => [.. Features(words, position)
            .Select(f => _featureIndex.GetValueOrDefault(f, -1))
            .Where(index => index >= 0)];

    private int Predict(int[] active)
    {
        var best = 0;
        var bestScore = double.NegativeInfinity;

        for (var label = 0; label < _labels.Count; label++)
        {
            var score = 0.0;
            foreach (var feature in active) score += _weights[feature, label];

            if (score <= bestScore) continue;
            bestScore = score;
            best = label;
        }

        return best;
    }

    /// <summary>The per-token label scores the CRF decodes over.</summary>
    private NdArray Emissions(IReadOnlyList<string> words)
    {
        var result = NdArray.Zeros(words.Count, _labels.Count);

        for (var i = 0; i < words.Count; i++)
        {
            var active = ActiveFeatures(words, i);
            for (var label = 0; label < _labels.Count; label++)
            {
                var score = 0.0;
                foreach (var feature in active) score += _weights[feature, label];
                result[i, label] = score;
            }
        }

        return result;
    }

    /// <summary>The entity spans a tag sequence describes, as (start, end, type).</summary>
    private static HashSet<(int Start, int End, string Type)> Spans(IReadOnlyList<string> tags)
    {
        var spans = new HashSet<(int, int, string)>();
        var start = -1;
        var type = "";

        for (var i = 0; i < tags.Count; i++)
        {
            var tag = tags[i];

            if (tag.StartsWith("B-", StringComparison.Ordinal))
            {
                if (start >= 0) spans.Add((start, i - 1, type));
                start = i;
                type = tag[2..];
            }
            else if (tag.StartsWith("I-", StringComparison.Ordinal))
            {
                // A continuation with no opener, or of the wrong type, starts its own span rather
                // than being dropped — the gold data may not obey the constraints the model does.
                if (start < 0 || tag[2..] != type) { if (start >= 0) spans.Add((start, i - 1, type)); start = i; type = tag[2..]; }
            }
            else if (start >= 0)
            {
                spans.Add((start, i - 1, type));
                start = -1;
            }
        }

        if (start >= 0) spans.Add((start, tags.Count - 1, type));
        return spans;
    }

    private static Entity Build(IReadOnlyList<string> words, int from, int to, string type,
        int startOffset, int endOffset)
        => new(string.Join(' ', words.Skip(from).Take(to - from)), type, startOffset,
            Math.Max(0, endOffset - startOffset));

    private void RequireFitted()
    {
        if (!IsFitted) throw new InvalidOperationException("The recogniser must be fitted before use.");
    }
}
