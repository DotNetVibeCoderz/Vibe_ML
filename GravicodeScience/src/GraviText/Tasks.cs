using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Linguistics;
using Gravicode.Science.GraviText.Tokenization;
using Gravicode.Science.GraviText.Vectorization;

namespace Gravicode.Science.GraviText.Tasks;

/// <summary>A predicted label and how confident the model was.</summary>
/// <param name="Label">The predicted class name.</param>
/// <param name="Confidence">Probability assigned to <paramref name="Label"/>.</param>
/// <param name="Scores">Probability of every class, by name.</param>
public sealed record Prediction(string Label, double Confidence, IReadOnlyDictionary<string, double> Scores)
{
    /// <inheritdoc />
    public override string ToString() => $"{Label} ({Confidence:P1})";
}

/// <summary>
/// A supervised text classifier: TF-IDF features into a logistic regression.
/// </summary>
/// <remarks>
/// This combination remains a strong baseline for topic and sentiment classification, and it has
/// two properties a transformer does not: it trains in seconds on a laptop, and its coefficients
/// are directly readable, so <see cref="TopFeatures"/> can show exactly which words drove a
/// decision.
/// </remarks>
public sealed class TextClassifier(VectorizerOptions? options = null, double learningRate = 0.5, int iterations = 800)
{
    private readonly TfidfVectorizer _vectorizer = new(options ?? DefaultOptions(), sublinearTf: true);
    private readonly LogisticRegression _model = new(learningRate, iterations, l2Penalty: 1e-4);
    private string[] _labels = [];

    private static VectorizerOptions DefaultOptions() => new()
    {
        Tokenizer = new RegexTokenizer(),
        StopWords = StopWords.Bilingual,
        MinNGram = 1,
        MaxNGram = 2,
        MinDocumentFrequency = 1,
    };

    /// <summary>The class names, in model order.</summary>
    public IReadOnlyList<string> Labels => _labels;

    /// <summary>Number of TF-IDF features.</summary>
    public int FeatureCount => _vectorizer.FeatureCount;

    /// <summary>True once the classifier has been trained.</summary>
    public bool IsTrained { get; private set; }

    /// <summary>Trains on labelled documents.</summary>
    public TextClassifier Train(IReadOnlyList<string> documents, IReadOnlyList<string> labels)
    {
        if (documents.Count != labels.Count)
            throw new ArgumentException($"Got {documents.Count} documents and {labels.Count} labels.");

        _labels = labels.Distinct().OrderBy(l => l, StringComparer.Ordinal).ToArray();
        var codes = NdArray.FromValues(labels.Select(l => (double)Array.IndexOf(_labels, l)));

        var features = _vectorizer.FitTransform(documents);
        _model.Fit(features, codes);
        IsTrained = true;
        return this;
    }

    /// <summary>Classifies one document.</summary>
    public Prediction Predict(string document)
    {
        RequireTrained();
        var features = _vectorizer.Transform([document]);
        var probabilities = _model.PredictProbabilities(features);

        var scores = new Dictionary<string, double>(StringComparer.Ordinal);
        var best = 0;
        for (var c = 0; c < _labels.Length; c++)
        {
            scores[_labels[c]] = probabilities[0, c];
            if (probabilities[0, c] > probabilities[0, best]) best = c;
        }
        return new Prediction(_labels[best], probabilities[0, best], scores);
    }

    /// <summary>Classifies several documents.</summary>
    public IReadOnlyList<Prediction> PredictBatch(IReadOnlyList<string> documents)
        => documents.Select(Predict).ToList();

    /// <summary>Accuracy on labelled data.</summary>
    public double Evaluate(IReadOnlyList<string> documents, IReadOnlyList<string> labels)
    {
        var predictions = PredictBatch(documents);
        var correct = predictions.Where((p, i) => p.Label == labels[i]).Count();
        return (double)correct / documents.Count;
    }

    /// <summary>
    /// The terms that push hardest toward a class, read straight off the model coefficients.
    /// </summary>
    public IReadOnlyList<(string Term, double Weight)> TopFeatures(string label, int count = 15)
    {
        RequireTrained();
        var index = Array.IndexOf(_labels, label);
        if (index < 0) throw new ArgumentException($"Unknown label '{label}'.");

        var coefficients = _model.CoefficientMatrix;
        // A two-class model has one coefficient row; the negative class is its mirror image.
        var row = coefficients.Shape[0] == 1 ? 0 : index;
        var sign = coefficients.Shape[0] == 1 && index == 0 ? -1.0 : 1.0;

        return Enumerable.Range(0, _vectorizer.FeatureCount)
            .Select(j => (Term: _vectorizer.Vocabulary[j], Weight: coefficients[row, j] * sign))
            .OrderByDescending(t => t.Weight)
            .Take(count)
            .ToList();
    }

    private void RequireTrained()
    {
        if (!IsTrained) throw new InvalidOperationException("The classifier must be trained before use.");
    }
}

/// <summary>
/// Sentiment analysis, either trained on labelled data or run straight from a lexicon.
/// </summary>
/// <remarks>
/// The lexicon path exists because sentiment is the one task where a decent answer is available
/// with no training data at all: a list of positive and negative words plus negation and
/// intensifier handling gets a long way. When labelled examples are available,
/// <see cref="Train"/> switches to the supervised classifier, which is materially better.
/// The lexicon covers English and Bahasa Indonesia.
/// </remarks>
public sealed class SentimentAnalyzer
{
    private readonly ITokenizer _tokenizer = new RegexTokenizer();
    private TextClassifier? _classifier;

    /// <summary>Words carrying positive sentiment, English and Indonesian.</summary>
    public static IReadOnlySet<string> PositiveLexicon { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "loved", "best", "brilliant",
        "perfect", "superb", "delightful", "enjoyable", "impressive", "outstanding", "beautiful", "happy",
        "recommend", "recommended", "worth", "solid", "charming", "engaging", "satisfying", "pleasant", "fun",
        "bagus", "baik", "hebat", "luar biasa", "mantap", "keren", "indah", "menyenangkan", "memuaskan",
        "puas", "suka", "senang", "cepat", "ramah", "bermanfaat", "berkualitas", "rekomendasi", "murah",
        "nyaman", "sempurna", "menarik", "cocok", "terbaik",
    };

    /// <summary>Words carrying negative sentiment, English and Indonesian.</summary>
    public static IReadOnlySet<string> NegativeLexicon { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "bad", "terrible", "awful", "horrible", "worst", "poor", "boring", "disappointing", "disappointed",
        "waste", "hate", "hated", "dull", "weak", "confusing", "annoying", "broken", "useless", "slow",
        "overpriced", "bland", "mediocre", "flawed", "painful", "regret",
        "buruk", "jelek", "parah", "mengecewakan", "kecewa", "lambat", "rusak", "gagal", "mahal", "susah",
        "sulit", "membosankan", "bosan", "tidak", "kurang", "keluhan", "menyebalkan", "payah", "percuma",
        "rugi", "kotor", "kasar",
    };

    private static readonly HashSet<string> Negations = new(StringComparer.OrdinalIgnoreCase)
    {
        "not", "no", "never", "none", "cannot", "can't", "don't", "doesn't", "didn't", "isn't", "wasn't",
        "tidak", "bukan", "tak", "jangan", "belum", "kurang",
    };

    private static readonly HashSet<string> Intensifiers = new(StringComparer.OrdinalIgnoreCase)
    {
        "very", "really", "extremely", "highly", "so", "too", "absolutely", "totally",
        "sangat", "sekali", "banget", "amat", "benar-benar", "paling",
    };

    /// <summary>True when a supervised model has been trained.</summary>
    public bool IsTrained => _classifier?.IsTrained ?? false;

    /// <summary>Trains a supervised model, which then takes precedence over the lexicon.</summary>
    public SentimentAnalyzer Train(IReadOnlyList<string> documents, IReadOnlyList<string> labels)
    {
        _classifier = new TextClassifier().Train(documents, labels);
        return this;
    }

    /// <summary>Classifies the sentiment of one document.</summary>
    public Prediction Analyze(string text)
        => _classifier is { IsTrained: true } ? _classifier.Predict(text) : AnalyzeWithLexicon(text);

    /// <summary>
    /// Scores sentiment from the lexicon alone, handling negation and intensifiers.
    /// </summary>
    public Prediction AnalyzeWithLexicon(string text)
    {
        var tokens = _tokenizer.Tokenize(text);
        var score = 0.0;
        var hits = 0;

        for (var i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            var polarity = PositiveLexicon.Contains(token) ? 1.0
                : NegativeLexicon.Contains(token) ? -1.0
                : 0.0;
            if (polarity == 0.0) continue;

            // Look back a couple of tokens for a negation or an intensifier.
            var multiplier = 1.0;
            for (var back = 1; back <= 2 && i - back >= 0; back++)
            {
                var previous = tokens[i - back];
                if (Negations.Contains(previous)) multiplier *= -1.0;
                else if (Intensifiers.Contains(previous)) multiplier *= 1.5;
            }

            score += polarity * multiplier;
            hits++;
        }

        var normalized = hits == 0 ? 0.0 : score / hits;
        var probability = MathUtil.Sigmoid(normalized * 2.0);
        var label = Math.Abs(normalized) < 1e-9 ? "neutral" : normalized > 0 ? "positive" : "negative";

        return new Prediction(label, Math.Max(probability, 1 - probability),
            new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["positive"] = probability,
                ["negative"] = 1 - probability,
            });
    }

    /// <summary>Classifies several documents.</summary>
    public IReadOnlyList<Prediction> AnalyzeBatch(IReadOnlyList<string> texts) => texts.Select(Analyze).ToList();

    /// <summary>The words that drove a supervised model toward a label.</summary>
    public IReadOnlyList<(string Term, double Weight)> Explain(string label, int count = 15)
        => _classifier?.TopFeatures(label, count)
           ?? throw new InvalidOperationException("Explanations require a trained model.");
}

/// <summary>An entity found in text.</summary>
/// <param name="Text">The matched surface form.</param>
/// <param name="Type">Entity type, e.g. <c>PERSON</c> or <c>LOCATION</c>.</param>
/// <param name="Start">Character offset where the match begins.</param>
/// <param name="Length">Length of the match in characters.</param>
public sealed record Entity(string Text, string Type, int Start, int Length)
{
    /// <inheritdoc />
    public override string ToString() => $"{Text} [{Type}]";
}

/// <summary>
/// Named entity recognition driven by gazetteers, capitalisation and contextual trigger words.
/// </summary>
/// <remarks>
/// <b>This is a rule-based recogniser, not a trained sequence model.</b> It reliably finds
/// entities that match its gazetteers or sit next to a trigger word ("PT", "Dr.", "in", "di"),
/// and it will miss novel entities that a CRF or a fine-tuned transformer would catch. It is
/// included because it needs no training data and is fully inspectable - extend it by adding to
/// <see cref="AddGazetteer"/> rather than by retraining.
/// </remarks>
public sealed class NamedEntityRecognizer
{
    private readonly Dictionary<string, HashSet<string>> _gazetteers = new(StringComparer.Ordinal);

    private static readonly HashSet<string> OrganizationTriggers = new(StringComparer.OrdinalIgnoreCase)
    {
        "pt", "cv", "tbk", "inc", "ltd", "llc", "corp", "corporation", "company", "university", "universitas",
        "institute", "institut", "foundation", "yayasan", "studios", "group", "holdings", "bank",
    };

    private static readonly HashSet<string> PersonTriggers = new(StringComparer.OrdinalIgnoreCase)
    {
        "mr", "mrs", "ms", "dr", "prof", "sir", "bapak", "pak", "ibu", "bu", "kang", "mas", "mbak", "haji",
    };

    private static readonly HashSet<string> LocationTriggers = new(StringComparer.OrdinalIgnoreCase)
    {
        "in", "at", "from", "to", "di", "ke", "dari", "kota", "kabupaten", "provinsi", "desa", "jalan",
    };

    /// <summary>Creates a recogniser with a small built-in gazetteer.</summary>
    public NamedEntityRecognizer()
    {
        AddGazetteer("LOCATION",
        [
            "Jakarta", "Bandung", "Surabaya", "Yogyakarta", "Medan", "Semarang", "Makassar", "Denpasar",
            "Bali", "Indonesia", "Singapore", "Malaysia", "London", "Paris", "Tokyo", "New York",
        ]);
        AddGazetteer("ORGANIZATION",
        [
            "Gravicode Studios", "Microsoft", "Google", "Amazon", "Apple", "Telkom", "Pertamina", "Gojek",
        ]);
    }

    /// <summary>Adds known surface forms for an entity type.</summary>
    public NamedEntityRecognizer AddGazetteer(string type, IEnumerable<string> entries)
    {
        if (!_gazetteers.TryGetValue(type, out var set))
            _gazetteers[type] = set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var entry in entries) set.Add(entry);
        return this;
    }

    /// <summary>Finds entities in <paramref name="text"/>, longest match first, without overlaps.</summary>
    public IReadOnlyList<Entity> Recognize(string text)
    {
        var found = new List<Entity>();

        // Gazetteer matches are the most reliable signal, so they run first and claim their spans.
        foreach (var (type, entries) in _gazetteers)
            foreach (var entry in entries.OrderByDescending(e => e.Length))
            {
                var start = 0;
                while ((start = text.IndexOf(entry, start, StringComparison.OrdinalIgnoreCase)) >= 0)
                {
                    if (IsWholeWord(text, start, entry.Length)) found.Add(new Entity(text[start..(start + entry.Length)], type, start, entry.Length));
                    start += entry.Length;
                }
            }

        found.AddRange(FindCapitalisedSpans(text));
        found.AddRange(FindNumericEntities(text));

        // Resolve overlaps by preferring the longest span, then the earliest.
        var accepted = new List<Entity>();
        foreach (var entity in found.OrderByDescending(e => e.Length).ThenBy(e => e.Start))
            if (!accepted.Any(a => entity.Start < a.Start + a.Length && a.Start < entity.Start + entity.Length))
                accepted.Add(entity);

        return accepted.OrderBy(e => e.Start).ToList();
    }

    private static bool IsWholeWord(string text, int start, int length)
    {
        var before = start == 0 || !char.IsLetterOrDigit(text[start - 1]);
        var end = start + length;
        var after = end >= text.Length || !char.IsLetterOrDigit(text[end]);
        return before && after;
    }

    private static IEnumerable<Entity> FindCapitalisedSpans(string text)
    {
        var words = new List<(string Word, int Start)>();
        var index = 0;
        while (index < text.Length)
        {
            if (!char.IsLetter(text[index])) { index++; continue; }
            var start = index;
            while (index < text.Length && (char.IsLetterOrDigit(text[index]) || text[index] == '.')) index++;
            words.Add((text[start..index].TrimEnd('.'), start));
        }

        for (var i = 0; i < words.Count; i++)
        {
            var (word, start) = words[i];
            if (word.Length == 0 || !char.IsUpper(word[0])) continue;
            // A capitalised word at the very start of the text is usually just a sentence opener.
            if (i == 0) continue;

            var previous = words[i - 1].Word;
            var length = word.Length;
            var j = i + 1;
            while (j < words.Count && words[j].Word.Length > 0 && char.IsUpper(words[j].Word[0]))
            {
                length = words[j].Start + words[j].Word.Length - start;
                j++;
            }

            var span = text.Substring(start, Math.Min(length, text.Length - start)).Trim();
            var type = PersonTriggers.Contains(previous) ? "PERSON"
                : LocationTriggers.Contains(previous) ? "LOCATION"
                : OrganizationTriggers.Contains(previous) ? "ORGANIZATION"
                : j - i > 1 ? "PERSON"
                : null;

            if (type is not null) yield return new Entity(span, type, start, span.Length);
            i = j - 1;
        }
    }

    private static IEnumerable<Entity> FindNumericEntities(string text)
    {
        var index = 0;
        while (index < text.Length)
        {
            if (!char.IsDigit(text[index])) { index++; continue; }
            var start = index;
            while (index < text.Length && (char.IsDigit(text[index]) || text[index] is '.' or ',' or '-' or '/'))
                index++;

            var span = text[start..index].TrimEnd('.', ',');
            if (span.Length == 0) continue;

            var type = span.Contains('-') || span.Contains('/') || span.Length == 4 ? "DATE" : "NUMBER";
            yield return new Entity(span, type, start, span.Length);
        }
    }
}

/// <summary>
/// Extractive summarisation by TextRank: sentences are ranked by how central they are in a graph
/// of sentence similarities.
/// </summary>
/// <remarks>
/// Extractive rather than abstractive: the summary is built from sentences that appear verbatim in
/// the source, so it can never hallucinate. The ranking is PageRank over a graph whose edges are
/// content-word overlap, which surfaces the sentences the rest of the document agrees with.
/// </remarks>
public sealed class TextRankSummarizer(double damping = 0.85, int iterations = 60, double tolerance = 1e-6)
{
    private readonly ITokenizer _tokenizer = new RegexTokenizer();

    /// <summary>Returns the <paramref name="sentenceCount"/> highest-ranked sentences, in original order.</summary>
    public IReadOnlyList<string> Summarize(string text, int sentenceCount = 3,
        IReadOnlySet<string>? stopWords = null)
    {
        var sentences = SentenceSplitter.Split(text);
        if (sentences.Count <= sentenceCount) return sentences;

        var stop = stopWords ?? StopWords.Bilingual;
        var tokenized = sentences
            .Select(s => new HashSet<string>(_tokenizer.Tokenize(s).Where(t => !stop.Contains(t)), StringComparer.Ordinal))
            .ToArray();

        var n = sentences.Count;
        var weights = new double[n, n];
        for (var i = 0; i < n; i++)
            for (var j = i + 1; j < n; j++)
            {
                var similarity = Overlap(tokenized[i], tokenized[j]);
                weights[i, j] = weights[j, i] = similarity;
            }

        var scores = Enumerable.Repeat(1.0 / n, n).ToArray();
        for (var iteration = 0; iteration < iterations; iteration++)
        {
            var updated = new double[n];
            var delta = 0.0;
            for (var i = 0; i < n; i++)
            {
                var acc = 0.0;
                for (var j = 0; j < n; j++)
                {
                    if (i == j || weights[j, i] == 0) continue;
                    var outbound = 0.0;
                    for (var k = 0; k < n; k++) outbound += weights[j, k];
                    if (outbound > 0) acc += weights[j, i] / outbound * scores[j];
                }
                updated[i] = (1 - damping) / n + damping * acc;
                delta += Math.Abs(updated[i] - scores[i]);
            }
            scores = updated;
            if (delta < tolerance) break;
        }

        return Enumerable.Range(0, n)
            .OrderByDescending(i => scores[i])
            .Take(sentenceCount)
            .OrderBy(i => i)
            .Select(i => sentences[i])
            .ToList();
    }

    /// <summary>Normalised content-word overlap between two sentences.</summary>
    private static double Overlap(HashSet<string> a, HashSet<string> b)
    {
        if (a.Count == 0 || b.Count == 0) return 0.0;
        var shared = a.Intersect(b, StringComparer.Ordinal).Count();
        // The log denominator is TextRank's length normalisation: it stops long sentences
        // dominating simply by containing more words.
        var denominator = Math.Log(a.Count + 1) + Math.Log(b.Count + 1);
        return denominator < 1e-9 ? 0.0 : shared / denominator;
    }
}

/// <summary>Ranks the terms that best characterise a document against a corpus.</summary>
public sealed class KeywordExtractor(VectorizerOptions? options = null)
{
    private readonly TfidfVectorizer _vectorizer = new(options ?? new VectorizerOptions
    {
        StopWords = StopWords.Bilingual,
        MinNGram = 1,
        MaxNGram = 2,
    }, sublinearTf: true);

    /// <summary>Learns the corpus statistics keywords are measured against.</summary>
    public KeywordExtractor Fit(IReadOnlyList<string> corpus)
    {
        _vectorizer.Fit(corpus);
        return this;
    }

    /// <summary>The highest TF-IDF terms of one document.</summary>
    public IReadOnlyList<(string Term, double Score)> Extract(string document, int count = 10)
    {
        var matrix = _vectorizer.Transform([document]);
        return _vectorizer.TopTerms(matrix, 0, count).Select(t => (t.Term, t.Weight)).ToList();
    }
}

/// <summary>
/// Distinguishes English from Bahasa Indonesia by which stop word list the text matches better.
/// </summary>
public static class LanguageDetector
{
    /// <summary>Returns <c>en</c>, <c>id</c> or <c>unknown</c>, with a confidence.</summary>
    public static (string Language, double Confidence) Detect(string text)
    {
        var tokens = new RegexTokenizer().Tokenize(text);
        if (tokens.Count == 0) return ("unknown", 0.0);

        // Words shared by both lists carry no signal, so only distinctive hits are counted.
        var english = tokens.Count(t => StopWords.English.Contains(t) && !StopWords.Indonesian.Contains(t));
        var indonesian = tokens.Count(t => StopWords.Indonesian.Contains(t) && !StopWords.English.Contains(t));

        var total = english + indonesian;
        if (total == 0) return ("unknown", 0.0);
        return english >= indonesian
            ? ("en", (double)english / total)
            : ("id", (double)indonesian / total);
    }
}
