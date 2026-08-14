using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Linguistics;
using Gravicode.Science.GraviText.Tokenization;

namespace Gravicode.Science.GraviText.Vectorization;

/// <summary>Shared settings for the bag-of-words vectorizers.</summary>
public sealed class VectorizerOptions
{
    /// <summary>Tokenizer applied to each document.</summary>
    public ITokenizer Tokenizer { get; set; } = new RegexTokenizer();

    /// <summary>Stop words to drop, or <c>null</c> to keep everything.</summary>
    public IReadOnlySet<string>? StopWords { get; set; }

    /// <summary>Optional stemming or lemmatization applied per token.</summary>
    public Func<string, string>? Stemmer { get; set; }

    /// <summary>Smallest n-gram size.</summary>
    public int MinNGram { get; set; } = 1;

    /// <summary>Largest n-gram size.</summary>
    public int MaxNGram { get; set; } = 1;

    /// <summary>Terms appearing in fewer documents than this are dropped.</summary>
    public int MinDocumentFrequency { get; set; } = 1;

    /// <summary>Terms appearing in more than this fraction of documents are dropped.</summary>
    public double MaxDocumentFrequencyRatio { get; set; } = 1.0;

    /// <summary>Cap on vocabulary size, keeping the most frequent terms.</summary>
    public int MaxFeatures { get; set; }

    /// <summary>Turns a document into its final term list.</summary>
    internal IReadOnlyList<string> Analyze(string document)
    {
        var tokens = Tokenizer.Tokenize(document);
        if (StopWords is not null) tokens = Linguistics.StopWords.Remove(tokens, StopWords);
        if (Stemmer is not null) tokens = tokens.Select(Stemmer).ToList();
        return MinNGram == 1 && MaxNGram == 1 ? tokens : NGrams.Range(tokens, MinNGram, MaxNGram);
    }
}

/// <summary>
/// Turns documents into term-count vectors.
/// </summary>
/// <remarks>
/// The document-frequency bounds are the useful part: <see cref="VectorizerOptions.MinDocumentFrequency"/>
/// drops typos and one-off terms that only add noise and dimensions, while
/// <see cref="VectorizerOptions.MaxDocumentFrequencyRatio"/> drops terms so common they carry no
/// signal - a corpus-specific stop word list, learned rather than hand-written.
/// </remarks>
public sealed class CountVectorizer(VectorizerOptions? options = null)
{
    private readonly VectorizerOptions _options = options ?? new VectorizerOptions();
    private string[] _vocabulary = [];
    private Dictionary<string, int> _index = new(StringComparer.Ordinal);

    /// <summary>The learned terms, in column order.</summary>
    public IReadOnlyList<string> Vocabulary => _vocabulary;

    /// <summary>Number of columns produced.</summary>
    public int FeatureCount => _vocabulary.Length;

    /// <summary>Number of documents each term appeared in.</summary>
    public IReadOnlyDictionary<string, int> DocumentFrequency { get; private set; } =
        new Dictionary<string, int>(StringComparer.Ordinal);

    /// <summary>Learns the vocabulary from a corpus.</summary>
    public CountVectorizer Fit(IReadOnlyList<string> documents)
    {
        var documentFrequency = new Dictionary<string, int>(StringComparer.Ordinal);
        var totalFrequency = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var document in documents)
        {
            var terms = _options.Analyze(document);
            foreach (var term in terms.Distinct(StringComparer.Ordinal))
            {
                documentFrequency.TryGetValue(term, out var df);
                documentFrequency[term] = df + 1;
            }
            foreach (var term in terms)
            {
                totalFrequency.TryGetValue(term, out var tf);
                totalFrequency[term] = tf + 1;
            }
        }

        var maxDocuments = _options.MaxDocumentFrequencyRatio * documents.Count;
        var kept = documentFrequency
            .Where(kv => kv.Value >= _options.MinDocumentFrequency && kv.Value <= maxDocuments)
            .Select(kv => kv.Key);

        if (_options.MaxFeatures > 0)
            kept = kept.OrderByDescending(t => totalFrequency[t]).Take(_options.MaxFeatures);

        _vocabulary = kept.OrderBy(t => t, StringComparer.Ordinal).ToArray();
        _index = _vocabulary.Select((t, i) => (t, i)).ToDictionary(p => p.t, p => p.i, StringComparer.Ordinal);
        DocumentFrequency = documentFrequency;
        return this;
    }

    /// <summary>Encodes documents as a dense count matrix.</summary>
    public NdArray Transform(IReadOnlyList<string> documents)
    {
        RequireFitted();
        var matrix = NdArray.Zeros(documents.Count, FeatureCount);
        for (var i = 0; i < documents.Count; i++)
            foreach (var term in _options.Analyze(documents[i]))
                if (_index.TryGetValue(term, out var j)) matrix[i, j] += 1.0;
        return matrix;
    }

    /// <summary>Encodes documents as a sparse count matrix, which is what real corpora need.</summary>
    public SparseMatrix TransformSparse(IReadOnlyList<string> documents)
    {
        RequireFitted();
        var triplets = new List<(int, int, double)>();
        for (var i = 0; i < documents.Count; i++)
        {
            var counts = new Dictionary<int, double>();
            foreach (var term in _options.Analyze(documents[i]))
                if (_index.TryGetValue(term, out var j))
                {
                    counts.TryGetValue(j, out var c);
                    counts[j] = c + 1;
                }
            foreach (var (j, c) in counts) triplets.Add((i, j, c));
        }
        return SparseMatrix.FromTriplets(documents.Count, FeatureCount, triplets);
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(IReadOnlyList<string> documents) { Fit(documents); return Transform(documents); }

    private void RequireFitted()
    {
        if (_vocabulary.Length == 0)
            throw new InvalidOperationException("The vectorizer has not been fitted, or the vocabulary is empty.");
    }
}

/// <summary>
/// Turns documents into TF-IDF vectors.
/// </summary>
/// <remarks>
/// Term frequency alone rewards long documents and common words. Multiplying by the inverse
/// document frequency, <c>log((1 + n) / (1 + df)) + 1</c>, down-weights terms that appear
/// everywhere and rewards terms that distinguish one document from the rest; the final L2
/// normalisation makes the vectors comparable by cosine similarity regardless of length.
/// </remarks>
public sealed class TfidfVectorizer(VectorizerOptions? options = null, bool sublinearTf = false, bool normalize = true)
{
    private readonly CountVectorizer _counts = new(options);
    private double[] _idf = [];

    /// <summary>The learned terms, in column order.</summary>
    public IReadOnlyList<string> Vocabulary => _counts.Vocabulary;

    /// <summary>Number of columns produced.</summary>
    public int FeatureCount => _counts.FeatureCount;

    /// <summary>The inverse document frequency of each term, in column order.</summary>
    public IReadOnlyList<double> InverseDocumentFrequency => _idf;

    /// <summary>Learns the vocabulary and the IDF weights.</summary>
    public TfidfVectorizer Fit(IReadOnlyList<string> documents)
    {
        _counts.Fit(documents);
        var n = documents.Count;
        _idf = new double[FeatureCount];

        for (var j = 0; j < FeatureCount; j++)
        {
            var term = _counts.Vocabulary[j];
            _counts.DocumentFrequency.TryGetValue(term, out var df);
            // The +1s are smoothing: they keep the weight finite for a term in every document
            // and for a term the transform sees but the fit did not.
            _idf[j] = Math.Log((1.0 + n) / (1.0 + df)) + 1.0;
        }
        return this;
    }

    /// <summary>Encodes documents as a TF-IDF matrix.</summary>
    public NdArray Transform(IReadOnlyList<string> documents)
    {
        var counts = _counts.Transform(documents);
        var result = NdArray.Zeros(counts.Shape[0], counts.Shape[1]);

        for (var i = 0; i < counts.Shape[0]; i++)
        {
            for (var j = 0; j < counts.Shape[1]; j++)
            {
                var tf = counts[i, j];
                if (tf == 0) continue;
                // Sublinear scaling stops a term repeated 100 times counting 100x a single mention.
                result[i, j] = (sublinearTf ? 1.0 + Math.Log(tf) : tf) * _idf[j];
            }

            if (!normalize) continue;
            var norm = 0.0;
            for (var j = 0; j < counts.Shape[1]; j++) norm += result[i, j] * result[i, j];
            norm = Math.Sqrt(norm);
            if (norm < 1e-12) continue;
            for (var j = 0; j < counts.Shape[1]; j++) result[i, j] /= norm;
        }
        return result;
    }

    /// <summary>
    /// Encodes documents as a sparse TF-IDF matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The same numbers as <see cref="Transform"/>, in CSR. For text this is not a micro-optimisation:
    /// a 30,000-word vocabulary is around 0.1% non-zero, so the dense form spends a thousand times
    /// the memory storing zeros. It is what makes a realistic vocabulary trainable at all — see
    /// <c>SparseLogisticRegression</c>, which consumes this directly.
    /// </para>
    /// <para>
    /// Normalisation is computed over the row's non-zeros, which is the same value the dense path
    /// gets: the zeros contribute nothing to a Euclidean norm.
    /// </para>
    /// </remarks>
    public SparseMatrix TransformSparse(IReadOnlyList<string> documents)
    {
        var counts = _counts.TransformSparse(documents);
        var triplets = new List<(int Row, int Column, double Value)>(counts.NonZeroCount);

        for (var i = 0; i < counts.Rows; i++)
        {
            var row = new List<(int Column, double Value)>();

            foreach (var (column, tf) in counts.Row(i))
            {
                if (tf == 0) continue;
                // Sublinear scaling stops a term repeated 100 times counting 100x a single mention.
                row.Add((column, (sublinearTf ? 1.0 + Math.Log(tf) : tf) * _idf[column]));
            }

            if (normalize)
            {
                var norm = Math.Sqrt(row.Sum(e => e.Value * e.Value));
                if (norm >= 1e-12)
                    for (var k = 0; k < row.Count; k++) row[k] = (row[k].Column, row[k].Value / norm);
            }

            foreach (var (column, value) in row) triplets.Add((i, column, value));
        }

        return SparseMatrix.FromTriplets(documents.Count, _counts.FeatureCount, triplets);
    }

    /// <summary>Fits and transforms in one call.</summary>
    public NdArray FitTransform(IReadOnlyList<string> documents) { Fit(documents); return Transform(documents); }

    /// <summary>Fits and transforms to a sparse matrix in one call.</summary>
    public SparseMatrix FitTransformSparse(IReadOnlyList<string> documents)
    {
        Fit(documents);
        return TransformSparse(documents);
    }

    /// <summary>The highest-weighted terms of one encoded document.</summary>
    public IReadOnlyList<(string Term, double Weight)> TopTerms(NdArray matrix, int row, int count = 10)
        => Enumerable.Range(0, FeatureCount)
            .Select(j => (Term: Vocabulary[j], Weight: matrix[row, j]))
            .Where(t => t.Weight > 0)
            .OrderByDescending(t => t.Weight)
            .Take(count)
            .ToList();
}

/// <summary>Similarity measures over document or word vectors.</summary>
public static class Similarity
{
    /// <summary>Cosine of the angle between two vectors; 1 means identical direction.</summary>
    public static double Cosine(NdArray a, NdArray b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Size; i++)
        {
            dot += a.At(i) * b.At(i);
            na += a.At(i) * a.At(i);
            nb += b.At(i) * b.At(i);
        }
        var denominator = Math.Sqrt(na) * Math.Sqrt(nb);
        return denominator < 1e-12 ? 0.0 : dot / denominator;
    }

    /// <summary>Cosine similarity between two rows of a matrix.</summary>
    public static double Cosine(NdArray matrix, int rowA, int rowB)
        => Cosine(matrix.Row(rowA).Copy(), matrix.Row(rowB).Copy());

    /// <summary>Overlap of two token sets, as intersection over union.</summary>
    public static double Jaccard(IEnumerable<string> a, IEnumerable<string> b)
    {
        var setA = new HashSet<string>(a, StringComparer.Ordinal);
        var setB = new HashSet<string>(b, StringComparer.Ordinal);
        if (setA.Count == 0 && setB.Count == 0) return 1.0;
        var intersection = setA.Intersect(setB).Count();
        return (double)intersection / (setA.Count + setB.Count - intersection);
    }

    /// <summary>Edit distance between two strings.</summary>
    public static int Levenshtein(string a, string b)
    {
        var previous = new int[b.Length + 1];
        var current = new int[b.Length + 1];
        for (var j = 0; j <= b.Length; j++) previous[j] = j;

        for (var i = 1; i <= a.Length; i++)
        {
            current[0] = i;
            for (var j = 1; j <= b.Length; j++)
                current[j] = Math.Min(
                    Math.Min(previous[j] + 1, current[j - 1] + 1),
                    previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            (previous, current) = (current, previous);
        }
        return previous[b.Length];
    }

    /// <summary>The rows most similar to <paramref name="query"/>, most similar first.</summary>
    public static IReadOnlyList<(int Index, double Score)> MostSimilar(NdArray matrix, NdArray query, int top = 5)
        => Enumerable.Range(0, matrix.Shape[0])
            .Select(i => (Index: i, Score: Cosine(matrix.Row(i).Copy(), query)))
            .OrderByDescending(t => t.Score)
            .Take(top)
            .ToList();
}
