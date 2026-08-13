using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Tokenization;

namespace Gravicode.Science.GraviText.Embeddings;

/// <summary>
/// A learned mapping from words to dense vectors, with the lookup and analogy operations that
/// make such vectors useful.
/// </summary>
public sealed class WordEmbeddings
{
    private readonly Dictionary<string, int> _index;
    private readonly string[] _words;
    private readonly NdArray _vectors;

    /// <summary>Wraps a vocabulary and its matching vector matrix.</summary>
    public WordEmbeddings(IReadOnlyList<string> words, NdArray vectors)
    {
        if (vectors.Shape[0] != words.Count)
            throw new ArgumentException($"Got {words.Count} words but {vectors.Shape[0]} vectors.");
        _words = words.ToArray();
        _vectors = vectors;
        _index = _words.Select((w, i) => (w, i)).ToDictionary(p => p.w, p => p.i, StringComparer.Ordinal);
    }

    /// <summary>Number of words.</summary>
    public int Count => _words.Length;

    /// <summary>Length of each vector.</summary>
    public int Dimensions => _vectors.Shape[1];

    /// <summary>The words, in row order.</summary>
    public IReadOnlyList<string> Words => _words;

    /// <summary>The full matrix, one word per row.</summary>
    public NdArray Matrix => _vectors;

    /// <summary>True when the word has a vector.</summary>
    public bool Contains(string word) => _index.ContainsKey(word);

    /// <summary>The vector for a word.</summary>
    public NdArray this[string word] => _index.TryGetValue(word, out var i)
        ? _vectors.Row(i).Copy()
        : throw new KeyNotFoundException($"'{word}' is not in the embedding vocabulary.");

    /// <summary>The vector for a word, or <c>null</c> when it is unknown.</summary>
    public NdArray? TryGet(string word) => _index.TryGetValue(word, out var i) ? _vectors.Row(i).Copy() : null;

    /// <summary>Cosine similarity between two words.</summary>
    public double Similarity(string a, string b) => Vectorization.Similarity.Cosine(this[a], this[b]);

    /// <summary>The words closest to <paramref name="word"/>, excluding itself.</summary>
    public IReadOnlyList<(string Word, double Score)> MostSimilar(string word, int top = 10)
        => MostSimilar(this[word], top, [word]);

    /// <summary>The words closest to an arbitrary vector.</summary>
    public IReadOnlyList<(string Word, double Score)> MostSimilar(NdArray vector, int top = 10,
        IReadOnlyCollection<string>? exclude = null)
    {
        var skip = exclude is null ? [] : new HashSet<string>(exclude, StringComparer.Ordinal);
        return Enumerable.Range(0, Count)
            .Where(i => !skip.Contains(_words[i]))
            .Select(i => (Word: _words[i], Score: Vectorization.Similarity.Cosine(_vectors.Row(i).Copy(), vector)))
            .OrderByDescending(t => t.Score)
            .Take(top)
            .ToList();
    }

    /// <summary>
    /// Solves the analogy <c>a is to b as c is to ?</c> by looking near <c>b - a + c</c>.
    /// </summary>
    public IReadOnlyList<(string Word, double Score)> Analogy(string a, string b, string c, int top = 5)
        => MostSimilar(this[b] - this[a] + this[c], top, [a, b, c]);

    /// <summary>
    /// Returns a copy with the mean vector subtracted from every row.
    /// </summary>
    /// <remarks>
    /// Trained word vectors share a large common component - every word gets pushed in the same
    /// direction by the negative sampling it has in common with the rest of the vocabulary. That
    /// component carries no meaning but dominates the cosine, so on small corpora every pair can
    /// look ~1.0 similar. Removing it (Mu and Viswanath's "all-but-the-top") restores the
    /// differences the specific components actually encode, and is standard post-processing.
    /// </remarks>
    public WordEmbeddings RemoveCommonComponent()
    {
        var mean = new double[Dimensions];
        for (var i = 0; i < Count; i++)
            for (var d = 0; d < Dimensions; d++) mean[d] += _vectors[i, d];
        for (var d = 0; d < Dimensions; d++) mean[d] /= Count;

        var centred = NdArray.Zeros(Count, Dimensions);
        for (var i = 0; i < Count; i++)
            for (var d = 0; d < Dimensions; d++) centred[i, d] = _vectors[i, d] - mean[d];

        return new WordEmbeddings(_words, centred);
    }

    /// <summary>The mean of a document's word vectors, the simplest usable sentence embedding.</summary>
    public NdArray Average(IReadOnlyList<string> tokens)
    {
        var sum = NdArray.Zeros(Dimensions);
        var counted = 0;
        foreach (var token in tokens)
        {
            var vector = TryGet(token);
            if (vector is null) continue;
            sum += vector;
            counted++;
        }
        return counted == 0 ? sum : sum / counted;
    }

    /// <summary>Writes the embeddings in the standard word2vec text format.</summary>
    public void Save(string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine($"{Count} {Dimensions}");
        for (var i = 0; i < Count; i++)
        {
            var values = string.Join(' ', Enumerable.Range(0, Dimensions).Select(j => _vectors[i, j].ToString("G9")));
            writer.WriteLine($"{_words[i]} {values}");
        }
    }

    /// <summary>Reads embeddings in the word2vec text format.</summary>
    public static WordEmbeddings Load(string path)
    {
        using var reader = new StreamReader(path);
        var header = reader.ReadLine()?.Split(' ') ?? throw new InvalidDataException("Empty embedding file.");
        var count = int.Parse(header[0]);
        var dimensions = int.Parse(header[1]);

        var words = new List<string>(count);
        var matrix = NdArray.Zeros(count, dimensions);

        for (var i = 0; i < count; i++)
        {
            var parts = reader.ReadLine()?.Split(' ') ?? throw new InvalidDataException($"Expected {count} rows.");
            words.Add(parts[0]);
            for (var j = 0; j < dimensions; j++) matrix[i, j] = double.Parse(parts[j + 1]);
        }
        return new WordEmbeddings(words, matrix);
    }
}

/// <summary>
/// Word2Vec skip-gram with negative sampling.
/// </summary>
/// <remarks>
/// The model learns by prediction: for each (centre, context) pair seen in the corpus it pushes
/// their vectors together, and for a handful of randomly drawn "negative" words it pushes them
/// apart. Two details do most of the work. Negative words are drawn from the unigram distribution
/// raised to the 3/4 power, which samples rare words more often than their frequency alone would.
/// And frequent words are randomly dropped by the subsampling rule, which both speeds training up
/// and stops "the" dominating every context window.
/// </remarks>
public sealed class Word2Vec(
    int dimensions = 100,
    int windowSize = 5,
    int minCount = 5,
    int negativeSamples = 5,
    double learningRate = 0.025,
    int epochs = 5,
    double subsampleThreshold = 1e-3,
    int seed = 42)
{
    private string[] _words = [];
    private Dictionary<string, int> _index = new(StringComparer.Ordinal);
    private NdArray _input = NdArray.Zeros(0, 0);
    private NdArray _output = NdArray.Zeros(0, 0);
    private double[] _samplingTable = [];
    private double[] _keepProbability = [];

    /// <summary>Vector length.</summary>
    public int Dimensions { get; } = dimensions;

    /// <summary>Context window radius.</summary>
    public int WindowSize { get; } = windowSize;

    /// <summary>Words seen fewer times than this are ignored.</summary>
    public int MinCount { get; } = minCount;

    /// <summary>Vocabulary size after filtering.</summary>
    public int VocabularySize => _words.Length;

    /// <summary>Average training loss of the last epoch.</summary>
    public double FinalLoss { get; private set; }

    /// <summary>Trains on a tokenized corpus.</summary>
    public WordEmbeddings Train(IReadOnlyList<IReadOnlyList<string>> corpus)
    {
        BuildVocabulary(corpus);
        if (VocabularySize == 0)
            throw new InvalidOperationException($"No word survived the minimum count of {MinCount}.");

        var rng = new GraviRandom(seed);
        // Small uniform init around zero: the model is symmetric at exactly zero and cannot learn.
        _input = rng.Uniform(-0.5 / Dimensions, 0.5 / Dimensions, VocabularySize, Dimensions);
        _output = NdArray.Zeros(VocabularySize, Dimensions);

        var encoded = corpus
            .Select(document => document.Where(_index.ContainsKey).Select(w => _index[w]).ToArray())
            .Where(document => document.Length > 1)
            .ToArray();

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            // Linear decay of the step size, as in the original implementation.
            var alpha = Math.Max(learningRate * (1.0 - (double)epoch / epochs), learningRate * 0.0001);
            var loss = 0.0;
            var pairs = 0;

            foreach (var document in encoded)
            {
                for (var position = 0; position < document.Length; position++)
                {
                    var centre = document[position];
                    if (rng.NextDouble() > _keepProbability[centre]) continue;

                    // A randomly shrunk window weights nearby words more heavily on average.
                    var span = 1 + rng.Next(WindowSize);
                    for (var offset = -span; offset <= span; offset++)
                    {
                        if (offset == 0) continue;
                        var contextPosition = position + offset;
                        if (contextPosition < 0 || contextPosition >= document.Length) continue;

                        loss += TrainPair(centre, document[contextPosition], alpha, rng);
                        pairs++;
                    }
                }
            }
            FinalLoss = pairs == 0 ? 0.0 : loss / pairs;
        }

        return new WordEmbeddings(_words, _input);
    }

    /// <summary>Trains on raw text, tokenizing it first.</summary>
    public WordEmbeddings Train(IReadOnlyList<string> documents, ITokenizer? tokenizer = null)
    {
        var tokenize = tokenizer ?? new RegexTokenizer();
        return Train(documents.Select(tokenize.Tokenize).ToList());
    }

    private double TrainPair(int centre, int context, double alpha, GraviRandom rng)
    {
        var hidden = new double[Dimensions];
        for (var d = 0; d < Dimensions; d++) hidden[d] = _input[centre, d];

        var gradient = new double[Dimensions];
        var loss = 0.0;

        // One positive example, then `negativeSamples` drawn from the noise distribution.
        for (var sample = 0; sample <= negativeSamples; sample++)
        {
            int target;
            double label;
            if (sample == 0) { target = context; label = 1.0; }
            else
            {
                target = DrawNegative(rng);
                if (target == context) continue;
                label = 0.0;
            }

            var score = 0.0;
            for (var d = 0; d < Dimensions; d++) score += hidden[d] * _output[target, d];
            var prediction = MathUtil.Sigmoid(score);

            var error = (label - prediction) * alpha;
            for (var d = 0; d < Dimensions; d++)
            {
                gradient[d] += error * _output[target, d];
                _output[target, d] += error * hidden[d];
            }

            var clamped = Math.Clamp(prediction, 1e-12, 1 - 1e-12);
            loss -= label > 0 ? Math.Log(clamped) : Math.Log(1 - clamped);
        }

        for (var d = 0; d < Dimensions; d++) _input[centre, d] += gradient[d];
        return loss;
    }

    private int DrawNegative(GraviRandom rng)
    {
        // Binary search over the cumulative noise distribution.
        var target = rng.NextDouble();
        var low = 0;
        var high = _samplingTable.Length - 1;
        while (low < high)
        {
            var mid = (low + high) / 2;
            if (_samplingTable[mid] < target) low = mid + 1; else high = mid;
        }
        return low;
    }

    private void BuildVocabulary(IReadOnlyList<IReadOnlyList<string>> corpus)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var total = 0L;
        foreach (var document in corpus)
            foreach (var token in document)
            {
                counts.TryGetValue(token, out var c);
                counts[token] = c + 1;
                total++;
            }

        _words = counts.Where(kv => kv.Value >= MinCount)
            .OrderByDescending(kv => kv.Value)
            .ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .Select(kv => kv.Key)
            .ToArray();
        _index = _words.Select((w, i) => (w, i)).ToDictionary(p => p.w, p => p.i, StringComparer.Ordinal);

        // Noise distribution: unigram frequency raised to 3/4.
        var weights = _words.Select(w => Math.Pow(counts[w], 0.75)).ToArray();
        var weightSum = weights.Sum();
        _samplingTable = new double[_words.Length];
        var cumulative = 0.0;
        for (var i = 0; i < _words.Length; i++)
        {
            cumulative += weights[i] / weightSum;
            _samplingTable[i] = cumulative;
        }

        // Subsampling: keep a frequent word with probability (sqrt(f/t) + 1) * t/f.
        _keepProbability = new double[_words.Length];
        for (var i = 0; i < _words.Length; i++)
        {
            var frequency = (double)counts[_words[i]] / total;
            _keepProbability[i] = subsampleThreshold <= 0 || frequency <= subsampleThreshold
                ? 1.0
                : Math.Min(1.0, (Math.Sqrt(frequency / subsampleThreshold) + 1) * subsampleThreshold / frequency);
        }
    }
}

/// <summary>
/// GloVe-style embeddings learned by factorising a word co-occurrence matrix.
/// </summary>
/// <remarks>
/// Where Word2Vec learns from individual (centre, context) events, GloVe first accumulates global
/// co-occurrence counts and then fits vectors so that a dot product reproduces the log count. The
/// weighting function caps the influence of very frequent pairs, which is what stops the fit being
/// dominated by a handful of function-word pairs.
/// </remarks>
public sealed class GloVe(
    int dimensions = 50,
    int windowSize = 5,
    int minCount = 5,
    int epochs = 30,
    double learningRate = 0.05,
    double maxCount = 100.0,
    double alpha = 0.75,
    int seed = 42)
{
    /// <summary>Trains on a tokenized corpus.</summary>
    public WordEmbeddings Train(IReadOnlyList<IReadOnlyList<string>> corpus)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var document in corpus)
            foreach (var token in document)
            {
                counts.TryGetValue(token, out var c);
                counts[token] = c + 1;
            }

        var words = counts.Where(kv => kv.Value >= minCount)
            .OrderByDescending(kv => kv.Value)
            .ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .Select(kv => kv.Key)
            .ToArray();
        if (words.Length == 0) throw new InvalidOperationException("No word survived the minimum count.");

        var index = words.Select((w, i) => (w, i)).ToDictionary(p => p.w, p => p.i, StringComparer.Ordinal);

        // Co-occurrences are weighted by 1/distance, so adjacent words count for more.
        var cooccurrence = new Dictionary<(int, int), double>();
        foreach (var document in corpus)
        {
            var ids = document.Where(index.ContainsKey).Select(w => index[w]).ToArray();
            for (var i = 0; i < ids.Length; i++)
                for (var offset = 1; offset <= windowSize && i + offset < ids.Length; offset++)
                {
                    var key = (ids[i], ids[i + offset]);
                    var reverse = (ids[i + offset], ids[i]);
                    cooccurrence.TryGetValue(key, out var a);
                    cooccurrence[key] = a + 1.0 / offset;
                    cooccurrence.TryGetValue(reverse, out var b);
                    cooccurrence[reverse] = b + 1.0 / offset;
                }
        }

        var rng = new GraviRandom(seed);
        var main = rng.Uniform(-0.5, 0.5, words.Length, dimensions);
        var context = rng.Uniform(-0.5, 0.5, words.Length, dimensions);
        var mainBias = rng.Uniform(-0.5, 0.5, words.Length);
        var contextBias = rng.Uniform(-0.5, 0.5, words.Length);

        var entries = cooccurrence.ToArray();
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            rng.Shuffle(entries);
            foreach (var ((i, j), count) in entries)
            {
                var weight = count < maxCount ? Math.Pow(count / maxCount, alpha) : 1.0;

                var prediction = mainBias.At(i) + contextBias.At(j);
                for (var d = 0; d < dimensions; d++) prediction += main[i, d] * context[j, d];

                var error = weight * (prediction - Math.Log(count));
                for (var d = 0; d < dimensions; d++)
                {
                    var gradientMain = error * context[j, d];
                    var gradientContext = error * main[i, d];
                    main[i, d] -= learningRate * gradientMain;
                    context[j, d] -= learningRate * gradientContext;
                }
                mainBias.SetAt(i, mainBias.At(i) - learningRate * error);
                contextBias.SetAt(j, contextBias.At(j) - learningRate * error);
            }
        }

        // The standard recipe sums the two vector sets rather than keeping only one.
        var combined = NdArray.Zeros(words.Length, dimensions);
        for (var i = 0; i < words.Length; i++)
            for (var d = 0; d < dimensions; d++)
                combined[i, d] = main[i, d] + context[i, d];

        return new WordEmbeddings(words, combined);
    }
}
