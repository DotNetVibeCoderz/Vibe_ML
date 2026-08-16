using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Compute;
using Gravicode.Science.GraviText.Embeddings;
using Gravicode.Science.GraviText.Linguistics;
using Gravicode.Science.GraviText.Tokenization;
using Gravicode.Science.GraviText.Transformers;
using Gravicode.Science.GraviText.Vectorization;

BenchmarkSwitcher.FromAssembly(typeof(TransformerInferenceBenchmark).Assembly).Run(args, DefaultConfig.Instance
    .AddJob(Job.ShortRun.WithWarmupCount(2).WithIterationCount(4))
    .WithOptions(ConfigOptions.DisableOptimizationsValidator));
return;

/// <summary>Shared corpus generation for the text benchmarks.</summary>
public static class Corpus
{
    private static readonly string[] Words =
    [
        "the", "model", "learns", "from", "data", "and", "produces", "an", "embedding", "vector",
        "graph", "network", "attention", "layer", "token", "sentence", "document", "corpus",
        "bagus", "sangat", "menarik", "hasil", "data", "model", "jaringan", "kata", "kalimat",
    ];

    /// <summary>Generates <paramref name="count"/> pseudo-documents of a fixed length.</summary>
    public static string[] Generate(int count, int wordsPerDocument, int seed = 42)
    {
        var rng = new GraviRandom(seed);
        var documents = new string[count];
        for (var i = 0; i < count; i++)
        {
            var words = new string[wordsPerDocument];
            for (var w = 0; w < wordsPerDocument; w++) words[w] = Words[rng.Next(Words.Length)];
            documents[i] = string.Join(' ', words);
        }
        return documents;
    }
}

/// <summary>
/// Transformer encoder inference cost as the sequence grows.
/// </summary>
/// <remarks>
/// Attention is quadratic in sequence length - every token attends to every other - so doubling
/// the sequence roughly quadruples the attention cost while the feed-forward part only doubles.
/// That is the scaling wall every long-context technique exists to work around, and it is visible
/// directly in these numbers. Batches are encoded in parallel because sequences are independent.
/// </remarks>
[MemoryDiagnoser]
public class TransformerInferenceBenchmark
{
    private TransformerModel _model = null!;
    private int[] _tokens = [];
    private string[] _documents = [];

    /// <summary>Sequence length in tokens.</summary>
    [Params(32, 64, 128)]
    public int SequenceLength { get; set; }

    /// <summary>Builds a small BERT-shaped encoder and a batch of documents.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var vocabulary = new Vocabulary();
        foreach (var word in Corpus.Generate(1, 200)[0].Split(' ').Distinct()) vocabulary.Add(word);

        var config = new TransformerConfig(vocabulary.Count, HiddenSize: 128, Layers: 4, Heads: 8,
            IntermediateSize: 512, MaxPositions: 256);
        _model = new TransformerModel(config, seed: 42).WithTokenizer(new WordPieceTokenizer(vocabulary));

        var rng = new GraviRandom(42);
        _tokens = Enumerable.Range(0, SequenceLength).Select(_ => rng.Next(vocabulary.Count)).ToArray();
        _documents = Corpus.Generate(16, SequenceLength / 2);

        Console.WriteLine($"[setup] {_model}");
        Console.WriteLine($"[setup] {Compute.DescribeDevices()}");
    }

    /// <summary>A single sequence through the encoder.</summary>
    [Benchmark(Baseline = true)]
    public double SingleSequence() => _model.Forward(_tokens)[0, 0];

    /// <summary>The same sequence pooled to one vector.</summary>
    [Benchmark]
    public double SingleSequencePooled() => _model.Encode(_tokens).At(0);

    /// <summary>Sixteen documents encoded in parallel.</summary>
    [Benchmark]
    public double BatchOfSixteen() => _model.EncodeBatch(_documents, SequenceLength)[0, 0];
}

/// <summary>
/// The matrix products inside an attention layer, measured on each backend.
/// </summary>
/// <remarks>
/// A transformer's cost is almost entirely dense matrix products, so this isolates them at the
/// shapes a real layer uses. It is the measurement that says whether routing a model's forward
/// pass through the GPU backend would pay for the transfers at a given hidden size.
/// </remarks>
[MemoryDiagnoser]
public class AttentionKernelBenchmark
{
    private double[] _input = [];
    private double[] _weights = [];
    private IComputeBackend? _gpu;

    /// <summary>Model width.</summary>
    [Params(128, 512, 768)]
    public int HiddenSize { get; set; }

    /// <summary>Sequence length.</summary>
    [Params(128, 512)]
    public int SequenceLength { get; set; }

    /// <summary>Allocates a projection input and weight matrix.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new GraviRandom(42);
        _input = rng.StandardNormal(SequenceLength, HiddenSize).ToArray();
        _weights = rng.StandardNormal(HiddenSize, HiddenSize).ToArray();
        _gpu = Compute.Gpu;
    }

    /// <summary>The query/key/value projection on the CPU.</summary>
    [Benchmark(Baseline = true)]
    public double ProjectionCpu()
        => Compute.Cpu.MatMul(_input, _weights, SequenceLength, HiddenSize, HiddenSize)[0];

    /// <summary>The same projection on the GPU, transfers included.</summary>
    [Benchmark]
    public double ProjectionGpu()
    {
        if (_gpu is null) return ProjectionCpu();
        return _gpu.MatMul(_input, _weights, SequenceLength, HiddenSize, HiddenSize)[0];
    }

    /// <summary>Releases the accelerator.</summary>
    [GlobalCleanup]
    public void Cleanup() => Compute.ReleaseGpu();
}

/// <summary>Tokenization and vectorization throughput, the part of a pipeline that runs on every request.</summary>
[MemoryDiagnoser]
public class TextPipelineBenchmark
{
    private string[] _documents = [];
    private WordPieceTokenizer _wordPiece = null!;
    private TfidfVectorizer _tfidf = null!;

    /// <summary>Number of documents processed per call.</summary>
    [Params(1_000, 10_000)]
    public int Documents { get; set; }

    /// <summary>Generates the corpus and fits the vectorizer once.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _documents = Corpus.Generate(Documents, 40);
        _wordPiece = new WordPieceTokenizer(WordPieceTokenizer.Train(_documents.Take(200).ToArray(), 500));
        _tfidf = new TfidfVectorizer(new VectorizerOptions { StopWords = StopWords.Bilingual, MaxNGram = 2 });
        _tfidf.Fit(_documents.Take(500).ToArray());
    }

    /// <summary>Regex word tokenization.</summary>
    [Benchmark(Baseline = true)]
    public int RegexTokenize()
    {
        var tokenizer = new RegexTokenizer();
        var total = 0;
        foreach (var document in _documents) total += tokenizer.Tokenize(document).Count;
        return total;
    }

    /// <summary>WordPiece sub-word tokenization, which does a longest-prefix search per word.</summary>
    [Benchmark]
    public int WordPieceTokenize()
    {
        var total = 0;
        foreach (var document in _documents) total += _wordPiece.Tokenize(document).Count;
        return total;
    }

    /// <summary>Porter stemming over every token.</summary>
    [Benchmark]
    public int Stem()
    {
        var tokenizer = new RegexTokenizer();
        var total = 0;
        foreach (var document in _documents) total += PorterStemmer.StemAll(tokenizer.Tokenize(document)).Count;
        return total;
    }

    /// <summary>TF-IDF transform into a dense matrix.</summary>
    [Benchmark]
    public int TfidfDense() => _tfidf.Transform(_documents).Shape[0];
}

/// <summary>Word2Vec training throughput as the corpus grows.</summary>
[MemoryDiagnoser]
public class EmbeddingTrainingBenchmark
{
    private IReadOnlyList<IReadOnlyList<string>> _corpus = [];

    /// <summary>Number of documents in the training corpus.</summary>
    [Params(500, 2_000)]
    public int Documents { get; set; }

    /// <summary>Tokenizes the corpus once.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var tokenizer = new RegexTokenizer();
        _corpus = Corpus.Generate(Documents, 60).Select(tokenizer.Tokenize).ToList();
    }

    /// <summary>Skip-gram with negative sampling, three epochs.</summary>
    [Benchmark(Baseline = true)]
    public int Word2Vec()
        => new Word2Vec(dimensions: 64, windowSize: 5, minCount: 2, epochs: 3, seed: 42).Train(_corpus).Count;

    /// <summary>GloVe, which builds a co-occurrence matrix first.</summary>
    [Benchmark]
    public int GloVe()
        => new GloVe(dimensions: 64, windowSize: 5, minCount: 2, epochs: 5, seed: 42).Train(_corpus).Count;
}
