using System.Diagnostics;
using MediaPipeNet.Diagnostics;
using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using MediaPipeNet.Serialization;
using MediaPipeNet.Tasks.Text.Tokenizers;

namespace MediaPipeNet.Tasks.Text;

/// <summary>Text classification model.</summary>
public enum TextClassifierModel
{
    /// <summary>MobileBERT fine-tuned on SST-2 (negative / positive). Most accurate (26 MB).</summary>
    Bert,
    /// <summary>Average word-embedding classifier on SST-2 (labels "0" = negative, "1" = positive). Tiny and very fast (0.6 MB).</summary>
    AverageWord,
}

/// <summary>Options of <see cref="TextClassifier"/>.</summary>
public sealed record TextClassifierOptions
{
    /// <summary>Model and runtime options.</summary>
    public BaseOptions BaseOptions { get; init; } = BaseOptions.Default;

    /// <summary>The model. Default <see cref="TextClassifierModel.Bert"/>.</summary>
    public TextClassifierModel Model { get; init; } = TextClassifierModel.Bert;

    /// <summary>Category filtering and ranking. Default: all categories.</summary>
    public ClassifierOptions Classifier { get; init; } = ClassifierOptions.Default;

    /// <summary>
    /// Path of your own ONNX text classifier of the <see cref="Model"/> architecture (e.g. a MediaPipe Model Maker export converted with
    /// <c>convert_models.py --custom</c>) used instead of the built-in one. Default null.
    /// </summary>
    public string? ModelPath { get; init; }

    /// <summary>
    /// Category names for a custom model, in output order. When null, <c>{model}.labels.txt</c> next to
    /// <see cref="ModelPath"/> is used if present, otherwise the built-in labels.
    /// </summary>
    public IReadOnlyList<string>? Labels { get; init; }

    /// <summary>
    /// Tokenizer vocabulary of a custom model (vocab.txt). When null, <c>{model}.vocab.txt</c> next to
    /// <see cref="ModelPath"/> is used if present, otherwise the built-in vocabulary of <see cref="Model"/>.
    /// </summary>
    public string? VocabularyPath { get; init; }
}

/// <summary>Result of <see cref="TextClassifier"/>.</summary>
/// <param name="Categories">Categories sorted by score.</param>
public sealed record TextClassificationResult(IReadOnlyList<Category> Categories)
{
    /// <summary>The best category, or null when every category was filtered out.</summary>
    public Category? TopCategory => Categories.Count > 0 ? Categories[0] : null;

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>
/// Text classification (sentiment) with MediaPipe's BERT or average word-embedding SST-2 models.
/// Thread-safe: concurrent calls share the model.
/// </summary>
/// <example>
/// <code>
/// using var classifier = TextClassifier.Create();
/// Console.WriteLine(classifier.Classify("It's beautiful outside.").TopCategory); // positive (99.9%)
/// </code>
/// </example>
public sealed class TextClassifier : IDisposable
{
    private const int BertSequenceLength = 128;
    private const int AverageWordSequenceLength = 256;
    private readonly OnnxModel _model;
    private readonly BertTokenizer? _bert;
    private readonly RegexTokenizer? _regex;
    private readonly IReadOnlyList<string> _labels;
    private readonly int _idsInput, _maskInput, _segmentInput;
    private readonly int _sequenceLength;

    private TextClassifier(TextClassifierOptions options, OnnxModel model)
    {
        Options = options;
        _model = model;
        var vocabulary = options.VocabularyPath
                         ?? (options.ModelPath is { } custom ? CustomModels.CompanionFile(custom, "vocab") : null);
        if (options.Model == TextClassifierModel.Bert)
        {
            _bert = vocabulary is null ? TextResources.BertTokenizer : new BertTokenizer(CustomModels.ReadLines(vocabulary));
            _labels = CustomModels.ResolveLabels(options.Labels, options.ModelPath, TextResources.BertClassifierLabels);
            (_idsInput, _maskInput, _segmentInput) = TextResources.BertInputs(model);
            _sequenceLength = model.Inputs[_idsInput].Shape[^1] is > 0 and var n ? n : BertSequenceLength;
        }
        else
        {
            _regex = vocabulary is null ? TextResources.AverageWordTokenizer : new RegexTokenizer(@"[^\w\']+", CustomModels.ReadLines(vocabulary));
            _labels = CustomModels.ResolveLabels(options.Labels, options.ModelPath, TextResources.AverageWordLabels);
            _sequenceLength = model.Inputs[0].Shape[^1] is > 0 and var n ? n : AverageWordSequenceLength;
        }
    }

    /// <summary>The options the classifier was created with.</summary>
    public TextClassifierOptions Options { get; }

    /// <summary>Creates the classifier (resolving the model synchronously).</summary>
    public static TextClassifier Create(TextClassifierOptions? options = null)
    {
        options ??= new TextClassifierOptions();
        return new TextClassifier(options, CustomModels.Load(options.BaseOptions, ModelFor(options.Model), options.ModelPath));
    }

    /// <summary>Creates the classifier, downloading the model asynchronously when needed.</summary>
    public static async Task<TextClassifier> CreateAsync(TextClassifierOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new TextClassifierOptions();
        return new TextClassifier(options, await CustomModels.LoadAsync(options.BaseOptions, ModelFor(options.Model), options.ModelPath, cancellationToken).ConfigureAwait(false));
    }

    private static ModelDescriptor ModelFor(TextClassifierModel model) =>
        model == TextClassifierModel.Bert ? ModelCatalog.BertClassifier : ModelCatalog.AverageWordClassifier;

    /// <summary>Classifies a text.</summary>
    public TextClassificationResult Classify(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        using var _ = TextTelemetry.Measure(nameof(TextClassifier));
        using var ctx = _model.RentContext();
        if (_bert is not null)
        {
            var (ids, mask, segments) = _bert.Encode(text, _sequenceLength);
            ids.CopyTo(ctx.GetInputInt32(_idsInput));
            mask.CopyTo(ctx.GetInputInt32(_maskInput));
            segments.CopyTo(ctx.GetInputInt32(_segmentInput));
        }
        else
        {
            _regex!.Encode(text, _sequenceLength).CopyTo(ctx.GetInputInt32(0));
        }
        ctx.Run();
        return new TextClassificationResult(Options.Classifier.Select(ctx.GetOutput(0), _labels));
    }

    /// <summary>Classifies a text on a worker thread.</summary>
    public Task<TextClassificationResult> ClassifyAsync(string text, CancellationToken cancellationToken = default) =>
        Task.Run(() => Classify(text), cancellationToken);

    /// <summary>Classifies many texts in parallel.</summary>
    public IReadOnlyList<TextClassificationResult> ClassifyBatch(IReadOnlyList<string> texts, int maxDegreeOfParallelism = -1)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var results = new TextClassificationResult[texts.Count];
        Parallel.For(0, texts.Count, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, i => results[i] = Classify(texts[i]));
        return results;
    }

    /// <inheritdoc />
    public void Dispose() => _model.Dispose();
}

/// <summary>Options of <see cref="TextEmbedder"/>.</summary>
public sealed record TextEmbedderOptions
{
    /// <summary>Model and runtime options.</summary>
    public BaseOptions BaseOptions { get; init; } = BaseOptions.Default;

    /// <summary>L2-normalize the embedding. Default false, as in MediaPipe.</summary>
    public bool L2Normalize { get; init; }

    /// <summary>Output a scalar-quantized (int8) embedding instead of floats. Default false.</summary>
    public bool Quantize { get; init; }
}

/// <summary>Result of <see cref="TextEmbedder"/>.</summary>
/// <param name="Embeddings">One embedding per model head (the MobileBERT embedder has one, 512-D).</param>
public sealed record TextEmbeddingResult(IReadOnlyList<Embedding> Embeddings)
{
    /// <summary>The first (usually only) embedding.</summary>
    public Embedding Embedding => Embeddings[0];

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>
/// Sentence embedding with MediaPipe's MobileBERT embedder (512-D) for semantic search, clustering and
/// duplicate detection. Compare vectors with <see cref="CosineSimilarity"/>. Thread-safe.
/// </summary>
/// <example>
/// <code>
/// using var embedder = TextEmbedder.Create();
/// double similarity = TextEmbedder.CosineSimilarity(embedder.Embed("I love sunny days.").Embedding,
///                                                   embedder.Embed("Sunny weather makes me happy.").Embedding);
/// </code>
/// </example>
public sealed class TextEmbedder : IDisposable
{
    private const int SequenceLength = 128;
    private readonly OnnxModel _model;
    private readonly int _idsInput, _maskInput, _segmentInput;

    private TextEmbedder(TextEmbedderOptions options, OnnxModel model)
    {
        Options = options;
        _model = model;
        (_idsInput, _maskInput, _segmentInput) = TextResources.BertInputs(model);
    }

    /// <summary>The options the embedder was created with.</summary>
    public TextEmbedderOptions Options { get; }

    /// <summary>Creates the embedder (resolving the model synchronously).</summary>
    public static TextEmbedder Create(TextEmbedderOptions? options = null)
    {
        options ??= new TextEmbedderOptions();
        return new TextEmbedder(options, ModelLoader.Load(options.BaseOptions, ModelCatalog.BertEmbedder));
    }

    /// <summary>Creates the embedder, downloading the model asynchronously when needed.</summary>
    public static async Task<TextEmbedder> CreateAsync(TextEmbedderOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new TextEmbedderOptions();
        return new TextEmbedder(options, await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.BertEmbedder, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Cosine similarity of two embeddings, in [−1, 1].</summary>
    public static double CosineSimilarity(Embedding a, Embedding b) => Embedding.CosineSimilarity(a, b);

    /// <summary>Embeds a text.</summary>
    public TextEmbeddingResult Embed(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        using var _ = TextTelemetry.Measure(nameof(TextEmbedder));
        using var ctx = _model.RentContext();
        var (ids, mask, segments) = TextResources.BertTokenizer.Encode(text, SequenceLength);
        ids.CopyTo(ctx.GetInputInt32(_idsInput));
        mask.CopyTo(ctx.GetInputInt32(_maskInput));
        segments.CopyTo(ctx.GetInputInt32(_segmentInput));
        ctx.Run();
        return new TextEmbeddingResult([Embedding.FromTensor(ctx.GetOutput(0), Options.L2Normalize, Options.Quantize)]);
    }

    /// <summary>Embeds a text on a worker thread.</summary>
    public Task<TextEmbeddingResult> EmbedAsync(string text, CancellationToken cancellationToken = default) =>
        Task.Run(() => Embed(text), cancellationToken);

    /// <summary>Embeds many texts in parallel.</summary>
    public IReadOnlyList<TextEmbeddingResult> EmbedBatch(IReadOnlyList<string> texts, int maxDegreeOfParallelism = -1)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var results = new TextEmbeddingResult[texts.Count];
        Parallel.For(0, texts.Count, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, i => results[i] = Embed(texts[i]));
        return results;
    }

    /// <inheritdoc />
    public void Dispose() => _model.Dispose();
}

/// <summary>A language prediction.</summary>
/// <param name="LanguageCode">BCP-47 style code (e.g. <c>en</c>, <c>id</c>, <c>zh-Latn</c>).</param>
/// <param name="Probability">Probability in [0, 1].</param>
public readonly record struct LanguagePrediction(string LanguageCode, float Probability);

/// <summary>Result of <see cref="LanguageDetector"/>.</summary>
/// <param name="Predictions">Languages sorted by probability.</param>
public sealed record LanguageDetectionResult(IReadOnlyList<LanguagePrediction> Predictions)
{
    /// <summary>The most likely language, or null.</summary>
    public LanguagePrediction? TopLanguage => Predictions.Count > 0 ? Predictions[0] : null;

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>Options of <see cref="LanguageDetector"/>.</summary>
public sealed record LanguageDetectorOptions
{
    /// <summary>Model and runtime options.</summary>
    public BaseOptions BaseOptions { get; init; } = BaseOptions.Default;

    /// <summary>Filtering and ranking of the 110 languages. Default: the top 3.</summary>
    public ClassifierOptions Classifier { get; init; } = new() { MaxResults = 3 };
}

/// <summary>
/// Language identification for 110 languages from character n-grams (MediaPipe's language detector).
/// The n-gram hashing runs in C# (<see cref="NGramHasher"/>); the embedding lookup and classifier run in
/// ONNX Runtime. Thread-safe.
/// </summary>
/// <example>
/// <code>
/// using var detector = LanguageDetector.Create();
/// Console.WriteLine(detector.Detect("Selamat pagi, apa kabar?").TopLanguage); // id (97.6%)
/// </code>
/// </example>
public sealed class LanguageDetector : IDisposable
{
    private readonly OnnxModel _model;
    private readonly NGramHasher _hasher = new();

    private LanguageDetector(LanguageDetectorOptions options, OnnxModel model)
    {
        Options = options;
        _model = model;
    }

    /// <summary>The options the detector was created with.</summary>
    public LanguageDetectorOptions Options { get; }

    /// <summary>The language codes the model knows, in output order (index 0 is "unknown").</summary>
    public static IReadOnlyList<string> Languages => TextResources.LanguageLabels;

    /// <summary>Creates the detector (resolving the model synchronously).</summary>
    public static LanguageDetector Create(LanguageDetectorOptions? options = null)
    {
        options ??= new LanguageDetectorOptions();
        return new LanguageDetector(options, ModelLoader.Load(options.BaseOptions, ModelCatalog.LanguageDetector));
    }

    /// <summary>Creates the detector, downloading the model asynchronously when needed.</summary>
    public static async Task<LanguageDetector> CreateAsync(LanguageDetectorOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new LanguageDetectorOptions();
        return new LanguageDetector(options, await ModelLoader.LoadAsync(options.BaseOptions, ModelCatalog.LanguageDetector, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Detects the language of a text.</summary>
    public LanguageDetectionResult Detect(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        using var _ = TextTelemetry.Measure(nameof(LanguageDetector));
        var (ids, tokens) = _hasher.Hash(text);
        var outputs = _model.RunDynamic([new DynamicTensor(_model.Inputs[0].Name, ids, [_hasher.NGramLengths.Count, tokens])]);
        var categories = Options.Classifier.Select(outputs[0].Data, TextResources.LanguageLabels);
        return new LanguageDetectionResult(categories.Select(c => new LanguagePrediction(c.CategoryName ?? c.Index.ToString(System.Globalization.CultureInfo.InvariantCulture), c.Score)).ToArray());
    }

    /// <summary>Detects the language of a text on a worker thread.</summary>
    public Task<LanguageDetectionResult> DetectAsync(string text, CancellationToken cancellationToken = default) =>
        Task.Run(() => Detect(text), cancellationToken);

    /// <inheritdoc />
    public void Dispose() => _model.Dispose();
}

internal static class TextResources
{
    private static readonly Lazy<BertTokenizer> s_bert = new(() => new BertTokenizer(Lines("bert_vocab.txt", keepEmpty: true)));
    private static readonly Lazy<RegexTokenizer> s_regex = new(() => new RegexTokenizer(@"[^\w\']+", Lines("average_word_vocab.txt")));
    private static readonly Lazy<string[]> s_bertLabels = new(() => Lines("bert_classifier_labels.txt"));
    private static readonly Lazy<string[]> s_averageWordLabels = new(() => Lines("average_word_labels.txt"));
    private static readonly Lazy<string[]> s_languages = new(() => Lines("language_labels.txt"));

    public static BertTokenizer BertTokenizer => s_bert.Value;
    public static RegexTokenizer AverageWordTokenizer => s_regex.Value;
    public static IReadOnlyList<string> BertClassifierLabels => s_bertLabels.Value;
    public static IReadOnlyList<string> AverageWordLabels => s_averageWordLabels.Value;
    public static IReadOnlyList<string> LanguageLabels => s_languages.Value;

    /// <summary>Finds the ids / mask / segment inputs of a BERT model by name.</summary>
    public static (int Ids, int Mask, int Segment) BertInputs(OnnxModel model) =>
        (Find(model, "word_ids", "input_ids"), Find(model, "mask"), Find(model, "type_ids", "segment"));

    private static int Find(OnnxModel model, params string[] fragments)
    {
        foreach (var f in fragments)
            if (model.FindInput(f) is >= 0 and var i) return i;
        throw new MediaPipeException($"Model '{model.Name}' has no input matching {string.Join(" / ", fragments)}.");
    }

    private static string[] Lines(string name, bool keepEmpty = false)
    {
        using var stream = typeof(TextResources).Assembly.GetManifestResourceStream("MediaPipeNet.Tasks.Text.Resources." + name)
                           ?? throw new InvalidOperationException($"Missing embedded resource {name}.");
        using var reader = new StreamReader(stream);
        var lines = reader.ReadToEnd().Split('\n').Select(l => l.TrimEnd('\r'));
        if (!keepEmpty) lines = lines.Where(l => l.Length > 0);
        var result = lines.ToArray();
        // A trailing newline yields one empty entry that is not a token.
        return keepEmpty && result.Length > 0 && result[^1].Length == 0 ? result[..^1] : result;
    }
}

internal static class TextTelemetry
{
    public static Scope Measure(string task) => new(task);

    public readonly struct Scope(string task) : IDisposable
    {
        private readonly long _start = Stopwatch.GetTimestamp();

        public void Dispose()
        {
            var tags = new KeyValuePair<string, object?>("task", task);
            MediaPipeTelemetry.TaskDuration.Record(Stopwatch.GetElapsedTime(_start).TotalMilliseconds, tags);
            MediaPipeTelemetry.FramesProcessed.Add(1, tags);
        }
    }
}
