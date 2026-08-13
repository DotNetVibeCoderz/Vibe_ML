using System.Diagnostics;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Embeddings;
using Gravicode.Science.GraviText.Linguistics;
using Gravicode.Science.GraviText.Tasks;
using Gravicode.Science.GraviText.Tokenization;
using Gravicode.Science.GraviText.Transformers;
using Gravicode.Science.GraviText.Vectorization;

Console.WriteLine(GraviInfo.Banner("GraviText"));

var datasets = Resolve("datasets") ?? throw new DirectoryNotFoundException("datasets directory not found.");
var screenshots = ResolveScreenshots();

var reviews = DataFrame.ReadCsv(Path.Combine(datasets, "imdb_reviews.csv"));
var documents = Enumerable.Range(0, reviews.RowCount).Select(i => reviews.Text("review")[i]!).ToArray();
var labels = Enumerable.Range(0, reviews.RowCount).Select(i => reviews.Text("sentiment")[i]!).ToArray();

// ---------------------------------------------------------------- tokenization
Section("1. Tokenization and normalisation");

const string sample = "Gravicode Studios membangun AI di .NET — hasilnya luar biasa!";
Console.WriteLine($"  input     : {sample}");
Console.WriteLine($"  whitespace: [{string.Join(" | ", new WhitespaceTokenizer().Tokenize(sample))}]");
Console.WriteLine($"  regex     : [{string.Join(" | ", new RegexTokenizer().Tokenize(sample))}]");
Console.WriteLine($"  normalised: {TextNormalizer.Normalize("  Café  RÉSUMÉ   naïve ")}");
Console.WriteLine();

Console.WriteLine("  Stemming:");
foreach (var word in new[] { "connection", "connecting", "connected", "ponies", "running" })
    Console.WriteLine($"    en  {word,-14}-> {PorterStemmer.Stem(word)}");
foreach (var word in new[] { "makanan", "membaca", "berlari", "pekerjaan", "menyapu" })
    Console.WriteLine($"    id  {word,-14}-> {IndonesianStemmer.Stem(word)}");
Console.WriteLine();

Console.WriteLine("  Language detection:");
foreach (var text in new[] { "The quick brown fox jumps over the lazy dog", "Solusi ini adalah ekosistem yang lengkap" })
{
    var (language, confidence) = LanguageDetector.Detect(text);
    Console.WriteLine($"    {language} ({confidence:P0})  \"{text[..Math.Min(45, text.Length)]}...\"");
}
Console.WriteLine();

// ---------------------------------------------------------------- subword
Section("2. WordPiece sub-words");

var learned = WordPieceTokenizer.Train(documents, vocabularySize: 900, minFrequency: 2);
var wordPiece = new WordPieceTokenizer(learned);
Console.WriteLine($"  learned a {learned.Count}-token vocabulary from {documents.Length} reviews");

foreach (var word in new[] { "wonderful", "disappointing", "mengecewakan" })
    Console.WriteLine($"    {word,-16}-> [{string.Join(" ", wordPiece.Tokenize(word))}]");
Console.WriteLine();

// ---------------------------------------------------------------- vectorization
Section("3. TF-IDF");

var tfidf = new TfidfVectorizer(new VectorizerOptions
{
    StopWords = StopWords.Bilingual,
    MinNGram = 1,
    MaxNGram = 2,
    MinDocumentFrequency = 2,
}, sublinearTf: true);

var matrix = tfidf.FitTransform(documents);
Console.WriteLine($"  {documents.Length} documents -> {tfidf.FeatureCount} features");
Console.WriteLine($"  matrix density: {(double)matrix.ToArray().Count(v => v != 0) / matrix.Size:P2}");
Console.WriteLine();
Console.WriteLine($"  Top terms of review 0 (\"{documents[0][..48]}...\"):");
foreach (var (term, weight) in tfidf.TopTerms(matrix, 0, 6))
    Console.WriteLine($"    {term,-22}{weight:F4}");
Console.WriteLine();

// ---------------------------------------------------------------- sentiment
Section("4. Sentiment analysis");

Console.WriteLine("  a) Lexicon, no training data required:");
foreach (var text in new[]
         {
             "an excellent and wonderful film",
             "this is not good at all",
             "produknya bagus dan sangat memuaskan",
             "sangat mengecewakan dan membosankan",
         })
    Console.WriteLine($"    {new SentimentAnalyzer().Analyze(text),-24} \"{text}\"");
Console.WriteLine();

Console.WriteLine("  b) Trained classifier:");
var indices = new GraviRandom(42).Permutation(documents.Length);
var trainCount = (int)(documents.Length * 0.75);
var trainDocs = indices.Take(trainCount).Select(i => documents[i]).ToArray();
var trainLabels = indices.Take(trainCount).Select(i => labels[i]).ToArray();
var testDocs = indices.Skip(trainCount).Select(i => documents[i]).ToArray();
var testLabels = indices.Skip(trainCount).Select(i => labels[i]).ToArray();

var watch = Stopwatch.StartNew();
var analyzer = new SentimentAnalyzer().Train(trainDocs, trainLabels);
watch.Stop();

var classifier = new TextClassifier().Train(trainDocs, trainLabels);
Console.WriteLine($"    trained on {trainCount} reviews in {watch.ElapsedMilliseconds} ms");
Console.WriteLine($"    training accuracy: {classifier.Evaluate(trainDocs, trainLabels):P2}");
Console.WriteLine($"    held-out accuracy: {classifier.Evaluate(testDocs, testLabels):P2}");
Console.WriteLine();

Console.WriteLine("    Terms the model learned to associate with each class:");
foreach (var label in classifier.Labels)
    Console.WriteLine($"      {label,-10}{string.Join(", ", classifier.TopFeatures(label, 8).Select(t => t.Term))}");
Console.WriteLine();

foreach (var text in new[] { "a brilliant and moving story", "dull, weak and disappointing" })
    Console.WriteLine($"    {analyzer.Analyze(text),-24} \"{text}\"");
Console.WriteLine();

// ---------------------------------------------------------------- entities
Section("5. Named entities");

const string article = "Proyek ini dipimpin oleh Kang Fadhil dari Gravicode Studios di Bandung sejak 2019, " +
                       "bekerja sama dengan Microsoft dan tim riset di Jakarta.";
Console.WriteLine($"  {article}");
Console.WriteLine();
foreach (var entity in new NamedEntityRecognizer().Recognize(article))
    Console.WriteLine($"    {entity.Text,-22}{entity.Type,-14}offset {entity.Start}");
Console.WriteLine();

// ---------------------------------------------------------------- summarization
Section("6. Extractive summarisation");

const string longText = """
    Gravicode Science is a data science ecosystem built for .NET developers.
    The numerical core provides n-dimensional arrays with SIMD acceleration and an optional GPU backend.
    On top of it sits a dataframe library with grouping, pivoting and time series resampling.
    The machine learning library covers preprocessing, supervised models, clustering and pipelines.
    Natural language processing adds tokenization, embeddings and transformer building blocks.
    Graph machine learning provides PageRank, node embeddings and graph neural networks.
    Probabilistic programming rounds it out with distributions, MCMC and variational inference.
    Every library ships with documentation in English and Bahasa Indonesia.
    """;

Console.WriteLine("  Three most central sentences:");
foreach (var sentence in new TextRankSummarizer().Summarize(longText, sentenceCount: 3))
    Console.WriteLine($"    - {sentence}");
Console.WriteLine();

Console.WriteLine("  Keywords:");
foreach (var (term, score) in new KeywordExtractor().Fit(documents).Extract(documents[0], 6))
    Console.WriteLine($"    {term,-22}{score:F4}");
Console.WriteLine();

// ---------------------------------------------------------------- embeddings
Section("7. Word embeddings");

var tokenized = documents.Select(d => new RegexTokenizer().Tokenize(d)).ToList();
watch.Restart();
var raw = new Word2Vec(dimensions: 64, windowSize: 4, minCount: 2, epochs: 25, seed: 42).Train(tokenized);
watch.Stop();

Console.WriteLine($"  trained {raw.Count} word vectors of {raw.Dimensions} dimensions in {watch.ElapsedMilliseconds} ms");

// Raw vectors share a large common direction that swamps the cosine on a corpus this small.
var probe = raw.Words.Take(40).ToArray();
var rawAverage = probe.SelectMany(a => probe.Where(b => b != a).Select(b => raw.Similarity(a, b))).Average();

var embeddings = raw.RemoveCommonComponent();
var centredAverage = probe.SelectMany(a => probe.Where(b => b != a).Select(b => embeddings.Similarity(a, b))).Average();

Console.WriteLine($"  mean pairwise cosine: {rawAverage:F3} raw -> {centredAverage:F3} after removing the common component");
Console.WriteLine();

foreach (var query in new[] { "excellent", "terrible", "bagus", "mengecewakan" })
{
    if (!embeddings.Contains(query)) continue;
    var neighbours = embeddings.MostSimilar(query, 5).Select(n => $"{n.Word} ({n.Score:F2})");
    Console.WriteLine($"    near '{query}': {string.Join(", ", neighbours)}");
}
Console.WriteLine("  (80 short reviews is a demo-sized corpus; real embeddings need millions of tokens)");
Console.WriteLine();

// ---------------------------------------------------------------- transformer
Section("8. Transformer encoder");

var config = new TransformerConfig(learned.Count, HiddenSize: 128, Layers: 4, Heads: 8,
    IntermediateSize: 512, MaxPositions: 128);
var model = new TransformerModel(config, seed: 42).WithTokenizer(wordPiece);
Console.WriteLine($"  {model}");
Console.WriteLine($"  NOTE: no pretrained weights are bundled, so these vectors are structurally");
Console.WriteLine($"        correct but not semantically meaningful. Call LoadWeights first for that.");

watch.Restart();
var encoded = model.EncodeBatch(documents.Take(16).ToArray(), maxLength: 64);
watch.Stop();
Console.WriteLine($"  encoded 16 documents (64 tokens each) in {watch.ElapsedMilliseconds} ms " +
                  $"-> {encoded.Shape[0]} x {encoded.Shape[1]}");

model.Forward([1, 2, 3, 4, 5, 6]);
var attention = model.AttentionMaps[0][0];
Console.WriteLine($"  layer 1 head 1 attention is a {attention.Shape[0]}x{attention.Shape[1]} row-stochastic matrix; " +
                  $"row 0 sums to {Enumerable.Range(0, attention.Shape[1]).Sum(j => attention[0, j]):F6}");
Console.WriteLine();

// ---------------------------------------------------------------- chart
Section("9. Embedding projection");

var vocabulary = embeddings.Words
    .Where(w => w.Length > 3)
    .Take(90)
    .ToArray();

var vectors = NdArray.Zeros(vocabulary.Length, embeddings.Dimensions);
for (var i = 0; i < vocabulary.Length; i++)
{
    var vector = embeddings[vocabulary[i]];
    for (var d = 0; d < embeddings.Dimensions; d++) vectors[i, d] = vector.At(d);
}

var tsne = new TStochasticNeighborEmbedding(components: 2, perplexity: 12, iterations: 400, seed: 42);
var projection = tsne.FitTransform(vectors);

var plot = new ScottPlot.Plot();
var xs = Enumerable.Range(0, vocabulary.Length).Select(i => projection[i, 0]).ToArray();
var ys = Enumerable.Range(0, vocabulary.Length).Select(i => projection[i, 1]).ToArray();

var scatter = plot.Add.ScatterPoints(xs, ys);
scatter.MarkerSize = 8;
for (var i = 0; i < vocabulary.Length; i += 3)
{
    var text = plot.Add.Text(vocabulary[i], xs[i], ys[i]);
    text.LabelFontSize = 9;
}

plot.Title($"GraviText - {vocabulary.Length} word vectors projected with t-SNE");
plot.SavePng(Path.Combine(screenshots, "gravitext_embeddings.png"), 1100, 850);
Console.WriteLine($"  saved {Path.Combine(screenshots, "gravitext_embeddings.png")}");
Console.WriteLine($"  final KL divergence: {tsne.FinalDivergence:F4}");

Console.WriteLine();
Console.WriteLine(GraviInfo.Attribution);
return;

static void Section(string title)
    => Console.WriteLine($"--- {title} " + new string('-', Math.Max(0, 60 - title.Length)));

static string? Resolve(string relative)
{
    var directory = new DirectoryInfo(AppContext.BaseDirectory);
    for (var depth = 0; directory is not null && depth < 12; depth++)
    {
        var candidate = Path.Combine(directory.FullName, relative);
        if (Directory.Exists(candidate)) return candidate;
        directory = directory.Parent;
    }
    return null;
}

static string ResolveScreenshots()
{
    var found = Resolve(Path.Combine("docs", "screenshots"));
    if (found is not null) return found;
    var fallback = Path.Combine(Environment.CurrentDirectory, "screenshots");
    Directory.CreateDirectory(fallback);
    return fallback;
}
