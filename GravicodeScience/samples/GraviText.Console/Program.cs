using System.Diagnostics;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Io;
using Gravicode.Science.GraviText.Embeddings;
using Gravicode.Science.GraviText.Linguistics;
using Gravicode.Science.GraviText.Tasks;
using Gravicode.Science.GraviText.Tokenization;
using Gravicode.Science.GraviText.Transformers;
using Gravicode.Science.GraviText.Vectorization;
using Gravicode.Science.GraviLearn;
using Gravicode.Science.GraviText.Sequence;
using Gravicode.Science.GraviText.Generation;

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

// ---------------------------------------------------------------- v0.4: BPE
Section("10. Byte-pair encoding");

// The corpus every BPE explanation uses, with the frequencies that make the merges predictable.
var bpeCorpus = new List<string>();
foreach (var (word, count) in new[] { ("low", 5), ("lower", 2), ("newest", 6), ("widest", 3) })
    for (var i = 0; i < count; i++) bpeCorpus.Add(word);

var bpe = BpeTokenizer.Train(bpeCorpus, vocabularySize: 60, minFrequency: 1);

Console.WriteLine($"  learned {bpe.Merges.Count} merges from a four-word corpus");
Console.WriteLine("  the first few, in the order they were learned:");
foreach (var (left, right) in bpe.Merges.Take(4))
    Console.WriteLine($"    {left} + {right} -> {left + right}");
Console.WriteLine("    'es' merges first because it occurs 9 times, in newest and widest together");

Console.WriteLine($"  'newest'  -> [{string.Join(", ", bpe.Encode("newest"))}]  (one token, it was frequent)");
Console.WriteLine($"  'lowest'  -> [{string.Join(", ", bpe.Encode("lowest"))}]  (never seen, but decomposes)");
Console.WriteLine($"  round trip: '{bpe.Decode(bpe.Encode("lowest"))}'");
Console.WriteLine("  The merge ORDER is the model - a token set alone cannot tokenize, which is why");
Console.WriteLine("  Save writes ranked merges rather than a vocabulary.");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: unigram
Section("11. Unigram (SentencePiece)");

string[] unigramCorpus =
[
    "the cat sat on the mat", "the dog sat on the log", "the cat and the dog",
    "a cat on a mat", "the mat and the log",
];

var unigram = UnigramTokenizer.Train(unigramCorpus, vocabularySize: 80, seedSize: 400);

Console.WriteLine($"  pruned a seed vocabulary down to {unigram.PieceCount} pieces by EM");
Console.WriteLine($"  'the cat sat' -> [{string.Join(", ", unigram.Encode("the cat sat"))}]");
Console.WriteLine($"  round trip is exact: '{unigram.Decode(unigram.Encode("the cat sat"))}'");
Console.WriteLine("    whitespace is ENCODED, not split on, so decoding is plain concatenation");

Console.WriteLine("  Segmentation is Viterbi over the piece lattice, so it is globally optimal.");
Console.WriteLine("  Because every piece carries a probability, alternatives can be sampled:");

var samplingRng = new GraviRandom(7);
var seen = new HashSet<string>(StringComparer.Ordinal);
for (var i = 0; i < 60; i++)
    seen.Add(string.Join(" | ", unigram.SampleEncoding("the cat sat", samplingRng, alpha: 0.2)));

foreach (var segmentation in seen.Take(4))
    Console.WriteLine($"    {segmentation}");
Console.WriteLine($"    {seen.Count} distinct segmentations of one sentence - this is subword");
Console.WriteLine("    regularisation, and BPE cannot do it, having no probabilities to sample from");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: CRF
Section("12. CRF sequence labelling");

var crfLabels = new[] { "O", "B-PER", "I-PER" };
var crf = new LinearChainCrf(crfLabels.Length);
crf.ApplyBioConstraints(crfLabels);

// Emissions that want an impossible sequence: O followed by I-PER.
var wanted = NdArray.Full(-10.0, 2, crfLabels.Length);
wanted[0, 0] = 10;      // O
wanted[1, 2] = 10;      // I-PER

Console.WriteLine("  Per-token emissions strongly want 'O' then 'I-PER', which the BIO scheme forbids:");
Console.WriteLine($"    independent argmax would give: O, I-PER");
Console.WriteLine($"    the CRF decodes:               {string.Join(", ", crf.Decode(wanted).Select(i => crfLabels[i]))}");
Console.WriteLine("  A learned penalty could be outvoted by a confident emission. Forbid gives the");
Console.WriteLine("  transition a score no path can recover from, so the output cannot be invalid.");

// Transitions the model learns, from featureless emissions - so anything it gets right
// came from the transition structure alone.
var alternating = Enumerable.Range(0, 40).Select(_ => NdArray.Zeros(6, 2)).ToList();
var alternatingTags = Enumerable.Range(0, 40).Select(_ => new[] { 0, 1, 0, 1, 0, 1 }).ToList();

var learnedCrf = new LinearChainCrf(2).Fit(alternating, alternatingTags, epochs: 150, learningRate: 0.5);
Console.WriteLine($"  Trained on alternating labels with ZERO emission signal:");
Console.WriteLine($"    0->1 scores {learnedCrf.Transition(0, 1):F3}, 0->0 scores {learnedCrf.Transition(0, 0):F3}");
Console.WriteLine($"    decoding featureless input gives: [{string.Join(", ", learnedCrf.Decode(NdArray.Zeros(6, 2)))}]");
Console.WriteLine("    everything it got right came from the transitions, not the observations");
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: trained NER
Section("13. Trained NER");

var nerPath = Path.Combine(Datasets.FindDatasetDirectory() ?? "datasets", "ner_conll.txt");
if (File.Exists(nerPath))
{
    var annotated = TaggedSentence.LoadConll(nerPath);
    var cut = (int)(annotated.Count * 0.75);
    var nerTrain = annotated.Take(cut).ToList();
    var nerTest = annotated.Skip(cut).ToList();

    var ner = new TrainedNer().Fit(nerTrain);

    Console.WriteLine($"  trained on {nerTrain.Count} sentences, {ner.FeatureCount} features, " +
                      $"{ner.Labels.Count} tags");
    Console.WriteLine($"  held-out entity score: {ner.Evaluate(nerTest)}");
    Console.WriteLine($"  token accuracy:        {ner.TokenAccuracy(nerTest):P2}  " +
                      "<- dominated by 'O', which is why entity F1 is the number to read");

    Console.WriteLine("  Names that appear NOWHERE in the corpus, recognised from shape and context:");
    foreach (var sentence in new[]
    {
        "Kartika Wijaya bekerja di Gravicode .",
        "Zulkarnain tinggal di Surabaya .",
        "Tim dari Bandung mengunjungi Tokopedia .",
    })
    {
        Console.WriteLine($"    \"{sentence}\"");
        foreach (var entity in ner.Recognize(sentence))
            Console.WriteLine($"        {entity.Type,-4} {entity.Text}");
    }

    Console.WriteLine("  The corpus is generated, so 98% F1 says the model learned the templates -");
    Console.WriteLine("  NOT that it would score 98% on newswire. Treat it as a working demonstration.");
}
else
{
    Console.WriteLine($"  datasets/ner_conll.txt not found; skipping.");
}
Console.WriteLine();

// ---------------------------------------------------------------- v0.4: decoder
Section("14. Decoder stack");

var decoderVocab = new Vocabulary();
foreach (var word in "the cat sat on a mat dog log and ran".Split(' ')) decoderVocab.Add(word);

var decoderConfig = new TransformerConfig(
    VocabularySize: decoderVocab.Count, HiddenSize: 32, Layers: 2, Heads: 4,
    IntermediateSize: 64, MaxPositions: 32);

var decoder = new TransformerDecoder(decoderConfig, decoderVocab, new GraviRandom(5));

// Causality is the one property you cannot see by reading generated text.
int[] first = [5, 6, 7, 8];
int[] second = [5, 6, 7, 12];

var stateA = decoder.Forward(first);
var stateB = decoder.Forward(second);

var maxDrift = 0.0;
for (var t = 0; t < 3; t++)
    for (var d = 0; d < stateA.Shape[1]; d++)
        maxDrift = Math.Max(maxDrift, Math.Abs(stateA[t, d] - stateB[t, d]));

Console.WriteLine("  Changing the LAST token and measuring how far the earlier states moved:");
Console.WriteLine($"    max drift across positions 0-2 = {maxDrift:E2}");
Console.WriteLine("    zero, because the attention is causally masked. Without the mask a decoder");
Console.WriteLine("    trains beautifully and generates nothing, having learned to read the future.");

// Untrained perplexity should be at or above the vocabulary size. A value well BELOW
// it would mean the model is already predicting, which for random weights means
// something is leaking - so the number being unimpressive is the point.
Console.WriteLine($"  perplexity on a short sequence: {decoder.Perplexity([5, 6, 7, 8, 9]):F2}");
Console.WriteLine($"    a uniform guess over {decoderVocab.Count} tokens would score {decoderVocab.Count}; " +
                  "random weights do no better, and");
Console.WriteLine("    that is what you want to see before training - anything lower would be a leak");

var generationRng = new GraviRandom(11);
Console.WriteLine("  Sampling strategies, from the same prompt and the same weights:");
Console.WriteLine($"    greedy      -> [{string.Join(", ", decoder.Generate([5, 6], 6, SamplingOptions.Greedy))}]");
Console.WriteLine($"    nucleus     -> [{string.Join(", ", decoder.Generate([5, 6], 6, SamplingOptions.Nucleus, generationRng))}]");
Console.WriteLine($"    temp 1.5    -> [{string.Join(", ", decoder.Generate([5, 6], 6, new SamplingOptions(Temperature: 1.5), generationRng))}]");
Console.WriteLine("  The weights are random, so the tokens are meaningless - what is being shown is");
Console.WriteLine("  that the sampling machinery works, not that the model has anything to say.");
Console.WriteLine();

// ---------------------------------------------------------------- v0.5: checkpoints
Section("15. Loading a pretrained checkpoint");

// No weights ship with this repository - licensing and size keep a real checkpoint out - so a
// small one is written here in PyTorch's (out, in) orientation and then loaded back.
var checkpointConfig = new TransformerConfig(
    VocabularySize: 40, HiddenSize: 16, Layers: 2, Heads: 2,
    IntermediateSize: 32, MaxPositions: 24);

var checkpointDirectory = Directory.CreateTempSubdirectory("gravitext-checkpoint");
try
{
    var checkpointPath = Path.Combine(checkpointDirectory.FullName, "tiny-bert.onnx");
    WriteCheckpoint(checkpointPath, checkpointConfig, skipLayer: -1);

    Console.WriteLine("  Run Inspect first on an unfamiliar file - names are a convention, and the file");
    Console.WriteLine("  is the only authority on which one it follows:");
    var contents = TransformerCheckpoint.Inspect(checkpointPath);
    Console.WriteLine($"    {contents.Count} tensors, for example");
    foreach (var (name, shape) in contents.Take(4))
        Console.WriteLine($"      {name,-58} [{string.Join(", ", shape)}]");
    Console.WriteLine();

    var restored = new TransformerModel(checkpointConfig);
    Console.WriteLine($"  before loading: HasPretrainedWeights = {restored.HasPretrainedWeights}");

    var report = TransformerCheckpoint.Load(restored, checkpointPath);
    Console.WriteLine($"  {report}");
    Console.WriteLine($"  after loading : HasPretrainedWeights = {restored.HasPretrainedWeights}");
    Console.WriteLine("  Every parameter, not just the embeddings: LoadOnnxWeights filled the embedding");
    Console.WriteLine("  tables only, which is enough to look up a word vector and not enough to run the");
    Console.WriteLine("  model - so the attention and feed-forward weights stayed random and the model");
    Console.WriteLine("  reported itself pretrained while producing noise shaped like a sentence.");
    Console.WriteLine();

    var states = restored.Forward([3, 9, 14, 2]);
    Console.WriteLine($"  running 4 token ids gives {states.Shape[0]} x {states.Shape[1]} hidden states, " +
                      $"first three of position 0: [{states[0, 0]:F6}, {states[0, 1]:F6}, {states[0, 2]:F6}]");
    Console.WriteLine("  The end-to-end check is not in this sample: tools/verify/checkpoint_interop.py");
    Console.WriteLine("  runs the same encoder in NumPy and agrees to 2.6e-07, which is float32 precision.");
    Console.WriteLine("  A load that is only nearly right agrees on nothing.");
    Console.WriteLine();

    Console.WriteLine("  Three things it refuses rather than absorbs:");

    var wrongWay = new TransformerModel(checkpointConfig);
    try
    {
        TransformerCheckpoint.Load(wrongWay, checkpointPath,
            CheckpointNames.HuggingFaceBert with { Transposed = false });
    }
    catch (InvalidDataException error)
    {
        Console.WriteLine($"    wrong transpose  -> {Trim(error.Message)}");
    }

    Console.WriteLine("      PyTorch stores (out, in) and computes x W^T; DenseLayer stores (in, out).");
    Console.WriteLine("      A 768x768 attention projection is square, so both readings are consistent and");
    Console.WriteLine("      the mistake loads silently. The check is made against the non-square");
    Console.WriteLine($"      feed-forward weight ({checkpointConfig.IntermediateSize}x{checkpointConfig.HiddenSize} here) before anything is written.");

    var partialPath = Path.Combine(checkpointDirectory.FullName, "partial.onnx");
    WriteCheckpoint(partialPath, checkpointConfig, skipLayer: 1);
    var partialModel = new TransformerModel(checkpointConfig);
    try
    {
        TransformerCheckpoint.Load(partialModel, partialPath);
    }
    catch (InvalidDataException error)
    {
        Console.WriteLine($"    partial export   -> {Trim(error.Message)}");
    }

    var lenient = new TransformerModel(checkpointConfig);
    var partialReport = TransformerCheckpoint.Load(lenient, partialPath, strict: false);
    Console.WriteLine($"      lenient mode loads and names the gap: {partialReport.Missing.Count} missing, " +
                      $"HasPretrainedWeights stays {lenient.HasPretrainedWeights}");
    Console.WriteLine($"      first missing: {partialReport.Missing[0]}");

    var wider = new TransformerConfig(
        VocabularySize: 40, HiddenSize: 32, Layers: 2, Heads: 2,
        IntermediateSize: 64, MaxPositions: 24);
    try
    {
        TransformerCheckpoint.Load(new TransformerModel(wider), checkpointPath);
    }
    catch (InvalidDataException error)
    {
        Console.WriteLine($"    wider model      -> {Trim(error.Message)}");
    }
    Console.WriteLine("      Shapes are checked rather than trusted, so a checkpoint for a different");
    Console.WriteLine("      architecture fails instead of loading its first columns and looking fine.");
    Console.WriteLine();

    Console.WriteLine("  Bring your own export - the tokenizer half is already in place:");
    Console.WriteLine("    torch.onnx.export(AutoModel.from_pretrained(\"bert-base-uncased\"), ...)");
    Console.WriteLine("    TransformerCheckpoint.Load(model, \"bert-base-uncased.onnx\");");
    Console.WriteLine("    CheckpointNames.Reprefixed(\"roberta.\")   // same layout, different model name");
}
finally { checkpointDirectory.Delete(recursive: true); }
Console.WriteLine();

Console.WriteLine();
Console.WriteLine(GraviInfo.Attribution);
return;

static void Section(string title)
    => Console.WriteLine($"--- {title} " + new string('-', Math.Max(0, 60 - title.Length)));

/// <summary>Keeps a thrown message to one readable line, with temp paths shortened to a filename.</summary>
static string Trim(string message)
{
    var line = System.Text.RegularExpressions.Regex.Replace(
        message.ReplaceLineEndings(" ").Trim(),
        @"'[^']*[\\/]([^'\\/]+)'", "'$1'");

    return line.Length <= 96 ? line : line[..96] + "...";
}

/// <summary>
/// Writes a checkpoint for <paramref name="config"/> in PyTorch's (out, in) orientation.
/// </summary>
/// <param name="skipLayer">Omit every tensor of this layer, to stand in for a partial export.</param>
/// <remarks>
/// The layer norms are written as scale 1 / shift 0 rather than as noise, because a norm scaled by
/// a random small number produces hidden states that are numerically fine and semantically empty,
/// and the sample would then be showing a load that worked on output that says nothing.
/// </remarks>
static void WriteCheckpoint(string path, TransformerConfig config, int skipLayer)
{
    var builder = new OnnxGraphBuilder("input", config.HiddenSize);
    var rng = new GraviRandom(7);

    NdArray Noise(int rows, int columns)
    {
        var array = NdArray.Zeros(rows, columns);
        for (var i = 0; i < array.Size; i++) array.SetAt(i, rng.Normal() * 0.05);
        return array;
    }

    NdArray Constant(int size, double value)
    {
        var array = NdArray.Zeros(size);
        for (var i = 0; i < size; i++) array.SetAt(i, value);
        return array;
    }

    NdArray Bias(int size)
    {
        var array = NdArray.Zeros(size);
        for (var i = 0; i < size; i++) array.SetAt(i, rng.Normal() * 0.01);
        return array;
    }

    builder.AddInitializer("bert.embeddings.word_embeddings.weight",
        Noise(config.VocabularySize, config.HiddenSize));
    builder.AddInitializer("bert.embeddings.position_embeddings.weight",
        Noise(config.MaxPositions, config.HiddenSize));
    builder.AddInitializer("bert.embeddings.LayerNorm.weight", Constant(config.HiddenSize, 1.0));
    builder.AddInitializer("bert.embeddings.LayerNorm.bias", Constant(config.HiddenSize, 0.0));

    for (var layer = 0; layer < config.Layers; layer++)
    {
        if (layer == skipLayer) continue;

        var prefix = $"bert.encoder.layer.{layer}";

        foreach (var part in new[]
        {
            "attention.self.query", "attention.self.key",
            "attention.self.value", "attention.output.dense",
        })
        {
            builder.AddInitializer($"{prefix}.{part}.weight", Noise(config.HiddenSize, config.HiddenSize));
            builder.AddInitializer($"{prefix}.{part}.bias", Bias(config.HiddenSize));
        }

        builder.AddInitializer($"{prefix}.attention.output.LayerNorm.weight", Constant(config.HiddenSize, 1.0));
        builder.AddInitializer($"{prefix}.attention.output.LayerNorm.bias", Constant(config.HiddenSize, 0.0));

        // PyTorch stores (out, in), so the expansion is (intermediate, hidden) and the
        // contraction is (hidden, intermediate) - the other way round from DenseLayer.
        builder.AddInitializer($"{prefix}.intermediate.dense.weight",
            Noise(config.IntermediateSize, config.HiddenSize));
        builder.AddInitializer($"{prefix}.intermediate.dense.bias", Bias(config.IntermediateSize));

        builder.AddInitializer($"{prefix}.output.dense.weight",
            Noise(config.HiddenSize, config.IntermediateSize));
        builder.AddInitializer($"{prefix}.output.dense.bias", Bias(config.HiddenSize));

        builder.AddInitializer($"{prefix}.output.LayerNorm.weight", Constant(config.HiddenSize, 1.0));
        builder.AddInitializer($"{prefix}.output.LayerNorm.bias", Constant(config.HiddenSize, 0.0));
    }

    builder.AddNode("Identity", ["input"], "output");
    builder.Save(path, "output", config.HiddenSize);
}

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
