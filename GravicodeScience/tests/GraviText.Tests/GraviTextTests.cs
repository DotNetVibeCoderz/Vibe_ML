using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Embeddings;
using Gravicode.Science.GraviText.Linguistics;
using Gravicode.Science.GraviText.Tasks;
using Gravicode.Science.GraviText.Tokenization;
using Gravicode.Science.GraviText.Transformers;
using Gravicode.Science.GraviText.Vectorization;
using Xunit;

namespace Gravicode.Science.Tests.GraviText;

public class TokenizationTests
{
    [Fact]
    public void WhitespaceTokenizer_SplitsOnSpaces()
    {
        var tokens = new WhitespaceTokenizer().Tokenize("  hello   brave new  world ");
        Assert.Equal(["hello", "brave", "new", "world"], tokens);
    }

    [Fact]
    public void RegexTokenizer_DropsPunctuationAndLowercases()
    {
        var tokens = new RegexTokenizer().Tokenize("Hello, World! It's 2024.");
        Assert.Equal(["hello", "world", "it's", "2024"], tokens);
    }

    [Fact]
    public void RegexTokenizer_HandlesIndonesianText()
    {
        var tokens = new RegexTokenizer().Tokenize("Gravicode Studios membangun AI di .NET!");
        Assert.Contains("gravicode", tokens);
        Assert.Contains("membangun", tokens);
        Assert.DoesNotContain("!", tokens);
    }

    [Fact]
    public void SentenceSplitter_SplitsOnTerminalPunctuation()
    {
        var sentences = SentenceSplitter.Split("First one. Second one! Third one? Done.");
        Assert.Equal(4, sentences.Count);
        Assert.Equal("First one.", sentences[0]);
    }

    [Fact]
    public void Vocabulary_ReservesTheSpecialTokens()
    {
        var vocabulary = new Vocabulary();
        Assert.Equal(5, vocabulary.Count);
        Assert.Equal(0, vocabulary.PadId);
        Assert.Equal(Vocabulary.UnknownToken, vocabulary[vocabulary.UnknownId]);
    }

    [Fact]
    public void Vocabulary_MapsUnknownTokensToUnk()
    {
        var vocabulary = new Vocabulary();
        vocabulary.Add("hello");
        Assert.NotEqual(vocabulary.UnknownId, vocabulary["hello"]);
        Assert.Equal(vocabulary.UnknownId, vocabulary["nonexistent"]);
    }

    [Fact]
    public void Vocabulary_BuildRespectsMinimumFrequency()
    {
        IReadOnlyList<string>[] documents =
        [
            ["a", "a", "a", "rare"],
            ["a", "b", "b"],
        ];

        var vocabulary = Vocabulary.Build(documents, minFrequency: 2);
        Assert.True(vocabulary.Contains("a"));
        Assert.True(vocabulary.Contains("b"));
        Assert.False(vocabulary.Contains("rare"));
    }

    [Fact]
    public void WordPiece_DecomposesUnseenWordsIntoKnownPieces()
    {
        var vocabulary = new Vocabulary();
        foreach (var piece in new[] { "play", "##ing", "##ed", "##er" }) vocabulary.Add(piece);

        var tokenizer = new WordPieceTokenizer(vocabulary);
        Assert.Equal(["play", "##ing"], tokenizer.Tokenize("playing"));
        Assert.Equal(["play", "##er"], tokenizer.Tokenize("player"));
    }

    [Fact]
    public void WordPiece_FallsBackToUnknownForUnreachableWords()
    {
        var vocabulary = new Vocabulary();
        vocabulary.Add("play");
        var tokenizer = new WordPieceTokenizer(vocabulary);
        Assert.Equal([Vocabulary.UnknownToken], tokenizer.Tokenize("zzzz"));
    }

    [Fact]
    public void WordPiece_EncodeAddsSpecialTokensAndPads()
    {
        var vocabulary = new Vocabulary();
        foreach (var piece in new[] { "the", "cat", "sat" }) vocabulary.Add(piece);

        var (ids, mask) = new WordPieceTokenizer(vocabulary).Encode("the cat sat", maxLength: 10);
        Assert.Equal(10, ids.Length);
        Assert.Equal(vocabulary.ClassId, ids[0]);
        Assert.Equal(vocabulary.SeparatorId, ids[4]);
        Assert.Equal(1, mask[0]);
        Assert.Equal(0, mask[9]);
    }

    [Fact]
    public void WordPiece_TrainingLearnsMergesFromACorpus()
    {
        var corpus = Enumerable.Repeat("lower lowest newer newest wider widest", 20).ToList();
        var vocabulary = WordPieceTokenizer.Train(corpus, vocabularySize: 40, minFrequency: 2);

        Assert.True(vocabulary.Count > 5);
        Assert.True(vocabulary.Count <= 40);
    }

    [Fact]
    public void TextNormalizer_StripsAccentsAndCollapsesWhitespace()
    {
        Assert.Equal("cafe resume", TextNormalizer.Normalize("  Café    Résumé  "));
        Assert.Equal("a b", TextNormalizer.Normalize("a...b", removePunctuation: true));
    }
}

public class LinguisticsTests
{
    [Theory]
    [InlineData("caresses", "caress")]
    [InlineData("ponies", "poni")]
    [InlineData("cats", "cat")]
    [InlineData("running", "run")]
    [InlineData("hopping", "hop")]
    [InlineData("connection", "connect")]
    [InlineData("relational", "relat")]
    public void PorterStemmer_MatchesTheReferenceOutput(string word, string expected)
    {
        Assert.Equal(expected, PorterStemmer.Stem(word));
    }

    [Fact]
    public void PorterStemmer_CollapsesAWordFamily()
    {
        var stems = new[] { "connect", "connected", "connecting", "connection", "connections" }
            .Select(PorterStemmer.Stem)
            .Distinct()
            .ToArray();
        Assert.Single(stems);
    }

    [Theory]
    [InlineData("makanan", "makan")]
    [InlineData("membaca", "baca")]
    [InlineData("dibaca", "baca")]
    [InlineData("berlari", "lari")]
    [InlineData("pekerjaan", "kerja")]
    [InlineData("menyapu", "sapu")]
    [InlineData("bukumu", "buku")]
    public void IndonesianStemmer_StripsStandardAffixes(string word, string expected)
    {
        Assert.Equal(expected, IndonesianStemmer.Stem(word));
    }

    [Fact]
    public void StopWords_CoverBothLanguages()
    {
        Assert.Contains("the", StopWords.English);
        Assert.Contains("yang", StopWords.Indonesian);
        Assert.Contains("the", StopWords.Bilingual);
        Assert.Contains("yang", StopWords.Bilingual);
    }

    [Fact]
    public void StopWords_RemoveDropsFunctionWords()
    {
        var kept = StopWords.Remove(["the", "quick", "brown", "fox"], StopWords.English);
        Assert.Equal(["quick", "brown", "fox"], kept);
    }

    [Fact]
    public void Lemmatizer_HandlesIrregularForms()
    {
        Assert.Equal("be", Lemmatizer.Lemmatize("were"));
        Assert.Equal("child", Lemmatizer.Lemmatize("children"));
        Assert.Equal("city", Lemmatizer.Lemmatize("cities"));
    }

    [Fact]
    public void NGrams_ProduceContiguousSequences()
    {
        var bigrams = NGrams.Extract(["a", "b", "c", "d"], 2);
        Assert.Equal(["a b", "b c", "c d"], bigrams);

        var range = NGrams.Range(["a", "b", "c"], 1, 2);
        Assert.Equal(5, range.Count);
    }

    [Fact]
    public void LanguageDetector_TellsEnglishFromIndonesian()
    {
        Assert.Equal("en", LanguageDetector.Detect("The quick brown fox jumps over the lazy dog.").Language);
        Assert.Equal("id", LanguageDetector.Detect("Solusi ini adalah ekosistem yang lengkap untuk para peneliti.").Language);
    }
}

public class VectorizationTests
{
    private static readonly string[] Corpus =
    [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are friends",
    ];

    [Fact]
    public void CountVectorizer_BuildsAVocabularyAndCountsTerms()
    {
        var vectorizer = new CountVectorizer();
        var matrix = vectorizer.FitTransform(Corpus);

        Assert.Equal(3, matrix.Shape[0]);
        Assert.Equal(vectorizer.FeatureCount, matrix.Shape[1]);

        var theColumn = vectorizer.Vocabulary.ToList().IndexOf("the");
        Assert.Equal(2.0, matrix[0, theColumn]);
    }

    [Fact]
    public void CountVectorizer_HonoursMinimumDocumentFrequency()
    {
        var vectorizer = new CountVectorizer(new VectorizerOptions { MinDocumentFrequency = 2 });
        vectorizer.Fit(Corpus);

        Assert.Contains("the", vectorizer.Vocabulary);
        Assert.DoesNotContain("mat", vectorizer.Vocabulary);
    }

    [Fact]
    public void CountVectorizer_SparseAndDenseAgree()
    {
        var vectorizer = new CountVectorizer();
        vectorizer.Fit(Corpus);

        var dense = vectorizer.Transform(Corpus);
        var sparse = vectorizer.TransformSparse(Corpus).ToDense();
        Assert.True(UFunc.AllClose(dense, sparse, 1e-12));
    }

    [Fact]
    public void TfidfVectorizer_DownWeightsUbiquitousTerms()
    {
        var vectorizer = new TfidfVectorizer();
        vectorizer.Fit(Corpus);

        var vocabulary = vectorizer.Vocabulary.ToList();
        var theIdf = vectorizer.InverseDocumentFrequency[vocabulary.IndexOf("the")];
        var matIdf = vectorizer.InverseDocumentFrequency[vocabulary.IndexOf("mat")];

        Assert.True(theIdf < matIdf);
    }

    [Fact]
    public void TfidfVectorizer_ProducesUnitLengthRows()
    {
        var matrix = new TfidfVectorizer().FitTransform(Corpus);
        for (var i = 0; i < matrix.Shape[0]; i++)
        {
            var norm = 0.0;
            for (var j = 0; j < matrix.Shape[1]; j++) norm += matrix[i, j] * matrix[i, j];
            Assert.Equal(1.0, Math.Sqrt(norm), 9);
        }
    }

    [Fact]
    public void Similarity_MeasuresBehaveAsExpected()
    {
        var a = NdArray.FromValues([1.0, 0.0]);
        var b = NdArray.FromValues([0.0, 1.0]);
        var c = NdArray.FromValues([2.0, 0.0]);

        Assert.Equal(0.0, Similarity.Cosine(a, b), 9);
        Assert.Equal(1.0, Similarity.Cosine(a, c), 9);
        // {x,y} and {y,z} share one item out of three distinct ones.
        Assert.Equal(1.0 / 3.0, Similarity.Jaccard(["x", "y"], ["y", "z"]), 9);
        Assert.Equal(3, Similarity.Levenshtein("kitten", "sitting"));
    }

    [Fact]
    public void SimilarDocumentsScoreHigherThanUnrelatedOnes()
    {
        var vectorizer = new TfidfVectorizer();
        var matrix = vectorizer.FitTransform(Corpus);

        // Documents 0 and 1 share "the ... sat on the"; document 2 shares almost nothing.
        Assert.True(Similarity.Cosine(matrix, 0, 1) > Similarity.Cosine(matrix, 0, 2));
    }
}

public class EmbeddingTests
{
    private static IReadOnlyList<IReadOnlyList<string>> ToyCorpus()
    {
        // Two strongly separated topics; the model should place each topic's words together.
        var royal = new[] { "king", "queen", "prince", "princess", "royal", "palace", "crown", "throne" };
        var food = new[] { "rice", "noodle", "soup", "spicy", "sweet", "kitchen", "recipe", "dinner" };

        var corpus = new List<IReadOnlyList<string>>();
        var rng = new GraviRandom(11);
        for (var i = 0; i < 400; i++)
        {
            var source = i % 2 == 0 ? royal : food;
            var sentence = new List<string>();
            for (var k = 0; k < 8; k++) sentence.Add(source[rng.Next(source.Length)]);
            corpus.Add(sentence);
        }
        return corpus;
    }

    [Fact]
    public void Word2Vec_LearnsVectorsForEveryFrequentWord()
    {
        var model = new Word2Vec(dimensions: 32, windowSize: 3, minCount: 2, epochs: 6, seed: 3);
        var embeddings = model.Train(ToyCorpus());

        Assert.Equal(16, embeddings.Count);
        Assert.Equal(32, embeddings.Dimensions);
        Assert.True(embeddings.Contains("king"));
    }

    [Fact]
    public void Word2Vec_PlacesTopicallyRelatedWordsCloserTogether()
    {
        var model = new Word2Vec(dimensions: 48, windowSize: 4, minCount: 2, epochs: 12, seed: 5);
        var embeddings = model.Train(ToyCorpus());

        var within = embeddings.Similarity("king", "queen");
        var across = embeddings.Similarity("king", "noodle");
        Assert.True(within > across, $"within-topic {within:F3} should beat across-topic {across:F3}");
    }

    [Fact]
    public void Word2Vec_MostSimilarExcludesTheQueryWord()
    {
        var model = new Word2Vec(dimensions: 32, minCount: 2, epochs: 6, seed: 7);
        var embeddings = model.Train(ToyCorpus());

        var neighbours = embeddings.MostSimilar("king", top: 3);
        Assert.Equal(3, neighbours.Count);
        Assert.DoesNotContain(neighbours, n => n.Word == "king");
    }

    [Fact]
    public void Word2Vec_RejectsACorpusWithNoFrequentWords()
    {
        // Every word in the toy corpus appears a few hundred times, so the cut has to be well above that.
        var model = new Word2Vec(minCount: 100_000);
        Assert.Throws<InvalidOperationException>(() => model.Train(ToyCorpus()));
    }

    [Fact]
    public void GloVe_LearnsAnEmbeddingOfTheRightShape()
    {
        var model = new GloVe(dimensions: 24, windowSize: 3, minCount: 2, epochs: 10, seed: 9);
        var embeddings = model.Train(ToyCorpus());

        Assert.Equal(16, embeddings.Count);
        Assert.Equal(24, embeddings.Dimensions);
    }

    [Fact]
    public void Embeddings_RoundTripThroughTheWord2VecTextFormat()
    {
        var model = new Word2Vec(dimensions: 16, minCount: 2, epochs: 3, seed: 13);
        var embeddings = model.Train(ToyCorpus());

        var path = Path.Combine(Path.GetTempPath(), $"gravi-{Guid.NewGuid():N}.vec");
        try
        {
            embeddings.Save(path);
            var loaded = WordEmbeddings.Load(path);

            Assert.Equal(embeddings.Count, loaded.Count);
            Assert.Equal(embeddings.Dimensions, loaded.Dimensions);
            Assert.Equal(embeddings["king"].At(0), loaded["king"].At(0), 6);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Embeddings_AverageProducesADocumentVector()
    {
        var model = new Word2Vec(dimensions: 16, minCount: 2, epochs: 3, seed: 15);
        var embeddings = model.Train(ToyCorpus());

        var vector = embeddings.Average(["king", "queen", "unknownword"]);
        Assert.Equal(16, vector.Size);
        Assert.False(double.IsNaN(vector.At(0)));
    }
}

public class TransformerTests
{
    private static (TransformerModel Model, WordPieceTokenizer Tokenizer) BuildTiny()
    {
        var vocabulary = new Vocabulary();
        foreach (var word in "the cat sat on mat dog log run fast slow".Split(' ')) vocabulary.Add(word);

        var config = new TransformerConfig(vocabulary.Count, HiddenSize: 32, Layers: 2, Heads: 4,
            IntermediateSize: 64, MaxPositions: 32);
        var tokenizer = new WordPieceTokenizer(vocabulary);
        return (new TransformerModel(config, seed: 17).WithTokenizer(tokenizer), tokenizer);
    }

    [Fact]
    public void Config_PresetsExposeTheExpectedShapes()
    {
        var config = TransformerConfig.Preset("bert-base", 30522);
        Assert.Equal(768, config.HiddenSize);
        Assert.Equal(12, config.Layers);
        Assert.Equal(12, config.Heads);
        Assert.Equal(64, config.HeadSize);
        Assert.True(config.ParameterCount > 10_000_000);
    }

    [Fact]
    public void Config_RejectsAnUnknownPreset()
    {
        var ex = Assert.Throws<ArgumentException>(() => TransformerConfig.Preset("gpt-9", 100));
        Assert.Contains("bert-base", ex.Message);
    }

    [Fact]
    public void Attention_RejectsAHiddenSizeThatIsNotDivisibleByHeads()
    {
        var config = new TransformerConfig(100, HiddenSize: 30, Heads: 4);
        Assert.Throws<ArgumentException>(() => new MultiHeadAttention(config, new GraviRandom(1)));
    }

    [Fact]
    public void LayerNorm_ProducesZeroMeanUnitVarianceRows()
    {
        var rng = new GraviRandom(19);
        var x = rng.Normal(5.0, 3.0, 4, 16);
        var normalized = new LayerNorm(16).Forward(x);

        for (var i = 0; i < 4; i++)
        {
            var row = normalized.Row(i).Copy();
            Assert.Equal(0.0, Statistics.Mean(row), 8);
            Assert.Equal(1.0, Statistics.Std(row), 5);
        }
    }

    [Fact]
    public void Forward_ProducesOneVectorPerPosition()
    {
        var (model, _) = BuildTiny();
        var hidden = model.Forward([1, 5, 6, 7]);

        Assert.Equal(4, hidden.Shape[0]);
        Assert.Equal(32, hidden.Shape[1]);
        for (var i = 0; i < hidden.Size; i++) Assert.False(double.IsNaN(hidden.At(i)));
    }

    [Fact]
    public void AttentionWeights_FormAProbabilityDistributionPerQuery()
    {
        var (model, _) = BuildTiny();
        model.Forward([1, 2, 3, 4, 5]);

        var maps = model.AttentionMaps;
        Assert.Equal(2, maps.Count);          // two layers
        Assert.Equal(4, maps[0].Length);      // four heads

        var head = maps[0][0];
        for (var i = 0; i < head.Shape[0]; i++)
        {
            var total = 0.0;
            for (var j = 0; j < head.Shape[1]; j++) total += head[i, j];
            Assert.Equal(1.0, total, 8);
        }
    }

    [Fact]
    public void AttentionMask_SuppressesPaddedPositions()
    {
        var (model, _) = BuildTiny();
        model.Forward([1, 2, 3, 4], [1, 1, 0, 0]);

        var head = model.AttentionMaps[0][0];
        // Masked columns receive essentially no attention weight.
        for (var i = 0; i < head.Shape[0]; i++)
        {
            Assert.True(head[i, 2] < 1e-6);
            Assert.True(head[i, 3] < 1e-6);
        }
    }

    [Fact]
    public void Encode_PoolsToASingleVector()
    {
        var (model, _) = BuildTiny();
        var vector = model.Encode("the cat sat on the mat", maxLength: 16);

        Assert.Equal(32, vector.Size);
        Assert.False(model.HasPretrainedWeights);
    }

    [Fact]
    public void Encode_IsDeterministicForTheSameInput()
    {
        var (model, _) = BuildTiny();
        var a = model.Encode("the cat sat", maxLength: 12);
        var b = model.Encode("the cat sat", maxLength: 12);
        Assert.True(UFunc.AllClose(a, b, 1e-12));
    }

    [Fact]
    public void EncodeBatch_ReturnsOneRowPerDocument()
    {
        var (model, _) = BuildTiny();
        var matrix = model.EncodeBatch(["the cat sat", "the dog run fast"], maxLength: 12);

        Assert.Equal(2, matrix.Shape[0]);
        Assert.Equal(32, matrix.Shape[1]);
    }

    [Fact]
    public void Encode_RequiresATokenizerForRawText()
    {
        var config = new TransformerConfig(50, HiddenSize: 16, Layers: 1, Heads: 2, IntermediateSize: 32);
        var model = new TransformerModel(config);
        Assert.Throws<InvalidOperationException>(() => model.Encode("text"));
    }

    [Fact]
    public void Weights_RoundTripAndMarkTheModelAsPretrained()
    {
        var (model, _) = BuildTiny();
        var path = Path.Combine(Path.GetTempPath(), $"gravi-{Guid.NewGuid():N}.json");
        try
        {
            var before = model.Encode("the cat sat", maxLength: 12);
            model.SaveWeights(path);
            model.LoadWeights(path);

            Assert.True(model.HasPretrainedWeights);
            Assert.True(UFunc.AllClose(before, model.Encode("the cat sat", maxLength: 12), 1e-10));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Forward_RejectsSequencesLongerThanThePositionTable()
    {
        var (model, _) = BuildTiny();
        Assert.Throws<ArgumentException>(() => model.Forward(new int[64]));
    }
}

public class TaskTests
{
    private static (string[] Documents, string[] Labels) SentimentCorpus()
    {
        string[] documents =
        [
            "this movie was excellent and the acting was brilliant",
            "a wonderful story with a perfect ending, highly recommend",
            "great film, beautiful and engaging from start to finish",
            "loved it, the best thing I have seen all year",
            "superb direction and an outstanding soundtrack",
            "film ini bagus sekali, ceritanya menarik dan memuaskan",
            "sangat keren dan berkualitas, saya suka",
            "terrible movie, boring and a complete waste of time",
            "awful acting, the worst film I have ever watched",
            "disappointing and dull, I regret watching it",
            "poor script, weak characters, avoid this one",
            "confusing plot and annoying dialogue throughout",
            "film ini mengecewakan dan membosankan sekali",
            "jelek, ceritanya parah dan bikin kecewa",
        ];

        string[] labels =
        [
            "positive", "positive", "positive", "positive", "positive", "positive", "positive",
            "negative", "negative", "negative", "negative", "negative", "negative", "negative",
        ];
        return (documents, labels);
    }

    [Fact]
    public void TextClassifier_LearnsTheTrainingCorpus()
    {
        var (documents, labels) = SentimentCorpus();
        var classifier = new TextClassifier().Train(documents, labels);

        Assert.Equal(["negative", "positive"], classifier.Labels);
        Assert.True(classifier.Evaluate(documents, labels) > 0.9);
    }

    [Fact]
    public void TextClassifier_GeneralisesToUnseenText()
    {
        var (documents, labels) = SentimentCorpus();
        var classifier = new TextClassifier().Train(documents, labels);

        Assert.Equal("positive", classifier.Predict("an excellent and wonderful film").Label);
        Assert.Equal("negative", classifier.Predict("a boring and terrible waste").Label);
    }

    [Fact]
    public void TextClassifier_ProbabilitiesSumToOne()
    {
        var (documents, labels) = SentimentCorpus();
        var prediction = new TextClassifier().Train(documents, labels).Predict("great film");
        Assert.Equal(1.0, prediction.Scores.Values.Sum(), 9);
    }

    [Fact]
    public void TextClassifier_ExposesTheTermsThatDroveTheDecision()
    {
        var (documents, labels) = SentimentCorpus();
        var classifier = new TextClassifier().Train(documents, labels);

        var top = classifier.TopFeatures("positive", 10);
        Assert.NotEmpty(top);
        Assert.True(top[0].Weight > 0);
    }

    [Fact]
    public void TextClassifier_MustBeTrainedFirst()
    {
        Assert.Throws<InvalidOperationException>(() => new TextClassifier().Predict("anything"));
    }

    [Fact]
    public void SentimentAnalyzer_WorksFromTheLexiconWithNoTraining()
    {
        var analyzer = new SentimentAnalyzer();
        Assert.Equal("positive", analyzer.Analyze("this is an excellent and wonderful product").Label);
        Assert.Equal("negative", analyzer.Analyze("terrible quality, a complete waste").Label);
        Assert.Equal("positive", analyzer.Analyze("produknya bagus dan sangat memuaskan").Label);
    }

    [Fact]
    public void SentimentAnalyzer_HandlesNegation()
    {
        var analyzer = new SentimentAnalyzer();
        Assert.Equal("positive", analyzer.AnalyzeWithLexicon("this is good").Label);
        Assert.Equal("negative", analyzer.AnalyzeWithLexicon("this is not good").Label);
    }

    [Fact]
    public void SentimentAnalyzer_PrefersTheTrainedModelOnceItExists()
    {
        var (documents, labels) = SentimentCorpus();
        var analyzer = new SentimentAnalyzer();
        Assert.False(analyzer.IsTrained);

        analyzer.Train(documents, labels);
        Assert.True(analyzer.IsTrained);
        Assert.Equal("positive", analyzer.Analyze("brilliant and engaging").Label);
    }

    [Fact]
    public void NamedEntityRecognizer_FindsGazetteerEntries()
    {
        var entities = new NamedEntityRecognizer()
            .Recognize("The team met in Jakarta and later visited Bandung.");

        Assert.Contains(entities, e => e.Text == "Jakarta" && e.Type == "LOCATION");
        Assert.Contains(entities, e => e.Text == "Bandung" && e.Type == "LOCATION");
    }

    [Fact]
    public void NamedEntityRecognizer_UsesTriggerWords()
    {
        var entities = new NamedEntityRecognizer()
            .Recognize("Proyek ini dipimpin oleh Kang Fadhil dari Gravicode Studios.");

        Assert.Contains(entities, e => e.Type == "PERSON" && e.Text.Contains("Fadhil"));
        Assert.Contains(entities, e => e.Type == "ORGANIZATION" && e.Text.Contains("Gravicode"));
    }

    [Fact]
    public void NamedEntityRecognizer_ProducesNonOverlappingSpans()
    {
        var entities = new NamedEntityRecognizer()
            .Recognize("Dr. Ana works in Jakarta since 2019 with Gravicode Studios.");

        for (var i = 1; i < entities.Count; i++)
            Assert.True(entities[i].Start >= entities[i - 1].Start + entities[i - 1].Length);
    }

    [Fact]
    public void NamedEntityRecognizer_AcceptsCustomGazetteers()
    {
        var entities = new NamedEntityRecognizer()
            .AddGazetteer("PRODUCT", ["GraviNum", "GraviFrame"])
            .Recognize("We benchmarked GraviNum against GraviFrame.");

        Assert.Equal(2, entities.Count(e => e.Type == "PRODUCT"));
    }

    [Fact]
    public void Summarizer_KeepsTheRequestedNumberOfSourceSentences()
    {
        const string article = """
            Gravicode Science is a data science ecosystem for .NET.
            It provides numerical arrays, dataframes and machine learning models.
            The numerical core uses SIMD instructions and an optional GPU backend.
            Dataframes support grouping, pivoting and time series resampling.
            Machine learning covers preprocessing, supervised models and clustering.
            Everything is documented in English and Bahasa Indonesia.
            """;

        var summary = new TextRankSummarizer().Summarize(article, sentenceCount: 2);
        var sentences = SentenceSplitter.Split(article);

        Assert.Equal(2, summary.Count);
        foreach (var sentence in summary) Assert.Contains(sentence, sentences);
    }

    [Fact]
    public void Summarizer_ReturnsEverythingWhenTheTextIsAlreadyShort()
    {
        var summary = new TextRankSummarizer().Summarize("One sentence only.", sentenceCount: 5);
        Assert.Single(summary);
    }

    [Fact]
    public void KeywordExtractor_SurfacesDistinctiveTerms()
    {
        string[] corpus =
        [
            "machine learning models learn patterns from data",
            "the weather forecast predicts rain tomorrow",
            "cooking recipes for a spicy noodle dinner",
        ];

        var keywords = new KeywordExtractor().Fit(corpus).Extract(corpus[2], count: 3);
        Assert.NotEmpty(keywords);
        Assert.Contains(keywords, k => k.Term.Contains("noodle") || k.Term.Contains("spicy") || k.Term.Contains("cooking"));
    }
}
