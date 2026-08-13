using System.ComponentModel;
using System.Text;
using Microsoft.SemanticKernel;

namespace ScienceAppGen.Plugins;

/// <summary>
/// The Gravicode.Science API surface, as a tool the assistant can consult.
/// </summary>
/// <remarks>
/// Without this the model writes plausible-looking calls that do not exist - the failure mode is
/// invented method names that compile in its head and not in the project. A curated reference is
/// far cheaper than a build-fix-rebuild cycle, and far more reliable than hoping the library was
/// in the training data.
/// </remarks>
public sealed class GravicodeReferencePlugin
{
    private static readonly Dictionary<string, string> Libraries = new(StringComparer.OrdinalIgnoreCase)
    {
        ["GraviNum"] = """
            GraviNum - arrays, linear algebra, random numbers, statistics.
            using Gravicode.Science.GraviNum;

            NdArray - shared buffer plus shape/strides/offset. Reshape, transpose and slice return
            VIEWS: mutating one view mutates the others. Copy() is the escape hatch.
              NdArray.Zeros(3, 4) / Ones / Full(v, shape) / Eye(n) / Arange(0, 10, 2) / Linspace(0, 1, 11)
              NdArray.FromValues([1.0, 2.0]) / FromArray(double[,]) / FromRows(IReadOnlyList<double[]>)
              a[i, j], a.At(flatIndex), a.SetAt(i, v), a.Shape, a.Rank, a.Size, a.T
              a.Reshape(2, -1), a.Ravel(), a.Transpose(1, 0), a.ExpandDims(0), a.Squeeze(), a.Copy()
              a.Slice(Slice.All, Slice.Range(1, 3)), Slice.At(i), Slice.Reversed
              a.Row(i), a.Column(j), a.Take(indices), a.ToArray(), a.To2DArray()
              Operators + - * / on arrays and scalars. NOTE: * is element-wise, NOT matrix product.
              a.Dot(b), a.Sqrt(), a.Exp(), a.Log(), a.Abs(), a.Pow(2), a.Sigmoid(), a.Relu(),
              a.Clip(lo, hi), a.Map(f), a.Sum(), a.Mean(), a.Min(), a.Max(), a.Std(), a.Var()

            LinAlg (static)
              Dot(a, b), Inner(x, y), Outer(x, y), Determinant(m), Trace(m), Diagonal(m)
              Inverse(m), Solve(a, b), PseudoInverse(m), LeastSquares(a, b)
              MatrixRank(m), ConditionNumber(m), Norm(v), Norm(v, p), MatrixPower(m, k), Kron(a, b)

            Decomposition (static)
              Lu(m) -> .Lower .Upper .Pivot .Solve(b)
              Qr(m) -> .Q .R .Solve(b)
              Cholesky(spd) -> lower triangular; throws when not positive definite
              Svd(m) -> .U .SingularValues .V .Reconstruct()
              SymmetricEigen(m) -> .Values .Vectors (descending)
              Eigenvalues(m) -> (Real, Imaginary)

            GraviRandom (seeded, reproducible)
              new GraviRandom(42); .NextDouble(), .Next(n), .Normal(mean, sd), .Uniform(lo, hi)
              .Gamma(shape, scale), .Beta(a, b), .Binomial(n, p), .Poisson(lambda), .Exponential(rate)
              Array forms: .StandardNormal(rows, cols), .Normal(m, sd, count), .Random(shape)
              .Permutation(n), .Shuffle(list), .Choice(n, count, replace)

            Statistics (static)
              Mean, Median, Mode, Var(a, ddof), Std, Percentile(a, 95), Quantile(a, 0.95)
              Skewness, Kurtosis, Describe(a), Sum(a, axis), Mean(a, axis), ArgMax(a, axis)
              Correlation(x, y), SpearmanCorrelation, CovarianceMatrix(m), CorrelationMatrix(m)
              Histogram(a, bins), Standardize(a), MinMaxScale(a)

            SparseMatrix - CSR. FromDense(m), FromTriplets(rows, cols, triplets), .Multiply(vector),
              .Multiply(matrix, denseIsMatrix: true), .Transpose(), .ToDense(), .Density

            Io: NdIO.SaveCsv/LoadCsv/SaveBinary/LoadBinary/SaveJson, MemoryMappedArray.Create/Open
            Compute: Compute.Cpu, Compute.Gpu, Compute.Best(n), Compute.DescribeDevices()
              GPU auto-dispatch is OFF by default; float64 on integrated GPUs is slower than the CPU.
            GraviInfo.Banner(module), GraviInfo.HardwareReport(), GraviInfo.Attribution
            """,

        ["GraviFrame"] = """
            GraviFrame - dataframes, group-by, joins, time series.
            using Gravicode.Science.GraviFrame;

            DataFrame is IMMUTABLE: every transformation returns a new frame.
              DataFrame.ReadCsv(path), ParseCsv(text), ReadCsvMemoryMapped(path)
              DataFrame.FromMatrix(ndarray, names), FromColumns(dictionary)
              df.RowCount, df.ColumnCount, df.Shape, df.ColumnNames, df.Info(), df.Describe()
              df["col"] -> Series;  df.Numeric("col") / .Text("col") / .DateTimes("col") / .Booleans("col")
              df.Head(n), Tail(n), Rows(start, count), Sample(n, seed)
              df.SelectColumns("a", "b"), Drop("a"), Rename("a", "b"), WithColumn(series)
              df.WithColumn("name", i => expression)
              df.Filter(row => row.Number("age") > 30), FilterBy("col", v => v > 0), Filter(bool[])
              df.SortBy("col", ascending), SortBy([("a", true), ("b", false)])
              df.DropMissing(), FillMissing(0.0), FillMissingWithMean(), MissingCounts()
              df.ToNdArray(), ToNdArray("a", "b")   // the bridge into GraviLearn

            Column types: NumericSeries (double[], NaN = missing), TextSeries, BooleanSeries, DateTimeSeries
              numeric.Sum/Mean/Min/Max/Median/Std/Var/Quantile(q)/Describe()
              numeric.FillMissingWithMedian(), ForwardFill(), BackwardFill(), Interpolate()
              numeric.Standardize(), MinMaxScale(), Clip(lo, hi), Rank(), Apply(f)
              text.Factorize() -> (codes, categories)

            GroupBy
              df.GroupBy("a").Mean("value") / .Sum / .Min / .Max / .Median / .Std / .Count("n")
              df.GroupBy("a", "b").Sum("value")                       // composite key
              df.GroupBy("a").Aggregate("median", "value")
              df.GroupBy("a").Aggregate("value", v => v.Max() - v.Min(), "range")
              df.GroupBy("a").AggregateMany([("v", "sum"), ("v", "mean")])

            Reshaping and joins
              df.Pivot("index", "columns", "values"), Pivot(..., aggregate: "sum")
              df.Melt(["id"], ["a", "b"]), Reshaping.OneHot(df, "col")
              left.Join(right, "id", JoinKind.Left), left.Merge(right, "leftKey", "rightKey")
              DataFrame.Concat([a, b])

            Time series (extension methods on NumericSeries)
              s.Shift(2), s.Diff(), s.PercentChange(), s.CumulativeSum()
              s.Rolling(window: 7).Mean() / .Sum() / .Std() / .Min() / .Max() / .Median()
              s.Rolling(7, minPeriods: 1).Mean(), s.Expanding().Mean(), s.ExponentialMovingAverageBySpan(20)
              Resampling.Resample(df, "date", ResampleFrequency.Monthly, "mean", ["col"])
              Frequencies: Hourly, Daily, Weekly, Monthly, Quarterly, Yearly

            Io.ParquetIO.Write(df, path), Read(path), Read(path, ["col1", "col2"])
            """,

        ["GraviLearn"] = """
            GraviLearn - preprocessing, models, pipelines, metrics.
            X is NdArray (samples, features); y is NdArray of length samples.

            using Gravicode.Science.GraviLearn;                 // Pipeline, Metrics, Datasets
            using Gravicode.Science.GraviLearn.Preprocessing;   // scalers, encoders, imputers
            using Gravicode.Science.GraviLearn.Decomposition;   // PCA, LDA, t-SNE
            using Gravicode.Science.GraviLearn.Linear;          // regressions, linear SVM
            using Gravicode.Science.GraviLearn.Trees;           // trees, forests, boosting
            using Gravicode.Science.GraviLearn.Neighbors;       // kNN, naive Bayes
            using Gravicode.Science.GraviLearn.Clustering;      // KMeans, DBSCAN, GMM
            using Gravicode.Science.GraviLearn.ModelSelection;  // Selection, GridSearch

            Preprocessing: new StandardScaler().FitTransform(x), MinMaxScaler, RobustScaler,
              Normalizer(p: 2), SimpleImputer(ImputationStrategy.Median), OneHotEncoder(dropFirst),
              PolynomialFeatures(degree: 2), LabelEncoder().FitTransform(y)

            Decomposition: new PCA(components: 10) or PrincipalComponentAnalysis(10)
              .FitTransform(x); .ExplainedVarianceRatio; .CumulativeExplainedVariance; .InverseTransform(z)
              new LinearDiscriminantAnalysis().FitTransform(x, y)
              new TStochasticNeighborEmbedding(2, perplexity: 30).FitTransform(x)   // visualisation only

            Models - all have .Fit(x, y) then .Predict(x); classifiers add .PredictProbabilities(x)
              LinearRegression(), RidgeRegression(alpha), LassoRegression(alpha)
              LogisticRegression(learningRate, maxIterations), LinearSupportVectorClassifier(c)
              DecisionTree(SplitCriterion.Gini, maxDepth), DecisionTreeRegressor(maxDepth)
              RandomForestClassifier(nTrees, maxDepth, seed) -> .OutOfBagScore, .FeatureImportances
              RandomForestRegressor(nTrees), GradientBoostingRegressor(nTrees, learningRate, maxDepth)
              GradientBoostingClassifier(nTrees)   // binary only
              KNearestNeighborsClassifier(k, DistanceMetric.Euclidean), KNearestNeighborsRegressor(k)
              GaussianNaiveBayes(), MultinomialNaiveBayes(alpha)   // scale features for kNN!

            Clustering: new KMeans(clusters, restarts, seed).FitPredict(x); .Centroids, .Inertia
              KMeans.ElbowCurve(x, maxK), Dbscan(epsilon, minSamples) -> label -1 is noise
              AgglomerativeClustering(clusters, Linkage.Average)
              GaussianMixture(components, seed) -> .Converged, .LogLikelihood, .Bic(x)

            Metrics (static): Accuracy, Precision, Recall, F1Score, Specificity, MatthewsCorrelation
              F1Average(yTrue, yPred, "weighted"), ConfusionMatrix, FormatConfusionMatrix
              ClassificationReport(yTrue, yPred, labelNames), RocAucScore, RocCurve, LogLoss
              MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, R2Score, RegressionReport
              SilhouetteScore(x, labels)

            Pipeline - keeps preprocessing with the model so cross-validation cannot leak.
              new Pipeline().Add(new StandardScaler()).Add(new PCA(10)).Add(new RandomForestClassifier(100))
              .Fit(x, y), .Predict(x), .Score(x, y). Only the LAST step may be an estimator.

            ModelSelection
              Selection.Split(x, y, testSize: 0.3, seed: 42, stratify: true) -> .TrainX .TestX .TrainY .TestY
              Selection.CrossValidate(() => new Model(), x, y, folds: 5, stratified: true)
              Selection.CrossValidatePipeline(() => BuildPipeline(), x, y, folds: 5)
              new GridSearch(p => new Model((int)p["k"]), folds: 5).AddParameter("k", 1, 3, 5).Fit(x, y)

            Datasets: LoadIris(), LoadTitanic(), LoadDigits()
              MakeBlobs(samples, features, centers, spread, seed)   // needs no data files
              MakeMoons(samples, noise), MakeRegression(samples, features, noise)
              Returned Dataset has .Features .Target .FeatureNames .TargetNames .LabelNames
            """,

        ["GraviText"] = """
            GraviText - tokenization, embeddings, transformers, NLP tasks. English and Indonesian.
            using Gravicode.Science.GraviText.Tokenization / .Linguistics / .Vectorization
                / .Embeddings / .Transformers / .Tasks;

            IMPORTANT: no pretrained transformer weights ship with the library. A new
            TransformerModel is randomly initialised, so its vectors are structurally valid but not
            semantically meaningful. For real semantics use Word2Vec or TfidfVectorizer, which learn
            from the user's own corpus, or call model.LoadWeights(path).

            Tokenization: new RegexTokenizer().Tokenize(text), WhitespaceTokenizer, CharacterTokenizer
              SentenceSplitter.Split(text), TextNormalizer.Normalize(text), StripAccents
              Vocabulary.Build(docs, minFrequency, maxSize)
              WordPieceTokenizer.Train(docs, vocabularySize) then new WordPieceTokenizer(vocab)
              tokenizer.Encode(text, maxLength) -> (ids, attentionMask)

            Linguistics: StopWords.English / .Indonesian / .Bilingual, StopWords.Remove(tokens, set)
              PorterStemmer.Stem(word), IndonesianStemmer.Stem(word), Lemmatizer.Lemmatize(word)
              NGrams.Extract(tokens, 2), NGrams.Range(tokens, 1, 3)
              LanguageDetector.Detect(text) -> (language, confidence)

            Vectorization: new TfidfVectorizer(options, sublinearTf: true).FitTransform(documents)
              new CountVectorizer(options).TransformSparse(documents)
              VectorizerOptions { StopWords, Stemmer, MinNGram, MaxNGram, MinDocumentFrequency,
                                  MaxDocumentFrequencyRatio, MaxFeatures }
              Similarity.Cosine(a, b), Jaccard(setA, setB), Levenshtein(s1, s2)

            Embeddings: new Word2Vec(dimensions, windowSize, minCount, epochs, seed).Train(documents)
              new GloVe(dimensions, windowSize).Train(documents)
              embeddings.Similarity(a, b), .MostSimilar(word, top), .Analogy(a, b, c), .Average(tokens)
              embeddings.RemoveCommonComponent()   // ESSENTIAL on small corpora, or every cosine is ~1.0

            Tasks: new SentimentAnalyzer().Analyze(text)          // lexicon, no training needed
              analyzer.Train(documents, labels)                    // then uses the supervised model
              new TextClassifier().Train(docs, labels).Predict(text) -> .Label .Confidence .Scores
              classifier.TopFeatures("label", 15)
              new NamedEntityRecognizer().AddGazetteer(type, entries).Recognize(text)  // rule-based
              new TextRankSummarizer().Summarize(article, sentenceCount)               // extractive
              new KeywordExtractor().Fit(corpus).Extract(document, count)
            """,

        ["GraviGraph"] = """
            GraviGraph - graph structures, algorithms, node embeddings, GNNs.
            using Gravicode.Science.GraviGraph / .Algorithms / .Embeddings / .Neural;

            Graph: new Graph(directed: false); g.AddNode(name, label), g.AddEdge(src, tgt, weight)
              g.NodeCount, EdgeCount, Density, Neighbors(i), Predecessors(i), Degree(i), HasEdge(a, b)
              g.NodeFeatures = ndarray;  g.NodeLabels;  g.NodeNames;  g.Classes
              g.ToSparseAdjacency(addSelfLoops: true, symmetricNormalize: true)   // GCN propagation
              g.Subgraph(nodes), g.AsUndirected(), g.Laplacian()
              Graph.Load(jsonPath), g.Save(path), Graph.LoadEdgeList(path)
              Generators: Graph.Random(n, p), ScaleFree(n, edgesPerNode), Communities(k, size), Cycle, Complete

            Algorithms (static GraphAlgorithms)
              BreadthFirstSearch(g, start), DepthFirstSearch, HopDistances(g, start)
              ShortestPaths(g, start) -> (Distances, Previous), ShortestPath(g, a, b)   // Dijkstra
              PageRank(g, damping), PersonalizedPageRank(g, seeds)
              DegreeCentrality, ClosenessCentrality, BetweennessCentrality, EigenvectorCentrality
              ConnectedComponents(g)          // WEAK connectivity on directed graphs
              StronglyConnectedComponents(g)  // Kosaraju
              TriangleCounts, ClusteringCoefficients, AverageClusteringCoefficient
              TopologicalSort(dag), LabelPropagation(g), Modularity(g, communities)

            Embeddings: new DeepWalk(dimensions, walksPerNode, walkLength, epochs).Train(graph)
              new Node2Vec(dimensions, p, q, walksPerNode).Train(graph)
                q < 1 explores outward (communities); q > 1 stays local (structural roles)
              CALL graph.AsUndirected() FIRST on citation graphs, or walks strand after one step.
              embeddings.Similarity(a, b), .MostSimilar(node, top), .LinkScore(a, b)

            Neural - all fully trained with hand-derived gradients
              new GraphConvolutionalNetwork(hiddenSize, learningRate, epochs, dropout, weightDecay, seed)
                .Train(graph, trainMask, validationMask, features)
                .Predict(), .PredictProbabilities(), .NodeEmbeddings(), .Score(graph, mask), .History
              new GraphSage(hiddenSize, epochs).Train(...)   // INDUCTIVE: .PredictInductive(newGraph, features)
              new GraphAttentionNetwork(hiddenSize, heads, epochs).Train(...)   // .AttentionWeights
              Two layers is the usual depth; beyond ~3 hops representations over-smooth.
            """,

        ["GraviProb"] = """
            GraviProb - distributions, MCMC, variational inference, probabilistic models.
            using Gravicode.Science.GraviProb;  using Gravicode.Science.GraviProb.Models;

            Distributions expose LogDensity (not density) because inference multiplies many of them.
              Distribution.Normal(mean, sd), Uniform(lo, hi), Bernoulli(p), Binomial(n, p),
              Poisson(rate), Gamma(shape, rate), Beta(a, b), Exponential(rate),
              StudentT(df, loc, scale), LogNormal(mu, sigma), HalfNormal(sigma), new Categorical(weights)
              d.LogDensity(x), d.Density(x), d.Sample(rng), d.Sample(rng, count), d.Mean, d.Variance, d.Cdf(x)
              Distribution.Beta(1, 1).PosteriorAfter(successes, failures)   // exact conjugate posterior

            Model building
              var model = new BayesianModel()
                  .AddDistribution("theta", Distribution.Beta(1, 1))
                  .AddObservation("data", DistributionSpec.Binomial(200, "theta"), 125);
              DistributionSpec.Binomial(n, "var"), Bernoulli("var"), Normal("mu", "sigma"),
                Normal("mu", 1.0), Poisson("rate"),
                DistributionSpec.From(v => new Gamma(v["a"], v["b"]), "a", "b")
              model.LogPosterior(values), model.PriorPredictive(draws)

            Inference
              model.SampleMCMC(iterations, chains, warmup, thin, seed)   // adaptive Metropolis-Hastings
              model.SampleGibbs(iterations, chains, warmup, seed)
              model.FitVariational(iterations, learningRate, monteCarloSamples, seed)

            PosteriorTrace
              posterior["theta"], .Chain("theta", 0), .Mean("theta"), .StandardDeviation, .Median
              .CredibleInterval("theta", 0.95), .HighestDensityInterval("theta", 0.95)   // prefer HDI
              .RHat("theta")   // near 1 = converged; above ~1.01 = not
              .EffectiveSampleSize("theta"), .AcceptanceRate, .Summary()
              .PosteriorPredictive((values, rng) => rng.Binomial(n, values["theta"]), draws, seed)

            Models
              new BayesianNetwork().AddVariable("rain", 0.8, 0.2)
                  .AddVariable("wet", 2, ["rain"], [[1.0, 0.0], [0.2, 0.8]])
                  .Infer("wet"), .Infer("rain", evidence), .Sample(rng)
              new HiddenMarkovModel(initial, transitions, emissions)
                  .LogLikelihood(obs), .Viterbi(obs), .StatePosteriors(obs), .Fit(sequences, iterations)
                  HiddenMarkovModel.Random(states, symbols, seed)
              new BayesianLinearRegression(priorPrecision, noisePrecision).Fit(x, y)
                  .CoefficientMeans, .Predict(x), .PredictWithUncertainty(x), .PredictInterval(x, 0.95)
            """,
    };

    [KernelFunction("GravicodeReference")]
    [Description("Returns the real API surface of a Gravicode.Science library: namespaces, types, " +
                 "method signatures and the gotchas. Call this BEFORE writing code against a " +
                 "library rather than guessing method names.")]
    public string GravicodeReference(
        [Description("Library name: GraviNum, GraviFrame, GraviLearn, GraviText, GraviGraph or " +
                     "GraviProb. Use 'all' for a short index of every library.")]
        string library)
    {
        if (string.Equals(library, "all", StringComparison.OrdinalIgnoreCase))
        {
            var builder = new StringBuilder("Gravicode.Science libraries:\n\n");
            builder.AppendLine("  GraviNum    arrays, linear algebra, decompositions, random, statistics");
            builder.AppendLine("  GraviFrame  dataframes, group-by, pivot, joins, time series (needs GraviNum)");
            builder.AppendLine("  GraviLearn  preprocessing, models, pipelines, metrics (needs GraviFrame)");
            builder.AppendLine("  GraviText   tokenization, embeddings, transformers, NLP (needs GraviLearn)");
            builder.AppendLine("  GraviGraph  graphs, algorithms, node embeddings, GNNs (needs GraviText)");
            builder.AppendLine("  GraviProb   distributions, MCMC, Bayesian models (needs GraviNum)");
            builder.AppendLine("\nCall GravicodeReference with one name for its full API surface.");
            return builder.ToString();
        }

        if (Libraries.TryGetValue(library, out var reference)) return reference;

        return $"No library called '{library}'. Available: {string.Join(", ", Libraries.Keys)}, or 'all'.";
    }

    [KernelFunction("GravicodeExample")]
    [Description("Returns a complete, working example program for a common task. Use it as the " +
                 "starting shape for generated code rather than inventing structure.")]
    public string GravicodeExample(
        [Description("Task: classification, regression, clustering, dataframe, timeseries, " +
                     "sentiment, graph or bayesian.")]
        string task)
    {
        var key = task.ToLowerInvariant();
        return key switch
        {
            "classification" => """
                using Gravicode.Science.GraviLearn;
                using Gravicode.Science.GraviLearn.ModelSelection;
                using Gravicode.Science.GraviLearn.Preprocessing;
                using Gravicode.Science.GraviLearn.Trees;

                var data = Datasets.LoadIris();
                var split = Selection.Split(data.Features, data.Target, testSize: 0.3, seed: 42, stratify: true);

                var pipeline = new Pipeline()
                    .Add(new StandardScaler())
                    .Add(new RandomForestClassifier(nTrees: 100, seed: 42));
                pipeline.Fit(split.TrainX, split.TrainY);

                Console.WriteLine($"accuracy: {pipeline.Score(split.TestX, split.TestY):P2}");
                Console.WriteLine(Metrics.ClassificationReport(
                    split.TestY, pipeline.Predict(split.TestX), data.LabelNames));
                """,

            "regression" => """
                using Gravicode.Science.GraviLearn;
                using Gravicode.Science.GraviLearn.Linear;
                using Gravicode.Science.GraviLearn.ModelSelection;

                var (x, y, truth) = Datasets.MakeRegression(samples: 500, features: 5, noise: 0.5, seed: 42);
                var split = Selection.Split(x, y, testSize: 0.25, seed: 42);

                var model = new LinearRegression();
                model.Fit(split.TrainX, split.TrainY);

                Console.WriteLine($"R2: {model.Score(split.TestX, split.TestY):F4}");
                Console.WriteLine(Metrics.RegressionReport(split.TestY, model.Predict(split.TestX), 5));
                """,

            "clustering" => """
                using Gravicode.Science.GraviLearn;
                using Gravicode.Science.GraviLearn.Clustering;
                using Gravicode.Science.GraviLearn.Preprocessing;

                var data = Datasets.MakeBlobs(samples: 400, features: 4, centers: 3, seed: 42);
                var x = new StandardScaler().FitTransform(data.Features);   // scaling is required

                var kmeans = new KMeans(clusters: 3, restarts: 10, seed: 42);
                var labels = kmeans.FitPredict(x);

                Console.WriteLine($"inertia    : {kmeans.Inertia:F2}");
                Console.WriteLine($"silhouette : {Metrics.SilhouetteScore(x, labels):F4}");
                """,

            "dataframe" => """
                using Gravicode.Science.GraviFrame;

                var df = DataFrame.ReadCsv("data.csv");
                Console.WriteLine(df.Info());

                var clean = df.WithColumn(df.Numeric("age").FillMissingWithMedian().Rename("age_filled"));
                Console.WriteLine(clean.GroupBy("category").Mean("value").SortBy("value", ascending: false));
                Console.WriteLine(clean.Describe());
                """,

            "timeseries" => """
                using Gravicode.Science.GraviFrame;

                var df = DataFrame.ReadCsv("prices.csv").SortBy("date");
                var close = df.Numeric("close");

                var enriched = df
                    .WithColumn(close.Rolling(window: 7).Mean().Rename("ma7"))
                    .WithColumn(close.PercentChange().Rename("daily_return"));

                Console.WriteLine(enriched.Tail(10));
                Console.WriteLine(Resampling.Resample(df, "date", ResampleFrequency.Monthly, "mean"));
                """,

            "sentiment" => """
                using Gravicode.Science.GraviText.Tasks;

                // No training data needed - the lexicon covers English and Indonesian.
                var analyzer = new SentimentAnalyzer();
                Console.WriteLine(analyzer.Analyze("produknya bagus dan sangat memuaskan"));

                // With labels the supervised model is materially better.
                var classifier = new TextClassifier().Train(documents, labels);
                Console.WriteLine(classifier.Predict("an excellent result"));
                Console.WriteLine(string.Join(", ", classifier.TopFeatures("positive", 10).Select(t => t.Term)));
                """,

            "graph" => """
                using Gravicode.Science.GraviGraph;
                using Gravicode.Science.GraviGraph.Algorithms;
                using Gravicode.Science.GraviGraph.Neural;

                var graph = Graph.Load("graph.json");
                var rank = GraphAlgorithms.PageRank(graph);

                var train = new Gravicode.Science.GraviNum.GraviRandom(42)
                    .Permutation(graph.NodeCount).Take(140).ToArray();

                var gcn = new GraphConvolutionalNetwork(hiddenSize: 16, epochs: 100, seed: 42)
                    .Train(graph, train, features: graph.NodeFeatures);
                Console.WriteLine($"accuracy: {gcn.Score(graph, test):P1}");
                """,

            "bayesian" => """
                using Gravicode.Science.GraviProb;

                var model = new BayesianModel()
                    .AddDistribution("theta", Distribution.Beta(1, 1))
                    .AddObservation("data", DistributionSpec.Binomial(200, "theta"), 125);

                var posterior = model.SampleMCMC(iterations: 20_000, chains: 4, seed: 42);
                Console.Write(posterior.Summary());

                var (low, high) = posterior.HighestDensityInterval("theta", 0.95);
                Console.WriteLine($"95% HDI: [{low:F4}, {high:F4}]");
                """,

            _ => "Unknown task. Try: classification, regression, clustering, dataframe, " +
                 "timeseries, sentiment, graph, bayesian.",
        };
    }
}
