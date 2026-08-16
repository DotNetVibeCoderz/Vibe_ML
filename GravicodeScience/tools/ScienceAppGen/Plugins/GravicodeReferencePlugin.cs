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

            Einsum (v0.4) - one notation for products, transposes, traces, contractions
              Einsum.Evaluate("ij,jk->ik", a, b)   matrix product
              Einsum.Evaluate("ji,jk->ik", a, b)   == Dot(a.T, b), but the axes are written down
              "ij->ji" transpose  "ii->i" diagonal  "ii->" trace  "ij->j" column sums
              "i,j->ij" outer  "ij,ij->" Frobenius  "bij,bjk->bik" batched
              Omitting "->" infers the output: letters appearing once, alphabetical.

            ComplexNdArray (v0.4) - complex arrays over an interleaved Complex buffer
              ComplexNdArray.FromParts(real, imaginary), .FromValues(...), .Zeros(shape)
              .Real() .Imaginary() .Magnitude() .Phase() .Power() .Conjugate() .Norm()
              .Transpose() .ConjugateTranspose()   <- A^H is what complex formulas mean, not A^T
              ComplexNdArray.Dot(a, b), ComplexNdArray.Inner(a, b)   Inner conjugates its FIRST arg
              .Fft() .Ifft() .Fft2() .Ifft2()      separable 2-D transform
              Reshape/Transpose COPY here, unlike NdArray views.

            SliceOps (v0.4, extension methods) - the rest of NumPy-style indexing
              array.Assign(value)                       fill a view with a scalar
              array.SliceEllipsis([], [Slice.At(0)])    name trailing axes, ignore leading ones
              array.TakeAlong(indices, axis: 1)         select along any axis (copies)
              array.AxisAt(axis, index)                 one position, axis dropped (a view)
              array.AxisRange(axis, start, stop)        a span, axis kept (a view)
              SliceOps.Select(condition, ifTrue, ifFalse)   element-wise choice, shape kept
              array.SetWhere(predicate, value)          masked write, in place
              array.IndicesWhere(predicate)             positions, not values
              array.FilterRows(row => ...)              whole rows, copied
              array.Clip(low, high)
              SliceShorthand.Last(n) / First(n) / Every(n) / DropLast(n)
              NdArray.Assign(NdArray) already existed and broadcasts - use it for arrays.

            Signal.Fft - any length (radix-2, Bluestein otherwise)
              Fft.Forward(complex[]), Fft.Inverse, Fft.ForwardReal(signal), Fft.InverseReal
              Fft.Magnitude(signal), Fft.FrequencyBins(n, sampleRate), Fft.Convolve(a, b)
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

            Windowing (v0.4) - window functions, partitioned, and time-series joins
              Windowing.Rank(df, ["customer"], "amount", descending: false, dense: false)
              Windowing.CumulativeSum(df, partitionBy, column)
              Windowing.RollingMean(df, partitionBy, column, window: 7)
              Windowing.Lag(df, partitionBy, column, offset: 1) / Lead(...)
              Windowing.AsOfJoin(left, right, on: "time", tolerance: null, suffix: "_right")
              Partitioning is the point: without it, Lag reaches across group boundaries.
              Incomplete rolling windows stay NaN. As-of is BACKWARD-ONLY (forward = look-ahead).
              An empty partition list treats the whole frame as one group, as SQL does.

            CategoricalSeries (v0.4) - dictionary-encoded text, optionally ordered
              CategoricalSeries.FromValues(name, values, categories: [...], ordered: true)
              new CategoricalSeries(name, codes, categories, ordered)    codes: -1 is missing
              .Categories .Codes .IsOrdered .CodeOf(s) .CategoryCounts()
              .ArgSort(descending: false)      by declared rank, not alphabetically
              .OneHot(dropFirst: false)        dropFirst leaves a baseline for a model w/ intercept
              .ToCodes()                       missing becomes NaN, NOT -1
              .ToText() .AddCategories(...) .RenameCategories(map) .ReorderCategories([...])
              .RemoveUnusedCategories()
              Categories are part of the column TYPE: Take keeps them all, and writing an
              unlisted value throws rather than widening the dictionary.

            Io.SqlReader / Io.SqlWriter (v0.4) - any ADO.NET provider, no package dependency
              SqlReader.Read(connection, sql, parameters, commandTimeout)
              SqlReader.FromReader(IDataReader)
              SqlWriter.Write(frame, connection, table, batchSize: 500)
              ALWAYS pass values via parameters, never string concatenation.
              Table and column names are checked as plain identifiers (they cannot be parameterised).

            Io.ExcelReader / Io.ExcelWriter (v0.4) - .xlsx without a spreadsheet library
              ExcelReader.Read(path, new ExcelOptions { SheetName = "Data", HasHeader = true })
              ExcelReader.SheetNames(path)
              ExcelWriter.Write(frame, path, sheetName: "Sheet1")
              Handles absent cells (gaps, not blanks), date serials, and the 1900 leap-year bug.

            Io.ArrowFile (v0.5) - Apache Arrow IPC, read and write, no package dependency
              ArrowFile.Write(frame, path)   ArrowFile.Write(frame, stream)
              ArrowFile.Read(path)           ArrowFile.Read(stream)
              Carries types and missing values across the boundary, so pyarrow/pandas read it
              directly. Verified BOTH WAYS against pyarrow - a round trip proves nothing here.
              Uncompressed: it trades file size for zero-copy reads.

            ChunkedFrame + Streaming (v0.5) - datasets larger than memory
              ChunkedFrame.FromCsv(path, chunkRows: 100_000, options)
              ChunkedFrame.FromFrame(frame, chunkRows)   .FromChunks(source, columns)
                .Chunks (IEnumerable<DataFrame>)  .ChunkRows  .ColumnNames
              Streaming.CountRows(source)
              Streaming.CountGroups(source, keys)          CALL THIS FIRST
              Streaming.GroupBy(source, keys, ("fare", "mean"), ("x", "sum"))
                sum, mean, min, max, count only. Median is refused - it cannot be done in
                bounded memory, and a version that kept every value would only look streaming.
              Streaming.Describe(source, columns) -> Count, Mean, StandardDeviation, Min, Max
              Streaming.Filter(source, (chunk, row) => ...)     bounded by what SURVIVES
              Streaming.FilterToFile(source, predicate, path)   bounded regardless
              Streaming.SortToFile(source, column, path, descending, temporaryDirectory)
              GroupBy memory is proportional to DISTINCT GROUPS, not rows: grouping a billion
              rows by country is trivial, grouping them by user id is not.
              Chunk size is a memory knob, not a parameter of the answer - column types are
              pinned from one sample rather than inferred per chunk.
              SortToFile writes CSV, so a SINGLE-column frame with missing values writes blank
              lines that a CSV reader cannot tell from padding. Two or more columns survive.
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

            Explain (v0.4) - namespace Gravicode.Science.GraviLearn.Explain
              PermutationImportance.Compute(model, x, y, repeats: 10, seed: 42)
              PermutationImportance.Ranked(...)   -> FeatureImportance(Feature, Mean, StandardDeviation)
              RUN IT ON HELD-OUT DATA; on the training set it measures memorisation.
              ShapleyValues.Exact(predict, instance, background)     <= 20 features
              ShapleyValues.Sample(predict, instance, background, samples: 200, seed: 42)
              ShapleyValues.Explain(predict, instances, background)  -> (PerRow, MeanAbsolute)
                predict is Func<NdArray, NdArray>: takes a batch, returns one value per row.
                Explain a PROBABILITY, not a class index - a step function attributes poorly.
                Attribution: .BaseValue .Contributions .Prediction .Ranked
              Calibration.Curve(probabilities, labels, bins: 10) -> CalibrationPoint list
              Calibration.ExpectedError(...), Calibration.BrierScore(...)
              new IsotonicRegression().Fit(x, y).Predict(x) / .PredictOne(v) / .Steps

            Resampling (v0.4) - namespace ...GraviLearn.Resampling
              Resampler.OverSample(x, y, seed), UnderSample(x, y, seed)
              Resampler.Smote(x, y, neighbours: 5, seed)     scale features first
              Resampler.ClassWeights(y) -> per-class multiplier, without touching the data
              Resampler.ClassBalance(y) -> (Label, Count) largest first
              RESAMPLE THE TRAINING SPLIT ONLY - doing it first leaks into the test set.

            Anomaly (v0.4) - namespace ...GraviLearn.Anomaly
              new OneClassSvm(nu: 0.05, kernel: SvmKernel.Rbf, gamma: null, tolerance: 1e-6).Fit(x)
                .Predict(x)  1 = normal, -1 = anomaly
                .DecisionFunction(x)  signed distance - use this to RANK alerts
                .SupportVectorCount .Gamma .Offset
              nu bounds the training outlier fraction above and the SV fraction below.
              Do not judge the model by scoring its own training data.

            Clustering.Hdbscan (v0.4) - density clustering with no single threshold
              new Hdbscan(minClusterSize: 5, minSamples: null).Fit(x)
                .Labels (-1 = noise) .ClusterCount .Probabilities .CoreDistances
                .FitPredict(x)
              Use it wherever a single DBSCAN eps cannot fit clusters of differing density.
              O(n^2) memory - a few thousand points is the practical ceiling.

            Linear.SparseLogisticRegression (v0.5) - trains on CSR without densifying
              new SparseLogisticRegression(learningRate: 1.0, maxIterations: 200,
                                           l2Penalty: 0.0, tolerance: 1e-7)
                .Fit(SparseMatrix x, NdArray y)   returns itself, so it chains
                .Predict(x) .PredictProbabilities(x) .Score(x, y)
                .Coefficients (rows = binary problems) .Intercepts .TopFeatures(count, problem)
                .IterationsRun .FinalLoss .Classes .FeatureCount
              Pair with TfidfVectorizer.FitTransformSparse / TransformSparse (GraviText).
              Same model as the dense path, not an approximation - compare COEFFICIENTS.
              Its weight vector is dense, so it bounds the FEATURE count, not the row count.
              On SEPARABLE data an unpenalised fit never converges: the MLE is at infinity.
              That is not a bug - add L2 to get a finite optimum.

            Distributed (v0.5) - namespace Gravicode.Science.GraviLearn.Distributed
              DataParallel.Partition(items, workers) -> Shard(Start, Count, End, Indices)
              DataParallel.AverageGradients(gradients, sampleCounts)   WEIGHTED, always
              DataParallel.Encode(NdArray) / Decode(byte[])
              IWorkerTransport: Publish(round, worker, payload), Collect(round), WorkerCount
                new InProcessTransport(workerCount)
                new FileTransport(directory, workerCount, timeout)   crosses machines
              new ParameterServer(transport).Contribute(worker, gradient, sampleCount)
                .Aggregate() .Round .Transport
              new DistributedForest(nTrees, maxDepth, seed).Fit(x, y, workers)
                .Predict(x) .PredictProbabilities(x) .Score(x, y) .Trees .TreeCount
              DistributedForest is BIT-IDENTICAL to single-process: tree t is seeded from
              seed + t * 7919, a function of its global index alone.
              A plain average of per-worker gradients equals the global one ONLY for equal
              shards, and Partition produces uneven ones whenever the count does not divide.
              Transports collect in WORKER order, not arrival order - float addition is not
              associative, so arrival order would make the answer depend on the scheduler.
              It splits computation, not memory: every worker fits on the whole training set.
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

            Tokenization (v0.4) - trainable sub-word tokenizers
              BpeTokenizer.Train(corpus, vocabularySize: 1000, minFrequency: 2, preTokenizer: null)
                .Encode(word) .Tokenize(text) .EncodeIds(text) .Decode(pieces) .DecodeIds(ids)
                .Merges .Vocabulary .Save(path) / BpeTokenizer.Load(path)
                The merge ORDER is the model; Save writes ranked merges, not a vocabulary.
                Words carry BpeTokenizer.EndOfWord on their last piece.
              UnigramTokenizer.Train(corpus, vocabularySize: 1000, seedSize: 10000)
                .Encode(text)      Viterbi - globally optimal, not greedy
                .SampleEncoding(text, rng, alpha: 0.2)   subword regularisation
                .Decode(pieces)    exactly reversible; whitespace is ENCODED, not split on
                .PieceCount .LogProbability(piece) .Save(path) / Load(path)

            Sequence (v0.4) - namespace Gravicode.Science.GraviText.Sequence
              new LinearChainCrf(labelCount)
                .Fit(emissionMatrices, tagSequences, epochs: 50, learningRate: 0.1, l2: 1e-4)
                .Decode(emissions)      Viterbi over the whole sequence
                .Marginals(emissions)   per-token confidence, forward-backward
                .LogPartition(emissions) .Score(...) .LogLikelihood(...)
                .Transition(from, to) .SetTransition(...) .Forbid(from, to) .ForbidStart(label)
                .ApplyBioConstraints(labelNames)   I-X may only follow B-X or I-X of the SAME type
              Emissions come from outside: any per-token scorer feeds it.
              The best sequence is NOT the sequence of best tokens.

            Tasks.TrainedNer (v0.4) - a learned entity tagger, unlike the rule-based recognizer
              TaggedSentence.LoadConll(path)   one token + BIO tag per line, blank line per sentence
              new TrainedNer().Fit(sentences, epochs: 30, learningRate: 0.1, crfEpochs: 60)
                .Tag(words) .Recognize(words) .Recognize(text)
                .Evaluate(sentences) -> EntityScore(Precision, Recall, F1, Predicted, Actual)
                .TokenAccuracy(sentences)   dominated by O - judge on entity F1 instead
              Features are shape-based, so it generalises to names never seen in training.

            Generation (v0.4) - namespace Gravicode.Science.GraviText.Generation
              new TransformerDecoder(config, vocabulary, rng)
                .Forward(tokenIds) .Logits(tokenIds) .NextTokenLogits(tokenIds)
                .Generate(prompt, maxNewTokens, options, rng, stopTokens)
                .Generate(text, tokenizer, maxNewTokens, options, rng)
                .CrossEntropy(tokenIds) .Perplexity(tokenIds)
              SamplingOptions(Temperature, TopK, TopP, RepetitionPenalty)
                SamplingOptions.Greedy, SamplingOptions.Nucleus
              CausalSelfAttention / TransformerDecoderLayer are the building blocks.
              Forward-only, like TransformerModel. Generation is quadratic (no KV cache).

            TransformerCheckpoint (v0.5) - loads EVERY parameter, not just the embeddings
              TransformerCheckpoint.Inspect(path) -> (Name, Shape)[]   RUN THIS FIRST
              TransformerCheckpoint.Load(model, path, names, strict: true) -> CheckpointReport
                CheckpointReport(.Loaded .Missing .Unused .IsComplete)
              CheckpointNames.HuggingFaceBert    bert.encoder.layer.{0}.attention.self.query.weight
              CheckpointNames.Unprefixed         encoder.layer.{0}....
              CheckpointNames.Reprefixed("roberta.")
              CheckpointNames has a Transposed flag: PyTorch nn.Linear stores (out, in) and
              computes x W^T, DenseLayer stores (in, out). Checked against the NON-SQUARE
              feed-forward weight, because a 768x768 projection accepts either reading.
              A PARTIAL load never sets HasPretrainedWeights, whatever strict mode was used.
              No weights ship with this repository. Export your own with torch.onnx.export.
              TransformerModel also exposes .Layers, .ReplaceTokenEmbeddings,
              .ReplacePositionEmbeddings and .MarkPretrained for a hand-rolled loader.

            Vectorization (v0.5) - sparse output
              new TfidfVectorizer(options).FitTransformSparse(documents) -> SparseMatrix
                .TransformSparse(documents)
              Feed it straight to GraviLearn's SparseLogisticRegression.
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

            HeterogeneousGraph (v0.4) - typed nodes and relations
              new HeterogeneousGraph()
                .AddNodeType(type, count) .CountOf(type)
                .AddEdge("user", "bought", "item", source, target, weight)
                .AddEdge(EdgeType, source, target, weight)
                .SetFeatures(type, matrix) .Features(type)   each type may differ in width
                .SetLabels(type, labels) .Labels(type)
                .SetEdgeFeatures(edgeType, matrix) .EdgeFeatures(edgeType)
                .AddReverseEdges(edgeType, reverseName)   a SEPARATE relation, not symmetry
                .Edges(edgeType) .IncomingTypes(nodeType) .ToHomogeneous(out offsets)
              EdgeType(Source, Relation, Target) - the TRIPLE identifies a relation.
              Node indices are LOCAL to their type: user 0 and item 0 are different nodes.
              new RelationalConvolution(graph, inputSizes, outputSize, rng).Forward(graph, inputs)
                R-GCN: one weight matrix per relation, in-degree normalised PER RELATION.

            TemporalGraph (v0.4) - timestamped edges
              new TemporalGraph(directed: true).AddEdge(source, target, time, weight)
              TemporalGraph.LoadCsv(path, directed, hasHeader)
                .TemporallyReachable(source, startTime, maxGap)   respects edge ORDERING
                .Snapshot(from, to) .SnapshotUpTo(time) .Windows(count) .Collapse()
                .TemporalEfficiency()   how much a static view overstates
                .TimeDecayedFeatures(features, asOf, halfLife)
                .Edges (time-sorted) .TimeRange .NodeCount .EdgeCount
              A static graph implies paths that the ordering forbids. SnapshotUpTo is the cut
              that stops a link predictor being trained on its own test set.

            GraphPooling / GraphClassifier (v0.4) - whole-graph tasks
              GraphPooling.Pool(nodeFeatures, PoolingKind.Mean | Sum | Max | MeanMax)
              GraphPooling.AttentionPool(nodeFeatures, gate) -> (Pooled, Weights)
              Every readout is permutation-invariant - graph nodes have no canonical numbering.
              Mean is size-invariant; Sum is not; Max detects presence.
              new GraphClassifier(inputSize, hiddenSize, layers, pooling, rng)
                .Fit(graphs, labels, features, regularisation) .Predict(graph) .Score(graph)
                .Embed(graph, nodeFeatures) .Accuracy(graphs, labels)

            NeighborSampler (v0.4) - GraphSAGE-style bounded sampling
              NeighborSampler.Sample(graph, targets, fanOut: [10, 5], rng, replace: false)
                -> SampledBlock(Nodes, Layers, TargetPositions)
              NeighborSampler.Batches(nodes, batchSize, rng)
              NeighborSampler.GatherFeatures(block, allFeatures)
              NeighborSampler.Aggregate(block, features, weights, activation)
              A SAGE layer takes TWICE its feature width (self and neighbourhood concatenated).
              Solves neighbourhood explosion, not memory: fan-out bounds the cost per target.
              NOTE: graph generators are static on Graph itself - Graph.Cycle(n), Graph.Complete(n),
              Graph.Random(nodes, p, seed) - and Random takes a SEED int, not a GraviRandom.
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

            MultivariateDistribution (v0.4) - distributions over vectors
              new MultivariateNormal(mean, covariance)
                MultivariateNormal.Standard(dim), .Diagonal(mean, variances)
                .LogDensity(x) .Sample(rng) .Sample(rng, count) .Mean .Covariance
                .CholeskyFactor .Conditional(unknown, observed, values)
                Everything runs off ONE Cholesky factor. Non-positive-definite is rejected.
              new Dirichlet(2, 3, 5) / Dirichlet.Symmetric(dim, concentration) / .Uniform(dim)
                .Mean .Covariance .Alpha .Concentration .Sample(rng)
                .Posterior(counts)   conjugate: the update is addition
                .Marginal(component) -> Beta
                Off-diagonal covariance is always negative (the components sum to one).
              new Multinomial(trials, p0, p1, ...)  .Sample(rng) .LogDensity(counts)

            GaussianProcess (v0.4) - a prior on the function, not on parameters
              new GaussianProcess(kernel, noise: 1e-6).Fit(x, y)
                .Predict(x) -> GpPrediction(.Mean .Variance .StandardDeviation .Interval(0.95))
                .PredictMean(x) .LogMarginalLikelihood() .SamplePosterior(x, count, rng)
              GaussianProcess.Optimise(x, y, lengthScales, noises)   grid search
              Kernels: new RbfKernel(lengthScale, variance), new MaternKernel(nu, ...)
                       new PeriodicKernel(period, ...), new SumKernel(a, b)
                nu must be 0.5, 1.5 or 2.5. The KERNEL is the model.
              Noise is NOT optional - it keeps the covariance invertible.
              x is rank 2 (samples, features) even for one input dimension.
              Cost is cubic in the observation count.

            KalmanFilter (v0.4) - linear-Gaussian state space
              KalmanFilter.LocalLevel(processVariance, observationVariance)
              KalmanFilter.LocalLinearTrend(levelVariance, slopeVariance, observationVariance)
              new KalmanFilter(transition, observation, processNoise, observationNoise)
                .Filter(observations) -> (Filtered, Predicted, LogLikelihood)
                .Smooth(observations)   uses the whole series - better, but look-ahead
                .Forecast(observations, horizon) .Simulate(steps, rng) .Observe(states)
              Observations are rank 2: (time, observationDimension).
              Only the RATIO of Q to R matters, so one number tunes the filter.
              LogLikelihood is what to maximise when fitting the variances.

            ModelComparison (v0.4) - out-of-sample predictive accuracy
              ModelComparison.Waic(logLikelihood)   (draws x observations) matrix
              ModelComparison.Loo(logLikelihood) -> LooResult(.Criterion .ParetoK)
                .IsReliable .UnreliableObservations    the diagnostic WAIC does not have
              ModelComparison.Compare(dictionary of name -> InformationCriterion)
              InformationCriterion(.Estimate .EffectiveParameters .StandardError .Pointwise)
              Deviance scale: LOWER IS BETTER. Only DIFFERENCES are interpretable, and only
              between models scored on the same observations.
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
                     "sentiment, graph, bayesian, explainability, anomaly, tokenizer, ner, " +
                     "forecasting, gaussianprocess, windowfunctions, heterogeneousgraph, " +
                     "arrow, outofcore, sparse, distributed or pretrained.")]
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

            "explainability" => """
                using Gravicode.Science.GraviLearn;
                using Gravicode.Science.GraviLearn.Explain;
                using Gravicode.Science.GraviLearn.ModelSelection;
                using Gravicode.Science.GraviLearn.Trees;
                using Gravicode.Science.GraviNum;

                var data = Datasets.LoadIris();
                var split = Selection.Split(data.Features, data.Target, testSize: 0.3, seed: 42, stratify: true);

                var model = new RandomForestClassifier(nTrees: 100, seed: 42);
                model.Fit(split.TrainX, split.TrainY);

                // Held-out data: on the training set this measures memorisation, not generalisation.
                foreach (var importance in PermutationImportance.Ranked(model, split.TestX, split.TestY, repeats: 10))
                    Console.WriteLine($"{data.FeatureNames[importance.Feature],-16} " +
                                      $"{importance.Mean,7:F4} +/- {importance.StandardDeviation:F4}");

                // Explaining one row. Attribute the PROBABILITY, not the class index: a step
                // function attributes poorly, because most perturbations do not move it at all.
                var instance = split.TestX.Row(0);

                NdArray AsRow(NdArray vector)
                {
                    var matrix = NdArray.Zeros(1, vector.Size);
                    for (var i = 0; i < vector.Size; i++) matrix[0, i] = vector.At(i);
                    return matrix;
                }

                var predicted = (int)model.Predict(AsRow(instance)).At(0);

                NdArray ClassProbability(NdArray batch)
                {
                    var probabilities = model.PredictProbabilities(batch);
                    var column = NdArray.Zeros(batch.Shape[0]);
                    for (var i = 0; i < batch.Shape[0]; i++) column.SetAt(i, probabilities[i, predicted]);
                    return column;
                }

                var attribution = ShapleyValues.Sample(ClassProbability, instance, split.TrainX, samples: 200);
                Console.WriteLine($"base {attribution.BaseValue:F4} -> prediction {attribution.Prediction:F4}");
                foreach (var (feature, contribution) in attribution.Ranked)
                    Console.WriteLine($"  {data.FeatureNames[feature],-16} {contribution,+8:F4}");
                """,

            "anomaly" => """
                using Gravicode.Science.GraviLearn.Anomaly;
                using Gravicode.Science.GraviLearn.Clustering;
                using Gravicode.Science.GraviNum;

                var rng = new GraviRandom(42);
                var normal = NdArray.Zeros(400, 2);
                for (var i = 0; i < 400; i++)
                {
                    normal[i, 0] = rng.Normal();
                    normal[i, 1] = rng.Normal();
                }

                // nu is a statement about contamination, not a tolerance to tune: it bounds the
                // training outlier fraction above and the support-vector fraction below.
                var detector = new OneClassSvm(nu: 0.05).Fit(normal);

                var probes = NdArray.FromArray(new double[,] { { 0, 0 }, { 5, 5 } });
                var scores = detector.DecisionFunction(probes);   // signed distance - ranks alerts

                for (var i = 0; i < probes.Shape[0]; i++)
                    Console.WriteLine($"({probes[i, 0]}, {probes[i, 1]}) {scores.At(i):F4} " +
                                      $"{(scores.At(i) >= 0 ? "normal" : "ANOMALY")}");

                // HDBSCAN where a single DBSCAN eps cannot fit clusters of differing density.
                var clusters = new Hdbscan(minClusterSize: 10).Fit(normal);
                Console.WriteLine($"{clusters.ClusterCount} clusters");
                """,

            "tokenizer" => """
                using Gravicode.Science.GraviNum;
                using Gravicode.Science.GraviText.Tokenization;

                string[] corpus = ["the cat sat on the mat", "the dog sat on the log"];

                // BPE: merge the most frequent adjacent pair, repeatedly. The merge ORDER is the
                // model, which is why Save writes ranked merges rather than a vocabulary.
                var bpe = BpeTokenizer.Train(corpus, vocabularySize: 200, minFrequency: 1);
                Console.WriteLine($"[{string.Join(", ", bpe.Encode("lowest"))}]");
                bpe.Save("merges.txt");

                // Unigram: prune a large candidate vocabulary by EM, segment by Viterbi. Whitespace
                // is encoded rather than split on, so decoding is exactly reversible.
                var unigram = UnigramTokenizer.Train(corpus, vocabularySize: 120, seedSize: 500);
                Console.WriteLine($"[{string.Join(", ", unigram.Encode("the cat sat"))}]");

                // Only unigram can sample alternatives - that is subword regularisation.
                var rng = new GraviRandom(7);
                Console.WriteLine(string.Join(" ", unigram.SampleEncoding("the cat sat", rng, alpha: 0.2)));
                """,

            "ner" => """
                using Gravicode.Science.GraviText.Tasks;

                // CoNLL columns: one token and its BIO tag per line, blank line between sentences.
                var sentences = TaggedSentence.LoadConll("datasets/ner_conll.txt");
                var cut = (int)(sentences.Count * 0.75);

                var ner = new TrainedNer().Fit(sentences.Take(cut).ToList());
                var heldOut = sentences.Skip(cut).ToList();

                // Score ENTITIES, not tokens: token accuracy is dominated by the O tag, so a model
                // predicting O everywhere scores above 85% on most corpora.
                Console.WriteLine(ner.Evaluate(heldOut));

                foreach (var entity in ner.Recognize("Kartika Wijaya bekerja di Gravicode ."))
                    Console.WriteLine($"{entity.Type,-4} {entity.Text}");
                """,

            "forecasting" => """
                using Gravicode.Science.GraviNum;
                using Gravicode.Science.GraviProb;

                // A local level model is an EWMA whose smoothing constant is derived from the
                // noise ratio rather than guessed - and it reports its own uncertainty.
                var filter = KalmanFilter.LocalLevel(processVariance: 0.05, observationVariance: 1.0);

                // Observations are rank 2: (time, observationDimension).
                var (truth, observations) = filter.Simulate(200, new GraviRandom(42));

                var filtered = filter.Filter(observations);   // past only - what a live system can do
                var smoothed = filter.Smooth(observations);   // whole series - better, but look-ahead

                Console.WriteLine($"log likelihood {filtered.LogLikelihood:F2}");

                // Only the RATIO of Q to R matters, so the likelihood fits it with one number.
                foreach (var q in new[] { 0.01, 0.05, 0.2 })
                    Console.WriteLine($"Q={q}: {KalmanFilter.LocalLevel(q, 1.0).Filter(observations).LogLikelihood:F2}");

                // No observations arrive during a forecast, so uncertainty grows as it must.
                foreach (var step in filter.Forecast(observations, horizon: 10))
                    Console.WriteLine($"{step.Mean.At(0):F4} +/- {step.StandardDeviation.At(0):F4}");
                """,

            "gaussianprocess" => """
                using Gravicode.Science.GraviNum;
                using Gravicode.Science.GraviProb;

                // x is rank 2 (samples, features) even in one input dimension.
                var x = NdArray.Zeros(12, 1);
                var y = NdArray.Zeros(12);
                for (var i = 0; i < 12; i++)
                {
                    var value = i * 2 * Math.PI / 12;
                    x[i, 0] = value;
                    y.SetAt(i, Math.Sin(value));
                }

                // The KERNEL is the model. Noise is not optional: it is what keeps the covariance
                // invertible when inputs are close together.
                var gp = new GaussianProcess(new RbfKernel(lengthScale: 1.0), noise: 1e-6).Fit(x, y);

                var probe = NdArray.Zeros(1, 1);
                probe[0, 0] = 1.5;

                var prediction = gp.Predict(probe);
                Console.WriteLine($"{prediction.Mean.At(0):F4} +/- {prediction.StandardDeviation.At(0):F4}");
                Console.WriteLine($"log marginal likelihood {gp.LogMarginalLikelihood():F4}");

                // Grid search, not gradient descent: the marginal likelihood is not concave and
                // has real local optima - one explaining the data as signal, another as noise.
                var tuned = GaussianProcess.Optimise(x, y);
                Console.WriteLine(tuned.Kernel.Name);
                """,

            "windowfunctions" => """
                using Gravicode.Science.GraviFrame;

                var sales = new DataFrame(
                [
                    new TextSeries("customer", ["a", "b", "a", "b"]),
                    new NumericSeries("amount", [10, 100, 20, 200]),
                ]);

                // Partitioning is the point: without it, Lag reaches across group boundaries and
                // each group's first row picks up the previous group's last. That is a real leak.
                var running = Windowing.CumulativeSum(sales, ["customer"], "amount");
                var previous = Windowing.Lag(sales, ["customer"], "amount");
                var rolling = Windowing.RollingMean(sales, ["customer"], "amount", window: 2);

                // As-of join: backward-only. Matching the NEAREST row in either direction is
                // look-ahead, and is how a backtest ends up predicting the past.
                var trades = new DataFrame([new NumericSeries("time", [10, 25])]);
                var quotes = new DataFrame(
                [
                    new NumericSeries("time", [5, 20]),
                    new NumericSeries("price", [100, 200]),
                ]);

                Console.WriteLine(Windowing.AsOfJoin(trades, quotes, "time"));
                """,

            "heterogeneousgraph" => """
                using Gravicode.Science.GraviGraph;
                using Gravicode.Science.GraviNum;

                var graph = new HeterogeneousGraph();
                graph.AddEdge("user", "viewed", "item", 0, 1);
                graph.AddEdge("user", "bought", "item", 1, 2);

                // Node indices are LOCAL to their type, which is what lets each type carry a
                // different feature width. An edge type is the TRIPLE, not the relation name.
                graph.SetFeatures("user", new GraviRandom(3).StandardNormal(4, 6));
                graph.SetFeatures("item", new GraviRandom(5).StandardNormal(5, 3));

                // Messages flow along edge direction only, so items cannot inform users without
                // the reverse relation - added separately, because it deserves its own weights.
                graph.AddReverseEdges(new EdgeType("user", "viewed", "item"));

                var layer = new RelationalConvolution(graph,
                    new Dictionary<string, int> { ["user"] = 6, ["item"] = 3 }, outputSize: 8);

                var output = layer.Forward(graph, new Dictionary<string, NdArray>
                {
                    ["user"] = graph.Features("user"),
                    ["item"] = graph.Features("item"),
                });

                Console.WriteLine($"user -> [{string.Join(", ", output["user"].Shape.ToArray())}]");
                """,

            "arrow" => """
                using Gravicode.Science.GraviFrame;
                using Gravicode.Science.GraviFrame.Io;

                var frame = new DataFrame(
                [
                    new TextSeries("symbol", ["BBCA", "TLKM", null]),
                    new NumericSeries("close", [9250.0, 3120.0, double.NaN]),
                    new BooleanSeries("halted", [false, true, null]),
                ]);

                ArrowFile.Write(frame, "quotes.arrow");
                var back = ArrowFile.Read("quotes.arrow");

                // Types and missing values cross the boundary intact, which is the point -
                // pyarrow reads this file directly, no CSV parsing and no type re-inference.
                Console.WriteLine(back);
                Console.WriteLine(string.Join(", ", back.ColumnNames.Select(n => $"{n}={back[n].DataType}")));
                Console.WriteLine($"missing preserved: {back["close"].IsMissing(2)}");

                // On the Python side:
                //   import pyarrow as pa
                //   pa.ipc.open_file("quotes.arrow").read_all().to_pandas()
                """,

            "outofcore" => """
                using Gravicode.Science.GraviFrame;

                // Reads the file a block at a time - at no point is more than chunkRows in memory.
                var chunked = ChunkedFrame.FromCsv("titanic.csv", chunkRows: 50_000);

                Console.WriteLine($"{Streaming.CountRows(chunked)} rows");

                // GroupBy memory is proportional to DISTINCT GROUPS, not rows, so check the
                // key's cardinality first: that number is the difference between a query that
                // runs and one that does not.
                Console.WriteLine($"{Streaming.CountGroups(chunked, ["pclass", "sex"])} groups");

                var summary = Streaming.GroupBy(chunked, ["pclass", "sex"],
                    ("fare", "mean"), ("survived", "mean"));
                Console.WriteLine(summary.SortBy([("pclass", true), ("sex", true)]).ToString(12));

                // One pass, constant memory, Welford's variance rather than E[x^2] - E[x]^2.
                foreach (var (name, stats) in Streaming.Describe(chunked, ["fare", "age"]))
                    Console.WriteLine($"{name}: n={stats.Count} mean={stats.Mean:F3} sd={stats.StandardDeviation:F3}");

                // External merge sort: two passes and disk space equal to the input. That is
                // the trade, and it is what lets a sort exceed memory.
                Streaming.SortToFile(chunked, "fare", "by-fare.csv", descending: true);
                Streaming.FilterToFile(chunked, (chunk, row) => chunk.Numeric("fare")[row] > 100, "rich.csv");
                """,

            "sparse" => """
                using Gravicode.Science.GraviLearn.Linear;
                using Gravicode.Science.GraviNum;
                using Gravicode.Science.GraviText.Vectorization;

                // A bag-of-words matrix is around 1% non-zero. The dense copy is almost entirely
                // zeros that cost the same to store and multiply as any other number - the memory
                // wall is why this exists, and the speed is a consequence.
                var vectoriser = new TfidfVectorizer(new VectorizerOptions { MaxFeatures = 30_000 });
                var x = vectoriser.FitTransformSparse(documents);

                var model = new SparseLogisticRegression(learningRate: 1.0, maxIterations: 200, l2Penalty: 0.01)
                    .Fit(x, labels);

                Console.WriteLine($"accuracy {model.Score(x, labels):P2} after {model.IterationsRun} iterations");

                // The practical reason to keep a linear model on text: a coefficient reads
                // directly as "this word moves the decision this far".
                foreach (var (feature, weight) in model.TopFeatures(15))
                    Console.WriteLine($"{vectoriser.Vocabulary[feature],-20} {weight,8:F4}");

                // Without L2 on separable data this never converges - the MLE is at infinity.
                """,

            "distributed" => """
                using Gravicode.Science.GraviLearn.Distributed;
                using Gravicode.Science.GraviNum;

                // Bit-identical to single-process training, not merely equivalent: tree t is
                // seeded from seed + t * 7919, a function of its global index alone, so a shard
                // boundary cannot change the answer.
                var forest = new DistributedForest(nTrees: 500, maxDepth: 12, seed: 42)
                    .Fit(x, y, workers: 8);

                Console.WriteLine($"accuracy {forest.Score(xTest, yTest):P2}");

                // Or the pieces, for a hand-written loop:
                var shards = DataParallel.Partition(items: x.Shape[0], workers: 8);

                using var transport = new InProcessTransport(workerCount: 8);
                var server = new ParameterServer(transport);

                foreach (var shard in shards)
                    server.Contribute(worker, GradientOver(shard), sampleCount: shard.Count);

                // WEIGHTED by sample count. A plain average equals the global gradient only for
                // equal shards, and Partition produces uneven ones whenever the count does not
                // divide - unweighted, the model trains to something slightly wrong that no
                // shape or convergence check would catch.
                var averaged = server.Aggregate();

                // FileTransport is the same interface over a shared directory: no broker, no
                // ports, and it crosses machines. Workers write then rename, so a collector
                // cannot read a half-written payload.
                """,

            "pretrained" => """
                using Gravicode.Science.GraviText.Transformers;

                // Names are a convention and the file is the only authority on which one it
                // follows, so look before loading.
                foreach (var (name, shape) in TransformerCheckpoint.Inspect("bert-base-uncased.onnx").Take(5))
                    Console.WriteLine($"{name} [{string.Join(", ", shape)}]");

                var model = new TransformerModel("bert-base", vocabularySize: 30522);
                var report = TransformerCheckpoint.Load(model, "bert-base-uncased.onnx");

                Console.WriteLine(report);                        // loaded 196, missing 0, unused 3
                Console.WriteLine(model.HasPretrainedWeights);

                // A different naming convention is usually the whole adaptation:
                //   CheckpointNames.Reprefixed("roberta.")
                //   CheckpointNames.Unprefixed

                // No weights ship with this repository. Export your own:
                //   from transformers import AutoModel
                //   import torch
                //   torch.onnx.export(AutoModel.from_pretrained("bert-base-uncased"),
                //                     torch.zeros(1, 8, dtype=torch.long),
                //                     "bert-base-uncased.onnx",
                //                     input_names=["input_ids"], opset_version=13)
                """,

            _ => "Unknown task. Try: classification, regression, clustering, dataframe, " +
                 "explainability, anomaly, tokenizer, ner, forecasting, gaussianprocess, " +
                 "windowfunctions, heterogeneousgraph, arrow, outofcore, sparse, distributed, " +
                 "pretrained, " +
                 "timeseries, sentiment, graph, bayesian.",
        };
    }
}
