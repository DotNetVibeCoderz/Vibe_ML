using System.Text;

namespace ScienceAppGen.Services;

/// <summary>A starting point offered by the New Project dialog.</summary>
/// <param name="Id">Stable identifier.</param>
/// <param name="Name">Display name.</param>
/// <param name="Category">Grouping in the dialog.</param>
/// <param name="Description">One line explaining what it produces.</param>
/// <param name="Libraries">Which Gravicode libraries it uses, for the spectrum indicator.</param>
/// <param name="Files">Relative path to file content.</param>
public sealed record ProjectTemplate(
    string Id,
    string Name,
    string Category,
    string Description,
    IReadOnlyList<string> Libraries,
    IReadOnlyDictionary<string, string> Files)
{
    /// <summary>
    /// The spectrum colour per library, so a card's bands read as its actual ingredients rather
    /// than as a count of anonymous ticks. These match the six bands in <c>Themes/Tokens.axaml</c>.
    /// </summary>
    public IReadOnlyList<string> LibraryColours => Libraries.Select(l => l switch
    {
        "GraviNum" => "#E0555A",
        "GraviFrame" => "#E08A3C",
        "GraviLearn" => "#D9C04A",
        "GraviText" => "#56B87F",
        "GraviGraph" => "#3FA9C9",
        "GraviProb" => "#7B7BD6",
        _ => "#586475",
    }).ToList();
}

/// <summary>
/// The project templates. Every one of them builds and runs as written.
/// </summary>
/// <remarks>
/// These are held in code rather than as loose files on disk so a template cannot go missing from
/// an installed copy, and so the project name can be substituted properly rather than by a
/// find-and-replace over a directory.
/// </remarks>
public static class TemplateService
{
    /// <summary>Every available template, in the order the dialog shows them.</summary>
    public static IReadOnlyList<ProjectTemplate> All { get; } = Build();

    /// <summary>Looks up a template by id.</summary>
    public static ProjectTemplate? Find(string id)
        => All.FirstOrDefault(t => string.Equals(t.Id, id, StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// Materialises a template into <paramref name="directory"/>, substituting the project name
    /// and wiring up the Gravicode.Science references.
    /// </summary>
    /// <remarks>
    /// The library references cannot be a fixed relative path: a project created in the user's
    /// Documents folder is nowhere near the repository, and <c>..\..\src\GraviNum</c> would point
    /// at a directory that does not exist. The path is therefore computed from the new project's
    /// location to wherever the libraries actually are.
    /// </remarks>
    public static void Create(ProjectTemplate template, string directory, string projectName)
    {
        Directory.CreateDirectory(directory);
        var source = ResolveLibraryPath(directory);

        foreach (var (relativePath, content) in template.Files)
        {
            var path = Path.Combine(directory, relativePath.Replace("$name$", projectName));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path,
                content.Replace("$name$", projectName).Replace("$gravicode$", source),
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        }
    }

    /// <summary>
    /// The path a generated project should use to reach the Gravicode.Science libraries, relative
    /// to <paramref name="projectDirectory"/> when that is shorter than an absolute path.
    /// </summary>
    public static string ResolveLibraryPath(string projectDirectory)
    {
        var source = FindLibrarySource();
        if (source is null) return "";

        // A relative path keeps the generated project portable as long as it stays put next to
        // the repository; an absolute one is used when they are on different roots.
        try
        {
            var relative = Path.GetRelativePath(projectDirectory, source);
            return relative.Length < source.Length ? relative : source;
        }
        catch (ArgumentException)
        {
            return source;
        }
    }

    /// <summary>Locates the repository's <c>src</c> directory, or null when it cannot be found.</summary>
    public static string? FindLibrarySource()
    {
        foreach (var start in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            var directory = new DirectoryInfo(start);
            for (var depth = 0; directory is not null && depth < 12; depth++)
            {
                var candidate = Path.Combine(directory.FullName, "src", "GraviNum", "GraviNum.csproj");
                if (File.Exists(candidate)) return Path.Combine(directory.FullName, "src");
                directory = directory.Parent;
            }
        }
        return null;
    }

    // ---------------------------------------------------------------- definitions

    /// <summary>
    /// Builds a csproj that references the given libraries.
    /// </summary>
    /// <remarks>
    /// <c>$gravicode$</c> is replaced at creation time with the real path to the libraries; see
    /// <see cref="ResolveLibraryPath"/>. Writing a fixed relative path here would break every
    /// project created outside the repository.
    /// </remarks>
    private static string Csproj(params string[] libraries)
    {
        var references = string.Join("\n", libraries.Select(l =>
            $"    <ProjectReference Include=\"$gravicode$\\{l}\\{l}.csproj\" />"));

        return $$"""
            <Project Sdk="Microsoft.NET.Sdk">

              <PropertyGroup>
                <OutputType>Exe</OutputType>
                <TargetFramework>net10.0</TargetFramework>
                <Nullable>enable</Nullable>
                <ImplicitUsings>enable</ImplicitUsings>
                <RootNamespace>{{"$name$"}}</RootNamespace>
              </PropertyGroup>

              <ItemGroup>
                <!-- Paths were resolved when the project was created. Replace them with
                     PackageReference entries once the Gravicode.Science packages are published. -->
            {{references}}
              </ItemGroup>

            </Project>
            """;
    }

    private static IReadOnlyList<ProjectTemplate> Build() =>
    [
        // ------------------------------------------------------------ blank
        new ProjectTemplate(
            "blank",
            "Blank console app",
            "Starting points",
            "An empty .NET console project with nothing assumed.",
            [],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = """
                    <Project Sdk="Microsoft.NET.Sdk">

                      <PropertyGroup>
                        <OutputType>Exe</OutputType>
                        <TargetFramework>net10.0</TargetFramework>
                        <Nullable>enable</Nullable>
                        <ImplicitUsings>enable</ImplicitUsings>
                        <RootNamespace>$name$</RootNamespace>
                      </PropertyGroup>

                    </Project>
                    """,
                ["Program.cs"] = """
                    Console.WriteLine("$name$");
                    """,
                ["README.md"] = """
                    # $name$

                    Created with ScienceAppGen.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // ------------------------------------------------------------ array maths
        new ProjectTemplate(
            "numerics",
            "Numerical computing",
            "Data science",
            "Arrays, linear algebra, decompositions and statistics with GraviNum.",
            ["GraviNum"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviNum;

                    Console.WriteLine(GraviInfo.Banner("$name$"));
                    Console.WriteLine(GraviInfo.HardwareReport());
                    Console.WriteLine();

                    // Arrays are views over one shared buffer: reshape and transpose move no data.
                    var a = NdArray.Arange(12).Reshape(3, 4);
                    Console.WriteLine(a);

                    // Broadcasting stretches the length-4 vector across all three rows.
                    Console.WriteLine(a + NdArray.Arange(4));

                    var m = NdArray.FromArray(new double[,] { { 4, 7, 2 }, { 3, 6, 1 }, { 2, 5, 9 } });
                    Console.WriteLine($"det(A)  = {LinAlg.Determinant(m):F4}");
                    Console.WriteLine($"rank(A) = {LinAlg.MatrixRank(m)}");

                    var svd = Decomposition.Svd(m);
                    Console.WriteLine($"singular values: {string.Join(", ", svd.SingularValues.ToArray().Select(v => v.ToString("F4")))}");

                    var rng = new GraviRandom(seed: 42);
                    var samples = rng.Normal(mean: 100, stdDev: 15, 50_000);
                    foreach (var (key, value) in Statistics.Describe(samples))
                        Console.WriteLine($"  {key,-8}{value,12:F4}");
                    """,
            }),

        // ------------------------------------------------------------ dataframes
        new ProjectTemplate(
            "dataframe",
            "Data analysis",
            "Data science",
            "Load a CSV, clean it, group it and summarise it with GraviFrame.",
            ["GraviNum", "GraviFrame"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame"),
                // Four-quote delimiter, because the generated code itself contains a raw string.
                ["Program.cs"] = """"
                    using Gravicode.Science.GraviFrame;
                    using Gravicode.Science.GraviNum;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    // Swap this for your own file. Column types are inferred from the contents.
                    var csv = """
                        region,product,units,revenue
                        North,Widget,12,240.5
                        North,Gadget,7,455.0
                        South,Widget,19,380.0
                        South,Gadget,3,195.0
                        East,Widget,25,500.0
                        East,Gadget,11,715.0
                        """;

                    var df = DataFrame.ParseCsv(csv);
                    Console.WriteLine(df);
                    Console.WriteLine();
                    Console.WriteLine(df.Info());

                    Console.WriteLine("Revenue by region:");
                    Console.WriteLine(df.GroupBy("region").Sum("revenue").SortBy("revenue", ascending: false));

                    Console.WriteLine();
                    Console.WriteLine("Units by region and product:");
                    Console.WriteLine(df.Pivot("region", "product", "units"));

                    Console.WriteLine();
                    Console.WriteLine(df.Describe());
                    """",
            }),

        // ------------------------------------------------------------ ml pipeline
        new ProjectTemplate(
            "ml-pipeline",
            "Machine learning pipeline",
            "Machine learning",
            "Split, scale, reduce, train and evaluate a classifier end to end.",
            ["GraviNum", "GraviFrame", "GraviLearn"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame", "GraviLearn"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviLearn;
                    using Gravicode.Science.GraviLearn.Decomposition;
                    using Gravicode.Science.GraviLearn.ModelSelection;
                    using Gravicode.Science.GraviLearn.Preprocessing;
                    using Gravicode.Science.GraviLearn.Trees;
                    using Gravicode.Science.GraviNum;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    // MakeBlobs needs no data files. Swap in Datasets.LoadIris() or your own matrix.
                    var data = Datasets.MakeBlobs(samples: 600, features: 8, centers: 3, spread: 1.5, seed: 42);
                    Console.WriteLine($"{data.SampleCount} samples, {data.FeatureCount} features, {data.TargetNames.Count} classes");

                    var split = Selection.Split(data.Features, data.Target, testSize: 0.3, seed: 42, stratify: true);

                    // Keeping the steps in a Pipeline is what stops cross-validation leaking test
                    // statistics into training: every fold refits the scaler from scratch.
                    var pipeline = new Pipeline()
                        .Add(new StandardScaler())
                        .Add(new PCA(components: 4))
                        .Add(new RandomForestClassifier(nTrees: 100, seed: 42));

                    pipeline.Fit(split.TrainX, split.TrainY);

                    Console.WriteLine($"test accuracy: {pipeline.Score(split.TestX, split.TestY):P2}");
                    Console.WriteLine();
                    Console.WriteLine(Metrics.ClassificationReport(split.TestY, pipeline.Predict(split.TestX), data.LabelNames));

                    var cv = Selection.CrossValidatePipeline(
                        () => new Pipeline()
                            .Add(new StandardScaler())
                            .Add(new PCA(components: 4))
                            .Add(new RandomForestClassifier(nTrees: 100, seed: 42)),
                        data.Features, data.Target, folds: 5, stratified: true);

                    Console.WriteLine($"cross-validated: {cv}");
                    """,
            }),

        // ------------------------------------------------------------ clustering
        new ProjectTemplate(
            "clustering",
            "Clustering and exploration",
            "Machine learning",
            "k-means, DBSCAN and a Gaussian mixture on unlabelled data, scored by silhouette.",
            ["GraviNum", "GraviLearn"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviLearn"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviLearn;
                    using Gravicode.Science.GraviLearn.Clustering;
                    using Gravicode.Science.GraviLearn.Preprocessing;
                    using Gravicode.Science.GraviNum;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    var data = Datasets.MakeBlobs(samples: 400, features: 4, centers: 3, spread: 1.0, seed: 42);

                    // Distance-based clustering needs scaled features, or whichever column has the
                    // largest units decides the answer on its own.
                    var x = new StandardScaler().FitTransform(data.Features);

                    Console.WriteLine("Elbow curve:");
                    foreach (var (k, inertia) in KMeans.ElbowCurve(x, maxK: 6, seed: 42))
                        Console.WriteLine($"  k={k}  inertia {inertia,9:F2}  {new string('#', (int)(inertia / 40))}");

                    var kmeans = new KMeans(clusters: 3, seed: 42);
                    var labels = kmeans.FitPredict(x);
                    Console.WriteLine($"\nk-means silhouette: {Metrics.SilhouetteScore(x, labels):F4}");

                    var dbscan = new Dbscan(epsilon: 1.2, minSamples: 5);
                    dbscan.Fit(x);
                    Console.WriteLine($"DBSCAN: {dbscan.ClusterCount} clusters, {dbscan.NoiseCount} noise points");

                    var mixture = new GaussianMixture(components: 3, seed: 42);
                    mixture.Fit(x);
                    Console.WriteLine($"Gaussian mixture: converged={mixture.Converged}, BIC={mixture.Bic(x):F1}");
                    """,
            }),

        // ------------------------------------------------------------ nlp
        new ProjectTemplate(
            "nlp",
            "Text and sentiment",
            "Natural language",
            "Tokenize, vectorise and classify text in English and Bahasa Indonesia.",
            ["GraviNum", "GraviLearn", "GraviText"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviLearn", "GraviText"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviNum;
                    using Gravicode.Science.GraviText.Linguistics;
                    using Gravicode.Science.GraviText.Tasks;
                    using Gravicode.Science.GraviText.Tokenization;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    var tokenizer = new RegexTokenizer();
                    Console.WriteLine(string.Join(" | ", tokenizer.Tokenize("Gravicode Studios membangun AI di .NET!")));

                    Console.WriteLine($"\nen: connecting -> {PorterStemmer.Stem("connecting")}");
                    Console.WriteLine($"id: berlari    -> {IndonesianStemmer.Stem("berlari")}");

                    // The lexicon path needs no training data at all.
                    var analyzer = new SentimentAnalyzer();
                    foreach (var text in new[]
                             {
                                 "an excellent and wonderful result",
                                 "this is not good",
                                 "produknya bagus dan sangat memuaskan",
                                 "sangat mengecewakan dan membosankan",
                             })
                        Console.WriteLine($"  {analyzer.Analyze(text),-24} \"{text}\"");

                    // With labels, the supervised classifier is materially better.
                    string[] documents =
                    [
                        "excellent product, works perfectly", "wonderful quality and fast delivery",
                        "great value, highly recommend", "bagus sekali dan sangat memuaskan",
                        "terrible quality, broke immediately", "awful experience, waste of money",
                        "poor design and slow support", "jelek dan mengecewakan sekali",
                    ];
                    string[] labels = ["positive", "positive", "positive", "positive",
                                       "negative", "negative", "negative", "negative"];

                    var classifier = new TextClassifier().Train(documents, labels);
                    Console.WriteLine($"\ntraining accuracy: {classifier.Evaluate(documents, labels):P1}");
                    Console.WriteLine($"prediction: {classifier.Predict("a brilliant and satisfying purchase")}");

                    foreach (var label in classifier.Labels)
                        Console.WriteLine($"  {label,-10}{string.Join(", ", classifier.TopFeatures(label, 6).Select(t => t.Term))}");
                    """,
            }),

        // ------------------------------------------------------------ graphs
        new ProjectTemplate(
            "graph",
            "Network analysis",
            "Graphs",
            "PageRank, centrality, communities and a graph neural network.",
            ["GraviNum", "GraviGraph"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviText", "GraviGraph"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviGraph;
                    using Gravicode.Science.GraviGraph.Algorithms;
                    using Gravicode.Science.GraviGraph.Neural;
                    using Gravicode.Science.GraviNum;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    // Swap for Graph.Load("your-graph.json") to use real data.
                    var graph = Graph.Communities(communities: 3, sizePerCommunity: 40,
                        internalP: 0.25, externalP: 0.01, seed: 42);
                    Console.WriteLine(graph);

                    var rank = GraphAlgorithms.PageRank(graph);
                    Console.WriteLine($"\nPageRank total mass: {rank.Sum():F6}");
                    foreach (var node in Enumerable.Range(0, graph.NodeCount).OrderByDescending(i => rank.At(i)).Take(5))
                        Console.WriteLine($"  node {node,4}  {rank.At(node):F6}  degree {graph.Degree(node)}");

                    var (components, _) = GraphAlgorithms.ConnectedComponents(graph);
                    Console.WriteLine($"\ncomponents      : {components}");
                    Console.WriteLine($"avg clustering  : {GraphAlgorithms.AverageClusteringCoefficient(graph):F4}");

                    var communities = GraphAlgorithms.LabelPropagation(graph, seed: 42);
                    Console.WriteLine($"modularity      : {GraphAlgorithms.Modularity(graph, communities):F4}");

                    // Node features: a noisy one-hot of the community, so the GCN has signal to find.
                    var rng = new GraviRandom(42);
                    var features = NdArray.Zeros(graph.NodeCount, 6);
                    for (var i = 0; i < graph.NodeCount; i++)
                    {
                        for (var j = 0; j < 6; j++) features[i, j] = rng.Normal(0, 0.5);
                        features[i, graph.NodeLabels[i]] += 1.0;
                    }
                    graph.NodeFeatures = features;

                    var order = rng.Permutation(graph.NodeCount);
                    var train = order.Take(30).ToArray();
                    var test = order.Skip(30).ToArray();

                    var gcn = new GraphConvolutionalNetwork(hiddenSize: 16, learningRate: 0.05, epochs: 80, seed: 42)
                        .Train(graph, train, test);

                    Console.WriteLine($"\nGCN test accuracy: {gcn.Score(graph, test):P1}");
                    Console.WriteLine($"loss {gcn.History!.Loss[0]:F4} -> {gcn.History.Loss[^1]:F4}");
                    """,
            }),

        // ------------------------------------------------------------ bayesian
        new ProjectTemplate(
            "bayesian",
            "Bayesian inference",
            "Statistics",
            "Build a model, sample the posterior with MCMC and check it against the exact answer.",
            ["GraviNum", "GraviProb"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviProb"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviNum;
                    using Gravicode.Science.GraviProb;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    const int trials = 200;
                    const int heads = 125;

                    var model = new BayesianModel()
                        .AddDistribution("theta", Distribution.Beta(1, 1))
                        .AddObservation("data", DistributionSpec.Binomial(trials, "theta"), heads);

                    Console.Write(model);

                    // Beta is conjugate to the binomial, so the exact posterior is available and
                    // the sampler can be checked against ground truth rather than trusted.
                    var exact = Distribution.Beta(1, 1).PosteriorAfter(heads, trials - heads);
                    Console.WriteLine($"exact posterior : {exact.Name}, mean {exact.Mean:F6}");

                    var posterior = model.SampleMCMC(iterations: 20_000, chains: 4, warmup: 10_000, seed: 42);
                    Console.WriteLine();
                    Console.Write(posterior.Summary());

                    Console.WriteLine($"\nsampled mean    : {posterior.Mean("theta"):F6}");
                    Console.WriteLine($"absolute error  : {Math.Abs(posterior.Mean("theta") - exact.Mean):E2}");

                    var (low, high) = posterior.HighestDensityInterval("theta", 0.95);
                    Console.WriteLine($"95% HDI         : [{low:F4}, {high:F4}]");

                    var aboveHalf = posterior["theta"].ToArray().Count(v => v > 0.5) / (double)posterior.TotalDraws;
                    Console.WriteLine($"P(theta > 0.5)  : {aboveHalf:P2}");

                    // Can the fitted model reproduce the data it was fitted to?
                    var replicated = posterior.PosteriorPredictive(
                        (values, rng) => rng.Binomial(trials, values["theta"]), draws: 5000, seed: 42);
                    Console.WriteLine($"\nposterior predictive mean: {Statistics.Mean(replicated):F2} (observed {heads})");
                    """,
            }),

        // ------------------------------------------------------------ time series
        new ProjectTemplate(
            "timeseries",
            "Time series analysis",
            "Data science",
            "Rolling windows, percentage change and calendar resampling.",
            ["GraviNum", "GraviFrame"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviFrame;
                    using Gravicode.Science.GraviNum;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    // A synthetic price series so the template runs with no data file.
                    var rng = new GraviRandom(42);
                    var rows = new List<string> { "date,close" };
                    var price = 100.0;
                    var day = new DateTime(2024, 1, 1);
                    for (var i = 0; i < 400; i++)
                    {
                        price *= Math.Exp(0.0004 + 0.015 * rng.Normal());
                        rows.Add($"{day.AddDays(i):yyyy-MM-dd},{price:F4}");
                    }

                    var df = DataFrame.ParseCsv(string.Join('\n', rows)).SortBy("date");
                    var close = df.Numeric("close");

                    var enriched = df
                        .WithColumn(close.Rolling(window: 7).Mean().Rename("ma7"))
                        .WithColumn(close.Rolling(window: 30).Mean().Rename("ma30"))
                        .WithColumn(close.PercentChange().Rename("daily_return"))
                        .WithColumn(close.Rolling(window: 30).Std().Rename("volatility30"));

                    Console.WriteLine(enriched.SelectColumns("date", "close", "ma7", "ma30").Tail(10));

                    var returns = enriched.Numeric("daily_return");
                    Console.WriteLine($"\nmean daily return : {returns.Mean():P4}");
                    Console.WriteLine($"annualised vol    : {returns.Std() * Math.Sqrt(252):P2}");

                    Console.WriteLine("\nMonthly means:");
                    Console.WriteLine(Resampling.Resample(df, "date", ResampleFrequency.Monthly, "mean", ["close"]));
                    """,
            }),

        // ------------------------------------------------------ model explanation
        new ProjectTemplate(
            "explainability",
            "Model explanation",
            "Machine learning",
            "Permutation importance, Shapley values and calibration for a fitted model.",
            ["GraviNum", "GraviLearn"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame", "GraviLearn"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviLearn;
                    using Gravicode.Science.GraviLearn.Explain;
                    using Gravicode.Science.GraviLearn.ModelSelection;
                    using Gravicode.Science.GraviLearn.Trees;
                    using Gravicode.Science.GraviNum;

                    // Datasets.LoadIris() needs the repository's datasets/ folder, so a project
                    // created elsewhere generates its data instead.
                    var data = Datasets.MakeBlobs(samples: 400, features: 4, centers: 3, spread: 2.5, seed: 42);
                    var split = Selection.Split(data.Features, data.Target, testSize: 0.3, seed: 42, stratify: true);

                    var model = new RandomForestClassifier(nTrees: 100, seed: 42);
                    model.Fit(split.TrainX, split.TrainY);
                    Console.WriteLine($"accuracy: {Metrics.Accuracy(split.TestY, model.Predict(split.TestX)):P2}\n");

                    // Which features does the model rely on? Measured on HELD-OUT data - on the
                    // training set this measures memorisation rather than what generalises.
                    Console.WriteLine("permutation importance:");
                    foreach (var importance in PermutationImportance.Ranked(model, split.TestX, split.TestY, repeats: 10))
                        Console.WriteLine($"  {data.FeatureNames[importance.Feature],-16} " +
                                          $"{importance.Mean,7:F4} +/- {importance.StandardDeviation:F4}");

                    // Why THIS row? Explaining the probability rather than the hard label - a class
                    // index is a step function, and attributing a step says far less than
                    // attributing the confidence behind it.
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

                    Console.WriteLine($"\nexplaining P({data.LabelNames[predicted]}) for one sample");
                    Console.WriteLine($"  base value {attribution.BaseValue:F4} (average over the background)");
                    foreach (var (feature, contribution) in attribution.Ranked)
                        Console.WriteLine($"  {data.FeatureNames[feature],-16} {contribution,+8:F4}");
                    Console.WriteLine($"  they sum to the prediction: {attribution.Prediction:F4}");

                    // A model can rank perfectly and still be badly calibrated. Accuracy and AUC
                    // cannot see it, because neither depends on the numbers themselves.
                    var probabilities = model.PredictProbabilities(split.TestX);
                    var positive = NdArray.Zeros(split.TestY.Size);
                    var binary = NdArray.Zeros(split.TestY.Size);
                    for (var i = 0; i < split.TestY.Size; i++)
                    {
                        positive.SetAt(i, probabilities[i, 0]);
                        binary.SetAt(i, split.TestY.At(i) == 0 ? 1 : 0);
                    }

                    Console.WriteLine($"\nexpected calibration error: {Calibration.ExpectedError(positive, binary):F4}");
                    Console.WriteLine($"Brier score               : {Calibration.BrierScore(positive, binary):F4}");
                    """,
                ["README.md"] = """
                    # $name$

                    Explaining a fitted model three ways.

                    - **Permutation importance** — which features the model relies on overall.
                      Run it on held-out data; on the training set it measures memorisation.
                    - **Shapley values** — why the model said *that*, for *this* row. Explains
                      the probability rather than the label, and the contributions sum to the
                      prediction exactly.
                    - **Calibration** — whether the probabilities mean what they say. A model
                      can rank perfectly and still be badly calibrated, and accuracy cannot see it.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // ------------------------------------------------------- anomaly detection
        new ProjectTemplate(
            "anomaly",
            "Anomaly detection",
            "Machine learning",
            "One-class SVM and HDBSCAN for novelty detection and density clustering.",
            ["GraviNum", "GraviLearn"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame", "GraviLearn"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviLearn.Anomaly;
                    using Gravicode.Science.GraviLearn.Clustering;
                    using Gravicode.Science.GraviNum;

                    var rng = new GraviRandom(42);

                    // Normal operating data: one cloud, no labels. Novelty detection is not
                    // classification with a class missing - there are no negatives to learn a
                    // boundary between, so the task is to wrap the normal data tightly.
                    var normal = NdArray.Zeros(400, 2);
                    for (var i = 0; i < 400; i++)
                    {
                        normal[i, 0] = rng.Normal();
                        normal[i, 1] = rng.Normal();
                    }

                    // nu says "about this fraction of the training data is contamination worth
                    // excluding". It bounds the outlier fraction above and the support-vector
                    // fraction below - it is not a tolerance to tune until the answer looks right.
                    var detector = new OneClassSvm(nu: 0.05).Fit(normal);
                    Console.WriteLine($"support vectors: {detector.SupportVectorCount} of 400, gamma {detector.Gamma:F4}");

                    var probes = NdArray.FromArray(new double[,] { { 0, 0 }, { 1, 1 }, { 5, 5 }, { -7, 2 } });
                    var scores = detector.DecisionFunction(probes);

                    Console.WriteLine("\nsigned distance from the boundary - use it to RANK alerts:");
                    for (var i = 0; i < probes.Shape[0]; i++)
                        Console.WriteLine($"  ({probes[i, 0],5:F1}, {probes[i, 1],5:F1})  {scores.At(i),9:F4}  " +
                                          $"{(scores.At(i) >= 0 ? "normal" : "ANOMALY")}");

                    // HDBSCAN finds structure without being told a density threshold. DBSCAN needs
                    // one eps for the whole dataset, which fails as soon as clusters differ in
                    // density - there is then no value that works.
                    var mixed = NdArray.Zeros(150, 2);
                    for (var i = 0; i < 50; i++)   { mixed[i, 0] = rng.Normal() * 0.3;      mixed[i, 1] = rng.Normal() * 0.3; }
                    for (var i = 50; i < 100; i++) { mixed[i, 0] = 3 + rng.Normal() * 0.3;  mixed[i, 1] = rng.Normal() * 0.3; }
                    for (var i = 100; i < 150; i++){ mixed[i, 0] = 25 + rng.Normal() * 3.0; mixed[i, 1] = 25 + rng.Normal() * 3.0; }

                    var clusters = new Hdbscan(minClusterSize: 10).Fit(mixed);
                    Console.WriteLine($"\nHDBSCAN found {clusters.ClusterCount} clusters");

                    var noise = 0;
                    for (var i = 0; i < clusters.Labels.Size; i++) if (clusters.Labels.At(i) < 0) noise++;
                    Console.WriteLine($"  {noise} points classed as noise");
                    Console.WriteLine("  membership strength is often more useful than the flat label:");
                    Console.WriteLine("  a point at 0.05 is nominally clustered and practically noise");
                    """,
                ["README.md"] = """
                    # $name$

                    Two unsupervised approaches to finding what does not belong.

                    - **One-class SVM** — learns the shape of "normal" from unlabelled data.
                      `nu` is the dial that matters: it bounds the training-set outlier fraction
                      above and the support-vector fraction below. Scale your features first.
                    - **HDBSCAN** — density clustering with no single density threshold, which is
                      what lets it handle clusters that differ in density. `minClusterSize` is a
                      question about the problem, not about the data's scale.

                    Evaluate the detector on data it has not seen: a support vector appears in its
                    own decision function, so training points score higher than they should.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // --------------------------------------------------------------- tokenizer
        new ProjectTemplate(
            "tokenizer",
            "Sub-word tokenizer",
            "Natural language",
            "Train a BPE or SentencePiece-style tokenizer on your own corpus.",
            ["GraviNum", "GraviText"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame", "GraviLearn", "GraviText"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviNum;
                    using Gravicode.Science.GraviText.Tokenization;

                    // Replace this with your own text. A real vocabulary wants far more than this.
                    string[] corpus =
                    [
                        "the cat sat on the mat", "the dog sat on the log",
                        "a cat and a dog", "the mat and the log",
                        "lowest newest widest slowest", "low lower lowest",
                    ];

                    // ---------------------------------------------------------------- BPE
                    // Merge the most frequent adjacent pair, over and over. The merge ORDER is the
                    // model: the same token set applied in a different order segments differently,
                    // which is why Save writes ranked merges rather than a vocabulary.
                    var bpe = BpeTokenizer.Train(corpus, vocabularySize: 200, minFrequency: 1);

                    Console.WriteLine($"BPE: {bpe.Merges.Count} merges, {bpe.Vocabulary.Count} tokens");
                    foreach (var word in new[] { "lowest", "sat", "unseen" })
                        Console.WriteLine($"  {word,-8} -> [{string.Join(", ", bpe.Encode(word))}]");

                    Console.WriteLine($"  round trip: '{bpe.Decode(bpe.Tokenize("the cat sat"))}'");

                    // ------------------------------------------------------------ unigram
                    // A different algorithm, not a variant: start from a large candidate set and
                    // prune downwards by EM. Segmentation is Viterbi, so it is globally optimal
                    // rather than a greedy artefact of the order rules were learned in.
                    var unigram = UnigramTokenizer.Train(corpus, vocabularySize: 120, seedSize: 500);

                    Console.WriteLine($"\nUnigram: {unigram.PieceCount} pieces");
                    Console.WriteLine($"  'the cat sat' -> [{string.Join(", ", unigram.Encode("the cat sat"))}]");
                    Console.WriteLine($"  round trip is exact: '{unigram.Decode(unigram.Encode("the cat sat"))}'");

                    // Because every piece carries a probability, alternatives can be SAMPLED. That
                    // is subword regularisation - training on several segmentations of the same
                    // sentence makes a model robust to the tokenizer's arbitrary choices.
                    var rng = new GraviRandom(7);
                    var seen = new HashSet<string>(StringComparer.Ordinal);
                    for (var i = 0; i < 50; i++)
                        seen.Add(string.Join(" ", unigram.SampleEncoding("the cat sat", rng, alpha: 0.2)));

                    Console.WriteLine($"  {seen.Count} distinct segmentations sampled from one sentence");

                    // Persist. BPE saves merges; unigram saves pieces with their log probabilities.
                    bpe.Save("merges.txt");
                    unigram.Save("unigram.model");
                    Console.WriteLine("\nsaved merges.txt and unigram.model");
                    """,
                ["README.md"] = """
                    # $name$

                    Two trainable sub-word tokenizers, by genuinely different routes.

                    - **BPE** merges the most frequent adjacent pair repeatedly. Simple, fast to
                      train, and what GPT uses. The merge *order* is the model — a vocabulary
                      alone cannot tokenize.
                    - **Unigram (SentencePiece)** prunes a large candidate vocabulary by EM and
                      segments by Viterbi, so the result is globally optimal. It can also *sample*
                      alternative segmentations, which BPE cannot.

                    Unigram encodes whitespace rather than splitting on it, so decoding is exactly
                    reversible and it works on languages that do not space their words.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // ------------------------------------------------------------ named entities
        new ProjectTemplate(
            "ner",
            "Named entity recognition",
            "Natural language",
            "Train a CRF-backed entity tagger on annotated text in CoNLL format.",
            ["GraviNum", "GraviText"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame", "GraviLearn", "GraviText"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviText.Tasks;

                    // CoNLL column format: one token and its BIO tag per line, blank lines between
                    // sentences. Point this at your own annotated data.
                    var path = args.Length > 0 ? args[0] : "../../datasets/ner_conll.txt";

                    if (!File.Exists(path))
                    {
                        Console.WriteLine($"No corpus at '{path}'. Pass one as the first argument.");
                        return;
                    }

                    var sentences = TaggedSentence.LoadConll(path);
                    var cut = (int)(sentences.Count * 0.75);
                    var train = sentences.Take(cut).ToList();
                    var test = sentences.Skip(cut).ToList();

                    Console.WriteLine($"{sentences.Count} sentences, {train.Count} for training\n");

                    var ner = new TrainedNer().Fit(train);
                    Console.WriteLine($"features: {ner.FeatureCount}, tags: {string.Join(", ", ner.Labels)}\n");

                    // Score whole ENTITIES, not tokens. Token accuracy is dominated by the O tag -
                    // a model predicting O everywhere scores above 85% on most corpora.
                    Console.WriteLine($"held out : {ner.Evaluate(test)}");
                    Console.WriteLine($"tokens   : {ner.TokenAccuracy(test):P2}  <- not the number to judge by\n");

                    foreach (var sentence in new[]
                    {
                        "Kartika Wijaya bekerja di Gravicode .",
                        "Tim dari Bandung mengunjungi Tokopedia .",
                    })
                    {
                        Console.WriteLine($"\"{sentence}\"");
                        foreach (var entity in ner.Recognize(sentence))
                            Console.WriteLine($"    {entity.Type,-4} {entity.Text}");
                    }
                    """,
                ["README.md"] = """
                    # $name$

                    A trained entity tagger: shape-based features scored per token, decoded as a
                    sequence by a linear-chain CRF.

                    Pass a CoNLL-format corpus as the first argument — one token and its BIO tag
                    per line, blank lines between sentences.

                    ```bash
                    dotnet run -- path/to/corpus.txt
                    ```

                    Two things worth knowing:

                    - **Features are shape-based, not identity-based.** Mapping capitals to `X` and
                      lower-case to `x` turns "Jakarta" and "Bandung" into the same pattern, so
                      evidence about one transfers to the other. That is what separates a trained
                      tagger from a gazetteer.
                    - **The CRF enforces the BIO scheme structurally.** An `I-PER` cannot follow an
                      `O`, because the transition is forbidden rather than merely penalised — a
                      learned penalty can always be outvoted by a confident emission.

                    Judge it on entity F1, never token accuracy.
                    """,
            }),

        // ------------------------------------------------------------ forecasting
        new ProjectTemplate(
            "forecasting",
            "State-space forecasting",
            "Statistics",
            "Kalman filtering, smoothing and forecasting with honest uncertainty.",
            ["GraviNum", "GraviProb"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviProb"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviNum;
                    using Gravicode.Science.GraviProb;

                    // A local level model is an exponentially weighted moving average whose
                    // smoothing constant is DERIVED from the noise ratio rather than guessed -
                    // and it reports its own uncertainty, which an EWMA does not.
                    var filter = KalmanFilter.LocalLevel(processVariance: 0.05, observationVariance: 1.0);

                    // Simulate so there is a truth to check against. Replace this with your series.
                    var (truth, observations) = filter.Simulate(200, new GraviRandom(42));

                    var filtered = filter.Filter(observations);
                    var smoothed = filter.Smooth(observations);

                    double Rmse(IReadOnlyList<double> estimate)
                    {
                        var total = 0.0;
                        for (var t = 0; t < estimate.Count; t++)
                        {
                            var error = estimate[t] - truth[t, 0];
                            total += error * error;
                        }
                        return Math.Sqrt(total / estimate.Count);
                    }

                    var raw = 0.0;
                    for (var t = 0; t < 200; t++)
                    {
                        var error = observations[t, 0] - truth[t, 0];
                        raw += error * error;
                    }

                    Console.WriteLine($"raw observations : {Math.Sqrt(raw / 200):F4}");
                    Console.WriteLine($"filtered         : {Rmse(filtered.Filtered.Select(s => s.Mean.At(0)).ToList()):F4}");
                    Console.WriteLine($"smoothed         : {Rmse(smoothed.Select(s => s.Mean.At(0)).ToList()):F4}");
                    Console.WriteLine("\nFiltering uses only the past, which is what a live system can do.");
                    Console.WriteLine("Smoothing uses the whole series - using smoothed states to test a");
                    Console.WriteLine("forecasting rule is a look-ahead error, and a common one.\n");

                    // Only the RATIO of Q to R matters, which is why the log likelihood can be used
                    // to fit it: the series decomposes into independent one-step prediction errors.
                    Console.WriteLine("choosing the process variance by likelihood:");
                    foreach (var q in new[] { 0.001, 0.01, 0.05, 0.2, 1.0 })
                    {
                        var candidate = KalmanFilter.LocalLevel(q, 1.0).Filter(observations);
                        Console.WriteLine($"  Q = {q,-6} log likelihood {candidate.LogLikelihood,10:F2}");
                    }

                    var forecast = filter.Forecast(observations, horizon: 20);
                    Console.WriteLine($"\n20-step forecast, with uncertainty that grows as it must:");
                    foreach (var step in new[] { 0, 4, 9, 19 })
                        Console.WriteLine($"  t+{step + 1,-3} {forecast[step].Mean.At(0),8:F4} " +
                                          $"+/- {forecast[step].StandardDeviation.At(0):F4}");

                    // A local linear trend adds a slope, which is never observed - it is inferred
                    // entirely from how the level moves, so it extrapolates rather than flattening.
                    Console.WriteLine("\nKalmanFilter.LocalLinearTrend adds an unobserved slope,");
                    Console.WriteLine("which is what lets a forecast continue a trend rather than flatten.");
                    """,
                ["README.md"] = """
                    # $name$

                    Linear-Gaussian state-space modelling. Within its assumptions the Kalman filter
                    is not a good method, it is *the* method: the exact posterior over the hidden
                    state, and the minimum-variance estimator among all estimators.

                    ```bash
                    dotnet run
                    ```

                    - **`Filter`** estimates each state from the past only — what a real-time
                      system can do.
                    - **`Smooth`** uses the whole series. Strictly better, and only available
                      after the fact.
                    - **`Forecast`** runs the predict step alone, so uncertainty grows with the
                      horizon. A forecast whose uncertainty does not grow is not a forecast.

                    Only the *ratio* of process to observation variance matters, so the model can
                    be tuned with one number — and `LogLikelihood` gives you a principled way to
                    choose it rather than guessing.
                    """,
            }),

        // ------------------------------------------------------------ notebook
        new ProjectTemplate(
            "notebook",
            "Interactive notebook",
            "Starting points",
            "A .NET Interactive notebook with charts, ready for Polyglot Notebooks in VS Code.",
            ["GraviNum", "GraviFrame", "GraviLearn"],
            new Dictionary<string, string>
            {
                ["$name$.ipynb"] = """
                    {
                      "cells": [
                        {
                          "cell_type": "markdown",
                          "metadata": {},
                          "source": ["# $name$\n", "\n", "Created with ScienceAppGen.\n", "\n",
                                     "Build Gravicode.Science in Release first so the assemblies exist."]
                        },
                        {
                          "cell_type": "code",
                          "execution_count": null,
                          "metadata": {},
                          "outputs": [],
                          "source": [
                            "#r \"../../src/GraviNum/bin/Release/net10.0/Gravicode.Science.GraviNum.dll\"\n",
                            "#r \"../../src/GraviFrame/bin/Release/net10.0/Gravicode.Science.GraviFrame.dll\"\n",
                            "#r \"../../src/GraviLearn/bin/Release/net10.0/Gravicode.Science.GraviLearn.dll\"\n",
                            "#r \"nuget: ScottPlot, 5.1.59\"\n",
                            "\n",
                            "using Gravicode.Science.GraviNum;\n",
                            "using Gravicode.Science.GraviFrame;\n",
                            "\n",
                            "Console.WriteLine(GraviInfo.HardwareReport());"
                          ]
                        },
                        {
                          "cell_type": "markdown",
                          "metadata": {},
                          "source": ["## Generate and describe some data"]
                        },
                        {
                          "cell_type": "code",
                          "execution_count": null,
                          "metadata": {},
                          "outputs": [],
                          "source": [
                            "var rng = new GraviRandom(42);\n",
                            "var samples = rng.Normal(100, 15, 10_000);\n",
                            "\n",
                            "foreach (var (key, value) in Statistics.Describe(samples))\n",
                            "    Console.WriteLine($\"{key,-8}{value,12:F4}\");"
                          ]
                        },
                        {
                          "cell_type": "markdown",
                          "metadata": {},
                          "source": ["## Plot it"]
                        },
                        {
                          "cell_type": "code",
                          "execution_count": null,
                          "metadata": {},
                          "outputs": [],
                          "source": [
                            "var (edges, counts) = Statistics.Histogram(samples, bins: 40);\n",
                            "var centres = Enumerable.Range(0, counts.Length)\n",
                            "    .Select(i => (edges[i] + edges[i + 1]) / 2).ToArray();\n",
                            "\n",
                            "var plot = new ScottPlot.Plot();\n",
                            "plot.Add.Bars(centres, counts.Select(c => (double)c).ToArray());\n",
                            "plot.Title(\"Sample distribution\");\n",
                            "plot.GetPngHtml(800, 450)"
                          ]
                        }
                      ],
                      "metadata": {
                        "kernelspec": { "display_name": ".NET (C#)", "language": "C#", "name": ".net-csharp" },
                        "language_info": { "name": "polyglot-notebook" }
                      },
                      "nbformat": 4,
                      "nbformat_minor": 4
                    }
                    """,
                ["README.md"] = """
                    # $name$

                    A .NET Interactive notebook created with ScienceAppGen.

                    Open `$name$.ipynb` in VS Code with the
                    [Polyglot Notebooks](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.dotnet-interactive-vscode)
                    extension.

                    Build the libraries first so the `#r` references resolve:

                    ```bash
                    dotnet build Gravicode.Science.sln -c Release
                    ```
                    """,
            }),

        // ------------------------------------------------------------ arrow interchange
        new ProjectTemplate(
            "arrow",
            "Arrow interchange",
            "Data science",
            "Exchange dataframes with pandas and pyarrow, and aggregate files larger than memory.",
            ["GraviNum", "GraviFrame"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviFrame;
                    using Gravicode.Science.GraviFrame.Io;
                    using Gravicode.Science.GraviNum;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    var frame = new DataFrame(
                    [
                        new TextSeries("symbol", ["BBCA", "TLKM", "ASII", null]),
                        new NumericSeries("close", [9250.0, 3120.0, double.NaN, 4410.0]),
                        new BooleanSeries("halted", [false, false, true, null]),
                        new DateTimeSeries("stamp",
                            [new DateTime(2024, 1, 2), new DateTime(2024, 1, 3), new DateTime(2024, 1, 4), null]),
                    ]);

                    ArrowFile.Write(frame, "quotes.arrow");
                    var back = ArrowFile.Read("quotes.arrow");

                    Console.WriteLine(back);
                    Console.WriteLine(string.Join(", ", back.ColumnNames.Select(n => $"{n}={back[n].DataType}")));
                    Console.WriteLine($"missing values survived: {back["close"].IsMissing(2)}");

                    // Read it from Python without a CSV parse and without re-inferring types:
                    //   import pyarrow as pa
                    //   pa.ipc.open_file("quotes.arrow").read_all().to_pandas()

                    // ------------------------------------------------------------------
                    // Out of core: the same queries over a file too big to load.

                    // 20,000 quotes written out, then read back 500 at a time. Stand in your
                    // own file here - nothing below cares how large it is.
                    var rng = new GraviRandom(42);
                    string[] tickers = ["BBCA", "TLKM", "ASII", "BBRI"];

                    var symbols = new string[20_000];
                    var prices = new double[20_000];
                    var volumes = new double[20_000];
                    for (var i = 0; i < 20_000; i++)
                    {
                        symbols[i] = tickers[i % tickers.Length];
                        prices[i] = 1000 + (i % tickers.Length) * 2000 + rng.Normal() * 150;
                        volumes[i] = Math.Abs(rng.Normal() * 1e6);
                    }

                    CsvWriter.Write(new DataFrame(
                    [
                        new TextSeries("symbol", symbols),
                        new NumericSeries("close", prices),
                        new NumericSeries("volume", volumes),
                    ]), "quotes.csv");

                    var chunked = ChunkedFrame.FromCsv("quotes.csv", chunkRows: 500);
                    Console.WriteLine($"{Streaming.CountRows(chunked)} rows, 500 at a time");

                    // One pass and constant memory. Variance comes from Welford's method rather
                    // than E[x^2] - E[x]^2, which subtracts two nearly equal large numbers and
                    // can return a negative variance.
                    foreach (var (name, stats) in Streaming.Describe(chunked, ["close", "volume"]))
                        Console.WriteLine($"  {name,-7} n={stats.Count} mean={stats.Mean,10:F2} sd={stats.StandardDeviation,10:F2}");

                    // GroupBy memory is proportional to DISTINCT GROUPS, not rows - so on an
                    // unfamiliar key, count them first. That number is the difference between a
                    // query that runs and one that runs out of memory.
                    Console.WriteLine($"{Streaming.CountGroups(chunked, ["symbol"])} distinct groups");
                    Console.WriteLine(Streaming.GroupBy(chunked, ["symbol"], ("close", "mean"), ("volume", "sum")));

                    // Chunk size is a memory knob, not a parameter of the answer - column types
                    // are pinned from one sample rather than inferred per chunk.
                    foreach (var size in new[] { 64, 500, 8_000 })
                        Console.WriteLine($"  chunkRows={size,6} -> mean close " +
                            $"{Streaming.Describe(ChunkedFrame.FromCsv("quotes.csv", chunkRows: size), ["close"])["close"].Mean:F9}");

                    // Two passes and disk space equal to the input. That is the trade, and it is
                    // what lets a sort exceed memory - do not reach for it when the data fits.
                    Streaming.SortToFile(chunked, "close", "by-close.csv", descending: true);
                    Console.WriteLine(DataFrame.ReadCsv("by-close.csv").Head(5));
                    """,
                ["README.md"] = """
                    # $name$

                    Arrow interchange and out-of-core aggregation with GraviFrame.

                    `ArrowFile` carries types and missing values across a language boundary, so
                    pandas reads the output directly. It is verified in both directions against
                    pyarrow rather than by round-tripping through itself, which for an interchange
                    format would prove nothing.

                    `ChunkedFrame` reads a block at a time and `Streaming` aggregates over the
                    blocks. Chunk size is a memory knob, not a parameter of the answer.

                    ```bash
                    dotnet run
                    ```

                    Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil
                    """,
            }),

        // ------------------------------------------------------------ sparse text
        new ProjectTemplate(
            "sparse-text",
            "Sparse text classification",
            "Machine learning",
            "Train a linear classifier on a bag-of-words matrix without densifying it.",
            ["GraviNum", "GraviFrame", "GraviLearn", "GraviText"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame", "GraviLearn", "GraviText"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviLearn.Linear;
                    using Gravicode.Science.GraviNum;
                    using Gravicode.Science.GraviText.Vectorization;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    string[] documents =
                    [
                        "layanan cepat dan ramah, sangat memuaskan",
                        "produk bagus sekali, akan beli lagi",
                        "pengiriman tepat waktu dan barang rapi",
                        "kualitas mantap, harga sepadan",
                        "barang rusak saat sampai, kecewa berat",
                        "pengiriman lambat dan tidak ada kabar",
                        "kualitas buruk, tidak sesuai deskripsi",
                        "pelayanan mengecewakan, tidak akan kembali",
                    ];

                    var labels = NdArray.FromValues([1, 1, 1, 1, 0, 0, 0, 0]);

                    // A bag-of-words matrix is around 1% non-zero. The dense copy is almost
                    // entirely zeros that cost the same to store and multiply as any other
                    // number - the memory wall is the reason for this path, not the speed.
                    var vectoriser = new TfidfVectorizer(new VectorizerOptions { MaxFeatures = 5_000 });
                    var x = vectoriser.FitTransformSparse(documents);

                    var density = 100.0 * x.NonZeroCount / ((double)x.Rows * x.Columns);
                    Console.WriteLine($"{x.Rows} documents x {x.Columns} terms, {density:F2}% non-zero");

                    // L2 is not optional on separable data: without it the maximum likelihood
                    // sits at infinity, the weights grow forever and the fit never converges.
                    var model = new SparseLogisticRegression(
                        learningRate: 1.0, maxIterations: 500, l2Penalty: 0.01).Fit(x, labels);

                    Console.WriteLine($"accuracy {model.Score(x, labels):P2} after {model.IterationsRun} iterations");
                    Console.WriteLine();

                    // The practical reason to keep a linear model on text: a coefficient reads
                    // directly as "this word moves the decision this far", which no tree
                    // ensemble or transformer offers.
                    Console.WriteLine("Terms pushing towards the positive class:");
                    foreach (var (feature, weight) in model.TopFeatures(8))
                        Console.WriteLine($"  {vectoriser.Vocabulary[feature],-16}{weight,8:F4}");

                    Console.WriteLine();
                    Console.WriteLine("Scoring new reviews:");
                    string[] fresh = ["barang bagus dan pengiriman cepat", "produk rusak dan pelayanan buruk"];
                    // One binary problem means ONE column of probabilities, not two: the
                    // second is 1 - the first and is not stored.
                    var scores = model.PredictProbabilities(vectoriser.TransformSparse(fresh));
                    for (var i = 0; i < fresh.Length; i++)
                        Console.WriteLine($"  {scores[i, 0]:P1} positive  '{fresh[i]}'");

                    Console.WriteLine();
                    Console.WriteLine(GraviInfo.Attribution);
                    """,
                ["README.md"] = """
                    # $name$

                    Sparse text classification with GraviText and GraviLearn.

                    `TfidfVectorizer.FitTransformSparse` returns a CSR matrix and
                    `SparseLogisticRegression` trains on it directly. It is the same model as the
                    dense path rather than an approximation, so the comparison worth making is
                    between coefficients — accuracy would agree even if the weights had drifted.

                    The weight vector stays dense, so this bounds the *feature* count, not the
                    number of documents.

                    ```bash
                    dotnet run
                    ```

                    Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil
                    """,
            }),

        // ------------------------------------------------------------ distributed
        new ProjectTemplate(
            "distributed",
            "Distributed training",
            "Machine learning",
            "Split training across workers, and average gradients the way that stays correct.",
            ["GraviNum", "GraviFrame", "GraviLearn"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame", "GraviLearn"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviLearn;
                    using Gravicode.Science.GraviLearn.Distributed;
                    using Gravicode.Science.GraviLearn.ModelSelection;
                    using Gravicode.Science.GraviNum;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    // MakeBlobs generates its data, so this runs anywhere. Datasets.LoadIris()
                    // needs the repository's datasets/ folder and will not resolve outside it.
                    var data = Datasets.MakeBlobs(samples: 2000, features: 8, centers: 3, seed: 42);
                    var split = Selection.Split(data.Features, data.Target, testSize: 0.3, seed: 42);

                    // Partition spreads the remainder rather than dumping it on the last worker,
                    // so shards differ by one whenever the count does not divide.
                    foreach (var shard in DataParallel.Partition(items: split.TrainX.Shape[0], workers: 4))
                        Console.WriteLine($"  {shard,-14} n={shard.Count}");
                    Console.WriteLine();

                    // Bit-identical to single-process training, not merely equivalent: tree t is
                    // seeded from seed + t * 7919, a function of its global index alone, so a
                    // shard boundary cannot change the answer.
                    var distributed = new DistributedForest(nTrees: 200, maxDepth: 12, seed: 42)
                        .Fit(split.TrainX, split.TrainY, workers: 4);
                    var single = new DistributedForest(nTrees: 200, maxDepth: 12, seed: 42)
                        .Fit(split.TrainX, split.TrainY, workers: 1);

                    var many = distributed.Predict(split.TestX);
                    var one = single.Predict(split.TestX);
                    var identical = true;
                    for (var i = 0; i < many.Size; i++) identical &= many.At(i) == one.At(i);

                    Console.WriteLine($"4 workers accuracy : {distributed.Score(split.TestX, split.TestY):P2}");
                    Console.WriteLine($"1 worker  accuracy : {single.Score(split.TestX, split.TestY):P2}");
                    Console.WriteLine($"bit-identical      : {identical}");
                    Console.WriteLine();

                    // The pieces, for a hand-written loop. Gradients are WEIGHTED by sample
                    // count: a plain average of per-worker means equals the global mean only for
                    // equal shards, and unweighted the model trains to something slightly wrong
                    // that no shape or convergence check would catch.
                    using var transport = new InProcessTransport(workerCount: 4);
                    var server = new ParameterServer(transport);

                    for (var worker = 0; worker < 4; worker++)
                        server.Contribute(worker, NdArray.FromValues([worker + 1.0, worker + 2.0]),
                            sampleCount: 10 * (worker + 1));

                    var averaged = server.Aggregate();
                    Console.WriteLine($"aggregate: [{averaged.At(0):F6}, {averaged.At(1):F6}]");

                    // FileTransport is the same interface over a shared directory: no broker, no
                    // ports, and it crosses machines. Workers write then rename, so a collector
                    // cannot read a half-written payload.
                    //   using var files = new FileTransport("/shared/exchange", workerCount: 8);

                    Console.WriteLine();
                    Console.WriteLine(GraviInfo.Attribution);
                    """,
                ["README.md"] = """
                    # $name$

                    Data-parallel training with GraviLearn.

                    `DistributedForest` splits the *computation*, not the memory: every worker
                    fits on the whole training set, which is the right direction for a forest
                    where the trees are the expensive part.

                    Two details are correctness requirements rather than refinements. Gradients
                    are weighted by sample count, because `Partition` produces uneven shards.
                    And transports collect in worker order rather than arrival order, because
                    floating-point addition is not associative and arrival order would make the
                    answer depend on the scheduler.

                    ```bash
                    dotnet run
                    ```

                    Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil
                    """,
            }),

        // ------------------------------------------------------------ pretrained weights
        new ProjectTemplate(
            "pretrained",
            "Pretrained transformer",
            "Natural language",
            "Load an exported BERT checkpoint into a TransformerModel and encode text with it.",
            ["GraviNum", "GraviFrame", "GraviLearn", "GraviText"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviNum", "GraviFrame", "GraviLearn", "GraviText"),
                ["Program.cs"] = """
                    using Gravicode.Science.GraviNum;
                    using Gravicode.Science.GraviText.Transformers;

                    Console.WriteLine(GraviInfo.Banner("$name$"));

                    // No weights ship with Gravicode.Science - licensing and size keep a real
                    // checkpoint out of the repository - so export your own first:
                    //
                    //   from transformers import AutoModel
                    //   import torch
                    //   torch.onnx.export(AutoModel.from_pretrained("bert-base-uncased"),
                    //                     torch.zeros(1, 8, dtype=torch.long),
                    //                     "bert-base-uncased.onnx",
                    //                     input_names=["input_ids"], opset_version=13)

                    const string checkpoint = "bert-base-uncased.onnx";

                    if (!File.Exists(checkpoint))
                    {
                        Console.WriteLine($"Export {checkpoint} first - see the comment above.");
                        return;
                    }

                    // Names are a convention and the file is the only authority on which one it
                    // follows, so look before loading. A tensor of shape [768, 768] could be a
                    // query, key or output projection, and nothing but the name says which.
                    foreach (var (name, shape) in TransformerCheckpoint.Inspect(checkpoint).Take(6))
                        Console.WriteLine($"  {name,-58} [{string.Join(", ", shape)}]");
                    Console.WriteLine();

                    var model = new TransformerModel("bert-base", vocabularySize: 30522);
                    var report = TransformerCheckpoint.Load(model, checkpoint);

                    Console.WriteLine(report);
                    Console.WriteLine($"HasPretrainedWeights = {model.HasPretrainedWeights}");

                    // A partial load never sets that flag, whichever mode was used: a model with
                    // three of twelve layers loaded produces output that is neither the
                    // checkpoint's nor a random model's, and nothing downstream could tell.

                    // A different naming convention is usually the whole adaptation:
                    //   TransformerCheckpoint.Load(model, path, CheckpointNames.Reprefixed("roberta."));
                    //   TransformerCheckpoint.Load(model, path, CheckpointNames.Unprefixed);

                    var hidden = model.Forward([101, 7592, 2088, 102]);
                    Console.WriteLine($"{hidden.Shape[0]} x {hidden.Shape[1]} hidden states");

                    Console.WriteLine();
                    Console.WriteLine(GraviInfo.Attribution);
                    """,
                ["README.md"] = """
                    # $name$

                    Loading pretrained transformer weights with GraviText.

                    `TransformerCheckpoint.Load` fills every parameter — embeddings, both layer
                    norms, all four attention projections and both feed-forward layers. The
                    older `LoadOnnxWeights` filled only the embedding tables, which is enough to
                    look up a word vector and not enough to run the model.

                    Three things it refuses rather than absorbs: a wrong transpose (checked
                    against the non-square feed-forward weight, because a 768×768 projection
                    accepts either reading), a partial checkpoint, and a mismatched architecture.

                    ```bash
                    dotnet run
                    ```

                    Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil
                    """,
            }),
    ];
}
