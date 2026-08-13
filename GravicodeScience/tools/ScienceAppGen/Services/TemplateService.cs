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
                            "plot.GetImageHtml(800, 450)"
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
    ];
}
