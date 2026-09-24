using System.Text;

namespace HFAppGen.Services;

/// <summary>A starting point offered by the New Project dialog.</summary>
/// <param name="Id">Stable identifier.</param>
/// <param name="Name">Display name.</param>
/// <param name="Category">Grouping in the dialog.</param>
/// <param name="Description">One line explaining what it produces.</param>
/// <param name="Libraries">Which HF.Net libraries it uses, for the spectrum indicator.</param>
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
    /// than as a count of anonymous ticks. They are the Lib1..Lib8 hues from <c>Themes/Tokens.axaml</c>, in pipeline order.
    /// </summary>
    public IReadOnlyList<string> LibraryColours => Libraries.Select(l => l switch
    {
        "GraviHub" => "#FFB454",
        "GraviTokenizers" => "#F2915C",
        "GraviDatasets" => "#E2726A",
        "GraviTransformers" => "#C86089",
        "GraviPEFT" => "#9E63A8",
        "GraviAccelerate" => "#7370BE",
        "GraviOptimum" => "#4E86C4",
        "GraviDiffusers" => "#57C7E3",
        _ => "#5D6875",
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
    /// and wiring up the HF.Net references.
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
                content.Replace("$name$", projectName).Replace("$hfnet$", source),
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        }
    }

    /// <summary>
    /// The path a generated project should use to reach the HF.Net libraries, relative
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
                var candidate = Path.Combine(directory.FullName, "src", "GraviHub", "GraviHub.csproj");
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
    /// <c>$hfnet$</c> is replaced at creation time with the real path to the libraries; see
    /// <see cref="ResolveLibraryPath"/>. Writing a fixed relative path here would break every
    /// project created outside the repository.
    /// </remarks>
    private static string Csproj(params string[] libraries)
    {
        var references = string.Join("\n", libraries.Select(l =>
            $"    <ProjectReference Include=\"$hfnet$\\{l}\\{l}.csproj\" />"));

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
                     PackageReference entries once the HF.Net packages are published. -->
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

                    Created with HFAppGen.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // ------------------------------------------------------------ sentiment
        new ProjectTemplate(
            "sentiment",
            "Sentiment analysis",
            "Text",
            "Classify text with a fine-tuned Hugging Face model, labels and all.",
            ["GraviTransformers"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviTransformers"),
                ["Program.cs"] = """
                    using Gravicode.HFNet.GraviTransformers;

                    // A fine-tuned checkpoint carries its own classification head and its label
                    // names, so nothing here has to be told what the classes are.
                    Console.WriteLine("Loading distilbert-base-uncased-finetuned-sst-2-english ...");

                    using var model = TransformerModel.Load("distilbert-base-uncased-finetuned-sst-2-english");

                    Console.WriteLine($"{model}");
                    Console.WriteLine($"labels: {string.Join(", ", model.Labels)}");
                    Console.WriteLine();

                    string[] samples =
                    [
                        "I absolutely loved this film.",
                        "A complete waste of time.",
                        "It was fine, nothing special.",
                    ];

                    foreach (var text in samples)
                    {
                        var best = model.Predict(text, topK: 1)[0];
                        Console.WriteLine($"{best.Label,-9} {best.Score,7:P2}  {text}");
                    }

                    Console.WriteLine();
                    Console.Write("Your text (blank to finish): ");

                    string? line;
                    while (!string.IsNullOrWhiteSpace(line = Console.ReadLine()))
                    {
                        foreach (var prediction in model.Predict(line))
                            Console.WriteLine($"  {prediction}");

                        Console.Write("Your text (blank to finish): ");
                    }
                    """,
                ["README.md"] = """
                    # $name$

                    Sentiment classification with HF.Net.

                    The first run downloads the model (about 260 MB) into the shared Hugging Face
                    cache; later runs revalidate it with a single request.

                    ```bash
                    dotnet run
                    ```

                    Set `HF_TOKEN` for private or gated models, and for a higher rate limit.
                    """,
            }),

        // ------------------------------------------------------------ semantic search
        new ProjectTemplate(
            "semantic-search",
            "Semantic search",
            "Text",
            "Embed a corpus once, then rank it against a query by cosine similarity.",
            ["GraviTransformers", "GraviDatasets"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviTransformers", "GraviDatasets"),
                ["Program.cs"] = """
                    using Gravicode.HFNet.GraviTransformers;
                    using Gravicode.Science.GraviNum;

                    using var model = TransformerModel.Load("bert-base-uncased");

                    string[] documents =
                    [
                        "The cat sat quietly on the mat.",
                        "Quarterly revenue exceeded analyst expectations.",
                        "A golden retriever played fetch in the park.",
                        "The board approved the merger this morning.",
                        "Kittens are small domestic cats.",
                    ];

                    // Embedded once. Every query after this is a handful of dot products, which is
                    // the whole point of doing it this way round.
                    Console.WriteLine($"Embedding {documents.Length} documents ...");
                    var corpus = model.EmbedBatch(documents);

                    while (true)
                    {
                        Console.Write("\nQuery (blank to finish): ");
                        var query = Console.ReadLine();
                        if (string.IsNullOrWhiteSpace(query)) break;

                        var vector = model.Embed(query);

                        var ranked = Enumerable.Range(0, documents.Length)
                            .Select(i => (Document: documents[i], Score: Cosine(corpus.Row(i), vector)))
                            .OrderByDescending(r => r.Score);

                        foreach (var (document, score) in ranked)
                            Console.WriteLine($"  {score:F4}  {document}");
                    }

                    static double Cosine(NdArray a, NdArray b)
                    {
                        double dot = 0, normA = 0, normB = 0;
                        for (var i = 0; i < a.Size; i++)
                        {
                            dot += a.At(i) * b.At(i);
                            normA += a.At(i) * a.At(i);
                            normB += b.At(i) * b.At(i);
                        }
                        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
                    }
                    """,
                ["README.md"] = """
                    # $name$

                    Semantic search over a small corpus.

                    Embeddings are mean-pooled over the tokens rather than taken from `[CLS]`: on a
                    model that has not been fine-tuned for sentence similarity, `[CLS]` is close to
                    constant and makes every pair of sentences look alike.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // ------------------------------------------------------------ fill mask
        new ProjectTemplate(
            "fill-mask",
            "Masked language model",
            "Text",
            "Ask a pretrained encoder to fill in a blank, and watch what it knows.",
            ["GraviTransformers"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviTransformers"),
                ["Program.cs"] = """
                    using Gravicode.HFNet.GraviTransformers;

                    using var model = TransformerModel.Load("bert-base-uncased");
                    Console.WriteLine($"{model}\n");

                    string[] prompts =
                    [
                        "The capital of France is [MASK].",
                        "He was a [MASK] player in the national team.",
                        "Water boils at one hundred [MASK].",
                    ];

                    foreach (var prompt in prompts)
                    {
                        Console.WriteLine(prompt);
                        foreach (var fill in model.FillMask(prompt, topK: 5))
                            Console.WriteLine($"    {fill.Token,-16} {fill.Score,7:P2}");

                        Console.WriteLine();
                    }

                    Console.WriteLine("Type a sentence containing [MASK] (blank to finish).");

                    string? line;
                    while (!string.IsNullOrWhiteSpace(line = Console.ReadLine()))
                    {
                        try
                        {
                            foreach (var fill in model.FillMask(line, topK: 5))
                                Console.WriteLine($"    {fill.Token,-16} {fill.Score,7:P2}");
                        }
                        catch (ArgumentException error)
                        {
                            Console.WriteLine($"    {error.Message}");
                        }
                    }
                    """,
                ["README.md"] = """
                    # $name$

                    Masked language modelling with a pretrained BERT.

                    This is also the sharpest check that a checkpoint loaded correctly: a model with
                    a transposed weight or a shifted position embedding still produces
                    plausible-looking vectors, but it does not answer "The capital of France is
                    [MASK]." with *paris*.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // ------------------------------------------------------------ tokenizer lab
        new ProjectTemplate(
            "tokenizer-lab",
            "Tokenizer laboratory",
            "Text",
            "Compare WordPiece, byte-level BPE and Unigram on the same text.",
            ["GraviTokenizers"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviTokenizers"),
                ["Program.cs"] = """
                    using Gravicode.HFNet.GraviTokenizers;

                    const string Text = "Tokenizers are unbelievably useful, aren't they? 🎉";

                    Console.WriteLine($"input: {Text}\n");

                    foreach (var (name, id) in new[]
                    {
                        ("WordPiece (BERT)", "bert-base-uncased"),
                        ("Byte-level BPE (GPT-2)", "gpt2"),
                    })
                    {
                        var tokenizer = HfTokenizer.FromPretrained(id);
                        var encoding = tokenizer.Encode(Text);

                        Console.WriteLine($"{name}  vocab={tokenizer.VocabularySize:N0}");
                        Console.WriteLine($"  {encoding.Length} tokens: {string.Join(' ', encoding.Tokens)}");
                        Console.WriteLine($"  ids    : {string.Join(' ', encoding.Ids)}");
                        Console.WriteLine($"  decoded: {tokenizer.Decode(encoding.Ids)}");
                        Console.WriteLine();
                    }

                    // Offsets point back into the ORIGINAL string, so a span can be reported as a
                    // substring rather than as token indices nobody can interpret.
                    var bert = HfTokenizer.FromPretrained("bert-base-uncased");
                    var detailed = bert.Encode(Text);

                    Console.WriteLine("token -> source span");
                    for (var i = 0; i < detailed.Length; i++)
                        Console.WriteLine($"  {detailed.Tokens[i],-14} {detailed.Offsets[i],-10} '{detailed.Span(Text, i)}'");

                    // Train one from scratch on your own corpus.
                    string[] corpus =
                    [
                        "mesin belajar dari data", "data melatih mesin",
                        "belajar mesin itu menarik", "model belajar dari contoh",
                    ];

                    var trained = Tokenizer.TrainWordPiece(corpus, vocabularySize: 120);
                    Console.WriteLine($"\ntrained vocab={trained.VocabularySize}");
                    Console.WriteLine($"  {string.Join(' ', trained.Encode("mesin belajar").Tokens)}");
                    """,
                ["README.md"] = """
                    # $name$

                    A side-by-side look at three tokenization schemes.

                    Use a model's *own* tokenizer whenever you have the model: a mismatched
                    tokenizer produces ids the model was never trained on, and nothing downstream
                    will tell you.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // ------------------------------------------------------------ hub explorer
        new ProjectTemplate(
            "hub-explorer",
            "Hub explorer",
            "Hugging Face Hub",
            "Search the Hub, inspect a repository, and read a checkpoint's tensors.",
            ["GraviHub"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviHub"),
                ["Program.cs"] = """
                    using Gravicode.HFNet.GraviHub;
                    using Gravicode.HFNet.GraviHub.Io;

                    Console.WriteLine(Hub.WhoAmI() is { } who
                        ? $"Signed in as {who}."
                        : "Anonymous - set HF_TOKEN for private repositories and a higher rate limit.");

                    Console.WriteLine($"Cache: {Hub.Cache.Root}\n");

                    Console.Write("Search the Hub for: ");
                    var query = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(query)) query = "sentiment";

                    var results = Hub.SearchModels(query, limit: 8);
                    for (var i = 0; i < results.Count; i++)
                        Console.WriteLine($"  [{i}] {results[i].Id,-52} {results[i].Downloads,12:N0}");

                    Console.Write("\nInspect which number? ");
                    if (!int.TryParse(Console.ReadLine(), out var choice) || choice >= results.Count) return;

                    var info = Hub.ModelInfo(results[choice].Id);

                    Console.WriteLine($"\n{info}");
                    Console.WriteLine($"  pipeline : {info.PipelineTag ?? "(none)"}");
                    Console.WriteLine($"  tags     : {string.Join(", ", info.Tags.Take(8))}");
                    Console.WriteLine($"  weights  : {(info.HasSafeTensors ? "safetensors" : "pickle only")}");

                    Console.WriteLine("\n  files:");
                    foreach (var file in info.Files.Take(15)) Console.WriteLine($"    {file}");

                    var weights = info.FilesWithExtension(".safetensors").FirstOrDefault();
                    if (weights.Path is null) return;

                    Console.WriteLine($"\nDownloading {weights.Path} ...");
                    var path = Hub.DownloadFile(info.Id, weights.Path);

                    // Memory-mapped: this reads the header, not the gigabytes behind it.
                    using var reader = SafeTensors.Open(path);
                    Console.WriteLine(reader);

                    foreach (var tensor in reader.Tensors.Take(10)) Console.WriteLine($"  {tensor}");
                    """,
                ["README.md"] = """
                    # $name$

                    An interactive tour of the Hugging Face Hub from .NET.

                    Opening a checkpoint memory-maps it, so listing four hundred tensors costs a
                    header read rather than a multi-gigabyte load.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // ------------------------------------------------------------ dataset pipeline
        new ProjectTemplate(
            "dataset-pipeline",
            "Dataset pipeline",
            "Data science",
            "Load, inspect, split and batch a dataset ready for training.",
            ["GraviDatasets"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviDatasets"),
                ["Program.cs"] = """
                    using Gravicode.HFNet.GraviDatasets;

                    // A bare name is a built-in dataset and needs no network; anything with a slash
                    // is a Hub id.
                    var data = Dataset.Load("titanic");

                    Console.WriteLine(data);
                    Console.WriteLine();

                    Console.WriteLine("columns:");
                    foreach (var (name, type) in data.Features) Console.WriteLine($"  {name,-16} {type}");

                    Console.WriteLine("\nsummary:");
                    Console.WriteLine(data.Describe());

                    var prepared = data
                        .SelectColumns("survived", "pclass", "age", "fare")
                        .Filter(row => !double.IsNaN(row.Number("age")));

                    Console.WriteLine($"\nafter dropping rows with no age: {prepared.Count} of {data.Count}");

                    // Shuffled before splitting by default. Many published CSVs are sorted by
                    // label, and an unshuffled split of one puts every positive case on one side.
                    var split = prepared.TrainTestSplit(testSize: 0.2, seed: 42);
                    Console.WriteLine(split);

                    var batches = 0;
                    foreach (var batch in split.Train.Batches(32, dropLast: true)) batches++;

                    Console.WriteLine($"\n{batches} full batches of 32 from the training split");
                    Console.WriteLine($"features as a matrix: {split.Train.ToMatrix("pclass", "age", "fare").Shape[0]} rows");
                    """,
                ["README.md"] = """
                    # $name$

                    Loading and preparing data with GraviDatasets.

                    Every operation returns a new dataset, so a split taken earlier cannot be
                    disturbed by a shuffle taken later - which is the kind of leak that inflates a
                    validation score without producing any visible error.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // ------------------------------------------------------------ onnx inference
        new ProjectTemplate(
            "onnx-inference",
            "ONNX inference",
            "Performance",
            "Run a Hugging Face ONNX export through ONNX Runtime and measure it.",
            ["GraviOptimum", "GraviTokenizers"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviOptimum", "GraviTokenizers"),
                ["Program.cs"] = """
                    using Gravicode.HFNet.GraviOptimum;
                    using Gravicode.HFNet.GraviTokenizers;

                    const string ModelId = "hf-internal-testing/tiny-random-BertModel";

                    // "auto" accepts a fallback to the CPU. Naming a provider explicitly throws
                    // when it is not installed, which is what stops a "CUDA" deployment quietly
                    // running on the CPU for months.
                    using var model = Optimum.Optimize(ModelId, target: "auto");

                    Console.WriteLine(model);
                    Console.WriteLine($"  running on {model.Session.ActualTarget}");

                    Console.WriteLine("  inputs:");
                    foreach (var spec in model.Session.Inputs) Console.WriteLine($"    {spec}");

                    Console.WriteLine("  outputs:");
                    foreach (var spec in model.Session.Outputs) Console.WriteLine($"    {spec}");

                    var tokenizer = HfTokenizer.FromPretrained(ModelId);
                    var feeds = Optimum.BuildEncoderInputs(tokenizer, "HF.Net runs ONNX on .NET.", model.Session);

                    Console.WriteLine();
                    foreach (var (name, tensor) in model.Run(feeds))
                        Console.WriteLine($"  {name}: [{string.Join(" x ", tensor.Shape.ToArray())}]");

                    // The warm-up is not optional: the first call builds the execution plan and
                    // allocates the arena, and timing it reports setup as if it were inference.
                    Console.WriteLine($"\n{model.Measure(feeds, iterations: 50)}");
                    """,
                ["README.md"] = """
                    # $name$

                    Hardware-aware inference with GraviOptimum.

                    The base `Microsoft.ML.OnnxRuntime` package ships the CPU provider only. For
                    CUDA add `Microsoft.ML.OnnxRuntime.Gpu`; for DirectML add
                    `Microsoft.ML.OnnxRuntime.DirectML`.

                    ```bash
                    dotnet run -c Release
                    ```
                    """,
            }),

        // ------------------------------------------------------------ lora
        new ProjectTemplate(
            "lora-adapter",
            "LoRA adapters",
            "Fine tuning",
            "Attach, merge and save low-rank adapters in the Hugging Face PEFT format.",
            ["GraviPEFT", "GraviTransformers"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviPEFT", "GraviTransformers"),
                ["Program.cs"] = """
                    using Gravicode.HFNet.GraviPEFT;
                    using Gravicode.HFNet.GraviTransformers;

                    using var model = TransformerModel.Load("prajjwal1/bert-tiny");
                    Console.WriteLine(model);

                    // Adapters start as exactly no change: every B matrix is zero, so the adapted
                    // model is identical to the base model until something trains it.
                    var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

                    var (adapter, encoder, fraction) = peft.ParameterEfficiency();
                    Console.WriteLine($"\nadapter {adapter:N0} of {encoder:N0} parameters = {fraction:P3}");

                    // Train a task head over the frozen encoder. The transformer forward pass
                    // dominates the cost and runs once per example, not once per epoch.
                    string[] texts =
                    [
                        "great product, works perfectly", "terrible quality, broke immediately",
                        "excellent value for money", "awful, would not buy again",
                        "very happy with this", "complete rubbish",
                    ];

                    string[] labels = ["positive", "negative", "positive", "negative", "positive", "negative"];

                    Console.WriteLine("\nTraining the head ...");
                    peft.FitHead(texts, labels);

                    Console.WriteLine($"training accuracy {peft.Score(texts, labels):P0}");

                    // Now train the adapters themselves, with a head, by backpropagating through
                    // the frozen encoder. This replaces the head FitHead trained.
                    Console.WriteLine("\nTraining the adapters ...");
                    var report = peft.Train(texts, labels, new TrainingOptions { Epochs = 10, BatchSize = 2, LearningRate = 2e-3 });
                    Console.WriteLine(report);
                    Console.WriteLine($"training accuracy {peft.Score(texts, labels):P0}");

                    foreach (var text in new[] { "this is excellent", "what a disappointment" })
                        Console.WriteLine($"  {text,-24} -> {peft.Predict(text)[0]}");

                    // Merging folds the adapters into the weights exactly, so serving costs what
                    // the base model costs. It cannot be undone from the merged weights.
                    peft.Merge();
                    peft.SaveAdapter("./adapter");

                    Console.WriteLine("\nWrote ./adapter/adapter_model.safetensors and adapter_config.json");
                    """,
                ["README.md"] = """
                    # $name$

                    Parameter-efficient fine tuning with GraviPEFT.

                    HF.Net can **train, apply, merge, save and load** LoRA adapters - including
                    adapters trained with PEFT in Python. `Train` fits the adapters and a
                    classification head together; `FitHead` fits a head alone over the frozen
                    encoder, which is the cheap baseline. Saved adapters use the PEFT layout, so
                    they load in Python too.

                    ```bash
                    dotnet run
                    ```
                    """,
            }),

        // ------------------------------------------------------------ diffusion
        new ProjectTemplate(
            "diffusion",
            "Text to image",
            "Generative",
            "Stable Diffusion through ONNX, with a choice of sampler.",
            ["GraviDiffusers"],
            new Dictionary<string, string>
            {
                ["$name$.csproj"] = Csproj("GraviDiffusers"),
                ["Program.cs"] = """
                    using Gravicode.HFNet.GraviDiffusers;
                    using SixLabors.ImageSharp;

                    // Needs a repository with the ONNX layout: text_encoder/, unet/, vae_decoder/.
                    // Convert one with: optimum-cli export onnx --model <id> <out>
                    const string ModelId = "OnnxStack/stable-diffusion-v1-5-onnx";

                    Console.Write("Prompt: ");
                    var prompt = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(prompt)) prompt = "a watercolour of a mountain village at dawn";

                    try
                    {
                        using var pipeline = DiffusionPipeline.FromPretrained(
                            ModelId,
                            scheduler: new EulerScheduler(),
                            progress: new Progress<Gravicode.HFNet.GraviHub.TransferProgress>(
                                p => { if (p.Fraction is 1) Console.WriteLine($"  fetched {p.Path}"); }));

                        Console.WriteLine($"\n{pipeline}");

                        using var image = pipeline.Generate(
                            prompt,
                            new GenerationOptions(Steps: 25, GuidanceScale: 7.5, Seed: 42),
                            onStep: (step, total) => Console.Write($"\r  step {step}/{total}   "));

                        image.Save("output.png");
                        Console.WriteLine("\n\nWrote output.png");
                    }
                    catch (Gravicode.HFNet.GraviHub.HubException error)
                    {
                        Console.WriteLine($"\n{error.Message}");
                    }

                    // The same prompt and seed give the same image: DDIM at eta = 0 and Euler are
                    // both deterministic, so a generation can be shared rather than just its result.
                    """,
                ["README.md"] = """
                    # $name$

                    Text to image with GraviDiffusers.

                    Width and height must be multiples of 8 - the latent grid is eight times smaller
                    than the image in each dimension. A guidance scale above 1 runs the UNet twice
                    per step, so it doubles the work.

                    ```bash
                    dotnet run -c Release
                    ```
                    """,
            }),

        // ------------------------------------------------------------ notebook: text
        new ProjectTemplate(
            "notebook-text",
            "Notebook: text analysis",
            "Notebooks",
            "A .NET Interactive notebook for exploring a text dataset with transformers.",
            ["GraviTransformers", "GraviDatasets", "GraviTokenizers"],
            new Dictionary<string, string>
            {
                ["$name$.ipynb"] = Notebook(
                    "# $name$\n\nText analysis with HF.Net.\n\nDibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.",
                    [
                        ("markdown", "## 1. References\n\nPoint these at your HF.Net build output, or at the published packages once they exist."),
                        ("code", "#r \"nuget: Gravicode.Science.GraviNum, 1.0.0\"\n#r \"nuget: Gravicode.Science.GraviFrame, 1.0.0\"\n\n// Replace with the paths to your built HF.Net assemblies:\n// #r \"../../src/GraviTransformers/bin/Debug/net10.0/Gravicode.HFNet.GraviTransformers.dll\"\n\nusing System;\nusing System.Linq;"),
                        ("markdown", "## 2. Load a dataset"),
                        ("code", "using Gravicode.HFNet.GraviDatasets;\n\nvar data = Dataset.Load(\"imdb\");\nConsole.WriteLine(data);\ndata.Head(5).Frame"),
                        ("markdown", "## 3. Tokenize\n\nOffsets point back into the original text, so a span can be reported as a substring."),
                        ("code", "using Gravicode.HFNet.GraviTokenizers;\n\nvar tokenizer = HfTokenizer.FromPretrained(\"bert-base-uncased\");\nvar text = data.TextColumn(\"review\")[0];\nvar encoding = tokenizer.Encode(text);\n\nConsole.WriteLine($\"{encoding.Length} tokens\");\nConsole.WriteLine(string.Join(' ', encoding.Tokens.Take(30)));"),
                        ("markdown", "## 4. Embed and compare\n\n`EmbedBatch` parallelises across inputs, which is where the throughput is."),
                        ("code", "using Gravicode.HFNet.GraviTransformers;\n\nusing var model = TransformerModel.Load(\"bert-base-uncased\");\n\nvar sample = data.TextColumn(\"review\").Take(16).ToList();\nvar vectors = model.EmbedBatch(sample);\n\nConsole.WriteLine($\"[{vectors.Shape[0]} x {vectors.Shape[1]}]\");"),
                        ("markdown", "## 5. What does the model know?"),
                        ("code", "foreach (var fill in model.FillMask(\"This movie was absolutely [MASK].\", topK: 8))\n    Console.WriteLine($\"{fill.Token,-16} {fill.Score:P2}\");"),
                    ]),
                ["README.md"] = """
                    # $name$

                    A .NET Interactive notebook.

                    ```bash
                    dotnet tool install -g Microsoft.dotnet-interactive
                    dotnet interactive jupyter install
                    jupyter lab $name$.ipynb
                    ```

                    Or open it in Visual Studio Code with the Polyglot Notebooks extension.
                    """,
            }),

        // ------------------------------------------------------------ notebook: benchmark
        new ProjectTemplate(
            "notebook-benchmark",
            "Notebook: performance study",
            "Notebooks",
            "A notebook comparing managed inference against ONNX Runtime, with plots.",
            ["GraviOptimum", "GraviTransformers", "GraviAccelerate"],
            new Dictionary<string, string>
            {
                ["$name$.ipynb"] = Notebook(
                    "# $name$\n\nWhere the time actually goes.\n\nDibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.",
                    [
                        ("markdown", "## 1. What hardware is this?"),
                        ("code", "using Gravicode.HFNet.GraviAccelerate;\n\nforeach (var device in Accelerator.Devices())\n    Console.WriteLine(device);"),
                        ("markdown", "## 2. Managed inference\n\nThe managed encoder is double precision end to end. It exists so a model can be inspected and understood in pure .NET, not to serve requests."),
                        ("code", "using Gravicode.HFNet.GraviTransformers;\n\nusing var model = TransformerModel.Load(\"prajjwal1/bert-tiny\");\nvar managed = Accelerator.Measure(() => model.Embed(\"a short sentence\"), iterations: 20, warmup: 40);\n\nConsole.WriteLine($\"managed: {managed.TotalMilliseconds:F2} ms\");"),
                        ("markdown", "## 3. ONNX Runtime\n\nSingle-precision kernels written for the hardware. This is the production path."),
                        ("code", "using Gravicode.HFNet.GraviOptimum;\nusing Gravicode.HFNet.GraviTokenizers;\n\nconst string Id = \"hf-internal-testing/tiny-random-BertModel\";\nusing var onnx = Optimum.Optimize(Id, \"auto\");\n\nvar tokenizer = HfTokenizer.FromPretrained(Id);\nvar feeds = Optimum.BuildEncoderInputs(tokenizer, \"a short sentence\", onnx.Session);\n\nConsole.WriteLine(onnx.Measure(feeds, iterations: 50));"),
                        ("markdown", "## 4. Quantisation\n\nThe error is measured by reading back what was written, not predicted from the format."),
                        ("code", "// Optimum.Quantize(source, destination, QuantizationLevel.BFloat16)\n//   bfloat16 keeps float32's exponent range; float16 has more mantissa bits but\n//   underflows below about 6e-5 and loses the tail of the weight distribution."),
                        ("markdown", "**Warm-up matters.** Tiered JIT recompiles a hot method after about thirty calls, so a measurement taken with two warm-up iterations times the wrong code entirely."),
                    ]),
                ["README.md"] = """
                    # $name$

                    A performance study as a notebook.

                    Never compare timings taken in separate runs on a laptop - thermal throttling
                    moves the numbers more than most code changes do. Run the variants alternately
                    in one process and take the best of many.
                    """,
            }),
    ];

    /// <summary>
    /// Builds a .NET Interactive notebook from a title and a list of cells.
    /// </summary>
    /// <param name="title">Markdown for the first cell.</param>
    /// <param name="cells">Each cell as (kind, source), where kind is "code" or "markdown".</param>
    /// <remarks>
    /// The notebook is assembled here rather than stored as a literal .ipynb so the project name
    /// substitutes into it properly and the JSON cannot drift out of shape by hand editing.
    /// </remarks>
    private static string Notebook(string title, IReadOnlyList<(string Kind, string Source)> cells)
    {
        var all = new List<(string Kind, string Source)> { ("markdown", title) };
        all.AddRange(cells);

        var rendered = all.Select(cell =>
        {
            var lines = System.Text.Json.JsonSerializer.Serialize(SplitKeepingNewlines(cell.Source));

            // Three dollars, so the doubled closing braces inside the JSON stay literal.
            return cell.Kind == "code"
                ? $$$"""
                      {
                       "cell_type": "code",
                       "execution_count": null,
                       "metadata": {"dotnet_interactive": {"language": "csharp"}},
                       "outputs": [],
                       "source": {{{lines}}}
                      }
                    """
                : $$$"""
                      {
                       "cell_type": "markdown",
                       "metadata": {},
                       "source": {{{lines}}}
                      }
                    """;
        });

        return $$"""
            {
             "cells": [
            {{string.Join(",\n", rendered)}}
             ],
             "metadata": {
              "kernelspec": {"display_name": ".NET (C#)", "language": "C#", "name": ".net-csharp"},
              "language_info": {"file_extension": ".cs", "mimetype": "text/x-csharp", "name": "C#", "pygments_lexer": "csharp", "version": "13.0"}
             },
             "nbformat": 4,
             "nbformat_minor": 5
            }
            """;
    }

    /// <summary>Splits a cell's text the way the notebook format wants it: one string per line, newline kept.</summary>
    private static string[] SplitKeepingNewlines(string source)
    {
        var lines = source.Split('\n');
        return [.. lines.Select((line, i) => i == lines.Length - 1 ? line : line + "\n")];
    }
}
