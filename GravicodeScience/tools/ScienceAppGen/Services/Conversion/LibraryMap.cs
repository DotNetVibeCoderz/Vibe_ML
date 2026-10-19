namespace ScienceAppGen.Services.Conversion;

/// <summary>A .NET library a Python package converts to.</summary>
/// <param name="Id">Short key, e.g. <c>SciNet</c>; also the reference-book key.</param>
/// <param name="PackageId">NuGet package id, or null for the base class library.</param>
/// <param name="Version">NuGet version.</param>
/// <param name="Usings">Namespaces a converted file typically needs.</param>
/// <param name="Colour">Spectrum token the UI paints the library with.</param>
public sealed record TargetLibrary(string Id, string? PackageId, string Version, IReadOnlyList<string> Usings, string Colour);

/// <summary>How one Python package is handled.</summary>
/// <param name="Package">Top-level Python package, e.g. <c>sklearn</c>.</param>
/// <param name="Target">What it becomes, or null when nothing in the stack covers it.</param>
/// <param name="Note">Shown in the UI and sent to the model when the mapping needs a caveat.</param>
public sealed record PackageMapping(string Package, TargetLibrary? Target, string? Note = null)
{
    /// <summary>True when the package has no .NET counterpart here.</summary>
    public bool IsUnsupported => Target is null;
}

/// <summary>
/// Maps Python packages onto Gravicode.Science, Visual.Net and the base class library.
/// </summary>
/// <remarks>
/// Converted projects reference the published NuGet packages rather than this repository's
/// sources, so they build anywhere - unlike the templates, which are generated next to the
/// repository and reference its projects. The versions here are the ones the converter was
/// verified against; bump them together with a release.
/// </remarks>
public static class LibraryMap
{
    /// <summary>Gravicode.Science version that converted projects reference.</summary>
    public const string ScienceVersion = "1.4.0";

    /// <summary>Visual.Net version that converted projects reference.</summary>
    public const string VisualNetVersion = "1.1.0";

    /// <summary>The base class library: no package, nothing to reference.</summary>
    public static readonly TargetLibrary Bcl = new("BCL", null, "", [], "TextMutedBrush");

    /// <summary>NumPy.</summary>
    public static readonly TargetLibrary GraviNum = new("GraviNum", "Gravicode.Science.GraviNum", ScienceVersion,
        ["Gravicode.Science.GraviNum", "Gravicode.Science.GraviNum.Io"], "Spec1");

    /// <summary>pandas.</summary>
    public static readonly TargetLibrary GraviFrame = new("GraviFrame", "Gravicode.Science.GraviFrame", ScienceVersion,
        ["Gravicode.Science.GraviFrame", "Gravicode.Science.GraviFrame.Io"], "Spec2");

    /// <summary>scikit-learn.</summary>
    public static readonly TargetLibrary GraviLearn = new("GraviLearn", "Gravicode.Science.GraviLearn", ScienceVersion,
        ["Gravicode.Science.GraviLearn", "Gravicode.Science.GraviLearn.Linear", "Gravicode.Science.GraviLearn.Trees",
         "Gravicode.Science.GraviLearn.Preprocessing", "Gravicode.Science.GraviLearn.ModelSelection",
         "Gravicode.Science.GraviLearn.Clustering", "Gravicode.Science.GraviLearn.Decomposition",
         "Gravicode.Science.GraviLearn.Neighbors", "Gravicode.Science.GraviLearn.Anomaly",
         "Gravicode.Science.GraviLearn.Explain", "Gravicode.Science.GraviLearn.Resampling"], "Spec3");

    /// <summary>Transformers, NLTK and friends.</summary>
    public static readonly TargetLibrary GraviText = new("GraviText", "Gravicode.Science.GraviText", ScienceVersion,
        ["Gravicode.Science.GraviText.Tokenization", "Gravicode.Science.GraviText.Embeddings",
         "Gravicode.Science.GraviText.Transformers", "Gravicode.Science.GraviText.Tasks",
         "Gravicode.Science.GraviText.Vectorization", "Gravicode.Science.GraviText.Linguistics",
         "Gravicode.Science.GraviText.Sequence", "Gravicode.Science.GraviText.Generation"], "Spec4");

    /// <summary>NetworkX, PyTorch Geometric, DGL.</summary>
    public static readonly TargetLibrary GraviGraph = new("GraviGraph", "Gravicode.Science.GraviGraph", ScienceVersion,
        ["Gravicode.Science.GraviGraph", "Gravicode.Science.GraviGraph.Algorithms",
         "Gravicode.Science.GraviGraph.Embeddings", "Gravicode.Science.GraviGraph.Neural"], "Spec5");

    /// <summary>PyMC, Stan, ArviZ.</summary>
    public static readonly TargetLibrary GraviProb = new("GraviProb", "Gravicode.Science.GraviProb", ScienceVersion,
        ["Gravicode.Science.GraviProb", "Gravicode.Science.GraviProb.Models"], "Spec6");

    /// <summary>SciPy.</summary>
    public static readonly TargetLibrary SciNet = new("SciNet", "Gravicode.Science.SciNet", ScienceVersion,
        ["Gravicode.Science.SciNet"], "AccentBrush");

    /// <summary>matplotlib.</summary>
    public static readonly TargetLibrary Matplotlib = new("Matplotlib.Net", "Gravicode.Matplotlib.Net", VisualNetVersion,
        ["Matplotlib.Net", "Matplotlib.Net.Artists", "Matplotlib.Net.Rendering", "Matplotlib.Net.Styling",
         "Visual.Net.Core.Colors"], "AccentBrightBrush");

    /// <summary>seaborn.</summary>
    public static readonly TargetLibrary Seaborn = new("Seaborn.Net", "Gravicode.Seaborn.Net", VisualNetVersion,
        ["Seaborn.Net", "Matplotlib.Net", "Matplotlib.Net.Artists", "Matplotlib.Net.Rendering",
         "Visual.Net.Core.Colors"], "AccentBrightBrush");

    /// <summary>plotly.</summary>
    public static readonly TargetLibrary Plotly = new("Plotly.Net", "Gravicode.Plotly.Net", VisualNetVersion,
        ["Plotly.Net", "Plotly.Net.Traces", "Visual.Net.Core.Colors"], "AccentBrightBrush");

    private static readonly Dictionary<string, (TargetLibrary? Target, string? Note)> Packages = new(StringComparer.Ordinal)
    {
        ["numpy"] = (GraviNum, null),
        ["pandas"] = (GraviFrame, null),
        ["sklearn"] = (GraviLearn, null),
        ["scipy"] = (SciNet, null),
        ["statsmodels"] = (SciNet, "statsmodels has no direct port: SciNet.Stats covers the tests and distributions, GraviLearn the regressions."),
        ["matplotlib"] = (Matplotlib, null),
        ["seaborn"] = (Seaborn, null),
        ["plotly"] = (Plotly, null),
        ["networkx"] = (GraviGraph, null),
        ["torch_geometric"] = (GraviGraph, "Model code converts to GraviGraph's GNN layers; pretrained torch checkpoints do not load."),
        ["dgl"] = (GraviGraph, "Model code converts to GraviGraph's GNN layers; pretrained checkpoints do not load."),
        ["pymc"] = (GraviProb, null),
        ["pymc3"] = (GraviProb, null),
        ["pystan"] = (GraviProb, "Stan programs are rewritten as GraviProb models, not compiled."),
        ["stan"] = (GraviProb, "Stan programs are rewritten as GraviProb models, not compiled."),
        ["arviz"] = (GraviProb, "Diagnostics map to GraviProb's summaries and WAIC/LOO."),
        ["transformers"] = (GraviText, "Architectures and tokenizers convert; hub downloads become explicit weight loading."),
        ["tokenizers"] = (GraviText, null),
        ["nltk"] = (GraviText, null),
        ["sentencepiece"] = (GraviText, null),
        ["gensim"] = (GraviText, null),
        ["torch"] = (null, "PyTorch has no counterpart here; tensor maths can be rewritten on GraviNum's autodiff tape by hand."),
        ["tensorflow"] = (null, "TensorFlow has no counterpart here."),
        ["keras"] = (null, "Keras has no counterpart here."),
        ["xgboost"] = (GraviLearn, "Becomes GraviLearn's gradient boosting; hyperparameters do not carry over one to one."),
        ["lightgbm"] = (GraviLearn, "Becomes GraviLearn's gradient boosting; hyperparameters do not carry over one to one."),
        ["requests"] = (Bcl, "Becomes HttpClient."),
        ["tqdm"] = (Bcl, "Progress bars are dropped or become Console output."),
    };

    /// <summary>Python standard-library modules: they map to the BCL.</summary>
    private static readonly HashSet<string> StandardLibrary = new(StringComparer.Ordinal)
    {
        "__future__", "abc", "argparse", "array", "asyncio", "base64", "bisect", "calendar", "collections",
        "concurrent", "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal", "enum", "functools",
        "gc", "glob", "gzip", "hashlib", "heapq", "io", "itertools", "json", "logging", "math", "multiprocessing",
        "operator", "os", "pathlib", "pickle", "pprint", "queue", "random", "re", "shutil", "statistics",
        "string", "struct", "subprocess", "sys", "tempfile", "textwrap", "threading", "time", "timeit",
        "typing", "typing_extensions", "unittest", "urllib", "uuid", "warnings", "zipfile", "fractions",
        "numbers", "secrets", "traceback", "inspect", "types", "weakref", "platform", "signal", "socket",
    };

    /// <summary>
    /// The <c>using</c> directives a converted module gets, decided here rather than by the model.
    /// </summary>
    /// <remarks>
    /// Left to itself the model guesses namespaces (<c>Visual.Net.Matplotlib</c>,
    /// <c>using Stats;</c>) and spends repair rounds on them. The lists above deliberately omit
    /// namespaces whose type names collide with another library's - <c>GraviNum.Signal</c> has an
    /// <c>Fft</c> like SciNet's, <c>Visual.Net.Core.Data</c> a <c>Datasets</c> like GraviLearn's.
    /// SciNet's sub-namespaces come in only when the module uses the matching scipy submodule.
    /// <para>
    /// The core data libraries are project-wide, not per module: a module that never imports
    /// pandas still receives the DataFrame another module loaded, and without the namespace the
    /// model declared its own <c>class DataFrame</c> - which then shadowed the real one in every
    /// file. GraviNum comes with any of them, because all of them are built on <c>NdArray</c>.
    /// </para>
    /// </remarks>
    /// <param name="module">The module being converted.</param>
    /// <param name="projectPackages">Every package mapping in the project.</param>
    public static IReadOnlyList<string> UsingsFor(ModuleAnalysis module, IEnumerable<PackageMapping> projectPackages)
    {
        var usings = new List<string>();
        var own = module.ExternalPackages.Select(Map).Where(m => m.Target is not null).Select(m => m.Target!).ToList();
        var shared = SharedTargets(projectPackages);

        if (own.Concat(shared).Any(t => t.Id.StartsWith("Gravi", StringComparison.Ordinal) || t == SciNet))
            usings.AddRange(GraviNum.Usings);
        foreach (var target in shared.Concat(own)) usings.AddRange(target.Usings);

        bool Uses(string prefix) => module.UsedSymbols.Any(s => s == prefix || s.StartsWith(prefix + ".", StringComparison.Ordinal));
        if (Uses("scipy.sparse")) usings.Add("Gravicode.Science.SciNet.Sparse");
        if (Uses("scipy.spatial")) usings.Add("Gravicode.Science.SciNet.Spatial");
        if (Uses("scipy.io")) usings.Add("Gravicode.Science.SciNet.IO");

        return usings.Distinct(StringComparer.Ordinal).ToList();
    }

    /// <summary>
    /// The Gravicode data libraries used anywhere in the project. Their objects cross module
    /// boundaries, so every module gets their usings and curated reference; SciNet and the plotting
    /// libraries stay with the modules that import them.
    /// </summary>
    public static IReadOnlyList<TargetLibrary> SharedTargets(IEnumerable<PackageMapping> projectPackages)
        => projectPackages.Select(p => p.Target)
            .OfType<TargetLibrary>()
            .Where(t => t == GraviNum || t == GraviFrame || t == GraviLearn || t == GraviText || t == GraviGraph || t == GraviProb)
            .Distinct()
            .ToList();

    /// <summary>How a top-level Python package is handled.</summary>
    public static PackageMapping Map(string package)
    {
        if (Packages.TryGetValue(package, out var mapped)) return new PackageMapping(package, mapped.Target, mapped.Note);
        if (StandardLibrary.Contains(package)) return new PackageMapping(package, Bcl);
        return new PackageMapping(package, null,
            $"'{package}' has no counterpart in Gravicode.Science or Visual.Net; the model will port what it can by hand and mark the rest.");
    }

    /// <summary>The distinct mappings for a set of modules, third-party packages first.</summary>
    public static IReadOnlyList<PackageMapping> MapAll(IEnumerable<ModuleAnalysis> modules)
        => modules.SelectMany(m => m.ExternalPackages)
            .Distinct(StringComparer.Ordinal)
            .Select(Map)
            .OrderBy(m => m.Target == Bcl ? 1 : 0)
            .ThenBy(m => m.Package, StringComparer.Ordinal)
            .ToList();

    /// <summary>
    /// The NuGet packages a converted project needs. Each Gravicode library pulls in its own
    /// dependencies, but listing them explicitly keeps the versions aligned at
    /// <see cref="ScienceVersion"/>: Visual.Net depends on an older Gravicode.Science, and an
    /// explicit reference is what makes NuGet pick the newer one.
    /// </summary>
    public static IReadOnlyList<TargetLibrary> PackagesFor(IEnumerable<PackageMapping> mappings)
    {
        var targets = mappings.Where(m => m.Target?.PackageId is not null).Select(m => m.Target!).ToHashSet();

        if (targets.Contains(Seaborn)) targets.Add(Matplotlib);
        if (targets.Contains(Matplotlib) || targets.Contains(Seaborn) || targets.Contains(Plotly))
        {
            targets.Add(GraviNum);
            targets.Add(GraviFrame);
            targets.Add(GraviLearn);
        }
        if (targets.Contains(GraviLearn) || targets.Contains(GraviFrame) || targets.Contains(SciNet)
            || targets.Contains(GraviProb) || targets.Contains(GraviText) || targets.Contains(GraviGraph))
            targets.Add(GraviNum);

        return targets.OrderBy(t => t.PackageId, StringComparer.Ordinal).ToList();
    }
}
