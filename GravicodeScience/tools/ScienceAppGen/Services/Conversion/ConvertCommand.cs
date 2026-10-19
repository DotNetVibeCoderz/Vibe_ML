using System.Runtime.InteropServices;
using ScienceAppGen.Models;

namespace ScienceAppGen.Services.Conversion;

/// <summary>
/// <c>ScienceAppGen --convert &lt;file-or-folder&gt;</c>: the converter without the window.
/// </summary>
/// <remarks>
/// Uses the same settings as the app (app.config, with <c>SCIENCEAPPGEN_APIKEY</c> and
/// <c>SCIENCEAPPGEN_ENDPOINT</c> winning over the file), so a key never has to appear on a command
/// line. ScienceAppGen is a WinExe, which has no console of its own on Windows; it attaches to the
/// parent's so the output appears in the terminal that started it.
/// </remarks>
public static partial class ConvertCommand
{
    /// <summary>The usage text.</summary>
    public const string Usage = """
        ScienceAppGen --convert <file.py | notebook.ipynb | folder> [options]

          --out <dir>         Output folder (default: <name>.CSharp next to the source)
          --name <Name>       C# project name (default: from the file or folder)
          --rounds <n>        Build-and-repair rounds (default 3)
          --no-build          Write the files without building
          --run               Run the program after a successful build
          --plan              Print the plan (files, names, packages) and stop - no model calls
          --prompt            Print each module's conversion prompt and stop - no model calls
          --provider <p>      OpenAI | AzureOpenAI | Anthropic | Google | Ollama (default: app.config)
          --model <m>         Model or deployment (default: app.config)
          --endpoint <url>    Endpoint (default: app.config or SCIENCEAPPGEN_ENDPOINT)

        The API key comes from app.config or the SCIENCEAPPGEN_APIKEY environment variable.
        """;

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool AttachConsole(int processId);

    /// <summary>True when the arguments ask for a headless conversion.</summary>
    public static bool IsRequested(string[] args)
        => args.Contains("--convert", StringComparer.OrdinalIgnoreCase)
           && !ConverterScreenshot.IsRequested(args);   // there --convert means "convert, then capture"

    /// <summary>Runs the command and returns the process exit code.</summary>
    public static async Task<int> RunAsync(string[] args)
    {
        if (OperatingSystem.IsWindows()) AttachConsole(-1);

        string? Option(string name)
        {
            var index = Array.FindIndex(args, a => a.Equals(name, StringComparison.OrdinalIgnoreCase));
            return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
        }
        bool Flag(string name) => args.Contains(name, StringComparer.OrdinalIgnoreCase);

        var source = Option("--convert");
        if (source is null || source.StartsWith("--", StringComparison.Ordinal))
        {
            Console.WriteLine(Usage);
            return 2;
        }

        try
        {
            var project = PythonProjectScanner.Scan(source);
            var plan = ConversionPlanner.Plan(project, Option("--name"));
            PrintPlan(plan);
            if (Flag("--plan")) return 0;
            if (Flag("--prompt"))
            {
                foreach (var module in plan.Modules)
                {
                    Console.WriteLine($"==================== {module.PythonPath}");
                    Console.WriteLine(ConversionPrompts.Convert(plan, module,
                        ApiReferenceBuilder.Build(module.Analysis, plan.Packages), new Dictionary<string, string>()));
                }
                return 0;
            }

            var settings = new ConfigurationService().Load();
            if (Option("--provider") is { } provider) settings.Provider = Enum.Parse<LlmProvider>(provider, ignoreCase: true);
            if (Option("--model") is { } model) settings.Model = model;
            if (Option("--endpoint") is { } endpoint) settings.Endpoint = endpoint;
            if (!settings.IsConfigured)
            {
                Console.Error.WriteLine(settings.MissingConfiguration);
                return 2;
            }

            var output = Option("--out") ?? DefaultOutput(project, plan);
            var options = new ConversionOptions
            {
                OutputDirectory = output,
                ProjectName = plan.ProjectName,
                Build = !Flag("--no-build"),
                RunAfterBuild = Flag("--run"),
                MaxRepairRounds = int.TryParse(Option("--rounds"), out var rounds) ? rounds : 3,
            };

            using var llm = new SemanticKernelLlmClient(settings);
            using var cancel = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancel.Cancel(); };

            var progress = new SynchronousProgress(Print);
            var result = await new PythonConverter(llm).ConvertAsync(plan, options, progress, cancel.Token);

            Console.WriteLine();
            Console.WriteLine(result.Succeeded ? "Converted." : "Converted with problems.");
            Console.WriteLine($"  output       {result.OutputDirectory}");
            Console.WriteLine($"  requests     {result.LlmCalls} ({result.RepairRounds} repair round(s), {result.Duration.TotalSeconds:F0} s)");
            Console.WriteLine($"  errors left  {result.RemainingErrors.Count}");
            Console.WriteLine($"  TODO notes   {result.Todos.Count}");
            if (result.ProgramExitCode is { } code) Console.WriteLine($"  program exit {code}");
            return result.Succeeded && (result.ProgramExitCode ?? 0) == 0 ? 0 : 1;
        }
        catch (Exception ex) when (ex is InvalidOperationException or FileNotFoundException or ArgumentException)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }
    }

    /// <summary><c>signal_analysis.py</c> → <c>SignalAnalysis.CSharp</c> beside it.</summary>
    public static string DefaultOutput(PythonProject project, ConversionPlan plan)
    {
        var parent = project.Files.Count == 1 && File.Exists(Path.Combine(project.Root, project.Files[0].RelativePath))
                     && project.Root == Path.GetDirectoryName(Path.Combine(project.Root, project.Files[0].RelativePath))
            ? project.Root
            : Path.GetDirectoryName(project.Root.TrimEnd(Path.DirectorySeparatorChar)) ?? project.Root;
        return Path.Combine(parent, plan.ProjectName + ".CSharp");
    }

    private static void PrintPlan(ConversionPlan plan)
    {
        Console.WriteLine($"{plan.ProjectName}: {plan.Modules.Count} module(s), {plan.Project.DataFiles.Count} data file(s)");
        foreach (var module in plan.Modules)
            Console.WriteLine($"  {module.PythonPath,-36} -> {module.CSharpPath}{(module.IsEntryPoint ? "  (entry point)" : "")}");
        foreach (var package in plan.Packages)
            Console.WriteLine($"  {package.Package,-16} -> {(package.Target is null ? "NO COUNTERPART" : package.Target.PackageId ?? ".NET")}");
        foreach (var skipped in plan.Project.Skipped) Console.WriteLine($"  skipped: {skipped}");
        Console.WriteLine();
    }

    private static void Print(ConversionEvent e)
    {
        var prefix = e.Kind switch
        {
            ConversionEventKind.Warning => "! ",
            ConversionEventKind.Error => "X ",
            ConversionEventKind.ProgramOutput => "  | ",
            ConversionEventKind.Module => "- ",
            _ => "",
        };
        // Build output is only interesting when it is an error; the parsed errors are reported anyway.
        if (e.Kind == ConversionEventKind.Build) return;
        Console.WriteLine(prefix + e.Message);
    }

    /// <summary>IProgress that calls back on the reporting thread (Progress&lt;T&gt; posts to a context the console does not have).</summary>
    private sealed class SynchronousProgress(Action<ConversionEvent> handler) : IProgress<ConversionEvent>
    {
        private readonly Lock _gate = new();

        public void Report(ConversionEvent value)
        {
            lock (_gate) handler(value);
        }
    }
}
