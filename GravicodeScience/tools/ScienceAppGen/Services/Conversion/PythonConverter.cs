using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;

namespace ScienceAppGen.Services.Conversion;

/// <summary>Options for one conversion.</summary>
public sealed class ConversionOptions
{
    /// <summary>Folder to write the C# project into. Must be empty or a previous conversion.</summary>
    public required string OutputDirectory { get; init; }

    /// <summary>C# project name; derived from the Python file or folder when null.</summary>
    public string? ProjectName { get; init; }

    /// <summary>Run <c>dotnet build</c> and the repair loop.</summary>
    public bool Build { get; init; } = true;

    /// <summary>How many build-and-repair rounds to try before giving up.</summary>
    public int MaxRepairRounds { get; init; } = 3;

    /// <summary>Run the program after a successful build and capture its output.</summary>
    public bool RunAfterBuild { get; init; }

    /// <summary>LLM requests in flight at once, for modules that do not depend on each other.</summary>
    public int MaxParallelRequests { get; init; } = 3;

    /// <summary>Kill the converted program after this long when <see cref="RunAfterBuild"/> is set.</summary>
    public TimeSpan RunTimeout { get; init; } = TimeSpan.FromMinutes(2);

    /// <summary>
    /// Keep every prompt, reply and build log under <c>.conversion/</c> in the output folder, so a
    /// result can be audited and a bad conversion diagnosed.
    /// </summary>
    public bool KeepTranscript { get; init; } = true;
}

/// <summary>What a conversion event is about.</summary>
public enum ConversionEventKind
{
    /// <summary>General progress.</summary>
    Info,
    /// <summary>A module's state changed; see <see cref="ConversionEvent.ModulePath"/> and <see cref="ConversionEvent.Status"/>.</summary>
    Module,
    /// <summary>A line of <c>dotnet</c> output.</summary>
    Build,
    /// <summary>Something the user should look at, but the conversion continues.</summary>
    Warning,
    /// <summary>A failure.</summary>
    Error,
    /// <summary>A line printed by the converted program.</summary>
    ProgramOutput,
}

/// <summary>The state of one module during a conversion.</summary>
public enum ModuleStatus
{
    /// <summary>Not started.</summary>
    Pending,
    /// <summary>Waiting for the model's first version.</summary>
    Converting,
    /// <summary>Written, not yet built.</summary>
    Converted,
    /// <summary>The build reported errors in it.</summary>
    HasErrors,
    /// <summary>Waiting for the model's corrected version.</summary>
    Repairing,
    /// <summary>The project builds, so this file compiles.</summary>
    Compiles,
    /// <summary>The model returned nothing usable, or errors remained after the last round.</summary>
    Failed,
}

/// <summary>A progress report.</summary>
/// <param name="Kind">What it is about.</param>
/// <param name="Message">Human-readable text.</param>
/// <param name="ModulePath">The Python path of the module concerned, for <see cref="ConversionEventKind.Module"/>.</param>
/// <param name="Status">The module's new state.</param>
/// <param name="CSharp">The module's current C# when it changed.</param>
public sealed record ConversionEvent(ConversionEventKind Kind, string Message, string? ModulePath = null,
    ModuleStatus? Status = null, string? CSharp = null);

/// <summary>The outcome of a conversion.</summary>
public sealed class ConversionResult
{
    /// <summary>Where the project was written.</summary>
    public required string OutputDirectory { get; init; }

    /// <summary>The generated csproj.</summary>
    public required string ProjectFile { get; init; }

    /// <summary>True when the last build succeeded (or the build was skipped and every file came back).</summary>
    public required bool Succeeded { get; init; }

    /// <summary>Errors left after the last round.</summary>
    public required IReadOnlyList<BuildDiagnostic> RemainingErrors { get; init; }

    /// <summary>Repair rounds used.</summary>
    public required int RepairRounds { get; init; }

    /// <summary>LLM requests made.</summary>
    public required int LlmCalls { get; init; }

    /// <summary><c>// TODO(convert):</c> notes the model left, as "file: note".</summary>
    public required IReadOnlyList<string> Todos { get; init; }

    /// <summary>The program's output when <see cref="ConversionOptions.RunAfterBuild"/> was set.</summary>
    public string? ProgramOutput { get; init; }

    /// <summary>The program's exit code when it was run.</summary>
    public int? ProgramExitCode { get; init; }

    /// <summary>Wall-clock time.</summary>
    public required TimeSpan Duration { get; init; }
}

/// <summary>
/// Converts a Python project to a C# project: plan, convert per module, build, repair.
/// </summary>
/// <remarks>
/// <para>
/// Modules are converted dependencies-first, and each prompt carries the public surface of the
/// files already written, so a caller is converted against the signatures its callee really got.
/// Modules with no dependency on each other go out in parallel.
/// </para>
/// <para>
/// The build is the judge. A conversion that "looks right" but does not compile is the common
/// failure, so after writing everything the project is built, errors are grouped by file, and each
/// failing file goes back to the model with its own errors and the API reference - up to
/// <see cref="ConversionOptions.MaxRepairRounds"/> times. A restore failure has no file to blame and
/// ends the loop immediately rather than burning rounds on it.
/// </para>
/// </remarks>
public sealed class PythonConverter(ILlmClient llm)
{
    /// <summary>Marks a folder as a previous conversion, which may be overwritten.</summary>
    public const string MarkerFile = ".converted-from-python";

    private int _calls;
    private string? _transcript;

    /// <summary>Runs a planned conversion.</summary>
    public async Task<ConversionResult> ConvertAsync(ConversionPlan plan, ConversionOptions options,
        IProgress<ConversionEvent>? progress = null, CancellationToken cancellationToken = default)
    {
        var clock = Stopwatch.StartNew();
        _calls = 0;
        void Report(ConversionEventKind kind, string message, PlannedModule? module = null, ModuleStatus? status = null, string? code = null)
            => progress?.Report(new ConversionEvent(kind, message, module?.PythonPath, status, code));

        var output = PrepareOutput(plan, options.OutputDirectory);
        _transcript = options.KeepTranscript ? Path.Combine(output, ".conversion") : null;
        if (_transcript is not null)
        {
            if (Directory.Exists(_transcript)) Directory.Delete(_transcript, recursive: true);
            Directory.CreateDirectory(_transcript);
        }
        var projectFile = Path.Combine(output, plan.ProjectName + ".csproj");
        await File.WriteAllTextAsync(projectFile, ConversionSupport.Csproj(plan), cancellationToken);
        CopyDataFiles(plan, output);
        Report(ConversionEventKind.Info, $"Writing {plan.ProjectName} to {output} with {llm.Description}.");

        foreach (var package in plan.Unsupported)
            Report(ConversionEventKind.Warning, $"{package.Package}: {package.Note}");

        // ------------------------------------------------------------ first pass, by dependency level
        var code = new ConcurrentDictionary<string, string>(StringComparer.Ordinal);       // C# path -> source
        var surfaces = new ConcurrentDictionary<string, string>(StringComparer.Ordinal);   // C# path -> public surface
        var references = plan.Modules.ToDictionary(m => m, m => ApiReferenceBuilder.Build(m.Analysis, plan.Packages));
        var failed = new HashSet<PlannedModule>();

        foreach (var level in Levels(plan))
        {
            using var gate = new SemaphoreSlim(Math.Max(1, options.MaxParallelRequests));
            var snapshot = new Dictionary<string, string>(surfaces, StringComparer.Ordinal);

            var tasks = level.Select(async module =>
            {
                await gate.WaitAsync(cancellationToken);
                try
                {
                    Report(ConversionEventKind.Module, $"Converting {module.PythonPath}", module, ModuleStatus.Converting);
                    var reply = await CallAsync(ConversionPrompts.Convert(plan, module, references[module], snapshot),
                        $"0-convert-{Slug(module.CSharpPath)}", cancellationToken);
                    var csharp = ConversionSupport.ExtractCode(reply);
                    if (csharp is null)
                    {
                        lock (failed) failed.Add(module);
                        Report(ConversionEventKind.Module, $"{module.PythonPath}: the model returned no C# code", module, ModuleStatus.Failed);
                        return;
                    }

                    csharp = await WriteAsync(plan, output, module, csharp, cancellationToken);
                    code[module.CSharpPath] = csharp;
                    surfaces[module.CSharpPath] = ConversionSupport.PublicSurface(csharp);
                    Report(ConversionEventKind.Module, $"Converted {module.PythonPath} -> {module.CSharpPath}", module, ModuleStatus.Converted, csharp);
                }
                finally
                {
                    gate.Release();
                }
            });
            await Task.WhenAll(tasks);
        }

        // ------------------------------------------------------------ build and repair
        // Names the project owns: a Python class called Pipeline may legitimately become one.
        var owned = plan.Modules.SelectMany(m => m.Analysis.Definitions.Where(d => d.Kind == "class")
                .Select(d => CSharpNaming.ToPascalCase(d.Name)))
            .Concat(plan.Modules.Select(m => m.ClassName))
            .ToHashSet(StringComparer.Ordinal);
        IReadOnlyList<BuildDiagnostic> errors = [];
        var rounds = 0;
        var built = false;
        string? programOutput = null;
        int? exitCode = null;

        if (options.Build && failed.Count < plan.Modules.Count)
        {
            for (var round = 0; ; round++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Report(ConversionEventKind.Info, round == 0 ? "Building..." : $"Rebuilding after repair round {round}...");
                var build = await DotnetCli.RunAsync("build -c Debug --nologo -v q", output,
                    line => Report(ConversionEventKind.Build, line), cancellationToken);
                if (_transcript is not null)
                    await File.WriteAllTextAsync(Path.Combine(_transcript, $"{round}-build.log"), build.Output, cancellationToken);

                // A stub of a library type compiles, so it is checked separately from the build.
                var shadows = plan.Modules
                    .Where(m => code.ContainsKey(m.CSharpPath))
                    .SelectMany(m => ConversionSupport.FindShadowedLibraryTypes(m.CSharpPath, code[m.CSharpPath],
                        LibraryMap.UsingsFor(m.Analysis, plan.Packages), owned))
                    .ToList();

                var crashed = false;
                if (build.Succeeded && shadows.Count == 0)
                {
                    built = true;
                    errors = [];
                    await PruneUsingsAsync(plan, output, code, Report, cancellationToken);
                    foreach (var module in plan.Modules.Where(m => !failed.Contains(m)))
                        Report(ConversionEventKind.Module, $"{module.CSharpPath} compiles", module, ModuleStatus.Compiles, code.GetValueOrDefault(module.CSharpPath));
                    Report(ConversionEventKind.Info, round == 0 ? "The project builds." : $"The project builds after {round} repair round(s).");
                    if (!options.RunAfterBuild) break;

                    // The Python original runs, so the conversion should too. An exception thrown
                    // in a converted file is one more thing to send back - compilers cannot see a
                    // wrong column name.
                    Report(ConversionEventKind.Info, "Running the converted program...");
                    using (var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
                    {
                        timeout.CancelAfter(options.RunTimeout);
                        var run = await DotnetCli.RunAsync("run --no-build", output,
                            line => Report(ConversionEventKind.ProgramOutput, line), timeout.Token);
                        programOutput = run.Output;
                        exitCode = run.ExitCode;
                        if (run.Succeeded)
                        {
                            Report(ConversionEventKind.Info, "The program ran to completion.");
                            break;
                        }

                        errors = ConversionSupport.ParseRuntimeFailure(run.Output, output, plan.Modules.Select(m => m.CSharpPath));
                        if (errors.Count == 0 || round >= options.MaxRepairRounds)
                        {
                            Report(ConversionEventKind.Warning, $"The program exited with code {run.ExitCode}"
                                + (errors.Count == 0 ? "." : $"; no repair rounds left. {errors[0].Message}"));
                            break;
                        }
                    }
                    crashed = true;
                    Report(ConversionEventKind.Warning, "The program builds but crashed: " + errors[0].Message);
                }
                else
                {
                    errors = [.. shadows, .. build.Succeeded ? [] : ConversionSupport.ParseBuildErrors(build.Output, output)];
                }

                if (build.Succeeded && !crashed)
                {
                    // It builds, but a file stubs a library type; if no rounds are left, the build stands.
                    built = round >= options.MaxRepairRounds;
                    Report(ConversionEventKind.Warning, $"The project builds, but {shadows.Count} declaration(s) shadow library types.");
                }
                var byModule = plan.Modules
                    .Select(m => (Module: m, Errors: (IReadOnlyList<BuildDiagnostic>)errors
                        .Where(e => string.Equals(e.File, m.CSharpPath, StringComparison.OrdinalIgnoreCase)).ToList()))
                    .Where(x => x.Errors.Count > 0)
                    .ToList();

                if (!crashed) Report(ConversionEventKind.Warning, $"Build failed with {errors.Count} error(s) in {byModule.Count} file(s).");
                foreach (var (module, moduleErrors) in byModule)
                    Report(ConversionEventKind.Module, $"{module.CSharpPath}: {moduleErrors.Count} error(s)", module, ModuleStatus.HasErrors);

                if (byModule.Count == 0)
                {
                    Report(ConversionEventKind.Error, "The build failed outside the converted files (restore or project errors); they cannot be repaired by the model:\n"
                        + string.Join("\n", errors.Take(10)));
                    break;
                }

                if (round >= options.MaxRepairRounds)
                {
                    Report(ConversionEventKind.Error, $"Errors remain after {round} repair round(s). The project is written; fix the rest in the editor.");
                    foreach (var (module, _) in byModule)
                        Report(ConversionEventKind.Module, $"{module.CSharpPath} still has errors", module, ModuleStatus.Failed);
                    break;
                }

                rounds = round + 1;
                var snapshot = new Dictionary<string, string>(surfaces, StringComparer.Ordinal);
                using var gate = new SemaphoreSlim(Math.Max(1, options.MaxParallelRequests));
                await Task.WhenAll(byModule.Select(async item =>
                {
                    await gate.WaitAsync(cancellationToken);
                    try
                    {
                        var (module, moduleErrors) = item;
                        Report(ConversionEventKind.Module, $"Repairing {module.CSharpPath} ({moduleErrors.Count} error(s))", module, ModuleStatus.Repairing);
                        var reply = await CallAsync(ConversionPrompts.Repair(plan, module, code[module.CSharpPath], moduleErrors,
                            references[module], snapshot, crashed ? programOutput : null), $"{rounds}-repair-{Slug(module.CSharpPath)}", cancellationToken);
                        var csharp = ConversionSupport.ExtractCode(reply);
                        if (csharp is null)
                        {
                            Report(ConversionEventKind.Warning, $"{module.CSharpPath}: the repair reply had no code; keeping the previous version.");
                            return;
                        }

                        csharp = await WriteAsync(plan, output, module, csharp, cancellationToken);
                        code[module.CSharpPath] = csharp;
                        surfaces[module.CSharpPath] = ConversionSupport.PublicSurface(csharp);
                        Report(ConversionEventKind.Module, $"Repaired {module.CSharpPath}", module, ModuleStatus.Converted, csharp);
                    }
                    finally
                    {
                        gate.Release();
                    }
                }));
            }

        }

        var todos = Todos(code);
        await File.WriteAllTextAsync(Path.Combine(output, "CONVERSION.md"),
            Report_(plan, llm.Description, built, options.Build, errors, rounds, _calls, todos, clock.Elapsed), cancellationToken);

        return new ConversionResult
        {
            OutputDirectory = output,
            ProjectFile = projectFile,
            Succeeded = failed.Count == 0 && (built || !options.Build),
            RemainingErrors = errors,
            RepairRounds = rounds,
            LlmCalls = _calls,
            Todos = todos,
            ProgramOutput = programOutput,
            ProgramExitCode = exitCode,
            Duration = clock.Elapsed,
        };
    }

    private async Task<string> CallAsync(string user, string name, CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref _calls);
        var clock = Stopwatch.StartNew();
        var reply = await llm.CompleteAsync(ConversionPrompts.System, user, cancellationToken);
        if (_transcript is not null)
        {
            await File.WriteAllTextAsync(Path.Combine(_transcript, name + ".prompt.md"), user, cancellationToken);
            await File.WriteAllTextAsync(Path.Combine(_transcript, name + ".reply.md"),
                $"<!-- {llm.Description}, {clock.Elapsed.TotalSeconds:F0} s -->\n{reply}", cancellationToken);
        }
        return reply;
    }

    private static string Slug(string path) => path.Replace('/', '_').Replace('\\', '_');

    /// <summary>
    /// Removes the using directives the files do not need, then proves it with a rebuild; on
    /// failure every file goes back to the version that built.
    /// </summary>
    private static async Task PruneUsingsAsync(ConversionPlan plan, string output, ConcurrentDictionary<string, string> code,
        Action<ConversionEventKind, string, PlannedModule?, ModuleStatus?, string?> report, CancellationToken cancellationToken)
    {
        var originals = new Dictionary<string, string>(code, StringComparer.Ordinal);
        var changed = false;
        foreach (var module in plan.Modules.Where(m => code.ContainsKey(m.CSharpPath)))
        {
            var pruned = ConversionSupport.PruneUnusedUsings(code[module.CSharpPath], ["Gravicode.Science.GraviNum"]);
            if (pruned == code[module.CSharpPath]) continue;
            code[module.CSharpPath] = pruned;
            await File.WriteAllTextAsync(Path.Combine(output, module.CSharpPath), pruned, new UTF8Encoding(false), cancellationToken);
            changed = true;
        }
        if (!changed) return;

        var rebuild = await DotnetCli.RunAsync("build -c Debug --nologo -v q", output, null, cancellationToken);
        if (rebuild.Succeeded) return;

        foreach (var (path, source) in originals)
        {
            code[path] = source;
            await File.WriteAllTextAsync(Path.Combine(output, path), source, new UTF8Encoding(false), cancellationToken);
        }
        await DotnetCli.RunAsync("build -c Debug --nologo -v q", output, null, cancellationToken);
        report(ConversionEventKind.Info, "Kept the full using list: trimming it broke the build.", null, null, null);
    }

    /// <summary>Groups modules so each level depends only on earlier levels.</summary>
    public static IReadOnlyList<IReadOnlyList<PlannedModule>> Levels(ConversionPlan plan)
    {
        var byName = plan.Modules.ToDictionary(m => m.Analysis.File.ModuleName, StringComparer.Ordinal);
        var level = new Dictionary<PlannedModule, int>();

        // Modules is already dependencies-first, so one pass in that order settles every level.
        foreach (var module in plan.Modules)
        {
            var dependencies = module.Analysis.LocalDependencies
                .Where(byName.ContainsKey)
                .Select(d => byName[d])
                .Where(level.ContainsKey);   // a cycle's back edge is ignored, as in the ordering
            level[module] = dependencies.Select(d => level[d] + 1).DefaultIfEmpty(0).Max();
        }

        // The entry point goes last on its own: it calls everything else.
        var entry = plan.Modules.FirstOrDefault(m => m.IsEntryPoint);
        if (entry is not null) level[entry] = level.Values.DefaultIfEmpty(0).Max() + (plan.Modules.Count > 1 ? 1 : 0);

        return level.GroupBy(kv => kv.Value).OrderBy(g => g.Key)
            .Select(g => (IReadOnlyList<PlannedModule>)g.Select(kv => kv.Key).ToList())
            .ToList();
    }

    private static string PrepareOutput(ConversionPlan plan, string directory)
    {
        var output = Path.GetFullPath(directory);
        var source = Path.GetFullPath(plan.Project.Root).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        if ((output + Path.DirectorySeparatorChar).StartsWith(source, StringComparison.OrdinalIgnoreCase) && plan.Project.Files.Count > 1)
            throw new InvalidOperationException("The output folder is inside the Python project; choose a folder outside it.");

        if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any()
            && !File.Exists(Path.Combine(output, MarkerFile)))
            throw new InvalidOperationException($"'{output}' is not empty. Choose an empty folder - a previous conversion's folder may be reused.");

        Directory.CreateDirectory(output);

        // A previous conversion's .cs files would otherwise be compiled alongside the new ones.
        foreach (var stale in Directory.EnumerateFiles(output, "*.cs", SearchOption.AllDirectories)
                     .Where(p => !ProjectService.IsIgnored(Path.GetRelativePath(output, p))))
            File.Delete(stale);
        foreach (var stale in Directory.EnumerateFiles(output, "*.csproj")) File.Delete(stale);

        File.WriteAllText(Path.Combine(output, MarkerFile), $"Converted from {plan.Project.Root}\n");
        return output;
    }

    private static void CopyDataFiles(ConversionPlan plan, string output)
    {
        foreach (var data in plan.Project.DataFiles)
        {
            var target = Path.Combine(output, data.RelativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(data.FullPath, target, overwrite: true);
        }
    }

    /// <summary>Normalises the usings and writes the file; returns what was written.</summary>
    private static async Task<string> WriteAsync(ConversionPlan plan, string output, PlannedModule module, string csharp, CancellationToken cancellationToken)
    {
        csharp = ConversionSupport.NormalizeUsings(csharp, LibraryMap.UsingsFor(module.Analysis, plan.Packages));
        var path = Path.Combine(output, module.CSharpPath);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllTextAsync(path, csharp, new UTF8Encoding(false), cancellationToken);
        return csharp;
    }

    private static IReadOnlyList<string> Todos(IReadOnlyDictionary<string, string> code)
        => code.OrderBy(kv => kv.Key, StringComparer.Ordinal)
            .SelectMany(kv => kv.Value.Split('\n')
                .Select((line, index) => (line, index))
                .Where(x => x.line.Contains("TODO(convert)", StringComparison.Ordinal))
                .Select(x => $"{kv.Key}:{x.index + 1}: {x.line.Trim().TrimStart('/').Trim()}"))
            .ToList();

    private static string Report_(ConversionPlan plan, string model, bool built, bool buildRequested,
        IReadOnlyList<BuildDiagnostic> errors, int rounds, int calls, IReadOnlyList<string> todos, TimeSpan duration)
    {
        var builder = new StringBuilder();
        builder.AppendLine($"# {plan.ProjectName}").AppendLine();
        builder.AppendLine($"Converted from Python (`{plan.Project.Root}`) by ScienceAppGen with `{model}` "
                           + $"in {duration.TotalSeconds:F0} s ({calls} model requests, {rounds} repair round(s)).").AppendLine();
        builder.AppendLine(buildRequested ? built ? "**Build: succeeded.**" : $"**Build: {errors.Count} error(s) remain.**" : "Build: skipped.").AppendLine();

        builder.AppendLine("| Python | C# |").AppendLine("|---|---|");
        foreach (var module in plan.Modules) builder.AppendLine($"| `{module.PythonPath}` | `{module.CSharpPath}` |");
        builder.AppendLine();

        builder.AppendLine("| Python package | Became |").AppendLine("|---|---|");
        foreach (var package in plan.Packages)
            builder.AppendLine($"| `{package.Package}` | {(package.Target is null ? "**no counterpart**" : package.Target.PackageId ?? ".NET base library")} |");
        builder.AppendLine();

        if (todos.Count > 0)
        {
            builder.AppendLine("## Left for you to check").AppendLine();
            foreach (var todo in todos) builder.AppendLine($"- {todo}");
            builder.AppendLine();
        }

        if (errors.Count > 0)
        {
            builder.AppendLine("## Remaining build errors").AppendLine();
            foreach (var error in errors.Take(50)) builder.AppendLine($"- {error}");
            builder.AppendLine();
        }

        builder.AppendLine("Run it with `dotnet run` from this folder. Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil.");
        return builder.ToString();
    }
}

/// <summary>Runs the dotnet CLI in a given folder.</summary>
public static class DotnetCli
{
    /// <summary>Runs <c>dotnet {arguments}</c>, reporting each line; cancelling kills the process tree.</summary>
    public static async Task<ProcessResult> RunAsync(string arguments, string workingDirectory, Action<string>? onLine,
        CancellationToken cancellationToken)
    {
        var info = new ProcessStartInfo
        {
            FileName = "dotnet",
            Arguments = arguments,
            WorkingDirectory = workingDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        info.Environment["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1";
        info.Environment["DOTNET_NOLOGO"] = "1";

        var collected = new StringBuilder();
        using var process = new Process { StartInfo = info };
        void OnData(object _, DataReceivedEventArgs e)
        {
            if (e.Data is null) return;
            lock (collected) collected.AppendLine(e.Data);
            onLine?.Invoke(e.Data);
        }
        process.OutputDataReceived += OnData;
        process.ErrorDataReceived += OnData;

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        try
        {
            await process.WaitForExitAsync(cancellationToken);
        }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
            return new ProcessResult(false, -1, collected + "\n(stopped)");
        }

        return new ProcessResult(process.ExitCode == 0, process.ExitCode, collected.ToString());
    }
}
