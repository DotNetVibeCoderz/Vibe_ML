namespace ScienceAppGen.Services.Conversion;

/// <summary>Where one Python module goes in the C# project.</summary>
/// <param name="Analysis">The analysed module.</param>
/// <param name="CSharpPath">Output path relative to the project root, e.g. <c>Pkg/DataLoader.cs</c>.</param>
/// <param name="ClassName">The static class holding the module's functions and globals.</param>
/// <param name="IsEntryPoint">True for the module that becomes <c>Program</c>.</param>
public sealed record PlannedModule(ModuleAnalysis Analysis, string CSharpPath, string ClassName, bool IsEntryPoint)
{
    /// <summary>The Python file's path.</summary>
    public string PythonPath => Analysis.File.RelativePath;

    /// <summary>Python name → C# name for every top-level definition.</summary>
    public IEnumerable<(string Python, string CSharp)> NameMap
        => Analysis.Definitions.Select(d => (d.Name, d.Kind == "class"
            ? CSharpNaming.ToPascalCase(d.Name)
            : $"{ClassName}.{CSharpNaming.ToPascalCase(d.Name)}"));
}

/// <summary>The whole conversion, decided before any LLM call.</summary>
public sealed class ConversionPlan
{
    /// <summary>The scanned Python project.</summary>
    public required PythonProject Project { get; init; }

    /// <summary>C# project name, which is also the root namespace.</summary>
    public required string ProjectName { get; init; }

    /// <summary>Modules in conversion order: every module after the local modules it imports.</summary>
    public required IReadOnlyList<PlannedModule> Modules { get; init; }

    /// <summary>How each third-party package is handled.</summary>
    public required IReadOnlyList<PackageMapping> Packages { get; init; }

    /// <summary>NuGet packages the generated csproj references.</summary>
    public IReadOnlyList<TargetLibrary> NuGetPackages => LibraryMap.PackagesFor(Packages);

    /// <summary>Packages with no counterpart, which the UI warns about before converting.</summary>
    public IEnumerable<PackageMapping> Unsupported => Packages.Where(p => p.IsUnsupported);
}

/// <summary>Builds a <see cref="ConversionPlan"/> from a scanned project.</summary>
/// <remarks>
/// Every name is fixed here, deterministically, and handed to the model as a table. Letting the
/// model choose names per file is what breaks multi-file conversions: <c>load_data</c> becomes
/// <c>LoadData</c> in one file and <c>DataLoader.Load</c> in the file that calls it.
/// </remarks>
public static class ConversionPlanner
{
    /// <summary>Plans a conversion.</summary>
    /// <param name="project">The scanned project.</param>
    /// <param name="projectName">Overrides the derived C# project name.</param>
    public static ConversionPlan Plan(PythonProject project, string? projectName = null)
    {
        var name = PythonProjectScanner.ProjectNameFrom(string.IsNullOrWhiteSpace(projectName) ? project.Name : projectName);
        var analyses = PythonImportAnalyzer.Analyze(project);
        var entry = ChooseEntryPoint(analyses);

        // Class names the project defines; a module whose PascalCase name matches one of them
        // takes a "Module" suffix (model.py defining class Model).
        var classNames = analyses.SelectMany(a => a.Definitions)
            .Where(d => d.Kind == "class")
            .Select(d => CSharpNaming.ToPascalCase(d.Name))
            .ToHashSet(StringComparer.Ordinal);
        classNames.Add(name);

        var taken = new HashSet<string>(StringComparer.Ordinal) { "Program" };
        var planned = new Dictionary<string, PlannedModule>(StringComparer.Ordinal);

        foreach (var analysis in analyses)
        {
            var isEntry = ReferenceEquals(analysis, entry);
            string className, path;

            if (isEntry)
            {
                className = "Program";
                path = "Program.cs";
            }
            else
            {
                var module = analysis.File.ModuleName;
                var last = module.Split('.')[^1];
                var baseName = CSharpNaming.ToPascalCase(last);
                if (baseName.Length == 0 || char.IsDigit(baseName[0])) baseName = "M" + baseName;
                if (classNames.Contains(baseName) || baseName == name) baseName += "Module";

                className = baseName;
                if (!taken.Add(className))
                {
                    // Two packages each with a utils.py: qualify with the package.
                    className = CSharpNaming.ToPascalCase(module.Replace('.', '_'));
                    for (var n = 2; !taken.Add(className); n++) className = baseName + n;
                }

                var folder = string.Join('/', module.Split('.')[..^1].Select(CSharpNaming.ToPascalCase));
                path = (folder.Length > 0 ? folder + "/" : "") + className + ".cs";
            }

            planned[analysis.File.ModuleName] = new PlannedModule(analysis, path, className, isEntry);
        }

        return new ConversionPlan
        {
            Project = project,
            ProjectName = name,
            Modules = TopologicalOrder(planned),
            Packages = LibraryMap.MapAll(analyses),
        };
    }

    /// <summary>
    /// The module that becomes <c>Program</c>: the only one, else <c>main.py</c>/<c>__main__.py</c>,
    /// else one with a main guard, else one nothing else imports.
    /// </summary>
    public static ModuleAnalysis ChooseEntryPoint(IReadOnlyList<ModuleAnalysis> analyses)
    {
        if (analyses.Count == 1) return analyses[0];

        static string Stem(ModuleAnalysis a) => a.File.ModuleName.Split('.')[^1];

        var byName = analyses.FirstOrDefault(a => Stem(a) is "__main__")
                     ?? analyses.FirstOrDefault(a => Stem(a) is "main" && !a.File.ModuleName.Contains('.'))
                     ?? analyses.FirstOrDefault(a => Stem(a) is "main" or "app" or "run");
        if (byName is not null) return byName;

        var imported = analyses.SelectMany(a => a.LocalDependencies).ToHashSet(StringComparer.Ordinal);
        var guarded = analyses.Where(a => a.HasMainGuard).ToList();
        return guarded.FirstOrDefault(a => !imported.Contains(a.File.ModuleName))
               ?? guarded.FirstOrDefault()
               ?? analyses.FirstOrDefault(a => !imported.Contains(a.File.ModuleName))
               ?? analyses[0];
    }

    /// <summary>Dependencies first; a cycle is broken in path order rather than refused.</summary>
    private static IReadOnlyList<PlannedModule> TopologicalOrder(Dictionary<string, PlannedModule> modules)
    {
        var order = new List<PlannedModule>();
        var state = new Dictionary<string, int>(StringComparer.Ordinal);   // 1 visiting, 2 done

        void Visit(string name)
        {
            if (state.TryGetValue(name, out var s) && s > 0) return;   // done, or a cycle back-edge
            state[name] = 1;
            foreach (var dependency in modules[name].Analysis.LocalDependencies.OrderBy(d => d, StringComparer.Ordinal))
                if (modules.ContainsKey(dependency)) Visit(dependency);
            state[name] = 2;
            order.Add(modules[name]);
        }

        foreach (var name in modules.Keys.OrderBy(k => k, StringComparer.Ordinal)) Visit(name);

        // The entry point is converted last: it calls everything else.
        var entry = order.FirstOrDefault(m => m.IsEntryPoint);
        if (entry is not null) { order.Remove(entry); order.Add(entry); }
        return order;
    }
}
