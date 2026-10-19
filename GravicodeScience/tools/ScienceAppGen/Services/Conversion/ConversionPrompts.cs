using System.Text;

namespace ScienceAppGen.Services.Conversion;

/// <summary>The system and user prompts for converting and repairing files.</summary>
public static class ConversionPrompts
{
    /// <summary>The converter's instructions. Fixed across files so the model applies one style.</summary>
    public const string System = """
        You convert Python data-science programs to C# 13 on .NET 10. The target libraries are
        Gravicode.Science (GraviNum = numpy, GraviFrame = pandas, GraviLearn = scikit-learn,
        SciNet = scipy, GraviText, GraviGraph, GraviProb) and Visual.Net for charts
        (Matplotlib.Net = matplotlib, Seaborn.Net = seaborn, Plotly.Net = plotly).

        Output: exactly ONE fenced ```csharp block holding the COMPLETE file, nothing omitted, no
        "..." placeholders. A short note may follow the block; nothing may precede it.

        Rules:
        - Use only APIs shown in the reference you are given, the .NET base class library, or plain
          C#. Never invent a Gravicode or Visual.Net method; if the reference has no equivalent,
          write it out by hand (loops, LINQ, Math) and put a comment starting
          `// TODO(convert):` saying what differs.
        - File-scoped namespace = the project namespace you are given. All files share it.
        - Use the class and member names in the naming table EXACTLY - other files are being
          converted with the same table and will call these names. Module-level functions become
          `public static` methods of the module's static class; module-level variables become
          `public static` fields or properties; Python classes become top-level C# classes in the
          same file with PascalCase members.
        - The module marked as the entry point becomes `public static class Program` with
          `public static void Main(string[] args)` (or `async Task Main`) holding the
          `if __name__ == "__main__":` block or the script's top-level statements.
        - The reference is exact. Do not hedge against it: no try/catch "in case the API differs",
          no reflection, no alternative code paths for a different library version.
        - Explicit, static types. No `dynamic`. Prefer `double[]` and GraviNum `NdArray` for numeric
          data, and `DataFrame` where pandas was used.
        - Keep behaviour and printed output: same computations, same order, same labels in prints,
          and NO prints the Python does not have. Match Python's format specs exactly:
          f"{x:.3f}" -> {x:F3}; f"{s:12s}" and f"{s:<12}" LEFT-align -> {s,-12}; f"{n:>8}" and
          a bare width on a number right-align -> {n,8}; f"{x:8.2f}" -> {x,8:F2}; print(a, b)
          joins with one space.
        - Keep file paths as they are in the Python source (relative paths stay relative).
        - Never declare a type the libraries already provide - DataFrame, Series, NdArray, Figure,
          Axes, Pipeline and the like come from the usings, even in a module whose Python file
          never imported them (it receives them from other modules). A file that declares one is
          sent back.
        - Never replace a target library with your own substitute: no hand-written Figure/Axes
          classes, no System.Drawing (it does not exist on .NET 10 outside Windows), no stub types
          that imitate GraviFrame or SciNet. If one call has no equivalent, port that call by hand -
          not the library.
        - Charts: plt.show() and fig.show() have no window here. Save to a file instead (PNG for
          matplotlib/seaborn, HTML for plotly) next to the program, named after the chart, and print
          the path - only in that case; a savefig() the Python already has needs no extra print.
        - Random numbers: use the library's seeded generator where the Python code seeds one. Exact
          random sequences will differ from numpy's; that is expected and needs no comment.
        - Python features: list/dict comprehensions -> LINQ or loops; tuples -> C# tuples or records;
          `with open(...)` -> File APIs / using; f-strings -> interpolation; argparse -> args parsing
          with sensible defaults; exceptions -> C# exceptions.
        """;

    /// <summary>The user prompt for converting one module.</summary>
    /// <param name="plan">The conversion plan.</param>
    /// <param name="module">The module to convert.</param>
    /// <param name="reference">API reference for this module.</param>
    /// <param name="convertedSurfaces">C# path → public surface of every file converted so far.</param>
    public static string Convert(ConversionPlan plan, PlannedModule module, string reference,
        IReadOnlyDictionary<string, string> convertedSurfaces)
    {
        var builder = new StringBuilder();
        AppendProjectContext(builder, plan, module, convertedSurfaces);

        builder.AppendLine("## API reference for this file").AppendLine();
        builder.AppendLine(reference.Length > 0 ? reference : "(only the .NET base class library is needed)").AppendLine();

        builder.AppendLine($"## Python source: {module.PythonPath}").AppendLine();
        if (module.Analysis.File.IsNotebook)
            builder.AppendLine("(A Jupyter notebook; its code cells are joined below. Treat it as one script.)").AppendLine();
        builder.AppendLine("```python").AppendLine(module.Analysis.File.Source.TrimEnd()).AppendLine("```").AppendLine();

        builder.AppendLine($"Convert it to `{module.CSharpPath}`. Reply with the complete file in one ```csharp block.");
        return builder.ToString();
    }

    /// <summary>The user prompt for repairing one file that does not compile.</summary>
    /// <param name="plan">The conversion plan.</param>
    /// <param name="module">The module whose C# file failed.</param>
    /// <param name="csharp">The current C# file.</param>
    /// <param name="errors">Its compiler errors.</param>
    /// <param name="reference">API reference for this module.</param>
    /// <param name="convertedSurfaces">C# path → public surface of every converted file.</param>
    /// <param name="programOutput">When the project built but the program crashed, what it printed.</param>
    public static string Repair(ConversionPlan plan, PlannedModule module, string csharp,
        IReadOnlyList<BuildDiagnostic> errors, string reference, IReadOnlyDictionary<string, string> convertedSurfaces,
        string? programOutput = null)
    {
        var builder = new StringBuilder();
        AppendProjectContext(builder, plan, module, convertedSurfaces);

        builder.AppendLine("## API reference for this file").AppendLine();
        builder.AppendLine(reference.Length > 0 ? reference : "(only the .NET base class library is needed)").AppendLine();

        builder.AppendLine($"## The original Python ({module.PythonPath})").AppendLine();
        builder.AppendLine("```python").AppendLine(module.Analysis.File.Source.TrimEnd()).AppendLine("```").AppendLine();

        var runtime = errors.Any(e => e.Code == "RUNTIME");
        builder.AppendLine(runtime
            ? $"## The current C# ({module.CSharpPath}), which compiles but crashes at run time"
            : $"## The current C# ({module.CSharpPath}), which does not compile").AppendLine();
        var lines = csharp.Replace("\r\n", "\n").Split('\n');
        builder.AppendLine("```csharp");
        for (var i = 0; i < lines.Length; i++) builder.Append((i + 1).ToString().PadLeft(4)).Append("| ").AppendLine(lines[i]);
        builder.AppendLine("```").AppendLine();

        builder.AppendLine(runtime ? "## Run-time failure" : "## Compiler errors").AppendLine();
        foreach (var error in errors.Take(40)) builder.Append("- ").AppendLine(error.ToString());
        if (errors.Count > 40) builder.AppendLine($"- ... and {errors.Count - 40} more of the same kinds");
        builder.AppendLine();

        if (runtime)
        {
            var printed = (programOutput ?? "").Replace("\r", "").Split('\n')
                .Where(l => !l.TrimStart().StartsWith("at ", StringComparison.Ordinal))
                .TakeLast(20);
            builder.AppendLine("The Python original runs without error, so the C# must too. Find the cause - a column name, "
                               + "an index, a shape or a type that differs from what the library really returns - using the "
                               + "API reference and the other files' signatures. What the program printed before failing:").AppendLine();
            builder.AppendLine("```").AppendLine(string.Join('\n', printed).TrimEnd()).AppendLine("```").AppendLine();
        }

        builder.AppendLine("""
            Fix every error. The line numbers above refer to the numbered listing. If an error says a
            member does not exist, use the API reference for the real name, or write the logic by hand
            with a `// TODO(convert):` note - do not guess another name. Keep everything that already
            compiles unchanged. Reply with the complete corrected file in one ```csharp block, without
            the line-number prefixes.
            """);
        return builder.ToString();
    }

    private static void AppendProjectContext(StringBuilder builder, ConversionPlan plan, PlannedModule module,
        IReadOnlyDictionary<string, string> convertedSurfaces)
    {
        builder.AppendLine($"Project: `{plan.ProjectName}` (namespace `{plan.ProjectName}`), {plan.Modules.Count} Python module(s).");
        builder.AppendLine($"This file: `{module.PythonPath}` -> `{module.CSharpPath}`"
                           + (module.IsEntryPoint ? " - the ENTRY POINT (class Program with Main)." : $", static class `{module.ClassName}`."));
        builder.AppendLine();

        var usings = LibraryMap.UsingsFor(module.Analysis, plan.Packages);
        if (usings.Count > 0)
        {
            builder.AppendLine("## Usings").AppendLine();
            builder.AppendLine("These are added to the file automatically, and are the only Gravicode / Visual.Net namespaces "
                               + "that exist for it - do not write other ones (System.* usings are yours to add):");
            foreach (var u in usings) builder.AppendLine($"- `using {u};`");
            builder.AppendLine();
        }

        builder.AppendLine("## Naming table (use exactly)").AppendLine();
        foreach (var other in plan.Modules)
        {
            builder.AppendLine($"- `{other.PythonPath}` -> `{other.CSharpPath}`, static class `{other.ClassName}`"
                               + (other.IsEntryPoint ? " (entry point)" : ""));
            foreach (var definition in other.Analysis.Definitions)
            {
                var name = CSharpNaming.ToPascalCase(definition.Name);
                builder.AppendLine(definition.Kind == "class"
                    ? $"  - class `{definition.Name}` -> top-level class `{name}` (in this file, NOT nested in `{other.ClassName}`)"
                    : $"  - `{definition.Name}` -> `{other.ClassName}.{name}`");
            }
        }
        builder.AppendLine();

        var dependencies = module.Analysis.LocalDependencies.ToHashSet(StringComparer.Ordinal);
        var relevant = plan.Modules
            .Where(m => !ReferenceEquals(m, module) && convertedSurfaces.ContainsKey(m.CSharpPath))
            .OrderByDescending(m => dependencies.Contains(m.Analysis.File.ModuleName))
            .ToList();
        if (relevant.Count > 0)
        {
            builder.AppendLine("## Already converted files - call these exact signatures").AppendLine();
            foreach (var other in relevant)
            {
                builder.AppendLine($"`{other.CSharpPath}`:").AppendLine("```csharp");
                builder.AppendLine(convertedSurfaces[other.CSharpPath].TrimEnd()).AppendLine("```");
            }
            builder.AppendLine();
        }

        var unsupported = plan.Unsupported.ToList();
        if (unsupported.Count > 0)
        {
            builder.AppendLine("## Packages with no .NET counterpart here").AppendLine();
            foreach (var package in unsupported) builder.AppendLine($"- {package.Package}: {package.Note}");
            builder.AppendLine();
        }
    }
}
