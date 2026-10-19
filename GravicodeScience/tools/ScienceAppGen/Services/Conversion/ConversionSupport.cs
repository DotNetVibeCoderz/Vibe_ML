using System.Text;
using System.Text.RegularExpressions;

namespace ScienceAppGen.Services.Conversion;

/// <summary>One compiler diagnostic from <c>dotnet build</c>.</summary>
/// <param name="File">Path relative to the project root, forward slashes, or "" when the diagnostic has no file.</param>
/// <param name="Line">1-based line, or 0.</param>
/// <param name="Column">1-based column, or 0.</param>
/// <param name="Code">e.g. <c>CS0117</c>.</param>
/// <param name="Message">The message text.</param>
public sealed record BuildDiagnostic(string File, int Line, int Column, string Code, string Message)
{
    /// <inheritdoc />
    public override string ToString() => File.Length == 0 ? $"{Code}: {Message}" : $"{File}({Line},{Column}): {Code}: {Message}";
}

/// <summary>Small, deterministic helpers around the LLM calls.</summary>
public static partial class ConversionSupport
{
    [GeneratedRegex(@"```(?:csharp|cs|c#)?[ \t]*\r?\n(.*?)\r?\n```", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex FencedBlock();

    [GeneratedRegex(@"^(?<file>.+?)\((?<line>\d+),(?<col>\d+)\):\s*error\s+(?<code>[A-Z]+\d+):\s*(?<msg>.*?)(?:\s+\[[^\]]+\])?$", RegexOptions.Multiline)]
    private static partial Regex ErrorLine();

    [GeneratedRegex(@"^\s*(?:error|.*:\s*error)\s+(?<code>(?:CS|NU|MSB|NETSDK)\d+):\s*(?<msg>.*?)(?:\s+\[[^\]]+\])?$", RegexOptions.Multiline)]
    private static partial Regex FilelessError();

    /// <summary>
    /// The C# file in a model reply: the longest fenced block, or the whole reply when it has no
    /// fence and looks like code. Null when nothing usable came back.
    /// </summary>
    /// <remarks>
    /// The longest block rather than the first, because models sometimes show a one-line usage
    /// example before the file itself.
    /// </remarks>
    public static string? ExtractCode(string reply)
    {
        var blocks = FencedBlock().Matches(reply).Select(m => m.Groups[1].Value).ToList();
        if (blocks.Count > 0) return blocks.MaxBy(b => b.Length)!.Trim() + "\n";

        var trimmed = reply.Trim();
        // An unclosed fence: a reply cut off by the token limit still carries most of the file.
        var open = trimmed.IndexOf("```", StringComparison.Ordinal);
        if (open >= 0)
        {
            var afterFence = trimmed.IndexOf('\n', open);
            if (afterFence > 0) trimmed = trimmed[(afterFence + 1)..];
        }

        var looksLikeCode = trimmed.Contains("namespace ", StringComparison.Ordinal)
                            || trimmed.Contains("using ", StringComparison.Ordinal)
                            || trimmed.Contains("class ", StringComparison.Ordinal);
        return looksLikeCode ? trimmed + "\n" : null;
    }

    /// <summary>
    /// Compiler errors from <c>dotnet build</c> output, de-duplicated (MSBuild prints each error
    /// twice: once as it happens and again in the summary) and made project-relative.
    /// </summary>
    public static IReadOnlyList<BuildDiagnostic> ParseBuildErrors(string output, string projectRoot)
    {
        // Windows output ends lines with \r, which stops `$` matching before the "[x.csproj]" suffix.
        output = output.Replace("\r", "");
        var root = Path.GetFullPath(projectRoot).TrimEnd('\\', '/') + Path.DirectorySeparatorChar;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var result = new List<BuildDiagnostic>();

        foreach (Match match in ErrorLine().Matches(output))
        {
            var file = match.Groups["file"].Value.Trim();
            if (Path.IsPathRooted(file) && file.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                file = file[root.Length..];
            file = file.Replace('\\', '/');

            var diagnostic = new BuildDiagnostic(file, int.Parse(match.Groups["line"].Value),
                int.Parse(match.Groups["col"].Value), match.Groups["code"].Value, match.Groups["msg"].Value.Trim());
            if (seen.Add(diagnostic.ToString())) result.Add(diagnostic);
        }

        // Restore and project-level failures have no file position, and would otherwise leave a
        // failed build with "no errors" to repair.
        if (result.Count == 0)
        {
            foreach (Match match in FilelessError().Matches(output))
            {
                var diagnostic = new BuildDiagnostic("", 0, 0, match.Groups["code"].Value, match.Groups["msg"].Value.Trim());
                if (seen.Add(diagnostic.ToString())) result.Add(diagnostic);
            }
        }

        return result;
    }

    private static readonly string[] LibraryPrefixes = ["Gravicode.", "Matplotlib.", "Seaborn.", "Plotly.", "Visual.Net"];

    private static readonly Lazy<(HashSet<string> Namespaces, HashSet<string> TypeNames)> Known = new(() =>
    {
        var assemblies = new[]
        {
            typeof(Gravicode.Science.GraviNum.NdArray).Assembly,
            typeof(Gravicode.Science.GraviFrame.DataFrame).Assembly,
            typeof(Gravicode.Science.GraviLearn.Pipeline).Assembly,
            typeof(Gravicode.Science.SciNet.Optimize).Assembly,
            typeof(Matplotlib.Net.Figure).Assembly,
            typeof(Seaborn.Net.Sns).Assembly,
            typeof(Plotly.Net.Chart).Assembly,
            typeof(Visual.Net.Core.Colors.Color).Assembly,
        };
        var types = assemblies.SelectMany(a => a.GetExportedTypes()).ToList();
        var namespaces = types.Select(t => t.Namespace).OfType<string>().ToHashSet(StringComparer.Ordinal);
        // The libraries ScienceAppGen does not reference are known from the library map.
        foreach (var target in new[] { LibraryMap.GraviText, LibraryMap.GraviGraph, LibraryMap.GraviProb })
            namespaces.UnionWith(target.Usings);
        return (namespaces, types.Where(t => !t.IsNested).Select(t => t.Name).ToHashSet(StringComparer.Ordinal));
    });

    /// <summary>
    /// Fixes the <c>using</c> block deterministically: adds every namespace in
    /// <paramref name="required"/> that is missing, and removes directives that name a Gravicode or
    /// Visual.Net namespace that does not exist (<c>using Visual.Net.Matplotlib;</c>) or a library
    /// type rather than a namespace (<c>using Stats;</c>). Each of those otherwise costs a repair round.
    /// </summary>
    public static string NormalizeUsings(string csharp, IEnumerable<string> required)
    {
        var (namespaces, typeNames) = Known.Value;
        var lines = csharp.Replace("\r\n", "\n").Split('\n').ToList();
        var present = new HashSet<string>(StringComparer.Ordinal);
        var lastUsing = -1;

        for (var i = 0; i < lines.Count; i++)
        {
            var line = lines[i].Trim();
            if (line.Length == 0 || line.StartsWith("//", StringComparison.Ordinal) || line.StartsWith('#')) continue;
            if (!line.StartsWith("using ", StringComparison.Ordinal) || !line.EndsWith(';')) break;   // end of the header

            var name = line["using ".Length..^1].Trim();
            if (name.StartsWith("static ", StringComparison.Ordinal) || name.Contains('='))
            {
                lastUsing = i;
                continue;
            }

            var bogusLibrary = LibraryPrefixes.Any(p => name.StartsWith(p, StringComparison.Ordinal)) && !namespaces.Contains(name);
            var typeAsNamespace = !name.Contains('.') && typeNames.Contains(name);
            if (bogusLibrary || typeAsNamespace)
            {
                lines.RemoveAt(i--);
                continue;
            }

            present.Add(name);
            lastUsing = i;
        }

        var missing = required.Where(r => !present.Contains(r)).Select(r => $"using {r};").ToList();
        if (missing.Count > 0) lines.InsertRange(lastUsing + 1, missing);
        return string.Join("\n", lines);
    }

    private static readonly Lazy<Dictionary<string, HashSet<string>>> TypesByNamespace = new(() =>
        new[]
        {
            typeof(Gravicode.Science.GraviNum.NdArray).Assembly,
            typeof(Gravicode.Science.GraviFrame.DataFrame).Assembly,
            typeof(Gravicode.Science.GraviLearn.Pipeline).Assembly,
            typeof(Gravicode.Science.SciNet.Optimize).Assembly,
            typeof(Matplotlib.Net.Figure).Assembly,
            typeof(Seaborn.Net.Sns).Assembly,
            typeof(Plotly.Net.Chart).Assembly,
            typeof(Visual.Net.Core.Colors.Color).Assembly,
        }
        .SelectMany(a => a.GetExportedTypes())
        .Where(t => !t.IsNested && t.Namespace is not null)
        .GroupBy(t => t.Namespace!)
        .ToDictionary(g => g.Key, g => g.Select(t => t.Name.Split('`')[0]).ToHashSet(StringComparer.Ordinal), StringComparer.Ordinal));

    [GeneratedRegex(@"^[ \t]*(?:(?:public|internal|private|protected|file|static|sealed|abstract|partial|readonly|unsafe|ref)\s+)*(?:class|struct|record(?:\s+(?:class|struct))?|interface|enum)\s+(?<name>[A-Za-z_]\w*)", RegexOptions.Multiline)]
    private static partial Regex TypeDeclaration();

    /// <summary>
    /// Types a converted file declares that already exist in the namespaces it imports - a stub
    /// <c>class DataFrame</c> compiles, shadows the real one for the whole project namespace, and
    /// breaks every other file's calls into it. Reported as synthetic <c>CONV001</c> errors so the
    /// file goes back for repair even when the build passes.
    /// </summary>
    /// <param name="csharpPath">The file's project-relative path.</param>
    /// <param name="csharp">Its source.</param>
    /// <param name="usings">The library namespaces it imports.</param>
    /// <param name="allowed">Names the project itself owns (its Python classes and module classes).</param>
    public static IReadOnlyList<BuildDiagnostic> FindShadowedLibraryTypes(string csharpPath, string csharp,
        IEnumerable<string> usings, IReadOnlySet<string> allowed)
    {
        var library = usings
            .Where(TypesByNamespace.Value.ContainsKey)
            .SelectMany(ns => TypesByNamespace.Value[ns].Select(name => (name, ns)))
            .GroupBy(x => x.name, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.First().ns, StringComparer.Ordinal);

        var text = csharp.Replace("\r\n", "\n");
        var result = new List<BuildDiagnostic>();
        foreach (Match match in TypeDeclaration().Matches(text))
        {
            var name = match.Groups["name"].Value;
            if (allowed.Contains(name) || !library.TryGetValue(name, out var ns)) continue;
            var line = text[..match.Index].Count(c => c == '\n') + 1;
            result.Add(new BuildDiagnostic(csharpPath, line, 1, "CONV001",
                $"Do not declare '{name}': it is {ns}.{name}, imported by this file's usings. Delete this declaration and use the library type."));
        }
        return result;
    }

    /// <summary>
    /// Drops library <c>using</c> directives whose namespace contributes no type name the file
    /// mentions. <see cref="NormalizeUsings"/> adds every namespace a module might need, which
    /// leaves a dozen unused lines at the top of every file; this runs after a successful build,
    /// and the caller rebuilds and reverts if the pruned version does not compile (a namespace
    /// that only contributes extension methods is the case this heuristic cannot see).
    /// </summary>
    /// <remarks>
    /// Each library's root namespace is kept regardless - that is where the extension methods
    /// live - and namespaces this process cannot inspect are never touched.
    /// </remarks>
    public static string PruneUnusedUsings(string csharp, IEnumerable<string> keep)
    {
        var keepSet = keep.ToHashSet(StringComparer.Ordinal);
        var body = csharp.Replace("\r\n", "\n");
        // Strings are deliberately not stripped: interpolation holes are code, and counting a name
        // that only appears in text merely keeps a using, which is the safe direction.
        var identifiers = Regex.Matches(
                string.Join("\n", body.Split('\n').Where(l => !l.TrimStart().StartsWith("using ", StringComparison.Ordinal))),
                @"[A-Za-z_]\w*")
            .Select(m => m.Value)
            .ToHashSet(StringComparer.Ordinal);

        var lines = body.Split('\n').ToList();
        for (var i = 0; i < lines.Count; i++)
        {
            var line = lines[i].Trim();
            if (line.Length == 0 || line.StartsWith("//", StringComparison.Ordinal)) continue;
            if (!line.StartsWith("using ", StringComparison.Ordinal) || !line.EndsWith(';')) break;

            var name = line["using ".Length..^1].Trim();
            if (keepSet.Contains(name) || !TypesByNamespace.Value.TryGetValue(name, out var types)) continue;
            if (!types.Any(identifiers.Contains)) lines.RemoveAt(i--);
        }
        return string.Join("\n", lines);
    }

    [GeneratedRegex(@"^Unhandled exception\.\s*(?<type>[\w.`]+):\s*(?<msg>.*)$", RegexOptions.Multiline)]
    private static partial Regex UnhandledException();

    [GeneratedRegex(@"^\s*at (?<member>.+?) in (?<file>.+?):line (?<line>\d+)\s*$", RegexOptions.Multiline)]
    private static partial Regex StackFrame();

    /// <summary>
    /// Turns an unhandled exception in the converted program into diagnostics on the converted
    /// files its stack passes through - the throw site first, then the first caller in a different
    /// file, because the fix belongs to one or the other (a wrong column name is read in the
    /// caller; a wrong shape may be produced in the callee). Empty when the failure is not an
    /// exception in our files (a timeout, a missing data file the user must supply, ...).
    /// </summary>
    public static IReadOnlyList<BuildDiagnostic> ParseRuntimeFailure(string output, string projectRoot, IEnumerable<string> convertedFiles)
    {
        output = output.Replace("\r", "");
        var exception = UnhandledException().Match(output);
        if (!exception.Success) return [];

        var root = Path.GetFullPath(projectRoot).TrimEnd('\\', '/') + Path.DirectorySeparatorChar;
        var files = convertedFiles.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var frames = StackFrame().Matches(output)
            .Select(m =>
            {
                var file = m.Groups["file"].Value.Trim();
                if (file.StartsWith(root, StringComparison.OrdinalIgnoreCase)) file = file[root.Length..];
                return (File: file.Replace('\\', '/'), Line: int.Parse(m.Groups["line"].Value), Member: m.Groups["member"].Value);
            })
            .Where(f => files.Contains(f.File))
            .ToList();
        if (frames.Count == 0) return [];

        var stack = string.Join(" <- ", frames.Select(f => $"{f.Member.Split('(')[0]} ({f.File}:{f.Line})"));
        var headline = $"{exception.Groups["type"].Value}: {exception.Groups["msg"].Value.Trim()}";
        var result = new List<BuildDiagnostic>
        {
            new(frames[0].File, frames[0].Line, 0, "RUNTIME", $"Unhandled exception at run time - {headline}. Stack: {stack}"),
        };
        var caller = frames.FirstOrDefault(f => !string.Equals(f.File, frames[0].File, StringComparison.OrdinalIgnoreCase));
        if (caller.File is not null)
            result.Add(new(caller.File, caller.Line, 0, "RUNTIME", $"This call led to an exception in {frames[0].File}:{frames[0].Line} - {headline}. Stack: {stack}"));
        return result;
    }

    /// <summary>
    /// The public surface of a converted file - type and member declarations, bodies removed - so
    /// files converted later call what was actually written rather than what was planned.
    /// </summary>
    public static string PublicSurface(string csharp)
    {
        var builder = new StringBuilder();
        foreach (var raw in csharp.Replace("\r\n", "\n").Split('\n'))
        {
            var line = raw.Trim();
            if (!line.StartsWith("public ", StringComparison.Ordinal) && !line.StartsWith("internal ", StringComparison.Ordinal)) continue;

            // Drop bodies: "public static int F(int x) => x * 2;" and "public int P { get; set; } = 3;" keep their heads.
            var arrow = line.IndexOf(" => ", StringComparison.Ordinal);
            if (arrow > 0) line = line[..arrow];
            var brace = line.IndexOf('{');
            if (brace > 0 && !line.Contains("get;", StringComparison.Ordinal) && !line.Contains("get ", StringComparison.Ordinal))
                line = line[..brace].TrimEnd();
            var equals = line.IndexOf(" = ", StringComparison.Ordinal);
            if (equals > 0 && !line.Contains('(', StringComparison.Ordinal)) line = line[..equals];

            builder.AppendLine(line.TrimEnd(';', ' '));
        }
        return builder.ToString();
    }

    /// <summary>The csproj for a converted project.</summary>
    /// <param name="plan">The conversion plan; supplies the packages and data files.</param>
    public static string Csproj(ConversionPlan plan)
    {
        var packages = string.Join("\n", plan.NuGetPackages.Select(p =>
            $"    <PackageReference Include=\"{p.PackageId}\" Version=\"{p.Version}\" />"));

        var data = plan.Project.DataFiles.Count == 0
            ? ""
            : """

              <ItemGroup>
                <!-- The Python program's data files, at the same relative paths, so its relative file
                     paths keep working both from the project folder (dotnet run) and from the output. -->
            """ + "\n" + string.Join("\n", plan.Project.DataFiles.Select(d =>
                  $"    <None Update=\"{d.RelativePath.Replace('/', '\\')}\" CopyToOutputDirectory=\"PreserveNewest\" />"))
              + "\n  </ItemGroup>\n";

        return $$"""
            <Project Sdk="Microsoft.NET.Sdk">

              <!--
                Converted from Python by ScienceAppGen's Python to C# converter.
                Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil.
              -->

              <PropertyGroup>
                <OutputType>Exe</OutputType>
                <TargetFramework>net10.0</TargetFramework>
                <Nullable>enable</Nullable>
                <ImplicitUsings>enable</ImplicitUsings>
                <RootNamespace>{{plan.ProjectName}}</RootNamespace>
                <!-- Converted numeric code is easier to read without nullable noise on every array. -->
                <NoWarn>$(NoWarn);CS8600;CS8602;CS8604;CS8618;CS8625</NoWarn>
              </PropertyGroup>

              <ItemGroup>
            {{packages}}
              </ItemGroup>
            {{data}}
            </Project>
            """;
    }
}
