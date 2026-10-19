using System.Text;
using System.Text.RegularExpressions;

namespace ScienceAppGen.Services.Conversion;

/// <summary>Python-to-C# identifier conventions shared by the planner and the prompts.</summary>
public static class CSharpNaming
{
    private static readonly HashSet<string> Keywords = new(StringComparer.Ordinal)
    {
        "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked", "class",
        "const", "continue", "decimal", "default", "delegate", "do", "double", "else", "enum", "event",
        "explicit", "extern", "false", "finally", "fixed", "float", "for", "foreach", "goto", "if",
        "implicit", "in", "int", "interface", "internal", "is", "lock", "long", "namespace", "new",
        "null", "object", "operator", "out", "override", "params", "private", "protected", "public",
        "readonly", "ref", "return", "sbyte", "sealed", "short", "sizeof", "stackalloc", "static",
        "string", "struct", "switch", "this", "throw", "true", "try", "typeof", "uint", "ulong",
        "unchecked", "unsafe", "ushort", "using", "virtual", "void", "volatile", "while",
    };

    /// <summary><c>load_sales_data</c> → <c>LoadSalesData</c>; <c>__init__</c> → <c>Init</c>; <c>plotXY</c> → <c>PlotXY</c>.</summary>
    public static string ToPascalCase(string identifier)
    {
        var builder = new StringBuilder(identifier.Length);
        var upperNext = true;
        foreach (var c in identifier)
        {
            if (!char.IsLetterOrDigit(c)) { upperNext = true; continue; }
            builder.Append(upperNext ? char.ToUpperInvariant(c) : c);
            upperNext = false;
        }
        return builder.ToString();
    }

    /// <summary>A C# identifier for a Python one, escaping keywords with <c>@</c>.</summary>
    public static string ToIdentifier(string name) => Keywords.Contains(name) ? "@" + name : name;
}

/// <summary>An <c>import</c> or <c>from ... import</c> in a module.</summary>
/// <param name="Module">The module imported from, fully dotted (<c>scipy.signal</c>), or the resolved local module.</param>
/// <param name="Alias">The name it is bound to locally, or null for a bare <c>import a.b</c>.</param>
/// <param name="ImportedName">For <c>from X import Y</c>, the <c>Y</c>.</param>
/// <param name="IsLocal">True when the import refers to another file of the project.</param>
public sealed record PythonImport(string Module, string? Alias, string? ImportedName, bool IsLocal);

/// <summary>A top-level <c>def</c> or <c>class</c> in a module.</summary>
/// <param name="Kind">"def" or "class".</param>
/// <param name="Name">Python name.</param>
/// <param name="Signature">The definition line, e.g. <c>def load(path, sep=",")</c>.</param>
public sealed record PythonDefinition(string Kind, string Name, string Signature);

/// <summary>What one module imports, uses and defines.</summary>
public sealed class ModuleAnalysis
{
    /// <summary>The analysed file.</summary>
    public required PythonSourceFile File { get; init; }

    /// <summary>Every import statement.</summary>
    public required IReadOnlyList<PythonImport> Imports { get; init; }

    /// <summary>
    /// Fully qualified names used from third-party modules, such as <c>scipy.signal.butter</c> or
    /// <c>matplotlib.pyplot.savefig</c> - what decides which API reference the prompt carries.
    /// </summary>
    public required IReadOnlySet<string> UsedSymbols { get; init; }

    /// <summary>Top-level functions and classes.</summary>
    public required IReadOnlyList<PythonDefinition> Definitions { get; init; }

    /// <summary>True when the module has an <c>if __name__ == "__main__":</c> block.</summary>
    public bool HasMainGuard { get; init; }

    /// <summary>Local modules this one imports.</summary>
    public IEnumerable<string> LocalDependencies => Imports.Where(i => i.IsLocal).Select(i => i.Module).Distinct();

    /// <summary>Top-level third-party packages, e.g. <c>numpy</c>, <c>scipy</c>.</summary>
    public IEnumerable<string> ExternalPackages
        => Imports.Where(i => !i.IsLocal).Select(i => i.Module.Split('.')[0]).Distinct(StringComparer.Ordinal);
}

/// <summary>
/// Reads imports, used names and definitions out of Python source with line-level patterns.
/// </summary>
/// <remarks>
/// This is not a Python parser and does not need to be: its output only chooses which API
/// reference goes into a prompt and how modules are named, and the LLM reads the real source.
/// Two things it must get right are handled explicitly - triple-quoted strings (a docstring that
/// says "import numpy" is not an import) and parenthesised multi-line <c>from</c> imports.
/// </remarks>
public static partial class PythonImportAnalyzer
{
    [GeneratedRegex(@"^import\s+(.+)$")]
    private static partial Regex ImportLine();

    [GeneratedRegex(@"^from\s+(\.*[\w.]*)\s+import\s+(.+)$")]
    private static partial Regex FromLine();

    [GeneratedRegex(@"^(async\s+def|def|class)\s+(\w+)\s*(\([^:]*\))?\s*(->\s*[^:]+)?:")]
    private static partial Regex DefinitionLine();

    [GeneratedRegex(@"^if\s+__name__\s*==\s*['""]__main__['""]\s*:", RegexOptions.Multiline)]
    private static partial Regex MainGuard();

    /// <summary>Analyses every module, resolving which imports are local to the project.</summary>
    public static IReadOnlyList<ModuleAnalysis> Analyze(PythonProject project)
    {
        var localModules = project.Files.Select(f => f.ModuleName).ToHashSet(StringComparer.Ordinal);
        return project.Files.Select(f => Analyze(f, localModules)).ToList();
    }

    /// <summary>Analyses one module.</summary>
    /// <param name="file">The module.</param>
    /// <param name="localModules">Dotted names of the project's own modules.</param>
    public static ModuleAnalysis Analyze(PythonSourceFile file, IReadOnlySet<string> localModules)
    {
        var lines = LogicalLines(file.Source);
        var imports = new List<PythonImport>();
        var definitions = new List<PythonDefinition>();
        // On the raw source: the string-stripped copy has blanked out "__main__".
        var mainGuard = MainGuard().IsMatch(file.Source);
        var package = file.ModuleName.Contains('.') ? file.ModuleName[..file.ModuleName.LastIndexOf('.')] : "";

        foreach (var (line, indented) in lines)
        {
            // Imports inside functions count too (a lazy "import matplotlib" is still a dependency).
            var import = ImportLine().Match(line);
            if (import.Success)
            {
                foreach (var part in SplitNames(import.Groups[1].Value))
                {
                    var (name, alias) = SplitAlias(part);
                    var local = ResolveLocal(name, package, localModules);
                    imports.Add(new PythonImport(local ?? name, alias, null, local is not null));
                }
                continue;
            }

            var from = FromLine().Match(line);
            if (from.Success)
            {
                var module = from.Groups[1].Value;
                foreach (var part in SplitNames(from.Groups[2].Value))
                {
                    var (name, alias) = SplitAlias(part);
                    if (name == "*") { imports.Add(Resolve(module, null, "*", package, localModules)); continue; }
                    imports.Add(Resolve(module, alias ?? name, name, package, localModules));
                }
                continue;
            }

            if (!indented)
            {
                var definition = DefinitionLine().Match(line);
                if (definition.Success)
                {
                    var kind = definition.Groups[1].Value.EndsWith("def", StringComparison.Ordinal) ? "def" : "class";
                    definitions.Add(new PythonDefinition(kind, definition.Groups[2].Value, line.TrimEnd(':').Trim()));
                }
            }
        }

        return new ModuleAnalysis
        {
            File = file,
            Imports = imports,
            UsedSymbols = CollectUsedSymbols(file.Source, imports),
            Definitions = definitions,
            HasMainGuard = mainGuard,
        };
    }

    private static PythonImport Resolve(string module, string? alias, string name, string package, IReadOnlySet<string> localModules)
    {
        // "from . import utils" and "from pkg import utils" may name a local module rather than a symbol.
        var asModule = module.Trim('.').Length == 0 ? name : module.TrimStart('.') + "." + name;
        var localModule = ResolveLocal(asModule, package, localModules, relative: module.StartsWith('.'));
        if (localModule is not null) return new PythonImport(localModule, alias, null, true);

        var local = ResolveLocal(module, package, localModules, relative: module.StartsWith('.'));
        return local is not null
            ? new PythonImport(local, alias, name, true)
            : new PythonImport(module, alias, name, false);
    }

    private static string? ResolveLocal(string module, string package, IReadOnlySet<string> localModules, bool relative = false)
    {
        var trimmed = module.TrimStart('.');
        if (trimmed.Length == 0) return null;
        if (relative && package.Length > 0 && localModules.Contains(package + "." + trimmed)) return package + "." + trimmed;
        if (localModules.Contains(trimmed)) return trimmed;
        // Sibling import inside a package ("import utils" from pkg/main.py, Python 2 style or sys.path tweaks).
        if (package.Length > 0 && localModules.Contains(package + "." + trimmed)) return package + "." + trimmed;
        return null;
    }

    /// <summary>
    /// Finds <c>alias.attr.attr</c> chains for every external import, and records names imported
    /// directly with <c>from</c>, as fully qualified dotted names.
    /// </summary>
    private static HashSet<string> CollectUsedSymbols(string source, IReadOnlyList<PythonImport> imports)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        var code = StripStringsAndComments(source);

        foreach (var import in imports.Where(i => !i.IsLocal))
        {
            if (import.ImportedName is not null)
            {
                if (import.ImportedName == "*") continue;
                var qualified = import.Module + "." + import.ImportedName;
                used.Add(qualified);
                // "from scipy import signal" then "signal.butter(...)": follow the alias.
                if (import.Alias is not null) AddChains(code, import.Alias, qualified, used);
                continue;
            }

            used.Add(import.Module);
            AddChains(code, import.Alias ?? import.Module, import.Module, used);
        }

        return used;
    }

    private static void AddChains(string code, string localName, string qualified, HashSet<string> used)
    {
        var pattern = new Regex(@"(?<![\w.])" + Regex.Escape(localName) + @"((?:\.[A-Za-z_]\w*)+)");
        foreach (Match match in pattern.Matches(code))
            used.Add(qualified + match.Groups[1].Value);
    }

    /// <summary>Joins bracket continuations and drops triple-quoted strings and comments.</summary>
    private static List<(string Line, bool Indented)> LogicalLines(string source)
    {
        var result = new List<(string, bool)>();
        var clean = StripStringsAndComments(source);
        var pending = new StringBuilder();
        var depth = 0;
        var indented = false;

        foreach (var raw in clean.Split('\n'))
        {
            if (pending.Length == 0)
            {
                if (raw.Trim().Length == 0) continue;
                indented = raw.Length > 0 && char.IsWhiteSpace(raw[0]);
            }

            var continued = raw.TrimEnd().EndsWith('\\');
            pending.Append(continued ? raw.TrimEnd()[..^1] : raw).Append(' ');
            foreach (var c in raw)
            {
                if (c is '(' or '[' or '{') depth++;
                else if (c is ')' or ']' or '}') depth = Math.Max(0, depth - 1);
            }

            if (depth > 0 || continued) continue;
            result.Add((Regex.Replace(pending.ToString().Trim(), @"\s+", " "), indented));
            pending.Clear();
        }

        if (pending.Length > 0) result.Add((pending.ToString().Trim(), indented));
        return result;
    }

    /// <summary>
    /// Replaces string literal contents and comments with spaces, keeping line breaks, so neither
    /// a docstring nor a commented-out line is read as code.
    /// </summary>
    public static string StripStringsAndComments(string source)
    {
        var output = new StringBuilder(source.Length);
        var i = 0;
        while (i < source.Length)
        {
            var c = source[i];
            if (c == '#')
            {
                while (i < source.Length && source[i] != '\n') { output.Append(' '); i++; }
                continue;
            }

            if (c is '"' or '\'')
            {
                var triple = i + 2 < source.Length && source[i + 1] == c && source[i + 2] == c;
                var quote = triple ? new string(c, 3) : c.ToString();
                output.Append(quote);
                i += quote.Length;
                while (i < source.Length)
                {
                    if (source[i] == '\\' && i + 1 < source.Length) { output.Append("  "); i += 2; continue; }
                    if (string.CompareOrdinal(source, i, quote, 0, quote.Length) == 0) { output.Append(quote); i += quote.Length; break; }
                    if (!triple && source[i] == '\n') break;   // unterminated: let the line end it
                    output.Append(source[i] == '\n' ? '\n' : ' ');
                    i++;
                }
                continue;
            }

            output.Append(c);
            i++;
        }
        return output.ToString();
    }

    private static IEnumerable<string> SplitNames(string list)
        => list.Trim().TrimStart('(').TrimEnd(')').Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    private static (string Name, string? Alias) SplitAlias(string part)
    {
        var pieces = part.Split(" as ", 2, StringSplitOptions.TrimEntries);
        return (pieces[0].Trim(), pieces.Length > 1 ? pieces[1].Trim() : null);
    }
}
