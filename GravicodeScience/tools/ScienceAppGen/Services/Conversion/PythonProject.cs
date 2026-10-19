using System.Text;
using System.Text.Json;

namespace ScienceAppGen.Services.Conversion;

/// <summary>One Python module of the project being converted.</summary>
/// <param name="RelativePath">Path relative to the project root, with forward slashes.</param>
/// <param name="Source">The module's Python source. For a notebook, its code cells joined.</param>
/// <param name="IsNotebook">True when the source came from an <c>.ipynb</c> file.</param>
public sealed record PythonSourceFile(string RelativePath, string Source, bool IsNotebook = false)
{
    /// <summary>The dotted module name, e.g. <c>pkg.data_loader</c> for <c>pkg/data_loader.py</c>.</summary>
    public string ModuleName
    {
        get
        {
            var withoutExtension = RelativePath[..^Path.GetExtension(RelativePath).Length];
            var dotted = withoutExtension.Replace('/', '.');
            return dotted.EndsWith(".__init__", StringComparison.Ordinal) ? dotted[..^".__init__".Length] : dotted;
        }
    }

    /// <summary>Number of source lines, for the UI.</summary>
    public int LineCount => Source.Length == 0 ? 0 : Source.Count(c => c == '\n') + 1;
}

/// <summary>A non-Python file the converted program is likely to read, copied alongside it.</summary>
/// <param name="RelativePath">Path relative to the project root, with forward slashes.</param>
/// <param name="FullPath">Where it is on disk.</param>
/// <param name="Length">Size in bytes.</param>
public sealed record PythonDataFile(string RelativePath, string FullPath, long Length);

/// <summary>Everything found in a Python file or folder, before any conversion.</summary>
public sealed class PythonProject
{
    /// <summary>The folder the relative paths are relative to.</summary>
    public required string Root { get; init; }

    /// <summary>A name for the C# project, derived from the file or folder.</summary>
    public required string Name { get; init; }

    /// <summary>The Python modules, in path order.</summary>
    public required IReadOnlyList<PythonSourceFile> Files { get; init; }

    /// <summary>Data files to copy next to the converted code.</summary>
    public required IReadOnlyList<PythonDataFile> DataFiles { get; init; }

    /// <summary>Contents of <c>requirements.txt</c>, when there is one.</summary>
    public string? Requirements { get; init; }

    /// <summary>Files that were found but deliberately left out, with the reason.</summary>
    public required IReadOnlyList<string> Skipped { get; init; }
}

/// <summary>
/// Finds the Python sources in a single file or a folder.
/// </summary>
/// <remarks>
/// Virtual environments, caches and version control are skipped by directory name, because a
/// project folder with a <c>.venv</c> inside it otherwise contributes thousands of site-packages
/// modules that must not be converted. Notebooks are read as their code cells, with IPython magics
/// and shell escapes commented out - they are not Python and would only confuse the conversion.
/// </remarks>
public static class PythonProjectScanner
{
    /// <summary>Data files above this size are listed but not copied.</summary>
    public const long MaxDataFileBytes = 50L * 1024 * 1024;

    /// <summary>Python sources above this size are refused: they do not fit a conversion prompt.</summary>
    public const int MaxSourceChars = 120_000;

    private static readonly HashSet<string> SkippedDirectories = new(StringComparer.OrdinalIgnoreCase)
    {
        ".git", ".hg", ".svn", ".venv", "venv", "env", ".env", "__pycache__", ".ipynb_checkpoints",
        "site-packages", "node_modules", ".mypy_cache", ".pytest_cache", ".tox", ".idea", ".vscode",
        "build", "dist", "bin", "obj", ".eggs",
    };

    private static readonly HashSet<string> DataExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".csv", ".tsv", ".txt", ".json", ".jsonl", ".xlsx", ".parquet", ".npy", ".npz", ".mat",
        ".h5", ".hdf5", ".nc", ".wav", ".arff", ".mtx", ".xml", ".yaml", ".yml", ".png", ".jpg",
        ".jpeg", ".dat",
    };

    /// <summary>Scans a <c>.py</c>/<c>.ipynb</c> file or a folder.</summary>
    /// <exception cref="FileNotFoundException">The path does not exist.</exception>
    /// <exception cref="InvalidOperationException">No Python source was found.</exception>
    public static PythonProject Scan(string path)
    {
        var full = Path.GetFullPath(path);

        if (File.Exists(full))
        {
            var extension = Path.GetExtension(full);
            if (!extension.Equals(".py", StringComparison.OrdinalIgnoreCase)
                && !extension.Equals(".ipynb", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"'{Path.GetFileName(full)}' is not a .py or .ipynb file.");

            var root = Path.GetDirectoryName(full)!;
            var skipped = new List<string>();
            var file = Read(root, full, skipped)
                       ?? throw new InvalidOperationException(skipped.FirstOrDefault() ?? "The file could not be read.");
            return new PythonProject
            {
                Root = root,
                Name = ProjectNameFrom(Path.GetFileNameWithoutExtension(full)),
                Files = [file],
                DataFiles = [],
                Skipped = skipped,
            };
        }

        if (!Directory.Exists(full))
            throw new FileNotFoundException($"'{path}' does not exist.");

        var files = new List<PythonSourceFile>();
        var data = new List<PythonDataFile>();
        var skippedFiles = new List<string>();

        foreach (var entry in Walk(full))
        {
            var relative = Path.GetRelativePath(full, entry).Replace('\\', '/');
            var extension = Path.GetExtension(entry);

            if (extension.Equals(".py", StringComparison.OrdinalIgnoreCase)
                || extension.Equals(".ipynb", StringComparison.OrdinalIgnoreCase))
            {
                // setup.py and conftest.py are packaging and test plumbing, not program logic.
                var name = Path.GetFileName(entry);
                if (name.Equals("setup.py", StringComparison.OrdinalIgnoreCase)
                    || name.Equals("conftest.py", StringComparison.OrdinalIgnoreCase))
                {
                    skippedFiles.Add($"{relative}: packaging/test plumbing");
                    continue;
                }

                var file = Read(full, entry, skippedFiles);
                if (file is null) continue;

                // An empty __init__.py only marks a package; there is nothing to convert.
                if (string.IsNullOrWhiteSpace(StripComments(file.Source))
                    && name.Equals("__init__.py", StringComparison.OrdinalIgnoreCase))
                    continue;

                files.Add(file);
            }
            else if (DataExtensions.Contains(extension) && !IsProjectMetadata(Path.GetFileName(entry)))
            {
                var length = new FileInfo(entry).Length;
                if (length > MaxDataFileBytes) skippedFiles.Add($"{relative}: data file over 50 MB, not copied");
                else data.Add(new PythonDataFile(relative, entry, length));
            }
        }

        if (files.Count == 0)
            throw new InvalidOperationException($"No Python files were found in '{full}'.");

        var requirements = Path.Combine(full, "requirements.txt");
        return new PythonProject
        {
            Root = full,
            Name = ProjectNameFrom(new DirectoryInfo(full).Name),
            Files = files.OrderBy(f => f.RelativePath, StringComparer.Ordinal).ToList(),
            DataFiles = data.OrderBy(f => f.RelativePath, StringComparer.Ordinal).ToList(),
            Requirements = File.Exists(requirements) ? File.ReadAllText(requirements) : null,
            Skipped = skippedFiles,
        };
    }

    private static IEnumerable<string> Walk(string root)
    {
        var pending = new Stack<string>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            var directory = pending.Pop();
            IEnumerable<string> children;
            try { children = Directory.EnumerateFiles(directory); }
            catch (UnauthorizedAccessException) { continue; }

            foreach (var file in children) yield return file;

            foreach (var sub in Directory.EnumerateDirectories(directory))
            {
                var name = Path.GetFileName(sub);
                if (SkippedDirectories.Contains(name) || name.EndsWith(".egg-info", StringComparison.OrdinalIgnoreCase)) continue;
                // A folder holding its own pyvenv.cfg is a virtual environment, whatever its name.
                if (File.Exists(Path.Combine(sub, "pyvenv.cfg"))) continue;
                pending.Push(sub);
            }
        }
    }

    private static PythonSourceFile? Read(string root, string path, List<string> skipped)
    {
        var relative = Path.GetRelativePath(root, path).Replace('\\', '/');
        string source;
        var notebook = path.EndsWith(".ipynb", StringComparison.OrdinalIgnoreCase);
        try
        {
            source = notebook ? NotebookToPython(File.ReadAllText(path)) : File.ReadAllText(path);
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            skipped.Add($"{relative}: could not be read ({ex.Message})");
            return null;
        }

        if (source.Length > MaxSourceChars)
        {
            skipped.Add($"{relative}: {source.Length:N0} characters is too large to convert in one piece - split it into modules");
            return null;
        }

        return new PythonSourceFile(relative, source.Replace("\r\n", "\n"), notebook);
    }

    /// <summary>requirements.txt, README.txt, LICENSE.txt: about the project, not read by it.</summary>
    private static bool IsProjectMetadata(string fileName)
        => fileName.StartsWith("requirements", StringComparison.OrdinalIgnoreCase)
           || fileName.StartsWith("readme", StringComparison.OrdinalIgnoreCase)
           || fileName.StartsWith("license", StringComparison.OrdinalIgnoreCase)
           || fileName.StartsWith("changelog", StringComparison.OrdinalIgnoreCase);

    /// <summary>The code cells of a notebook as one Python module, magics commented out.</summary>
    public static string NotebookToPython(string json)
    {
        using var document = JsonDocument.Parse(json);
        var builder = new StringBuilder();
        if (!document.RootElement.TryGetProperty("cells", out var cells)) return "";

        var index = 0;
        foreach (var cell in cells.EnumerateArray())
        {
            if (!cell.TryGetProperty("cell_type", out var type) || type.GetString() != "code") continue;
            index++;

            var text = cell.TryGetProperty("source", out var src)
                ? src.ValueKind == JsonValueKind.Array
                    ? string.Concat(src.EnumerateArray().Select(l => l.GetString()))
                    : src.GetString() ?? ""
                : "";

            builder.Append("# ---- cell ").Append(index).Append('\n');
            foreach (var line in text.Replace("\r\n", "\n").Split('\n'))
            {
                var trimmed = line.TrimStart();
                builder.Append(trimmed.StartsWith('%') || trimmed.StartsWith('!') ? "# (notebook magic) " + line : line);
                builder.Append('\n');
            }
        }

        return builder.ToString();
    }

    /// <summary>A valid C# project name: PascalCase, letters and digits only, not starting with a digit.</summary>
    public static string ProjectNameFrom(string raw)
    {
        var name = CSharpNaming.ToPascalCase(raw);
        if (name.Length == 0) name = "Converted";
        return char.IsDigit(name[0]) ? "App" + name : name;
    }

    private static string StripComments(string source)
        => string.Join('\n', source.Split('\n').Where(l => !l.TrimStart().StartsWith('#')));
}
