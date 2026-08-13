using System.Collections.ObjectModel;
using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using ScienceAppGen.Models;
using ScienceAppGen.Services;

namespace ScienceAppGen.ViewModels;

/// <summary>One open file.</summary>
public sealed partial class OpenDocument : ObservableObject
{
    /// <summary>Absolute path.</summary>
    public required string FullPath { get; init; }

    /// <summary>File name, shown on the tab.</summary>
    public required string Name { get; init; }

    /// <summary>The AvaloniaEdit syntax highlighting name, or null for plain text.</summary>
    public required string? SyntaxName { get; init; }

    [ObservableProperty] private string _text = "";
    [ObservableProperty] private bool _isModified;

    /// <summary>Tab caption, with a dot when there are unsaved changes.</summary>
    public string Caption => IsModified ? Name + " •" : Name;

    partial void OnIsModifiedChanged(bool value) => OnPropertyChanged(nameof(Caption));
}

/// <summary>
/// The editor: open documents, saving, and the formatting and navigation commands.
/// </summary>
public sealed partial class EditorViewModel(AppSettings settings) : ObservableObject
{
    /// <summary>Every open document.</summary>
    public ObservableCollection<OpenDocument> Documents { get; } = [];

    [ObservableProperty] private OpenDocument? _active;
    [ObservableProperty] private int _caretLine = 1;
    [ObservableProperty] private int _caretColumn = 1;
    [ObservableProperty] private bool _showLineNumbers = settings.ShowLineNumbers;

    /// <summary>Opens a file, or focuses it when it is already open.</summary>
    public void OpenFile(string path)
    {
        var existing = Documents.FirstOrDefault(d =>
            string.Equals(d.FullPath, path, StringComparison.OrdinalIgnoreCase));

        if (existing is not null)
        {
            Active = existing;
            return;
        }

        if (!File.Exists(path)) return;

        // Reading a binary file into the editor produces noise, not content.
        if (IsProbablyBinary(path)) return;

        var document = new OpenDocument
        {
            FullPath = path,
            Name = Path.GetFileName(path),
            SyntaxName = SyntaxFor(path),
            Text = File.ReadAllText(path),
        };

        Documents.Add(document);
        Active = document;
    }

    /// <summary>Saves the active document.</summary>
    public bool SaveActive()
    {
        if (Active is null) return false;

        File.WriteAllText(Active.FullPath, Active.Text,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        Active.IsModified = false;
        return true;
    }

    /// <summary>Saves every modified document.</summary>
    public int SaveAll()
    {
        var saved = 0;
        foreach (var document in Documents.Where(d => d.IsModified))
        {
            File.WriteAllText(document.FullPath, document.Text,
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            document.IsModified = false;
            saved++;
        }
        return saved;
    }

    /// <summary>Closes a document.</summary>
    public void Close(OpenDocument document)
    {
        var index = Documents.IndexOf(document);
        Documents.Remove(document);

        if (!ReferenceEquals(Active, document)) return;
        Active = Documents.Count == 0 ? null : Documents[Math.Clamp(index, 0, Documents.Count - 1)];
    }

    /// <summary>Closes every document.</summary>
    public void CloseAll()
    {
        Documents.Clear();
        Active = null;
    }

    /// <summary>Reloads a document from disk, discarding unsaved changes.</summary>
    public void Reload(string path)
    {
        var document = Documents.FirstOrDefault(d =>
            string.Equals(d.FullPath, path, StringComparison.OrdinalIgnoreCase));

        if (document is null || !File.Exists(path)) return;

        document.Text = File.ReadAllText(path);
        document.IsModified = false;
    }

    /// <summary>
    /// Re-indents the active document.
    /// </summary>
    /// <remarks>
    /// This is a brace-depth reformatter, not a C# parser: it fixes indentation and trims trailing
    /// whitespace, and deliberately leaves everything else alone. Anything more would need Roslyn,
    /// and getting it half-right would be worse than not doing it.
    /// </remarks>
    public string FormatActive()
    {
        if (Active is null) return "No file is open.";

        var extension = Path.GetExtension(Active.FullPath).ToLowerInvariant();
        if (extension is not (".cs" or ".json" or ".axaml" or ".xaml" or ".xml" or ".csproj"))
            return $"Formatting is not supported for {extension} files.";

        var indent = new string(' ', settings.TabSize);
        var builder = new StringBuilder();
        var depth = 0;

        foreach (var raw in Active.Text.Split('\n'))
        {
            var line = raw.TrimEnd('\r', ' ', '\t');
            var trimmed = line.TrimStart();

            if (trimmed.Length == 0)
            {
                builder.Append('\n');
                continue;
            }

            // A closing brace belongs at the parent's level, so dedent before writing it.
            if (trimmed.StartsWith('}') || trimmed.StartsWith(']') || trimmed.StartsWith("</"))
                depth = Math.Max(0, depth - 1);

            builder.Append(string.Concat(Enumerable.Repeat(indent, depth))).Append(trimmed).Append('\n');

            var opens = trimmed.Count(c => c is '{' or '[');
            var closes = trimmed.Count(c => c is '}' or ']');
            if (opens > closes) depth++;
        }

        Active.Text = builder.ToString().TrimEnd('\n') + "\n";
        Active.IsModified = true;
        return "Formatted.";
    }

    /// <summary>The syntax highlighting definition for a path, or null when there is none.</summary>
    public static string? SyntaxFor(string path) => Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".cs" => "C#",
        ".json" or ".ipynb" => "JavaScript",
        ".xml" or ".xaml" or ".axaml" or ".csproj" or ".config" => "XML",
        ".md" => "MarkDown",
        ".py" => "Python",
        ".js" => "JavaScript",
        ".html" or ".htm" => "HTML",
        ".css" => "CSS",
        _ => null,
    };

    /// <summary>Sniffs for a NUL byte, which text files do not contain.</summary>
    private static bool IsProbablyBinary(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            Span<byte> buffer = stackalloc byte[512];
            var read = stream.Read(buffer);
            return buffer[..read].Contains((byte)0);
        }
        catch
        {
            return true;
        }
    }
}
