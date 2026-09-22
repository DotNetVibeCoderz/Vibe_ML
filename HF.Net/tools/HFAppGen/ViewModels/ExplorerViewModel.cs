using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using HFAppGen.Services;

namespace HFAppGen.ViewModels;

/// <summary>One entry in the file tree.</summary>
public sealed partial class FileNode : ObservableObject
{
    /// <summary>Display name.</summary>
    public required string Name { get; init; }

    /// <summary>Absolute path.</summary>
    public required string FullPath { get; init; }

    /// <summary>True for a directory.</summary>
    public required bool IsDirectory { get; init; }

    /// <summary>Child nodes; empty for files.</summary>
    public ObservableCollection<FileNode> Children { get; } = [];

    [ObservableProperty] private bool _isExpanded;

    /// <summary>A short glyph standing in for a file-type icon.</summary>
    public string Glyph => IsDirectory
        ? (IsExpanded ? "▾" : "▸")
        : Path.GetExtension(Name).ToLowerInvariant() switch
        {
            ".cs" => "#",
            ".csproj" or ".sln" => "◈",
            ".json" => "{}",
            ".md" => "¶",
            ".ipynb" => "▣",
            ".csv" => "▤",
            ".xaml" or ".axaml" => "◇",
            _ => "·",
        };

    /// <summary>The spectrum band this file type is tinted with.</summary>
    public string AccentBrush => IsDirectory ? "TextMutedBrush" : Path.GetExtension(Name).ToLowerInvariant() switch
    {
        ".cs" => "Spec5",
        ".csproj" or ".sln" => "Spec3",
        ".json" => "Spec2",
        ".md" => "Spec4",
        ".ipynb" => "Spec6",
        ".csv" => "Spec1",
        _ => "TextFaintBrush",
    };
}

/// <summary>
/// The project file tree.
/// </summary>
/// <remarks>
/// Directories are read on demand as they expand rather than eagerly, so opening a project with a
/// large <c>bin</c> tree does not stall the UI. Build output and version control are filtered out
/// by <see cref="ProjectService.IsIgnored"/>.
/// </remarks>
public sealed partial class ExplorerViewModel(ProjectService projects) : ObservableObject
{
    /// <summary>Raised when the user activates a file.</summary>
    public event Action<string>? FileOpenRequested;

    /// <summary>Top-level nodes.</summary>
    public ObservableCollection<FileNode> Roots { get; } = [];

    [ObservableProperty] private FileNode? _selected;
    [ObservableProperty] private string _emptyMessage = "No project open";

    /// <summary>Rebuilds the tree from the open project.</summary>
    public void Refresh()
    {
        Roots.Clear();

        if (projects.ProjectRoot is null)
        {
            EmptyMessage = "No project open";
            return;
        }

        var root = new FileNode
        {
            Name = projects.ProjectName!,
            FullPath = projects.ProjectRoot,
            IsDirectory = true,
            IsExpanded = true,
        };

        Populate(root);
        Roots.Add(root);
        EmptyMessage = "";
    }

    /// <summary>Fills a directory node's children, one level deep.</summary>
    public static void Populate(FileNode node)
    {
        node.Children.Clear();
        if (!Directory.Exists(node.FullPath)) return;

        try
        {
            foreach (var directory in Directory.EnumerateDirectories(node.FullPath)
                         .Where(d => !ProjectService.IsIgnored(d))
                         .OrderBy(d => d, StringComparer.OrdinalIgnoreCase))
            {
                var child = new FileNode
                {
                    Name = Path.GetFileName(directory),
                    FullPath = directory,
                    IsDirectory = true,
                };
                // One placeholder level so the expander chevron appears without a full walk.
                Populate(child, depth: 1);
                node.Children.Add(child);
            }

            foreach (var file in Directory.EnumerateFiles(node.FullPath)
                         .OrderBy(f => f, StringComparer.OrdinalIgnoreCase))
            {
                node.Children.Add(new FileNode
                {
                    Name = Path.GetFileName(file),
                    FullPath = file,
                    IsDirectory = false,
                });
            }
        }
        catch (UnauthorizedAccessException)
        {
            // A directory we cannot read simply shows as empty.
        }
    }

    private static void Populate(FileNode node, int depth)
    {
        if (depth <= 0) return;
        Populate(node);
    }

    /// <summary>Raises <see cref="FileOpenRequested"/> for a file node.</summary>
    public void Activate(FileNode node)
    {
        if (node.IsDirectory)
        {
            node.IsExpanded = !node.IsExpanded;
            if (node.IsExpanded && node.Children.Count == 0) Populate(node);
            return;
        }

        FileOpenRequested?.Invoke(node.FullPath);
    }
}
