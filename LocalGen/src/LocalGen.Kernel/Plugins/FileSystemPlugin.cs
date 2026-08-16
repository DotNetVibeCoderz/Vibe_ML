using System.ComponentModel;
using System.Text;
using LocalGen.Core;
using Microsoft.SemanticKernel;

namespace LocalGen.Kernel.Plugins;

/// <summary>
/// Reads and writes files inside the directories the user has allowed.
/// </summary>
/// <remarks>
/// Every path goes through <see cref="PathGuard"/>, and reads are size-capped: a model that asks
/// for a 2 GB log would otherwise blow out both memory and the context window.
/// </remarks>
public sealed class FileSystemPlugin(PathGuard guard)
{
    private const string ToolName = "FileSystem";

    /// <summary>Reads are capped well below the context window of any model LocalGen serves.</summary>
    private const int MaxReadBytes = 512 * 1024;

    [KernelFunction("list_directory")]
    [Description("Lists the files and folders in a directory.")]
    public string ListDirectory(
        [Description("Directory path. Relative paths resolve against the workspace.")] string path = ".")
    {
        try
        {
            var resolved = guard.Resolve(path, ToolName);

            if (!Directory.Exists(resolved))
            {
                return $"Error: '{resolved}' is not an existing directory.";
            }

            var builder = new StringBuilder();
            builder.Append("Contents of ").AppendLine(resolved);

            foreach (var directory in Directory.EnumerateDirectories(resolved).Order())
            {
                builder.Append("  [dir]  ").AppendLine(Path.GetFileName(directory));
            }

            foreach (var file in Directory.EnumerateFiles(resolved).Order())
            {
                var info = new FileInfo(file);
                builder.Append("  [file] ")
                       .Append(info.Name)
                       .Append("  (")
                       .Append(FormatSize(info.Length))
                       .AppendLine(")");
            }

            return builder.ToString();
        }
        catch (ToolPermissionException ex)
        {
            return $"Error: {ex.Message}";
        }
        catch (IOException ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("read_file")]
    [Description("Reads a text file and returns its contents.")]
    public async Task<string> ReadFileAsync(
        [Description("Path to the file")] string path,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var resolved = guard.Resolve(path, ToolName);

            if (!File.Exists(resolved))
            {
                return $"Error: '{resolved}' does not exist.";
            }

            var info = new FileInfo(resolved);
            if (info.Length > MaxReadBytes)
            {
                return $"Error: '{info.Name}' is {FormatSize(info.Length)}, over the " +
                       $"{FormatSize(MaxReadBytes)} read limit. Read a smaller file or use a search tool.";
            }

            return await File.ReadAllTextAsync(resolved, cancellationToken).ConfigureAwait(false);
        }
        catch (ToolPermissionException ex)
        {
            return $"Error: {ex.Message}";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("write_file")]
    [Description("Writes text to a file, creating or overwriting it. Parent folders are created as needed.")]
    public async Task<string> WriteFileAsync(
        [Description("Path to the file")] string path,
        [Description("Content to write")] string content,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var resolved = guard.Resolve(path, ToolName);

            var directory = Path.GetDirectoryName(resolved);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            await File.WriteAllTextAsync(resolved, content, cancellationToken).ConfigureAwait(false);
            return $"Wrote {content.Length:N0} characters to {resolved}.";
        }
        catch (ToolPermissionException ex)
        {
            return $"Error: {ex.Message}";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("append_file")]
    [Description("Appends text to the end of a file, creating it if it does not exist.")]
    public async Task<string> AppendFileAsync(
        [Description("Path to the file")] string path,
        [Description("Content to append")] string content,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var resolved = guard.Resolve(path, ToolName);

            var directory = Path.GetDirectoryName(resolved);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            await File.AppendAllTextAsync(resolved, content, cancellationToken).ConfigureAwait(false);
            return $"Appended {content.Length:N0} characters to {resolved}.";
        }
        catch (ToolPermissionException ex)
        {
            return $"Error: {ex.Message}";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("file_exists")]
    [Description("Reports whether a file or directory exists.")]
    public string Exists([Description("Path to check")] string path)
    {
        try
        {
            var resolved = guard.Resolve(path, ToolName);

            return File.Exists(resolved) ? "file"
                : Directory.Exists(resolved) ? "directory"
                : "missing";
        }
        catch (ToolPermissionException ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("delete_file")]
    [Description("Deletes a file. Directories are not deleted by this function.")]
    public string DeleteFile([Description("Path to the file")] string path)
    {
        try
        {
            var resolved = guard.Resolve(path, ToolName);

            if (!File.Exists(resolved))
            {
                return $"Error: '{resolved}' does not exist.";
            }

            File.Delete(resolved);
            return $"Deleted {resolved}.";
        }
        catch (ToolPermissionException ex)
        {
            return $"Error: {ex.Message}";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("search_files")]
    [Description("Finds files matching a glob pattern under a directory.")]
    public string SearchFiles(
        [Description("Directory to search")] string path,
        [Description("Glob pattern, e.g. \"*.md\" or \"**/*.cs\"")] string pattern,
        [Description("Search subdirectories too")] bool recursive = true)
    {
        try
        {
            var resolved = guard.Resolve(path, ToolName);

            if (!Directory.Exists(resolved))
            {
                return $"Error: '{resolved}' is not an existing directory.";
            }

            var matches = Directory
                .EnumerateFiles(
                    resolved,
                    pattern.Replace("**/", string.Empty),
                    recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly)
                .Take(200)
                .ToList();

            return matches.Count == 0
                ? $"No files matched '{pattern}' under {resolved}."
                : string.Join('\n', matches);
        }
        catch (ToolPermissionException ex)
        {
            return $"Error: {ex.Message}";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("workspace_path")]
    [Description("The workspace directory that relative paths resolve against.")]
    public string WorkspacePath() => guard.Workspace;

    private static string FormatSize(long bytes) => bytes switch
    {
        < 1024 => $"{bytes} B",
        < 1024 * 1024 => $"{bytes / 1024.0:N1} KB",
        < 1024L * 1024 * 1024 => $"{bytes / (1024.0 * 1024):N1} MB",
        _ => $"{bytes / (1024.0 * 1024 * 1024):N2} GB"
    };
}
