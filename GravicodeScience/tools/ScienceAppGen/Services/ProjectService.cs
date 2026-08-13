using System.Diagnostics;
using System.Text;

namespace ScienceAppGen.Services;

/// <summary>The outcome of running an external tool.</summary>
/// <param name="Succeeded">Whether the process exited with zero.</param>
/// <param name="ExitCode">The process exit code.</param>
/// <param name="Output">Everything the process wrote, stdout and stderr interleaved.</param>
public sealed record ProcessResult(bool Succeeded, int ExitCode, string Output);

/// <summary>
/// Everything that touches the project on disk: file access, and the dotnet CLI.
/// </summary>
/// <remarks>
/// Every path the assistant supplies is resolved against the open project root and then checked to
/// be inside it. That check is the security boundary of the whole tool surface: the model can name
/// any path it likes, and a traversal like <c>../../../etc</c> is rejected rather than followed.
/// </remarks>
public sealed class ProjectService(LogService logs)
{
    /// <summary>The open project's root directory, or null when nothing is open.</summary>
    public string? ProjectRoot { get; private set; }

    /// <summary>The open project's display name.</summary>
    public string? ProjectName => ProjectRoot is null ? null : new DirectoryInfo(ProjectRoot).Name;

    /// <summary>True when a project is open.</summary>
    public bool HasProject => ProjectRoot is not null;

    /// <summary>Raised when the open project changes, so the explorer can rebuild.</summary>
    public event Action<string?>? ProjectChanged;

    /// <summary>Opens a project directory.</summary>
    public void OpenProject(string directory)
    {
        if (!Directory.Exists(directory))
            throw new DirectoryNotFoundException($"'{directory}' does not exist.");

        ProjectRoot = Path.GetFullPath(directory);
        logs.Info("project", $"Opened {ProjectRoot}");
        ProjectChanged?.Invoke(ProjectRoot);
    }

    /// <summary>Closes the open project.</summary>
    public void CloseProject()
    {
        if (ProjectRoot is null) return;
        logs.Info("project", $"Closed {ProjectName}");
        ProjectRoot = null;
        ProjectChanged?.Invoke(null);
    }

    /// <summary>
    /// Resolves a project-relative path and refuses anything that escapes the project root.
    /// </summary>
    public string ResolvePath(string relativePath)
    {
        if (ProjectRoot is null)
            throw new InvalidOperationException("No project is open.");

        var combined = Path.GetFullPath(Path.Combine(ProjectRoot, relativePath));

        // Compare with a trailing separator so "/proj-evil" cannot pass as inside "/proj".
        var root = ProjectRoot.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        if (!combined.StartsWith(root, StringComparison.OrdinalIgnoreCase)
            && !string.Equals(combined, ProjectRoot, StringComparison.OrdinalIgnoreCase))
        {
            throw new UnauthorizedAccessException(
                $"'{relativePath}' resolves outside the project directory and was refused.");
        }

        return combined;
    }

    /// <summary>Reads a file inside the project.</summary>
    public string ReadFile(string relativePath) => File.ReadAllText(ResolvePath(relativePath));

    /// <summary>Writes a file inside the project, creating directories as needed.</summary>
    public void WriteFile(string relativePath, string content)
    {
        var path = ResolvePath(relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        logs.Info("file", $"Wrote {relativePath} ({content.Length:N0} chars)");
    }

    /// <summary>Deletes a file inside the project.</summary>
    public void DeleteFile(string relativePath)
    {
        var path = ResolvePath(relativePath);
        if (File.Exists(path))
        {
            File.Delete(path);
            logs.Info("file", $"Deleted {relativePath}");
        }
    }

    /// <summary>Lists project files, skipping build output and version control.</summary>
    public IReadOnlyList<string> ListFiles(string? subdirectory = null, string pattern = "*")
    {
        if (ProjectRoot is null) return [];

        var start = subdirectory is null ? ProjectRoot : ResolvePath(subdirectory);
        if (!Directory.Exists(start)) return [];

        return Directory.EnumerateFiles(start, pattern, SearchOption.AllDirectories)
            .Where(p => !IsIgnored(p))
            .Select(p => Path.GetRelativePath(ProjectRoot, p).Replace('\\', '/'))
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    /// <summary>True for paths inside bin, obj, .git, .vs or node_modules.</summary>
    public static bool IsIgnored(string path)
    {
        var parts = path.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return parts.Any(part =>
            part.Equals("bin", StringComparison.OrdinalIgnoreCase)
            || part.Equals("obj", StringComparison.OrdinalIgnoreCase)
            || part.Equals(".git", StringComparison.OrdinalIgnoreCase)
            || part.Equals(".vs", StringComparison.OrdinalIgnoreCase)
            || part.Equals("node_modules", StringComparison.OrdinalIgnoreCase));
    }

    // ---------------------------------------------------------------- dotnet CLI

    /// <summary>Builds the open project.</summary>
    public Task<ProcessResult> BuildAsync(CancellationToken cancellationToken = default)
        => RunDotnetAsync("build -c Debug --nologo", cancellationToken);

    /// <summary>Runs the open project.</summary>
    public Task<ProcessResult> RunAsync(CancellationToken cancellationToken = default)
        => RunDotnetAsync("run --no-build", cancellationToken);

    /// <summary>Publishes the open project as a self-contained folder.</summary>
    public Task<ProcessResult> PublishAsync(CancellationToken cancellationToken = default)
        => RunDotnetAsync("publish -c Release -o publish --nologo", cancellationToken);

    /// <summary>Runs an arbitrary dotnet command in the project directory, streaming output to the logs.</summary>
    public async Task<ProcessResult> RunDotnetAsync(string arguments, CancellationToken cancellationToken = default)
    {
        if (ProjectRoot is null)
            return new ProcessResult(false, -1, "No project is open.");

        logs.Build("dotnet", $"> dotnet {arguments}");

        var info = new ProcessStartInfo
        {
            FileName = "dotnet",
            Arguments = arguments,
            WorkingDirectory = ProjectRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        var collected = new StringBuilder();
        try
        {
            using var process = new Process { StartInfo = info, EnableRaisingEvents = true };

            process.OutputDataReceived += (_, e) =>
            {
                if (e.Data is null) return;
                collected.AppendLine(e.Data);
                logs.Build("dotnet", e.Data);
            };
            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data is null) return;
                collected.AppendLine(e.Data);
                // The CLI writes plenty of non-fatal noise to stderr; the exit code decides.
                logs.Warning("dotnet", e.Data);
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync(cancellationToken);

            var succeeded = process.ExitCode == 0;
            if (succeeded) logs.Success("dotnet", $"Finished with exit code 0");
            else logs.Error("dotnet", $"Failed with exit code {process.ExitCode}");

            return new ProcessResult(succeeded, process.ExitCode, collected.ToString());
        }
        catch (OperationCanceledException)
        {
            logs.Warning("dotnet", "Cancelled.");
            return new ProcessResult(false, -1, "Cancelled.");
        }
        catch (Exception ex)
        {
            logs.Error("dotnet", ex.Message);
            return new ProcessResult(false, -1, ex.Message);
        }
    }
}
