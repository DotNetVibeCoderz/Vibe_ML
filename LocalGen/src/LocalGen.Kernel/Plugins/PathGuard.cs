using LocalGen.Core;
using LocalGen.Core.Configuration;

namespace LocalGen.Kernel.Plugins;

/// <summary>
/// Confines file-touching kernel functions to directories the user has approved.
/// </summary>
/// <remarks>
/// The file system, download and code execution functions all act on paths chosen by the model.
/// Without a check, a prompt-injected instruction could read SSH keys or overwrite files anywhere
/// the process can reach. Paths are fully resolved before comparison so that <c>..</c> segments
/// and symlink-style tricks cannot escape an allowed root.
/// </remarks>
public sealed class PathGuard
{
    private readonly string[] _allowedRoots;

    public PathGuard(LocalGenOptions options)
    {
        var roots = new List<string>
        {
            // The LocalGen workspace is always writable; it is what the tools are for.
            Path.GetFullPath(Path.Combine(options.DataDirectory, "workspace"))
        };

        roots.AddRange(options.Tools.AllowedPaths
            .Where(static p => !string.IsNullOrWhiteSpace(p))
            .Select(Path.GetFullPath));

        _allowedRoots = [.. roots.Distinct(StringComparer.OrdinalIgnoreCase)];

        Workspace = roots[0];
        Directory.CreateDirectory(Workspace);
    }

    /// <summary>Default working directory for tools, always inside an allowed root.</summary>
    public string Workspace { get; }

    public IReadOnlyList<string> AllowedRoots => _allowedRoots;

    /// <summary>
    /// Resolves a model-supplied path and verifies it stays inside an allowed root.
    /// Relative paths are interpreted against the workspace.
    /// </summary>
    /// <exception cref="ToolPermissionException">The path falls outside every allowed root.</exception>
    public string Resolve(string path, string toolName)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ToolPermissionException(toolName, "a path is required.");
        }

        var resolved = Path.GetFullPath(
            Path.IsPathRooted(path) ? path : Path.Combine(Workspace, path));

        if (!IsAllowed(resolved))
        {
            throw new ToolPermissionException(
                toolName,
                $"'{resolved}' is outside the allowed directories. " +
                $"Allowed: {string.Join(", ", _allowedRoots)}. " +
                "Add more under LocalGen:Tools:AllowedPaths if this is intended.");
        }

        return resolved;
    }

    private bool IsAllowed(string fullPath)
    {
        foreach (var root in _allowedRoots)
        {
            if (fullPath.Equals(root, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            // The separator check stops "/data/models-evil" from matching the root "/data/models".
            var prefix = root.EndsWith(Path.DirectorySeparatorChar)
                ? root
                : root + Path.DirectorySeparatorChar;

            if (fullPath.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }
}
