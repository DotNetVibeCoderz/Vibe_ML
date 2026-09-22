namespace Gravicode.HFNet.GraviHub;

/// <summary>
/// The on-disk store behind every download: one directory per repository revision, plus a sidecar
/// file per artifact holding the ETag it was fetched with.
/// </summary>
/// <remarks>
/// <para>
/// The layout is deliberately readable — <c>models/bert-base-uncased/main/model.safetensors</c> —
/// rather than the blob/snapshot/symlink arrangement the Python client uses. Symlinks need elevation
/// or Developer Mode on Windows, and a cache a user cannot inspect with a file browser is one they
/// cannot clear when a download goes wrong.
/// </para>
/// <para>
/// The cost is that two revisions sharing an identical file store it twice. That is the right trade
/// here: revisions of a model are rarely both held, and correctness of a plain directory tree beats
/// deduplication that fails on a stock Windows box.
/// </para>
/// </remarks>
public sealed class HubCache
{
    private const string ETagDirectory = ".etags";

    /// <summary>Creates a cache rooted at a directory.</summary>
    /// <param name="root">
    /// Where to keep files. When null the environment decides: <c>HF_HUB_CACHE</c>, then
    /// <c>HF_HOME</c>, then a per-user default.
    /// </param>
    public HubCache(string? root = null)
    {
        Root = root ?? DefaultRoot();
    }

    /// <summary>The directory this cache lives in.</summary>
    public string Root { get; }

    /// <summary>Resolves the default cache location from the environment.</summary>
    /// <remarks>
    /// <c>HF_HUB_CACHE</c> and <c>HF_HOME</c> are honoured so that a machine already set up for the
    /// Python tooling does not end up with two copies of every checkpoint.
    /// </remarks>
    public static string DefaultRoot()
    {
        var explicitCache = Environment.GetEnvironmentVariable("HF_HUB_CACHE");
        if (!string.IsNullOrWhiteSpace(explicitCache)) return explicitCache;

        var home = Environment.GetEnvironmentVariable("HF_HOME");
        if (!string.IsNullOrWhiteSpace(home)) return Path.Combine(home, "hub");

        var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(local))
        {
            local = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
        }

        return Path.Combine(local, "huggingface", "hub");
    }

    /// <summary>The directory holding one repository revision.</summary>
    public string DirectoryFor(string repoId, RepoKind kind, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var segment = kind switch
        {
            RepoKind.Model => "models",
            RepoKind.Dataset => "datasets",
            RepoKind.Space => "spaces",
            _ => "models",
        };

        return Path.Combine(Root, segment, Sanitize(repoId), Sanitize(revision));
    }

    /// <summary>The local path a repository file is cached at.</summary>
    /// <param name="repoId">The repository id.</param>
    /// <param name="kind">Model, dataset or Space.</param>
    /// <param name="revision">The branch, tag or commit.</param>
    /// <param name="fileName">The repository-relative path.</param>
    /// <remarks>
    /// The repository-relative path keeps its own subdirectories, so a sharded checkpoint lands
    /// with its index beside it and can be opened as if it had been cloned.
    /// </remarks>
    public string PathFor(string repoId, RepoKind kind, string revision, string fileName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);

        var relative = fileName.Replace('\\', '/');
        var parts = relative.Split('/', StringSplitOptions.RemoveEmptyEntries);

        // A repository path is attacker-controlled as far as this process is concerned: a file
        // named ../../../evil would otherwise escape the cache root entirely.
        foreach (var part in parts)
        {
            if (part is "." or "..")
            {
                throw new HubException($"'{fileName}' contains a relative segment and will not be cached.")
                {
                    RepoId = repoId,
                };
            }
        }

        return Path.Combine(DirectoryFor(repoId, kind, revision), Path.Combine([.. parts.Select(Sanitize)]));
    }

    /// <summary>Whether a file is already in the cache.</summary>
    public bool Contains(string repoId, RepoKind kind, string revision, string fileName)
        => File.Exists(PathFor(repoId, kind, revision, fileName));

    /// <summary>Reads the ETag a cached file was fetched with, or <c>null</c>.</summary>
    public string? ReadETag(string cachedPath)
    {
        var sidecar = ETagPath(cachedPath);
        return File.Exists(sidecar) ? File.ReadAllText(sidecar).Trim() : null;
    }

    /// <summary>Records the ETag a file was fetched with.</summary>
    public void WriteETag(string cachedPath, string etag)
    {
        var sidecar = ETagPath(cachedPath);
        Directory.CreateDirectory(Path.GetDirectoryName(sidecar)!);
        File.WriteAllText(sidecar, etag);
    }

    /// <summary>Total size of everything in the cache, in bytes.</summary>
    public long SizeInBytes()
    {
        if (!Directory.Exists(Root)) return 0;

        long total = 0;
        foreach (var file in Directory.EnumerateFiles(Root, "*", SearchOption.AllDirectories))
        {
            try
            {
                total += new FileInfo(file).Length;
            }
            catch (IOException)
            {
                // A file being written by another process is not worth failing a size report over.
            }
        }
        return total;
    }

    /// <summary>Deletes a repository revision from the cache.</summary>
    /// <returns>True when something was removed.</returns>
    public bool Evict(string repoId, RepoKind kind, string revision = "main")
    {
        var directory = DirectoryFor(repoId, kind, revision);
        if (!Directory.Exists(directory)) return false;

        Directory.Delete(directory, recursive: true);
        return true;
    }

    /// <summary>Deletes everything in the cache.</summary>
    public void Clear()
    {
        if (Directory.Exists(Root)) Directory.Delete(Root, recursive: true);
    }

    private string ETagPath(string cachedPath)
    {
        var directory = Path.GetDirectoryName(cachedPath)!;
        return Path.Combine(directory, ETagDirectory, Path.GetFileName(cachedPath) + ".etag");
    }

    /// <summary>Makes one path segment safe to write on any filesystem.</summary>
    private static string Sanitize(string segment)
    {
        Span<char> buffer = stackalloc char[segment.Length];
        var invalid = Path.GetInvalidFileNameChars();

        for (var i = 0; i < segment.Length; i++)
        {
            buffer[i] = Array.IndexOf(invalid, segment[i]) >= 0 ? '_' : segment[i];
        }

        return new string(buffer);
    }

    /// <inheritdoc />
    public override string ToString() => $"HubCache at {Root} ({SizeInBytes() / (1024.0 * 1024):0.#} MB)";
}
