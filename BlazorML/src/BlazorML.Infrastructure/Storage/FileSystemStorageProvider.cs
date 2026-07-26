using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Domain;

namespace BlazorML.Infrastructure.Storage;

/// <summary>
/// Stores objects as files under a root directory. The default provider, and the only one that
/// works with no configuration at all.
/// </summary>
public sealed class FileSystemStorageProvider : IStorageProvider
{
    private readonly string _root;

    public FileSystemStorageProvider(FileSystemStorageOptions options, string contentRoot)
    {
        _root = Path.IsPathRooted(options.RootPath)
            ? options.RootPath
            : Path.Combine(contentRoot, options.RootPath);

        Directory.CreateDirectory(_root);
    }

    public StorageProviderKind Kind => StorageProviderKind.FileSystem;

    public async Task<string> WriteAsync(string key, Stream content, string? contentType = null, CancellationToken ct = default)
    {
        var path = Resolve(key);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);

        await using var file = File.Create(path);
        await content.CopyToAsync(file, ct);
        return key;
    }

    public Task<Stream> ReadAsync(string key, CancellationToken ct = default)
    {
        var path = Resolve(key);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"'{key}' is not in storage.", key);
        }

        return Task.FromResult<Stream>(File.OpenRead(path));
    }

    public Task<bool> ExistsAsync(string key, CancellationToken ct = default) =>
        Task.FromResult(File.Exists(Resolve(key)));

    public Task DeleteAsync(string key, CancellationToken ct = default)
    {
        var path = Resolve(key);
        if (File.Exists(path))
        {
            File.Delete(path);
        }

        return Task.CompletedTask;
    }

    public Task<IReadOnlyList<StorageItem>> ListAsync(string prefix, CancellationToken ct = default)
    {
        var directory = Resolve(prefix);
        if (!Directory.Exists(directory))
        {
            return Task.FromResult<IReadOnlyList<StorageItem>>(Array.Empty<StorageItem>());
        }

        var items = Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories)
            .Select(path =>
            {
                var info = new FileInfo(path);
                var key = Path.GetRelativePath(_root, path).Replace('\\', '/');
                return new StorageItem(key, info.Length, info.LastWriteTimeUtc);
            })
            .ToList();

        return Task.FromResult<IReadOnlyList<StorageItem>>(items);
    }

    /// <summary>
    /// The file system has no public URL, so this points at the app's own download route.
    /// The route checks authorisation, which a raw file path never could.
    /// </summary>
    public Task<string> GetUrlAsync(string key, TimeSpan? lifetime = null, CancellationToken ct = default) =>
        Task.FromResult($"/storage/{Uri.EscapeDataString(key)}");

    /// <summary>Blocks <c>..</c> traversal: a crafted key must not reach outside the root.</summary>
    private string Resolve(string key)
    {
        var normalised = key.Replace('\\', '/').TrimStart('/');
        var combined = Path.GetFullPath(Path.Combine(_root, normalised));
        var rootFull = Path.GetFullPath(_root);

        if (!combined.StartsWith(rootFull, StringComparison.OrdinalIgnoreCase))
        {
            throw new UnauthorizedAccessException($"Storage key '{key}' resolves outside the storage root.");
        }

        return combined;
    }
}
