using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;

namespace SplatStudio.Infrastructure.Storage;

/// <summary>
/// Stores files on local disk under <see cref="FileSystemStorageOptions.RootPath"/>.
/// Best for single-instance/dev deployments. Files are served back to the
/// browser through a dedicated static-file mapping (see Program.cs) rather
/// than a generic wwwroot exposure, so uploads can live outside the web root.
/// </summary>
public class FileSystemStorageService : IStorageService
{
    private readonly string _rootPath;
    private readonly string _publicBasePath;

    public StorageProviderType ProviderType => StorageProviderType.FileSystem;

    public FileSystemStorageService(FileSystemStorageOptions options, string contentRootPath)
    {
        _rootPath = Path.IsPathRooted(options.RootPath)
            ? options.RootPath
            : Path.Combine(contentRootPath, options.RootPath);
        _publicBasePath = options.PublicBasePath.TrimEnd('/');
        Directory.CreateDirectory(_rootPath);
    }

    private string ResolvePath(string key)
    {
        // Defend against path traversal — keys are generated server-side but
        // never trust a value that ends up concatenated into a file path.
        var safeKey = key.Replace("\\", "/").TrimStart('/');
        if (safeKey.Contains(".."))
            throw new ArgumentException("Invalid storage key.", nameof(key));

        var fullPath = Path.GetFullPath(Path.Combine(_rootPath, safeKey));
        if (!fullPath.StartsWith(Path.GetFullPath(_rootPath), StringComparison.Ordinal))
            throw new ArgumentException("Invalid storage key.", nameof(key));

        return fullPath;
    }

    public async Task SaveAsync(string key, Stream content, string contentType, CancellationToken ct = default)
    {
        var path = ResolvePath(key);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await using var fileStream = File.Create(path);
        await content.CopyToAsync(fileStream, ct);
    }

    public Task<Stream> OpenReadAsync(string key, CancellationToken ct = default)
    {
        var path = ResolvePath(key);
        Stream stream = File.OpenRead(path);
        return Task.FromResult(stream);
    }

    public Task DeleteAsync(string key, CancellationToken ct = default)
    {
        var path = ResolvePath(key);
        if (File.Exists(path)) File.Delete(path);
        return Task.CompletedTask;
    }

    public Task<bool> ExistsAsync(string key, CancellationToken ct = default) =>
        Task.FromResult(File.Exists(ResolvePath(key)));

    public Task<string> GetPublicUrlAsync(string key, CancellationToken ct = default) =>
        Task.FromResult($"{_publicBasePath}/{key.Replace("\\", "/").TrimStart('/')}");
}
