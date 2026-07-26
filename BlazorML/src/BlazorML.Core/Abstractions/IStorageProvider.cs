using BlazorML.Core.Domain;

namespace BlazorML.Core.Abstractions;

/// <summary>
/// The single door to bulk bytes. Datasets, uploaded chat attachments and serialised ML.NET
/// models all move through this, so swapping FileSystem for S3 is a configuration change.
/// </summary>
public interface IStorageProvider
{
    StorageProviderKind Kind { get; }

    Task<string> WriteAsync(string key, Stream content, string? contentType = null, CancellationToken ct = default);

    Task<Stream> ReadAsync(string key, CancellationToken ct = default);

    Task<bool> ExistsAsync(string key, CancellationToken ct = default);

    Task DeleteAsync(string key, CancellationToken ct = default);

    Task<IReadOnlyList<StorageItem>> ListAsync(string prefix, CancellationToken ct = default);

    /// <summary>
    /// A URL a browser can fetch the object from. Providers that cannot serve directly return a
    /// route into the app's own download handler instead of a signed URL.
    /// </summary>
    Task<string> GetUrlAsync(string key, TimeSpan? lifetime = null, CancellationToken ct = default);
}

public sealed record StorageItem(string Key, long SizeBytes, DateTimeOffset? LastModified);

/// <summary>Resolves the provider named in configuration, and any provider by kind for migrations.</summary>
public interface IStorageProviderFactory
{
    IStorageProvider Current { get; }
    IStorageProvider Resolve(StorageProviderKind kind);
}
