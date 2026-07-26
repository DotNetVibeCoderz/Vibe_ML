using System.Collections.Concurrent;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Domain;

namespace BlazorML.Infrastructure.Storage;

/// <summary>
/// Builds providers on demand and caches them per kind. Because storage settings are editable at
/// runtime, the cache is dropped whenever the Storage section changes.
/// </summary>
public sealed class StorageProviderFactory : IStorageProviderFactory, IDisposable
{
    private readonly ISettingsService _settings;
    private readonly string _contentRoot;
    private readonly ConcurrentDictionary<StorageProviderKind, IStorageProvider> _cache = new();

    public StorageProviderFactory(ISettingsService settings, string contentRoot)
    {
        _settings = settings;
        _contentRoot = contentRoot;
        _settings.SectionChanged += OnSectionChanged;
    }

    public IStorageProvider Current
    {
        get
        {
            var options = Options();
            return Resolve(options.Provider);
        }
    }

    public IStorageProvider Resolve(StorageProviderKind kind) =>
        _cache.GetOrAdd(kind, Build);

    private IStorageProvider Build(StorageProviderKind kind)
    {
        var options = Options();

        return kind switch
        {
            StorageProviderKind.AzureBlob => new AzureBlobStorageProvider(options.AzureBlob),
            StorageProviderKind.S3 => new S3StorageProvider(options.S3, StorageProviderKind.S3),
            StorageProviderKind.MinIO => new S3StorageProvider(options.MinIO, StorageProviderKind.MinIO),
            _ => new FileSystemStorageProvider(options.FileSystem, _contentRoot)
        };
    }

    private StorageOptions Options() =>
        _settings.GetAsync<StorageOptions>(SettingsSections.Storage).GetAwaiter().GetResult();

    private void OnSectionChanged(string section)
    {
        if (section == SettingsSections.Storage)
        {
            _cache.Clear();
        }
    }

    public void Dispose() => _settings.SectionChanged -= OnSectionChanged;
}
