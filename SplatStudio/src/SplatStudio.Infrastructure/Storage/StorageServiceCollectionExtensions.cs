using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;

namespace SplatStudio.Infrastructure.Storage;

public static class StorageServiceCollectionExtensions
{
    /// <summary>
    /// Registers exactly one <see cref="IStorageService"/> implementation —
    /// the one named by "Storage:Provider" — as a singleton. Every other
    /// part of the app injects <see cref="IStorageService"/> and never
    /// knows or cares which backend is actually running.
    /// </summary>
    public static IServiceCollection AddSplatStorage(this IServiceCollection services, IConfiguration configuration, string contentRootPath)
    {
        var options = configuration.GetSection(StorageOptions.SectionName).Get<StorageOptions>() ?? new StorageOptions();

        services.AddSingleton(options);

        if (!Enum.TryParse<StorageProviderType>(options.Provider, ignoreCase: true, out var providerType))
            providerType = StorageProviderType.FileSystem;

        services.AddSingleton<IStorageService>(_ => providerType switch
        {
            StorageProviderType.FileSystem => new FileSystemStorageService(options.FileSystem, contentRootPath),
            StorageProviderType.AzureBlob => new AzureBlobStorageService(options.AzureBlob),
            StorageProviderType.S3 => new S3StorageService(options.S3, StorageProviderType.S3),
            StorageProviderType.MinIO => new S3StorageService(options.MinIO, StorageProviderType.MinIO),
            _ => new FileSystemStorageService(options.FileSystem, contentRootPath)
        });

        return services;
    }
}
