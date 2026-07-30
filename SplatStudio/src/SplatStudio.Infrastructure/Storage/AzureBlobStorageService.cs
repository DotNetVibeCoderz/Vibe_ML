using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;

namespace SplatStudio.Infrastructure.Storage;

/// <summary>Stores files in an Azure Storage blob container.</summary>
public class AzureBlobStorageService : IStorageService
{
    private readonly BlobContainerClient _container;

    public StorageProviderType ProviderType => StorageProviderType.AzureBlob;

    public AzureBlobStorageService(AzureBlobStorageOptions options)
    {
        var serviceClient = new BlobServiceClient(options.ConnectionString);
        _container = serviceClient.GetBlobContainerClient(options.ContainerName);
        _container.CreateIfNotExists(PublicAccessType.Blob);
    }

    public async Task SaveAsync(string key, Stream content, string contentType, CancellationToken ct = default)
    {
        var blob = _container.GetBlobClient(key);
        await blob.UploadAsync(content, new BlobUploadOptions
        {
            HttpHeaders = new BlobHttpHeaders { ContentType = contentType }
        }, ct);
    }

    public async Task<Stream> OpenReadAsync(string key, CancellationToken ct = default)
    {
        var blob = _container.GetBlobClient(key);
        var download = await blob.DownloadStreamingAsync(cancellationToken: ct);
        return download.Value.Content;
    }

    public async Task DeleteAsync(string key, CancellationToken ct = default)
    {
        var blob = _container.GetBlobClient(key);
        await blob.DeleteIfExistsAsync(cancellationToken: ct);
    }

    public async Task<bool> ExistsAsync(string key, CancellationToken ct = default)
    {
        var blob = _container.GetBlobClient(key);
        var response = await blob.ExistsAsync(ct);
        return response.Value;
    }

    public Task<string> GetPublicUrlAsync(string key, CancellationToken ct = default)
    {
        var blob = _container.GetBlobClient(key);
        return Task.FromResult(blob.Uri.ToString());
    }
}
