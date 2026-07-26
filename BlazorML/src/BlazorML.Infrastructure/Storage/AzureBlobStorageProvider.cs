using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Azure.Storage.Sas;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Domain;

namespace BlazorML.Infrastructure.Storage;

public sealed class AzureBlobStorageProvider : IStorageProvider
{
    private readonly BlobContainerClient _container;
    private bool _containerChecked;

    public AzureBlobStorageProvider(AzureBlobStorageOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.ConnectionString))
        {
            throw new InvalidOperationException(
                "Azure Blob storage is selected but no connection string is set. Add one under Settings → Storage.");
        }

        _container = new BlobServiceClient(options.ConnectionString).GetBlobContainerClient(options.Container);
    }

    public StorageProviderKind Kind => StorageProviderKind.AzureBlob;

    public async Task<string> WriteAsync(string key, Stream content, string? contentType = null, CancellationToken ct = default)
    {
        await EnsureContainerAsync(ct);
        var blob = _container.GetBlobClient(key);

        await blob.UploadAsync(content, new BlobUploadOptions
        {
            HttpHeaders = contentType is null ? null : new BlobHttpHeaders { ContentType = contentType }
        }, ct);

        return key;
    }

    public async Task<Stream> ReadAsync(string key, CancellationToken ct = default)
    {
        var response = await _container.GetBlobClient(key).DownloadStreamingAsync(cancellationToken: ct);
        return response.Value.Content;
    }

    public async Task<bool> ExistsAsync(string key, CancellationToken ct = default) =>
        await _container.GetBlobClient(key).ExistsAsync(ct);

    public async Task DeleteAsync(string key, CancellationToken ct = default) =>
        await _container.GetBlobClient(key).DeleteIfExistsAsync(cancellationToken: ct);

    public async Task<IReadOnlyList<StorageItem>> ListAsync(string prefix, CancellationToken ct = default)
    {
        var items = new List<StorageItem>();
        await foreach (var blob in _container.GetBlobsAsync(
            BlobTraits.None, BlobStates.None, prefix, ct))
        {
            items.Add(new StorageItem(blob.Name, blob.Properties.ContentLength ?? 0, blob.Properties.LastModified));
        }

        return items;
    }

    public Task<string> GetUrlAsync(string key, TimeSpan? lifetime = null, CancellationToken ct = default)
    {
        var blob = _container.GetBlobClient(key);

        // A SAS needs a shared-key credential; with a token credential we fall back to the app's
        // own authorised download route rather than handing out a URL that would 403.
        if (!blob.CanGenerateSasUri)
        {
            return Task.FromResult($"/storage/{Uri.EscapeDataString(key)}");
        }

        var expiry = DateTimeOffset.UtcNow.Add(lifetime ?? TimeSpan.FromHours(1));
        var uri = blob.GenerateSasUri(BlobSasPermissions.Read, expiry);
        return Task.FromResult(uri.ToString());
    }

    private async Task EnsureContainerAsync(CancellationToken ct)
    {
        if (_containerChecked)
        {
            return;
        }

        await _container.CreateIfNotExistsAsync(cancellationToken: ct);
        _containerChecked = true;
    }
}
