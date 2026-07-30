using Amazon.S3;
using Amazon.S3.Model;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;

namespace SplatStudio.Infrastructure.Storage;

/// <summary>
/// Storage provider backed by the S3 API. Used for both real AWS S3 and
/// any S3-compatible server (MinIO, Cloudflare R2, Wasabi, ...) — the only
/// difference is <see cref="S3CompatibleStorageOptions.ServiceUrl"/> and
/// <see cref="S3CompatibleStorageOptions.ForcePathStyle"/>.
/// </summary>
public class S3StorageService : IStorageService
{
    private readonly IAmazonS3 _client;
    private readonly string _bucket;
    private readonly string? _publicBaseUrl;
    private readonly StorageProviderType _providerType;

    public StorageProviderType ProviderType => _providerType;

    public S3StorageService(S3CompatibleStorageOptions options, StorageProviderType providerType)
    {
        _providerType = providerType;
        _bucket = options.BucketName;
        _publicBaseUrl = options.PublicBaseUrl?.TrimEnd('/');

        var config = new AmazonS3Config
        {
            ForcePathStyle = options.ForcePathStyle,
            RegionEndpoint = string.IsNullOrWhiteSpace(options.ServiceUrl)
                ? Amazon.RegionEndpoint.GetBySystemName(options.Region)
                : null
        };
        if (!string.IsNullOrWhiteSpace(options.ServiceUrl))
            config.ServiceURL = options.ServiceUrl;

        _client = new AmazonS3Client(options.AccessKey, options.SecretKey, config);

        EnsureBucketExists();
    }

    private void EnsureBucketExists()
    {
        // Startup-time, one-shot check — blocking here is fine and keeps the
        // DI registration synchronous; everything after this is async.
        var exists = Amazon.S3.Util.AmazonS3Util.DoesS3BucketExistV2Async(_client, _bucket).GetAwaiter().GetResult();
        if (!exists)
        {
            _client.PutBucketAsync(new PutBucketRequest { BucketName = _bucket }).GetAwaiter().GetResult();
        }
    }

    public async Task SaveAsync(string key, Stream content, string contentType, CancellationToken ct = default)
    {
        await _client.PutObjectAsync(new PutObjectRequest
        {
            BucketName = _bucket,
            Key = key,
            InputStream = content,
            ContentType = contentType,
            AutoCloseStream = false
        }, ct);
    }

    public async Task<Stream> OpenReadAsync(string key, CancellationToken ct = default)
    {
        var response = await _client.GetObjectAsync(new GetObjectRequest { BucketName = _bucket, Key = key }, ct);
        return response.ResponseStream;
    }

    public async Task DeleteAsync(string key, CancellationToken ct = default)
    {
        await _client.DeleteObjectAsync(new DeleteObjectRequest { BucketName = _bucket, Key = key }, ct);
    }

    public async Task<bool> ExistsAsync(string key, CancellationToken ct = default)
    {
        try
        {
            await _client.GetObjectMetadataAsync(new GetObjectMetadataRequest { BucketName = _bucket, Key = key }, ct);
            return true;
        }
        catch (AmazonS3Exception ex) when (ex.StatusCode == System.Net.HttpStatusCode.NotFound)
        {
            return false;
        }
    }

    public Task<string> GetPublicUrlAsync(string key, CancellationToken ct = default)
    {
        if (!string.IsNullOrWhiteSpace(_publicBaseUrl))
            return Task.FromResult($"{_publicBaseUrl}/{key}");

        // Bucket not public (or no CDN configured) — hand back a time-limited signed URL.
        var url = _client.GetPreSignedURL(new GetPreSignedUrlRequest
        {
            BucketName = _bucket,
            Key = key,
            Expires = DateTime.UtcNow.AddHours(2),
            Verb = HttpVerb.GET
        });
        return Task.FromResult(url);
    }
}
