using Amazon;
using Amazon.Runtime;
using Amazon.S3;
using Amazon.S3.Model;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Domain;

namespace BlazorML.Infrastructure.Storage;

/// <summary>
/// Serves both AWS S3 and MinIO. They differ only in endpoint and addressing style, so one
/// implementation covers both rather than duplicating the object plumbing twice.
/// </summary>
public sealed class S3StorageProvider : IStorageProvider
{
    private readonly IAmazonS3 _client;
    private readonly string _bucket;
    private bool _bucketChecked;

    public S3StorageProvider(S3StorageOptions options, StorageProviderKind kind)
    {
        Kind = kind;
        _bucket = options.Bucket;

        var config = new AmazonS3Config { ForcePathStyle = options.ForcePathStyle };

        if (!string.IsNullOrWhiteSpace(options.ServiceUrl))
        {
            config.ServiceURL = options.ServiceUrl;
            config.AuthenticationRegion = options.Region;
        }
        else
        {
            config.RegionEndpoint = RegionEndpoint.GetBySystemName(options.Region);
        }

        _client = string.IsNullOrWhiteSpace(options.AccessKey)
            ? new AmazonS3Client(config)
            : new AmazonS3Client(new BasicAWSCredentials(options.AccessKey, options.SecretKey), config);
    }

    public StorageProviderKind Kind { get; }

    public async Task<string> WriteAsync(string key, Stream content, string? contentType = null, CancellationToken ct = default)
    {
        await EnsureBucketAsync(ct);

        await _client.PutObjectAsync(new PutObjectRequest
        {
            BucketName = _bucket,
            Key = key,
            InputStream = content,
            ContentType = contentType,
            AutoCloseStream = false
        }, ct);

        return key;
    }

    public async Task<Stream> ReadAsync(string key, CancellationToken ct = default)
    {
        var response = await _client.GetObjectAsync(_bucket, key, ct);

        // Copy out so the caller is not tied to the lifetime of the HTTP response.
        var buffer = new MemoryStream();
        await response.ResponseStream.CopyToAsync(buffer, ct);
        buffer.Position = 0;
        return buffer;
    }

    public async Task<bool> ExistsAsync(string key, CancellationToken ct = default)
    {
        try
        {
            await _client.GetObjectMetadataAsync(_bucket, key, ct);
            return true;
        }
        catch (AmazonS3Exception e) when (e.StatusCode == System.Net.HttpStatusCode.NotFound)
        {
            return false;
        }
    }

    public async Task DeleteAsync(string key, CancellationToken ct = default) =>
        await _client.DeleteObjectAsync(_bucket, key, ct);

    public async Task<IReadOnlyList<StorageItem>> ListAsync(string prefix, CancellationToken ct = default)
    {
        var items = new List<StorageItem>();
        var request = new ListObjectsV2Request { BucketName = _bucket, Prefix = prefix };

        ListObjectsV2Response response;
        do
        {
            response = await _client.ListObjectsV2Async(request, ct);
            items.AddRange(response.S3Objects.Select(o =>
                new StorageItem(o.Key, o.Size ?? 0, o.LastModified)));
            request.ContinuationToken = response.NextContinuationToken;
        }
        while (response.IsTruncated == true);

        return items;
    }

    public Task<string> GetUrlAsync(string key, TimeSpan? lifetime = null, CancellationToken ct = default)
    {
        var url = _client.GetPreSignedURL(new GetPreSignedUrlRequest
        {
            BucketName = _bucket,
            Key = key,
            Expires = DateTime.UtcNow.Add(lifetime ?? TimeSpan.FromHours(1)),
            Verb = HttpVerb.GET
        });

        return Task.FromResult(url);
    }

    private async Task EnsureBucketAsync(CancellationToken ct)
    {
        if (_bucketChecked)
        {
            return;
        }

        try
        {
            await _client.PutBucketAsync(new PutBucketRequest { BucketName = _bucket }, ct);
        }
        catch (AmazonS3Exception e) when (
            e.ErrorCode is "BucketAlreadyOwnedByYou" or "BucketAlreadyExists")
        {
            // Already there, which is the outcome we wanted.
        }

        _bucketChecked = true;
    }
}
