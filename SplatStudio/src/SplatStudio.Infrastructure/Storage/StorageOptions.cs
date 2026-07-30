namespace SplatStudio.Infrastructure.Storage;

/// <summary>Root "Storage" configuration section.</summary>
public class StorageOptions
{
    public const string SectionName = "Storage";

    /// <summary>FileSystem | AzureBlob | S3 | MinIO</summary>
    public string Provider { get; set; } = "FileSystem";

    public FileSystemStorageOptions FileSystem { get; set; } = new();
    public AzureBlobStorageOptions AzureBlob { get; set; } = new();
    public S3CompatibleStorageOptions S3 { get; set; } = new();
    public S3CompatibleStorageOptions MinIO { get; set; } = new();
}

public class FileSystemStorageOptions
{
    /// <summary>Folder on disk (relative to the web app's content root) where files are stored.</summary>
    public string RootPath { get; set; } = "App_Data/storage";

    /// <summary>URL prefix the app serves these files under (see the static-file middleware mapping in Program.cs).</summary>
    public string PublicBasePath { get; set; } = "/media";
}

public class AzureBlobStorageOptions
{
    public string ConnectionString { get; set; } = string.Empty;
    public string ContainerName { get; set; } = "splatstudio";
}

/// <summary>
/// Shared options shape for both real AWS S3 and any S3-compatible service
/// (MinIO, Cloudflare R2, Wasabi, ...). Leave ServiceUrl empty for AWS.
/// </summary>
public class S3CompatibleStorageOptions
{
    public string ServiceUrl { get; set; } = string.Empty;
    public string AccessKey { get; set; } = string.Empty;
    public string SecretKey { get; set; } = string.Empty;
    public string BucketName { get; set; } = "splatstudio";
    public string Region { get; set; } = "us-east-1";

    /// <summary>MinIO and most self-hosted S3-compatible servers require path-style addressing.</summary>
    public bool ForcePathStyle { get; set; } = true;

    /// <summary>If set, used to build public URLs instead of asking the SDK to build one (useful behind a CDN/reverse proxy).</summary>
    public string? PublicBaseUrl { get; set; }
}
