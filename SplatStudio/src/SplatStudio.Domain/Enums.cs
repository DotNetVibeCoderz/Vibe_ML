namespace SplatStudio.Domain.Enums;

/// <summary>
/// Lifecycle status of a single image -> Gaussian splat conversion job.
/// </summary>
public enum SplatStatus
{
    Queued = 0,
    Processing = 1,
    Completed = 2,
    Failed = 3
}

/// <summary>
/// Supported object/blob storage backends. The active backend is selected
/// at startup via configuration ("Storage:Provider") and resolved through
/// IStorageService — application code never depends on a concrete provider.
/// </summary>
public enum StorageProviderType
{
    FileSystem = 0,
    AzureBlob = 1,
    S3 = 2,
    MinIO = 3
}

/// <summary>
/// Supported relational database backends. Selected via configuration
/// ("Database:Provider"); EF Core's provider-agnostic LINQ surface means
/// the rest of the app is written once and runs against all three.
/// </summary>
public enum DatabaseProviderType
{
    Sqlite = 0,
    SqlServer = 1,
    MySql = 2
}

/// <summary>
/// Which engine produced a splat scene. Kept on the record so the gallery
/// can be transparent with users about reconstruction quality/provenance.
/// </summary>
public enum SplatEngineType
{
    LocalHeuristic = 0,
    ExternalApi = 1
}
