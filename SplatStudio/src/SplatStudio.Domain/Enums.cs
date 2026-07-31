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
    ExternalApi = 1,
    /// <summary>Same heuristic as <see cref="LocalHeuristic"/>, evaluated on the GPU via ILGPU.</summary>
    Gpu = 2,
    /// <summary>A hosted multi-view 3DGS service reached over HTTP.</summary>
    HostedPhotoreal = 3,
    /// <summary>A hosted image-to-mesh service (TRELLIS, Hunyuan3D, Rodin and friends).</summary>
    HostedMesh = 4,

    /// <summary>
    /// Demo geometry written by the seeder, not produced from an image by any engine. Kept
    /// distinct so the gallery never implies a model generated something it didn't.
    /// </summary>
    SampleData = 5
}

/// <summary>
/// What the uploader asked for. This is a user-facing choice made at upload time, not an
/// implementation detail: the three modes produce genuinely different artifacts with very
/// different cost and quality, so the scene records which one it came from.
/// </summary>
public enum ConversionMode
{
    /// <summary>
    /// The offline depth heuristic. Instant, free, runs on this machine, and is a 2.5D
    /// approximation rather than reconstruction.
    /// </summary>
    HeuristicSplat = 0,

    /// <summary>
    /// Real multi-view-quality Gaussian splatting via a hosted service. Produces a .splat
    /// like <see cref="HeuristicSplat"/>, so the same viewer renders it.
    /// </summary>
    PhotorealSplat = 1,

    /// <summary>
    /// Image-to-3D-object generation via a hosted service. Produces a textured mesh, not a
    /// point cloud, so it is stored and rendered down a separate path.
    /// </summary>
    Mesh = 2
}

/// <summary>
/// Which kind of file a conversion produced. Splats and meshes need different storage
/// prefixes and different browser viewers, so the pipeline branches on this rather than
/// inferring it from the mode.
/// </summary>
public enum ConversionArtifactKind
{
    Splat = 0,
    Mesh = 1
}
