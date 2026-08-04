using SplatStudio.Domain.Enums;

namespace SplatStudio.Domain.Entities;

/// <summary>
/// A user-uploaded source photo. The binary bytes never live in the
/// database — only the logical storage key (resolved through IStorageService)
/// and lightweight metadata used by the UI.
/// </summary>
public class ImageAsset
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string UserId { get; set; } = string.Empty;

    public string OriginalFileName { get; set; } = string.Empty;
    public string ContentType { get; set; } = "image/jpeg";
    public long SizeBytes { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }

    /// <summary>Storage key for the original upload (provider-agnostic).</summary>
    public string StorageKey { get; set; } = string.Empty;
    /// <summary>Storage key for a small downscaled thumbnail used in the gallery.</summary>
    public string ThumbnailStorageKey { get; set; } = string.Empty;

    public DateTime UploadedAtUtc { get; set; } = DateTime.UtcNow;

    public SplatScene? SplatScene { get; set; }
}

/// <summary>
/// A generated 3D Gaussian-splat "world" produced from one <see cref="ImageAsset"/>.
/// This is the primary aggregate shown in the gallery and the 3D viewer.
/// </summary>
public class SplatScene
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public Guid ImageAssetId { get; set; }
    public ImageAsset? ImageAsset { get; set; }

    public string UserId { get; set; } = string.Empty;

    public string Title { get; set; } = string.Empty;
    public string? Description { get; set; }

    public SplatStatus Status { get; set; } = SplatStatus.Queued;
    public SplatEngineType Engine { get; set; } = SplatEngineType.LocalHeuristic;
    public string? ErrorMessage { get; set; }

    /// <summary>What the uploader asked for. Chosen at upload time and never changed after.</summary>
    public ConversionMode Mode { get; set; } = ConversionMode.HeuristicSplat;

    /// <summary>
    /// Whether this scene's output is a point cloud or a mesh. Derived from the mode when the
    /// conversion completes, and stored so the viewer doesn't have to re-derive it.
    /// </summary>
    public ConversionArtifactKind ArtifactKind { get; set; } = ConversionArtifactKind.Splat;

    /// <summary>Storage key of the generated binary .splat file. Null for mesh scenes.</summary>
    public string? SplatStorageKey { get; set; }

    /// <summary>Storage key of the generated mesh (.glb). Null for splat scenes.</summary>
    public string? MeshStorageKey { get; set; }

    /// <summary>
    /// The storage key holding this scene's renderable output, whichever kind it is. The
    /// viewer and the delete path both want "the artifact" without caring which column.
    /// </summary>
    public string? OutputStorageKey =>
        ArtifactKind == ConversionArtifactKind.Mesh ? MeshStorageKey : SplatStorageKey;
    /// <summary>Number of Gaussian splats baked into the file — shown in the UI as a quality indicator.</summary>
    public int PointCount { get; set; }

    /// <summary>Wall-clock time the conversion took, in milliseconds. 0 until it completes.</summary>
    public int ConversionMilliseconds { get; set; }

    public bool IsPublic { get; set; } = true;
    public int ViewCount { get; set; }

    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
    public DateTime? CompletedAtUtc { get; set; }

    public List<Comment> Comments { get; set; } = new();
    public List<Rating> Ratings { get; set; } = new();

    public double AverageRating => Ratings.Count == 0 ? 0 : Math.Round(Ratings.Average(r => r.Stars), 1);
}

public class Comment
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid SplatSceneId { get; set; }
    public SplatScene? SplatScene { get; set; }

    public string UserId { get; set; } = string.Empty;
    public string AuthorDisplayName { get; set; } = string.Empty;

    public string Content { get; set; } = string.Empty;
    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}

public class Rating
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid SplatSceneId { get; set; }
    public SplatScene? SplatScene { get; set; }

    public string UserId { get; set; } = string.Empty;

    /// <summary>1-5 stars. One rating per user per scene (enforced by a unique index).</summary>
    public int Stars { get; set; }
    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}
