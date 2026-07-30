using SplatStudio.Domain.Enums;

namespace SplatStudio.Application.Abstractions;

/// <summary>
/// Provider-agnostic object storage port. Concrete implementations live in
/// Infrastructure (FileSystem, AzureBlob, S3, MinIO) and are selected at
/// startup by <c>StorageServiceFactory</c> based on configuration. No other
/// layer is allowed to reference a concrete storage SDK directly.
/// </summary>
public interface IStorageService
{
    StorageProviderType ProviderType { get; }

    /// <summary>Writes content under the given logical key (e.g. "images/2026/06/abc.jpg").</summary>
    Task SaveAsync(string key, Stream content, string contentType, CancellationToken ct = default);

    Task<Stream> OpenReadAsync(string key, CancellationToken ct = default);

    Task DeleteAsync(string key, CancellationToken ct = default);

    Task<bool> ExistsAsync(string key, CancellationToken ct = default);

    /// <summary>
    /// A URL the browser can load directly (public container/bucket URL, a
    /// pre-signed S3/MinIO URL, or an internal app route for FileSystem).
    /// </summary>
    Task<string> GetPublicUrlAsync(string key, CancellationToken ct = default);
}

/// <summary>
/// Result of converting one source image into a binary .splat point cloud.
/// </summary>
public record GaussianSplatGenerationResult(
    bool Success,
    byte[]? SplatFileBytes,
    int PointCount,
    string? ErrorMessage);

/// <summary>
/// Port for the actual image -> Gaussian splat reconstruction step.
/// <see cref="SplatStudio.Infrastructure.Splatting.LocalHeuristicSplatEngine"/> ships a
/// fast, fully offline, single-image depth-heuristic implementation. For
/// production-grade multi-view reconstruction, implement this interface
/// against an external trainer (nerfstudio/gsplat, Luma AI, KIRI Engine,
/// etc.) — see <see cref="SplatStudio.Infrastructure.Splatting.ExternalApiSplatEngine"/>
/// for the integration point.
/// </summary>
public interface IGaussianSplatEngine
{
    SplatEngineType EngineType { get; }

    Task<GaussianSplatGenerationResult> GenerateAsync(
        Stream imageStream,
        int maxOutputPoints,
        CancellationToken ct = default);
}

/// <summary>
/// Lightweight in-process work queue that decouples the (fast) HTTP upload
/// request from the (slower) splat generation step. Backed by a bounded
/// <see cref="System.Threading.Channels.Channel{T}"/> in Infrastructure and
/// drained by a single <c>BackgroundService</c>.
/// </summary>
public interface IConversionQueue
{
    void QueueSceneConversion(Guid splatSceneId);

    Task<Guid> DequeueAsync(CancellationToken ct);
}

/// <summary>
/// Push-style notification so open Blazor Server circuits (e.g. a gallery
/// page) can refresh themselves the instant a queued scene finishes
/// processing, without polling the database.
/// </summary>
public interface ISceneUpdateNotifier
{
    event Func<Guid, Task>? SceneUpdated;

    Task NotifySceneUpdatedAsync(Guid sceneId);
}

/// <summary>
/// Minimal mail port used for password-reset links. The default
/// implementation just logs/writes the message to disk so the app is
/// usable out of the box without SMTP credentials; configure
/// "Email:Provider": "Smtp" for real delivery.
/// </summary>
public interface IAppEmailSender
{
    Task SendAsync(string toEmail, string subject, string htmlBody, CancellationToken ct = default);
}

/// <summary>Result of inspecting + thumbnailing a freshly uploaded image.</summary>
public record ProcessedImageResult(int Width, int Height, byte[] ThumbnailBytes, string ThumbnailContentType);

/// <summary>
/// Upload-time image inspection port (dimensions + a small JPEG thumbnail
/// for the gallery grid). Kept behind a port so the Web layer never has to
/// reference an imaging SDK directly.
/// </summary>
public interface IImageProcessingService
{
    Task<ProcessedImageResult> ProcessUploadAsync(Stream imageStream, int thumbnailMaxDimension, CancellationToken ct = default);
}
