using LocalGen.Core.Models;
using LocalGen.Runtime.Models;

namespace LocalGen.Runtime.Downloads;

/// <summary>Where a downloaded model landed, plus what the store should call it.</summary>
public sealed record DownloadResult
{
    public required string FilePath { get; init; }

    public required string ModelId { get; init; }

    /// <summary>
    /// Where the multimodal projector landed, when the repository shipped one. Empty otherwise.
    /// </summary>
    public string ProjectorPath { get; init; } = string.Empty;

    public string Publisher { get; init; } = string.Empty;
}

/// <summary>Fetches model weights for a reference into the models directory.</summary>
public interface IModelDownloader
{
    ValueTask<DownloadResult> DownloadAsync(
        ModelReference reference,
        string destinationDirectory,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default);
}
