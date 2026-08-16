using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using LocalGen.Core;
using LocalGen.Core.Models;
using LocalGen.Runtime.Models;
using Microsoft.Extensions.Logging;

namespace LocalGen.Runtime.Downloads;

/// <summary>
/// Downloads GGUF weights from HuggingFace.
/// </summary>
/// <remarks>
/// Model files routinely run to several gigabytes over connections that drop, so transfers are
/// resumable: bytes land in a <c>.part</c> file and a retry sends a Range header for whatever is
/// already on disk instead of starting again.
/// </remarks>
public sealed class HuggingFaceDownloader : IModelDownloader
{
    public const string HttpClientName = "huggingface";

    private const string ApiBase = "https://huggingface.co/api";
    private const string ResolveBase = "https://huggingface.co";

    /// <summary>Quantizations tried in order when a reference names a repo but not a file.</summary>
    private static readonly string[] PreferredQuantizations =
        ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q6_K", "Q8_0"];

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILogger<HuggingFaceDownloader> _logger;

    public HuggingFaceDownloader(
        IHttpClientFactory httpClientFactory,
        ILogger<HuggingFaceDownloader> logger)
    {
        _httpClientFactory = httpClientFactory;
        _logger = logger;
    }

    public async ValueTask<DownloadResult> DownloadAsync(
        ModelReference reference,
        string destinationDirectory,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (reference.Provider != ModelReference.HuggingFace)
        {
            throw new LocalGenException(
                $"HuggingFaceDownloader cannot handle '{reference.Provider}' references.");
        }

        using var http = _httpClientFactory.CreateClient(HttpClientName);

        // Listed once and used twice: to choose the weights when the reference names none, and to
        // spot a projector. Naming a file explicitly used to skip the listing entirely, which is
        // how a vision model could be pulled without the half of it that reads images.
        var files = await ListFilesAsync(http, reference.Repository, cancellationToken).ConfigureAwait(false);

        var fileName = string.IsNullOrEmpty(reference.File)
            ? ChooseFile(files, reference.Repository)
            : reference.File;

        Directory.CreateDirectory(destinationDirectory);

        var modelId = BuildModelId(reference.Repository, Path.GetFileName(fileName));

        // A sharded model is one logical set of weights split across numbered files. llama.cpp
        // opens the first shard and finds the rest beside it, so every part has to be fetched or
        // loading fails partway through with a missing-tensor error.
        var parts = ShardNaming.Expand(fileName);

        if (parts.Count > 1)
        {
            _logger.LogInformation(
                "{File} is part 1 of {Count} shards; fetching all of them.",
                Path.GetFileName(fileName), parts.Count);
        }

        string? firstShardPath = null;

        foreach (var part in parts)
        {
            var localName = Path.GetFileName(part);
            var destination = Path.Combine(destinationDirectory, localName);

            firstShardPath ??= destination;

            if (File.Exists(destination))
            {
                var existing = new FileInfo(destination).Length;

                _logger.LogInformation("{File} is already present; skipping download.", localName);

                progress?.Report(new DownloadProgress
                {
                    ModelId = modelId,
                    FileName = localName,
                    Status = "already downloaded",
                    BytesDownloaded = existing,
                    TotalBytes = existing
                });

                continue;
            }

            var url = $"{ResolveBase}/{reference.Repository}/resolve/main/{part}?download=true";

            await TransferAsync(http, url, destination, modelId, localName, progress, cancellationToken)
                .ConfigureAwait(false);
        }

        var projectorPath = await DownloadProjectorAsync(
            http, reference.Repository, files, firstShardPath!, modelId, progress, cancellationToken)
            .ConfigureAwait(false);

        return new DownloadResult
        {
            // The store registers the first shard; llama.cpp resolves the others from it.
            FilePath = firstShardPath!,
            ProjectorPath = projectorPath ?? string.Empty,
            ModelId = modelId,
            Publisher = reference.Repository.Split('/')[0]
        };
    }

    /// <summary>
    /// Fetches the multimodal projector, if the repository has one, and saves it under the name
    /// that pairs it with these weights.
    /// </summary>
    /// <remarks>
    /// A vision model is two files and neither names the other. Pulling only the language weights
    /// gives a model that loads, answers text, and silently ignores every image — so the projector
    /// comes along automatically rather than waiting to be asked for.
    /// </remarks>
    private async Task<string?> DownloadProjectorAsync(
        HttpClient http,
        string repository,
        IReadOnlyList<string> files,
        string modelPath,
        string modelId,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var projectors = files
            .Where(f => f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            .Where(ProjectorNaming.IsProjector)
            .ToList();

        if (projectors.Count == 0)
        {
            return null;
        }

        // Repositories often ship the projector at several precisions. It is small next to the
        // weights, so the most accurate one is worth the bytes.
        var chosen = projectors.FirstOrDefault(
                         f => f.Contains("f16", StringComparison.OrdinalIgnoreCase))
                     ?? projectors.FirstOrDefault(
                         f => f.Contains("f32", StringComparison.OrdinalIgnoreCase))
                     ?? projectors[0];

        var localName = ProjectorNaming.CanonicalNameFor(Path.GetFileName(modelPath));
        var destination = Path.Combine(Path.GetDirectoryName(modelPath)!, localName);

        if (File.Exists(destination))
        {
            _logger.LogInformation("Projector {File} is already present; skipping.", localName);
            return destination;
        }

        _logger.LogInformation(
            "{Repository} is a vision model; fetching its projector {File}.", repository, chosen);

        var url = $"{ResolveBase}/{repository}/resolve/main/{chosen}?download=true";

        try
        {
            await TransferAsync(http, url, destination, modelId, localName, progress, cancellationToken)
                .ConfigureAwait(false);

            return destination;
        }
        catch (HttpRequestException ex)
        {
            // The weights are already on disk and usable for text. Failing the whole pull over the
            // projector would throw away a download that may have taken an hour.
            _logger.LogWarning(
                ex, "Could not fetch the projector for {Repository}; the model will be text-only.",
                repository);

            return null;
        }
    }

    /// <summary>
    /// Streams the file to a <c>.part</c> sibling, resuming from whatever is already there.
    /// </summary>
    private async Task TransferAsync(
        HttpClient http,
        string url,
        string destination,
        string modelId,
        string fileName,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var partPath = destination + ".part";
        var existingBytes = File.Exists(partPath) ? new FileInfo(partPath).Length : 0;

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (existingBytes > 0)
        {
            request.Headers.Range = new RangeHeaderValue(existingBytes, null);
            _logger.LogInformation("Resuming {File} at {Bytes:N0} bytes", fileName, existingBytes);
        }

        using var response = await http
            .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);

        // The server ignored the Range header, so start over rather than concatenating garbage.
        if (existingBytes > 0 && response.StatusCode != HttpStatusCode.PartialContent)
        {
            _logger.LogWarning("Server did not honour the resume request; restarting {File}.", fileName);
            existingBytes = 0;
            File.Delete(partPath);
        }

        response.EnsureSuccessStatusCode();

        var totalBytes = (response.Content.Headers.ContentLength ?? 0) + existingBytes;

        await using (var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
        await using (var target = new FileStream(
            partPath,
            existingBytes > 0 ? FileMode.Append : FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 1 << 20,
            useAsync: true))
        {
            var buffer = new byte[1 << 20];
            var downloaded = existingBytes;
            var stopwatch = Stopwatch.StartNew();
            var lastReport = TimeSpan.Zero;

            int read;
            while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await target.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                downloaded += read;

                // Reporting on every chunk would flood the UI; a few updates per second is enough.
                if (stopwatch.Elapsed - lastReport > TimeSpan.FromMilliseconds(250))
                {
                    lastReport = stopwatch.Elapsed;
                    progress?.Report(new DownloadProgress
                    {
                        ModelId = modelId,
                        FileName = fileName,
                        BytesDownloaded = downloaded,
                        TotalBytes = totalBytes,
                        BytesPerSecond = (downloaded - existingBytes) / Math.Max(stopwatch.Elapsed.TotalSeconds, 0.001),
                        Status = "downloading"
                    });
                }
            }
        }

        File.Move(partPath, destination, overwrite: true);

        progress?.Report(new DownloadProgress
        {
            ModelId = modelId,
            FileName = fileName,
            BytesDownloaded = totalBytes,
            TotalBytes = totalBytes,
            Status = "complete"
        });

        _logger.LogInformation("Downloaded {File} to {Path}", fileName, destination);
    }

    /// <summary>
    /// Picks a GGUF file from a repository, preferring the quantization that gives the best
    /// size/quality trade-off for local inference.
    /// </summary>
    private string ChooseFile(IReadOnlyList<string> files, string repository)
    {
        // Sharded models collapse to their first file, which stands for the whole set — the
        // downloader fetches the rest, so a split model is a normal choice here. Projectors are
        // excluded: they hold an image encoder and no language weights, so one can never be the
        // model itself.
        var ggufFiles = ShardNaming
            .CollapseShards(files
                .Where(f => f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                .Where(f => !ProjectorNaming.IsProjector(f)))
            .ToList();

        if (ggufFiles.Count == 0)
        {
            throw new LocalGenException(
                $"'{repository}' contains no GGUF weights. " +
                "Name a file explicitly, e.g. huggingface:owner/repo/model.Q4_K_M.gguf");
        }

        foreach (var quantization in PreferredQuantizations)
        {
            var match = ggufFiles.FirstOrDefault(
                f => f.Contains(quantization, StringComparison.OrdinalIgnoreCase));

            if (match is not null)
            {
                _logger.LogInformation("Selected {File} from {Repository}", match, repository);
                return match;
            }
        }

        return ggufFiles[0];
    }

    private static async Task<IReadOnlyList<string>> ListFilesAsync(
        HttpClient http,
        string repository,
        CancellationToken cancellationToken)
    {
        using var response = await http
            .GetAsync($"{ApiBase}/models/{repository}", cancellationToken)
            .ConfigureAwait(false);

        if (response.StatusCode == HttpStatusCode.NotFound)
        {
            throw new LocalGenException($"HuggingFace repository '{repository}' was not found.");
        }

        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        if (!document.RootElement.TryGetProperty("siblings", out var siblings))
        {
            return [];
        }

        return [.. siblings
            .EnumerateArray()
            .Select(s => s.TryGetProperty("rfilename", out var name) ? name.GetString() : null)
            .Where(static name => !string.IsNullOrEmpty(name))
            .Select(static name => name!)];
    }

    /// <summary>
    /// Builds an id such as <c>qwen2.5-7b-instruct:q4_k_m</c> from the repo and file names,
    /// so pulled models read the same way as they do in other local inference tools.
    /// </summary>
    private static string BuildModelId(string repository, string fileName)
    {
        var repoName = repository.Split('/')[^1];

        foreach (var suffix in (string[])["-GGUF", "-gguf", ".GGUF"])
        {
            if (repoName.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
            {
                repoName = repoName[..^suffix.Length];
                break;
            }
        }

        var quantization = Quantization.FromFileName(fileName);
        var tag = quantization == Quantization.Unknown
            ? "latest"
            : quantization.Name.ToLowerInvariant();

        return $"{repoName.ToLowerInvariant()}:{tag}";
    }
}
