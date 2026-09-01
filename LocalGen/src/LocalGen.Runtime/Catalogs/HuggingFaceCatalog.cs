using System.Net;
using System.Text.Json;
using LocalGen.Core.Configuration;
using LocalGen.Core.Models;
using LocalGen.Runtime.Downloads;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalGen.Runtime.Catalogs;

/// <summary>
/// Model discovery against the HuggingFace Hub, backing the Model Gallery screen.
/// Search is restricted to GGUF repositories because those are what LocalGen can serve directly.
/// </summary>
public sealed class HuggingFaceCatalog : IModelCatalog
{
    private const string ApiBase = "https://huggingface.co/api";

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly LocalGenOptions _options;
    private readonly ILogger<HuggingFaceCatalog> _logger;

    public HuggingFaceCatalog(
        IHttpClientFactory httpClientFactory,
        IOptions<LocalGenOptions> options,
        ILogger<HuggingFaceCatalog> logger)
    {
        _httpClientFactory = httpClientFactory;
        _options = options.Value;
        _logger = logger;
    }

    public string Provider => "huggingface";

    public string DisplayName => "Hugging Face";

    public async ValueTask<IReadOnlyList<CatalogEntry>> SearchAsync(
        string query,
        int limit = 25,
        CancellationToken cancellationToken = default)
    {
        if (_options.Runtime.OfflineMode)
        {
            _logger.LogDebug("Offline mode is on; returning no catalogue results.");
            return [];
        }

        using var http = _httpClientFactory.CreateClient(HuggingFaceDownloader.HttpClientName);

        var url = $"{ApiBase}/models" +
                  $"?search={Uri.EscapeDataString(query)}" +
                  "&filter=gguf" +
                  "&sort=downloads&direction=-1" +
                  $"&limit={Math.Clamp(limit, 1, 100)}";

        try
        {
            using var response = await http.GetAsync(url, cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            await using var stream = await response.Content
                .ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            using var document = await JsonDocument
                .ParseAsync(stream, cancellationToken: cancellationToken).ConfigureAwait(false);

            return [.. document.RootElement.EnumerateArray().Select(ReadSummary)];
        }
        catch (HttpRequestException ex)
        {
            _logger.LogWarning(ex, "Hugging Face search failed for '{Query}'", query);
            return [];
        }
    }

    public async ValueTask<CatalogEntry?> GetAsync(
        string reference,
        CancellationToken cancellationToken = default)
    {
        if (_options.Runtime.OfflineMode)
        {
            return null;
        }

        var repository = reference.StartsWith("huggingface:", StringComparison.OrdinalIgnoreCase)
            ? reference["huggingface:".Length..]
            : reference;

        using var http = _httpClientFactory.CreateClient(HuggingFaceDownloader.HttpClientName);

        try
        {
            // blobs=true is what puts a "size" on each sibling; without it every variant reports
            // zero bytes and the gallery cannot tell a 4 GB quantization from a 40 GB one.
            using var response = await http
                .GetAsync($"{ApiBase}/models/{repository}?blobs=true", cancellationToken)
                .ConfigureAwait(false);

            if (response.StatusCode == HttpStatusCode.NotFound)
            {
                return null;
            }

            response.EnsureSuccessStatusCode();

            await using var stream = await response.Content
                .ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            using var document = await JsonDocument
                .ParseAsync(stream, cancellationToken: cancellationToken).ConfigureAwait(false);

            var entry = ReadSummary(document.RootElement);
            return entry with { Variants = ReadVariants(document.RootElement, repository) };
        }
        catch (HttpRequestException ex)
        {
            _logger.LogWarning(ex, "Could not load Hugging Face model '{Repository}'", repository);
            return null;
        }
    }

    private CatalogEntry ReadSummary(JsonElement element)
    {
        var id = element.TryGetProperty("id", out var idElement) ? idElement.GetString() ?? string.Empty : string.Empty;
        var owner = id.Contains('/') ? id.Split('/')[0] : string.Empty;

        return new CatalogEntry
        {
            Reference = $"huggingface:{id}",
            Name = id,
            Publisher = owner,
            Provider = Provider,
            Downloads = element.TryGetProperty("downloads", out var downloads) ? downloads.GetInt64() : 0,
            Likes = element.TryGetProperty("likes", out var likes) ? likes.GetInt64() : 0,
            UpdatedAt = element.TryGetProperty("lastModified", out var modified)
                && modified.TryGetDateTimeOffset(out var timestamp)
                ? timestamp
                : null,
            Tags = element.TryGetProperty("tags", out var tags) && tags.ValueKind == JsonValueKind.Array
                ? [.. tags.EnumerateArray()
                    .Select(static t => t.GetString())
                    .Where(static t => !string.IsNullOrEmpty(t))
                    .Select(static t => t!)]
                : []
        };
    }

    /// <summary>
    /// Turns the repository file listing into the quantization choices shown in the model
    /// browser. A split model appears once, as its first shard, with the size of the whole set.
    /// </summary>
    private static IReadOnlyList<CatalogVariant> ReadVariants(JsonElement element, string repository)
    {
        if (!element.TryGetProperty("siblings", out var siblings) ||
            siblings.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        // Sizes are gathered first so a sharded model can report the total of all its parts
        // rather than the size of its first file, which would understate it badly.
        var sizes = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
        var names = new List<string>();

        foreach (var sibling in siblings.EnumerateArray())
        {
            if (!sibling.TryGetProperty("rfilename", out var nameElement))
            {
                continue;
            }

            var fileName = nameElement.GetString();

            if (string.IsNullOrEmpty(fileName) ||
                !fileName.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            names.Add(fileName);
            sizes[fileName] = sibling.TryGetProperty("size", out var size) && size.TryGetInt64(out var bytes)
                ? bytes
                : 0;
        }

        var variants = new List<CatalogVariant>();

        foreach (var fileName in Downloads.ShardNaming.CollapseShards(names))
        {
            var parts = Downloads.ShardNaming.Expand(fileName);
            var totalBytes = parts.Sum(part => sizes.GetValueOrDefault(part));

            variants.Add(new CatalogVariant
            {
                Reference = $"huggingface:{repository}/{fileName}",
                FileName = parts.Count > 1 ? $"{fileName}  (+{parts.Count - 1} shards)" : fileName,
                SizeBytes = totalBytes,
                Quantization = Quantization.FromFileName(fileName)
            });
        }

        // Smallest first, which is the order people scan when choosing a quantization.
        return [.. variants.OrderBy(static v => v.Quantization.BitsPerWeight).ThenBy(static v => v.FileName)];
    }
}
