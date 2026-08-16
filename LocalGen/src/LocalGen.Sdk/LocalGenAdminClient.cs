using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using LocalGen.Core.Models;
using LocalGen.Core.Protocol;

namespace LocalGen.Sdk;

/// <summary>
/// The management half of the API: models, engines, status and metrics.
/// </summary>
/// <remarks>
/// Kept separate from <see cref="LocalGenClient"/>'s OpenAI-compatible surface so it is obvious
/// which calls are portable to other providers and which are LocalGen-only.
/// </remarks>
public sealed class LocalGenAdminClient
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    private readonly HttpClient _http;

    internal LocalGenAdminClient(HttpClient http) => _http = http;

    /// <summary>Service state, uptime and which models are resident.</summary>
    public async Task<ServiceStatusDto> GetStatusAsync(CancellationToken cancellationToken = default) =>
        await _http.GetFromJsonAsync<ServiceStatusDto>("api/status", SerializerOptions, cancellationToken)
            .ConfigureAwait(false)
        ?? throw new LocalGenClientException("The server returned an empty status.");

    /// <summary>Installed models with LocalGen's own metadata, richer than <c>/v1/models</c>.</summary>
    public async Task<IReadOnlyList<ModelDescriptor>> ListModelsAsync(
        CancellationToken cancellationToken = default) =>
        await _http.GetFromJsonAsync<List<ModelDescriptor>>("api/models", SerializerOptions, cancellationToken)
            .ConfigureAwait(false) ?? [];

    /// <summary>
    /// Downloads a model, yielding progress as it goes. The server streams updates because a pull
    /// runs for minutes.
    /// </summary>
    public async IAsyncEnumerable<PullUpdate> PullModelAsync(
        string reference,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "api/models/pull")
        {
            Content = JsonContent.Create(new { reference }, options: SerializerOptions)
        };

        using var response = await _http
            .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);

        await LocalGenClient.EnsureSuccessAsync(response, cancellationToken).ConfigureAwait(false);

        await using var stream = await response.Content
            .ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);

        await foreach (var update in ServerSentEvents
            .ReadAsync<PullUpdate>(stream, cancellationToken)
            .ConfigureAwait(false))
        {
            yield return update;
        }
    }

    public async Task<bool> RemoveModelAsync(string id, CancellationToken cancellationToken = default)
    {
        using var response = await _http
            .DeleteAsync($"api/models/{Uri.EscapeDataString(id)}", cancellationToken)
            .ConfigureAwait(false);

        return response.IsSuccessStatusCode;
    }

    /// <summary>Loads a model into memory ahead of the first request.</summary>
    public async Task LoadModelAsync(string id, CancellationToken cancellationToken = default)
    {
        using var response = await _http
            .PostAsync($"api/models/{Uri.EscapeDataString(id)}/load", null, cancellationToken)
            .ConfigureAwait(false);

        await LocalGenClient.EnsureSuccessAsync(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<bool> UnloadModelAsync(string id, CancellationToken cancellationToken = default)
    {
        using var response = await _http
            .PostAsync($"api/models/{Uri.EscapeDataString(id)}/unload", null, cancellationToken)
            .ConfigureAwait(false);

        return response.IsSuccessStatusCode;
    }

    /// <summary>The model's Modelfile, as text.</summary>
    public async Task<string> GetModelfileAsync(string id, CancellationToken cancellationToken = default) =>
        await _http.GetStringAsync($"api/models/{Uri.EscapeDataString(id)}/modelfile", cancellationToken)
            .ConfigureAwait(false);

    /// <summary>Installed backends, their availability and the recommendation for this machine.</summary>
    public async Task<EnginesDto> GetEnginesAsync(CancellationToken cancellationToken = default) =>
        await _http.GetFromJsonAsync<EnginesDto>("api/engines", SerializerOptions, cancellationToken)
            .ConfigureAwait(false)
        ?? throw new LocalGenClientException("The server returned no engine information.");

    /// <summary>Inference and resource telemetry for the monitoring dashboard.</summary>
    public async Task<JsonElement> GetMetricsAsync(CancellationToken cancellationToken = default) =>
        await _http.GetFromJsonAsync<JsonElement>("api/metrics", SerializerOptions, cancellationToken)
            .ConfigureAwait(false);

    /// <summary>Recent server log entries.</summary>
    public async Task<IReadOnlyList<LogEntryDto>> GetLogsAsync(
        int limit = 200,
        CancellationToken cancellationToken = default) =>
        await _http.GetFromJsonAsync<List<LogEntryDto>>(
            $"api/logs?limit={limit}", SerializerOptions, cancellationToken).ConfigureAwait(false) ?? [];

    /// <summary>Searches the remote model catalogues.</summary>
    public async Task<IReadOnlyList<CatalogEntry>> SearchCatalogAsync(
        string query,
        int limit = 25,
        CancellationToken cancellationToken = default) =>
        await _http.GetFromJsonAsync<List<CatalogEntry>>(
            $"api/catalog/search?q={Uri.EscapeDataString(query)}&limit={limit}",
            SerializerOptions,
            cancellationToken).ConfigureAwait(false) ?? [];

    /// <summary>Full detail for a catalogue entry, including its downloadable quantizations.</summary>
    public async Task<CatalogEntry?> GetCatalogEntryAsync(
        string reference,
        CancellationToken cancellationToken = default)
    {
        using var response = await _http
            .GetAsync($"api/catalog/entry?reference={Uri.EscapeDataString(reference)}", cancellationToken)
            .ConfigureAwait(false);

        return response.IsSuccessStatusCode
            ? await response.Content
                .ReadFromJsonAsync<CatalogEntry>(SerializerOptions, cancellationToken).ConfigureAwait(false)
            : null;
    }
}

public sealed record ServiceStatusDto
{
    public string Status { get; init; } = string.Empty;

    public string Version { get; init; } = string.Empty;

    public DateTimeOffset StartedAt { get; init; }

    public TimeSpan Uptime { get; init; }

    public string BaseUrl { get; init; } = string.Empty;

    public bool OfflineMode { get; init; }

    public string DataDirectory { get; init; } = string.Empty;

    public IReadOnlyList<LoadedModelDto> LoadedModels { get; init; } = [];
}

public sealed record LoadedModelDto
{
    public string Id { get; init; } = string.Empty;

    public string Engine { get; init; } = string.Empty;

    public string Device { get; init; } = string.Empty;

    public int ContextSize { get; init; }

    public DateTimeOffset LoadedAt { get; init; }

    public DateTimeOffset LastUsedAt { get; init; }

    public long RequestCount { get; init; }

    public int ActiveRequests { get; init; }
}

public sealed record EnginesDto
{
    public IReadOnlyList<EngineDto> Engines { get; init; } = [];

    public string RecommendedEngine { get; init; } = string.Empty;

    public string RecommendedDevice { get; init; } = string.Empty;

    public string Rationale { get; init; } = string.Empty;
}

public sealed record EngineDto
{
    public string Kind { get; init; } = string.Empty;

    public string DisplayName { get; init; } = string.Empty;

    public string Description { get; init; } = string.Empty;

    public string Recommendation { get; init; } = string.Empty;

    public bool IsAvailable { get; init; }

    public bool IsDefault { get; init; }

    public string UnavailableReason { get; init; } = string.Empty;

    public string Version { get; init; } = string.Empty;

    public IReadOnlyList<string> Devices { get; init; } = [];

    public IReadOnlyList<string> Formats { get; init; } = [];

    public bool SupportsGrammar { get; init; }

    public bool SupportsEmbeddings { get; init; }

    public bool SupportsMultiGpu { get; init; }
}

public sealed record LogEntryDto
{
    public DateTimeOffset Timestamp { get; init; }

    public string Level { get; init; } = string.Empty;

    public string Category { get; init; } = string.Empty;

    public string Message { get; init; } = string.Empty;

    public string? Exception { get; init; }
}

/// <summary>One progress frame from a model pull.</summary>
public sealed record PullUpdate
{
    /// <summary>Empty while downloading; <c>success</c> or <c>error</c> on the final frame.</summary>
    public string? Status { get; init; }

    public string? ModelId { get; init; }

    public string? FileName { get; init; }

    public long BytesDownloaded { get; init; }

    public long TotalBytes { get; init; }

    public double BytesPerSecond { get; init; }

    public string? Error { get; init; }

    public ModelDescriptor? Model { get; init; }

    public double Percentage => TotalBytes > 0
        ? Math.Clamp(BytesDownloaded * 100.0 / TotalBytes, 0, 100)
        : 0;

    public bool IsComplete => Status is "success" or "error";
}
