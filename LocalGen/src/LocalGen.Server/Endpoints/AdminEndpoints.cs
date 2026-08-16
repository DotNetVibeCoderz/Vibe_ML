using System.Text.Json;
using LocalGen.Core;
using LocalGen.Core.Configuration;
using LocalGen.Core.Diagnostics;
using LocalGen.Core.Models;
using LocalGen.Core.Protocol;
using LocalGen.Runtime.Diagnostics;
using LocalGen.Runtime.Engines;
using LocalGen.Runtime.Sessions;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalGen.Server.Endpoints;

/// <summary>
/// LocalGen's own management API, outside the OpenAI-compatible surface. The Admin Control
/// desktop app, the CLI and the Blazor UI are all built on these endpoints.
/// </summary>
public static class AdminEndpoints
{
    public static IEndpointRouteBuilder MapAdminEndpoints(this IEndpointRouteBuilder app)
    {
        var api = app.MapGroup("/api").WithTags("Admin");

        api.MapGet("/status", GetStatusAsync).WithSummary("Service status and loaded models.");
        api.MapGet("/health", static () => TypedResults.Ok(new { status = "ok" }))
           .WithSummary("Liveness probe.");

        api.MapGet("/engines", GetEnginesAsync).WithSummary("Installed backends and their availability.");
        api.MapGet("/metrics", GetMetrics).WithSummary("Inference and resource telemetry.");
        api.MapGet("/logs", GetLogs).WithSummary("Recent server log entries.");
        api.MapDelete("/logs", ClearLogs).WithSummary("Clears the in-memory log buffer.");

        var models = api.MapGroup("/models");
        models.MapGet("/", ListModelsAsync).WithSummary("Installed models.");
        models.MapPost("/pull", PullModelAsync).WithSummary("Downloads a model, streaming progress.");
        models.MapDelete("/{id}", RemoveModelAsync).WithSummary("Deletes an installed model.");
        models.MapPost("/{id}/load", LoadModelAsync).WithSummary("Loads a model into memory.");
        models.MapPost("/{id}/unload", UnloadModelAsync).WithSummary("Unloads a model from memory.");
        models.MapGet("/{id}/modelfile", GetModelfileAsync).WithSummary("The model's Modelfile.");

        var catalog = api.MapGroup("/catalog");
        catalog.MapGet("/search", SearchCatalogAsync).WithSummary("Searches remote model catalogues.");
        catalog.MapGet("/entry", GetCatalogEntryAsync).WithSummary("Details for one catalogue entry.");

        return app;
    }

    private static Ok<ServiceStatus> GetStatusAsync(
        ModelSessionManager sessions,
        IOptions<LocalGenOptions> options)
    {
        var settings = options.Value;

        return TypedResults.Ok(new ServiceStatus
        {
            Status = "running",
            Version = typeof(AdminEndpoints).Assembly.GetName().Version?.ToString(3) ?? "0.1.0",
            StartedAt = ProcessStartedAt,
            Uptime = DateTimeOffset.UtcNow - ProcessStartedAt,
            BaseUrl = settings.Server.BaseUrl,
            OfflineMode = settings.Runtime.OfflineMode,
            DataDirectory = settings.DataDirectory,
            LoadedModels =
            [
                .. sessions.Loaded.Select(static m => new LoadedModelInfo
                {
                    Id = m.ModelId,
                    Engine = m.Session.Engine.ToString(),
                    Device = m.Session.Device.ToString(),
                    ContextSize = m.Session.ContextSize,
                    LoadedAt = m.LoadedAt,
                    LastUsedAt = m.LastUsedAt,
                    RequestCount = Interlocked.Read(ref m.RequestCount),
                    ActiveRequests = m.ActiveRequests
                })
            ]
        });
    }

    private static async Task<Ok<EnginesResponse>> GetEnginesAsync(
        EngineRegistry registry,
        CancellationToken cancellationToken)
    {
        var statuses = await registry.GetStatusAsync(cancellationToken).ConfigureAwait(false);
        var recommendation = await registry.RecommendAsync(cancellationToken).ConfigureAwait(false);

        return TypedResults.Ok(new EnginesResponse
        {
            Engines =
            [
                .. statuses.Select(static s => new EngineInfo
                {
                    Kind = s.Descriptor.Kind.ToString(),
                    DisplayName = s.Descriptor.DisplayName,
                    Description = s.Descriptor.Description,
                    Recommendation = s.Descriptor.Recommendation,
                    IsAvailable = s.Availability.IsAvailable,
                    IsDefault = s.IsDefault,
                    UnavailableReason = s.Availability.Reason,
                    Version = s.Availability.Version,
                    Devices = [.. s.Availability.AvailableDevices.Select(static d => d.ToString())],
                    Formats = [.. s.Descriptor.Capabilities.Formats.Select(static f => f.ToString())],
                    SupportsGrammar = s.Descriptor.Capabilities.SupportsGrammar,
                    SupportsEmbeddings = s.Descriptor.Capabilities.SupportsEmbeddings,
                    SupportsMultiGpu = s.Descriptor.Capabilities.SupportsMultiGpu
                })
            ],
            RecommendedEngine = recommendation.Engine.ToString(),
            RecommendedDevice = recommendation.Device.ToString(),
            Rationale = recommendation.Rationale
        });
    }

    private static Ok<MetricsResponse> GetMetrics(InferenceMetrics metrics) =>
        TypedResults.Ok(new MetricsResponse
        {
            Summary = metrics.Summarize(),
            RecentInference = metrics.RecentInference(),
            RecentResources = metrics.RecentResources()
        });

    private static Ok<IReadOnlyList<LogEntry>> GetLogs(
        InMemoryLogStore store,
        int limit = 200,
        LogLevel level = LogLevel.Information) =>
        TypedResults.Ok(store.Recent(limit, level));

    private static NoContent ClearLogs(InMemoryLogStore store)
    {
        store.Clear();
        return TypedResults.NoContent();
    }

    private static async Task<Ok<IReadOnlyList<ModelDescriptor>>> ListModelsAsync(
        IModelStore store,
        CancellationToken cancellationToken) =>
        TypedResults.Ok(await store.ListAsync(cancellationToken).ConfigureAwait(false));

    /// <summary>
    /// Pulls a model, streaming progress as server-sent events. Downloads run for minutes, so a
    /// plain request/response would either time out or leave the caller blind.
    /// </summary>
    private static async Task PullModelAsync(
        HttpContext context,
        PullRequest request,
        IModelStore store,
        CancellationToken cancellationToken)
    {
        context.Response.Headers.ContentType = "text/event-stream";
        context.Response.Headers.CacheControl = "no-cache";
        context.Response.Headers["X-Accel-Buffering"] = "no";

        // The download callback fires from a background thread; the channel hands events back to
        // the request thread in order.
        var channel = System.Threading.Channels.Channel.CreateUnbounded<object>();

        var progress = new Progress<DownloadProgress>(p => channel.Writer.TryWrite(p));

        var pullTask = Task.Run(async () =>
        {
            try
            {
                var model = await store.PullAsync(request.Reference, progress, cancellationToken)
                    .ConfigureAwait(false);

                channel.Writer.TryWrite(new { status = "success", model });
            }
            catch (Exception ex)
            {
                channel.Writer.TryWrite(new { status = "error", error = ex.Message });
            }
            finally
            {
                channel.Writer.TryComplete();
            }
        }, CancellationToken.None);

        await foreach (var update in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
        {
            var json = JsonSerializer.Serialize(update, OpenAiJson.Options);
            await context.Response.WriteAsync($"data: {json}\n\n", cancellationToken).ConfigureAwait(false);
            await context.Response.Body.FlushAsync(cancellationToken).ConfigureAwait(false);
        }

        await pullTask.ConfigureAwait(false);
        await context.Response.WriteAsync("data: [DONE]\n\n", cancellationToken).ConfigureAwait(false);
    }

    private static async Task<Results<NoContent, NotFound>> RemoveModelAsync(
        string id,
        IModelStore store,
        ModelSessionManager sessions,
        CancellationToken cancellationToken)
    {
        // Unload first: deleting weights out from under a loaded session would crash the backend.
        await sessions.UnloadAsync(id, cancellationToken).ConfigureAwait(false);

        return await store.RemoveAsync(id, cancellationToken).ConfigureAwait(false)
            ? TypedResults.NoContent()
            : TypedResults.NotFound();
    }

    private static async Task<Results<Ok<LoadedModelInfo>, NotFound<string>>> LoadModelAsync(
        string id,
        ModelSessionManager sessions,
        CancellationToken cancellationToken)
    {
        try
        {
            using var lease = await sessions.AcquireAsync(id, null, cancellationToken).ConfigureAwait(false);

            return TypedResults.Ok(new LoadedModelInfo
            {
                Id = lease.Session.Model.Id,
                Engine = lease.Session.Engine.ToString(),
                Device = lease.Session.Device.ToString(),
                ContextSize = lease.Session.ContextSize,
                LoadedAt = DateTimeOffset.UtcNow,
                LastUsedAt = DateTimeOffset.UtcNow
            });
        }
        catch (ModelNotFoundException ex)
        {
            return TypedResults.NotFound(ex.Message);
        }
    }

    private static async Task<Results<NoContent, NotFound>> UnloadModelAsync(
        string id,
        ModelSessionManager sessions,
        CancellationToken cancellationToken) =>
        await sessions.UnloadAsync(id, cancellationToken).ConfigureAwait(false)
            ? TypedResults.NoContent()
            : TypedResults.NotFound();

    private static async Task<Results<ContentHttpResult, NotFound>> GetModelfileAsync(
        string id,
        IModelStore store,
        CancellationToken cancellationToken)
    {
        var modelfile = await store.GetModelfileAsync(id, cancellationToken).ConfigureAwait(false);

        return modelfile is null
            ? TypedResults.NotFound()
            : TypedResults.Text(
                LocalGen.Core.Modelfiles.ModelfileParser.Render(modelfile),
                "text/plain");
    }

    private static async Task<Ok<IReadOnlyList<CatalogEntry>>> SearchCatalogAsync(
        string q,
        IEnumerable<IModelCatalog> catalogs,
        int limit = 25,
        string? provider = null,
        CancellationToken cancellationToken = default)
    {
        var results = new List<CatalogEntry>();

        foreach (var catalog in catalogs)
        {
            if (provider is not null &&
                !string.Equals(catalog.Provider, provider, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            results.AddRange(await catalog.SearchAsync(q, limit, cancellationToken).ConfigureAwait(false));
        }

        return TypedResults.Ok<IReadOnlyList<CatalogEntry>>(results);
    }

    private static async Task<Results<Ok<CatalogEntry>, NotFound>> GetCatalogEntryAsync(
        string reference,
        IEnumerable<IModelCatalog> catalogs,
        CancellationToken cancellationToken)
    {
        foreach (var catalog in catalogs)
        {
            var entry = await catalog.GetAsync(reference, cancellationToken).ConfigureAwait(false);
            if (entry is not null)
            {
                return TypedResults.Ok(entry);
            }
        }

        return TypedResults.NotFound();
    }

    private static readonly DateTimeOffset ProcessStartedAt = DateTimeOffset.UtcNow;
}

public sealed record PullRequest(string Reference);

public sealed record ServiceStatus
{
    public required string Status { get; init; }

    public required string Version { get; init; }

    public DateTimeOffset StartedAt { get; init; }

    public TimeSpan Uptime { get; init; }

    public required string BaseUrl { get; init; }

    public bool OfflineMode { get; init; }

    public required string DataDirectory { get; init; }

    public IReadOnlyList<LoadedModelInfo> LoadedModels { get; init; } = [];
}

public sealed record LoadedModelInfo
{
    public required string Id { get; init; }

    public required string Engine { get; init; }

    public required string Device { get; init; }

    public int ContextSize { get; init; }

    public DateTimeOffset LoadedAt { get; init; }

    public DateTimeOffset LastUsedAt { get; init; }

    public long RequestCount { get; init; }

    public int ActiveRequests { get; init; }
}

public sealed record EnginesResponse
{
    public IReadOnlyList<EngineInfo> Engines { get; init; } = [];

    public required string RecommendedEngine { get; init; }

    public required string RecommendedDevice { get; init; }

    public required string Rationale { get; init; }
}

public sealed record EngineInfo
{
    public required string Kind { get; init; }

    public required string DisplayName { get; init; }

    public required string Description { get; init; }

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

public sealed record MetricsResponse
{
    public required MetricsSummary Summary { get; init; }

    public IReadOnlyList<InferenceSample> RecentInference { get; init; } = [];

    public IReadOnlyList<ResourceSample> RecentResources { get; init; } = [];
}
