using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Enums;
using SplatStudio.Infrastructure.Data;
using SplatStudio.Infrastructure.Splatting;

namespace SplatStudio.Infrastructure.BackgroundProcessing;

/// <summary>
/// Single long-running background worker. Kept to one worker (rather than
/// a pool) deliberately: image -> splat conversion is CPU-bound, and on a
/// modest web-app instance running many of these in parallel would starve
/// the Blazor Server SignalR circuits of CPU. Scale out by running this
/// app on more instances, not by adding workers per-instance.
/// </summary>
public class ConversionBackgroundService : BackgroundService
{
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly IConversionQueue _queue;
    private readonly ILogger<ConversionBackgroundService> _logger;

    public ConversionBackgroundService(
        IServiceScopeFactory scopeFactory,
        IConversionQueue queue,
        ILogger<ConversionBackgroundService> logger)
    {
        _scopeFactory = scopeFactory;
        _queue = queue;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            Guid sceneId;
            try
            {
                sceneId = await _queue.DequeueAsync(stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            try
            {
                await ProcessSceneAsync(sceneId, stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Unhandled error converting scene {SceneId}", sceneId);
            }
        }
    }

    private async Task ProcessSceneAsync(Guid sceneId, CancellationToken ct)
    {
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
        var storage = scope.ServiceProvider.GetRequiredService<IStorageService>();
        var catalog = scope.ServiceProvider.GetRequiredService<IConversionEngineCatalog>();
        var notifier = scope.ServiceProvider.GetRequiredService<ISceneUpdateNotifier>();

        var scene = await db.SplatScenes.FindAsync(new object[] { sceneId }, ct);
        if (scene is null)
        {
            _logger.LogWarning("Scene {SceneId} no longer exists, skipping", sceneId);
            return;
        }

        var imageAsset = await db.ImageAssets.FindAsync(new object[] { scene.ImageAssetId }, ct);
        if (imageAsset is null)
        {
            scene.Status = SplatStatus.Failed;
            scene.ErrorMessage = "Source image record is missing.";
            await db.SaveChangesAsync(ct);
            await notifier.NotifySceneUpdatedAsync(sceneId);
            return;
        }

        // The mode was chosen by the uploader and is fixed for the life of the scene.
        var engine = catalog.Resolve(scene.Mode);
        if (engine is null || !engine.IsAvailable)
        {
            // Configuration can change between the upload and the worker picking the job up,
            // so this is re-checked here rather than trusted from the upload page.
            scene.Status = SplatStatus.Failed;
            scene.ErrorMessage = engine?.UnavailableReason
                ?? $"No conversion engine is registered for {scene.Mode}.";
            await db.SaveChangesAsync(ct);
            await notifier.NotifySceneUpdatedAsync(sceneId);
            return;
        }

        scene.Status = SplatStatus.Processing;
        await db.SaveChangesAsync(ct);
        await notifier.NotifySceneUpdatedAsync(sceneId);

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        ConversionOutput result;
        await using (var imageStream = await storage.OpenReadAsync(imageAsset.StorageKey, ct))
        {
            result = await engine.ConvertAsync(imageStream, ct);
        }
        stopwatch.Stop();

        // Record which engine actually produced this scene — the configured engine can differ
        // from the effective one (the GPU engine falls back to CPU when no device is present),
        // and the gallery is transparent about provenance.
        scene.Engine = result.Engine;

        if (result.Success && result.Content is not null)
        {
            var prefix = result.Kind == ConversionArtifactKind.Mesh ? "meshes" : "splats";
            var key = $"{prefix}/{scene.UserId}/{scene.Id}{result.FileExtension}";

            await using (var stream = new MemoryStream(result.Content))
                await storage.SaveAsync(key, stream, result.ContentType, ct);

            scene.ArtifactKind = result.Kind;
            if (result.Kind == ConversionArtifactKind.Mesh)
            {
                scene.MeshStorageKey = key;
                scene.SplatStorageKey = null;
            }
            else
            {
                scene.SplatStorageKey = key;
                scene.MeshStorageKey = null;
            }

            scene.PointCount = result.PointCount;
            scene.Status = SplatStatus.Completed;
            scene.CompletedAtUtc = DateTime.UtcNow;
            scene.ErrorMessage = null;
            scene.ConversionMilliseconds = (int)stopwatch.ElapsedMilliseconds;

            _logger.LogInformation(
                "Converted scene {SceneId} in {Mode} mode with {Engine}: {Detail} in {Elapsed} ms.",
                sceneId, scene.Mode, result.Engine,
                result.Kind == ConversionArtifactKind.Mesh
                    ? $"{FormatBytes(result.Content.Length)} mesh"
                    : $"{result.PointCount:N0} splats",
                stopwatch.ElapsedMilliseconds);
        }
        else
        {
            scene.Status = SplatStatus.Failed;
            scene.ErrorMessage = result.ErrorMessage ?? "Conversion failed for an unknown reason.";
            _logger.LogWarning("Scene {SceneId} ({Mode}) failed: {Error}",
                sceneId, scene.Mode, scene.ErrorMessage);
        }

        await db.SaveChangesAsync(ct);
        await notifier.NotifySceneUpdatedAsync(sceneId);
    }

    private static string FormatBytes(int bytes) => bytes switch
    {
        < 1024 => $"{bytes} B",
        < 1024 * 1024 => $"{bytes / 1024.0:0.#} KB",
        _ => $"{bytes / (1024.0 * 1024.0):0.#} MB"
    };
}
