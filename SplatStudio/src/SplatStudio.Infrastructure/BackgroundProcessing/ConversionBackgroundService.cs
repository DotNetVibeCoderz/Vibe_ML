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
    private readonly SplattingOptions _options;
    private readonly ILogger<ConversionBackgroundService> _logger;

    public ConversionBackgroundService(
        IServiceScopeFactory scopeFactory,
        IConversionQueue queue,
        SplattingOptions options,
        ILogger<ConversionBackgroundService> logger)
    {
        _scopeFactory = scopeFactory;
        _queue = queue;
        _options = options;
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
        var engine = scope.ServiceProvider.GetRequiredService<IGaussianSplatEngine>();
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

        scene.Status = SplatStatus.Processing;
        await db.SaveChangesAsync(ct);
        await notifier.NotifySceneUpdatedAsync(sceneId);

        GaussianSplatGenerationResult result;
        await using (var imageStream = await storage.OpenReadAsync(imageAsset.StorageKey, ct))
        {
            result = await engine.GenerateAsync(imageStream, _options.MaxPoints, ct);
        }

        if (result.Success && result.SplatFileBytes is not null)
        {
            var splatKey = $"splats/{scene.UserId}/{scene.Id}.splat";
            await using var splatStream = new MemoryStream(result.SplatFileBytes);
            await storage.SaveAsync(splatKey, splatStream, "application/octet-stream", ct);

            scene.SplatStorageKey = splatKey;
            scene.PointCount = result.PointCount;
            scene.Status = SplatStatus.Completed;
            scene.CompletedAtUtc = DateTime.UtcNow;
            scene.ErrorMessage = null;
        }
        else
        {
            scene.Status = SplatStatus.Failed;
            scene.ErrorMessage = result.ErrorMessage ?? "Conversion failed for an unknown reason.";
        }

        await db.SaveChangesAsync(ct);
        await notifier.NotifySceneUpdatedAsync(sceneId);
    }
}
