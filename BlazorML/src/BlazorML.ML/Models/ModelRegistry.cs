using BlazorML.Core.Abstractions;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;

namespace BlazorML.ML.Models;

/// <summary>
/// Stores trained models and keeps their version history. Registering the same name again
/// creates the next version rather than overwriting, so a published endpoint keeps serving the
/// version it was pinned to.
/// </summary>
public sealed class ModelRegistry(AppDbContext db, IStorageProviderFactory storage) : IModelRegistry
{
    public async Task<TrainedModel> RegisterAsync(string name, string? description, MlTask task,
        string algorithm, string? labelColumn, Stream modelZip, string? metricsJson, string? inputSchemaJson,
        string? experimentId, string? runId, string? ownerId, CancellationToken ct = default)
    {
        var previous = await db.TrainedModels
            .Where(m => m.Name == name)
            .OrderByDescending(m => m.Version)
            .FirstOrDefaultAsync(ct);

        var model = new TrainedModel
        {
            Name = name,
            Description = description,
            Task = task,
            Algorithm = algorithm,
            LabelColumn = labelColumn,
            Version = (previous?.Version ?? 0) + 1,
            MetricsJson = metricsJson,
            InputSchemaJson = inputSchemaJson,
            ExperimentId = experimentId,
            RunId = runId,
            OwnerId = ownerId,
            SizeBytes = modelZip.CanSeek ? modelZip.Length : 0
        };

        model.StorageKey = $"models/{model.Id}/model.zip";

        if (modelZip.CanSeek)
        {
            modelZip.Position = 0;
        }

        await storage.Current.WriteAsync(model.StorageKey, modelZip, "application/zip", ct);

        db.TrainedModels.Add(model);
        await db.SaveChangesAsync(ct);

        return model;
    }

    public async Task<Stream> OpenAsync(string modelId, CancellationToken ct = default)
    {
        var model = await db.TrainedModels.AsNoTracking().FirstOrDefaultAsync(m => m.Id == modelId, ct)
            ?? throw new InvalidOperationException($"Model '{modelId}' no longer exists.");

        return await storage.Current.ReadAsync(model.StorageKey, ct);
    }

    public async Task<IReadOnlyList<TrainedModel>> VersionsAsync(string name, CancellationToken ct = default) =>
        await db.TrainedModels.AsNoTracking()
            .Where(m => m.Name == name)
            .OrderByDescending(m => m.Version)
            .ToListAsync(ct);
}
