using BlazorML.Core.Designer;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;

namespace BlazorML.Web.Services;

/// <summary>Create, save and version experiments. Every save writes a snapshot.</summary>
public sealed class ExperimentService(AppDbContext db)
{
    public async Task<Experiment> CreateAsync(string name, string? ownerId, CancellationToken ct = default)
    {
        var experiment = new Experiment
        {
            Name = name,
            OwnerId = ownerId,
            GraphJson = new ExperimentGraph().ToJson()
        };

        db.Experiments.Add(experiment);

        db.ExperimentVersions.Add(new ExperimentVersion
        {
            ExperimentId = experiment.Id,
            Version = 1,
            GraphJson = experiment.GraphJson,
            Note = "Dibuat",
            OwnerId = ownerId
        });

        await db.SaveChangesAsync(ct);
        return experiment;
    }

    public async Task<Experiment?> GetAsync(string id, CancellationToken ct = default) =>
        await db.Experiments.FirstOrDefaultAsync(e => e.Id == id, ct);

    public async Task<List<Experiment>> ListAsync(bool templates = false, CancellationToken ct = default) =>
        await db.Experiments.AsNoTracking()
            .Where(e => e.IsTemplate == templates)
            .OrderByDescending(e => e.UpdatedAt ?? e.CreatedAt)
            .ToListAsync(ct);

    public async Task SaveGraphAsync(string experimentId, ExperimentGraph graph, string? note,
        CancellationToken ct = default)
    {
        var experiment = await db.Experiments.FirstOrDefaultAsync(e => e.Id == experimentId, ct)
            ?? throw new InvalidOperationException("That experiment no longer exists.");

        var json = graph.ToJson();

        // An unchanged graph does not deserve a new version: the history stays meaningful.
        if (json == experiment.GraphJson)
        {
            return;
        }

        experiment.GraphJson = json;
        experiment.Version++;
        experiment.Task = InferTask(graph);

        db.ExperimentVersions.Add(new ExperimentVersion
        {
            ExperimentId = experiment.Id,
            Version = experiment.Version,
            GraphJson = json,
            Note = note,
            OwnerId = experiment.OwnerId
        });

        await db.SaveChangesAsync(ct);
    }

    /// <summary>Copies a template into a working experiment the user owns.</summary>
    public async Task<Experiment> CloneAsync(string sourceId, string? ownerId, string? newName = null,
        CancellationToken ct = default)
    {
        var source = await db.Experiments.AsNoTracking().FirstOrDefaultAsync(e => e.Id == sourceId, ct)
            ?? throw new InvalidOperationException("That experiment no longer exists.");

        var copy = new Experiment
        {
            Name = newName ?? $"{source.Name} (salinan)",
            Description = source.Description,
            GraphJson = source.GraphJson,
            Task = source.Task,
            Tags = source.Tags,
            OwnerId = ownerId
        };

        db.Experiments.Add(copy);

        db.ExperimentVersions.Add(new ExperimentVersion
        {
            ExperimentId = copy.Id,
            Version = 1,
            GraphJson = copy.GraphJson,
            Note = $"Disalin dari {source.Name}",
            OwnerId = ownerId
        });

        await db.SaveChangesAsync(ct);
        return copy;
    }

    public async Task DeleteAsync(string id, CancellationToken ct = default)
    {
        var experiment = await db.Experiments.FirstOrDefaultAsync(e => e.Id == id, ct);
        if (experiment is null)
        {
            return;
        }

        db.Experiments.Remove(experiment);
        await db.SaveChangesAsync(ct);
    }

    public async Task<List<ExperimentVersion>> VersionsAsync(string experimentId, CancellationToken ct = default) =>
        await db.ExperimentVersions.AsNoTracking()
            .Where(v => v.ExperimentId == experimentId)
            .OrderByDescending(v => v.Version)
            .ToListAsync(ct);

    public async Task RestoreVersionAsync(string experimentId, int version, CancellationToken ct = default)
    {
        var snapshot = await db.ExperimentVersions.AsNoTracking()
            .FirstOrDefaultAsync(v => v.ExperimentId == experimentId && v.Version == version, ct)
            ?? throw new InvalidOperationException($"Version {version} was not found.");

        var experiment = await db.Experiments.FirstOrDefaultAsync(e => e.Id == experimentId, ct)
            ?? throw new InvalidOperationException("That experiment no longer exists.");

        experiment.GraphJson = snapshot.GraphJson;
        experiment.Version++;

        // Restoring is itself a new version, so the history stays append-only and auditable.
        db.ExperimentVersions.Add(new ExperimentVersion
        {
            ExperimentId = experimentId,
            Version = experiment.Version,
            GraphJson = snapshot.GraphJson,
            Note = $"Dikembalikan ke versi {version}",
            OwnerId = experiment.OwnerId
        });

        await db.SaveChangesAsync(ct);
    }

    public async Task<List<ExperimentRun>> RunsAsync(string experimentId, int take = 20,
        CancellationToken ct = default) =>
        await db.ExperimentRuns.AsNoTracking()
            .Where(r => r.ExperimentId == experimentId)
            .OrderByDescending(r => r.CreatedAt)
            .Take(take)
            .ToListAsync(ct);

    /// <summary>Reads the experiment's task from whichever algorithm is on the canvas.</summary>
    private static MlTask InferTask(ExperimentGraph graph)
    {
        foreach (var node in graph.Nodes)
        {
            var module = Core.Modules.ModuleCatalog.Find(node.ModuleId);
            if (module is { Task: not MlTask.None })
            {
                return module.Task;
            }

            if (node.ModuleId == "train.autoML" &&
                Enum.TryParse<MlTask>(node.Parameters.GetValueOrDefault("task"), out var autoTask))
            {
                return autoTask;
            }
        }

        return MlTask.None;
    }
}
