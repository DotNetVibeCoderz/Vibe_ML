using System.Text.Json;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;

namespace BlazorML.Infrastructure.Datasets;

public sealed class DatasetService(
    AppDbContext db,
    IStorageProviderFactory storage,
    ISettingsService settings) : IDatasetService
{
    public async Task<TabularData> LoadAsync(string datasetId, int rowLimit = 0, CancellationToken ct = default)
    {
        var dataset = await db.Datasets.AsNoTracking().FirstOrDefaultAsync(d => d.Id == datasetId, ct)
            ?? throw new InvalidOperationException($"Dataset '{datasetId}' no longer exists.");

        var training = await settings.GetAsync<TrainingOptions>(SettingsSections.Training, ct);
        var cap = rowLimit > 0 ? Math.Min(rowLimit, training.MaxRowsInMemory) : training.MaxRowsInMemory;

        await using var stream = await storage.Current.ReadAsync(dataset.StorageKey, ct);
        return await TabularSerializer.ReadAsync(stream, dataset.Format, cap, ct);
    }

    public async Task<TabularData> PreviewAsync(string datasetId, int rows = 50, CancellationToken ct = default) =>
        await LoadAsync(datasetId, rows, ct);

    public async Task<Dataset> ImportAsync(string name, Stream content, DatasetFormat format,
        DatasetSourceKind source, string? ownerId, CancellationToken ct = default)
    {
        // Buffer once: the bytes are needed both for profiling and for the write to storage.
        var buffer = new MemoryStream();
        await content.CopyToAsync(buffer, ct);
        buffer.Position = 0;

        var table = await TabularSerializer.ReadAsync(buffer, format, 0, ct);
        buffer.Position = 0;

        var dataset = new Dataset
        {
            Name = name,
            Format = format,
            Source = source,
            OwnerId = ownerId,
            SizeBytes = buffer.Length,
            RowCount = table.RowCount,
            ColumnCount = table.ColumnCount,
            ProfileJson = JsonSerializer.Serialize(table.Profile())
        };

        dataset.StorageKey = $"datasets/{dataset.Id}/{Sanitise(name)}.{Extension(format)}";
        await storage.Current.WriteAsync(dataset.StorageKey, buffer, ContentType(format), ct);

        db.Datasets.Add(dataset);
        await db.SaveChangesAsync(ct);
        return dataset;
    }

    public async Task<Dataset> SaveAsync(string name, TabularData data, DatasetFormat format,
        string? ownerId, CancellationToken ct = default)
    {
        await using var stream = await TabularSerializer.WriteAsync(data, format, ct);

        var dataset = new Dataset
        {
            Name = name,
            Format = format,
            Source = DatasetSourceKind.Generated,
            OwnerId = ownerId,
            SizeBytes = stream.Length,
            RowCount = data.RowCount,
            ColumnCount = data.ColumnCount,
            ProfileJson = JsonSerializer.Serialize(data.Profile())
        };

        dataset.StorageKey = $"datasets/{dataset.Id}/{Sanitise(name)}.{Extension(format)}";
        stream.Position = 0;
        await storage.Current.WriteAsync(dataset.StorageKey, stream, ContentType(format), ct);

        db.Datasets.Add(dataset);
        await db.SaveChangesAsync(ct);
        return dataset;
    }

    private static string Sanitise(string name)
    {
        var cleaned = new string(name.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '-').ToArray());
        return cleaned.Trim('-').ToLowerInvariant() is { Length: > 0 } s ? s : "dataset";
    }

    private static string Extension(DatasetFormat format) => format switch
    {
        DatasetFormat.Tsv => "tsv",
        DatasetFormat.Json => "json",
        DatasetFormat.Parquet => "parquet",
        _ => "csv"
    };

    private static string ContentType(DatasetFormat format) => format switch
    {
        DatasetFormat.Json => "application/json",
        DatasetFormat.Tsv => "text/tab-separated-values",
        _ => "text/csv"
    };
}
