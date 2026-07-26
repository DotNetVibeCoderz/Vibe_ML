using System.ComponentModel;
using System.Globalization;
using System.Text.Json;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Data;
using BlazorML.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;

namespace BlazorML.Agents.Plugins;

/// <summary>
/// Lets Profesor Wicak look at the actual data before giving advice. Without these the model can
/// only guess at what a column contains, and guessing is what makes an assistant useless.
/// </summary>
public sealed class DataPlugin(IServiceScopeFactory scopeFactory)
{
    [KernelFunction, Description("Lists the datasets in the workspace with their size and column count.")]
    public async Task<string> ListDatasets(CancellationToken ct = default)
    {
        using var scope = scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var datasets = await db.Datasets.AsNoTracking()
            .OrderByDescending(d => d.CreatedAt)
            .Take(50)
            .Select(d => new { d.Id, d.Name, d.RowCount, d.ColumnCount, d.Description })
            .ToListAsync(ct);

        return datasets.Count == 0
            ? "There are no datasets in the workspace yet."
            : JsonSerializer.Serialize(datasets);
    }

    [KernelFunction, Description("Describes a dataset: every column, its type, how many values are missing, and its range.")]
    public async Task<string> DescribeDataset(
        [Description("The dataset id, from ListDatasets.")] string datasetId,
        CancellationToken ct = default)
    {
        using var scope = scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var dataset = await db.Datasets.AsNoTracking().FirstOrDefaultAsync(d => d.Id == datasetId, ct);

        if (dataset is null)
        {
            return $"No dataset with id '{datasetId}'. Call ListDatasets to see what is available.";
        }

        if (string.IsNullOrWhiteSpace(dataset.ProfileJson))
        {
            return $"'{dataset.Name}' has {dataset.RowCount:N0} rows and {dataset.ColumnCount} columns, " +
                   "but no column profile was stored for it.";
        }

        var profile = JsonSerializer.Deserialize<Core.Analysis.DatasetProfile>(dataset.ProfileJson);

        return JsonSerializer.Serialize(new
        {
            dataset.Name,
            dataset.Description,
            rows = dataset.RowCount,
            columns = profile?.Columns.Select(c => new
            {
                c.Name,
                type = c.DataType.ToString(),
                missing = c.MissingCount,
                distinct = c.DistinctCount,
                c.Min,
                c.Max,
                c.Mean,
                topValues = c.TopValues.Take(5).Select(t => $"{t.Value} ({t.Count})")
            })
        });
    }

    [KernelFunction, Description("Reads rows from a dataset, optionally filtered, so you can look at real values.")]
    public async Task<string> QueryDataset(
        [Description("The dataset id.")] string datasetId,
        [Description("Comma-separated column names. Leave empty for every column.")] string? columns = null,
        [Description("Optional filter, written as 'column operator value', e.g. 'age > 30' or 'city = Bandung'.")]
        string? filter = null,
        [Description("Maximum rows to return, at most 100.")] int limit = 20,
        CancellationToken ct = default)
    {
        using var scope = scopeFactory.CreateScope();
        var datasets = scope.ServiceProvider.GetRequiredService<IDatasetService>();

        TabularData table;
        try
        {
            table = await datasets.LoadAsync(datasetId, 0, ct);
        }
        catch (InvalidOperationException e)
        {
            return e.Message;
        }

        var rows = table.ToDictionaries();

        if (!string.IsNullOrWhiteSpace(filter))
        {
            var parsed = ParseFilter(filter);
            if (parsed is null)
            {
                return $"I could not read the filter '{filter}'. Write it as 'column operator value'.";
            }

            rows = rows.Where(r => Passes(r, parsed.Value)).ToList();
        }

        if (!string.IsNullOrWhiteSpace(columns))
        {
            var wanted = columns.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);

            rows = rows.Select(r => r.Where(kv => wanted.Contains(kv.Key))
                .ToDictionary(kv => kv.Key, kv => kv.Value)).ToList();
        }

        var capped = rows.Take(Math.Clamp(limit, 1, 100)).ToList();

        return JsonSerializer.Serialize(new
        {
            matched = rows.Count,
            returned = capped.Count,
            rows = capped
        });
    }

    [KernelFunction, Description("Counts how many rows fall into each value of a column — useful for checking class balance.")]
    public async Task<string> ValueCounts(
        [Description("The dataset id.")] string datasetId,
        [Description("The column to count.")] string column,
        CancellationToken ct = default)
    {
        using var scope = scopeFactory.CreateScope();
        var datasets = scope.ServiceProvider.GetRequiredService<IDatasetService>();

        TabularData table;
        try
        {
            table = await datasets.LoadAsync(datasetId, 0, ct);
        }
        catch (InvalidOperationException e)
        {
            return e.Message;
        }

        if (!table.Has(column))
        {
            return $"'{column}' is not a column in this dataset. It has: " +
                   string.Join(", ", table.Columns.Select(c => c.Name));
        }

        var index = table.IndexOf(column);
        var counts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var row in table.Rows)
        {
            var key = Convert.ToString(index < row.Length ? row[index] : null, CultureInfo.InvariantCulture);
            key = string.IsNullOrWhiteSpace(key) ? "(empty)" : key;
            counts[key] = counts.GetValueOrDefault(key) + 1;
        }

        var ordered = counts.OrderByDescending(kv => kv.Value).Take(40).ToList();

        return JsonSerializer.Serialize(new
        {
            column,
            distinct = counts.Count,
            total = table.RowCount,
            counts = ordered.ToDictionary(kv => kv.Key, kv => kv.Value)
        });
    }

    private static (string Column, string Op, string Value)? ParseFilter(string filter)
    {
        foreach (var op in (string[])[">=", "<=", "!=", "=", ">", "<", " contains "])
        {
            var index = filter.IndexOf(op, StringComparison.OrdinalIgnoreCase);
            if (index > 0)
            {
                return (filter[..index].Trim(), op.Trim(), filter[(index + op.Length)..].Trim());
            }
        }

        return null;
    }

    private static bool Passes(Dictionary<string, object?> row, (string Column, string Op, string Value) filter)
    {
        if (!row.TryGetValue(filter.Column, out var cell))
        {
            return false;
        }

        var text = Convert.ToString(cell, CultureInfo.InvariantCulture) ?? string.Empty;
        var left = TabularData.ToNumber(cell);
        var right = TabularData.ToNumber(filter.Value);

        if (left.HasValue && right.HasValue)
        {
            return filter.Op switch
            {
                ">" => left > right,
                ">=" => left >= right,
                "<" => left < right,
                "<=" => left <= right,
                "!=" => Math.Abs(left.Value - right.Value) > 1e-9,
                _ => Math.Abs(left.Value - right.Value) < 1e-9
            };
        }

        return filter.Op switch
        {
            "!=" => !string.Equals(text, filter.Value, StringComparison.OrdinalIgnoreCase),
            "contains" => text.Contains(filter.Value, StringComparison.OrdinalIgnoreCase),
            _ => string.Equals(text, filter.Value, StringComparison.OrdinalIgnoreCase)
        };
    }
}
