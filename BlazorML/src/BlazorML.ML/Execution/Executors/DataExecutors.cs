using System.Data.Common;
using System.Text;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Datasets;
using Microsoft.Extensions.DependencyInjection;
using System.Net.Http;

namespace BlazorML.ML.Execution.Executors;

/// <summary>Handles the four ways rows enter an experiment.</summary>
public sealed class DataInputExecutor : IModuleExecutor
{
    public bool CanExecute(string moduleId) => moduleId.StartsWith("data.", StringComparison.Ordinal);

    public async Task<object?[]> ExecuteAsync(ModuleExecutionContext ctx)
    {
        var table = ctx.Module.Id switch
        {
            "data.dataset" => await FromWorkspace(ctx),
            "data.sql" => await FromSql(ctx),
            "data.web" => await FromWeb(ctx),
            "data.manual" => FromManual(ctx),
            _ => throw new NotSupportedException($"Unknown input module '{ctx.Module.Id}'.")
        };

        ctx.Log(Core.Domain.LogLevel.Info, $"Read {table.RowCount:N0} rows and {table.ColumnCount} columns.");
        return [table];
    }

    private static async Task<TabularData> FromWorkspace(ModuleExecutionContext ctx)
    {
        var datasetId = ctx.RequireParam("datasetId");
        var datasets = ctx.Services.GetRequiredService<IDatasetService>();
        return await datasets.LoadAsync(datasetId, ctx.ParamInt("sampleRows", 0), ctx.Ct);
    }

    private static async Task<TabularData> FromSql(ModuleExecutionContext ctx)
    {
        var provider = Enum.TryParse<DatabaseProviderKind>(ctx.Param("provider"), out var p)
            ? p : DatabaseProviderKind.Sqlite;

        var connectionString = ctx.RequireParam("connectionString");
        var query = ctx.RequireParam("query");

        await using var connection = SqlConnections.Create(provider, connectionString);
        await connection.OpenAsync(ctx.Ct);

        await using var command = connection.CreateCommand();
        command.CommandText = query;

        await using var reader = await command.ExecuteReaderAsync(ctx.Ct);
        var table = new TabularData();

        for (var i = 0; i < reader.FieldCount; i++)
        {
            table.AddColumn(reader.GetName(i));
        }

        while (await reader.ReadAsync(ctx.Ct))
        {
            var row = table.NewRow();
            for (var i = 0; i < reader.FieldCount && i < row.Length; i++)
            {
                row[i] = await reader.IsDBNullAsync(i, ctx.Ct) ? null : reader.GetValue(i);
            }

            table.Rows.Add(row);
        }

        table.InferTypes();
        return table;
    }

    private static async Task<TabularData> FromWeb(ModuleExecutionContext ctx)
    {
        var url = ctx.RequireParam("url");
        var format = Enum.TryParse<DatasetFormat>(ctx.Param("format"), out var f) ? f : DatasetFormat.Csv;

        var factory = ctx.Services.GetRequiredService<IHttpClientFactory>();
        using var client = factory.CreateClient("modules");

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        foreach (var line in (ctx.Param("headers") ?? string.Empty)
                     .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var separator = line.IndexOf(':');
            if (separator > 0)
            {
                request.Headers.TryAddWithoutValidation(line[..separator].Trim(), line[(separator + 1)..].Trim());
            }
        }

        using var response = await client.SendAsync(request, ctx.Ct);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(ctx.Ct);
        return await TabularSerializer.ReadAsync(stream, format, 0, ctx.Ct);
    }

    private static TabularData FromManual(ModuleExecutionContext ctx)
    {
        var csv = ctx.RequireParam("csv");
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(csv));
        return TabularSerializer.ReadAsync(stream, DatasetFormat.Csv, 0, ctx.Ct).GetAwaiter().GetResult();
    }
}

/// <summary>Opens an ADO.NET connection for whichever database the user picked.</summary>
public static class SqlConnections
{
    public static DbConnection Create(DatabaseProviderKind provider, string connectionString) => provider switch
    {
        DatabaseProviderKind.SqlServer => new Microsoft.Data.SqlClient.SqlConnection(connectionString),
        DatabaseProviderKind.MySql => new MySql.Data.MySqlClient.MySqlConnection(connectionString),
        DatabaseProviderKind.PostgreSql => new Npgsql.NpgsqlConnection(connectionString),
        _ => new Microsoft.Data.Sqlite.SqliteConnection(connectionString)
    };
}

/// <summary>Writes results back out of the experiment: a saved dataset or a registered model.</summary>
public sealed class OutputExecutor : IModuleExecutor
{
    public bool CanExecute(string moduleId) => moduleId.StartsWith("out.", StringComparison.Ordinal);

    public async Task<object?[]> ExecuteAsync(ModuleExecutionContext ctx)
    {
        switch (ctx.Module.Id)
        {
            case "out.saveDataset":
            {
                var table = ctx.RequireTable(0, "Dataset");
                var name = ctx.Param("name") ?? $"{ctx.Node.Label} output";
                var format = Enum.TryParse<DatasetFormat>(ctx.Param("format"), out var f) ? f : DatasetFormat.Csv;

                var datasets = ctx.Services.GetRequiredService<IDatasetService>();
                var saved = await datasets.SaveAsync(name, table, format, ctx.UserId, ctx.Ct);

                ctx.Log(Core.Domain.LogLevel.Info,
                    $"Saved '{saved.Name}' with {saved.RowCount:N0} rows to the workspace.");
                return [];
            }

            case "out.registerModel":
            {
                var handle = ctx.Require<TrainedModelHandle>(0, "Trained model");
                var registry = ctx.Services.GetRequiredService<IModelRegistry>();
                var name = ctx.Param("name") ?? $"{handle.Algorithm} model";

                await using var buffer = new MemoryStream();
                ctx.Ml.Model.Save(handle.Transformer, handle.InputSchema, buffer);
                buffer.Position = 0;

                var metrics = handle.TrainingMetrics is null
                    ? null
                    : System.Text.Json.JsonSerializer.Serialize(handle.TrainingMetrics);

                var schema = Core.Domain.ModelInputSchema.Serialise(
                    DescribeInputs(handle));

                var model = await registry.RegisterAsync(name, ctx.Param("description"), handle.Task,
                    handle.Algorithm, handle.LabelColumn, buffer, metrics, schema,
                    null, null, ctx.UserId, ctx.Ct);

                ctx.Log(Core.Domain.LogLevel.Info, $"Registered '{model.Name}' as version {model.Version}.");
                return [];
            }

            default:
                throw new NotSupportedException($"Unknown output module '{ctx.Module.Id}'.");
        }
    }

    /// <summary>
    /// Describes the model's features for whoever has to call it later: name, type, and one real
    /// value taken from the training data.
    /// <para>
    /// The example is picked from the first row that actually has one, because the first row of a
    /// real dataset is as likely to hold a gap as any other, and a null example teaches nobody
    /// anything about the column.
    /// </para>
    /// </summary>
    private static List<ModelInputField> DescribeInputs(TrainedModelHandle handle)
    {
        var sample = handle.TrainingSample;

        return handle.FeatureColumns.Select(name =>
        {
            var column = sample?.Columns.FirstOrDefault(c => c.Name == name);

            return new ModelInputField
            {
                Name = name,
                Type = column?.DataType,
                Example = FirstValue(sample, name)
            };
        }).ToList();
    }

    private static System.Text.Json.JsonElement? FirstValue(TabularData? table, string column)
    {
        if (table is null || !table.Has(column))
        {
            return null;
        }

        for (var row = 0; row < Math.Min(table.RowCount, 50); row++)
        {
            var value = table.Value(row, column);

            if (!TabularData.IsBlank(value))
            {
                return System.Text.Json.JsonSerializer.SerializeToElement(value);
            }
        }

        return null;
    }
}
