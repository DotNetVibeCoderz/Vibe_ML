using BlazorML.Core.Abstractions;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.ML.Training;
using BlazorML.Web.Services;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;

namespace BlazorML.Web.Tests;

/// <summary>
/// Boots the real application against a throwaway database and storage folder, trains a small
/// model through the same services the designer uses, and publishes it. The API tests then hit
/// the running app exactly as an outside caller would.
/// </summary>
public sealed class ScoringApiFixture : WebApplicationFactory<Program>, IAsyncLifetime
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"blazorml-test-{Guid.NewGuid():n}");

    public string Slug { get; private set; } = string.Empty;
    public string ApiKey { get; private set; } = string.Empty;
    public string StoppedSlug { get; private set; } = string.Empty;

    /// <summary>The seeded administrator, as created by <c>DataSeeder</c>.</summary>
    public const string Email = "admin@gravicode.com";
    public const string Password = "StudioML#2026";

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        Directory.CreateDirectory(_root);
        builder.UseContentRoot(_root);

        builder.UseSetting("Database:Provider", "Sqlite");
        builder.UseSetting("Database:ConnectionStrings:Sqlite", $"Data Source={Path.Combine(_root, "test.db")}");
        builder.UseSetting("Storage:Provider", "FileSystem");
        builder.UseSetting("Storage:FileSystem:RootPath", "storage");
    }

    // Explicit implementation: xunit's IAsyncLifetime is Task-based while
    // WebApplicationFactory's own DisposeAsync is ValueTask-based, so the two cannot share a
    // signature.
    Task IAsyncLifetime.InitializeAsync() => StartAsync();

    async Task IAsyncLifetime.DisposeAsync()
    {
        await DisposeAsync();

        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (IOException)
        {
            // A file the host still holds open is not worth failing a test run over.
        }
    }

    private async Task StartAsync()
    {
        // Touching the client is what actually starts the host.
        _ = CreateClient();

        using var scope = Services.CreateScope();
        var registry = scope.ServiceProvider.GetRequiredService<IModelRegistry>();
        var endpoints = scope.ServiceProvider.GetRequiredService<EndpointService>();

        var model = await TrainAndRegisterAsync(registry);

        var (live, key) = await endpoints.PublishAsync(model.Id, "Uji Skor", "Endpoint untuk pengujian", null);
        Slug = live.Slug;
        ApiKey = key;

        var (stopped, _) = await endpoints.PublishAsync(model.Id, "Uji Berhenti", null, null);
        await endpoints.SetStatusAsync(stopped.Id, EndpointStatus.Stopped);
        StoppedSlug = stopped.Slug;
    }

    /// <summary>Trains on a separable signal so the predictions are checkable, not just present.</summary>
    private static async Task<TrainedModel> TrainAndRegisterAsync(IModelRegistry registry)
    {
        var random = new Random(3);
        var data = TabularData.WithColumns("pendapatan", "usia", "beli");

        for (var i = 0; i < 200; i++)
        {
            var income = random.Next(1, 100);
            data.AddRow(income, random.Next(18, 70), income > 55 ? "1" : "0");
        }

        var ml = new MLContext(seed: 42);

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ml, data, "beli", MlTask.BinaryClassification, out _);

        var pipeline = PipelineBuilder
            .BuildFeaturization(ml, data, MlTask.BinaryClassification, "beli", null, out var features)
            .Append(ml.BinaryClassification.Trainers.FastTree(
                labelColumnName: PipelineBuilder.Label, featureColumnName: PipelineBuilder.Features));

        var transformer = pipeline.Fit(view);

        using var buffer = new MemoryStream();
        ml.Model.Save(transformer, view.Schema, buffer);
        buffer.Position = 0;

        // The enriched shape the registrar writes: name, type and a real value. The snippet tests
        // depend on it, and so does anything that generates a payload from a published model.
        var schema = ModelInputSchema.Serialise(features.Select(f => new ModelInputField
        {
            Name = f,
            Type = data.Columns.First(c => c.Name == f).DataType,
            Example = System.Text.Json.JsonSerializer.SerializeToElement(data.Value(0, f))
        }));

        return await registry.RegisterAsync("uji-churn", "Model untuk pengujian API",
            MlTask.BinaryClassification, "Boosted Decision Tree", "beli",
            buffer, null, schema, null, null, null);
    }

}
