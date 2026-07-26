using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Data;
using BlazorML.Core.Designer;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using BlazorML.Infrastructure.Datasets;
using BlazorML.Infrastructure.Settings;
using BlazorML.Infrastructure.Storage;
using BlazorML.ML.Execution;
using BlazorML.ML.Execution.Executors;
using BlazorML.ML.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace BlazorML.Agents.Tests;

/// <summary>
/// A real workspace on disk — SQLite plus file storage in a temp folder — so the plugins are
/// exercised against the same services the running app gives them. None of this needs a model
/// provider: the plugins that read data and edit the canvas are ordinary code.
/// </summary>
public sealed class WorkspaceFixture : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"blazorml-agents-{Guid.NewGuid():n}");

    public ServiceProvider Services { get; }
    public IServiceScopeFactory Scopes => Services.GetRequiredService<IServiceScopeFactory>();

    public string DatasetId { get; private set; } = string.Empty;
    public string ExperimentId { get; private set; } = string.Empty;

    public WorkspaceFixture()
    {
        Directory.CreateDirectory(_root);

        var configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["Storage:Provider"] = "FileSystem",
                ["Storage:FileSystem:RootPath"] = "storage",
                ["Chat:Provider"] = "OpenAI",
                ["Tools:EnableWebSearch"] = "true",
                ["Tools:EnableUrlScraping"] = "true"
            })
            .Build();

        var services = new ServiceCollection();
        services.AddLogging();
        services.AddHttpClient("agent");
        services.AddHttpClient("modules");
        services.AddSingleton<IConfiguration>(configuration);
        services.AddDbContext<AppDbContext>(b =>
            b.UseSqlite($"Data Source={Path.Combine(_root, "agents.db")}"));
        services.AddSingleton<ISettingsService, SettingsService>();
        services.AddSingleton<IStorageProviderFactory>(sp =>
            new StorageProviderFactory(sp.GetRequiredService<ISettingsService>(), _root));
        services.AddScoped<IDatasetService, DatasetService>();
        services.AddScoped<IModelRegistry, ModelRegistry>();
        services.AddScoped<IExperimentRunner, ExperimentRunner>();
        services.AddScoped<IModuleExecutor, DataInputExecutor>();
        services.AddScoped<IModuleExecutor, TransformExecutor>();
        services.AddScoped<IModuleExecutor, AlgorithmExecutor>();
        services.AddScoped<IModuleExecutor, TrainingExecutor>();
        services.AddScoped<IModuleExecutor, ScoringExecutor>();

        Services = services.BuildServiceProvider();

        Seed();
    }

    private void Seed()
    {
        using var scope = Services.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        db.Database.EnsureCreated();

        var datasets = scope.ServiceProvider.GetRequiredService<IDatasetService>();

        var table = TabularData.WithColumns("umur", "kota", "churn");
        var random = new Random(5);

        for (var i = 0; i < 60; i++)
        {
            table.AddRow(
                random.Next(18, 70),
                (string[])["Bandung", "Jakarta"] is var c ? c[i % 2] : "?",
                // Deliberately lopsided so a class-balance question has something to report.
                i < 45 ? "0" : "1");
        }

        DatasetId = datasets.SaveAsync("Pelanggan Uji", table, DatasetFormat.Csv, null)
            .GetAwaiter().GetResult().Id;

        var experiment = new Experiment
        {
            Name = "Eksperimen Uji",
            GraphJson = new ExperimentGraph().ToJson()
        };

        db.Experiments.Add(experiment);
        db.SaveChanges();

        ExperimentId = experiment.Id;
    }

    /// <summary>Reads the experiment's graph straight from the database, bypassing the plugin.</summary>
    public ExperimentGraph Graph(string? experimentId = null)
    {
        using var scope = Services.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var experiment = db.Experiments.AsNoTracking()
            .First(e => e.Id == (experimentId ?? ExperimentId));

        return ExperimentGraph.FromJson(experiment.GraphJson);
    }

    public Experiment Experiment(string? experimentId = null)
    {
        using var scope = Services.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        return db.Experiments.AsNoTracking().First(e => e.Id == (experimentId ?? ExperimentId));
    }

    public int VersionCount(string? experimentId = null)
    {
        using var scope = Services.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        return db.ExperimentVersions.Count(v => v.ExperimentId == (experimentId ?? ExperimentId));
    }

    /// <summary>A fresh, empty experiment so a test can mutate without disturbing the others.</summary>
    public string NewExperiment(string name)
    {
        using var scope = Services.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var experiment = new Experiment { Name = name, GraphJson = new ExperimentGraph().ToJson() };
        db.Experiments.Add(experiment);
        db.SaveChanges();

        return experiment.Id;
    }

    public void Dispose()
    {
        Services.Dispose();

        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (IOException)
        {
            // A leftover temp folder is not worth failing a run over.
        }
    }
}
