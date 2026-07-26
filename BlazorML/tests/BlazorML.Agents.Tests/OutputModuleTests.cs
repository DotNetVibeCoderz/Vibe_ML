using System.Text.Json;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.Core.Modules;
using BlazorML.Infrastructure.Data;
using BlazorML.ML.Execution;
using BlazorML.ML.Execution.Executors;
using BlazorML.ML.Training;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;

namespace BlazorML.Agents.Tests;

/// <summary>
/// Save Dataset and Register Model — the two modules that write back out of an experiment. Both
/// need a real workspace behind them, which is why they live here rather than with the other
/// executor tests. Neither had ever been executed.
/// </summary>
public class OutputModuleTests(WorkspaceFixture workspace) : IClassFixture<WorkspaceFixture>
{
    private readonly OutputExecutor _outputs = new();
    private readonly AlgorithmExecutor _algorithms = new();
    private readonly TrainingExecutor _training = new();

    /// <summary>An execution context wired to the fixture's real services.</summary>
    private ModuleExecutionContext Context(string moduleId, IServiceScope scope,
        (string Name, string? Value)[]? parameters, params object?[] inputs)
    {
        var module = ModuleCatalog.Find(moduleId)!;

        var node = new Core.Designer.GraphNode
        {
            ModuleId = moduleId,
            Label = module.Name,
            Parameters = module.BuildDefaultParameters()
        };

        foreach (var (name, value) in parameters ?? [])
        {
            node.Parameters[name] = value;
        }

        return new ModuleExecutionContext
        {
            Ml = new MLContext(seed: 42),
            Node = node,
            Module = module,
            Services = scope.ServiceProvider,
            Inputs = inputs,
            Ct = CancellationToken.None,
            Log = (_, _) => { }
        };
    }

    private static TabularData Sample()
    {
        var table = TabularData.WithColumns("kota", "jumlah");
        table.AddRow("Bandung", 10);
        table.AddRow("Jakarta", 20);
        table.AddRow("Surabaya", null);

        return table;
    }

    // ------------------------------------------------------------ save dataset

    [Fact]
    public async Task Save_dataset_writes_a_dataset_that_can_be_read_back()
    {
        using var scope = workspace.Scopes.CreateScope();

        var name = $"Keluaran Uji {Guid.NewGuid():n}"[..24];
        var context = Context("out.saveDataset", scope,
            [("name", name), ("format", "Csv")], Sample());

        var results = await _outputs.ExecuteAsync(context);

        // A terminal module produces no ports; its whole effect is the write.
        Assert.Empty(results);

        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        var saved = await db.Datasets.FirstOrDefaultAsync(d => d.Name == name);

        Assert.NotNull(saved);
        Assert.Equal(3, saved!.RowCount);
        Assert.Equal(2, saved.ColumnCount);

        // Round trip: what came out has to be what went in.
        var datasets = scope.ServiceProvider.GetRequiredService<IDatasetService>();
        var reloaded = await datasets.LoadAsync(saved.Id);

        Assert.Equal(3, reloaded.RowCount);
        Assert.Equal("Bandung", reloaded.Text(0, "kota"));
        Assert.True(TabularData.IsBlank(reloaded.Value(2, "jumlah")));
    }

    [Fact]
    public async Task Save_dataset_stores_a_column_profile_alongside_it()
    {
        using var scope = workspace.Scopes.CreateScope();

        var name = $"Profil {Guid.NewGuid():n}"[..18];
        await _outputs.ExecuteAsync(Context("out.saveDataset", scope,
            [("name", name), ("format", "Csv")], Sample()));

        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        var saved = await db.Datasets.FirstAsync(d => d.Name == name);

        Assert.False(string.IsNullOrWhiteSpace(saved.ProfileJson));

        var profile = JsonSerializer.Deserialize<Core.Analysis.DatasetProfile>(saved.ProfileJson!)!;

        Assert.Equal(2, profile.Columns.Count);
        Assert.Equal(1, profile.Column("jumlah")!.MissingCount);
    }

    [Theory]
    [InlineData("Csv")]
    [InlineData("Json")]
    [InlineData("Parquet")]
    public async Task Save_dataset_honours_the_format_it_was_given(string format)
    {
        using var scope = workspace.Scopes.CreateScope();

        var name = $"Fmt{format}{Guid.NewGuid():n}"[..16];
        await _outputs.ExecuteAsync(Context("out.saveDataset", scope,
            [("name", name), ("format", format)], Sample()));

        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        var saved = await db.Datasets.FirstAsync(d => d.Name == name);

        Assert.Equal(Enum.Parse<DatasetFormat>(format), saved.Format);
        Assert.EndsWith(format.ToLowerInvariant(), saved.StorageKey);

        var reloaded = await scope.ServiceProvider.GetRequiredService<IDatasetService>()
            .LoadAsync(saved.Id);

        Assert.Equal(3, reloaded.RowCount);
    }

    // ---------------------------------------------------------- register model

    private TrainedModelHandle TrainSomething(IServiceScope scope)
    {
        var random = new Random(67);
        var data = TabularData.WithColumns("x", "kelas");

        for (var i = 0; i < 120; i++)
        {
            var x = random.Next(0, 100);
            data.AddRow(x, x > 50 ? "1" : "0");
        }

        var spec = (TrainerSpec)_algorithms
            .ExecuteAsync(Context("algo.bin.fastTree", scope, null))
            .GetAwaiter().GetResult()[0]!;

        return (TrainedModelHandle)_training
            .ExecuteAsync(Context("train.model", scope, [("labelColumn", "kelas")], spec, data))
            .GetAwaiter().GetResult()[0]!;
    }

    [Fact]
    public async Task Register_model_stores_the_model_and_its_schema()
    {
        using var scope = workspace.Scopes.CreateScope();

        var handle = TrainSomething(scope);
        var name = $"Model {Guid.NewGuid():n}"[..18];

        await _outputs.ExecuteAsync(Context("out.registerModel", scope,
            [("name", name), ("description", "Ditulis oleh tes")], handle));

        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        var model = await db.TrainedModels.FirstOrDefaultAsync(m => m.Name == name);

        Assert.NotNull(model);
        Assert.Equal(1, model!.Version);
        Assert.Equal(MlTask.BinaryClassification, model.Task);
        Assert.Equal("kelas", model.LabelColumn);
        Assert.False(string.IsNullOrWhiteSpace(model.InputSchemaJson));
    }

    /// <summary>
    /// The stored file has to be a model an endpoint can actually load. Recording a row in the
    /// database while writing something unusable would only surface in production.
    /// </summary>
    [Fact]
    public async Task A_registered_model_can_be_loaded_and_scored_with()
    {
        using var scope = workspace.Scopes.CreateScope();

        var handle = TrainSomething(scope);
        var name = $"Muat {Guid.NewGuid():n}"[..16];

        await _outputs.ExecuteAsync(Context("out.registerModel", scope, [("name", name)], handle));

        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        var model = await db.TrainedModels.FirstAsync(m => m.Name == name);

        var registry = scope.ServiceProvider.GetRequiredService<IModelRegistry>();
        await using var stream = await registry.OpenAsync(model.Id);

        var ml = new MLContext(seed: 42);
        var transformer = ml.Model.Load(stream, out _);

        var rows = TabularData.WithColumns("x", "kelas");
        rows.AddRow(90, "0");
        rows.AddRow(10, "0");

        using var bridge = new MlDataBridge();
        var view = bridge.ToDataView(ml, rows, "kelas", MlTask.BinaryClassification, out _);
        var scored = MlDataBridge.FromDataView(transformer.Transform(view));

        Assert.Equal(2, scored.RowCount);
        Assert.True(scored.Has("PredictedLabel"));
    }

    [Fact]
    public async Task Registering_the_same_name_again_makes_the_next_version()
    {
        using var scope = workspace.Scopes.CreateScope();

        var handle = TrainSomething(scope);
        var name = $"Versi {Guid.NewGuid():n}"[..18];

        await _outputs.ExecuteAsync(Context("out.registerModel", scope, [("name", name)], handle));
        await _outputs.ExecuteAsync(Context("out.registerModel", scope, [("name", name)], handle));

        var registry = scope.ServiceProvider.GetRequiredService<IModelRegistry>();
        var versions = await registry.VersionsAsync(name);

        // Two rows, not one overwritten: an endpoint pinned to v1 must keep serving v1.
        Assert.Equal(2, versions.Count);
        Assert.Equal([2, 1], versions.Select(v => v.Version));
        Assert.NotEqual(versions[0].StorageKey, versions[1].StorageKey);
    }

    [Fact]
    public async Task Register_model_refuses_when_nothing_is_wired_to_it()
    {
        using var scope = workspace.Scopes.CreateScope();

        // One unwired port, not an absent input array — what the engine hands a module whose
        // upstream edge the user never drew.
        var unwired = new object?[] { null };

        var error = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _outputs.ExecuteAsync(Context("out.registerModel", scope, [("name", "kosong")], unwired)));

        Assert.Contains("Trained model", error.Message);
    }
}
