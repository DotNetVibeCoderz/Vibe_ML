using System.Text.Json;
using BlazorML.Agents.Plugins;
using Microsoft.Extensions.DependencyInjection;

namespace BlazorML.Agents.Tests;

/// <summary>
/// These functions are how Profesor Wicak edits the canvas. A model calling them can pass
/// anything at all, so the guards matter more here than almost anywhere else: a wrong port, a
/// connection that loops, a setting that is not a valid choice. None of this needs a provider —
/// the plugin is ordinary code that the model happens to call.
/// </summary>
public class DesignerPluginTests(WorkspaceFixture workspace) : IClassFixture<WorkspaceFixture>
{
    private DesignerPlugin Plugin => new(workspace.Scopes);

    private async Task<string> AddAsync(string experimentId, string moduleId)
    {
        var reply = await Plugin.AddModule(experimentId, moduleId);

        // "Added 'Import Dataset' with node id ab12cd34."
        return reply.Split("node id ").Last().TrimEnd('.');
    }

    // ------------------------------------------------------------- discovery

    [Fact]
    public async Task ListModules_returns_ids_that_actually_exist_in_the_catalog()
    {
        // The model builds flows from whatever this returns; an invented id would make every
        // following call fail.
        var json = Plugin.ListModules();
        var modules = JsonDocument.Parse(json).RootElement;

        Assert.True(modules.GetArrayLength() > 50);

        foreach (var module in modules.EnumerateArray())
        {
            var id = module.GetProperty("Id").GetString();
            Assert.NotNull(Core.Modules.ModuleCatalog.Find(id));
        }

        await Task.CompletedTask;
    }

    [Fact]
    public void ListModules_filters_by_category_and_by_search_term()
    {
        var algorithms = JsonDocument.Parse(Plugin.ListModules("Algorithm")).RootElement;
        Assert.All(algorithms.EnumerateArray(),
            m => Assert.Equal("Algorithm", m.GetProperty("category").GetString()));

        var found = JsonDocument.Parse(Plugin.ListModules(search: "split")).RootElement;
        Assert.Contains(found.EnumerateArray(), m => m.GetProperty("Id").GetString() == "tf.splitData");
    }

    [Fact]
    public async Task GetExperiment_reports_the_nodes_and_edges_that_are_really_stored()
    {
        var id = workspace.NewExperiment("baca-balik");
        var import = await AddAsync(id, "data.dataset");
        var clean = await AddAsync(id, "tf.cleanMissing");
        await Plugin.ConnectModules(id, import, clean);

        var body = JsonDocument.Parse(await Plugin.GetExperiment(id)).RootElement;

        Assert.Equal(2, body.GetProperty("nodes").GetArrayLength());
        Assert.Equal(1, body.GetProperty("edges").GetArrayLength());
    }

    // -------------------------------------------------------------- building

    [Fact]
    public async Task AddModule_puts_a_node_on_the_canvas_with_its_default_parameters()
    {
        var id = workspace.NewExperiment("tambah");
        var nodeId = await AddAsync(id, "tf.splitData");

        var node = workspace.Graph(id).Node(nodeId);

        Assert.NotNull(node);
        Assert.Equal("tf.splitData", node!.ModuleId);
        Assert.Equal("0.8", node.Parameters["fraction"]);
    }

    [Fact]
    public async Task AddModule_refuses_an_id_that_is_not_in_the_catalog()
    {
        var reply = await Plugin.AddModule(workspace.ExperimentId, "modul.karangan");

        Assert.Contains("not a known module", reply);
        Assert.Contains("ListModules", reply);
    }

    [Fact]
    public async Task ConnectModules_wires_two_nodes_together()
    {
        var id = workspace.NewExperiment("sambung");
        var import = await AddAsync(id, "data.dataset");
        var clean = await AddAsync(id, "tf.cleanMissing");

        var reply = await Plugin.ConnectModules(id, import, clean);

        Assert.Contains("Connected", reply);
        Assert.Single(workspace.Graph(id).Edges);
    }

    /// <summary>
    /// The model has no way to know a dataset cannot be fed into an algorithm's model port; the
    /// plugin has to say so rather than storing a graph that fails at run time.
    /// </summary>
    [Fact]
    public async Task ConnectModules_refuses_a_port_type_mismatch_and_explains_it()
    {
        var id = workspace.NewExperiment("tipe-salah");
        var import = await AddAsync(id, "data.dataset");
        var train = await AddAsync(id, "train.model");

        // Port 0 of Train Model expects an algorithm, not a dataset.
        var reply = await Plugin.ConnectModules(id, import, train, 0, 0);

        Assert.Contains("Dataset", reply);
        Assert.Contains("UntrainedModel", reply);
        Assert.Empty(workspace.Graph(id).Edges);
    }

    [Fact]
    public async Task ConnectModules_refuses_a_port_number_that_does_not_exist()
    {
        var id = workspace.NewExperiment("port-salah");
        var import = await AddAsync(id, "data.dataset");
        var clean = await AddAsync(id, "tf.cleanMissing");

        var reply = await Plugin.ConnectModules(id, import, clean, 5, 0);

        Assert.Contains("out of range", reply);
        Assert.Empty(workspace.Graph(id).Edges);
    }

    [Fact]
    public async Task ConnectModules_refuses_a_connection_that_would_create_a_loop()
    {
        var id = workspace.NewExperiment("lingkaran");
        var a = await AddAsync(id, "tf.cleanMissing");
        var b = await AddAsync(id, "tf.normalize");

        await Plugin.ConnectModules(id, a, b);
        var reply = await Plugin.ConnectModules(id, b, a);

        Assert.Contains("loop", reply);
        // The rejected edge must not be left behind.
        Assert.Single(workspace.Graph(id).Edges);
    }

    [Fact]
    public async Task Connecting_to_an_input_that_is_already_wired_replaces_it()
    {
        // An input port carries one value. Two edges into the same port would leave the run
        // engine picking arbitrarily.
        var id = workspace.NewExperiment("ganti");
        var first = await AddAsync(id, "data.dataset");
        var second = await AddAsync(id, "data.manual");
        var clean = await AddAsync(id, "tf.cleanMissing");

        await Plugin.ConnectModules(id, first, clean);
        await Plugin.ConnectModules(id, second, clean);

        var edges = workspace.Graph(id).Edges;

        Assert.Single(edges);
        Assert.Equal(second, edges[0].SourceNodeId);
    }

    // ------------------------------------------------------------ parameters

    [Fact]
    public async Task SetParameter_stores_the_value()
    {
        var id = workspace.NewExperiment("param");
        var node = await AddAsync(id, "tf.splitData");

        var reply = await Plugin.SetParameter(id, node, "fraction", "0.6");

        Assert.Contains("Set", reply);
        Assert.Equal("0.6", workspace.Graph(id).Node(node)!.Parameters["fraction"]);
    }

    [Fact]
    public async Task SetParameter_rejects_an_unknown_name_and_lists_what_is_available()
    {
        var id = workspace.NewExperiment("param-salah");
        var node = await AddAsync(id, "tf.splitData");

        var reply = await Plugin.SetParameter(id, node, "tidak_ada", "x");

        Assert.Contains("not a setting", reply);
        Assert.Contains("fraction", reply);
    }

    [Fact]
    public async Task SetParameter_rejects_a_value_outside_a_choice_list()
    {
        var id = workspace.NewExperiment("pilihan-salah");
        var node = await AddAsync(id, "tf.cleanMissing");

        var reply = await Plugin.SetParameter(id, node, "strategy", "ngawur");

        Assert.Contains("not valid", reply);
        Assert.Contains("median", reply);
        Assert.NotEqual("ngawur", workspace.Graph(id).Node(node)!.Parameters["strategy"]);
    }

    // --------------------------------------------------------------- removal

    [Fact]
    public async Task RemoveModule_takes_its_connections_with_it()
    {
        var id = workspace.NewExperiment("hapus");
        var import = await AddAsync(id, "data.dataset");
        var clean = await AddAsync(id, "tf.cleanMissing");
        await Plugin.ConnectModules(id, import, clean);

        var reply = await Plugin.RemoveModule(id, clean);

        Assert.Contains("Removed", reply);
        Assert.Single(workspace.Graph(id).Nodes);
        Assert.Empty(workspace.Graph(id).Edges);
    }

    [Fact]
    public async Task RemoveModule_reports_a_node_that_is_not_there()
    {
        Assert.Contains("No node with id", await Plugin.RemoveModule(workspace.ExperimentId, "tidak-ada"));
    }

    // ------------------------------------------------------------ versioning

    /// <summary>
    /// Every edit the assistant makes has to be reviewable and reversible in the same way a
    /// person's edit is. If a mutation skipped the version log there would be no way to see
    /// what the bot changed.
    /// </summary>
    [Fact]
    public async Task Every_mutation_records_a_new_version()
    {
        var id = workspace.NewExperiment("versi");
        var before = workspace.VersionCount(id);
        var beforeVersion = workspace.Experiment(id).Version;

        var node = await AddAsync(id, "tf.splitData");
        await Plugin.SetParameter(id, node, "fraction", "0.7");
        await Plugin.RemoveModule(id, node);

        Assert.Equal(before + 3, workspace.VersionCount(id));
        Assert.Equal(beforeVersion + 3, workspace.Experiment(id).Version);
    }

    [Fact]
    public async Task Version_notes_say_that_the_assistant_made_the_change()
    {
        var id = workspace.NewExperiment("catatan");
        await AddAsync(id, "tf.normalize");

        using var scope = workspace.Scopes.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<Infrastructure.Data.AppDbContext>();

        var note = db.ExperimentVersions
            .Where(v => v.ExperimentId == id)
            .OrderByDescending(v => v.Version)
            .Select(v => v.Note)
            .First();

        Assert.Contains("Wicak", note);
    }

    // ------------------------------------------------------------ validation

    [Fact]
    public async Task ValidateExperiment_reports_an_input_that_is_not_connected()
    {
        var id = workspace.NewExperiment("validasi");
        await AddAsync(id, "tf.cleanMissing");

        var reply = await Plugin.ValidateExperiment(id);

        Assert.Contains("Dataset", reply);
        Assert.Contains("Error", reply);
    }

    [Fact]
    public async Task ValidateExperiment_says_a_complete_flow_is_ready()
    {
        var id = workspace.NewExperiment("lengkap");
        var import = await AddAsync(id, "data.dataset");
        await Plugin.SetParameter(id, import, "datasetId", workspace.DatasetId);

        var reply = await Plugin.ValidateExperiment(id);

        Assert.Contains("ready to run", reply);
    }

    [Fact]
    public async Task Functions_report_a_missing_experiment_rather_than_throwing()
    {
        // A model will pass a stale id sooner or later; that must come back as a sentence it can
        // act on, not an exception that ends the conversation.
        Assert.Contains("No experiment", await Plugin.GetExperiment("tidak-ada"));
        Assert.Contains("No experiment", await Plugin.AddModule("tidak-ada", "tf.normalize"));
        Assert.Contains("No experiment", await Plugin.ValidateExperiment("tidak-ada"));
        Assert.Contains("No experiment", await Plugin.RunExperiment("tidak-ada"));
    }

    // ------------------------------------------------------------------- run

    [Fact]
    public async Task RunExperiment_executes_the_flow_the_assistant_built_and_reports_metrics()
    {
        // The end-to-end claim: a model can go from an empty canvas to a scored result using
        // nothing but these functions.
        var id = workspace.NewExperiment("jalankan");

        var import = await AddAsync(id, "data.dataset");
        await Plugin.SetParameter(id, import, "datasetId", workspace.DatasetId);

        var split = await AddAsync(id, "tf.splitData");
        var algorithm = await AddAsync(id, "algo.bin.fastTree");
        var train = await AddAsync(id, "train.model");
        var score = await AddAsync(id, "score.model");
        var evaluate = await AddAsync(id, "score.evaluate");

        await Plugin.SetParameter(id, split, "stratifyColumn", "churn");
        await Plugin.SetParameter(id, train, "labelColumn", "churn");

        await Plugin.ConnectModules(id, import, split);
        await Plugin.ConnectModules(id, algorithm, train, 0, 0);
        await Plugin.ConnectModules(id, split, train, 0, 1);
        await Plugin.ConnectModules(id, train, score, 0, 0);
        await Plugin.ConnectModules(id, split, score, 1, 1);
        await Plugin.ConnectModules(id, score, evaluate);

        Assert.Contains("ready to run", await Plugin.ValidateExperiment(id));

        var result = JsonDocument.Parse(await Plugin.RunExperiment(id)).RootElement;

        Assert.Equal("Succeeded", result.GetProperty("status").GetString());
        Assert.True(result.GetProperty("metrics").EnumerateObject().Any());
    }
}
