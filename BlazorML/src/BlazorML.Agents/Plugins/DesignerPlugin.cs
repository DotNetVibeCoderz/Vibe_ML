using System.ComponentModel;
using System.Text.Json;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Designer;
using BlazorML.Core.Domain;
using BlazorML.Core.Modules;
using BlazorML.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;

namespace BlazorML.Agents.Plugins;

/// <summary>
/// The functions that make Profesor Wicak able to build experiments rather than only describe
/// them. Every change goes through <see cref="ExperimentGraph"/> and is saved as a new version,
/// so anything the assistant does can be inspected and rolled back like a human's edit.
/// </summary>
public sealed class DesignerPlugin(IServiceScopeFactory scopeFactory)
{
    [KernelFunction, Description(
        "Lists the modules available in the designer. Call this before building a flow so you use real module ids.")]
    public string ListModules(
        [Description("Optional category: DataInput, DataTransform, LlmAction, Algorithm, Training, Evaluation, Script, Output.")]
        string? category = null,
        [Description("Optional search term, matched against name, description and keywords.")]
        string? search = null)
    {
        var modules = ModuleCatalog.All.AsEnumerable();

        if (!string.IsNullOrWhiteSpace(category) && Enum.TryParse<ModuleCategory>(category, true, out var parsed))
        {
            modules = modules.Where(m => m.Category == parsed);
        }

        if (!string.IsNullOrWhiteSpace(search))
        {
            modules = modules.Intersect(ModuleCatalog.Search(search));
        }

        return JsonSerializer.Serialize(modules.Select(m => new
        {
            m.Id,
            m.Name,
            category = m.Category.ToString(),
            task = m.Task == MlTask.None ? null : m.Task.ToString(),
            m.Description,
            inputs = m.Inputs.Select(p => new { p.Name, type = p.Type.ToString(), p.Required }),
            outputs = m.Outputs.Select(p => new { p.Name, type = p.Type.ToString() }),
            parameters = m.Parameters.Select(p => new
            {
                p.Name,
                p.Label,
                kind = p.Kind.ToString(),
                p.Default,
                choices = p.Choices.Select(c => c.Value)
            })
        }));
    }

    [KernelFunction, Description("Shows the current flow of an experiment: its modules, their settings and how they are wired.")]
    public async Task<string> GetExperiment(
        [Description("The experiment id.")] string experimentId,
        CancellationToken ct = default)
    {
        using var scope = scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var experiment = await db.Experiments.AsNoTracking().FirstOrDefaultAsync(e => e.Id == experimentId, ct);
        if (experiment is null)
        {
            return $"No experiment with id '{experimentId}'.";
        }

        var graph = ExperimentGraph.FromJson(experiment.GraphJson);

        return JsonSerializer.Serialize(new
        {
            experiment.Name,
            experiment.Description,
            version = experiment.Version,
            nodes = graph.Nodes.Select(n => new
            {
                n.Id,
                n.ModuleId,
                n.Label,
                parameters = n.Parameters.Where(p => p.Value is not null)
                    .ToDictionary(p => p.Key, p => p.Value)
            }),
            edges = graph.Edges.Select(e => new
            {
                from = $"{e.SourceNodeId}:{e.SourcePort}",
                to = $"{e.TargetNodeId}:{e.TargetPort}"
            })
        });
    }

    [KernelFunction, Description(
        "Adds a module to an experiment and returns the new node id. Use ListModules first to get a valid moduleId.")]
    public async Task<string> AddModule(
        [Description("The experiment id.")] string experimentId,
        [Description("The module id, for example 'algo.bin.fastTree'.")] string moduleId,
        [Description("Optional label for the node. Defaults to the module name.")] string? label = null,
        CancellationToken ct = default)
    {
        var module = ModuleCatalog.Find(moduleId);
        if (module is null)
        {
            return $"'{moduleId}' is not a known module. Call ListModules to see the valid ids.";
        }

        return await MutateAsync(experimentId, graph =>
        {
            var node = new GraphNode
            {
                ModuleId = moduleId,
                Label = label ?? module.Name,
                Parameters = module.BuildDefaultParameters()
            };

            // Lay the node out in a column per category so an agent-built graph is readable
            // without the user having to rearrange it by hand.
            var column = (int)module.Category;
            var inColumn = graph.Nodes.Count(n =>
                ModuleCatalog.Find(n.ModuleId)?.Category == module.Category);

            node.X = 80 + column * 260;
            node.Y = 80 + inColumn * 140;

            graph.Nodes.Add(node);
            return $"Added '{node.Label}' with node id {node.Id}.";
        }, $"Wicak added {module.Name}", ct);
    }

    [KernelFunction, Description("Connects one module's output to another module's input.")]
    public async Task<string> ConnectModules(
        [Description("The experiment id.")] string experimentId,
        [Description("Node id the data comes from.")] string sourceNodeId,
        [Description("Node id the data goes to.")] string targetNodeId,
        [Description("Which output port of the source, counting from 0.")] int sourcePort = 0,
        [Description("Which input port of the target, counting from 0.")] int targetPort = 0,
        CancellationToken ct = default)
    {
        return await MutateAsync(experimentId, graph =>
        {
            var source = graph.Node(sourceNodeId);
            var target = graph.Node(targetNodeId);

            if (source is null || target is null)
            {
                return $"One of the nodes was not found: {sourceNodeId}, {targetNodeId}.";
            }

            var sourceModule = ModuleCatalog.Find(source.ModuleId);
            var targetModule = ModuleCatalog.Find(target.ModuleId);

            if (sourceModule is null || targetModule is null)
            {
                return "One of the nodes refers to a module that no longer exists.";
            }

            if (sourcePort < 0 || sourcePort >= sourceModule.Outputs.Count)
            {
                return $"'{source.Label}' has {sourceModule.Outputs.Count} outputs, so port {sourcePort} is out of range.";
            }

            if (targetPort < 0 || targetPort >= targetModule.Inputs.Count)
            {
                return $"'{target.Label}' has {targetModule.Inputs.Count} inputs, so port {targetPort} is out of range.";
            }

            var outputType = sourceModule.Outputs[sourcePort].Type;
            var inputType = targetModule.Inputs[targetPort].Type;

            if (outputType != inputType && outputType != PortType.Any && inputType != PortType.Any)
            {
                return $"'{sourceModule.Outputs[sourcePort].Name}' carries {outputType} but " +
                       $"'{targetModule.Inputs[targetPort].Name}' expects {inputType}.";
            }

            // One value per input port: a second connection replaces the first rather than
            // silently leaving two edges fighting over the same slot.
            graph.Edges.RemoveAll(e => e.TargetNodeId == targetNodeId && e.TargetPort == targetPort);

            graph.Edges.Add(new GraphEdge
            {
                SourceNodeId = sourceNodeId,
                SourcePort = sourcePort,
                TargetNodeId = targetNodeId,
                TargetPort = targetPort
            });

            if (!graph.TryTopologicalSort(out _, out var cycle))
            {
                graph.Edges.RemoveAll(e =>
                    e.SourceNodeId == sourceNodeId && e.TargetNodeId == targetNodeId && e.TargetPort == targetPort);

                return $"That connection would create a loop: {cycle}";
            }

            return $"Connected '{source.Label}' to '{target.Label}'.";
        }, "Wicak connected two modules", ct);
    }

    [KernelFunction, Description("Sets one setting on a module, for example the label column or the number of trees.")]
    public async Task<string> SetParameter(
        [Description("The experiment id.")] string experimentId,
        [Description("The node id.")] string nodeId,
        [Description("The parameter name, as returned by ListModules.")] string name,
        [Description("The value to set.")] string value,
        CancellationToken ct = default)
    {
        return await MutateAsync(experimentId, graph =>
        {
            var node = graph.Node(nodeId);
            if (node is null)
            {
                return $"No node with id '{nodeId}'.";
            }

            var module = ModuleCatalog.Find(node.ModuleId);
            var parameter = module?.Parameters.FirstOrDefault(p => p.Name == name);

            if (parameter is null)
            {
                var available = module is null
                    ? string.Empty
                    : " Available: " + string.Join(", ", module.Parameters.Select(p => p.Name));

                return $"'{name}' is not a setting on '{node.Label}'.{available}";
            }

            if (parameter.Kind == ParameterKind.Choice &&
                parameter.Choices.Count > 0 &&
                !parameter.Choices.Any(c => string.Equals(c.Value, value, StringComparison.OrdinalIgnoreCase)))
            {
                return $"'{value}' is not valid for {parameter.Label}. Choose one of: " +
                       string.Join(", ", parameter.Choices.Select(c => c.Value));
            }

            node.Parameters[name] = value;
            return $"Set {parameter.Label} on '{node.Label}' to {value}.";
        }, $"Wicak set {name}", ct);
    }

    [KernelFunction, Description("Removes a module and every connection attached to it.")]
    public async Task<string> RemoveModule(
        [Description("The experiment id.")] string experimentId,
        [Description("The node id to remove.")] string nodeId,
        CancellationToken ct = default)
    {
        return await MutateAsync(experimentId, graph =>
        {
            var node = graph.Node(nodeId);
            if (node is null)
            {
                return $"No node with id '{nodeId}'.";
            }

            graph.Nodes.Remove(node);
            var removed = graph.Edges.RemoveAll(e => e.SourceNodeId == nodeId || e.TargetNodeId == nodeId);

            return $"Removed '{node.Label}' and {removed} connection{(removed == 1 ? string.Empty : "s")}.";
        }, "Wicak removed a module", ct);
    }

    [KernelFunction, Description(
        "Checks an experiment for problems — missing connections, unset settings, loops — without running it.")]
    public async Task<string> ValidateExperiment(
        [Description("The experiment id.")] string experimentId,
        CancellationToken ct = default)
    {
        using var scope = scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        var runner = scope.ServiceProvider.GetRequiredService<IExperimentRunner>();

        var experiment = await db.Experiments.AsNoTracking().FirstOrDefaultAsync(e => e.Id == experimentId, ct);
        if (experiment is null)
        {
            return $"No experiment with id '{experimentId}'.";
        }

        var issues = await runner.ValidateAsync(ExperimentGraph.FromJson(experiment.GraphJson), ct);

        return issues.Count == 0
            ? "The experiment looks ready to run."
            : JsonSerializer.Serialize(issues.Select(i => new
            {
                severity = i.Severity.ToString(),
                node = i.NodeId,
                i.Message
            }));
    }

    [KernelFunction, Description("Runs an experiment and reports the result.")]
    public async Task<string> RunExperiment(
        [Description("The experiment id.")] string experimentId,
        CancellationToken ct = default)
    {
        using var scope = scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        var runner = scope.ServiceProvider.GetRequiredService<IExperimentRunner>();

        var experiment = await db.Experiments.AsNoTracking().FirstOrDefaultAsync(e => e.Id == experimentId, ct);
        if (experiment is null)
        {
            return $"No experiment with id '{experimentId}'.";
        }

        var run = await runner.RunAsync(experimentId, experiment.OwnerId, null, ct);

        return JsonSerializer.Serialize(new
        {
            status = run.Status.ToString(),
            seconds = Math.Round(run.DurationSeconds, 1),
            run.Error,
            metrics = string.IsNullOrWhiteSpace(run.MetricsJson)
                ? null
                : JsonSerializer.Deserialize<Dictionary<string, double>>(run.MetricsJson)
        });
    }

    /// <summary>
    /// Loads the graph, applies a change, and saves it as a new version. Centralised so every
    /// mutation is versioned the same way and none of them can forget to bump the version.
    /// </summary>
    private async Task<string> MutateAsync(string experimentId, Func<ExperimentGraph, string> change,
        string versionNote, CancellationToken ct)
    {
        using var scope = scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var experiment = await db.Experiments.FirstOrDefaultAsync(e => e.Id == experimentId, ct);
        if (experiment is null)
        {
            return $"No experiment with id '{experimentId}'.";
        }

        var graph = ExperimentGraph.FromJson(experiment.GraphJson);
        var message = change(graph);

        experiment.GraphJson = graph.ToJson();
        experiment.Version++;

        db.ExperimentVersions.Add(new ExperimentVersion
        {
            ExperimentId = experiment.Id,
            Version = experiment.Version,
            GraphJson = experiment.GraphJson,
            Note = versionNote,
            OwnerId = experiment.OwnerId
        });

        await db.SaveChangesAsync(ct);
        return message;
    }
}
