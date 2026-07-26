using System.Diagnostics;
using System.Text.Json;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Data;
using BlazorML.Core.Designer;
using BlazorML.Core.Domain;
using BlazorML.Core.Modules;
using BlazorML.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;

namespace BlazorML.ML.Execution;

/// <summary>
/// Walks an experiment graph in dependency order, runs each node, and records what happened.
/// One instance handles one run.
/// </summary>
public sealed class ExperimentRunner(
    AppDbContext db,
    IServiceProvider services,
    IEnumerable<IModuleExecutor> executors,
    ISettingsService settings) : IExperimentRunner
{
    private readonly List<IModuleExecutor> _executors = executors.ToList();

    public async Task<ExperimentRun> RunAsync(string experimentId, string? userId,
        IProgress<RunProgress>? progress = null, CancellationToken ct = default)
    {
        var experiment = await db.Experiments.FirstOrDefaultAsync(e => e.Id == experimentId, ct)
            ?? throw new InvalidOperationException($"Experiment '{experimentId}' no longer exists.");

        var graph = ExperimentGraph.FromJson(experiment.GraphJson);
        var options = await settings.GetAsync<TrainingOptions>(SettingsSections.Training, ct);

        var run = new ExperimentRun
        {
            ExperimentId = experimentId,
            ExperimentVersion = experiment.Version,
            OwnerId = userId,
            Status = RunStatus.Running,
            StartedAt = DateTimeOffset.UtcNow
        };

        db.ExperimentRuns.Add(run);
        experiment.LastStatus = RunStatus.Running;
        experiment.LastRunAt = run.StartedAt;
        await db.SaveChangesAsync(ct);

        var results = new Dictionary<string, NodeRunResult>();
        var logs = new List<RunLogEntry>();
        var sequence = 0;

        // Values produced so far, keyed by "nodeId:portIndex".
        var outputs = new Dictionary<string, object?>();

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromMinutes(options.RunTimeoutMinutes));

        var ml = new MLContext(seed: 42);
        var stopwatch = Stopwatch.StartNew();

        try
        {
            if (!graph.TryTopologicalSort(out var ordered, out var sortError))
            {
                throw new InvalidOperationException(sortError ?? "The graph could not be ordered.");
            }

            if (ordered.Count == 0)
            {
                throw new InvalidOperationException("There is nothing on the canvas to run yet.");
            }

            foreach (var node in ordered)
            {
                timeout.Token.ThrowIfCancellationRequested();

                var module = ModuleCatalog.Find(node.ModuleId);
                if (module is null)
                {
                    throw new InvalidOperationException(
                        $"'{node.Label}' refers to an unknown module '{node.ModuleId}'.");
                }

                // A node whose upstream failed is skipped, not failed: it never got its input.
                if (HasFailedUpstream(graph, node.Id, results))
                {
                    results[node.Id] = new NodeRunResult
                    {
                        State = NodeRunState.Skipped,
                        Message = "Skipped because a step it depends on did not finish."
                    };

                    progress?.Report(new RunProgress(node.Id, NodeRunState.Skipped));
                    continue;
                }

                progress?.Report(new RunProgress(node.Id, NodeRunState.Running));

                var nodeStopwatch = Stopwatch.StartNew();

                try
                {
                    var inputs = ResolveInputs(graph, node, module, outputs);

                    var context = new ModuleExecutionContext
                    {
                        Ml = ml,
                        Node = node,
                        Module = module,
                        Services = services,
                        Inputs = inputs,
                        UserId = userId,
                        Ct = timeout.Token,
                        Log = (level, message) => logs.Add(new RunLogEntry
                        {
                            RunId = run.Id,
                            NodeId = node.Id,
                            Level = level,
                            Message = message,
                            Sequence = sequence++
                        })
                    };

                    var executor = _executors.FirstOrDefault(e => e.CanExecute(module.Id))
                        ?? throw new InvalidOperationException(
                            $"No executor is registered for '{module.Name}'.");

                    var produced = await executor.ExecuteAsync(context);

                    for (var port = 0; port < produced.Length; port++)
                    {
                        outputs[$"{node.Id}:{port}"] = produced[port];
                    }

                    nodeStopwatch.Stop();

                    var result = Describe(produced, options.PreviewRows);
                    result.State = NodeRunState.Succeeded;
                    result.DurationSeconds = nodeStopwatch.Elapsed.TotalSeconds;
                    results[node.Id] = result;

                    progress?.Report(new RunProgress(node.Id, NodeRunState.Succeeded, null, result));
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception e)
                {
                    nodeStopwatch.Stop();

                    var result = new NodeRunResult
                    {
                        State = NodeRunState.Failed,
                        DurationSeconds = nodeStopwatch.Elapsed.TotalSeconds,
                        Message = e.Message
                    };

                    results[node.Id] = result;
                    logs.Add(new RunLogEntry
                    {
                        RunId = run.Id,
                        NodeId = node.Id,
                        Level = Core.Domain.LogLevel.Error,
                        Message = e.Message,
                        Sequence = sequence++
                    });

                    progress?.Report(new RunProgress(node.Id, NodeRunState.Failed, e.Message, result));
                }
            }

            var failed = results.Values.Count(r => r.State == NodeRunState.Failed);
            run.Status = failed == 0 ? RunStatus.Succeeded : RunStatus.Failed;
            run.Error = failed == 0
                ? null
                : $"{failed} step{(failed == 1 ? string.Empty : "s")} did not finish. Open the log for the reason.";
        }
        catch (OperationCanceledException)
        {
            run.Status = ct.IsCancellationRequested ? RunStatus.Cancelled : RunStatus.Failed;
            run.Error = ct.IsCancellationRequested
                ? "The run was cancelled."
                : $"The run passed its {options.RunTimeoutMinutes} minute limit and was stopped.";
        }
        catch (Exception e)
        {
            run.Status = RunStatus.Failed;
            run.Error = e.Message;

            logs.Add(new RunLogEntry
            {
                RunId = run.Id,
                Level = Core.Domain.LogLevel.Error,
                Message = e.Message,
                Sequence = sequence++
            });
        }
        finally
        {
            stopwatch.Stop();

            run.FinishedAt = DateTimeOffset.UtcNow;
            run.DurationSeconds = stopwatch.Elapsed.TotalSeconds;
            run.NodeResultsJson = JsonSerializer.Serialize(results);
            run.MetricsJson = JsonSerializer.Serialize(CollectHeadlineMetrics(results));

            db.RunLogs.AddRange(logs);

            experiment.LastStatus = run.Status;
            await db.SaveChangesAsync(CancellationToken.None);
        }

        return run;
    }

    /// <summary>
    /// Fills the input array for a node from the edges wired into it. An unwired optional port
    /// stays null; an unwired required port is reported by the module itself, which can give a
    /// better message than a generic one here.
    /// </summary>
    private static object?[] ResolveInputs(ExperimentGraph graph, GraphNode node, ModuleDescriptor module,
        Dictionary<string, object?> outputs)
    {
        var inputs = new object?[module.Inputs.Count];

        foreach (var edge in graph.InboundEdges(node.Id))
        {
            if (edge.TargetPort < 0 || edge.TargetPort >= inputs.Length)
            {
                continue;
            }

            inputs[edge.TargetPort] = outputs.GetValueOrDefault($"{edge.SourceNodeId}:{edge.SourcePort}");
        }

        return inputs;
    }

    private static bool HasFailedUpstream(ExperimentGraph graph, string nodeId,
        Dictionary<string, NodeRunResult> results) =>
        graph.InboundEdges(nodeId).Any(e =>
            results.TryGetValue(e.SourceNodeId, out var upstream) &&
            upstream.State is NodeRunState.Failed or NodeRunState.Skipped);

    /// <summary>Summarises a node's outputs for the canvas badge and the preview panel.</summary>
    private static NodeRunResult Describe(object?[] outputs, int previewRows)
    {
        var result = new NodeRunResult();

        foreach (var output in outputs)
        {
            switch (output)
            {
                case TabularData table when result.Preview is null:
                    result.RowsOut = table.RowCount;
                    result.ColumnsOut = table.ColumnCount;
                    result.Preview = table.Preview(previewRows);
                    break;

                case MetricsBundle bundle:
                    result.Metrics ??= new Dictionary<string, double>();
                    foreach (var (name, value) in bundle.Values)
                    {
                        result.Metrics[name] = value;
                    }

                    result.Evaluation ??= bundle.Evaluation;
                    result.Preview ??= bundle.Table?.Preview(previewRows);
                    break;

                case TrainedModelHandle handle:
                    result.Message = $"{handle.Algorithm} · {handle.FeatureColumns.Count} features";
                    break;
            }
        }

        return result;
    }

    private static Dictionary<string, double> CollectHeadlineMetrics(Dictionary<string, NodeRunResult> results)
    {
        var metrics = new Dictionary<string, double>();

        foreach (var result in results.Values.Where(r => r.Metrics is not null))
        {
            foreach (var (name, value) in result.Metrics!)
            {
                // First writer wins so the earliest evaluation node in the graph sets the headline.
                metrics.TryAdd(name, value);
            }
        }

        return metrics;
    }

    // ------------------------------------------------------------------------ validation

    public Task<IReadOnlyList<GraphIssue>> ValidateAsync(ExperimentGraph graph, CancellationToken ct = default)
    {
        var issues = new List<GraphIssue>();

        if (graph.Nodes.Count == 0)
        {
            issues.Add(new GraphIssue(null, GraphIssueSeverity.Warning,
                "The canvas is empty. Drag a module from the palette to start."));

            return Task.FromResult<IReadOnlyList<GraphIssue>>(issues);
        }

        if (!graph.TryTopologicalSort(out _, out var sortError))
        {
            issues.Add(new GraphIssue(null, GraphIssueSeverity.Error, sortError!));
        }

        foreach (var node in graph.Nodes)
        {
            var module = ModuleCatalog.Find(node.ModuleId);

            if (module is null)
            {
                issues.Add(new GraphIssue(node.Id, GraphIssueSeverity.Error,
                    $"'{node.Label}' refers to a module that is no longer available."));
                continue;
            }

            var wired = graph.InboundEdges(node.Id).Select(e => e.TargetPort).ToHashSet();

            for (var port = 0; port < module.Inputs.Count; port++)
            {
                if (module.Inputs[port].Required && !wired.Contains(port))
                {
                    issues.Add(new GraphIssue(node.Id, GraphIssueSeverity.Error,
                        $"'{node.Label}' has nothing connected to its {module.Inputs[port].Name} input."));
                }
            }

            foreach (var parameter in module.Parameters)
            {
                var hasValue = node.Parameters.TryGetValue(parameter.Name, out var value)
                               && !string.IsNullOrWhiteSpace(value);

                // Only flag parameters with no default: those are the ones the user must supply.
                if (!hasValue && string.IsNullOrWhiteSpace(parameter.Default) && IsRequired(parameter))
                {
                    issues.Add(new GraphIssue(node.Id, GraphIssueSeverity.Error,
                        $"'{node.Label}' still needs a value for {parameter.Label}."));
                }
            }

            // A trainer wired into the wrong Train Model variant is a common and confusing mistake.
            if (module.Id is "train.model" or "train.clustering")
            {
                var algorithmEdge = graph.InboundEdges(node.Id).FirstOrDefault(e => e.TargetPort == 0);
                var source = algorithmEdge is null ? null : ModuleCatalog.Find(graph.Node(algorithmEdge.SourceNodeId)?.ModuleId);

                if (source is not null && module.Id == "train.clustering" && source.Task != MlTask.Clustering)
                {
                    issues.Add(new GraphIssue(node.Id, GraphIssueSeverity.Error,
                        $"'{source.Name}' is not a clustering algorithm. Use Train Model instead."));
                }
                else if (source is not null && module.Id == "train.model" && source.Task == MlTask.Clustering)
                {
                    issues.Add(new GraphIssue(node.Id, GraphIssueSeverity.Error,
                        $"'{source.Name}' needs Train Clustering Model, which does not take a label."));
                }
            }
        }

        var terminals = graph.Nodes.Where(n =>
        {
            var module = ModuleCatalog.Find(n.ModuleId);
            return module is not null && module.Outputs.Count > 0 && !graph.OutboundEdges(n.Id).Any();
        }).ToList();

        if (terminals.Count == graph.Nodes.Count && graph.Nodes.Count > 1)
        {
            issues.Add(new GraphIssue(null, GraphIssueSeverity.Warning,
                "None of the modules are connected to each other yet."));
        }

        return Task.FromResult<IReadOnlyList<GraphIssue>>(issues);
    }

    /// <summary>
    /// Requiredness is declared on the spec. A field that is only shown conditionally is never
    /// reported, because the user cannot fill in something they cannot see.
    /// </summary>
    private static bool IsRequired(ParameterSpec parameter) =>
        parameter.Required && parameter.VisibleWhen is null;
}
