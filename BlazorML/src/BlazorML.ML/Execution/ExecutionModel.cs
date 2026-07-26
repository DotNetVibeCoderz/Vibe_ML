using System.Globalization;
using BlazorML.Core.Analysis;
using BlazorML.Core.Data;
using BlazorML.Core.Designer;
using BlazorML.Core.Domain;
using BlazorML.Core.Modules;
using BlazorML.ML.Training;
using Microsoft.ML;

namespace BlazorML.ML.Execution;

/// <summary>
/// Everything one module needs while it runs: its resolved inputs, its parameters, and a way to
/// say what it is doing. Executors receive this and return one value per declared output port.
/// </summary>
public sealed class ModuleExecutionContext
{
    public required MLContext Ml { get; init; }
    public required GraphNode Node { get; init; }
    public required ModuleDescriptor Module { get; init; }
    public required IServiceProvider Services { get; init; }
    public required CancellationToken Ct { get; init; }

    /// <summary>One entry per declared input port; null where an optional port is unwired.</summary>
    public required object?[] Inputs { get; init; }

    public required Action<Core.Domain.LogLevel, string> Log { get; init; }

    /// <summary>The owner of the run, so anything the module persists is attributed correctly.</summary>
    public string? UserId { get; init; }

    public TabularData? Table(int port) => port < Inputs.Length ? Inputs[port] as TabularData : null;

    public TabularData RequireTable(int port, string portName) =>
        Table(port) ?? throw new InvalidOperationException(
            $"'{Node.Label}' needs a dataset on its {portName} input. Connect one and run again.");

    public T Require<T>(int port, string portName) where T : class =>
        (port < Inputs.Length ? Inputs[port] as T : null) ?? throw new InvalidOperationException(
            $"'{Node.Label}' needs a {typeof(T).Name} on its {portName} input. Connect one and run again.");

    // -------------------------------------------------------------- parameter access

    public string? Param(string name)
    {
        if (Node.Parameters.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value))
        {
            return value;
        }

        var fallback = Module.Parameters.FirstOrDefault(p => p.Name == name)?.Default;
        return string.IsNullOrWhiteSpace(fallback) ? null : fallback;
    }

    public string RequireParam(string name)
    {
        var label = Module.Parameters.FirstOrDefault(p => p.Name == name)?.Label ?? name;
        return Param(name) ?? throw new InvalidOperationException(
            $"'{Node.Label}' is missing a value for {label}.");
    }

    public int ParamInt(string name, int fallback) =>
        int.TryParse(Param(name), NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : fallback;

    public double ParamDouble(string name, double fallback) =>
        double.TryParse(Param(name), NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : fallback;

    public bool ParamBool(string name, bool fallback) =>
        bool.TryParse(Param(name), out var v) ? v : fallback;

    /// <summary>Splits a column-list parameter, tolerating commas, newlines and stray spaces.</summary>
    public List<string> ParamList(string name) =>
        Param(name)?.Split([',', '\n', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToList() ?? [];
}

public interface IModuleExecutor
{
    bool CanExecute(string moduleId);

    /// <summary>Returns one value per declared output port, in port order.</summary>
    Task<object?[]> ExecuteAsync(ModuleExecutionContext context);
}

/// <summary>
/// What travels on a <see cref="PortType.TrainedModel"/> edge: the fitted transformer plus the
/// context needed to score with it and to describe it when it is registered.
/// </summary>
public sealed class TrainedModelHandle
{
    public required ITransformer Transformer { get; init; }
    public required DataViewSchema InputSchema { get; init; }
    public required MlTask Task { get; init; }
    public required string Algorithm { get; init; }
    public string? LabelColumn { get; init; }
    public List<string> FeatureColumns { get; init; } = new();

    /// <summary>The table the model was fitted on, kept for feature importance and previews.</summary>
    public TabularData? TrainingSample { get; init; }

    public EvaluationResult? TrainingMetrics { get; set; }
}

/// <summary>
/// What travels on a <see cref="PortType.Metrics"/> edge. Carries the full evaluation result when
/// there is one, and a flat table for things like a sweep leaderboard.
/// </summary>
public sealed class MetricsBundle
{
    public string Title { get; init; } = "Metrics";
    public EvaluationResult? Evaluation { get; init; }
    public TabularData? Table { get; init; }
    public Dictionary<string, double> Values { get; init; } = new();
}
