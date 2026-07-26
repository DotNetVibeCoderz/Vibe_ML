using System.Globalization;
using BlazorML.Core.Domain;

namespace BlazorML.ML.Training;

/// <summary>
/// What an algorithm node puts on its <see cref="PortType.UntrainedModel"/> port: which trainer
/// to use and the settings the user chose. Kept as plain values so it can be logged, compared
/// across sweep iterations, and recorded on the run.
/// </summary>
public sealed class TrainerSpec
{
    public required string ModuleId { get; init; }
    public required string DisplayName { get; init; }
    public required MlTask Task { get; init; }
    public Dictionary<string, string?> Parameters { get; init; } = new();

    public TrainerSpec With(string name, string? value)
    {
        var copy = new TrainerSpec
        {
            ModuleId = ModuleId,
            DisplayName = DisplayName,
            Task = Task,
            Parameters = new Dictionary<string, string?>(Parameters)
        };

        copy.Parameters[name] = value;
        return copy;
    }

    public int Int(string name, int fallback) =>
        Parameters.TryGetValue(name, out var v) && int.TryParse(v, NumberStyles.Any, CultureInfo.InvariantCulture, out var parsed)
            ? parsed : fallback;

    public float Float(string name, float fallback) =>
        Parameters.TryGetValue(name, out var v) && float.TryParse(v, NumberStyles.Any, CultureInfo.InvariantCulture, out var parsed)
            ? parsed : fallback;

    public double Double(string name, double fallback) =>
        Parameters.TryGetValue(name, out var v) && double.TryParse(v, NumberStyles.Any, CultureInfo.InvariantCulture, out var parsed)
            ? parsed : fallback;

    public bool Bool(string name, bool fallback) =>
        Parameters.TryGetValue(name, out var v) && bool.TryParse(v, out var parsed) ? parsed : fallback;

    public string? Text(string name) =>
        Parameters.TryGetValue(name, out var v) && !string.IsNullOrWhiteSpace(v) ? v : null;

    public override string ToString() =>
        $"{DisplayName} ({string.Join(", ", Parameters.Where(p => p.Value is not null).Select(p => $"{p.Key}={p.Value}"))})";
}
