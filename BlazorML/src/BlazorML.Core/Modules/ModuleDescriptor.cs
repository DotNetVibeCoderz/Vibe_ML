using BlazorML.Core.Domain;

namespace BlazorML.Core.Modules;

/// <summary>
/// Static description of one draggable module. The palette, the node renderer, the parameter
/// inspector and the run engine all read from this, so a new module is added in exactly one place.
/// </summary>
public sealed class ModuleDescriptor
{
    public required string Id { get; init; }
    public required string Name { get; init; }
    public required ModuleCategory Category { get; init; }

    /// <summary>One line, written for someone deciding whether to drop this on the canvas.</summary>
    public required string Description { get; init; }

    /// <summary>Set for algorithm modules so the trainer can be matched to a Train Model node.</summary>
    public MlTask Task { get; init; } = MlTask.None;

    public IReadOnlyList<PortSpec> Inputs { get; init; } = Array.Empty<PortSpec>();
    public IReadOnlyList<PortSpec> Outputs { get; init; } = Array.Empty<PortSpec>();
    public IReadOnlyList<ParameterSpec> Parameters { get; init; } = Array.Empty<ParameterSpec>();

    /// <summary>Search terms beyond the name, so "regresi" and "linear" both find the same module.</summary>
    public IReadOnlyList<string> Keywords { get; init; } = Array.Empty<string>();

    public bool IsExperimental { get; init; }

    public string DefaultsFor(string parameterName) =>
        Parameters.FirstOrDefault(p => p.Name == parameterName)?.Default ?? string.Empty;

    public Dictionary<string, string?> BuildDefaultParameters() =>
        Parameters.ToDictionary(p => p.Name, p => (string?)p.Default);
}

public sealed class PortSpec
{
    public required string Name { get; init; }
    public required PortType Type { get; init; }
    public bool Required { get; init; } = true;
    public string? Description { get; init; }
}

public sealed class ParameterSpec
{
    public required string Name { get; init; }
    public required string Label { get; init; }
    public required ParameterKind Kind { get; init; }
    public string? Default { get; init; }

    /// <summary>Options for <see cref="ParameterKind.Choice"/>, value and display label.</summary>
    public IReadOnlyList<ChoiceOption> Choices { get; init; } = Array.Empty<ChoiceOption>();

    public double? Min { get; init; }
    public double? Max { get; init; }
    public double? Step { get; init; }

    /// <summary>Shown under the field. Explains the effect of the setting, not its type.</summary>
    public string? Help { get; init; }

    /// <summary>
    /// True when the module genuinely cannot run without a value. Declared rather than inferred:
    /// guessing from "has no default" marked optional fields like a positive-class override or a
    /// rename target as required, and validation that cries wolf is validation people stop reading.
    /// </summary>
    public bool Required { get; init; }

    /// <summary>Only show this field when the named parameter equals <see cref="VisibleWhenValue"/>.</summary>
    public string? VisibleWhen { get; init; }
    public string? VisibleWhenValue { get; init; }

    /// <summary>For code parameters: which language the editor should treat the value as.</summary>
    public ScriptLanguage? Language { get; init; }
}

public sealed record ChoiceOption(string Value, string Label);
