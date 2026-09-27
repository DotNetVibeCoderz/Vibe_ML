using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using Microsoft.Extensions.Logging;

namespace MediaPipeNet.Tasks;

/// <summary>Options shared by every task: where models come from and how they run.</summary>
public sealed record BaseOptions
{
    /// <summary>Default options (default <see cref="ModelStore"/>, automatic execution provider).</summary>
    public static BaseOptions Default { get; } = new();

    /// <summary>Model store used to resolve model files; <see cref="ModelStore.Default"/> when null.</summary>
    public ModelStore? ModelStore { get; init; }

    /// <summary>A directory searched first for model files (e.g. a folder you ship yourself).</summary>
    public string? ModelDirectory { get; init; }

    /// <summary>Explicit model file per model id (see <see cref="ModelCatalog"/>), overriding every other source.</summary>
    public IReadOnlyDictionary<string, string>? ModelPaths { get; init; }

    /// <summary>Execution provider and threading.</summary>
    public InferenceOptions Inference { get; init; } = InferenceOptions.Default;

    /// <summary>Logger factory for task diagnostics.</summary>
    public ILoggerFactory? LoggerFactory { get; init; }
}
