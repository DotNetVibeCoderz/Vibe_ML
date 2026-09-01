using LocalGen.Core.Engines;
using LocalGen.Core.Inference;

namespace LocalGen.Core.Modelfiles;

/// <summary>
/// A declarative model configuration, in the spirit of a Dockerfile: it names a base model and
/// layers prompt, sampling and context settings on top. Parsed by <see cref="ModelfileParser"/>.
/// </summary>
public sealed record Modelfile
{
    /// <summary>
    /// Base model: a path to local weights, or a reference such as
    /// <c>huggingface:Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf</c>.
    /// </summary>
    public required string From { get; init; }

    /// <summary>System prompt prepended to every conversation.</summary>
    public string? System { get; init; }

    /// <summary>Chat template override, in the model's own templating dialect.</summary>
    public string? Template { get; init; }

    /// <summary>LoRA adapters applied on top of the base weights, in order.</summary>
    public IReadOnlyList<string> Adapters { get; init; } = [];

    /// <summary>Sampling defaults, which per-request options may override.</summary>
    public GenerationOptions Parameters { get; init; } = GenerationOptions.Default;

    /// <summary>Load-time settings such as context size and GPU offload.</summary>
    public ModelLoadOptions LoadOptions { get; init; } = ModelLoadOptions.Default;

    /// <summary>Marks the model as an embedding model rather than a chat model.</summary>
    public bool IsEmbedding { get; init; }

    public string? License { get; init; }

    /// <summary>Few-shot examples seeded into every new conversation.</summary>
    public IReadOnlyList<ChatMessage> Messages { get; init; } = [];

    /// <summary>Free-form metadata from unrecognised <c>PARAMETER</c> keys.</summary>
    public IReadOnlyDictionary<string, string> Extras { get; init; } =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
}

/// <summary>Raised when a Modelfile cannot be parsed.</summary>
public sealed class ModelfileException(string message, int line)
    : Exception($"Modelfile error on line {line}: {message}")
{
    public int Line { get; } = line;
}
