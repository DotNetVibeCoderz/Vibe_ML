using LocalGen.Core.Inference;
using LocalGen.Core.Models;

namespace LocalGen.Core.Engines;

/// <summary>
/// A pluggable inference backend. Implementations own model loading and token generation;
/// everything above this interface (API, SDK, CLI, UI) is written against these contracts only,
/// so backend-specific types never leak upwards.
/// </summary>
public interface IInferenceEngine
{
    EngineDescriptor Descriptor { get; }

    /// <summary>
    /// Whether this backend can run on the current machine — native libraries present,
    /// runtime installed, drivers available. Probed once at startup and surfaced in the UI.
    /// </summary>
    ValueTask<EngineAvailability> ProbeAsync(CancellationToken cancellationToken = default);

    /// <summary>Whether this backend is able to serve the given model.</summary>
    bool CanServe(ModelDescriptor model);

    /// <summary>
    /// Loads weights and returns a session ready for inference. Callers are expected to keep
    /// sessions alive and reuse them; loading is expensive.
    /// </summary>
    ValueTask<IModelSession> LoadAsync(
        ModelDescriptor model,
        ModelLoadOptions options,
        CancellationToken cancellationToken = default);
}

/// <summary>Outcome of probing a backend on the current machine.</summary>
public sealed record EngineAvailability
{
    public required bool IsAvailable { get; init; }

    /// <summary>Why the backend is unavailable, shown to the user. Empty when available.</summary>
    public string Reason { get; init; } = string.Empty;

    /// <summary>Devices actually usable right now, which may be narrower than the declared capabilities.</summary>
    public IReadOnlyList<DeviceKind> AvailableDevices { get; init; } = [];

    /// <summary>
    /// The individual accelerators the backend registered, in the order a tensor split addresses
    /// them. Its length is what decides whether multi-GPU is available on this machine at all.
    /// </summary>
    public IReadOnlyList<AcceleratorDevice> Accelerators { get; init; } = [];

    /// <summary>Backend or native library version, for the About and diagnostics screens.</summary>
    public string Version { get; init; } = string.Empty;

    public static EngineAvailability Available(
        string version = "",
        params DeviceKind[] devices) => new()
    {
        IsAvailable = true,
        Version = version,
        AvailableDevices = devices.Length > 0 ? devices : [DeviceKind.Cpu]
    };

    public static EngineAvailability Unavailable(string reason) => new()
    {
        IsAvailable = false,
        Reason = reason
    };
}

/// <summary>
/// A loaded model, ready to generate. Sessions are expensive to create and safe to share;
/// implementations must serialise concurrent generation internally.
/// </summary>
public interface IModelSession : IAsyncDisposable
{
    ModelDescriptor Model { get; }

    EngineKind Engine { get; }

    DeviceKind Device { get; }

    /// <summary>Effective context window for this session, after load-time overrides.</summary>
    int ContextSize { get; }

    /// <summary>
    /// Generates a response incrementally. This is the primitive every other entry point is
    /// built on — <see cref="InferenceSessionExtensions.CompleteAsync"/> aggregates it.
    /// </summary>
    IAsyncEnumerable<ChatStreamChunk> StreamAsync(
        ChatRequest request,
        CancellationToken cancellationToken = default);

    /// <summary>Produces embedding vectors. Throws when the model has no embedding support.</summary>
    ValueTask<EmbeddingResponse> EmbedAsync(
        EmbeddingRequest request,
        CancellationToken cancellationToken = default);

    /// <summary>Token count for a string under this model's tokenizer.</summary>
    int CountTokens(string text);
}

public static class InferenceSessionExtensions
{
    /// <summary>
    /// Drains a streaming generation into a single response. Kept as an extension so backends
    /// only ever implement the streaming path.
    /// </summary>
    public static async Task<ChatResponse> CompleteAsync(
        this IModelSession session,
        ChatRequest request,
        CancellationToken cancellationToken = default)
    {
        var startedAt = System.Diagnostics.Stopwatch.GetTimestamp();
        var text = new System.Text.StringBuilder();
        var toolCalls = new List<ToolCall>();
        var finishReason = FinishReason.Stop;
        var usage = TokenUsage.Empty;

        await foreach (var chunk in session.StreamAsync(request, cancellationToken).ConfigureAwait(false))
        {
            text.Append(chunk.Delta);

            if (chunk.ToolCalls.Count > 0)
            {
                toolCalls.AddRange(chunk.ToolCalls);
            }

            if (chunk.FinishReason != FinishReason.None)
            {
                finishReason = chunk.FinishReason;
            }

            if (chunk.Usage is not null)
            {
                usage = chunk.Usage;
            }
        }

        return new ChatResponse
        {
            Model = request.Model,
            FinishReason = finishReason,
            Usage = usage,
            Duration = System.Diagnostics.Stopwatch.GetElapsedTime(startedAt),
            Message = new ChatMessage
            {
                Role = ChatRole.Assistant,
                Content = [new ContentPart.Text(text.ToString())],
                ToolCalls = toolCalls
            }
        };
    }
}
