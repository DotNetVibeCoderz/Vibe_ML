using System.Runtime.CompilerServices;
using System.Text.Json;
using LocalGen.Core.Engines;
using LocalGen.Core.Inference;
using LocalGen.Core.Models;
using LocalGen.Core.Protocol;
using LocalGen.Sdk;

namespace LocalGen.Engines.FoundryLocal;

/// <summary>
/// A session backed by a remote OpenAI-compatible endpoint.
/// </summary>
/// <remarks>
/// Used for backends that run their own server process rather than loading weights in-process —
/// Foundry Local, and any external OpenAI-compatible gateway. Reusing
/// <see cref="LocalGenClient"/> here means the streaming and error handling are the same code
/// paths the SDK already exercises.
/// </remarks>
internal sealed class OpenAiHttpSession : IModelSession
{
    private readonly LocalGenClient _client;
    private readonly bool _ownsClient;
    private readonly string _remoteModelId;

    public OpenAiHttpSession(
        ModelDescriptor model,
        LocalGenClient client,
        string remoteModelId,
        EngineKind engine,
        bool ownsClient = true)
    {
        Model = model;
        Engine = engine;
        _client = client;
        _remoteModelId = remoteModelId;
        _ownsClient = ownsClient;
    }

    public ModelDescriptor Model { get; }

    public EngineKind Engine { get; }

    /// <summary>The remote process owns placement, so there is no local device to report.</summary>
    public DeviceKind Device => DeviceKind.Auto;

    public int ContextSize => Model.ContextLength;

    public async IAsyncEnumerable<ChatStreamChunk> StreamAsync(
        ChatRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var wire = new ChatCompletionRequest
        {
            Model = _remoteModelId,
            Temperature = request.Options.Temperature,
            TopP = request.Options.TopP,
            MaxTokens = request.Options.MaxTokens,
            PresencePenalty = request.Options.PresencePenalty,
            FrequencyPenalty = request.Options.FrequencyPenalty,
            Seed = request.Options.Seed,
            StreamOptions = new StreamOptions { IncludeUsage = true },
            Stop = request.Options.StopSequences.Count > 0
                ? JsonSerializer.SerializeToElement(request.Options.StopSequences)
                : null,
            Messages = [.. request.Messages.Select(ToWireMessage)],
            Tools = request.Tools.Count == 0
                ? null
                : [.. request.Tools.Select(static tool => new OpenAiTool
                {
                    Function = new OpenAiFunction
                    {
                        Name = tool.Name,
                        Description = tool.Description,
                        Parameters = JsonSerializer.Deserialize<JsonElement>(tool.ParametersJsonSchema)
                    }
                })]
        };

        await foreach (var chunk in _client.StreamChatAsync(wire, cancellationToken).ConfigureAwait(false))
        {
            var choice = chunk.Choices.FirstOrDefault();

            yield return new ChatStreamChunk
            {
                Delta = choice?.Delta?.Content ?? string.Empty,
                FinishReason = OpenAiMapper.FromFinishReason(choice?.FinishReason),
                ToolCalls = choice?.Delta?.ToolCalls is { Count: > 0 } calls
                    ? [.. calls.Select(static c => new ToolCall(c.Id, c.Function.Name, c.Function.Arguments))]
                    : [],
                Usage = chunk.Usage is null ? null : new TokenUsage
                {
                    PromptTokens = chunk.Usage.PromptTokens,
                    CompletionTokens = chunk.Usage.CompletionTokens
                }
            };
        }
    }

    private static OpenAiMessage ToWireMessage(ChatMessage message) => new()
    {
        Role = message.Role switch
        {
            ChatRole.System => "system",
            ChatRole.Assistant => "assistant",
            ChatRole.Tool => "tool",
            _ => "user"
        },
        Name = message.Name,
        ToolCallId = message.ToolCallId,
        Content = JsonSerializer.SerializeToElement(message.Text),
        ToolCalls = message.ToolCalls.Count == 0
            ? null
            : [.. message.ToolCalls.Select(static (call, index) => new OpenAiToolCall
            {
                Id = call.Id,
                Index = index,
                Function = new OpenAiFunctionCall { Name = call.Name, Arguments = call.ArgumentsJson }
            })]
    };

    public async ValueTask<EmbeddingResponse> EmbedAsync(
        EmbeddingRequest request,
        CancellationToken cancellationToken = default)
    {
        var response = await _client
            .EmbedAsync(request.Inputs, _remoteModelId, cancellationToken)
            .ConfigureAwait(false);

        return new EmbeddingResponse
        {
            Model = request.Model,
            Embeddings = [.. response.Data.Select(static d => new ReadOnlyMemory<float>(d.Embedding))],
            Usage = new TokenUsage { PromptTokens = response.Usage?.PromptTokens ?? 0 }
        };
    }

    /// <summary>
    /// Approximates the token count. The remote tokenizer is not reachable, and roughly four
    /// characters per token holds well enough for the budgeting this is used for.
    /// </summary>
    public int CountTokens(string text) =>
        string.IsNullOrEmpty(text) ? 0 : (int)Math.Ceiling(text.Length / 4.0);

    public ValueTask DisposeAsync()
    {
        if (_ownsClient)
        {
            _client.Dispose();
        }

        return ValueTask.CompletedTask;
    }
}
