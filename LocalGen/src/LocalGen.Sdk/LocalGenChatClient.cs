using System.Runtime.CompilerServices;
using System.Text.Json;
using LocalGen.Core.Protocol;
using Microsoft.Extensions.AI;

namespace LocalGen.Sdk;

/// <summary>
/// Adapts <see cref="LocalGenClient"/> to <see cref="IChatClient"/>.
/// </summary>
/// <remarks>
/// <c>Microsoft.Extensions.AI</c> is the common abstraction that Semantic Kernel, the Agent
/// Framework and most .NET AI middleware build on. Implementing it means LocalGen drops into
/// those pipelines — including the function-invoking and caching decorators — without any
/// LocalGen-specific glue.
/// </remarks>
public sealed class LocalGenChatClient : IChatClient
{
    private readonly LocalGenClient _client;
    private readonly string _modelId;
    private readonly bool _ownsClient;

    public LocalGenChatClient(LocalGenClient client, string modelId, bool ownsClient = false)
    {
        _client = client;
        _modelId = modelId;
        _ownsClient = ownsClient;
    }

    public LocalGenChatClient(string endpoint, string modelId, string? apiKey = null)
        : this(new LocalGenClient(endpoint, apiKey), modelId, ownsClient: true)
    {
    }

    public async Task<Microsoft.Extensions.AI.ChatResponse> GetResponseAsync(
        IEnumerable<Microsoft.Extensions.AI.ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var request = BuildRequest(messages, options);
        var response = await _client.ChatAsync(request, cancellationToken).ConfigureAwait(false);

        var choice = response.Choices.FirstOrDefault();
        var text = LocalGenClient.ReadContent(choice?.Message?.Content);

        var message = new Microsoft.Extensions.AI.ChatMessage(
            Microsoft.Extensions.AI.ChatRole.Assistant,
            text);

        foreach (var call in choice?.Message?.ToolCalls ?? [])
        {
            message.Contents.Add(new FunctionCallContent(
                call.Id,
                call.Function.Name,
                ParseArguments(call.Function.Arguments)));
        }

        return new Microsoft.Extensions.AI.ChatResponse(message)
        {
            ResponseId = response.Id,
            ModelId = response.Model,
            CreatedAt = DateTimeOffset.FromUnixTimeSeconds(response.Created),
            FinishReason = MapFinishReason(choice?.FinishReason),
            Usage = response.Usage is null ? null : new UsageDetails
            {
                InputTokenCount = response.Usage.PromptTokens,
                OutputTokenCount = response.Usage.CompletionTokens,
                TotalTokenCount = response.Usage.TotalTokens
            }
        };
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<Microsoft.Extensions.AI.ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var request = BuildRequest(messages, options);

        await foreach (var chunk in _client
            .StreamChatAsync(request, cancellationToken)
            .ConfigureAwait(false))
        {
            var choice = chunk.Choices.FirstOrDefault();
            if (choice?.Delta is null)
            {
                continue;
            }

            var update = new ChatResponseUpdate
            {
                ResponseId = chunk.Id,
                ModelId = chunk.Model,
                Role = Microsoft.Extensions.AI.ChatRole.Assistant,
                FinishReason = MapFinishReason(choice.FinishReason)
            };

            if (!string.IsNullOrEmpty(choice.Delta.Content))
            {
                update.Contents.Add(new Microsoft.Extensions.AI.TextContent(choice.Delta.Content));
            }

            foreach (var call in choice.Delta.ToolCalls ?? [])
            {
                update.Contents.Add(new FunctionCallContent(
                    call.Id,
                    call.Function.Name,
                    ParseArguments(call.Function.Arguments)));
            }

            if (chunk.Usage is not null)
            {
                update.Contents.Add(new UsageContent(new UsageDetails
                {
                    InputTokenCount = chunk.Usage.PromptTokens,
                    OutputTokenCount = chunk.Usage.CompletionTokens,
                    TotalTokenCount = chunk.Usage.TotalTokens
                }));
            }

            if (update.Contents.Count > 0 || update.FinishReason is not null)
            {
                yield return update;
            }
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceKey is not null)
        {
            return null;
        }

        return serviceType == typeof(LocalGenClient) ? _client
            : serviceType.IsInstanceOfType(this) ? this
            : null;
    }

    private ChatCompletionRequest BuildRequest(
        IEnumerable<Microsoft.Extensions.AI.ChatMessage> messages,
        ChatOptions? options) => new()
        {
            Model = options?.ModelId ?? _modelId,
            Messages = [.. messages.Select(ToOpenAiMessage)],
            Temperature = options?.Temperature,
            TopP = options?.TopP,
            TopK = options?.TopK,
            MaxTokens = options?.MaxOutputTokens,
            PresencePenalty = options?.PresencePenalty,
            FrequencyPenalty = options?.FrequencyPenalty,
            Seed = options?.Seed,
            Stop = options?.StopSequences is { Count: > 0 } stops
                ? JsonSerializer.SerializeToElement(stops)
                : null,
            Tools = options?.Tools?.OfType<AIFunction>().Select(static function => new OpenAiTool
            {
                Function = new OpenAiFunction
                {
                    Name = function.Name,
                    Description = function.Description,
                    Parameters = function.JsonSchema
                }
            }).ToList()
        };

    private static OpenAiMessage ToOpenAiMessage(Microsoft.Extensions.AI.ChatMessage message)
    {
        var role = message.Role.Value.ToLowerInvariant() switch
        {
            "system" => "system",
            "assistant" => "assistant",
            "tool" => "tool",
            _ => "user"
        };

        var result = new OpenAiMessage
        {
            Role = role,
            Content = JsonSerializer.SerializeToElement(message.Text),
            Name = message.AuthorName
        };

        var toolCalls = message.Contents.OfType<FunctionCallContent>().ToList();
        if (toolCalls.Count > 0)
        {
            result.ToolCalls =
            [
                .. toolCalls.Select(static (call, index) => new OpenAiToolCall
                {
                    Id = call.CallId,
                    Index = index,
                    Function = new OpenAiFunctionCall
                    {
                        Name = call.Name,
                        Arguments = JsonSerializer.Serialize(call.Arguments ?? new Dictionary<string, object?>())
                    }
                })
            ];
        }

        // A tool result carries the id of the call it answers, which the server needs to pair them.
        var resultContent = message.Contents.OfType<FunctionResultContent>().FirstOrDefault();
        if (resultContent is not null)
        {
            result.ToolCallId = resultContent.CallId;
            result.Content = JsonSerializer.SerializeToElement(resultContent.Result?.ToString() ?? string.Empty);
        }

        return result;
    }

    private static IDictionary<string, object?>? ParseArguments(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return null;
        }

        try
        {
            return JsonSerializer.Deserialize<Dictionary<string, object?>>(json);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static ChatFinishReason? MapFinishReason(string? reason) => reason switch
    {
        "stop" => ChatFinishReason.Stop,
        "length" => ChatFinishReason.Length,
        "tool_calls" => ChatFinishReason.ToolCalls,
        "content_filter" => ChatFinishReason.ContentFilter,
        _ => null
    };

    public void Dispose()
    {
        if (_ownsClient)
        {
            _client.Dispose();
        }
    }
}
