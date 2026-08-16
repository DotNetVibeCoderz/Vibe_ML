using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using LocalGen.Core.Engines;
using LocalGen.Core.Inference;
using LocalGen.Runtime.Sessions;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace LocalGen.Kernel;

/// <summary>
/// Exposes locally loaded models to Semantic Kernel as an <see cref="IChatCompletionService"/>.
/// </summary>
/// <remarks>
/// This binds to <see cref="ModelSessionManager"/> rather than going back out over HTTP, so the
/// desktop Playground shares the same loaded weights as the web service instead of loading a
/// second copy. Tool calls are surfaced as <see cref="FunctionCallContent"/>; invocation is left
/// to <see cref="LocalGenAgent"/>, which keeps the transport layer free of orchestration policy.
/// </remarks>
public sealed class LocalGenChatCompletionService : IChatCompletionService
{
    private readonly ModelSessionManager _sessions;
    private readonly string _defaultModelId;

    public LocalGenChatCompletionService(ModelSessionManager sessions, string defaultModelId)
    {
        _sessions = sessions;
        _defaultModelId = defaultModelId;
    }

    public IReadOnlyDictionary<string, object?> Attributes { get; } =
        new Dictionary<string, object?> { ["Provider"] = "LocalGen" };

    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Microsoft.SemanticKernel.Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var request = BuildRequest(chatHistory, executionSettings, kernel);

        using var lease = await _sessions
            .AcquireAsync(request.Model, null, cancellationToken)
            .ConfigureAwait(false);

        var response = await lease.Session.CompleteAsync(request, cancellationToken).ConfigureAwait(false);

        var items = new ChatMessageContentItemCollection();

        if (!string.IsNullOrEmpty(response.Message.Text))
        {
            items.Add(new TextContent(response.Message.Text));
        }

        foreach (var call in response.Message.ToolCalls)
        {
            items.Add(ToFunctionCall(call));
        }

        return
        [
            new ChatMessageContent(AuthorRole.Assistant, items)
            {
                ModelId = response.Model,
                Metadata = new Dictionary<string, object?>
                {
                    ["Usage"] = response.Usage,
                    ["FinishReason"] = response.FinishReason.ToString(),
                    ["Engine"] = lease.Session.Engine.ToString()
                }
            }
        ];
    }

    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Microsoft.SemanticKernel.Kernel? kernel = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var request = BuildRequest(chatHistory, executionSettings, kernel);

        using var lease = await _sessions
            .AcquireAsync(request.Model, null, cancellationToken)
            .ConfigureAwait(false);

        await foreach (var chunk in lease.Session
            .StreamAsync(request, cancellationToken)
            .ConfigureAwait(false))
        {
            if (chunk.Delta.Length > 0)
            {
                yield return new StreamingChatMessageContent(AuthorRole.Assistant, chunk.Delta)
                {
                    ModelId = request.Model
                };
            }

            // Tool calls only become actionable once complete, so they ride on their own chunk.
            foreach (var call in chunk.ToolCalls)
            {
                var content = new StreamingChatMessageContent(AuthorRole.Assistant, null)
                {
                    ModelId = request.Model
                };

                content.Items.Add(new StreamingFunctionCallUpdateContent(
                    call.Id,
                    call.Name,
                    call.ArgumentsJson));

                yield return content;
            }
        }
    }

    /// <summary>
    /// Maps SK's chat history and settings onto a LocalGen request, including any kernel
    /// functions the caller made available through <see cref="FunctionChoiceBehavior"/>.
    /// </summary>
    private ChatRequest BuildRequest(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings,
        Microsoft.SemanticKernel.Kernel? kernel)
    {
        var settings = LocalGenPromptExecutionSettings.From(executionSettings);

        return new ChatRequest
        {
            Model = settings.ModelId ?? executionSettings?.ModelId ?? _defaultModelId,
            Messages = [.. chatHistory.Select(ToChatMessage)],
            Tools = ResolveTools(executionSettings, kernel, chatHistory),
            Options = settings.ToGenerationOptions()
        };
    }

    /// <summary>
    /// Reads the functions the caller advertised. SK models this through the choice behaviour
    /// rather than a plain list, so that a caller can expose a subset of the kernel's plugins.
    /// </summary>
    private static IReadOnlyList<ToolDefinition> ResolveTools(
        PromptExecutionSettings? executionSettings,
        Microsoft.SemanticKernel.Kernel? kernel,
        ChatHistory chatHistory)
    {
        if (executionSettings?.FunctionChoiceBehavior is not { } behavior || kernel is null)
        {
            return [];
        }

        var configuration = behavior.GetConfiguration(
            new FunctionChoiceBehaviorConfigurationContext(chatHistory) { Kernel = kernel });

        if (configuration.Functions is not { Count: > 0 } functions)
        {
            return [];
        }

        return
        [
            .. functions.Select(static function => new ToolDefinition
            {
                // Plugin-qualified names keep two plugins from colliding on a common verb.
                Name = string.IsNullOrEmpty(function.PluginName)
                    ? function.Name
                    : $"{function.PluginName}-{function.Name}",
                Description = function.Description,
                ParametersJsonSchema = BuildParameterSchema(function.Metadata)
            })
        ];
    }

    /// <summary>Renders a kernel function's parameters as the JSON Schema the model is shown.</summary>
    private static string BuildParameterSchema(KernelFunctionMetadata metadata)
    {
        var properties = new Dictionary<string, object>();
        var required = new List<string>();

        foreach (var parameter in metadata.Parameters)
        {
            // A parameter's own schema is richer than anything reconstructed from its CLR type.
            if (parameter.Schema is { } schema)
            {
                properties[parameter.Name] = JsonSerializer.Deserialize<JsonElement>(schema.ToString());
            }
            else
            {
                properties[parameter.Name] = new Dictionary<string, object>
                {
                    ["type"] = MapJsonType(parameter.ParameterType),
                    ["description"] = parameter.Description ?? string.Empty
                };
            }

            if (parameter.IsRequired)
            {
                required.Add(parameter.Name);
            }
        }

        return JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["type"] = "object",
            ["properties"] = properties,
            ["required"] = required
        });
    }

    private static string MapJsonType(Type? type) => type switch
    {
        null => "string",
        _ when type == typeof(bool) => "boolean",
        _ when type == typeof(int) || type == typeof(long) => "integer",
        _ when type == typeof(double) || type == typeof(float) || type == typeof(decimal) => "number",
        _ when type.IsArray => "array",
        _ when type == typeof(string) => "string",
        _ => "object"
    };

    private static FunctionCallContent ToFunctionCall(ToolCall call)
    {
        // Names are round-tripped through the "Plugin-Function" convention used above.
        var separator = call.Name.IndexOf('-');
        var pluginName = separator > 0 ? call.Name[..separator] : null;
        var functionName = separator > 0 ? call.Name[(separator + 1)..] : call.Name;

        KernelArguments? arguments = null;

        try
        {
            var parsed = JsonSerializer.Deserialize<Dictionary<string, object?>>(call.ArgumentsJson);
            if (parsed is not null)
            {
                arguments = [];
                foreach (var (key, value) in parsed)
                {
                    arguments[key] = value is JsonElement element ? ReadJsonValue(element) : value;
                }
            }
        }
        catch (JsonException)
        {
            // Leave the arguments null; SK reports the resulting binding failure to the model,
            // which is more useful than throwing here.
        }

        return new FunctionCallContent(functionName, pluginName, call.Id, arguments);
    }

    private static object? ReadJsonValue(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.String => element.GetString(),
        JsonValueKind.Number => element.TryGetInt64(out var l) ? l : element.GetDouble(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Null => null,
        _ => element.GetRawText()
    };

    /// <summary>Flattens an SK message, including tool results, into a LocalGen message.</summary>
    private static Core.Inference.ChatMessage ToChatMessage(ChatMessageContent message)
    {
        var role = message.Role.Label.ToLowerInvariant() switch
        {
            "system" or "developer" => ChatRole.System,
            "assistant" => ChatRole.Assistant,
            "tool" => ChatRole.Tool,
            _ => ChatRole.User
        };

        var parts = new List<ContentPart>();
        var toolCalls = new List<ToolCall>();
        string? toolCallId = null;

        foreach (var item in message.Items)
        {
            switch (item)
            {
                case TextContent text when !string.IsNullOrEmpty(text.Text):
                    parts.Add(new ContentPart.Text(text.Text));
                    break;

                case ImageContent { Data: { } data } image:
                    parts.Add(new ContentPart.Image(data, image.MimeType ?? "image/png"));
                    break;

                // An image referenced by URL rather than inlined. The bytes are left for the host
                // to resolve, so an attachment is stored once instead of copied into every turn.
                case ImageContent { Uri: { } uri } image:
                    parts.Add(new ContentPart.Image(
                        ReadOnlyMemory<byte>.Empty,
                        image.MimeType ?? "image/png",
                        uri.ToString()));
                    break;

                case FunctionCallContent call:
                    toolCalls.Add(new ToolCall(
                        call.Id ?? Guid.NewGuid().ToString("N"),
                        string.IsNullOrEmpty(call.PluginName)
                            ? call.FunctionName
                            : $"{call.PluginName}-{call.FunctionName}",
                        JsonSerializer.Serialize(call.Arguments ?? [])));
                    break;

                case FunctionResultContent result:
                    toolCallId = result.CallId;
                    parts.Add(new ContentPart.Text(result.Result?.ToString() ?? string.Empty));
                    break;
            }
        }

        if (parts.Count == 0 && !string.IsNullOrEmpty(message.Content))
        {
            parts.Add(new ContentPart.Text(message.Content));
        }

        return new Core.Inference.ChatMessage
        {
            Role = role,
            Content = parts,
            ToolCalls = toolCalls,
            ToolCallId = toolCallId,
            Name = message.AuthorName
        };
    }
}
