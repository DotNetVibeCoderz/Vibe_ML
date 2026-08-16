using System.Text.Json;
using LocalGen.Core.Inference;
using LocalGen.Core.Models;

namespace LocalGen.Core.Protocol;

/// <summary>
/// Translates between the OpenAI wire format and LocalGen's domain types. Keeping the mapping in
/// one place is what lets the wire contract stay frozen while the internal model evolves.
/// </summary>
public static class OpenAiMapper
{
    /// <summary>Converts an inbound chat completion request into a domain request.</summary>
    public static ChatRequest ToChatRequest(ChatCompletionRequest request)
    {
        var messages = request.Messages.Select(ToChatMessage).ToList();

        return new ChatRequest
        {
            Model = request.Model,
            Messages = messages,
            Tools = request.Tools?.Select(ToToolDefinition).ToList() ?? [],
            Options = new GenerationOptions
            {
                Temperature = request.Temperature,
                TopP = request.TopP,
                TopK = request.TopK,
                PresencePenalty = request.PresencePenalty,
                FrequencyPenalty = request.FrequencyPenalty,
                // max_completion_tokens supersedes max_tokens in newer clients.
                MaxTokens = request.MaxCompletionTokens ?? request.MaxTokens,
                StopSequences = ReadStopSequences(request.Stop),
                Seed = request.Seed is { } seed ? unchecked((uint)seed) : null,
                JsonMode = request.ResponseFormat?.Type is "json_object" or "json_schema"
            }
        };
    }

    public static ChatMessage ToChatMessage(OpenAiMessage message)
    {
        var role = message.Role.ToLowerInvariant() switch
        {
            "system" or "developer" => ChatRole.System,
            "assistant" => ChatRole.Assistant,
            "tool" or "function" => ChatRole.Tool,
            _ => ChatRole.User
        };

        return new ChatMessage
        {
            Role = role,
            Name = message.Name,
            ToolCallId = message.ToolCallId,
            Content = ReadContent(message.Content),
            ToolCalls = message.ToolCalls?
                .Select(static c => new ToolCall(c.Id, c.Function.Name, c.Function.Arguments))
                .ToList() ?? []
        };
    }

    /// <summary>
    /// Reads the polymorphic <c>content</c> field, which is either a string or an array of typed
    /// parts. Image parts arrive as data URIs or as remote URLs; only data URIs can be decoded
    /// without a network fetch, so remote URLs are passed through as text for the model to see.
    /// </summary>
    private static IReadOnlyList<ContentPart> ReadContent(JsonElement? content)
    {
        if (content is not { } element || element.ValueKind == JsonValueKind.Null)
        {
            return [];
        }

        if (element.ValueKind == JsonValueKind.String)
        {
            return [new ContentPart.Text(element.GetString() ?? string.Empty)];
        }

        if (element.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        var parts = new List<ContentPart>();

        foreach (var item in element.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object ||
                !item.TryGetProperty("type", out var typeElement))
            {
                continue;
            }

            switch (typeElement.GetString())
            {
                case "text" when item.TryGetProperty("text", out var text):
                    parts.Add(new ContentPart.Text(text.GetString() ?? string.Empty));
                    break;

                case "image_url" when item.TryGetProperty("image_url", out var imageUrl):
                    var url = imageUrl.TryGetProperty("url", out var urlElement)
                        ? urlElement.GetString()
                        : null;

                    if (!string.IsNullOrEmpty(url))
                    {
                        parts.Add(ReadImage(url));
                    }

                    break;
            }
        }

        return parts;
    }

    /// <summary>
    /// Reads an <c>image_url</c> part. A data URI is decoded here; any other URL is carried as a
    /// source with no bytes, because fetching it is I/O and this layer performs none. The host
    /// resolves the ones it can serve — see <c>ResolveAttachments</c> in the server.
    /// </summary>
    private static ContentPart ReadImage(string url)
    {
        const string dataPrefix = "data:";

        if (!url.StartsWith(dataPrefix, StringComparison.OrdinalIgnoreCase))
        {
            return new ContentPart.Image(
                ReadOnlyMemory<byte>.Empty,
                GuessMediaTypeFromUrl(url),
                Source: url);
        }

        var comma = url.IndexOf(',');
        if (comma < 0)
        {
            return new ContentPart.Text("[malformed image data URI]");
        }

        var header = url[dataPrefix.Length..comma];
        var mediaType = header.Split(';')[0];

        try
        {
            return new ContentPart.Image(Convert.FromBase64String(url[(comma + 1)..]), mediaType);
        }
        catch (FormatException)
        {
            return new ContentPart.Text("[image data URI was not valid base64]");
        }
    }

    private static string GuessMediaTypeFromUrl(string url)
    {
        // Query strings and fragments are stripped so "photo.png?v=2" still reads as PNG.
        var path = url.Split('?')[0].Split('#')[0];

        return Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".png" => "image/png",
            ".jpg" or ".jpeg" => "image/jpeg",
            ".gif" => "image/gif",
            ".webp" => "image/webp",
            ".bmp" => "image/bmp",
            _ => "image/png"
        };
    }

    private static ToolDefinition ToToolDefinition(OpenAiTool tool) => new()
    {
        Name = tool.Function.Name,
        Description = tool.Function.Description,
        ParametersJsonSchema = tool.Function.Parameters?.GetRawText()
                               ?? """{"type":"object","properties":{}}"""
    };

    /// <summary>The <c>stop</c> field is either a single string or an array of them.</summary>
    public static IReadOnlyList<string> ReadStopSequences(JsonElement? stop) => stop switch
    {
        null => [],
        { ValueKind: JsonValueKind.String } element =>
            element.GetString() is { Length: > 0 } value ? [value] : [],
        { ValueKind: JsonValueKind.Array } element =>
            [.. element.EnumerateArray()
                .Select(static e => e.GetString())
                .Where(static s => !string.IsNullOrEmpty(s))
                .Select(static s => s!)],
        _ => []
    };

    /// <summary>Reads the <c>input</c> field of an embeddings request into plain strings.</summary>
    public static IReadOnlyList<string> ReadEmbeddingInputs(JsonElement input) => input.ValueKind switch
    {
        JsonValueKind.String => [input.GetString() ?? string.Empty],
        JsonValueKind.Array =>
            [.. input.EnumerateArray()
                .Where(static e => e.ValueKind == JsonValueKind.String)
                .Select(static e => e.GetString() ?? string.Empty)],
        _ => []
    };

    public static string ToFinishReason(FinishReason reason) => reason switch
    {
        FinishReason.Stop => "stop",
        FinishReason.Length => "length",
        FinishReason.ToolCalls => "tool_calls",
        FinishReason.Cancelled => "stop",
        FinishReason.Error => "error",
        _ => "stop"
    };

    public static FinishReason FromFinishReason(string? reason) => reason switch
    {
        "stop" => FinishReason.Stop,
        "length" => FinishReason.Length,
        "tool_calls" or "function_call" => FinishReason.ToolCalls,
        "error" => FinishReason.Error,
        null or "" => FinishReason.None,
        _ => FinishReason.Stop
    };

    /// <summary>Builds the non-streaming response body for a completed generation.</summary>
    public static ChatCompletionResponse ToCompletionResponse(
        ChatResponse response,
        string id,
        string? systemFingerprint = null) => new()
    {
        Id = id,
        Model = response.Model,
        Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        SystemFingerprint = systemFingerprint,
        Choices =
        [
            new ChatChoice
            {
                Index = 0,
                FinishReason = ToFinishReason(response.FinishReason),
                Message = new OpenAiMessage
                {
                    Role = "assistant",
                    Content = JsonSerializer.SerializeToElement(response.Message.Text),
                    ToolCalls = response.Message.ToolCalls.Count > 0
                        ? [.. response.Message.ToolCalls.Select(static (call, index) => new OpenAiToolCall
                        {
                            Id = call.Id,
                            Index = index,
                            Function = new OpenAiFunctionCall
                            {
                                Name = call.Name,
                                Arguments = call.ArgumentsJson
                            }
                        })]
                        : null
                }
            }
        ],
        Usage = ToUsage(response.Usage)
    };

    public static OpenAiUsage ToUsage(TokenUsage usage) => new()
    {
        PromptTokens = usage.PromptTokens,
        CompletionTokens = usage.CompletionTokens,
        TotalTokens = usage.TotalTokens
    };

    /// <summary>Projects a model onto <c>/v1/models</c>, including the LocalGen extension block.</summary>
    public static ModelData ToModelData(ModelDescriptor model) => new()
    {
        Id = model.Id,
        Created = (model.DownloadedAt ?? DateTimeOffset.UnixEpoch).ToUnixTimeSeconds(),
        OwnedBy = string.IsNullOrEmpty(model.Publisher) ? "localgen" : model.Publisher,
        LocalGen = new LocalGenModelInfo
        {
            Name = model.Name,
            Format = model.Format.ToString(),
            Quantization = model.Quantization.Name,
            SizeBytes = model.SizeBytes,
            ParameterCountB = model.ParameterCountB,
            ContextLength = model.ContextLength,
            Capabilities = [.. model.Capabilities.Select(static c => c.ToString())],
            Engines = [.. model.SupportedEngines.Select(static e => e.ToString())]
        }
    };
}
