using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Services;

namespace HFAppGen.Services;

/// <summary>
/// Semantic Kernel chat completion over the Anthropic Messages API.
/// </summary>
/// <remarks>
/// <para>
/// Semantic Kernel ships first-party connectors for OpenAI, Azure OpenAI, Google and Ollama, but
/// there is no official Anthropic one, so Claude support is written here against the raw Messages
/// API.
/// </para>
/// <para>
/// Three things differ from the OpenAI shape and are handled explicitly: the system prompt is a
/// top-level <c>system</c> field rather than a message with a role, tool results travel as
/// <c>tool_result</c> content blocks inside a user message rather than as their own role, and
/// <c>max_tokens</c> is required rather than optional.
/// </para>
/// </remarks>
public sealed class AnthropicChatCompletionService : IChatCompletionService
{
    private const string ApiVersion = "2023-06-01";

    private readonly HttpClient _http;
    private readonly string _model;
    private readonly int _maxTokens;

    /// <summary>Creates a service for one Claude model.</summary>
    public AnthropicChatCompletionService(string apiKey, string model, string? endpoint = null,
        int maxTokens = 8192, TimeSpan? timeout = null)
    {
        _model = model;
        _maxTokens = maxTokens;

        _http = new HttpClient
        {
            BaseAddress = new Uri(string.IsNullOrWhiteSpace(endpoint) ? "https://api.anthropic.com" : endpoint),
            Timeout = timeout ?? TimeSpan.FromMinutes(3),
        };
        _http.DefaultRequestHeaders.Add("x-api-key", apiKey);
        _http.DefaultRequestHeaders.Add("anthropic-version", ApiVersion);
        _http.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, object?> Attributes { get; } =
        new Dictionary<string, object?> { [AIServiceExtensions.ModelIdKey] = "anthropic" };

    /// <inheritdoc />
    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var request = BuildRequest(chatHistory, executionSettings, stream: false);

        using var response = await _http.PostAsync("/v1/messages",
            new StringContent(request.ToJsonString(), Encoding.UTF8, "application/json"),
            cancellationToken);

        var body = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException($"Anthropic returned {(int)response.StatusCode}: {body}");

        var payload = JsonNode.Parse(body);
        var text = new StringBuilder();
        foreach (var block in payload?["content"]?.AsArray() ?? [])
        {
            if ((string?)block?["type"] == "text") text.Append((string?)block["text"]);
        }

        return [new ChatMessageContent(AuthorRole.Assistant, text.ToString())];
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var request = BuildRequest(chatHistory, executionSettings, stream: true);

        using var message = new HttpRequestMessage(HttpMethod.Post, "/v1/messages")
        {
            Content = new StringContent(request.ToJsonString(), Encoding.UTF8, "application/json"),
        };

        using var response = await _http.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            var error = await response.Content.ReadAsStringAsync(cancellationToken);
            throw new HttpRequestException($"Anthropic returned {(int)response.StatusCode}: {error}");
        }

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        // Server-sent events: data lines carry JSON, everything else is framing.
        while (!reader.EndOfStream)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var line = await reader.ReadLineAsync(cancellationToken);
            if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("data:", StringComparison.Ordinal)) continue;

            var json = line[5..].Trim();
            if (json is "[DONE]") break;

            JsonNode? node;
            try { node = JsonNode.Parse(json); }
            catch (JsonException) { continue; }

            if ((string?)node?["type"] != "content_block_delta") continue;

            var text = (string?)node["delta"]?["text"];
            if (!string.IsNullOrEmpty(text))
                yield return new StreamingChatMessageContent(AuthorRole.Assistant, text);
        }
    }

    private JsonObject BuildRequest(ChatHistory history, PromptExecutionSettings? settings, bool stream)
    {
        var messages = new JsonArray();
        var system = new StringBuilder();

        foreach (var message in history)
        {
            var text = message.Content ?? "";

            // Anthropic takes the system prompt as a top-level field, not as a message.
            if (message.Role == AuthorRole.System)
            {
                if (system.Length > 0) system.Append("\n\n");
                system.Append(text);
                continue;
            }

            if (string.IsNullOrWhiteSpace(text)) continue;

            var role = message.Role == AuthorRole.Assistant ? "assistant" : "user";

            // Consecutive same-role messages are rejected, so they are merged.
            if (messages.Count > 0 && (string?)messages[^1]!["role"] == role)
            {
                var existing = messages[^1]!["content"]!.AsArray();
                existing.Add(new JsonObject { ["type"] = "text", ["text"] = text });
                continue;
            }

            messages.Add(new JsonObject
            {
                ["role"] = role,
                ["content"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = text }),
            });
        }

        // The API requires at least one message and requires it to be from the user.
        if (messages.Count == 0)
            messages.Add(new JsonObject
            {
                ["role"] = "user",
                ["content"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = "Hello" }),
            });

        var request = new JsonObject
        {
            ["model"] = _model,
            ["max_tokens"] = _maxTokens,
            ["messages"] = messages,
            ["stream"] = stream,
        };

        if (system.Length > 0) request["system"] = system.ToString();

        if (settings?.ExtensionData?.TryGetValue("temperature", out var temperature) == true
            && temperature is not null)
        {
            request["temperature"] = Convert.ToDouble(temperature);
        }

        return request;
    }
}
