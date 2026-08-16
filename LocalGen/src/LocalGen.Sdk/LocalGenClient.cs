using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using LocalGen.Core.Protocol;

namespace LocalGen.Sdk;

/// <summary>Connection settings for <see cref="LocalGenClient"/>.</summary>
public sealed class LocalGenClientOptions
{
    /// <summary>Base address of the server. Defaults to the LocalGen loopback address.</summary>
    public string Endpoint { get; set; } = "http://127.0.0.1:11434";

    /// <summary>Bearer token, when the server requires one.</summary>
    public string? ApiKey { get; set; }

    /// <summary>Model used when a call does not name one.</summary>
    public string? DefaultModel { get; set; }

    /// <summary>
    /// Per-request timeout. Generous by default: a cold model load on CPU can take a minute
    /// before the first token appears.
    /// </summary>
    public TimeSpan Timeout { get; set; } = TimeSpan.FromMinutes(10);
}

/// <summary>
/// A typed client for the LocalGen API.
/// </summary>
/// <remarks>
/// Speaks the OpenAI wire format, so it also works against OpenAI itself or any compatible
/// gateway — handy for code that has to fall back to a hosted model. The LocalGen-specific
/// management calls live on <see cref="Admin"/> and simply fail against other servers.
/// </remarks>
public sealed class LocalGenClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttpClient;
    private readonly LocalGenClientOptions _options;

    public LocalGenClient(LocalGenClientOptions? options = null)
        : this(new HttpClient(), options, ownsHttpClient: true)
    {
    }

    public LocalGenClient(string endpoint, string? apiKey = null)
        : this(new LocalGenClientOptions { Endpoint = endpoint, ApiKey = apiKey })
    {
    }

    public LocalGenClient(
        HttpClient httpClient,
        LocalGenClientOptions? options = null,
        bool ownsHttpClient = false)
    {
        _options = options ?? new LocalGenClientOptions();
        _http = httpClient;
        _ownsHttpClient = ownsHttpClient;

        // A caller-supplied HttpClient may already be configured and shared, so only fill in
        // what has not been set rather than overwriting their settings.
        _http.BaseAddress ??= new Uri(_options.Endpoint.TrimEnd('/') + "/");

        if (ownsHttpClient)
        {
            _http.Timeout = _options.Timeout;
        }

        if (!string.IsNullOrEmpty(_options.ApiKey) &&
            _http.DefaultRequestHeaders.Authorization is null)
        {
            _http.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _options.ApiKey);
        }

        Admin = new LocalGenAdminClient(_http);
    }

    /// <summary>LocalGen-specific management operations: models, engines, status, metrics.</summary>
    public LocalGenAdminClient Admin { get; }

    /// <summary>
    /// The settings this client was built with. Exposed so callers can read the configured
    /// endpoint or default model back rather than tracking them separately.
    /// </summary>
    public LocalGenClientOptions Options => _options;

    /// <summary>Sends a chat completion and returns the assistant's reply.</summary>
    public async Task<ChatCompletionResponse> ChatAsync(
        ChatCompletionRequest request,
        CancellationToken cancellationToken = default)
    {
        request.Model = ResolveModel(request.Model);
        request.Stream = false;

        using var response = await _http
            .PostAsJsonAsync("v1/chat/completions", request, OpenAiJson.Options, cancellationToken)
            .ConfigureAwait(false);

        await EnsureSuccessAsync(response, cancellationToken).ConfigureAwait(false);

        return await response.Content
            .ReadFromJsonAsync<ChatCompletionResponse>(OpenAiJson.Options, cancellationToken)
            .ConfigureAwait(false)
            ?? throw new LocalGenClientException("The server returned an empty response.");
    }

    /// <summary>Convenience overload for a single-prompt exchange.</summary>
    public async Task<string> ChatAsync(
        string prompt,
        string? systemPrompt = null,
        string? model = null,
        CancellationToken cancellationToken = default)
    {
        var messages = new List<OpenAiMessage>();

        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            messages.Add(new OpenAiMessage
            {
                Role = "system",
                Content = JsonSerializer.SerializeToElement(systemPrompt)
            });
        }

        messages.Add(new OpenAiMessage
        {
            Role = "user",
            Content = JsonSerializer.SerializeToElement(prompt)
        });

        var response = await ChatAsync(
            new ChatCompletionRequest { Model = ResolveModel(model), Messages = messages },
            cancellationToken).ConfigureAwait(false);

        return ReadContent(response.Choices.FirstOrDefault()?.Message?.Content);
    }

    /// <summary>Streams a chat completion token by token.</summary>
    public async IAsyncEnumerable<ChatCompletionResponse> StreamChatAsync(
        ChatCompletionRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        request.Model = ResolveModel(request.Model);
        request.Stream = true;

        using var message = new HttpRequestMessage(HttpMethod.Post, "v1/chat/completions")
        {
            Content = JsonContent.Create(request, options: OpenAiJson.Options)
        };

        using var response = await _http
            .SendAsync(message, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);

        await EnsureSuccessAsync(response, cancellationToken).ConfigureAwait(false);

        await using var stream = await response.Content
            .ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);

        await foreach (var chunk in ServerSentEvents
            .ReadAsync<ChatCompletionResponse>(stream, cancellationToken)
            .ConfigureAwait(false))
        {
            yield return chunk;
        }
    }

    /// <summary>Streams just the text of a chat completion, which is what most callers want.</summary>
    public async IAsyncEnumerable<string> StreamTextAsync(
        string prompt,
        string? systemPrompt = null,
        string? model = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var messages = new List<OpenAiMessage>();

        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            messages.Add(new OpenAiMessage
            {
                Role = "system",
                Content = JsonSerializer.SerializeToElement(systemPrompt)
            });
        }

        messages.Add(new OpenAiMessage
        {
            Role = "user",
            Content = JsonSerializer.SerializeToElement(prompt)
        });

        var request = new ChatCompletionRequest { Model = ResolveModel(model), Messages = messages };

        await foreach (var chunk in StreamChatAsync(request, cancellationToken).ConfigureAwait(false))
        {
            var delta = chunk.Choices.FirstOrDefault()?.Delta?.Content;
            if (!string.IsNullOrEmpty(delta))
            {
                yield return delta;
            }
        }
    }

    /// <summary>Produces embedding vectors for one or more inputs.</summary>
    public async Task<EmbeddingsResponse> EmbedAsync(
        IEnumerable<string> inputs,
        string? model = null,
        CancellationToken cancellationToken = default)
    {
        var request = new EmbeddingsRequest
        {
            Model = ResolveModel(model),
            Input = JsonSerializer.SerializeToElement(inputs.ToArray())
        };

        using var response = await _http
            .PostAsJsonAsync("v1/embeddings", request, OpenAiJson.Options, cancellationToken)
            .ConfigureAwait(false);

        await EnsureSuccessAsync(response, cancellationToken).ConfigureAwait(false);

        return await response.Content
            .ReadFromJsonAsync<EmbeddingsResponse>(OpenAiJson.Options, cancellationToken)
            .ConfigureAwait(false)
            ?? throw new LocalGenClientException("The server returned an empty response.");
    }

    /// <summary>Embeds a single string and returns its vector.</summary>
    public async Task<float[]> EmbedAsync(
        string input,
        string? model = null,
        CancellationToken cancellationToken = default)
    {
        var response = await EmbedAsync([input], model, cancellationToken).ConfigureAwait(false);
        return response.Data.FirstOrDefault()?.Embedding ?? [];
    }

    /// <summary>Lists the models the server can serve.</summary>
    public async Task<IReadOnlyList<ModelData>> ListModelsAsync(
        CancellationToken cancellationToken = default)
    {
        var response = await _http
            .GetFromJsonAsync<ModelListResponse>("v1/models", OpenAiJson.Options, cancellationToken)
            .ConfigureAwait(false);

        return response?.Data ?? [];
    }

    /// <summary>Checks that the server is reachable, without throwing.</summary>
    public async Task<bool> PingAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            using var response = await _http.GetAsync("api/health", cancellationToken).ConfigureAwait(false);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            return false;
        }
    }

    private string ResolveModel(string? model) =>
        !string.IsNullOrWhiteSpace(model) ? model
        : !string.IsNullOrWhiteSpace(_options.DefaultModel) ? _options.DefaultModel
        : throw new LocalGenClientException(
            "No model was specified and LocalGenClientOptions.DefaultModel is not set.");

    /// <summary>Reads the polymorphic message content field into plain text.</summary>
    internal static string ReadContent(JsonElement? content) => content switch
    {
        null => string.Empty,
        { ValueKind: JsonValueKind.String } element => element.GetString() ?? string.Empty,
        { ValueKind: JsonValueKind.Array } element => string.Concat(
            element.EnumerateArray()
                .Where(static part => part.TryGetProperty("text", out _))
                .Select(static part => part.GetProperty("text").GetString())),
        _ => string.Empty
    };

    /// <summary>Surfaces the server's OpenAI-shaped error body rather than a bare status code.</summary>
    internal static async Task EnsureSuccessAsync(
        HttpResponseMessage response,
        CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode)
        {
            return;
        }

        var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            var error = JsonSerializer.Deserialize<OpenAiErrorResponse>(body, OpenAiJson.Options);
            if (!string.IsNullOrEmpty(error?.Error.Message))
            {
                throw new LocalGenClientException(error.Error.Message, response.StatusCode, error.Error.Code);
            }
        }
        catch (JsonException)
        {
            // Not an OpenAI error body; fall through to the generic message.
        }

        throw new LocalGenClientException(
            $"The server returned {(int)response.StatusCode} {response.ReasonPhrase}. {body}",
            response.StatusCode);
    }

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _http.Dispose();
        }
    }
}

/// <summary>An error returned by a LocalGen server.</summary>
public sealed class LocalGenClientException(
    string message,
    System.Net.HttpStatusCode? statusCode = null,
    string? errorCode = null) : Exception(message)
{
    public System.Net.HttpStatusCode? StatusCode { get; } = statusCode;

    public string? ErrorCode { get; } = errorCode;
}

/// <summary>Reads a <c>text/event-stream</c> body into deserialized payloads.</summary>
internal static class ServerSentEvents
{
    private const string DataPrefix = "data:";
    private const string DoneSentinel = "[DONE]";

    public static async IAsyncEnumerable<T> ReadAsync<T>(
        Stream stream,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var reader = new StreamReader(stream, Encoding.UTF8);

        while (!reader.EndOfStream)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false);

            // Blank lines separate events; comments start with ':'.
            if (string.IsNullOrWhiteSpace(line) || !line.StartsWith(DataPrefix, StringComparison.Ordinal))
            {
                continue;
            }

            var payload = line[DataPrefix.Length..].Trim();

            if (payload is DoneSentinel)
            {
                yield break;
            }

            T? value;

            try
            {
                value = JsonSerializer.Deserialize<T>(payload, OpenAiJson.Options);
            }
            catch (JsonException)
            {
                // A malformed frame should not abort a long stream.
                continue;
            }

            if (value is not null)
            {
                yield return value;
            }
        }
    }
}
