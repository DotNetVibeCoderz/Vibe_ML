using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LocalGen.Core.Configuration;
using LocalGen.Core.Models;
using LocalGen.Desktop.Services;
using Microsoft.Extensions.Options;

namespace LocalGen.Desktop.ViewModels;

/// <summary>One endpoint offered in the tester, with a payload that runs as-is.</summary>
public sealed record ApiEndpoint
{
    public required string Method { get; init; }

    public required string Path { get; init; }

    /// <summary>Short label shown in the list.</summary>
    public required string Title { get; init; }

    /// <summary>What the endpoint does, and what to look for in the response.</summary>
    public required string Description { get; init; }

    /// <summary>Grouping header: <c>OpenAI</c>, <c>Management</c> or <c>Files</c>.</summary>
    public required string Group { get; init; }

    /// <summary>Request body template. <c>{model}</c> is replaced with the selected model.</summary>
    public string Body { get; init; } = string.Empty;

    /// <summary>Query or path parameters worth knowing about.</summary>
    public string Parameters { get; init; } = string.Empty;

    /// <summary>Whether the response is a server-sent event stream.</summary>
    public bool Streams { get; init; }

    /// <summary>Whether the request is a multipart upload rather than JSON.</summary>
    public bool Uploads { get; init; }

    public bool HasBody => !string.IsNullOrEmpty(Body);

    public string Signature => $"{Method} {Path}";
}

/// <summary>
/// A request bench for the HTTP API.
/// </summary>
/// <remarks>
/// LocalGen's central claim is that it is a drop-in replacement for the OpenAI API. That claim is
/// only worth anything if it can be checked, so this sends real requests over HTTP — not through
/// the SDK — and shows the raw response. Every example is a payload that runs as it stands
/// against the selected model.
/// </remarks>
public sealed partial class ApiTestViewModel : ViewModelBase
{
    private readonly LocalGenOptions _options;
    private readonly IModelStore _store;
    private readonly IFileDialogService _dialogs;
    private readonly HttpClient _http;

    private CancellationTokenSource? _request;

    [ObservableProperty]
    private ApiEndpoint? _selectedEndpoint;

    [ObservableProperty]
    private string _url = string.Empty;

    [ObservableProperty]
    private string _method = "GET";

    [ObservableProperty]
    private string _requestBody = string.Empty;

    [ObservableProperty]
    private string _apiKey = string.Empty;

    [ObservableProperty]
    private string _responseBody = string.Empty;

    [ObservableProperty]
    private string _responseStatus = string.Empty;

    [ObservableProperty]
    private string _responseMeta = string.Empty;

    [ObservableProperty]
    private bool _responseFailed;

    [ObservableProperty]
    private bool _isSending;

    [ObservableProperty]
    private ModelDescriptor? _selectedModel;

    [ObservableProperty]
    private string _uploadPath = string.Empty;

    public ApiTestViewModel(
        IOptions<LocalGenOptions> options,
        IModelStore store,
        IFileDialogService dialogs)
    {
        _options = options.Value;
        _store = store;
        _dialogs = dialogs;

        // No timeout: a cold model load can take minutes before the first byte, and cancelling is
        // the user's job here rather than the client's.
        _http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };

        Endpoints = new ObservableCollection<ApiEndpoint>(BuildCatalog());
        SelectedEndpoint = Endpoints[0];
    }

    public ObservableCollection<ApiEndpoint> Endpoints { get; }

    public ObservableCollection<ModelDescriptor> Models { get; } = [];

    public IReadOnlyList<string> Methods { get; } = ["GET", "POST", "PUT", "DELETE"];

    public string BaseUrl => _options.Server.BaseUrl;

    public override async Task InitializeAsync()
    {
        var models = await _store.ListAsync().ConfigureAwait(true);

        Models.Clear();
        foreach (var model in models)
        {
            Models.Add(model);
        }

        SelectedModel ??= Models.FirstOrDefault();

        // Re-apply the selection so the example payload picks up the resolved model name.
        ApplyEndpoint(SelectedEndpoint);
    }

    partial void OnSelectedEndpointChanged(ApiEndpoint? value) => ApplyEndpoint(value);

    partial void OnSelectedModelChanged(ModelDescriptor? value) => ApplyEndpoint(SelectedEndpoint);

    private void ApplyEndpoint(ApiEndpoint? endpoint)
    {
        if (endpoint is null)
        {
            return;
        }

        Method = endpoint.Method;
        Url = $"{BaseUrl}{endpoint.Path}".Replace("{model}", SelectedModel?.Id ?? "your-model");
        RequestBody = endpoint.Body.Replace("{model}", SelectedModel?.Id ?? "your-model");

        ResponseBody = string.Empty;
        ResponseStatus = string.Empty;
        ResponseMeta = string.Empty;
        ResponseFailed = false;
        ErrorMessage = string.Empty;
    }

    [RelayCommand]
    private async Task PickUploadAsync()
    {
        var paths = await _dialogs.PickDocumentsAsync().ConfigureAwait(true);
        UploadPath = paths.FirstOrDefault() ?? string.Empty;
    }

    [RelayCommand]
    private void Cancel() => _request?.Cancel();

    /// <summary>Copies the current request as a curl command, for pasting into a terminal or a bug report.</summary>
    [RelayCommand]
    private async Task CopyAsCurlAsync()
    {
        var builder = new StringBuilder($"curl -X {Method} \"{Url}\"");

        if (!string.IsNullOrWhiteSpace(ApiKey))
        {
            builder.Append(" \\\n  -H \"Authorization: Bearer ").Append(ApiKey).Append('"');
        }

        if (SelectedEndpoint?.Uploads == true)
        {
            builder.Append(" \\\n  -F \"file=@")
                   .Append(string.IsNullOrEmpty(UploadPath) ? "path/to/file" : UploadPath)
                   .Append('"');
        }
        else if (!string.IsNullOrWhiteSpace(RequestBody))
        {
            builder.Append(" \\\n  -H \"Content-Type: application/json\"")
                   .Append(" \\\n  -d '")
                   .Append(RequestBody.Replace("'", "'\\''"))
                   .Append('\'');
        }

        if (SelectedEndpoint?.Streams == true)
        {
            // Without -N curl buffers the stream and the point of testing it is lost.
            builder.Insert(4, " -N");
        }

        var clipboard = Avalonia.Application.Current?.ApplicationLifetime
            is Avalonia.Controls.ApplicationLifetimes.IClassicDesktopStyleApplicationLifetime { MainWindow: { } window }
            ? window.Clipboard
            : null;

        if (clipboard is not null)
        {
            await clipboard.SetTextAsync(builder.ToString());
            ResponseMeta = "copied as curl";
        }
    }

    [RelayCommand]
    private async Task SendAsync()
    {
        if (IsSending || SelectedEndpoint is null)
        {
            return;
        }

        IsSending = true;
        ResponseBody = string.Empty;
        ResponseStatus = string.Empty;
        ResponseMeta = string.Empty;
        ResponseFailed = false;
        ErrorMessage = string.Empty;

        _request = new CancellationTokenSource();
        var stopwatch = Stopwatch.StartNew();

        try
        {
            using var message = new HttpRequestMessage(new HttpMethod(Method), Url);

            if (!string.IsNullOrWhiteSpace(ApiKey))
            {
                message.Headers.Authorization = new AuthenticationHeaderValue("Bearer", ApiKey);
            }

            if (SelectedEndpoint.Uploads)
            {
                if (string.IsNullOrWhiteSpace(UploadPath) || !File.Exists(UploadPath))
                {
                    ErrorMessage = "Choose a file to upload first.";
                    return;
                }

                var content = new MultipartFormDataContent();
                var bytes = await File.ReadAllBytesAsync(UploadPath, _request.Token).ConfigureAwait(true);

                content.Add(new ByteArrayContent(bytes), "file", Path.GetFileName(UploadPath));
                message.Content = content;
            }
            else if (!string.IsNullOrWhiteSpace(RequestBody) && Method is not "GET" and not "DELETE")
            {
                message.Content = new StringContent(RequestBody, Encoding.UTF8, "application/json");
            }

            using var response = await _http
                .SendAsync(message, HttpCompletionOption.ResponseHeadersRead, _request.Token)
                .ConfigureAwait(true);

            ResponseStatus = $"{(int)response.StatusCode} {response.ReasonPhrase}";
            ResponseFailed = !response.IsSuccessStatusCode;

            var isStream = response.Content.Headers.ContentType?.MediaType == "text/event-stream";

            if (isStream)
            {
                await ReadStreamAsync(response, stopwatch, _request.Token).ConfigureAwait(true);
            }
            else
            {
                var body = await response.Content.ReadAsStringAsync(_request.Token).ConfigureAwait(true);

                ResponseBody = Prettify(body);
                ResponseMeta = $"{stopwatch.ElapsedMilliseconds:N0} ms · {body.Length:N0} bytes";
            }
        }
        catch (OperationCanceledException)
        {
            ResponseMeta = $"cancelled after {stopwatch.ElapsedMilliseconds:N0} ms";
        }
        catch (Exception ex)
        {
            ResponseFailed = true;
            ErrorMessage = ex.Message;
            ResponseStatus = "request failed";
        }
        finally
        {
            IsSending = false;
        }
    }

    /// <summary>
    /// Reads a server-sent event body, appending frames as they arrive so the response pane shows
    /// the stream rather than only its end.
    /// </summary>
    private async Task ReadStreamAsync(
        HttpResponseMessage response,
        Stopwatch stopwatch,
        CancellationToken cancellationToken)
    {
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(true);
        using var reader = new StreamReader(stream, Encoding.UTF8);

        var builder = new StringBuilder();
        var frames = 0;
        var firstFrame = TimeSpan.Zero;

        while (!reader.EndOfStream)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(true);

            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            if (firstFrame == TimeSpan.Zero)
            {
                firstFrame = stopwatch.Elapsed;
            }

            frames++;
            builder.AppendLine(line);

            // Updated as it goes; a stream that only appeared at the end would prove nothing
            // about whether the server actually streams.
            ResponseBody = builder.ToString();
            ResponseMeta = $"{frames} frame(s) · {stopwatch.ElapsedMilliseconds:N0} ms";
        }

        ResponseMeta =
            $"{frames} frame(s) · first at {firstFrame.TotalMilliseconds:N0} ms · " +
            $"{stopwatch.ElapsedMilliseconds:N0} ms total";
    }

    /// <summary>Formats a JSON body for reading; anything else is shown as it arrived.</summary>
    private static string Prettify(string body)
    {
        if (string.IsNullOrWhiteSpace(body))
        {
            return "(empty response)";
        }

        try
        {
            using var document = JsonDocument.Parse(body);
            return JsonSerializer.Serialize(document.RootElement, new JsonSerializerOptions
            {
                WriteIndented = true
            });
        }
        catch (JsonException)
        {
            return body;
        }
    }

    /// <summary>
    /// The endpoint catalogue. Each example is a payload that runs as it stands, because a
    /// template the user has to repair first is not much of a test.
    /// </summary>
    private static IReadOnlyList<ApiEndpoint> BuildCatalog() =>
    [
        new ApiEndpoint
        {
            Group = "OpenAI",
            Method = "POST",
            Path = "/v1/chat/completions",
            Title = "Chat completion",
            Description = "The main endpoint. Returns one choice with the assistant's message and token usage.",
            Parameters = "model, messages, temperature, top_p, top_k, max_tokens, stop, seed, n, user",
            Body = """
                {
                  "model": "{model}",
                  "messages": [
                    { "role": "system", "content": "You are concise." },
                    { "role": "user", "content": "Explain quantization in one sentence." }
                  ],
                  "temperature": 0.7,
                  "max_tokens": 200
                }
                """
        },
        new ApiEndpoint
        {
            Group = "OpenAI",
            Method = "POST",
            Path = "/v1/chat/completions",
            Title = "Chat completion — streaming",
            Description =
                "Server-sent events. Each frame is a chat.completion.chunk; the role arrives on " +
                "the first delta and the stream ends with data: [DONE].",
            Parameters = "stream, stream_options.include_usage",
            Streams = true,
            Body = """
                {
                  "model": "{model}",
                  "messages": [
                    { "role": "user", "content": "Count from one to five." }
                  ],
                  "max_tokens": 100,
                  "stream": true,
                  "stream_options": { "include_usage": true }
                }
                """
        },
        new ApiEndpoint
        {
            Group = "OpenAI",
            Method = "POST",
            Path = "/v1/chat/completions",
            Title = "Function calling",
            Description =
                "Advertises a tool. A model that decides to use it answers with finish_reason " +
                "\"tool_calls\" and an empty content, with the call in tool_calls.",
            Parameters = "tools, tool_choice",
            Body = """
                {
                  "model": "{model}",
                  "messages": [
                    { "role": "user", "content": "What is the weather in Jakarta? Use the tool." }
                  ],
                  "temperature": 0.2,
                  "max_tokens": 200,
                  "tools": [
                    {
                      "type": "function",
                      "function": {
                        "name": "get_weather",
                        "description": "Returns the current weather for a city.",
                        "parameters": {
                          "type": "object",
                          "properties": { "city": { "type": "string", "description": "City name" } },
                          "required": ["city"]
                        }
                      }
                    }
                  ]
                }
                """
        },
        new ApiEndpoint
        {
            Group = "OpenAI",
            Method = "POST",
            Path = "/v1/chat/completions",
            Title = "Structured output (JSON mode)",
            Description =
                "On the LlamaSharp backend this constrains decoding with a JSON grammar, so the " +
                "reply cannot be prose wrapped around an object.",
            Parameters = "response_format.type = json_object | json_schema",
            Body = """
                {
                  "model": "{model}",
                  "messages": [
                    { "role": "system", "content": "Reply with JSON matching {\"language\": string, \"year\": number, \"paradigms\": string[]}" },
                    { "role": "user", "content": "Describe the C# programming language." }
                  ],
                  "temperature": 0.1,
                  "max_tokens": 200,
                  "response_format": { "type": "json_object" }
                }
                """
        },
        new ApiEndpoint
        {
            Group = "OpenAI",
            Method = "POST",
            Path = "/v1/chat/completions",
            Title = "Image attachment",
            Description =
                "Multimodal content. Upload a file first, then paste its URL here. LocalGen loads " +
                "the bytes for URLs it serves itself; remote URLs are not fetched, so offline " +
                "mode holds.",
            Parameters = "content as an array of text and image_url parts",
            Body = """
                {
                  "model": "{model}",
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        { "type": "text", "text": "What does this image show?" },
                        { "type": "image_url",
                          "image_url": { "url": "http://127.0.0.1:11434/api/files/REPLACE_WITH_UPLOADED_ID.png" } }
                      ]
                    }
                  ],
                  "max_tokens": 200
                }
                """
        },
        new ApiEndpoint
        {
            Group = "OpenAI",
            Method = "POST",
            Path = "/v1/completions",
            Title = "Legacy completion",
            Description = "The pre-chat endpoint. The prompt is wrapped in a single user turn.",
            Parameters = "model, prompt, max_tokens, temperature, top_p, stop, stream, seed",
            Body = """
                {
                  "model": "{model}",
                  "prompt": "Write one sentence about local inference:",
                  "max_tokens": 80,
                  "temperature": 0.7
                }
                """
        },
        new ApiEndpoint
        {
            Group = "OpenAI",
            Method = "POST",
            Path = "/v1/embeddings",
            Title = "Embeddings",
            Description =
                "Vectors for one or more inputs. Needs an embedding model — the LlamaSharp " +
                "backend only; ONNX Runtime GenAI has no embedding path.",
            Parameters = "model, input (string or array), encoding_format, dimensions",
            Body = """
                {
                  "model": "nomic-embed-text",
                  "input": ["first sentence", "second sentence"]
                }
                """
        },
        new ApiEndpoint
        {
            Group = "OpenAI",
            Method = "GET",
            Path = "/v1/models",
            Title = "List models",
            Description =
                "Standard OpenAI shape, plus a localgen block that other clients ignore: " +
                "quantization, size, context length and which engines can serve it.",
            Parameters = "none"
        },
        new ApiEndpoint
        {
            Group = "OpenAI",
            Method = "GET",
            Path = "/v1/models/{model}",
            Title = "Retrieve one model",
            Description = "Details for a single model. Returns 404 with code model_not_found when it is not installed.",
            Parameters = "id in the path"
        },
        new ApiEndpoint
        {
            Group = "Management",
            Method = "GET",
            Path = "/api/health",
            Title = "Health",
            Description = "Liveness probe. Never requires an API key, so load balancers work without one.",
            Parameters = "none"
        },
        new ApiEndpoint
        {
            Group = "Management",
            Method = "GET",
            Path = "/api/status",
            Title = "Service status",
            Description = "State, uptime, data directory, and which models are resident with their engine and device.",
            Parameters = "none"
        },
        new ApiEndpoint
        {
            Group = "Management",
            Method = "GET",
            Path = "/api/engines",
            Title = "Engines",
            Description =
                "Installed backends, whether each probed successfully, the devices it can use, " +
                "and a recommendation derived from this machine.",
            Parameters = "none"
        },
        new ApiEndpoint
        {
            Group = "Management",
            Method = "GET",
            Path = "/api/metrics",
            Title = "Metrics",
            Description = "Inference and resource telemetry: throughput, latency, token counts, CPU and GPU.",
            Parameters = "none"
        },
        new ApiEndpoint
        {
            Group = "Management",
            Method = "GET",
            Path = "/api/models",
            Title = "Installed models",
            Description = "Full LocalGen metadata for each installed model, richer than /v1/models.",
            Parameters = "none"
        },
        new ApiEndpoint
        {
            Group = "Management",
            Method = "POST",
            Path = "/api/models/pull",
            Title = "Pull a model",
            Description =
                "Downloads a model, streaming progress as server-sent events because a pull runs " +
                "for minutes. Naming a repository without a file lets LocalGen choose the quantization.",
            Parameters = "reference",
            Streams = true,
            Body = """
                {
                  "reference": "huggingface:bartowski/SmolLM2-135M-Instruct-GGUF"
                }
                """
        },
        new ApiEndpoint
        {
            Group = "Management",
            Method = "GET",
            Path = "/api/logs?limit=50&level=Information",
            Title = "Logs",
            Description = "Recent server log entries, newest last.",
            Parameters = "limit, level (Debug | Information | Warning | Error)"
        },
        new ApiEndpoint
        {
            Group = "Management",
            Method = "GET",
            Path = "/api/catalog/search?q=qwen&limit=10",
            Title = "Search the catalogue",
            Description = "Searches the remote model catalogues. Returns nothing in offline mode.",
            Parameters = "q, limit, provider"
        },
        new ApiEndpoint
        {
            Group = "Files",
            Method = "POST",
            Path = "/api/files",
            Title = "Upload an attachment",
            Description =
                "multipart/form-data with one file part. Files are content-addressed by hash, so " +
                "uploading the same bytes twice returns the same id. Capped at 64 MB.",
            Parameters = "file (multipart part)",
            Uploads = true
        },
        new ApiEndpoint
        {
            Group = "Files",
            Method = "GET",
            Path = "/api/files/REPLACE_WITH_UPLOADED_ID",
            Title = "Download an attachment",
            Description = "Serves the file with its media type, and supports range requests for seeking video and audio.",
            Parameters = "id in the path"
        }
    ];
}
