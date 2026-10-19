using System.Net;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using ScienceAppGen.Models;

namespace ScienceAppGen.Services.Conversion;

/// <summary>One prompt in, one reply out. The seam the converter's tests replace.</summary>
public interface ILlmClient
{
    /// <summary>Provider and model, for logs and the conversion report.</summary>
    string Description { get; }

    /// <summary>Sends a system and a user message and returns the reply text.</summary>
    Task<string> CompleteAsync(string system, string user, CancellationToken cancellationToken);
}

/// <summary>
/// <see cref="ILlmClient"/> over the configured Semantic Kernel provider, without plugins.
/// </summary>
/// <remarks>
/// Conversion is one request per file with no tool use, so function calling stays off: a model
/// offered tools tends to spend turns calling them instead of writing the file. Throttling and
/// gateway errors are retried with backoff - a ten-file project makes ten to forty calls, and a
/// single 429 should not throw the whole conversion away.
/// </remarks>
public sealed class SemanticKernelLlmClient : ILlmClient, IDisposable
{
    private readonly AppSettings _settings;
    private readonly HttpClient _http;
    private readonly IChatCompletionService _chat;
    private readonly int _maxOutputTokens;

    /// <summary>Creates a client for the given settings.</summary>
    /// <param name="settings">Provider, model and key.</param>
    /// <param name="maxOutputTokens">Room for a whole file plus a reasoning model's thinking.</param>
    /// <param name="timeout">Per request.</param>
    public SemanticKernelLlmClient(AppSettings settings, int maxOutputTokens = 32_000, TimeSpan? timeout = null)
    {
        _settings = settings;
        _maxOutputTokens = maxOutputTokens;
        _http = new HttpClient
        {
            Timeout = timeout ?? TimeSpan.FromSeconds(Math.Max(settings.TimeoutSeconds, 600)),
        };
        _chat = LlmFactory.CreatePlainKernel(settings, _http).GetRequiredService<IChatCompletionService>();
    }

    /// <inheritdoc />
    public string Description => $"{_settings.Provider}/{_settings.Model}";

    /// <inheritdoc />
    public async Task<string> CompleteAsync(string system, string user, CancellationToken cancellationToken)
    {
        var history = new ChatHistory();
        history.AddSystemMessage(system);
        history.AddUserMessage(user);
        var execution = LlmFactory.CreateExecutionSettings(_settings, functionCalling: false, maxTokens: _maxOutputTokens);

        for (var attempt = 1; ; attempt++)
        {
            try
            {
                var reply = await _chat.GetChatMessageContentAsync(history, execution, cancellationToken: cancellationToken);
                return reply.Content ?? "";
            }
            catch (Exception ex) when (attempt < 4 && IsTransient(ex) && !cancellationToken.IsCancellationRequested)
            {
                await Task.Delay(TimeSpan.FromSeconds(5 * Math.Pow(2, attempt - 1)), cancellationToken);
            }
        }
    }

    private static bool IsTransient(Exception ex) => ex switch
    {
        HttpOperationException http => http.StatusCode is null or HttpStatusCode.TooManyRequests
            or HttpStatusCode.InternalServerError or HttpStatusCode.BadGateway
            or HttpStatusCode.ServiceUnavailable or HttpStatusCode.GatewayTimeout,
        HttpRequestException => true,
        TaskCanceledException => true,   // the HttpClient timeout, not the user (filtered by the caller)
        _ => ex.InnerException is not null && IsTransient(ex.InnerException),
    };

    /// <inheritdoc />
    public void Dispose() => _http.Dispose();
}
