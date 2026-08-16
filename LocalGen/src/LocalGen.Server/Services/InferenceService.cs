using System.Diagnostics;
using System.Runtime.CompilerServices;
using LocalGen.Core.Diagnostics;
using LocalGen.Core.Engines;
using LocalGen.Core.Inference;
using LocalGen.Runtime.Sessions;
using Microsoft.Extensions.Logging;

namespace LocalGen.Server.Services;

/// <summary>
/// Runs inference requests against loaded models and records telemetry.
/// </summary>
/// <remarks>
/// This is the single choke point where a request meets a model: it holds the session lease for
/// the whole generation (including the streaming enumeration, which outlives the method call) and
/// records one <see cref="InferenceSample"/> per request whether it succeeds, fails or is cancelled.
/// Endpoints stay free of lifetime and metrics concerns as a result.
/// </remarks>
public sealed class InferenceService
{
    private readonly ModelSessionManager _sessions;
    private readonly InferenceMetrics _metrics;
    private readonly ILogger<InferenceService> _logger;

    public InferenceService(
        ModelSessionManager sessions,
        InferenceMetrics metrics,
        ILogger<InferenceService> logger)
    {
        _sessions = sessions;
        _metrics = metrics;
        _logger = logger;
    }

    /// <summary>Streams a chat completion, recording metrics as the generation proceeds.</summary>
    public async IAsyncEnumerable<ChatStreamChunk> StreamAsync(
        ChatRequest request,
        EngineKind? engine = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var lease = await _sessions
            .AcquireAsync(request.Model, engine, cancellationToken)
            .ConfigureAwait(false);

        var session = lease.Session;
        var stopwatch = Stopwatch.StartNew();
        var timeToFirstToken = TimeSpan.Zero;
        var usage = TokenUsage.Empty;
        var failed = false;

        // The enumerator is stepped manually so that a fault or cancellation can be recorded
        // before rethrowing — `yield return` is not allowed inside a try/catch.
        await using var enumerator = session
            .StreamAsync(request, cancellationToken)
            .GetAsyncEnumerator(cancellationToken);

        while (true)
        {
            ChatStreamChunk chunk;

            try
            {
                if (!await enumerator.MoveNextAsync().ConfigureAwait(false))
                {
                    break;
                }

                chunk = enumerator.Current;
            }
            catch (OperationCanceledException)
            {
                Record(session, request, stopwatch, timeToFirstToken, usage, failed: false);
                throw;
            }
            catch (Exception ex)
            {
                failed = true;
                _logger.LogError(ex, "Generation failed for {Model}", request.Model);
                Record(session, request, stopwatch, timeToFirstToken, usage, failed);
                throw;
            }

            if (timeToFirstToken == TimeSpan.Zero && chunk.Delta.Length > 0)
            {
                timeToFirstToken = stopwatch.Elapsed;
            }

            if (chunk.Usage is not null)
            {
                usage = chunk.Usage;
            }

            yield return chunk;
        }

        Record(session, request, stopwatch, timeToFirstToken, usage, failed);
    }

    /// <summary>Runs a chat completion to completion and returns the whole response.</summary>
    public async Task<ChatResponse> CompleteAsync(
        ChatRequest request,
        EngineKind? engine = null,
        CancellationToken cancellationToken = default)
    {
        var text = new System.Text.StringBuilder();
        var toolCalls = new List<ToolCall>();
        var finishReason = FinishReason.Stop;
        var usage = TokenUsage.Empty;
        var stopwatch = Stopwatch.StartNew();

        await foreach (var chunk in StreamAsync(request, engine, cancellationToken).ConfigureAwait(false))
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
            Duration = stopwatch.Elapsed,
            Message = new ChatMessage
            {
                Role = ChatRole.Assistant,
                Content = [new ContentPart.Text(text.ToString())],
                ToolCalls = toolCalls
            }
        };
    }

    public async Task<EmbeddingResponse> EmbedAsync(
        EmbeddingRequest request,
        CancellationToken cancellationToken = default)
    {
        using var lease = await _sessions
            .AcquireAsync(request.Model, null, cancellationToken)
            .ConfigureAwait(false);

        var stopwatch = Stopwatch.StartNew();
        var response = await lease.Session.EmbedAsync(request, cancellationToken).ConfigureAwait(false);

        _metrics.Record(new InferenceSample
        {
            Model = request.Model,
            Engine = lease.Session.Engine.ToString(),
            PromptTokens = response.Usage.PromptTokens,
            CompletionTokens = 0,
            TotalDuration = stopwatch.Elapsed,
            TimeToFirstToken = stopwatch.Elapsed
        });

        return response;
    }

    private void Record(
        IModelSession session,
        ChatRequest request,
        Stopwatch stopwatch,
        TimeSpan timeToFirstToken,
        TokenUsage usage,
        bool failed) =>
        _metrics.Record(new InferenceSample
        {
            Model = request.Model,
            Engine = session.Engine.ToString(),
            PromptTokens = usage.PromptTokens,
            CompletionTokens = usage.CompletionTokens,
            TimeToFirstToken = timeToFirstToken,
            TotalDuration = stopwatch.Elapsed,
            Failed = failed
        });
}
