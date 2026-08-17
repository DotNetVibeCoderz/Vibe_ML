using System.Diagnostics;
using System.Runtime.CompilerServices;
using LocalGen.Core.Diagnostics;
using LocalGen.Core.Engines;
using LocalGen.Core.Inference;
using LocalGen.Runtime.Sessions;
using LocalGen.Server.Tenancy;
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
    private readonly ApiKeyRegistry _keys;
    private readonly TenantContext _tenants;
    private readonly ResponseCache _cache;
    private readonly ILogger<InferenceService> _logger;

    public InferenceService(
        ModelSessionManager sessions,
        InferenceMetrics metrics,
        ApiKeyRegistry keys,
        TenantContext tenants,
        ResponseCache cache,
        ILogger<InferenceService> logger)
    {
        _sessions = sessions;
        _metrics = metrics;
        _keys = keys;
        _tenants = tenants;
        _cache = cache;
        _logger = logger;
    }

    /// <summary>
    /// Applies the calling key's model allow-list. Enforced here rather than in the middleware
    /// because this is where the model is known — the middleware would have to buffer and parse
    /// the request body to learn it, and would still miss every non-HTTP caller.
    /// </summary>
    private ApiTenant? Authorize(string model)
    {
        var tenant = _tenants.Current;

        if (tenant is not null)
        {
            ApiKeyRegistry.EnsureModelAllowed(tenant, model);
        }

        return tenant;
    }

    /// <summary>Streams a chat completion, recording metrics as the generation proceeds.</summary>
    public async IAsyncEnumerable<ChatStreamChunk> StreamAsync(
        ChatRequest request,
        EngineKind? engine = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var tenant = Authorize(request.Model);

        // Checked before the lease is taken, so a hit costs nothing at all — it neither waits
        // behind the per-model queue nor keeps a model resident that would otherwise be evicted.
        // The cache is shared across keys deliberately: the entry is a pure function of the
        // prompt the caller supplied, so serving one tenant from another's entry reveals nothing
        // the second tenant did not already provide.
        var cacheKey = _cache.IsCacheable(request) ? ResponseCache.ComputeKey(request) : null;

        if (cacheKey is not null && _cache.TryGet(cacheKey) is { } cached)
        {
            _logger.LogDebug("Serving {Model} from the response cache.", request.Model);

            foreach (var chunk in Replay(cached))
            {
                yield return chunk;
            }

            yield break;
        }

        using var lease = await _sessions
            .AcquireAsync(request.Model, engine, cancellationToken)
            .ConfigureAwait(false);

        var session = lease.Session;
        var stopwatch = Stopwatch.StartNew();
        var timeToFirstToken = TimeSpan.Zero;
        var usage = TokenUsage.Empty;
        var failed = false;
        var generated = new System.Text.StringBuilder();
        var generatedCalls = new List<ToolCall>();
        var finishReason = FinishReason.Stop;

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
                Record(session, tenant, request, stopwatch, timeToFirstToken, usage, failed: false);
                throw;
            }
            catch (Exception ex)
            {
                failed = true;
                _logger.LogError(ex, "Generation failed for {Model}", request.Model);
                Record(session, tenant, request, stopwatch, timeToFirstToken, usage, failed);
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

            if (cacheKey is not null)
            {
                generated.Append(chunk.Delta);
                generatedCalls.AddRange(chunk.ToolCalls);

                if (chunk.FinishReason != FinishReason.None)
                {
                    finishReason = chunk.FinishReason;
                }
            }

            yield return chunk;
        }

        Record(session, tenant, request, stopwatch, timeToFirstToken, usage, failed);

        // Stored only on a clean run: the loop above rethrows on fault and on cancellation, so
        // reaching here means the generation finished and is safe to replay.
        if (cacheKey is not null)
        {
            _cache.Set(cacheKey, new CachedCompletion
            {
                Text = generated.ToString(),
                ToolCalls = generatedCalls,
                FinishReason = finishReason,
                Usage = usage
            });
        }
    }

    /// <summary>
    /// Turns a cached generation back into the chunk sequence a caller expects.
    /// </summary>
    /// <remarks>
    /// The text arrives as one delta rather than being re-split into the tokens that originally
    /// produced it: the token boundaries were never part of the contract, and inventing a delay to
    /// imitate them would spend the latency the cache exists to save.
    /// </remarks>
    private static IEnumerable<ChatStreamChunk> Replay(CachedCompletion cached)
    {
        if (cached.Text.Length > 0)
        {
            yield return new ChatStreamChunk { Delta = cached.Text };
        }

        if (cached.ToolCalls.Count > 0)
        {
            yield return new ChatStreamChunk { ToolCalls = cached.ToolCalls };
        }

        yield return new ChatStreamChunk
        {
            FinishReason = cached.FinishReason,
            Usage = cached.Usage
        };
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
        var tenant = Authorize(request.Model);

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
            TimeToFirstToken = stopwatch.Elapsed,
            Tenant = tenant?.Name ?? string.Empty
        });

        if (tenant is not null)
        {
            _keys.RecordTokens(tenant, response.Usage.TotalTokens);
        }

        return response;
    }

    private void Record(
        IModelSession session,
        ApiTenant? tenant,
        ChatRequest request,
        Stopwatch stopwatch,
        TimeSpan timeToFirstToken,
        TokenUsage usage,
        bool failed)
    {
        _metrics.Record(new InferenceSample
        {
            Model = request.Model,
            Engine = session.Engine.ToString(),
            PromptTokens = usage.PromptTokens,
            CompletionTokens = usage.CompletionTokens,
            TimeToFirstToken = timeToFirstToken,
            TotalDuration = stopwatch.Elapsed,
            Failed = failed,
            Tenant = tenant?.Name ?? string.Empty
        });

        if (tenant is not null)
        {
            // Charged even when the generation failed part-way: the tokens were still decoded,
            // and not charging them would make a loop of failing requests free.
            _keys.RecordTokens(tenant, usage.TotalTokens);
        }
    }
}
