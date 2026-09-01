using LocalGen.Core.Configuration;
using LocalGen.Core.Protocol;
using LocalGen.Server.Tenancy;
using Microsoft.Extensions.Options;

namespace LocalGen.Server;

/// <summary>
/// Authenticates the bearer token and holds each key to its quotas.
/// </summary>
/// <remarks>
/// LocalGen binds to loopback by default, so a key is only needed when the service is exposed on
/// a network — but once it is, an unauthenticated inference endpoint is an open door to the
/// machine's compute and its tool functions. Authentication and quota enforcement live in the same
/// middleware because the quota is a property of the key: there is no point at which a request is
/// authenticated but not yet attributed.
/// </remarks>
public sealed class ApiKeyMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ApiKeyRegistry _registry;
    private readonly TelemetryOptions _telemetry;

    public ApiKeyMiddleware(
        RequestDelegate next,
        ApiKeyRegistry registry,
        IOptions<LocalGenOptions> options)
    {
        _next = next;
        _registry = registry;
        _telemetry = options.Value.Telemetry;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        if (IsOpenPath(context.Request.Path.Value ?? string.Empty))
        {
            await _next(context).ConfigureAwait(false);
            return;
        }

        if (!TryReadKey(context, out var presented) || _registry.Resolve(presented) is not { } tenant)
        {
            await WriteErrorAsync(
                context,
                StatusCodes.Status401Unauthorized,
                "Incorrect API key provided.",
                "invalid_request_error",
                "invalid_api_key").ConfigureAwait(false);
            return;
        }

        var decision = _registry.TryAdmit(
            tenant,
            DateTimeOffset.UtcNow,
            chargesTokens: GeneratesTokens(context.Request.Path));

        if (!decision.IsAllowed)
        {
            // Seconds, rounded up: a Retry-After of 0 invites an immediate retry that would fail
            // again, and the header has no sub-second form.
            context.Response.Headers.RetryAfter =
                Math.Max(1, (int)Math.Ceiling(decision.RetryAfter.TotalSeconds)).ToString();

            await WriteErrorAsync(
                context,
                StatusCodes.Status429TooManyRequests,
                decision.Reason,
                "rate_limit_exceeded",
                "rate_limit_exceeded").ConfigureAwait(false);
            return;
        }

        TenantContext.Attach(context, tenant);

        try
        {
            await _next(context).ConfigureAwait(false);
        }
        finally
        {
            // Runs after the response body is fully written, so a streamed completion holds its
            // concurrency slot for as long as it is actually occupying a model.
            _registry.Complete(tenant);
        }
    }

    /// <summary>
    /// Paths served without a key: the banner and health, so probes and load balancers work, and
    /// optionally the metrics scrape, which a Prometheus server inside a cluster is rarely given
    /// credentials for.
    /// </summary>
    private bool IsOpenPath(string path)
    {
        if (path is "/" || path.StartsWith("/api/health", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        return !_telemetry.RequireApiKeyForMetrics
            && _telemetry.PrometheusEnabled
            && path.Equals(_telemetry.MetricsPath, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Whether a path can spend tokens, and so should be held to the daily token budget.
    /// </summary>
    /// <remarks>
    /// Only the three generating endpoints qualify. Listing models, reading status and reading
    /// usage cost nothing, and refusing them once a budget is spent would leave the caller unable
    /// to find out what happened.
    /// </remarks>
    private static bool GeneratesTokens(PathString path) =>
        path.StartsWithSegments("/v1/chat/completions") ||
        path.StartsWithSegments("/v1/completions") ||
        path.StartsWithSegments("/v1/embeddings");

    /// <summary>Accepts the OpenAI <c>Authorization: Bearer</c> header or an <c>X-API-Key</c> header.</summary>
    private static bool TryReadKey(HttpContext context, out string key)
    {
        var authorization = context.Request.Headers.Authorization.ToString();

        if (authorization.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
        {
            key = authorization["Bearer ".Length..].Trim();
            return key.Length > 0;
        }

        key = context.Request.Headers["X-API-Key"].ToString();
        return key.Length > 0;
    }

    private static async Task WriteErrorAsync(
        HttpContext context,
        int statusCode,
        string message,
        string type,
        string code)
    {
        context.Response.StatusCode = statusCode;
        await context.Response.WriteAsJsonAsync(
            OpenAiErrorResponse.Create(message, type, code),
            OpenAiJson.Options).ConfigureAwait(false);
    }
}
