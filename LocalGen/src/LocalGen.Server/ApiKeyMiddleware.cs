using System.Security.Cryptography;
using System.Text;
using LocalGen.Core.Configuration;
using LocalGen.Core.Protocol;
using Microsoft.Extensions.Options;

namespace LocalGen.Server;

/// <summary>
/// Enforces the optional bearer token. LocalGen binds to loopback by default, so a key is only
/// needed when the service is exposed on a network — but once it is, an unauthenticated inference
/// endpoint is an open door to the machine's compute and its tool functions.
/// </summary>
public sealed class ApiKeyMiddleware
{
    private readonly RequestDelegate _next;
    private readonly byte[] _expectedKey;

    public ApiKeyMiddleware(RequestDelegate next, IOptions<LocalGenOptions> options)
    {
        _next = next;
        _expectedKey = Encoding.UTF8.GetBytes(options.Value.Server.ApiKey ?? string.Empty);
    }

    public async Task InvokeAsync(HttpContext context)
    {
        // Health and the service banner stay open so probes and load balancers work without a key.
        var path = context.Request.Path.Value ?? string.Empty;
        if (path is "/" || path.StartsWith("/api/health", StringComparison.OrdinalIgnoreCase))
        {
            await _next(context).ConfigureAwait(false);
            return;
        }

        if (!TryReadKey(context, out var presented) || !IsValid(presented))
        {
            context.Response.StatusCode = StatusCodes.Status401Unauthorized;
            await context.Response.WriteAsJsonAsync(
                OpenAiErrorResponse.Create(
                    "Incorrect API key provided.",
                    "invalid_request_error",
                    "invalid_api_key"),
                OpenAiJson.Options).ConfigureAwait(false);
            return;
        }

        await _next(context).ConfigureAwait(false);
    }

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

    /// <summary>Compared in fixed time so the key cannot be recovered by timing the comparison.</summary>
    private bool IsValid(string presented) =>
        CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(presented), _expectedKey);
}
