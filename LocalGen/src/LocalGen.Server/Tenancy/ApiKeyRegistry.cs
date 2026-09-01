using System.Security.Cryptography;
using System.Text;
using LocalGen.Core;
using LocalGen.Core.Configuration;
using Microsoft.Extensions.Options;

namespace LocalGen.Server.Tenancy;

/// <summary>
/// Resolves presented API keys to tenants and holds their quota state.
/// </summary>
/// <remarks>
/// Registered as a singleton: the counters are the server's memory of what each key has spent, so
/// they have to outlive the request that spent it. Keys come from configuration rather than a
/// database because LocalGen is a single-instance product — an operator sharing one machine with
/// a handful of colleagues, not a SaaS control plane.
/// </remarks>
public sealed class ApiKeyRegistry
{
    private readonly List<(byte[] Key, ApiTenant Tenant)> _tenants = [];

    public ApiKeyRegistry(IOptions<LocalGenOptions> options)
    {
        var server = options.Value.Server;

        foreach (var descriptor in server.ApiKeys)
        {
            if (string.IsNullOrWhiteSpace(descriptor.Key) || !descriptor.Enabled)
            {
                continue;
            }

            _tenants.Add((Encoding.UTF8.GetBytes(descriptor.Key), new ApiTenant(descriptor)));
        }

        // The single-tenant key is modelled as one more tenant with no limits, so everything
        // downstream deals in tenants and never has to special-case the simple deployment.
        if (!string.IsNullOrEmpty(server.ApiKey))
        {
            _tenants.Add((
                Encoding.UTF8.GetBytes(server.ApiKey),
                new ApiTenant(new ApiKeyDescriptor { Key = server.ApiKey, Name = "default" })));
        }
    }

    /// <summary>Whether any key is configured. When false the API is open and unmetered.</summary>
    public bool IsEnabled => _tenants.Count > 0;

    /// <summary>Every configured tenant, for the Admin Control's usage view.</summary>
    public IReadOnlyList<ApiTenant> Tenants => [.. _tenants.Select(static t => t.Tenant)];

    /// <summary>
    /// Matches a presented secret against every configured key.
    /// </summary>
    /// <remarks>
    /// Every candidate is compared even after one matches, and each comparison is fixed-time, so
    /// neither the value of a key nor how many are configured can be recovered by timing the call.
    /// </remarks>
    public ApiTenant? Resolve(string presented)
    {
        var candidate = Encoding.UTF8.GetBytes(presented);
        ApiTenant? matched = null;

        foreach (var (key, tenant) in _tenants)
        {
            if (CryptographicOperations.FixedTimeEquals(candidate, key))
            {
                matched ??= tenant;
            }
        }

        return matched;
    }

    /// <summary>
    /// Admits a request against the key's rate and concurrency limits, or explains the refusal.
    /// </summary>
    /// <remarks>
    /// The request is counted here rather than on completion so that a burst arriving together is
    /// throttled as it lands. Callers must pair a successful admission with <see cref="Complete"/>.
    /// </remarks>
    /// <param name="chargesTokens">
    /// Whether this path can actually generate tokens. The daily token budget is a budget on
    /// compute, so it is not applied to requests that consume none: a caller who has spent their
    /// allowance must still be able to list models and — above all — read back the usage figures
    /// that explain why they are being refused.
    /// </param>
    public QuotaDecision TryAdmit(ApiTenant tenant, DateTimeOffset now, bool chargesTokens = true)
    {
        var limits = tenant.Descriptor;

        if (chargesTokens && limits.TokensPerDay > 0 && tenant.Tokens.Sum(now) >= limits.TokensPerDay)
        {
            return QuotaDecision.Denied(
                $"Daily token quota of {limits.TokensPerDay:N0} for '{tenant.Name}' is exhausted.",
                tenant.Tokens.TimeUntilRelease(now));
        }

        if (limits.RequestsPerMinute > 0 && tenant.Requests.Sum(now) >= limits.RequestsPerMinute)
        {
            return QuotaDecision.Denied(
                $"Rate limit of {limits.RequestsPerMinute:N0} requests/minute for '{tenant.Name}' exceeded.",
                tenant.Requests.TimeUntilRelease(now));
        }

        if (limits.MaxConcurrentRequests > 0)
        {
            // Checked by incrementing first: testing and then incrementing would let two requests
            // arriving together both observe room for one and both be admitted.
            if (tenant.EnterRequest() > limits.MaxConcurrentRequests)
            {
                tenant.ExitRequest();

                return QuotaDecision.Denied(
                    $"'{tenant.Name}' already has {limits.MaxConcurrentRequests} requests in flight.",
                    TimeSpan.FromSeconds(1));
            }
        }
        else
        {
            tenant.EnterRequest();
        }

        tenant.Requests.Add(1, now);
        return QuotaDecision.Allow;
    }

    /// <summary>Releases the concurrency slot taken by <see cref="TryAdmit"/>.</summary>
    public void Complete(ApiTenant tenant) => tenant.ExitRequest();

    /// <summary>Charges tokens to a key once a generation has reported its usage.</summary>
    public void RecordTokens(ApiTenant tenant, long tokens) =>
        tenant.Tokens.Add(tokens, DateTimeOffset.UtcNow);

    /// <summary>Throws when the key may not use the model it asked for.</summary>
    public static void EnsureModelAllowed(ApiTenant tenant, string modelId)
    {
        if (!tenant.AllowsModel(modelId))
        {
            throw new ModelForbiddenException(modelId, tenant.Name);
        }
    }
}

/// <summary>Outcome of a quota check.</summary>
public readonly record struct QuotaDecision(bool IsAllowed, string Reason, TimeSpan RetryAfter)
{
    public static QuotaDecision Allow { get; } = new(true, string.Empty, TimeSpan.Zero);

    public static QuotaDecision Denied(string reason, TimeSpan retryAfter) =>
        new(false, reason, retryAfter);
}
