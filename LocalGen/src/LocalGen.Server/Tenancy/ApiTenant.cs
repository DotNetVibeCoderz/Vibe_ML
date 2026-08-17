using LocalGen.Core.Configuration;

namespace LocalGen.Server.Tenancy;

/// <summary>
/// One authenticated caller and the live state of its quotas.
/// </summary>
/// <remarks>
/// The identity a request runs under is the API key, because that is the only identity the OpenAI
/// wire protocol carries. One of these exists per configured key for the lifetime of the server,
/// so the counters inside it survive across requests — which is the point of a quota.
/// </remarks>
public sealed class ApiTenant
{
    private int _inFlight;

    internal ApiTenant(ApiKeyDescriptor descriptor)
    {
        Descriptor = descriptor;
        Name = string.IsNullOrWhiteSpace(descriptor.Name) ? "unnamed" : descriptor.Name;

        // Ten buckets over a minute bounds the rollover error at six seconds; hourly buckets are
        // ample for a daily token budget, where being an hour early is not a meaningful unfairness.
        Requests = new SlidingWindowCounter(TimeSpan.FromMinutes(1), 10);
        Tokens = new SlidingWindowCounter(TimeSpan.FromDays(1), 24);
    }

    /// <summary>Label used in logs, metrics and errors. Never the secret itself.</summary>
    public string Name { get; }

    public ApiKeyDescriptor Descriptor { get; }

    internal SlidingWindowCounter Requests { get; }

    internal SlidingWindowCounter Tokens { get; }

    /// <summary>Requests this key currently has open, including streams still being written.</summary>
    public int InFlight => Volatile.Read(ref _inFlight);

    /// <summary>Tokens charged to this key in the last rolling day.</summary>
    public long TokensToday => Tokens.Sum(DateTimeOffset.UtcNow);

    /// <summary>Requests served for this key in the last rolling minute.</summary>
    public long RequestsThisMinute => Requests.Sum(DateTimeOffset.UtcNow);

    internal int EnterRequest() => Interlocked.Increment(ref _inFlight);

    internal void ExitRequest() => Interlocked.Decrement(ref _inFlight);

    /// <summary>Whether this key may address the given model.</summary>
    public bool AllowsModel(string modelId) =>
        Descriptor.AllowedModels.Length == 0 ||
        Descriptor.AllowedModels.Contains(modelId, StringComparer.OrdinalIgnoreCase);

    /// <summary>The unmetered identity used when no keys are configured at all.</summary>
    internal static ApiTenant Anonymous { get; } = new(new ApiKeyDescriptor { Name = "anonymous" });
}
