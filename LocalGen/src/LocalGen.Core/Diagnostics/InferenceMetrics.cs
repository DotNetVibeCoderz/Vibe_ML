using System.Collections.Concurrent;
using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace LocalGen.Core.Diagnostics;

/// <summary>A completed generation, recorded for the monitoring dashboard.</summary>
public sealed record InferenceSample
{
    public required string Model { get; init; }

    public required string Engine { get; init; }

    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    public int PromptTokens { get; init; }

    public int CompletionTokens { get; init; }

    /// <summary>Time until the first token reached the caller — what users perceive as latency.</summary>
    public TimeSpan TimeToFirstToken { get; init; }

    public TimeSpan TotalDuration { get; init; }

    public bool Failed { get; init; }

    /// <summary>
    /// Name of the API key the request was billed to. Empty when the server has no keys
    /// configured, or when the caller is in-process rather than over HTTP.
    /// </summary>
    public string Tenant { get; init; } = string.Empty;

    /// <summary>Generation throughput, excluding prompt processing.</summary>
    public double TokensPerSecond
    {
        get
        {
            var generationTime = TotalDuration - TimeToFirstToken;
            return generationTime.TotalSeconds > 0.001
                ? CompletionTokens / generationTime.TotalSeconds
                : 0;
        }
    }
}

/// <summary>A point-in-time reading of host resources.</summary>
public sealed record ResourceSample
{
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    public double CpuPercent { get; init; }

    public long MemoryUsedBytes { get; init; }

    public long MemoryTotalBytes { get; init; }

    public IReadOnlyList<GpuSample> Gpus { get; init; } = [];
}

public sealed record GpuSample
{
    public required string Name { get; init; }

    public int Index { get; init; }

    public double UtilizationPercent { get; init; }

    public long MemoryUsedBytes { get; init; }

    public long MemoryTotalBytes { get; init; }

    public double? TemperatureCelsius { get; init; }

    public double? PowerWatts { get; init; }
}

/// <summary>
/// Collects inference and resource telemetry. Publishes to <see cref="Meter"/> for OpenTelemetry
/// exporters while also keeping a bounded in-memory window for the live dashboard, which needs
/// recent history rather than aggregates.
/// </summary>
public sealed class InferenceMetrics : IDisposable
{
    public const string MeterName = "LocalGen.Inference";

    private const int MaxSamples = 500;

    private readonly Meter _meter;
    private readonly Counter<long> _requests;
    private readonly Counter<long> _promptTokens;
    private readonly Counter<long> _completionTokens;
    private readonly Histogram<double> _timeToFirstToken;
    private readonly Histogram<double> _tokensPerSecond;

    private readonly ConcurrentQueue<InferenceSample> _inferenceSamples = new();
    private readonly ConcurrentQueue<ResourceSample> _resourceSamples = new();

    public InferenceMetrics()
    {
        _meter = new Meter(MeterName, "1.0.0");
        _requests = _meter.CreateCounter<long>("localgen.requests", "requests", "Inference requests served.");
        _promptTokens = _meter.CreateCounter<long>("localgen.tokens.prompt", "tokens", "Prompt tokens processed.");
        _completionTokens = _meter.CreateCounter<long>("localgen.tokens.completion", "tokens", "Tokens generated.");
        _timeToFirstToken = _meter.CreateHistogram<double>("localgen.ttft", "ms", "Time to first token.");
        _tokensPerSecond = _meter.CreateHistogram<double>("localgen.throughput", "tokens/s", "Generation throughput.");
    }

    /// <summary>Raised on every recorded generation so dashboards can update without polling.</summary>
    public event EventHandler<InferenceSample>? InferenceRecorded;

    /// <summary>Raised on every resource poll.</summary>
    public event EventHandler<ResourceSample>? ResourceRecorded;

    public void Record(InferenceSample sample)
    {
        // Every tag here is drawn from a bounded set — installed models, compiled-in engines,
        // configured key names — so the exported series cannot fan out with traffic.
        var tags = new TagList
        {
            { "model", sample.Model },
            { "engine", sample.Engine },
            { "status", sample.Failed ? "error" : "ok" }
        };

        if (sample.Tenant.Length > 0)
        {
            tags.Add("tenant", sample.Tenant);
        }

        _requests.Add(1, tags);
        _promptTokens.Add(sample.PromptTokens, tags);
        _completionTokens.Add(sample.CompletionTokens, tags);

        if (!sample.Failed)
        {
            _timeToFirstToken.Record(sample.TimeToFirstToken.TotalMilliseconds, tags);
            _tokensPerSecond.Record(sample.TokensPerSecond, tags);
        }

        Enqueue(_inferenceSamples, sample);
        InferenceRecorded?.Invoke(this, sample);
    }

    public void Record(ResourceSample sample)
    {
        Enqueue(_resourceSamples, sample);
        ResourceRecorded?.Invoke(this, sample);
    }

    public IReadOnlyList<InferenceSample> RecentInference() => [.. _inferenceSamples];

    public IReadOnlyList<ResourceSample> RecentResources() => [.. _resourceSamples];

    /// <summary>Aggregate view over the retained window, for the dashboard summary tiles.</summary>
    public MetricsSummary Summarize()
    {
        var samples = _inferenceSamples.ToArray();
        if (samples.Length == 0)
        {
            return new MetricsSummary();
        }

        var succeeded = samples.Where(static s => !s.Failed).ToArray();

        return new MetricsSummary
        {
            TotalRequests = samples.Length,
            FailedRequests = samples.Length - succeeded.Length,
            TotalPromptTokens = samples.Sum(static s => (long)s.PromptTokens),
            TotalCompletionTokens = samples.Sum(static s => (long)s.CompletionTokens),
            AverageTimeToFirstToken = succeeded.Length > 0
                ? TimeSpan.FromMilliseconds(succeeded.Average(static s => s.TimeToFirstToken.TotalMilliseconds))
                : TimeSpan.Zero,
            AverageTokensPerSecond = succeeded.Length > 0
                ? succeeded.Average(static s => s.TokensPerSecond)
                : 0,
            ModelUsage = samples
                .GroupBy(static s => s.Model)
                .ToDictionary(static g => g.Key, static g => (long)g.Count())
        };
    }

    /// <summary>Keeps a bounded window so a long-running server does not grow without limit.</summary>
    private static void Enqueue<T>(ConcurrentQueue<T> queue, T item)
    {
        queue.Enqueue(item);
        while (queue.Count > MaxSamples && queue.TryDequeue(out _))
        {
        }
    }

    public void Dispose() => _meter.Dispose();
}

public sealed record MetricsSummary
{
    public int TotalRequests { get; init; }

    public int FailedRequests { get; init; }

    public long TotalPromptTokens { get; init; }

    public long TotalCompletionTokens { get; init; }

    public TimeSpan AverageTimeToFirstToken { get; init; }

    public double AverageTokensPerSecond { get; init; }

    public IReadOnlyDictionary<string, long> ModelUsage { get; init; } =
        new Dictionary<string, long>();
}
