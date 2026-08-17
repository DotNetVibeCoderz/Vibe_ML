using System.Diagnostics.Metrics;
using LocalGen.Core.Configuration;
using LocalGen.Core.Diagnostics;
using LocalGen.Server.Services;
using LocalGen.Server.Tenancy;
using Microsoft.Extensions.DependencyInjection;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

namespace LocalGen.Server.Telemetry;

/// <summary>
/// Exports the metrics LocalGen already collects to Prometheus and OTLP.
/// </summary>
/// <remarks>
/// <see cref="InferenceMetrics"/> has always published to a <see cref="Meter"/>; nothing was
/// listening. This adds the listeners rather than a second measurement path, so the numbers on the
/// Admin Control dashboard and the numbers a Grafana panel draws come from the same recording.
/// </remarks>
public static class TelemetryExtensions
{
    /// <summary>Meter carrying the gauges derived from server state rather than from events.</summary>
    public const string StateMeterName = "LocalGen.Server";

    public static IServiceCollection AddLocalGenTelemetry(
        this IServiceCollection services,
        LocalGenOptions options)
    {
        var telemetry = options.Telemetry;
        var hasOtlp = !string.IsNullOrWhiteSpace(telemetry.OtlpEndpoint);

        if (!telemetry.PrometheusEnabled && !hasOtlp)
        {
            return services;
        }

        services.AddSingleton<StateMetrics>();

        var builder = services
            .AddOpenTelemetry()
            .ConfigureResource(resource => resource.AddService(
                serviceName: telemetry.ServiceName,
                serviceVersion: typeof(TelemetryExtensions).Assembly.GetName().Version?.ToString(3)
                    ?? "0.1.0"));

        builder.WithMetrics(metrics =>
        {
            metrics
                .AddMeter(InferenceMetrics.MeterName)
                .AddMeter(StateMeterName)
                .AddAspNetCoreInstrumentation()
                .AddRuntimeInstrumentation();

            // Latency buckets sized for local inference: a fast first token is tens of
            // milliseconds and a slow whole response is tens of seconds, which the default
            // buckets (topping out at 10 s) would flatten into one saturated bin.
            metrics.AddView(
                instrumentName: "localgen.ttft",
                new ExplicitBucketHistogramConfiguration
                {
                    Boundaries = [10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000]
                });

            metrics.AddView(
                instrumentName: "localgen.throughput",
                new ExplicitBucketHistogramConfiguration
                {
                    Boundaries = [1, 5, 10, 20, 40, 80, 160, 320, 640]
                });

            if (telemetry.PrometheusEnabled)
            {
                metrics.AddPrometheusExporter();
            }

            if (hasOtlp)
            {
                metrics.AddOtlpExporter((exporter, _) => Configure(exporter, telemetry));
            }
        });

        if (telemetry.Traces && hasOtlp)
        {
            // Traces go to OTLP only: Prometheus has nowhere to put a span, and exporting them
            // when no collector is configured would cost work with no destination.
            builder.WithTracing(tracing => tracing
                .AddAspNetCoreInstrumentation(instrumentation =>
                    // Scrapes are frequent, uniform and uninteresting; keeping them would drown
                    // the real inference traces.
                    instrumentation.Filter = context =>
                        !context.Request.Path.StartsWithSegments(telemetry.MetricsPath))
                .AddOtlpExporter(exporter => Configure(exporter, telemetry)));
        }

        return services;
    }

    private static void Configure(
        OpenTelemetry.Exporter.OtlpExporterOptions exporter,
        TelemetryOptions telemetry)
    {
        exporter.Endpoint = new Uri(telemetry.OtlpEndpoint!);
        exporter.Protocol = telemetry.OtlpProtocol.Equals("httpprotobuf", StringComparison.OrdinalIgnoreCase)
            ? OpenTelemetry.Exporter.OtlpExportProtocol.HttpProtobuf
            : OpenTelemetry.Exporter.OtlpExportProtocol.Grpc;
    }
}

/// <summary>
/// Publishes the parts of the server's state that are levels rather than events.
/// </summary>
/// <remarks>
/// Loaded models, cache occupancy and per-key consumption are not produced by anything happening;
/// they are simply true at a moment. Observable instruments read them when the exporter collects,
/// which keeps the values honest without a polling loop of our own.
/// </remarks>
internal sealed class StateMetrics : IDisposable
{
    private readonly Meter _meter;

    public StateMetrics(ResponseCache cache, ApiKeyRegistry keys)
    {
        _meter = new Meter(TelemetryExtensions.StateMeterName, "1.0.0");

        _meter.CreateObservableCounter(
            "localgen.cache.hits", () => cache.Hits, "responses",
            "Responses served from the cache.");

        _meter.CreateObservableCounter(
            "localgen.cache.misses", () => cache.Misses, "responses",
            "Cacheable requests that had to be generated.");

        _meter.CreateObservableGauge(
            "localgen.cache.entries", () => cache.Count, "entries",
            "Responses currently held in the cache.");

        _meter.CreateObservableGauge(
            "localgen.tenant.requests_per_minute",
            () => keys.Tenants.Select(t => new Measurement<long>(
                t.RequestsThisMinute, new KeyValuePair<string, object?>("tenant", t.Name))),
            "requests",
            "Requests served for each key in the last rolling minute.");

        _meter.CreateObservableGauge(
            "localgen.tenant.tokens_today",
            () => keys.Tenants.Select(t => new Measurement<long>(
                t.TokensToday, new KeyValuePair<string, object?>("tenant", t.Name))),
            "tokens",
            "Tokens charged to each key in the last rolling day.");

        _meter.CreateObservableGauge(
            "localgen.tenant.in_flight",
            () => keys.Tenants.Select(t => new Measurement<long>(
                t.InFlight, new KeyValuePair<string, object?>("tenant", t.Name))),
            "requests",
            "Requests each key currently has open.");
    }

    public void Dispose() => _meter.Dispose();
}
