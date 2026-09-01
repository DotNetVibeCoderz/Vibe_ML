using LocalGen.Core.Configuration;
using LocalGen.Core.Diagnostics;
using LocalGen.Runtime.Diagnostics;
using LocalGen.Runtime.Sessions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalGen.Server.Services;

/// <summary>
/// The background loop that keeps a long-running server healthy: it samples host resources for
/// the monitoring dashboard and unloads models that have gone idle, so an unattended machine does
/// not sit on gigabytes of VRAM indefinitely.
/// </summary>
public sealed class MaintenanceService : BackgroundService
{
    private readonly ModelSessionManager _sessions;
    private readonly SystemMonitor _monitor;
    private readonly InferenceMetrics _metrics;
    private readonly LocalGenOptions _options;
    private readonly ILogger<MaintenanceService> _logger;

    public MaintenanceService(
        ModelSessionManager sessions,
        SystemMonitor monitor,
        InferenceMetrics metrics,
        IOptions<LocalGenOptions> options,
        ILogger<MaintenanceService> logger)
    {
        _sessions = sessions;
        _monitor = monitor;
        _metrics = metrics;
        _options = options.Value;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        await PreloadAsync(stoppingToken).ConfigureAwait(false);

        using var timer = new PeriodicTimer(_options.Runtime.MetricsInterval);
        var lastEviction = DateTimeOffset.UtcNow;

        while (await timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false))
        {
            try
            {
                if (_options.Runtime.CollectMetrics)
                {
                    _metrics.Record(await _monitor.SampleAsync(stoppingToken).ConfigureAwait(false));
                }

                // Eviction is checked far less often than metrics are sampled; scanning loaded
                // models every couple of seconds would be pure overhead.
                if (DateTimeOffset.UtcNow - lastEviction > TimeSpan.FromSeconds(30))
                {
                    lastEviction = DateTimeOffset.UtcNow;
                    await _sessions.EvictIdleAsync(stoppingToken).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                // A failed sample must never take the server down.
                _logger.LogWarning(ex, "Maintenance tick failed.");
            }
        }
    }

    /// <summary>
    /// Loads the configured model at startup so the first request does not pay the load cost.
    /// </summary>
    private async Task PreloadAsync(CancellationToken cancellationToken)
    {
        var modelId = _options.Runtime.PreloadModel;

        if (string.IsNullOrWhiteSpace(modelId))
        {
            return;
        }

        try
        {
            _logger.LogInformation("Preloading {Model}…", modelId);
            using var lease = await _sessions.AcquireAsync(modelId, null, cancellationToken)
                .ConfigureAwait(false);
            _logger.LogInformation("Preloaded {Model}.", modelId);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not preload {Model}; it will load on first use.", modelId);
        }
    }
}
