using System.Diagnostics;
using System.Globalization;
using LocalGen.Core.Diagnostics;
using Microsoft.Extensions.Logging;

namespace LocalGen.Runtime.Diagnostics;

/// <summary>
/// Samples host resources for the monitoring dashboard.
/// </summary>
/// <remarks>
/// GPU counters come from <c>nvidia-smi</c> rather than a native binding: it ships with every
/// NVIDIA driver, needs no extra dependency, and degrades to "no GPU data" on machines without
/// one. CPU is reported for the LocalGen process, which is the number that actually matters when
/// diagnosing inference throughput.
/// </remarks>
public sealed class SystemMonitor
{
    private readonly ILogger<SystemMonitor> _logger;

    private TimeSpan _lastProcessorTime;
    private DateTimeOffset _lastSampledAt;

    /// <summary>Cached so a machine without NVIDIA tooling is only probed once.</summary>
    private bool? _nvidiaSmiAvailable;

    public SystemMonitor(ILogger<SystemMonitor> logger)
    {
        _logger = logger;
        _lastProcessorTime = Process.GetCurrentProcess().TotalProcessorTime;
        _lastSampledAt = DateTimeOffset.UtcNow;
    }

    public async ValueTask<ResourceSample> SampleAsync(CancellationToken cancellationToken = default)
    {
        var process = Process.GetCurrentProcess();
        var now = DateTimeOffset.UtcNow;

        var elapsed = now - _lastSampledAt;
        var processorTime = process.TotalProcessorTime;

        // CPU percentage is a delta over wall-clock time, normalised by core count so the value
        // stays in 0–100 regardless of how many cores the machine has.
        var cpuPercent = elapsed.TotalMilliseconds > 0
            ? (processorTime - _lastProcessorTime).TotalMilliseconds
              / (elapsed.TotalMilliseconds * Environment.ProcessorCount) * 100
            : 0;

        _lastProcessorTime = processorTime;
        _lastSampledAt = now;

        var memoryInfo = GC.GetGCMemoryInfo();

        return new ResourceSample
        {
            Timestamp = now,
            CpuPercent = Math.Clamp(cpuPercent, 0, 100),
            MemoryUsedBytes = process.WorkingSet64,
            MemoryTotalBytes = memoryInfo.TotalAvailableMemoryBytes,
            Gpus = await SampleGpusAsync(cancellationToken).ConfigureAwait(false)
        };
    }

    private async ValueTask<IReadOnlyList<GpuSample>> SampleGpusAsync(CancellationToken cancellationToken)
    {
        if (_nvidiaSmiAvailable == false)
        {
            return [];
        }

        try
        {
            var output = await RunNvidiaSmiAsync(cancellationToken).ConfigureAwait(false);
            if (output is null)
            {
                _nvidiaSmiAvailable = false;
                return [];
            }

            _nvidiaSmiAvailable = true;
            return ParseNvidiaSmi(output);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogDebug(ex, "GPU sampling failed; continuing without GPU metrics.");
            _nvidiaSmiAvailable = false;
            return [];
        }
    }

    private static async Task<string?> RunNvidiaSmiAsync(CancellationToken cancellationToken)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = "nvidia-smi",
            Arguments =
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw " +
                "--format=csv,noheader,nounits",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = Process.Start(startInfo);
        if (process is null)
        {
            return null;
        }

        var readTask = process.StandardOutput.ReadToEndAsync(cancellationToken);

        // A hung driver must not stall the metrics loop.
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(5));

        try
        {
            await process.WaitForExitAsync(timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                process.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException)
            {
                // Already exited.
            }

            return null;
        }

        return process.ExitCode == 0 ? await readTask.ConfigureAwait(false) : null;
    }

    /// <summary>Parses one CSV row per GPU, in the column order requested above.</summary>
    private static IReadOnlyList<GpuSample> ParseNvidiaSmi(string output)
    {
        var samples = new List<GpuSample>();

        foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var fields = line.Split(',', StringSplitOptions.TrimEntries);
            if (fields.Length < 5)
            {
                continue;
            }

            samples.Add(new GpuSample
            {
                Index = ParseInt(fields[0]),
                Name = fields[1],
                UtilizationPercent = ParseDouble(fields[2]),
                // nvidia-smi reports memory in MiB.
                MemoryUsedBytes = (long)(ParseDouble(fields[3]) * 1024 * 1024),
                MemoryTotalBytes = (long)(ParseDouble(fields[4]) * 1024 * 1024),
                TemperatureCelsius = fields.Length > 5 ? ParseNullableDouble(fields[5]) : null,
                PowerWatts = fields.Length > 6 ? ParseNullableDouble(fields[6]) : null
            });
        }

        return samples;
    }

    private static int ParseInt(string value) =>
        int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) ? parsed : 0;

    private static double ParseDouble(string value) =>
        double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ? parsed : 0;

    /// <summary>nvidia-smi prints <c>[N/A]</c> for counters a card does not expose.</summary>
    private static double? ParseNullableDouble(string value) =>
        double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ? parsed : null;
}
