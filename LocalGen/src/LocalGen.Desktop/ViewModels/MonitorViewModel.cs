using System.Collections.ObjectModel;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LocalGen.Core.Diagnostics;

namespace LocalGen.Desktop.ViewModels;

/// <summary>Per-model usage, for the breakdown table.</summary>
public sealed record ModelUsageRow(string Model, long Requests, double Share);

/// <summary>A GPU as reported by the driver.</summary>
public sealed partial class GpuRow(GpuSample sample) : ObservableObject
{
    public string Name => sample.Name;

    public int Index => sample.Index;

    public double UtilizationPercent => sample.UtilizationPercent;

    public string Utilization => $"{sample.UtilizationPercent:N0}%";

    public string Memory =>
        $"{Converters.FormatBytes(sample.MemoryUsedBytes)} / {Converters.FormatBytes(sample.MemoryTotalBytes)}";

    public double MemoryPercent => sample.MemoryTotalBytes > 0
        ? sample.MemoryUsedBytes * 100.0 / sample.MemoryTotalBytes
        : 0;

    public string Temperature => sample.TemperatureCelsius is { } value ? $"{value:N0} °C" : "—";

    public string Power => sample.PowerWatts is { } value ? $"{value:N0} W" : "—";
}

/// <summary>
/// The monitoring dashboard: throughput, latency, token usage and hardware load.
/// </summary>
/// <remarks>
/// Charts are driven by the metrics collector's retained window rather than by a database, so the
/// dashboard shows the recent past and costs nothing when nobody is looking at it.
/// </remarks>
public sealed partial class MonitorViewModel : ViewModelBase
{
    private const int TraceLength = 90;

    private readonly InferenceMetrics _metrics;
    private readonly LocalGen.Runtime.Diagnostics.SystemMonitor _monitor;
    private readonly DispatcherTimer _timer;

    [ObservableProperty]
    private int _totalRequests;

    [ObservableProperty]
    private int _failedRequests;

    [ObservableProperty]
    private string _promptTokens = "0";

    [ObservableProperty]
    private string _completionTokens = "0";

    [ObservableProperty]
    private string _averageThroughput = "—";

    [ObservableProperty]
    private string _averageLatency = "—";

    [ObservableProperty]
    private string _cpuUsage = "—";

    [ObservableProperty]
    private double _cpuPercent;

    [ObservableProperty]
    private string _processMemory = "—";

    [ObservableProperty]
    private bool _hasGpu;

    public MonitorViewModel(
        InferenceMetrics metrics,
        LocalGen.Runtime.Diagnostics.SystemMonitor monitor)
    {
        _metrics = metrics;
        _monitor = monitor;

        _timer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(2) };
        _timer.Tick += async (_, _) => await SampleAsync();
    }

    /// <summary>
    /// Takes a hardware reading, then refreshes the view.
    /// </summary>
    /// <remarks>
    /// The server's maintenance loop samples resources too, but it only runs while the service is
    /// started. Sampling here as well means the dashboard reports CPU and GPU whenever it is open,
    /// rather than showing dashes until someone happens to start the web service.
    /// </remarks>
    private async Task SampleAsync()
    {
        try
        {
            _metrics.Record(await _monitor.SampleAsync().ConfigureAwait(true));
        }
        catch (Exception ex)
        {
            ErrorMessage = ex.Message;
        }

        Refresh();
    }

    /// <summary>Generation throughput over recent requests.</summary>
    public ObservableCollection<double> ThroughputTrace { get; } = [];

    /// <summary>Time to first token, in milliseconds, over recent requests.</summary>
    public ObservableCollection<double> LatencyTrace { get; } = [];

    /// <summary>CPU utilisation of the LocalGen process.</summary>
    public ObservableCollection<double> CpuTrace { get; } = [];

    public ObservableCollection<GpuRow> Gpus { get; } = [];

    public ObservableCollection<ModelUsageRow> ModelUsage { get; } = [];

    public override async Task InitializeAsync()
    {
        await SampleAsync().ConfigureAwait(true);

        // The timer only runs once this screen has been opened; other screens do not pay for it.
        _timer.Start();
    }

    [RelayCommand]
    private void Refresh()
    {
        var summary = _metrics.Summarize();

        TotalRequests = summary.TotalRequests;
        FailedRequests = summary.FailedRequests;
        PromptTokens = summary.TotalPromptTokens.ToString("N0");
        CompletionTokens = summary.TotalCompletionTokens.ToString("N0");

        AverageThroughput = summary.AverageTokensPerSecond > 0
            ? $"{summary.AverageTokensPerSecond:N1} tok/s"
            : "—";

        AverageLatency = summary.AverageTimeToFirstToken > TimeSpan.Zero
            ? $"{summary.AverageTimeToFirstToken.TotalMilliseconds:N0} ms"
            : "—";

        Replace(ThroughputTrace, _metrics.RecentInference()
            .Where(static s => !s.Failed && s.TokensPerSecond > 0)
            .TakeLast(TraceLength)
            .Select(static s => s.TokensPerSecond));

        Replace(LatencyTrace, _metrics.RecentInference()
            .Where(static s => !s.Failed && s.TimeToFirstToken > TimeSpan.Zero)
            .TakeLast(TraceLength)
            .Select(static s => s.TimeToFirstToken.TotalMilliseconds));

        var resources = _metrics.RecentResources();

        Replace(CpuTrace, resources.TakeLast(TraceLength).Select(static s => s.CpuPercent));

        if (resources.Count > 0)
        {
            var latest = resources[^1];

            CpuPercent = latest.CpuPercent;
            CpuUsage = $"{latest.CpuPercent:N0}%";
            ProcessMemory = Converters.FormatBytes(latest.MemoryUsedBytes);

            Gpus.Clear();
            foreach (var gpu in latest.Gpus)
            {
                Gpus.Add(new GpuRow(gpu));
            }

            HasGpu = Gpus.Count > 0;
        }

        ModelUsage.Clear();
        var total = Math.Max(summary.ModelUsage.Values.Sum(), 1);

        foreach (var (model, count) in summary.ModelUsage.OrderByDescending(static pair => pair.Value))
        {
            ModelUsage.Add(new ModelUsageRow(model, count, count * 100.0 / total));
        }
    }

    /// <summary>
    /// Refreshes a bound collection in place. Clearing and re-adding keeps the binding alive,
    /// which reassigning the property would not.
    /// </summary>
    private static void Replace(ObservableCollection<double> target, IEnumerable<double> values)
    {
        target.Clear();

        foreach (var value in values)
        {
            target.Add(value);
        }
    }
}
