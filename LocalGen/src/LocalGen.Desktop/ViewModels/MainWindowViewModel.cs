using System.Collections.ObjectModel;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LocalGen.Core.Diagnostics;
using LocalGen.Desktop.Services;
using LocalGen.Runtime.Sessions;
using LocalGen.Server;

namespace LocalGen.Desktop.ViewModels;

/// <summary>One destination in the navigation rail.</summary>
public sealed partial class NavigationItem(string label, string glyph, ViewModelBase target) : ObservableObject
{
    public string Label { get; } = label;

    /// <summary>A geometry path, so the rail needs no icon font or image assets.</summary>
    public string Glyph { get; } = glyph;

    public ViewModelBase Target { get; } = target;
}

/// <summary>
/// The shell: navigation plus the readout strip that stays visible on every screen.
/// </summary>
/// <remarks>
/// The readout is the application's signature. Local inference is the one setting where the
/// machine is visible, so throughput, the loaded model and the service state are always on
/// screen rather than buried in a status page you have to go and find.
/// </remarks>
public sealed partial class MainWindowViewModel : ViewModelBase
{
    private const int SparklineSampleCount = 60;

    private readonly ServerController _server;
    private readonly ModelSessionManager _sessions;
    private readonly InferenceMetrics _metrics;
    private readonly ThemeService _theme;
    private readonly DispatcherTimer _timer;

    [ObservableProperty]
    private NavigationItem? _selectedItem;

    [ObservableProperty]
    private string _serviceState = "stopped";

    /// <summary>Drives the LED style: <c>stopped</c>, <c>running</c>, <c>busy</c> or <c>faulted</c>.</summary>
    [ObservableProperty]
    private string _ledState = "stopped";

    [ObservableProperty]
    private string _serviceAddress = "not listening";

    [ObservableProperty]
    private string _loadedModelSummary = "no model loaded";

    [ObservableProperty]
    private string _throughput = "—";

    [ObservableProperty]
    private string _deviceSummary = "—";

    [ObservableProperty]
    private string _memorySummary = "—";

    [ObservableProperty]
    private bool _isDarkTheme;

    public MainWindowViewModel(
        ServerController server,
        ModelSessionManager sessions,
        InferenceMetrics metrics,
        ThemeService theme,
        ServiceViewModel service,
        PlaygroundViewModel playground,
        ModelsViewModel models,
        EngineViewModel engine,
        ApiTestViewModel apiTest,
        MonitorViewModel monitor,
        AboutViewModel about)
    {
        _server = server;
        _sessions = sessions;
        _metrics = metrics;
        _theme = theme;

        IsDarkTheme = theme.IsDark;

        Items =
        [
            new NavigationItem("Service", Icons.Power, service),
            new NavigationItem("Playground", Icons.Chat, playground),
            new NavigationItem("Models", Icons.Layers, models),
            new NavigationItem("Engine", Icons.Chip, engine),
            new NavigationItem("API test", Icons.Terminal, apiTest),
            new NavigationItem("Monitor", Icons.Pulse, monitor),
            new NavigationItem("About", Icons.Info, about)
        ];

        SelectedItem = Items[0];

        _server.StateChanged += (_, _) => Dispatcher.UIThread.Post(RefreshReadout);
        _metrics.InferenceRecorded += OnInferenceRecorded;

        // One timer drives the whole readout. Polling at 1 Hz is enough for a status strip and
        // avoids a per-screen refresh loop competing for the UI thread.
        _timer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _timer.Tick += (_, _) => RefreshReadout();
        _timer.Start();

        RefreshReadout();
    }

    public ObservableCollection<NavigationItem> Items { get; }

    /// <summary>Recent tokens-per-second readings, oldest first, for the sparkline.</summary>
    public ObservableCollection<double> ThroughputHistory { get; } = [];

    public string Version =>
        typeof(MainWindowViewModel).Assembly.GetName().Version?.ToString(3) ?? "0.1.0";

    partial void OnSelectedItemChanged(NavigationItem? value)
    {
        if (value is not null)
        {
            // Fire-and-forget is deliberate: navigation must not block on a screen's first load,
            // and ViewModelBase.RunAsync already surfaces failures on the screen itself.
            _ = value.Target.InitializeAsync();
        }
    }

    [RelayCommand]
    private void ToggleTheme()
    {
        _theme.Toggle();
        IsDarkTheme = _theme.IsDark;
    }

    private void OnInferenceRecorded(object? sender, InferenceSample sample) =>
        Dispatcher.UIThread.Post(() =>
        {
            if (sample.Failed || sample.TokensPerSecond <= 0)
            {
                return;
            }

            ThroughputHistory.Add(sample.TokensPerSecond);

            while (ThroughputHistory.Count > SparklineSampleCount)
            {
                ThroughputHistory.RemoveAt(0);
            }

            Throughput = $"{sample.TokensPerSecond:N1} tok/s";
        });

    private void RefreshReadout()
    {
        var loaded = _sessions.Loaded;
        var busy = loaded.Any(static m => m.ActiveRequests > 0);

        ServiceState = _server.State switch
        {
            ServerState.Running => "running",
            ServerState.Starting => "starting",
            ServerState.Stopping => "stopping",
            ServerState.Faulted => "faulted",
            _ => "stopped"
        };

        LedState = _server.State switch
        {
            ServerState.Faulted => "faulted",
            ServerState.Running when busy => "busy",
            ServerState.Running => "running",
            ServerState.Starting or ServerState.Stopping => "busy",
            _ => busy ? "busy" : "stopped"
        };

        ServiceAddress = _server.BaseUrl ?? "not listening";

        if (loaded.Count == 0)
        {
            LoadedModelSummary = "no model loaded";
            DeviceSummary = "—";
            MemorySummary = "—";
            return;
        }

        var primary = loaded[0];

        LoadedModelSummary = loaded.Count == 1
            ? primary.ModelId
            : $"{primary.ModelId} +{loaded.Count - 1}";

        DeviceSummary = $"{primary.Session.Engine} · {primary.Session.Device}";
        MemorySummary = $"ctx {primary.Session.ContextSize:N0}";
    }

    /// <summary>Stops the service and releases loaded models before the process exits.</summary>
    public async Task ShutdownAsync()
    {
        _timer.Stop();

        await _server.StopAsync().ConfigureAwait(false);
        await _sessions.DisposeAsync().ConfigureAwait(false);
    }
}

/// <summary>
/// Icon geometries, kept in one place so the rail needs no font or image dependency.
/// Drawn on a 24×24 grid.
/// </summary>
internal static class Icons
{
    public const string Power =
        "M12 3 L12 12 M7.5 5.5 A7 7 0 1 0 16.5 5.5";

    public const string Chat =
        "M4 5 H20 V16 H10 L5.5 20 V16 H4 Z";

    public const string Layers =
        "M12 3 L21 8 L12 13 L3 8 Z M3 12.5 L12 17.5 L21 12.5 M3 16.5 L12 21.5 L21 16.5";

    public const string Chip =
        "M7 7 H17 V17 H7 Z M9.5 4 V7 M14.5 4 V7 M9.5 17 V20 M14.5 17 V20 " +
        "M4 9.5 H7 M4 14.5 H7 M17 9.5 H20 M17 14.5 H20";

    public const string Pulse =
        "M3 12 H7 L9.5 5 L14 19 L16.5 12 H21";

    public const string Terminal =
        "M3 5 H21 V19 H3 Z M6.5 9.5 L9.5 12 L6.5 14.5 M12.5 15 H17";

    public const string Info =
        "M12 3 A9 9 0 1 0 12 21 A9 9 0 1 0 12 3 M12 11 V16.5 M12 7.5 V8.5";

    public const string Sun =
        "M12 7.5 A4.5 4.5 0 1 0 12 16.5 A4.5 4.5 0 1 0 12 7.5 M12 2 V4 M12 20 V22 " +
        "M2 12 H4 M20 12 H22 M5 5 L6.5 6.5 M17.5 17.5 L19 19 M19 5 L17.5 6.5 M6.5 17.5 L5 19";

    public const string Moon =
        "M20 14.5 A9 9 0 1 1 9.5 4 A7 7 0 0 0 20 14.5 Z";
}
