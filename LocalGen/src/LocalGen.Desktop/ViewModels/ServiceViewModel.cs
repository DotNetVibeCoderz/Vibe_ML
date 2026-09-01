using System.Collections.ObjectModel;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LocalGen.Core.Configuration;
using LocalGen.Desktop.Services;
using LocalGen.Runtime.Diagnostics;
using LocalGen.Runtime.Sessions;
using LocalGen.Server;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalGen.Desktop.ViewModels;

/// <summary>
/// Start, stop and observe the web service, and read its log.
/// </summary>
/// <remarks>
/// The service runs inside this process, so starting it here makes the models the Playground has
/// already loaded available over HTTP without loading them twice.
/// </remarks>
public sealed partial class ServiceViewModel : ViewModelBase
{
    /// <summary>Log lines retained in the panel. Older lines age out of the shared store anyway.</summary>
    private const int MaxDisplayedLogEntries = 500;

    private readonly ServerController _server;
    private readonly InMemoryLogStore _logs;
    private readonly ModelSessionManager _sessions;
    private readonly LocalGenOptions _options;

    [ObservableProperty]
    private string _state = "stopped";

    [ObservableProperty]
    private bool _isRunning;

    [ObservableProperty]
    private bool _canStart = true;

    [ObservableProperty]
    private string _address = "—";

    [ObservableProperty]
    private string _dataDirectory = string.Empty;

    [ObservableProperty]
    private bool _followLog = true;

    [ObservableProperty]
    private LogLevel _minimumLevel = LogLevel.Information;

    public ServiceViewModel(
        ServerController server,
        InMemoryLogStore logs,
        ModelSessionManager sessions,
        IOptions<LocalGenOptions> options)
    {
        _server = server;
        _logs = logs;
        _sessions = sessions;
        _options = options.Value;

        DataDirectory = _options.DataDirectory;

        _server.StateChanged += (_, _) => Dispatcher.UIThread.Post(Refresh);
        _logs.EntryAdded += OnLogEntryAdded;

        Refresh();
    }

    public ObservableCollection<LogEntry> LogEntries { get; } = [];

    public ObservableCollection<LoadedModel> LoadedModels { get; } = [];

    public IReadOnlyList<LogLevel> LogLevels { get; } =
        [LogLevel.Debug, LogLevel.Information, LogLevel.Warning, LogLevel.Error];

    public string ConfiguredEndpoint => _options.Server.BaseUrl;

    public string OpenAiEndpoint => $"{_options.Server.BaseUrl}/v1";

    public bool RequiresApiKey => !string.IsNullOrEmpty(_options.Server.ApiKey);

    public bool IsOffline => _options.Runtime.OfflineMode;

    public override Task InitializeAsync()
    {
        ReloadLog();
        Refresh();
        return Task.CompletedTask;
    }

    [RelayCommand]
    private Task StartAsync() => RunAsync(async () =>
    {
        await _server.StartAsync().ConfigureAwait(true);
        Refresh();
    });

    [RelayCommand]
    private Task StopAsync() => RunAsync(async () =>
    {
        await _server.StopAsync().ConfigureAwait(true);
        Refresh();
    });

    [RelayCommand]
    private Task RestartAsync() => RunAsync(async () =>
    {
        await _server.RestartAsync().ConfigureAwait(true);
        Refresh();
    });

    [RelayCommand]
    private Task UnloadAllAsync() => RunAsync(async () =>
    {
        foreach (var model in _sessions.Loaded)
        {
            await _sessions.UnloadAsync(model.ModelId).ConfigureAwait(true);
        }

        Refresh();
    });

    [RelayCommand]
    private void ClearLog()
    {
        _logs.Clear();
        LogEntries.Clear();
    }

    partial void OnMinimumLevelChanged(LogLevel value) => ReloadLog();

    private void ReloadLog()
    {
        LogEntries.Clear();

        foreach (var entry in _logs.Recent(MaxDisplayedLogEntries, MinimumLevel))
        {
            LogEntries.Add(entry);
        }
    }

    private void OnLogEntryAdded(object? sender, LogEntry entry)
    {
        if (entry.Level < MinimumLevel)
        {
            return;
        }

        // Log lines arrive on whichever thread produced them.
        Dispatcher.UIThread.Post(() =>
        {
            LogEntries.Add(entry);

            while (LogEntries.Count > MaxDisplayedLogEntries)
            {
                LogEntries.RemoveAt(0);
            }
        });
    }

    private void Refresh()
    {
        State = _server.State switch
        {
            ServerState.Running => "running",
            ServerState.Starting => "starting",
            ServerState.Stopping => "stopping",
            ServerState.Faulted => "faulted",
            _ => "stopped"
        };

        IsRunning = _server.State == ServerState.Running;
        CanStart = _server.State is ServerState.Stopped or ServerState.Faulted;
        Address = _server.BaseUrl ?? "—";

        LoadedModels.Clear();
        foreach (var model in _sessions.Loaded)
        {
            LoadedModels.Add(model);
        }
    }
}
