using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LocalGen.Core.Configuration;
using LocalGen.Core.Engines;
using LocalGen.Runtime.Engines;
using Microsoft.Extensions.Options;

namespace LocalGen.Desktop.ViewModels;

/// <summary>One backend as presented on the Engine screen.</summary>
public sealed partial class EngineRow(EngineStatus status, bool isSelected) : ObservableObject
{
    [ObservableProperty]
    private bool _isSelected = isSelected;

    public EngineStatus Status { get; } = status;

    public EngineKind Kind => Status.Descriptor.Kind;

    public string DisplayName => Status.Descriptor.DisplayName;

    public string Description => Status.Descriptor.Description;

    public string Recommendation => Status.Descriptor.Recommendation;

    public bool IsAvailable => Status.Availability.IsAvailable;

    public string StateLabel => IsAvailable ? "available" : "unavailable";

    public string UnavailableReason => Status.Availability.Reason;

    public bool HasReason => !string.IsNullOrEmpty(Status.Availability.Reason);

    public string Devices => string.Join(" · ", Status.Availability.AvailableDevices);

    public string Formats => string.Join(" · ", Status.Descriptor.Capabilities.Formats);

    public string Build => Status.Availability.Version;

    public bool HasBuild => !string.IsNullOrEmpty(Status.Availability.Version);

    public bool SupportsGrammar => Status.Descriptor.Capabilities.SupportsGrammar;

    public bool SupportsEmbeddings => Status.Descriptor.Capabilities.SupportsEmbeddings;

    public bool SupportsMultiGpu => Status.Descriptor.Capabilities.SupportsMultiGpu;
}

/// <summary>
/// Picks the inference backend and how it uses the machine.
/// </summary>
/// <remarks>
/// Backend choice is the setting with the largest effect on speed, and the one users have the
/// least basis to decide. The screen therefore leads with a concrete recommendation derived from
/// what actually probed successfully on this machine, rather than presenting four equal options.
/// </remarks>
public sealed partial class EngineViewModel : ViewModelBase
{
    private readonly EngineRegistry _registry;
    private readonly LocalGenOptions _options;

    [ObservableProperty]
    private string _recommendationHeadline = string.Empty;

    [ObservableProperty]
    private string _recommendationRationale = string.Empty;

    [ObservableProperty]
    private DeviceKind _selectedDevice;

    [ObservableProperty]
    private int _gpuLayers;

    [ObservableProperty]
    private int _contextSize;

    [ObservableProperty]
    private int _threads;

    [ObservableProperty]
    private string _saveHint = string.Empty;

    public EngineViewModel(EngineRegistry registry, IOptions<LocalGenOptions> options)
    {
        _registry = registry;
        _options = options.Value;

        SelectedDevice = _options.Engine.Device;
        GpuLayers = _options.Engine.GpuLayers ?? 0;
        ContextSize = _options.Engine.ContextSize ?? 0;
        Threads = _options.Engine.Threads ?? 0;
    }

    public ObservableCollection<EngineRow> Engines { get; } = [];

    public IReadOnlyList<DeviceKind> Devices { get; } =
        [DeviceKind.Auto, DeviceKind.Cpu, DeviceKind.Cuda, DeviceKind.Vulkan, DeviceKind.Metal, DeviceKind.DirectML];

    public override Task InitializeAsync() => RefreshAsync();

    [RelayCommand]
    private Task RefreshAsync() => RunAsync(async () =>
    {
        var statuses = await _registry.GetStatusAsync().ConfigureAwait(true);
        var recommendation = await _registry.RecommendAsync().ConfigureAwait(true);

        Engines.Clear();
        foreach (var status in statuses)
        {
            Engines.Add(new EngineRow(status, status.Descriptor.Kind == _options.Engine.Default));
        }

        RecommendationHeadline = $"{recommendation.Engine} on {recommendation.Device}";
        RecommendationRationale = recommendation.Rationale;
    });

    [RelayCommand]
    private void Select(EngineRow? row)
    {
        if (row is null || !row.IsAvailable)
        {
            return;
        }

        foreach (var engine in Engines)
        {
            engine.IsSelected = engine.Kind == row.Kind;
        }

        _options.Engine.Default = row.Kind;
        NoteChange();
    }

    [RelayCommand]
    private void Apply()
    {
        // Applied to the live options object, which the session manager reads on the next load.
        // Models already resident keep the settings they were loaded with.
        _options.Engine.Device = SelectedDevice;
        _options.Engine.GpuLayers = GpuLayers > 0 ? GpuLayers : null;
        _options.Engine.ContextSize = ContextSize > 0 ? ContextSize : null;
        _options.Engine.Threads = Threads > 0 ? Threads : null;

        NoteChange();
    }

    [RelayCommand]
    private void UseRecommended() => _ = RunAsync(async () =>
    {
        var recommendation = await _registry.RecommendAsync().ConfigureAwait(true);

        SelectedDevice = recommendation.Device;
        _options.Engine.Default = recommendation.Engine;
        _options.Engine.Device = recommendation.Device;

        foreach (var engine in Engines)
        {
            engine.IsSelected = engine.Kind == recommendation.Engine;
        }

        NoteChange();
    });

    private void NoteChange() =>
        SaveHint = "Applied. Models already in memory keep their current settings — " +
                   "unload and reload one to use the new configuration.";
}
