using System.Collections.ObjectModel;
using System.Globalization;
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

    public string Accelerators => string.Join(" · ", Status.Availability.Accelerators.Select(a => a.Name));

    public bool HasAccelerators => Status.Availability.Accelerators.Count > 0;

    /// <summary>Whether this backend has more than one GPU to split a model across.</summary>
    public bool CanSplit => Status.Availability.Accelerators.Count > 1;

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

    /// <summary>Per-GPU weights as typed, e.g. <c>0.6, 0.4</c>. Empty leaves the split to llama.cpp.</summary>
    [ObservableProperty]
    private string _tensorSplitText = string.Empty;

    [ObservableProperty]
    private GpuSplitMode _splitMode;

    [ObservableProperty]
    private int _mainGpu;

    /// <summary>
    /// What the typed split resolves to against the GPUs actually present, refreshed as it is
    /// typed. A split is otherwise invisible until a model loads, which is a slow way to find a
    /// typo in a number.
    /// </summary>
    [ObservableProperty]
    private string _splitPreview = string.Empty;

    [ObservableProperty]
    private bool _splitPreviewIsWarning;

    /// <summary>Accelerators the selected backend registered, which is what a split can address.</summary>
    private IReadOnlyList<AcceleratorDevice> _accelerators = [];

    public EngineViewModel(EngineRegistry registry, IOptions<LocalGenOptions> options)
    {
        _registry = registry;
        _options = options.Value;

        SelectedDevice = _options.Engine.Device;
        GpuLayers = _options.Engine.GpuLayers ?? 0;
        ContextSize = _options.Engine.ContextSize ?? 0;
        Threads = _options.Engine.Threads ?? 0;
        SplitMode = _options.Engine.SplitMode;
        MainGpu = _options.Engine.MainGpu ?? 0;
        TensorSplitText = string.Join(
            ", ",
            _options.Engine.TensorSplit.Select(v => v.ToString("0.###", CultureInfo.InvariantCulture)));
    }

    public ObservableCollection<EngineRow> Engines { get; } = [];

    public IReadOnlyList<DeviceKind> Devices { get; } =
        [DeviceKind.Auto, DeviceKind.Cpu, DeviceKind.Cuda, DeviceKind.Vulkan, DeviceKind.Metal, DeviceKind.DirectML];

    public IReadOnlyList<GpuSplitMode> SplitModes { get; } =
        [GpuSplitMode.Auto, GpuSplitMode.Layer, GpuSplitMode.Row, GpuSplitMode.None];

    /// <summary>The split controls are only shown when there is more than one GPU to address.</summary>
    public bool MultiGpuAvailable => _accelerators.Count > 1;

    /// <summary>
    /// What the active backend can actually see, which is not the same as what is in the machine:
    /// a CPU-only build registers no GPU at all, and a masked device is invisible here too.
    /// </summary>
    public string AcceleratorSummary => _accelerators.Count switch
    {
        0 => "The active backend has registered no GPU, so models run on the CPU.",
        1 => $"One GPU is registered ({_accelerators[0].Name}). Splitting a model needs a second one.",
        var count => $"{count} GPUs registered: {string.Join(", ", _accelerators.Select(a => a.Name))}."
    };

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

        // The accelerators of the backend that will actually serve, not of every backend probed:
        // a split is resolved against the devices the loaded native library registered.
        UseAccelerators(
            statuses.FirstOrDefault(s => s.Descriptor.Kind == _options.Engine.Default)?
                .Availability.Accelerators ?? []);

        RecommendationHeadline = $"{recommendation.Engine} on {recommendation.Device}";
        RecommendationRationale = recommendation.Rationale;
    });

    partial void OnTensorSplitTextChanged(string value) => UpdateSplitPreview();

    partial void OnSplitModeChanged(GpuSplitMode value) => UpdateSplitPreview();

    private void UpdateSplitPreview()
    {
        if (!TensorSplitPlan.TryParseWeights(TensorSplitText, out var weights))
        {
            SplitPreview = "Not a list of numbers. Write one weight per GPU, e.g. 0.6, 0.4";
            SplitPreviewIsWarning = true;
            return;
        }

        var plan = TensorSplitPlan.Create(weights, _accelerators.Count);

        SplitPreviewIsWarning = plan.Warnings.Count > 0;
        SplitPreview = plan.Warnings.Count > 0
            ? string.Join(" ", plan.Warnings)
            : $"Weights: {plan.Describe(_accelerators)}";
    }

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

        // The split settings describe the backend that will serve, so switching backend changes
        // which devices they address — and whether they apply at all.
        UseAccelerators(row.Status.Availability.Accelerators);

        NoteChange();
    }

    private void UseAccelerators(IReadOnlyList<AcceleratorDevice> accelerators)
    {
        _accelerators = accelerators;

        OnPropertyChanged(nameof(MultiGpuAvailable));
        OnPropertyChanged(nameof(AcceleratorSummary));
        UpdateSplitPreview();
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
        _options.Engine.SplitMode = SplitMode;
        _options.Engine.MainGpu = MainGpu > 0 ? MainGpu : null;

        // An unreadable split is left alone rather than written as an empty one: silently
        // discarding what the user typed would look like it had been accepted.
        if (TensorSplitPlan.TryParseWeights(TensorSplitText, out var weights))
        {
            _options.Engine.TensorSplit = [.. weights];
        }

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
