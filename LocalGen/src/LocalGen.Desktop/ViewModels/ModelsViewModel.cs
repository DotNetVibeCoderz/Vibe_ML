using System.Collections.ObjectModel;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LocalGen.Core.Models;
using LocalGen.Runtime.Sessions;

namespace LocalGen.Desktop.ViewModels;

/// <summary>A downloadable variant, as shown in the gallery detail pane.</summary>
public sealed partial class VariantRow(CatalogVariant variant) : ObservableObject
{
    public CatalogVariant Variant { get; } = variant;

    public string Quantization => Variant.Quantization.Name;

    public string Size => Converters.FormatBytes(Variant.SizeBytes);

    public string Notes => Variant.Quantization.Notes;

    public string FileName => Variant.FileName;
}

/// <summary>
/// The model browser: what is installed, what the catalogues offer, and download progress.
/// </summary>
/// <remarks>
/// The gallery and the local list are one screen rather than two because the decision a user is
/// actually making — which quantization to keep on disk — needs both in view: what is already
/// there, and what it would cost to add another.
/// </remarks>
public sealed partial class ModelsViewModel : ViewModelBase
{
    private readonly IModelStore _store;
    private readonly IEnumerable<IModelCatalog> _catalogs;
    private readonly ModelSessionManager _sessions;
    private readonly IModelQuantizer? _quantizer;

    private CancellationTokenSource? _download;
    private CancellationTokenSource? _quantize;

    [ObservableProperty]
    private string _searchQuery = string.Empty;

    [ObservableProperty]
    private ModelDescriptor? _selectedInstalled;

    [ObservableProperty]
    private CatalogEntry? _selectedCatalogEntry;

    [ObservableProperty]
    private bool _isSearching;

    [ObservableProperty]
    private bool _isDownloading;

    [ObservableProperty]
    private double _downloadPercentage;

    [ObservableProperty]
    private string _downloadStatus = string.Empty;

    [ObservableProperty]
    private string _totalSize = "—";

    [ObservableProperty]
    private string _selectedQuantization = string.Empty;

    [ObservableProperty]
    private bool _isQuantizing;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasQuantizeProgress))]
    private string _quantizeStatus = string.Empty;

    [ObservableProperty]
    private bool _allowRequantize;

    /// <summary>Whether <see cref="QuantizeStatus"/> is reporting a failure rather than progress.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasQuantizeProgress))]
    private bool _quantizeFailed;

    /// <summary>There is something to say and it is not a failure.</summary>
    public bool HasQuantizeProgress => !QuantizeFailed && !string.IsNullOrEmpty(QuantizeStatus);

    /// <param name="quantizers">
    /// Taken as a collection because only the llama.cpp backend registers one: an ONNX-only or
    /// Foundry-only install resolves an empty sequence, where a plain dependency would not resolve
    /// at all. The panel then hides itself rather than failing.
    /// </param>
    public ModelsViewModel(
        IModelStore store,
        IEnumerable<IModelCatalog> catalogs,
        ModelSessionManager sessions,
        IEnumerable<IModelQuantizer> quantizers)
    {
        _store = store;
        _catalogs = catalogs;
        _sessions = sessions;
        _quantizer = quantizers.FirstOrDefault();

        foreach (var quantization in _quantizer?.Supported ?? [])
        {
            QuantizationTargets.Add(quantization.Name);
        }

        SelectedQuantization = QuantizationTargets.Contains("Q4_K_M")
            ? "Q4_K_M"
            : QuantizationTargets.FirstOrDefault() ?? string.Empty;

        _store.Changed += (_, _) => Dispatcher.UIThread.Post(() => _ = RefreshAsync());
    }

    /// <summary>Whether any installed backend can quantize; drives the panel's visibility.</summary>
    public bool CanQuantize => _quantizer is not null;

    public ObservableCollection<string> QuantizationTargets { get; } = [];

    public ObservableCollection<ModelDescriptor> Installed { get; } = [];

    public ObservableCollection<CatalogEntry> SearchResults { get; } = [];

    public ObservableCollection<VariantRow> Variants { get; } = [];

    public override Task InitializeAsync() => RefreshAsync();

    [RelayCommand]
    private Task RefreshAsync() => RunAsync(async () =>
    {
        var models = await _store.ListAsync().ConfigureAwait(true);

        Installed.Clear();
        foreach (var model in models)
        {
            Installed.Add(model);
        }

        TotalSize = Converters.FormatBytes(models.Sum(static m => m.SizeBytes));
    });

    [RelayCommand]
    private async Task SearchAsync()
    {
        if (string.IsNullOrWhiteSpace(SearchQuery))
        {
            SearchResults.Clear();
            return;
        }

        IsSearching = true;
        ErrorMessage = string.Empty;

        try
        {
            SearchResults.Clear();

            foreach (var catalog in _catalogs)
            {
                var results = await catalog.SearchAsync(SearchQuery, 30).ConfigureAwait(true);

                foreach (var entry in results)
                {
                    SearchResults.Add(entry);
                }
            }

            if (SearchResults.Count == 0)
            {
                ErrorMessage = $"Nothing matched “{SearchQuery}”. Try a shorter query, " +
                               "or check that LocalGen is not in offline mode.";
            }
        }
        catch (Exception ex)
        {
            ErrorMessage = ex.Message;
        }
        finally
        {
            IsSearching = false;
        }
    }

    /// <summary>Loads the variants for the selected catalogue entry so the user can pick a quantization.</summary>
    partial void OnSelectedCatalogEntryChanged(CatalogEntry? value)
    {
        Variants.Clear();

        if (value is null)
        {
            return;
        }

        _ = RunAsync(async () =>
        {
            foreach (var catalog in _catalogs)
            {
                if (!string.Equals(catalog.Provider, value.Provider, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                var detail = await catalog.GetAsync(value.Reference).ConfigureAwait(true);

                foreach (var variant in detail?.Variants ?? [])
                {
                    Variants.Add(new VariantRow(variant));
                }

                break;
            }
        });
    }

    [RelayCommand]
    private async Task DownloadAsync(object? parameter)
    {
        var reference = parameter switch
        {
            VariantRow row => row.Variant.Reference,
            CatalogEntry entry => entry.Reference,
            string text => text,
            _ => SelectedCatalogEntry?.Reference
        };

        if (string.IsNullOrWhiteSpace(reference))
        {
            return;
        }

        _download?.Cancel();
        _download = new CancellationTokenSource();

        IsDownloading = true;
        DownloadPercentage = 0;
        DownloadStatus = "starting…";
        ErrorMessage = string.Empty;

        // Progress arrives from the download thread; Progress<T> marshals it back to the UI thread.
        var progress = new Progress<DownloadProgress>(update =>
        {
            DownloadPercentage = update.Percentage;
            DownloadStatus = update.TotalBytes > 0
                ? $"{Converters.FormatBytes(update.BytesDownloaded)} of {Converters.FormatBytes(update.TotalBytes)}" +
                  $" · {Converters.FormatBytes((long)update.BytesPerSecond)}/s"
                : update.Status;
        });

        try
        {
            var model = await _store.PullAsync(reference, progress, _download.Token).ConfigureAwait(true);

            DownloadStatus = $"installed {model.Id}";
            await RefreshAsync().ConfigureAwait(true);
        }
        catch (OperationCanceledException)
        {
            DownloadStatus = "cancelled";
        }
        catch (Exception ex)
        {
            ErrorMessage = ex.Message;
            DownloadStatus = "failed";
        }
        finally
        {
            IsDownloading = false;
        }
    }

    [RelayCommand]
    private void CancelDownload() => _download?.Cancel();

    [RelayCommand]
    private Task RemoveAsync(ModelDescriptor? model) => RunAsync(async () =>
    {
        if (model is null)
        {
            return;
        }

        // Unload first: deleting the file under a live session would crash the backend.
        await _sessions.UnloadAsync(model.Id).ConfigureAwait(true);
        await _store.RemoveAsync(model.Id).ConfigureAwait(true);
        await RefreshAsync().ConfigureAwait(true);
    });

    [RelayCommand]
    private Task LoadAsync(ModelDescriptor? model) => RunAsync(async () =>
    {
        if (model is null)
        {
            return;
        }

        using var lease = await _sessions.AcquireAsync(model.Id).ConfigureAwait(true);
    });

    [RelayCommand]
    private Task UnloadAsync(ModelDescriptor? model) => RunAsync(async () =>
    {
        if (model is not null)
        {
            await _sessions.UnloadAsync(model.Id).ConfigureAwait(true);
        }
    });

    /// <summary>
    /// Converts the selected model to a smaller quantization, writing a new file beside it.
    /// </summary>
    /// <remarks>
    /// The source is left alone: a quantization is a lossy derivative, and someone who dislikes
    /// the result needs the original still there to try a different target.
    /// </remarks>
    [RelayCommand]
    private async Task QuantizeAsync(ModelDescriptor? model)
    {
        var source = model ?? SelectedInstalled;

        if (_quantizer is null || source is null || string.IsNullOrEmpty(SelectedQuantization))
        {
            return;
        }

        var sourcePath = source.Path;

        // Failures report through QuantizeStatus rather than the shared error banner, which sits
        // over the catalogue column — an answer to a button press belongs beside the button.
        if (string.IsNullOrEmpty(sourcePath) || !File.Exists(sourcePath))
        {
            Fail($"{source.Id} has no local weights to quantize.");
            return;
        }

        var targetPath = QuantizationNaming.BuildOutputPath(sourcePath, SelectedQuantization);

        if (File.Exists(targetPath))
        {
            Fail($"{Path.GetFileName(targetPath)} already exists. " +
                 "Delete it first, or pick a different quantization.");
            return;
        }

        _quantize?.Cancel();
        _quantize = new CancellationTokenSource();

        IsQuantizing = true;
        QuantizeFailed = false;
        QuantizeStatus = $"quantizing {source.Name} to {SelectedQuantization}…";

        try
        {
            // llama.cpp reports progress per tensor through the log rather than as a fraction, so
            // there is nothing honest to bind a progress bar to here.
            var result = await _quantizer.QuantizeAsync(
                new QuantizationRequest
                {
                    SourcePath = sourcePath,
                    TargetPath = targetPath,
                    Quantization = SelectedQuantization,
                    AllowRequantize = AllowRequantize,
                    QuantizeOutputTensor = true
                },
                _quantize.Token).ConfigureAwait(true);

            QuantizeStatus =
                $"wrote {Path.GetFileName(result.TargetPath)} — " +
                $"{Converters.FormatBytes(result.SourceBytes)} → {Converters.FormatBytes(result.TargetBytes)} " +
                $"({result.Reduction:P0} smaller) in {result.Duration.TotalSeconds:N0}s";

            // Registering it explicitly rather than relying on the directory scan: the store only
            // rescans when its cache is cold, so a refresh alone would not show the new file until
            // the next restart.
            await _store.PullAsync(result.TargetPath).ConfigureAwait(true);
            await RefreshAsync().ConfigureAwait(true);
        }
        catch (OperationCanceledException)
        {
            QuantizeStatus = "cancelled";
        }
        catch (Exception ex)
        {
            Fail(ex.Message);
        }
        finally
        {
            IsQuantizing = false;
        }
    }

    private void Fail(string message)
    {
        QuantizeFailed = true;
        QuantizeStatus = message;
    }

    [RelayCommand]
    private void CancelQuantize() => _quantize?.Cancel();
}
