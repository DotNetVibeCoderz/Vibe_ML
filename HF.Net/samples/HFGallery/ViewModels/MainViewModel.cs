using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HFGallery.Cases;
using HFGallery.Controls;

namespace HFGallery.ViewModels;

/// <summary>One row in the rail.</summary>
public sealed partial class CaseItem : ObservableObject
{
    internal CaseItem(GalleryCase source) => Source = source;

    /// <summary>The case this row runs.</summary>
    public GalleryCase Source { get; }

    /// <summary>The case's name.</summary>
    public string Title => Source.Title;

    /// <summary>Which library it exercises.</summary>
    public string Library => Source.Library;

    /// <summary>What running it costs, as a word.</summary>
    public string Cost => Source.Cost switch
    {
        CaseCost.Instant => "instant",
        CaseCost.Light => "seconds",
        _ => "downloads a model",
    };
}

/// <summary>Drives the whole window.</summary>
public sealed partial class MainViewModel : ObservableObject
{
    private CancellationTokenSource? _running;

    /// <summary>Builds the view model with the full catalog and selects the first case.</summary>
    public MainViewModel()
    {
        Cases = [.. Catalog.All.Select(c => new CaseItem(c))];
        Selected = Cases[0];
    }

    /// <summary>Every case, in rail order.</summary>
    public ObservableCollection<CaseItem> Cases { get; }

    [ObservableProperty]
    private CaseItem _selected;

    [ObservableProperty]
    private string _input = "";

    [ObservableProperty]
    private string _second = "";

    [ObservableProperty]
    private string _status = "";

    [ObservableProperty]
    private bool _isBusy;

    [ObservableProperty]
    private CaseResult? _result;

    /// <summary>The bars, when the current result has any.</summary>
    public IReadOnlyList<Datum>? Bars => Result?.Bars;

    /// <summary>The scatter points, when the current result has any.</summary>
    public IReadOnlyList<Point2>? Scatter => Result?.Scatter;

    /// <summary>The curves, when the current result has any.</summary>
    public IReadOnlyList<Series>? Lines => Result?.Lines;

    /// <summary>The treemap slices, when the current result has any.</summary>
    public IReadOnlyList<Slice>? Treemap => Result?.Treemap;

    /// <summary>The marked passage, when the current result has one.</summary>
    public SpanText? Spans => Result?.Spans;

    /// <summary>The key-value table beside the visualisation.</summary>
    public IReadOnlyList<Fact>? Facts => Result?.Facts;

    /// <summary>The sentence above the visualisation.</summary>
    public string Summary => Result?.Summary ?? "";

    /// <summary>What the bar values are measured in.</summary>
    public string Unit => Result?.Unit ?? "%";

    /// <summary>What the line chart's horizontal axis counts.</summary>
    public string XLabel => Result?.XLabel ?? "";

    /// <summary>Category names for the legend, in palette order.</summary>
    public IReadOnlyList<string>? Legend => Result?.Legend;

    /// <summary>Whether the current case takes a second input.</summary>
    public bool HasSecond => Selected.Source.DefaultSecondInput is not null;

    /// <summary>Whether the current case takes any input at all.</summary>
    public bool HasInput => Selected.Source.DefaultInput is not null;

    partial void OnSelectedChanged(CaseItem value)
    {
        _running?.Cancel();

        Input = value.Source.DefaultInput ?? "";
        Second = value.Source.DefaultSecondInput ?? "";
        Result = null;
        Status = value.Source.Model.Length > 0 ? $"Model: {value.Source.Model}" : "No model needed.";

        OnPropertyChanged(nameof(HasSecond));
        OnPropertyChanged(nameof(HasInput));
    }

    partial void OnResultChanged(CaseResult? value)
    {
        foreach (var name in new[]
        {
            nameof(Bars), nameof(Scatter), nameof(Lines), nameof(Treemap),
            nameof(Spans), nameof(Facts), nameof(Summary), nameof(Unit), nameof(XLabel),
            nameof(Legend),
        })
        {
            OnPropertyChanged(name);
        }
    }

    /// <summary>Runs the selected case and shows whatever comes back.</summary>
    [RelayCommand]
    private async Task RunAsync()
    {
        // A second run while one is in flight cancels the first rather than queueing: the user
        // changed their mind about the input, and the older answer is no longer the one they want.
        _running?.Cancel();
        var cancellation = new CancellationTokenSource();
        _running = cancellation;

        IsBusy = true;
        Result = null;
        Status = "Running…";

        var progress = new Progress<string>(line =>
        {
            if (!cancellation.IsCancellationRequested) Status = line;
        });

        try
        {
            Result = await Selected.Source
                .RunAsync(Input, Second, progress, cancellation.Token)
                .ConfigureAwait(true);

            Status = Selected.Source.Model.Length > 0
                ? $"Done. Model: {Selected.Source.Model}"
                : "Done.";
        }
        catch (OperationCanceledException)
        {
            Status = "Cancelled.";
        }
        catch (Exception error)
        {
            // Surfaced rather than swallowed: a gallery whose case silently does nothing is
            // indistinguishable from a library that silently does nothing.
            Status = $"Failed: {error.Message}";
        }
        finally
        {
            if (ReferenceEquals(_running, cancellation)) IsBusy = false;
            cancellation.Dispose();
        }
    }
}
