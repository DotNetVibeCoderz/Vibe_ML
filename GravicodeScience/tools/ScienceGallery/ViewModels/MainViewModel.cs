using System.Collections.ObjectModel;
using System.Diagnostics;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery.ViewModels;

/// <summary>The whole gallery: filters, the visible cards, and the open use case.</summary>
public sealed partial class MainViewModel : ObservableObject
{
    private readonly List<ShowcaseCard> _cards;

    /// <summary>Builds the gallery from every showcase in the assembly.</summary>
    public MainViewModel()
    {
        _cards = ShowcaseRegistry.All.Select(s => new ShowcaseCard(s)).ToList();

        Domains =
        [
            new DomainFilter(null, "All use cases", _cards.Count),
            .. DomainInfo.All.Select(d => new DomainFilter(d, d.Name, _cards.Count(c => c.Showcase.Domain == d.Domain))),
        ];
        Libraries = LibraryInfo.All.Select(l => new LibraryFilter(l, _cards.Count(c => c.Showcase.Libraries.Contains(l.Library)), Refilter)).ToList();
        _selectedDomain = Domains[0];
        Refilter();
    }

    /// <summary>Domains for the sidebar, "All use cases" first.</summary>
    public IReadOnlyList<DomainFilter> Domains { get; }

    /// <summary>Library toggles for the sidebar.</summary>
    public IReadOnlyList<LibraryFilter> Libraries { get; }

    /// <summary>The cards that pass every filter.</summary>
    public ObservableCollection<ShowcaseCard> Visible { get; } = [];

    /// <summary>How many use cases exist, for the header.</summary>
    public int TotalCount => _cards.Count;

    [ObservableProperty]
    private DomainFilter _selectedDomain;

    [ObservableProperty]
    private string _searchText = "";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsDetailOpen), nameof(IsBrowsing))]
    private ShowcaseDetail? _detail;

    /// <summary>True while a use case is open.</summary>
    public bool IsDetailOpen => Detail is not null;

    /// <summary>True while the card grid is showing.</summary>
    public bool IsBrowsing => Detail is null;

    /// <summary>What the grid is showing, in words.</summary>
    public string ResultsCaption { get; private set; } = "";

    /// <summary>Shown when nothing matches.</summary>
    public bool IsEmpty => Visible.Count == 0;

    partial void OnSelectedDomainChanged(DomainFilter value) => Refilter();

    partial void OnSearchTextChanged(string value) => Refilter();

    /// <summary>Opens a use case and runs it.</summary>
    [RelayCommand]
    private async Task Open(ShowcaseCard card)
    {
        var detail = new ShowcaseDetail(card.Showcase);
        Detail = detail;
        await detail.RunCommand.ExecuteAsync(null);
    }

    /// <summary>Goes back to the grid.</summary>
    [RelayCommand]
    private void Close() => Detail = null;

    /// <summary>Clears every filter.</summary>
    [RelayCommand]
    private void ClearFilters()
    {
        foreach (var library in Libraries) library.SetSilently(false);
        SearchText = "";
        SelectedDomain = Domains[0];
        Refilter();
    }

    private void Refilter()
    {
        var required = Libraries.Where(l => l.IsChecked).Select(l => l.Info.Library).ToList();
        var terms = SearchText.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        var matches = _cards.Where(c =>
            (SelectedDomain?.Info is null || c.Showcase.Domain == SelectedDomain.Info.Domain) &&
            required.All(r => c.Showcase.Libraries.Contains(r)) &&
            terms.All(t => c.SearchText.Contains(t, StringComparison.OrdinalIgnoreCase)));

        Visible.Clear();
        foreach (var card in matches) Visible.Add(card);

        var scope = SelectedDomain?.Info?.Name ?? "all domains";
        var libraries = required.Count == 0 ? "" : $", using {string.Join(" + ", required)}";
        ResultsCaption = $"{Visible.Count} of {_cards.Count} use cases · {scope}{libraries}";
        OnPropertyChanged(nameof(ResultsCaption));
        OnPropertyChanged(nameof(IsEmpty));
    }
}

/// <summary>A sidebar domain entry. <see cref="Info"/> is null for "All use cases".</summary>
public sealed record DomainFilter(DomainInfo? Info, string Name, int Count)
{
    /// <summary>The count, as shown.</summary>
    public string CountText => Count.ToString();
}

/// <summary>A sidebar library toggle.</summary>
public sealed partial class LibraryFilter : ObservableObject
{
    private readonly Action _changed;
    private bool _silent;

    /// <summary>Creates the toggle.</summary>
    public LibraryFilter(LibraryInfo info, int count, Action changed)
    {
        Info = info;
        Count = count;
        _changed = changed;
    }

    /// <summary>The library.</summary>
    public LibraryInfo Info { get; }

    /// <summary>How many use cases use it.</summary>
    public int Count { get; }

    /// <summary>"pandas", "SciPy", ...</summary>
    public string Analogue => Info.Analogue;

    /// <summary>The single library this toggle stands for, for its own spectrum swatch.</summary>
    public GraviLibrary[] Self => [Info.Library];

    [ObservableProperty]
    private bool _isChecked;

    partial void OnIsCheckedChanged(bool value)
    {
        if (!_silent) _changed();
    }

    /// <summary>Changes the state without refiltering (used when clearing every filter at once).</summary>
    public void SetSilently(bool value)
    {
        _silent = true;
        IsChecked = value;
        _silent = false;
    }
}

/// <summary>A card in the grid.</summary>
public sealed class ShowcaseCard(Showcase showcase)
{
    /// <summary>The use case.</summary>
    public Showcase Showcase { get; } = showcase;

    /// <summary>Its title.</summary>
    public string Title => Showcase.Title;

    /// <summary>Its one-line summary.</summary>
    public string Summary => Showcase.Summary;

    /// <summary>The domain eyebrow.</summary>
    public string DomainName => DomainInfo.Of(Showcase.Domain).Name.ToUpperInvariant();

    /// <summary>The libraries it uses.</summary>
    public GraviLibrary[] Libraries => Showcase.Libraries;

    /// <summary>The libraries, in words.</summary>
    public string LibraryNames => string.Join(" · ", Showcase.Libraries);

    /// <summary>Everything the search box looks at.</summary>
    public string SearchText { get; } =
        $"{showcase.Title} {showcase.Summary} {showcase.Scenario} {DomainInfo.Of(showcase.Domain).Name} {string.Join(' ', showcase.Libraries)}";
}

/// <summary>An open use case: its source, and the result of running it.</summary>
public sealed partial class ShowcaseDetail(Showcase showcase) : ObservableObject
{
    /// <summary>The use case.</summary>
    public Showcase Showcase { get; } = showcase;

    /// <summary>The verbatim source.</summary>
    public string Source { get; } = ShowcaseRegistry.SourceOf(showcase);

    /// <summary>The title.</summary>
    public string Title => Showcase.Title;

    /// <summary>Who does this, and why.</summary>
    public string Scenario => Showcase.Scenario;

    /// <summary>The domain eyebrow.</summary>
    public string DomainName => DomainInfo.Of(Showcase.Domain).Name.ToUpperInvariant();

    /// <summary>The libraries it uses.</summary>
    public GraviLibrary[] Libraries => Showcase.Libraries;

    /// <summary>Library names with their Python analogues, for the header.</summary>
    public string LibraryLine => string.Join("   ", Showcase.Libraries.Select(l => $"{l} ({LibraryInfo.Of(l).Analogue})"));

    /// <summary>The source file name, for the code panel.</summary>
    public string SourceName => "Showcases/" + Showcase.SourceName;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasChart))]
    private Bitmap? _chart;

    /// <summary>The PNG bytes behind <see cref="Chart"/>, for "Save chart".</summary>
    public byte[]? ChartPng { get; private set; }

    /// <summary>True when the run produced a chart.</summary>
    public bool HasChart => Chart is not null;

    /// <summary>The key numbers.</summary>
    public ObservableCollection<MetricItem> Metrics { get; } = [];

    /// <summary>Findings, as text.</summary>
    public ObservableCollection<string> Lines { get; } = [];

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError))]
    private string? _error;

    /// <summary>True when the last run failed.</summary>
    public bool HasError => Error is not null;

    [ObservableProperty]
    private bool _isRunning;

    [ObservableProperty]
    private string _status = "Not run yet";

    /// <summary>Runs the use case off the UI thread and shows what it produced.</summary>
    [RelayCommand]
    private async Task Run()
    {
        if (IsRunning) return;
        IsRunning = true;
        Error = null;
        Status = "Running…";

        try
        {
            var output = new ShowcaseOutput();
            var watch = Stopwatch.StartNew();
            await Task.Run(() => Showcase.Run(output));
            watch.Stop();

            var png = output.Figure is { } figure ? await Task.Run(() => figure.ToPng(110)) : null;

            Dispatcher.UIThread.VerifyAccess();
            Metrics.Clear();
            foreach (var (label, value, note) in output.Metrics) Metrics.Add(new MetricItem(label, value, note));
            Lines.Clear();
            foreach (var line in output.Lines) Lines.Add(line);
            ChartPng = png;
            Chart = png is null ? null : new Bitmap(new MemoryStream(png));
            Status = $"Ran in {FormatTime(watch.Elapsed)} on this machine";
        }
        catch (Exception exception)
        {
            var inner = exception is AggregateException { InnerException: { } first } ? first : exception;
            Error = $"{inner.GetType().Name}: {inner.Message}";
            Status = "The run failed";
        }
        finally
        {
            IsRunning = false;
        }
    }

    private static string FormatTime(TimeSpan elapsed) => elapsed.TotalMilliseconds < 1000
        ? $"{elapsed.TotalMilliseconds:F0} ms"
        : $"{elapsed.TotalSeconds:F2} s";
}

/// <summary>A key number from a run.</summary>
public sealed record MetricItem(string Label, string Value, string? Note)
{
    /// <summary>True when there is a note to show.</summary>
    public bool HasNote => !string.IsNullOrEmpty(Note);
}
