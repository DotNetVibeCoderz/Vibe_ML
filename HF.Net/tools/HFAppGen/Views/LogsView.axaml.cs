using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
using HFAppGen.Services;

namespace HFAppGen.Views;

/// <summary>The build and tool output panel.</summary>
public partial class LogsView : UserControl
{
    private LogService? _logs;

    /// <summary>Creates the view.</summary>
    public LogsView()
    {
        AvaloniaXamlLoader.Load(this);
        DataContextChanged += OnDataContextChanged;
    }

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        if (_logs is not null) _logs.EntryAdded -= OnEntryAdded;

        _logs = DataContext as LogService;
        if (_logs is not null) _logs.EntryAdded += OnEntryAdded;
    }

    private void OnEntryAdded(LogEntry entry)
    {
        // Scrolling has to wait for the item container to exist.
        Dispatcher.UIThread.Post(
            () => this.FindControl<ScrollViewer>("Scroller")?.ScrollToEnd(),
            DispatcherPriority.Background);
    }

    private void OnClear(object? sender, RoutedEventArgs e) => _logs?.Clear();
}
