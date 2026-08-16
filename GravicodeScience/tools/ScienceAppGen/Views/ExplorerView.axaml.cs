using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using ScienceAppGen.ViewModels;

namespace ScienceAppGen.Views;

/// <summary>The project file tree.</summary>
public partial class ExplorerView : UserControl
{
    /// <summary>Creates the view.</summary>
    public ExplorerView() => AvaloniaXamlLoader.Load(this);

    private void OnRefresh(object? sender, RoutedEventArgs e)
        => (DataContext as ExplorerViewModel)?.Refresh();

    private void OnItemActivated(object? sender, RoutedEventArgs e)
    {
        if (DataContext is ExplorerViewModel model && model.Selected is { } node)
            model.Activate(node);
    }
}
