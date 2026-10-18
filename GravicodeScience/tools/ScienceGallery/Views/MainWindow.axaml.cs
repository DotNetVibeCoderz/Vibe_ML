using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input.Platform;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Avalonia.Styling;
using AvaloniaEdit;
using AvaloniaEdit.Highlighting;
using ScienceGallery.Models;
using ScienceGallery.ViewModels;

namespace ScienceGallery.Views;

/// <summary>The gallery window.</summary>
public partial class MainWindow : Window
{
    private MainViewModel? _model;

    /// <summary>Creates the window.</summary>
    public MainWindow()
    {
        InitializeComponent();

        var code = this.FindControl<TextEditor>("CodeView")!;
        code.SyntaxHighlighting = HighlightingManager.Instance.GetDefinition("C#");
        ApplyEditorTheme();
        ActualThemeVariantChanged += (_, _) => ApplyEditorTheme();

        DataContextChanged += (_, _) => Attach(DataContext as MainViewModel);
        SizeChanged += (_, _) => FitColumns();
    }

    /// <summary>Selection colour comes from AvaloniaEdit, not the theme dictionaries, so set it per theme.</summary>
    private void ApplyEditorTheme()
    {
        var code = this.FindControl<TextEditor>("CodeView")!;
        var variant = Application.Current?.ActualThemeVariant ?? ThemeVariant.Light;
        if (Application.Current?.TryGetResource("EditorSelection", variant, out var brush) == true && brush is IBrush selection)
            code.TextArea.SelectionBrush = selection;
    }

    /// <summary>Every library, for the wordmark's fully lit strip.</summary>
    public GraviLibrary[] AllLibraries { get; } = Enum.GetValues<GraviLibrary>();

    private void Attach(MainViewModel? model)
    {
        if (_model is not null) _model.PropertyChanged -= OnModelChanged;
        _model = model;
        if (_model is not null) _model.PropertyChanged += OnModelChanged;
    }

    private void OnModelChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(MainViewModel.Detail)) return;
        var code = this.FindControl<TextEditor>("CodeView")!;
        code.Text = _model?.Detail?.Source ?? "";
        code.ScrollToHome();
        this.FindControl<ScrollViewer>("DetailScroll")!.Offset = default;
        this.FindControl<TextBlock>("CopyNotice")!.Text = "";
    }

    /// <summary>Three cards a row on a normal window, two when narrow, four when wide.</summary>
    private void FitColumns() => FitColumnsFor(Bounds.Width);

    /// <summary>Picks the card columns for a window <paramref name="width"/> wide.</summary>
    public void FitColumnsFor(double width)
    {
        var available = width - 272 - 88;
        var columns = available switch { < 760 => 2, < 1500 => 3, _ => 4 };
        foreach (var grid in this.GetVisualDescendantsOfType<UniformGrid>()) grid.Columns = columns;
    }

    private void OnThemeToggled(object? sender, RoutedEventArgs e)
    {
        if (Application.Current is null) return;
        Application.Current.RequestedThemeVariant = (sender as ToggleSwitch)?.IsChecked == true ? ThemeVariant.Dark : ThemeVariant.Light;
    }

    /// <summary>Switches the theme from code (used by the screenshot tour).</summary>
    public void SetDark(bool dark) => this.FindControl<ToggleSwitch>("ThemeSwitch")!.IsChecked = dark;

    private async void OnCopyCode(object? sender, RoutedEventArgs e)
    {
        if (_model?.Detail is not { } detail || Clipboard is null) return;
        await Clipboard.SetTextAsync(detail.Source);
        this.FindControl<TextBlock>("CopyNotice")!.Text = "Code copied";
    }

    private async void OnSaveChart(object? sender, RoutedEventArgs e)
    {
        if (_model?.Detail is not { ChartPng: { } png } detail) return;
        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save chart as PNG",
            SuggestedFileName = detail.Showcase.GetType().Name + ".png",
            FileTypeChoices = [new FilePickerFileType("PNG image") { Patterns = ["*.png"] }],
        });
        if (file is null) return;
        await using var stream = await file.OpenWriteAsync();
        await stream.WriteAsync(png);
        this.FindControl<TextBlock>("CopyNotice")!.Text = "Chart saved";
    }
}

/// <summary>Visual-tree helper without pulling in the whole extension namespace at call sites.</summary>
internal static class VisualTreeHelpers
{
    public static IEnumerable<T> GetVisualDescendantsOfType<T>(this Avalonia.Visual root) where T : Avalonia.Visual =>
        Avalonia.VisualTree.VisualExtensions.GetVisualDescendants(root).OfType<T>();
}

/// <summary>Turns "#RRGGBB" into a brush, for library colours bound from the model.</summary>
public sealed class HexBrushConverter : Avalonia.Data.Converters.IValueConverter
{
    /// <inheritdoc />
    public object? Convert(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo culture) =>
        value is string hex ? new SolidColorBrush(Color.Parse(hex)) : null;

    /// <inheritdoc />
    public object? ConvertBack(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo culture) =>
        throw new NotSupportedException();
}
