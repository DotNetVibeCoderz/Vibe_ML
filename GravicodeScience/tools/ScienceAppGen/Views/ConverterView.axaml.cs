using System.ComponentModel;
using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using AvaloniaEdit;
using AvaloniaEdit.Highlighting;
using ScienceAppGen.Services;
using ScienceAppGen.ViewModels;

namespace ScienceAppGen.Views;

/// <summary>The Python to C# page.</summary>
public partial class ConverterView : UserControl
{
    private readonly TextEditor _python;
    private readonly TextEditor _csharp;
    private readonly ScrollViewer _activity;
    private ConverterViewModel? _model;
    private ConvertedModuleItem? _shown;

    /// <summary>Creates the page.</summary>
    public ConverterView()
    {
        AvaloniaXamlLoader.Load(this);
        _python = this.FindControl<TextEditor>("PythonEditor")!;
        _csharp = this.FindControl<TextEditor>("CSharpEditor")!;
        _activity = this.FindControl<ScrollViewer>("ActivityScroll")!;

        _python.SyntaxHighlighting = HighlightingManager.Instance.GetDefinition("Python");
        _csharp.SyntaxHighlighting = HighlightingManager.Instance.GetDefinition("C#");

        ApplyTheme();
        ActualThemeVariantChanged += (_, _) => ApplyTheme();
        DataContextChanged += OnDataContextChanged;
    }

    private ShellViewModel? Shell => TopLevel.GetTopLevel(this)?.DataContext as ShellViewModel;

    /// <summary>Same treatment as the main editor: the palette and AvaloniaEdit's own brushes follow the theme.</summary>
    private void ApplyTheme()
    {
        SyntaxTheme.Apply(ActualThemeVariant);
        foreach (var editor in new[] { _python, _csharp })
        {
            var view = editor.TextArea.TextView;
            view.CurrentLineBackground = Brush("EditorCurrentLine");
            view.CurrentLineBorder = new Pen(Brush("EditorCurrentLine"));
            editor.TextArea.SelectionBrush = Brush("EditorSelection");
            editor.TextArea.SelectionBorder = null;
            view.Redraw();
        }
    }

    private IBrush? Brush(string key) =>
        this.TryFindResource(key, ActualThemeVariant, out var value) ? value as IBrush : null;

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        if (_model is not null)
        {
            _model.PropertyChanged -= OnModelChanged;
            _model.Activity.CollectionChanged -= OnActivityChanged;
        }

        _model = DataContext as ConverterViewModel;
        if (_model is null) return;

        _model.PropertyChanged += OnModelChanged;
        _model.Activity.CollectionChanged += OnActivityChanged;
        ShowModule(_model.SelectedModule);
    }

    private void OnModelChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(ConverterViewModel.SelectedModule)) ShowModule(_model?.SelectedModule);
    }

    /// <summary>Puts a module's Python and C# into the two editors and follows its C# as it changes.</summary>
    private void ShowModule(ConvertedModuleItem? module)
    {
        if (_shown is not null) _shown.PropertyChanged -= OnModuleChanged;
        _shown = module;

        _python.Text = module?.PythonSource ?? "";
        _csharp.Text = module?.CSharp ?? "";
        _python.ScrollToHome();
        _csharp.ScrollToHome();

        if (module is not null) module.PropertyChanged += OnModuleChanged;
    }

    private void OnModuleChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(ConvertedModuleItem.CSharp) && sender == _shown)
            _csharp.Text = _shown?.CSharp ?? "";
    }

    private void OnActivityChanged(object? sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
        => _activity.ScrollToEnd();

    // ---------------------------------------------------------------- source and output

    private async void OnChooseFile(object? sender, RoutedEventArgs e)
    {
        var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
        if (storage is null || _model is null) return;

        var files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Choose a Python script or notebook",
            AllowMultiple = false,
            FileTypeFilter =
            [
                new FilePickerFileType("Python") { Patterns = ["*.py", "*.ipynb"] },
            ],
        });
        if (files.Count > 0) _model.LoadSource(files[0].Path.LocalPath);
    }

    private async void OnChooseFolder(object? sender, RoutedEventArgs e)
    {
        var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
        if (storage is null || _model is null) return;

        var folders = await storage.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Choose a Python project folder",
            AllowMultiple = false,
        });
        if (folders.Count > 0) _model.LoadSource(folders[0].Path.LocalPath);
    }

    private async void OnChooseOutput(object? sender, RoutedEventArgs e)
    {
        var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
        if (storage is null || _model is null) return;

        var folders = await storage.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Where should the C# project go?",
            AllowMultiple = false,
        });
        if (folders.Count > 0) _model.OutputDirectory = folders[0].Path.LocalPath;
    }

    // ---------------------------------------------------------------- actions

    private async void OnConvert(object? sender, RoutedEventArgs e)
    {
        if (_model is null) return;
        Shell?.Logs.Info("convert", $"Converting {_model.SourcePath}");
        await _model.ConvertAsync();
    }

    private void OnCancel(object? sender, RoutedEventArgs e) => _model?.Cancel();

    private void OnOpenResult(object? sender, RoutedEventArgs e) => _model?.OpenResult();

    private void OnShowFolder(object? sender, RoutedEventArgs e)
    {
        var folder = _model?.OutputDirectory.Trim();
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder)) return;
        try
        {
            Process.Start(new ProcessStartInfo { FileName = folder, UseShellExecute = true });
        }
        catch (Exception ex)
        {
            Shell?.Logs.Warning("convert", $"Could not open the folder: {ex.Message}");
        }
    }

    private void OnBack(object? sender, RoutedEventArgs e)
    {
        if (Shell is { } shell) shell.IsConverterOpen = false;
    }
}
