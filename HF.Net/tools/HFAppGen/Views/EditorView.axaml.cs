using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Media;
using AvaloniaEdit;
using AvaloniaEdit.Highlighting;
using HFAppGen.Services;
using HFAppGen.ViewModels;

namespace HFAppGen.Views;

/// <summary>
/// The code editor.
/// </summary>
/// <remarks>
/// AvaloniaEdit owns its own document, so text is synchronised explicitly rather than bound.
/// A re-entrancy guard keeps the two from ping-ponging when either side changes.
/// </remarks>
public partial class EditorView : UserControl
{
    private TextEditor? _editor;
    private EditorViewModel? _model;
    private OpenDocument? _bound;
    private bool _updating;

    /// <summary>Creates the view.</summary>
    public EditorView()
    {
        AvaloniaXamlLoader.Load(this);
        _editor = this.FindControl<TextEditor>("Editor");

        if (_editor is not null)
        {
            _editor.TextChanged += OnEditorTextChanged;
            _editor.TextArea.Caret.PositionChanged += OnCaretMoved;
            _editor.Options.IndentationSize = 4;
            _editor.Options.ConvertTabsToSpaces = true;
            _editor.Options.HighlightCurrentLine = true;
        }

        // The theme is fixed at startup from app.config, but ActualThemeVariantChanged also
        // covers a settings change, and the editor's own brushes have to follow it.
        ApplyTheme();
        ActualThemeVariantChanged += (_, _) => ApplyTheme();

        DataContextChanged += OnDataContextChanged;
    }

    /// <summary>
    /// Repaints the syntax colours and the editor's own decorations for the current theme.
    /// </summary>
    /// <remarks>
    /// Selection and current-line come from AvaloniaEdit rather than from the control template,
    /// so they do not pick up the theme dictionaries and have to be set here alongside the
    /// highlighting palette.
    /// </remarks>
    private void ApplyTheme()
    {
        SyntaxTheme.Apply(ActualThemeVariant);

        if (_editor is null) return;

        var view = _editor.TextArea.TextView;
        view.CurrentLineBackground = Brush("EditorCurrentLine");
        view.CurrentLineBorder = new Pen(Brush("EditorCurrentLine"));
        _editor.TextArea.SelectionBrush = Brush("EditorSelection");
        _editor.TextArea.SelectionBorder = null;

        // A repaint is needed because the definition objects changed in place; the editor has
        // no way to know its colours are no longer the ones it drew with.
        view.Redraw();
    }

    private IBrush? Brush(string key) =>
        this.TryFindResource(key, ActualThemeVariant, out var value) ? value as IBrush : null;

    /// <summary>The underlying editor control, for the shell's go-to-line command.</summary>
    public TextEditor? Control => _editor;

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        if (_model is not null) _model.PropertyChanged -= OnModelPropertyChanged;

        _model = DataContext as EditorViewModel;
        if (_model is not null) _model.PropertyChanged += OnModelPropertyChanged;

        BindActiveDocument();
    }

    private void OnModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(EditorViewModel.Active)) BindActiveDocument();
        else if (e.PropertyName is nameof(EditorViewModel.ShowLineNumbers) && _editor is not null && _model is not null)
            _editor.ShowLineNumbers = _model.ShowLineNumbers;
    }

    private void BindActiveDocument()
    {
        if (_editor is null || _model is null) return;

        _bound = _model.Active;
        _updating = true;

        if (_bound is null)
        {
            _editor.Text = "";
            _editor.IsVisible = false;
        }
        else
        {
            _editor.IsVisible = true;
            _editor.Text = _bound.Text;
            _editor.SyntaxHighlighting = _bound.SyntaxName is null
                ? null
                : HighlightingManager.Instance.GetDefinition(_bound.SyntaxName);
        }

        _updating = false;
    }

    private void OnEditorTextChanged(object? sender, EventArgs e)
    {
        if (_updating || _editor is null || _bound is null) return;

        _bound.Text = _editor.Text;
        _bound.IsModified = true;
    }

    private void OnCaretMoved(object? sender, EventArgs e)
    {
        if (_editor is null || _model is null) return;

        _model.CaretLine = _editor.TextArea.Caret.Line;
        _model.CaretColumn = _editor.TextArea.Caret.Column;
    }

    private void OnTabPressed(object? sender, PointerPressedEventArgs e)
    {
        if (sender is Control { DataContext: OpenDocument document } && _model is not null)
            _model.Active = document;
    }

    private void OnCloseTab(object? sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: OpenDocument document } && _model is not null)
            _model.Close(document);

        e.Handled = true;
    }

    /// <summary>Re-reads the active document, after something changed it outside the editor.</summary>
    public void ReloadFromModel()
    {
        if (_editor is null || _bound is null) return;

        var caret = _editor.TextArea.Caret.Line;
        _updating = true;
        _editor.Text = _bound.Text;
        _updating = false;

        _editor.TextArea.Caret.Line = Math.Clamp(caret, 1, Math.Max(1, _editor.Document.LineCount));
    }

    /// <summary>Moves the caret to a line and scrolls it into view.</summary>
    public void GoToLine(int line)
    {
        if (_editor is null) return;

        var target = Math.Clamp(line, 1, Math.Max(1, _editor.Document.LineCount));
        _editor.TextArea.Caret.Line = target;
        _editor.TextArea.Caret.Column = 1;
        _editor.ScrollToLine(target);
        _editor.Focus();
    }
}
