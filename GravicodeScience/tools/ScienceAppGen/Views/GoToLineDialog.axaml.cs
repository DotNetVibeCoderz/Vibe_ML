using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;

namespace ScienceAppGen.Views;

/// <summary>Asks for a line number.</summary>
public partial class GoToLineDialog : Window
{
    /// <summary>Creates the dialog.</summary>
    public GoToLineDialog()
    {
        AvaloniaXamlLoader.Load(this);
        Opened += (_, _) => this.FindControl<TextBox>("LineBox")?.Focus();
    }

    private void OnKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter) { e.Handled = true; Accept(); }
        else if (e.Key == Key.Escape) { e.Handled = true; Close(null); }
    }

    private void OnGo(object? sender, RoutedEventArgs e) => Accept();

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(null);

    private void Accept()
    {
        var text = this.FindControl<TextBox>("LineBox")?.Text;
        Close(int.TryParse(text, out var line) && line > 0 ? line : null);
    }
}
