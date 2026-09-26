using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using MediaPipeNet.Gallery.Services;

namespace MediaPipeNet.Gallery.Controls;

/// <summary>Small factory helpers so views read like layout descriptions.</summary>
public static class Ui
{
    public static TextBlock Text(string text, string? classes = null, double? size = null)
    {
        var t = new TextBlock { Text = text, TextWrapping = TextWrapping.Wrap };
        if (classes is not null) foreach (var c in classes.Split(' ')) t.Classes.Add(c);
        if (size is { } s) t.FontSize = s;
        return t;
    }

    public static T With<T>(this T control, string classes) where T : Control
    {
        foreach (var c in classes.Split(' ', StringSplitOptions.RemoveEmptyEntries)) control.Classes.Add(c);
        return control;
    }

    public static Border Panel(Control child, string classes = "panel") => new Border { Child = child }.With(classes);

    public static Button Button(string text, string classes, EventHandler<Avalonia.Interactivity.RoutedEventArgs> onClick)
    {
        var b = new Button { Content = text }.With(classes);
        b.Click += onClick;
        return b;
    }

    public static StackPanel Stack(Orientation orientation, double spacing, params Control[] children)
    {
        var s = new StackPanel { Orientation = orientation, Spacing = spacing };
        s.Children.AddRange(children);
        return s;
    }

    /// <summary>The standard page header: mono eyebrow, display title, muted lede.</summary>
    public static StackPanel Header(string eyebrow, string title, string lede, double maxWidth = 760) =>
        Stack(Orientation.Vertical, 6,
            Text(eyebrow, "eyebrow"),
            Text(title, "display"),
            new TextBlock { Text = lede, MaxWidth = maxWidth, HorizontalAlignment = HorizontalAlignment.Left, Margin = new Thickness(0, 4, 0, 0) }.With("lede"));

    /// <summary>A labeled meter row used in result panels.</summary>
    public static Control Meter(ResultRow row)
    {
        var grid = new Grid { ColumnDefinitions = new ColumnDefinitions("*,Auto"), Margin = new Thickness(0, 3) };
        grid.Children.Add(Text(row.Label));
        var value = Text(row.Value, "mono");
        value.HorizontalAlignment = HorizontalAlignment.Right;
        Grid.SetColumn(value, 1);
        grid.Children.Add(value);
        if (row.Fraction is not { } f) return grid;
        var track = new Border { Height = 4, CornerRadius = new CornerRadius(2), Margin = new Thickness(0, 4, 0, 2) };
        track.Bind(Border.BackgroundProperty, track.GetResourceObservable("RuleBrush"));
        var fill = new Border { Height = 4, CornerRadius = new CornerRadius(2), HorizontalAlignment = HorizontalAlignment.Left };
        fill.Bind(Border.BackgroundProperty, fill.GetResourceObservable("CobaltBrush"));
        var host = new Grid();
        host.Children.Add(track);
        host.Children.Add(fill);
        host.SizeChanged += (_, e) => fill.Width = Math.Clamp(f, 0, 1) * e.NewSize.Width;
        return Stack(Orientation.Vertical, 0, grid, host);
    }
}
