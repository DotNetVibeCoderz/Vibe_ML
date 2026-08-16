using System.Globalization;
using Avalonia.Controls;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace LocalGen.Desktop.ViewModels;

/// <summary>
/// Value converters used by the shell.
/// </summary>
/// <remarks>
/// The LED's appearance is driven by style classes rather than by binding a brush, so that the
/// pulse animation can be attached to the <c>busy</c> class in the stylesheet instead of being
/// re-implemented in code.
/// </remarks>
public static class Converters
{
    public static readonly IValueConverter EqualsRunning = new StateConverter("running");
    public static readonly IValueConverter EqualsBusy = new StateConverter("busy");
    public static readonly IValueConverter EqualsStopped = new StateConverter("stopped");
    public static readonly IValueConverter EqualsFaulted = new StateConverter("faulted");

    /// <summary>Shows the sun when dark is active, since the control switches to light.</summary>
    public static readonly IValueConverter ThemeGlyph = new FuncValueConverter<bool, Geometry?>(
        isDark => Geometry.Parse(isDark ? Icons.Sun : Icons.Moon));

    public static readonly IValueConverter BytesToSize =
        new FuncValueConverter<long, string>(FormatBytes);

    public static readonly IValueConverter InvertBoolean =
        new FuncValueConverter<bool, bool>(static value => !value);

    public static readonly IValueConverter IsNotEmpty =
        new FuncValueConverter<string?, bool>(static value => !string.IsNullOrWhiteSpace(value));

    /// <summary>The user's own turns sit on the right, the way a conversation reads.</summary>
    public static readonly IValueConverter UserAlignment =
        new FuncValueConverter<bool, Avalonia.Layout.HorizontalAlignment>(
            static isUser => isUser
                ? Avalonia.Layout.HorizontalAlignment.Right
                : Avalonia.Layout.HorizontalAlignment.Left);

    /// <summary>
    /// Transcript colouring by turn kind. Tool calls get the signal wash because they are the
    /// machine acting on the user's behalf — the thing a local playground exists to make visible.
    /// </summary>
    public static readonly IValueConverter EntryBackground =
        new ThemeBrushConverter(kind => kind switch
        {
            "user" => "SurfaceOverlay",
            "tool" => "SignalWash",
            "error" => "AlertWash",
            _ => "SurfaceRaised"
        });

    public static readonly IValueConverter EntryBorder =
        new ThemeBrushConverter(kind => kind switch
        {
            "tool" => "SignalDim",
            "error" => "Alert",
            _ => "Edge"
        });

    public static readonly IValueConverter EntryForeground =
        new ThemeBrushConverter(kind => kind switch
        {
            "error" => "Alert",
            "tool" => "TextSecondary",
            _ => "TextPrimary"
        });

    /// <summary>Tool output is machine text, so it takes the mono face.</summary>
    public static readonly IValueConverter EntryFont =
        new FuncValueConverter<bool, FontFamily>(static isTool => isTool
            ? new FontFamily("Cascadia Mono,Consolas,JetBrains Mono,SF Mono,Menlo,DejaVu Sans Mono,monospace")
            : new FontFamily("avares://Avalonia.Fonts.Inter/Assets#Inter"));

    public static readonly IValueConverter EntryFontSize =
        new FuncValueConverter<bool, double>(static isTool => isTool ? 11.5 : 13.5);

    /// <summary>Marks a staged attachment as a picture or a page, at a glance.</summary>
    public static readonly IValueConverter AttachmentGlyph =
        new FuncValueConverter<bool, string>(static isImage => isImage ? "🖼" : "▤");

    /// <summary>
    /// HTTP method colouring in the API tester. Reading is neutral; anything that writes or
    /// deletes carries the signal colour, so a destructive request is not one careless click away.
    /// </summary>
    public static readonly IValueConverter MethodForeground =
        new ThemeBrushConverter(method => method switch
        {
            "GET" => "Data",
            "DELETE" => "Alert",
            _ => "Signal"
        });

    public static readonly IValueConverter MethodBorder =
        new ThemeBrushConverter(method => method switch
        {
            "GET" => "DataDim",
            "DELETE" => "Alert",
            _ => "SignalDim"
        });

    public static readonly IValueConverter StatusForeground =
        new ThemeBrushConverter(failed =>
            string.Equals(failed, bool.TrueString, StringComparison.OrdinalIgnoreCase)
                ? "Alert"
                : "Success");

    /// <summary>Highlights the selected engine panel without adding a second visual device.</summary>
    public static readonly IValueConverter SelectionBorder =
        new ThemeBrushConverter(selected =>
            string.Equals(selected, bool.TrueString, StringComparison.OrdinalIgnoreCase)
                ? "SignalDim"
                : "Edge");

    public static string FormatBytes(long bytes) => bytes switch
    {
        <= 0 => "—",
        < 1024 => $"{bytes} B",
        < 1024L * 1024 => $"{bytes / 1024.0:N0} KB",
        < 1024L * 1024 * 1024 => $"{bytes / (1024.0 * 1024):N0} MB",
        _ => $"{bytes / (1024.0 * 1024 * 1024):N2} GB"
    };

    /// <summary>
    /// Resolves a theme brush by resource key so transcript colours follow the active theme
    /// instead of being frozen at the value they had when the control was created.
    /// </summary>
    private sealed class ThemeBrushConverter(Func<string, string> selectKey) : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            // ToString rather than a cast: callers pass both the entry kind and a bool.
            var key = selectKey(value?.ToString() ?? string.Empty);

            return Avalonia.Application.Current?.TryFindResource(
                key,
                Avalonia.Application.Current.ActualThemeVariant,
                out var brush) == true
                ? brush
                : null;
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
            throw new NotSupportedException();
    }

    private sealed class StateConverter(string expected) : IValueConverter
    {
        public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
            string.Equals(value as string, expected, StringComparison.OrdinalIgnoreCase);

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
            throw new NotSupportedException();
    }
}
