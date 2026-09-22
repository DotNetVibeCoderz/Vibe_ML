using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace HFAppGen.Views;

/// <summary>
/// Turns a hex colour string into a brush.
/// </summary>
/// <remarks>
/// Needed because a resource key cannot be bound dynamically: <c>{DynamicResource {Binding …}}</c>
/// is not a thing, so per-item colours travel as hex strings and are converted here.
/// </remarks>
public sealed class HexBrushConverter : IValueConverter
{
    /// <summary>The shared instance referenced from XAML.</summary>
    public static HexBrushConverter Instance { get; } = new();

    /// <inheritdoc />
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not string hex || hex.Length == 0) return Brushes.Transparent;

        try { return new SolidColorBrush(Color.Parse(hex)); }
        catch (FormatException) { return Brushes.Transparent; }
    }

    /// <inheritdoc />
    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
