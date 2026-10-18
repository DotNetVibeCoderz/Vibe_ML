using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using ScienceGallery.Models;

namespace ScienceGallery.Controls;

/// <summary>
/// The gallery's signature: a showcase's libraries drawn as emission lines on a spectrum.
/// </summary>
/// <remarks>
/// <para>
/// Each Gravi library is a real emission line (its NuGet icon colour), placed at its wavelength
/// between 380 and 680 nm the way a spectroscope would show it. Lines a showcase uses are lit; the
/// rest stay as faint ticks, so a strip shows at a glance both what a use case needs and what it
/// does not.
/// </para>
/// <para>
/// SciNet is the instrument rather than a line, so when it is used the whole band glows with a
/// faint continuum behind the lines.
/// </para>
/// </remarks>
public sealed class SpectrumStrip : Control
{
    private const double Low = 380, High = 680;

    /// <summary>The libraries to light.</summary>
    public static readonly StyledProperty<IEnumerable<GraviLibrary>?> LibrariesProperty =
        AvaloniaProperty.Register<SpectrumStrip, IEnumerable<GraviLibrary>?>(nameof(Libraries));

    /// <summary>The ground behind the lines.</summary>
    public static readonly StyledProperty<IBrush?> GroundProperty =
        AvaloniaProperty.Register<SpectrumStrip, IBrush?>(nameof(Ground), new SolidColorBrush(Color.Parse("#16132E")));

    /// <summary>Colour of unlit lines.</summary>
    public static readonly StyledProperty<IBrush?> UnlitProperty =
        AvaloniaProperty.Register<SpectrumStrip, IBrush?>(nameof(Unlit), new SolidColorBrush(Color.Parse("#3A3660")));

    static SpectrumStrip()
    {
        AffectsRender<SpectrumStrip>(LibrariesProperty, GroundProperty, UnlitProperty);
    }

    /// <inheritdoc cref="LibrariesProperty"/>
    public IEnumerable<GraviLibrary>? Libraries
    {
        get => GetValue(LibrariesProperty);
        set => SetValue(LibrariesProperty, value);
    }

    /// <inheritdoc cref="GroundProperty"/>
    public IBrush? Ground
    {
        get => GetValue(GroundProperty);
        set => SetValue(GroundProperty, value);
    }

    /// <inheritdoc cref="UnlitProperty"/>
    public IBrush? Unlit
    {
        get => GetValue(UnlitProperty);
        set => SetValue(UnlitProperty, value);
    }

    /// <summary>Horizontal position of a wavelength inside a strip of <paramref name="width"/>.</summary>
    public static double XOf(double wavelength, double width, double inset) =>
        inset + (wavelength - Low) / (High - Low) * (width - 2 * inset);

    /// <inheritdoc />
    public override void Render(DrawingContext context)
    {
        var bounds = new Rect(Bounds.Size);
        if (bounds.Width < 4 || bounds.Height < 2) return;

        var radius = Math.Min(bounds.Height / 2, 6);
        context.DrawRectangle(Ground, null, bounds, radius, radius);

        var lit = new HashSet<GraviLibrary>(Libraries ?? []);
        var inset = Math.Max(6, bounds.Height * 0.4);

        if (lit.Contains(GraviLibrary.SciNet))
        {
            var continuum = new LinearGradientBrush
            {
                StartPoint = new RelativePoint(0, 0, RelativeUnit.Relative),
                EndPoint = new RelativePoint(1, 0, RelativeUnit.Relative),
                Opacity = 0.42,
            };
            // Violet to red: the continuum a white-light source throws, behind the discrete lines.
            foreach (var (offset, hex) in new[] { (0.0, "#7B2FBF"), (0.2, "#3E4FD8"), (0.38, "#1B95BC"),
                                                   (0.55, "#2E9E68"), (0.7, "#E8C317"), (0.85, "#E8761F"), (1.0, "#D9363E") })
                continuum.GradientStops.Add(new GradientStop(Color.Parse(hex), offset));
            var band = bounds.Deflate(new Thickness(inset * 0.6, bounds.Height * 0.3));
            context.DrawRectangle(continuum, null, band, 2, 2);
        }

        foreach (var info in LibraryInfo.All)
        {
            if (info.Wavelength is not { } wavelength) continue;
            var x = XOf(wavelength, bounds.Width, inset);
            var on = lit.Contains(info.Library);
            var colour = Color.Parse(info.Hex);

            if (on)
            {
                // A soft halo first, then the line: a lit line reads as light, not as ink.
                var halo = new SolidColorBrush(colour, 0.32);
                var haloWidth = Math.Max(5, bounds.Height * 0.35);
                context.DrawRectangle(halo, null, new Rect(x - haloWidth / 2, 2, haloWidth, bounds.Height - 4), 2, 2);
                var lineWidth = Math.Max(2.2, bounds.Height * 0.11);
                context.DrawRectangle(new SolidColorBrush(Lighten(colour)), null,
                    new Rect(x - lineWidth / 2, 1.5, lineWidth, bounds.Height - 3), 1, 1);
            }
            else
            {
                var tick = Math.Max(1, bounds.Height * 0.05);
                var h = bounds.Height * 0.38;
                context.DrawRectangle(Unlit, null, new Rect(x - tick / 2, (bounds.Height - h) / 2, tick, h));
            }
        }
    }

    /// <summary>Lit lines are drawn a little brighter than the icon colour, as emitted light.</summary>
    private static Color Lighten(Color c) => Color.FromRgb(
        (byte)(c.R + (255 - c.R) * 0.18), (byte)(c.G + (255 - c.G) * 0.18), (byte)(c.B + (255 - c.B) * 0.18));
}
