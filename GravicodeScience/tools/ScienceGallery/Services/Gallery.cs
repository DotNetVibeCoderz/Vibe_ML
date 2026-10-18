using System.Reflection;
using Matplotlib.Net;
using Matplotlib.Net.Styling;
using ScienceGallery.Models;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Services;

/// <summary>Finds every showcase in the assembly and reads their source files back.</summary>
public static class ShowcaseRegistry
{
    private static readonly Lazy<IReadOnlyList<Showcase>> _all = new(() =>
        typeof(Showcase).Assembly.GetTypes()
            .Where(t => t is { IsAbstract: false, IsClass: true } && typeof(Showcase).IsAssignableFrom(t))
            .Select(t => (Showcase)Activator.CreateInstance(t)!)
            .OrderBy(s => s.Domain).ThenBy(s => s.Title, StringComparer.Ordinal)
            .ToList());

    /// <summary>Every showcase, grouped by domain then sorted by title.</summary>
    public static IReadOnlyList<Showcase> All => _all.Value;

    /// <summary>The verbatim source of <paramref name="showcase"/>, as compiled into the app.</summary>
    public static string SourceOf(Showcase showcase)
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("Showcase:" + showcase.SourceName);
        if (stream is null) return $"// Source for {showcase.SourceName} was not embedded.";
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd().Replace("\r\n", "\n");
    }
}

/// <summary>Locates the repository's <c>datasets/</c> folder for showcases that use real data.</summary>
public static class Datasets
{
    private static readonly Lazy<string?> _root = new(() =>
    {
        foreach (var start in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            var directory = new DirectoryInfo(start);
            for (var depth = 0; directory is not null && depth < 12; depth++)
            {
                var candidate = System.IO.Path.Combine(directory.FullName, "datasets");
                if (File.Exists(System.IO.Path.Combine(candidate, "iris.csv"))) return candidate;
                directory = directory.Parent;
            }
        }
        return null;
    });

    /// <summary>The full path of <paramref name="file"/> inside <c>datasets/</c>.</summary>
    /// <exception cref="FileNotFoundException">When the gallery is run outside the repository.</exception>
    public static string Path(string file) => _root.Value is { } root
        ? System.IO.Path.Combine(root, file)
        : throw new FileNotFoundException(
            $"This showcase reads datasets/{file}. Run the gallery from inside the GravicodeScience repository.");
}

/// <summary>The chart style every showcase shares, so the gallery reads as one publication.</summary>
public static class GalleryCharts
{
    /// <summary>Ink used for text and axes, taken from the app's palette.</summary>
    public const string Ink = "#16132E";

    /// <summary>
    /// The library colours, in a cycle that keeps neighbouring series distinguishable.
    /// </summary>
    public static IReadOnlyList<Color> Cycle { get; } =
    [
        Color.Parse("#1B95BC"), Color.Parse("#D9363E"), Color.Parse("#2E9E68"), Color.Parse("#E8A317"),
        Color.Parse("#4F46D6"), Color.Parse("#9634C8"), Color.Parse("#5F6B7A"),
    ];

    /// <summary>A figure in the gallery style.</summary>
    public static Figure Figure(double widthInches = 8, double heightInches = 4.6) =>
        new(widthInches, heightInches, 110, Style()) { TightLayout = true };

    /// <summary>The style itself, for showcases that build figures their own way.</summary>
    public static PlotStyle Style()
    {
        var style = PlotStyle.SeabornWhiteGrid().Clone();
        style.FigureFaceColor = Color.Parse("#FFFFFF");
        style.AxesFaceColor = Color.Parse("#FFFFFF");
        style.AxesEdgeColor = Color.Parse("#C9D0D8");
        style.Spines = (true, true, false, false);
        style.GridVisible = true;
        style.GridColor = Color.Parse("#E4E8ED");
        style.TickColor = Color.Parse("#8A94A3");
        style.TickLabelColor = Color.Parse("#4A5363");
        style.TextColor = Color.Parse(Ink);
        style.ColorCycle = Cycle;
        style.Colormap = "viridis";
        style.LineWidth = 2.0;
        // Charts are drawn at full width and then fitted into a card, so text is set larger than
        // matplotlib's default to stay readable at the size the card shows it.
        style.FontSize = 13;
        return style;
    }

    /// <summary>A library's colour, for when a series stands for that library.</summary>
    public static Color Of(GraviLibrary library) => Color.Parse(LibraryInfo.Of(library).Hex);
}
