using Avalonia;
using HFGallery.Cases;

namespace HFGallery;

/// <summary>Entry point for HF Gallery.</summary>
public static class Program
{
    /// <summary>Starts the gallery, or runs one case headlessly.</summary>
    /// <param name="args">
    /// <c>--run &lt;case title&gt;</c> runs a single case in the terminal and exits; <c>--list</c>
    /// prints the catalog. Everything else opens the window.
    /// </param>
    [STAThread]
    public static int Main(string[] args)
    {
        if (args.Contains("--list", StringComparer.OrdinalIgnoreCase))
        {
            foreach (var entry in Catalog.All)
            {
                Console.WriteLine($"{entry.Title,-22} {entry.Library,-18} {entry.Model}");
            }

            return 0;
        }

        var index = Array.FindIndex(args, a => a.Equals("--run", StringComparison.OrdinalIgnoreCase));
        if (index >= 0 && index + 1 < args.Length)
        {
            return RunHeadless(args[index + 1]).GetAwaiter().GetResult();
        }

        Light = args.Contains("--light", StringComparer.OrdinalIgnoreCase);

        var opened = Array.FindIndex(args, a => a.Equals("--open", StringComparison.OrdinalIgnoreCase));
        if (opened >= 0 && opened + 1 < args.Length) StartOn = args[opened + 1];

        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        return 0;
    }

    /// <summary>
    /// A case title to select and run as soon as the window opens, from <c>--open</c>.
    /// </summary>
    /// <remarks>
    /// Screenshots of a gallery whose every panel starts empty are useless, and driving the window
    /// with synthetic clicks to fill them is fragile. This makes the interesting state reachable
    /// from the command line.
    /// </remarks>
    public static string? StartOn { get; private set; }

    /// <summary>Whether <c>--light</c> asked for the light theme.</summary>
    /// <remarks>
    /// The light palette is its own set of steps rather than an inversion of the dark one, so it
    /// needs a way to be actually looked at - a theme nobody can open is a theme nobody checked.
    /// </remarks>
    public static bool Light { get; private set; }

    /// <summary>Configures Avalonia. Referenced by the designer as well as <see cref="Main"/>.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>()
        .UsePlatformDetect()
        .WithInterFont()
        .LogToTrace();

    /// <summary>
    /// Runs one case without a window and prints what it produced.
    /// </summary>
    /// <remarks>
    /// Every case is real work against a real model, and a window is a poor place to find out that
    /// a checkpoint stopped loading. This path is what CI and a terminal can drive.
    /// </remarks>
    private static async Task<int> RunHeadless(string title)
    {
        var entry = Catalog.All.FirstOrDefault(
            c => c.Title.StartsWith(title, StringComparison.OrdinalIgnoreCase));

        if (entry is null)
        {
            Console.Error.WriteLine($"No case starts with \"{title}\". Try --list.");
            return 2;
        }

        var progress = new Progress<string>(Console.WriteLine);

        try
        {
            var result = await entry.RunAsync(
                entry.DefaultInput ?? "", entry.DefaultSecondInput ?? "", progress, default);

            Console.WriteLine();
            Console.WriteLine(result.Summary);

            foreach (var bar in result.Bars ?? [])
            {
                Console.WriteLine($"  {bar.Label,-44} {bar.Value:F4}");
            }

            foreach (var (key, value) in result.Facts ?? [])
            {
                Console.WriteLine($"  {key,-14} {value}");
            }

            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"{entry.Title} failed: {error.Message}");
            return 1;
        }
    }
}
