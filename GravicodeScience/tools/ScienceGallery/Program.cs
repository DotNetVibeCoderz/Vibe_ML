using System.Diagnostics;
using Avalonia;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery;

/// <summary>Entry point for Gravicode Science Gallery.</summary>
public static class Program
{
    /// <summary>Starts the application, or with <c>--run-all</c> runs every use case without a window.</summary>
    [STAThread]
    public static int Main(string[] args)
    {
        if (args.Contains("--run-all")) return RunAll();
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        return 0;
    }

    /// <summary>Configures Avalonia. Also used by the designer.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>()
        .UsePlatformDetect()
        .WithInterFont()
        .LogToTrace();

    /// <summary>
    /// Runs every showcase and renders its chart, headless. Exits non-zero if any fails, so CI can
    /// check that every example the gallery shows actually works.
    /// </summary>
    private static int RunAll()
    {
        var failures = 0;
        foreach (var showcase in ShowcaseRegistry.All)
        {
            var output = new ShowcaseOutput();
            var watch = Stopwatch.StartNew();
            try
            {
                showcase.Run(output);
                var png = output.Figure?.ToPng(110);
                watch.Stop();
                Console.WriteLine($"  ok    {watch.ElapsedMilliseconds,6} ms  {showcase.Domain,-10} {showcase.GetType().Name}" +
                                  $"  ({output.Metrics.Count} metrics{(png is null ? "" : $", chart {png.Length / 1024} KB")})");
                if (ShowcaseRegistry.SourceOf(showcase).StartsWith("// Source", StringComparison.Ordinal))
                    throw new InvalidOperationException("source file not embedded");
            }
            catch (Exception exception)
            {
                failures++;
                Console.WriteLine($"  FAIL            {showcase.Domain,-10} {showcase.GetType().Name}: {exception.GetType().Name}: {exception.Message}");
            }
        }
        Console.WriteLine($"{ShowcaseRegistry.All.Count - failures} of {ShowcaseRegistry.All.Count} use cases ran.");
        return failures == 0 ? 0 : 1;
    }
}
