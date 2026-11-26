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
        if (args.Contains("--run-all"))
        {
            // `--run-all --only Dask` runs just the use cases whose class name contains "Dask".
            var only = Array.IndexOf(args, "--only");
            return RunAll(only >= 0 && only + 1 < args.Length ? args[only + 1] : null, args.Contains("--verbose"));
        }
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
    private static int RunAll(string? only, bool verbose)
    {
        var failures = 0;
        var selected = ShowcaseRegistry.All
            .Where(s => only is null || s.GetType().Name.Contains(only, StringComparison.OrdinalIgnoreCase)).ToList();
        foreach (var showcase in selected)
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
                if (verbose)
                {
                    foreach (var (label, value, note) in output.Metrics) Console.WriteLine($"          {label}: {value}{(note is null ? "" : $"  ({note})")}");
                    foreach (var line in output.Lines) Console.WriteLine("          " + line.Replace("\n", "\n          "));
                }
                if (ShowcaseRegistry.SourceOf(showcase).StartsWith("// Source", StringComparison.Ordinal))
                    throw new InvalidOperationException("source file not embedded");
            }
            catch (Exception exception)
            {
                failures++;
                Console.WriteLine($"  FAIL            {showcase.Domain,-10} {showcase.GetType().Name}: {exception.GetType().Name}: {exception.Message}");
            }
        }
        Console.WriteLine($"{selected.Count - failures} of {selected.Count} use cases ran.");
        return failures == 0 ? 0 : 1;
    }
}
