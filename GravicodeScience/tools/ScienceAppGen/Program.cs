using Avalonia;

namespace ScienceAppGen;

/// <summary>Entry point for the ScienceAppGen desktop application.</summary>
public static class Program
{
    /// <summary>
    /// Starts the Avalonia application, or runs a headless conversion for <c>--convert</c>
    /// (see <see cref="Services.Conversion.ConvertCommand"/>).
    /// </summary>
    [STAThread]
    public static int Main(string[] args)
    {
        if (Services.Conversion.ConvertCommand.IsRequested(args))
            return Services.Conversion.ConvertCommand.RunAsync(args).GetAwaiter().GetResult();

        return BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    /// <summary>Configures Avalonia. Referenced by the designer as well as <see cref="Main"/>.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>()
        .UsePlatformDetect()
        .WithInterFont()
        .LogToTrace();
}
