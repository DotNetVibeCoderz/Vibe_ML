using Avalonia;

namespace ScienceAppGen;

/// <summary>Entry point for the ScienceAppGen desktop application.</summary>
public static class Program
{
    /// <summary>Starts the Avalonia application.</summary>
    [STAThread]
    public static void Main(string[] args) => BuildAvaloniaApp()
        .StartWithClassicDesktopLifetime(args);

    /// <summary>Configures Avalonia. Referenced by the designer as well as <see cref="Main"/>.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>()
        .UsePlatformDetect()
        .WithInterFont()
        .LogToTrace();
}
