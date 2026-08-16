using Avalonia;

namespace LocalGen.Desktop;

internal static class Program
{
    /// <summary>
    /// Avalonia needs an STA thread and its own initialisation before any UI type is touched,
    /// so nothing here may reference controls or the app's services.
    /// </summary>
    [STAThread]
    public static void Main(string[] args) =>
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
