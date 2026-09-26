using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Avalonia.Styling;
using MediaPipeNet.Gallery.Services;

namespace MediaPipeNet.Gallery;

public sealed class App : Application
{
    /// <summary>Folder for automated screenshots (set by <c>--screenshots</c>).</summary>
    public static string? ScreenshotDirectory { get; set; }

    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        ApplyTheme();
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            desktop.MainWindow = new MainWindow();
        base.OnFrameworkInitializationCompleted();
    }

    public static void ApplyTheme()
    {
        if (Current is not null)
            Current.RequestedThemeVariant = AppSettings.Current.Theme == "Dark" ? ThemeVariant.Dark : ThemeVariant.Light;
    }
}
