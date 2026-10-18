using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using ScienceGallery.Services;
using ScienceGallery.ViewModels;
using ScienceGallery.Views;

namespace ScienceGallery;

/// <summary>The gallery application.</summary>
public partial class App : Application
{
    /// <inheritdoc />
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    /// <inheritdoc />
    public override void OnFrameworkInitializationCompleted()
    {
        SyntaxTheme.Apply(ActualThemeVariant);
        ActualThemeVariantChanged += (_, _) => SyntaxTheme.Apply(ActualThemeVariant);

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var window = new MainWindow { DataContext = new MainViewModel() };
            desktop.MainWindow = window;

            // `ScienceGallery --capture <folder>` renders the documentation screenshots and exits.
            var capture = Array.IndexOf(desktop.Args ?? [], "--capture");
            if (capture >= 0 && capture + 1 < desktop.Args!.Length)
                window.Opened += async (_, _) => await ScreenshotTour.RunAsync(window, desktop.Args[capture + 1], desktop);
        }

        base.OnFrameworkInitializationCompleted();
    }
}
