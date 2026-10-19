using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Avalonia.Styling;
using ScienceAppGen.Services;
using ScienceAppGen.ViewModels;
using ScienceAppGen.Views;

namespace ScienceAppGen;

/// <summary>The Avalonia application object; owns the services the whole app shares.</summary>
public partial class App : Application
{
    /// <summary>Reads and writes <c>app.config</c>.</summary>
    public static ConfigurationService Configuration { get; } = new();

    /// <summary>Collects log lines from every part of the app.</summary>
    public static LogService Logs { get; } = new();

    /// <inheritdoc />
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    /// <inheritdoc />
    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var settings = Configuration.Load();
            RequestedThemeVariant = settings.Theme.Equals("Light", StringComparison.OrdinalIgnoreCase)
                ? ThemeVariant.Light
                : ThemeVariant.Dark;

            var shell = new ShellViewModel(Configuration, Logs, settings);
            var window = new MainWindow { DataContext = shell };
            desktop.MainWindow = window;

            var args = desktop.Args ?? [];
            if (Services.Conversion.ConverterScreenshot.IsRequested(args))
            {
                // A capture must not rewrite the user's layout or last project in app.config.
                window.Opened += async (_, _) => await Services.Conversion.ConverterScreenshot.RunAsync(args, window, shell, desktop);
            }
            else
            {
                desktop.ShutdownRequested += (_, _) => shell.PersistLayout();
            }
        }

        base.OnFrameworkInitializationCompleted();
    }
}
