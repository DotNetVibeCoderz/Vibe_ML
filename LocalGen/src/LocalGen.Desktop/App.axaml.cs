using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using LocalGen.Desktop.Services;
using LocalGen.Desktop.ViewModels;
using LocalGen.Desktop.Views;
using Microsoft.Extensions.DependencyInjection;

namespace LocalGen.Desktop;

public partial class App : Application
{
    /// <summary>
    /// The application's services. Resolved once at startup and exposed so views created by the
    /// XAML loader — which cannot take constructor arguments — can reach them.
    /// </summary>
    public static IServiceProvider Services { get; private set; } = default!;

    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        Services = DesktopServices.Build();

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var shell = Services.GetRequiredService<MainWindowViewModel>();

            desktop.MainWindow = new MainWindow { DataContext = shell };

            // The in-process server keeps models loaded, so it has to be told to shut down
            // before the process exits or native memory is torn down mid-generation.
            desktop.ShutdownRequested += (_, _) => shell.ShutdownAsync().GetAwaiter().GetResult();
        }

        base.OnFrameworkInitializationCompleted();
    }
}
