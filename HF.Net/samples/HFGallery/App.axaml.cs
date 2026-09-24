using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Avalonia.Styling;
using Avalonia.Threading;
using HFGallery.ViewModels;
using HFGallery.Views;

namespace HFGallery;

/// <summary>The application.</summary>
public sealed partial class App : Application
{
    /// <inheritdoc />
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    /// <inheritdoc />
    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            if (Program.Light) RequestedThemeVariant = ThemeVariant.Light;

            var model = new MainViewModel();

            if (Program.StartOn is { Length: > 0 } wanted)
            {
                var match = model.Cases.FirstOrDefault(
                    c => c.Title.StartsWith(wanted, StringComparison.OrdinalIgnoreCase));

                if (match is not null)
                {
                    model.Selected = match;

                    // Queued rather than awaited: the run needs a window to report progress into.
                    Dispatcher.UIThread.Post(
                        () => model.RunCommand.Execute(null), DispatcherPriority.Background);
                }
            }

            desktop.MainWindow = new MainWindow { DataContext = model };
        }

        base.OnFrameworkInitializationCompleted();
    }
}
