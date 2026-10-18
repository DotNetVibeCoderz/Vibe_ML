using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using ScienceGallery.Models;
using ScienceGallery.ViewModels;
using ScienceGallery.Views;

namespace ScienceGallery.Services;

/// <summary>
/// Drives the real window through a few states and saves each one, for the documentation.
/// </summary>
/// <remarks>
/// Run with <c>ScienceGallery --capture docs/screenshots</c>. The window renders itself into an
/// offscreen bitmap, so the screenshots are exact pixels of the app rather than a mock-up, and they
/// are reproducible: regenerate them after any UI change.
/// </remarks>
public static class ScreenshotTour
{
    /// <summary>Use cases opened for the detail screenshots, by class name.</summary>
    private static readonly string[] DetailShowcases = ["HeartRateFromEcg", "CellCounting", "PortfolioFrontier", "ProductionLineControlChart"];

    /// <summary>Takes the screenshots and closes the app.</summary>
    public static async Task RunAsync(MainWindow window, string folder, IClassicDesktopStyleApplicationLifetime lifetime)
    {
        Directory.CreateDirectory(folder);
        var model = (MainViewModel)window.DataContext!;

        try
        {
            await Settle();
            Save(window, folder, "gallery-home.png");

            model.SelectedDomain = model.Domains.First(d => d.Info?.Domain == Domain.Research);
            await Settle();
            Save(window, folder, "gallery-domain.png");

            model.ClearFiltersCommand.Execute(null);
            model.Libraries.First(l => l.Info.Library == GraviLibrary.SciNet).IsChecked = true;
            model.Libraries.First(l => l.Info.Library == GraviLibrary.GraviFrame).IsChecked = true;
            await Settle();
            Save(window, folder, "gallery-filter.png");
            model.ClearFiltersCommand.Execute(null);

            var index = 1;
            foreach (var name in DetailShowcases)
            {
                var card = model.Visible.FirstOrDefault(c => c.Showcase.GetType().Name == name);
                if (card is null) continue;
                await model.OpenCommand.ExecuteAsync(card);
                await Settle();
                Save(window, folder, $"gallery-detail-{index++}.png");
            }

            window.SetDark(true);
            await Settle();
            Save(window, folder, "gallery-detail-dark.png");
            model.CloseCommand.Execute(null);
            await Settle();
            Save(window, folder, "gallery-home-dark.png");
        }
        finally
        {
            lifetime.Shutdown();
        }
    }

    private static async Task Settle()
    {
        await Task.Delay(700);
        await Dispatcher.UIThread.InvokeAsync(() => { }, DispatcherPriority.Background);
    }

    /// <summary>
    /// Lays the window's content out at a fixed 1440×900 and renders it, so the screenshots look
    /// the same whatever screen they are taken on (a laptop clamps the real window smaller).
    /// </summary>
    private static void Save(MainWindow window, string folder, string name)
    {
        const double width = 1440, height = 900;
        var content = (Control)window.Content!;
        window.FitColumnsFor(width);
        content.Measure(new Size(width, height));
        content.Arrange(new Rect(0, 0, width, height));

        var scale = window.RenderScaling;
        var size = new PixelSize((int)(width * scale), (int)(height * scale));
        using var bitmap = new RenderTargetBitmap(size, new Vector(96 * scale, 96 * scale));
        bitmap.Render(content);
        bitmap.Save(Path.Combine(folder, name), new PngBitmapEncoderOptions());

        // Give the window back its real size on the next layout pass.
        window.FitColumnsFor(window.Bounds.Width);
        content.InvalidateMeasure();
    }
}
