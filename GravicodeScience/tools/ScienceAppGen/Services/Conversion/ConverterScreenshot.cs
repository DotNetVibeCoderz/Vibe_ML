using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using ScienceAppGen.ViewModels;
using ScienceAppGen.Views;

namespace ScienceAppGen.Services.Conversion;

/// <summary>
/// <c>ScienceAppGen --capture-converter &lt;source&gt; &lt;folder&gt; [--convert]</c>: drives the
/// converter page through the real view model and renders it to PNGs.
/// </summary>
/// <remarks>
/// With <c>--convert</c> this is an end-to-end test of the page, not a mock-up: it calls the
/// configured model, builds and runs the result exactly as pressing Convert does, then captures
/// the finished state. The content is rendered at a fixed 1600×1000 so the images do not depend on
/// the screen they were taken on.
/// </remarks>
public static class ConverterScreenshot
{
    /// <summary>True when the arguments ask for a capture.</summary>
    public static bool IsRequested(string[] args) => args.Contains("--capture-converter", StringComparer.OrdinalIgnoreCase);

    /// <summary>Runs the capture and shuts the application down.</summary>
    public static async Task RunAsync(string[] args, MainWindow window, ShellViewModel shell,
        IClassicDesktopStyleApplicationLifetime lifetime)
    {
        var index = Array.FindIndex(args, a => a.Equals("--capture-converter", StringComparison.OrdinalIgnoreCase));
        var source = Path.GetFullPath(args[index + 1]);
        var folder = Path.GetFullPath(args[index + 2]);
        var convert = args.Contains("--convert", StringComparer.OrdinalIgnoreCase);
        var name = Path.GetFileNameWithoutExtension(source.TrimEnd(Path.DirectorySeparatorChar));
        Directory.CreateDirectory(folder);

        try
        {
            await Settle();
            shell.IsConverterOpen = true;
            shell.Converter.LoadSource(source);
            if (convert) shell.Converter.OutputDirectory = Path.Combine(folder, "converted", shell.Converter.ProjectName);
            await Settle();
            Save(window, folder, $"converter-{name}-plan.png");

            if (convert)
            {
                await shell.Converter.ConvertAsync();
                shell.Converter.SelectedModule = shell.Converter.Modules.FirstOrDefault(m => m.Module.IsEntryPoint)
                                                 ?? shell.Converter.Modules.FirstOrDefault();
                await Settle();
                Save(window, folder, $"converter-{name}-done.png");
                Console.WriteLine(shell.Converter.ResultSummary);
            }
        }
        finally
        {
            lifetime.Shutdown();
        }
    }

    private static async Task Settle()
    {
        await Task.Delay(800);
        await Dispatcher.UIThread.InvokeAsync(() => { }, DispatcherPriority.Background);
    }

    private static void Save(Window window, string folder, string name)
    {
        const double width = 1600, height = 1000;
        var content = (Control)window.Content!;
        content.Measure(new Size(width, height));
        content.Arrange(new Rect(0, 0, width, height));

        var scale = window.RenderScaling;
        using var bitmap = new RenderTargetBitmap(new PixelSize((int)(width * scale), (int)(height * scale)),
            new Vector(96 * scale, 96 * scale));
        bitmap.Render(content);
        bitmap.Save(Path.Combine(folder, name), new PngBitmapEncoderOptions());

        content.InvalidateMeasure();
    }
}
