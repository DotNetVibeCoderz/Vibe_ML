using Avalonia;
using MediaPipeNet.Gallery.Services;

namespace MediaPipeNet.Gallery;

// MediaPipe.Net Gallery — created by Gravicode Studios, led by Kang Fadhil.
//
//   MediaPipeNet.Gallery                          interactive app
//   MediaPipeNet.Gallery --screenshots <dir>      renders every page to PNG and exits (docs)
//   MediaPipeNet.Gallery --lang id --theme Dark   start in Indonesian / dark theme (not persisted)
internal static class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            switch (args[i])
            {
                case "--screenshots":
                    App.ScreenshotDirectory = Path.GetFullPath(args[i + 1]);
                    AppSettings.UseTransient(new AppSettings());
                    break;
                case "--lang":
                    AppSettings.Current.Language = args[i + 1];
                    break;
                case "--theme":
                    AppSettings.Current.Theme = args[i + 1];
                    break;
            }
        }
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
