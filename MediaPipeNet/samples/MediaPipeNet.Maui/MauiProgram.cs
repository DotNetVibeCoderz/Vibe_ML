// MediaPipe.NET on .NET MAUI — created by Gravicode Studios, led by Kang Fadhil.
namespace MediaPipeNet.Maui;

public static class MauiProgram
{
    public static MauiApp CreateMauiApp() =>
        MauiApp.CreateBuilder().UseMauiApp<App>().Build();
}

public sealed class App : Application
{
    protected override Window CreateWindow(IActivationState? activationState) =>
        new(new NavigationPage(new MainPage())) { Title = "MediaPipe.NET" };
}
