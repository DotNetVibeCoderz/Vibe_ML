using LocalGen.Desktop.ViewModels;
using LocalGen.Engines.LlamaSharp;
using LocalGen.Engines.Onnx;
using LocalGen.Kernel;
using LocalGen.Rag;
using LocalGen.Runtime;
using LocalGen.Runtime.Diagnostics;
using LocalGen.Server;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace LocalGen.Desktop.Services;

/// <summary>
/// Composes the desktop application's services.
/// </summary>
/// <remarks>
/// Admin Control hosts the inference runtime in its own process rather than shelling out to the
/// server executable. That is what lets the Playground and the web service share one set of
/// loaded weights: starting the service from the Service screen exposes the same models the
/// Playground is already using, instead of loading a second copy into memory.
/// </remarks>
public static class DesktopServices
{
    public static IServiceProvider Build()
    {
        var configuration = new ConfigurationBuilder()
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
            .AddEnvironmentVariables("LOCALGEN_")
            .Build();

        var services = new ServiceCollection();

        // One log store, shared by the runtime and the Logs panel.
        var logStore = new InMemoryLogStore();
        services.AddSingleton(logStore);

        services.AddLogging(logging =>
        {
            logging.AddProvider(new InMemoryLoggerProvider(logStore));
            logging.SetMinimumLevel(LogLevel.Information);
            logging.AddFilter("Microsoft.AspNetCore", LogLevel.Warning);
            logging.AddFilter("llama.cpp", LogLevel.Debug);
        });

        services.AddSingleton<IConfiguration>(configuration);
        services.AddLocalGenRuntime(configuration);
        services.AddLocalGenKernel();
        services.AddLocalGenRag();

        services.AddLlamaSharpEngine();
        services.AddOnnxEngine();

        services.AddSingleton<ServerController>();
        services.AddSingleton<ThemeService>();
        services.AddSingleton<IFileDialogService, FileDialogService>();

        services.AddSingleton<MainWindowViewModel>();
        services.AddSingleton<ServiceViewModel>();
        services.AddSingleton<PlaygroundViewModel>();
        services.AddSingleton<ModelsViewModel>();
        services.AddSingleton<EngineViewModel>();
        services.AddSingleton<ApiTestViewModel>();
        services.AddSingleton<MonitorViewModel>();
        services.AddSingleton<AboutViewModel>();

        return services.BuildServiceProvider();
    }
}

/// <summary>
/// Owns the embedded web service so the Service panel can start, stop and report on it.
/// </summary>
public sealed class ServerController : IAsyncDisposable
{
    private readonly IConfiguration _configuration;
    private readonly LocalGenServerHost _host = new();

    public ServerController(IConfiguration configuration)
    {
        _configuration = configuration;
        _host.StateChanged += (_, state) => StateChanged?.Invoke(this, state);
    }

    public event EventHandler<ServerState>? StateChanged;

    public ServerState State => _host.State;

    public string? BaseUrl => _host.BaseUrl;

    public Task StartAsync(CancellationToken cancellationToken = default) =>
        _host.StartAsync(_configuration, cancellationToken);

    public Task StopAsync(CancellationToken cancellationToken = default) =>
        _host.StopAsync(cancellationToken);

    public Task RestartAsync(CancellationToken cancellationToken = default) =>
        _host.RestartAsync(_configuration, cancellationToken);

    public ValueTask DisposeAsync() => _host.DisposeAsync();
}
