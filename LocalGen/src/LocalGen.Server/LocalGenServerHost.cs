using LocalGen.Core.Configuration;
using LocalGen.Core.Protocol;
using LocalGen.Engines.LlamaSharp;
using LocalGen.Engines.Onnx;
using LocalGen.Runtime;
using LocalGen.Runtime.Diagnostics;
using LocalGen.Server.Endpoints;
using LocalGen.Server.Services;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Diagnostics;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalGen.Server;

/// <summary>What the service is currently doing, as shown in Admin Control.</summary>
public enum ServerState
{
    Stopped,
    Starting,
    Running,
    Stopping,
    Faulted
}

/// <summary>
/// Builds and controls the LocalGen web service.
/// </summary>
/// <remarks>
/// The server is exposed as a controllable object rather than only a <c>Main</c> method because
/// the Avalonia Admin Control hosts it in-process: its Start/Stop/Status panel drives this class
/// directly, which avoids shipping a second process and keeps model sessions alive in the same
/// memory space the Playground already uses.
/// </remarks>
public sealed class LocalGenServerHost : IAsyncDisposable
{
    private readonly object _stateLock = new();

    private WebApplication? _app;
    private ServerState _state = ServerState.Stopped;

    /// <summary>Log buffer shared with the UI, kept across restarts so history is not lost.</summary>
    public InMemoryLogStore Logs { get; } = new();

    public ServerState State
    {
        get
        {
            lock (_stateLock)
            {
                return _state;
            }
        }
        private set
        {
            lock (_stateLock)
            {
                _state = value;
            }

            StateChanged?.Invoke(this, value);
        }
    }

    public event EventHandler<ServerState>? StateChanged;

    /// <summary>Base address the service is listening on, once running.</summary>
    public string? BaseUrl { get; private set; }

    /// <summary>The running application's services, for callers that host the server in-process.</summary>
    public IServiceProvider? Services => _app?.Services;

    /// <summary>Starts the service. Does nothing if it is already running.</summary>
    public async Task StartAsync(
        IConfiguration? configuration = null,
        CancellationToken cancellationToken = default)
    {
        if (State is ServerState.Running or ServerState.Starting)
        {
            return;
        }

        State = ServerState.Starting;

        try
        {
            _app = Build(configuration);
            await _app.StartAsync(cancellationToken).ConfigureAwait(false);

            var options = _app.Services.GetRequiredService<IOptions<LocalGenOptions>>().Value;
            BaseUrl = options.Server.BaseUrl;

            State = ServerState.Running;

            _app.Services.GetRequiredService<ILogger<LocalGenServerHost>>()
                .LogInformation("LocalGen is listening on {BaseUrl}", BaseUrl);
        }
        catch
        {
            State = ServerState.Faulted;
            throw;
        }
    }

    /// <summary>Stops the service and releases every loaded model.</summary>
    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        if (_app is null || State is ServerState.Stopped or ServerState.Stopping)
        {
            return;
        }

        State = ServerState.Stopping;

        try
        {
            await _app.StopAsync(cancellationToken).ConfigureAwait(false);
            await _app.DisposeAsync().ConfigureAwait(false);
        }
        finally
        {
            _app = null;
            BaseUrl = null;
            State = ServerState.Stopped;
        }
    }

    public async Task RestartAsync(
        IConfiguration? configuration = null,
        CancellationToken cancellationToken = default)
    {
        await StopAsync(cancellationToken).ConfigureAwait(false);
        await StartAsync(configuration, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Composes the web application. Also used by the test host and the CLI's serve command.</summary>
    public WebApplication Build(IConfiguration? configuration = null)
    {
        var builder = WebApplication.CreateBuilder();

        if (configuration is not null)
        {
            builder.Configuration.AddConfiguration(configuration);
        }

        var options = new LocalGenOptions();
        builder.Configuration.GetSection(LocalGenOptions.SectionName).Bind(options);
        LocalGenPaths.EnsureCreated(options);

        builder.WebHost.UseUrls(options.Server.BaseUrl);

        builder.Logging.ClearProviders();
        builder.Logging.AddSimpleConsole(c => c.SingleLine = true);
        // Feeds the Admin Control log panel.
        builder.Logging.AddProvider(new InMemoryLoggerProvider(Logs));
        builder.Logging.SetMinimumLevel(LogLevel.Information);
        builder.Logging.AddFilter("Microsoft.AspNetCore", LogLevel.Warning);

        builder.Services.AddSingleton(Logs);
        builder.Services.AddLocalGenRuntime(builder.Configuration);

        // Backends register themselves; a deployment can trim either one out.
        builder.Services.AddLlamaSharpEngine(preferGpu: options.Engine.Device != Core.Engines.DeviceKind.Cpu);
        builder.Services.AddOnnxEngine();

        builder.Services.AddSingleton<InferenceService>();
        builder.Services.AddSingleton<AttachmentResolver>();
        builder.Services.AddHostedService<MaintenanceService>();

        builder.Services.ConfigureHttpJsonOptions(json =>
        {
            json.SerializerOptions.PropertyNamingPolicy = OpenAiJson.Options.PropertyNamingPolicy;
            json.SerializerOptions.DefaultIgnoreCondition = OpenAiJson.Options.DefaultIgnoreCondition;
            json.SerializerOptions.PropertyNameCaseInsensitive = true;
        });

        builder.Services.AddCors(cors => cors.AddDefaultPolicy(policy =>
        {
            if (options.Server.CorsOrigins.Contains("*"))
            {
                policy.AllowAnyOrigin();
            }
            else
            {
                policy.WithOrigins(options.Server.CorsOrigins).AllowCredentials();
            }

            policy.AllowAnyHeader().AllowAnyMethod();
        }));

        builder.Services.AddEndpointsApiExplorer();

        var app = builder.Build();

        app.UseCors();
        app.UseExceptionHandler(handler => handler.Run(WriteProblemAsync));

        if (!string.IsNullOrEmpty(options.Server.ApiKey))
        {
            app.UseMiddleware<ApiKeyMiddleware>();
        }

        app.MapOpenAiEndpoints();
        app.MapAdminEndpoints();
        app.MapFileEndpoints();

        // A friendly root, so hitting the base URL in a browser is not a 404.
        app.MapGet("/", () => Results.Json(new
        {
            name = "LocalGen",
            description = "Local AI inference engine with an OpenAI-compatible API.",
            version = typeof(LocalGenServerHost).Assembly.GetName().Version?.ToString(3) ?? "0.1.0",
            author = "Gravicode Studios — led by Kang Fadhil",
            endpoints = new[] { "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/models", "/api/status" }
        }));

        return app;
    }

    /// <summary>Renders unhandled exceptions in the OpenAI error shape clients expect.</summary>
    private static async Task WriteProblemAsync(HttpContext context)
    {
        var feature = context.Features.Get<IExceptionHandlerFeature>();
        var exception = feature?.Error;

        var (status, type) = exception switch
        {
            Core.ModelNotFoundException => (StatusCodes.Status404NotFound, "model_not_found"),
            Core.EngineNotAvailableException => (StatusCodes.Status503ServiceUnavailable, "engine_unavailable"),
            Core.OfflineModeException => (StatusCodes.Status403Forbidden, "offline_mode"),
            Core.ToolPermissionException => (StatusCodes.Status403Forbidden, "tool_forbidden"),
            Core.LocalGenException => (StatusCodes.Status400BadRequest, "invalid_request_error"),
            _ => (StatusCodes.Status500InternalServerError, "server_error")
        };

        context.Response.StatusCode = status;
        await context.Response.WriteAsJsonAsync(
            OpenAiErrorResponse.Create(exception?.Message ?? "Unexpected error.", type),
            OpenAiJson.Options).ConfigureAwait(false);
    }

    public async ValueTask DisposeAsync() => await StopAsync().ConfigureAwait(false);
}
