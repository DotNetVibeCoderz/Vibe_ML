using LocalGen.Core.Configuration;
using LocalGen.Core.Engines;
using LocalGen.Core.Inference;
using LocalGen.Core.Models;
using LocalGen.Engines.LlamaSharp;
using LocalGen.Runtime;
using LocalGen.Runtime.Sessions;
using LocalGen.Sdk;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace LocalGen.Cli;

/// <summary>
/// How the CLI reaches LocalGen: through a running server, or by loading models itself.
/// </summary>
/// <remarks>
/// Both modes matter. When the service is already running it must be reused, or the CLI would
/// load a second copy of the weights and double the memory footprint. When nothing is running,
/// requiring the user to start a daemon first would be friction for a one-off <c>localgen run</c>.
/// So the CLI probes for a server and falls back to loading models in-process.
/// </remarks>
public interface ILocalGenBridge : IAsyncDisposable
{
    /// <summary>Description of the active mode, shown in command output.</summary>
    string Mode { get; }

    ValueTask<IReadOnlyList<ModelDescriptor>> ListModelsAsync(CancellationToken cancellationToken);

    ValueTask<ModelDescriptor?> GetModelAsync(string id, CancellationToken cancellationToken);

    ValueTask<ModelDescriptor> PullAsync(
        string reference,
        IProgress<DownloadProgress> progress,
        CancellationToken cancellationToken);

    ValueTask<bool> RemoveAsync(string id, CancellationToken cancellationToken);

    IAsyncEnumerable<ChatStreamChunk> StreamAsync(ChatRequest request, CancellationToken cancellationToken);
}

/// <summary>Talks to a running LocalGen server over HTTP.</summary>
public sealed class RemoteBridge(LocalGenClient client, string endpoint) : ILocalGenBridge
{
    public string Mode => $"connected to {endpoint}";

    public async ValueTask<IReadOnlyList<ModelDescriptor>> ListModelsAsync(
        CancellationToken cancellationToken) =>
        await client.Admin.ListModelsAsync(cancellationToken).ConfigureAwait(false);

    public async ValueTask<ModelDescriptor?> GetModelAsync(string id, CancellationToken cancellationToken)
    {
        var models = await client.Admin.ListModelsAsync(cancellationToken).ConfigureAwait(false);
        return models.FirstOrDefault(m =>
            string.Equals(m.Id, id, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(m.Name, id, StringComparison.OrdinalIgnoreCase));
    }

    public async ValueTask<ModelDescriptor> PullAsync(
        string reference,
        IProgress<DownloadProgress> progress,
        CancellationToken cancellationToken)
    {
        ModelDescriptor? result = null;

        await foreach (var update in client.Admin
            .PullModelAsync(reference, cancellationToken)
            .ConfigureAwait(false))
        {
            if (update.Status == "error")
            {
                throw new LocalGenClientException(update.Error ?? "The pull failed.");
            }

            if (update.Status == "success")
            {
                result = update.Model;
                continue;
            }

            progress.Report(new DownloadProgress
            {
                ModelId = update.ModelId ?? reference,
                FileName = update.FileName ?? string.Empty,
                BytesDownloaded = update.BytesDownloaded,
                TotalBytes = update.TotalBytes,
                BytesPerSecond = update.BytesPerSecond
            });
        }

        return result ?? throw new LocalGenClientException(
            "The pull finished without reporting a model.");
    }

    public async ValueTask<bool> RemoveAsync(string id, CancellationToken cancellationToken) =>
        await client.Admin.RemoveModelAsync(id, cancellationToken).ConfigureAwait(false);

    public async IAsyncEnumerable<ChatStreamChunk> StreamAsync(
        ChatRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var wire = new Core.Protocol.ChatCompletionRequest
        {
            Model = request.Model,
            Temperature = request.Options.Temperature,
            TopP = request.Options.TopP,
            MaxTokens = request.Options.MaxTokens,
            Messages =
            [
                .. request.Messages.Select(static m => new Core.Protocol.OpenAiMessage
                {
                    Role = m.Role.ToString().ToLowerInvariant(),
                    Content = System.Text.Json.JsonSerializer.SerializeToElement(m.Text)
                })
            ]
        };

        await foreach (var chunk in client.StreamChatAsync(wire, cancellationToken).ConfigureAwait(false))
        {
            var choice = chunk.Choices.FirstOrDefault();

            yield return new ChatStreamChunk
            {
                Delta = choice?.Delta?.Content ?? string.Empty,
                FinishReason = Core.Protocol.OpenAiMapper.FromFinishReason(choice?.FinishReason),
                Usage = chunk.Usage is null ? null : new TokenUsage
                {
                    PromptTokens = chunk.Usage.PromptTokens,
                    CompletionTokens = chunk.Usage.CompletionTokens
                }
            };
        }
    }

    public ValueTask DisposeAsync()
    {
        client.Dispose();
        return ValueTask.CompletedTask;
    }
}

/// <summary>Loads models directly in the CLI process, for when no server is running.</summary>
public sealed class EmbeddedBridge : ILocalGenBridge
{
    private readonly ServiceProvider _services;
    private readonly IModelStore _store;
    private readonly ModelSessionManager _sessions;

    private EmbeddedBridge(ServiceProvider services)
    {
        _services = services;
        _store = services.GetRequiredService<IModelStore>();
        _sessions = services.GetRequiredService<ModelSessionManager>();
    }

    public string Mode => "running in-process (no server detected)";

    /// <summary>
    /// The in-process container. Exposed for the few commands that need runtime services the
    /// HTTP API does not cover, such as creating a model from a Modelfile.
    /// </summary>
    public IServiceProvider Services => _services;

    public static EmbeddedBridge Create(bool verbose)
    {
        var configuration = new ConfigurationBuilder()
            .AddJsonFile("appsettings.json", optional: true)
            .AddEnvironmentVariables("LOCALGEN_")
            .Build();

        var services = new ServiceCollection();

        services.AddLogging(logging =>
        {
            logging.AddSimpleConsole(console => console.SingleLine = true);
            // llama.cpp is chatty at startup; the CLI's own output is what the user came for.
            logging.SetMinimumLevel(verbose ? LogLevel.Debug : LogLevel.Warning);
        });

        services.AddLocalGenRuntime(configuration);
        services.AddLlamaSharpEngine();

        return new EmbeddedBridge(services.BuildServiceProvider());
    }

    public async ValueTask<IReadOnlyList<ModelDescriptor>> ListModelsAsync(
        CancellationToken cancellationToken) =>
        await _store.ListAsync(cancellationToken).ConfigureAwait(false);

    public async ValueTask<ModelDescriptor?> GetModelAsync(
        string id,
        CancellationToken cancellationToken) =>
        await _store.GetAsync(id, cancellationToken).ConfigureAwait(false);

    public async ValueTask<ModelDescriptor> PullAsync(
        string reference,
        IProgress<DownloadProgress> progress,
        CancellationToken cancellationToken) =>
        await _store.PullAsync(reference, progress, cancellationToken).ConfigureAwait(false);

    public async ValueTask<bool> RemoveAsync(string id, CancellationToken cancellationToken) =>
        await _store.RemoveAsync(id, cancellationToken).ConfigureAwait(false);

    public async IAsyncEnumerable<ChatStreamChunk> StreamAsync(
        ChatRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var lease = await _sessions
            .AcquireAsync(request.Model, null, cancellationToken)
            .ConfigureAwait(false);

        await foreach (var chunk in lease.Session
            .StreamAsync(request, cancellationToken)
            .ConfigureAwait(false))
        {
            yield return chunk;
        }
    }

    public async ValueTask DisposeAsync()
    {
        await _sessions.DisposeAsync().ConfigureAwait(false);
        await _services.DisposeAsync().ConfigureAwait(false);
    }
}

/// <summary>Picks the bridge to use, preferring a server that is already running.</summary>
public static class BridgeFactory
{
    public static async Task<ILocalGenBridge> CreateAsync(
        string? endpoint,
        bool verbose,
        CancellationToken cancellationToken = default)
    {
        endpoint ??= Environment.GetEnvironmentVariable("LOCALGEN_HOST")
                     ?? new ServerOptions().BaseUrl;

        var client = new LocalGenClient(new LocalGenClientOptions
        {
            Endpoint = endpoint,
            ApiKey = Environment.GetEnvironmentVariable("LOCALGEN_API_KEY"),
            // The probe must fail fast; the full timeout applies to real calls afterwards.
            Timeout = TimeSpan.FromSeconds(2)
        });

        if (await client.PingAsync(cancellationToken).ConfigureAwait(false))
        {
            // Rebuild with a working timeout now that the endpoint is known to be live.
            client.Dispose();

            return new RemoteBridge(
                new LocalGenClient(new LocalGenClientOptions
                {
                    Endpoint = endpoint,
                    ApiKey = Environment.GetEnvironmentVariable("LOCALGEN_API_KEY")
                }),
                endpoint);
        }

        client.Dispose();
        return EmbeddedBridge.Create(verbose);
    }
}
