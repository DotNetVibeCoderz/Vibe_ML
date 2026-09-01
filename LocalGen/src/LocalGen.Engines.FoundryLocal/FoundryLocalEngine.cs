using LocalGen.Core;
using LocalGen.Core.Engines;
using LocalGen.Core.Models;
using LocalGen.Sdk;
using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.Logging;

namespace LocalGen.Engines.FoundryLocal;

/// <summary>
/// Serves models through Microsoft Foundry Local.
/// </summary>
/// <remarks>
/// Foundry Local manages its own model cache and runs its own OpenAI-compatible web service with
/// hardware-specific execution providers already selected. LocalGen therefore drives it rather
/// than loading weights itself: this engine starts the Foundry service, ensures the requested
/// model is cached and loaded, then proxies inference over HTTP.
/// </remarks>
public sealed class FoundryLocalEngine : IInferenceEngine
{
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger<FoundryLocalEngine> _logger;
    private readonly SemaphoreSlim _initLock = new(1, 1);

    private EngineAvailability? _availability;
    private string? _endpoint;

    public FoundryLocalEngine(ILoggerFactory loggerFactory)
    {
        _loggerFactory = loggerFactory;
        _logger = loggerFactory.CreateLogger<FoundryLocalEngine>();
    }

    public EngineDescriptor Descriptor { get; } = new()
    {
        Kind = EngineKind.FoundryLocal,
        DisplayName = "Foundry Local",
        Description =
            "Runs models through Microsoft Foundry Local, which manages its own catalogue and " +
            "picks the best execution provider for the hardware it finds.",
        Recommendation =
            "Choose this on Copilot+ PCs and machines with an NPU, or when you already use the " +
            "Foundry catalogue. Foundry Local must be installed separately.",
        Capabilities = new EngineCapabilities
        {
            SupportsStreaming = true,
            SupportsEmbeddings = true,
            SupportsGrammar = false,
            SupportsToolCalling = true,
            SupportsVision = false,
            SupportsMultiGpu = false,
            Formats = [ModelFormat.FoundryAlias, ModelFormat.Onnx],
            Devices = [DeviceKind.Auto]
        }
    };

    public async ValueTask<EngineAvailability> ProbeAsync(CancellationToken cancellationToken = default)
    {
        if (_availability is not null)
        {
            return _availability;
        }

        await _initLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_availability is not null)
            {
                return _availability;
            }

            return _availability = await ProbeCoreAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _initLock.Release();
        }
    }

    private async Task<EngineAvailability> ProbeCoreAsync(CancellationToken cancellationToken)
    {
        try
        {
            var manager = FoundryLocalManager.Instance;

            if (!FoundryLocalManager.IsInitialized)
            {
                await FoundryLocalManager
                    .CreateAsync(
                        new Configuration { AppName = "LocalGen" },
                        _loggerFactory.CreateLogger("FoundryLocal"),
                        cancellationToken)
                    .ConfigureAwait(false);
            }

            await manager.StartWebServiceAsync(cancellationToken).ConfigureAwait(false);

            _endpoint = manager.Urls?.FirstOrDefault();

            if (string.IsNullOrEmpty(_endpoint))
            {
                return EngineAvailability.Unavailable(
                    "Foundry Local started but did not report a service address.");
            }

            _logger.LogInformation("Foundry Local is serving at {Endpoint}", _endpoint);

            return new EngineAvailability
            {
                IsAvailable = true,
                Version = "Foundry Local",
                AvailableDevices = [DeviceKind.Auto]
            };
        }
        catch (FoundryLocalException ex)
        {
            _logger.LogInformation(ex, "Foundry Local is not usable on this machine.");
            return EngineAvailability.Unavailable(
                $"Foundry Local reported: {ex.Message}. Install it with 'winget install Microsoft.FoundryLocal'.");
        }
        catch (Exception ex) when (ex is DllNotFoundException or FileNotFoundException or TypeInitializationException)
        {
            return EngineAvailability.Unavailable(
                "Foundry Local is not installed. Install it with 'winget install Microsoft.FoundryLocal'.");
        }
    }

    public bool CanServe(ModelDescriptor model) =>
        model.Format == ModelFormat.FoundryAlias ||
        model.SupportedEngines.Contains(EngineKind.FoundryLocal) ||
        model.Source.StartsWith("foundry:", StringComparison.OrdinalIgnoreCase);

    public async ValueTask<IModelSession> LoadAsync(
        ModelDescriptor model,
        ModelLoadOptions options,
        CancellationToken cancellationToken = default)
    {
        var availability = await ProbeAsync(cancellationToken).ConfigureAwait(false);

        if (!availability.IsAvailable || _endpoint is null)
        {
            throw new EngineNotAvailableException(Descriptor.DisplayName, availability.Reason);
        }

        // The descriptor's path holds the Foundry alias rather than a file path.
        var alias = string.IsNullOrEmpty(model.Path) ? model.Name : model.Path;

        try
        {
            var catalog = await FoundryLocalManager.Instance
                .GetCatalogAsync(cancellationToken)
                .ConfigureAwait(false);

            var foundryModel = await catalog.GetModelAsync(alias, cancellationToken).ConfigureAwait(false)
                ?? throw new ModelLoadException(model.Id, $"Foundry Local has no model aliased '{alias}'.");

            if (!await foundryModel.IsCachedAsync(cancellationToken).ConfigureAwait(false))
            {
                _logger.LogInformation("Downloading {Alias} through Foundry Local…", alias);
                await foundryModel.DownloadAsync(null, cancellationToken).ConfigureAwait(false);
            }

            if (!await foundryModel.IsLoadedAsync(cancellationToken).ConfigureAwait(false))
            {
                await foundryModel.LoadAsync(cancellationToken).ConfigureAwait(false);
            }

            _logger.LogInformation("Foundry Local is serving {Alias} ({Id})", alias, foundryModel.Id);

            var client = new LocalGenClient(new LocalGenClientOptions
            {
                Endpoint = _endpoint,
                DefaultModel = foundryModel.Id
            });

            var descriptor = model with
            {
                ContextLength = foundryModel.Info?.ContextLength is { } context && context > 0
                    ? (int)Math.Min(context, int.MaxValue)
                    : model.ContextLength
            };

            return new OpenAiHttpSession(descriptor, client, foundryModel.Id, EngineKind.FoundryLocal);
        }
        catch (FoundryLocalException ex)
        {
            throw new ModelLoadException(model.Id, ex.Message, ex);
        }
    }

    /// <summary>Lists what the Foundry catalogue offers, for the Model Gallery.</summary>
    public async Task<IReadOnlyList<CatalogEntry>> ListCatalogAsync(
        CancellationToken cancellationToken = default)
    {
        var availability = await ProbeAsync(cancellationToken).ConfigureAwait(false);

        if (!availability.IsAvailable)
        {
            return [];
        }

        try
        {
            var catalog = await FoundryLocalManager.Instance
                .GetCatalogAsync(cancellationToken)
                .ConfigureAwait(false);

            var models = await catalog.ListModelsAsync(cancellationToken).ConfigureAwait(false);

            return
            [
                .. models.Select(static model => new CatalogEntry
                {
                    Reference = $"foundry:{model.Alias}",
                    Name = model.Info?.DisplayName ?? model.Alias,
                    Publisher = model.Info?.Publisher ?? "Microsoft",
                    Description = model.Info?.Task ?? string.Empty,
                    Provider = "foundry",
                    Tags = model.Info?.Runtime is { } runtime
                        ? [runtime.ToString() ?? string.Empty]
                        : []
                })
            ];
        }
        catch (FoundryLocalException)
        {
            return [];
        }
    }
}
