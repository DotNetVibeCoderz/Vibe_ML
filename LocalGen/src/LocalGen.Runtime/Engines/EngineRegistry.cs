using LocalGen.Core;
using LocalGen.Core.Configuration;
using LocalGen.Core.Engines;
using LocalGen.Core.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalGen.Runtime.Engines;

/// <summary>An engine plus its probed state, as shown on the Engine settings screen.</summary>
public sealed record EngineStatus
{
    public required EngineDescriptor Descriptor { get; init; }

    public required EngineAvailability Availability { get; init; }

    /// <summary>Whether this is the engine new models load on by default.</summary>
    public bool IsDefault { get; init; }
}

/// <summary>
/// Chooses which backend serves a model and reports what is installed.
/// </summary>
/// <remarks>
/// Selection is deliberate rather than first-match: a model may declare preferred engines, the
/// user has a configured default, and some backends only handle certain formats. Getting this
/// wrong means a GGUF file silently lands on a backend that cannot read it.
/// </remarks>
public sealed class EngineRegistry
{
    private readonly IReadOnlyList<IInferenceEngine> _engines;
    private readonly LocalGenOptions _options;
    private readonly ILogger<EngineRegistry> _logger;

    public EngineRegistry(
        IEnumerable<IInferenceEngine> engines,
        IOptions<LocalGenOptions> options,
        ILogger<EngineRegistry> logger)
    {
        _engines = [.. engines];
        _options = options.Value;
        _logger = logger;
    }

    public IReadOnlyList<IInferenceEngine> Engines => _engines;

    /// <summary>Probes every registered backend, for the engine picker and the About screen.</summary>
    public async ValueTask<IReadOnlyList<EngineStatus>> GetStatusAsync(
        CancellationToken cancellationToken = default)
    {
        var statuses = new List<EngineStatus>(_engines.Count);

        foreach (var engine in _engines)
        {
            statuses.Add(new EngineStatus
            {
                Descriptor = engine.Descriptor,
                Availability = await engine.ProbeAsync(cancellationToken).ConfigureAwait(false),
                IsDefault = engine.Descriptor.Kind == _options.Engine.Default
            });
        }

        return statuses;
    }

    /// <summary>
    /// Resolves the backend for a model. Order of preference: an explicit override, the model's
    /// own declared engines, the configured default, then any backend that can serve the format.
    /// </summary>
    public async ValueTask<IInferenceEngine> SelectAsync(
        ModelDescriptor model,
        EngineKind? preferred = null,
        CancellationToken cancellationToken = default)
    {
        var candidates = new List<EngineKind>();

        if (preferred is { } explicitKind)
        {
            candidates.Add(explicitKind);
        }

        candidates.AddRange(model.SupportedEngines);
        candidates.Add(_options.Engine.Default);

        foreach (var kind in candidates.Distinct())
        {
            var engine = _engines.FirstOrDefault(e => e.Descriptor.Kind == kind);

            if (engine is null || !engine.CanServe(model))
            {
                continue;
            }

            var availability = await engine.ProbeAsync(cancellationToken).ConfigureAwait(false);
            if (availability.IsAvailable)
            {
                return engine;
            }

            _logger.LogDebug(
                "Skipping {Engine} for {Model}: {Reason}",
                kind, model.Id, availability.Reason);
        }

        // Nothing preferred worked — fall back to whatever can actually serve the model.
        foreach (var engine in _engines.Where(e => e.CanServe(model)))
        {
            var availability = await engine.ProbeAsync(cancellationToken).ConfigureAwait(false);
            if (availability.IsAvailable)
            {
                _logger.LogInformation(
                    "Falling back to {Engine} for {Model}.",
                    engine.Descriptor.Kind, model.Id);
                return engine;
            }
        }

        throw new EngineNotAvailableException(
            _options.Engine.Default.ToString(),
            BuildFailureExplanation(model));
    }

    /// <summary>
    /// Explains why no backend matched, distinguishing "nothing supports this format" from
    /// "the right backend is installed but its native libraries are missing".
    /// </summary>
    private string BuildFailureExplanation(ModelDescriptor model)
    {
        var capable = _engines.Where(e => e.CanServe(model)).ToList();

        if (capable.Count == 0)
        {
            var formats = string.Join(
                ", ",
                _engines.SelectMany(e => e.Descriptor.Capabilities.Formats).Distinct());

            return $"no installed backend can serve a {model.Format} model (supported formats: {formats}).";
        }

        return $"{string.Join(", ", capable.Select(e => e.Descriptor.DisplayName))} could serve this " +
               "model but reported themselves unavailable — check that the matching native backend " +
               "package is installed.";
    }

    /// <summary>
    /// Suggests the backend best suited to this machine, used by the Engine screen's
    /// recommendation hint.
    /// </summary>
    public async ValueTask<EngineRecommendation> RecommendAsync(
        CancellationToken cancellationToken = default)
    {
        var statuses = await GetStatusAsync(cancellationToken).ConfigureAwait(false);
        var available = statuses.Where(static s => s.Availability.IsAvailable).ToList();

        if (available.Count == 0)
        {
            return new EngineRecommendation
            {
                Engine = EngineKind.LlamaSharp,
                Device = DeviceKind.Cpu,
                Rationale =
                    "No backend is currently usable. Install LLamaSharp.Backend.Cpu (or .Cuda12 for " +
                    "an NVIDIA GPU) to get started."
            };
        }

        // A GPU-capable llama.cpp build is the best default: widest model coverage, and GPU
        // offload is worth far more than any other difference between the backends.
        var llama = available.FirstOrDefault(static s => s.Descriptor.Kind == EngineKind.LlamaSharp);

        if (llama is not null)
        {
            var gpu = llama.Availability.AvailableDevices
                .FirstOrDefault(static d => d is DeviceKind.Cuda or DeviceKind.Metal or DeviceKind.Vulkan);

            return gpu != default
                ? new EngineRecommendation
                {
                    Engine = EngineKind.LlamaSharp,
                    Device = gpu,
                    Rationale = $"LlamaSharp with {gpu} offload gives the best throughput on this machine."
                }
                : new EngineRecommendation
                {
                    Engine = EngineKind.LlamaSharp,
                    Device = DeviceKind.Cpu,
                    Rationale =
                        "LlamaSharp on CPU. Installing LLamaSharp.Backend.Cuda12 or .Vulkan would " +
                        "let LocalGen use the GPU and run several times faster."
                };
        }

        var fallback = available[0];
        return new EngineRecommendation
        {
            Engine = fallback.Descriptor.Kind,
            Device = fallback.Availability.AvailableDevices.FirstOrDefault(),
            Rationale = fallback.Descriptor.Recommendation
        };
    }
}

/// <summary>Advice shown next to the engine picker.</summary>
public sealed record EngineRecommendation
{
    public required EngineKind Engine { get; init; }

    public required DeviceKind Device { get; init; }

    public required string Rationale { get; init; }
}
