using System.Collections.Concurrent;
using System.Diagnostics;
using LocalGen.Core;
using LocalGen.Core.Configuration;
using LocalGen.Core.Engines;
using LocalGen.Core.Models;
using LocalGen.Runtime.Engines;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalGen.Runtime.Sessions;

/// <summary>A model held in memory, with the bookkeeping needed to decide when to evict it.</summary>
public sealed record LoadedModel
{
    public required IModelSession Session { get; init; }

    public required DateTimeOffset LoadedAt { get; init; }

    public DateTimeOffset LastUsedAt { get; set; }

    public long RequestCount;

    /// <summary>Non-zero while requests are in flight; a busy model is never evicted.</summary>
    public int ActiveRequests;

    public string ModelId => Session.Model.Id;
}

/// <summary>
/// Owns the lifetime of loaded models.
/// </summary>
/// <remarks>
/// Weights take seconds to load and gigabytes of memory, so sessions are cached and shared across
/// requests. The cache is bounded by <see cref="ServerOptions.MaxLoadedModels"/> and evicts the
/// least recently used model — never one with requests in flight — which is what keeps a laptop
/// with 8 GB of VRAM from being pushed into swap by a second model.
/// </remarks>
public sealed class ModelSessionManager : IAsyncDisposable
{
    private readonly IModelStore _store;
    private readonly EngineRegistry _registry;
    private readonly LocalGenOptions _options;
    private readonly ILogger<ModelSessionManager> _logger;

    private readonly ConcurrentDictionary<string, LoadedModel> _loaded = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Serialises loads so two concurrent first-requests cannot load the same model twice.</summary>
    private readonly SemaphoreSlim _loadLock = new(1, 1);

    public ModelSessionManager(
        IModelStore store,
        EngineRegistry registry,
        IOptions<LocalGenOptions> options,
        ILogger<ModelSessionManager> logger)
    {
        _store = store;
        _registry = registry;
        _options = options.Value;
        _logger = logger;
    }

    /// <summary>Models currently resident, for the status screen and <c>localgen ps</c>.</summary>
    public IReadOnlyList<LoadedModel> Loaded => [.. _loaded.Values];

    /// <summary>
    /// Returns a ready session for a model, loading it if necessary. The returned lease must be
    /// disposed so the manager knows the request finished and the model may be evicted again.
    /// </summary>
    public async ValueTask<SessionLease> AcquireAsync(
        string modelId,
        EngineKind? preferredEngine = null,
        CancellationToken cancellationToken = default)
    {
        if (_loaded.TryGetValue(modelId, out var cached))
        {
            return Lease(cached);
        }

        await _loadLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Another caller may have loaded it while this one waited for the lock.
            if (_loaded.TryGetValue(modelId, out cached))
            {
                return Lease(cached);
            }

            var model = await _store.GetAsync(modelId, cancellationToken).ConfigureAwait(false)
                        ?? throw new ModelNotFoundException(modelId);

            await EvictIfNeededAsync(cancellationToken).ConfigureAwait(false);

            var engine = await _registry.SelectAsync(model, preferredEngine, cancellationToken)
                .ConfigureAwait(false);

            var loadOptions = BuildLoadOptions(model);

            _logger.LogInformation(
                "Loading {Model} on {Engine}…", model.Id, engine.Descriptor.DisplayName);

            var stopwatch = Stopwatch.StartNew();
            var session = await engine.LoadAsync(model, loadOptions, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation(
                "Loaded {Model} in {Elapsed:N1}s (context {Context} tokens, {Device})",
                model.Id, stopwatch.Elapsed.TotalSeconds, session.ContextSize, session.Device);

            var entry = new LoadedModel
            {
                Session = session,
                LoadedAt = DateTimeOffset.UtcNow,
                LastUsedAt = DateTimeOffset.UtcNow
            };

            _loaded[model.Id] = entry;
            return Lease(entry);
        }
        finally
        {
            _loadLock.Release();
        }
    }

    /// <summary>Unloads a model immediately, waiting for in-flight requests to drain.</summary>
    public async ValueTask<bool> UnloadAsync(string modelId, CancellationToken cancellationToken = default)
    {
        if (!_loaded.TryRemove(modelId, out var entry))
        {
            return false;
        }

        await DisposeEntryAsync(entry, cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("Unloaded {Model}", modelId);
        return true;
    }

    /// <summary>
    /// Evicts models idle for longer than the configured timeout. Called on a timer by the
    /// hosting service so a machine left running does not hold VRAM indefinitely.
    /// </summary>
    public async ValueTask EvictIdleAsync(CancellationToken cancellationToken = default)
    {
        var cutoff = DateTimeOffset.UtcNow - _options.Server.ModelIdleTimeout;

        foreach (var entry in _loaded.Values)
        {
            if (entry.ActiveRequests > 0 || entry.LastUsedAt > cutoff)
            {
                continue;
            }

            if (_loaded.TryRemove(entry.ModelId, out var removed))
            {
                _logger.LogInformation(
                    "Evicting {Model}: idle for {Idle:N0}s",
                    removed.ModelId, (DateTimeOffset.UtcNow - removed.LastUsedAt).TotalSeconds);

                await DisposeEntryAsync(removed, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>Makes room for one more model by evicting the least recently used idle one.</summary>
    private async ValueTask EvictIfNeededAsync(CancellationToken cancellationToken)
    {
        while (_loaded.Count >= _options.Server.MaxLoadedModels)
        {
            var victim = _loaded.Values
                .Where(static e => e.ActiveRequests == 0)
                .OrderBy(static e => e.LastUsedAt)
                .FirstOrDefault();

            if (victim is null)
            {
                // Everything resident is busy. Loading anyway risks running out of memory, but
                // refusing the request outright is worse, so the limit is treated as advisory.
                _logger.LogWarning(
                    "All {Count} loaded models are busy; exceeding MaxLoadedModels for this request.",
                    _loaded.Count);
                return;
            }

            if (_loaded.TryRemove(victim.ModelId, out var removed))
            {
                _logger.LogInformation("Evicting {Model} to stay within MaxLoadedModels.", removed.ModelId);
                await DisposeEntryAsync(removed, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>Layers configured engine defaults under the model's own requirements.</summary>
    private ModelLoadOptions BuildLoadOptions(ModelDescriptor model) => new()
    {
        Device = _options.Engine.Device,
        GpuLayerCount = _options.Engine.GpuLayers,
        ContextSize = _options.Engine.ContextSize ?? model.ContextLength,
        ThreadCount = _options.Engine.Threads,
        TensorSplit = _options.Engine.TensorSplit,
        EmbeddingMode = model.Supports(ModelCapability.Embedding)
                        && !model.Supports(ModelCapability.Chat)
    };

    private SessionLease Lease(LoadedModel entry)
    {
        Interlocked.Increment(ref entry.ActiveRequests);
        Interlocked.Increment(ref entry.RequestCount);
        entry.LastUsedAt = DateTimeOffset.UtcNow;
        return new SessionLease(entry);
    }

    /// <summary>
    /// Disposes a session once its in-flight requests finish. Tearing down a session mid-decode
    /// would free native memory the generation loop is still reading.
    /// </summary>
    private async ValueTask DisposeEntryAsync(LoadedModel entry, CancellationToken cancellationToken)
    {
        var deadline = DateTimeOffset.UtcNow + TimeSpan.FromMinutes(2);

        while (entry.ActiveRequests > 0 && DateTimeOffset.UtcNow < deadline)
        {
            await Task.Delay(100, cancellationToken).ConfigureAwait(false);
        }

        if (entry.ActiveRequests > 0)
        {
            _logger.LogWarning(
                "{Model} still has {Count} active request(s) after the drain timeout; disposing anyway.",
                entry.ModelId, entry.ActiveRequests);
        }

        await entry.Session.DisposeAsync().ConfigureAwait(false);
    }

    public async ValueTask DisposeAsync()
    {
        foreach (var entry in _loaded.Values)
        {
            await entry.Session.DisposeAsync().ConfigureAwait(false);
        }

        _loaded.Clear();
        _loadLock.Dispose();
    }
}

/// <summary>
/// Borrowed access to a loaded session. Disposing releases the model for eviction, so callers
/// must scope the lease to the whole request — including the streaming enumeration.
/// </summary>
public sealed class SessionLease : IDisposable
{
    private readonly LoadedModel _entry;
    private bool _released;

    internal SessionLease(LoadedModel entry) => _entry = entry;

    public IModelSession Session => _entry.Session;

    public void Dispose()
    {
        if (_released)
        {
            return;
        }

        _released = true;
        _entry.LastUsedAt = DateTimeOffset.UtcNow;
        Interlocked.Decrement(ref _entry.ActiveRequests);
    }
}
