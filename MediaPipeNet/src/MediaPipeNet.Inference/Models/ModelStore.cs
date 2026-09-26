using System.Collections.Concurrent;
using System.Reflection;
using System.Security.Cryptography;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace MediaPipeNet.Inference.Models;

/// <summary>
/// Resolves pretrained models to verified local files by asking a chain of
/// <see cref="IModelProvider"/>s in order. Every file is checked against the SHA-256 recorded in its
/// <see cref="ModelDescriptor"/> before use (once per process and file).
/// </summary>
/// <remarks>
/// The default chain is:
/// <list type="number">
/// <item>the directory in the <c>MEDIAPIPENET_MODELS</c> environment variable, when set;</item>
/// <item>the application's <c>models</c> folder, filled by the <c>Gravicode.MediaPipeNet.Models.*</c> packages;</item>
/// <item>the user cache (<see cref="DefaultCacheDirectory"/>);</item>
/// <item>nuget.org — the model package is downloaded and extracted into the cache.</item>
/// </list>
/// </remarks>
public sealed class ModelStore
{
    /// <summary>Environment variable naming an extra model directory searched first.</summary>
    public const string ModelDirectoryEnvironmentVariable = "MEDIAPIPENET_MODELS";

    private static ModelStore? s_default;
    private readonly ConcurrentDictionary<string, DateTime> _verified = new(StringComparer.OrdinalIgnoreCase);
    private readonly ILogger _logger;

    internal static readonly HttpClient SharedHttpClient = new() { Timeout = TimeSpan.FromMinutes(10) };

    static ModelStore()
    {
        SharedHttpClient.DefaultRequestHeaders.UserAgent.ParseAdd($"MediaPipeNet/{LibraryVersion}");
    }

    /// <summary>Creates a store over the given providers.</summary>
    public ModelStore(IEnumerable<IModelProvider> providers, bool verifyChecksums = true, ILogger? logger = null)
    {
        ArgumentNullException.ThrowIfNull(providers);
        Providers = providers.ToArray();
        VerifyChecksums = verifyChecksums;
        _logger = logger ?? NullLogger.Instance;
    }

    /// <summary>The process-wide default store (see remarks). Replace it to change where tasks load models from.</summary>
    public static ModelStore Default
    {
        get => s_default ??= CreateDefault();
        set => s_default = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>The per-user model cache: <c>%LOCALAPPDATA%/MediaPipeNet/models</c> or <c>~/.local/share/MediaPipeNet/models</c>.</summary>
    public static string DefaultCacheDirectory { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData, Environment.SpecialFolderOption.Create) is { Length: > 0 } d
            ? d : Path.GetTempPath(),
        "MediaPipeNet", "models");

    /// <summary>The MediaPipe.NET version, used to pick matching model packages.</summary>
    public static string LibraryVersion { get; } =
        (typeof(ModelStore).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "0.1.0")
        .Split('+')[0];

    /// <summary>The providers asked, in order.</summary>
    public IReadOnlyList<IModelProvider> Providers { get; }

    /// <summary>Whether files are checked against their SHA-256 before use.</summary>
    public bool VerifyChecksums { get; }

    /// <summary>Creates the default provider chain.</summary>
    /// <param name="modelDirectory">Extra directory searched first (e.g. your own model folder).</param>
    /// <param name="cacheDirectory">Cache directory; defaults to <see cref="DefaultCacheDirectory"/>.</param>
    /// <param name="allowDownload">Whether missing models may be downloaded from nuget.org.</param>
    /// <param name="logger">Optional logger.</param>
    public static ModelStore CreateDefault(string? modelDirectory = null, string? cacheDirectory = null, bool allowDownload = true, ILogger? logger = null)
    {
        cacheDirectory ??= DefaultCacheDirectory;
        var providers = new List<IModelProvider>();
        if (!string.IsNullOrEmpty(modelDirectory)) providers.Add(new DirectoryModelProvider(modelDirectory));
        if (Environment.GetEnvironmentVariable(ModelDirectoryEnvironmentVariable) is { Length: > 0 } env) providers.Add(new DirectoryModelProvider(env));
        providers.Add(new BundledModelProvider());
        providers.Add(new DirectoryModelProvider(cacheDirectory));
        if (allowDownload) providers.Add(new NuGetModelProvider(cacheDirectory));
        return new ModelStore(providers, verifyChecksums: true, logger);
    }

    /// <summary>Returns a verified local path for <paramref name="model"/>, downloading it when needed.</summary>
    /// <exception cref="ModelNotFoundException">No provider could supply a valid file.</exception>
    public async ValueTask<string> GetModelPathAsync(ModelDescriptor model, IProgress<ModelDownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        var errors = new List<string>();
        foreach (var provider in Providers)
        {
            string? path;
            try
            {
                path = await provider.TryGetModelPathAsync(model, progress, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception e) when (e is HttpRequestException or IOException or InvalidDataException or TaskCanceledException && !cancellationToken.IsCancellationRequested)
            {
                errors.Add($"{provider.Name}: {e.Message}");
                _logger.LogWarning(e, "Model {Model}: provider {Provider} failed", model.Id, provider.Name);
                continue;
            }
            if (path is null) continue;

            if (!VerifyChecksums || await IsValidAsync(path, model, progress, cancellationToken).ConfigureAwait(false))
            {
                _logger.LogDebug("Model {Model} resolved from {Provider}: {Path}", model.Id, provider.Name, path);
                return path;
            }
            errors.Add($"{provider.Name}: checksum mismatch for {path}");
            _logger.LogWarning("Model {Model}: checksum mismatch in {Path}", model.Id, path);
            if (provider.IsRemote) TryDelete(path);
        }
        throw new ModelNotFoundException(
            $"Model '{model.Id}' ({model.FileName}) could not be resolved. Install the '{model.PackageId}' NuGet package, " +
            $"set {ModelDirectoryEnvironmentVariable}, or allow downloads. Tried: {string.Join("; ", Providers.Select(p => p.Name))}." +
            (errors.Count > 0 ? " Errors: " + string.Join("; ", errors) : string.Empty));
    }

    /// <summary>Synchronous variant of <see cref="GetModelPathAsync"/> (may block on a download).</summary>
    public string GetModelPath(ModelDescriptor model) =>
        GetModelPathAsync(model).AsTask().ConfigureAwait(false).GetAwaiter().GetResult();

    /// <summary>Returns the path of a locally available (non-remote) copy, or null. Does not verify.</summary>
    public string? FindLocal(ModelDescriptor model)
    {
        foreach (var provider in Providers.Where(p => !p.IsRemote))
        {
            var path = provider.TryGetModelPathAsync(model, null, CancellationToken.None).AsTask().GetAwaiter().GetResult();
            if (path is not null) return path;
        }
        return null;
    }

    /// <summary>Makes sure every given model is available locally (downloading as needed).</summary>
    public async Task EnsureModelsAsync(IEnumerable<ModelDescriptor> models, IProgress<ModelDownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        foreach (var m in models) await GetModelPathAsync(m, progress, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Computes the lower-case hex SHA-256 of a file.</summary>
    public static async Task<string> ComputeSha256Async(string path, CancellationToken cancellationToken = default)
    {
        await using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, useAsync: true);
        var hash = await SHA256.HashDataAsync(fs, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexStringLower(hash);
    }

    private async ValueTask<bool> IsValidAsync(string path, ModelDescriptor model, IProgress<ModelDownloadProgress>? progress, CancellationToken ct)
    {
        var info = new FileInfo(path);
        if (_verified.TryGetValue(path, out var stamp) && stamp == info.LastWriteTimeUtc) return true;
        if (info.Length != model.SizeBytes) return false;
        progress?.Report(new ModelDownloadProgress(model, info.Length, info.Length, "verifying"));
        var hash = await ComputeSha256Async(path, ct).ConfigureAwait(false);
        if (!string.Equals(hash, model.Sha256, StringComparison.OrdinalIgnoreCase)) return false;
        _verified[path] = info.LastWriteTimeUtc;
        return true;
    }

    private static void TryDelete(string path)
    {
        try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }
}
