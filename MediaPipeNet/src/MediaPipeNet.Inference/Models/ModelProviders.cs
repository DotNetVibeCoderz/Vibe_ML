using System.IO.Compression;
using System.Reflection;

namespace MediaPipeNet.Inference.Models;

/// <summary>
/// A place models can come from: a local directory, the application's output folder, embedded
/// resources, a model cache fed from NuGet, an HTTP mirror, ...
/// </summary>
public interface IModelProvider
{
    /// <summary>Short description used in logs and error messages.</summary>
    string Name { get; }

    /// <summary>True when the provider may hit the network.</summary>
    bool IsRemote { get; }

    /// <summary>
    /// Returns a local file path for <paramref name="model"/>, fetching it if necessary, or null when
    /// this provider does not have it. Integrity is verified by <see cref="ModelStore"/>.
    /// </summary>
    ValueTask<string?> TryGetModelPathAsync(ModelDescriptor model, IProgress<ModelDownloadProgress>? progress, CancellationToken cancellationToken);
}

/// <summary>Looks for models in a directory (e.g. a folder you manage yourself).</summary>
/// <param name="directory">The directory holding <c>*.onnx</c> files.</param>
public sealed class DirectoryModelProvider(string directory) : IModelProvider
{
    /// <summary>The directory searched.</summary>
    public string Directory { get; } = directory ?? throw new ArgumentNullException(nameof(directory));

    /// <inheritdoc />
    public string Name => $"directory '{Directory}'";

    /// <inheritdoc />
    public bool IsRemote => false;

    /// <inheritdoc />
    public ValueTask<string?> TryGetModelPathAsync(ModelDescriptor model, IProgress<ModelDownloadProgress>? progress, CancellationToken cancellationToken)
    {
        var path = Path.Combine(Directory, model.FileName);
        return ValueTask.FromResult(File.Exists(path) ? path : null);
    }
}

/// <summary>
/// Finds models copied next to the application by the <c>Gravicode.MediaPipeNet.Models.*</c> NuGet packages
/// (<c>&lt;app&gt;/models/*.onnx</c>).
/// </summary>
public sealed class BundledModelProvider : IModelProvider
{
    /// <summary>Folder name (under the application base directory) used by the model packages.</summary>
    public const string FolderName = "models";

    /// <inheritdoc />
    public string Name => $"bundled '{Path.Combine(AppContext.BaseDirectory, FolderName)}'";

    /// <inheritdoc />
    public bool IsRemote => false;

    /// <inheritdoc />
    public ValueTask<string?> TryGetModelPathAsync(ModelDescriptor model, IProgress<ModelDownloadProgress>? progress, CancellationToken cancellationToken)
    {
        var path = Path.Combine(AppContext.BaseDirectory, FolderName, model.FileName);
        return ValueTask.FromResult(File.Exists(path) ? path : null);
    }
}

/// <summary>
/// Extracts models embedded as manifest resources (resource name ending in the model file name)
/// into a cache directory.
/// </summary>
/// <param name="assembly">Assembly containing the resources.</param>
/// <param name="cacheDirectory">Where extracted files are written.</param>
public sealed class EmbeddedResourceModelProvider(Assembly assembly, string cacheDirectory) : IModelProvider
{
    /// <inheritdoc />
    public string Name => $"embedded resources of {assembly.GetName().Name}";

    /// <inheritdoc />
    public bool IsRemote => false;

    /// <inheritdoc />
    public async ValueTask<string?> TryGetModelPathAsync(ModelDescriptor model, IProgress<ModelDownloadProgress>? progress, CancellationToken cancellationToken)
    {
        var resource = assembly.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(model.FileName, StringComparison.OrdinalIgnoreCase));
        if (resource is null) return null;
        var target = Path.Combine(cacheDirectory, model.FileName);
        if (File.Exists(target) && new FileInfo(target).Length == model.SizeBytes) return target;
        System.IO.Directory.CreateDirectory(cacheDirectory);
        await using var src = assembly.GetManifestResourceStream(resource)!;
        await FileUtil.WriteAtomicallyAsync(target, src, cancellationToken).ConfigureAwait(false);
        return target;
    }
}

/// <summary>
/// Downloads model files from an HTTP mirror (<c>{baseUrl}/{fileName}</c>) into a cache directory.
/// </summary>
public sealed class HttpModelProvider : IModelProvider
{
    private readonly Uri _baseUrl;
    private readonly string _cacheDirectory;
    private readonly HttpClient _http;

    /// <summary>Creates the provider.</summary>
    public HttpModelProvider(Uri baseUrl, string cacheDirectory, HttpClient? httpClient = null)
    {
        _baseUrl = baseUrl ?? throw new ArgumentNullException(nameof(baseUrl));
        _cacheDirectory = cacheDirectory;
        _http = httpClient ?? ModelStore.SharedHttpClient;
    }

    /// <inheritdoc />
    public string Name => $"http mirror {_baseUrl}";

    /// <inheritdoc />
    public bool IsRemote => true;

    /// <inheritdoc />
    public async ValueTask<string?> TryGetModelPathAsync(ModelDescriptor model, IProgress<ModelDownloadProgress>? progress, CancellationToken cancellationToken)
    {
        var target = Path.Combine(_cacheDirectory, model.FileName);
        if (File.Exists(target)) return target;
        var url = new Uri(_baseUrl.ToString().TrimEnd('/') + "/" + model.FileName);
        using var response = await _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode) return null;
        System.IO.Directory.CreateDirectory(_cacheDirectory);
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        await FileUtil.CopyWithProgressAsync(stream, target, model, response.Content.Headers.ContentLength, progress, cancellationToken).ConfigureAwait(false);
        return target;
    }
}

/// <summary>
/// Downloads the <c>Gravicode.MediaPipeNet.Models.*</c> package that ships a model from a NuGet v3 feed
/// (nuget.org by default), extracts the model files into the cache directory and returns the path.
/// Every model of the package is extracted, so sibling models need no further download.
/// </summary>
public sealed class NuGetModelProvider : IModelProvider
{
    /// <summary>The nuget.org flat-container endpoint.</summary>
    public static readonly Uri NuGetOrgFlatContainer = new("https://api.nuget.org/v3-flatcontainer/");

    private readonly Uri _flatContainer;
    private readonly string _cacheDirectory;
    private readonly string _version;
    private readonly HttpClient _http;
    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>Creates the provider.</summary>
    /// <param name="cacheDirectory">Where model files are extracted.</param>
    /// <param name="packageVersion">Model package version; defaults to this library's version.</param>
    /// <param name="flatContainer">NuGet v3 flat-container base URL.</param>
    /// <param name="httpClient">HTTP client to use.</param>
    public NuGetModelProvider(string cacheDirectory, string? packageVersion = null, Uri? flatContainer = null, HttpClient? httpClient = null)
    {
        _cacheDirectory = cacheDirectory;
        _version = packageVersion ?? ModelStore.LibraryVersion;
        _flatContainer = flatContainer ?? NuGetOrgFlatContainer;
        _http = httpClient ?? ModelStore.SharedHttpClient;
    }

    /// <inheritdoc />
    public string Name => $"NuGet feed {_flatContainer} (version {_version})";

    /// <inheritdoc />
    public bool IsRemote => true;

    /// <inheritdoc />
    public async ValueTask<string?> TryGetModelPathAsync(ModelDescriptor model, IProgress<ModelDownloadProgress>? progress, CancellationToken cancellationToken)
    {
        var target = Path.Combine(_cacheDirectory, model.FileName);
        if (File.Exists(target)) return target;

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (File.Exists(target)) return target;
            string id = model.PackageId.ToLowerInvariant(), ver = _version.ToLowerInvariant();
            var url = new Uri(_flatContainer, $"{id}/{ver}/{id}.{ver}.nupkg");
            using var response = await _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode) return null;

            System.IO.Directory.CreateDirectory(_cacheDirectory);
            var nupkg = Path.Combine(_cacheDirectory, $"{id}.{ver}.nupkg.download");
            await using (var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
            {
                await FileUtil.CopyWithProgressAsync(stream, nupkg, model, response.Content.Headers.ContentLength, progress, cancellationToken).ConfigureAwait(false);
            }
            try
            {
                progress?.Report(new ModelDownloadProgress(model, 0, null, "extracting"));
                using var zip = ZipFile.OpenRead(nupkg);
                foreach (var entry in zip.Entries)
                {
                    if (!entry.FullName.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)) continue;
                    var dest = Path.Combine(_cacheDirectory, entry.Name);
                    if (File.Exists(dest)) continue;
                    await using var es = entry.Open();
                    await FileUtil.WriteAtomicallyAsync(dest, es, cancellationToken).ConfigureAwait(false);
                }
            }
            finally
            {
                File.Delete(nupkg);
            }
            return File.Exists(target) ? target : null;
        }
        finally
        {
            _gate.Release();
        }
    }
}

internal static class FileUtil
{
    public static async Task WriteAtomicallyAsync(string target, Stream source, CancellationToken ct)
    {
        var temp = target + "." + Guid.NewGuid().ToString("N")[..8] + ".tmp";
        try
        {
            await using (var fs = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1 << 16, useAsync: true))
                await source.CopyToAsync(fs, ct).ConfigureAwait(false);
            File.Move(temp, target, overwrite: true);
        }
        finally
        {
            if (File.Exists(temp)) File.Delete(temp);
        }
    }

    public static async Task CopyWithProgressAsync(Stream source, string target, ModelDescriptor model, long? total, IProgress<ModelDownloadProgress>? progress, CancellationToken ct)
    {
        var temp = target + ".part";
        var buffer = new byte[1 << 16];
        long received = 0;
        try
        {
            await using (var fs = new FileStream(temp, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16, useAsync: true))
            {
                int read;
                while ((read = await source.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
                {
                    await fs.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                    received += read;
                    progress?.Report(new ModelDownloadProgress(model, received, total, "downloading"));
                }
            }
            File.Move(temp, target, overwrite: true);
        }
        finally
        {
            if (File.Exists(temp)) File.Delete(temp);
        }
    }
}
