using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using Microsoft.Extensions.Logging;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Resolves and loads task models according to <see cref="BaseOptions"/>.</summary>
public static class ModelLoader
{
    /// <summary>Resolves the file of <paramref name="model"/>: explicit path → model directory → model store.</summary>
    public static async ValueTask<string> ResolvePathAsync(BaseOptions options, ModelDescriptor model, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(model);
        if (options.ModelPaths is not null && options.ModelPaths.TryGetValue(model.Id, out var explicitPath))
            return File.Exists(explicitPath) ? explicitPath : throw new ModelNotFoundException($"Model file for '{model.Id}' not found: {explicitPath}");
        if (options.ModelDirectory is not null)
        {
            var p = Path.Combine(options.ModelDirectory, model.FileName);
            if (File.Exists(p)) return p;
        }
        return await (options.ModelStore ?? ModelStore.Default).GetModelPathAsync(model, null, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Loads <paramref name="model"/> (resolving and possibly downloading it first).</summary>
    public static async ValueTask<OnnxModel> LoadAsync(BaseOptions options, ModelDescriptor model, CancellationToken cancellationToken = default)
    {
        var path = await ResolvePathAsync(options, model, cancellationToken).ConfigureAwait(false);
        return OnnxModel.Load(path, options.Inference, options.LoggerFactory?.CreateLogger("MediaPipeNet.Inference"), model.Id);
    }

    /// <summary>Synchronous variant of <see cref="LoadAsync"/>.</summary>
    public static OnnxModel Load(BaseOptions options, ModelDescriptor model) =>
        LoadAsync(options, model).AsTask().ConfigureAwait(false).GetAwaiter().GetResult();
}
