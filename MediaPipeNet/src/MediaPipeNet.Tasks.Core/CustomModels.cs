using MediaPipeNet.Inference;
using MediaPipeNet.Inference.Models;
using Microsoft.Extensions.Logging;

namespace MediaPipeNet.Tasks;

/// <summary>
/// Support for your own models (for example MediaPipe Model Maker exports converted with
/// <c>tools/model-conversion/convert_models.py --custom my_model.tflite</c>): tasks that take a
/// <c>ModelPath</c> load that file instead of the catalog model, and read their labels from the
/// <c>Labels</c> option or from the <c>{model}.labels.txt</c> file written by the converter.
/// </summary>
public static class CustomModels
{
    /// <summary>Loads <paramref name="customModelPath"/> when set, otherwise the catalog <paramref name="model"/>.</summary>
    public static async ValueTask<OnnxModel> LoadAsync(BaseOptions options, ModelDescriptor model, string? customModelPath, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (string.IsNullOrEmpty(customModelPath)) return await ModelLoader.LoadAsync(options, model, cancellationToken).ConfigureAwait(false);
        if (!File.Exists(customModelPath)) throw new ModelNotFoundException($"Custom model not found: {customModelPath}");
        return OnnxModel.Load(customModelPath, options.Inference, options.LoggerFactory?.CreateLogger("MediaPipeNet.Inference"),
            Path.GetFileNameWithoutExtension(customModelPath));
    }

    /// <summary>Synchronous variant of <see cref="LoadAsync"/>.</summary>
    public static OnnxModel Load(BaseOptions options, ModelDescriptor model, string? customModelPath) =>
        LoadAsync(options, model, customModelPath).AsTask().ConfigureAwait(false).GetAwaiter().GetResult();

    /// <summary>
    /// The labels to use: <paramref name="labels"/> when given; for a custom model the companion
    /// <c>{model}.labels.txt</c> when it exists; otherwise <paramref name="builtIn"/>.
    /// </summary>
    public static IReadOnlyList<string> ResolveLabels(IReadOnlyList<string>? labels, string? customModelPath, IReadOnlyList<string> builtIn)
    {
        if (labels is not null) return labels;
        if (!string.IsNullOrEmpty(customModelPath) && CompanionFile(customModelPath, "labels") is { } file) return ReadLines(file);
        return builtIn;
    }

    /// <summary>Path of <c>{model}.{kind}.txt</c> next to a model file, or null.</summary>
    public static string? CompanionFile(string modelPath, string kind)
    {
        ArgumentNullException.ThrowIfNull(modelPath);
        var path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", $"{Path.GetFileNameWithoutExtension(modelPath)}.{kind}.txt");
        return File.Exists(path) ? path : null;
    }

    /// <summary>Reads a label or vocabulary file (one entry per line; trailing empty lines ignored).</summary>
    public static string[] ReadLines(string path)
    {
        var lines = File.ReadAllLines(path);
        int n = lines.Length;
        while (n > 0 && lines[n - 1].Length == 0) n--;
        return lines[..n];
    }
}
