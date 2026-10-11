using System.Text.Json;
using Gravicode.HFNet.GraviHub;

namespace Gravicode.HFNet.GraviTransformers.Vision;

/// <summary>
/// A vision encoder's <c>config.json</c>, read into the dimensions the model needs.
/// </summary>
/// <remarks>
/// Separate from <see cref="PretrainedConfig"/> because the two disagree about what a model has: a
/// text encoder is defined by a vocabulary and a maximum position count, a vision encoder by an
/// image size, a patch size and a channel count. Sharing one type would mean four required
/// properties that are meaningless on half the models that use it.
/// </remarks>
public sealed record VisionConfig
{
    private static readonly string[] Supported = ["vit", "deit"];

    /// <summary>The architecture family, such as <c>vit</c>.</summary>
    public required string ModelType { get; init; }

    /// <summary>What the checkpoint declares it is, such as <c>ViTForImageClassification</c>.</summary>
    public IReadOnlyList<string> Architectures { get; init; } = [];

    /// <summary>Width of every hidden state.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>How many encoder blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>How many attention heads per block.</summary>
    public required int Heads { get; init; }

    /// <summary>Width of the feed-forward layer inside each block.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>The square edge length of the input, in pixels.</summary>
    public required int ImageSize { get; init; }

    /// <summary>The square edge length of one patch, in pixels.</summary>
    public required int PatchSize { get; init; }

    /// <summary>How many colour channels the model expects.</summary>
    public int Channels { get; init; } = 3;

    /// <summary>Epsilon inside every layer norm.</summary>
    public double LayerNormEpsilon { get; init; } = 1e-12;

    /// <summary>The feed-forward activation, as <c>hidden_act</c> names it.</summary>
    /// <remarks>
    /// <c>gelu</c> means the exact, erf-based GELU; the tanh approximation is a different name
    /// (<c>gelu_new</c>, <c>gelu_pytorch_tanh</c>). The two differ by up to about 1e-3 per value,
    /// which is enough to move a probability in the third decimal place.
    /// </remarks>
    public string Activation { get; init; } = "gelu";

    /// <summary>Class names by index, where the checkpoint publishes them.</summary>
    public IReadOnlyDictionary<int, string> IdToLabel { get; init; } = new Dictionary<int, string>();

    /// <summary>How many patches one image becomes.</summary>
    public int Patches => Grid * Grid;

    /// <summary>How many patches fit along one edge.</summary>
    public int Grid => ImageSize / PatchSize;

    /// <summary>How many classes the head predicts.</summary>
    public int LabelCount => IdToLabel.Count;

    /// <summary>The name of a class, or its index where the checkpoint names none.</summary>
    public string Label(int index)
        => IdToLabel.TryGetValue(index, out var name) ? name : index.ToString();

    /// <summary>Downloads and reads a repository's <c>config.json</c>.</summary>
    /// <param name="repoId">A model id such as <c>google/vit-base-patch16-224</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static VisionConfig FromPretrained(string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var config = Load(Hub.DownloadFile(repoId, "config.json", revision));
        Require(config, repoId);

        return config;
    }

    /// <summary>Reads a <c>config.json</c> from disk.</summary>
    public static VisionConfig Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        var imageSize = Dimension(root, "image_size", "input_size");
        var patchSize = Dimension(root, "patch_size");

        if (imageSize % patchSize != 0)
        {
            throw new InvalidDataException(
                $"A {imageSize}px image does not divide into {patchSize}px patches. One of the two "
                + "is wrong in the config, and a partial patch is not something to round away.");
        }

        return new VisionConfig
        {
            ModelType = Text(root, "model_type") ?? "vit",
            Architectures = root.TryGetProperty("architectures", out var architectures)
                    && architectures.ValueKind == JsonValueKind.Array
                ? [.. architectures.EnumerateArray().Select(a => a.GetString() ?? "")]
                : [],

            HiddenSize = Dimension(root, "hidden_size"),
            Layers = Dimension(root, "num_hidden_layers"),
            Heads = Dimension(root, "num_attention_heads"),
            IntermediateSize = Dimension(root, "intermediate_size"),
            ImageSize = imageSize,
            PatchSize = patchSize,

            Channels = Optional(root, "num_channels") ?? 3,
            LayerNormEpsilon = root.TryGetProperty("layer_norm_eps", out var epsilon)
                ? epsilon.GetDouble()
                : 1e-12,
            Activation = Text(root, "hidden_act") ?? "gelu",

            IdToLabel = ReadLabels(root),
        };
    }

    /// <summary>Throws unless this is an architecture the vision encoder can actually run.</summary>
    /// <remarks>
    /// The check is on the family, not on the head: a ViT with a detection head still has a ViT
    /// encoder inside it and its features are still readable. What cannot be run is a Swin or a
    /// ConvNeXt, whose blocks are a different shape entirely.
    /// </remarks>
    public static void Require(VisionConfig config, string repoId)
    {
        ArgumentNullException.ThrowIfNull(config);

        if (Supported.Contains(config.ModelType, StringComparer.OrdinalIgnoreCase)) return;

        throw new NotSupportedException(
            $"'{repoId}' is a '{config.ModelType}' model. This runs plain patch-embedding vision "
            + $"transformers ({string.Join(", ", Supported)}); a windowed or convolutional backbone "
            + "needs a different encoder. Export it to ONNX and use GraviOptimum instead.");
    }

    private static int Dimension(JsonElement root, params string[] names)
    {
        var value = Optional(root, names);
        if (value is not null) return value.Value;

        throw new InvalidDataException(
            $"The config declares none of {string.Join(", ", names)}. Guessing a default would "
            + "build a model of the wrong shape, which loads and then returns noise.");
    }

    private static int? Optional(JsonElement root, params string[] names)
    {
        foreach (var name in names)
        {
            if (!root.TryGetProperty(name, out var value)) continue;

            if (value.ValueKind == JsonValueKind.Number) return value.GetInt32();

            // A few exports write image_size as [height, width]; a square model only needs one.
            if (value.ValueKind == JsonValueKind.Array)
            {
                foreach (var element in value.EnumerateArray()) return element.GetInt32();
            }
        }

        return null;
    }

    private static string? Text(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    private static Dictionary<int, string> ReadLabels(JsonElement root)
    {
        var labels = new Dictionary<int, string>();
        if (!root.TryGetProperty("id2label", out var map) || map.ValueKind != JsonValueKind.Object)
        {
            return labels;
        }

        foreach (var entry in map.EnumerateObject())
        {
            if (int.TryParse(entry.Name, out var index))
            {
                labels[index] = entry.Value.GetString() ?? index.ToString();
            }
        }

        return labels;
    }

    /// <inheritdoc />
    public override string ToString()
        => $"{ModelType} ({Layers}L {HiddenSize}H {Heads} heads, {ImageSize}px in "
            + $"{PatchSize}px patches, {LabelCount} classes)";
}
