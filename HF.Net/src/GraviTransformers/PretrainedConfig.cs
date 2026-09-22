using System.Text.Json;
using Gravicode.HFNet.GraviHub;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>
/// A model's <c>config.json</c>: the architecture, its dimensions, and its label names.
/// </summary>
/// <remarks>
/// Field names differ between families for the same quantity - BERT writes
/// <c>num_hidden_layers</c>, DistilBERT writes <c>n_layers</c>, GPT-2 writes <c>n_layer</c> - so
/// each dimension is read from a list of aliases rather than from one key. Defaulting a missing
/// dimension would be worse than failing: a model built with twelve layers where the checkpoint has
/// six loads "successfully" and returns noise.
/// </remarks>
public sealed record PretrainedConfig
{
    /// <summary>The architecture family, for example <c>bert</c>, <c>roberta</c> or <c>distilbert</c>.</summary>
    public required string ModelType { get; init; }

    /// <summary>The architecture class names the checkpoint declares, when it declares any.</summary>
    public IReadOnlyList<string> Architectures { get; init; } = [];

    /// <summary>Width of the hidden states.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of encoder layers.</summary>
    public required int Layers { get; init; }

    /// <summary>Number of attention heads per layer.</summary>
    public required int Heads { get; init; }

    /// <summary>Width of the feed-forward block's inner layer.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Number of entries in the vocabulary.</summary>
    public required int VocabularySize { get; init; }

    /// <summary>How many position embeddings the model has, which caps the input length.</summary>
    public int MaxPositions { get; init; } = 512;

    /// <summary>Epsilon used by the layer norms.</summary>
    public double LayerNormEpsilon { get; init; } = 1e-12;

    /// <summary>The activation in the feed-forward block.</summary>
    public string Activation { get; init; } = "gelu";

    /// <summary>Class index to label name, when the checkpoint has a classification head.</summary>
    public IReadOnlyDictionary<int, string> IdToLabel { get; init; } = new Dictionary<int, string>();

    /// <summary>
    /// The offset added to position ids before looking up a position embedding.
    /// </summary>
    /// <remarks>
    /// RoBERTa reserves the first two position slots for padding and starts real positions at 2,
    /// which is why its <c>max_position_embeddings</c> is 514 rather than 512. Ignoring the offset
    /// shifts every position embedding by two and degrades output without any shape mismatch to
    /// catch it.
    /// </remarks>
    public int PositionOffset { get; init; }

    /// <summary>Number of classes, or 0 when the model has no classification head.</summary>
    public int LabelCount => IdToLabel.Count;

    /// <summary>Hidden size per attention head.</summary>
    public int HeadSize => HiddenSize / Heads;

    /// <summary>Reads a <c>config.json</c> from disk.</summary>
    /// <param name="path">Path to the file.</param>
    public static PretrainedConfig Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        var modelType = ReadString(root, "model_type") ?? "bert";
        var hidden = ReadInt(root, "hidden_size", "dim", "d_model", "n_embd")
            ?? throw new InvalidDataException($"'{path}' declares no hidden size.");

        var labels = new Dictionary<int, string>();
        if (root.TryGetProperty("id2label", out var id2label) && id2label.ValueKind == JsonValueKind.Object)
        {
            foreach (var entry in id2label.EnumerateObject())
            {
                if (int.TryParse(entry.Name, out var index)) labels[index] = entry.Value.GetString() ?? entry.Name;
            }
        }

        return new PretrainedConfig
        {
            ModelType = modelType,
            Architectures = root.TryGetProperty("architectures", out var architectures)
                && architectures.ValueKind == JsonValueKind.Array
                ? [.. architectures.EnumerateArray().Select(a => a.GetString() ?? "")]
                : [],
            HiddenSize = hidden,
            Layers = ReadInt(root, "num_hidden_layers", "n_layers", "n_layer", "num_layers")
                ?? throw new InvalidDataException($"'{path}' declares no layer count."),
            Heads = ReadInt(root, "num_attention_heads", "n_heads", "n_head")
                ?? throw new InvalidDataException($"'{path}' declares no head count."),

            // DistilBERT calls it hidden_dim; where nothing is declared, four times the hidden size
            // is the convention every BERT-family model follows.
            IntermediateSize = ReadInt(root, "intermediate_size", "hidden_dim", "n_inner") ?? hidden * 4,
            VocabularySize = ReadInt(root, "vocab_size")
                ?? throw new InvalidDataException($"'{path}' declares no vocabulary size."),
            MaxPositions = ReadInt(root, "max_position_embeddings", "n_positions", "max_seq_length") ?? 512,
            LayerNormEpsilon = ReadDouble(root, "layer_norm_eps", "layer_norm_epsilon") ?? 1e-12,
            Activation = ReadString(root, "hidden_act", "activation", "activation_function") ?? "gelu",
            IdToLabel = labels,
            PositionOffset = modelType is "roberta" or "xlm-roberta" or "camembert" ? 2 : 0,
        };
    }

    /// <summary>Downloads and reads a Hub model's <c>config.json</c>.</summary>
    /// <param name="repoId">A model id.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static PretrainedConfig FromPretrained(string repoId, string revision = "main")
        => Load(Hub.DownloadFile(repoId, "config.json", revision));

    private static int? ReadInt(JsonElement root, params string[] names)
    {
        foreach (var name in names)
        {
            if (root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number
                && value.TryGetInt32(out var number))
            {
                return number;
            }
        }
        return null;
    }

    private static double? ReadDouble(JsonElement root, params string[] names)
    {
        foreach (var name in names)
        {
            if (root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number)
            {
                return value.GetDouble();
            }
        }
        return null;
    }

    private static string? ReadString(JsonElement root, params string[] names)
    {
        foreach (var name in names)
        {
            if (root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String)
            {
                return value.GetString();
            }
        }
        return null;
    }

    /// <inheritdoc />
    public override string ToString()
        => $"{ModelType} ({Layers}L x {HiddenSize}H x {Heads} heads, vocab {VocabularySize:N0}"
            + (LabelCount > 0 ? $", {LabelCount} labels)" : ")");
}
