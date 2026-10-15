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

    /// <summary>The ESM-specific settings, present only when <see cref="ModelType"/> is <c>esm</c>.</summary>
    public EsmOptions? Esm { get; init; }

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

        var esm = string.Equals(modelType, "esm", StringComparison.OrdinalIgnoreCase)
            ? new EsmOptions
            {
                PositionEmbeddingType = ReadString(root, "position_embedding_type") ?? "absolute",
                TokenDropout = ReadBool(root, "token_dropout") ?? false,
                MaskTokenId = ReadInt(root, "mask_token_id") ?? 32,
                PadTokenId = ReadInt(root, "pad_token_id") ?? 1,
                EmbeddingNormBefore = ReadBool(root, "emb_layer_norm_before") ?? false,
                RopeTheta = ReadDouble(root, "rope_theta") ?? 10000.0,
                BiasedFeedForward = ReadBool(root, "add_bias_fnn") ?? true,
                IsFoldingModel = ReadBool(root, "is_folding_model") ?? false,
            }
            : null;

        return new PretrainedConfig
        {
            ModelType = modelType,
            Esm = esm,
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
            PositionOffset = modelType is "roberta" or "xlm-roberta" or "camembert" ? 2
                : esm is { PositionEmbeddingType: "absolute" } ? esm.PadTokenId + 1
                : 0,
        };
    }

    private static bool? ReadBool(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? value.GetBoolean()
            : null;

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

/// <summary>
/// The parts of an ESM <c>config.json</c> that a BERT config does not have.
/// </summary>
/// <remarks>
/// ESM (ESM-1b, ESM-1v, ESM-2, the Nucleotide Transformer) shares BERT's parameter names but not its
/// forward pass: it is pre-norm, ESM-2 rotates queries and keys instead of adding position embeddings,
/// and masked tokens are zeroed and the rest rescaled. Each of those is read from here rather than
/// guessed, because each one, wrong, still gives plausible numbers.
/// </remarks>
public sealed record EsmOptions
{
    /// <summary><c>rotary</c> (ESM-2, Nucleotide Transformer v2) or <c>absolute</c> (ESM-1b, Nucleotide Transformer v1).</summary>
    public string PositionEmbeddingType { get; init; } = "absolute";

    /// <summary>
    /// Whether <c>&lt;mask&gt;</c> embeddings are zeroed and the rest scaled by
    /// <c>(1 - 0.15 * 0.8) / (1 - masked / tokens)</c>, as ESM was trained.
    /// </summary>
    public bool TokenDropout { get; init; }

    /// <summary>The id of <c>&lt;mask&gt;</c>.</summary>
    public int MaskTokenId { get; init; } = 32;

    /// <summary>The id of <c>&lt;pad&gt;</c>; absolute positions start at one past it.</summary>
    public int PadTokenId { get; init; } = 1;

    /// <summary>Whether a layer norm runs on the embeddings before the first layer (ESM-1b).</summary>
    public bool EmbeddingNormBefore { get; init; }

    /// <summary>The rotary base, 10000 on every published ESM.</summary>
    public double RopeTheta { get; init; } = 10000.0;

    /// <summary>False for the Nucleotide Transformer v2's gated (SwiGLU) feed-forward, which is not supported.</summary>
    public bool BiasedFeedForward { get; init; } = true;

    /// <summary>True for ESMFold, a structure model rather than a language model; not supported.</summary>
    public bool IsFoldingModel { get; init; }
}
