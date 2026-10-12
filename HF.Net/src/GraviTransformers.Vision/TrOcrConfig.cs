using System.Text.Json;
using Gravicode.HFNet.GraviHub;

namespace Gravicode.HFNet.GraviTransformers.Vision;

/// <summary>The text decoder of a TrOCR checkpoint: the <c>decoder</c> half of its <c>config.json</c>.</summary>
/// <remarks>
/// TrOCR's decoder is RoBERTa-shaped but generates: causal self-attention, then cross-attention to
/// the image, then a feed-forward layer, each followed by its residual and a <b>post</b>-norm. The
/// checkpoints differ in the details this record carries - learned or sinusoidal positions, whether
/// the token vectors are scaled by <c>sqrt(d_model)</c>, a norm over the embeddings or not - and
/// every one of them changes the output while loading the same tensors.
/// </remarks>
public sealed record TrOcrDecoderConfig
{
    /// <summary>How many tokens the output layer scores.</summary>
    public required int VocabularySize { get; init; }

    /// <summary>Width of every hidden state (<c>d_model</c>).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>How many decoder blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Attention heads, in both the self- and the cross-attention.</summary>
    public required int Heads { get; init; }

    /// <summary>Width of the feed-forward layer (<c>decoder_ffn_dim</c>).</summary>
    public required int FeedForwardSize { get; init; }

    /// <summary>Width of the encoder states the cross-attention reads (<c>cross_attention_hidden_size</c>).</summary>
    public required int CrossAttentionSize { get; init; }

    /// <summary>The feed-forward activation (<c>activation_function</c>): <c>relu</c> on trocr-small, <c>gelu</c> on trocr-base.</summary>
    public string Activation { get; init; } = "gelu";

    /// <summary>The longest text the position table covers.</summary>
    public int MaxPositions { get; init; } = 512;

    /// <summary>Whether token vectors are multiplied by <c>sqrt(HiddenSize)</c> before positions are added.</summary>
    public bool ScaleEmbedding { get; init; }

    /// <summary>Whether a layer norm runs over the summed embeddings before the first block.</summary>
    public bool LayerNormEmbedding { get; init; } = true;

    /// <summary>Learned positions (offset by two, as fairseq numbers them) rather than fixed sinusoids.</summary>
    public bool LearnedPositions { get; init; } = true;

    /// <summary>Whether the output layer is the token table itself.</summary>
    public bool TieWordEmbeddings { get; init; } = true;

    /// <summary>The padding id, which also anchors the sinusoidal positions.</summary>
    public int PadTokenId { get; init; } = 1;

    /// <summary>Width of one attention head.</summary>
    public int HeadSize => HiddenSize / Heads;

    internal static TrOcrDecoderConfig Read(JsonElement root)
    {
        var hidden = Required(root, "d_model", "hidden_size");
        var heads = Required(root, "decoder_attention_heads");
        if (hidden % heads != 0)
        {
            throw new InvalidDataException($"d_model {hidden} does not divide into {heads} attention heads.");
        }

        return new TrOcrDecoderConfig
        {
            VocabularySize = Required(root, "vocab_size"),
            HiddenSize = hidden,
            Layers = Required(root, "decoder_layers"),
            Heads = heads,
            FeedForwardSize = Required(root, "decoder_ffn_dim"),
            CrossAttentionSize = Int(root, "cross_attention_hidden_size") ?? hidden,
            Activation = Text(root, "activation_function") ?? "gelu",
            MaxPositions = Int(root, "max_position_embeddings") ?? 512,
            ScaleEmbedding = Bool(root, "scale_embedding") ?? false,
            LayerNormEmbedding = Bool(root, "layernorm_embedding") ?? true,
            LearnedPositions = Bool(root, "use_learned_position_embeddings") ?? true,
            TieWordEmbeddings = Bool(root, "tie_word_embeddings") ?? true,
            PadTokenId = Int(root, "pad_token_id") ?? 1,
        };
    }

    internal static int Required(JsonElement root, params string[] names)
    {
        foreach (var name in names)
        {
            if (Int(root, name) is { } value) return value;
        }

        throw new InvalidDataException(
            $"The decoder config declares none of {string.Join(", ", names)}; a default would build a model of the wrong shape.");
    }

    internal static int? Int(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : null;

    internal static bool? Bool(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? value.GetBoolean()
            : null;

    internal static double? Double(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetDouble() : null;

    internal static string? Text(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
}

/// <summary>
/// A TrOCR checkpoint's configuration: the image encoder, the text decoder, and how text is generated.
/// </summary>
/// <remarks>
/// The generation settings come from <c>generation_config.json</c> where the repository has one, as
/// transformers reads them, falling back to the ids in <c>config.json</c>. A length limit that neither
/// file sets is transformers' own default: 20 new tokens.
/// </remarks>
public sealed record TrOcrConfig
{
    /// <summary>The ViT or DeiT image encoder.</summary>
    public required VisionConfig Encoder { get; init; }

    /// <summary>The text decoder.</summary>
    public required TrOcrDecoderConfig Decoder { get; init; }

    /// <summary>The token every text starts from (<c>decoder_start_token_id</c>) - <c>&lt;/s&gt;</c> on TrOCR, not <c>&lt;s&gt;</c>.</summary>
    public int DecoderStartTokenId { get; init; } = 2;

    /// <summary>The tokens that end a text.</summary>
    public IReadOnlyList<int> EndTokenIds { get; init; } = [2];

    /// <summary>The beginning-of-sequence id.</summary>
    public int BosTokenId { get; init; } = 0;

    /// <summary>The padding id.</summary>
    public int PadTokenId { get; init; } = 1;

    /// <summary>The most tokens to generate after the start token.</summary>
    public int MaxNewTokens { get; init; } = 20;

    /// <summary>Beams for beam search; 1 is greedy decoding.</summary>
    public int Beams { get; init; } = 1;

    /// <summary>The exponent of the length a finished beam's score is divided by.</summary>
    public double LengthPenalty { get; init; } = 1.0;

    /// <summary>Whether beam search stops as soon as there are as many finished texts as beams.</summary>
    public bool EarlyStopping { get; init; }

    /// <summary>Downloads and reads a repository's <c>config.json</c> and <c>generation_config.json</c>.</summary>
    /// <param name="repoId">A model id such as <c>microsoft/trocr-small-printed</c>.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    public static TrOcrConfig FromPretrained(string repoId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        string? generation;
        try { generation = Hub.DownloadFile(repoId, "generation_config.json", revision); }
        catch (HubException) { generation = null; }

        return Load(Hub.DownloadFile(repoId, "config.json", revision), generation, repoId);
    }

    /// <summary>Reads the two files from disk.</summary>
    /// <param name="configPath">The model's <c>config.json</c>.</param>
    /// <param name="generationPath">Its <c>generation_config.json</c>, or <c>null</c> for none.</param>
    /// <param name="source">Named in errors.</param>
    public static TrOcrConfig Load(string configPath, string? generationPath = null, string? source = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(configPath);
        source ??= configPath;

        using var document = JsonDocument.Parse(File.ReadAllText(configPath));
        var root = document.RootElement;

        if (!root.TryGetProperty("encoder", out var encoderNode) || !root.TryGetProperty("decoder", out var decoderNode))
        {
            throw new NotSupportedException(
                $"'{source}' is not a vision encoder-decoder: its config has no 'encoder' and 'decoder'. "
                + "Use VisionTransformer for a plain image encoder.");
        }

        var decoderType = TrOcrDecoderConfig.Text(decoderNode, "model_type");
        if (decoderType is not null && decoderType != "trocr")
        {
            throw new NotSupportedException(
                $"'{source}' pairs its image encoder with a '{decoderType}' decoder. Only TrOCR's decoder is supported.");
        }

        var encoder = VisionConfig.Read(encoderNode);
        VisionConfig.Require(encoder, source);
        var decoder = TrOcrDecoderConfig.Read(decoderNode);

        // Token ids: the top level, then the decoder's own, then TrOCR's defaults.
        int Id(string name, int fallback)
            => TrOcrDecoderConfig.Int(root, name) ?? TrOcrDecoderConfig.Int(decoderNode, name) ?? fallback;

        var config = new TrOcrConfig
        {
            Encoder = encoder,
            Decoder = decoder,
            DecoderStartTokenId = Id("decoder_start_token_id", 2),
            EndTokenIds = EndTokens(root) ?? EndTokens(decoderNode) ?? [2],
            BosTokenId = Id("bos_token_id", 0),
            PadTokenId = Id("pad_token_id", 1),
        };

        return generationPath is not null && File.Exists(generationPath) ? config.WithGeneration(generationPath) : config;
    }

    /// <summary>Applies a <c>generation_config.json</c>, which overrides <c>config.json</c> wherever it speaks.</summary>
    private TrOcrConfig WithGeneration(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        // max_new_tokens wins over max_length, which counts the start token too.
        var maxNew = TrOcrDecoderConfig.Int(root, "max_new_tokens")
            ?? (TrOcrDecoderConfig.Int(root, "max_length") is { } total ? total - 1 : MaxNewTokens);

        return this with
        {
            DecoderStartTokenId = TrOcrDecoderConfig.Int(root, "decoder_start_token_id") ?? DecoderStartTokenId,
            EndTokenIds = EndTokens(root) ?? EndTokenIds,
            BosTokenId = TrOcrDecoderConfig.Int(root, "bos_token_id") ?? BosTokenId,
            PadTokenId = TrOcrDecoderConfig.Int(root, "pad_token_id") ?? PadTokenId,
            MaxNewTokens = Math.Max(1, maxNew),
            Beams = TrOcrDecoderConfig.Int(root, "num_beams") ?? Beams,
            LengthPenalty = TrOcrDecoderConfig.Double(root, "length_penalty") ?? LengthPenalty,
            EarlyStopping = TrOcrDecoderConfig.Bool(root, "early_stopping") ?? EarlyStopping,
        };
    }

    private static int[]? EndTokens(JsonElement root)
    {
        if (!root.TryGetProperty("eos_token_id", out var value)) return null;

        return value.ValueKind switch
        {
            JsonValueKind.Number => [value.GetInt32()],
            JsonValueKind.Array => [.. value.EnumerateArray().Select(e => e.GetInt32())],
            _ => null,
        };
    }

    /// <inheritdoc />
    public override string ToString()
        => $"TrOCR ({Encoder.ModelType} {Encoder.Layers}L {Encoder.HiddenSize}H encoder, "
            + $"{Decoder.Layers}L {Decoder.HiddenSize}H decoder, vocab {Decoder.VocabularySize})";
}
