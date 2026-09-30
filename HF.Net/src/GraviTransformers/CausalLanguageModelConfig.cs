using System.Text.Json;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>The decoder block families <see cref="CausalLanguageModel"/> runs.</summary>
public enum DecoderFamily
{
    /// <summary>GPT-2: learned positions, LayerNorm, GELU feed-forward, <c>Conv1D</c> weights.</summary>
    Gpt2,

    /// <summary>
    /// Llama and its descendants - Mistral, Qwen2, Qwen3, SmolLM, TinyLlama: rotary positions, RMSNorm,
    /// a gated SiLU feed-forward, and grouped-query attention.
    /// </summary>
    Llama,

    /// <summary>GPT-NeoX and Pythia: rotary positions on part of each head, LayerNorm, parallel residual.</summary>
    GptNeoX,
}

/// <summary>A decoder checkpoint's <c>config.json</c>, read into what the forward pass needs.</summary>
public sealed record CausalLanguageModelConfig
{
    /// <summary>The <c>model_type</c> the file declares.</summary>
    public required string ModelType { get; init; }

    /// <summary>Which block the layers are.</summary>
    public required DecoderFamily Family { get; init; }

    /// <summary>Token vocabulary size.</summary>
    public required int VocabularySize { get; init; }

    /// <summary>How many tokens the model can attend over, prompt and continuation together.</summary>
    public required int MaxPositions { get; init; }

    /// <summary>Width of every hidden state.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>How many blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Query heads per block.</summary>
    public required int Heads { get; init; }

    /// <summary>
    /// Key and value heads per block. Fewer than <see cref="Heads"/> is grouped-query attention: each
    /// key/value head serves a group of query heads, and the cache is that much smaller.
    /// </summary>
    public int KeyValueHeads { get; init; }

    /// <summary>Width of one head.</summary>
    public int HeadSize { get; init; }

    /// <summary>Width of the feed-forward layer.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>The feed-forward activation: <c>gelu_new</c> for GPT-2, <c>silu</c> for Llama, <c>gelu</c> for Pythia.</summary>
    public required string Activation { get; init; }

    /// <summary>Epsilon inside every norm.</summary>
    public required double LayerNormEpsilon { get; init; }

    /// <summary>The end-of-text tokens generation stops at; the first is <see cref="EndTokenId"/>.</summary>
    public IReadOnlyList<int> EndTokenIds { get; init; } = [];

    /// <summary>The first end-of-text token, or <c>null</c> when the config names none.</summary>
    public int? EndTokenId => EndTokenIds.Count > 0 ? EndTokenIds[0] : null;

    /// <summary>How many dimensions of each head are rotated; the rest pass through. All of them, except on Pythia.</summary>
    public int RotaryDimensions { get; init; }

    /// <summary>The rotary base, <c>rope_theta</c>.</summary>
    public double RopeTheta { get; init; } = 10000;

    /// <summary>The rotary scaling, <c>rope_type</c>: <c>default</c>, <c>linear</c> or <c>llama3</c>.</summary>
    public string RopeType { get; init; } = "default";

    /// <summary>The rotary scaling's parameters - <c>factor</c>, <c>low_freq_factor</c> and so on.</summary>
    public IReadOnlyDictionary<string, double> RopeParameters { get; init; } = new Dictionary<string, double>();

    /// <summary>Whether the query, key and value projections have biases - Qwen2's do.</summary>
    public bool AttentionBias { get; init; }

    /// <summary>Whether the attention output projection has a bias.</summary>
    public bool OutputBias { get; init; }

    /// <summary>Whether the feed-forward projections have biases.</summary>
    public bool MlpBias { get; init; }

    /// <summary>Whether each head's query and key are RMS-normalized before rotation - Qwen3's addition.</summary>
    public bool QueryKeyNorm { get; init; }

    /// <summary>Mistral's sliding window: each token sees at most this many, itself included. <c>null</c> for none.</summary>
    public int? SlidingWindow { get; init; }

    /// <summary>Pythia's parallel residual: attention and feed-forward both read the block's input.</summary>
    public bool ParallelResidual { get; init; }

    /// <summary>Whether the output layer is the token embedding table.</summary>
    public bool TieEmbeddings { get; init; }

    /// <summary>The <c>model_type</c> values this runs.</summary>
    public static IReadOnlySet<string> Supported { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "gpt2", "llama", "mistral", "qwen2", "qwen3", "gpt_neox",
    };

    /// <summary>Reads a <c>config.json</c>, and a <c>generation_config.json</c> beside it for extra end tokens.</summary>
    /// <exception cref="NotSupportedException">It describes a model this does not run.</exception>
    public static CausalLanguageModelConfig Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        var type = Text(root, "model_type") ?? "";
        if (!Supported.Contains(type))
        {
            throw new NotSupportedException(
                $"This is a '{type}' model. CausalLanguageModel runs GPT-2, Llama, Mistral, Qwen2, Qwen3 and "
                + "GPT-NeoX (Pythia) checkpoints and their fine-tunes. Other decoders - Gemma, Phi-3, Falcon, "
                + "mixture-of-experts models - have different blocks; export them to ONNX and use GraviOptimum.");
        }

        var end = EndTokens(root);
        var generation = Path.Combine(Path.GetDirectoryName(path) ?? ".", "generation_config.json");
        if (File.Exists(generation))
        {
            using var extra = JsonDocument.Parse(File.ReadAllText(generation));
            foreach (var id in EndTokens(extra.RootElement)) if (!end.Contains(id)) end.Add(id);
        }

        return type.ToLowerInvariant() switch
        {
            "gpt2" => Gpt2(root, type, end),
            "gpt_neox" => NeoX(root, type, end),
            _ => Llama(root, type, end),
        };
    }

    private static CausalLanguageModelConfig Gpt2(JsonElement root, string type, List<int> end)
    {
        if (Flag(root, "scale_attn_by_inverse_layer_idx"))
        {
            throw new NotSupportedException("scale_attn_by_inverse_layer_idx is set; this runs GPT-2's standard attention scaling only.");
        }

        var hidden = Int(root, "n_embd");
        var heads = Int(root, "n_head");
        return new CausalLanguageModelConfig
        {
            ModelType = type,
            Family = DecoderFamily.Gpt2,
            VocabularySize = Int(root, "vocab_size"),
            MaxPositions = Int(root, "n_positions"),
            HiddenSize = hidden,
            Layers = Int(root, "n_layer"),
            Heads = heads,
            KeyValueHeads = heads,
            HeadSize = hidden / heads,
            IntermediateSize = OptionalInt(root, "n_inner") ?? 4 * hidden,
            Activation = Text(root, "activation_function") ?? "gelu_new",
            LayerNormEpsilon = OptionalDouble(root, "layer_norm_epsilon") ?? 1e-5,
            EndTokenIds = end,
            TieEmbeddings = true,
        };
    }

    private static CausalLanguageModelConfig Llama(JsonElement root, string type, List<int> end)
    {
        var hidden = Int(root, "hidden_size");
        var heads = Int(root, "num_attention_heads");
        var headSize = OptionalInt(root, "head_dim") ?? hidden / heads;
        var qwen2 = type.Equals("qwen2", StringComparison.OrdinalIgnoreCase);
        var qwen3 = type.Equals("qwen3", StringComparison.OrdinalIgnoreCase);

        var (theta, ropeType, ropeParameters, partial) = Rope(root);

        // Qwen2 always has query/key/value biases; Llama's and Qwen3's follow attention_bias. The
        // sliding window only applies when a model uses it - Mistral always, Qwen2 when told to.
        var window = OptionalInt(root, "sliding_window");
        if (type.StartsWith("qwen", StringComparison.OrdinalIgnoreCase) && !Flag(root, "use_sliding_window")) window = null;

        return new CausalLanguageModelConfig
        {
            ModelType = type,
            Family = DecoderFamily.Llama,
            VocabularySize = Int(root, "vocab_size"),
            MaxPositions = OptionalInt(root, "max_position_embeddings") ?? 4096,
            HiddenSize = hidden,
            Layers = Int(root, "num_hidden_layers"),
            Heads = heads,
            KeyValueHeads = OptionalInt(root, "num_key_value_heads") ?? heads,
            HeadSize = headSize,
            IntermediateSize = Int(root, "intermediate_size"),
            Activation = Text(root, "hidden_act") ?? "silu",
            LayerNormEpsilon = OptionalDouble(root, "rms_norm_eps") ?? 1e-6,
            EndTokenIds = end,
            RotaryDimensions = (int)(headSize * partial),
            RopeTheta = theta,
            RopeType = ropeType,
            RopeParameters = ropeParameters,
            AttentionBias = qwen2 || Flag(root, "attention_bias"),
            OutputBias = !qwen2 && Flag(root, "attention_bias"),
            MlpBias = Flag(root, "mlp_bias"),
            QueryKeyNorm = qwen3,
            SlidingWindow = window,
            TieEmbeddings = Flag(root, "tie_word_embeddings"),
        };
    }

    private static CausalLanguageModelConfig NeoX(JsonElement root, string type, List<int> end)
    {
        var hidden = Int(root, "hidden_size");
        var heads = Int(root, "num_attention_heads");
        var headSize = hidden / heads;
        var (theta, ropeType, ropeParameters, partial) = Rope(root);
        var rotaryPercent = OptionalDouble(root, "rotary_pct") ?? partial;

        return new CausalLanguageModelConfig
        {
            ModelType = type,
            Family = DecoderFamily.GptNeoX,
            VocabularySize = Int(root, "vocab_size"),
            MaxPositions = OptionalInt(root, "max_position_embeddings") ?? 2048,
            HiddenSize = hidden,
            Layers = Int(root, "num_hidden_layers"),
            Heads = heads,
            KeyValueHeads = heads,
            HeadSize = headSize,
            IntermediateSize = Int(root, "intermediate_size"),
            Activation = Text(root, "hidden_act") ?? "gelu",
            LayerNormEpsilon = OptionalDouble(root, "layer_norm_eps") ?? 1e-5,
            EndTokenIds = end,
            RotaryDimensions = (int)(headSize * rotaryPercent),
            RopeTheta = OptionalDouble(root, "rotary_emb_base") ?? theta,
            RopeType = ropeType,
            RopeParameters = ropeParameters,
            AttentionBias = !root.TryGetProperty("attention_bias", out var bias) || bias.ValueKind != JsonValueKind.False,
            OutputBias = true,
            MlpBias = true,
            ParallelResidual = !root.TryGetProperty("use_parallel_residual", out var parallel) || parallel.ValueKind != JsonValueKind.False,
            TieEmbeddings = Flag(root, "tie_word_embeddings"),
        };
    }

    /// <summary>
    /// The rotary settings, from transformers 5's <c>rope_parameters</c> or the older <c>rope_theta</c>
    /// and <c>rope_scaling</c> - both are in use on the Hub.
    /// </summary>
    private static (double Theta, string Type, Dictionary<string, double> Parameters, double Partial) Rope(JsonElement root)
    {
        var theta = OptionalDouble(root, "rope_theta") ?? 10000;
        var type = "default";
        var parameters = new Dictionary<string, double>(StringComparer.Ordinal);
        var partial = OptionalDouble(root, "partial_rotary_factor") ?? 1.0;

        foreach (var name in (string[])["rope_scaling", "rope_parameters"])
        {
            if (!root.TryGetProperty(name, out var node) || node.ValueKind != JsonValueKind.Object) continue;

            foreach (var property in node.EnumerateObject())
            {
                if (property.Value.ValueKind == JsonValueKind.Number) parameters[property.Name] = property.Value.GetDouble();
            }

            type = Text(node, "rope_type") ?? Text(node, "type") ?? type;
        }

        if (parameters.TryGetValue("rope_theta", out var fromParameters)) theta = fromParameters;
        if (parameters.TryGetValue("partial_rotary_factor", out var partialFromParameters)) partial = partialFromParameters;

        if (type is not ("default" or "linear" or "llama3"))
        {
            throw new NotSupportedException(
                $"The rotary scaling '{type}' is not implemented; default, linear and llama3 are. Dynamic, YaRN and "
                + "LongRoPE scalings change the positions with the length of the text.");
        }

        return (theta, type, parameters, partial);
    }

    private static List<int> EndTokens(JsonElement root)
    {
        var ids = new List<int>();
        if (!root.TryGetProperty("eos_token_id", out var eos)) return ids;

        if (eos.ValueKind == JsonValueKind.Number) ids.Add(eos.GetInt32());
        else if (eos.ValueKind == JsonValueKind.Array) ids.AddRange(eos.EnumerateArray().Where(e => e.ValueKind == JsonValueKind.Number).Select(e => e.GetInt32()));

        return ids;
    }

    private static string? Text(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    private static bool Flag(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.True;

    private static int? OptionalInt(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : null;

    private static double? OptionalDouble(JsonElement root, string name)
        => root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetDouble() : null;

    private static int Int(JsonElement root, string name)
        => OptionalInt(root, name)
            ?? throw new InvalidDataException($"The config has no '{name}'; guessing it would build a model of the wrong shape.");
}
