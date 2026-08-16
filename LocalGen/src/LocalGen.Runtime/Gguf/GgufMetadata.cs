namespace LocalGen.Runtime.Gguf;

/// <summary>Metadata read from a GGUF file's header, without loading the weights.</summary>
public sealed record GgufMetadata
{
    /// <summary>Model architecture, e.g. <c>llama</c>, <c>qwen2</c>, <c>gemma3</c>.</summary>
    public string Architecture { get; init; } = string.Empty;

    /// <summary>Name recorded by the converter, which is often nicer than the file name.</summary>
    public string Name { get; init; } = string.Empty;

    /// <summary>Trained context length in tokens.</summary>
    public int ContextLength { get; init; }

    /// <summary>Hidden size, which is also the embedding dimension for embedding models.</summary>
    public int EmbeddingLength { get; init; }

    public int BlockCount { get; init; }

    /// <summary>Total parameters, or zero when the file does not record it.</summary>
    public long ParameterCount { get; init; }

    /// <summary>Jinja chat template baked into the file, when present.</summary>
    public string? ChatTemplate { get; init; }

    /// <summary>Quantization tag derived from the file type, e.g. <c>Q4_K_M</c>.</summary>
    public string QuantizationType { get; init; } = string.Empty;

    /// <summary>Size label such as <c>7B</c>, when the converter recorded one.</summary>
    public string SizeLabel { get; init; } = string.Empty;

    public int TensorCount { get; init; }

    public uint Version { get; init; }

    /// <summary>Every key/value pair, for the model detail view.</summary>
    public IReadOnlyDictionary<string, string> All { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>True when the file looks like an embedding model rather than a chat model.</summary>
    public bool LooksLikeEmbeddingModel =>
        Architecture.Contains("bert", StringComparison.OrdinalIgnoreCase) ||
        Name.Contains("embed", StringComparison.OrdinalIgnoreCase) ||
        All.ContainsKey("bert.pooling_type");
}
