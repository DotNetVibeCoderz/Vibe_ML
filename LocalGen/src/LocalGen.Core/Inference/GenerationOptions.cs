namespace LocalGen.Core.Inference;

/// <summary>
/// Sampling and decoding knobs. Every value is nullable so that a request can express
/// "leave this at whatever the Modelfile or engine default is" instead of silently
/// overriding it with a framework default.
/// </summary>
public sealed record GenerationOptions
{
    public float? Temperature { get; init; }

    public float? TopP { get; init; }

    public int? TopK { get; init; }

    public float? MinP { get; init; }

    /// <summary>Penalty applied to tokens already present in the context. 1.0 disables it.</summary>
    public float? RepeatPenalty { get; init; }

    /// <summary>How many previous tokens the repeat penalty looks back over.</summary>
    public int? RepeatLastN { get; init; }

    public float? PresencePenalty { get; init; }

    public float? FrequencyPenalty { get; init; }

    /// <summary>Upper bound on generated tokens. Null means "until EOS or context is full".</summary>
    public int? MaxTokens { get; init; }

    /// <summary>Strings that halt generation when produced.</summary>
    public IReadOnlyList<string> StopSequences { get; init; } = [];

    /// <summary>Fixed seed for reproducible sampling. Null means non-deterministic.</summary>
    public uint? Seed { get; init; }

    /// <summary>Context window override, in tokens.</summary>
    public int? ContextSize { get; init; }

    /// <summary>Request a JSON-shaped response. Engines that cannot constrain grammar ignore it.</summary>
    public bool JsonMode { get; init; }

    /// <summary>GBNF grammar constraining the output. Supported by the LlamaSharp engine.</summary>
    public string? Grammar { get; init; }

    public static readonly GenerationOptions Default = new();

    /// <summary>
    /// Layers <paramref name="overrides"/> on top of this instance; any value the override
    /// leaves null keeps the current setting. Used to merge Modelfile defaults with per-request options.
    /// </summary>
    public GenerationOptions Merge(GenerationOptions? overrides)
    {
        if (overrides is null)
        {
            return this;
        }

        return new GenerationOptions
        {
            Temperature = overrides.Temperature ?? Temperature,
            TopP = overrides.TopP ?? TopP,
            TopK = overrides.TopK ?? TopK,
            MinP = overrides.MinP ?? MinP,
            RepeatPenalty = overrides.RepeatPenalty ?? RepeatPenalty,
            RepeatLastN = overrides.RepeatLastN ?? RepeatLastN,
            PresencePenalty = overrides.PresencePenalty ?? PresencePenalty,
            FrequencyPenalty = overrides.FrequencyPenalty ?? FrequencyPenalty,
            MaxTokens = overrides.MaxTokens ?? MaxTokens,
            StopSequences = overrides.StopSequences.Count > 0 ? overrides.StopSequences : StopSequences,
            Seed = overrides.Seed ?? Seed,
            ContextSize = overrides.ContextSize ?? ContextSize,
            JsonMode = overrides.JsonMode || JsonMode,
            Grammar = overrides.Grammar ?? Grammar
        };
    }
}

/// <summary>A tool the model may call, described with a JSON Schema parameter block.</summary>
public sealed record ToolDefinition
{
    public required string Name { get; init; }

    public string? Description { get; init; }

    /// <summary>JSON Schema describing the call arguments.</summary>
    public string ParametersJsonSchema { get; init; } = """{"type":"object","properties":{}}""";
}

/// <summary>A chat completion request, independent of any wire protocol.</summary>
public sealed record ChatRequest
{
    /// <summary>Identifier of the model to run, as known to the model store.</summary>
    public required string Model { get; init; }

    public required IReadOnlyList<ChatMessage> Messages { get; init; }

    public GenerationOptions Options { get; init; } = GenerationOptions.Default;

    public IReadOnlyList<ToolDefinition> Tools { get; init; } = [];
}

/// <summary>A completed, non-streaming chat response.</summary>
public sealed record ChatResponse
{
    public required ChatMessage Message { get; init; }

    public FinishReason FinishReason { get; init; } = FinishReason.Stop;

    public TokenUsage Usage { get; init; } = TokenUsage.Empty;

    public required string Model { get; init; }

    /// <summary>Wall-clock time spent producing this response.</summary>
    public TimeSpan Duration { get; init; }
}

/// <summary>One incremental piece of a streaming response.</summary>
public sealed record ChatStreamChunk
{
    /// <summary>Newly generated text, if any. Empty on chunks that only carry a tool call or finish reason.</summary>
    public string Delta { get; init; } = string.Empty;

    public IReadOnlyList<ToolCall> ToolCalls { get; init; } = [];

    public FinishReason FinishReason { get; init; } = FinishReason.None;

    /// <summary>Populated on the final chunk only.</summary>
    public TokenUsage? Usage { get; init; }
}

/// <summary>Request for one or more embedding vectors.</summary>
public sealed record EmbeddingRequest
{
    public required string Model { get; init; }

    public required IReadOnlyList<string> Inputs { get; init; }
}

/// <summary>Embedding vectors in the same order as the request inputs.</summary>
public sealed record EmbeddingResponse
{
    public required IReadOnlyList<ReadOnlyMemory<float>> Embeddings { get; init; }

    public required string Model { get; init; }

    public TokenUsage Usage { get; init; } = TokenUsage.Empty;
}
