using System.Text.Json;
using System.Text.Json.Serialization;
using LocalGen.Core.Inference;
using Microsoft.SemanticKernel;

namespace LocalGen.Kernel;

/// <summary>
/// LocalGen's sampling settings in Semantic Kernel's shape.
/// </summary>
/// <remarks>
/// SK passes settings as a loosely typed <see cref="PromptExecutionSettings"/> whose extra keys
/// live in <see cref="PromptExecutionSettings.ExtensionData"/>. <see cref="From"/> accepts either
/// this concrete type or a generic settings object, so callers configured from JSON keep working.
/// </remarks>
public sealed class LocalGenPromptExecutionSettings : PromptExecutionSettings
{
    [JsonPropertyName("temperature")]
    public float? Temperature { get; set; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; set; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; set; }

    [JsonPropertyName("min_p")]
    public float? MinP { get; set; }

    [JsonPropertyName("repeat_penalty")]
    public float? RepeatPenalty { get; set; }

    [JsonPropertyName("presence_penalty")]
    public float? PresencePenalty { get; set; }

    [JsonPropertyName("frequency_penalty")]
    public float? FrequencyPenalty { get; set; }

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; set; }

    [JsonPropertyName("stop")]
    public IList<string>? StopSequences { get; set; }

    [JsonPropertyName("seed")]
    public uint? Seed { get; set; }

    [JsonPropertyName("json_mode")]
    public bool JsonMode { get; set; }

    [JsonPropertyName("grammar")]
    public string? Grammar { get; set; }

    public GenerationOptions ToGenerationOptions() => new()
    {
        Temperature = Temperature,
        TopP = TopP,
        TopK = TopK,
        MinP = MinP,
        RepeatPenalty = RepeatPenalty,
        PresencePenalty = PresencePenalty,
        FrequencyPenalty = FrequencyPenalty,
        MaxTokens = MaxTokens,
        StopSequences = StopSequences is null ? [] : [.. StopSequences],
        Seed = Seed,
        JsonMode = JsonMode,
        Grammar = Grammar
    };

    /// <summary>Normalises any <see cref="PromptExecutionSettings"/> into this concrete type.</summary>
    public static LocalGenPromptExecutionSettings From(PromptExecutionSettings? settings)
    {
        switch (settings)
        {
            case null:
                return new LocalGenPromptExecutionSettings();

            case LocalGenPromptExecutionSettings typed:
                return typed;
        }

        // Round-trip through JSON so extension data lands on the matching properties.
        var json = JsonSerializer.Serialize(settings);
        var parsed = JsonSerializer.Deserialize<LocalGenPromptExecutionSettings>(json)
                     ?? new LocalGenPromptExecutionSettings();

        parsed.ModelId ??= settings.ModelId;
        parsed.ServiceId ??= settings.ServiceId;
        parsed.FunctionChoiceBehavior ??= settings.FunctionChoiceBehavior;

        return parsed;
    }
}
