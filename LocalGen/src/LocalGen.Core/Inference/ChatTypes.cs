using System.Text.Json.Serialization;

namespace LocalGen.Core.Inference;

/// <summary>Speaker role for a single turn in a conversation.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<ChatRole>))]
public enum ChatRole
{
    System,
    User,
    Assistant,
    Tool
}

/// <summary>
/// A single piece of content inside a message. A message may carry several parts so that
/// text, images and documents can travel together for multimodal and RAG scenarios.
/// </summary>
public abstract record ContentPart
{
    public sealed record Text(string Value) : ContentPart;

    /// <summary>
    /// Raw image bytes plus its media type, e.g. <c>image/png</c>.
    /// </summary>
    /// <param name="Source">
    /// Where the image can be fetched from, when it was uploaded rather than inlined. The bytes
    /// are what a vision model reads; the URL is what a transcript renders and what travels in an
    /// OpenAI <c>image_url</c> part, so both are carried rather than one being derived.
    /// </param>
    public sealed record Image(
        ReadOnlyMemory<byte> Data,
        string MediaType,
        string? Source = null) : ContentPart;

    /// <summary>
    /// A document (PDF, markdown, code…) already extracted to text.
    /// </summary>
    /// <param name="Source">URL the original file can be downloaded from, when it was uploaded.</param>
    public sealed record Document(
        string FileName,
        string ExtractedText,
        string? Source = null) : ContentPart;
}

/// <summary>A tool/function call requested by the model.</summary>
public sealed record ToolCall(string Id, string Name, string ArgumentsJson);

/// <summary>One turn in a conversation.</summary>
public sealed record ChatMessage
{
    public required ChatRole Role { get; init; }

    public IReadOnlyList<ContentPart> Content { get; init; } = [];

    /// <summary>Optional speaker name, used by some chat templates.</summary>
    public string? Name { get; init; }

    /// <summary>Tool calls the assistant asked for. Only meaningful on assistant turns.</summary>
    public IReadOnlyList<ToolCall> ToolCalls { get; init; } = [];

    /// <summary>Identifies which <see cref="ToolCall"/> this message answers. Only on tool turns.</summary>
    public string? ToolCallId { get; init; }

    /// <summary>Concatenation of every textual part, which is what chat templates render.</summary>
    public string Text =>
        string.Concat(Content.OfType<ContentPart.Text>().Select(static p => p.Value));

    public static ChatMessage System(string text) => Create(ChatRole.System, text);

    public static ChatMessage User(string text) => Create(ChatRole.User, text);

    public static ChatMessage Assistant(string text) => Create(ChatRole.Assistant, text);

    public static ChatMessage Tool(string toolCallId, string result) => new()
    {
        Role = ChatRole.Tool,
        ToolCallId = toolCallId,
        Content = [new ContentPart.Text(result)]
    };

    private static ChatMessage Create(ChatRole role, string text) => new()
    {
        Role = role,
        Content = [new ContentPart.Text(text)]
    };
}

/// <summary>Why generation stopped.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<FinishReason>))]
public enum FinishReason
{
    /// <summary>Generation is still in flight — only valid on streaming deltas.</summary>
    None,

    /// <summary>The model emitted an end-of-sequence token or a configured stop string.</summary>
    Stop,

    /// <summary>The <c>MaxTokens</c> budget was exhausted.</summary>
    Length,

    /// <summary>The model asked to invoke one or more tools.</summary>
    ToolCalls,

    /// <summary>The caller cancelled the request.</summary>
    Cancelled,

    /// <summary>Generation failed; inspect the accompanying error.</summary>
    Error
}

/// <summary>Token accounting for a single request.</summary>
public sealed record TokenUsage
{
    public int PromptTokens { get; init; }

    public int CompletionTokens { get; init; }

    public int TotalTokens => PromptTokens + CompletionTokens;

    public static readonly TokenUsage Empty = new();

    public TokenUsage Add(TokenUsage other) => new()
    {
        PromptTokens = PromptTokens + other.PromptTokens,
        CompletionTokens = CompletionTokens + other.CompletionTokens
    };
}
