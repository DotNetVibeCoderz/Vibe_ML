using System.Text;
using System.Text.Json;
using LocalGen.Core.Inference;

namespace LocalGen.Core.Prompting;

/// <summary>
/// How one family of models expresses a tool call.
/// </summary>
/// <remarks>
/// No local backend has a structured function-call channel, so a call is always text the model
/// emits and LocalGen parses back out. What differs between families is the shape of that text:
/// Qwen and Hermes wrap a JSON object in <c>&lt;tool_call&gt;</c>, Llama 3.1 prefixes it with
/// <c>&lt;|python_tag|&gt;</c> and names the arguments <c>parameters</c>, Mistral emits an array
/// after <c>[TOOL_CALLS]</c>.
///
/// Asking a model to use a convention it was not trained on works, but not as reliably as asking
/// for the one it already knows — the tokens are in its vocabulary and the format is in its
/// fine-tuning data. Detecting the family from the chat template and speaking its own dialect is
/// what "native tool calling" means here; the Hermes form remains the fallback for anything
/// unrecognised, which is what most instruction-tuned models understand from instructions alone.
/// </remarks>
public sealed record ToolDialect
{
    public required string Name { get; init; }

    /// <summary>
    /// Marker that begins a call. Empty for families that emit a bare JSON object with nothing
    /// in front of it, which is what Llama 3 does.
    /// </summary>
    public required string OpenTag { get; init; }

    /// <summary>
    /// A marker accepted at the start of a call but not required. Llama 3.1 prefixes its
    /// <em>built-in</em> tools with <c>&lt;|python_tag|&gt;</c> while emitting custom function
    /// calls bare, and the same model will do either depending on what it was asked for.
    /// </summary>
    public string OptionalPrefix { get; init; } = string.Empty;

    /// <summary>Whether a call is introduced by a marker at all.</summary>
    public bool HasOpenTag => OpenTag.Length > 0;

    /// <summary>
    /// Marker that ends a call. Empty when the family ends the call with the end-of-turn token
    /// instead, in which case the payload runs to the end of the generation.
    /// </summary>
    public string CloseTag { get; init; } = string.Empty;

    /// <summary>Property the call arguments live under — <c>arguments</c> for most, <c>parameters</c> for Llama 3.</summary>
    public string ArgumentsProperty { get; init; } = "arguments";

    /// <summary>Whether the payload is an array of calls rather than a single object.</summary>
    public bool PayloadIsArray { get; init; }

    /// <summary>Whether a call is terminated by a marker or by the end of the turn.</summary>
    public bool HasCloseTag => CloseTag.Length > 0;

    /// <summary>Qwen 2.5, Hermes, and the general fallback.</summary>
    public static readonly ToolDialect Hermes = new()
    {
        Name = "hermes",
        OpenTag = "<tool_call>",
        CloseTag = "</tool_call>"
    };

    /// <summary>
    /// Llama 3.1 and 3.2.
    /// </summary>
    /// <remarks>
    /// Llama's published template asks for "JSON for a function call" and nothing else: the model
    /// emits a bare object naming its arguments <c>parameters</c>, and the turn ends on
    /// <c>&lt;|eot_id|&gt;</c>, which llama.cpp consumes as end-of-sequence rather than surfacing
    /// as text. So there is no marker on either side — the whole reply is the call. This was
    /// checked against the real templates rather than assumed: <c>&lt;|python_tag|&gt;</c> appears
    /// nowhere in them, being reserved for the built-in search and interpreter tools.
    /// </remarks>
    public static readonly ToolDialect Llama3 = new()
    {
        Name = "llama3",
        OpenTag = string.Empty,
        OptionalPrefix = "<|python_tag|>",
        ArgumentsProperty = "parameters"
    };

    /// <summary>Mistral and Nemo, which emit a JSON array of calls after a bare marker.</summary>
    public static readonly ToolDialect Mistral = new()
    {
        Name = "mistral",
        OpenTag = "[TOOL_CALLS]",
        PayloadIsArray = true
    };

    /// <summary>
    /// Picks the dialect from the chat template baked into the model file.
    /// </summary>
    /// <remarks>
    /// The template is the most reliable signal available without loading the weights: a model
    /// trained to emit <c>[TOOL_CALLS]</c> has that string in the template it was published with.
    /// The model's name would be a guess by comparison — quantizers rename files freely.
    /// </remarks>
    public static ToolDialect Detect(string? chatTemplate)
    {
        if (string.IsNullOrWhiteSpace(chatTemplate))
        {
            return Hermes;
        }

        if (chatTemplate.Contains("[TOOL_CALLS]", StringComparison.Ordinal))
        {
            return Mistral;
        }

        // Checked before the family token below, because a fine-tune may keep a base model's
        // architecture while replacing its tool convention outright. Hermes 3 is a Llama 3 model
        // that speaks ChatML and <tool_call>; reading the architecture first would get it wrong.
        if (chatTemplate.Contains("<tool_call>", StringComparison.Ordinal))
        {
            return Hermes;
        }

        // Llama 3 leaves no marker in the output to recognise, so the family is identified from
        // the template's own header tokens. That is a stronger signal than the presence of a
        // tools branch: a published template may omit the branch and the model still emits bare
        // JSON with `parameters`, because that is what it was fine-tuned on.
        if (chatTemplate.Contains("<|start_header_id|>", StringComparison.Ordinal))
        {
            return Llama3;
        }

        return Hermes;
    }

    /// <summary>
    /// Renders the instruction block appended to the system prompt when tools are available.
    /// </summary>
    public string BuildInstructions(IReadOnlyList<ToolDefinition> tools)
    {
        if (tools.Count == 0)
        {
            return string.Empty;
        }

        var sb = new StringBuilder();
        sb.AppendLine("You have access to the following tools:");
        sb.AppendLine();

        foreach (var tool in tools)
        {
            sb.Append("- ").Append(tool.Name);

            if (!string.IsNullOrWhiteSpace(tool.Description))
            {
                sb.Append(": ").Append(tool.Description);
            }

            sb.AppendLine();
            sb.Append("  parameters: ").AppendLine(Minify(tool.ParametersJsonSchema));
        }

        sb.AppendLine();
        sb.AppendLine("To call a tool, reply with exactly this and nothing else:");
        sb.AppendLine(Example());
        sb.AppendLine("Call one tool at a time and wait for its result before continuing.");
        sb.AppendLine("If no tool is needed, answer the user directly without the tags.");

        return sb.ToString();
    }

    /// <summary>The one-line example of a call, in this family's own form.</summary>
    private string Example()
    {
        // Concatenated rather than interpolated: the example ends in "{…}}", and a raw
        // interpolated literal reads those closing braces as an escape rather than as content.
        var body = "{\"name\": \"<tool name>\", \"" + ArgumentsProperty + "\": {…}}";
        var payload = PayloadIsArray ? $"[{body}]" : body;

        return $"{OpenTag}{payload}{CloseTag}";
    }

    /// <summary>
    /// Extracts tool calls from generated text, returning the calls and the text with the call
    /// blocks removed — which is what the caller should surface as assistant content.
    /// </summary>
    public (IReadOnlyList<ToolCall> Calls, string RemainingText) Parse(string output)
    {
        if (!HasOpenTag)
        {
            return ParseMarkerless(output);
        }

        if (!output.Contains(OpenTag, StringComparison.Ordinal))
        {
            return ([], output);
        }

        var calls = new List<ToolCall>();
        var remaining = new StringBuilder();
        var cursor = 0;

        while (cursor < output.Length)
        {
            var start = output.IndexOf(OpenTag, cursor, StringComparison.Ordinal);

            if (start < 0)
            {
                remaining.Append(output.AsSpan(cursor));
                break;
            }

            remaining.Append(output.AsSpan(cursor, start - cursor));

            var payloadStart = start + OpenTag.Length;
            var end = HasCloseTag
                ? output.IndexOf(CloseTag, payloadStart, StringComparison.Ordinal)
                : -1;

            // No closing marker — either the family does not use one, or the model emitted the
            // call and stopped. Both are common, so the payload gets a chance to parse on its
            // own and only genuinely truncated JSON is kept as text.
            if (end < 0)
            {
                if (TryParseUnterminated(output[payloadStart..], calls.Count, out var recovered))
                {
                    calls.AddRange(recovered);
                }
                else
                {
                    remaining.Append(output.AsSpan(start));
                }

                break;
            }

            var payload = output[payloadStart..end].Trim();

            if (TryParsePayload(payload, calls.Count, out var parsed))
            {
                calls.AddRange(parsed);
            }
            else
            {
                // Malformed JSON is more useful to the user as visible text than silently dropped.
                remaining.Append(output.AsSpan(start, end + CloseTag.Length - start));
            }

            cursor = end + CloseTag.Length;
        }

        return (calls, remaining.ToString().Trim());
    }

    /// <summary>
    /// Reads a call from a family that uses no marker at all: the reply either is a call or is
    /// prose, with nothing to tell them apart but the shape of the text.
    /// </summary>
    /// <remarks>
    /// The JSON is required to be the very first thing in the reply. Scanning for a brace
    /// anywhere would turn any answer that merely discusses JSON — "you could send
    /// {"name": "x"}" — into a phantom tool call, which is a far worse failure than missing a
    /// call the model buried inside a sentence it was told not to write.
    /// </remarks>
    private (IReadOnlyList<ToolCall> Calls, string RemainingText) ParseMarkerless(string output)
    {
        var trimmed = output.TrimStart();

        if (OptionalPrefix.Length > 0 && trimmed.StartsWith(OptionalPrefix, StringComparison.Ordinal))
        {
            trimmed = trimmed[OptionalPrefix.Length..].TrimStart();
        }

        var opening = PayloadIsArray ? '[' : '{';

        if (!trimmed.StartsWith(opening) ||
            !TryParseUnterminated(trimmed, 0, out var calls))
        {
            return ([], output);
        }

        return (calls, string.Empty);
    }

    /// <summary>
    /// Recovers a call whose terminator never arrived, by reading one balanced JSON value out of
    /// the payload and ignoring any prose that follows it.
    /// </summary>
    internal bool TryParseUnterminated(string payload, int ordinal, out IReadOnlyList<ToolCall> calls)
    {
        calls = [];

        var trimmed = payload.Trim();
        var opening = PayloadIsArray ? '[' : '{';
        var closing = PayloadIsArray ? ']' : '}';
        var start = trimmed.IndexOf(opening);

        if (start < 0)
        {
            return false;
        }

        var depth = 0;
        var inString = false;
        var escaped = false;

        for (var i = start; i < trimmed.Length; i++)
        {
            var character = trimmed[i];

            if (escaped)
            {
                escaped = false;
                continue;
            }

            if (character == '\\' && inString)
            {
                escaped = true;
                continue;
            }

            if (character == '"')
            {
                inString = !inString;
                continue;
            }

            if (inString)
            {
                continue;
            }

            if (character == opening)
            {
                depth++;
            }
            else if (character == closing)
            {
                depth--;

                if (depth == 0)
                {
                    return TryParsePayload(trimmed[start..(i + 1)], ordinal, out calls);
                }
            }
        }

        return false;
    }

    /// <summary>Parses a payload that is either one call object or an array of them.</summary>
    private bool TryParsePayload(string payload, int ordinal, out IReadOnlyList<ToolCall> calls)
    {
        calls = [];

        try
        {
            using var document = JsonDocument.Parse(payload);
            var root = document.RootElement;

            if (root.ValueKind == JsonValueKind.Array)
            {
                var parsed = new List<ToolCall>();

                foreach (var element in root.EnumerateArray())
                {
                    if (!TryReadCall(element, ordinal + parsed.Count, out var call))
                    {
                        return false;
                    }

                    parsed.Add(call);
                }

                if (parsed.Count == 0)
                {
                    return false;
                }

                calls = parsed;
                return true;
            }

            if (TryReadCall(root, ordinal, out var single))
            {
                calls = [single];
                return true;
            }

            return false;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private bool TryReadCall(JsonElement element, int ordinal, out ToolCall call)
    {
        call = null!;

        if (element.ValueKind != JsonValueKind.Object ||
            !element.TryGetProperty("name", out var nameElement) ||
            nameElement.ValueKind != JsonValueKind.String)
        {
            return false;
        }

        var name = nameElement.GetString();

        if (string.IsNullOrWhiteSpace(name))
        {
            return false;
        }

        // The dialect's own property is preferred, but the other spelling is accepted too: a
        // model asked for one convention will sometimes answer in the other, and refusing a
        // call that is otherwise perfectly well-formed helps nobody.
        var arguments = element.TryGetProperty(ArgumentsProperty, out var argumentsElement)
            || element.TryGetProperty(AlternateArgumentsProperty, out argumentsElement)
                ? argumentsElement.GetRawText()
                : "{}";

        call = new ToolCall($"call_{ordinal}_{name}", name, arguments);
        return true;
    }

    private string AlternateArgumentsProperty =>
        ArgumentsProperty == "arguments" ? "parameters" : "arguments";

    private static string Minify(string json)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            return JsonSerializer.Serialize(document.RootElement);
        }
        catch (JsonException)
        {
            return json;
        }
    }
}
