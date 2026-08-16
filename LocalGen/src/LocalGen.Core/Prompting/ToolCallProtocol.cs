using System.Text;
using System.Text.Json;
using LocalGen.Core.Inference;

namespace LocalGen.Core.Prompting;

/// <summary>
/// Tool calling for backends that only produce raw text. llama.cpp and ONNX Runtime have no
/// native notion of a function call, so the tool schema is injected into the prompt and the
/// model is asked to answer with a JSON block, which is parsed back out here.
/// </summary>
/// <remarks>
/// Models with a native tool-calling chat template still benefit: the instructions are
/// consistent with what most instruction-tuned models were trained on.
/// </remarks>
public static class ToolCallProtocol
{
    /// <summary>Sentinel the model wraps a call in. Chosen to be unlikely in ordinary prose.</summary>
    public const string OpenTag = "<tool_call>";

    public const string CloseTag = "</tool_call>";

    /// <summary>
    /// Renders the instruction block appended to the system prompt when tools are available.
    /// </summary>
    public static string BuildInstructions(IReadOnlyList<ToolDefinition> tools)
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
        sb.Append(OpenTag).Append("""{"name": "<tool name>", "arguments": {…}}""").AppendLine(CloseTag);
        sb.AppendLine("Call one tool at a time and wait for its result before continuing.");
        sb.AppendLine("If no tool is needed, answer the user directly without the tags.");

        return sb.ToString();
    }

    /// <summary>
    /// Extracts tool calls from generated text. Returns the calls plus the text with the call
    /// blocks stripped, which is what the caller should surface as assistant content.
    /// </summary>
    public static (IReadOnlyList<ToolCall> Calls, string RemainingText) Parse(string output)
    {
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
            var end = output.IndexOf(CloseTag, payloadStart, StringComparison.Ordinal);

            // No closing tag. Models frequently emit the call and then stop, so the payload gets
            // a chance to parse on its own; only genuinely truncated JSON is kept as text.
            if (end < 0)
            {
                if (TryParseUnterminated(output[payloadStart..], calls.Count, out var recovered))
                {
                    calls.Add(recovered);
                }
                else
                {
                    remaining.Append(output.AsSpan(start));
                }

                break;
            }

            var payload = output[payloadStart..end].Trim();
            if (TryParseCall(payload, calls.Count, out var call))
            {
                calls.Add(call);
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
    /// Recovers a tool call whose closing tag never arrived.
    /// </summary>
    /// <remarks>
    /// Instruction-tuned models routinely emit the opening tag and the JSON, then stop — the
    /// end-of-sequence token arrives where the closing tag should be. Treating that as prose
    /// would surface raw JSON to the user and silently lose the call, so the payload is parsed on
    /// its own and only kept as text when it is genuinely incomplete.
    /// </remarks>
    internal static bool TryParseUnterminated(string payload, int ordinal, out ToolCall call)
    {
        // Trailing prose or a stray token after the JSON is common; the object is taken from the
        // first '{' to its matching '}' rather than assuming the payload is JSON end to end.
        var trimmed = payload.Trim();
        var start = trimmed.IndexOf('{');

        if (start < 0)
        {
            call = null!;
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

            switch (character)
            {
                case '\\' when inString:
                    escaped = true;
                    break;

                case '"':
                    inString = !inString;
                    break;

                case '{' when !inString:
                    depth++;
                    break;

                case '}' when !inString:
                    depth--;

                    if (depth == 0)
                    {
                        return TryParseCall(trimmed[start..(i + 1)], ordinal, out call);
                    }

                    break;
            }
        }

        call = null!;
        return false;
    }

    private static bool TryParseCall(string payload, int ordinal, out ToolCall call)
    {
        call = null!;

        try
        {
            using var document = JsonDocument.Parse(payload);
            var root = document.RootElement;

            if (root.ValueKind != JsonValueKind.Object ||
                !root.TryGetProperty("name", out var nameElement) ||
                nameElement.ValueKind != JsonValueKind.String)
            {
                return false;
            }

            var name = nameElement.GetString();
            if (string.IsNullOrWhiteSpace(name))
            {
                return false;
            }

            var arguments = root.TryGetProperty("arguments", out var argumentsElement)
                ? argumentsElement.GetRawText()
                : "{}";

            call = new ToolCall($"call_{ordinal}_{name}", name, arguments);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

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
