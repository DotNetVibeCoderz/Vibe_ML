using LocalGen.Core.Inference;

namespace LocalGen.Core.Prompting;

/// <summary>
/// Tool calling for backends that only produce raw text. llama.cpp and ONNX Runtime have no
/// native notion of a function call, so the tool schema is injected into the prompt and the
/// model is asked to answer with a JSON block, which is parsed back out here.
/// </summary>
/// <remarks>
/// This is the default dialect — the one Qwen, Hermes and most instruction-tuned models either
/// know already or follow readily from instructions. Families with a convention of their own are
/// served by <see cref="ToolDialect"/>, which this type is the fallback case of; the members here
/// remain because they are the shape the rest of the codebase and its tests were written against.
/// </remarks>
public static class ToolCallProtocol
{
    /// <summary>Sentinel the model wraps a call in. Chosen to be unlikely in ordinary prose.</summary>
    public const string OpenTag = "<tool_call>";

    public const string CloseTag = "</tool_call>";

    /// <summary>The dialect these members delegate to.</summary>
    public static ToolDialect Default => ToolDialect.Hermes;

    /// <summary>
    /// Renders the instruction block appended to the system prompt when tools are available.
    /// </summary>
    public static string BuildInstructions(IReadOnlyList<ToolDefinition> tools) =>
        ToolDialect.Hermes.BuildInstructions(tools);

    /// <summary>
    /// Extracts tool calls from generated text. Returns the calls plus the text with the call
    /// blocks stripped, which is what the caller should surface as assistant content.
    /// </summary>
    public static (IReadOnlyList<ToolCall> Calls, string RemainingText) Parse(string output) =>
        ToolDialect.Hermes.Parse(output);

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
        if (ToolDialect.Hermes.TryParseUnterminated(payload, ordinal, out var calls) && calls.Count > 0)
        {
            call = calls[0];
            return true;
        }

        call = null!;
        return false;
    }
}
