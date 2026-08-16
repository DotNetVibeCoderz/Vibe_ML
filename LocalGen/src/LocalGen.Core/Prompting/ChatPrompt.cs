using System.Text;
using LocalGen.Core.Inference;

namespace LocalGen.Core.Prompting;

/// <summary>
/// Prepares a conversation for a text-completion backend: folds tool definitions and tool
/// results into the message list, then renders it with the model's chat template when the
/// backend exposes one, or with ChatML as a fallback.
/// </summary>
public static class ChatPrompt
{
    /// <summary>
    /// Rewrites messages so a plain text model can consume them: tool instructions are merged
    /// into the system turn and tool results become user turns the model can read.
    /// </summary>
    public static IReadOnlyList<ChatMessage> Prepare(ChatRequest request)
    {
        var messages = new List<ChatMessage>(request.Messages.Count + 1);
        var toolInstructions = ToolCallProtocol.BuildInstructions(request.Tools);
        var systemInjected = false;

        foreach (var message in request.Messages)
        {
            switch (message.Role)
            {
                case ChatRole.System when !systemInjected && toolInstructions.Length > 0:
                    messages.Add(ChatMessage.System($"{message.Text}\n\n{toolInstructions}"));
                    systemInjected = true;
                    break;

                case ChatRole.Tool:
                    // Most templates have no tool role, so results are surfaced as a user turn.
                    messages.Add(ChatMessage.User($"Tool result: {message.Text}"));
                    break;

                case ChatRole.Assistant when message.ToolCalls.Count > 0:
                    messages.Add(ChatMessage.Assistant(RenderAssistantToolCalls(message)));
                    break;

                default:
                    messages.Add(FlattenContent(message));
                    break;
            }
        }

        // No system turn existed to merge the tool instructions into.
        if (!systemInjected && toolInstructions.Length > 0)
        {
            messages.Insert(0, ChatMessage.System(toolInstructions));
        }

        return messages;
    }

    /// <summary>
    /// ChatML rendering, used when the model file carries no chat template of its own.
    /// Recognised by nearly every modern instruction-tuned model.
    /// </summary>
    public static string RenderChatMl(IReadOnlyList<ChatMessage> messages)
    {
        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            var role = message.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Assistant => "assistant",
                ChatRole.Tool => "user",
                _ => "user"
            };

            sb.Append("<|im_start|>").Append(role).Append('\n')
              .Append(message.Text).Append("<|im_end|>\n");
        }

        sb.Append("<|im_start|>assistant\n");
        return sb.ToString();
    }

    /// <summary>Stop strings that keep a ChatML-rendered prompt from running past its turn.</summary>
    public static readonly IReadOnlyList<string> ChatMlStopSequences =
        ["<|im_end|>", "<|im_start|>"];

    /// <summary>
    /// Collapses multimodal parts into text. Images are replaced with a placeholder because
    /// text-only backends cannot see them; vision-capable sessions handle images separately.
    /// </summary>
    private static ChatMessage FlattenContent(ChatMessage message)
    {
        if (message.Content.All(static p => p is ContentPart.Text))
        {
            return message;
        }

        var sb = new StringBuilder();
        foreach (var part in message.Content)
        {
            switch (part)
            {
                case ContentPart.Text text:
                    sb.Append(text.Value);
                    break;

                case ContentPart.Document document:
                    sb.AppendLine().Append("--- ").Append(document.FileName);

                    if (!string.IsNullOrEmpty(document.Source))
                    {
                        sb.Append(" (").Append(document.Source).Append(')');
                    }

                    sb.AppendLine(" ---").AppendLine(document.ExtractedText);
                    break;

                case ContentPart.Image image:
                    // The model cannot see it, but naming the attachment is more useful than
                    // dropping it — the user can then be told what was ignored.
                    sb.AppendLine().Append("[an image was attached");

                    if (!string.IsNullOrEmpty(image.Source))
                    {
                        sb.Append(": ").Append(image.Source);
                    }

                    sb.Append("; this model cannot process images]");
                    break;
            }
        }

        return message with { Content = [new ContentPart.Text(sb.ToString())] };
    }

    private static string RenderAssistantToolCalls(ChatMessage message)
    {
        var sb = new StringBuilder(message.Text);

        foreach (var call in message.ToolCalls)
        {
            sb.Append(ToolCallProtocol.OpenTag)
              .Append("{\"name\": \"").Append(call.Name)
              .Append("\", \"arguments\": ").Append(call.ArgumentsJson).Append('}')
              .Append(ToolCallProtocol.CloseTag);
        }

        return sb.ToString();
    }
}
