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
    /// <param name="dialect">
    /// The model family's tool-call convention. Defaults to the Hermes form when the caller does
    /// not know the family — which is what an unrecognised model is asked to use.
    /// </param>
    /// <param name="imageMarker">
    /// Placeholder written where an image appears, for a session that can actually see it — the
    /// backend replaces each marker with the encoded image. Null means the model is text-only,
    /// and images are described in words instead.
    /// </param>
    public static IReadOnlyList<ChatMessage> Prepare(
        ChatRequest request,
        ToolDialect? dialect = null,
        string? imageMarker = null)
    {
        var convention = dialect ?? ToolDialect.Hermes;
        var messages = new List<ChatMessage>(request.Messages.Count + 1);
        var toolInstructions = convention.BuildInstructions(request.Tools);
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
                    messages.Add(ChatMessage.Assistant(RenderAssistantToolCalls(message, convention)));
                    break;

                default:
                    messages.Add(FlattenContent(message, imageMarker));
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
    /// Collapses multimodal parts into text.
    /// </summary>
    /// <remarks>
    /// An image becomes <paramref name="imageMarker"/> when the session can see it — the marker
    /// is where the backend splices the encoded image into the token sequence, so its position
    /// in the text is what puts the picture in the right place in the conversation. Without a
    /// marker the image is described in words instead, which is all a text-only model can use.
    /// </remarks>
    private static ChatMessage FlattenContent(ChatMessage message, string? imageMarker)
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

                case ContentPart.Image image when imageMarker is not null:
                    sb.Append(imageMarker);
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

    /// <summary>
    /// Replays a previous assistant turn's tool calls in the dialect the model speaks, so that
    /// the transcript it reads back matches the form it was asked to produce.
    /// </summary>
    private static string RenderAssistantToolCalls(ChatMessage message, ToolDialect dialect)
    {
        var sb = new StringBuilder(message.Text);

        foreach (var call in message.ToolCalls)
        {
            var body = $"{{\"name\": \"{call.Name}\", \"{dialect.ArgumentsProperty}\": {call.ArgumentsJson}}}";

            sb.Append(dialect.OpenTag)
              .Append(dialect.PayloadIsArray ? $"[{body}]" : body)
              .Append(dialect.CloseTag);
        }

        return sb.ToString();
    }
}
