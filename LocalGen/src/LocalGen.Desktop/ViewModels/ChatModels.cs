using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using LocalGen.Core.Inference;
using Microsoft.SemanticKernel.ChatCompletion;

namespace LocalGen.Desktop.ViewModels;

/// <summary>What an attachment is, which decides how it reaches the model.</summary>
public enum AttachmentKind
{
    /// <summary>Sent as an image content part; a vision model reads the pixels.</summary>
    Image,

    /// <summary>Extracted to text and sent as a document part, with a link in the message.</summary>
    Document
}

/// <summary>A file attached to the next message.</summary>
public sealed partial class Attachment : ObservableObject
{
    public required AttachmentKind Kind { get; init; }

    public required string FileName { get; init; }

    /// <summary>Address the uploaded file is served from.</summary>
    public required string Url { get; init; }

    public long SizeBytes { get; init; }

    /// <summary>Extracted text, for documents. Empty for images.</summary>
    public string ExtractedText { get; init; } = string.Empty;

    /// <summary>Media type of the stored file, e.g. <c>image/png</c>.</summary>
    public string MediaType { get; init; } = "application/octet-stream";

    public bool IsImage => Kind == AttachmentKind.Image;

    public string SizeLabel => Converters.FormatBytes(SizeBytes);

    /// <summary>
    /// The markdown that puts this attachment in the transcript — an image renders inline, a
    /// document becomes a link the user can open.
    /// </summary>
    public string ToMarkdown() =>
        Kind == AttachmentKind.Image ? $"![{FileName}]({Url})" : $"[{FileName}]({Url})";
}

/// <summary>One turn shown in the transcript.</summary>
public sealed partial class ChatEntry : ObservableObject
{
    [ObservableProperty]
    private string _text = string.Empty;

    /// <summary><c>user</c>, <c>assistant</c>, <c>tool</c> or <c>error</c>.</summary>
    public required string Kind { get; init; }

    public DateTimeOffset Timestamp { get; } = DateTimeOffset.Now;

    /// <summary>Tool name, when this entry records a tool call.</summary>
    public string? Tool { get; init; }

    public bool IsUser => Kind == "user";

    public bool IsAssistant => Kind == "assistant";

    public bool IsTool => Kind == "tool";

    public bool IsError => Kind == "error";

    /// <summary>
    /// Whether to render this turn as markdown. Tool output is machine text — often JSON or a
    /// stack trace — and running it through a markdown renderer would mangle it.
    /// </summary>
    public bool RendersMarkdown => !IsTool && !IsError;

    public string Speaker => Kind switch
    {
        "user" => "You",
        "tool" => Tool ?? "Tool",
        "error" => "Error",
        _ => "Assistant"
    };
}

/// <summary>A named conversation with its own history, attachments and settings.</summary>
public sealed partial class ChatSession(string name) : ObservableObject
{
    [ObservableProperty]
    private string _name = name;

    public System.Collections.ObjectModel.ObservableCollection<ChatEntry> Entries { get; } = [];

    /// <summary>Attachments staged for the next message.</summary>
    public System.Collections.ObjectModel.ObservableCollection<Attachment> Pending { get; } = [];

    public ChatHistory History { get; } = [];

    public DateTimeOffset CreatedAt { get; } = DateTimeOffset.Now;

    public string Summary
    {
        get
        {
            var turns = Entries.Count(static e => e.IsUser);

            return turns switch
            {
                0 => "empty",
                1 => "1 turn",
                _ => $"{turns} turns"
            };
        }
    }

    /// <summary>Builds the message sent to the model, with attachments as content parts.</summary>
    public ChatMessage BuildUserMessage(string text)
    {
        var parts = new List<ContentPart>();

        // Documents go before the prompt so the model reads the material first and the question
        // second, which is the order that produces answers grounded in the attachment.
        foreach (var attachment in Pending.Where(static a => a.Kind == AttachmentKind.Document))
        {
            parts.Add(new ContentPart.Document(
                attachment.FileName,
                attachment.ExtractedText,
                attachment.Url));
        }

        foreach (var attachment in Pending.Where(static a => a.Kind == AttachmentKind.Image))
        {
            // No bytes here: the source URL is resolved by whichever host serves the file, so an
            // image is stored once rather than copied into every message that references it.
            parts.Add(new ContentPart.Image(
                ReadOnlyMemory<byte>.Empty,
                attachment.MediaType,
                attachment.Url));
        }

        parts.Add(new ContentPart.Text(text));

        return new ChatMessage { Role = ChatRole.User, Content = parts };
    }

    /// <summary>Renders the user's turn for the transcript, with attachments as markdown.</summary>
    public string BuildTranscriptText(string text)
    {
        if (Pending.Count == 0)
        {
            return text;
        }

        var builder = new StringBuilder();

        foreach (var attachment in Pending)
        {
            builder.AppendLine(attachment.ToMarkdown()).AppendLine();
        }

        return builder.Append(text).ToString();
    }
}
