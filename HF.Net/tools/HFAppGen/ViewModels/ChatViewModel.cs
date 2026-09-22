using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using HFAppGen.Models;
using HFAppGen.Services;

namespace HFAppGen.ViewModels;

/// <summary>Who said something in the transcript.</summary>
public enum ChatRole
{
    /// <summary>The person using the app.</summary>
    User,

    /// <summary>Jack.</summary>
    Assistant,

    /// <summary>A note about a tool call or an error.</summary>
    System,
}

/// <summary>One turn in the transcript.</summary>
public sealed partial class ChatMessage : ObservableObject
{
    /// <summary>Who is speaking.</summary>
    public required ChatRole Role { get; init; }

    /// <summary>When it was said.</summary>
    public DateTime Timestamp { get; init; } = DateTime.Now;

    /// <summary>Any images attached to a user turn.</summary>
    public List<string> Attachments { get; init; } = [];

    [ObservableProperty] private string _text = "";
    [ObservableProperty] private bool _isStreaming;

    /// <summary>
    /// <see cref="Text"/> with inline markdown emphasis removed.
    /// </summary>
    /// <remarks>
    /// The transcript is plain text, so raw markers would show up literally as
    /// <c>**like this**</c>. Emphasis and inline-code markers are stripped; fenced code blocks
    /// keep their fences, because losing the boundary between prose and code would be worse
    /// than an extra pair of backticks.
    /// </remarks>
    public string Display => Strip(Text);

    partial void OnTextChanged(string value) => OnPropertyChanged(nameof(Display));

    private static string Strip(string text)
    {
        if (string.IsNullOrEmpty(text)) return text;

        var builder = new System.Text.StringBuilder(text.Length);
        var insideFence = false;

        foreach (var line in text.Split('\n'))
        {
            if (line.TrimStart().StartsWith("```", StringComparison.Ordinal)) insideFence = !insideFence;

            // Inside a fence the characters are code, not formatting.
            builder.Append(insideFence ? line : StripInline(line)).Append('\n');
        }

        return builder.ToString().TrimEnd('\n');
    }

    private static string StripInline(string line)
    {
        var builder = new System.Text.StringBuilder(line.Length);

        for (var i = 0; i < line.Length; i++)
        {
            if (line[i] == '*' || line[i] == '`' || line[i] == '_')
            {
                // A marker only counts when it is not surrounded by whitespace on both sides,
                // so "a * b" and a bullet "- * item" survive untouched.
                var beforeIsSpace = i == 0 || char.IsWhiteSpace(line[i - 1]);
                var afterIsSpace = i + 1 >= line.Length || char.IsWhiteSpace(line[i + 1]);
                if (!(beforeIsSpace && afterIsSpace)) continue;
            }
            builder.Append(line[i]);
        }

        return builder.ToString();
    }

    /// <summary>Display name for the speaker.</summary>
    public string Speaker => Role switch
    {
        ChatRole.User => "You",
        ChatRole.Assistant => AssistantService.Name,
        _ => "System",
    };

    /// <summary>Theme brush key for the speaker's accent.</summary>
    public string AccentBrush => Role switch
    {
        ChatRole.User => "AccentBrush",
        ChatRole.Assistant => "Spec6",
        _ => "Spec2",
    };

    /// <summary>Clock time for the header.</summary>
    public string Time => Timestamp.ToString("HH:mm");

    /// <summary>Whether the attachment strip should be shown.</summary>
    public bool HasAttachments => Attachments.Count > 0;

    /// <summary>Attachment file names, for display.</summary>
    public string AttachmentSummary => string.Join(", ", Attachments.Select(Path.GetFileName));
}

/// <summary>
/// The chat panel: transcript, composer, attachments and the model picker.
/// </summary>
public sealed partial class ChatViewModel : ObservableObject
{
    private readonly AssistantService _assistant;
    private readonly ProjectService _projects;
    private readonly LogService _logs;
    private readonly AppSettings _settings;
    private CancellationTokenSource? _cancellation;

    /// <summary>Creates the panel and seeds the greeting.</summary>
    public ChatViewModel(AssistantService assistant, ProjectService projects, LogService logs, AppSettings settings)
    {
        _assistant = assistant;
        _projects = projects;
        _logs = logs;
        _settings = settings;

        Models = new ObservableCollection<string>(
            settings.AvailableModels.Count > 0 ? settings.AvailableModels : [settings.Model]);

        if (!Models.Contains(settings.Model)) Models.Insert(0, settings.Model);
        _selectedModel = settings.Model;

        AddGreeting();
    }

    /// <summary>The transcript.</summary>
    public ObservableCollection<ChatMessage> Messages { get; } = [];

    /// <summary>Models offered in the picker.</summary>
    public ObservableCollection<string> Models { get; }

    /// <summary>Images staged for the next message.</summary>
    public ObservableCollection<string> PendingAttachments { get; } = [];

    [ObservableProperty] private string _composer = "";
    [ObservableProperty] private bool _isBusy;
    [ObservableProperty] private string _selectedModel;
    [ObservableProperty] private string _statusText = "";

    partial void OnSelectedModelChanged(string value)
    {
        if (string.IsNullOrWhiteSpace(value) || value == _settings.Model) return;

        _settings.Model = value;
        _assistant.Configure(_settings);
        _logs.Info("assistant", $"Switched to {value}.");
    }

    /// <summary>True when there is something to send and nothing in flight.</summary>
    public bool CanSend => !IsBusy && !string.IsNullOrWhiteSpace(Composer);

    partial void OnComposerChanged(string value) => OnPropertyChanged(nameof(CanSend));

    partial void OnIsBusyChanged(bool value) => OnPropertyChanged(nameof(CanSend));

    /// <summary>Sends the composer contents and streams the reply.</summary>
    public async Task SendAsync()
    {
        if (!CanSend) return;

        var text = Composer.Trim();
        var attachments = PendingAttachments.ToList();

        Composer = "";
        PendingAttachments.Clear();

        Messages.Add(new ChatMessage { Role = ChatRole.User, Text = text, Attachments = attachments });

        if (!_settings.IsConfigured)
        {
            Messages.Add(new ChatMessage
            {
                Role = ChatRole.System,
                Text = _settings.MissingConfiguration ?? "The assistant is not configured.",
            });
            return;
        }

        var reply = new ChatMessage { Role = ChatRole.Assistant, IsStreaming = true };
        Messages.Add(reply);

        IsBusy = true;
        StatusText = "Thinking…";
        _cancellation = new CancellationTokenSource();

        try
        {
            _assistant.Configure(_settings);

            await foreach (var chunk in _assistant.SendAsync(text, _settings, attachments, _cancellation.Token))
            {
                if (chunk.ToolName is not null)
                {
                    StatusText = $"Running {chunk.ToolName}…";
                    Messages.Insert(Messages.IndexOf(reply), new ChatMessage
                    {
                        Role = ChatRole.System,
                        Text = $"⟶ {chunk.ToolName}",
                    });
                    continue;
                }

                reply.Text += chunk.Text;
                StatusText = "Writing…";
            }
        }
        catch (Exception ex)
        {
            reply.Text += $"\n\n**Something went wrong.** {ex.Message}";
            _logs.Error("assistant", ex.Message);
        }
        finally
        {
            reply.IsStreaming = false;
            IsBusy = false;
            StatusText = "";
            _cancellation?.Dispose();
            _cancellation = null;
        }

        if (string.IsNullOrWhiteSpace(reply.Text))
            reply.Text = "_(no reply)_";
    }

    /// <summary>Cancels an in-flight request.</summary>
    public void Cancel()
    {
        _cancellation?.Cancel();
        StatusText = "Cancelling…";
    }

    /// <summary>Clears the transcript and the model's memory of it.</summary>
    public void ClearThread()
    {
        Messages.Clear();
        _assistant.ResetHistory(_settings);
        AddGreeting();
        _logs.Info("assistant", "Conversation cleared.");
    }

    /// <summary>
    /// Notes a project change in the transcript, when nothing has been said yet.
    /// </summary>
    /// <remarks>
    /// The greeting reports which project is open, and it would be stale the moment one is
    /// opened afterwards. Rewriting it is only safe while the greeting is still the only message;
    /// once there is a conversation, the transcript is history and stays as it was.
    /// </remarks>
    public void NoteProjectChanged()
    {
        if (Messages.Count != 1 || Messages[0].Role != ChatRole.Assistant) return;

        Messages.Clear();
        AddGreeting();
    }

    /// <summary>Stages images for the next message.</summary>
    public void Attach(IEnumerable<string> paths)
    {
        foreach (var path in paths)
        {
            if (!File.Exists(path) || PendingAttachments.Contains(path)) continue;
            PendingAttachments.Add(path);
        }
    }

    /// <summary>Removes a staged image.</summary>
    public void RemoveAttachment(string path) => PendingAttachments.Remove(path);

    private void AddGreeting()
    {
        var project = _projects.HasProject
            ? $"You have {_projects.ProjectName} open."
            : "No project is open yet.";

        Messages.Add(new ChatMessage
        {
            Role = ChatRole.Assistant,
            Text = $"""
                I'm {AssistantService.FullName}. I build .NET applications on HF.Net - Hugging Face models, tokenizers, datasets and ONNX inference - and I write the files and run the build myself rather than handing you code to paste.

                {project}

                Try asking for:
                  · a sentiment classifier over a CSV, using a fine-tuned model from the Hub
                  · semantic search: embed a corpus once, then rank it against a query
                  · a notebook comparing managed inference against ONNX Runtime
                  · a tokenizer trained on my own text, saved to disk

                Ctrl+Enter sends.
                """,
        });
    }
}
