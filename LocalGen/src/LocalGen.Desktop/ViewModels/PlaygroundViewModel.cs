using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LocalGen.Core.Configuration;
using LocalGen.Core.Models;
using LocalGen.Desktop.Services;
using LocalGen.Kernel;
using LocalGen.Kernel.Mcp;
using LocalGen.Kernel.Skills;
using LocalGen.Rag.Documents;
using LocalGen.Runtime.Files;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace LocalGen.Desktop.ViewModels;

/// <summary>A built-in kernel function, as a checkbox on the tools panel.</summary>
public sealed partial class ToolToggle(string key, string label, string description) : ObservableObject
{
    [ObservableProperty]
    private bool _isEnabled = true;

    public string Key { get; } = key;

    public string Label { get; } = label;

    public string Description { get; } = description;
}

/// <summary>
/// The Playground: chat against a local model with kernel functions, skills, MCP servers and
/// file attachments.
/// </summary>
/// <remarks>
/// Built on <see cref="LocalGenAgent"/> so tool calls and their results stream into the transcript
/// as they happen. Seeing which tool ran, with what arguments, and what came back is the point of
/// a local playground — it is the part a hosted chat interface cannot show you.
/// </remarks>
public sealed partial class PlaygroundViewModel : ViewModelBase
{
    private readonly LocalGenKernelFactory _kernels;
    private readonly IModelStore _store;
    private readonly SkillStore _skills;
    private readonly McpManager _mcp;
    private readonly FileStore _files;
    private readonly DocumentExtractor _extractor;
    private readonly IFileDialogService _dialogs;
    private readonly LocalGenOptions _options;

    private CancellationTokenSource? _generation;

    [ObservableProperty]
    private ModelDescriptor? _selectedModel;

    [ObservableProperty]
    private ChatSession? _activeSession;

    [ObservableProperty]
    private string _prompt = string.Empty;

    [ObservableProperty]
    private string _systemPrompt = "You are a helpful assistant running locally on the user's machine. " +
                                   "Use markdown — tables, fenced code blocks and links — where it makes the answer clearer.";

    [ObservableProperty]
    private double _temperature = 0.7;

    [ObservableProperty]
    private int _maxTokens = 1024;

    [ObservableProperty]
    private double _topP = 0.9;

    [ObservableProperty]
    private bool _isGenerating;

    [ObservableProperty]
    private string _generatedCode = string.Empty;

    [ObservableProperty]
    private string _skillQuery = string.Empty;

    [ObservableProperty]
    private string _attachmentStatus = string.Empty;

    // Panel visibility. Both start open; hiding one gives the transcript the whole window, which
    // is what you want once a conversation is under way and the settings are set.
    [ObservableProperty]
    private bool _isSessionsVisible = true;

    [ObservableProperty]
    private bool _isConfigVisible = true;

    public PlaygroundViewModel(
        LocalGenKernelFactory kernels,
        IModelStore store,
        SkillStore skills,
        McpManager mcp,
        FileStore files,
        DocumentExtractor extractor,
        IFileDialogService dialogs,
        IOptions<LocalGenOptions> options)
    {
        _kernels = kernels;
        _store = store;
        _skills = skills;
        _mcp = mcp;
        _files = files;
        _extractor = extractor;
        _dialogs = dialogs;
        _options = options.Value;

        Tools =
        [
            new ToolToggle("math", "Math", "Exact arithmetic, percentages and statistics."),
            new ToolToggle("search", "Internet search", "Web search through Tavily."),
            new ToolToggle("scrape", "Web scrape", "Fetch a page and read its text."),
            new ToolToggle("download", "Download", "Save a file from a URL into the workspace."),
            new ToolToggle("code", "Code execution", "Run Python, JavaScript, C#, Bash or PowerShell."),
            new ToolToggle("time", "Time and date", "Current time, date arithmetic, time zones."),
            new ToolToggle("files", "File system", "Read and write files in the allowed directories."),
            new ToolToggle("skills", "Skills", "Load installed skills, their assets and scripts.")
        ];

        NewSession();
    }

    public ObservableCollection<ModelDescriptor> Models { get; } = [];

    public ObservableCollection<ChatSession> Sessions { get; } = [];

    public ObservableCollection<ToolToggle> Tools { get; }

    public ObservableCollection<Skill> InstalledSkills { get; } = [];

    public ObservableCollection<SkillGalleryEntry> GallerySkills { get; } = [];

    public ObservableCollection<McpServerDefinition> McpServers { get; } = [];

    /// <summary>Prompts that exercise the tools and markdown, so a new user has somewhere to start.</summary>
    public IReadOnlyList<string> SamplePrompts { get; } =
    [
        "What is 17.5% of 84,320? Show the calculation.",
        "Compare Q4_K_M, Q5_K_M and Q8_0 in a markdown table: size, quality and when to pick each.",
        "Write a Python script that plots a sine wave, run it, and tell me what it printed.",
        "How many days until 1 January next year?",
        "Search the web for the newest GGUF quantization formats and summarise the trade-offs.",
        "List the files in the workspace and tell me which is largest.",
        "Reply with JSON only: {\"language\": string, \"paradigms\": string[]} describing C#."
    ];

    public override async Task InitializeAsync()
    {
        await RefreshModelsAsync().ConfigureAwait(true);
        RefreshSkills();
        RefreshMcp();
    }

    // ─────────────────────────────  Panels  ─────────────────────────────

    [RelayCommand]
    private void ToggleSessions() => IsSessionsVisible = !IsSessionsVisible;

    [RelayCommand]
    private void ToggleConfig() => IsConfigVisible = !IsConfigVisible;

    // ─────────────────────────────  Sessions  ─────────────────────────────

    [RelayCommand]
    private void NewSession()
    {
        var session = new ChatSession($"Session {Sessions.Count + 1}");
        Sessions.Add(session);
        ActiveSession = session;
    }

    [RelayCommand]
    private void RemoveSession(ChatSession? session)
    {
        if (session is null)
        {
            return;
        }

        Sessions.Remove(session);

        // Never leave the screen without a conversation to type into.
        if (Sessions.Count == 0)
        {
            NewSession();
        }
        else if (ActiveSession == session)
        {
            ActiveSession = Sessions[^1];
        }
    }

    /// <summary>Empties the conversation but keeps the session and its name.</summary>
    [RelayCommand]
    private void ResetSession()
    {
        if (ActiveSession is null)
        {
            return;
        }

        ActiveSession.Entries.Clear();
        ActiveSession.History.Clear();
        ActiveSession.Pending.Clear();

        AttachmentStatus = string.Empty;
        ErrorMessage = string.Empty;

        OnPropertyChanged(nameof(ActiveSession));
    }

    // ─────────────────────────────  Attachments  ─────────────────────────────

    [RelayCommand]
    private Task AttachImageAsync() => AttachAsync(AttachmentKind.Image);

    [RelayCommand]
    private Task AttachDocumentAsync() => AttachAsync(AttachmentKind.Document);

    /// <summary>
    /// Uploads the picked files and stages them for the next message.
    /// </summary>
    /// <remarks>
    /// Files are uploaded rather than inlined so they have an address: the transcript renders an
    /// image by URL, a document link is clickable, and the same file attached to several messages
    /// is stored once.
    /// </remarks>
    private async Task AttachAsync(AttachmentKind kind)
    {
        if (ActiveSession is null)
        {
            return;
        }

        var paths = kind == AttachmentKind.Image
            ? await _dialogs.PickImagesAsync().ConfigureAwait(true)
            : await _dialogs.PickDocumentsAsync().ConfigureAwait(true);

        if (paths.Count == 0)
        {
            return;
        }

        AttachmentStatus = $"attaching {paths.Count} file(s)…";
        ErrorMessage = string.Empty;

        var attached = 0;

        foreach (var path in paths)
        {
            try
            {
                var stored = await _files.SaveAsync(path).ConfigureAwait(true);

                var extracted = string.Empty;

                if (kind == AttachmentKind.Document)
                {
                    // Extracted here rather than at send time so a file that cannot be read is
                    // reported while the user is still choosing, not after they hit Send.
                    var document = await _extractor.ExtractAsync(path).ConfigureAwait(true);
                    extracted = document.Text;
                }

                ActiveSession.Pending.Add(new Attachment
                {
                    Kind = kind,
                    FileName = stored.FileName,
                    Url = $"{_options.Server.BaseUrl}{stored.RelativeUrl}",
                    SizeBytes = stored.SizeBytes,
                    MediaType = stored.MediaType,
                    ExtractedText = extracted
                });

                attached++;
            }
            catch (Exception ex)
            {
                // One unreadable file should not abandon the rest of the selection.
                ErrorMessage = $"{Path.GetFileName(path)}: {ex.Message}";
            }
        }

        AttachmentStatus = attached > 0
            ? $"{ActiveSession.Pending.Count} attachment(s) staged"
            : string.Empty;
    }

    [RelayCommand]
    private void RemoveAttachment(Attachment? attachment)
    {
        if (attachment is null || ActiveSession is null)
        {
            return;
        }

        ActiveSession.Pending.Remove(attachment);

        AttachmentStatus = ActiveSession.Pending.Count > 0
            ? $"{ActiveSession.Pending.Count} attachment(s) staged"
            : string.Empty;
    }

    // ─────────────────────────────  Chat  ─────────────────────────────

    [RelayCommand]
    private async Task SendAsync()
    {
        if (IsGenerating || ActiveSession is null)
        {
            return;
        }

        var session = ActiveSession;
        var input = Prompt.Trim();

        // An attachment on its own is a reasonable turn — "look at this" is implied.
        if (input.Length == 0 && session.Pending.Count == 0)
        {
            return;
        }

        if (SelectedModel is null)
        {
            ErrorMessage = "Choose a model first. If the list is empty, download one from the Models screen.";
            return;
        }

        Prompt = string.Empty;
        ErrorMessage = string.Empty;
        IsGenerating = true;

        _generation = new CancellationTokenSource();

        session.Entries.Add(new ChatEntry
        {
            Kind = "user",
            Text = session.BuildTranscriptText(input)
        });

        // The system prompt heads the history so edits take effect on the next turn rather than
        // only on a fresh session.
        if (session.History.Count == 0 && !string.IsNullOrWhiteSpace(SystemPrompt))
        {
            session.History.AddSystemMessage(SystemPrompt);
        }

        session.History.Add(BuildHistoryMessage(session, input));

        session.Pending.Clear();
        AttachmentStatus = string.Empty;

        var reply = new ChatEntry { Kind = "assistant", Text = string.Empty };
        session.Entries.Add(reply);

        try
        {
            var agent = await _kernels
                .CreateAgentAsync(SelectedModel.Id, BuildToolSelection(), cancellationToken: _generation.Token)
                .ConfigureAwait(true);

            var settings = new LocalGenPromptExecutionSettings
            {
                Temperature = (float)Temperature,
                TopP = (float)TopP,
                MaxTokens = MaxTokens,
                FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(autoInvoke: false)
            };

            await foreach (var evt in agent
                .StreamAsync(session.History, settings, _generation.Token)
                .ConfigureAwait(true))
            {
                switch (evt)
                {
                    case AgentEvent.TextDelta delta:
                        reply.Text += delta.Text;
                        break;

                    case AgentEvent.ToolCallStarted started:
                        session.Entries.Add(new ChatEntry
                        {
                            Kind = "tool",
                            Tool = started.Tool,
                            Text = $"called with {started.ArgumentsJson}"
                        });
                        break;

                    case AgentEvent.ToolCallCompleted completed:
                        session.Entries.Add(new ChatEntry
                        {
                            Kind = "tool",
                            Tool = completed.Tool,
                            Text = completed.Error is null
                                ? $"returned in {completed.Duration.TotalMilliseconds:N0} ms\n\n{Trim(completed.Result)}"
                                : $"failed: {completed.Error}"
                        });

                        // A tool call means the assistant will keep going, so the transcript
                        // needs a fresh bubble for what it says next.
                        reply = new ChatEntry { Kind = "assistant", Text = string.Empty };
                        session.Entries.Add(reply);
                        break;

                    case AgentEvent.Failed failed:
                        session.Entries.Add(new ChatEntry { Kind = "error", Text = failed.Message });
                        break;
                }
            }

            // Remove the trailing bubble when the model finished on a tool result.
            if (string.IsNullOrEmpty(reply.Text))
            {
                session.Entries.Remove(reply);
            }
        }
        catch (OperationCanceledException)
        {
            reply.Text += "\n\n_[stopped]_";
        }
        catch (Exception ex)
        {
            session.Entries.Add(new ChatEntry { Kind = "error", Text = ex.Message });
        }
        finally
        {
            IsGenerating = false;
            OnPropertyChanged(nameof(ActiveSession));
        }
    }

    /// <summary>
    /// Builds the Semantic Kernel message, carrying attachments as content items so they reach
    /// the model rather than only the transcript.
    /// </summary>
    private static ChatMessageContent BuildHistoryMessage(ChatSession session, string text)
    {
        if (session.Pending.Count == 0)
        {
            return new ChatMessageContent(AuthorRole.User, text);
        }

        var items = new ChatMessageContentItemCollection();

        // Documents first: the model should read the material before the question.
        foreach (var attachment in session.Pending.Where(static a => a.Kind == AttachmentKind.Document))
        {
            items.Add(new TextContent(
                $"--- {attachment.FileName} ({attachment.Url}) ---\n{attachment.ExtractedText}"));
        }

        foreach (var attachment in session.Pending.Where(static a => a.Kind == AttachmentKind.Image))
        {
            items.Add(new ImageContent(new Uri(attachment.Url)) { MimeType = attachment.MediaType });
        }

        items.Add(new TextContent(string.IsNullOrWhiteSpace(text)
            ? "Describe the attached files."
            : text));

        return new ChatMessageContent(AuthorRole.User, items);
    }

    [RelayCommand]
    private void Stop() => _generation?.Cancel();

    [RelayCommand]
    private void UseSamplePrompt(string? sample)
    {
        if (!string.IsNullOrWhiteSpace(sample))
        {
            Prompt = sample;
        }
    }

    private ToolSelection BuildToolSelection()
    {
        bool Enabled(string key) => Tools.First(t => t.Key == key).IsEnabled;

        return new ToolSelection
        {
            Math = Enabled("math"),
            InternetSearch = Enabled("search"),
            WebScrape = Enabled("scrape"),
            Download = Enabled("download"),
            CodeExecution = Enabled("code"),
            TimeAndDate = Enabled("time"),
            FileSystem = Enabled("files"),
            Skills = Enabled("skills"),
            McpServers = [.. McpServers.Where(static s => s.Enabled).Select(static s => s.Name)]
        };
    }

    private static string Trim(string value) =>
        value.Length <= 600 ? value : value[..600] + " …";

    // ─────────────────────────────  Models  ─────────────────────────────

    [RelayCommand]
    private Task RefreshModelsAsync() => RunAsync(async () =>
    {
        var models = await _store.ListAsync().ConfigureAwait(true);

        Models.Clear();
        foreach (var model in models.Where(static m => m.Supports(ModelCapability.Chat)))
        {
            Models.Add(model);
        }

        SelectedModel ??= Models.FirstOrDefault();
    });

    // ─────────────────────────────  Skills  ─────────────────────────────

    [RelayCommand]
    private void RefreshSkills()
    {
        InstalledSkills.Clear();

        foreach (var skill in _skills.List())
        {
            InstalledSkills.Add(skill);
        }
    }

    [RelayCommand]
    private Task SearchSkillsAsync() => RunAsync(async () =>
    {
        var entries = await _skills.SearchGalleryAsync(SkillQuery).ConfigureAwait(true);

        GallerySkills.Clear();
        foreach (var entry in entries)
        {
            GallerySkills.Add(entry);
        }

        if (GallerySkills.Count == 0)
        {
            ErrorMessage = "The skill gallery returned nothing. Check your connection, " +
                           "or drop a skill folder into the skills directory by hand.";
        }
    });

    [RelayCommand]
    private Task InstallSkillAsync(SkillGalleryEntry? entry) => RunAsync(async () =>
    {
        if (entry is null)
        {
            return;
        }

        await _skills.InstallAsync(entry.DownloadUrl, entry.Name).ConfigureAwait(true);
        RefreshSkills();
    });

    [RelayCommand]
    private void RemoveSkill(Skill? skill)
    {
        if (skill is not null && _skills.Remove(skill.Name))
        {
            RefreshSkills();
        }
    }

    // ─────────────────────────────  MCP  ─────────────────────────────

    [RelayCommand]
    private void RefreshMcp()
    {
        McpServers.Clear();

        var configured = _mcp.ListDefinitions();

        foreach (var definition in configured)
        {
            McpServers.Add(definition);
        }

        // The curated gallery is offered alongside whatever the user has configured, so the
        // panel is useful before they know any server names.
        foreach (var suggestion in McpManager.Gallery)
        {
            if (!configured.Any(d => string.Equals(d.Name, suggestion.Name, StringComparison.OrdinalIgnoreCase)))
            {
                McpServers.Add(suggestion with { Enabled = false });
            }
        }
    }

    [RelayCommand]
    private void ToggleMcp(McpServerDefinition? definition)
    {
        if (definition is null)
        {
            return;
        }

        _mcp.AddDefinition(definition with { Enabled = !definition.Enabled });
        RefreshMcp();
    }

    // ─────────────────────────────  SDK code generation  ─────────────────────────────

    /// <summary>
    /// Emits a runnable snippet wired to the current model and settings. The Playground is where
    /// people work out what they want; this turns that configuration into code without them
    /// having to translate the sliders back into parameters by hand.
    /// </summary>
    [RelayCommand]
    private void GenerateCode(string? flavour)
    {
        var model = SelectedModel?.Id ?? "your-model";
        var endpoint = _options.Server.BaseUrl;

        GeneratedCode = flavour switch
        {
            "sdk" => $$"""
                // dotnet add package LocalGen.Sdk
                using LocalGen.Sdk;

                using var client = new LocalGenClient(new LocalGenClientOptions
                {
                    Endpoint = "{{endpoint}}",
                    DefaultModel = "{{model}}"
                });

                await foreach (var token in client.StreamTextAsync(
                    prompt: "Explain quantization in one paragraph.",
                    systemPrompt: {{Quote(SystemPrompt)}}))
                {
                    Console.Write(token);
                }
                """,

            // Three '$' so the emitted prompt template's own {{$input}} placeholder stays literal.
            "kernel" => $$$"""
                // dotnet add package Microsoft.SemanticKernel
                // dotnet add package LocalGen.Sdk
                using LocalGen.Sdk;
                using Microsoft.Extensions.AI;
                using Microsoft.SemanticKernel;

                var builder = Kernel.CreateBuilder();
                builder.Services.AddSingleton<IChatClient>(
                    new LocalGenChatClient("{{{endpoint}}}", "{{{model}}}"));

                var kernel = builder.Build();

                var answer = await kernel.InvokePromptAsync(
                    "Summarise the following in three bullets: {{$input}}",
                    new KernelArguments { ["input"] = "…" });

                Console.WriteLine(answer);
                """,

            "curl" => $$"""
                curl {{endpoint}}/v1/chat/completions \
                  -H "Content-Type: application/json" \
                  -d '{
                    "model": "{{model}}",
                    "messages": [
                      {"role": "system", "content": {{Quote(SystemPrompt)}}},
                      {"role": "user", "content": "Explain quantization in one paragraph."}
                    ],
                    "temperature": {{Temperature:0.##}},
                    "max_tokens": {{MaxTokens}},
                    "stream": true
                  }'
                """,

            "openai" => $$"""
                # pip install openai
                from openai import OpenAI

                client = OpenAI(base_url="{{endpoint}}/v1", api_key="not-needed")

                stream = client.chat.completions.create(
                    model="{{model}}",
                    messages=[
                        {"role": "system", "content": {{Quote(SystemPrompt)}}},
                        {"role": "user", "content": "Explain quantization in one paragraph."},
                    ],
                    temperature={{Temperature:0.##}},
                    max_tokens={{MaxTokens}},
                    stream=True,
                )

                for chunk in stream:
                    print(chunk.choices[0].delta.content or "", end="")
                """,

            _ => string.Empty
        };
    }

    private static string Quote(string value) =>
        "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n") + "\"";
}
