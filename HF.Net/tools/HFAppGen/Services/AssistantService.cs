using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using HFAppGen.Models;
using HFAppGen.Plugins;

namespace HFAppGen.Services;

/// <summary>A chunk of assistant output, or a note about a tool it called.</summary>
/// <param name="Text">Text to append to the reply.</param>
/// <param name="ToolName">Set when this chunk reports a tool call rather than prose.</param>
public readonly record struct AssistantChunk(string Text, string? ToolName = null);

/// <summary>
/// Jack, the Code Bender: the Semantic Kernel wiring behind the chat panel.
/// </summary>
/// <remarks>
/// <para>
/// The kernel is rebuilt whenever the provider, model or key changes, because a Semantic Kernel
/// instance binds its chat service at construction time.
/// </para>
/// <para>
/// Function calling is what makes this an app builder rather than a chat window:
/// <c>FunctionChoiceBehavior.Auto</c> lets the model decide to create a project, write files and
/// build them, and the kernel executes those calls and feeds the results back automatically.
/// </para>
/// </remarks>
public sealed class AssistantService(
    ProjectService projects,
    LogService logs)
{
    private Kernel? _kernel;
    private IChatCompletionService? _chat;
    private string _signature = "";

    /// <summary>The conversation so far, including the system prompt.</summary>
    public ChatHistory History { get; private set; } = [];

    /// <summary>The file-manipulation plugin, exposed so the UI can watch for changes.</summary>
    public ProjectPlugin? ProjectTools { get; private set; }

    /// <summary>True once a kernel has been built successfully.</summary>
    public bool IsReady => _kernel is not null && _chat is not null;

    /// <summary>The assistant's name, used throughout the UI.</summary>
    public const string Name = "Jack";

    /// <summary>The assistant's full title.</summary>
    public const string FullName = "Jack - The Code Bender";

    /// <summary>
    /// Builds or rebuilds the kernel for the current settings. Safe to call repeatedly.
    /// </summary>
    public void Configure(AppSettings settings)
    {
        var signature = $"{settings.Provider}|{settings.Model}|{settings.Endpoint}|{settings.ApiKey.Length}";
        if (_kernel is not null && signature == _signature) return;

        var builder = Kernel.CreateBuilder();

        switch (settings.Provider)
        {
            case LlmProvider.AzureOpenAI:
                builder.AddAzureOpenAIChatCompletion(
                    deploymentName: settings.Model,
                    endpoint: settings.Endpoint,
                    apiKey: settings.ApiKey);
                break;

            case LlmProvider.OpenAI:
                if (string.IsNullOrWhiteSpace(settings.Endpoint))
                    builder.AddOpenAIChatCompletion(settings.Model, settings.ApiKey);
                else
                    builder.AddOpenAIChatCompletion(settings.Model, new Uri(settings.Endpoint), settings.ApiKey);
                break;

            case LlmProvider.Google:
                builder.AddGoogleAIGeminiChatCompletion(settings.Model, settings.ApiKey);
                break;

            case LlmProvider.Ollama:
                builder.AddOllamaChatCompletion(settings.Model, new Uri(settings.Endpoint));
                break;

            case LlmProvider.Anthropic:
                // No official Semantic Kernel connector exists for Anthropic, so this is our own.
                builder.Services.AddSingleton<IChatCompletionService>(
                    new AnthropicChatCompletionService(
                        settings.ApiKey, settings.Model, settings.Endpoint,
                        settings.MaxTokens, TimeSpan.FromSeconds(settings.TimeoutSeconds)));
                break;

            default:
                throw new NotSupportedException($"Provider {settings.Provider} is not supported.");
        }

        var kernel = builder.Build();

        ProjectTools = new ProjectPlugin(projects, logs);
        kernel.Plugins.AddFromObject(ProjectTools, "Project");
        kernel.Plugins.AddFromObject(new CommonToolsPlugin(settings, logs), "Tools");
        kernel.Plugins.AddFromObject(new HFNetReferencePlugin(), "HFNet");

        _kernel = kernel;
        _chat = kernel.GetRequiredService<IChatCompletionService>();
        _signature = signature;

        ResetHistory(settings);

        var functions = kernel.Plugins.SelectMany(p => p).Count();
        logs.Info("assistant", $"{FullName} ready on {settings.Provider}/{settings.Model} with {functions} tools.");
    }

    /// <summary>Clears the conversation, keeping the system prompt.</summary>
    public void ResetHistory(AppSettings settings)
    {
        History = [];
        History.AddSystemMessage(string.IsNullOrWhiteSpace(settings.SystemPrompt)
            ? $"You are {FullName}, a coding assistant that builds .NET applications."
            : settings.SystemPrompt);
    }

    /// <summary>
    /// Sends a message and streams the reply.
    /// </summary>
    /// <param name="message">What the user typed.</param>
    /// <param name="settings">Current settings, for temperature and token limits.</param>
    /// <param name="imagePaths">Optional images to attach.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    public async IAsyncEnumerable<AssistantChunk> SendAsync(
        string message,
        AppSettings settings,
        IReadOnlyList<string>? imagePaths = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_kernel is null || _chat is null)
        {
            yield return new AssistantChunk("The assistant is not configured. Open Tools > Settings.");
            yield break;
        }

        AddUserMessage(message, imagePaths);

        var execution = BuildExecutionSettings(settings);
        var reply = new System.Text.StringBuilder();
        var seenTools = new HashSet<string>(StringComparer.Ordinal);

        var enumerator = _chat
            .GetStreamingChatMessageContentsAsync(History, execution, _kernel, cancellationToken)
            .GetAsyncEnumerator(cancellationToken);

        try
        {
            while (true)
            {
                // C# forbids `yield` inside catch, so each step reports its outcome and the
                // yielding happens afterwards, outside the try.
                var step = await NextAsync(enumerator);

                if (step.Failure is not null)
                {
                    yield return new AssistantChunk(step.Failure);
                    yield break;
                }

                if (step.Finished) break;

                // Surface tool calls so the user sees what the assistant is doing to their project.
                foreach (var tool in ExtractToolNames(step.Chunk))
                {
                    if (!seenTools.Add(tool)) continue;
                    logs.Assistant($"calling {tool}");
                    yield return new AssistantChunk("", tool);
                }

                var content = step.Chunk?.Content;
                if (string.IsNullOrEmpty(content)) continue;

                reply.Append(content);
                yield return new AssistantChunk(content);
            }
        }
        finally
        {
            await enumerator.DisposeAsync();
        }

        if (reply.Length > 0) History.AddAssistantMessage(reply.ToString());
    }

    /// <summary>One step of the stream: a chunk, the end of it, or a message explaining a failure.</summary>
    private readonly record struct StreamStep(
        StreamingChatMessageContent? Chunk, bool Finished, string? Failure);

    private async Task<StreamStep> NextAsync(IAsyncEnumerator<StreamingChatMessageContent> enumerator)
    {
        try
        {
            return await enumerator.MoveNextAsync()
                ? new StreamStep(enumerator.Current, false, null)
                : new StreamStep(null, true, null);
        }
        catch (OperationCanceledException)
        {
            return new StreamStep(null, true, "\n\n_(cancelled)_");
        }
        catch (Exception ex)
        {
            logs.Error("assistant", ex.Message);
            return new StreamStep(null, true, $"\n\n**Request failed.** {ex.Message}");
        }
    }

    private void AddUserMessage(string message, IReadOnlyList<string>? imagePaths)
    {
        // The open project can change at any time, so the context is refreshed on every turn
        // rather than baked into the system prompt when the kernel was built.
        History.AddSystemMessage(BuildContextSuffix());

        if (imagePaths is null || imagePaths.Count == 0)
        {
            History.AddUserMessage(message);
            return;
        }

        var items = new ChatMessageContentItemCollection { new TextContent(message) };
        foreach (var path in imagePaths)
        {
            try
            {
                var bytes = File.ReadAllBytes(path);
                items.Add(new ImageContent(bytes, MimeTypeFor(path)));
                logs.Info("assistant", $"Attached {Path.GetFileName(path)} ({bytes.Length / 1024} KB)");
            }
            catch (Exception ex)
            {
                logs.Warning("assistant", $"Could not attach {path}: {ex.Message}");
            }
        }

        History.Add(new ChatMessageContent(AuthorRole.User, items));
    }

    /// <summary>
    /// True for the reasoning-model families, which changed the request contract.
    /// </summary>
    /// <remarks>
    /// o1, o3, o4 and gpt-5 reject <c>max_tokens</c> outright - "Unsupported parameter:
    /// 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead" - and
    /// most of them accept only the default temperature. Semantic Kernel 1.79 still maps
    /// <c>MaxTokens</c> to the legacy field, so the newer name is injected through
    /// <c>ExtraBody</c> and the legacy one is left unset.
    /// </remarks>
    internal static bool UsesCompletionTokenLimit(string model)
    {
        var name = model.ToLowerInvariant();
        return name.StartsWith("o1", StringComparison.Ordinal)
               || name.StartsWith("o3", StringComparison.Ordinal)
               || name.StartsWith("o4", StringComparison.Ordinal)
               || name.StartsWith("gpt-5", StringComparison.Ordinal);
    }

    private PromptExecutionSettings BuildExecutionSettings(AppSettings settings)
    {
        var reasoning = UsesCompletionTokenLimit(settings.Model);

        switch (settings.Provider)
        {
            case LlmProvider.AzureOpenAI:
            {
                var openAi = new AzureOpenAIPromptExecutionSettings
                {
                    FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(),
                };
                ApplyLimits(openAi, settings, reasoning);
                return openAi;
            }

            case LlmProvider.OpenAI:
            {
                var openAi = new OpenAIPromptExecutionSettings
                {
                    FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(),
                };
                ApplyLimits(openAi, settings, reasoning);
                return openAi;
            }

            default:
                // Google and Ollama read temperature from extension data; so does the custom
                // Anthropic service.
                return new PromptExecutionSettings
                {
                    ExtensionData = new Dictionary<string, object> { ["temperature"] = settings.Temperature },
                    FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(),
                };
        }
    }

    private static void ApplyLimits(OpenAIPromptExecutionSettings target, AppSettings settings, bool reasoning)
    {
        if (reasoning)
        {
            target.ExtraBody = new Dictionary<string, object?>
            {
                ["max_completion_tokens"] = settings.MaxTokens,
            };
            // Temperature is left at the service default: these models reject anything else.
            return;
        }

        target.Temperature = settings.Temperature;
        target.MaxTokens = settings.MaxTokens;
    }

    /// <summary>Pulls function names out of a streaming chunk's metadata, whatever shape it takes.</summary>
    private static IEnumerable<string> ExtractToolNames(StreamingChatMessageContent? chunk)
    {
        if (chunk?.Metadata is null) yield break;

        if (!chunk.Metadata.TryGetValue("ChatResponseMessage.FunctionToolCalls", out var raw) || raw is null)
            yield break;

        if (raw is System.Collections.IEnumerable sequence and not string)
        {
            foreach (var item in sequence)
            {
                var name = item?.GetType().GetProperty("FunctionName")?.GetValue(item)?.ToString()
                           ?? item?.GetType().GetProperty("Name")?.GetValue(item)?.ToString();
                if (!string.IsNullOrWhiteSpace(name)) yield return name;
            }
        }
    }

    private string BuildContextSuffix()
    {
        var context = projects.HasProject
            ? $"Currently open project: '{projects.ProjectName}' at {projects.ProjectRoot}."
            : "No project is open. Use CreateProject when the user asks for an application.";

        return $"[context] {context} Today is {DateTime.Now:dddd, d MMMM yyyy}.";
    }

    private static string MimeTypeFor(string path) => Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".png" => "image/png",
        ".gif" => "image/gif",
        ".webp" => "image/webp",
        ".bmp" => "image/bmp",
        _ => "image/jpeg",
    };
}
