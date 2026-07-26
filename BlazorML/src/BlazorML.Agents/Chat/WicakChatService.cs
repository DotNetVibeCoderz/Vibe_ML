using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using BlazorML.Agents.Kernel;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using WicakOptions = BlazorML.Core.Configuration.ChatOptions;

namespace BlazorML.Agents.Chat;

/// <summary>
/// Profesor Wicak's conversation layer: sessions, history, attachments and streaming.
/// </summary>
public sealed class WicakChatService(
    AppDbContext db,
    KernelFactory kernels,
    ISettingsService settings)
{
    public async Task<List<ChatSession>> SessionsAsync(string? ownerId, CancellationToken ct = default) =>
        await db.ChatSessions.AsNoTracking()
            .Where(s => s.OwnerId == ownerId)
            .OrderByDescending(s => s.UpdatedAt ?? s.CreatedAt)
            .ToListAsync(ct);

    public async Task<ChatSession> CreateSessionAsync(string? ownerId, string? experimentId,
        CancellationToken ct = default)
    {
        var options = await settings.GetAsync<WicakOptions>(SettingsSections.Chat, ct);

        var session = new ChatSession
        {
            OwnerId = ownerId,
            ExperimentId = experimentId,
            Provider = options.Provider,
            Model = await kernels.ModelNameAsync(options.Provider, ct),
            Title = "Sesi baru"
        };

        db.ChatSessions.Add(session);
        await db.SaveChangesAsync(ct);
        return session;
    }

    public async Task<List<ChatMessage>> MessagesAsync(string sessionId, CancellationToken ct = default) =>
        await db.ChatMessages.AsNoTracking()
            .Where(m => m.SessionId == sessionId)
            .OrderBy(m => m.Sequence)
            .ToListAsync(ct);

    public async Task DeleteSessionAsync(string sessionId, CancellationToken ct = default)
    {
        var session = await db.ChatSessions.FirstOrDefaultAsync(s => s.Id == sessionId, ct);
        if (session is null)
        {
            return;
        }

        db.ChatSessions.Remove(session);
        await db.SaveChangesAsync(ct);
    }

    /// <summary>Clears the messages but keeps the session, so the user does not lose their place.</summary>
    public async Task ResetSessionAsync(string sessionId, CancellationToken ct = default)
    {
        var messages = await db.ChatMessages.Where(m => m.SessionId == sessionId).ToListAsync(ct);
        db.ChatMessages.RemoveRange(messages);

        var session = await db.ChatSessions.FirstOrDefaultAsync(s => s.Id == sessionId, ct);
        if (session is not null)
        {
            session.Title = "Sesi baru";
        }

        await db.SaveChangesAsync(ct);
    }

    /// <summary>
    /// Streams a reply, saving both the question and the answer. The chunks are yielded as they
    /// arrive so the panel can render progressively rather than sitting blank.
    /// </summary>
    public async IAsyncEnumerable<string> SendAsync(string sessionId, string prompt,
        List<ChatAttachment>? attachments, string? experimentId,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var options = await settings.GetAsync<WicakOptions>(SettingsSections.Chat, ct);

        var session = await db.ChatSessions.FirstOrDefaultAsync(s => s.Id == sessionId, ct)
            ?? throw new InvalidOperationException("That chat session no longer exists.");

        var history = await db.ChatMessages
            .Where(m => m.SessionId == sessionId)
            .OrderBy(m => m.Sequence)
            .ToListAsync(ct);

        var sequence = history.Count == 0 ? 0 : history[^1].Sequence + 1;

        var question = new ChatMessage
        {
            SessionId = sessionId,
            Role = ChatRole.User,
            Content = prompt,
            Sequence = sequence++,
            OwnerId = session.OwnerId,
            AttachmentsJson = attachments is { Count: > 0 } ? JsonSerializer.Serialize(attachments) : null
        };

        db.ChatMessages.Add(question);

        // Name the session from its first question so the list is scannable.
        if (history.Count == 0)
        {
            session.Title = prompt.Length <= 60 ? prompt : prompt[..57] + "…";
        }

        await db.SaveChangesAsync(ct);

        var chat = BuildHistory(options, history, prompt, attachments, experimentId);

        var kernel = await kernels.CreateAsync(session.Provider, options.EnableFunctionCalling, ct);
        var service = kernel.GetRequiredService<IChatCompletionService>();

        var executionSettings = new PromptExecutionSettings
        {
            FunctionChoiceBehavior = options.EnableFunctionCalling
                ? FunctionChoiceBehavior.Auto()
                : FunctionChoiceBehavior.None(),
            ExtensionData = new Dictionary<string, object>
            {
                ["temperature"] = options.Temperature,
                ["max_tokens"] = options.MaxTokens
            }
        };

        var answer = new StringBuilder();

        await foreach (var chunk in service.GetStreamingChatMessageContentsAsync(
                           chat, executionSettings, kernel, ct))
        {
            if (string.IsNullOrEmpty(chunk.Content))
            {
                continue;
            }

            answer.Append(chunk.Content);
            yield return chunk.Content;
        }

        db.ChatMessages.Add(new ChatMessage
        {
            SessionId = sessionId,
            Role = ChatRole.Assistant,
            Content = answer.ToString(),
            Sequence = sequence,
            OwnerId = session.OwnerId
        });

        session.UpdatedAt = DateTimeOffset.UtcNow;
        await db.SaveChangesAsync(CancellationToken.None);
    }

    /// <summary>
    /// Assembles what the model sees: the persona, the recent turns, and the current question
    /// with any attachments. Only the last <see cref="WicakOptions.HistoryTurns"/> turns are
    /// replayed, which keeps a long session from growing the request without bound.
    /// </summary>
    private static ChatHistory BuildHistory(WicakOptions options, List<ChatMessage> history,
        string prompt, List<ChatAttachment>? attachments, string? experimentId)
    {
        var system = new StringBuilder(options.SystemPrompt);

        if (!string.IsNullOrWhiteSpace(experimentId))
        {
            system.AppendLine();
            system.AppendLine();
            system.AppendLine($"The user currently has experiment '{experimentId}' open. " +
                              "When they ask about \"this experiment\" or ask you to change the flow, use that id.");
        }

        var chat = new ChatHistory(system.ToString());

        foreach (var message in history.TakeLast(options.HistoryTurns * 2))
        {
            if (message.Role == ChatRole.User)
            {
                chat.AddUserMessage(message.Content);
            }
            else if (message.Role == ChatRole.Assistant)
            {
                chat.AddAssistantMessage(message.Content);
            }
        }

        if (attachments is { Count: > 0 })
        {
            var items = new ChatMessageContentItemCollection { new TextContent(prompt) };

            foreach (var attachment in attachments)
            {
                if (attachment.Kind == AttachmentKind.Image)
                {
                    // Images go in as image content the model can actually look at.
                    items.Add(new ImageContent(new Uri(attachment.Url)));
                }
                else
                {
                    // Documents go in as a link in the text: the model can fetch them with
                    // ReadFileFromUrl if it decides the contents matter.
                    items.Add(new TextContent(
                        $"\n\nAttached document: {attachment.Name} — {attachment.Url}"));
                }
            }

            chat.AddUserMessage(items);
        }
        else
        {
            chat.AddUserMessage(prompt);
        }

        return chat;
    }
}
