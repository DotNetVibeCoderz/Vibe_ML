using BlazorML.Agents.Kernel;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Domain;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using WicakOptions = BlazorML.Core.Configuration.ChatOptions;

namespace BlazorML.Agents.Chat;

/// <summary>
/// Bridges the LLM data modules to whichever provider is configured. Deliberately plain
/// completion with no tools: an LLM Sentiment node should label a column, not decide to go and
/// rewrite the user's experiment.
/// </summary>
public sealed class LlmActionRunner(KernelFactory kernels, ISettingsService settings) : ILlmActionRunner
{
    public bool IsConfigured
    {
        get
        {
            var options = settings.GetAsync<WicakOptions>(SettingsSections.Chat)
                .GetAwaiter().GetResult();

            return Enum.GetValues<ChatProviderKind>().Any(options.IsProviderConfigured);
        }
    }

    public async Task<string> CompleteAsync(string systemPrompt, string userPrompt, string? provider = null,
        double temperature = 0.2, CancellationToken ct = default)
    {
        var kind = Enum.TryParse<ChatProviderKind>(provider, true, out var parsed)
            ? parsed
            : (ChatProviderKind?)null;

        var kernel = await kernels.CreateAsync(kind, withPlugins: false, ct);
        var service = kernel.GetRequiredService<IChatCompletionService>();

        var chat = new ChatHistory(systemPrompt);
        chat.AddUserMessage(userPrompt);

        var settingsForCall = new PromptExecutionSettings
        {
            FunctionChoiceBehavior = FunctionChoiceBehavior.None(),
            ExtensionData = new Dictionary<string, object> { ["temperature"] = temperature }
        };

        var reply = await service.GetChatMessageContentAsync(chat, settingsForCall, kernel, ct);
        return reply.Content ?? string.Empty;
    }
}
