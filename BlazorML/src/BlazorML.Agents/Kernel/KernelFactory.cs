using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using WicakOptions = BlazorML.Core.Configuration.ChatOptions;
using BlazorML.Core.Domain;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace BlazorML.Agents.Kernel;

/// <summary>
/// Builds a Semantic Kernel for whichever provider the workspace is configured to use.
/// <para>
/// Three of the four providers have a first-party connector. Anthropic does not — Microsoft has
/// never shipped <c>Connectors.Anthropic</c> — so it goes through <c>Anthropic.SDK</c>, which
/// exposes an <c>IChatClient</c> that Semantic Kernel adapts natively.
/// </para>
/// </summary>
public sealed class KernelFactory(ISettingsService settings, IServiceProvider services)
{
    /// <summary>
    /// Builds a kernel and registers the plugins on it. <paramref name="withPlugins"/> is false
    /// for the plain-completion path used by the LLM data modules, which must not call tools.
    /// </summary>
    public async Task<Microsoft.SemanticKernel.Kernel> CreateAsync(ChatProviderKind? provider = null,
        bool withPlugins = true, CancellationToken ct = default)
    {
        var options = await settings.GetAsync<WicakOptions>(SettingsSections.Chat, ct);
        var kind = provider ?? options.Provider;

        if (!options.IsProviderConfigured(kind))
        {
            throw new InvalidOperationException(
                $"{Describe(kind)} is not configured yet. Add its credentials under Settings → Profesor Wicak.");
        }

        var builder = Microsoft.SemanticKernel.Kernel.CreateBuilder();

        switch (kind)
        {
            case ChatProviderKind.OpenAI:
                if (string.IsNullOrWhiteSpace(options.OpenAI.Endpoint))
                {
                    builder.AddOpenAIChatCompletion(options.OpenAI.Model, options.OpenAI.ApiKey);
                }
                else
                {
                    // A custom endpoint covers Azure OpenAI-compatible gateways and local proxies.
                    builder.AddOpenAIChatCompletion(options.OpenAI.Model,
                        new Uri(options.OpenAI.Endpoint), options.OpenAI.ApiKey);
                }

                break;

            case ChatProviderKind.Gemini:
                builder.AddGoogleAIGeminiChatCompletion(options.Gemini.Model, options.Gemini.ApiKey);
                break;

            case ChatProviderKind.Ollama:
                builder.AddOllamaChatCompletion(options.Ollama.Model, new Uri(options.Ollama.Endpoint));
                break;

            case ChatProviderKind.Anthropic:
                builder.Services.AddSingleton<IChatCompletionService>(_ =>
                {
                    // MessagesEndpoint implements IChatClient explicitly, so it has to be cast
                    // before the Microsoft.Extensions.AI builder extensions are visible on it.
                    IChatClient endpoint = new Anthropic.SDK.AnthropicClient(options.Anthropic.ApiKey).Messages;

                    var client = endpoint.AsBuilder()
                        .ConfigureOptions(o => o.ModelId ??= options.Anthropic.Model)
                        .Build();

                    return client.AsChatCompletionService();
                });

                break;

            default:
                throw new NotSupportedException($"Provider '{kind}' is not supported.");
        }

        var kernel = builder.Build();

        if (withPlugins)
        {
            foreach (var plugin in services.GetServices<KernelPlugin>())
            {
                kernel.Plugins.Add(plugin);
            }
        }

        return kernel;
    }

    /// <summary>The model name in force for a provider, shown next to the session in the chat panel.</summary>
    public async Task<string> ModelNameAsync(ChatProviderKind kind, CancellationToken ct = default)
    {
        var options = await settings.GetAsync<WicakOptions>(SettingsSections.Chat, ct);

        return kind switch
        {
            ChatProviderKind.OpenAI => options.OpenAI.Model,
            ChatProviderKind.Anthropic => options.Anthropic.Model,
            ChatProviderKind.Gemini => options.Gemini.Model,
            ChatProviderKind.Ollama => options.Ollama.Model,
            _ => "unknown"
        };
    }

    public static string Describe(ChatProviderKind kind) => kind switch
    {
        ChatProviderKind.OpenAI => "OpenAI",
        ChatProviderKind.Anthropic => "Anthropic",
        ChatProviderKind.Gemini => "Google Gemini",
        ChatProviderKind.Ollama => "Ollama",
        _ => kind.ToString()
    };
}
