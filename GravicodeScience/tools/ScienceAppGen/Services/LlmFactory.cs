using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using ScienceAppGen.Models;

namespace ScienceAppGen.Services;

/// <summary>
/// Turns <see cref="AppSettings"/> into a Semantic Kernel chat service and the matching request
/// settings. Shared by the assistant and the Python converter so the two cannot drift apart on a
/// provider quirk.
/// </summary>
public static class LlmFactory
{
    /// <summary>Registers the configured provider's chat completion service on a kernel builder.</summary>
    /// <param name="builder">The kernel builder.</param>
    /// <param name="settings">Provider, model, key and endpoint.</param>
    /// <param name="httpClient">
    /// Supplies the timeout. The connectors otherwise use HttpClient's 100-second default, which a
    /// reasoning model writing a long file exceeds.
    /// </param>
    public static IKernelBuilder AddChatCompletion(IKernelBuilder builder, AppSettings settings, HttpClient? httpClient = null)
    {
        switch (settings.Provider)
        {
            case LlmProvider.AzureOpenAI:
                builder.AddAzureOpenAIChatCompletion(
                    deploymentName: settings.Model,
                    endpoint: settings.Endpoint,
                    apiKey: settings.ApiKey,
                    httpClient: httpClient);
                break;

            case LlmProvider.OpenAI:
                if (string.IsNullOrWhiteSpace(settings.Endpoint))
                    builder.AddOpenAIChatCompletion(settings.Model, settings.ApiKey, httpClient: httpClient);
                else
                    builder.AddOpenAIChatCompletion(settings.Model, new Uri(settings.Endpoint), settings.ApiKey, httpClient: httpClient);
                break;

            case LlmProvider.Google:
                builder.AddGoogleAIGeminiChatCompletion(settings.Model, settings.ApiKey, httpClient: httpClient);
                break;

            case LlmProvider.Ollama:
                if (httpClient is null) builder.AddOllamaChatCompletion(settings.Model, new Uri(settings.Endpoint));
                else
                {
                    httpClient.BaseAddress ??= new Uri(settings.Endpoint);
                    builder.AddOllamaChatCompletion(settings.Model, httpClient);
                }
                break;

            case LlmProvider.Anthropic:
                // No official Semantic Kernel connector exists for Anthropic, so this is our own.
                builder.Services.AddSingleton<IChatCompletionService>(
                    new AnthropicChatCompletionService(
                        settings.ApiKey, settings.Model, settings.Endpoint,
                        settings.MaxTokens, httpClient?.Timeout ?? TimeSpan.FromSeconds(settings.TimeoutSeconds)));
                break;

            default:
                throw new NotSupportedException($"Provider {settings.Provider} is not supported.");
        }

        return builder;
    }

    /// <summary>A kernel with only the chat service - no plugins, so no function calling.</summary>
    public static Kernel CreatePlainKernel(AppSettings settings, HttpClient? httpClient = null)
        => AddChatCompletion(Kernel.CreateBuilder(), settings, httpClient).Build();

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
    public static bool UsesCompletionTokenLimit(string model)
    {
        var name = model.ToLowerInvariant();
        return name.StartsWith("o1", StringComparison.Ordinal)
               || name.StartsWith("o3", StringComparison.Ordinal)
               || name.StartsWith("o4", StringComparison.Ordinal)
               || name.StartsWith("gpt-5", StringComparison.Ordinal);
    }

    /// <summary>Request settings for the configured provider.</summary>
    /// <param name="settings">Current settings, for temperature and token limits.</param>
    /// <param name="functionCalling">Let the model call kernel functions (the assistant does; the converter does not).</param>
    /// <param name="maxTokens">Overrides <see cref="AppSettings.MaxTokens"/> when set.</param>
    public static PromptExecutionSettings CreateExecutionSettings(AppSettings settings, bool functionCalling, int? maxTokens = null)
    {
        var reasoning = UsesCompletionTokenLimit(settings.Model);
        var limit = maxTokens ?? settings.MaxTokens;
        var choice = functionCalling ? FunctionChoiceBehavior.Auto() : null;

        switch (settings.Provider)
        {
            case LlmProvider.AzureOpenAI:
            {
                var openAi = new AzureOpenAIPromptExecutionSettings { FunctionChoiceBehavior = choice };
                ApplyLimits(openAi, settings.Temperature, limit, reasoning);
                return openAi;
            }

            case LlmProvider.OpenAI:
            {
                var openAi = new OpenAIPromptExecutionSettings { FunctionChoiceBehavior = choice };
                ApplyLimits(openAi, settings.Temperature, limit, reasoning);
                return openAi;
            }

            default:
                // Google and Ollama read temperature from extension data; so does the custom
                // Anthropic service.
                return new PromptExecutionSettings
                {
                    ExtensionData = new Dictionary<string, object>
                    {
                        ["temperature"] = settings.Temperature,
                        ["max_tokens"] = limit,
                    },
                    FunctionChoiceBehavior = choice,
                };
        }
    }

    private static void ApplyLimits(OpenAIPromptExecutionSettings target, double temperature, int maxTokens, bool reasoning)
    {
        if (reasoning)
        {
            target.ExtraBody = new Dictionary<string, object?> { ["max_completion_tokens"] = maxTokens };
            // Temperature is left at the service default: these models reject anything else.
            return;
        }

        target.Temperature = temperature;
        target.MaxTokens = maxTokens;
    }
}
