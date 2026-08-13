namespace ScienceAppGen.Models;

/// <summary>The LLM back ends ScienceAppGen can talk to.</summary>
public enum LlmProvider
{
    /// <summary>api.openai.com, or any OpenAI-compatible gateway via a custom endpoint.</summary>
    OpenAI,

    /// <summary>An Azure OpenAI deployment. Requires an endpoint.</summary>
    AzureOpenAI,

    /// <summary>Anthropic Claude, through a hand-written chat completion service.</summary>
    Anthropic,

    /// <summary>Google Gemini.</summary>
    Google,

    /// <summary>A local Ollama server. Requires an endpoint, needs no key.</summary>
    Ollama,
}

/// <summary>
/// Everything in <c>app.config</c>, as a strongly typed object.
/// </summary>
/// <remarks>
/// This is a plain mutable record rather than a set of scattered <c>ConfigurationManager</c>
/// lookups so the settings dialog can bind to it, validate it and write the whole thing back in
/// one step. <see cref="Services.ConfigurationService"/> owns loading and saving.
/// </remarks>
public sealed class AppSettings
{
    // ---------------------------------------------------------------- llm

    /// <summary>Which back end to talk to.</summary>
    public LlmProvider Provider { get; set; } = LlmProvider.AzureOpenAI;

    /// <summary>Model or deployment name.</summary>
    public string Model { get; set; } = "gpt-5-mini";

    /// <summary>API key. Blank when the provider needs none, as with Ollama.</summary>
    public string ApiKey { get; set; } = "";

    /// <summary>Base endpoint. Required for Azure OpenAI and Ollama.</summary>
    public string Endpoint { get; set; } = "";

    /// <summary>Sampling temperature.</summary>
    public double Temperature { get; set; } = 0.3;

    /// <summary>Cap on generated tokens per reply.</summary>
    public int MaxTokens { get; set; } = 8192;

    /// <summary>How long to wait for a reply before giving up.</summary>
    public int TimeoutSeconds { get; set; } = 180;

    /// <summary>The models offered in the chat panel's picker.</summary>
    public List<string> AvailableModels { get; set; } = [];

    /// <summary>The assistant's instructions.</summary>
    public string SystemPrompt { get; set; } = "";

    // ---------------------------------------------------------------- tools

    /// <summary>Tavily key for <c>SearchInternet</c>. Search is disabled when blank.</summary>
    public string TavilyApiKey { get; set; } = "";

    /// <summary>How many search results to return.</summary>
    public int TavilyMaxResults { get; set; } = 5;

    /// <summary>Timeout for search and page fetches.</summary>
    public int WebTimeoutSeconds { get; set; } = 30;

    /// <summary>Whether the assistant may write files without asking each time.</summary>
    public bool AutoApproveFileWrites { get; set; } = true;

    // ---------------------------------------------------------------- editor

    /// <summary>Whether the editor gutter shows line numbers.</summary>
    public bool ShowLineNumbers { get; set; } = true;

    /// <summary>Editor font stack.</summary>
    public string EditorFontFamily { get; set; } = "Cascadia Mono,Consolas,monospace";

    /// <summary>Editor font size.</summary>
    public double EditorFontSize { get; set; } = 13;

    /// <summary>Spaces per indent level.</summary>
    public int TabSize { get; set; } = 4;

    /// <summary>Whether long lines wrap.</summary>
    public bool WordWrap { get; set; }

    /// <summary>Whether the line under the caret is tinted.</summary>
    public bool HighlightCurrentLine { get; set; } = true;

    // ---------------------------------------------------------------- workspace

    /// <summary>The project to reopen on launch.</summary>
    public string LastProject { get; set; } = "";

    /// <summary>Where the New Project dialog starts.</summary>
    public string ProjectsRoot { get; set; } = "";

    /// <summary>Chat panel width in pixels.</summary>
    public double ChatPanelWidth { get; set; } = 420;

    /// <summary>Whether the chat panel is shown.</summary>
    public bool ChatPanelVisible { get; set; } = true;

    /// <summary>Explorer width in pixels.</summary>
    public double ExplorerWidth { get; set; } = 260;

    /// <summary>Logs panel height in pixels.</summary>
    public double LogsHeight { get; set; } = 180;

    /// <summary>Dark or Light.</summary>
    public string Theme { get; set; } = "Dark";

    /// <summary>
    /// Whether the current provider has everything it needs to make a call.
    /// </summary>
    public bool IsConfigured => Provider switch
    {
        LlmProvider.Ollama => !string.IsNullOrWhiteSpace(Endpoint) && !string.IsNullOrWhiteSpace(Model),
        LlmProvider.AzureOpenAI => !string.IsNullOrWhiteSpace(ApiKey)
                                   && !string.IsNullOrWhiteSpace(Endpoint)
                                   && !string.IsNullOrWhiteSpace(Model),
        _ => !string.IsNullOrWhiteSpace(ApiKey) && !string.IsNullOrWhiteSpace(Model),
    };

    /// <summary>A one-line explanation of what is still missing, or null when ready.</summary>
    public string? MissingConfiguration
    {
        get
        {
            if (string.IsNullOrWhiteSpace(Model)) return "No model set. Open Tools > Settings.";
            if (Provider == LlmProvider.Ollama)
                return string.IsNullOrWhiteSpace(Endpoint)
                    ? "Ollama needs an endpoint, usually http://localhost:11434."
                    : null;
            if (string.IsNullOrWhiteSpace(ApiKey))
                return $"{Provider} needs an API key. Open Tools > Settings.";
            if (Provider == LlmProvider.AzureOpenAI && string.IsNullOrWhiteSpace(Endpoint))
                return "Azure OpenAI needs the resource endpoint, e.g. https://your-resource.openai.azure.com/.";
            return null;
        }
    }
}
