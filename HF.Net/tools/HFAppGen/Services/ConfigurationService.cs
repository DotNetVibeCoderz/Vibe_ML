using System.Globalization;
using System.Xml.Linq;
using HFAppGen.Models;

namespace HFAppGen.Services;

/// <summary>
/// Loads and saves <see cref="AppSettings"/> from <c>app.config</c>.
/// </summary>
/// <remarks>
/// The file is read and written as XML rather than through <c>ConfigurationManager</c>, which can
/// read <c>appSettings</c> but cannot write them back without a round trip that reorders and
/// reformats the file. Editing the XML directly keeps comments and layout intact, so a config
/// edited in the UI still reads like the one shipped in source control.
/// <para>
/// Environment variables win over the file, so a key can be supplied without ever writing it to
/// disk: <c>HFAPPGEN_APIKEY</c>, <c>HFAPPGEN_ENDPOINT</c>, <c>TAVILY_API_KEY</c>.
/// </para>
/// </remarks>
public sealed class ConfigurationService
{
    private readonly string _configPath;

    /// <summary>Creates a service bound to the running application's config file.</summary>
    public ConfigurationService(string? configPath = null)
        => _configPath = configPath ?? Path.Combine(AppContext.BaseDirectory, "app.config");

    /// <summary>Where the settings are read from and written to.</summary>
    public string ConfigPath => _configPath;

    /// <summary>Reads the settings, falling back to defaults for anything absent or malformed.</summary>
    public AppSettings Load()
    {
        var settings = new AppSettings();
        var values = ReadAppSettings();

        settings.Provider = ParseEnum(values, "llm.provider", LlmProvider.AzureOpenAI);
        settings.Model = Get(values, "llm.model", settings.Model);
        settings.ApiKey = Environment.GetEnvironmentVariable("HFAPPGEN_APIKEY")
                          ?? Get(values, "llm.apiKey", "");
        settings.Endpoint = Environment.GetEnvironmentVariable("HFAPPGEN_ENDPOINT")
                            ?? Get(values, "llm.endpoint", "");
        settings.Temperature = ParseDouble(values, "llm.temperature", 0.3);
        settings.MaxTokens = ParseInt(values, "llm.maxTokens", 8192);
        settings.TimeoutSeconds = ParseInt(values, "llm.timeoutSeconds", 180);
        settings.SystemPrompt = Get(values, "llm.systemPrompt", "");
        settings.AvailableModels = Get(values, "llm.availableModels", "")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToList();

        settings.TavilyApiKey = Environment.GetEnvironmentVariable("TAVILY_API_KEY")
                                ?? Get(values, "tools.tavilyApiKey", "");
        settings.TavilyMaxResults = ParseInt(values, "tools.tavilyMaxResults", 5);
        settings.WebTimeoutSeconds = ParseInt(values, "tools.webTimeoutSeconds", 30);
        settings.AutoApproveFileWrites = ParseBool(values, "tools.autoApproveFileWrites", true);

        settings.ShowLineNumbers = ParseBool(values, "editor.showLineNumbers", true);
        settings.EditorFontFamily = Get(values, "editor.fontFamily", settings.EditorFontFamily);
        settings.EditorFontSize = ParseDouble(values, "editor.fontSize", 13);
        settings.TabSize = ParseInt(values, "editor.tabSize", 4);
        settings.WordWrap = ParseBool(values, "editor.wordWrap", false);
        settings.HighlightCurrentLine = ParseBool(values, "editor.highlightCurrentLine", true);

        settings.LastProject = Get(values, "workspace.lastProject", "");
        settings.ProjectsRoot = Get(values, "workspace.projectsRoot", "");
        settings.ChatPanelWidth = ParseDouble(values, "workspace.chatPanelWidth", 420);
        settings.ChatPanelVisible = ParseBool(values, "workspace.chatPanelVisible", true);
        settings.ExplorerWidth = ParseDouble(values, "workspace.explorerWidth", 260);
        settings.LogsHeight = ParseDouble(values, "workspace.logsHeight", 180);
        settings.Theme = Get(values, "workspace.theme", "Dark");

        if (string.IsNullOrWhiteSpace(settings.ProjectsRoot))
        {
            settings.ProjectsRoot = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "HFAppGen");
        }

        return settings;
    }

    /// <summary>Writes the settings back, preserving comments and element order.</summary>
    public void Save(AppSettings settings)
    {
        XDocument document;
        if (File.Exists(_configPath))
        {
            document = XDocument.Load(_configPath, LoadOptions.PreserveWhitespace);
        }
        else
        {
            document = new XDocument(new XElement("configuration", new XElement("appSettings")));
        }

        var appSettings = document.Root?.Element("appSettings");
        if (appSettings is null)
        {
            appSettings = new XElement("appSettings");
            document.Root!.Add(appSettings);
        }

        void Set(string key, string value)
        {
            var existing = appSettings.Elements("add")
                .FirstOrDefault(e => string.Equals((string?)e.Attribute("key"), key, StringComparison.Ordinal));

            if (existing is not null) existing.SetAttributeValue("value", value);
            else appSettings.Add(new XElement("add", new XAttribute("key", key), new XAttribute("value", value)));
        }

        Set("llm.provider", settings.Provider.ToString());
        Set("llm.model", settings.Model);
        Set("llm.apiKey", settings.ApiKey);
        Set("llm.endpoint", settings.Endpoint);
        Set("llm.temperature", settings.Temperature.ToString(CultureInfo.InvariantCulture));
        Set("llm.maxTokens", settings.MaxTokens.ToString(CultureInfo.InvariantCulture));
        Set("llm.timeoutSeconds", settings.TimeoutSeconds.ToString(CultureInfo.InvariantCulture));
        Set("llm.availableModels", string.Join(',', settings.AvailableModels));
        Set("llm.systemPrompt", settings.SystemPrompt);

        Set("tools.tavilyApiKey", settings.TavilyApiKey);
        Set("tools.tavilyMaxResults", settings.TavilyMaxResults.ToString(CultureInfo.InvariantCulture));
        Set("tools.webTimeoutSeconds", settings.WebTimeoutSeconds.ToString(CultureInfo.InvariantCulture));
        Set("tools.autoApproveFileWrites", settings.AutoApproveFileWrites ? "true" : "false");

        Set("editor.showLineNumbers", settings.ShowLineNumbers ? "true" : "false");
        Set("editor.fontFamily", settings.EditorFontFamily);
        Set("editor.fontSize", settings.EditorFontSize.ToString(CultureInfo.InvariantCulture));
        Set("editor.tabSize", settings.TabSize.ToString(CultureInfo.InvariantCulture));
        Set("editor.wordWrap", settings.WordWrap ? "true" : "false");
        Set("editor.highlightCurrentLine", settings.HighlightCurrentLine ? "true" : "false");

        Set("workspace.lastProject", settings.LastProject);
        Set("workspace.projectsRoot", settings.ProjectsRoot);
        Set("workspace.chatPanelWidth", settings.ChatPanelWidth.ToString("0", CultureInfo.InvariantCulture));
        Set("workspace.chatPanelVisible", settings.ChatPanelVisible ? "true" : "false");
        Set("workspace.explorerWidth", settings.ExplorerWidth.ToString("0", CultureInfo.InvariantCulture));
        Set("workspace.logsHeight", settings.LogsHeight.ToString("0", CultureInfo.InvariantCulture));
        Set("workspace.theme", settings.Theme);

        document.Save(_configPath);
    }

    private Dictionary<string, string> ReadAppSettings()
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!File.Exists(_configPath)) return values;

        try
        {
            var document = XDocument.Load(_configPath);
            foreach (var element in document.Root?.Element("appSettings")?.Elements("add") ?? [])
            {
                var key = (string?)element.Attribute("key");
                var value = (string?)element.Attribute("value");
                if (key is not null) values[key] = value ?? "";
            }
        }
        catch (Exception)
        {
            // A malformed config should not stop the app from starting; defaults apply instead.
        }
        return values;
    }

    private static string Get(Dictionary<string, string> values, string key, string fallback)
        => values.TryGetValue(key, out var value) && value.Length > 0 ? value : fallback;

    private static int ParseInt(Dictionary<string, string> values, string key, int fallback)
        => values.TryGetValue(key, out var value)
           && int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
            ? parsed : fallback;

    private static double ParseDouble(Dictionary<string, string> values, string key, double fallback)
        => values.TryGetValue(key, out var value)
           && double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed)
            ? parsed : fallback;

    private static bool ParseBool(Dictionary<string, string> values, string key, bool fallback)
        => values.TryGetValue(key, out var value) && bool.TryParse(value, out var parsed) ? parsed : fallback;

    private static TEnum ParseEnum<TEnum>(Dictionary<string, string> values, string key, TEnum fallback)
        where TEnum : struct
        => values.TryGetValue(key, out var value) && Enum.TryParse<TEnum>(value, ignoreCase: true, out var parsed)
            ? parsed : fallback;
}
