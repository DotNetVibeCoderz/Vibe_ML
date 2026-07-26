using System.Text.Json;
using BlazorML.Core.Configuration;
using BlazorML.Core.Domain;
using WicakOptions = BlazorML.Core.Configuration.ChatOptions;

namespace BlazorML.Agents.Tests;

/// <summary>
/// Reads the <c>appsettings.json</c> that actually ships, rather than a default-constructed
/// options object. A fresh install is the state most users meet first, and the defaults in code
/// and the values in the file can drift apart — which is only visible from here.
/// <para>
/// The file is deserialised directly rather than through <c>IConfiguration</c>: the fluent
/// builder extensions resolve against a different <c>IConfigurationBuilder</c> than the one this
/// project ends up with, and the whole configuration stack is more machinery than a check on a
/// JSON file needs.
/// </para>
/// </summary>
public class ShippedSettingsTests
{
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        // Enums are written by name in the settings file, as IConfiguration reads them.
        Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() }
    };

    private static JsonElement Root { get; } = Load();

    private static JsonElement Load()
    {
        var path = Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..",
            "src", "BlazorML.Web", "appsettings.json"));

        Assert.True(File.Exists(path), $"appsettings.json was not found at {path}.");

        return JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone();
    }

    private static T Section<T>(string name) where T : new()
    {
        Assert.True(Root.TryGetProperty(name, out var section), $"Section '{name}' is missing.");

        return section.Deserialize<T>(Json) ?? new T();
    }

    /// <summary>
    /// Regression test, and the reason this file exists. A real OpenAI key and a real Tavily key
    /// were once sitting in this file — which the README tells people to clone. Nothing here is
    /// gitignored away, so the guard has to be a failing test rather than a convention.
    /// </summary>
    [Fact]
    public void No_credential_is_committed_to_the_repository()
    {
        var chat = Section<WicakOptions>(SettingsSections.Chat);

        Assert.Empty(chat.OpenAI.ApiKey);
        Assert.Empty(chat.Anthropic.ApiKey);
        Assert.Empty(chat.Gemini.ApiKey);

        Assert.Empty(Section<ToolOptions>(SettingsSections.Tools).TavilyApiKey);

        var storage = Section<StorageOptions>(SettingsSections.Storage);
        Assert.Empty(storage.AzureBlob.ConnectionString);
        Assert.Empty(storage.S3.SecretKey);
        Assert.Empty(storage.MinIO.SecretKey);
    }

    /// <summary>
    /// What the chat panel keys off. Getting it wrong meant a brand-new workspace showed an
    /// assistant that looked ready and failed on first use.
    /// </summary>
    [Fact]
    public void Out_of_the_box_no_model_provider_counts_as_configured()
    {
        var chat = Section<WicakOptions>(SettingsSections.Chat);

        foreach (var provider in Enum.GetValues<ChatProviderKind>())
        {
            Assert.False(chat.IsProviderConfigured(provider),
                $"{provider} counts as configured in the shipped appsettings.json.");
        }
    }

    [Fact]
    public void Every_settings_section_the_app_reads_is_present()
    {
        foreach (var section in (string[])
                 [SettingsSections.Workspace, SettingsSections.Database, SettingsSections.Storage,
                  SettingsSections.Chat, SettingsSections.Tools, SettingsSections.Training,
                  SettingsSections.Scripting])
        {
            Assert.True(Root.TryGetProperty(section, out _), $"'{section}' is missing.");
        }
    }

    [Fact]
    public void All_four_databases_ship_with_a_connection_string_to_edit()
    {
        var database = Section<DatabaseOptions>(SettingsSections.Database);

        foreach (var provider in Enum.GetValues<DatabaseProviderKind>())
        {
            Assert.True(database.ConnectionStrings.ContainsKey(provider.ToString()),
                $"No connection string entry for {provider}.");
        }
    }

    [Fact]
    public void The_shipped_defaults_are_the_safe_ones()
    {
        Assert.Equal(DatabaseProviderKind.Sqlite,
            Section<DatabaseOptions>(SettingsSections.Database).Provider);

        Assert.Equal(StorageProviderKind.FileSystem,
            Section<StorageOptions>(SettingsSections.Storage).Provider);

        // Scripting that shells out to an interpreter is off unless asked for.
        Assert.False(Section<ScriptingOptions>(SettingsSections.Scripting).EnableR);
    }
}
