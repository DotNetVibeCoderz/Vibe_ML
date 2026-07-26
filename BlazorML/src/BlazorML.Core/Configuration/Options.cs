using BlazorML.Core.Domain;

namespace BlazorML.Core.Configuration;

/// <summary>
/// Every section below maps to one <c>appsettings.json</c> block and to one card on the Settings
/// page. <see cref="Abstractions.ISettingsService"/> reads the file first, then overlays anything
/// the user has changed in the app, so nobody has to edit JSON on the server to change a key.
/// </summary>
public static class SettingsSections
{
    public const string Workspace = "Workspace";
    public const string Database = "Database";
    public const string Storage = "Storage";
    public const string Chat = "Chat";
    public const string Tools = "Tools";
    public const string Training = "Training";
    public const string Scripting = "Scripting";
    public const string Email = "Email";
}

/// <summary>
/// Outgoing mail. Only used for password resets today, so it is entirely optional: without a
/// server the reset flow still works, and the link is written to the application log for a
/// self-hosted administrator to pass on.
/// </summary>
public sealed class EmailOptions
{
    public bool Enabled { get; set; }

    public string Host { get; set; } = string.Empty;
    public int Port { get; set; } = 587;
    public bool UseStartTls { get; set; } = true;

    public string Username { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;

    public string FromAddress { get; set; } = string.Empty;
    public string FromName { get; set; } = "Blazor ML Studio";

    /// <summary>How long a password-reset link stays valid.</summary>
    public int ResetLinkValidHours { get; set; } = 2;

    public bool IsConfigured => Enabled && !string.IsNullOrWhiteSpace(Host)
                                        && !string.IsNullOrWhiteSpace(FromAddress);
}

public sealed class WorkspaceOptions
{
    public string Name { get; set; } = "Blazor ML Studio";
    public string Tagline { get; set; } = "Rancang, latih, dan terbitkan model tanpa menulis kode.";

    /// <summary>Two-letter UI language: <c>id</c> or <c>en</c>.</summary>
    public string DefaultLanguage { get; set; } = "id";

    /// <summary><c>system</c>, <c>light</c> or <c>dark</c>.</summary>
    public string DefaultTheme { get; set; } = "system";

    public bool AllowRegistration { get; set; } = true;
    public string BuiltBy { get; set; } = "Gravicode Studios";
    public string LedBy { get; set; } = "Kang Fadhil";
}

public sealed class DatabaseOptions
{
    public DatabaseProviderKind Provider { get; set; } = DatabaseProviderKind.Sqlite;

    /// <summary>Connection string per provider, so switching does not lose the others.</summary>
    public Dictionary<string, string> ConnectionStrings { get; set; } = new()
    {
        ["Sqlite"] = "Data Source=blazorml.db",
        ["SqlServer"] = "Server=localhost;Database=BlazorML;Trusted_Connection=True;TrustServerCertificate=True",
        ["MySql"] = "Server=localhost;Database=blazorml;User=root;Password=",
        ["PostgreSql"] = "Host=localhost;Database=blazorml;Username=postgres;Password="
    };

    public bool SeedSampleData { get; set; } = true;
    public int CommandTimeoutSeconds { get; set; } = 60;

    public string Current => ConnectionStrings.TryGetValue(Provider.ToString(), out var cs)
        ? cs
        : "Data Source=blazorml.db";
}

public sealed class StorageOptions
{
    public StorageProviderKind Provider { get; set; } = StorageProviderKind.FileSystem;

    public FileSystemStorageOptions FileSystem { get; set; } = new();
    public AzureBlobStorageOptions AzureBlob { get; set; } = new();
    public S3StorageOptions S3 { get; set; } = new();
    public S3StorageOptions MinIO { get; set; } = new() { ServiceUrl = "http://localhost:9000", ForcePathStyle = true };

    /// <summary>Largest upload accepted, in megabytes.</summary>
    public int MaxUploadMegabytes { get; set; } = 256;
}

public sealed class FileSystemStorageOptions
{
    /// <summary>Relative paths resolve under the app's content root.</summary>
    public string RootPath { get; set; } = "App_Data/storage";
}

public sealed class AzureBlobStorageOptions
{
    public string ConnectionString { get; set; } = string.Empty;
    public string Container { get; set; } = "blazorml";
}

public sealed class S3StorageOptions
{
    public string AccessKey { get; set; } = string.Empty;
    public string SecretKey { get; set; } = string.Empty;
    public string Bucket { get; set; } = "blazorml";
    public string Region { get; set; } = "us-east-1";

    /// <summary>Set for MinIO or any other S3-compatible server. Empty means real AWS S3.</summary>
    public string ServiceUrl { get; set; } = string.Empty;

    /// <summary>MinIO needs path-style addressing; AWS does not.</summary>
    public bool ForcePathStyle { get; set; }
}

/// <summary>Profesor Wicak's configuration. Persona and model live here so they are editable in the UI.</summary>
public sealed class ChatOptions
{
    public bool Enabled { get; set; } = true;
    public string AssistantName { get; set; } = "Profesor Wicak";

    public ChatProviderKind Provider { get; set; } = ChatProviderKind.OpenAI;

    public string SystemPrompt { get; set; } =
        """
        Kamu adalah Profesor Wicak, data scientist senior yang mendampingi pengguna di Blazor ML Studio.

        Cara kerjamu:
        - Jawab dalam bahasa yang dipakai pengguna. Kalau dia menulis bahasa Indonesia, jawab bahasa Indonesia.
        - Sebelum menyarankan algoritma, periksa dulu datanya lewat fungsi yang tersedia. Jangan menebak isi kolom.
        - Kalau pengguna minta dibuatkan atau diubah alur eksperimen, kerjakan lewat fungsi designer,
          lalu jelaskan singkat apa yang kamu susun dan kenapa.
        - Sebutkan angka yang kamu lihat, bukan kesan umum. "AUC 0.82 pada 1.200 baris" lebih berguna
          daripada "hasilnya lumayan".
        - Kalau sebuah pendekatan berisiko (data terlalu sedikit, kelas timpang, kebocoran label),
          katakan sekali dengan jelas, lalu tetap bantu kerjakan.
        - Format jawaban dengan markdown: tabel untuk perbandingan, blok kode untuk kode.
        """;

    public double Temperature { get; set; } = 0.4;
    public int MaxTokens { get; set; } = 4096;

    /// <summary>How many past turns are replayed to the model on each request.</summary>
    public int HistoryTurns { get; set; } = 12;

    public bool EnableFunctionCalling { get; set; } = true;
    public bool StreamResponses { get; set; } = true;

    public OpenAiOptions OpenAI { get; set; } = new();
    public AnthropicOptions Anthropic { get; set; } = new();
    public GeminiOptions Gemini { get; set; } = new();
    public OllamaOptions Ollama { get; set; } = new();

    public bool IsProviderConfigured(ChatProviderKind kind) => kind switch
    {
        ChatProviderKind.OpenAI => !string.IsNullOrWhiteSpace(OpenAI.ApiKey),
        ChatProviderKind.Anthropic => !string.IsNullOrWhiteSpace(Anthropic.ApiKey),
        ChatProviderKind.Gemini => !string.IsNullOrWhiteSpace(Gemini.ApiKey),
        ChatProviderKind.Ollama => Ollama.Enabled && !string.IsNullOrWhiteSpace(Ollama.Endpoint),
        _ => false
    };
}

public sealed class OpenAiOptions
{
    public string ApiKey { get; set; } = string.Empty;
    public string Model { get; set; } = "gpt-4o";
    public string? Endpoint { get; set; }
}

public sealed class AnthropicOptions
{
    public string ApiKey { get; set; } = string.Empty;
    public string Model { get; set; } = "claude-sonnet-4-5";
}

public sealed class GeminiOptions
{
    public string ApiKey { get; set; } = string.Empty;
    public string Model { get; set; } = "gemini-2.0-flash";
}

public sealed class OllamaOptions
{
    /// <summary>
    /// Must be turned on deliberately. The endpoint below has a sensible default, so treating
    /// "an endpoint is set" as "a provider is available" made the workspace look configured on a
    /// fresh install — the chat panel showed itself as connected and then failed on the first
    /// message. Running a local Ollama is a claim only the user can make.
    /// </summary>
    public bool Enabled { get; set; }

    public string Endpoint { get; set; } = "http://localhost:11434";
    public string Model { get; set; } = "llama3.2";
}

/// <summary>Credentials for the kernel functions that reach outside the app.</summary>
public sealed class ToolOptions
{
    public string TavilyApiKey { get; set; } = string.Empty;
    public bool EnableWebSearch { get; set; } = true;
    public bool EnableUrlScraping { get; set; } = true;
    public int WebRequestTimeoutSeconds { get; set; } = 30;
    public int MaxSearchResults { get; set; } = 5;
}

public sealed class TrainingOptions
{
    /// <summary>Runs beyond this queue behind the ones already going.</summary>
    public int MaxConcurrentRuns { get; set; } = 2;

    public int DefaultAutoMlSeconds { get; set; } = 60;
    public int RunTimeoutMinutes { get; set; } = 30;

    /// <summary>Rows kept in the per-node preview shown on the canvas.</summary>
    public int PreviewRows { get; set; } = 10;

    /// <summary>Guard against loading a file large enough to exhaust the server's memory.</summary>
    public int MaxRowsInMemory { get; set; } = 1_000_000;
}

public sealed class ScriptingOptions
{
    public bool EnablePython { get; set; } = true;
    public bool EnableR { get; set; }
    public bool EnableCSharp { get; set; } = true;
    public bool EnableJavaScript { get; set; } = true;

    /// <summary>Path to <c>python3xx.dll</c> / <c>libpython3.x.so</c>. Empty lets Python.NET search.</summary>
    public string PythonDll { get; set; } = string.Empty;
    public string RscriptPath { get; set; } = "Rscript";
    public int DefaultTimeoutSeconds { get; set; } = 120;
}
