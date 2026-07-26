using BlazorML.Core.Data;
using BlazorML.Core.Designer;
using BlazorML.Core.Domain;

namespace BlazorML.Core.Abstractions;

/// <summary>
/// Configuration that can be edited from inside the app. Reads fall through to
/// <c>appsettings.json</c>; writes go to the database so they survive a restart without
/// anyone editing a file on the server.
/// </summary>
public interface ISettingsService
{
    Task<T> GetAsync<T>(string section, CancellationToken ct = default) where T : class, new();

    Task SaveAsync<T>(string section, T value, CancellationToken ct = default) where T : class;

    /// <summary>Drops the cached overlay so the next read reflects a change made elsewhere.</summary>
    void Invalidate(string? section = null);

    event Action<string>? SectionChanged;
}

/// <summary>
/// Protects the secret-bearing settings the app writes to its own database — provider API keys,
/// storage credentials, connection strings typed into a module. Implemented in the web layer on
/// top of ASP.NET Core data protection, so the key ring is managed for us.
/// </summary>
public interface ISecretProtector
{
    string Protect(string plaintext);

    /// <summary>Returns null when the value cannot be unprotected, e.g. after a key-ring reset.</summary>
    string? Unprotect(string protectedText);
}

/// <summary>
/// Sends the few transactional messages the app produces — currently only password resets.
/// </summary>
public interface IEmailSender
{
    /// <summary>
    /// False when no mail server is configured. The caller still behaves identically; this only
    /// decides whether a self-hosted administrator is told to look in the log for the link.
    /// </summary>
    bool IsConfigured { get; }

    Task SendAsync(string toAddress, string subject, string htmlBody, CancellationToken ct = default);
}

/// <summary>Loads a registered dataset into memory and writes new ones back.</summary>
public interface IDatasetService
{
    Task<TabularData> LoadAsync(string datasetId, int rowLimit = 0, CancellationToken ct = default);

    Task<Dataset> ImportAsync(string name, Stream content, DatasetFormat format,
        DatasetSourceKind source, string? ownerId, CancellationToken ct = default);

    Task<Dataset> SaveAsync(string name, TabularData data, DatasetFormat format,
        string? ownerId, CancellationToken ct = default);

    Task<TabularData> PreviewAsync(string datasetId, int rows = 50, CancellationToken ct = default);
}

/// <summary>
/// Runs one experiment graph. Progress is pushed through <paramref name="progress"/> so the
/// designer can light up nodes while the run is still going.
/// </summary>
public interface IExperimentRunner
{
    Task<ExperimentRun> RunAsync(string experimentId, string? userId,
        IProgress<RunProgress>? progress = null, CancellationToken ct = default);

    /// <summary>Validates wiring and required parameters without executing anything.</summary>
    Task<IReadOnlyList<GraphIssue>> ValidateAsync(ExperimentGraph graph, CancellationToken ct = default);
}

public sealed record RunProgress(string NodeId, NodeRunState State, string? Message = null,
    NodeRunResult? Result = null);

public sealed record GraphIssue(string? NodeId, GraphIssueSeverity Severity, string Message);

public enum GraphIssueSeverity
{
    Warning = 0,
    Error = 1
}

/// <summary>
/// Lets the ML layer call a language model without depending on the agent layer. Implemented in
/// <c>BlazorML.Agents</c> and wired up at startup, so the LLM modules stay optional.
/// </summary>
public interface ILlmActionRunner
{
    /// <summary>True when at least one provider has usable credentials.</summary>
    bool IsConfigured { get; }

    /// <summary>
    /// Sends one prompt and returns the raw completion. <paramref name="provider"/> is the
    /// module's override, or null to use the workspace default.
    /// </summary>
    Task<string> CompleteAsync(string systemPrompt, string userPrompt, string? provider = null,
        double temperature = 0.2, CancellationToken ct = default);
}

/// <summary>Executes a user script module against the rows flowing through it.</summary>
public interface IScriptRunner
{
    ScriptLanguage Language { get; }

    /// <summary>True when the runtime this language needs is actually present on the machine.</summary>
    Task<bool> IsAvailableAsync(CancellationToken ct = default);

    /// <summary>Explains what is missing when <see cref="IsAvailableAsync"/> is false.</summary>
    string UnavailableReason { get; }

    Task<TabularData> RunAsync(string code, TabularData? input1, TabularData? input2,
        TimeSpan timeout, CancellationToken ct = default);
}

/// <summary>Saves and loads trained ML.NET models, and keeps the version history.</summary>
public interface IModelRegistry
{
    Task<TrainedModel> RegisterAsync(string name, string? description, MlTask task, string algorithm,
        string? labelColumn, Stream modelZip, string? metricsJson, string? inputSchemaJson,
        string? experimentId, string? runId, string? ownerId, CancellationToken ct = default);

    Task<Stream> OpenAsync(string modelId, CancellationToken ct = default);

    Task<IReadOnlyList<TrainedModel>> VersionsAsync(string name, CancellationToken ct = default);
}
