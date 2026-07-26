using System.ComponentModel.DataAnnotations;

namespace BlazorML.Core.Domain;

/// <summary>Base for every persisted aggregate: string ids keep all four database providers happy.</summary>
public abstract class EntityBase
{
    [Key]
    [MaxLength(40)]
    public string Id { get; set; } = Guid.NewGuid().ToString("n");

    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
    public DateTimeOffset? UpdatedAt { get; set; }

    [MaxLength(450)]
    public string? OwnerId { get; set; }
}

public class Dataset : EntityBase
{
    [MaxLength(200)]
    public string Name { get; set; } = string.Empty;

    [MaxLength(2000)]
    public string? Description { get; set; }

    public DatasetFormat Format { get; set; } = DatasetFormat.Csv;
    public DatasetSourceKind Source { get; set; } = DatasetSourceKind.Upload;

    /// <summary>Key inside the configured storage provider, not a local path.</summary>
    [MaxLength(1000)]
    public string StorageKey { get; set; } = string.Empty;

    public long SizeBytes { get; set; }
    public int RowCount { get; set; }
    public int ColumnCount { get; set; }

    /// <summary>Serialised <see cref="Analysis.DatasetProfile"/>, computed on import.</summary>
    public string? ProfileJson { get; set; }

    [MaxLength(500)]
    public string? Tags { get; set; }

    public bool IsSample { get; set; }
}

public class Experiment : EntityBase
{
    [MaxLength(200)]
    public string Name { get; set; } = string.Empty;

    [MaxLength(2000)]
    public string? Description { get; set; }

    /// <summary>Serialised <see cref="Designer.ExperimentGraph"/>.</summary>
    public string GraphJson { get; set; } = "{}";

    public int Version { get; set; } = 1;
    public MlTask Task { get; set; } = MlTask.None;
    public RunStatus LastStatus { get; set; } = RunStatus.Draft;
    public DateTimeOffset? LastRunAt { get; set; }

    [MaxLength(500)]
    public string? Tags { get; set; }

    public bool IsTemplate { get; set; }

    [MaxLength(200)]
    public string? TemplateCategory { get; set; }

    public ICollection<ExperimentRun> Runs { get; set; } = new List<ExperimentRun>();
    public ICollection<ExperimentVersion> Versions { get; set; } = new List<ExperimentVersion>();
}

/// <summary>Immutable snapshot of a graph, written every time an experiment is saved or run.</summary>
public class ExperimentVersion : EntityBase
{
    [MaxLength(40)]
    public string ExperimentId { get; set; } = string.Empty;
    public Experiment? Experiment { get; set; }

    public int Version { get; set; }
    public string GraphJson { get; set; } = "{}";

    [MaxLength(500)]
    public string? Note { get; set; }
}

public class ExperimentRun : EntityBase
{
    [MaxLength(40)]
    public string ExperimentId { get; set; } = string.Empty;
    public Experiment? Experiment { get; set; }

    public int ExperimentVersion { get; set; }
    public RunStatus Status { get; set; } = RunStatus.Queued;
    public DateTimeOffset? StartedAt { get; set; }
    public DateTimeOffset? FinishedAt { get; set; }

    public double DurationSeconds { get; set; }

    [MaxLength(4000)]
    public string? Error { get; set; }

    /// <summary>Per-node state and metrics, serialised as <c>Dictionary&lt;string, NodeRunResult&gt;</c>.</summary>
    public string? NodeResultsJson { get; set; }

    /// <summary>Primary metrics of the run, promoted for leaderboard sorting.</summary>
    public string? MetricsJson { get; set; }

    public ICollection<RunLogEntry> Logs { get; set; } = new List<RunLogEntry>();
}

public class RunLogEntry : EntityBase
{
    [MaxLength(40)]
    public string RunId { get; set; } = string.Empty;
    public ExperimentRun? Run { get; set; }

    [MaxLength(40)]
    public string? NodeId { get; set; }

    public LogLevel Level { get; set; } = LogLevel.Info;

    [MaxLength(4000)]
    public string Message { get; set; } = string.Empty;

    public int Sequence { get; set; }
}

public class TrainedModel : EntityBase
{
    [MaxLength(200)]
    public string Name { get; set; } = string.Empty;

    [MaxLength(2000)]
    public string? Description { get; set; }

    [MaxLength(40)]
    public string? ExperimentId { get; set; }

    [MaxLength(40)]
    public string? RunId { get; set; }

    public MlTask Task { get; set; }

    [MaxLength(200)]
    public string Algorithm { get; set; } = string.Empty;

    [MaxLength(200)]
    public string? LabelColumn { get; set; }

    /// <summary>Storage key of the serialised ML.NET zip.</summary>
    [MaxLength(1000)]
    public string StorageKey { get; set; } = string.Empty;

    public int Version { get; set; } = 1;

    /// <summary>Evaluation metrics captured at training time.</summary>
    public string? MetricsJson { get; set; }

    /// <summary>Input schema so the scoring API can build a request contract.</summary>
    public string? InputSchemaJson { get; set; }

    public long SizeBytes { get; set; }
}

public class ServiceEndpoint : EntityBase
{
    [MaxLength(200)]
    public string Name { get; set; } = string.Empty;

    [MaxLength(2000)]
    public string? Description { get; set; }

    /// <summary>URL-safe segment the scoring route is mounted on.</summary>
    [MaxLength(120)]
    public string Slug { get; set; } = string.Empty;

    [MaxLength(40)]
    public string ModelId { get; set; } = string.Empty;
    public TrainedModel? Model { get; set; }

    public EndpointStatus Status { get; set; } = EndpointStatus.Stopped;
    public long CallCount { get; set; }
    public DateTimeOffset? LastCalledAt { get; set; }

    public ICollection<ApiKey> ApiKeys { get; set; } = new List<ApiKey>();
}

public class ApiKey : EntityBase
{
    [MaxLength(200)]
    public string Name { get; set; } = string.Empty;

    /// <summary>SHA-256 of the key; the plaintext is shown once at creation and never stored.</summary>
    [MaxLength(100)]
    public string KeyHash { get; set; } = string.Empty;

    /// <summary>First characters of the key so a user can tell keys apart in the list.</summary>
    [MaxLength(20)]
    public string Prefix { get; set; } = string.Empty;

    [MaxLength(40)]
    public string? EndpointId { get; set; }
    public ServiceEndpoint? Endpoint { get; set; }

    public DateTimeOffset? ExpiresAt { get; set; }
    public DateTimeOffset? LastUsedAt { get; set; }
    public bool IsRevoked { get; set; }
}

public class ChatSession : EntityBase
{
    [MaxLength(200)]
    public string Title { get; set; } = "Sesi baru";

    [MaxLength(40)]
    public string? ExperimentId { get; set; }

    public ChatProviderKind Provider { get; set; }

    [MaxLength(120)]
    public string? Model { get; set; }

    public ICollection<ChatMessage> Messages { get; set; } = new List<ChatMessage>();
}

public class ChatMessage : EntityBase
{
    [MaxLength(40)]
    public string SessionId { get; set; } = string.Empty;
    public ChatSession? Session { get; set; }

    public ChatRole Role { get; set; }

    public string Content { get; set; } = string.Empty;

    /// <summary>Serialised list of <see cref="ChatAttachment"/>.</summary>
    public string? AttachmentsJson { get; set; }

    /// <summary>Names of kernel functions invoked while producing this message.</summary>
    [MaxLength(1000)]
    public string? ToolCalls { get; set; }

    public int Sequence { get; set; }
    public int TokenCount { get; set; }
}

public class ChatAttachment
{
    public AttachmentKind Kind { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Url { get; set; } = string.Empty;
    public string? ContentType { get; set; }
    public long SizeBytes { get; set; }
}

/// <summary>A ready-made experiment plus its dataset, surfaced in the marketplace.</summary>
public class MarketplaceItem : EntityBase
{
    [MaxLength(200)]
    public string Name { get; set; } = string.Empty;

    [MaxLength(2000)]
    public string? Summary { get; set; }

    [MaxLength(120)]
    public string Category { get; set; } = string.Empty;

    public MlTask Task { get; set; }

    /// <summary>Emoji or short glyph used as the card mark.</summary>
    [MaxLength(16)]
    public string? Glyph { get; set; }

    [MaxLength(40)]
    public string? ExperimentId { get; set; }

    [MaxLength(40)]
    public string? DatasetId { get; set; }

    public int Installs { get; set; }

    [MaxLength(500)]
    public string? Tags { get; set; }
}

/// <summary>Key/value overrides that let configuration be edited from inside the app.</summary>
public class AppSetting : EntityBase
{
    [MaxLength(200)]
    public string Key { get; set; } = string.Empty;

    public string? Value { get; set; }

    [MaxLength(120)]
    public string Section { get; set; } = string.Empty;

    public bool IsSecret { get; set; }
}

public class AuditEntry : EntityBase
{
    [MaxLength(120)]
    public string Action { get; set; } = string.Empty;

    [MaxLength(120)]
    public string? EntityType { get; set; }

    [MaxLength(40)]
    public string? EntityId { get; set; }

    [MaxLength(200)]
    public string? Actor { get; set; }

    [MaxLength(2000)]
    public string? Detail { get; set; }
}
