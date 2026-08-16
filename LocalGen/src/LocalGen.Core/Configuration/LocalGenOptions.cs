using LocalGen.Core.Engines;

namespace LocalGen.Core.Configuration;

/// <summary>Root configuration, bound from the <c>LocalGen</c> configuration section.</summary>
public sealed class LocalGenOptions
{
    public const string SectionName = "LocalGen";

    /// <summary>Directory holding models, manifests and runtime state.</summary>
    public string DataDirectory { get; set; } = LocalGenPaths.DefaultDataDirectory;

    public EngineOptions Engine { get; set; } = new();

    public ServerOptions Server { get; set; } = new();

    public RuntimeOptions Runtime { get; set; } = new();

    public ToolOptions Tools { get; set; } = new();

    public RagOptions Rag { get; set; } = new();

    public string ModelsDirectory => Path.Combine(DataDirectory, "models");

    public string SkillsDirectory => Path.Combine(DataDirectory, "skills");

    public string SessionsDirectory => Path.Combine(DataDirectory, "sessions");

    public string LogsDirectory => Path.Combine(DataDirectory, "logs");

    public string ManifestPath => Path.Combine(DataDirectory, "manifest.json");
}

public sealed class EngineOptions
{
    /// <summary>Backend used when a model does not pin one. LlamaSharp is the product default.</summary>
    public EngineKind Default { get; set; } = EngineKind.LlamaSharp;

    public DeviceKind Device { get; set; } = DeviceKind.Auto;

    /// <summary>Layers offloaded to GPU. Null offloads everything when a GPU is in use.</summary>
    public int? GpuLayers { get; set; }

    public int? ContextSize { get; set; }

    public int? Threads { get; set; }

    /// <summary>Per-GPU weight split for tensor parallelism. Empty means single GPU.</summary>
    public float[] TensorSplit { get; set; } = [];

    /// <summary>Base address of the remote endpoint when <see cref="Default"/> is RemoteOpenAI.</summary>
    public string? RemoteEndpoint { get; set; }

    public string? RemoteApiKey { get; set; }
}

public sealed class ServerOptions
{
    public string Host { get; set; } = "127.0.0.1";

    public int Port { get; set; } = 11434;

    /// <summary>Optional bearer token. When set, every API call must present it.</summary>
    public string? ApiKey { get; set; }

    /// <summary>Origins allowed by CORS. <c>*</c> permits any.</summary>
    public string[] CorsOrigins { get; set; } = ["*"];

    /// <summary>How long an idle model stays resident before it is unloaded.</summary>
    public TimeSpan ModelIdleTimeout { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>Maximum models held in memory at once.</summary>
    public int MaxLoadedModels { get; set; } = 2;

    /// <summary>Requests served concurrently per loaded model.</summary>
    public int MaxConcurrentRequests { get; set; } = 4;

    public string BaseUrl => $"http://{Host}:{Port}";
}

public sealed class RuntimeOptions
{
    /// <summary>Blocks every outbound network call, for air-gapped and edge deployments.</summary>
    public bool OfflineMode { get; set; }

    /// <summary>Model loaded eagerly at startup so the first request is not slow.</summary>
    public string? PreloadModel { get; set; }

    public bool CollectMetrics { get; set; } = true;

    /// <summary>How often GPU/CPU counters are sampled for the monitoring dashboard.</summary>
    public TimeSpan MetricsInterval { get; set; } = TimeSpan.FromSeconds(2);
}

/// <summary>Controls the built-in kernel functions. All are enabled by default per the product spec.</summary>
public sealed class ToolOptions
{
    public bool Math { get; set; } = true;

    public bool InternetSearch { get; set; } = true;

    public bool Download { get; set; } = true;

    public bool WebScrape { get; set; } = true;

    public bool CodeExecution { get; set; } = true;

    public bool TimeAndDate { get; set; } = true;

    public bool FileSystem { get; set; } = true;

    /// <summary>API key for Tavily, which backs the internet search function.</summary>
    public string? TavilyApiKey { get; set; }

    /// <summary>
    /// Directories the file system and code execution functions may touch. Empty means the
    /// LocalGen workspace only — these functions are powerful, so the default stays narrow.
    /// </summary>
    public string[] AllowedPaths { get; set; } = [];

    /// <summary>Wall-clock limit for a single code execution.</summary>
    public TimeSpan CodeExecutionTimeout { get; set; } = TimeSpan.FromMinutes(2);

    /// <summary>
    /// Whether executed code may install SDKs, runtimes and packages. Off by default because
    /// it lets the model change the host machine.
    /// </summary>
    public bool AllowDependencyInstall { get; set; }
}

public sealed class RagOptions
{
    /// <summary>Vector backend: <c>sqlite</c>, <c>qdrant</c>, <c>chroma</c>, <c>azureaisearch</c> or <c>inmemory</c>.</summary>
    public string Provider { get; set; } = "sqlite";

    public string ConnectionString { get; set; } = string.Empty;

    /// <summary>Model used to embed chunks and queries.</summary>
    public string EmbeddingModel { get; set; } = "nomic-embed-text";

    public int ChunkSize { get; set; } = 1000;

    public int ChunkOverlap { get; set; } = 200;

    /// <summary>Chunks retrieved per query.</summary>
    public int TopK { get; set; } = 5;

    /// <summary>Minimum cosine similarity for a chunk to be considered relevant.</summary>
    public double MinRelevance { get; set; } = 0.5;
}
