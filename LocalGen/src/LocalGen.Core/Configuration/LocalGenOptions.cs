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

    public TelemetryOptions Telemetry { get; set; } = new();

    public CacheOptions Cache { get; set; } = new();

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

    /// <summary>
    /// Per-GPU weight split for tensor parallelism. Empty leaves the backend to divide the model
    /// in proportion to free VRAM, which is the right answer for identical cards.
    /// </summary>
    /// <remarks>
    /// Relative weights on any scale: <c>0.6, 0.4</c> and <c>60, 40</c> mean the same thing. Set
    /// it when the cards differ in size, or to keep part of one card free for something else.
    /// </remarks>
    public float[] TensorSplit { get; set; } = [];

    /// <summary>
    /// How a model is divided between GPUs. <c>Layer</c> gives each card a run of layers;
    /// <c>Row</c> splits every tensor across all of them and needs a fast interconnect to pay off.
    /// </summary>
    public GpuSplitMode SplitMode { get; set; } = GpuSplitMode.Auto;

    /// <summary>
    /// GPU that holds the tensors which are not split. Null means device 0. Setting it matters
    /// when the cards differ: the KV cache lands here, so it should be the one with room.
    /// </summary>
    public int? MainGpu { get; set; }

    /// <summary>
    /// Decode concurrent requests together against one context instead of queueing them per model.
    /// </summary>
    /// <remarks>
    /// Off by default. It is a real throughput win when requests overlap, but the context window
    /// is then shared between them rather than belonging to one request at a time, so it changes
    /// how much room a long conversation has. Vision models keep the serialised path regardless —
    /// their image encoding does not batch.
    /// </remarks>
    public bool BatchedInference { get; set; }

    /// <summary>
    /// Requests decoded together when <see cref="BatchedInference"/> is on. Null follows
    /// <see cref="ServerOptions.MaxConcurrentRequests"/>.
    /// </summary>
    public int? MaxBatchedSequences { get; set; }

    /// <summary>Base address of the remote endpoint when <see cref="Default"/> is RemoteOpenAI.</summary>
    public string? RemoteEndpoint { get; set; }

    public string? RemoteApiKey { get; set; }
}

public sealed class ServerOptions
{
    public string Host { get; set; } = "127.0.0.1";

    public int Port { get; set; } = 11434;

    /// <summary>
    /// Optional bearer token for single-tenant use. When set, every API call must present it.
    /// </summary>
    /// <remarks>
    /// Kept alongside <see cref="ApiKeys"/> rather than folded into it: the overwhelmingly common
    /// deployment is one operator with one key, and making them write a tenant entry for that
    /// would be ceremony. A key here behaves as an unlimited tenant named <c>default</c>.
    /// </remarks>
    public string? ApiKey { get; set; }

    /// <summary>
    /// Named keys with their own quotas, for serving more than one caller from one instance.
    /// Empty means single-tenant behaviour driven by <see cref="ApiKey"/>.
    /// </summary>
    public ApiKeyDescriptor[] ApiKeys { get; set; } = [];

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

/// <summary>One API key and the limits that apply to whoever holds it.</summary>
/// <remarks>
/// Quotas are expressed per key rather than per user because the key is the only identity the
/// OpenAI wire protocol carries. A zero on any limit means "unmetered" — that is the default, so
/// adding a key to share access does not silently start throttling it.
/// </remarks>
public sealed class ApiKeyDescriptor
{
    /// <summary>The secret the caller presents. Compared in fixed time.</summary>
    public string Key { get; set; } = string.Empty;

    /// <summary>Label for logs, metrics and the Admin Control. Not a secret.</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Lets a key be revoked without deleting its quota configuration.</summary>
    public bool Enabled { get; set; } = true;

    /// <summary>Requests allowed in any rolling minute. Zero is unmetered.</summary>
    public int RequestsPerMinute { get; set; }

    /// <summary>Tokens — prompt plus completion — allowed per rolling day. Zero is unmetered.</summary>
    public long TokensPerDay { get; set; }

    /// <summary>Requests this key may have in flight at once. Zero is unmetered.</summary>
    public int MaxConcurrentRequests { get; set; }

    /// <summary>Models this key may address. Empty allows every model the server has.</summary>
    public string[] AllowedModels { get; set; } = [];
}

/// <summary>
/// Caching of completed responses, keyed on the prompt and the sampling settings.
/// </summary>
public sealed class CacheOptions
{
    /// <summary>Off by default: a cache changes observable behaviour, so it is opted into.</summary>
    public bool Enabled { get; set; }

    /// <summary>Entries retained before the least recently used one is evicted.</summary>
    public int MaxEntries { get; set; } = 256;

    public TimeSpan Ttl { get; set; } = TimeSpan.FromMinutes(30);

    /// <summary>
    /// Whether sampled generations are cached as well as greedy ones. Off by default: with a
    /// temperature above zero the caller asked for variety, and replaying one answer forever
    /// would quietly take that away. Greedy and seeded requests are reproducible anyway, so
    /// caching them changes only the latency.
    /// </summary>
    public bool CacheNonDeterministic { get; set; }
}

/// <summary>
/// Export of the metrics LocalGen already collects. Both exporters are off by default so that a
/// desktop install opens no ports and dials nothing.
/// </summary>
public sealed class TelemetryOptions
{
    /// <summary>Serves the Prometheus exposition format at <see cref="MetricsPath"/>.</summary>
    public bool PrometheusEnabled { get; set; }

    public string MetricsPath { get; set; } = "/metrics";

    /// <summary>
    /// Whether the scrape endpoint sits behind the API key. Off by default because a Prometheus
    /// server scraping a pod inside a cluster is not usually given one, and the endpoint exposes
    /// counters rather than content.
    /// </summary>
    public bool RequireApiKeyForMetrics { get; set; }

    /// <summary>OTLP collector address. Empty leaves the OTLP exporter unregistered.</summary>
    public string? OtlpEndpoint { get; set; }

    /// <summary><c>grpc</c> or <c>httpprotobuf</c>.</summary>
    public string OtlpProtocol { get; set; } = "grpc";

    /// <summary>Whether request traces are exported alongside metrics.</summary>
    public bool Traces { get; set; } = true;

    /// <summary>Identifies this instance in the collector.</summary>
    public string ServiceName { get; set; } = "localgen";
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
