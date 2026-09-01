using System.Text.Json.Serialization;

namespace LocalGen.Core.Engines;

/// <summary>Identifies an inference backend implementation.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<EngineKind>))]
public enum EngineKind
{
    /// <summary>llama.cpp via LLamaSharp. The default backend; runs GGUF on CPU or GPU.</summary>
    LlamaSharp,

    /// <summary>ONNX Runtime GenAI. Strong on DirectML/NPU hardware and quantized ONNX models.</summary>
    OnnxRuntime,

    /// <summary>Microsoft Foundry Local, which manages its own catalogue and serving process.</summary>
    FoundryLocal,

    /// <summary>Any OpenAI-compatible HTTP endpoint used as a remote backend.</summary>
    RemoteOpenAI
}

/// <summary>Compute device an engine should run on.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<DeviceKind>))]
public enum DeviceKind
{
    /// <summary>Let the engine pick, preferring GPU when one is usable.</summary>
    Auto,
    Cpu,
    Cuda,
    Vulkan,
    Metal,
    DirectML,
    Npu
}

/// <summary>What a backend can do. Callers check this before offering a feature in the UI.</summary>
public sealed record EngineCapabilities
{
    public bool SupportsStreaming { get; init; } = true;

    public bool SupportsEmbeddings { get; init; }

    /// <summary>Whether the backend can constrain output to a grammar (GBNF) or JSON schema.</summary>
    public bool SupportsGrammar { get; init; }

    public bool SupportsToolCalling { get; init; }

    public bool SupportsVision { get; init; }

    /// <summary>Whether weights can be split across several GPUs (tensor parallelism).</summary>
    public bool SupportsMultiGpu { get; init; }

    /// <summary>Model file formats the backend can load.</summary>
    public IReadOnlyList<ModelFormat> Formats { get; init; } = [];

    public IReadOnlyList<DeviceKind> Devices { get; init; } = [DeviceKind.Cpu];
}

/// <summary>Weight file format.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<ModelFormat>))]
public enum ModelFormat
{
    Unknown,

    /// <summary>llama.cpp GGUF, usually quantized.</summary>
    Gguf,

    /// <summary>HuggingFace safetensors.</summary>
    Safetensors,

    /// <summary>ONNX graph, optionally with an external .onnx.data blob.</summary>
    Onnx,

    /// <summary>Managed by Foundry Local; the path is an alias rather than a file.</summary>
    FoundryAlias
}

/// <summary>Static description of a backend, surfaced in the Engine settings screen.</summary>
public sealed record EngineDescriptor
{
    public required EngineKind Kind { get; init; }

    /// <summary>Human-readable name shown in the UI.</summary>
    public required string DisplayName { get; init; }

    public required string Description { get; init; }

    public EngineCapabilities Capabilities { get; init; } = new();

    /// <summary>Guidance shown next to the engine picker so users can choose sensibly.</summary>
    public string Recommendation { get; init; } = string.Empty;
}

/// <summary>How a model should be loaded into memory.</summary>
public sealed record ModelLoadOptions
{
    public DeviceKind Device { get; init; } = DeviceKind.Auto;

    /// <summary>
    /// Transformer layers to offload to the GPU. Null means "all of them" when a GPU is
    /// selected, which is the right default for models that fit in VRAM.
    /// </summary>
    public int? GpuLayerCount { get; init; }

    /// <summary>Context window in tokens. Null falls back to the model's trained context length.</summary>
    public int? ContextSize { get; init; }

    /// <summary>Batch size used while ingesting the prompt.</summary>
    public int? BatchSize { get; init; }

    public int? ThreadCount { get; init; }

    /// <summary>
    /// Relative weight of the model to place on each GPU, for multi-GPU tensor splitting. Empty
    /// leaves the backend to divide the model in proportion to free VRAM.
    /// </summary>
    /// <remarks>
    /// Any positive scale works: <c>0.6, 0.4</c> and <c>60, 40</c> describe the same split.
    /// <see cref="TensorSplitPlan"/> normalises it against the devices actually present.
    /// </remarks>
    public IReadOnlyList<float> TensorSplit { get; init; } = [];

    /// <summary>How the model is divided between GPUs when more than one is present.</summary>
    public GpuSplitMode SplitMode { get; init; } = GpuSplitMode.Auto;

    /// <summary>
    /// Device holding the tensors that are not split — the KV cache and small intermediates.
    /// Null leaves the backend's default, which is device 0.
    /// </summary>
    public int? MainGpu { get; init; }

    /// <summary>Keep weights memory-mapped instead of copying them into RAM.</summary>
    public bool UseMemoryMap { get; init; } = true;

    /// <summary>Lock weights in RAM to stop the OS paging them out.</summary>
    public bool UseMemoryLock { get; init; }

    /// <summary>Load the model for embedding rather than text generation.</summary>
    public bool EmbeddingMode { get; init; }

    /// <summary>
    /// Decode several requests together against one context instead of queueing them.
    /// </summary>
    /// <remarks>
    /// Off by default. Batching raises total throughput markedly when requests overlap, but the
    /// context is shared between them: <see cref="MaxSequences"/> concurrent requests each get
    /// roughly a <see cref="ContextSize"/>-divided-by-that share of the window, and a long
    /// conversation that would have fitted on its own can run out of room. That is a trade an
    /// operator should make deliberately.
    /// </remarks>
    public bool BatchedInference { get; init; }

    /// <summary>Requests decoded together when <see cref="BatchedInference"/> is on.</summary>
    public int MaxSequences { get; init; } = 4;

    public static readonly ModelLoadOptions Default = new();
}
