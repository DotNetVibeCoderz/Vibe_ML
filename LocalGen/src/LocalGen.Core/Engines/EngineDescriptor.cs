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

    /// <summary>Fraction of the model to place on each GPU, for multi-GPU tensor splitting.</summary>
    public IReadOnlyList<float> TensorSplit { get; init; } = [];

    /// <summary>Keep weights memory-mapped instead of copying them into RAM.</summary>
    public bool UseMemoryMap { get; init; } = true;

    /// <summary>Lock weights in RAM to stop the OS paging them out.</summary>
    public bool UseMemoryLock { get; init; }

    /// <summary>Load the model for embedding rather than text generation.</summary>
    public bool EmbeddingMode { get; init; }

    public static readonly ModelLoadOptions Default = new();
}
