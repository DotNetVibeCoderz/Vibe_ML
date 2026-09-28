namespace MediaPipeNet.Inference;

/// <summary>The hardware backend (ONNX Runtime execution provider) used to run models.</summary>
public enum ExecutionProvider
{
    /// <summary>Pick the best available backend: CUDA → DirectML → CoreML → NNAPI → CPU.</summary>
    Auto = 0,
    /// <summary>Portable CPU execution (always available).</summary>
    Cpu,
    /// <summary>NVIDIA GPUs through CUDA (requires the MediaPipeNet.Cuda package and a CUDA runtime).</summary>
    Cuda,
    /// <summary>Any DirectX 12 GPU on Windows (requires the MediaPipeNet.DirectML package).</summary>
    DirectML,
    /// <summary>Apple Neural Engine / GPU on macOS and iOS.</summary>
    CoreML,
    /// <summary>
    /// Android Neural Networks API (NPU / GPU / DSP) on Android 8.1+. Available when the app runs the Android build of
    /// ONNX Runtime (the Microsoft.ML.OnnxRuntime package inside a .NET for Android / MAUI app).
    /// </summary>
    Nnapi,
}

/// <summary>Controls how models are executed.</summary>
public sealed record InferenceOptions
{
    /// <summary>Default options: automatic provider selection, ONNX Runtime's default threading.</summary>
    public static InferenceOptions Default { get; } = new();

    /// <summary>The requested execution provider. <see cref="ExecutionProvider.Auto"/> probes in order CUDA → DirectML → CoreML → NNAPI → CPU.</summary>
    public ExecutionProvider Provider { get; init; } = ExecutionProvider.Auto;

    /// <summary>GPU device index for CUDA / DirectML.</summary>
    public int DeviceId { get; init; }

    /// <summary>Threads used inside one operator. 0 lets ONNX Runtime choose (number of physical cores).</summary>
    public int IntraOpThreads { get; init; }

    /// <summary>Threads used to run independent operators in parallel. 0 lets ONNX Runtime choose.</summary>
    public int InterOpThreads { get; init; }

    /// <summary>
    /// When the requested provider fails to initialize, fall back to CPU instead of throwing.
    /// Always true for <see cref="ExecutionProvider.Auto"/>.
    /// </summary>
    public bool FallbackToCpu { get; init; } = true;

    /// <summary>Enables ONNX Runtime's per-session profiling output (JSON trace files).</summary>
    public bool EnableProfiling { get; init; }

    /// <summary>Maximum number of idle I/O contexts kept per model for concurrent callers.</summary>
    public int MaxPooledContexts { get; init; } = Math.Max(2, Environment.ProcessorCount / 2);

    /// <summary>
    /// Bind each context's input and output buffers to the session once (ONNX Runtime I/O binding) and
    /// reuse the binding for every run. Saves per-call marshaling; most useful with GPU providers.
    /// </summary>
    public bool UseIoBinding { get; init; }

    /// <summary>
    /// Preferred numeric precision of the models. <see cref="ModelPrecision.Float16"/> halves model size
    /// (best on GPUs); <see cref="ModelPrecision.Int8"/> quarters it (dynamic weight quantization, CPU).
    /// Models without such a variant fall back to float32.
    /// </summary>
    public ModelPrecision Precision { get; init; } = ModelPrecision.Float32;
}

/// <summary>Numeric precision variant of a model.</summary>
public enum ModelPrecision
{
    /// <summary>The reference float32 model (default).</summary>
    Float32 = 0,
    /// <summary>Weights stored as float16 (half the size; exact on GPUs, cast on CPU).</summary>
    Float16,
    /// <summary>Weights quantized to 8-bit integers (a quarter of the size).</summary>
    Int8,
}
